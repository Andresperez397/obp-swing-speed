"""Run the pre-specified lab analyses in ANALYSIS_PLAN.md (Q1-Q3); write tables to reports/tables/."""
from __future__ import annotations

import os

# Each worker fits many small models; pin BLAS/OpenMP to one thread to avoid oversubscription.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swing import data, models, stats  # noqa: E402

OUT = data.ROOT / "reports" / "tables"
N_REPEATS = 20
N_JOBS = min(8, os.cpu_count() or 1)
SPECS = [  # (label, block, learner)
    ("B0 mean", None, "mean"),
    ("B1 body + bat", "B1", "ols"),
    ("B2 + age, level", "B2", "ols"),
    ("B3 + pelvis/trunk (enet)", "B3", "enet"),
    ("B3 + pelvis/trunk (gbm)", "B3", "gbm"),
    ("B4 + arm/hand (enet)", "B4", "enet"),
    ("B4 + arm/hand (gbm)", "B4", "gbm"),
]
GAINS = [  # (label, model, reference)
    ("body + bat over mean", "B1 body + bat", "B0 mean"),
    ("age, level over body + bat", "B2 + age, level", "B1 body + bat"),
    ("pelvis/trunk (enet) over B2", "B3 + pelvis/trunk (enet)", "B2 + age, level"),
    ("pelvis/trunk (gbm) over B2", "B3 + pelvis/trunk (gbm)", "B2 + age, level"),
    ("arm/hand (enet) over B3 enet", "B4 + arm/hand (enet)", "B3 + pelvis/trunk (enet)"),
    ("arm/hand (gbm) over B3 gbm", "B4 + arm/hand (gbm)", "B3 + pelvis/trunk (gbm)"),
    ("gbm minus enet at B3", "B3 + pelvis/trunk (gbm)", "B3 + pelvis/trunk (enet)"),
    ("gbm minus enet at B4", "B4 + arm/hand (gbm)", "B4 + arm/hand (enet)"),
]
TAUS = [0.5, 0.75, 0.9]


def one_repeat(df, blocks, scheme, seed):
    preds = {label: models.oof_predictions(df, blocks[b] if b else [], kind, data.OUTCOME,
                                           data.GROUP, seed=seed, scheme=scheme)
             for label, b, kind in SPECS}
    y = df[data.OUTCOME].to_numpy()
    rows = [{"scheme": scheme, "seed": seed, "model": k, "rmse": models.rmse(y, v),
             "r2": models.r2_vs(y, v, preds["B0 mean"])} for k, v in preds.items()]
    return rows, (preds if seed == 0 else None)


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = data.load()
    blocks = data.blocks()
    results: dict = {"n_swings": len(df), "n_hitters": int(df[data.GROUP].nunique()),
                     "block_sizes": {k: len(v) for k, v in blocks.items()}, "n_jobs": N_JOBS}
    with open(OUT / "feature_blocks.json", "w") as f:
        json.dump(blocks, f, indent=2)

    print("Q1 ICC", flush=True)
    results["q1_icc"] = {o: stats.icc(df, o, data.GROUP) for o in (data.OUTCOME, data.EV)}

    print(f"Q2 cross-validation on {N_JOBS} workers", flush=True)
    jobs = [(scheme, s) for scheme in ("group", "random") for s in range(N_REPEATS)]
    out = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(one_repeat)(df, blocks, sc, s) for sc, s in jobs)
    rows = [r for rs, _ in out for r in rs]
    oof = next(p for (rs, p), (sc, s) in zip(out, jobs, strict=True) if sc == "group" and s == 0)
    cv = pd.DataFrame(rows)
    cv.to_csv(OUT / "cv_repeats.csv", index=False)
    grouped, random_cv = cv[cv.scheme == "group"], cv[cv.scheme == "random"]

    y = df[data.OUTCOME].to_numpy()
    boot_df = pd.DataFrame({"g": df[data.GROUP].to_numpy(), "y": y, **oof})

    def stat_fn(sub):
        return {f"{m}|{label}": (models.rmse(sub["y"], sub[label]) if m == "rmse"
                                 else models.r2_vs(sub["y"], sub[label], sub["B0 mean"]))
                for label, _, _ in SPECS for m in ("rmse", "r2")}

    boot = stats.cluster_bootstrap(boot_df, "g", stat_fn, n_boot=2000, seed=0)
    point = stat_fn(boot_df)
    table = []
    for label, _, _ in SPECS:
        gm = grouped[grouped.model == label][["rmse", "r2"]]
        rm = random_cv[random_cv.model == label][["rmse", "r2"]].mean()
        pm = boot_df.groupby("g")[list(dict.fromkeys(["y", label, "B0 mean"]))].mean()
        table.append({
            "model": label, "rmse_grouped": gm["rmse"].mean(), "r2_grouped": gm["r2"].mean(),
            "r2_grouped_sd": gm["r2"].std(), "r2_seed0": point[f"r2|{label}"],
            "r2_ci_low": np.percentile(boot[f"r2|{label}"], 2.5),
            "r2_ci_high": np.percentile(boot[f"r2|{label}"], 97.5),
            "rmse_ci_low": np.percentile(boot[f"rmse|{label}"], 2.5),
            "rmse_ci_high": np.percentile(boot[f"rmse|{label}"], 97.5),
            "r2_hitter_level": models.r2_vs(pm["y"], pm[label], pm["B0 mean"]),
            "r2_random": rm["r2"], "leakage_inflation_r2": rm["r2"] - gm["r2"].mean()})
    pd.DataFrame(table).to_csv(OUT / "model_table.csv", index=False)
    gains = []
    for label, m, ref in GAINS:
        d = boot[f"r2|{m}"] - boot[f"r2|{ref}"]
        gains.append({"comparison": label, "gain_r2": point[f"r2|{m}"] - point[f"r2|{ref}"],
                      "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5),
                      "adds_information": bool(np.percentile(d, 2.5) > 0)})
    pd.DataFrame(gains).to_csv(OUT / "block_gains.csv", index=False)
    boot_df.to_csv(OUT / "oof_predictions_seed0.csv", index=False)

    print("Q3 exit velocity", flush=True)
    ev = stats.add_within_between(df.dropna(subset=[data.EV]), [data.OUTCOME, "attack_angle_contact_x"],
                                  data.GROUP)
    terms = [f"{data.OUTCOME}_between", f"{data.OUTCOME}_within", "attack_angle_contact_x_within",
             "attack_angle_contact_x_between", "mass_kg", "lefty"]
    res_a = stats.fit_mixed(f"{data.EV} ~ " + " + ".join(terms), ev, data.GROUP)
    stats.mixed_table(res_a, terms).to_csv(OUT / "q3a_ev_mixed.csv", index=False)
    evp = ev.dropna(subset=["pitch_speed_mph"])
    res_b = stats.fit_mixed(f"{data.EV} ~ " + " + ".join(terms + ["pitch_speed_mph"]), evp, data.GROUP)
    stats.mixed_table(res_b, terms + ["pitch_speed_mph"]).to_csv(OUT / "q3b_ev_mixed_pitch.csv", index=False)
    results["q3_n"] = {"q3a": int(len(ev)), "q3b": int(len(evp))}

    qd = evp[[data.EV, "sweet_spot_speed_mph", "pitch_speed_mph", data.GROUP]]
    qrows = []
    for tau in TAUS:
        est = stats.collision_q(qd[data.EV], qd["sweet_spot_speed_mph"], qd["pitch_speed_mph"], tau)
        bs = stats.cluster_bootstrap(qd, data.GROUP, lambda s, t=tau: stats.collision_q(
            s[data.EV], s["sweet_spot_speed_mph"], s["pitch_speed_mph"], t), n_boot=2000, seed=0)
        qrows.append({"setting": "lab", "tau": tau, "q": est, "ci_low": np.percentile(bs, 2.5),
                      "ci_high": np.percentile(bs, 97.5), "n": int(len(qd)),
                      "clusters": int(qd[data.GROUP].nunique())})
    pd.DataFrame(qrows).to_csv(OUT / "q3c_collision_q_lab.csv", index=False)

    results["runtime_s"] = round(time.time() - t0, 1)
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(json.dumps(results, indent=2, default=float))
    print(pd.DataFrame(table).round(3).to_string())
    print(pd.DataFrame(gains).round(3).to_string())
    print(stats.mixed_table(res_a, terms).round(3).to_string())
    print(pd.DataFrame(qrows).round(3).to_string())


if __name__ == "__main__":
    main()
