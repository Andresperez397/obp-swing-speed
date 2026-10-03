"""EXPLORATORY, pre-specified (ANALYSIS_PLAN.md section 6): collision efficiency q in MLB vs the lab.

MLB: 2025 regular-season balls in play with Statcast bat tracking, bunts excluded. Pitch speed at
the plate is approximated as 0.92 x release speed (sensitivity: 0.90 and 1.00, point estimates).
q is the slope of a no-intercept quantile regression of (EV - bat speed) on (pitch + bat speed).
Intervals: batter-cluster bootstrap (1,000) for MLB; hitter-cluster bootstrap (1,000, same seed)
for the lab, so the lab-minus-MLB difference has a bootstrap interval.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swing import data, stats  # noqa: E402

OUT = ROOT / "reports" / "tables"
TAUS = [0.5, 0.75, 0.9]
PLATE_FACTOR = 0.92
SENSITIVITY = [0.90, 1.00]
N_BOOT = 1000


def mlb_balls_in_play() -> tuple[pd.DataFrame, dict]:
    sc = pd.read_parquet(ROOT / "data" / "raw" / "statcast_2025.parquet")
    log = {"pitches": len(sc), "with_bat_speed": int(sc["bat_speed"].notna().sum())}
    bip = sc[sc["description"] == "hit_into_play"]
    log["balls_in_play"] = len(bip)
    is_bunt = bip["events"].fillna("").str.contains("bunt")
    log["bunts_excluded"] = int(is_bunt.sum())
    bip = bip[~is_bunt].dropna(subset=["bat_speed", "launch_speed", "release_speed"])
    log["analysis_rows"] = len(bip)
    log["batters"] = int(bip["batter"].nunique())
    return bip, log


def boot_q(d, ev, bat, pitch, group, tau, seed):
    return stats.cluster_bootstrap(d, group, lambda s: stats.collision_q(s[ev], s[bat], s[pitch], tau),
                                   n_boot=N_BOOT, seed=seed).to_numpy()


def main() -> None:
    bip, log = mlb_balls_in_play()
    bip = bip.assign(v_pitch=PLATE_FACTOR * bip["release_speed"])
    lab = data.load().dropna(subset=[data.EV, "pitch_speed_mph"])
    lab = lab[[data.EV, "sweet_spot_speed_mph", "pitch_speed_mph", data.GROUP]]

    rows, diffs = [], []
    tasks = [(tau, "mlb") for tau in TAUS] + [(tau, "lab") for tau in TAUS]

    def run(tau, which):
        if which == "mlb":
            return boot_q(bip, "launch_speed", "bat_speed", "v_pitch", "batter", tau, seed=0)
        return boot_q(lab, data.EV, "sweet_spot_speed_mph", "pitch_speed_mph", data.GROUP, tau, seed=0)

    boots = Parallel(n_jobs=min(6, os.cpu_count() or 1))(delayed(run)(t, w) for t, w in tasks)
    b = dict(zip(tasks, boots, strict=True))
    for tau in TAUS:
        q_mlb = stats.collision_q(bip["launch_speed"], bip["bat_speed"], bip["v_pitch"], tau)
        q_lab = stats.collision_q(lab[data.EV], lab["sweet_spot_speed_mph"], lab["pitch_speed_mph"], tau)
        for setting, q, bs in (("mlb", q_mlb, b[(tau, "mlb")]), ("lab", q_lab, b[(tau, "lab")])):
            rows.append({"setting": setting, "tau": tau, "q": q, "ci_low": np.percentile(bs, 2.5),
                         "ci_high": np.percentile(bs, 97.5),
                         "n": len(bip) if setting == "mlb" else len(lab)})
        d = b[(tau, "lab")] - b[(tau, "mlb")]
        diffs.append({"tau": tau, "lab_minus_mlb": q_lab - q_mlb, "ci_low": np.percentile(d, 2.5),
                      "ci_high": np.percentile(d, 97.5)})
    sens = [{"plate_factor": f, "tau": tau,
             "q_mlb": stats.collision_q(bip["launch_speed"], bip["bat_speed"], f * bip["release_speed"], tau)}
            for f in SENSITIVITY for tau in TAUS]

    desc = {"mlb_bat_speed": bip["bat_speed"].describe().round(2).to_dict(),
            "mlb_launch_speed": bip["launch_speed"].describe().round(2).to_dict(),
            "mlb_release_speed": bip["release_speed"].describe().round(2).to_dict(),
            "lab_sweet_spot_speed": lab["sweet_spot_speed_mph"].describe().round(2).to_dict(),
            "lab_exit_velo": lab[data.EV].describe().round(2).to_dict()}
    pd.DataFrame(rows).to_csv(OUT / "q4_collision_q_lab_vs_mlb.csv", index=False)
    pd.DataFrame(diffs).to_csv(OUT / "q4_lab_minus_mlb.csv", index=False)
    pd.DataFrame(sens).to_csv(OUT / "q4_mlb_plate_speed_sensitivity.csv", index=False)
    with open(OUT / "q4_summary.json", "w") as f:
        json.dump({"mlb_filtering": log, "descriptives": desc, "plate_factor": PLATE_FACTOR,
                   "n_boot": N_BOOT}, f, indent=2, default=float)
    print(json.dumps(log, indent=2))
    print(pd.DataFrame(rows).round(3).to_string())
    print(pd.DataFrame(diffs).round(3).to_string())
    print(pd.DataFrame(sens).round(3).to_string())


if __name__ == "__main__":
    main()
