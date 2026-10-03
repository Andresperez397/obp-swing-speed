"""Mixed models, collision efficiency and bootstrap intervals (ANALYSIS_PLAN.md sections 4-6)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg


def fit_mixed(formula: str, df: pd.DataFrame, group: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(formula, df, groups=df[group])
        for method in ("lbfgs", "powell", "nm"):
            try:
                res = model.fit(reml=True, method=method)
                if res.converged:
                    return res
            except Exception:  # noqa: BLE001 - try the next optimizer
                continue
    raise RuntimeError(f"mixed model did not converge: {formula}")


def icc(df: pd.DataFrame, outcome: str, group: str, n_boot: int = 1000, seed: int = 0) -> dict:
    d = df[[outcome, group]].dropna()
    res = fit_mixed(f"{outcome} ~ 1", d, group)
    tau2, sigma2 = float(res.cov_re.iloc[0, 0]), float(res.scale)
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(d[group].to_numpy(), return_inverse=True)
    mu = float(res.fe_params.iloc[0])
    sim = d[[group]].copy()
    boots = []
    for _ in range(n_boot):
        sim["y"] = mu + rng.normal(0, np.sqrt(tau2), len(uniq))[inv] + rng.normal(0, np.sqrt(sigma2), len(d))
        try:
            r = fit_mixed("y ~ 1", sim, group)
        except RuntimeError:
            continue
        t, s = float(r.cov_re.iloc[0, 0]), float(r.scale)
        boots.append(t / (t + s))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"icc": tau2 / (tau2 + sigma2), "ci_low": float(lo), "ci_high": float(hi),
            "between_sd": float(np.sqrt(tau2)), "within_sd": float(np.sqrt(sigma2)),
            "n": int(len(d)), "n_boot_ok": len(boots)}


def add_within_between(df: pd.DataFrame, cols: list[str], group: str) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        d[f"{c}_between"] = d.groupby(group)[c].transform("mean")
        d[f"{c}_within"] = d[c] - d[f"{c}_between"]
    return d


def mixed_table(res, terms: list[str]) -> pd.DataFrame:
    ci = res.conf_int()
    return pd.DataFrame([{"term": t, "estimate": res.params[t], "ci_low": ci.loc[t, 0],
                          "ci_high": ci.loc[t, 1], "p": res.pvalues[t]} for t in terms])


def collision_q(ev, v_bat, v_pitch, tau: float) -> float:
    """No-intercept quantile regression of (EV - v_bat) on (v_pitch + v_bat); slope = q."""
    y = np.asarray(ev, dtype=float) - np.asarray(v_bat, dtype=float)
    x = (np.asarray(v_pitch, dtype=float) + np.asarray(v_bat, dtype=float)).reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(QuantReg(y, x).fit(q=tau, max_iter=5000).params[0])


def cluster_bootstrap(df: pd.DataFrame, group: str, stat_fn, n_boot: int = 2000, seed: int = 0):
    """Resample whole clusters with replacement; stat_fn(sub_df) -> float or dict of floats."""
    rng = np.random.default_rng(seed)
    gvals = df[group].to_numpy()
    idx_by_g = {g: np.flatnonzero(gvals == g) for g in pd.unique(gvals)}
    keys = np.array(list(idx_by_g))
    out = []
    for _ in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        out.append(stat_fn(df.iloc[np.concatenate([idx_by_g[k] for k in pick])]))
    return pd.DataFrame(out) if isinstance(out[0], dict) else pd.Series(out)
