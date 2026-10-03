"""Data audit, run BEFORE any outcome modeling. Writes reports/tables/data_audit.json.

Checks: linkage across files, one session per hitter, missingness, value ranges, exact and
near-duplicate variables (|r| > 0.995), HitTrax pitch-speed validity, and agreement between the
two bat-speed sources. Only descriptive statistics; no model is fit here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def main() -> None:
    m = pd.read_csv(RAW / "metadata.csv")
    p = pd.read_csv(RAW / "poi_metrics.csv")
    h = pd.read_csv(RAW / "hittrax.csv")
    out: dict = {}

    out["rows"] = {"metadata": len(m), "poi": len(p), "hittrax": len(h)}
    out["hitters"] = int(m["user"].nunique())
    out["swings_per_hitter"] = m.groupby("user").size().describe().round(2).to_dict()
    out["one_session_per_hitter"] = bool((m.groupby("user")["session"].nunique() == 1).all())
    out["linkage"] = {
        "poi_in_metadata": float(p["session_swing"].isin(m["session_swing"]).mean()),
        "hittrax_in_metadata": float(h["session_swing"].isin(m["session_swing"]).mean()),
        "unique_swing_ids": bool(m["session_swing"].is_unique and p["session_swing"].is_unique),
    }
    out["playing_level_swings"] = m["highest_playing_level"].value_counts().to_dict()
    out["playing_level_hitters"] = m.groupby("user")["highest_playing_level"].first().value_counts().to_dict()
    out["hitter_side_hitters"] = m.groupby("user")["hitter_side"].first().value_counts().to_dict()

    miss = p.isna().mean()
    out["poi_missing_share"] = {k: round(float(v), 4) for k, v in miss[miss > 0].items()}
    out["force_plate_columns_in_poi"] = [c for c in p.columns
                                         if any(k in c.lower() for k in ("grf", "force", "rfd"))]

    num = p.select_dtypes("number").drop(columns=["session"])
    corr = num.corr().abs()
    dups = []
    cols = list(num.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if corr.loc[a, b] > 0.995:
                dups.append({"a": a, "b": b, "abs_r": round(float(corr.loc[a, b]), 4),
                             "identical": bool(np.allclose(num[a], num[b], equal_nan=True))})
    out["near_duplicate_pairs"] = dups

    key = ["bat_speed_mph_max_x", "bat_speed_mph_contact_x", "blast_bat_speed_mph_x",
           "exit_velo_mph_x", "hand_speed_mag_max_x", "hand_speed_blast_bat_mph_max_x",
           "attack_angle_contact_x"]
    out["ranges"] = p[key].describe().T[["count", "mean", "std", "min", "max"]].round(2).to_dict(orient="index")
    out["metadata_ranges"] = m[["session_mass_lbs", "session_height_in", "athlete_age",
                                "bat_weight_oz", "bat_length_in"]].describe().T[
        ["mean", "min", "max"]].round(2).to_dict(orient="index")

    zero = int((h["pitch"] <= 0).sum())
    valid = h.loc[h["pitch"] > 0, "pitch"]
    out["hittrax_pitch_speed"] = {"zero_or_negative": zero, "valid_n": int(valid.size),
                                  "median": round(float(valid.median()), 1),
                                  "p05": round(float(valid.quantile(0.05)), 1),
                                  "p95": round(float(valid.quantile(0.95)), 1)}
    both = p[["blast_bat_speed_mph_x", "bat_speed_mph_contact_x"]].dropna()
    out["blast_vs_marker_bat_speed_r"] = round(float(both.corr().iloc[0, 1]), 3)

    path = ROOT / "reports" / "tables" / "data_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps({k: out[k] for k in ("rows", "hitters", "linkage", "poi_missing_share",
                                          "hittrax_pitch_speed", "blast_vs_marker_bat_speed_r",
                                          "playing_level_hitters", "hitter_side_hitters")},
                     indent=2))
    print("near-duplicate pairs:", len(dups))


if __name__ == "__main__":
    main()
