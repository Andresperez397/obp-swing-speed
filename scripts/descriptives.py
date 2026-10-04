"""Descriptive numbers quoted in the README that are not model outputs.

Writes reports/tables/descriptives.json: predictor counts per block, the within-hitter agreement
between bat-speed measures (same system and an independent sensor), and the lab pitch-speed range.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swing import data  # noqa: E402


def main() -> None:
    df = data.load()
    roles = Counter(data.column_roles().values())
    g = df.groupby(data.GROUP)
    within_max = df[data.OUTCOME] - g[data.OUTCOME].transform("mean")
    within_ss = df["sweet_spot_speed_mph"] - g["sweet_spot_speed_mph"].transform("mean")
    # Independent device: the Blast Motion sensor (shares no markers with motion capture).
    within_blast = df["blast_bat_speed_mph_x"] - g["blast_bat_speed_mph_x"].transform("mean")
    out = {
        "column_roles": dict(roles),
        "trunk_predictors": roles["trunk"], "arm_predictors": roles["arm"],
        "within_hitter_r_max_vs_sweet_spot_speed": round(float(within_max.corr(within_ss)), 3),
        "within_hitter_r_max_vs_blast_sensor": round(float(within_max.corr(within_blast)), 3),
        "within_hitter_sd_max_bat_speed": round(float(within_max.std()), 2),
        "lab_pitch_speed_valid": df["pitch_speed_mph"].describe().round(2).to_dict(),
        "sweet_spot_speed": df["sweet_spot_speed_mph"].describe().round(2).to_dict(),
    }
    path = ROOT / "reports" / "tables" / "descriptives.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
