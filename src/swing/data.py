"""Load and join the OBP hitting files; define the cleaned variables and predictor blocks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

OUTCOME = "bat_speed_mph_max_x"
EV = "exit_velo_mph_x"
GROUP = "hitter_id"
ID_COLS = ["session_swing", "session"]

# Bat or ball measurements: never predictors of bat speed (ANALYSIS_PLAN.md section 3).
LEAKAGE = [
    "bat_speed_mph_contact_x", "blast_bat_speed_mph_x",
    "sweet_spot_velo_mph_contact_x", "sweet_spot_velo_mph_contact_y", "sweet_spot_velo_mph_contact_z",
    "bat_speed_xy_max_x", "bat_max_x", "bat_min_x", "attack_angle_contact_x", EV,
]

# One variable kept per duplicate / near-duplicate pair (|r| > 0.995; DATA_AUDIT.md item 2).
DROP_DUPLICATES = [
    "bat_torso_angle_connection_x",            # == bat_torso_angle_ds_y
    "hand_speed_mag_seq_max_x",                # == hand_speed_mag_max_x
    "pelvis_angle_fm_x", "pelvis_angle_fm_y", "pelvis_angle_fm_z",   # K-Vest copies of pelvis_fm_*
    "pelvis_angle_fp_x", "pelvis_angle_fp_y", "pelvis_angle_fp_z",   # ... of pelvis_launchpos_*
    "torso_angle_fm_x", "torso_angle_fm_y", "torso_angle_fm_z",      # ... of torso_fm_*
    "torso_angle_fp_x", "torso_angle_fp_y", "torso_angle_fp_z",      # ... of torso_launchpos_*
    "upper_arm_speed_mag_seq_max_x", "upper_arm_speed_mag_swing_max_velo_x",  # == ..._max_x
    "x_factor_fm_z", "x_factor_fp_z",          # == torso_pelvis_fm_x / torso_pelvis_launchpos_x
    "pelvis_angular_velocity_swing_max_x",     # == pelvis_angular_velocity_seq_max_x
    "torso_angular_velocity_swing_max_x",      # == torso_angular_velocity_seq_max_x
    "torso_angular_velocity_stride_max_x",     # r = 0.998 with torso_angular_velocity_fp_x
    "upper_arm_speed_mag_stride_max_velo_x",   # r = 0.998 with upper_arm_speed_mag_fp_x
]

TRUNK_PREFIXES = ("pelvis_", "torso_", "x_factor_", "lead_knee_", "rear_hip_", "max_cog_velo")
ARM_PREFIXES = ("upper_arm_speed_", "hand_speed_", "lead_wrist_", "rear_elbow_", "rear_shoulder_",
                "bat_torso_angle_ds_")
BODY = ["mass_kg", "height_m", "bat_weight_oz", "bat_length_in", "lefty"]
LEVELS = ["high_school", "independent", "milb"]  # reference level: college


def load(raw_dir: Path = RAW) -> pd.DataFrame:
    poi = pd.read_csv(raw_dir / "poi_metrics.csv")
    meta = pd.read_csv(raw_dir / "metadata.csv")
    trax = pd.read_csv(raw_dir / "hittrax.csv")
    meta = meta.rename(columns={"user": GROUP})
    keep = ["session_swing", GROUP, "session_mass_lbs", "session_height_in", "athlete_age",
            "highest_playing_level", "hitter_side", "bat_weight_oz", "bat_length_in"]
    df = poi.merge(meta[keep], on="session_swing", how="inner", validate="one_to_one")
    if len(df) != len(poi):
        raise ValueError(f"join dropped rows: {len(poi)} -> {len(df)}")
    df = df.merge(trax.rename(columns={"pitch": "pitch_speed_mph", "la": "launch_angle"})[
        ["session_swing", "pitch_speed_mph", "launch_angle"]], on="session_swing", how="left",
        validate="one_to_one")
    df.loc[df["pitch_speed_mph"] <= 0, "pitch_speed_mph"] = np.nan  # DATA_AUDIT.md item 4
    derived = {
        "mass_kg": df["session_mass_lbs"] * 0.45359237,
        "height_m": df["session_height_in"] * 0.0254,
        "lefty": (df["hitter_side"] == "L").astype(int),
        **{f"level_{lvl}": (df["highest_playing_level"] == lvl).astype(int) for lvl in LEVELS},
        "sweet_spot_speed_mph": np.sqrt(df["sweet_spot_velo_mph_contact_x"] ** 2
                                        + df["sweet_spot_velo_mph_contact_y"] ** 2
                                        + df["sweet_spot_velo_mph_contact_z"] ** 2),
    }
    df = pd.concat([df, pd.DataFrame(derived, index=df.index)], axis=1)
    return df


def poi_columns() -> list[str]:
    return list(pd.read_csv(RAW / "poi_metrics.csv", nrows=1).columns)


def trunk_columns(cols: list[str]) -> list[str]:
    return [c for c in cols if c.startswith(TRUNK_PREFIXES) and c not in DROP_DUPLICATES
            and c not in LEAKAGE]


def arm_columns(cols: list[str]) -> list[str]:
    return [c for c in cols if c.startswith(ARM_PREFIXES) and c not in DROP_DUPLICATES
            and c not in LEAKAGE]


def blocks(cols: list[str] | None = None) -> dict[str, list[str]]:
    cols = cols or poi_columns()
    b1 = BODY
    b2 = b1 + ["athlete_age"] + [f"level_{lvl}" for lvl in LEVELS]
    b3 = b2 + trunk_columns(cols)
    b4 = b3 + arm_columns(cols)
    return {"B1": b1, "B2": b2, "B3": b3, "B4": b4}


def column_roles(cols: list[str] | None = None) -> dict[str, str]:
    """Every POI column mapped to exactly one role (checked by a test)."""
    cols = cols or poi_columns()
    roles = {}
    for c in cols:
        if c in ID_COLS:
            roles[c] = "id"
        elif c == OUTCOME:
            roles[c] = "outcome"
        elif c in LEAKAGE:
            roles[c] = "leakage"
        elif c in DROP_DUPLICATES:
            roles[c] = "duplicate"
        elif c.startswith(TRUNK_PREFIXES):
            roles[c] = "trunk"
        elif c.startswith(ARM_PREFIXES):
            roles[c] = "arm"
        else:
            roles[c] = "UNASSIGNED"
    return roles
