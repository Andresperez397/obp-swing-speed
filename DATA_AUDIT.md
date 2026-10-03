# Data audit

Run before any outcome modeling. All numbers come from `scripts/data_audit.py`, which writes `reports/tables/data_audit.json`. Data: The OpenBiomechanics Project, baseball hitting, pinned to commit `44b98da` (hitting files last changed 2022-12-19).

## Coverage and linkage
- 677 swings from 98 hitters, one lab session each, 4–9 swings per hitter (median 8).
- **Hitters by level:** 74 college, 12 high school, 8 MiLB, 4 independent.
- **Handedness:** 71 right-handed and 27 left-handed hitters.
- `poi_metrics`, `metadata` and `hittrax` link one-to-one on `session_swing`. HitTrax covers 604 of the 677 swings. Swing IDs are unique.
- Swings were hit off a pitching machine (documented as about 65 mph from about 40 ft).

## Missingness
- `exit_velo_mph_x`: 5 swings (0.7%). The outcome is not imputed; those swings drop out of exit-velocity analyses.
- `blast_bat_speed_mph_x`: 30 swings (4.4%). Not used as an outcome (see below).
- No other point-of-interest metric has missing values.

## Problems found and how they're handled
1. **No force-plate variables in the summary files.**
   - The hitting point-of-interest file has no ground-reaction-force columns; they exist only in the large full-signal release archives.
   - The portfolio plan's "ground reaction forces" block is therefore not part of this project. Body and segment kinematics are analyzed instead.
2. **23 duplicate or near-duplicate variable pairs (|r| > 0.995).**
   - The same quantity appears under two naming conventions, for example K-Vest `pelvis_angle_fm_*` and `pelvis_fm_*`. Some pairs differ only by sign or offset.
   - Keeping both would double-count information and hide collinearity.
   - **Rule:** keep one variable per pair, using the fixed list in `src/swing/data.py`. The K-Vest copies, the `_seq_max`/`_swing_max` repeats and the `_stride_max` near-copies are dropped.
3. **Mislabeled units in the data dictionary.**
   - `hand_speed_mag_*` and `upper_arm_speed_mag_*` are angular speeds (around 1,800 and 1,200 °/s).
   - `hand_speed_blast_bat_mph_max_x` is a linear speed (around 21 mph), even though the dictionary says °/s.
   - Values are used as recorded, and units are stated from their magnitudes.
4. **Invalid HitTrax pitch speeds.**
   - 51 of 604 HitTrax swings (8%) record a pitch speed of 0 mph; these are set to missing.
   - Valid speeds have a median of 59.2 mph (5th–95th percentile 55.0–62.9), below the documented "about 65 mph".
5. **The two bat-speed sources disagree.**
   - The Blast Motion sensor and the marker-based bat speed at contact correlate only r = 0.72.
   - The primary outcome is the **marker-based maximum bat speed** (`bat_speed_mph_max_x`): it's complete and comes from the motion-capture system itself.

## Leakage-risk columns (never predictors of bat speed)
These are bat or ball measurements, not body mechanics:
- `bat_speed_mph_contact_x`
- `blast_bat_speed_mph_x`
- `sweet_spot_velo_mph_contact_x/y/z`
- `bat_speed_xy_max_x` (bat angular velocity)
- `bat_max_x` and `bat_min_x`
- `attack_angle_contact_x` (bat path at contact)
- `exit_velo_mph_x`

## Definitions checked for the physics comparison
- Statcast's public bat speed is measured at the bat's sweet spot. The lab's closest equivalent is the **sweet-spot speed at contact**, the magnitude of `sweet_spot_velo_mph_contact_x/y/z`, so that is used for the collision-efficiency comparison.
- Statcast `release_speed` is pitch speed at release, not at the plate. The plan states how this is handled.

## Statcast data (for the exploratory lab-to-MLB comparison)
- 2025 regular season, public Baseball Savant data, fetched one day per request.
- The download fails loudly on an error page or the 25,000-row cap.
- It is checked against MLB's official schedule.
- Columns used: bat speed, swing length, attack angle, launch speed and angle, pitch release speed, and outcome descriptions.
