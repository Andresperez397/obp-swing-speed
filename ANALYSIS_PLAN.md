# Analysis Plan (frozen before any outcome modeling)

**Project:** Where does bat speed come from, and how much of it becomes exit velocity?
**Data:** OpenBiomechanics Project hitting point-of-interest metrics (Driveline Baseball, CC BY-NC-SA 4.0), pinned to commit `44b98da`; public 2025 Statcast bat tracking for the exploratory comparison.
**Author:** Andres Perez
**Written:** 2026-10-03. This plan is committed to git before any outcome model is fit.

**Before this plan (disclosed):** a data audit (`DATA_AUDIT.md`), plus a single descriptive check during planning: the swing-level correlation between exit velocity and bat speed is r = 0.71. No model was fit.

## 1. Questions

- **Q1. Where does the variation live?** What share of the variance in bat speed and in exit velocity lies between hitters versus swing to swing?
- **Q2. What predicts bat speed for a hitter the model has never seen?**
  - The model is built up in blocks: body and bat size → age and playing level → pelvis and trunk kinematics → arm and hand kinematics.
  - For each block: how much does it add, and do flexible models beat linear ones once both are tuned fairly?
- **Q3. How much of exit velocity is bat speed?**
  - Within-hitter versus between-hitter effects of bat speed on exit velocity.
  - The bat-ball collision efficiency *q* in the lab.
- **Q4 (exploratory, pre-specified). Does the lab relationship hold in MLB?** The collision efficiency estimated from 2025 MLB balls in play with Statcast bat tracking, compared with the lab's.

## 2. Units, outcomes, cleaning

- **Unit:** the swing, nested in the hitter (98 hitters, 677 swings).
- **Primary outcome:** `bat_speed_mph_max_x`, the marker-based maximum bat speed.
- **Secondary outcome:** `exit_velo_mph_x`. The 5 swings missing exit velocity are dropped from exit-velocity analyses.
- **Duplicates:** the 23 duplicate or near-duplicate variable pairs are reduced to one variable each, using the fixed list in `src/swing/data.py`.
- **HitTrax pitch speed:** values of 0 mph or less are set to missing.
- **Sweet-spot speed at contact:** the magnitude of `sweet_spot_velo_mph_contact_x/y/z`, used only in Q3c and Q4.

## 3. Q2: predictor blocks for bat speed

**Never predictors (bat or ball measurements):**
- `bat_speed_mph_contact_x`
- `blast_bat_speed_mph_x`
- `sweet_spot_velo_mph_contact_*`
- `bat_speed_xy_max_x`
- `bat_max_x`, `bat_min_x`
- `attack_angle_contact_x`
- `exit_velo_mph_x`

| Block | Adds | Learner |
|---|---|---|
| B0 | none (training-fold mean) | mean |
| B1 | body mass, height, bat weight, bat length, handedness | OLS |
| B2 | B1 + age + playing level (indicator variables) | OLS |
| B3 | B2 + pelvis and trunk kinematics: pelvis and torso angles, angular velocities, x-factor and torso–pelvis separation, lead-knee and rear-hip angles, centre-of-gravity velocity | elastic net; gradient-boosted trees |
| B4 | B3 + arm and hand kinematics: upper-arm and hand speeds, lead-wrist, rear-elbow and rear-shoulder angles, bat–torso connection angle at downswing | elastic net; gradient-boosted trees |

**Why B4 is interpreted with care:** the hands hold the bat, so hand speed is mechanically close to bat speed. B4's gain shows how much of bat speed is already present at the hands. B3 is the main answer to "does the body generate it?".

**Fair tuning, applied to every flexible model:**
- **Elastic net:** penalty and mix chosen by inner GroupKFold (5 folds, by hitter) within each training fold.
- **Gradient-boosted trees:** `HistGradientBoostingRegressor` with `early_stopping=False`. Depth {2, 3}, learning rate {0.03, 0.1}, rounds {100, 300} and minimum leaf size {10, 30} are chosen by the same inner GroupKFold.
- **Preprocessing:** median imputation, winsorizing at the 1st and 99th percentiles, and standardizing, all fit within training folds only.

## 4. Validation and uncertainty

- **Primary scheme:** GroupKFold by hitter, 10 folds, repeated 20 times with random hitter-to-fold assignment. Metrics are out-of-fold RMSE (mph) and R² relative to the training-fold mean, averaged over repeats.
- **Intervals:** a hitter-cluster bootstrap (2,000 resamples) on the out-of-fold predictions of the first repeat. Predictions are held fixed, so intervals reflect which hitters were sampled.
- **Decision rule:** a block "adds information" only if the 95% interval of its R² gain excludes zero.
- **Leakage comparison:** the same models under ordinary 10-fold KFold over swings. The difference in R² is reported as inflation.

## 5. Q1 and Q3: mixed models

**Q1:** a random-intercept model `y ~ 1 + (1 | hitter)` fit by REML for bat speed and for exit velocity. ICC with a parametric-bootstrap 95% CI (1,000 resamples).

**Q3a:** `EV ~ bat_speed_between + bat_speed_within + attack_angle_within + attack_angle_between + mass + lefty + (1 | hitter)`, where "between" is the hitter mean and "within" is the deviation from it. Wald 95% CIs, REML.

**Q3b, sensitivity:** Q3a with HitTrax pitch speed added, on swings with a valid pitch speed.

**Q3c, collision efficiency *q*:**
- **Physics:** for a squared-up collision, EV = q·v_pitch + (1 + q)·v_bat, so EV − v_bat = q·(v_pitch + v_bat).
- **Lab estimate:** fit *q* by quantile regression with no intercept on that relation at τ = 0.90, as a stand-in for squared-up contact. Uses sweet-spot speed at contact as v_bat and HitTrax pitch speed as v_pitch.
- **Also reported:** τ = 0.50 and τ = 0.75 for context.
- **Interval:** 95% CI from a hitter-cluster bootstrap (2,000 resamples).

## 6. Q4: exploratory MLB comparison (pre-specified)

- **Data:** 2025 regular-season MLB balls in play (`description == "hit_into_play"`) with non-missing bat speed, launch speed and release speed. Bunts are excluded.
- **Pitch speed at the plate:** approximated as 0.92 × release speed. Sensitivity analyses use 0.90 and 1.00. This conversion is an approximation and is labelled as one.
- **Model:** the same no-intercept quantile regression of (EV − bat speed) on (v_pitch + bat speed) at τ = 0.90, 0.75 and 0.50.
- **Interval:** batter-cluster bootstrap (1,000 resamples).
- **Reported:** q_MLB, q_lab, and their difference with a 95% interval.
- **Status:** exploratory. Differences can come from bat material (unknown for the lab), pitch speed (about 59 vs about 90 mph), measurement systems and population. They are discussed, not explained away.

## 7. Pre-declared interpretation rules

- All five Q2 models are reported, whatever the results; the best is not chosen by test performance.
- Flexible models are tuned with the same inner-grouped effort as the elastic net (§3).
- A trivial baseline (B0) is always shown.
- Every post-plan change goes in `DEVIATIONS.md` with a date and reason.

## 8. Known limitations (stated in advance)

- **Sample:** mostly college hitters at one private facility; one session each; hitting off a machine at about 59 mph.
- **Bat speed:** bat speed and its kinematics come from marker-based tracking; the bat's material isn't documented.
- **Q4 scope:** exploratory only. Two different measurement systems and populations are compared.
- **Small sample:** with 98 hitters, R² gains smaller than about 0.1 may not be distinguishable from zero.
