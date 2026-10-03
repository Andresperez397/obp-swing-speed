# Deviations from ANALYSIS_PLAN.md

Every change made after the plan was frozen is logged here, dated, with its reason.

## Analysis

No pre-specified analysis was changed, dropped or added. All five Q2 models, both outcomes in Q1, Q3a–c and Q4 were run exactly as written.

## Implementation notes (no effect on results)

- **2026-10-03:** the first run of `run_analysis.py` was interrupted when the working session restarted. It was rerun from the start with identical code. Results are written only at the end of a run, so no partial output was used.
- **2026-10-03:** `run_mlb_comparison.py` recomputes the lab bootstrap with 1,000 hitter resamples (seed 0), matching the 1,000 MLB batter resamples, so the lab-minus-MLB difference has a paired bootstrap interval. The lab-only Q3c table uses 2,000 resamples, as the plan states. Both give the same point estimates; the intervals differ only in the third decimal.

## Differences from the portfolio-level plan (written before the data audit)

- The portfolio plan listed a "ground reaction forces" block. The data audit found no force-plate variables in the hitting summary files (they exist only in the raw full-signal archives), so this frozen analysis plan never included one.
