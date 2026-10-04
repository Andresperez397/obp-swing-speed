# Deviations from ANALYSIS_PLAN.md

Every change made after the plan was frozen is logged here, dated, with its reason.

## Analysis

No pre-specified analysis was changed, dropped or added. All five Q2 models, both outcomes in Q1, Q3a–c and Q4 were run exactly as written.

## Implementation notes (no effect on results)

- **2026-10-03:** the first run of `run_analysis.py` was interrupted when the working session restarted. It was rerun from the start with identical code. Results are written only at the end of a run, so no partial output was used.
- **2026-10-03:** `run_mlb_comparison.py` recomputes the lab bootstrap with 1,000 hitter resamples (seed 0), matching the 1,000 MLB batter resamples, so the lab-minus-MLB difference has a paired bootstrap interval. The lab-only Q3c table uses 2,000 resamples, as the plan states. Both give the same point estimates; the intervals differ only in the third decimal.

## Differences from the portfolio-level plan (written before the data audit)

- The portfolio plan listed a "ground reaction forces" block. The data audit found no force-plate variables in the hitting summary files (they exist only in the raw full-signal archives), so this frozen analysis plan never included one.

## Final review corrections (2026-10-03; interpretation only, no analysis changes)

- **Within-hitter slope:** the README had said the flatter within-hitter slope "isn't explained by measurement noise", citing r = 0.89 between two bat-speed measures. Both measures come from the same motion-capture markers, so they share error. Against an independent device (Blast sensor), within-hitter agreement is only r = 0.28 (`scripts/descriptives.py`). The README now says the data can't separate contact quality from measurement noise, and that noise would flatten the slope.
- **Lab vs MLB efficiency gap:** "lab hitters square the ball up more consistently" is now presented as a plausible reading, not a finding.
- **Reference value:** the collision-efficiency chart had shaded 0.20–0.25 as "typical (Nathan 2003)". The paper gives q ≈ 0.2 for a typical sweet-spot collision, so the chart now shows a single reference line at 0.2.
