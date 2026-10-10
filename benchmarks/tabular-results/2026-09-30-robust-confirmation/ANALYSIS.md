# Four-family robust incumbent confirmation

**The incumbent does not pass the frozen confirmation gates.** Gains are
mostly neutral, two thyroid classification losses exceed the 5% harm
threshold, and three candidate CatBoost cells time out. Better results
than some references on this small cohort do not establish broad SOTA.

All **192 cells were attempted** in **10.77356 active hours**. **177 scored
successfully**: raw and OpenFE each complete 48/48, FeatCopilot 45/48,
and AutoFeat 36/48. The 15 unscored cells are explicit 900-second
timeouts, not replaced by baseline losses or dropped from coverage.
There were no total-budget skips.

## Frozen scope and matched preprocessing

The four project-recorded unexposed observational origins were reserved
before this run: Facebook Cosmetics Posts (regression), Differentiated
Thyroid Recurrence (binary), Assam Medical Entrance and Personality Drug
Consumption (multiclass). Rights, schemas, retrospective timing and
exclusions are retained in the locked suite and release audit.

Every method uses full rows, three IID folds, seed 2028, the existing
three-setting learner grids, eight threads, a 900-second cell deadline and
4,096 MiB sampled RSS. FeatCopilot is the unchanged **single-fit,
frequency-aligned robust** incumbent: not the bagged or independent-gate
variants. Its candidate/width/repeat limits remain 1,000/50/3.

Native-width OpenFE and AutoFeat have the same final numeric-support caps
as the incumbent and raw learner. Their generation inputs and upstream
search remain unchanged, including declared training-median preprocessing.
Original numeric missingness is restored before the final learner fits
its caps/imputer. This does not claim that upstream discovery itself
uses FeatCopilot's clipping policy.

This cohort has **one regression family**, so no regression-family
uncertainty interval or general regression-superiority claim is supported.
Its IID/retrospective tasks are not prospective clinical or forecasting
validation.

## Candidate loss reductions against matched robust raw

Positive percentages mean lower log loss or RMSE.

| Learner | Classification mean | Complete classification families | Regression reduction, one family |
|---|---|---|---|
| Linear | +0.56% | 3/3 | -0.01% |
| LightGBM | +0.0051% | 3/3 | 0.00% |
| CatBoost | **-7.15%** | **2/3** | -2.77% |
| XGBoost | **-2.61%** | 3/3 | +3.82% |

The CatBoost mean excludes its entirely timed-out personality family and
must not be compared with a three-family reference mean. Linear's only
greater-than-1% classification gain is Assam (+1.68%); LightGBM's
classification changes are all within the 1% tie tolerance.

Thyroid/CatBoost log loss rises **0.303167 to 0.344869**, a **13.76%**
increase. Thyroid/XGBoost rises **0.337072 to 0.363657**, a **7.89%**
increase. These violate the predeclared family-harm criterion.

## Reference comparisons and coverage

On all three classification families, OpenFE's mean changes versus robust
raw are -8.28% linear, -6.52% LightGBM and -3.18% XGBoost. The incumbent
therefore avoids some reference harm, but that is not equivalent to
consistently improving the raw learner.

On the **two common CatBoost classification families**, incumbent mean
reduction is -7.15%, OpenFE +1.52%, and AutoFeat +0.85%.
On the two common XGBoost classification families, incumbent is -3.94%
and AutoFeat -1.47%. Exact common-family comparisons are retained in
[decision.json](decision.json); no cross-coverage leaderboard is used.

For Facebook/XGBoost, incumbent reduces RMSE by +3.82%, versus OpenFE
-1.36% and AutoFeat -41.69%. This is one useful result on one family,
not evidence of general regression superiority.

## Failure and promotion decision

All Personality Drug Consumption AutoFeat cells timed out: three folds
times four learners, **12 failures**. FeatCopilot's three CatBoost cells
on that family also timed out. Each received the full 900-second cell
allowance. No source change, raised deadline, post-outcome retry or family
substitution was applied.

The frozen gates fail on candidate/reference coverage, negative or neutral
stratum means, and greater-than-5% family harms. **Defaults remain
unchanged and no candidate is promoted.** These four families are now
exposed and cannot be recycled as fresh confirmation for a revision.

## Evidence

All 192 cell identities are unique; failures contain no invented loss.
Start/end/current source hashes matched before further work:
`2c56b1386d67e0a812324482b57ba4f345e5d404f32c3b2cf437765456036e15`.
Exact result/source checksum and bounded archive reconstruction passed.

- [REPORT.md](REPORT.md): full configuration, losses, costs and coverage.
- [archive.json](archive.json): ordered lossless result parts.
- [auxiliary_archive.json](auxiliary_archive.json): bounded manifest/source.
- [decision.json](decision.json): coverage-qualified failed promotion.
- [Locked suite](../../suites/cycle2_robust_confirmation_20260930.json)
  and [release audit](../../suites/cycle2_robust_confirmation_release_20260930.json).

The current 72-hour cycle has used **22.77030 hours**, leaving
**49.22970 hours**. No live LLM calls were made. The next research question
is prediction-level variance/calibration with matched raw controls,
not another attempt to relabel these confirmation outcomes.
