# Higher-retention LightGBM prediction ensembles

**Retaining more training rows improves the overall tradeoff, but does not
eliminate split-sensitive harm.** Across all three seeds, plain FE85
averaging gains 7.60% over matched raw85 and 7.31% over full-training
single raw, with worst family losses of 2.59% and 4.95%, respectively.
CKD still has substantial per-seed harm. This remains a development
candidate, not a safe default or SOTA result.

All **486 prediction outcomes succeeded** from **324 physical jobs** in
**9.12791 active hours**. All **162 single-model control losses match
the archived `cv3` studies exactly**. The nine datasets, all 81 outer
split identities, learner grids and dependency versions are unchanged.

## One fixed sampling intervention

Each ensemble retains three fitted complete learning procedures. Their
base-training bags use approximately 85% of the allowed training rows,
with complementary approximately 15% calibration holdouts. All
hyperparameter search, feature selection, frequency state, clipping,
imputation and model fitting stay within the base-training side.

The three bags overlap; calibration rows can appear in zero, one, two
or three holdouts. These are not a three-fold partition or pooled OOF
predictions. Actual sizes, masks, coverage and overlaps are recorded.
No cases are deduplicated by value, inconvenient splits redrawn, or
calibration methods selected from outcomes.

Plain and sigmoid views share the exact same retained bases. Sigmoid
remains diagnostic-only. The existing `cv3` default and frozen
linear-single policy are unchanged.

## Complete three-seed comparison

Family-level ratios average all relevant fold losses before averaging
families. Positive values mean lower log loss.

| Seed | FE85 vs raw85 mean | Worst family vs raw85 | FE85 vs single raw mean | Worst family vs single raw |
|---|---|---|---|---|
| 42 | +10.55% | -0.32% | +8.41% | -3.72% |
| 137 | +4.02% | **-8.53%** | +4.87% | **-16.88%** |
| 2029 | +6.10% | **-28.97%** | +7.30% | **-22.50%** |
| All three | **+7.60%** | **-2.59%** | **+7.31%** | **-4.95%** |

The per-seed violations are CKD. Do not present the combined mean as
proof that every resampling split is safe. Aggregate ratios are not the
arithmetic average of the per-seed percentages.

Against the previously measured FE `cv3` ensemble on the same splits,
FE85 improves equally weighted family-average relative loss by **1.35%**
overall. The difference is positive on each seed on average, but includes
large individual tradeoffs: Wisconsin/seed42 worsens 15.32% versus FEcv3,
and Banknote/seed137 worsens 12.03%.

## What changed, and what was not established

On the same three seeds, CKD's loss increase versus single raw is reduced
from **10.54% for FEcv3 to 4.95% for FE85**. The previously reported
23.66% CV3 harm used only seeds 137/2029 and is not the matched
three-seed comparator. Exact per-seed contrasts remain in
[decision.json](decision.json).

More training data also means more member overlap and less diversity.
This experiment therefore does not isolate sample size as the sole
cause of improvement. Min/max caps, the candidate catalogue, selection
rule, search values and equal prediction weights were not changed.

Sigmoid still worsens mean log loss and Brier sum versus the plain
views of the same bases: log-loss increases are 0.026098 for raw and
0.026194 for FE. Its smaller calibration holdouts do not make it a
preferred variant, and no post-result isotonic/temperature sweep is applied.

## Integrity, costs and next boundary

All 486 identities are unique and scored. All 162 paired bundles have
three shared base identities and mode-consistent diagnostics. Source
remained:
`ec3d1f9c93d39ff9e6667085571be2194cfcf73197d274ca6c653095f5757ef5`.
Exact source/result hashes and bounded archive reconstruction passed.
Physical bundle training is counted once, not once per prediction view.

- [REPORT.md](REPORT.md): full contrasts and actual sampling membership.
- [archive.json](archive.json): ordered lossless results.
- [auxiliary_archive.json](auxiliary_archive.json): bounded manifest/source.
- [decision.json](decision.json): every seed's matched and single-raw harms.

Freeze FE85 for LightGBM and retain single FE for linear models for a
separate two-origin **standard-IID, entity-independence-unverified**
confirmation, subject to its locked source and tier checks. No choice
depends on which dataset is being predicted. The new tier was explicitly
approved before acquisition and does not support patient/entity-independent,
clinical, prospective or broad SOTA claims.

The current cycle has used **46.62553 of 72 hours**; **25.37447 hours
remain**. No additional LLM calls were made.
