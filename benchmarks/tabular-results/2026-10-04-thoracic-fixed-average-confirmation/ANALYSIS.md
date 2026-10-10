# Thoracic Surgery: qualified fixed-average confirmation

**The authoritative confirmation decision is BLOCKED**, because one
AutoFeat fold failed. Fixed averaging completed all three folds and
slightly improved log loss versus raw, but the missing comparison cannot
be replaced by a successful-fold mean or ignored.

All **15 outcomes** and **nine physical jobs** are recorded, using
**0.34997 active hours**. Four methods score 3/3 folds; AutoFeat scores
2/3 and raises a NaN-input error in its internal LogisticRegressionCV on
fold 2. No scores were substituted and no post-result retry occurred.

## Fixed method and restricted claim scope

The single source is Thoracic Surgery, UCI277/OpenML1506 version1/task9969:
470 records, sixteen reviewed predictors and the original one-year binary
outcome. Original ten-fold source identities remain recorded; the user-
approved experiment uses three newly registered IID folds, seed 2031,
full rows, matched fixed learner grids and final numeric-support caps.

This is **`standard_iid_unverified_entity`** evidence only. Patient/entity
independence, clinical/prospective generalization, pooling into a verified
tier and automatic SOTA/default promotion are not authorized. One
family's three folds are not three independent datasets.

The fixed policy averages the full-training raw and FE probabilities
50:50. It learns no mixture weight and uses no OOF blend fitting or
calibration. Both native FE references retain their declared generation
algorithms and matched final clipping.

## Complete comparisons

| Method | Complete folds | Mean log loss |
|---|---:|---:|
| Raw robust LightGBM | 3/3 | 0.510180 |
| FE single | 3/3 | 0.520143 |
| **Fixed 50:50 average** | **3/3** | **0.508215** |
| OpenFE-inductive-native-robust | 3/3 | 0.510180 |
| AutoFeat-native-robust | **2/3** | **Undefined for the locked comparison** |

The fixed average improves 0.385% versus raw and OpenFE and 2.293%
versus FE single. The raw harm criterion and those complete comparisons
pass, but complete AutoFeat coverage is required by the frozen protocol.
Consequently, the overall decision remains **blocked**, not passed.

Raw and FE predictions are identical on folds 0 and 1. On fold 2, raw
loss is 0.528401, FE loss is 0.558289 and their fixed average scores
0.522507. This is a small, qualified result rather than a broad claim
of superiority.

## Reference failure and preservation

AutoFeat fold 2 reports:
`ValueError: Input X contains NaN. LogisticRegressionCV does not accept missing values ...`

The error occurred inside the declared upstream reference path. It must
not be repaired by dropping the fold, using the raw score, changing
feature definitions, or presenting a modified adapter as the original
method. A separately named, justified compatibility fix could be
investigated later, with the original failure and decision preserved.

## Integrity and resources

All 15 identities are unique, failures have no invented loss, and every
outcome retains the restricted tier. Source remained:
`1466feb7980e3b327845ffaf14ac7d0dee4ad4ccb9e9d4c5fd3afa36f8bb2249`.
Exact result/source checksums and bounded reconstruction passed.

- [REPORT.md](REPORT.md): authoritative blocked decision and all provenance.
- [archive.json](archive.json): exact ordered results.
- [auxiliary_archive.json](auxiliary_archive.json): bounded source/manifest.
- [Locked suite](../../suites/thoracic_fixed_average_confirmation_20261004.json)
  and [release audit](../../suites/thoracic_fixed_average_confirmation_release_20261004.json).

Thoracic is now exposed. Defaults remain unchanged and no SOTA claim
is established. Current-cycle measurement totals **60.92824 of 72 hours**,
leaving **11.07176 hours**. No new LLM calls were made.
