# Cross-fitted categorical target-encoding ablation

**Target encoding is not a general improvement on this panel.** Its matched
FE gains can look favorable because the target-encoded raw comparator
becomes weaker, especially for linear CKD. Keep one-hot as the default;
do not promote target encoding based only on its matched-raw percentage.

All **216 cells succeeded in 3.98840 active hours**. All **108 one-hot
control losses exactly reproduce the earlier fixed-split results**.
Nine exposed classification families used three folds, seed 42,
linear/LightGBM, unchanged row/candidate/feature limits and learner grids.

## One preprocessing change, with explicit boundaries

Target mode uses sklearn's cross-fitted categorical encoding with fixed
five-fold stratification, automatic smoothing and the existing integer
seed. The learner trains on the matrix returned by `fit_transform`;
held-out/inference rows use the stored full-training map through
`transform`. Every internal feature-selection and final model fit owns
its own encoder. Encoded values never enter the candidate formula pool.

Target-mode categorical preparation preserves scalar types rather than
merging integers/booleans with similarly spelled strings. The historical
one-hot path is unchanged. This correction and raw-type categorical
literals belong to the declared target-mode pipeline; they are not an
unreported change to the OHE controls.

## Four-arm results

Mean primary log-loss reductions, averaging within each of nine families
before averaging families:

| Learner | Target raw vs OHE raw | OHE FE vs OHE raw | Target FE vs OHE raw | Target FE vs target raw |
|---|---|---|---|---|
| Linear | **-16.73%** | +16.12% | +0.89% | +15.52% |
| LightGBM | -1.22% | +8.12% | +8.08% | +8.92% |

The last column is necessary to measure incremental FE value, but not
sufficient to choose the whole pipeline. For linear models, it conceals
the substantial loss caused by replacing the stronger OHE representation.
No smoothing, fold-count or cardinality-based encoder sweep follows.

## CKD illustrates the control issue

Mean CKD/linear log loss:

| Representation | Loss |
|---|---:|
| OHE raw | 0.014859 |
| OHE FE | 0.012371 |
| Target-encoded raw | 0.036953 |
| Target-encoded FE | 0.032223 |

Target FE improves 12.80% versus target raw, but is **116.86% worse
than OHE raw** and **160.47% worse than OHE FE**. Lower-dimensional,
cross-fitted encoding can still remove useful category distinctions or
change model regularization geometry. These results do not establish a
leakage defect or isolate the causal mechanism.

German Credit/linear also worsens versus OHE raw by 2.93% with target FE.
Online Shoppers improves modestly from +2.27% OHE FE to +3.32% target FE
versus OHE raw. There is no uniform dominance.

## Activation, dimensions and cost

Target encoding is active on five families and inactive on the four
numeric-only families. Each target method/learner has fifteen active and
twelve inactive outer-fit cells. Inactive cases retain numeric behavior;
the option does not fabricate a categorical encoding stage.

[encoding_diagnostics.json](encoding_diagnostics.json) preserves actual
categorical input/output widths, target type, encoder row/class counts,
cross-fitted array dtype/bytes and total encoded width. These are model
input dimensions, not generated-feature counts or worker peak RSS.
No category vocabularies, target priors or encoding matrices are published.

Mean pipeline seconds are 28.53 to 30.26 for OHE versus target linear FE,
and 60.95 to 62.32 for LightGBM FE. Target cross-fitting adds preprocessing
work, not five extra downstream learner fits.

## Integrity and decision

All 216 identities are unique and scored. Source remained:
`6ab4fe2df607ced7a8ad23e160f03688144b7e70d7d79bc0292c28818404b82d`.
Exact result/source checksums, bounded reconstruction and all unchanged
OHE controls were verified.

- [REPORT.md](REPORT.md): raw/OHE and matched-target summaries.
- [archive.json](archive.json): ordered lossless result parts.
- [auxiliary_archive.json](auxiliary_archive.json): source/manifest parts.
- [decision.json](decision.json): no default promotion.
- [encoding_diagnostics.json](encoding_diagnostics.json): actual encoded widths.

The API remains explicit and experimental; existing one-hot defaults and
native CatBoost semantics are unchanged. This is development evidence,
not a new confirmation or SOTA result.

Current-cycle measured execution totals **55.37356 of 72 hours**, leaving
**16.62644 hours**. No live LLM calls occurred.
