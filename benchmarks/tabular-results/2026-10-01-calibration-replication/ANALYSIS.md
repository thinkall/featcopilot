# Frozen classifier-policy replication: seeds 137 and 2029

**Matched FE gains replicate, but the full policy still does not pass the
tail-harm criterion against the existing single raw learner.** Keep the
learner choices frozen in this analysis: linear uses single robust FE,
and LightGBM uses the plain three-fold FE prediction ensemble. No
dataset-specific switches or favorable calibration substitutions are made.

All **648 prediction outcomes succeeded** from **432 physical jobs** in
**9.95263 active hours**. Source and dependency identities match the
seed-42 study. All 216 paired bundles retain the exact same three bases
for plain/calibrated views. These are additional splits of nine exposed
origins, not new datasets or independent confirmation.

## Preselected policy, without post-result switching

Loss reductions are family-level ratios after averaging all relevant
fold losses within each family.

| Seed | Policy | Mean vs matched raw | Worst vs matched raw | Mean vs single raw | Worst vs single raw |
|---|---|---|---|---|---|
| 137 | Linear single FE | +17.96% | -2.35% | +17.96% | -2.35% |
| 2029 | Linear single FE | +18.06% | -0.67% | +18.06% | -0.67% |
| Both | Linear single FE | **+18.06%** | **-1.00%** | +18.06% | -1.00% |
| 137 | LightGBM plain FE ensemble | +5.63% | -3.18% | +2.17% | **-40.13%** |
| 2029 | LightGBM plain FE ensemble | +7.92% | -1.54% | +6.90% | **-12.47%** |
| Both | LightGBM plain FE ensemble | **+6.89%** | **-0.59%** | +4.90% | **-23.66%** |

The matched LightGBM comparator is the raw prediction ensemble, not the
full-training single raw model. The former can itself lose accuracy or
confidence through its smaller base-training partitions.

## The remaining CKD tradeoff

CKD/LightGBM shows why both comparisons are necessary:

| Seed | Single raw log loss | Raw ensemble | FE ensemble |
|---|---|---|---|
| 137 | 0.017498 | 0.023764 | 0.024520 |
| 2029 | 0.025772 | 0.031515 | 0.028986 |

FE's value relative to the matched ensemble is broadly positive, but
changing the model architecture still harms this family relative to the
existing single raw learner. The absolute losses are small; the
predeclared relative-harm criterion nevertheless fails and is not
retroactively loosened.

Wisconsin's improvement persists for the LightGBM FE ensemble: +4.81%
and +8.86% versus matched raw ensemble, and +15.44%/+16.21% versus single
raw. That useful result does not justify hiding CKD through a
dataset-specific architecture rule.

## Calibration remains a diagnostic, not a rescued candidate

Sigmoid again worsens equally weighted family-average log loss and Brier
sum relative to the plain views of the exact same fitted models. It is
not selected for any dataset after inspecting these outcomes.
Single, plain and sigmoid raw/FE results all remain in [REPORT.md](REPORT.md).

The linear single-FE policy is a promising development component;
LightGBM plain FE averaging has reproducible matched gains but an
unresolved training-size/tail-risk tradeoff. Neither conclusion is a
fresh-origin SOTA claim.

## Integrity and resources

All 648 identities are unique and scored. All 432 physical checkpoint
records are present, with shared training counted once. Source remained:
`738d4fc39b57cb5fb3557070ed1cb7460c809114099656e9deb87e8a5262a40e`.
Source/result checksums and exact bounded reconstruction passed.

- [REPORT.md](REPORT.md): all six architectures and paired contrasts.
- [archive.json](archive.json): ordered lossless results.
- [auxiliary_archive.json](auxiliary_archive.json): bounded manifest/source.
- [decision.json](decision.json): the frozen policy's per-seed and combined
  matched/single-raw comparisons, without architecture switching.

The current cycle has used **37.49762 of 72 hours**, leaving
**34.50238 hours**. No additional LLM calls occurred.
The next hypothesis may test three higher-retention training subsamples
to examine the training-size tradeoff, but must be specified before any
new outcomes; no fraction sweep or claim that sample size is the proven
cause is warranted.
