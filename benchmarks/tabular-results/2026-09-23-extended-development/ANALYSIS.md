# Extended candidate development comparison

**Adding candidate families alone did not close the boosted-model gap and
introduced a serious linear-model regression. Extended modes remain opt-in;
none is promoted to the default and no SOTA claim is warranted.**

All 576 requested cells completed successfully in **10.64 active hours**,
with unchanged source. The fixed comparison used the same 16 development
families, three folds, seed 42, linear/LightGBM, 1,000 candidate and 50-addition
budgets, eight threads and 600 seconds per cell. Reserved confirmation
candidates were not evaluated.

## Observational primary-loss reductions

Positive means lower loss than the matched raw learner. Classification uses
log loss over nine families; regression uses RMSE over six families.

| Task / learner | Standard | Paired | Extended | Extended + paired | OpenFE-inductive |
|---|---|---|---|---|---|
| Classification / linear | +14.09% | +14.18% | +11.48% | +8.70% | +11.24% |
| Classification / LightGBM | -0.33% | +1.56% | +0.99% | +1.81% | +9.71% |
| Regression / linear | +10.65% | +11.37% | +9.81% | +11.29% | +8.28% |
| Regression / LightGBM | +1.62% | +0.97% | +1.91% | +1.70% | +2.56% |

Energy Efficiency simulation remains separate. Extended-paired linear
improved it by +38.01%, while OpenFE-inductive achieved +43.87%.
Extended-paired LightGBM worsened it by -2.34%, versus OpenFE's +8.05%.

## Gains do not erase the regressions

Extended candidates improved several cases, including Banknote/linear
(+95.62% relative loss reduction), houses/LightGBM (about +5.2%), and
Combined Cycle Power/LightGBM (+6.92%). However:

- Chronic Kidney Disease/linear worsened by **46.28%** under extended-paired,
  versus +4.55% for the standard numeric pool.
- Extended modes reduced the previous Spambase/linear gains.
- The boosted-classification mean remains materially behind the reference.

The low baseline losses on some classification tasks make relative gains
sensitive; per-dataset absolute losses remain in [REPORT.md](REPORT.md).

## Development-only diagnosis

Training-derived selection records expose redundancy: houses rankings contain
multiple minima with the same longitude column that are identical to that
original column on the training partition. Other monotone variants consume
several nearby ranks. More operator choices do not automatically yield more
independent predictive signal.

The next bounded hypothesis is exact **training-only** candidate deduplication
and joint learner-importance screening conditioned on original features.
Independent internal validation must still select the final subset. This is
not a claim that training importance or these diagnoses predict unseen results.
No dataset-specific formula or confirmation-driven choice is introduced.

## Evidence

All 576 cell IDs are unique and scored. The exact result, manifest and source
archives were verified.

- [REPORT.md](REPORT.md) contains paired results and individual-family means.
- [archive.json](archive.json) lists lossless result parts.
- [auxiliary_archive.json](auxiliary_archive.json) lists bounded manifest/source
  parts and original checksums.

Raw large artifacts remain local for resume; numbered public archives preserve
their exact contents. The preceding [full-reference comparison](../2026-09-22-full-reference-development/ANALYSIS.md)
is unchanged and retains AutoFeat's resource/fitting failures.
