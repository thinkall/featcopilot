# Parameter-aligned pipeline development ablation

**Alignment fixes a major linear-classification failure on this development
panel, but does not improve most aggregate strata or remove boosted harm.
It remains a benchmark-only experiment, not a production default or SOTA
result.**

All **288 cells succeeded in 3.85811 active hours**. The original sixteen
development families, three folds, seed 42, 5,000-row cap, 1,000 candidates,
50 additions, three internal repeats and matched learner grids are unchanged.
Raw, frequency-paired and frequency-aligned use the same splits in this run.

## What changed

Frequency-paired builds features using the learner's fixed middle setting,
then tunes the downstream model across three settings. Frequency-aligned
instead independently fits features for each of those same three settings,
using the matching learner parameters during feature selection. Inner
validation chooses the full configuration, and outer refitting preserves
its selection-estimator parameters.

All extra feature fits count against the same cell deadline. Records include
each trial's parameters, loss, width and selection report. Every one of the
**96 aligned cells** was checked: the chosen parameters match the lowest
recorded inner loss and the final selection-estimator configuration.
Feature sets can still change when training data grow; alignment is not an
extrapolation guard or a no-harm guarantee.

## Same-run observational results

Positive percentages mean lower primary loss than the matched raw learner.
Classification covers nine families and regression six.

| Task / learner | Frequency-paired | Frequency-aligned |
|---|---|---|
| Classification / linear | -0.30% | **+14.96%** |
| Classification / LightGBM | **+8.26%** | +8.12% |
| Regression / linear | **+13.54%** | +12.70% |
| Regression / LightGBM | **+2.49%** | +1.39% |

Aligned medians in the same order are +2.55%, 0.00%, +10.00% and +1.32%.
Aligned linear-classification W/T/L is 6/3/0 versus paired 6/2/1. The
boosted-classification median remains zero, and mean benefits remain
concentrated in a few tasks.

Energy Efficiency remains separate simulation evidence: aligned reductions
are +81.39% linear and +6.32% LightGBM, compared with +82.07%/+5.93% paired.

## Imputation correction changes the historical comparison

Both predictive arms now preserve numeric missingness until their own
training-local preprocessing. The earlier frequency experiment imputed the
whole supplied training frame before internal holdouts. Its predictive
scores therefore are **not** an unchanged control for this experiment.

All **96 raw losses reproduce exactly**. Ten historical frequency-paired
cell losses change, confined to Chronic Kidney Disease and Auto MPG, which
have missing numeric values. Preserve those earlier scores and their source
archives; do not restore premature imputation because it happened to score
better.

For CKD/linear, the corrected paired log loss is **0.035217**, versus raw
**0.016247** and aligned **0.015710**. Thus alignment changes a **116.77%
loss increase** into a **3.30% reduction** against the same-run raw learner.
The earlier imputed frequency-paired result was 0.012448; comparing only
that historical number with aligned would confound two changes.

Other tradeoffs remain:

- Spambase/linear improves from +14.10% paired to +29.32% aligned.
- Wisconsin Diagnostic/LightGBM remains **22.76% worse than raw** in both.
- Wine/linear changes from +2.29% paired to **-1.97%** aligned.
- Combined Cycle Power/LightGBM falls from +8.26% to +4.68%.
- Concrete/LightGBM changes from -3.37% to **-4.12%**.

## Cost

Mean pipeline seconds per family/cell:

| Task / learner | Paired | Aligned |
|---|---|---|
| Classification / linear | 12.72 | 22.22 |
| Classification / LightGBM | 33.15 | 55.23 |
| Regression / linear | 4.62 | 7.40 |
| Regression / LightGBM | 12.69 | 23.41 |

Alignment trades roughly 1.6-1.8 times the measured pipeline cost for
mixed quality changes on this panel. Times include additional feature
fitting but exclude some supervisor/process startup overhead.

## Evidence and next boundary

All 288 identities are unique and scored. Start/end/current source hashes
matched at verification:
`b7f0b38ae2f73780da84bb1452dcb28e1cec3bf1ec4c44487008fd2d0be2d4c1`.
Exact result/source checksums and bounded archive reconstruction passed.

- [REPORT.md](REPORT.md): full configuration, family losses and intervals.
- [archive.json](archive.json): ordered lossless result parts.
- [auxiliary_archive.json](auxiliary_archive.json): manifest/source parts.

The original confirmation cohorts remain exposed, and their failed
promotion decisions are unchanged. One final bounded retrospective check
will compare raw/paired/aligned on the same six training-only hardware and
temperature audit partitions, never the original outer-test rows. It cannot
serve as fresh confirmation for the revised pipeline.

Current-cycle measurements total **65.37595 hours**, including the prior
training-only audit. With the separate 0.25-hour operational probe allowance,
**6.37405 hours remain**. Defaults remain legacy and SOTA is not established.
