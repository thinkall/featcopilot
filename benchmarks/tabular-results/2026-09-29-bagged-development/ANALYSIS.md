# Five-bag mean-rank development experiment

**Resampled ranking is not a general improvement on this panel.** It
improves boosted regression but reduces three other mean-loss gains,
worsens the largest classification harm, and costs roughly three times
as much. Keep it experimental; do not promote it or open the reserved
confirmation cohort on the strength of these results.

All **288 cells succeeded in 6.36447 active hours**. All **192 matched
control losses reproduce the preceding robust-development run exactly**.
Sixteen exposed development families used three folds, seed 42, unchanged
5,000-row/1,000-candidate/50-addition limits, three internal validation
repeats, and the same three learner settings. Eight threads, 600-second
per-cell deadlines and 4,096 MiB sampled RSS were retained.

## Fixed change and prior art

The new arm averages candidate-level importance ranks from five 85%
training subsamples. Every bag fits its own frequency mappings, caps,
imputer, encoder and intended learner. Scope-level exact filtering fixes
the candidate universe; bag-local unusable/duplicate candidates map to
zero importance and worst tied ranks without regeneration or refill.
Candidates never positive in any bag are ineligible.

The actual aggregate-ranked prefixes are validated inside the existing
holdouts; positive-count final ranking repeats the same procedure. This is
not a post-validation majority filter. The existing single-fit arm and
matched capped raw control are unchanged.

[AutoFeat's repeated selection](https://github.com/cod3licious/autofeat/blob/639793a6e620f2b8da21f32620a1c94d1d7c50bc/src/autofeat/featsel.py#L179-L276)
motivated the resampling idea. It does not establish that mean-rank
aggregation will work, and this experiment supplies no formal stability-
selection, false-positive, or no-harm guarantee.

## Observational loss reduction against capped raw

Classification uses log loss across nine families; regression uses RMSE
across six. Positive percentages mean lower loss than `baseline-robust`.

| Task / learner | Single-fit aligned FE | Five-bag aligned FE |
|---|---|---|
| Classification / linear | **+16.12%** | +14.84% |
| Classification / LightGBM | **+8.12%** | +7.55% |
| Regression / linear | **+14.97%** | +14.77% |
| Regression / LightGBM | +1.39% | **+2.60%** |

Bagged medians in the same order are +2.17%, +0.33%, +10.47% and +1.48%.
The small number of repeatedly examined development families and
descriptive bootstrap intervals do not justify superiority claims.

The separate Energy Efficiency simulation improves from +6.32% to +9.59%
with LightGBM, while linear decreases from +81.39% to +80.61%. It is not
included in the observational means.

## Tail harm and instability remain

Wisconsin Diagnostic/LightGBM worsens from **22.76%** above raw loss to
**32.90%** above raw loss. CKD/linear stays better than capped raw, but
loses most of its previous gain: +16.74% becomes +4.69%, a **14.48%**
loss increase versus the single-fit engineered control.

Useful changes include Concrete/LightGBM moving from -4.12% to +0.15%
versus capped raw, and Banknote/linear from +90.67% to +93.60%.
Those wins do not erase worse aggregate quality or larger tail harm.

Mean pairwise selected-name Jaccard, first averaged within each family:

| Task / learner | Single-fit | Five-bag | Selected-count standard deviation, single to bagged |
|---|---|---|---|
| Classification / linear | 0.389 | 0.416 | 6.13 to 5.23 |
| Classification / LightGBM | 0.326 | 0.314 | 7.13 to 3.24 |
| Regression / linear | 0.405 | 0.367 | 3.93 to 7.68 |
| Regression / LightGBM | 0.263 | 0.301 | 9.70 to 12.54 |

Stability and accuracy do not move together. A steadier feature count does
not imply steadier identities, and steadier identities do not imply better
loss. Empty/empty selected-name sets score one; these diagnostics do not
measure equality of fitted mapping values or predictive interchangeability.

## Cost and decision

Mean pipeline seconds per family/cell:

| Task / learner | Single-fit | Five-bag |
|---|---|---|
| Classification / linear | 28.18 | 89.54 |
| Classification / LightGBM | 61.22 | 177.27 |
| Regression / linear | 10.12 | 29.54 |
| Regression / LightGBM | 26.25 | 70.58 |

The measured increase is approximately **2.7-3.2 times**. Extra ranking
fits are charged to the same deadlines; this is not free ensembling.
All 96 bagged cells retain their three tuning trials, chosen-parameter
transfer and five-bag diagnostics. Those records were checked for
consistency before publication.

[decision.json](decision.json) records that three observational mean-gain
strata decline and stability is mixed. **Default promotion is rejected.**
The four metadata-reserved fresh families remain unopened.

## Evidence and next hypothesis

All 288 cell identities are unique. Start/end/current source hashes matched
before further work:
`0242853ec6aef62dfcb6dcd91e083f7302f7811a4aca80af36dcb33f73f27b61`.
Exact source/result checksums, bounded archive reconstruction, and all
control-loss comparisons passed.

- [REPORT.md](REPORT.md): matched summaries and all family losses.
- [archive.json](archive.json): ordered lossless result parts.
- [auxiliary_archive.json](auxiliary_archive.json): bounded manifest/source parts.
- [stability.json](stability.json): all family stability diagnostics.
- [decision.json](decision.json): unsuccessful promotion decision.

Do not respond by sweeping bag counts or sample fractions on these results.
The next research question is whether holding back an independent part of
training data from all feature search, then validating and freezing a
specific proposed plan, reduces selection optimism and refit mismatch.
That is a separate hypothesis, not a demonstrated explanation or fix.

This is the first measured experiment in the newly authorized cycle:
**6.36447 of 72 hours used; 65.63553 hours remain**. No live LLM calls occurred.
