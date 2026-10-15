# Seoul whole-day temporal development evaluation

**All 18 cells completed. Predictive FE helped the linear model modestly but
did not improve LightGBM in this run.**

This is retrospective demand estimation conditional on observed weather,
**not ahead-of-time forecasting**. Date-only buckets preserve complete days;
Hour remains a predictor. The reviewed target is `Rented Bike Count`, overriding
the inconsistent upstream target-role metadata. `Functioning Day` is excluded.
No fine-grained label-availability horizon is inferred.

The uncapped dataset contains 8,760 hourly observations in 365 day buckets.
Three forward outer splits, seed 42, and training-only forward inner selection
were used. The run took approximately 5.27 minutes, with unchanged source and
no failures or timeouts.

| Learner | Raw RMSE | Predictive RMSE | Predictive loss reduction | Polynomial loss reduction |
|---|---|---|---|---|
| Linear | 578.66 | 557.17 | +3.71% | -0.74% |
| LightGBM | 545.98 | 545.98 | +0.00% | +8.43% |

Predictive LightGBM retained the original feature width on all three folds;
the polynomial control reached RMSE 499.94. This is an informative
development result, not evidence that the predictive strategy dominates
alternative feature representations or forecasting methods.

See [REPORT.md](REPORT.md), [manifest.json](manifest.json), [source.zip](source.zip)
and [archive.json](archive.json). Byte-exact result reconstruction, source
archive checksum, unique requested cells and split provenance were verified.

True forecasting still requires explicit prediction-time feature availability,
label horizons and appropriate temporal baselines; this experiment does not
claim those capabilities merely because its outer split is chronological.
