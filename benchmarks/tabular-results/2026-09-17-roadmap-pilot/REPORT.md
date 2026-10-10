# Tabular development benchmark

**Benchmark measurement, not a SOTA claim.** Dataset identity, admission roles and split policies
are recorded in the manifest. Legacy manifests are not automatically rights-reviewed or confirmation data.

Primary loss: classification log loss; regression RMSE (lower is better).
Learners are tuned separately per method on inner validation, never selected using outer-test scores.
Synthetic diagnostics are reported separately. All requested failures count toward coverage.

OpenFE-stateless is an explicitly restricted variant, not full OpenFE. AutoFeat uses numeric generation.
All methods retain original categorical features for identical downstream one-hot preprocessing.
FeatCopilot-task changes only explicit task routing; FeatCopilot-predictive uses learner-aligned holdout selection.
The predictive method fits a separate representation per learner with its predeclared middle search candidate.
Each protocol-v3 worker has one learner and an independent budget. Native CatBoost uses categorical data;
its internal predictive-selection estimator uses numeric preprocessing, a disclosed representation difference.

## Configuration

```json
{
  "datasets": [
    "uci-adult",
    "uci-german-credit",
    "uci-online-shoppers",
    "uci-dry-bean",
    "uci-steel-faults",
    "uci-concrete-strength",
    "uci-energy-efficiency",
    "houses",
    "wine"
  ],
  "folds": 3,
  "seeds": [
    42,
    137
  ],
  "max_rows": 5000,
  "max_added_features": 50,
  "max_candidates": 1000,
  "threads": 8,
  "memory_limit_mb": 4096,
  "case_timeout": 600.0,
  "synthetic": false,
  "stability_repeats": 3,
  "screening": "auto",
  "allow_confirmation": false,
  "detailed_diagnostics": false,
  "methods": [
    "baseline",
    "featcopilot",
    "featcopilot-task",
    "featcopilot-predictive",
    "polynomial",
    "openfe-stateless",
    "autofeat"
  ],
  "learners": [
    "linear",
    "lightgbm",
    "catboost",
    "xgboost"
  ],
  "suite": [
    "C:\\code\\featcopilot\\benchmarks\\suites\\uci_development.json",
    "C:\\code\\featcopilot\\benchmarks\\tabular_suite.json"
  ]
}
```

## Paired dataset-level results

Loss reduction is 1 - mean(method loss) / mean(baseline loss), first averaged within each dataset.
W/T/L uses a predeclared 1% relative-loss tolerance, not a significance test.
95% intervals bootstrap origin families (not dependent folds), are descriptive and not multiplicity-adjusted.
Rows with different complete-pair coverage must not be ranked against one another.

| Source / role | Task | Learner | Method | Paired datasets / families | Cells OK/all | W/T/L | Mean loss reduction | 95% CI |
|---|---|---|---|---|---|---|---|---|
| real_world / development | classification | catboost | autofeat | 3/5 / 3/5 | 18/30 | 0/3/0 | -0.45% | [-0.98%, -0.06%] |
| real_world / development | classification | catboost | baseline | 5/5 / 5/5 | 30/30 | 0/5/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | classification | catboost | featcopilot | 5/5 / 5/5 | 30/30 | 0/5/0 | +0.21% | [-0.17%, +0.64%] |
| real_world / development | classification | catboost | featcopilot-predictive | 5/5 / 5/5 | 30/30 | 1/4/0 | +0.92% | [-0.15%, +2.90%] |
| real_world / development | classification | catboost | featcopilot-task | 5/5 / 5/5 | 30/30 | 0/5/0 | +0.21% | [-0.17%, +0.64%] |
| real_world / development | classification | catboost | openfe-stateless | 5/5 / 5/5 | 30/30 | 2/2/1 | +0.61% | [-1.18%, +2.51%] |
| real_world / development | classification | catboost | polynomial | 5/5 / 5/5 | 30/30 | 0/3/2 | -1.34% | [-2.76%, -0.33%] |
| real_world / development | classification | lightgbm | autofeat | 3/5 / 3/5 | 18/30 | 0/1/2 | -1.76% | [-2.67%, -0.44%] |
| real_world / development | classification | lightgbm | baseline | 5/5 / 5/5 | 30/30 | 0/5/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | classification | lightgbm | featcopilot | 5/5 / 5/5 | 30/30 | 0/5/0 | +0.01% | [-0.08%, +0.08%] |
| real_world / development | classification | lightgbm | featcopilot-predictive | 5/5 / 5/5 | 30/30 | 0/5/0 | +0.10% | [-0.09%, +0.32%] |
| real_world / development | classification | lightgbm | featcopilot-task | 5/5 / 5/5 | 30/30 | 0/5/0 | +0.01% | [-0.08%, +0.08%] |
| real_world / development | classification | lightgbm | openfe-stateless | 5/5 / 5/5 | 30/30 | 0/0/5 | -2.08% | [-2.75%, -1.40%] |
| real_world / development | classification | lightgbm | polynomial | 5/5 / 5/5 | 30/30 | 0/1/4 | -3.31% | [-4.88%, -1.88%] |
| real_world / development | classification | linear | autofeat | 3/5 / 3/5 | 18/30 | 1/2/0 | +0.51% | [+0.01%, +1.32%] |
| real_world / development | classification | linear | baseline | 5/5 / 5/5 | 30/30 | 0/5/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | classification | linear | featcopilot | 5/5 / 5/5 | 30/30 | 0/5/0 | -0.02% | [-0.12%, +0.09%] |
| real_world / development | classification | linear | featcopilot-predictive | 5/5 / 5/5 | 30/30 | 3/2/0 | +1.60% | [+0.01%, +3.53%] |
| real_world / development | classification | linear | featcopilot-task | 5/5 / 5/5 | 30/30 | 0/5/0 | -0.02% | [-0.12%, +0.09%] |
| real_world / development | classification | linear | openfe-stateless | 5/5 / 5/5 | 30/30 | 2/3/0 | +1.45% | [-0.14%, +3.75%] |
| real_world / development | classification | linear | polynomial | 5/5 / 5/5 | 30/30 | 3/2/0 | +1.16% | [-0.24%, +2.90%] |
| real_world / development | classification | xgboost | autofeat | 3/5 / 3/5 | 18/30 | 0/3/0 | -0.29% | [-0.93%, +0.57%] |
| real_world / development | classification | xgboost | baseline | 5/5 / 5/5 | 30/30 | 0/5/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | classification | xgboost | featcopilot | 5/5 / 5/5 | 30/30 | 0/5/0 | +0.03% | [-0.00%, +0.09%] |
| real_world / development | classification | xgboost | featcopilot-predictive | 5/5 / 5/5 | 30/30 | 1/4/0 | +0.40% | [-0.09%, +1.31%] |
| real_world / development | classification | xgboost | featcopilot-task | 5/5 / 5/5 | 30/30 | 0/5/0 | +0.03% | [-0.00%, +0.09%] |
| real_world / development | classification | xgboost | openfe-stateless | 5/5 / 5/5 | 30/30 | 2/1/2 | -0.20% | [-2.03%, +1.62%] |
| real_world / development | classification | xgboost | polynomial | 5/5 / 5/5 | 30/30 | 0/1/4 | -2.31% | [-3.57%, -1.26%] |
| real_world / development | regression | catboost | autofeat | 3/3 / 3/3 | 18/18 | 3/0/0 | +2.47% | [+1.01%, +4.41%] |
| real_world / development | regression | catboost | baseline | 3/3 / 3/3 | 18/18 | 0/3/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | regression | catboost | featcopilot | 3/3 / 3/3 | 18/18 | 0/3/0 | +0.01% | [+0.00%, +0.03%] |
| real_world / development | regression | catboost | featcopilot-predictive | 3/3 / 3/3 | 18/18 | 1/2/0 | +1.63% | [+0.00%, +4.50%] |
| real_world / development | regression | catboost | featcopilot-task | 3/3 / 3/3 | 18/18 | 0/3/0 | +0.07% | [+0.00%, +0.20%] |
| real_world / development | regression | catboost | openfe-stateless | 3/3 / 3/3 | 18/18 | 2/1/0 | +2.69% | [+0.74%, +6.15%] |
| real_world / development | regression | catboost | polynomial | 3/3 / 3/3 | 18/18 | 0/3/0 | +0.01% | [-0.75%, +0.52%] |
| real_world / development | regression | lightgbm | autofeat | 3/3 / 3/3 | 18/18 | 2/0/1 | +0.68% | [-4.21%, +5.16%] |
| real_world / development | regression | lightgbm | baseline | 3/3 / 3/3 | 18/18 | 0/3/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | regression | lightgbm | featcopilot | 3/3 / 3/3 | 18/18 | 0/3/0 | -0.21% | [-0.90%, +0.26%] |
| real_world / development | regression | lightgbm | featcopilot-predictive | 3/3 / 3/3 | 18/18 | 0/3/0 | +0.17% | [-0.46%, +0.62%] |
| real_world / development | regression | lightgbm | featcopilot-task | 3/3 / 3/3 | 18/18 | 0/3/0 | -0.37% | [-0.90%, +0.00%] |
| real_world / development | regression | lightgbm | openfe-stateless | 3/3 / 3/3 | 18/18 | 2/0/1 | +1.35% | [-3.92%, +6.42%] |
| real_world / development | regression | lightgbm | polynomial | 3/3 / 3/3 | 18/18 | 1/0/2 | -0.92% | [-2.97%, +1.68%] |
| real_world / development | regression | linear | autofeat | 3/3 / 3/3 | 18/18 | 2/0/1 | -2.47% | [-49.69%, +38.63%] |
| real_world / development | regression | linear | baseline | 3/3 / 3/3 | 18/18 | 0/3/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | regression | linear | featcopilot | 3/3 / 3/3 | 18/18 | 1/2/0 | +1.74% | [+0.00%, +5.19%] |
| real_world / development | regression | linear | featcopilot-predictive | 3/3 / 3/3 | 18/18 | 2/0/1 | +5.71% | [-1.05%, +10.75%] |
| real_world / development | regression | linear | featcopilot-task | 3/3 / 3/3 | 18/18 | 1/2/0 | +1.75% | [+0.00%, +5.19%] |
| real_world / development | regression | linear | openfe-stateless | 3/3 / 3/3 | 18/18 | 2/0/1 | +10.48% | [-3.02%, +31.82%] |
| real_world / development | regression | linear | polynomial | 3/3 / 3/3 | 18/18 | 3/0/0 | +11.68% | [+1.73%, +25.39%] |
| real_world / development | regression | xgboost | autofeat | 3/3 / 3/3 | 18/18 | 2/1/0 | +2.79% | [+0.22%, +6.27%] |
| real_world / development | regression | xgboost | baseline | 3/3 / 3/3 | 18/18 | 0/3/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | regression | xgboost | featcopilot | 3/3 / 3/3 | 18/18 | 0/2/1 | -0.30% | [-1.08%, +0.18%] |
| real_world / development | regression | xgboost | featcopilot-predictive | 3/3 / 3/3 | 18/18 | 1/2/0 | +0.65% | [-0.01%, +1.94%] |
| real_world / development | regression | xgboost | featcopilot-task | 3/3 / 3/3 | 18/18 | 0/2/1 | -0.34% | [-1.08%, +0.05%] |
| real_world / development | regression | xgboost | openfe-stateless | 3/3 / 3/3 | 18/18 | 2/1/0 | +2.93% | [-0.49%, +8.02%] |
| real_world / development | regression | xgboost | polynomial | 3/3 / 3/3 | 18/18 | 1/2/0 | +0.27% | [-0.55%, +1.60%] |
| simulation / development | regression | catboost | autofeat | 1/1 / 1/1 | 6/6 | 1/0/0 | +3.34% | n/a |
| simulation / development | regression | catboost | baseline | 1/1 / 1/1 | 6/6 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | catboost | featcopilot | 1/1 / 1/1 | 6/6 | 1/0/0 | +1.31% | n/a |
| simulation / development | regression | catboost | featcopilot-predictive | 1/1 / 1/1 | 6/6 | 0/1/0 | -0.42% | n/a |
| simulation / development | regression | catboost | featcopilot-task | 1/1 / 1/1 | 6/6 | 1/0/0 | +1.31% | n/a |
| simulation / development | regression | catboost | openfe-stateless | 1/1 / 1/1 | 6/6 | 1/0/0 | +2.72% | n/a |
| simulation / development | regression | catboost | polynomial | 1/1 / 1/1 | 6/6 | 1/0/0 | +3.40% | n/a |
| simulation / development | regression | lightgbm | autofeat | 1/1 / 1/1 | 6/6 | 0/0/1 | -2.65% | n/a |
| simulation / development | regression | lightgbm | baseline | 1/1 / 1/1 | 6/6 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | lightgbm | featcopilot | 1/1 / 1/1 | 6/6 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | lightgbm | featcopilot-predictive | 1/1 / 1/1 | 6/6 | 0/1/0 | -0.98% | n/a |
| simulation / development | regression | lightgbm | featcopilot-task | 1/1 / 1/1 | 6/6 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | lightgbm | openfe-stateless | 1/1 / 1/1 | 6/6 | 1/0/0 | +4.88% | n/a |
| simulation / development | regression | lightgbm | polynomial | 1/1 / 1/1 | 6/6 | 0/0/1 | -3.71% | n/a |
| simulation / development | regression | linear | autofeat | 1/1 / 1/1 | 6/6 | 1/0/0 | +24.56% | n/a |
| simulation / development | regression | linear | baseline | 1/1 / 1/1 | 6/6 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | linear | featcopilot | 1/1 / 1/1 | 6/6 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | linear | featcopilot-predictive | 1/1 / 1/1 | 6/6 | 1/0/0 | +34.06% | n/a |
| simulation / development | regression | linear | featcopilot-task | 1/1 / 1/1 | 6/6 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | linear | openfe-stateless | 1/1 / 1/1 | 6/6 | 1/0/0 | +22.60% | n/a |
| simulation / development | regression | linear | polynomial | 1/1 / 1/1 | 6/6 | 1/0/0 | +25.49% | n/a |
| simulation / development | regression | xgboost | autofeat | 1/1 / 1/1 | 6/6 | 0/0/1 | -3.08% | n/a |
| simulation / development | regression | xgboost | baseline | 1/1 / 1/1 | 6/6 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | xgboost | featcopilot | 1/1 / 1/1 | 6/6 | 0/1/0 | +0.65% | n/a |
| simulation / development | regression | xgboost | featcopilot-predictive | 1/1 / 1/1 | 6/6 | 0/1/0 | +0.81% | n/a |
| simulation / development | regression | xgboost | featcopilot-task | 1/1 / 1/1 | 6/6 | 0/1/0 | +0.65% | n/a |
| simulation / development | regression | xgboost | openfe-stateless | 1/1 / 1/1 | 6/6 | 0/0/1 | -2.47% | n/a |
| simulation / development | regression | xgboost | polynomial | 1/1 / 1/1 | 6/6 | 0/0/1 | -3.25% | n/a |

## Per-dataset means (complete folds only)

| Dataset | Learner | Method | Primary loss | FE seconds (inner + outer) | Pipeline seconds | Features |
|---|---|---|---|---|---|---|
| houses | catboost | autofeat | 0.238647 | 35.34 | 47.45 | 31.7 |
| houses | catboost | baseline | 0.249658 | 0.02 | 5.43 | 8.0 |
| houses | catboost | featcopilot | 0.249658 | 12.14 | 17.30 | 8.0 |
| houses | catboost | featcopilot-predictive | 0.238424 | 53.85 | 62.44 | 23.8 |
| houses | catboost | featcopilot-task | 0.249658 | 12.33 | 17.64 | 8.0 |
| houses | catboost | openfe-stateless | 0.234299 | 55.98 | 75.02 | 58.0 |
| houses | catboost | polynomial | 0.251534 | 0.04 | 16.21 | 44.0 |
| houses | lightgbm | autofeat | 0.233427 | 34.55 | 35.72 | 31.5 |
| houses | lightgbm | baseline | 0.246137 | 0.02 | 1.11 | 8.0 |
| houses | lightgbm | featcopilot | 0.246137 | 12.27 | 13.29 | 8.0 |
| houses | lightgbm | featcopilot-predictive | 0.244603 | 18.85 | 19.53 | 13.8 |
| houses | lightgbm | featcopilot-task | 0.246137 | 11.65 | 12.66 | 8.0 |
| houses | lightgbm | openfe-stateless | 0.230344 | 56.62 | 57.80 | 58.0 |
| houses | lightgbm | polynomial | 0.249722 | 0.04 | 1.64 | 44.0 |
| houses | linear | autofeat | 0.507063 | 35.23 | 35.30 | 31.0 |
| houses | linear | baseline | 0.338740 | 0.02 | 0.08 | 8.0 |
| houses | linear | featcopilot | 0.338740 | 12.60 | 12.64 | 8.0 |
| houses | linear | featcopilot-predictive | 0.313565 | 3.01 | 3.09 | 15.5 |
| houses | linear | featcopilot-task | 0.338740 | 12.68 | 12.72 | 8.0 |
| houses | linear | openfe-stateless | 0.348967 | 56.15 | 56.26 | 58.0 |
| houses | linear | polynomial | 0.311927 | 0.04 | 0.13 | 44.0 |
| houses | xgboost | autofeat | 0.233752 | 35.01 | 37.79 | 31.0 |
| houses | xgboost | baseline | 0.249377 | 0.02 | 1.25 | 8.0 |
| houses | xgboost | featcopilot | 0.249377 | 11.94 | 13.17 | 8.0 |
| houses | xgboost | featcopilot-predictive | 0.244543 | 30.77 | 32.71 | 22.3 |
| houses | xgboost | featcopilot-task | 0.249377 | 12.27 | 13.55 | 8.0 |
| houses | xgboost | openfe-stateless | 0.229388 | 56.31 | 61.16 | 58.0 |
| houses | xgboost | polynomial | 0.250757 | 0.04 | 3.78 | 44.0 |
| uci-adult | catboost | autofeat | 0.316553 | 116.28 | 167.18 | 21.5 |
| uci-adult | catboost | baseline | 0.313477 | 0.06 | 48.60 | 13.0 |
| uci-adult | catboost | featcopilot | 0.314143 | 7.14 | 54.89 | 15.5 |
| uci-adult | catboost | featcopilot-predictive | 0.313477 | 48.33 | 93.79 | 13.0 |
| uci-adult | catboost | featcopilot-task | 0.314143 | 6.93 | 53.98 | 15.5 |
| uci-adult | catboost | openfe-stateless | 0.318939 | 56.11 | 111.81 | 60.0 |
| uci-adult | catboost | polynomial | 0.317871 | 0.09 | 51.07 | 28.0 |
| uci-adult | lightgbm | autofeat | 0.316713 | 111.17 | 112.35 | 21.2 |
| uci-adult | lightgbm | baseline | 0.309933 | 0.06 | 1.29 | 13.0 |
| uci-adult | lightgbm | featcopilot | 0.309933 | 6.91 | 8.19 | 15.5 |
| uci-adult | lightgbm | featcopilot-predictive | 0.309314 | 17.19 | 18.00 | 13.3 |
| uci-adult | lightgbm | featcopilot-task | 0.309933 | 7.04 | 8.37 | 15.5 |
| uci-adult | lightgbm | openfe-stateless | 0.319018 | 55.79 | 56.95 | 60.0 |
| uci-adult | lightgbm | polynomial | 0.317243 | 0.09 | 1.51 | 28.0 |
| uci-adult | linear | autofeat | 0.332964 | 113.00 | 113.33 | 21.5 |
| uci-adult | linear | baseline | 0.337434 | 0.06 | 0.36 | 13.0 |
| uci-adult | linear | featcopilot | 0.336869 | 6.92 | 7.23 | 15.5 |
| uci-adult | linear | featcopilot-predictive | 0.338041 | 49.93 | 50.28 | 13.2 |
| uci-adult | linear | featcopilot-task | 0.336869 | 6.93 | 7.23 | 15.5 |
| uci-adult | linear | openfe-stateless | 0.336708 | 54.04 | 54.56 | 60.0 |
| uci-adult | linear | polynomial | 0.333546 | 0.09 | 0.46 | 28.0 |
| uci-adult | xgboost | autofeat | 0.313076 | 113.08 | 114.30 | 21.5 |
| uci-adult | xgboost | baseline | 0.310181 | 0.06 | 0.95 | 13.0 |
| uci-adult | xgboost | featcopilot | 0.310181 | 7.16 | 8.20 | 15.5 |
| uci-adult | xgboost | featcopilot-predictive | 0.310662 | 18.92 | 19.76 | 14.7 |
| uci-adult | xgboost | featcopilot-task | 0.310181 | 6.94 | 7.84 | 15.5 |
| uci-adult | xgboost | openfe-stateless | 0.317250 | 54.24 | 56.08 | 60.0 |
| uci-adult | xgboost | polynomial | 0.315305 | 0.09 | 1.29 | 28.0 |
| uci-concrete-strength | catboost | autofeat | 4.547339 | 33.92 | 40.01 | 22.8 |
| uci-concrete-strength | catboost | baseline | 4.640088 | 0.02 | 3.40 | 8.0 |
| uci-concrete-strength | catboost | featcopilot | 4.639672 | 4.25 | 7.68 | 8.3 |
| uci-concrete-strength | catboost | featcopilot-predictive | 4.621483 | 34.58 | 36.75 | 8.0 |
| uci-concrete-strength | catboost | featcopilot-task | 4.639672 | 4.22 | 7.66 | 8.3 |
| uci-concrete-strength | catboost | openfe-stateless | 4.605908 | 55.55 | 68.15 | 58.0 |
| uci-concrete-strength | catboost | polynomial | 4.628595 | 0.04 | 8.41 | 44.0 |
| uci-concrete-strength | lightgbm | autofeat | 4.744836 | 34.92 | 35.70 | 22.3 |
| uci-concrete-strength | lightgbm | baseline | 4.553248 | 0.02 | 0.92 | 8.0 |
| uci-concrete-strength | lightgbm | featcopilot | 4.594321 | 4.20 | 5.01 | 8.3 |
| uci-concrete-strength | lightgbm | featcopilot-predictive | 4.574290 | 11.70 | 12.17 | 16.3 |
| uci-concrete-strength | lightgbm | featcopilot-task | 4.594321 | 4.44 | 5.26 | 8.3 |
| uci-concrete-strength | lightgbm | openfe-stateless | 4.731527 | 56.13 | 56.75 | 58.0 |
| uci-concrete-strength | lightgbm | polynomial | 4.688414 | 0.04 | 1.12 | 44.0 |
| uci-concrete-strength | linear | autofeat | 6.466043 | 34.66 | 34.70 | 23.0 |
| uci-concrete-strength | linear | baseline | 10.536313 | 0.02 | 0.07 | 8.0 |
| uci-concrete-strength | linear | featcopilot | 9.989839 | 4.26 | 4.30 | 8.3 |
| uci-concrete-strength | linear | featcopilot-predictive | 9.403466 | 2.59 | 2.64 | 35.0 |
| uci-concrete-strength | linear | featcopilot-task | 9.989839 | 4.28 | 4.32 | 8.3 |
| uci-concrete-strength | linear | openfe-stateless | 7.183593 | 56.18 | 56.23 | 58.0 |
| uci-concrete-strength | linear | polynomial | 7.861125 | 0.04 | 0.08 | 44.0 |
| uci-concrete-strength | xgboost | autofeat | 4.812138 | 34.18 | 35.61 | 22.3 |
| uci-concrete-strength | xgboost | baseline | 4.822602 | 0.02 | 0.88 | 8.0 |
| uci-concrete-strength | xgboost | featcopilot | 4.874598 | 4.18 | 5.03 | 8.3 |
| uci-concrete-strength | xgboost | featcopilot-predictive | 4.821767 | 20.66 | 21.44 | 9.0 |
| uci-concrete-strength | xgboost | featcopilot-task | 4.874598 | 4.23 | 5.08 | 8.3 |
| uci-concrete-strength | xgboost | openfe-stateless | 4.846150 | 56.08 | 58.75 | 58.0 |
| uci-concrete-strength | xgboost | polynomial | 4.834286 | 0.04 | 1.93 | 44.0 |
| uci-dry-bean | catboost | baseline | 0.211585 | 0.04 | 39.36 | 16.0 |
| uci-dry-bean | catboost | featcopilot | 0.209533 | 4.22 | 36.87 | 16.0 |
| uci-dry-bean | catboost | featcopilot-predictive | 0.212098 | 326.33 | 367.95 | 24.3 |
| uci-dry-bean | catboost | featcopilot-task | 0.209533 | 4.35 | 37.09 | 16.0 |
| uci-dry-bean | catboost | openfe-stateless | 0.205597 | 69.55 | 181.31 | 66.0 |
| uci-dry-bean | catboost | polynomial | 0.211710 | 0.08 | 163.86 | 66.0 |
| uci-dry-bean | lightgbm | baseline | 0.245189 | 0.04 | 3.86 | 16.0 |
| uci-dry-bean | lightgbm | featcopilot | 0.244917 | 4.26 | 7.91 | 16.0 |
| uci-dry-bean | lightgbm | featcopilot-predictive | 0.245189 | 88.70 | 91.81 | 16.0 |
| uci-dry-bean | lightgbm | featcopilot-task | 0.244917 | 4.26 | 7.88 | 16.0 |
| uci-dry-bean | lightgbm | openfe-stateless | 0.251926 | 68.47 | 73.33 | 66.0 |
| uci-dry-bean | lightgbm | polynomial | 0.253877 | 0.08 | 5.47 | 66.0 |
| uci-dry-bean | linear | baseline | 0.216617 | 0.04 | 0.43 | 16.0 |
| uci-dry-bean | linear | featcopilot | 0.216617 | 4.27 | 4.64 | 16.0 |
| uci-dry-bean | linear | featcopilot-predictive | 0.214239 | 51.60 | 52.03 | 21.0 |
| uci-dry-bean | linear | featcopilot-task | 0.216617 | 4.24 | 4.61 | 16.0 |
| uci-dry-bean | linear | openfe-stateless | 0.217392 | 68.69 | 69.57 | 66.0 |
| uci-dry-bean | linear | polynomial | 0.216806 | 0.08 | 0.98 | 66.0 |
| uci-dry-bean | xgboost | baseline | 0.223610 | 0.04 | 4.54 | 16.0 |
| uci-dry-bean | xgboost | featcopilot | 0.223293 | 4.23 | 8.65 | 16.0 |
| uci-dry-bean | xgboost | featcopilot-predictive | 0.223610 | 121.74 | 126.02 | 16.0 |
| uci-dry-bean | xgboost | featcopilot-task | 0.223293 | 4.35 | 8.72 | 16.0 |
| uci-dry-bean | xgboost | openfe-stateless | 0.220007 | 68.78 | 78.11 | 66.0 |
| uci-dry-bean | xgboost | polynomial | 0.226686 | 0.08 | 10.57 | 66.0 |
| uci-energy-efficiency | catboost | autofeat | 0.484408 | 29.30 | 67.02 | 18.3 |
| uci-energy-efficiency | catboost | baseline | 0.501127 | 0.02 | 34.58 | 8.0 |
| uci-energy-efficiency | catboost | featcopilot | 0.494581 | 1.33 | 34.90 | 8.0 |
| uci-energy-efficiency | catboost | featcopilot-predictive | 0.503230 | 19.82 | 52.70 | 17.2 |
| uci-energy-efficiency | catboost | featcopilot-task | 0.494581 | 1.46 | 36.87 | 8.0 |
| uci-energy-efficiency | catboost | openfe-stateless | 0.487472 | 53.81 | 90.25 | 43.3 |
| uci-energy-efficiency | catboost | polynomial | 0.484084 | 0.03 | 37.90 | 29.0 |
| uci-energy-efficiency | lightgbm | autofeat | 0.509861 | 28.49 | 29.12 | 17.3 |
| uci-energy-efficiency | lightgbm | baseline | 0.496701 | 0.02 | 0.86 | 8.0 |
| uci-energy-efficiency | lightgbm | featcopilot | 0.496701 | 1.63 | 2.51 | 8.0 |
| uci-energy-efficiency | lightgbm | featcopilot-predictive | 0.501569 | 9.34 | 9.69 | 8.0 |
| uci-energy-efficiency | lightgbm | featcopilot-task | 0.496701 | 1.46 | 2.19 | 8.0 |
| uci-energy-efficiency | lightgbm | openfe-stateless | 0.472445 | 53.26 | 53.60 | 43.3 |
| uci-energy-efficiency | lightgbm | polynomial | 0.515124 | 0.03 | 0.90 | 29.0 |
| uci-energy-efficiency | linear | autofeat | 2.150119 | 29.43 | 29.48 | 17.8 |
| uci-energy-efficiency | linear | baseline | 2.850262 | 0.02 | 0.08 | 8.0 |
| uci-energy-efficiency | linear | featcopilot | 2.850262 | 1.35 | 1.41 | 8.0 |
| uci-energy-efficiency | linear | featcopilot-predictive | 1.879492 | 2.78 | 2.86 | 33.8 |
| uci-energy-efficiency | linear | featcopilot-task | 2.850262 | 1.35 | 1.40 | 8.0 |
| uci-energy-efficiency | linear | openfe-stateless | 2.205962 | 52.45 | 52.51 | 43.3 |
| uci-energy-efficiency | linear | polynomial | 2.123600 | 0.03 | 0.09 | 29.0 |
| uci-energy-efficiency | xgboost | autofeat | 0.439343 | 28.17 | 28.82 | 16.3 |
| uci-energy-efficiency | xgboost | baseline | 0.426229 | 0.02 | 0.69 | 8.0 |
| uci-energy-efficiency | xgboost | featcopilot | 0.423469 | 1.58 | 2.29 | 8.0 |
| uci-energy-efficiency | xgboost | featcopilot-predictive | 0.422777 | 14.80 | 15.42 | 12.3 |
| uci-energy-efficiency | xgboost | featcopilot-task | 0.423469 | 1.34 | 1.99 | 8.0 |
| uci-energy-efficiency | xgboost | openfe-stateless | 0.436763 | 52.24 | 52.90 | 43.3 |
| uci-energy-efficiency | xgboost | polynomial | 0.440073 | 0.03 | 0.74 | 29.0 |
| uci-german-credit | catboost | autofeat | 0.508798 | 69.06 | 118.62 | 31.8 |
| uci-german-credit | catboost | baseline | 0.507292 | 0.04 | 48.55 | 20.0 |
| uci-german-credit | catboost | featcopilot | 0.508366 | 4.55 | 53.48 | 22.0 |
| uci-german-credit | catboost | featcopilot-predictive | 0.507292 | 51.87 | 98.32 | 20.0 |
| uci-german-credit | catboost | featcopilot-task | 0.508366 | 4.70 | 54.19 | 22.0 |
| uci-german-credit | catboost | openfe-stateless | 0.510838 | 55.45 | 115.71 | 69.5 |
| uci-german-credit | catboost | polynomial | 0.511313 | 0.06 | 57.19 | 48.0 |
| uci-german-credit | lightgbm | autofeat | 0.531450 | 67.62 | 68.52 | 27.7 |
| uci-german-credit | lightgbm | baseline | 0.517633 | 0.04 | 1.05 | 20.0 |
| uci-german-credit | lightgbm | featcopilot | 0.517208 | 4.55 | 5.53 | 22.0 |
| uci-german-credit | lightgbm | featcopilot-predictive | 0.518757 | 12.88 | 13.51 | 20.3 |
| uci-german-credit | lightgbm | featcopilot-task | 0.517208 | 4.55 | 5.47 | 22.0 |
| uci-german-credit | lightgbm | openfe-stateless | 0.530133 | 56.01 | 56.68 | 69.5 |
| uci-german-credit | lightgbm | polynomial | 0.536921 | 0.06 | 1.13 | 48.0 |
| uci-german-credit | linear | autofeat | 0.497693 | 69.47 | 69.67 | 30.7 |
| uci-german-credit | linear | baseline | 0.498642 | 0.04 | 0.25 | 20.0 |
| uci-german-credit | linear | featcopilot | 0.499649 | 4.49 | 4.67 | 22.0 |
| uci-german-credit | linear | featcopilot-predictive | 0.500325 | 44.88 | 45.15 | 21.7 |
| uci-german-credit | linear | featcopilot-task | 0.499649 | 4.59 | 4.78 | 22.0 |
| uci-german-credit | linear | openfe-stateless | 0.499197 | 54.72 | 55.01 | 69.5 |
| uci-german-credit | linear | polynomial | 0.502424 | 0.06 | 0.30 | 48.0 |
| uci-german-credit | xgboost | autofeat | 0.516854 | 70.70 | 71.48 | 27.7 |
| uci-german-credit | xgboost | baseline | 0.519836 | 0.04 | 0.72 | 20.0 |
| uci-german-credit | xgboost | featcopilot | 0.519836 | 4.54 | 5.20 | 22.0 |
| uci-german-credit | xgboost | featcopilot-predictive | 0.519836 | 15.56 | 16.25 | 20.0 |
| uci-german-credit | xgboost | featcopilot-task | 0.519836 | 4.67 | 5.38 | 22.0 |
| uci-german-credit | xgboost | openfe-stateless | 0.532180 | 54.45 | 55.61 | 69.5 |
| uci-german-credit | xgboost | polynomial | 0.536344 | 0.06 | 1.08 | 48.0 |
| uci-online-shoppers | catboost | autofeat | 0.379893 | 53.10 | 101.08 | 11.7 |
| uci-online-shoppers | catboost | baseline | 0.379654 | 0.05 | 46.00 | 11.0 |
| uci-online-shoppers | catboost | featcopilot | 0.379654 | 6.00 | 51.82 | 11.0 |
| uci-online-shoppers | catboost | featcopilot-predictive | 0.379654 | 48.13 | 93.74 | 11.0 |
| uci-online-shoppers | catboost | featcopilot-task | 0.379654 | 6.04 | 52.60 | 11.0 |
| uci-online-shoppers | catboost | openfe-stateless | 0.382939 | 55.27 | 110.69 | 46.7 |
| uci-online-shoppers | catboost | polynomial | 0.381032 | 0.07 | 49.17 | 21.0 |
| uci-online-shoppers | lightgbm | autofeat | 0.381605 | 52.97 | 54.01 | 11.7 |
| uci-online-shoppers | lightgbm | baseline | 0.379942 | 0.05 | 1.24 | 11.0 |
| uci-online-shoppers | lightgbm | featcopilot | 0.379942 | 6.05 | 7.19 | 11.0 |
| uci-online-shoppers | lightgbm | featcopilot-predictive | 0.379942 | 16.84 | 17.61 | 11.0 |
| uci-online-shoppers | lightgbm | featcopilot-task | 0.379942 | 6.07 | 7.35 | 11.0 |
| uci-online-shoppers | lightgbm | openfe-stateless | 0.384249 | 56.30 | 57.34 | 46.7 |
| uci-online-shoppers | lightgbm | polynomial | 0.383083 | 0.07 | 1.37 | 21.0 |
| uci-online-shoppers | linear | autofeat | 0.394913 | 52.34 | 52.61 | 11.7 |
| uci-online-shoppers | linear | baseline | 0.394934 | 0.05 | 0.31 | 11.0 |
| uci-online-shoppers | linear | featcopilot | 0.394934 | 6.17 | 6.41 | 11.0 |
| uci-online-shoppers | linear | featcopilot-predictive | 0.386222 | 48.53 | 48.82 | 12.7 |
| uci-online-shoppers | linear | featcopilot-task | 0.394934 | 6.03 | 6.27 | 11.0 |
| uci-online-shoppers | linear | openfe-stateless | 0.388335 | 55.69 | 56.11 | 46.7 |
| uci-online-shoppers | linear | polynomial | 0.390906 | 0.07 | 0.35 | 21.0 |
| uci-online-shoppers | xgboost | autofeat | 0.380738 | 52.25 | 53.12 | 12.0 |
| uci-online-shoppers | xgboost | baseline | 0.378759 | 0.05 | 0.94 | 11.0 |
| uci-online-shoppers | xgboost | featcopilot | 0.378759 | 6.10 | 7.02 | 11.0 |
| uci-online-shoppers | xgboost | featcopilot-predictive | 0.378759 | 17.31 | 18.06 | 11.8 |
| uci-online-shoppers | xgboost | featcopilot-task | 0.378759 | 6.05 | 6.87 | 11.0 |
| uci-online-shoppers | xgboost | openfe-stateless | 0.381974 | 53.50 | 54.95 | 46.7 |
| uci-online-shoppers | xgboost | polynomial | 0.382313 | 0.07 | 1.16 | 21.0 |
| uci-steel-faults | catboost | baseline | 0.568000 | 0.07 | 42.89 | 27.0 |
| uci-steel-faults | catboost | featcopilot | 0.565174 | 6.18 | 44.77 | 27.3 |
| uci-steel-faults | catboost | featcopilot-predictive | 0.540553 | 322.72 | 368.46 | 33.7 |
| uci-steel-faults | catboost | featcopilot-task | 0.565174 | 6.23 | 44.66 | 27.3 |
| uci-steel-faults | catboost | openfe-stateless | 0.547937 | 79.68 | 188.32 | 77.0 |
| uci-steel-faults | catboost | polynomial | 0.591117 | 0.10 | 136.82 | 77.0 |
| uci-steel-faults | lightgbm | baseline | 0.554473 | 0.07 | 3.61 | 27.0 |
| uci-steel-faults | lightgbm | featcopilot | 0.555357 | 6.29 | 9.79 | 27.3 |
| uci-steel-faults | lightgbm | featcopilot-predictive | 0.551653 | 80.54 | 83.70 | 30.3 |
| uci-steel-faults | lightgbm | featcopilot-task | 0.555357 | 6.20 | 9.64 | 27.3 |
| uci-steel-faults | lightgbm | openfe-stateless | 0.560844 | 78.20 | 82.37 | 77.0 |
| uci-steel-faults | lightgbm | polynomial | 0.588347 | 0.11 | 5.01 | 77.0 |
| uci-steel-faults | linear | baseline | 0.777794 | 0.07 | 0.53 | 27.0 |
| uci-steel-faults | linear | featcopilot | 0.778191 | 6.19 | 6.63 | 27.3 |
| uci-steel-faults | linear | featcopilot-predictive | 0.737326 | 52.33 | 52.93 | 38.2 |
| uci-steel-faults | linear | featcopilot-task | 0.778191 | 6.27 | 6.71 | 27.3 |
| uci-steel-faults | linear | openfe-stateless | 0.732542 | 77.90 | 78.77 | 77.0 |
| uci-steel-faults | linear | polynomial | 0.742922 | 0.10 | 1.00 | 77.0 |
| uci-steel-faults | xgboost | baseline | 0.546028 | 0.07 | 4.81 | 27.0 |
| uci-steel-faults | xgboost | featcopilot | 0.546068 | 6.23 | 10.97 | 27.3 |
| uci-steel-faults | xgboost | featcopilot-predictive | 0.534125 | 117.69 | 122.95 | 33.0 |
| uci-steel-faults | xgboost | featcopilot-task | 0.546068 | 6.21 | 10.99 | 27.3 |
| uci-steel-faults | xgboost | openfe-stateless | 0.530355 | 78.53 | 87.35 | 77.0 |
| uci-steel-faults | xgboost | polynomial | 0.570194 | 0.11 | 9.94 | 77.0 |
| wine | catboost | autofeat | 0.641301 | 41.57 | 56.44 | 44.5 |
| wine | catboost | baseline | 0.647849 | 0.03 | 5.87 | 11.0 |
| wine | catboost | featcopilot | 0.647647 | 9.33 | 15.15 | 11.8 |
| wine | catboost | featcopilot-predictive | 0.647849 | 55.69 | 60.50 | 11.0 |
| wine | catboost | featcopilot-task | 0.646539 | 12.48 | 19.09 | 13.8 |
| wine | catboost | openfe-stateless | 0.640144 | 56.09 | 76.45 | 61.0 |
| wine | catboost | polynomial | 0.644467 | 0.05 | 20.91 | 61.0 |
| wine | lightgbm | autofeat | 0.645233 | 43.53 | 44.82 | 52.0 |
| wine | lightgbm | baseline | 0.652311 | 0.03 | 1.11 | 11.0 |
| wine | lightgbm | featcopilot | 0.650608 | 9.29 | 10.33 | 11.8 |
| wine | lightgbm | featcopilot-predictive | 0.649981 | 22.81 | 23.46 | 11.0 |
| wine | lightgbm | featcopilot-task | 0.653684 | 12.70 | 13.78 | 13.8 |
| wine | lightgbm | openfe-stateless | 0.642139 | 56.86 | 58.13 | 61.0 |
| wine | lightgbm | polynomial | 0.641383 | 0.06 | 1.83 | 61.0 |
| wine | linear | autofeat | 0.710603 | 40.83 | 40.90 | 40.7 |
| wine | linear | baseline | 0.737438 | 0.03 | 0.09 | 11.0 |
| wine | linear | featcopilot | 0.737134 | 9.33 | 9.38 | 11.8 |
| wine | linear | featcopilot-predictive | 0.745215 | 3.74 | 3.82 | 20.5 |
| wine | linear | featcopilot-task | 0.736980 | 13.07 | 13.12 | 13.8 |
| wine | linear | openfe-stateless | 0.717963 | 57.48 | 57.59 | 61.0 |
| wine | linear | polynomial | 0.724653 | 0.05 | 0.16 | 61.0 |
| wine | xgboost | autofeat | 0.638029 | 41.97 | 45.63 | 44.3 |
| wine | xgboost | baseline | 0.650277 | 0.03 | 1.33 | 11.0 |
| wine | xgboost | featcopilot | 0.649080 | 9.29 | 10.72 | 11.8 |
| wine | xgboost | featcopilot-predictive | 0.650335 | 32.93 | 34.48 | 11.0 |
| wine | xgboost | featcopilot-task | 0.649960 | 13.01 | 14.48 | 13.8 |
| wine | xgboost | openfe-stateless | 0.641981 | 57.49 | 62.27 | 61.0 |
| wine | xgboost | polynomial | 0.639845 | 0.05 | 5.97 | 61.0 |

## Failures / timeouts

- uci-dry-bean / 42 / fold 0 / autofeat / xgboost: timeout: Case exceeded 600.0s wall-clock limit
- uci-dry-bean / 42 / fold 0 / autofeat / catboost: timeout: Case exceeded 600.0s wall-clock limit
- uci-dry-bean / 42 / fold 0 / autofeat / lightgbm: timeout: Case exceeded 600.0s wall-clock limit
- uci-steel-faults / 42 / fold 0 / autofeat / catboost: timeout: Case exceeded 600.0s wall-clock limit
- uci-steel-faults / 42 / fold 0 / autofeat / lightgbm: timeout: Case exceeded 600.0s wall-clock limit
- uci-steel-faults / 42 / fold 0 / autofeat / linear: timeout: Case exceeded 600.0s wall-clock limit
- uci-dry-bean / 42 / fold 0 / autofeat / linear: timeout: Case exceeded 600.0s wall-clock limit
- uci-steel-faults / 42 / fold 0 / autofeat / xgboost: timeout: Case exceeded 600.0s wall-clock limit
- uci-steel-faults / 137 / fold 0 / autofeat / lightgbm: timeout: Case exceeded 600.0s wall-clock limit
- uci-dry-bean / 137 / fold 0 / autofeat / xgboost: timeout: Case exceeded 600.0s wall-clock limit
- uci-dry-bean / 137 / fold 0 / autofeat / catboost: timeout: Case exceeded 600.0s wall-clock limit
- uci-steel-faults / 137 / fold 0 / autofeat / xgboost: timeout: Case exceeded 600.0s wall-clock limit
- uci-steel-faults / 137 / fold 0 / autofeat / catboost: timeout: Case exceeded 600.0s wall-clock limit
- uci-dry-bean / 137 / fold 0 / autofeat / lightgbm: timeout: Case exceeded 600.0s wall-clock limit
- uci-dry-bean / 137 / fold 0 / autofeat / linear: timeout: Case exceeded 600.0s wall-clock limit
- uci-steel-faults / 137 / fold 0 / autofeat / linear: timeout: Case exceeded 600.0s wall-clock limit
- uci-steel-faults / 42 / fold 1 / autofeat / catboost: timeout: Case exceeded 600.0s wall-clock limit
- uci-steel-faults / 42 / fold 1 / autofeat / linear: timeout: Case exceeded 600.0s wall-clock limit
- uci-steel-faults / 42 / fold 1 / autofeat / lightgbm: timeout: Case exceeded 600.0s wall-clock limit
- uci-dry-bean / 42 / fold 1 / autofeat / linear: timeout: Case exceeded 600.0s wall-clock limit
- uci-dry-bean / 42 / fold 1 / autofeat / lightgbm: timeout: Case exceeded 600.0s wall-clock limit
- uci-steel-faults / 42 / fold 1 / autofeat / xgboost: timeout: Case exceeded 600.0s wall-clock limit
- uci-dry-bean / 42 / fold 1 / autofeat / xgboost: timeout: Case exceeded 600.0s wall-clock limit
- uci-dry-bean / 42 / fold 1 / autofeat / catboost: timeout: Case exceeded 600.0s wall-clock limit
- uci-steel-faults / 137 / fold 1 / autofeat / xgboost: timeout: Case exceeded 600.0s wall-clock limit
- uci-steel-faults / 137 / fold 1 / autofeat / linear: timeout: Case exceeded 600.0s wall-clock limit
- uci-steel-faults / 137 / fold 1 / autofeat / lightgbm: timeout: Case exceeded 600.0s wall-clock limit
- uci-dry-bean / 137 / fold 1 / autofeat / catboost: timeout: Case exceeded 600.0s wall-clock limit
- uci-steel-faults / 137 / fold 1 / autofeat / catboost: timeout: Case exceeded 600.0s wall-clock limit
- uci-dry-bean / 137 / fold 1 / autofeat / xgboost: timeout: Case exceeded 600.0s wall-clock limit
- uci-dry-bean / 137 / fold 1 / autofeat / linear: timeout: Case exceeded 600.0s wall-clock limit
- uci-dry-bean / 137 / fold 1 / autofeat / lightgbm: timeout: Case exceeded 600.0s wall-clock limit
- uci-dry-bean / 42 / fold 2 / autofeat / xgboost: timeout: Case exceeded 600.0s wall-clock limit
- uci-steel-faults / 42 / fold 2 / autofeat / xgboost: timeout: Case exceeded 600.0s wall-clock limit
- uci-dry-bean / 42 / fold 2 / autofeat / linear: timeout: Case exceeded 600.0s wall-clock limit
- uci-steel-faults / 42 / fold 2 / autofeat / catboost: timeout: Case exceeded 600.0s wall-clock limit
- uci-dry-bean / 42 / fold 2 / autofeat / lightgbm: timeout: Case exceeded 600.0s wall-clock limit
- uci-steel-faults / 42 / fold 2 / autofeat / linear: timeout: Case exceeded 600.0s wall-clock limit
- uci-steel-faults / 42 / fold 2 / autofeat / lightgbm: timeout: Case exceeded 600.0s wall-clock limit
- uci-dry-bean / 42 / fold 2 / autofeat / catboost: timeout: Case exceeded 600.0s wall-clock limit
- uci-steel-faults / 137 / fold 2 / autofeat / lightgbm: timeout: Case exceeded 600.0s wall-clock limit
- uci-dry-bean / 137 / fold 2 / autofeat / catboost: timeout: Case exceeded 600.0s wall-clock limit
- uci-steel-faults / 137 / fold 2 / autofeat / catboost: timeout: Case exceeded 600.0s wall-clock limit
- uci-steel-faults / 137 / fold 2 / autofeat / linear: timeout: Case exceeded 600.0s wall-clock limit
- uci-dry-bean / 137 / fold 2 / autofeat / xgboost: timeout: Case exceeded 600.0s wall-clock limit
- uci-dry-bean / 137 / fold 2 / autofeat / lightgbm: timeout: Case exceeded 600.0s wall-clock limit
- uci-dry-bean / 137 / fold 2 / autofeat / linear: timeout: Case exceeded 600.0s wall-clock limit
- uci-steel-faults / 137 / fold 2 / autofeat / xgboost: timeout: Case exceeded 600.0s wall-clock limit

Raw cell metrics and provenance: local `results.json`. Completed runs also write `archive.json`,
which lists the authoritative lossless gzip parts and their uncompressed checksum.
