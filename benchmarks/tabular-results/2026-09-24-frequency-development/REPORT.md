# Tabular development benchmark

**Benchmark measurement, not a SOTA claim.** Dataset identity, admission roles and split policies
are recorded in the manifest. Legacy manifests are not automatically rights-reviewed or confirmation data.

Primary loss: classification log loss; regression RMSE (lower is better).
Learners are tuned separately per method on inner validation, never selected using outer-test scores.
Synthetic diagnostics are reported separately. All requested failures count toward coverage.

OpenFE-stateless is an explicitly restricted variant, not full OpenFE. AutoFeat uses numeric generation.
OpenFE-inductive searches all supported operator families but freezes training statistics at inference;
its native variant retains all selected additions, while the width-matched variant retains the top-ranked budget.
AutoFeat-default uses native generation/selection defaults and output width with train-fitted categorical preprocessing.
All methods retain original categorical features for identical downstream one-hot preprocessing.
FeatCopilot-task changes only explicit task routing; FeatCopilot-predictive uses learner-aligned holdout selection.
FeatCopilot-paired is an opt-in development variant using paired-gain uncertainty instead of absolute-loss uncertainty.
Extended variants add balanced numeric pairs and frozen training-observed categorical pair indicators; defaults remain standard.
Importance variants fit a training-only joint learner after exact candidate deduplication; holdout subset validation remains independent.
Frequency-paired adds bounded frozen unary-frequency candidates to importance-paired; mappings never learn from validation or test rows.
Unselected and selection-only variants isolate the legacy stages; unselected output retains the generator's full planned width.
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
    "uci-spambase",
    "uci-banknote-authentication",
    "uci-combined-cycle-power",
    "uci-real-estate-sindian",
    "uci-auto-mpg",
    "uci-chronic-kidney-disease",
    "uci-wisconsin-diagnostic",
    "houses",
    "wine"
  ],
  "folds": 3,
  "seeds": [
    42
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
  "live_llm": false,
  "llm_ledger": null,
  "allow_soft_llm_tokens": false,
  "methods": [
    "baseline",
    "featcopilot-importance-paired",
    "featcopilot-frequency-paired",
    "openfe-inductive"
  ],
  "learners": [
    "linear",
    "lightgbm"
  ],
  "suite": [
    "C:\\code\\featcopilot\\benchmarks\\suites\\frequency_development_replay_20260924.json",
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
| real_world / development | classification | lightgbm | baseline | 9/9 / 9/9 | 27/27 | 0/9/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | classification | lightgbm | featcopilot-frequency-paired | 9/9 / 9/9 | 27/27 | 2/6/1 | +8.31% | [-5.15%, +25.45%] |
| real_world / development | classification | lightgbm | featcopilot-importance-paired | 9/9 / 9/9 | 27/27 | 2/5/2 | +8.58% | [-5.09%, +25.77%] |
| real_world / development | classification | lightgbm | openfe-inductive | 9/9 / 9/9 | 27/27 | 3/1/5 | +9.71% | [-1.53%, +25.56%] |
| real_world / development | classification | linear | baseline | 9/9 / 9/9 | 27/27 | 0/9/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | classification | linear | featcopilot-frequency-paired | 9/9 / 9/9 | 27/27 | 7/2/0 | +15.27% | [+2.56%, +34.92%] |
| real_world / development | classification | linear | featcopilot-importance-paired | 9/9 / 9/9 | 27/27 | 6/1/2 | +12.38% | [-1.66%, +34.84%] |
| real_world / development | classification | linear | openfe-inductive | 9/9 / 9/9 | 27/27 | 6/2/1 | +11.24% | [+1.34%, +27.99%] |
| real_world / development | regression | lightgbm | baseline | 6/6 / 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | regression | lightgbm | featcopilot-frequency-paired | 6/6 / 6/6 | 18/18 | 4/1/1 | +2.89% | [-0.51%, +6.35%] |
| real_world / development | regression | lightgbm | featcopilot-importance-paired | 6/6 / 6/6 | 18/18 | 4/2/0 | +2.06% | [+0.28%, +4.57%] |
| real_world / development | regression | lightgbm | openfe-inductive | 6/6 / 6/6 | 18/18 | 3/2/1 | +2.56% | [-0.03%, +5.26%] |
| real_world / development | regression | linear | baseline | 6/6 / 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | regression | linear | featcopilot-frequency-paired | 6/6 / 6/6 | 18/18 | 6/0/0 | +12.83% | [+4.71%, +25.56%] |
| real_world / development | regression | linear | featcopilot-importance-paired | 6/6 / 6/6 | 18/18 | 6/0/0 | +13.37% | [+4.88%, +26.35%] |
| real_world / development | regression | linear | openfe-inductive | 6/6 / 6/6 | 18/18 | 5/1/0 | +8.28% | [+1.93%, +18.21%] |
| simulation / development | regression | lightgbm | baseline | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | lightgbm | featcopilot-frequency-paired | 1/1 / 1/1 | 3/3 | 1/0/0 | +5.93% | n/a |
| simulation / development | regression | lightgbm | featcopilot-importance-paired | 1/1 / 1/1 | 3/3 | 1/0/0 | +5.91% | n/a |
| simulation / development | regression | lightgbm | openfe-inductive | 1/1 / 1/1 | 3/3 | 1/0/0 | +8.05% | n/a |
| simulation / development | regression | linear | baseline | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | linear | featcopilot-frequency-paired | 1/1 / 1/1 | 3/3 | 1/0/0 | +82.07% | n/a |
| simulation / development | regression | linear | featcopilot-importance-paired | 1/1 / 1/1 | 3/3 | 1/0/0 | +76.99% | n/a |
| simulation / development | regression | linear | openfe-inductive | 1/1 / 1/1 | 3/3 | 1/0/0 | +43.87% | n/a |

## Per-dataset means (complete folds only)

| Dataset | Learner | Method | Primary loss | FE seconds (inner + outer) | Pipeline seconds | Features |
|---|---|---|---|---|---|---|
| houses | lightgbm | baseline | 0.244681 | 0.02 | 1.50 | 8.0 |
| houses | lightgbm | featcopilot-frequency-paired | 0.224109 | 16.71 | 17.51 | 33.0 |
| houses | lightgbm | featcopilot-importance-paired | 0.224733 | 14.03 | 14.93 | 33.0 |
| houses | lightgbm | openfe-inductive | 0.227345 | 143.22 | 144.55 | 58.0 |
| houses | linear | baseline | 0.338541 | 0.02 | 0.08 | 8.0 |
| houses | linear | featcopilot-frequency-paired | 0.327903 | 6.17 | 6.27 | 26.3 |
| houses | linear | featcopilot-importance-paired | 0.326582 | 3.45 | 3.54 | 41.3 |
| houses | linear | openfe-inductive | 0.319067 | 145.00 | 145.12 | 58.0 |
| uci-adult | lightgbm | baseline | 0.309826 | 0.06 | 1.64 | 13.0 |
| uci-adult | lightgbm | featcopilot-frequency-paired | 0.309826 | 19.57 | 20.44 | 13.0 |
| uci-adult | lightgbm | featcopilot-importance-paired | 0.309826 | 17.24 | 18.16 | 13.0 |
| uci-adult | lightgbm | openfe-inductive | 0.322537 | 147.12 | 148.42 | 63.0 |
| uci-adult | linear | baseline | 0.340533 | 0.06 | 0.39 | 13.0 |
| uci-adult | linear | featcopilot-frequency-paired | 0.333355 | 15.30 | 15.71 | 16.3 |
| uci-adult | linear | featcopilot-importance-paired | 0.335560 | 12.77 | 13.19 | 18.0 |
| uci-adult | linear | openfe-inductive | 0.332074 | 149.58 | 150.36 | 63.0 |
| uci-auto-mpg | lightgbm | baseline | 2.924286 | 0.02 | 1.13 | 7.0 |
| uci-auto-mpg | lightgbm | featcopilot-frequency-paired | 2.941751 | 5.73 | 6.03 | 17.0 |
| uci-auto-mpg | lightgbm | featcopilot-importance-paired | 2.936235 | 5.45 | 5.73 | 27.0 |
| uci-auto-mpg | lightgbm | openfe-inductive | 2.906142 | 131.93 | 132.18 | 8.0 |
| uci-auto-mpg | linear | baseline | 3.372414 | 0.02 | 0.08 | 7.0 |
| uci-auto-mpg | linear | featcopilot-frequency-paired | 3.015777 | 2.71 | 2.80 | 28.7 |
| uci-auto-mpg | linear | featcopilot-importance-paired | 2.950793 | 2.56 | 2.64 | 30.3 |
| uci-auto-mpg | linear | openfe-inductive | 3.335482 | 133.21 | 133.27 | 8.0 |
| uci-banknote-authentication | lightgbm | baseline | 0.015250 | 0.01 | 1.29 | 4.0 |
| uci-banknote-authentication | lightgbm | featcopilot-frequency-paired | 0.005267 | 6.68 | 7.05 | 45.7 |
| uci-banknote-authentication | lightgbm | featcopilot-importance-paired | 0.005345 | 5.96 | 6.34 | 45.7 |
| uci-banknote-authentication | lightgbm | openfe-inductive | 0.004879 | 137.11 | 137.56 | 53.0 |
| uci-banknote-authentication | linear | baseline | 0.029459 | 0.01 | 0.11 | 4.0 |
| uci-banknote-authentication | linear | featcopilot-frequency-paired | 0.003435 | 3.06 | 3.17 | 14.0 |
| uci-banknote-authentication | linear | featcopilot-importance-paired | 0.001147 | 2.41 | 2.51 | 29.0 |
| uci-banknote-authentication | linear | openfe-inductive | 0.007443 | 140.73 | 140.86 | 53.0 |
| uci-chronic-kidney-disease | lightgbm | baseline | 0.035504 | 0.05 | 1.24 | 24.0 |
| uci-chronic-kidney-disease | lightgbm | featcopilot-frequency-paired | 0.023974 | 8.98 | 9.85 | 30.7 |
| uci-chronic-kidney-disease | lightgbm | featcopilot-importance-paired | 0.022914 | 8.49 | 9.35 | 42.3 |
| uci-chronic-kidney-disease | lightgbm | openfe-inductive | 0.025707 | 131.72 | 132.06 | 25.0 |
| uci-chronic-kidney-disease | linear | baseline | 0.016247 | 0.05 | 0.24 | 24.0 |
| uci-chronic-kidney-disease | linear | featcopilot-frequency-paired | 0.012448 | 6.73 | 7.49 | 41.3 |
| uci-chronic-kidney-disease | linear | featcopilot-importance-paired | 0.017848 | 6.10 | 6.84 | 41.3 |
| uci-chronic-kidney-disease | linear | openfe-inductive | 0.014241 | 140.93 | 141.11 | 25.0 |
| uci-combined-cycle-power | lightgbm | baseline | 3.382905 | 0.01 | 1.60 | 4.0 |
| uci-combined-cycle-power | lightgbm | featcopilot-frequency-paired | 3.103413 | 12.56 | 13.42 | 29.0 |
| uci-combined-cycle-power | lightgbm | featcopilot-importance-paired | 3.324975 | 10.25 | 11.04 | 14.0 |
| uci-combined-cycle-power | lightgbm | openfe-inductive | 3.155407 | 150.57 | 151.78 | 47.7 |
| uci-combined-cycle-power | linear | baseline | 4.474993 | 0.01 | 0.06 | 4.0 |
| uci-combined-cycle-power | linear | featcopilot-frequency-paired | 4.060389 | 4.31 | 4.41 | 54.0 |
| uci-combined-cycle-power | linear | featcopilot-importance-paired | 4.105000 | 2.42 | 2.52 | 54.0 |
| uci-combined-cycle-power | linear | openfe-inductive | 4.130092 | 142.79 | 142.90 | 47.7 |
| uci-concrete-strength | lightgbm | baseline | 4.553620 | 0.02 | 1.45 | 8.0 |
| uci-concrete-strength | lightgbm | featcopilot-frequency-paired | 4.707254 | 9.91 | 10.35 | 9.7 |
| uci-concrete-strength | lightgbm | featcopilot-importance-paired | 4.571635 | 9.30 | 9.74 | 26.3 |
| uci-concrete-strength | lightgbm | openfe-inductive | 4.645433 | 146.03 | 146.67 | 58.0 |
| uci-concrete-strength | linear | baseline | 10.559292 | 0.02 | 0.07 | 8.0 |
| uci-concrete-strength | linear | featcopilot-frequency-paired | 5.970942 | 3.57 | 3.64 | 44.7 |
| uci-concrete-strength | linear | featcopilot-importance-paired | 5.895346 | 2.92 | 2.99 | 49.7 |
| uci-concrete-strength | linear | openfe-inductive | 7.188370 | 141.23 | 141.29 | 58.0 |
| uci-dry-bean | lightgbm | baseline | 0.248388 | 0.04 | 4.05 | 16.0 |
| uci-dry-bean | lightgbm | featcopilot-frequency-paired | 0.247101 | 91.95 | 95.31 | 16.7 |
| uci-dry-bean | lightgbm | featcopilot-importance-paired | 0.247101 | 86.62 | 89.98 | 16.7 |
| uci-dry-bean | lightgbm | openfe-inductive | 0.254248 | 160.93 | 166.20 | 66.0 |
| uci-dry-bean | linear | baseline | 0.216633 | 0.05 | 0.47 | 16.0 |
| uci-dry-bean | linear | featcopilot-frequency-paired | 0.214246 | 20.88 | 21.35 | 19.3 |
| uci-dry-bean | linear | featcopilot-importance-paired | 0.210888 | 15.89 | 16.41 | 21.3 |
| uci-dry-bean | linear | openfe-inductive | 0.216792 | 153.66 | 154.53 | 66.0 |
| uci-energy-efficiency | lightgbm | baseline | 0.479742 | 0.02 | 1.36 | 8.0 |
| uci-energy-efficiency | lightgbm | featcopilot-frequency-paired | 0.451304 | 7.72 | 8.15 | 41.3 |
| uci-energy-efficiency | lightgbm | featcopilot-importance-paired | 0.451394 | 8.06 | 8.53 | 33.0 |
| uci-energy-efficiency | lightgbm | openfe-inductive | 0.441135 | 136.51 | 136.88 | 57.0 |
| uci-energy-efficiency | linear | baseline | 2.844552 | 0.02 | 0.08 | 8.0 |
| uci-energy-efficiency | linear | featcopilot-frequency-paired | 0.510095 | 3.28 | 3.39 | 58.0 |
| uci-energy-efficiency | linear | featcopilot-importance-paired | 0.654416 | 2.52 | 2.63 | 58.0 |
| uci-energy-efficiency | linear | openfe-inductive | 1.596539 | 139.63 | 139.70 | 57.0 |
| uci-german-credit | lightgbm | baseline | 0.519777 | 0.05 | 1.52 | 20.0 |
| uci-german-credit | lightgbm | featcopilot-frequency-paired | 0.523848 | 15.00 | 15.89 | 20.7 |
| uci-german-credit | lightgbm | featcopilot-importance-paired | 0.523848 | 13.95 | 14.83 | 20.7 |
| uci-german-credit | lightgbm | openfe-inductive | 0.532456 | 141.34 | 142.02 | 70.0 |
| uci-german-credit | linear | baseline | 0.498452 | 0.05 | 0.27 | 20.0 |
| uci-german-credit | linear | featcopilot-frequency-paired | 0.498452 | 8.42 | 8.95 | 20.0 |
| uci-german-credit | linear | featcopilot-importance-paired | 0.498452 | 7.30 | 7.81 | 20.0 |
| uci-german-credit | linear | openfe-inductive | 0.499680 | 141.89 | 142.18 | 70.0 |
| uci-online-shoppers | lightgbm | baseline | 0.380834 | 0.06 | 1.70 | 11.0 |
| uci-online-shoppers | lightgbm | featcopilot-frequency-paired | 0.382172 | 19.19 | 20.05 | 11.3 |
| uci-online-shoppers | lightgbm | featcopilot-importance-paired | 0.382172 | 16.39 | 17.26 | 11.3 |
| uci-online-shoppers | lightgbm | openfe-inductive | 0.384828 | 150.81 | 152.12 | 61.0 |
| uci-online-shoppers | linear | baseline | 0.395244 | 0.05 | 0.31 | 11.0 |
| uci-online-shoppers | linear | featcopilot-frequency-paired | 0.387539 | 13.77 | 14.12 | 13.0 |
| uci-online-shoppers | linear | featcopilot-importance-paired | 0.384799 | 11.83 | 12.18 | 17.7 |
| uci-online-shoppers | linear | openfe-inductive | 0.388122 | 152.74 | 153.55 | 61.0 |
| uci-real-estate-sindian | lightgbm | baseline | 7.448276 | 0.02 | 1.07 | 6.0 |
| uci-real-estate-sindian | lightgbm | featcopilot-frequency-paired | 7.266136 | 5.29 | 5.53 | 8.3 |
| uci-real-estate-sindian | lightgbm | featcopilot-importance-paired | 7.323031 | 5.14 | 5.40 | 15.0 |
| uci-real-estate-sindian | lightgbm | openfe-inductive | 7.448276 | 133.13 | 133.38 | 7.0 |
| uci-real-estate-sindian | linear | baseline | 8.793747 | 0.02 | 0.06 | 6.0 |
| uci-real-estate-sindian | linear | featcopilot-frequency-paired | 8.066223 | 2.47 | 2.53 | 9.7 |
| uci-real-estate-sindian | linear | featcopilot-importance-paired | 7.952655 | 2.34 | 2.39 | 11.3 |
| uci-real-estate-sindian | linear | openfe-inductive | 8.792728 | 133.41 | 133.45 | 7.0 |
| uci-spambase | lightgbm | baseline | 0.141042 | 0.16 | 2.46 | 57.0 |
| uci-spambase | lightgbm | featcopilot-frequency-paired | 0.140386 | 38.57 | 39.98 | 57.3 |
| uci-spambase | lightgbm | featcopilot-importance-paired | 0.140386 | 30.80 | 32.24 | 57.3 |
| uci-spambase | lightgbm | openfe-inductive | 0.142694 | 224.89 | 226.63 | 107.0 |
| uci-spambase | linear | baseline | 0.239802 | 0.15 | 0.41 | 57.0 |
| uci-spambase | linear | featcopilot-frequency-paired | 0.205999 | 18.79 | 19.08 | 75.3 |
| uci-spambase | linear | featcopilot-importance-paired | 0.195665 | 8.33 | 8.61 | 75.3 |
| uci-spambase | linear | openfe-inductive | 0.230953 | 227.72 | 228.07 | 107.0 |
| uci-steel-faults | lightgbm | baseline | 0.557708 | 0.07 | 4.09 | 27.0 |
| uci-steel-faults | lightgbm | featcopilot-frequency-paired | 0.559112 | 84.44 | 88.00 | 30.3 |
| uci-steel-faults | lightgbm | featcopilot-importance-paired | 0.563370 | 76.04 | 79.51 | 28.7 |
| uci-steel-faults | lightgbm | openfe-inductive | 0.552266 | 164.71 | 168.87 | 77.0 |
| uci-steel-faults | linear | baseline | 0.764914 | 0.07 | 0.54 | 27.0 |
| uci-steel-faults | linear | featcopilot-frequency-paired | 0.715271 | 21.40 | 22.10 | 34.0 |
| uci-steel-faults | linear | featcopilot-importance-paired | 0.735209 | 17.55 | 18.24 | 32.3 |
| uci-steel-faults | linear | openfe-inductive | 0.707839 | 161.50 | 162.31 | 77.0 |
| uci-wisconsin-diagnostic | lightgbm | baseline | 0.120432 | 0.08 | 1.30 | 30.0 |
| uci-wisconsin-diagnostic | lightgbm | featcopilot-frequency-paired | 0.147848 | 9.29 | 9.67 | 33.3 |
| uci-wisconsin-diagnostic | lightgbm | featcopilot-importance-paired | 0.146930 | 7.87 | 8.24 | 33.3 |
| uci-wisconsin-diagnostic | lightgbm | openfe-inductive | 0.118061 | 140.14 | 140.47 | 31.0 |
| uci-wisconsin-diagnostic | linear | baseline | 0.089368 | 0.07 | 0.21 | 30.0 |
| uci-wisconsin-diagnostic | linear | featcopilot-frequency-paired | 0.089368 | 5.87 | 6.04 | 30.0 |
| uci-wisconsin-diagnostic | linear | featcopilot-importance-paired | 0.092795 | 4.07 | 4.24 | 31.7 |
| uci-wisconsin-diagnostic | linear | openfe-inductive | 0.090266 | 139.30 | 139.41 | 31.0 |
| wine | lightgbm | baseline | 0.654417 | 0.03 | 1.47 | 11.0 |
| wine | lightgbm | featcopilot-frequency-paired | 0.639870 | 23.52 | 24.70 | 52.7 |
| wine | lightgbm | featcopilot-importance-paired | 0.643800 | 17.37 | 18.48 | 52.7 |
| wine | lightgbm | openfe-inductive | 0.634993 | 147.46 | 148.78 | 61.0 |
| wine | linear | baseline | 0.737508 | 0.03 | 0.09 | 11.0 |
| wine | linear | featcopilot-frequency-paired | 0.720650 | 7.67 | 7.81 | 44.3 |
| wine | linear | featcopilot-importance-paired | 0.721247 | 4.02 | 4.16 | 44.3 |
| wine | linear | openfe-inductive | 0.714151 | 144.32 | 144.44 | 61.0 |

## Failures / timeouts

None.

Raw cell metrics and provenance: local `results.json`. Completed runs also write `archive.json`,
which lists the authoritative lossless gzip parts and their uncompressed checksum.
