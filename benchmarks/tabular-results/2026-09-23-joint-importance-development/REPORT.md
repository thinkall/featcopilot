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
    "featcopilot-predictive",
    "featcopilot-importance",
    "featcopilot-importance-paired",
    "openfe-inductive"
  ],
  "learners": [
    "linear",
    "lightgbm"
  ],
  "suite": [
    "C:\\code\\featcopilot\\benchmarks\\suites\\development_expansion_20260922.json",
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
| real_world / development | classification | lightgbm | featcopilot-importance | 9/9 / 9/9 | 27/27 | 2/6/1 | +4.71% | [-5.09%, +15.45%] |
| real_world / development | classification | lightgbm | featcopilot-importance-paired | 9/9 / 9/9 | 27/27 | 2/5/2 | +8.58% | [-5.09%, +25.77%] |
| real_world / development | classification | lightgbm | featcopilot-predictive | 9/9 / 9/9 | 27/27 | 1/7/1 | -0.33% | [-1.24%, +0.29%] |
| real_world / development | classification | lightgbm | openfe-inductive | 9/9 / 9/9 | 27/27 | 3/1/5 | +9.71% | [-1.53%, +25.56%] |
| real_world / development | classification | linear | baseline | 9/9 / 9/9 | 27/27 | 0/9/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | classification | linear | featcopilot-importance | 9/9 / 9/9 | 27/27 | 5/2/2 | +11.63% | [-2.23%, +34.10%] |
| real_world / development | classification | linear | featcopilot-importance-paired | 9/9 / 9/9 | 27/27 | 6/1/2 | +12.38% | [-1.66%, +34.84%] |
| real_world / development | classification | linear | featcopilot-predictive | 9/9 / 9/9 | 27/27 | 5/4/0 | +14.09% | [+1.06%, +34.04%] |
| real_world / development | classification | linear | openfe-inductive | 9/9 / 9/9 | 27/27 | 6/2/1 | +11.24% | [+1.34%, +27.99%] |
| real_world / development | regression | lightgbm | baseline | 6/6 / 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | regression | lightgbm | featcopilot-importance | 6/6 / 6/6 | 18/18 | 4/1/1 | +1.66% | [-0.60%, +4.58%] |
| real_world / development | regression | lightgbm | featcopilot-importance-paired | 6/6 / 6/6 | 18/18 | 4/2/0 | +2.06% | [+0.28%, +4.57%] |
| real_world / development | regression | lightgbm | featcopilot-predictive | 6/6 / 6/6 | 18/18 | 3/3/0 | +1.62% | [+0.35%, +3.57%] |
| real_world / development | regression | lightgbm | openfe-inductive | 6/6 / 6/6 | 18/18 | 3/2/1 | +2.56% | [-0.03%, +5.26%] |
| real_world / development | regression | linear | baseline | 6/6 / 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | regression | linear | featcopilot-importance | 6/6 / 6/6 | 18/18 | 5/1/0 | +10.97% | [+2.63%, +24.11%] |
| real_world / development | regression | linear | featcopilot-importance-paired | 6/6 / 6/6 | 18/18 | 6/0/0 | +13.37% | [+4.88%, +26.35%] |
| real_world / development | regression | linear | featcopilot-predictive | 6/6 / 6/6 | 18/18 | 6/0/0 | +10.65% | [+4.20%, +20.22%] |
| real_world / development | regression | linear | openfe-inductive | 6/6 / 6/6 | 18/18 | 5/1/0 | +8.28% | [+1.93%, +18.21%] |
| simulation / development | regression | lightgbm | baseline | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | lightgbm | featcopilot-importance | 1/1 / 1/1 | 3/3 | 1/0/0 | +5.17% | n/a |
| simulation / development | regression | lightgbm | featcopilot-importance-paired | 1/1 / 1/1 | 3/3 | 1/0/0 | +5.91% | n/a |
| simulation / development | regression | lightgbm | featcopilot-predictive | 1/1 / 1/1 | 3/3 | 0/0/1 | -2.85% | n/a |
| simulation / development | regression | lightgbm | openfe-inductive | 1/1 / 1/1 | 3/3 | 1/0/0 | +8.05% | n/a |
| simulation / development | regression | linear | baseline | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | linear | featcopilot-importance | 1/1 / 1/1 | 3/3 | 1/0/0 | +76.71% | n/a |
| simulation / development | regression | linear | featcopilot-importance-paired | 1/1 / 1/1 | 3/3 | 1/0/0 | +76.99% | n/a |
| simulation / development | regression | linear | featcopilot-predictive | 1/1 / 1/1 | 3/3 | 1/0/0 | +32.04% | n/a |
| simulation / development | regression | linear | openfe-inductive | 1/1 / 1/1 | 3/3 | 1/0/0 | +43.87% | n/a |

## Per-dataset means (complete folds only)

| Dataset | Learner | Method | Primary loss | FE seconds (inner + outer) | Pipeline seconds | Features |
|---|---|---|---|---|---|---|
| houses | lightgbm | baseline | 0.244681 | 0.02 | 1.40 | 8.0 |
| houses | lightgbm | featcopilot-importance | 0.224733 | 13.46 | 14.27 | 33.0 |
| houses | lightgbm | featcopilot-importance-paired | 0.224733 | 14.53 | 15.39 | 33.0 |
| houses | lightgbm | featcopilot-predictive | 0.241225 | 19.97 | 20.77 | 16.3 |
| houses | lightgbm | openfe-inductive | 0.227345 | 116.65 | 117.90 | 58.0 |
| houses | linear | baseline | 0.338541 | 0.02 | 0.08 | 8.0 |
| houses | linear | featcopilot-importance | 0.328247 | 2.95 | 3.04 | 41.3 |
| houses | linear | featcopilot-importance-paired | 0.326582 | 2.99 | 3.07 | 41.3 |
| houses | linear | featcopilot-predictive | 0.319025 | 3.70 | 3.79 | 11.3 |
| houses | linear | openfe-inductive | 0.319067 | 116.93 | 117.04 | 58.0 |
| uci-adult | lightgbm | baseline | 0.309826 | 0.06 | 1.52 | 13.0 |
| uci-adult | lightgbm | featcopilot-importance | 0.309826 | 15.74 | 16.61 | 13.0 |
| uci-adult | lightgbm | featcopilot-importance-paired | 0.309826 | 15.60 | 16.42 | 13.0 |
| uci-adult | lightgbm | featcopilot-predictive | 0.309826 | 15.30 | 16.11 | 13.0 |
| uci-adult | lightgbm | openfe-inductive | 0.322537 | 125.97 | 127.26 | 63.0 |
| uci-adult | linear | baseline | 0.340533 | 0.06 | 0.36 | 13.0 |
| uci-adult | linear | featcopilot-importance | 0.338376 | 111.64 | 112.05 | 13.3 |
| uci-adult | linear | featcopilot-importance-paired | 0.335560 | 113.24 | 113.64 | 18.0 |
| uci-adult | linear | featcopilot-predictive | 0.340533 | 105.52 | 105.86 | 13.0 |
| uci-adult | linear | openfe-inductive | 0.332074 | 125.73 | 126.49 | 63.0 |
| uci-auto-mpg | lightgbm | baseline | 2.924286 | 0.02 | 1.00 | 7.0 |
| uci-auto-mpg | lightgbm | featcopilot-importance | 2.924286 | 4.75 | 5.00 | 7.0 |
| uci-auto-mpg | lightgbm | featcopilot-importance-paired | 2.936235 | 5.41 | 5.68 | 27.0 |
| uci-auto-mpg | lightgbm | featcopilot-predictive | 2.924286 | 7.40 | 7.65 | 7.0 |
| uci-auto-mpg | lightgbm | openfe-inductive | 2.906142 | 109.51 | 109.76 | 8.0 |
| uci-auto-mpg | linear | baseline | 3.372414 | 0.02 | 0.08 | 7.0 |
| uci-auto-mpg | linear | featcopilot-importance | 3.034497 | 2.23 | 2.31 | 15.3 |
| uci-auto-mpg | linear | featcopilot-importance-paired | 2.950793 | 2.40 | 2.48 | 30.3 |
| uci-auto-mpg | linear | featcopilot-predictive | 2.993322 | 3.14 | 3.21 | 12.0 |
| uci-auto-mpg | linear | openfe-inductive | 3.335482 | 108.78 | 108.84 | 8.0 |
| uci-banknote-authentication | lightgbm | baseline | 0.015250 | 0.01 | 1.08 | 4.0 |
| uci-banknote-authentication | lightgbm | featcopilot-importance | 0.010164 | 5.50 | 5.81 | 10.7 |
| uci-banknote-authentication | lightgbm | featcopilot-importance-paired | 0.005345 | 5.64 | 6.02 | 45.7 |
| uci-banknote-authentication | lightgbm | featcopilot-predictive | 0.015795 | 8.56 | 8.98 | 37.3 |
| uci-banknote-authentication | lightgbm | openfe-inductive | 0.004879 | 115.41 | 115.86 | 53.0 |
| uci-banknote-authentication | linear | baseline | 0.029459 | 0.01 | 0.11 | 4.0 |
| uci-banknote-authentication | linear | featcopilot-importance | 0.001147 | 96.52 | 96.62 | 29.0 |
| uci-banknote-authentication | linear | featcopilot-importance-paired | 0.001147 | 97.29 | 97.39 | 29.0 |
| uci-banknote-authentication | linear | featcopilot-predictive | 0.003119 | 98.63 | 98.73 | 29.0 |
| uci-banknote-authentication | linear | openfe-inductive | 0.007443 | 115.62 | 115.74 | 53.0 |
| uci-chronic-kidney-disease | lightgbm | baseline | 0.035504 | 0.05 | 1.09 | 24.0 |
| uci-chronic-kidney-disease | lightgbm | featcopilot-importance | 0.024097 | 7.69 | 8.55 | 27.3 |
| uci-chronic-kidney-disease | lightgbm | featcopilot-importance-paired | 0.022914 | 8.07 | 8.94 | 42.3 |
| uci-chronic-kidney-disease | lightgbm | featcopilot-predictive | 0.035504 | 11.37 | 11.86 | 24.0 |
| uci-chronic-kidney-disease | lightgbm | openfe-inductive | 0.025707 | 113.06 | 113.39 | 25.0 |
| uci-chronic-kidney-disease | linear | baseline | 0.016247 | 0.05 | 0.22 | 24.0 |
| uci-chronic-kidney-disease | linear | featcopilot-importance | 0.017848 | 101.32 | 102.05 | 41.3 |
| uci-chronic-kidney-disease | linear | featcopilot-importance-paired | 0.017848 | 101.36 | 102.09 | 41.3 |
| uci-chronic-kidney-disease | linear | featcopilot-predictive | 0.015507 | 100.20 | 100.53 | 26.0 |
| uci-chronic-kidney-disease | linear | openfe-inductive | 0.014241 | 111.91 | 112.08 | 25.0 |
| uci-combined-cycle-power | lightgbm | baseline | 3.382905 | 0.01 | 1.27 | 4.0 |
| uci-combined-cycle-power | lightgbm | featcopilot-importance | 3.324975 | 9.80 | 10.52 | 14.0 |
| uci-combined-cycle-power | lightgbm | featcopilot-importance-paired | 3.324975 | 9.49 | 10.24 | 14.0 |
| uci-combined-cycle-power | lightgbm | featcopilot-predictive | 3.173097 | 13.57 | 14.16 | 16.0 |
| uci-combined-cycle-power | lightgbm | openfe-inductive | 3.155407 | 115.14 | 116.18 | 47.7 |
| uci-combined-cycle-power | linear | baseline | 4.474993 | 0.01 | 0.06 | 4.0 |
| uci-combined-cycle-power | linear | featcopilot-importance | 4.157545 | 2.44 | 2.51 | 24.0 |
| uci-combined-cycle-power | linear | featcopilot-importance-paired | 4.105000 | 2.24 | 2.34 | 54.0 |
| uci-combined-cycle-power | linear | featcopilot-predictive | 4.139593 | 3.30 | 3.36 | 22.3 |
| uci-combined-cycle-power | linear | openfe-inductive | 4.130092 | 116.21 | 116.30 | 47.7 |
| uci-concrete-strength | lightgbm | baseline | 4.553620 | 0.02 | 1.21 | 8.0 |
| uci-concrete-strength | lightgbm | featcopilot-importance | 4.667819 | 7.93 | 8.31 | 8.0 |
| uci-concrete-strength | lightgbm | featcopilot-importance-paired | 4.571635 | 8.67 | 9.09 | 26.3 |
| uci-concrete-strength | lightgbm | featcopilot-predictive | 4.491002 | 10.83 | 11.27 | 8.0 |
| uci-concrete-strength | lightgbm | openfe-inductive | 4.645433 | 115.67 | 116.25 | 58.0 |
| uci-concrete-strength | linear | baseline | 10.559292 | 0.02 | 0.07 | 8.0 |
| uci-concrete-strength | linear | featcopilot-importance | 6.073268 | 2.65 | 2.72 | 41.3 |
| uci-concrete-strength | linear | featcopilot-importance-paired | 5.895346 | 2.42 | 2.49 | 49.7 |
| uci-concrete-strength | linear | featcopilot-predictive | 7.054075 | 3.07 | 3.12 | 28.3 |
| uci-concrete-strength | linear | openfe-inductive | 7.188370 | 114.21 | 114.27 | 58.0 |
| uci-dry-bean | lightgbm | baseline | 0.248388 | 0.04 | 3.88 | 16.0 |
| uci-dry-bean | lightgbm | featcopilot-importance | 0.248388 | 77.26 | 80.55 | 16.0 |
| uci-dry-bean | lightgbm | featcopilot-importance-paired | 0.247101 | 86.29 | 89.67 | 16.7 |
| uci-dry-bean | lightgbm | featcopilot-predictive | 0.248388 | 74.36 | 77.52 | 16.0 |
| uci-dry-bean | lightgbm | openfe-inductive | 0.254248 | 131.05 | 135.87 | 66.0 |
| uci-dry-bean | linear | baseline | 0.216633 | 0.04 | 0.43 | 16.0 |
| uci-dry-bean | linear | featcopilot-importance | 0.214285 | 109.16 | 109.61 | 18.0 |
| uci-dry-bean | linear | featcopilot-importance-paired | 0.210888 | 112.36 | 112.83 | 21.3 |
| uci-dry-bean | linear | featcopilot-predictive | 0.214540 | 107.98 | 108.43 | 24.3 |
| uci-dry-bean | linear | openfe-inductive | 0.216792 | 128.75 | 129.60 | 66.0 |
| uci-energy-efficiency | lightgbm | baseline | 0.479742 | 0.02 | 1.08 | 8.0 |
| uci-energy-efficiency | lightgbm | featcopilot-importance | 0.454951 | 6.32 | 6.70 | 11.3 |
| uci-energy-efficiency | lightgbm | featcopilot-importance-paired | 0.451394 | 6.36 | 6.78 | 33.0 |
| uci-energy-efficiency | lightgbm | featcopilot-predictive | 0.493421 | 9.18 | 9.50 | 8.0 |
| uci-energy-efficiency | lightgbm | openfe-inductive | 0.441135 | 111.99 | 112.32 | 57.0 |
| uci-energy-efficiency | linear | baseline | 2.844552 | 0.02 | 0.08 | 8.0 |
| uci-energy-efficiency | linear | featcopilot-importance | 0.662624 | 2.44 | 2.54 | 49.7 |
| uci-energy-efficiency | linear | featcopilot-importance-paired | 0.654416 | 2.46 | 2.57 | 58.0 |
| uci-energy-efficiency | linear | featcopilot-predictive | 1.933146 | 3.32 | 3.40 | 33.0 |
| uci-energy-efficiency | linear | openfe-inductive | 1.596539 | 113.58 | 113.65 | 57.0 |
| uci-german-credit | lightgbm | baseline | 0.519777 | 0.04 | 1.30 | 20.0 |
| uci-german-credit | lightgbm | featcopilot-importance | 0.523848 | 12.77 | 13.60 | 20.7 |
| uci-german-credit | lightgbm | featcopilot-importance-paired | 0.523848 | 13.26 | 14.10 | 20.7 |
| uci-german-credit | lightgbm | featcopilot-predictive | 0.522024 | 11.90 | 12.50 | 20.7 |
| uci-german-credit | lightgbm | openfe-inductive | 0.532456 | 119.25 | 119.94 | 70.0 |
| uci-german-credit | linear | baseline | 0.498452 | 0.04 | 0.24 | 20.0 |
| uci-german-credit | linear | featcopilot-importance | 0.498452 | 102.32 | 102.82 | 20.0 |
| uci-german-credit | linear | featcopilot-importance-paired | 0.498452 | 103.30 | 103.80 | 20.0 |
| uci-german-credit | linear | featcopilot-predictive | 0.501817 | 99.79 | 100.06 | 23.3 |
| uci-german-credit | linear | openfe-inductive | 0.499680 | 119.43 | 119.73 | 70.0 |
| uci-online-shoppers | lightgbm | baseline | 0.380834 | 0.06 | 1.50 | 11.0 |
| uci-online-shoppers | lightgbm | featcopilot-importance | 0.382172 | 14.49 | 15.30 | 11.3 |
| uci-online-shoppers | lightgbm | featcopilot-importance-paired | 0.382172 | 15.16 | 15.98 | 11.3 |
| uci-online-shoppers | lightgbm | featcopilot-predictive | 0.380834 | 14.68 | 15.42 | 11.0 |
| uci-online-shoppers | lightgbm | openfe-inductive | 0.384828 | 127.19 | 128.45 | 61.0 |
| uci-online-shoppers | linear | baseline | 0.395244 | 0.05 | 0.31 | 11.0 |
| uci-online-shoppers | linear | featcopilot-importance | 0.386346 | 109.49 | 109.81 | 14.7 |
| uci-online-shoppers | linear | featcopilot-importance-paired | 0.384799 | 109.54 | 109.87 | 17.7 |
| uci-online-shoppers | linear | featcopilot-predictive | 0.385326 | 105.94 | 106.24 | 13.3 |
| uci-online-shoppers | linear | openfe-inductive | 0.388122 | 124.59 | 125.37 | 61.0 |
| uci-real-estate-sindian | lightgbm | baseline | 7.448276 | 0.02 | 0.97 | 6.0 |
| uci-real-estate-sindian | lightgbm | featcopilot-importance | 7.349739 | 4.90 | 5.14 | 14.3 |
| uci-real-estate-sindian | lightgbm | featcopilot-importance-paired | 7.323031 | 5.12 | 5.38 | 15.0 |
| uci-real-estate-sindian | lightgbm | featcopilot-predictive | 7.448276 | 6.90 | 7.12 | 6.0 |
| uci-real-estate-sindian | lightgbm | openfe-inductive | 7.448276 | 109.81 | 110.03 | 7.0 |
| uci-real-estate-sindian | linear | baseline | 8.793747 | 0.02 | 0.06 | 6.0 |
| uci-real-estate-sindian | linear | featcopilot-importance | 8.592077 | 1.99 | 2.04 | 6.3 |
| uci-real-estate-sindian | linear | featcopilot-importance-paired | 7.952655 | 2.12 | 2.17 | 11.3 |
| uci-real-estate-sindian | linear | featcopilot-predictive | 8.378888 | 3.15 | 3.20 | 6.7 |
| uci-real-estate-sindian | linear | openfe-inductive | 8.792728 | 110.20 | 110.24 | 7.0 |
| uci-spambase | lightgbm | baseline | 0.141042 | 0.15 | 2.35 | 57.0 |
| uci-spambase | lightgbm | featcopilot-importance | 0.141042 | 27.40 | 28.79 | 57.0 |
| uci-spambase | lightgbm | featcopilot-importance-paired | 0.140386 | 28.90 | 30.25 | 57.3 |
| uci-spambase | lightgbm | featcopilot-predictive | 0.141042 | 36.75 | 38.11 | 57.0 |
| uci-spambase | lightgbm | openfe-inductive | 0.142694 | 190.48 | 192.15 | 107.0 |
| uci-spambase | linear | baseline | 0.239802 | 0.15 | 0.40 | 57.0 |
| uci-spambase | linear | featcopilot-importance | 0.199929 | 106.67 | 106.93 | 67.3 |
| uci-spambase | linear | featcopilot-importance-paired | 0.195665 | 107.35 | 107.62 | 75.3 |
| uci-spambase | linear | featcopilot-predictive | 0.175298 | 110.20 | 110.48 | 90.3 |
| uci-spambase | linear | openfe-inductive | 0.230953 | 188.23 | 188.56 | 107.0 |
| uci-steel-faults | lightgbm | baseline | 0.557708 | 0.07 | 3.70 | 27.0 |
| uci-steel-faults | lightgbm | featcopilot-importance | 0.557708 | 65.57 | 68.51 | 27.0 |
| uci-steel-faults | lightgbm | featcopilot-importance-paired | 0.563370 | 71.01 | 74.09 | 28.7 |
| uci-steel-faults | lightgbm | featcopilot-predictive | 0.552069 | 75.33 | 78.50 | 33.7 |
| uci-steel-faults | lightgbm | openfe-inductive | 0.552266 | 140.19 | 144.25 | 77.0 |
| uci-steel-faults | linear | baseline | 0.764914 | 0.07 | 0.52 | 27.0 |
| uci-steel-faults | linear | featcopilot-importance | 0.752259 | 109.87 | 110.48 | 27.7 |
| uci-steel-faults | linear | featcopilot-importance-paired | 0.735209 | 111.52 | 112.15 | 32.3 |
| uci-steel-faults | linear | featcopilot-predictive | 0.740742 | 109.38 | 109.93 | 29.0 |
| uci-steel-faults | linear | openfe-inductive | 0.707839 | 141.18 | 141.98 | 77.0 |
| uci-wisconsin-diagnostic | lightgbm | baseline | 0.120432 | 0.07 | 1.26 | 30.0 |
| uci-wisconsin-diagnostic | lightgbm | featcopilot-importance | 0.146930 | 7.49 | 7.87 | 33.3 |
| uci-wisconsin-diagnostic | lightgbm | featcopilot-importance-paired | 0.146930 | 7.48 | 7.86 | 33.3 |
| uci-wisconsin-diagnostic | lightgbm | featcopilot-predictive | 0.120432 | 23.63 | 24.00 | 30.0 |
| uci-wisconsin-diagnostic | lightgbm | openfe-inductive | 0.118061 | 113.97 | 114.27 | 31.0 |
| uci-wisconsin-diagnostic | linear | baseline | 0.089368 | 0.07 | 0.20 | 30.0 |
| uci-wisconsin-diagnostic | linear | featcopilot-importance | 0.092795 | 99.94 | 100.11 | 31.7 |
| uci-wisconsin-diagnostic | linear | featcopilot-importance-paired | 0.092795 | 98.60 | 98.76 | 31.7 |
| uci-wisconsin-diagnostic | linear | featcopilot-predictive | 0.089368 | 100.75 | 100.90 | 30.0 |
| uci-wisconsin-diagnostic | linear | openfe-inductive | 0.090266 | 112.72 | 112.82 | 31.0 |
| wine | lightgbm | baseline | 0.654417 | 0.03 | 1.35 | 11.0 |
| wine | lightgbm | featcopilot-importance | 0.646005 | 16.02 | 16.93 | 19.3 |
| wine | lightgbm | featcopilot-importance-paired | 0.643800 | 16.16 | 17.20 | 52.7 |
| wine | lightgbm | featcopilot-predictive | 0.649759 | 20.25 | 20.97 | 11.0 |
| wine | lightgbm | openfe-inductive | 0.634993 | 117.21 | 118.46 | 61.0 |
| wine | linear | baseline | 0.737508 | 0.03 | 0.09 | 11.0 |
| wine | linear | featcopilot-importance | 0.731082 | 3.67 | 3.77 | 27.7 |
| wine | linear | featcopilot-importance-paired | 0.721247 | 3.70 | 3.83 | 44.3 |
| wine | linear | featcopilot-predictive | 0.726537 | 4.09 | 4.18 | 28.0 |
| wine | linear | openfe-inductive | 0.714151 | 116.85 | 116.97 | 61.0 |

## Failures / timeouts

None.

Raw cell metrics and provenance: local `results.json`. Completed runs also write `archive.json`,
which lists the authoritative lossless gzip parts and their uncompressed checksum.
