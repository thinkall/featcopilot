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
    "featcopilot-paired",
    "openfe-inductive",
    "autofeat-default"
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
| real_world / development | classification | lightgbm | autofeat-default | 4/9 / 4/9 | 12/27 | 3/0/1 | +9.33% | [-1.07%, +19.73%] |
| real_world / development | classification | lightgbm | baseline | 9/9 / 9/9 | 27/27 | 0/9/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | classification | lightgbm | featcopilot-paired | 9/9 / 9/9 | 27/27 | 3/5/1 | +1.56% | [-0.52%, +4.38%] |
| real_world / development | classification | lightgbm | featcopilot-predictive | 9/9 / 9/9 | 27/27 | 1/7/1 | -0.33% | [-1.24%, +0.29%] |
| real_world / development | classification | lightgbm | openfe-inductive | 9/9 / 9/9 | 27/27 | 3/1/5 | +9.71% | [-1.53%, +25.56%] |
| real_world / development | classification | linear | autofeat-default | 4/9 / 4/9 | 12/27 | 2/0/2 | -9.43% | [-82.93%, +59.60%] |
| real_world / development | classification | linear | baseline | 9/9 / 9/9 | 27/27 | 0/9/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | classification | linear | featcopilot-paired | 9/9 / 9/9 | 27/27 | 7/1/1 | +14.18% | [+1.37%, +34.21%] |
| real_world / development | classification | linear | featcopilot-predictive | 9/9 / 9/9 | 27/27 | 5/4/0 | +14.09% | [+1.06%, +34.04%] |
| real_world / development | classification | linear | openfe-inductive | 9/9 / 9/9 | 27/27 | 6/2/1 | +11.24% | [+1.34%, +27.99%] |
| real_world / development | regression | lightgbm | autofeat-default | 6/6 / 6/6 | 18/18 | 3/1/2 | +1.03% | [-0.76%, +2.90%] |
| real_world / development | regression | lightgbm | baseline | 6/6 / 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | regression | lightgbm | featcopilot-paired | 6/6 / 6/6 | 18/18 | 2/2/2 | +0.97% | [-1.03%, +3.43%] |
| real_world / development | regression | lightgbm | featcopilot-predictive | 6/6 / 6/6 | 18/18 | 3/3/0 | +1.62% | [+0.35%, +3.57%] |
| real_world / development | regression | lightgbm | openfe-inductive | 6/6 / 6/6 | 18/18 | 3/2/1 | +2.56% | [-0.03%, +5.26%] |
| real_world / development | regression | linear | autofeat-default | 6/6 / 6/6 | 18/18 | 5/0/1 | +6.12% | [-14.74%, +24.38%] |
| real_world / development | regression | linear | baseline | 6/6 / 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | regression | linear | featcopilot-paired | 6/6 / 6/6 | 18/18 | 5/1/0 | +11.37% | [+4.56%, +20.97%] |
| real_world / development | regression | linear | featcopilot-predictive | 6/6 / 6/6 | 18/18 | 6/0/0 | +10.65% | [+4.20%, +20.22%] |
| real_world / development | regression | linear | openfe-inductive | 6/6 / 6/6 | 18/18 | 5/1/0 | +8.28% | [+1.93%, +18.21%] |
| simulation / development | regression | lightgbm | autofeat-default | 1/1 / 1/1 | 3/3 | 0/0/1 | -20.30% | n/a |
| simulation / development | regression | lightgbm | baseline | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | lightgbm | featcopilot-paired | 1/1 / 1/1 | 3/3 | 0/0/1 | -2.85% | n/a |
| simulation / development | regression | lightgbm | featcopilot-predictive | 1/1 / 1/1 | 3/3 | 0/0/1 | -2.85% | n/a |
| simulation / development | regression | lightgbm | openfe-inductive | 1/1 / 1/1 | 3/3 | 1/0/0 | +8.05% | n/a |
| simulation / development | regression | linear | autofeat-default | 1/1 / 1/1 | 3/3 | 1/0/0 | +28.20% | n/a |
| simulation / development | regression | linear | baseline | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | linear | featcopilot-paired | 1/1 / 1/1 | 3/3 | 1/0/0 | +39.75% | n/a |
| simulation / development | regression | linear | featcopilot-predictive | 1/1 / 1/1 | 3/3 | 1/0/0 | +32.04% | n/a |
| simulation / development | regression | linear | openfe-inductive | 1/1 / 1/1 | 3/3 | 1/0/0 | +43.87% | n/a |

## Per-dataset means (complete folds only)

| Dataset | Learner | Method | Primary loss | FE seconds (inner + outer) | Pipeline seconds | Features |
|---|---|---|---|---|---|---|
| houses | lightgbm | autofeat-default | 0.233479 | 47.11 | 48.34 | 25.7 |
| houses | lightgbm | baseline | 0.244681 | 0.02 | 1.33 | 8.0 |
| houses | lightgbm | featcopilot-paired | 0.239706 | 19.55 | 20.51 | 29.7 |
| houses | lightgbm | featcopilot-predictive | 0.241225 | 19.07 | 19.85 | 16.3 |
| houses | lightgbm | openfe-inductive | 0.227345 | 90.97 | 92.27 | 58.0 |
| houses | linear | autofeat-default | 0.470964 | 48.86 | 48.92 | 36.0 |
| houses | linear | baseline | 0.338541 | 0.02 | 0.08 | 8.0 |
| houses | linear | featcopilot-paired | 0.315165 | 3.62 | 3.70 | 15.0 |
| houses | linear | featcopilot-predictive | 0.319025 | 3.29 | 3.37 | 11.3 |
| houses | linear | openfe-inductive | 0.319067 | 88.81 | 88.92 | 58.0 |
| uci-adult | lightgbm | baseline | 0.309826 | 0.06 | 1.50 | 13.0 |
| uci-adult | lightgbm | featcopilot-paired | 0.312879 | 16.64 | 17.47 | 29.7 |
| uci-adult | lightgbm | featcopilot-predictive | 0.309826 | 15.71 | 16.51 | 13.0 |
| uci-adult | lightgbm | openfe-inductive | 0.322537 | 101.66 | 103.00 | 63.0 |
| uci-adult | linear | baseline | 0.340533 | 0.06 | 0.36 | 13.0 |
| uci-adult | linear | featcopilot-paired | 0.333762 | 78.79 | 79.16 | 18.0 |
| uci-adult | linear | featcopilot-predictive | 0.340533 | 77.71 | 78.06 | 13.0 |
| uci-adult | linear | openfe-inductive | 0.332074 | 98.11 | 98.90 | 63.0 |
| uci-auto-mpg | lightgbm | autofeat-default | 2.841589 | 39.35 | 40.04 | 14.0 |
| uci-auto-mpg | lightgbm | baseline | 2.924286 | 0.02 | 0.88 | 7.0 |
| uci-auto-mpg | lightgbm | featcopilot-paired | 2.981749 | 7.39 | 7.66 | 15.3 |
| uci-auto-mpg | lightgbm | featcopilot-predictive | 2.924286 | 7.23 | 7.48 | 7.0 |
| uci-auto-mpg | lightgbm | openfe-inductive | 2.906142 | 79.81 | 80.06 | 8.0 |
| uci-auto-mpg | linear | autofeat-default | 2.857214 | 39.64 | 39.70 | 17.0 |
| uci-auto-mpg | linear | baseline | 3.372414 | 0.02 | 0.08 | 7.0 |
| uci-auto-mpg | linear | featcopilot-paired | 2.993322 | 3.05 | 3.12 | 12.0 |
| uci-auto-mpg | linear | featcopilot-predictive | 2.993322 | 3.25 | 3.32 | 12.0 |
| uci-auto-mpg | linear | openfe-inductive | 3.335482 | 81.73 | 81.79 | 8.0 |
| uci-banknote-authentication | lightgbm | autofeat-default | 0.014929 | 90.68 | 91.68 | 16.7 |
| uci-banknote-authentication | lightgbm | baseline | 0.015250 | 0.01 | 1.02 | 4.0 |
| uci-banknote-authentication | lightgbm | featcopilot-paired | 0.013493 | 8.78 | 9.20 | 20.7 |
| uci-banknote-authentication | lightgbm | featcopilot-predictive | 0.015795 | 8.76 | 9.20 | 37.3 |
| uci-banknote-authentication | lightgbm | openfe-inductive | 0.004879 | 87.12 | 87.58 | 53.0 |
| uci-banknote-authentication | linear | autofeat-default | 0.004910 | 91.31 | 91.41 | 21.7 |
| uci-banknote-authentication | linear | baseline | 0.029459 | 0.01 | 0.11 | 4.0 |
| uci-banknote-authentication | linear | featcopilot-paired | 0.003119 | 71.87 | 71.99 | 29.0 |
| uci-banknote-authentication | linear | featcopilot-predictive | 0.003119 | 71.12 | 71.23 | 29.0 |
| uci-banknote-authentication | linear | openfe-inductive | 0.007443 | 88.88 | 89.00 | 53.0 |
| uci-chronic-kidney-disease | lightgbm | autofeat-default | 0.028635 | 156.53 | 157.43 | 126.0 |
| uci-chronic-kidney-disease | lightgbm | baseline | 0.035504 | 0.05 | 0.96 | 24.0 |
| uci-chronic-kidney-disease | lightgbm | featcopilot-paired | 0.036179 | 11.49 | 11.97 | 24.3 |
| uci-chronic-kidney-disease | lightgbm | featcopilot-predictive | 0.035504 | 10.72 | 11.20 | 24.0 |
| uci-chronic-kidney-disease | lightgbm | openfe-inductive | 0.025707 | 83.23 | 83.55 | 25.0 |
| uci-chronic-kidney-disease | linear | autofeat-default | 0.034304 | 156.81 | 157.02 | 125.7 |
| uci-chronic-kidney-disease | linear | baseline | 0.016247 | 0.05 | 0.22 | 24.0 |
| uci-chronic-kidney-disease | linear | featcopilot-paired | 0.015507 | 71.25 | 71.57 | 26.0 |
| uci-chronic-kidney-disease | linear | featcopilot-predictive | 0.015507 | 73.19 | 73.51 | 26.0 |
| uci-chronic-kidney-disease | linear | openfe-inductive | 0.014241 | 82.37 | 82.54 | 25.0 |
| uci-combined-cycle-power | lightgbm | autofeat-default | 3.449861 | 40.54 | 41.70 | 14.3 |
| uci-combined-cycle-power | lightgbm | baseline | 3.382905 | 0.01 | 1.26 | 4.0 |
| uci-combined-cycle-power | lightgbm | featcopilot-paired | 3.162279 | 14.60 | 15.31 | 19.0 |
| uci-combined-cycle-power | lightgbm | featcopilot-predictive | 3.173097 | 14.54 | 15.24 | 16.0 |
| uci-combined-cycle-power | lightgbm | openfe-inductive | 3.155407 | 87.59 | 88.71 | 47.7 |
| uci-combined-cycle-power | linear | autofeat-default | 4.169598 | 39.43 | 39.48 | 13.7 |
| uci-combined-cycle-power | linear | baseline | 4.474993 | 0.01 | 0.06 | 4.0 |
| uci-combined-cycle-power | linear | featcopilot-paired | 4.110473 | 2.89 | 2.98 | 54.0 |
| uci-combined-cycle-power | linear | featcopilot-predictive | 4.139593 | 2.59 | 2.65 | 22.3 |
| uci-combined-cycle-power | linear | openfe-inductive | 4.130092 | 87.77 | 87.87 | 47.7 |
| uci-concrete-strength | lightgbm | autofeat-default | 4.556039 | 50.94 | 51.85 | 30.3 |
| uci-concrete-strength | lightgbm | baseline | 4.553620 | 0.02 | 1.04 | 8.0 |
| uci-concrete-strength | lightgbm | featcopilot-paired | 4.621215 | 11.50 | 12.00 | 16.3 |
| uci-concrete-strength | lightgbm | featcopilot-predictive | 4.491002 | 10.79 | 11.22 | 8.0 |
| uci-concrete-strength | lightgbm | openfe-inductive | 4.645433 | 87.29 | 87.90 | 58.0 |
| uci-concrete-strength | linear | autofeat-default | 6.075692 | 53.53 | 53.58 | 27.3 |
| uci-concrete-strength | linear | baseline | 10.559292 | 0.02 | 0.07 | 8.0 |
| uci-concrete-strength | linear | featcopilot-paired | 7.056526 | 2.93 | 2.99 | 28.3 |
| uci-concrete-strength | linear | featcopilot-predictive | 7.054075 | 2.93 | 2.98 | 28.3 |
| uci-concrete-strength | linear | openfe-inductive | 7.188370 | 88.13 | 88.19 | 58.0 |
| uci-dry-bean | lightgbm | baseline | 0.248388 | 0.04 | 3.89 | 16.0 |
| uci-dry-bean | lightgbm | featcopilot-paired | 0.246992 | 84.32 | 87.47 | 19.7 |
| uci-dry-bean | lightgbm | featcopilot-predictive | 0.248388 | 75.55 | 78.77 | 16.0 |
| uci-dry-bean | lightgbm | openfe-inductive | 0.254248 | 101.06 | 105.97 | 66.0 |
| uci-dry-bean | linear | baseline | 0.216633 | 0.04 | 0.43 | 16.0 |
| uci-dry-bean | linear | featcopilot-paired | 0.213023 | 77.79 | 78.39 | 66.0 |
| uci-dry-bean | linear | featcopilot-predictive | 0.214540 | 80.59 | 81.05 | 24.3 |
| uci-dry-bean | linear | openfe-inductive | 0.216792 | 100.18 | 101.05 | 66.0 |
| uci-energy-efficiency | lightgbm | autofeat-default | 0.577120 | 46.26 | 47.04 | 27.7 |
| uci-energy-efficiency | lightgbm | baseline | 0.479742 | 0.02 | 0.93 | 8.0 |
| uci-energy-efficiency | lightgbm | featcopilot-paired | 0.493421 | 9.18 | 9.53 | 8.0 |
| uci-energy-efficiency | lightgbm | featcopilot-predictive | 0.493421 | 8.85 | 9.17 | 8.0 |
| uci-energy-efficiency | lightgbm | openfe-inductive | 0.441135 | 85.47 | 85.83 | 57.0 |
| uci-energy-efficiency | linear | autofeat-default | 2.042312 | 46.92 | 46.98 | 27.7 |
| uci-energy-efficiency | linear | baseline | 2.844552 | 0.02 | 0.08 | 8.0 |
| uci-energy-efficiency | linear | featcopilot-paired | 1.713867 | 3.41 | 3.50 | 58.0 |
| uci-energy-efficiency | linear | featcopilot-predictive | 1.933146 | 3.15 | 3.23 | 33.0 |
| uci-energy-efficiency | linear | openfe-inductive | 1.596539 | 86.21 | 86.27 | 57.0 |
| uci-german-credit | lightgbm | autofeat-default | 0.541834 | 324.19 | 325.33 | 81.3 |
| uci-german-credit | lightgbm | baseline | 0.519777 | 0.04 | 1.17 | 20.0 |
| uci-german-credit | lightgbm | featcopilot-paired | 0.522024 | 12.66 | 13.28 | 20.7 |
| uci-german-credit | lightgbm | featcopilot-predictive | 0.522024 | 12.50 | 13.13 | 20.7 |
| uci-german-credit | lightgbm | openfe-inductive | 0.532456 | 91.21 | 91.90 | 70.0 |
| uci-german-credit | linear | autofeat-default | 0.556221 | 315.93 | 316.16 | 81.7 |
| uci-german-credit | linear | baseline | 0.498452 | 0.04 | 0.25 | 20.0 |
| uci-german-credit | linear | featcopilot-paired | 0.499897 | 74.52 | 74.79 | 28.3 |
| uci-german-credit | linear | featcopilot-predictive | 0.501817 | 72.93 | 73.20 | 23.3 |
| uci-german-credit | linear | openfe-inductive | 0.499680 | 90.57 | 90.87 | 70.0 |
| uci-online-shoppers | lightgbm | baseline | 0.380834 | 0.05 | 1.38 | 11.0 |
| uci-online-shoppers | lightgbm | featcopilot-paired | 0.381682 | 15.89 | 16.63 | 19.7 |
| uci-online-shoppers | lightgbm | featcopilot-predictive | 0.380834 | 14.70 | 15.47 | 11.0 |
| uci-online-shoppers | lightgbm | openfe-inductive | 0.384828 | 99.40 | 100.71 | 61.0 |
| uci-online-shoppers | linear | baseline | 0.395244 | 0.05 | 0.31 | 11.0 |
| uci-online-shoppers | linear | featcopilot-paired | 0.385196 | 80.15 | 80.45 | 14.7 |
| uci-online-shoppers | linear | featcopilot-predictive | 0.385326 | 77.52 | 77.81 | 13.3 |
| uci-online-shoppers | linear | openfe-inductive | 0.388122 | 96.94 | 97.71 | 61.0 |
| uci-real-estate-sindian | lightgbm | autofeat-default | 7.539188 | 38.38 | 39.08 | 11.3 |
| uci-real-estate-sindian | lightgbm | baseline | 7.448276 | 0.02 | 0.84 | 6.0 |
| uci-real-estate-sindian | lightgbm | featcopilot-paired | 7.464846 | 7.09 | 7.34 | 7.7 |
| uci-real-estate-sindian | lightgbm | featcopilot-predictive | 7.448276 | 6.69 | 6.92 | 6.0 |
| uci-real-estate-sindian | lightgbm | openfe-inductive | 7.448276 | 81.05 | 81.29 | 7.0 |
| uci-real-estate-sindian | linear | autofeat-default | 7.960999 | 39.94 | 39.98 | 10.7 |
| uci-real-estate-sindian | linear | baseline | 8.793747 | 0.02 | 0.06 | 6.0 |
| uci-real-estate-sindian | linear | featcopilot-paired | 8.055879 | 2.52 | 2.56 | 16.3 |
| uci-real-estate-sindian | linear | featcopilot-predictive | 8.378888 | 2.36 | 2.40 | 6.7 |
| uci-real-estate-sindian | linear | openfe-inductive | 8.792728 | 80.65 | 80.69 | 7.0 |
| uci-spambase | lightgbm | baseline | 0.141042 | 0.15 | 2.14 | 57.0 |
| uci-spambase | lightgbm | featcopilot-paired | 0.141557 | 41.52 | 42.92 | 58.7 |
| uci-spambase | lightgbm | featcopilot-predictive | 0.141042 | 36.62 | 37.92 | 57.0 |
| uci-spambase | lightgbm | openfe-inductive | 0.142694 | 151.96 | 153.71 | 107.0 |
| uci-spambase | linear | baseline | 0.239802 | 0.15 | 0.40 | 57.0 |
| uci-spambase | linear | featcopilot-paired | 0.179039 | 80.61 | 80.91 | 93.7 |
| uci-spambase | linear | featcopilot-predictive | 0.175298 | 83.61 | 83.90 | 90.3 |
| uci-spambase | linear | openfe-inductive | 0.230953 | 152.84 | 153.17 | 107.0 |
| uci-steel-faults | lightgbm | baseline | 0.557708 | 0.07 | 3.74 | 27.0 |
| uci-steel-faults | lightgbm | featcopilot-paired | 0.545787 | 74.68 | 77.89 | 32.0 |
| uci-steel-faults | lightgbm | featcopilot-predictive | 0.552069 | 74.48 | 77.68 | 33.7 |
| uci-steel-faults | lightgbm | openfe-inductive | 0.552266 | 113.33 | 117.57 | 77.0 |
| uci-steel-faults | linear | baseline | 0.764914 | 0.07 | 0.52 | 27.0 |
| uci-steel-faults | linear | featcopilot-paired | 0.733784 | 82.47 | 83.06 | 32.3 |
| uci-steel-faults | linear | featcopilot-predictive | 0.740742 | 79.83 | 80.37 | 29.0 |
| uci-steel-faults | linear | openfe-inductive | 0.707839 | 113.13 | 113.91 | 77.0 |
| uci-wisconsin-diagnostic | lightgbm | autofeat-default | 0.096205 | 386.14 | 386.86 | 40.3 |
| uci-wisconsin-diagnostic | lightgbm | baseline | 0.120432 | 0.07 | 1.00 | 30.0 |
| uci-wisconsin-diagnostic | lightgbm | featcopilot-paired | 0.115976 | 25.46 | 25.83 | 38.3 |
| uci-wisconsin-diagnostic | lightgbm | featcopilot-predictive | 0.120432 | 22.96 | 23.31 | 30.0 |
| uci-wisconsin-diagnostic | lightgbm | openfe-inductive | 0.118061 | 84.68 | 84.99 | 31.0 |
| uci-wisconsin-diagnostic | linear | autofeat-default | 0.087852 | 388.10 | 388.20 | 39.0 |
| uci-wisconsin-diagnostic | linear | baseline | 0.089368 | 0.07 | 0.20 | 30.0 |
| uci-wisconsin-diagnostic | linear | featcopilot-paired | 0.090855 | 73.45 | 73.60 | 33.3 |
| uci-wisconsin-diagnostic | linear | featcopilot-predictive | 0.089368 | 73.72 | 73.87 | 30.0 |
| uci-wisconsin-diagnostic | linear | openfe-inductive | 0.090266 | 86.29 | 86.39 | 31.0 |
| wine | lightgbm | autofeat-default | 0.640975 | 56.74 | 58.17 | 42.0 |
| wine | lightgbm | baseline | 0.654417 | 0.03 | 1.20 | 11.0 |
| wine | lightgbm | featcopilot-paired | 0.648129 | 21.18 | 22.01 | 27.7 |
| wine | lightgbm | featcopilot-predictive | 0.649759 | 20.23 | 20.92 | 11.0 |
| wine | lightgbm | openfe-inductive | 0.634993 | 89.32 | 90.59 | 61.0 |
| wine | linear | autofeat-default | 0.724187 | 58.70 | 58.78 | 48.7 |
| wine | linear | baseline | 0.737508 | 0.03 | 0.09 | 11.0 |
| wine | linear | featcopilot-paired | 0.734912 | 4.42 | 4.52 | 47.7 |
| wine | linear | featcopilot-predictive | 0.726537 | 3.96 | 4.05 | 28.0 |
| wine | linear | openfe-inductive | 0.714151 | 89.53 | 89.65 | 61.0 |

## Failures / timeouts

- uci-steel-faults / 42 / fold 0 / autofeat-default / linear: timeout: Case exceeded 600.0s wall-clock limit
- uci-dry-bean / 42 / fold 0 / autofeat-default / lightgbm: timeout: Case exceeded 600.0s wall-clock limit
- uci-online-shoppers / 42 / fold 0 / autofeat-default / lightgbm: error: ValueError: Input X contains NaN. LogisticRegressionCV does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is 
- uci-spambase / 42 / fold 0 / autofeat-default / linear: memory_limit: Worker tree exceeded the configured RSS memory limit
- uci-adult / 42 / fold 0 / autofeat-default / lightgbm: error: ValueError: Input X contains NaN. LogisticRegressionCV does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is 
- uci-steel-faults / 42 / fold 0 / autofeat-default / lightgbm: timeout: Case exceeded 600.0s wall-clock limit
- uci-online-shoppers / 42 / fold 0 / autofeat-default / linear: error: ValueError: Input X contains NaN. LogisticRegressionCV does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is 
- uci-adult / 42 / fold 0 / autofeat-default / linear: error: ValueError: Input X contains NaN. LogisticRegressionCV does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is 
- uci-spambase / 42 / fold 0 / autofeat-default / lightgbm: memory_limit: Worker tree exceeded the configured RSS memory limit
- uci-dry-bean / 42 / fold 0 / autofeat-default / linear: timeout: Case exceeded 600.0s wall-clock limit
- uci-adult / 42 / fold 1 / autofeat-default / linear: error: ValueError: Input X contains NaN. LogisticRegressionCV does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is 
- uci-dry-bean / 42 / fold 1 / autofeat-default / lightgbm: timeout: Case exceeded 600.0s wall-clock limit
- uci-spambase / 42 / fold 1 / autofeat-default / linear: memory_limit: Worker tree exceeded the configured RSS memory limit
- uci-spambase / 42 / fold 1 / autofeat-default / lightgbm: memory_limit: Worker tree exceeded the configured RSS memory limit
- uci-steel-faults / 42 / fold 1 / autofeat-default / linear: timeout: Case exceeded 600.0s wall-clock limit
- uci-adult / 42 / fold 1 / autofeat-default / lightgbm: error: ValueError: Input X contains NaN. LogisticRegressionCV does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is 
- uci-online-shoppers / 42 / fold 1 / autofeat-default / linear: error: ValueError: Input X contains NaN. LogisticRegressionCV does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is 
- uci-steel-faults / 42 / fold 1 / autofeat-default / lightgbm: timeout: Case exceeded 600.0s wall-clock limit
- uci-online-shoppers / 42 / fold 1 / autofeat-default / lightgbm: error: ValueError: Input X contains NaN. LogisticRegressionCV does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is 
- uci-dry-bean / 42 / fold 1 / autofeat-default / linear: timeout: Case exceeded 600.0s wall-clock limit
- uci-steel-faults / 42 / fold 2 / autofeat-default / lightgbm: timeout: Case exceeded 600.0s wall-clock limit
- uci-adult / 42 / fold 2 / autofeat-default / linear: error: ValueError: Input X contains NaN. LogisticRegressionCV does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is 
- uci-online-shoppers / 42 / fold 2 / autofeat-default / lightgbm: error: ValueError: Input X contains NaN. LogisticRegressionCV does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is 
- uci-adult / 42 / fold 2 / autofeat-default / lightgbm: error: ValueError: Input X contains NaN. LogisticRegressionCV does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is 
- uci-spambase / 42 / fold 2 / autofeat-default / linear: memory_limit: Worker tree exceeded the configured RSS memory limit
- uci-dry-bean / 42 / fold 2 / autofeat-default / linear: timeout: Case exceeded 600.0s wall-clock limit
- uci-steel-faults / 42 / fold 2 / autofeat-default / linear: timeout: Case exceeded 600.0s wall-clock limit
- uci-online-shoppers / 42 / fold 2 / autofeat-default / linear: error: ValueError: Input X contains NaN. LogisticRegressionCV does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is 
- uci-spambase / 42 / fold 2 / autofeat-default / lightgbm: memory_limit: Worker tree exceeded the configured RSS memory limit
- uci-dry-bean / 42 / fold 2 / autofeat-default / lightgbm: timeout: Case exceeded 600.0s wall-clock limit

Raw cell metrics and provenance: local `results.json`. Completed runs also write `archive.json`,
which lists the authoritative lossless gzip parts and their uncompressed checksum.
