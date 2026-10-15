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
    "featcopilot-paired",
    "featcopilot-extended",
    "featcopilot-extended-paired",
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
| real_world / development | classification | lightgbm | featcopilot-extended | 9/9 / 9/9 | 27/27 | 2/7/0 | +0.99% | [-0.13%, +2.76%] |
| real_world / development | classification | lightgbm | featcopilot-extended-paired | 9/9 / 9/9 | 27/27 | 3/5/1 | +1.81% | [+0.06%, +4.56%] |
| real_world / development | classification | lightgbm | featcopilot-paired | 9/9 / 9/9 | 27/27 | 3/5/1 | +1.56% | [-0.52%, +4.38%] |
| real_world / development | classification | lightgbm | featcopilot-predictive | 9/9 / 9/9 | 27/27 | 1/7/1 | -0.33% | [-1.24%, +0.29%] |
| real_world / development | classification | lightgbm | openfe-inductive | 9/9 / 9/9 | 27/27 | 3/1/5 | +9.71% | [-1.53%, +25.56%] |
| real_world / development | classification | linear | baseline | 9/9 / 9/9 | 27/27 | 0/9/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | classification | linear | featcopilot-extended | 9/9 / 9/9 | 27/27 | 5/3/1 | +11.48% | [-2.73%, +34.18%] |
| real_world / development | classification | linear | featcopilot-extended-paired | 9/9 / 9/9 | 27/27 | 5/3/1 | +8.70% | [-12.13%, +34.30%] |
| real_world / development | classification | linear | featcopilot-paired | 9/9 / 9/9 | 27/27 | 7/1/1 | +14.18% | [+1.37%, +34.21%] |
| real_world / development | classification | linear | featcopilot-predictive | 9/9 / 9/9 | 27/27 | 5/4/0 | +14.09% | [+1.06%, +34.04%] |
| real_world / development | classification | linear | openfe-inductive | 9/9 / 9/9 | 27/27 | 6/2/1 | +11.24% | [+1.34%, +27.99%] |
| real_world / development | regression | lightgbm | baseline | 6/6 / 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | regression | lightgbm | featcopilot-extended | 6/6 / 6/6 | 18/18 | 2/3/1 | +1.91% | [-0.37%, +4.46%] |
| real_world / development | regression | lightgbm | featcopilot-extended-paired | 6/6 / 6/6 | 18/18 | 3/1/2 | +1.70% | [-1.08%, +4.52%] |
| real_world / development | regression | lightgbm | featcopilot-paired | 6/6 / 6/6 | 18/18 | 2/2/2 | +0.97% | [-1.03%, +3.43%] |
| real_world / development | regression | lightgbm | featcopilot-predictive | 6/6 / 6/6 | 18/18 | 3/3/0 | +1.62% | [+0.35%, +3.57%] |
| real_world / development | regression | lightgbm | openfe-inductive | 6/6 / 6/6 | 18/18 | 3/2/1 | +2.56% | [-0.03%, +5.26%] |
| real_world / development | regression | linear | baseline | 6/6 / 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | regression | linear | featcopilot-extended | 6/6 / 6/6 | 18/18 | 6/0/0 | +9.81% | [+3.92%, +19.32%] |
| real_world / development | regression | linear | featcopilot-extended-paired | 6/6 / 6/6 | 18/18 | 5/1/0 | +11.29% | [+4.06%, +20.87%] |
| real_world / development | regression | linear | featcopilot-paired | 6/6 / 6/6 | 18/18 | 5/1/0 | +11.37% | [+4.56%, +20.97%] |
| real_world / development | regression | linear | featcopilot-predictive | 6/6 / 6/6 | 18/18 | 6/0/0 | +10.65% | [+4.20%, +20.22%] |
| real_world / development | regression | linear | openfe-inductive | 6/6 / 6/6 | 18/18 | 5/1/0 | +8.28% | [+1.93%, +18.21%] |
| simulation / development | regression | lightgbm | baseline | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | lightgbm | featcopilot-extended | 1/1 / 1/1 | 3/3 | 0/0/1 | -2.85% | n/a |
| simulation / development | regression | lightgbm | featcopilot-extended-paired | 1/1 / 1/1 | 3/3 | 0/0/1 | -2.34% | n/a |
| simulation / development | regression | lightgbm | featcopilot-paired | 1/1 / 1/1 | 3/3 | 0/0/1 | -2.85% | n/a |
| simulation / development | regression | lightgbm | featcopilot-predictive | 1/1 / 1/1 | 3/3 | 0/0/1 | -2.85% | n/a |
| simulation / development | regression | lightgbm | openfe-inductive | 1/1 / 1/1 | 3/3 | 1/0/0 | +8.05% | n/a |
| simulation / development | regression | linear | baseline | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | linear | featcopilot-extended | 1/1 / 1/1 | 3/3 | 1/0/0 | +32.65% | n/a |
| simulation / development | regression | linear | featcopilot-extended-paired | 1/1 / 1/1 | 3/3 | 1/0/0 | +38.01% | n/a |
| simulation / development | regression | linear | featcopilot-paired | 1/1 / 1/1 | 3/3 | 1/0/0 | +39.75% | n/a |
| simulation / development | regression | linear | featcopilot-predictive | 1/1 / 1/1 | 3/3 | 1/0/0 | +32.04% | n/a |
| simulation / development | regression | linear | openfe-inductive | 1/1 / 1/1 | 3/3 | 1/0/0 | +43.87% | n/a |

## Per-dataset means (complete folds only)

| Dataset | Learner | Method | Primary loss | FE seconds (inner + outer) | Pipeline seconds | Features |
|---|---|---|---|---|---|---|
| houses | lightgbm | baseline | 0.244681 | 0.02 | 1.35 | 8.0 |
| houses | lightgbm | featcopilot-extended | 0.231745 | 22.39 | 23.36 | 28.0 |
| houses | lightgbm | featcopilot-extended-paired | 0.232031 | 23.11 | 24.19 | 49.7 |
| houses | lightgbm | featcopilot-paired | 0.239706 | 19.48 | 20.46 | 29.7 |
| houses | lightgbm | featcopilot-predictive | 0.241225 | 19.25 | 20.04 | 16.3 |
| houses | lightgbm | openfe-inductive | 0.227345 | 105.59 | 106.81 | 58.0 |
| houses | linear | baseline | 0.338541 | 0.02 | 0.08 | 8.0 |
| houses | linear | featcopilot-extended | 0.318972 | 4.07 | 4.14 | 11.3 |
| houses | linear | featcopilot-extended-paired | 0.310813 | 4.05 | 4.14 | 28.3 |
| houses | linear | featcopilot-paired | 0.315165 | 3.53 | 3.62 | 15.0 |
| houses | linear | featcopilot-predictive | 0.319025 | 3.36 | 3.44 | 11.3 |
| houses | linear | openfe-inductive | 0.319067 | 103.64 | 103.76 | 58.0 |
| uci-adult | lightgbm | baseline | 0.309826 | 0.06 | 1.59 | 13.0 |
| uci-adult | lightgbm | featcopilot-extended | 0.309826 | 25.03 | 25.98 | 13.0 |
| uci-adult | lightgbm | featcopilot-extended-paired | 0.307939 | 26.69 | 27.63 | 16.7 |
| uci-adult | lightgbm | featcopilot-paired | 0.312879 | 17.22 | 18.08 | 29.7 |
| uci-adult | lightgbm | featcopilot-predictive | 0.309826 | 15.41 | 16.23 | 13.0 |
| uci-adult | lightgbm | openfe-inductive | 0.322537 | 111.04 | 112.37 | 63.0 |
| uci-adult | linear | baseline | 0.340533 | 0.06 | 0.36 | 13.0 |
| uci-adult | linear | featcopilot-extended | 0.340533 | 98.34 | 98.73 | 13.0 |
| uci-adult | linear | featcopilot-extended-paired | 0.335285 | 99.70 | 100.10 | 23.0 |
| uci-adult | linear | featcopilot-paired | 0.333762 | 93.07 | 93.45 | 18.0 |
| uci-adult | linear | featcopilot-predictive | 0.340533 | 96.63 | 96.97 | 13.0 |
| uci-adult | linear | openfe-inductive | 0.332074 | 119.15 | 119.95 | 63.0 |
| uci-auto-mpg | lightgbm | baseline | 2.924286 | 0.02 | 0.90 | 7.0 |
| uci-auto-mpg | lightgbm | featcopilot-extended | 2.924286 | 8.57 | 8.83 | 7.0 |
| uci-auto-mpg | lightgbm | featcopilot-extended-paired | 3.008955 | 8.48 | 8.78 | 23.7 |
| uci-auto-mpg | lightgbm | featcopilot-paired | 2.981749 | 7.65 | 7.91 | 15.3 |
| uci-auto-mpg | lightgbm | featcopilot-predictive | 2.924286 | 7.27 | 7.53 | 7.0 |
| uci-auto-mpg | lightgbm | openfe-inductive | 2.906142 | 93.92 | 94.18 | 8.0 |
| uci-auto-mpg | linear | baseline | 3.372414 | 0.02 | 0.07 | 7.0 |
| uci-auto-mpg | linear | featcopilot-extended | 3.130025 | 3.31 | 3.38 | 9.3 |
| uci-auto-mpg | linear | featcopilot-extended-paired | 3.040278 | 3.33 | 3.41 | 12.3 |
| uci-auto-mpg | linear | featcopilot-paired | 2.993322 | 3.46 | 3.53 | 12.0 |
| uci-auto-mpg | linear | featcopilot-predictive | 2.993322 | 3.40 | 3.47 | 12.0 |
| uci-auto-mpg | linear | openfe-inductive | 3.335482 | 96.48 | 96.53 | 8.0 |
| uci-banknote-authentication | lightgbm | baseline | 0.015250 | 0.01 | 1.12 | 4.0 |
| uci-banknote-authentication | lightgbm | featcopilot-extended | 0.014091 | 8.99 | 9.37 | 12.7 |
| uci-banknote-authentication | lightgbm | featcopilot-extended-paired | 0.013413 | 8.88 | 9.24 | 14.0 |
| uci-banknote-authentication | lightgbm | featcopilot-paired | 0.013493 | 8.94 | 9.38 | 20.7 |
| uci-banknote-authentication | lightgbm | featcopilot-predictive | 0.015795 | 8.58 | 9.00 | 37.3 |
| uci-banknote-authentication | lightgbm | openfe-inductive | 0.004879 | 102.97 | 103.43 | 53.0 |
| uci-banknote-authentication | linear | baseline | 0.029459 | 0.01 | 0.11 | 4.0 |
| uci-banknote-authentication | linear | featcopilot-extended | 0.001290 | 89.98 | 90.08 | 45.7 |
| uci-banknote-authentication | linear | featcopilot-extended-paired | 0.001290 | 86.30 | 86.40 | 45.7 |
| uci-banknote-authentication | linear | featcopilot-paired | 0.003119 | 85.84 | 85.95 | 29.0 |
| uci-banknote-authentication | linear | featcopilot-predictive | 0.003119 | 84.99 | 85.10 | 29.0 |
| uci-banknote-authentication | linear | openfe-inductive | 0.007443 | 105.79 | 105.92 | 53.0 |
| uci-chronic-kidney-disease | lightgbm | baseline | 0.035504 | 0.05 | 1.03 | 24.0 |
| uci-chronic-kidney-disease | lightgbm | featcopilot-extended | 0.034860 | 23.74 | 24.49 | 24.7 |
| uci-chronic-kidney-disease | lightgbm | featcopilot-extended-paired | 0.034633 | 23.02 | 23.77 | 27.7 |
| uci-chronic-kidney-disease | lightgbm | featcopilot-paired | 0.036179 | 11.72 | 12.21 | 24.3 |
| uci-chronic-kidney-disease | lightgbm | featcopilot-predictive | 0.035504 | 11.12 | 11.59 | 24.0 |
| uci-chronic-kidney-disease | lightgbm | openfe-inductive | 0.025707 | 98.21 | 98.54 | 25.0 |
| uci-chronic-kidney-disease | linear | baseline | 0.016247 | 0.05 | 0.22 | 24.0 |
| uci-chronic-kidney-disease | linear | featcopilot-extended | 0.018647 | 93.68 | 94.27 | 57.7 |
| uci-chronic-kidney-disease | linear | featcopilot-extended-paired | 0.023765 | 89.44 | 90.04 | 41.0 |
| uci-chronic-kidney-disease | linear | featcopilot-paired | 0.015507 | 87.07 | 87.41 | 26.0 |
| uci-chronic-kidney-disease | linear | featcopilot-predictive | 0.015507 | 88.80 | 89.13 | 26.0 |
| uci-chronic-kidney-disease | linear | openfe-inductive | 0.014241 | 98.88 | 99.05 | 25.0 |
| uci-combined-cycle-power | lightgbm | baseline | 3.382905 | 0.01 | 1.37 | 4.0 |
| uci-combined-cycle-power | lightgbm | featcopilot-extended | 3.148677 | 14.54 | 15.24 | 12.3 |
| uci-combined-cycle-power | lightgbm | featcopilot-extended-paired | 3.148677 | 14.85 | 15.60 | 12.3 |
| uci-combined-cycle-power | lightgbm | featcopilot-paired | 3.162279 | 14.74 | 15.46 | 19.0 |
| uci-combined-cycle-power | lightgbm | featcopilot-predictive | 3.173097 | 14.41 | 15.09 | 16.0 |
| uci-combined-cycle-power | lightgbm | openfe-inductive | 3.155407 | 104.40 | 105.57 | 47.7 |
| uci-combined-cycle-power | linear | baseline | 4.474993 | 0.01 | 0.06 | 4.0 |
| uci-combined-cycle-power | linear | featcopilot-extended | 4.143538 | 2.85 | 2.91 | 22.3 |
| uci-combined-cycle-power | linear | featcopilot-extended-paired | 4.105969 | 2.82 | 2.92 | 45.7 |
| uci-combined-cycle-power | linear | featcopilot-paired | 4.110473 | 2.81 | 2.91 | 54.0 |
| uci-combined-cycle-power | linear | featcopilot-predictive | 4.139593 | 2.67 | 2.73 | 22.3 |
| uci-combined-cycle-power | linear | openfe-inductive | 4.130092 | 102.86 | 102.96 | 47.7 |
| uci-concrete-strength | lightgbm | baseline | 4.553620 | 0.02 | 1.09 | 8.0 |
| uci-concrete-strength | lightgbm | featcopilot-extended | 4.619720 | 12.73 | 13.17 | 8.3 |
| uci-concrete-strength | lightgbm | featcopilot-extended-paired | 4.458800 | 13.50 | 14.02 | 24.7 |
| uci-concrete-strength | lightgbm | featcopilot-paired | 4.621215 | 11.72 | 12.22 | 16.3 |
| uci-concrete-strength | lightgbm | featcopilot-predictive | 4.491002 | 10.97 | 11.44 | 8.0 |
| uci-concrete-strength | lightgbm | openfe-inductive | 4.645433 | 103.26 | 103.86 | 58.0 |
| uci-concrete-strength | linear | baseline | 10.559292 | 0.02 | 0.07 | 8.0 |
| uci-concrete-strength | linear | featcopilot-extended | 7.146132 | 3.42 | 3.48 | 20.0 |
| uci-concrete-strength | linear | featcopilot-extended-paired | 7.094121 | 3.43 | 3.49 | 28.3 |
| uci-concrete-strength | linear | featcopilot-paired | 7.056526 | 3.02 | 3.07 | 28.3 |
| uci-concrete-strength | linear | featcopilot-predictive | 7.054075 | 3.06 | 3.12 | 28.3 |
| uci-concrete-strength | linear | openfe-inductive | 7.188370 | 101.36 | 101.42 | 58.0 |
| uci-dry-bean | lightgbm | baseline | 0.248388 | 0.05 | 4.09 | 16.0 |
| uci-dry-bean | lightgbm | featcopilot-extended | 0.248388 | 84.21 | 87.31 | 16.0 |
| uci-dry-bean | lightgbm | featcopilot-extended-paired | 0.246992 | 100.83 | 104.17 | 19.7 |
| uci-dry-bean | lightgbm | featcopilot-paired | 0.246992 | 85.51 | 88.68 | 19.7 |
| uci-dry-bean | lightgbm | featcopilot-predictive | 0.248388 | 75.24 | 78.41 | 16.0 |
| uci-dry-bean | lightgbm | openfe-inductive | 0.254248 | 119.90 | 124.69 | 66.0 |
| uci-dry-bean | linear | baseline | 0.216633 | 0.04 | 0.43 | 16.0 |
| uci-dry-bean | linear | featcopilot-extended | 0.214321 | 94.87 | 95.34 | 32.7 |
| uci-dry-bean | linear | featcopilot-extended-paired | 0.215113 | 95.14 | 95.76 | 32.7 |
| uci-dry-bean | linear | featcopilot-paired | 0.213023 | 93.20 | 93.79 | 66.0 |
| uci-dry-bean | linear | featcopilot-predictive | 0.214540 | 94.06 | 94.50 | 24.3 |
| uci-dry-bean | linear | openfe-inductive | 0.216792 | 119.47 | 120.35 | 66.0 |
| uci-energy-efficiency | lightgbm | baseline | 0.479742 | 0.02 | 1.15 | 8.0 |
| uci-energy-efficiency | lightgbm | featcopilot-extended | 0.493421 | 9.78 | 10.11 | 8.0 |
| uci-energy-efficiency | lightgbm | featcopilot-extended-paired | 0.490971 | 9.88 | 10.23 | 24.7 |
| uci-energy-efficiency | lightgbm | featcopilot-paired | 0.493421 | 8.93 | 9.25 | 8.0 |
| uci-energy-efficiency | lightgbm | featcopilot-predictive | 0.493421 | 10.15 | 10.54 | 8.0 |
| uci-energy-efficiency | lightgbm | openfe-inductive | 0.441135 | 98.90 | 99.25 | 57.0 |
| uci-energy-efficiency | linear | baseline | 2.844552 | 0.02 | 0.09 | 8.0 |
| uci-energy-efficiency | linear | featcopilot-extended | 1.915893 | 3.55 | 3.64 | 41.3 |
| uci-energy-efficiency | linear | featcopilot-extended-paired | 1.763395 | 3.60 | 3.69 | 58.0 |
| uci-energy-efficiency | linear | featcopilot-paired | 1.713867 | 3.53 | 3.61 | 58.0 |
| uci-energy-efficiency | linear | featcopilot-predictive | 1.933146 | 3.48 | 3.56 | 33.0 |
| uci-energy-efficiency | linear | openfe-inductive | 1.596539 | 99.25 | 99.32 | 57.0 |
| uci-german-credit | lightgbm | baseline | 0.519777 | 0.05 | 1.26 | 20.0 |
| uci-german-credit | lightgbm | featcopilot-extended | 0.523334 | 22.48 | 23.29 | 21.7 |
| uci-german-credit | lightgbm | featcopilot-extended-paired | 0.525551 | 22.49 | 23.28 | 22.0 |
| uci-german-credit | lightgbm | featcopilot-paired | 0.522024 | 12.69 | 13.31 | 20.7 |
| uci-german-credit | lightgbm | featcopilot-predictive | 0.522024 | 12.48 | 13.09 | 20.7 |
| uci-german-credit | lightgbm | openfe-inductive | 0.532456 | 106.04 | 106.73 | 70.0 |
| uci-german-credit | linear | baseline | 0.498452 | 0.04 | 0.24 | 20.0 |
| uci-german-credit | linear | featcopilot-extended | 0.498452 | 92.28 | 92.72 | 20.0 |
| uci-german-credit | linear | featcopilot-extended-paired | 0.498452 | 91.55 | 91.99 | 20.0 |
| uci-german-credit | linear | featcopilot-paired | 0.499897 | 89.51 | 89.79 | 28.3 |
| uci-german-credit | linear | featcopilot-predictive | 0.501817 | 86.94 | 87.22 | 23.3 |
| uci-german-credit | linear | openfe-inductive | 0.499680 | 105.33 | 105.63 | 70.0 |
| uci-online-shoppers | lightgbm | baseline | 0.380834 | 0.05 | 1.46 | 11.0 |
| uci-online-shoppers | lightgbm | featcopilot-extended | 0.380834 | 21.43 | 22.22 | 11.0 |
| uci-online-shoppers | lightgbm | featcopilot-extended-paired | 0.380834 | 21.88 | 22.65 | 11.0 |
| uci-online-shoppers | lightgbm | featcopilot-paired | 0.381682 | 15.33 | 16.06 | 19.7 |
| uci-online-shoppers | lightgbm | featcopilot-predictive | 0.380834 | 15.51 | 16.31 | 11.0 |
| uci-online-shoppers | lightgbm | openfe-inductive | 0.384828 | 113.48 | 114.79 | 61.0 |
| uci-online-shoppers | linear | baseline | 0.395244 | 0.06 | 0.33 | 11.0 |
| uci-online-shoppers | linear | featcopilot-extended | 0.385326 | 96.66 | 96.99 | 13.3 |
| uci-online-shoppers | linear | featcopilot-extended-paired | 0.385196 | 96.67 | 97.00 | 14.7 |
| uci-online-shoppers | linear | featcopilot-paired | 0.385196 | 92.41 | 92.72 | 14.7 |
| uci-online-shoppers | linear | featcopilot-predictive | 0.385326 | 91.49 | 91.79 | 13.3 |
| uci-online-shoppers | linear | openfe-inductive | 0.388122 | 112.23 | 113.00 | 61.0 |
| uci-real-estate-sindian | lightgbm | baseline | 7.448276 | 0.02 | 0.90 | 6.0 |
| uci-real-estate-sindian | lightgbm | featcopilot-extended | 7.448276 | 8.91 | 9.18 | 6.0 |
| uci-real-estate-sindian | lightgbm | featcopilot-extended-paired | 7.592642 | 8.27 | 8.54 | 14.3 |
| uci-real-estate-sindian | lightgbm | featcopilot-paired | 7.464846 | 7.24 | 7.48 | 7.7 |
| uci-real-estate-sindian | lightgbm | featcopilot-predictive | 7.448276 | 6.68 | 6.90 | 6.0 |
| uci-real-estate-sindian | lightgbm | openfe-inductive | 7.448276 | 99.04 | 99.29 | 7.0 |
| uci-real-estate-sindian | linear | baseline | 8.793747 | 0.02 | 0.06 | 6.0 |
| uci-real-estate-sindian | linear | featcopilot-extended | 8.378978 | 2.70 | 2.75 | 6.7 |
| uci-real-estate-sindian | linear | featcopilot-extended-paired | 7.976423 | 2.82 | 2.87 | 9.7 |
| uci-real-estate-sindian | linear | featcopilot-paired | 8.055879 | 2.65 | 2.70 | 16.3 |
| uci-real-estate-sindian | linear | featcopilot-predictive | 8.378888 | 2.50 | 2.54 | 6.7 |
| uci-real-estate-sindian | linear | openfe-inductive | 8.792728 | 99.19 | 99.23 | 7.0 |
| uci-spambase | lightgbm | baseline | 0.141042 | 0.15 | 2.21 | 57.0 |
| uci-spambase | lightgbm | featcopilot-extended | 0.141042 | 38.04 | 39.41 | 57.0 |
| uci-spambase | lightgbm | featcopilot-extended-paired | 0.141024 | 43.10 | 44.49 | 57.3 |
| uci-spambase | lightgbm | featcopilot-paired | 0.141557 | 43.68 | 45.28 | 58.7 |
| uci-spambase | lightgbm | featcopilot-predictive | 0.141042 | 37.02 | 38.35 | 57.0 |
| uci-spambase | lightgbm | openfe-inductive | 0.142694 | 173.23 | 174.94 | 107.0 |
| uci-spambase | linear | baseline | 0.239802 | 0.15 | 0.41 | 57.0 |
| uci-spambase | linear | featcopilot-extended | 0.202149 | 97.03 | 97.30 | 67.3 |
| uci-spambase | linear | featcopilot-extended-paired | 0.198103 | 98.56 | 98.85 | 77.3 |
| uci-spambase | linear | featcopilot-paired | 0.179039 | 94.96 | 95.26 | 93.7 |
| uci-spambase | linear | featcopilot-predictive | 0.175298 | 98.70 | 98.98 | 90.3 |
| uci-spambase | linear | openfe-inductive | 0.230953 | 178.89 | 179.23 | 107.0 |
| uci-steel-faults | lightgbm | baseline | 0.557708 | 0.07 | 3.85 | 27.0 |
| uci-steel-faults | lightgbm | featcopilot-extended | 0.556585 | 72.11 | 75.52 | 30.3 |
| uci-steel-faults | lightgbm | featcopilot-extended-paired | 0.548895 | 74.10 | 77.51 | 30.7 |
| uci-steel-faults | lightgbm | featcopilot-paired | 0.545787 | 74.33 | 77.44 | 32.0 |
| uci-steel-faults | lightgbm | featcopilot-predictive | 0.552069 | 72.55 | 75.51 | 33.7 |
| uci-steel-faults | lightgbm | openfe-inductive | 0.552266 | 127.26 | 131.38 | 77.0 |
| uci-steel-faults | linear | baseline | 0.764914 | 0.07 | 0.52 | 27.0 |
| uci-steel-faults | linear | featcopilot-extended | 0.740212 | 93.74 | 94.29 | 30.3 |
| uci-steel-faults | linear | featcopilot-extended-paired | 0.713074 | 94.70 | 95.28 | 33.7 |
| uci-steel-faults | linear | featcopilot-paired | 0.733784 | 100.19 | 100.79 | 32.3 |
| uci-steel-faults | linear | featcopilot-predictive | 0.740742 | 94.52 | 95.08 | 29.0 |
| uci-steel-faults | linear | openfe-inductive | 0.707839 | 128.08 | 128.87 | 77.0 |
| uci-wisconsin-diagnostic | lightgbm | baseline | 0.120432 | 0.07 | 1.07 | 30.0 |
| uci-wisconsin-diagnostic | lightgbm | featcopilot-extended | 0.120432 | 23.40 | 23.75 | 30.0 |
| uci-wisconsin-diagnostic | lightgbm | featcopilot-extended-paired | 0.120296 | 23.55 | 23.91 | 38.3 |
| uci-wisconsin-diagnostic | lightgbm | featcopilot-paired | 0.115976 | 25.78 | 26.14 | 38.3 |
| uci-wisconsin-diagnostic | lightgbm | featcopilot-predictive | 0.120432 | 23.65 | 24.00 | 30.0 |
| uci-wisconsin-diagnostic | lightgbm | openfe-inductive | 0.118061 | 98.73 | 99.06 | 31.0 |
| uci-wisconsin-diagnostic | linear | baseline | 0.089368 | 0.07 | 0.21 | 30.0 |
| uci-wisconsin-diagnostic | linear | featcopilot-extended | 0.089368 | 87.98 | 88.13 | 30.0 |
| uci-wisconsin-diagnostic | linear | featcopilot-extended-paired | 0.089368 | 90.88 | 91.02 | 30.0 |
| uci-wisconsin-diagnostic | linear | featcopilot-paired | 0.090855 | 91.11 | 91.26 | 33.3 |
| uci-wisconsin-diagnostic | linear | featcopilot-predictive | 0.089368 | 93.53 | 93.68 | 30.0 |
| uci-wisconsin-diagnostic | linear | openfe-inductive | 0.090266 | 100.85 | 100.95 | 31.0 |
| wine | lightgbm | baseline | 0.654417 | 0.03 | 1.28 | 11.0 |
| wine | lightgbm | featcopilot-extended | 0.649759 | 23.91 | 24.65 | 11.0 |
| wine | lightgbm | featcopilot-extended-paired | 0.648986 | 26.15 | 27.06 | 12.7 |
| wine | lightgbm | featcopilot-paired | 0.648129 | 21.30 | 22.12 | 27.7 |
| wine | lightgbm | featcopilot-predictive | 0.649759 | 19.81 | 20.49 | 11.0 |
| wine | lightgbm | openfe-inductive | 0.634993 | 105.82 | 107.11 | 61.0 |
| wine | linear | baseline | 0.737508 | 0.03 | 0.09 | 11.0 |
| wine | linear | featcopilot-extended | 0.726671 | 4.87 | 4.95 | 28.0 |
| wine | linear | featcopilot-extended-paired | 0.742393 | 5.17 | 5.28 | 47.7 |
| wine | linear | featcopilot-paired | 0.734912 | 4.78 | 4.88 | 47.7 |
| wine | linear | featcopilot-predictive | 0.726537 | 4.04 | 4.12 | 28.0 |
| wine | linear | openfe-inductive | 0.714151 | 104.24 | 104.36 | 61.0 |

## Failures / timeouts

None.

Raw cell metrics and provenance: local `results.json`. Completed runs also write `archive.json`,
which lists the authoritative lossless gzip parts and their uncompressed checksum.
