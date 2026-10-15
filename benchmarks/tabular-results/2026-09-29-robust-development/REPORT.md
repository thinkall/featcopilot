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
Frequency-aligned separately fits each declared learner/selection setting during inner tuning and refits the winning configuration.
Robust variants cap numeric model inputs to their fitted training minimum/maximum; incremental FE value is reported against baseline-robust separately.
Unselected and selection-only variants isolate the legacy stages; unselected output retains the generator's full planned width.
Other predictive variants fit a separate representation per learner with its predeclared middle search candidate.
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
    "baseline-robust",
    "featcopilot-frequency-aligned",
    "featcopilot-frequency-aligned-robust"
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
| real_world / development | classification | lightgbm | baseline-robust | 9/9 / 9/9 | 27/27 | 0/9/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | classification | lightgbm | featcopilot-frequency-aligned | 9/9 / 9/9 | 27/27 | 2/5/2 | +8.12% | [-5.51%, +25.53%] |
| real_world / development | classification | lightgbm | featcopilot-frequency-aligned-robust | 9/9 / 9/9 | 27/27 | 2/5/2 | +8.12% | [-5.51%, +25.53%] |
| real_world / development | classification | linear | baseline | 9/9 / 9/9 | 27/27 | 0/9/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | classification | linear | baseline-robust | 9/9 / 9/9 | 27/27 | 4/5/0 | +1.55% | [+0.31%, +3.46%] |
| real_world / development | classification | linear | featcopilot-frequency-aligned | 9/9 / 9/9 | 27/27 | 6/3/0 | +14.96% | [+1.66%, +35.00%] |
| real_world / development | classification | linear | featcopilot-frequency-aligned-robust | 9/9 / 9/9 | 27/27 | 7/1/1 | +17.43% | [+3.14%, +37.90%] |
| real_world / development | regression | lightgbm | baseline | 6/6 / 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | regression | lightgbm | baseline-robust | 6/6 / 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | regression | lightgbm | featcopilot-frequency-aligned | 6/6 / 6/6 | 18/18 | 3/1/2 | +1.39% | [-1.76%, +4.72%] |
| real_world / development | regression | lightgbm | featcopilot-frequency-aligned-robust | 6/6 / 6/6 | 18/18 | 3/1/2 | +1.39% | [-1.76%, +4.72%] |
| real_world / development | regression | linear | baseline | 6/6 / 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | regression | linear | baseline-robust | 6/6 / 6/6 | 18/18 | 0/6/0 | +0.10% | [+0.01%, +0.20%] |
| real_world / development | regression | linear | featcopilot-frequency-aligned | 6/6 / 6/6 | 18/18 | 5/0/1 | +12.70% | [+3.21%, +25.55%] |
| real_world / development | regression | linear | featcopilot-frequency-aligned-robust | 6/6 / 6/6 | 18/18 | 6/0/0 | +15.06% | [+6.84%, +27.69%] |
| simulation / development | regression | lightgbm | baseline | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | lightgbm | baseline-robust | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | lightgbm | featcopilot-frequency-aligned | 1/1 / 1/1 | 3/3 | 1/0/0 | +6.32% | n/a |
| simulation / development | regression | lightgbm | featcopilot-frequency-aligned-robust | 1/1 / 1/1 | 3/3 | 1/0/0 | +6.32% | n/a |
| simulation / development | regression | linear | baseline | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | linear | baseline-robust | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | linear | featcopilot-frequency-aligned | 1/1 / 1/1 | 3/3 | 1/0/0 | +81.39% | n/a |
| simulation / development | regression | linear | featcopilot-frequency-aligned-robust | 1/1 / 1/1 | 3/3 | 1/0/0 | +81.39% | n/a |

## Incremental feature value against the matched robust raw learner

These reductions use baseline-robust on the same complete splits, not the unclipped raw baseline above.
Clipping gains alone are not counted as feature-engineering gains. Missing robust controls remain unpaired.

| Source / role | Task | Learner | Method | Paired datasets / families | Cells OK/all | W/T/L | Mean loss reduction | 95% CI |
|---|---|---|---|---|---|---|---|---|
| real_world / development | classification | lightgbm | baseline-robust | 9/9 / 9/9 | 27/27 | 0/9/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | classification | lightgbm | featcopilot-frequency-aligned-robust | 9/9 / 9/9 | 27/27 | 2/5/2 | +8.12% | [-5.51%, +25.53%] |
| real_world / development | classification | linear | baseline-robust | 9/9 / 9/9 | 27/27 | 0/9/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | classification | linear | featcopilot-frequency-aligned-robust | 9/9 / 9/9 | 27/27 | 7/1/1 | +16.12% | [+2.17%, +36.78%] |
| real_world / development | regression | lightgbm | baseline-robust | 6/6 / 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | regression | lightgbm | featcopilot-frequency-aligned-robust | 6/6 / 6/6 | 18/18 | 3/1/2 | +1.39% | [-1.76%, +4.72%] |
| real_world / development | regression | linear | baseline-robust | 6/6 / 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | regression | linear | featcopilot-frequency-aligned-robust | 6/6 / 6/6 | 18/18 | 6/0/0 | +14.97% | [+6.82%, +27.59%] |
| simulation / development | regression | lightgbm | baseline-robust | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | lightgbm | featcopilot-frequency-aligned-robust | 1/1 / 1/1 | 3/3 | 1/0/0 | +6.32% | n/a |
| simulation / development | regression | linear | baseline-robust | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | linear | featcopilot-frequency-aligned-robust | 1/1 / 1/1 | 3/3 | 1/0/0 | +81.39% | n/a |

## Per-dataset means (complete folds only)

| Dataset | Learner | Method | Primary loss | FE seconds (inner + outer) | Pipeline seconds | Features |
|---|---|---|---|---|---|---|
| houses | lightgbm | baseline | 0.244681 | 0.02 | 3.10 | 8.0 |
| houses | lightgbm | baseline-robust | 0.244681 | 0.02 | 3.08 | 8.0 |
| houses | lightgbm | featcopilot-frequency-aligned | 0.225023 | 36.31 | 37.18 | 33.0 |
| houses | lightgbm | featcopilot-frequency-aligned-robust | 0.225023 | 39.35 | 40.75 | 33.0 |
| houses | linear | baseline | 0.338541 | 0.02 | 1.59 | 8.0 |
| houses | linear | baseline-robust | 0.338534 | 0.02 | 1.60 | 8.0 |
| houses | linear | featcopilot-frequency-aligned | 0.328029 | 10.34 | 10.43 | 26.3 |
| houses | linear | featcopilot-frequency-aligned-robust | 0.313249 | 12.51 | 13.21 | 41.3 |
| uci-adult | lightgbm | baseline | 0.309826 | 0.06 | 3.20 | 13.0 |
| uci-adult | lightgbm | baseline-robust | 0.309826 | 0.05 | 3.23 | 13.0 |
| uci-adult | lightgbm | featcopilot-frequency-aligned | 0.312557 | 38.87 | 39.80 | 31.3 |
| uci-adult | lightgbm | featcopilot-frequency-aligned-robust | 0.312557 | 39.62 | 40.78 | 31.3 |
| uci-adult | linear | baseline | 0.340533 | 0.06 | 1.88 | 13.0 |
| uci-adult | linear | baseline-robust | 0.340534 | 0.05 | 1.88 | 13.0 |
| uci-adult | linear | featcopilot-frequency-aligned | 0.331842 | 27.44 | 27.86 | 24.7 |
| uci-adult | linear | featcopilot-frequency-aligned-robust | 0.331841 | 29.28 | 29.96 | 24.7 |
| uci-auto-mpg | lightgbm | baseline | 2.924286 | 0.02 | 2.66 | 7.0 |
| uci-auto-mpg | lightgbm | baseline-robust | 2.924286 | 0.01 | 2.64 | 7.0 |
| uci-auto-mpg | lightgbm | featcopilot-frequency-aligned | 3.009249 | 8.95 | 9.23 | 17.0 |
| uci-auto-mpg | lightgbm | featcopilot-frequency-aligned-robust | 3.009249 | 9.95 | 10.34 | 17.0 |
| uci-auto-mpg | linear | baseline | 3.372414 | 0.02 | 1.59 | 7.0 |
| uci-auto-mpg | linear | baseline-robust | 3.366601 | 0.01 | 1.88 | 7.0 |
| uci-auto-mpg | linear | featcopilot-frequency-aligned | 2.938272 | 4.30 | 4.39 | 20.3 |
| uci-auto-mpg | linear | featcopilot-frequency-aligned-robust | 2.941279 | 5.56 | 5.91 | 18.7 |
| uci-banknote-authentication | lightgbm | baseline | 0.015250 | 0.01 | 2.74 | 4.0 |
| uci-banknote-authentication | lightgbm | baseline-robust | 0.015250 | 0.01 | 2.73 | 4.0 |
| uci-banknote-authentication | lightgbm | featcopilot-frequency-aligned | 0.005112 | 11.35 | 11.75 | 37.3 |
| uci-banknote-authentication | lightgbm | featcopilot-frequency-aligned-robust | 0.005112 | 11.99 | 12.62 | 37.3 |
| uci-banknote-authentication | linear | baseline | 0.029459 | 0.01 | 1.63 | 4.0 |
| uci-banknote-authentication | linear | baseline-robust | 0.029393 | 0.01 | 1.61 | 4.0 |
| uci-banknote-authentication | linear | featcopilot-frequency-aligned | 0.003110 | 5.10 | 5.20 | 12.3 |
| uci-banknote-authentication | linear | featcopilot-frequency-aligned-robust | 0.002743 | 5.98 | 6.40 | 30.7 |
| uci-chronic-kidney-disease | lightgbm | baseline | 0.035504 | 0.05 | 2.74 | 24.0 |
| uci-chronic-kidney-disease | lightgbm | baseline-robust | 0.035504 | 0.04 | 2.78 | 24.0 |
| uci-chronic-kidney-disease | lightgbm | featcopilot-frequency-aligned | 0.024128 | 16.48 | 17.30 | 44.0 |
| uci-chronic-kidney-disease | lightgbm | featcopilot-frequency-aligned-robust | 0.024128 | 20.05 | 22.22 | 44.0 |
| uci-chronic-kidney-disease | linear | baseline | 0.016247 | 0.05 | 1.71 | 24.0 |
| uci-chronic-kidney-disease | linear | baseline-robust | 0.014859 | 0.04 | 1.75 | 24.0 |
| uci-chronic-kidney-disease | linear | featcopilot-frequency-aligned | 0.015710 | 11.39 | 12.04 | 27.7 |
| uci-chronic-kidney-disease | linear | featcopilot-frequency-aligned-robust | 0.012371 | 15.75 | 18.29 | 28.0 |
| uci-combined-cycle-power | lightgbm | baseline | 3.382905 | 0.01 | 3.11 | 4.0 |
| uci-combined-cycle-power | lightgbm | baseline-robust | 3.382905 | 0.01 | 3.06 | 4.0 |
| uci-combined-cycle-power | lightgbm | featcopilot-frequency-aligned | 3.224508 | 25.82 | 26.68 | 24.0 |
| uci-combined-cycle-power | lightgbm | featcopilot-frequency-aligned-robust | 3.224508 | 26.83 | 27.90 | 24.0 |
| uci-combined-cycle-power | linear | baseline | 4.474993 | 0.01 | 1.56 | 4.0 |
| uci-combined-cycle-power | linear | baseline-robust | 4.474864 | 0.01 | 1.63 | 4.0 |
| uci-combined-cycle-power | linear | featcopilot-frequency-aligned | 4.062855 | 6.87 | 6.97 | 54.0 |
| uci-combined-cycle-power | linear | featcopilot-frequency-aligned-robust | 4.062886 | 7.87 | 8.46 | 54.0 |
| uci-concrete-strength | lightgbm | baseline | 4.553620 | 0.02 | 2.79 | 8.0 |
| uci-concrete-strength | lightgbm | baseline-robust | 4.553620 | 0.01 | 2.79 | 8.0 |
| uci-concrete-strength | lightgbm | featcopilot-frequency-aligned | 4.741218 | 14.75 | 15.16 | 34.7 |
| uci-concrete-strength | lightgbm | featcopilot-frequency-aligned-robust | 4.741218 | 16.79 | 17.70 | 34.7 |
| uci-concrete-strength | linear | baseline | 10.559292 | 0.02 | 1.59 | 8.0 |
| uci-concrete-strength | linear | baseline-robust | 10.558518 | 0.01 | 1.57 | 8.0 |
| uci-concrete-strength | linear | featcopilot-frequency-aligned | 6.104725 | 5.30 | 5.37 | 44.7 |
| uci-concrete-strength | linear | featcopilot-frequency-aligned-robust | 5.850409 | 7.14 | 8.16 | 58.0 |
| uci-dry-bean | lightgbm | baseline | 0.248388 | 0.05 | 6.10 | 16.0 |
| uci-dry-bean | lightgbm | baseline-robust | 0.248388 | 0.03 | 5.70 | 16.0 |
| uci-dry-bean | lightgbm | featcopilot-frequency-aligned | 0.248178 | 151.62 | 154.96 | 16.3 |
| uci-dry-bean | lightgbm | featcopilot-frequency-aligned-robust | 0.248178 | 157.00 | 161.48 | 16.3 |
| uci-dry-bean | linear | baseline | 0.216633 | 0.04 | 1.96 | 16.0 |
| uci-dry-bean | linear | baseline-robust | 0.215886 | 0.03 | 2.00 | 16.0 |
| uci-dry-bean | linear | featcopilot-frequency-aligned | 0.215898 | 37.99 | 38.45 | 19.3 |
| uci-dry-bean | linear | featcopilot-frequency-aligned-robust | 0.213348 | 45.16 | 47.72 | 19.3 |
| uci-energy-efficiency | lightgbm | baseline | 0.479742 | 0.02 | 2.71 | 8.0 |
| uci-energy-efficiency | lightgbm | baseline-robust | 0.479742 | 0.02 | 3.07 | 8.0 |
| uci-energy-efficiency | lightgbm | featcopilot-frequency-aligned | 0.449412 | 11.37 | 11.76 | 41.3 |
| uci-energy-efficiency | lightgbm | featcopilot-frequency-aligned-robust | 0.449412 | 12.91 | 13.69 | 41.3 |
| uci-energy-efficiency | linear | baseline | 2.844552 | 0.02 | 1.85 | 8.0 |
| uci-energy-efficiency | linear | baseline-robust | 2.844552 | 0.02 | 1.58 | 8.0 |
| uci-energy-efficiency | linear | featcopilot-frequency-aligned | 0.529265 | 5.56 | 5.67 | 58.0 |
| uci-energy-efficiency | linear | featcopilot-frequency-aligned-robust | 0.529265 | 6.44 | 7.33 | 58.0 |
| uci-german-credit | lightgbm | baseline | 0.519777 | 0.05 | 3.00 | 20.0 |
| uci-german-credit | lightgbm | baseline-robust | 0.519777 | 0.04 | 3.02 | 20.0 |
| uci-german-credit | lightgbm | featcopilot-frequency-aligned | 0.521541 | 23.46 | 24.36 | 20.3 |
| uci-german-credit | lightgbm | featcopilot-frequency-aligned-robust | 0.521541 | 26.18 | 27.60 | 20.3 |
| uci-german-credit | linear | baseline | 0.498452 | 0.04 | 1.79 | 20.0 |
| uci-german-credit | linear | baseline-robust | 0.498670 | 0.04 | 1.78 | 20.0 |
| uci-german-credit | linear | featcopilot-frequency-aligned | 0.498452 | 13.67 | 14.18 | 20.0 |
| uci-german-credit | linear | featcopilot-frequency-aligned-robust | 0.498670 | 16.52 | 17.52 | 20.0 |
| uci-online-shoppers | lightgbm | baseline | 0.380834 | 0.05 | 3.11 | 11.0 |
| uci-online-shoppers | lightgbm | baseline-robust | 0.380834 | 0.04 | 3.24 | 11.0 |
| uci-online-shoppers | lightgbm | featcopilot-frequency-aligned | 0.388462 | 35.73 | 36.57 | 13.0 |
| uci-online-shoppers | lightgbm | featcopilot-frequency-aligned-robust | 0.388462 | 36.02 | 36.95 | 13.0 |
| uci-online-shoppers | linear | baseline | 0.395244 | 0.05 | 1.84 | 11.0 |
| uci-online-shoppers | linear | baseline-robust | 0.395017 | 0.04 | 1.82 | 11.0 |
| uci-online-shoppers | linear | featcopilot-frequency-aligned | 0.385988 | 23.93 | 24.27 | 13.3 |
| uci-online-shoppers | linear | featcopilot-frequency-aligned-robust | 0.386049 | 25.81 | 26.32 | 13.3 |
| uci-real-estate-sindian | lightgbm | baseline | 7.448276 | 0.02 | 2.61 | 6.0 |
| uci-real-estate-sindian | lightgbm | baseline-robust | 7.448276 | 0.01 | 2.98 | 6.0 |
| uci-real-estate-sindian | lightgbm | featcopilot-frequency-aligned | 7.266136 | 9.14 | 9.42 | 8.3 |
| uci-real-estate-sindian | lightgbm | featcopilot-frequency-aligned-robust | 7.266136 | 9.98 | 10.50 | 8.3 |
| uci-real-estate-sindian | linear | baseline | 8.793747 | 0.02 | 1.57 | 6.0 |
| uci-real-estate-sindian | linear | baseline-robust | 8.764502 | 0.01 | 1.58 | 6.0 |
| uci-real-estate-sindian | linear | featcopilot-frequency-aligned | 7.845634 | 3.84 | 3.89 | 11.0 |
| uci-real-estate-sindian | linear | featcopilot-frequency-aligned-robust | 7.699602 | 5.23 | 5.44 | 14.3 |
| uci-spambase | lightgbm | baseline | 0.141042 | 0.15 | 3.98 | 57.0 |
| uci-spambase | lightgbm | baseline-robust | 0.141042 | 0.09 | 3.97 | 57.0 |
| uci-spambase | lightgbm | featcopilot-frequency-aligned | 0.140386 | 71.52 | 72.85 | 57.3 |
| uci-spambase | lightgbm | featcopilot-frequency-aligned-robust | 0.140386 | 80.41 | 84.39 | 57.3 |
| uci-spambase | linear | baseline | 0.239802 | 0.15 | 1.89 | 57.0 |
| uci-spambase | linear | baseline-robust | 0.235432 | 0.09 | 2.03 | 57.0 |
| uci-spambase | linear | featcopilot-frequency-aligned | 0.169484 | 32.97 | 33.25 | 98.7 |
| uci-spambase | linear | featcopilot-frequency-aligned-robust | 0.166747 | 42.21 | 46.09 | 98.7 |
| uci-steel-faults | lightgbm | baseline | 0.557708 | 0.07 | 5.52 | 27.0 |
| uci-steel-faults | lightgbm | baseline-robust | 0.557708 | 0.05 | 5.47 | 27.0 |
| uci-steel-faults | lightgbm | featcopilot-frequency-aligned | 0.557708 | 121.20 | 124.43 | 27.0 |
| uci-steel-faults | lightgbm | featcopilot-frequency-aligned-robust | 0.557708 | 127.26 | 131.96 | 27.0 |
| uci-steel-faults | linear | baseline | 0.764914 | 0.07 | 2.03 | 27.0 |
| uci-steel-faults | linear | baseline-robust | 0.754344 | 0.05 | 2.06 | 27.0 |
| uci-steel-faults | linear | featcopilot-frequency-aligned | 0.708988 | 34.26 | 34.88 | 33.7 |
| uci-steel-faults | linear | featcopilot-frequency-aligned-robust | 0.703115 | 40.29 | 42.76 | 33.7 |
| uci-wisconsin-diagnostic | lightgbm | baseline | 0.120432 | 0.07 | 2.80 | 30.0 |
| uci-wisconsin-diagnostic | lightgbm | baseline-robust | 0.120432 | 0.04 | 2.83 | 30.0 |
| uci-wisconsin-diagnostic | lightgbm | featcopilot-frequency-aligned | 0.147848 | 16.76 | 17.14 | 33.3 |
| uci-wisconsin-diagnostic | lightgbm | featcopilot-frequency-aligned-robust | 0.147848 | 21.86 | 24.88 | 33.3 |
| uci-wisconsin-diagnostic | linear | baseline | 0.089368 | 0.07 | 1.71 | 30.0 |
| uci-wisconsin-diagnostic | linear | baseline-robust | 0.087878 | 0.05 | 1.86 | 30.0 |
| uci-wisconsin-diagnostic | linear | featcopilot-frequency-aligned | 0.089368 | 9.55 | 9.72 | 30.0 |
| uci-wisconsin-diagnostic | linear | featcopilot-frequency-aligned-robust | 0.091637 | 14.71 | 16.53 | 33.3 |
| wine | lightgbm | baseline | 0.654417 | 0.03 | 2.98 | 11.0 |
| wine | lightgbm | baseline-robust | 0.654417 | 0.02 | 3.01 | 11.0 |
| wine | lightgbm | featcopilot-frequency-aligned | 0.653127 | 42.89 | 43.82 | 36.3 |
| wine | lightgbm | featcopilot-frequency-aligned-robust | 0.653127 | 44.82 | 47.01 | 36.3 |
| wine | linear | baseline | 0.737508 | 0.03 | 1.60 | 11.0 |
| wine | linear | baseline-robust | 0.737010 | 0.02 | 1.64 | 11.0 |
| wine | linear | featcopilot-frequency-aligned | 0.752055 | 13.45 | 13.59 | 61.0 |
| wine | linear | featcopilot-frequency-aligned-robust | 0.709145 | 16.73 | 18.33 | 61.0 |

## Failures / timeouts

None.

Raw cell metrics and provenance: local `results.json`. Completed runs also write `archive.json`,
which lists the authoritative lossless gzip parts and their uncompressed checksum.
