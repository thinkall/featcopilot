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
    "featcopilot-frequency-paired",
    "featcopilot-frequency-aligned"
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
| real_world / development | classification | lightgbm | featcopilot-frequency-aligned | 9/9 / 9/9 | 27/27 | 2/5/2 | +8.12% | [-5.51%, +25.53%] |
| real_world / development | classification | lightgbm | featcopilot-frequency-paired | 9/9 / 9/9 | 27/27 | 2/6/1 | +8.26% | [-5.15%, +25.37%] |
| real_world / development | classification | linear | baseline | 9/9 / 9/9 | 27/27 | 0/9/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | classification | linear | featcopilot-frequency-aligned | 9/9 / 9/9 | 27/27 | 6/3/0 | +14.96% | [+1.66%, +35.00%] |
| real_world / development | classification | linear | featcopilot-frequency-paired | 9/9 / 9/9 | 27/27 | 6/2/1 | -0.30% | [-36.51%, +31.14%] |
| real_world / development | regression | lightgbm | baseline | 6/6 / 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | regression | lightgbm | featcopilot-frequency-aligned | 6/6 / 6/6 | 18/18 | 3/1/2 | +1.39% | [-1.76%, +4.72%] |
| real_world / development | regression | lightgbm | featcopilot-frequency-paired | 6/6 / 6/6 | 18/18 | 4/0/2 | +2.49% | [-1.32%, +6.35%] |
| real_world / development | regression | linear | baseline | 6/6 / 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | regression | linear | featcopilot-frequency-aligned | 6/6 / 6/6 | 18/18 | 5/0/1 | +12.70% | [+3.21%, +25.55%] |
| real_world / development | regression | linear | featcopilot-frequency-paired | 6/6 / 6/6 | 18/18 | 6/0/0 | +13.54% | [+4.73%, +26.12%] |
| simulation / development | regression | lightgbm | baseline | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | lightgbm | featcopilot-frequency-aligned | 1/1 / 1/1 | 3/3 | 1/0/0 | +6.32% | n/a |
| simulation / development | regression | lightgbm | featcopilot-frequency-paired | 1/1 / 1/1 | 3/3 | 1/0/0 | +5.93% | n/a |
| simulation / development | regression | linear | baseline | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | linear | featcopilot-frequency-aligned | 1/1 / 1/1 | 3/3 | 1/0/0 | +81.39% | n/a |
| simulation / development | regression | linear | featcopilot-frequency-paired | 1/1 / 1/1 | 3/3 | 1/0/0 | +82.07% | n/a |

## Per-dataset means (complete folds only)

| Dataset | Learner | Method | Primary loss | FE seconds (inner + outer) | Pipeline seconds | Features |
|---|---|---|---|---|---|---|
| houses | lightgbm | baseline | 0.244681 | 0.02 | 1.62 | 8.0 |
| houses | lightgbm | featcopilot-frequency-aligned | 0.225023 | 36.28 | 37.13 | 33.0 |
| houses | lightgbm | featcopilot-frequency-paired | 0.224109 | 17.07 | 17.89 | 33.0 |
| houses | linear | baseline | 0.338541 | 0.02 | 0.08 | 8.0 |
| houses | linear | featcopilot-frequency-aligned | 0.328029 | 10.29 | 10.38 | 26.3 |
| houses | linear | featcopilot-frequency-paired | 0.327903 | 6.13 | 6.22 | 26.3 |
| uci-adult | lightgbm | baseline | 0.309826 | 0.06 | 1.76 | 13.0 |
| uci-adult | lightgbm | featcopilot-frequency-aligned | 0.312557 | 37.85 | 38.71 | 31.3 |
| uci-adult | lightgbm | featcopilot-frequency-paired | 0.309826 | 20.11 | 20.94 | 13.0 |
| uci-adult | linear | baseline | 0.340533 | 0.06 | 0.36 | 13.0 |
| uci-adult | linear | featcopilot-frequency-aligned | 0.331842 | 27.47 | 27.89 | 24.7 |
| uci-adult | linear | featcopilot-frequency-paired | 0.333355 | 15.52 | 15.93 | 16.3 |
| uci-auto-mpg | lightgbm | baseline | 2.924286 | 0.02 | 1.18 | 7.0 |
| uci-auto-mpg | lightgbm | featcopilot-frequency-aligned | 3.009249 | 8.81 | 9.09 | 17.0 |
| uci-auto-mpg | lightgbm | featcopilot-frequency-paired | 3.012853 | 5.89 | 6.16 | 17.0 |
| uci-auto-mpg | linear | baseline | 3.372414 | 0.02 | 0.08 | 7.0 |
| uci-auto-mpg | linear | featcopilot-frequency-aligned | 2.938272 | 4.24 | 4.32 | 20.3 |
| uci-auto-mpg | linear | featcopilot-frequency-paired | 2.872009 | 2.89 | 2.98 | 33.7 |
| uci-banknote-authentication | lightgbm | baseline | 0.015250 | 0.01 | 1.39 | 4.0 |
| uci-banknote-authentication | lightgbm | featcopilot-frequency-aligned | 0.005112 | 11.35 | 11.74 | 37.3 |
| uci-banknote-authentication | lightgbm | featcopilot-frequency-paired | 0.005267 | 6.86 | 7.24 | 45.7 |
| uci-banknote-authentication | linear | baseline | 0.029459 | 0.01 | 0.11 | 4.0 |
| uci-banknote-authentication | linear | featcopilot-frequency-aligned | 0.003110 | 4.94 | 5.05 | 12.3 |
| uci-banknote-authentication | linear | featcopilot-frequency-paired | 0.003435 | 3.25 | 3.35 | 14.0 |
| uci-chronic-kidney-disease | lightgbm | baseline | 0.035504 | 0.05 | 1.29 | 24.0 |
| uci-chronic-kidney-disease | lightgbm | featcopilot-frequency-aligned | 0.024128 | 16.01 | 16.83 | 44.0 |
| uci-chronic-kidney-disease | lightgbm | featcopilot-frequency-paired | 0.024128 | 9.25 | 10.06 | 44.0 |
| uci-chronic-kidney-disease | linear | baseline | 0.016247 | 0.05 | 0.23 | 24.0 |
| uci-chronic-kidney-disease | linear | featcopilot-frequency-aligned | 0.015710 | 11.35 | 12.00 | 27.7 |
| uci-chronic-kidney-disease | linear | featcopilot-frequency-paired | 0.035217 | 6.70 | 7.36 | 26.3 |
| uci-combined-cycle-power | lightgbm | baseline | 3.382905 | 0.01 | 1.63 | 4.0 |
| uci-combined-cycle-power | lightgbm | featcopilot-frequency-aligned | 3.224508 | 26.08 | 26.94 | 24.0 |
| uci-combined-cycle-power | lightgbm | featcopilot-frequency-paired | 3.103413 | 12.21 | 13.03 | 29.0 |
| uci-combined-cycle-power | linear | baseline | 4.474993 | 0.01 | 0.06 | 4.0 |
| uci-combined-cycle-power | linear | featcopilot-frequency-aligned | 4.062855 | 6.82 | 6.93 | 54.0 |
| uci-combined-cycle-power | linear | featcopilot-frequency-paired | 4.060389 | 4.29 | 4.39 | 54.0 |
| uci-concrete-strength | lightgbm | baseline | 4.553620 | 0.02 | 1.37 | 8.0 |
| uci-concrete-strength | lightgbm | featcopilot-frequency-aligned | 4.741218 | 14.92 | 15.32 | 34.7 |
| uci-concrete-strength | lightgbm | featcopilot-frequency-paired | 4.707254 | 9.70 | 10.10 | 9.7 |
| uci-concrete-strength | linear | baseline | 10.559292 | 0.02 | 0.07 | 8.0 |
| uci-concrete-strength | linear | featcopilot-frequency-aligned | 6.104725 | 5.28 | 5.36 | 44.7 |
| uci-concrete-strength | linear | featcopilot-frequency-paired | 5.970942 | 3.66 | 3.73 | 44.7 |
| uci-dry-bean | lightgbm | baseline | 0.248388 | 0.05 | 4.17 | 16.0 |
| uci-dry-bean | lightgbm | featcopilot-frequency-aligned | 0.248178 | 153.16 | 156.56 | 16.3 |
| uci-dry-bean | lightgbm | featcopilot-frequency-paired | 0.247101 | 92.21 | 95.46 | 16.7 |
| uci-dry-bean | linear | baseline | 0.216633 | 0.05 | 0.43 | 16.0 |
| uci-dry-bean | linear | featcopilot-frequency-aligned | 0.215898 | 37.95 | 38.41 | 19.3 |
| uci-dry-bean | linear | featcopilot-frequency-paired | 0.214246 | 20.02 | 20.48 | 19.3 |
| uci-energy-efficiency | lightgbm | baseline | 0.479742 | 0.02 | 1.34 | 8.0 |
| uci-energy-efficiency | lightgbm | featcopilot-frequency-aligned | 0.449412 | 11.53 | 11.93 | 41.3 |
| uci-energy-efficiency | lightgbm | featcopilot-frequency-paired | 0.451304 | 7.76 | 8.16 | 41.3 |
| uci-energy-efficiency | linear | baseline | 2.844552 | 0.02 | 0.08 | 8.0 |
| uci-energy-efficiency | linear | featcopilot-frequency-aligned | 0.529265 | 5.10 | 5.22 | 58.0 |
| uci-energy-efficiency | linear | featcopilot-frequency-paired | 0.510095 | 3.39 | 3.50 | 58.0 |
| uci-german-credit | lightgbm | baseline | 0.519777 | 0.04 | 1.52 | 20.0 |
| uci-german-credit | lightgbm | featcopilot-frequency-aligned | 0.521541 | 22.92 | 23.79 | 20.3 |
| uci-german-credit | lightgbm | featcopilot-frequency-paired | 0.523848 | 14.62 | 15.46 | 20.7 |
| uci-german-credit | linear | baseline | 0.498452 | 0.05 | 0.25 | 20.0 |
| uci-german-credit | linear | featcopilot-frequency-aligned | 0.498452 | 13.72 | 14.24 | 20.0 |
| uci-german-credit | linear | featcopilot-frequency-paired | 0.498452 | 8.01 | 8.51 | 20.0 |
| uci-online-shoppers | lightgbm | baseline | 0.380834 | 0.05 | 1.71 | 11.0 |
| uci-online-shoppers | lightgbm | featcopilot-frequency-aligned | 0.388462 | 34.79 | 35.58 | 13.0 |
| uci-online-shoppers | lightgbm | featcopilot-frequency-paired | 0.382172 | 18.88 | 19.68 | 11.3 |
| uci-online-shoppers | linear | baseline | 0.395244 | 0.05 | 0.31 | 11.0 |
| uci-online-shoppers | linear | featcopilot-frequency-aligned | 0.385988 | 24.23 | 24.57 | 13.3 |
| uci-online-shoppers | linear | featcopilot-frequency-paired | 0.387539 | 13.98 | 14.32 | 13.0 |
| uci-real-estate-sindian | lightgbm | baseline | 7.448276 | 0.02 | 1.18 | 6.0 |
| uci-real-estate-sindian | lightgbm | featcopilot-frequency-aligned | 7.266136 | 8.73 | 8.99 | 8.3 |
| uci-real-estate-sindian | lightgbm | featcopilot-frequency-paired | 7.266136 | 6.19 | 6.45 | 8.3 |
| uci-real-estate-sindian | linear | baseline | 8.793747 | 0.02 | 0.06 | 6.0 |
| uci-real-estate-sindian | linear | featcopilot-frequency-aligned | 7.845634 | 3.77 | 3.83 | 11.0 |
| uci-real-estate-sindian | linear | featcopilot-frequency-paired | 8.066223 | 2.68 | 2.73 | 9.7 |
| uci-spambase | lightgbm | baseline | 0.141042 | 0.16 | 2.44 | 57.0 |
| uci-spambase | lightgbm | featcopilot-frequency-aligned | 0.140386 | 72.66 | 74.06 | 57.3 |
| uci-spambase | lightgbm | featcopilot-frequency-paired | 0.140386 | 37.34 | 38.70 | 57.3 |
| uci-spambase | linear | baseline | 0.239802 | 0.15 | 0.41 | 57.0 |
| uci-spambase | linear | featcopilot-frequency-aligned | 0.169484 | 32.92 | 33.20 | 98.7 |
| uci-spambase | linear | featcopilot-frequency-paired | 0.205999 | 18.38 | 18.64 | 75.3 |
| uci-steel-faults | lightgbm | baseline | 0.557708 | 0.07 | 4.07 | 27.0 |
| uci-steel-faults | lightgbm | featcopilot-frequency-aligned | 0.557708 | 119.31 | 122.50 | 27.0 |
| uci-steel-faults | lightgbm | featcopilot-frequency-paired | 0.559112 | 77.69 | 80.96 | 30.3 |
| uci-steel-faults | linear | baseline | 0.764914 | 0.07 | 0.52 | 27.0 |
| uci-steel-faults | linear | featcopilot-frequency-aligned | 0.708988 | 34.37 | 35.00 | 33.7 |
| uci-steel-faults | linear | featcopilot-frequency-paired | 0.715271 | 19.25 | 19.90 | 34.0 |
| uci-wisconsin-diagnostic | lightgbm | baseline | 0.120432 | 0.07 | 1.45 | 30.0 |
| uci-wisconsin-diagnostic | lightgbm | featcopilot-frequency-aligned | 0.147848 | 16.87 | 17.25 | 33.3 |
| uci-wisconsin-diagnostic | lightgbm | featcopilot-frequency-paired | 0.147848 | 9.46 | 9.84 | 33.3 |
| uci-wisconsin-diagnostic | linear | baseline | 0.089368 | 0.07 | 0.21 | 30.0 |
| uci-wisconsin-diagnostic | linear | featcopilot-frequency-aligned | 0.089368 | 9.46 | 9.63 | 30.0 |
| uci-wisconsin-diagnostic | linear | featcopilot-frequency-paired | 0.089368 | 5.81 | 5.97 | 30.0 |
| wine | lightgbm | baseline | 0.654417 | 0.03 | 1.55 | 11.0 |
| wine | lightgbm | featcopilot-frequency-aligned | 0.653127 | 42.09 | 42.99 | 36.3 |
| wine | lightgbm | featcopilot-frequency-paired | 0.639870 | 21.45 | 22.51 | 52.7 |
| wine | linear | baseline | 0.737508 | 0.03 | 0.09 | 11.0 |
| wine | linear | featcopilot-frequency-aligned | 0.752055 | 13.47 | 13.61 | 61.0 |
| wine | linear | featcopilot-frequency-paired | 0.720650 | 7.54 | 7.67 | 44.3 |

## Failures / timeouts

None.

Raw cell metrics and provenance: local `results.json`. Completed runs also write `archive.json`,
which lists the authoritative lossless gzip parts and their uncompressed checksum.
