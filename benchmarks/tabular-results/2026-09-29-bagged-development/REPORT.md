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
Bagged-aligned-robust averages five training-only candidate-importance rankings before subset validation and during final ranking; it makes no formal stability-selection error-control claim.
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
    "baseline-robust",
    "featcopilot-frequency-aligned-robust",
    "featcopilot-bagged-aligned-robust"
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

Reference method: `baseline-robust`, determined by the declared methods, not observed success or scores.
Loss reduction is 1 - mean(method loss) / mean(baseline loss), first averaged within each dataset.
W/T/L uses a predeclared 1% relative-loss tolerance, not a significance test.
95% intervals bootstrap origin families (not dependent folds), are descriptive and not multiplicity-adjusted.
Rows with different complete-pair coverage must not be ranked against one another.

| Source / role | Task | Learner | Method | Paired datasets / families | Cells OK/all | W/T/L | Mean loss reduction | 95% CI |
|---|---|---|---|---|---|---|---|---|
| real_world / development | classification | lightgbm | baseline-robust | 9/9 / 9/9 | 27/27 | 0/9/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | classification | lightgbm | featcopilot-bagged-aligned-robust | 9/9 / 9/9 | 27/27 | 2/5/2 | +7.55% | [-7.55%, +26.44%] |
| real_world / development | classification | lightgbm | featcopilot-frequency-aligned-robust | 9/9 / 9/9 | 27/27 | 2/5/2 | +8.12% | [-5.51%, +25.53%] |
| real_world / development | classification | linear | baseline-robust | 9/9 / 9/9 | 27/27 | 0/9/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | classification | linear | featcopilot-bagged-aligned-robust | 9/9 / 9/9 | 27/27 | 6/2/1 | +14.84% | [+0.67%, +36.24%] |
| real_world / development | classification | linear | featcopilot-frequency-aligned-robust | 9/9 / 9/9 | 27/27 | 7/1/1 | +16.12% | [+2.17%, +36.78%] |
| real_world / development | regression | lightgbm | baseline-robust | 6/6 / 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | regression | lightgbm | featcopilot-bagged-aligned-robust | 6/6 / 6/6 | 18/18 | 4/1/1 | +2.60% | [-0.29%, +5.66%] |
| real_world / development | regression | lightgbm | featcopilot-frequency-aligned-robust | 6/6 / 6/6 | 18/18 | 3/1/2 | +1.39% | [-1.76%, +4.72%] |
| real_world / development | regression | linear | baseline-robust | 6/6 / 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | regression | linear | featcopilot-bagged-aligned-robust | 6/6 / 6/6 | 18/18 | 6/0/0 | +14.77% | [+6.79%, +26.98%] |
| real_world / development | regression | linear | featcopilot-frequency-aligned-robust | 6/6 / 6/6 | 18/18 | 6/0/0 | +14.97% | [+6.82%, +27.59%] |
| simulation / development | regression | lightgbm | baseline-robust | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | lightgbm | featcopilot-bagged-aligned-robust | 1/1 / 1/1 | 3/3 | 1/0/0 | +9.59% | n/a |
| simulation / development | regression | lightgbm | featcopilot-frequency-aligned-robust | 1/1 / 1/1 | 3/3 | 1/0/0 | +6.32% | n/a |
| simulation / development | regression | linear | baseline-robust | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | linear | featcopilot-bagged-aligned-robust | 1/1 / 1/1 | 3/3 | 1/0/0 | +80.61% | n/a |
| simulation / development | regression | linear | featcopilot-frequency-aligned-robust | 1/1 / 1/1 | 3/3 | 1/0/0 | +81.39% | n/a |

## Per-dataset means (complete folds only)

| Dataset | Learner | Method | Primary loss | FE seconds (inner + outer) | Pipeline seconds | Features |
|---|---|---|---|---|---|---|
| houses | lightgbm | baseline-robust | 0.244681 | 0.02 | 3.15 | 8.0 |
| houses | lightgbm | featcopilot-bagged-aligned-robust | 0.224675 | 120.49 | 122.63 | 33.0 |
| houses | lightgbm | featcopilot-frequency-aligned-robust | 0.225023 | 39.53 | 40.94 | 33.0 |
| houses | linear | baseline-robust | 0.338534 | 0.02 | 1.64 | 8.0 |
| houses | linear | featcopilot-bagged-aligned-robust | 0.308228 | 43.51 | 45.04 | 41.3 |
| houses | linear | featcopilot-frequency-aligned-robust | 0.313249 | 13.03 | 13.74 | 41.3 |
| uci-adult | lightgbm | baseline-robust | 0.309826 | 0.05 | 3.33 | 13.0 |
| uci-adult | lightgbm | featcopilot-bagged-aligned-robust | 0.309883 | 99.57 | 101.12 | 13.3 |
| uci-adult | lightgbm | featcopilot-frequency-aligned-robust | 0.312557 | 39.91 | 41.05 | 31.3 |
| uci-adult | linear | baseline-robust | 0.340534 | 0.05 | 1.93 | 13.0 |
| uci-adult | linear | featcopilot-bagged-aligned-robust | 0.333426 | 93.45 | 94.56 | 23.3 |
| uci-adult | linear | featcopilot-frequency-aligned-robust | 0.331841 | 29.48 | 30.17 | 24.7 |
| uci-auto-mpg | lightgbm | baseline-robust | 2.924286 | 0.01 | 2.66 | 7.0 |
| uci-auto-mpg | lightgbm | featcopilot-bagged-aligned-robust | 2.999535 | 18.49 | 19.15 | 24.0 |
| uci-auto-mpg | lightgbm | featcopilot-frequency-aligned-robust | 3.009249 | 10.19 | 10.60 | 17.0 |
| uci-auto-mpg | linear | baseline-robust | 3.366601 | 0.01 | 1.64 | 7.0 |
| uci-auto-mpg | linear | featcopilot-bagged-aligned-robust | 2.954785 | 12.01 | 12.56 | 28.7 |
| uci-auto-mpg | linear | featcopilot-frequency-aligned-robust | 2.941279 | 5.62 | 5.96 | 18.7 |
| uci-banknote-authentication | lightgbm | baseline-robust | 0.015250 | 0.01 | 2.80 | 4.0 |
| uci-banknote-authentication | lightgbm | featcopilot-bagged-aligned-robust | 0.004797 | 22.43 | 23.15 | 37.3 |
| uci-banknote-authentication | lightgbm | featcopilot-frequency-aligned-robust | 0.005112 | 12.22 | 12.88 | 37.3 |
| uci-banknote-authentication | linear | baseline-robust | 0.029393 | 0.01 | 1.97 | 4.0 |
| uci-banknote-authentication | linear | featcopilot-bagged-aligned-robust | 0.001880 | 12.73 | 13.41 | 37.3 |
| uci-banknote-authentication | linear | featcopilot-frequency-aligned-robust | 0.002743 | 6.41 | 6.83 | 30.7 |
| uci-chronic-kidney-disease | lightgbm | baseline-robust | 0.035504 | 0.04 | 2.82 | 24.0 |
| uci-chronic-kidney-disease | lightgbm | featcopilot-bagged-aligned-robust | 0.023790 | 63.87 | 71.41 | 35.7 |
| uci-chronic-kidney-disease | lightgbm | featcopilot-frequency-aligned-robust | 0.024128 | 20.51 | 22.71 | 44.0 |
| uci-chronic-kidney-disease | linear | baseline-robust | 0.014859 | 0.04 | 1.78 | 24.0 |
| uci-chronic-kidney-disease | linear | featcopilot-bagged-aligned-robust | 0.014162 | 55.69 | 61.22 | 25.0 |
| uci-chronic-kidney-disease | linear | featcopilot-frequency-aligned-robust | 0.012371 | 15.87 | 18.41 | 28.0 |
| uci-combined-cycle-power | lightgbm | baseline-robust | 3.382905 | 0.01 | 3.07 | 4.0 |
| uci-combined-cycle-power | lightgbm | featcopilot-bagged-aligned-robust | 3.150208 | 58.58 | 59.64 | 19.0 |
| uci-combined-cycle-power | lightgbm | featcopilot-frequency-aligned-robust | 3.224508 | 28.58 | 29.73 | 24.0 |
| uci-combined-cycle-power | linear | baseline-robust | 4.474864 | 0.01 | 1.63 | 4.0 |
| uci-combined-cycle-power | linear | featcopilot-bagged-aligned-robust | 4.067353 | 21.37 | 22.27 | 54.0 |
| uci-combined-cycle-power | linear | featcopilot-frequency-aligned-robust | 4.062886 | 8.07 | 8.67 | 54.0 |
| uci-concrete-strength | lightgbm | baseline-robust | 4.553620 | 0.01 | 3.20 | 8.0 |
| uci-concrete-strength | lightgbm | featcopilot-bagged-aligned-robust | 4.546701 | 42.71 | 44.16 | 26.3 |
| uci-concrete-strength | lightgbm | featcopilot-frequency-aligned-robust | 4.741218 | 16.54 | 17.44 | 34.7 |
| uci-concrete-strength | linear | baseline-robust | 10.558518 | 0.01 | 1.65 | 8.0 |
| uci-concrete-strength | linear | featcopilot-bagged-aligned-robust | 6.009590 | 19.96 | 21.87 | 58.0 |
| uci-concrete-strength | linear | featcopilot-frequency-aligned-robust | 5.850409 | 7.22 | 8.25 | 58.0 |
| uci-dry-bean | lightgbm | baseline-robust | 0.248388 | 0.03 | 5.70 | 16.0 |
| uci-dry-bean | lightgbm | featcopilot-bagged-aligned-robust | 0.246206 | 469.69 | 475.75 | 19.3 |
| uci-dry-bean | lightgbm | featcopilot-frequency-aligned-robust | 0.248178 | 160.29 | 164.86 | 16.3 |
| uci-dry-bean | linear | baseline-robust | 0.215886 | 0.03 | 2.03 | 16.0 |
| uci-dry-bean | linear | featcopilot-bagged-aligned-robust | 0.216189 | 137.44 | 141.38 | 22.7 |
| uci-dry-bean | linear | featcopilot-frequency-aligned-robust | 0.213348 | 45.81 | 48.39 | 19.3 |
| uci-energy-efficiency | lightgbm | baseline-robust | 0.479742 | 0.02 | 2.81 | 8.0 |
| uci-energy-efficiency | lightgbm | featcopilot-bagged-aligned-robust | 0.433741 | 26.11 | 27.65 | 37.0 |
| uci-energy-efficiency | lightgbm | featcopilot-frequency-aligned-robust | 0.449412 | 12.73 | 13.49 | 41.3 |
| uci-energy-efficiency | linear | baseline-robust | 2.844552 | 0.02 | 1.93 | 8.0 |
| uci-energy-efficiency | linear | featcopilot-bagged-aligned-robust | 0.551666 | 13.54 | 14.94 | 49.7 |
| uci-energy-efficiency | linear | featcopilot-frequency-aligned-robust | 0.529265 | 6.51 | 7.40 | 58.0 |
| uci-german-credit | lightgbm | baseline-robust | 0.519777 | 0.04 | 3.10 | 20.0 |
| uci-german-credit | lightgbm | featcopilot-bagged-aligned-robust | 0.523848 | 68.02 | 70.99 | 20.7 |
| uci-german-credit | lightgbm | featcopilot-frequency-aligned-robust | 0.521541 | 26.28 | 27.69 | 20.3 |
| uci-german-credit | linear | baseline-robust | 0.498670 | 0.04 | 1.79 | 20.0 |
| uci-german-credit | linear | featcopilot-bagged-aligned-robust | 0.498670 | 48.65 | 50.55 | 20.0 |
| uci-german-credit | linear | featcopilot-frequency-aligned-robust | 0.498670 | 16.26 | 17.25 | 20.0 |
| uci-online-shoppers | lightgbm | baseline-robust | 0.380834 | 0.04 | 3.19 | 11.0 |
| uci-online-shoppers | lightgbm | featcopilot-bagged-aligned-robust | 0.386777 | 90.12 | 91.41 | 11.3 |
| uci-online-shoppers | lightgbm | featcopilot-frequency-aligned-robust | 0.388462 | 37.30 | 38.29 | 13.0 |
| uci-online-shoppers | linear | baseline-robust | 0.395017 | 0.04 | 2.14 | 11.0 |
| uci-online-shoppers | linear | featcopilot-bagged-aligned-robust | 0.386441 | 76.47 | 77.22 | 14.7 |
| uci-online-shoppers | linear | featcopilot-frequency-aligned-robust | 0.386049 | 26.21 | 26.73 | 13.3 |
| uci-real-estate-sindian | lightgbm | baseline-robust | 7.448276 | 0.01 | 2.81 | 6.0 |
| uci-real-estate-sindian | lightgbm | featcopilot-bagged-aligned-robust | 7.322690 | 18.59 | 19.14 | 7.7 |
| uci-real-estate-sindian | lightgbm | featcopilot-frequency-aligned-robust | 7.266136 | 10.20 | 10.74 | 8.3 |
| uci-real-estate-sindian | linear | baseline-robust | 8.764502 | 0.01 | 1.63 | 6.0 |
| uci-real-estate-sindian | linear | featcopilot-bagged-aligned-robust | 7.728018 | 11.84 | 12.18 | 14.3 |
| uci-real-estate-sindian | linear | featcopilot-frequency-aligned-robust | 7.699602 | 5.29 | 5.50 | 14.3 |
| uci-spambase | lightgbm | baseline-robust | 0.141042 | 0.10 | 4.09 | 57.0 |
| uci-spambase | lightgbm | featcopilot-bagged-aligned-robust | 0.140386 | 246.93 | 258.68 | 57.3 |
| uci-spambase | lightgbm | featcopilot-frequency-aligned-robust | 0.140386 | 81.61 | 85.59 | 57.3 |
| uci-spambase | linear | baseline-robust | 0.235432 | 0.10 | 2.11 | 57.0 |
| uci-spambase | linear | featcopilot-bagged-aligned-robust | 0.165815 | 157.75 | 165.43 | 98.7 |
| uci-spambase | linear | featcopilot-frequency-aligned-robust | 0.166747 | 42.54 | 46.46 | 98.7 |
| uci-steel-faults | lightgbm | baseline-robust | 0.557708 | 0.05 | 5.78 | 27.0 |
| uci-steel-faults | lightgbm | featcopilot-bagged-aligned-robust | 0.555843 | 417.78 | 425.26 | 27.3 |
| uci-steel-faults | lightgbm | featcopilot-frequency-aligned-robust | 0.557708 | 127.46 | 132.21 | 27.0 |
| uci-steel-faults | linear | baseline-robust | 0.754344 | 0.05 | 2.09 | 27.0 |
| uci-steel-faults | linear | featcopilot-bagged-aligned-robust | 0.706971 | 147.26 | 151.52 | 30.7 |
| uci-steel-faults | linear | featcopilot-frequency-aligned-robust | 0.703115 | 40.08 | 42.56 | 33.7 |
| uci-wisconsin-diagnostic | lightgbm | baseline-robust | 0.120432 | 0.05 | 3.01 | 30.0 |
| uci-wisconsin-diagnostic | lightgbm | featcopilot-bagged-aligned-robust | 0.160054 | 70.52 | 77.68 | 31.7 |
| uci-wisconsin-diagnostic | lightgbm | featcopilot-frequency-aligned-robust | 0.147848 | 22.58 | 25.69 | 33.3 |
| uci-wisconsin-diagnostic | linear | baseline-robust | 0.087878 | 0.05 | 1.82 | 30.0 |
| uci-wisconsin-diagnostic | linear | featcopilot-bagged-aligned-robust | 0.092011 | 46.85 | 50.54 | 33.7 |
| uci-wisconsin-diagnostic | linear | featcopilot-frequency-aligned-robust | 0.091637 | 14.96 | 16.80 | 33.3 |
| wine | lightgbm | baseline-robust | 0.654417 | 0.02 | 3.03 | 11.0 |
| wine | lightgbm | featcopilot-bagged-aligned-robust | 0.646032 | 155.33 | 158.73 | 36.3 |
| wine | lightgbm | featcopilot-frequency-aligned-robust | 0.653127 | 45.84 | 48.09 | 36.3 |
| wine | linear | baseline-robust | 0.737010 | 0.02 | 1.70 | 11.0 |
| wine | linear | featcopilot-bagged-aligned-robust | 0.711673 | 60.78 | 63.30 | 52.7 |
| wine | linear | featcopilot-frequency-aligned-robust | 0.709145 | 17.02 | 18.63 | 61.0 |

## Failures / timeouts

None.

Raw cell metrics and provenance: local `results.json`. Completed runs also write `archive.json`,
which lists the authoritative lossless gzip parts and their uncompressed checksum.
