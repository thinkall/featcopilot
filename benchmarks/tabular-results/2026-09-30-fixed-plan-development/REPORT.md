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
Plan-gated-robust reserves a discovery-independent gate, tests one frozen proposal, and refits only unsupervised mapping state after its decision.
Its matched baseline-discovery-robust uses the same discovery-only raw-model tuning; it is not interchangeable with the ordinary robust baseline.
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
    "baseline-discovery-robust",
    "featcopilot-plan-gated-robust"
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
| real_world / development | classification | lightgbm | baseline-discovery-robust | 9/9 / 9/9 | 27/27 | 2/6/1 | -0.45% | [-1.93%, +0.55%] |
| real_world / development | classification | lightgbm | baseline-robust | 9/9 / 9/9 | 27/27 | 0/9/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | classification | lightgbm | featcopilot-frequency-aligned-robust | 9/9 / 9/9 | 27/27 | 2/5/2 | +8.12% | [-5.51%, +25.53%] |
| real_world / development | classification | lightgbm | featcopilot-plan-gated-robust | 9/9 / 9/9 | 27/27 | 2/5/2 | +3.20% | [-11.99%, +19.34%] |
| real_world / development | classification | linear | baseline-discovery-robust | 9/9 / 9/9 | 27/27 | 3/5/1 | +0.59% | [-0.57%, +2.06%] |
| real_world / development | classification | linear | baseline-robust | 9/9 / 9/9 | 27/27 | 0/9/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | classification | linear | featcopilot-frequency-aligned-robust | 9/9 / 9/9 | 27/27 | 7/1/1 | +16.12% | [+2.17%, +36.78%] |
| real_world / development | classification | linear | featcopilot-plan-gated-robust | 9/9 / 9/9 | 27/27 | 5/2/2 | +11.38% | [-7.23%, +35.45%] |
| real_world / development | regression | lightgbm | baseline-discovery-robust | 6/6 / 6/6 | 18/18 | 0/5/1 | -0.94% | [-2.85%, +0.23%] |
| real_world / development | regression | lightgbm | baseline-robust | 6/6 / 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | regression | lightgbm | featcopilot-frequency-aligned-robust | 6/6 / 6/6 | 18/18 | 3/1/2 | +1.39% | [-1.76%, +4.72%] |
| real_world / development | regression | lightgbm | featcopilot-plan-gated-robust | 6/6 / 6/6 | 18/18 | 2/3/1 | +1.27% | [-1.38%, +4.35%] |
| real_world / development | regression | linear | baseline-discovery-robust | 6/6 / 6/6 | 18/18 | 0/6/0 | -0.10% | [-0.25%, +0.00%] |
| real_world / development | regression | linear | baseline-robust | 6/6 / 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | regression | linear | featcopilot-frequency-aligned-robust | 6/6 / 6/6 | 18/18 | 6/0/0 | +14.97% | [+6.82%, +27.59%] |
| real_world / development | regression | linear | featcopilot-plan-gated-robust | 6/6 / 6/6 | 18/18 | 6/0/0 | +13.09% | [+5.02%, +25.75%] |
| simulation / development | regression | lightgbm | baseline-discovery-robust | 1/1 / 1/1 | 3/3 | 1/0/0 | +1.47% | n/a |
| simulation / development | regression | lightgbm | baseline-robust | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | lightgbm | featcopilot-frequency-aligned-robust | 1/1 / 1/1 | 3/3 | 1/0/0 | +6.32% | n/a |
| simulation / development | regression | lightgbm | featcopilot-plan-gated-robust | 1/1 / 1/1 | 3/3 | 0/0/1 | -5.54% | n/a |
| simulation / development | regression | linear | baseline-discovery-robust | 1/1 / 1/1 | 3/3 | 0/1/0 | -0.03% | n/a |
| simulation / development | regression | linear | baseline-robust | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | linear | featcopilot-frequency-aligned-robust | 1/1 / 1/1 | 3/3 | 1/0/0 | +81.39% | n/a |
| simulation / development | regression | linear | featcopilot-plan-gated-robust | 1/1 / 1/1 | 3/3 | 1/0/0 | +81.22% | n/a |

## Fixed-plan method against discovery-matched raw tuning

These reductions use baseline-discovery-robust with identical discovery-only raw hyperparameter tuning.
Both methods reserve the same gate rows from discovery; only the feature method uses them to accept or reject one plan.

| Source / role | Task | Learner | Method | Paired datasets / families | Cells OK/all | W/T/L | Mean loss reduction | 95% CI |
|---|---|---|---|---|---|---|---|---|
| real_world / development | classification | lightgbm | baseline-discovery-robust | 9/9 / 9/9 | 27/27 | 0/9/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | classification | lightgbm | featcopilot-plan-gated-robust | 9/9 / 9/9 | 27/27 | 2/6/1 | +3.64% | [-11.96%, +19.29%] |
| real_world / development | classification | linear | baseline-discovery-robust | 9/9 / 9/9 | 27/27 | 0/9/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | classification | linear | featcopilot-plan-gated-robust | 9/9 / 9/9 | 27/27 | 5/3/1 | +10.57% | [-9.31%, +35.49%] |
| real_world / development | regression | lightgbm | baseline-discovery-robust | 6/6 / 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | regression | lightgbm | featcopilot-plan-gated-robust | 6/6 / 6/6 | 18/18 | 3/2/1 | +2.18% | [-0.15%, +4.81%] |
| real_world / development | regression | linear | baseline-discovery-robust | 6/6 / 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | regression | linear | featcopilot-plan-gated-robust | 6/6 / 6/6 | 18/18 | 6/0/0 | +13.18% | [+5.04%, +25.85%] |
| simulation / development | regression | lightgbm | baseline-discovery-robust | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | lightgbm | featcopilot-plan-gated-robust | 1/1 / 1/1 | 3/3 | 0/0/1 | -7.12% | n/a |
| simulation / development | regression | linear | baseline-discovery-robust | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| simulation / development | regression | linear | featcopilot-plan-gated-robust | 1/1 / 1/1 | 3/3 | 1/0/0 | +81.23% | n/a |

## Per-dataset means (complete folds only)

| Dataset | Learner | Method | Primary loss | FE seconds (inner + outer) | Pipeline seconds | Features |
|---|---|---|---|---|---|---|
| houses | lightgbm | baseline-discovery-robust | 0.245126 | 0.02 | 1.53 | 8.0 |
| houses | lightgbm | baseline-robust | 0.244681 | 0.02 | 3.17 | 8.0 |
| houses | lightgbm | featcopilot-frequency-aligned-robust | 0.225023 | 39.32 | 40.72 | 33.0 |
| houses | lightgbm | featcopilot-plan-gated-robust | 0.226589 | 32.29 | 34.83 | 33.0 |
| houses | linear | baseline-discovery-robust | 0.338528 | 0.02 | 0.09 | 8.0 |
| houses | linear | baseline-robust | 0.338534 | 0.02 | 1.68 | 8.0 |
| houses | linear | featcopilot-frequency-aligned-robust | 0.313249 | 12.97 | 13.68 | 41.3 |
| houses | linear | featcopilot-plan-gated-robust | 0.302418 | 9.48 | 9.74 | 49.7 |
| uci-adult | lightgbm | baseline-discovery-robust | 0.309826 | 0.07 | 1.78 | 13.0 |
| uci-adult | lightgbm | baseline-robust | 0.309826 | 0.05 | 3.46 | 13.0 |
| uci-adult | lightgbm | featcopilot-frequency-aligned-robust | 0.312557 | 39.97 | 41.12 | 31.3 |
| uci-adult | lightgbm | featcopilot-plan-gated-robust | 0.309826 | 30.52 | 32.95 | 13.0 |
| uci-adult | linear | baseline-discovery-robust | 0.340534 | 0.06 | 0.35 | 13.0 |
| uci-adult | linear | baseline-robust | 0.340534 | 0.05 | 1.98 | 13.0 |
| uci-adult | linear | featcopilot-frequency-aligned-robust | 0.331841 | 30.05 | 30.74 | 24.7 |
| uci-adult | linear | featcopilot-plan-gated-robust | 0.336879 | 21.94 | 22.57 | 14.7 |
| uci-auto-mpg | lightgbm | baseline-discovery-robust | 2.903341 | 0.02 | 1.14 | 7.0 |
| uci-auto-mpg | lightgbm | baseline-robust | 2.924286 | 0.01 | 2.72 | 7.0 |
| uci-auto-mpg | lightgbm | featcopilot-frequency-aligned-robust | 3.009249 | 10.57 | 10.99 | 17.0 |
| uci-auto-mpg | lightgbm | featcopilot-plan-gated-robust | 2.953397 | 6.95 | 8.33 | 23.7 |
| uci-auto-mpg | linear | baseline-discovery-robust | 3.371927 | 0.02 | 0.08 | 7.0 |
| uci-auto-mpg | linear | baseline-robust | 3.366601 | 0.01 | 1.69 | 7.0 |
| uci-auto-mpg | linear | featcopilot-frequency-aligned-robust | 2.941279 | 5.68 | 6.03 | 18.7 |
| uci-auto-mpg | linear | featcopilot-plan-gated-robust | 3.202689 | 3.99 | 4.14 | 9.0 |
| uci-banknote-authentication | lightgbm | baseline-discovery-robust | 0.015067 | 0.02 | 1.20 | 4.0 |
| uci-banknote-authentication | lightgbm | baseline-robust | 0.015250 | 0.01 | 2.83 | 4.0 |
| uci-banknote-authentication | lightgbm | featcopilot-frequency-aligned-robust | 0.005112 | 12.44 | 13.10 | 37.3 |
| uci-banknote-authentication | lightgbm | featcopilot-plan-gated-robust | 0.007972 | 8.99 | 10.89 | 30.7 |
| uci-banknote-authentication | linear | baseline-discovery-robust | 0.029393 | 0.02 | 0.12 | 4.0 |
| uci-banknote-authentication | linear | baseline-robust | 0.029393 | 0.01 | 1.70 | 4.0 |
| uci-banknote-authentication | linear | featcopilot-frequency-aligned-robust | 0.002743 | 6.17 | 6.59 | 30.7 |
| uci-banknote-authentication | linear | featcopilot-plan-gated-robust | 0.001783 | 4.10 | 4.45 | 29.0 |
| uci-chronic-kidney-disease | lightgbm | baseline-discovery-robust | 0.035504 | 0.05 | 1.24 | 24.0 |
| uci-chronic-kidney-disease | lightgbm | baseline-robust | 0.035504 | 0.04 | 2.91 | 24.0 |
| uci-chronic-kidney-disease | lightgbm | featcopilot-frequency-aligned-robust | 0.024128 | 20.46 | 22.65 | 44.0 |
| uci-chronic-kidney-disease | lightgbm | featcopilot-plan-gated-robust | 0.024025 | 15.00 | 16.64 | 27.3 |
| uci-chronic-kidney-disease | linear | baseline-discovery-robust | 0.014859 | 0.05 | 0.24 | 24.0 |
| uci-chronic-kidney-disease | linear | baseline-robust | 0.014859 | 0.04 | 1.87 | 24.0 |
| uci-chronic-kidney-disease | linear | featcopilot-frequency-aligned-robust | 0.012371 | 15.90 | 18.44 | 28.0 |
| uci-chronic-kidney-disease | linear | featcopilot-plan-gated-robust | 0.012905 | 13.24 | 13.71 | 29.0 |
| uci-combined-cycle-power | lightgbm | baseline-discovery-robust | 3.382905 | 0.02 | 1.48 | 4.0 |
| uci-combined-cycle-power | lightgbm | baseline-robust | 3.382905 | 0.01 | 3.08 | 4.0 |
| uci-combined-cycle-power | lightgbm | featcopilot-frequency-aligned-robust | 3.224508 | 26.55 | 27.62 | 24.0 |
| uci-combined-cycle-power | lightgbm | featcopilot-plan-gated-robust | 3.224531 | 22.64 | 25.24 | 19.0 |
| uci-combined-cycle-power | linear | baseline-discovery-robust | 4.474952 | 0.01 | 0.07 | 4.0 |
| uci-combined-cycle-power | linear | baseline-robust | 4.474864 | 0.01 | 1.66 | 4.0 |
| uci-combined-cycle-power | linear | featcopilot-frequency-aligned-robust | 4.062886 | 8.06 | 8.67 | 54.0 |
| uci-combined-cycle-power | linear | featcopilot-plan-gated-robust | 4.076208 | 5.62 | 5.90 | 54.0 |
| uci-concrete-strength | lightgbm | baseline-discovery-robust | 4.806046 | 0.02 | 1.26 | 8.0 |
| uci-concrete-strength | lightgbm | baseline-robust | 4.553620 | 0.01 | 2.99 | 8.0 |
| uci-concrete-strength | lightgbm | featcopilot-frequency-aligned-robust | 4.741218 | 17.44 | 18.38 | 34.7 |
| uci-concrete-strength | lightgbm | featcopilot-plan-gated-robust | 4.701587 | 13.08 | 14.82 | 9.7 |
| uci-concrete-strength | linear | baseline-discovery-robust | 10.558518 | 0.02 | 0.08 | 8.0 |
| uci-concrete-strength | linear | baseline-robust | 10.558518 | 0.01 | 1.67 | 8.0 |
| uci-concrete-strength | linear | featcopilot-frequency-aligned-robust | 5.850409 | 7.30 | 8.33 | 58.0 |
| uci-concrete-strength | linear | featcopilot-plan-gated-robust | 5.976845 | 5.22 | 5.43 | 49.7 |
| uci-dry-bean | lightgbm | baseline-discovery-robust | 0.248388 | 0.04 | 4.15 | 16.0 |
| uci-dry-bean | lightgbm | baseline-robust | 0.248388 | 0.03 | 5.71 | 16.0 |
| uci-dry-bean | lightgbm | featcopilot-frequency-aligned-robust | 0.248178 | 160.08 | 164.74 | 16.3 |
| uci-dry-bean | lightgbm | featcopilot-plan-gated-robust | 0.248432 | 136.35 | 144.07 | 16.3 |
| uci-dry-bean | linear | baseline-discovery-robust | 0.217692 | 0.04 | 0.42 | 16.0 |
| uci-dry-bean | linear | baseline-robust | 0.215886 | 0.03 | 2.07 | 16.0 |
| uci-dry-bean | linear | featcopilot-frequency-aligned-robust | 0.213348 | 44.97 | 47.55 | 19.3 |
| uci-dry-bean | linear | featcopilot-plan-gated-robust | 0.217692 | 33.20 | 34.16 | 16.0 |
| uci-energy-efficiency | lightgbm | baseline-discovery-robust | 0.472684 | 0.02 | 1.27 | 8.0 |
| uci-energy-efficiency | lightgbm | baseline-robust | 0.479742 | 0.02 | 2.81 | 8.0 |
| uci-energy-efficiency | lightgbm | featcopilot-frequency-aligned-robust | 0.449412 | 13.52 | 14.33 | 41.3 |
| uci-energy-efficiency | lightgbm | featcopilot-plan-gated-robust | 0.506330 | 9.71 | 11.44 | 25.0 |
| uci-energy-efficiency | linear | baseline-discovery-robust | 2.845353 | 0.02 | 0.09 | 8.0 |
| uci-energy-efficiency | linear | baseline-robust | 2.844552 | 0.02 | 1.69 | 8.0 |
| uci-energy-efficiency | linear | featcopilot-frequency-aligned-robust | 0.529265 | 6.54 | 7.44 | 58.0 |
| uci-energy-efficiency | linear | featcopilot-plan-gated-robust | 0.534205 | 4.72 | 4.99 | 58.0 |
| uci-german-credit | lightgbm | baseline-discovery-robust | 0.550998 | 0.05 | 1.46 | 20.0 |
| uci-german-credit | lightgbm | baseline-robust | 0.519777 | 0.04 | 3.03 | 20.0 |
| uci-german-credit | lightgbm | featcopilot-frequency-aligned-robust | 0.521541 | 26.09 | 27.48 | 20.3 |
| uci-german-credit | lightgbm | featcopilot-plan-gated-robust | 0.550998 | 20.04 | 22.10 | 20.0 |
| uci-german-credit | linear | baseline-discovery-robust | 0.509433 | 0.05 | 0.27 | 20.0 |
| uci-german-credit | linear | baseline-robust | 0.498670 | 0.04 | 1.87 | 20.0 |
| uci-german-credit | linear | featcopilot-frequency-aligned-robust | 0.498670 | 16.40 | 17.39 | 20.0 |
| uci-german-credit | linear | featcopilot-plan-gated-robust | 0.509433 | 13.10 | 13.57 | 20.0 |
| uci-online-shoppers | lightgbm | baseline-discovery-robust | 0.380834 | 0.05 | 1.65 | 11.0 |
| uci-online-shoppers | lightgbm | baseline-robust | 0.380834 | 0.04 | 3.33 | 11.0 |
| uci-online-shoppers | lightgbm | featcopilot-frequency-aligned-robust | 0.388462 | 36.84 | 37.83 | 13.0 |
| uci-online-shoppers | lightgbm | featcopilot-plan-gated-robust | 0.380834 | 27.83 | 30.26 | 11.0 |
| uci-online-shoppers | linear | baseline-discovery-robust | 0.395974 | 0.05 | 0.29 | 11.0 |
| uci-online-shoppers | linear | baseline-robust | 0.395017 | 0.04 | 2.19 | 11.0 |
| uci-online-shoppers | linear | featcopilot-frequency-aligned-robust | 0.386049 | 26.15 | 26.67 | 13.3 |
| uci-online-shoppers | linear | featcopilot-plan-gated-robust | 0.392516 | 18.80 | 19.35 | 11.3 |
| uci-real-estate-sindian | lightgbm | baseline-discovery-robust | 7.464846 | 0.02 | 1.08 | 6.0 |
| uci-real-estate-sindian | lightgbm | baseline-robust | 7.448276 | 0.01 | 2.73 | 6.0 |
| uci-real-estate-sindian | lightgbm | featcopilot-frequency-aligned-robust | 7.266136 | 10.56 | 11.11 | 8.3 |
| uci-real-estate-sindian | lightgbm | featcopilot-plan-gated-robust | 7.464846 | 6.77 | 8.15 | 6.0 |
| uci-real-estate-sindian | linear | baseline-discovery-robust | 8.803991 | 0.02 | 0.07 | 6.0 |
| uci-real-estate-sindian | linear | baseline-robust | 8.764502 | 0.01 | 1.66 | 6.0 |
| uci-real-estate-sindian | linear | featcopilot-frequency-aligned-robust | 7.699602 | 5.25 | 5.46 | 14.3 |
| uci-real-estate-sindian | linear | featcopilot-plan-gated-robust | 7.992817 | 3.59 | 3.73 | 12.7 |
| uci-spambase | lightgbm | baseline-discovery-robust | 0.141773 | 0.13 | 2.23 | 57.0 |
| uci-spambase | lightgbm | baseline-robust | 0.141042 | 0.10 | 4.11 | 57.0 |
| uci-spambase | lightgbm | featcopilot-frequency-aligned-robust | 0.140386 | 81.85 | 85.85 | 57.3 |
| uci-spambase | lightgbm | featcopilot-plan-gated-robust | 0.141773 | 59.41 | 63.01 | 57.0 |
| uci-spambase | linear | baseline-discovery-robust | 0.231947 | 0.12 | 0.47 | 57.0 |
| uci-spambase | linear | baseline-robust | 0.235432 | 0.10 | 2.12 | 57.0 |
| uci-spambase | linear | featcopilot-frequency-aligned-robust | 0.166747 | 42.95 | 46.88 | 98.7 |
| uci-spambase | linear | featcopilot-plan-gated-robust | 0.172201 | 33.94 | 34.86 | 90.3 |
| uci-steel-faults | lightgbm | baseline-discovery-robust | 0.557708 | 0.07 | 3.69 | 27.0 |
| uci-steel-faults | lightgbm | baseline-robust | 0.557708 | 0.05 | 5.55 | 27.0 |
| uci-steel-faults | lightgbm | featcopilot-frequency-aligned-robust | 0.557708 | 128.88 | 133.73 | 27.0 |
| uci-steel-faults | lightgbm | featcopilot-plan-gated-robust | 0.557708 | 98.60 | 105.62 | 27.0 |
| uci-steel-faults | linear | baseline-discovery-robust | 0.744395 | 0.07 | 0.53 | 27.0 |
| uci-steel-faults | linear | baseline-robust | 0.754344 | 0.05 | 2.17 | 27.0 |
| uci-steel-faults | linear | featcopilot-frequency-aligned-robust | 0.703115 | 40.53 | 43.04 | 33.7 |
| uci-steel-faults | linear | featcopilot-plan-gated-robust | 0.724258 | 31.33 | 32.32 | 30.3 |
| uci-wisconsin-diagnostic | lightgbm | baseline-discovery-robust | 0.118848 | 0.06 | 1.30 | 30.0 |
| uci-wisconsin-diagnostic | lightgbm | baseline-robust | 0.120432 | 0.05 | 2.95 | 30.0 |
| uci-wisconsin-diagnostic | lightgbm | featcopilot-frequency-aligned-robust | 0.147848 | 22.07 | 25.13 | 33.3 |
| uci-wisconsin-diagnostic | lightgbm | featcopilot-plan-gated-robust | 0.174300 | 17.08 | 18.83 | 32.0 |
| uci-wisconsin-diagnostic | linear | baseline-discovery-robust | 0.082863 | 0.06 | 0.25 | 30.0 |
| uci-wisconsin-diagnostic | linear | baseline-robust | 0.087878 | 0.05 | 1.88 | 30.0 |
| uci-wisconsin-diagnostic | linear | featcopilot-frequency-aligned-robust | 0.091637 | 14.84 | 16.67 | 33.3 |
| uci-wisconsin-diagnostic | linear | featcopilot-plan-gated-robust | 0.117957 | 12.87 | 13.32 | 31.7 |
| wine | lightgbm | baseline-discovery-robust | 0.656946 | 0.03 | 1.49 | 11.0 |
| wine | lightgbm | baseline-robust | 0.654417 | 0.02 | 3.08 | 11.0 |
| wine | lightgbm | featcopilot-frequency-aligned-robust | 0.653127 | 45.67 | 47.93 | 36.3 |
| wine | lightgbm | featcopilot-plan-gated-robust | 0.654324 | 36.89 | 39.30 | 27.7 |
| wine | linear | baseline-discovery-robust | 0.737010 | 0.03 | 0.11 | 11.0 |
| wine | linear | baseline-robust | 0.737010 | 0.02 | 1.69 | 11.0 |
| wine | linear | featcopilot-frequency-aligned-robust | 0.709145 | 17.02 | 18.64 | 61.0 |
| wine | linear | featcopilot-plan-gated-robust | 0.723079 | 12.62 | 12.88 | 36.0 |

## Failures / timeouts

None.

Raw cell metrics and provenance: local `results.json`. Completed runs also write `archive.json`,
which lists the authoritative lossless gzip parts and their uncompressed checksum.
