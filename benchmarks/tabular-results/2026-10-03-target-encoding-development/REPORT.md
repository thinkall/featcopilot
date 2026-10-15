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
Target-encoding variants cross-fit categorical model inputs inside each training partition; their incremental FE comparison uses baseline-target-robust.
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
    "uci-spambase",
    "uci-banknote-authentication",
    "uci-chronic-kidney-disease",
    "uci-wisconsin-diagnostic"
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
    "baseline-target-robust",
    "featcopilot-frequency-aligned-robust",
    "featcopilot-frequency-aligned-target-robust"
  ],
  "learners": [
    "linear",
    "lightgbm"
  ],
  "suite": [
    "C:\\code\\featcopilot\\benchmarks\\suites\\frequency_development_replay_20260924.json"
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
| real_world / development | classification | lightgbm | baseline-target-robust | 9/9 / 9/9 | 27/27 | 0/7/2 | -1.22% | [-3.40%, +0.03%] |
| real_world / development | classification | lightgbm | featcopilot-frequency-aligned-robust | 9/9 / 9/9 | 27/27 | 2/5/2 | +8.12% | [-5.51%, +25.53%] |
| real_world / development | classification | lightgbm | featcopilot-frequency-aligned-target-robust | 9/9 / 9/9 | 27/27 | 2/5/2 | +8.08% | [-5.25%, +25.52%] |
| real_world / development | classification | linear | baseline-robust | 9/9 / 9/9 | 27/27 | 0/9/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | classification | linear | baseline-target-robust | 9/9 / 9/9 | 27/27 | 1/6/2 | -16.73% | [-49.92%, +0.30%] |
| real_world / development | classification | linear | featcopilot-frequency-aligned-robust | 9/9 / 9/9 | 27/27 | 7/1/1 | +16.12% | [+2.17%, +36.78%] |
| real_world / development | classification | linear | featcopilot-frequency-aligned-target-robust | 9/9 / 9/9 | 27/27 | 6/0/3 | +0.89% | [-35.67%, +31.80%] |

## FE versus identically target-encoded raw inputs

Reference: baseline-target-robust; encoder changes alone are not counted as FE improvement.
Target encoding affects only model preprocessing, not raw feature-plan width or candidate operands.

| Source / role | Task | Learner | Method | Paired datasets / families | Cells OK/all | W/T/L | Mean loss reduction | 95% CI |
|---|---|---|---|---|---|---|---|---|
| real_world / development | classification | lightgbm | baseline-target-robust | 9/9 / 9/9 | 27/27 | 0/9/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | classification | lightgbm | featcopilot-frequency-aligned-target-robust | 9/9 / 9/9 | 27/27 | 2/6/1 | +8.92% | [-5.02%, +26.34%] |
| real_world / development | classification | linear | baseline-target-robust | 9/9 / 9/9 | 27/27 | 0/9/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / development | classification | linear | featcopilot-frequency-aligned-target-robust | 9/9 / 9/9 | 27/27 | 7/1/1 | +15.52% | [+1.66%, +36.17%] |

## Per-dataset means (complete folds only)

| Dataset | Learner | Method | Primary loss | FE seconds (inner + outer) | Pipeline seconds | Features |
|---|---|---|---|---|---|---|
| uci-adult | lightgbm | baseline-robust | 0.309826 | 0.05 | 3.80 | 13.0 |
| uci-adult | lightgbm | baseline-target-robust | 0.314360 | 0.01 | 3.82 | 13.0 |
| uci-adult | lightgbm | featcopilot-frequency-aligned-robust | 0.312557 | 40.07 | 41.20 | 31.3 |
| uci-adult | lightgbm | featcopilot-frequency-aligned-target-robust | 0.314638 | 40.29 | 41.81 | 29.7 |
| uci-adult | linear | baseline-robust | 0.340534 | 0.05 | 2.34 | 13.0 |
| uci-adult | linear | baseline-target-robust | 0.341054 | 0.01 | 2.41 | 13.0 |
| uci-adult | linear | featcopilot-frequency-aligned-robust | 0.331841 | 30.18 | 30.87 | 24.7 |
| uci-adult | linear | featcopilot-frequency-aligned-target-robust | 0.334998 | 32.24 | 33.40 | 18.0 |
| uci-banknote-authentication | lightgbm | baseline-robust | 0.015250 | 0.01 | 3.37 | 4.0 |
| uci-banknote-authentication | lightgbm | baseline-target-robust | 0.015250 | 0.01 | 3.23 | 4.0 |
| uci-banknote-authentication | lightgbm | featcopilot-frequency-aligned-robust | 0.005112 | 12.96 | 13.63 | 37.3 |
| uci-banknote-authentication | lightgbm | featcopilot-frequency-aligned-target-robust | 0.005112 | 12.60 | 13.23 | 37.3 |
| uci-banknote-authentication | linear | baseline-robust | 0.029393 | 0.01 | 2.10 | 4.0 |
| uci-banknote-authentication | linear | baseline-target-robust | 0.029393 | 0.01 | 2.06 | 4.0 |
| uci-banknote-authentication | linear | featcopilot-frequency-aligned-robust | 0.002743 | 6.44 | 6.85 | 30.7 |
| uci-banknote-authentication | linear | featcopilot-frequency-aligned-target-robust | 0.002743 | 6.48 | 6.91 | 30.7 |
| uci-chronic-kidney-disease | lightgbm | baseline-robust | 0.035504 | 0.04 | 3.22 | 24.0 |
| uci-chronic-kidney-disease | lightgbm | baseline-target-robust | 0.038943 | 0.02 | 3.35 | 24.0 |
| uci-chronic-kidney-disease | lightgbm | featcopilot-frequency-aligned-robust | 0.024128 | 21.07 | 23.26 | 44.0 |
| uci-chronic-kidney-disease | lightgbm | featcopilot-frequency-aligned-target-robust | 0.024888 | 22.73 | 25.80 | 27.3 |
| uci-chronic-kidney-disease | linear | baseline-robust | 0.014859 | 0.04 | 2.23 | 24.0 |
| uci-chronic-kidney-disease | linear | baseline-target-robust | 0.036953 | 0.02 | 2.26 | 24.0 |
| uci-chronic-kidney-disease | linear | featcopilot-frequency-aligned-robust | 0.012371 | 16.29 | 18.84 | 28.0 |
| uci-chronic-kidney-disease | linear | featcopilot-frequency-aligned-target-robust | 0.032223 | 19.29 | 23.42 | 59.0 |
| uci-dry-bean | lightgbm | baseline-robust | 0.248388 | 0.03 | 6.58 | 16.0 |
| uci-dry-bean | lightgbm | baseline-target-robust | 0.248388 | 0.03 | 6.06 | 16.0 |
| uci-dry-bean | lightgbm | featcopilot-frequency-aligned-robust | 0.248178 | 159.62 | 164.18 | 16.3 |
| uci-dry-bean | lightgbm | featcopilot-frequency-aligned-target-robust | 0.248178 | 159.98 | 164.60 | 16.3 |
| uci-dry-bean | linear | baseline-robust | 0.215886 | 0.03 | 2.40 | 16.0 |
| uci-dry-bean | linear | baseline-target-robust | 0.215886 | 0.03 | 2.38 | 16.0 |
| uci-dry-bean | linear | featcopilot-frequency-aligned-robust | 0.213348 | 46.06 | 48.63 | 19.3 |
| uci-dry-bean | linear | featcopilot-frequency-aligned-target-robust | 0.213348 | 46.03 | 48.62 | 19.3 |
| uci-german-credit | lightgbm | baseline-robust | 0.519777 | 0.04 | 3.42 | 20.0 |
| uci-german-credit | lightgbm | baseline-target-robust | 0.520140 | 0.02 | 3.54 | 20.0 |
| uci-german-credit | lightgbm | featcopilot-frequency-aligned-robust | 0.521541 | 26.14 | 27.50 | 20.3 |
| uci-german-credit | lightgbm | featcopilot-frequency-aligned-target-robust | 0.520140 | 27.40 | 29.68 | 20.0 |
| uci-german-credit | linear | baseline-robust | 0.498670 | 0.04 | 2.20 | 20.0 |
| uci-german-credit | linear | baseline-target-robust | 0.515391 | 0.02 | 2.35 | 20.0 |
| uci-german-credit | linear | featcopilot-frequency-aligned-robust | 0.498670 | 16.82 | 17.82 | 20.0 |
| uci-german-credit | linear | featcopilot-frequency-aligned-target-robust | 0.513302 | 19.94 | 21.91 | 20.3 |
| uci-online-shoppers | lightgbm | baseline-robust | 0.380834 | 0.04 | 4.07 | 11.0 |
| uci-online-shoppers | lightgbm | baseline-target-robust | 0.380481 | 0.01 | 4.24 | 11.0 |
| uci-online-shoppers | lightgbm | featcopilot-frequency-aligned-robust | 0.388462 | 37.22 | 38.15 | 13.0 |
| uci-online-shoppers | lightgbm | featcopilot-frequency-aligned-target-robust | 0.380481 | 38.25 | 39.50 | 11.0 |
| uci-online-shoppers | linear | baseline-robust | 0.395017 | 0.04 | 2.27 | 11.0 |
| uci-online-shoppers | linear | baseline-target-robust | 0.393365 | 0.01 | 2.39 | 11.0 |
| uci-online-shoppers | linear | featcopilot-frequency-aligned-robust | 0.386049 | 26.50 | 27.02 | 13.3 |
| uci-online-shoppers | linear | featcopilot-frequency-aligned-target-robust | 0.381901 | 30.33 | 31.18 | 17.7 |
| uci-spambase | lightgbm | baseline-robust | 0.141042 | 0.10 | 4.46 | 57.0 |
| uci-spambase | lightgbm | baseline-target-robust | 0.141042 | 0.10 | 4.52 | 57.0 |
| uci-spambase | lightgbm | featcopilot-frequency-aligned-robust | 0.140386 | 79.47 | 83.39 | 57.3 |
| uci-spambase | lightgbm | featcopilot-frequency-aligned-target-robust | 0.140386 | 81.94 | 85.90 | 57.3 |
| uci-spambase | linear | baseline-robust | 0.235432 | 0.10 | 2.46 | 57.0 |
| uci-spambase | linear | baseline-target-robust | 0.235432 | 0.10 | 2.44 | 57.0 |
| uci-spambase | linear | featcopilot-frequency-aligned-robust | 0.166747 | 42.81 | 46.71 | 98.7 |
| uci-spambase | linear | featcopilot-frequency-aligned-target-robust | 0.166747 | 42.91 | 46.85 | 98.7 |
| uci-steel-faults | lightgbm | baseline-robust | 0.557708 | 0.05 | 5.97 | 27.0 |
| uci-steel-faults | lightgbm | baseline-target-robust | 0.557103 | 0.04 | 6.05 | 27.0 |
| uci-steel-faults | lightgbm | featcopilot-frequency-aligned-robust | 0.557708 | 126.96 | 131.66 | 27.0 |
| uci-steel-faults | lightgbm | featcopilot-frequency-aligned-target-robust | 0.557103 | 129.73 | 135.02 | 27.0 |
| uci-steel-faults | linear | baseline-robust | 0.754344 | 0.05 | 2.51 | 27.0 |
| uci-steel-faults | linear | baseline-target-robust | 0.745258 | 0.04 | 2.55 | 27.0 |
| uci-steel-faults | linear | featcopilot-frequency-aligned-robust | 0.703115 | 40.47 | 42.95 | 33.7 |
| uci-steel-faults | linear | featcopilot-frequency-aligned-target-robust | 0.707955 | 40.08 | 42.85 | 35.3 |
| uci-wisconsin-diagnostic | lightgbm | baseline-robust | 0.120432 | 0.04 | 3.28 | 30.0 |
| uci-wisconsin-diagnostic | lightgbm | baseline-target-robust | 0.120432 | 0.05 | 3.30 | 30.0 |
| uci-wisconsin-diagnostic | lightgbm | featcopilot-frequency-aligned-robust | 0.147848 | 22.53 | 25.57 | 33.3 |
| uci-wisconsin-diagnostic | lightgbm | featcopilot-frequency-aligned-target-robust | 0.147848 | 22.35 | 25.38 | 33.3 |
| uci-wisconsin-diagnostic | linear | baseline-robust | 0.087878 | 0.05 | 2.25 | 30.0 |
| uci-wisconsin-diagnostic | linear | baseline-target-robust | 0.087878 | 0.05 | 2.50 | 30.0 |
| uci-wisconsin-diagnostic | linear | featcopilot-frequency-aligned-robust | 0.091637 | 15.23 | 17.05 | 33.3 |
| uci-wisconsin-diagnostic | linear | featcopilot-frequency-aligned-target-robust | 0.091637 | 15.35 | 17.20 | 33.3 |

## Failures / timeouts

None.

Raw cell metrics and provenance: local `results.json`. Completed runs also write `archive.json`,
which lists the authoritative lossless gzip parts and their uncompressed checksum.
