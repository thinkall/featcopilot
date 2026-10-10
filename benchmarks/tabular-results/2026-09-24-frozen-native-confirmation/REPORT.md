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
  "datasets": null,
  "folds": 3,
  "seeds": [
    2026
  ],
  "max_rows": 0,
  "max_added_features": 50,
  "max_candidates": 1000,
  "threads": 8,
  "memory_limit_mb": 4096,
  "case_timeout": 900.0,
  "synthetic": false,
  "stability_repeats": 3,
  "screening": "auto",
  "allow_confirmation": true,
  "detailed_diagnostics": false,
  "live_llm": false,
  "llm_ledger": null,
  "allow_soft_llm_tokens": false,
  "methods": [
    "baseline",
    "featcopilot-importance-paired",
    "openfe-inductive-native",
    "autofeat-native"
  ],
  "learners": [
    "linear",
    "lightgbm"
  ],
  "suite": [
    "C:\\code\\featcopilot\\benchmarks\\suites\\final_reserved_confirmation_20260924.json"
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
| real_world / confirmation | classification | lightgbm | autofeat-native | 6/9 / 6/9 | 8/11 | 3/1/2 | +2.62% | [-1.24%, +6.26%] |
| real_world / confirmation | classification | lightgbm | baseline | 9/9 / 9/9 | 11/11 | 0/9/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | classification | lightgbm | featcopilot-importance-paired | 9/9 / 9/9 | 11/11 | 2/7/0 | +1.13% | [+0.00%, +2.88%] |
| real_world / confirmation | classification | lightgbm | openfe-inductive-native | 9/9 / 9/9 | 11/11 | 4/2/3 | +9.39% | [-1.51%, +25.13%] |
| real_world / confirmation | classification | linear | autofeat-native | 6/9 / 6/9 | 8/11 | 3/2/1 | +6.20% | [-0.08%, +12.60%] |
| real_world / confirmation | classification | linear | baseline | 9/9 / 9/9 | 11/11 | 0/9/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | classification | linear | featcopilot-importance-paired | 9/9 / 9/9 | 11/11 | 4/4/1 | +7.09% | [+1.26%, +13.57%] |
| real_world / confirmation | classification | linear | openfe-inductive-native | 9/9 / 9/9 | 11/11 | 3/3/3 | -5.86% | [-22.48%, +4.71%] |
| real_world / confirmation | regression | lightgbm | autofeat-native | 2/2 / 2/2 | 4/4 | 1/1/0 | +3.24% | [+0.53%, +5.95%] |
| real_world / confirmation | regression | lightgbm | baseline | 2/2 / 2/2 | 4/4 | 0/2/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | regression | lightgbm | featcopilot-importance-paired | 2/2 / 2/2 | 4/4 | 0/1/1 | -0.95% | [-2.33%, +0.43%] |
| real_world / confirmation | regression | lightgbm | openfe-inductive-native | 2/2 / 2/2 | 4/4 | 2/0/0 | +2.17% | [+1.83%, +2.51%] |
| real_world / confirmation | regression | linear | autofeat-native | 2/2 / 2/2 | 4/4 | 0/0/2 | -7.31% | [-13.45%, -1.18%] |
| real_world / confirmation | regression | linear | baseline | 2/2 / 2/2 | 4/4 | 0/2/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | regression | linear | featcopilot-importance-paired | 2/2 / 2/2 | 4/4 | 0/2/0 | +0.34% | [+0.00%, +0.69%] |
| real_world / confirmation | regression | linear | openfe-inductive-native | 2/2 / 2/2 | 4/4 | 1/0/1 | -1.33% | [-4.30%, +1.64%] |
| simulation / confirmation | regression | lightgbm | autofeat-native | 0/1 / 0/1 | 0/1 | 0/0/0 | n/a | n/a |
| simulation / confirmation | regression | lightgbm | baseline | 1/1 / 1/1 | 1/1 | 0/1/0 | +0.00% | n/a |
| simulation / confirmation | regression | lightgbm | featcopilot-importance-paired | 1/1 / 1/1 | 1/1 | 1/0/0 | +5.79% | n/a |
| simulation / confirmation | regression | lightgbm | openfe-inductive-native | 1/1 / 1/1 | 1/1 | 1/0/0 | +5.08% | n/a |
| simulation / confirmation | regression | linear | autofeat-native | 0/1 / 0/1 | 0/1 | 0/0/0 | n/a | n/a |
| simulation / confirmation | regression | linear | baseline | 1/1 / 1/1 | 1/1 | 0/1/0 | +0.00% | n/a |
| simulation / confirmation | regression | linear | featcopilot-importance-paired | 1/1 / 1/1 | 1/1 | 1/0/0 | +7.60% | n/a |
| simulation / confirmation | regression | linear | openfe-inductive-native | 1/1 / 1/1 | 1/1 | 1/0/0 | +7.53% | n/a |

## Per-dataset means (complete folds only)

| Dataset | Learner | Method | Primary loss | FE seconds (inner + outer) | Pipeline seconds | Features |
|---|---|---|---|---|---|---|
| tabarena-46907 | lightgbm | autofeat-native | 699.972030 | 50.11 | 51.20 | 13.0 |
| tabarena-46907 | lightgbm | baseline | 744.224731 | 0.02 | 1.31 | 7.0 |
| tabarena-46907 | lightgbm | featcopilot-importance-paired | 761.601999 | 10.51 | 11.38 | 7.0 |
| tabarena-46907 | lightgbm | openfe-inductive-native | 725.547132 | 122.32 | 123.40 | 72.0 |
| tabarena-46907 | linear | autofeat-native | 737.736132 | 52.10 | 52.16 | 13.0 |
| tabarena-46907 | linear | baseline | 729.153137 | 0.02 | 0.08 | 7.0 |
| tabarena-46907 | linear | featcopilot-importance-paired | 724.129366 | 2.44 | 2.53 | 12.0 |
| tabarena-46907 | linear | openfe-inductive-native | 717.222692 | 122.26 | 122.35 | 72.0 |
| tabarena-46913 | lightgbm | autofeat-native | 0.585720 | 62.01 | 62.82 | 7.0 |
| tabarena-46913 | lightgbm | baseline | 0.563041 | 0.01 | 1.06 | 4.0 |
| tabarena-46913 | lightgbm | featcopilot-importance-paired | 0.563041 | 5.78 | 6.07 | 4.0 |
| tabarena-46913 | lightgbm | openfe-inductive-native | 0.565104 | 120.35 | 120.71 | 25.0 |
| tabarena-46913 | linear | autofeat-native | 0.483081 | 61.15 | 61.24 | 4.0 |
| tabarena-46913 | linear | baseline | 0.483081 | 0.01 | 0.10 | 4.0 |
| tabarena-46913 | linear | featcopilot-importance-paired | 0.496938 | 106.28 | 106.37 | 9.0 |
| tabarena-46913 | linear | openfe-inductive-native | 0.479647 | 118.85 | 118.95 | 25.0 |
| tabarena-46930 | lightgbm | autofeat-native | 0.219294 | 410.88 | 412.30 | 72.0 |
| tabarena-46930 | lightgbm | baseline | 0.234332 | 0.08 | 1.67 | 30.0 |
| tabarena-46930 | lightgbm | featcopilot-importance-paired | 0.234332 | 21.68 | 22.50 | 30.0 |
| tabarena-46930 | lightgbm | openfe-inductive-native | 0.211708 | 138.77 | 152.84 | 2016.0 |
| tabarena-46930 | linear | autofeat-native | 0.244230 | 415.41 | 415.61 | 65.0 |
| tabarena-46930 | linear | baseline | 0.296839 | 0.08 | 0.26 | 30.0 |
| tabarena-46930 | linear | featcopilot-importance-paired | 0.277242 | 110.84 | 111.09 | 35.0 |
| tabarena-46930 | linear | openfe-inductive-native | 0.295277 | 137.54 | 140.16 | 2016.0 |
| tabarena-46941 | lightgbm | autofeat-native | 0.471995 | 101.01 | 102.46 | 12.0 |
| tabarena-46941 | lightgbm | baseline | 0.507376 | 0.02 | 1.56 | 6.0 |
| tabarena-46941 | lightgbm | featcopilot-importance-paired | 0.507376 | 14.80 | 15.70 | 6.0 |
| tabarena-46941 | lightgbm | openfe-inductive-native | 0.416646 | 125.16 | 126.44 | 81.0 |
| tabarena-46941 | linear | autofeat-native | 0.730114 | 101.85 | 102.01 | 11.0 |
| tabarena-46941 | linear | baseline | 0.777171 | 0.02 | 0.12 | 6.0 |
| tabarena-46941 | linear | featcopilot-importance-paired | 0.673545 | 107.91 | 108.12 | 31.0 |
| tabarena-46941 | linear | openfe-inductive-native | 0.683053 | 127.21 | 127.67 | 81.0 |
| tabarena-46949 | lightgbm | baseline | 3.991828 | 0.05 | 2.05 | 9.0 |
| tabarena-46949 | lightgbm | featcopilot-importance-paired | 3.760614 | 32.45 | 35.37 | 59.0 |
| tabarena-46949 | lightgbm | openfe-inductive-native | 3.789015 | 142.24 | 150.16 | 201.0 |
| tabarena-46949 | linear | baseline | 5.147532 | 0.05 | 0.18 | 9.0 |
| tabarena-46949 | linear | featcopilot-importance-paired | 4.756303 | 10.78 | 11.45 | 59.0 |
| tabarena-46949 | linear | openfe-inductive-native | 4.759841 | 143.86 | 146.34 | 201.0 |
| tabarena-46950 | lightgbm | baseline | 0.097912 | 0.18 | 2.20 | 64.0 |
| tabarena-46950 | lightgbm | featcopilot-importance-paired | 0.090186 | 36.70 | 37.99 | 89.0 |
| tabarena-46950 | lightgbm | openfe-inductive-native | 0.030861 | 258.87 | 277.68 | 2064.0 |
| tabarena-46950 | linear | baseline | 0.321696 | 0.19 | 0.61 | 64.0 |
| tabarena-46950 | linear | featcopilot-importance-paired | 0.236967 | 119.89 | 120.24 | 89.0 |
| tabarena-46950 | linear | openfe-inductive-native | 0.293665 | 261.56 | 268.09 | 2064.0 |
| tabarena-46956 | lightgbm | autofeat-native | 0.218479 | 290.81 | 292.10 | 29.0 |
| tabarena-46956 | lightgbm | baseline | 0.218956 | 0.04 | 1.75 | 15.0 |
| tabarena-46956 | lightgbm | featcopilot-importance-paired | 0.218956 | 15.78 | 16.63 | 15.0 |
| tabarena-46956 | lightgbm | openfe-inductive-native | 0.219196 | 132.95 | 134.70 | 270.0 |
| tabarena-46956 | linear | autofeat-native | 0.212900 | 308.77 | 308.95 | 29.0 |
| tabarena-46956 | linear | baseline | 0.211901 | 0.04 | 0.22 | 15.0 |
| tabarena-46956 | linear | featcopilot-importance-paired | 0.210464 | 108.93 | 109.23 | 16.0 |
| tabarena-46956 | linear | openfe-inductive-native | 0.227296 | 131.41 | 131.98 | 270.0 |
| tabarena-46960 | lightgbm | baseline | 0.884446 | 0.08 | 2.78 | 18.0 |
| tabarena-46960 | lightgbm | featcopilot-importance-paired | 0.884446 | 40.25 | 42.28 | 18.0 |
| tabarena-46960 | lightgbm | openfe-inductive-native | 0.910201 | 188.85 | 197.93 | 459.0 |
| tabarena-46960 | linear | baseline | 0.893117 | 0.08 | 0.85 | 18.0 |
| tabarena-46960 | linear | featcopilot-importance-paired | 0.893117 | 127.78 | 128.78 | 18.0 |
| tabarena-46960 | linear | openfe-inductive-native | 0.955104 | 193.64 | 202.22 | 459.0 |
| tabarena-46962 | lightgbm | baseline | 0.082978 | 0.27 | 2.86 | 94.0 |
| tabarena-46962 | lightgbm | featcopilot-importance-paired | 0.082978 | 33.00 | 34.48 | 94.0 |
| tabarena-46962 | lightgbm | openfe-inductive-native | 0.088595 | 297.42 | 320.11 | 2535.0 |
| tabarena-46962 | linear | baseline | 0.107230 | 0.26 | 0.70 | 94.0 |
| tabarena-46962 | linear | featcopilot-importance-paired | 0.107230 | 119.71 | 120.10 | 94.0 |
| tabarena-46962 | linear | openfe-inductive-native | 0.178274 | 299.86 | 311.05 | 2535.0 |
| tabarena-46963 | lightgbm | autofeat-native | 0.217851 | 149.50 | 151.48 | 55.0 |
| tabarena-46963 | lightgbm | baseline | 0.238037 | 0.02 | 1.84 | 9.0 |
| tabarena-46963 | lightgbm | featcopilot-importance-paired | 0.238037 | 18.71 | 19.77 | 9.0 |
| tabarena-46963 | lightgbm | openfe-inductive-native | 0.233079 | 128.70 | 129.81 | 57.0 |
| tabarena-46963 | linear | autofeat-native | 0.278578 | 144.09 | 144.39 | 44.0 |
| tabarena-46963 | linear | baseline | 0.334959 | 0.02 | 0.19 | 9.0 |
| tabarena-46963 | linear | featcopilot-importance-paired | 0.269005 | 109.49 | 109.80 | 34.0 |
| tabarena-46963 | linear | openfe-inductive-native | 0.315200 | 125.38 | 125.69 | 57.0 |
| uci-heart-failure-clinical | lightgbm | autofeat-native | 0.730441 | 60.42 | 61.24 | 26.0 |
| uci-heart-failure-clinical | lightgbm | baseline | 0.713600 | 0.02 | 1.11 | 11.0 |
| uci-heart-failure-clinical | lightgbm | featcopilot-importance-paired | 0.697502 | 5.25 | 5.61 | 11.3 |
| uci-heart-failure-clinical | lightgbm | openfe-inductive-native | 0.738140 | 123.64 | 123.94 | 12.0 |
| uci-heart-failure-clinical | linear | autofeat-native | 0.559833 | 60.07 | 60.20 | 24.7 |
| uci-heart-failure-clinical | linear | baseline | 0.543781 | 0.02 | 0.15 | 11.0 |
| uci-heart-failure-clinical | linear | featcopilot-importance-paired | 0.543781 | 107.28 | 107.47 | 11.0 |
| uci-heart-failure-clinical | linear | openfe-inductive-native | 0.545139 | 120.73 | 120.86 | 12.0 |
| uci-student-performance | lightgbm | autofeat-native | 2.931193 | 120.19 | 121.31 | 93.7 |
| uci-student-performance | lightgbm | baseline | 2.946800 | 0.06 | 1.37 | 30.0 |
| uci-student-performance | lightgbm | featcopilot-importance-paired | 2.934063 | 12.78 | 13.96 | 33.3 |
| uci-student-performance | lightgbm | openfe-inductive-native | 2.892811 | 128.02 | 128.59 | 245.3 |
| uci-student-performance | linear | autofeat-native | 3.141022 | 117.01 | 117.17 | 84.3 |
| uci-student-performance | linear | baseline | 2.768656 | 0.06 | 0.18 | 30.0 |
| uci-student-performance | linear | featcopilot-importance-paired | 2.768656 | 6.61 | 7.35 | 30.0 |
| uci-student-performance | linear | openfe-inductive-native | 2.887734 | 127.34 | 127.53 | 245.3 |

## Failures / timeouts

- tabarena-46950 / 2026 / fold 0 / autofeat-native / lightgbm: memory_limit: Worker tree exceeded the configured RSS memory limit
- tabarena-46962 / 2026 / fold 0 / autofeat-native / lightgbm: memory_limit: Worker tree exceeded the configured RSS memory limit
- tabarena-46949 / 2026 / fold 0 / autofeat-native / linear: memory_limit: Worker tree exceeded the configured RSS memory limit
- tabarena-46950 / 2026 / fold 0 / autofeat-native / linear: memory_limit: Worker tree exceeded the configured RSS memory limit
- tabarena-46960 / 2026 / fold 0 / autofeat-native / lightgbm: timeout: Case exceeded 900.0s wall-clock limit
- tabarena-46949 / 2026 / fold 0 / autofeat-native / lightgbm: memory_limit: Worker tree exceeded the configured RSS memory limit
- tabarena-46960 / 2026 / fold 0 / autofeat-native / linear: timeout: Case exceeded 900.0s wall-clock limit
- tabarena-46962 / 2026 / fold 0 / autofeat-native / linear: memory_limit: Worker tree exceeded the configured RSS memory limit

Raw cell metrics and provenance: local `results.json`. Completed runs also write `archive.json`,
which lists the authoritative lossless gzip parts and their uncompressed checksum.
