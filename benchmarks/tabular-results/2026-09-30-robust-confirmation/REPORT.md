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
  "datasets": null,
  "folds": 3,
  "seeds": [
    2028
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
    "baseline-robust",
    "featcopilot-frequency-aligned-robust",
    "openfe-inductive-native-robust",
    "autofeat-native-robust"
  ],
  "learners": [
    "linear",
    "lightgbm",
    "catboost",
    "xgboost"
  ],
  "suite": [
    "C:\\code\\featcopilot\\benchmarks\\suites\\cycle2_robust_confirmation_20260930.json"
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
| real_world / confirmation | classification | catboost | autofeat-native-robust | 2/3 / 2/3 | 6/9 | 1/1/0 | +0.85% | [+0.27%, +1.43%] |
| real_world / confirmation | classification | catboost | baseline-robust | 3/3 / 3/3 | 9/9 | 0/3/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | classification | catboost | featcopilot-frequency-aligned-robust | 2/3 / 2/3 | 6/9 | 0/1/1 | -7.15% | [-13.76%, -0.54%] |
| real_world / confirmation | classification | catboost | openfe-inductive-native-robust | 3/3 / 3/3 | 9/9 | 1/2/0 | +0.89% | [-0.37%, +2.66%] |
| real_world / confirmation | classification | lightgbm | autofeat-native-robust | 2/3 / 2/3 | 6/9 | 0/1/1 | -0.77% | [-1.08%, -0.46%] |
| real_world / confirmation | classification | lightgbm | baseline-robust | 3/3 / 3/3 | 9/9 | 0/3/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | classification | lightgbm | featcopilot-frequency-aligned-robust | 3/3 / 3/3 | 9/9 | 0/3/0 | +0.01% | [-0.20%, +0.22%] |
| real_world / confirmation | classification | lightgbm | openfe-inductive-native-robust | 3/3 / 3/3 | 9/9 | 0/1/2 | -6.52% | [-10.00%, +0.39%] |
| real_world / confirmation | classification | linear | autofeat-native-robust | 2/3 / 2/3 | 6/9 | 1/0/1 | -2.07% | [-5.93%, +1.79%] |
| real_world / confirmation | classification | linear | baseline-robust | 3/3 / 3/3 | 9/9 | 0/3/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | classification | linear | featcopilot-frequency-aligned-robust | 3/3 / 3/3 | 9/9 | 1/2/0 | +0.56% | [+0.00%, +1.68%] |
| real_world / confirmation | classification | linear | openfe-inductive-native-robust | 3/3 / 3/3 | 9/9 | 0/1/2 | -8.28% | [-17.32%, +0.28%] |
| real_world / confirmation | classification | xgboost | autofeat-native-robust | 2/3 / 2/3 | 6/9 | 0/0/2 | -1.47% | [-1.90%, -1.03%] |
| real_world / confirmation | classification | xgboost | baseline-robust | 3/3 / 3/3 | 9/9 | 0/3/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | classification | xgboost | featcopilot-frequency-aligned-robust | 3/3 / 3/3 | 9/9 | 0/2/1 | -2.61% | [-7.89%, +0.07%] |
| real_world / confirmation | classification | xgboost | openfe-inductive-native-robust | 3/3 / 3/3 | 9/9 | 0/1/2 | -3.18% | [-5.60%, +0.64%] |
| real_world / confirmation | regression | catboost | autofeat-native-robust | 1/1 / 1/1 | 3/3 | 0/0/1 | -1.08% | n/a |
| real_world / confirmation | regression | catboost | baseline-robust | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| real_world / confirmation | regression | catboost | featcopilot-frequency-aligned-robust | 1/1 / 1/1 | 3/3 | 0/0/1 | -2.77% | n/a |
| real_world / confirmation | regression | catboost | openfe-inductive-native-robust | 1/1 / 1/1 | 3/3 | 0/0/1 | -1.75% | n/a |
| real_world / confirmation | regression | lightgbm | autofeat-native-robust | 1/1 / 1/1 | 3/3 | 1/0/0 | +1.72% | n/a |
| real_world / confirmation | regression | lightgbm | baseline-robust | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| real_world / confirmation | regression | lightgbm | featcopilot-frequency-aligned-robust | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| real_world / confirmation | regression | lightgbm | openfe-inductive-native-robust | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.75% | n/a |
| real_world / confirmation | regression | linear | autofeat-native-robust | 1/1 / 1/1 | 3/3 | 0/1/0 | -0.06% | n/a |
| real_world / confirmation | regression | linear | baseline-robust | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| real_world / confirmation | regression | linear | featcopilot-frequency-aligned-robust | 1/1 / 1/1 | 3/3 | 0/1/0 | -0.01% | n/a |
| real_world / confirmation | regression | linear | openfe-inductive-native-robust | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.48% | n/a |
| real_world / confirmation | regression | xgboost | autofeat-native-robust | 1/1 / 1/1 | 3/3 | 0/0/1 | -41.69% | n/a |
| real_world / confirmation | regression | xgboost | baseline-robust | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| real_world / confirmation | regression | xgboost | featcopilot-frequency-aligned-robust | 1/1 / 1/1 | 3/3 | 1/0/0 | +3.82% | n/a |
| real_world / confirmation | regression | xgboost | openfe-inductive-native-robust | 1/1 / 1/1 | 3/3 | 0/0/1 | -1.36% | n/a |

## Per-dataset means (complete folds only)

| Dataset | Learner | Method | Primary loss | FE seconds (inner + outer) | Pipeline seconds | Features |
|---|---|---|---|---|---|---|
| assam-medical-entrance | catboost | autofeat-native-robust | 0.994059 | 158.23 | 250.67 | 60.0 |
| assam-medical-entrance | catboost | baseline-robust | 0.996720 | 0.02 | 83.56 | 10.0 |
| assam-medical-entrance | catboost | featcopilot-frequency-aligned-robust | 1.002142 | 240.13 | 336.33 | 12.0 |
| assam-medical-entrance | catboost | openfe-inductive-native-robust | 0.993009 | 162.49 | 251.63 | 33.3 |
| assam-medical-entrance | lightgbm | autofeat-native-robust | 1.110886 | 158.09 | 162.11 | 57.3 |
| assam-medical-entrance | lightgbm | baseline-robust | 1.105759 | 0.02 | 3.84 | 10.0 |
| assam-medical-entrance | lightgbm | featcopilot-frequency-aligned-robust | 1.103332 | 32.91 | 34.36 | 10.7 |
| assam-medical-entrance | lightgbm | openfe-inductive-native-robust | 1.216381 | 162.95 | 165.82 | 33.3 |
| assam-medical-entrance | linear | autofeat-native-robust | 1.088320 | 156.72 | 158.83 | 57.3 |
| assam-medical-entrance | linear | baseline-robust | 1.027394 | 0.02 | 2.14 | 10.0 |
| assam-medical-entrance | linear | featcopilot-frequency-aligned-robust | 1.010098 | 11.52 | 11.97 | 15.0 |
| assam-medical-entrance | linear | openfe-inductive-native-robust | 1.107431 | 161.83 | 163.69 | 33.3 |
| assam-medical-entrance | xgboost | autofeat-native-robust | 1.088574 | 154.59 | 158.43 | 57.0 |
| assam-medical-entrance | xgboost | baseline-robust | 1.077436 | 0.02 | 3.37 | 10.0 |
| assam-medical-entrance | xgboost | featcopilot-frequency-aligned-robust | 1.077436 | 37.13 | 38.77 | 10.0 |
| assam-medical-entrance | xgboost | openfe-inductive-native-robust | 1.137779 | 161.42 | 165.37 | 33.3 |
| differentiated-thyroid-recurrence | catboost | autofeat-native-robust | 0.298829 | 88.78 | 156.23 | 67.3 |
| differentiated-thyroid-recurrence | catboost | baseline-robust | 0.303167 | 0.02 | 66.25 | 14.0 |
| differentiated-thyroid-recurrence | catboost | featcopilot-frequency-aligned-robust | 0.344869 | 117.75 | 194.24 | 24.0 |
| differentiated-thyroid-recurrence | catboost | openfe-inductive-native-robust | 0.295095 | 162.20 | 223.79 | 15.0 |
| differentiated-thyroid-recurrence | lightgbm | autofeat-native-robust | 0.383694 | 87.75 | 90.75 | 66.0 |
| differentiated-thyroid-recurrence | lightgbm | baseline-robust | 0.379613 | 0.02 | 3.04 | 14.0 |
| differentiated-thyroid-recurrence | lightgbm | featcopilot-frequency-aligned-robust | 0.379613 | 14.20 | 15.06 | 14.0 |
| differentiated-thyroid-recurrence | lightgbm | openfe-inductive-native-robust | 0.378120 | 160.54 | 162.48 | 15.0 |
| differentiated-thyroid-recurrence | linear | autofeat-native-robust | 0.292239 | 88.78 | 90.82 | 67.0 |
| differentiated-thyroid-recurrence | linear | baseline-robust | 0.297576 | 0.02 | 1.86 | 14.0 |
| differentiated-thyroid-recurrence | linear | featcopilot-frequency-aligned-robust | 0.297576 | 10.07 | 10.75 | 14.0 |
| differentiated-thyroid-recurrence | linear | openfe-inductive-native-robust | 0.296747 | 161.68 | 163.98 | 15.0 |
| differentiated-thyroid-recurrence | xgboost | autofeat-native-robust | 0.343481 | 89.12 | 91.59 | 65.3 |
| differentiated-thyroid-recurrence | xgboost | baseline-robust | 0.337072 | 0.02 | 2.31 | 14.0 |
| differentiated-thyroid-recurrence | xgboost | featcopilot-frequency-aligned-robust | 0.363657 | 15.86 | 16.99 | 14.0 |
| differentiated-thyroid-recurrence | xgboost | openfe-inductive-native-robust | 0.334929 | 159.73 | 161.88 | 15.0 |
| facebook-cosmetics-posts | catboost | autofeat-native-robust | 367.235509 | 65.05 | 126.43 | 20.3 |
| facebook-cosmetics-posts | catboost | baseline-robust | 363.298773 | 0.01 | 58.32 | 7.0 |
| facebook-cosmetics-posts | catboost | featcopilot-frequency-aligned-robust | 373.352779 | 63.63 | 125.09 | 10.3 |
| facebook-cosmetics-posts | catboost | openfe-inductive-native-robust | 369.641990 | 160.41 | 224.77 | 8.0 |
| facebook-cosmetics-posts | lightgbm | autofeat-native-robust | 363.001146 | 64.81 | 67.36 | 18.7 |
| facebook-cosmetics-posts | lightgbm | baseline-robust | 369.364837 | 0.01 | 2.89 | 7.0 |
| facebook-cosmetics-posts | lightgbm | featcopilot-frequency-aligned-robust | 369.364837 | 12.08 | 12.48 | 7.0 |
| facebook-cosmetics-posts | lightgbm | openfe-inductive-native-robust | 366.589960 | 160.95 | 162.86 | 8.0 |
| facebook-cosmetics-posts | linear | autofeat-native-robust | 347.531838 | 65.58 | 67.25 | 17.0 |
| facebook-cosmetics-posts | linear | baseline-robust | 347.318033 | 0.01 | 1.74 | 7.0 |
| facebook-cosmetics-posts | linear | featcopilot-frequency-aligned-robust | 347.353495 | 5.24 | 5.40 | 7.3 |
| facebook-cosmetics-posts | linear | openfe-inductive-native-robust | 345.654498 | 158.72 | 160.37 | 8.0 |
| facebook-cosmetics-posts | xgboost | autofeat-native-robust | 554.409886 | 65.95 | 68.46 | 19.0 |
| facebook-cosmetics-posts | xgboost | baseline-robust | 391.284426 | 0.01 | 2.37 | 7.0 |
| facebook-cosmetics-posts | xgboost | featcopilot-frequency-aligned-robust | 376.335862 | 18.41 | 19.05 | 7.7 |
| facebook-cosmetics-posts | xgboost | openfe-inductive-native-robust | 396.616317 | 159.52 | 161.86 | 8.0 |
| personality-drug-consumption | catboost | baseline-robust | 1.508996 | 0.03 | 76.23 | 12.0 |
| personality-drug-consumption | catboost | openfe-inductive-native-robust | 1.514590 | 172.56 | 475.60 | 340.3 |
| personality-drug-consumption | lightgbm | baseline-robust | 1.634710 | 0.03 | 6.13 | 12.0 |
| personality-drug-consumption | lightgbm | featcopilot-frequency-aligned-robust | 1.638050 | 92.73 | 95.99 | 12.3 |
| personality-drug-consumption | lightgbm | openfe-inductive-native-robust | 1.797344 | 172.45 | 187.70 | 340.3 |
| personality-drug-consumption | linear | baseline-robust | 1.527908 | 0.03 | 2.25 | 12.0 |
| personality-drug-consumption | linear | featcopilot-frequency-aligned-robust | 1.527908 | 14.59 | 15.10 | 12.0 |
| personality-drug-consumption | linear | openfe-inductive-native-robust | 1.792597 | 171.60 | 178.62 | 340.3 |
| personality-drug-consumption | xgboost | baseline-robust | 1.569406 | 0.03 | 4.86 | 12.0 |
| personality-drug-consumption | xgboost | featcopilot-frequency-aligned-robust | 1.568384 | 162.09 | 165.52 | 12.3 |
| personality-drug-consumption | xgboost | openfe-inductive-native-robust | 1.640982 | 172.84 | 206.12 | 340.3 |

## Failures / timeouts

- personality-drug-consumption / 2028 / fold 0 / autofeat-native-robust / linear: timeout: Case exceeded 900.0s wall-clock limit
- personality-drug-consumption / 2028 / fold 0 / autofeat-native-robust / catboost: timeout: Case exceeded 900.0s wall-clock limit
- personality-drug-consumption / 2028 / fold 0 / autofeat-native-robust / lightgbm: timeout: Case exceeded 900.0s wall-clock limit
- personality-drug-consumption / 2028 / fold 0 / featcopilot-frequency-aligned-robust / catboost: timeout: Case exceeded 900.0s wall-clock limit
- personality-drug-consumption / 2028 / fold 0 / autofeat-native-robust / xgboost: timeout: Case exceeded 900.0s wall-clock limit
- personality-drug-consumption / 2028 / fold 1 / autofeat-native-robust / lightgbm: timeout: Case exceeded 900.0s wall-clock limit
- personality-drug-consumption / 2028 / fold 1 / autofeat-native-robust / xgboost: timeout: Case exceeded 900.0s wall-clock limit
- personality-drug-consumption / 2028 / fold 1 / autofeat-native-robust / linear: timeout: Case exceeded 900.0s wall-clock limit
- personality-drug-consumption / 2028 / fold 1 / autofeat-native-robust / catboost: timeout: Case exceeded 900.0s wall-clock limit
- personality-drug-consumption / 2028 / fold 1 / featcopilot-frequency-aligned-robust / catboost: timeout: Case exceeded 900.0s wall-clock limit
- personality-drug-consumption / 2028 / fold 2 / autofeat-native-robust / xgboost: timeout: Case exceeded 900.0s wall-clock limit
- personality-drug-consumption / 2028 / fold 2 / featcopilot-frequency-aligned-robust / catboost: timeout: Case exceeded 900.0s wall-clock limit
- personality-drug-consumption / 2028 / fold 2 / autofeat-native-robust / lightgbm: timeout: Case exceeded 900.0s wall-clock limit
- personality-drug-consumption / 2028 / fold 2 / autofeat-native-robust / linear: timeout: Case exceeded 900.0s wall-clock limit
- personality-drug-consumption / 2028 / fold 2 / autofeat-native-robust / catboost: timeout: Case exceeded 900.0s wall-clock limit

Raw cell metrics and provenance: local `results.json`. Completed runs also write `archive.json`,
which lists the authoritative lossless gzip parts and their uncompressed checksum.
