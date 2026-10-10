# Tabular development benchmark

**Benchmark measurement, not a SOTA claim.** Dataset identity, admission roles and split policies
are recorded in the manifest. Legacy manifests are not automatically rights-reviewed or confirmation data.

Primary loss: classification log loss; regression RMSE (lower is better).
Learners are tuned separately per method on inner validation, never selected using outer-test scores.
Synthetic diagnostics are reported separately. All requested failures count toward coverage.

OpenFE-stateless is an explicitly restricted variant, not full OpenFE. AutoFeat uses numeric generation.
All methods retain original categorical features for identical downstream one-hot preprocessing.
FeatCopilot-task changes only explicit task routing; FeatCopilot-predictive uses learner-aligned holdout selection.
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
  "case_timeout": 600.0,
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
    "featcopilot-predictive"
  ],
  "learners": [
    "linear",
    "lightgbm",
    "catboost",
    "xgboost"
  ],
  "suite": [
    "C:\\code\\featcopilot\\benchmarks\\suites\\tabarena_confirmation_candidates_20260920.json"
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
| real_world / confirmation | classification | catboost | baseline | 2/4 / 2/4 | 60/120 | 0/2/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | classification | catboost | featcopilot-predictive | 2/4 / 2/4 | 60/120 | 0/2/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | classification | lightgbm | baseline | 2/4 / 2/4 | 60/120 | 0/2/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | classification | lightgbm | featcopilot-predictive | 2/4 / 2/4 | 60/120 | 0/2/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | classification | linear | baseline | 2/4 / 2/4 | 60/120 | 0/2/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | classification | linear | featcopilot-predictive | 2/4 / 2/4 | 60/120 | 0/2/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | classification | xgboost | baseline | 2/4 / 2/4 | 60/120 | 0/2/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | classification | xgboost | featcopilot-predictive | 2/4 / 2/4 | 60/120 | 0/2/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | regression | catboost | baseline | 2/2 / 2/2 | 60/60 | 0/2/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | regression | catboost | featcopilot-predictive | 2/2 / 2/2 | 60/60 | 1/1/0 | +5.79% | [-0.09%, +11.66%] |
| real_world / confirmation | regression | lightgbm | baseline | 2/2 / 2/2 | 60/60 | 0/2/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | regression | lightgbm | featcopilot-predictive | 2/2 / 2/2 | 60/60 | 1/1/0 | +3.26% | [-0.27%, +6.78%] |
| real_world / confirmation | regression | linear | baseline | 2/2 / 2/2 | 60/60 | 0/2/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | regression | linear | featcopilot-predictive | 2/2 / 2/2 | 60/60 | 1/1/0 | +9.71% | [+0.06%, +19.36%] |
| real_world / confirmation | regression | xgboost | baseline | 2/2 / 2/2 | 60/60 | 0/2/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | regression | xgboost | featcopilot-predictive | 2/2 / 2/2 | 60/60 | 1/1/0 | +0.93% | [-0.24%, +2.10%] |

## Per-dataset means (complete folds only)

| Dataset | Learner | Method | Primary loss | FE seconds (inner + outer) | Pipeline seconds | Features |
|---|---|---|---|---|---|---|
| tabarena-46904 | catboost | baseline | 1.855722 | 0.02 | 49.49 | 5.0 |
| tabarena-46904 | catboost | featcopilot-predictive | 1.639322 | 28.10 | 80.07 | 12.2 |
| tabarena-46904 | lightgbm | baseline | 1.823837 | 0.02 | 1.09 | 5.0 |
| tabarena-46904 | lightgbm | featcopilot-predictive | 1.700118 | 10.65 | 11.30 | 34.3 |
| tabarena-46904 | linear | baseline | 4.751407 | 0.01 | 0.07 | 5.0 |
| tabarena-46904 | linear | featcopilot-predictive | 3.831699 | 2.94 | 3.02 | 31.7 |
| tabarena-46904 | xgboost | baseline | 1.791910 | 0.02 | 0.83 | 5.0 |
| tabarena-46904 | xgboost | featcopilot-predictive | 1.754257 | 15.65 | 16.51 | 17.9 |
| tabarena-46954 | catboost | baseline | 0.879616 | 0.02 | 4.19 | 6.0 |
| tabarena-46954 | catboost | featcopilot-predictive | 0.880413 | 30.04 | 32.99 | 6.0 |
| tabarena-46954 | lightgbm | baseline | 0.907517 | 0.02 | 0.98 | 6.0 |
| tabarena-46954 | lightgbm | featcopilot-predictive | 0.909951 | 10.45 | 10.82 | 6.1 |
| tabarena-46954 | linear | baseline | 0.956630 | 0.02 | 0.06 | 6.0 |
| tabarena-46954 | linear | featcopilot-predictive | 0.956084 | 2.47 | 2.52 | 8.8 |
| tabarena-46954 | xgboost | baseline | 0.919857 | 0.02 | 0.77 | 6.0 |
| tabarena-46954 | xgboost | featcopilot-predictive | 0.922080 | 16.75 | 17.45 | 8.7 |
| tabarena-46958 | catboost | baseline | 0.132611 | 0.21 | 206.97 | 60.0 |
| tabarena-46958 | catboost | featcopilot-predictive | 0.132611 | 8.09 | 214.06 | 60.0 |
| tabarena-46958 | lightgbm | baseline | 0.120119 | 0.21 | 2.89 | 60.0 |
| tabarena-46958 | lightgbm | featcopilot-predictive | 0.120119 | 5.73 | 7.81 | 60.0 |
| tabarena-46958 | linear | baseline | 0.154131 | 0.21 | 0.92 | 60.0 |
| tabarena-46958 | linear | featcopilot-predictive | 0.154131 | 46.71 | 47.44 | 60.0 |
| tabarena-46958 | xgboost | baseline | 0.121782 | 0.21 | 3.01 | 60.0 |
| tabarena-46958 | xgboost | featcopilot-predictive | 0.121782 | 6.30 | 8.96 | 60.0 |
| tabarena-46969 | catboost | baseline | 0.145687 | 0.59 | 12.01 | 86.0 |
| tabarena-46969 | catboost | featcopilot-predictive | 0.145687 | 16.28 | 26.33 | 86.0 |
| tabarena-46969 | lightgbm | baseline | 0.150774 | 0.59 | 3.20 | 86.0 |
| tabarena-46969 | lightgbm | featcopilot-predictive | 0.150774 | 7.89 | 9.93 | 86.0 |
| tabarena-46969 | linear | baseline | 0.167665 | 0.59 | 1.98 | 86.0 |
| tabarena-46969 | linear | featcopilot-predictive | 0.167665 | 50.13 | 51.57 | 86.0 |
| tabarena-46969 | xgboost | baseline | 0.151050 | 0.59 | 2.82 | 86.0 |
| tabarena-46969 | xgboost | featcopilot-predictive | 0.151050 | 7.59 | 9.71 | 86.0 |

## Failures / timeouts

- tabarena-46937 / 2026 / fold 0 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 0 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 0 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 0 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 0 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 0 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 0 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 0 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 0 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 0 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 0 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 0 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 0 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 0 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 0 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 0 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 1 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 1 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 1 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 1 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 1 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 1 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 1 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 1 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 1 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 1 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 1 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 1 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 1 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 1 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 1 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 1 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 2 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 2 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 2 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 2 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 2 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 2 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 2 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 2 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 2 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 2 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 2 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 2 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 2 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 2 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 2 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 2 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 3 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 3 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 3 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 3 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 3 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 3 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 3 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 3 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 3 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 3 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 3 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 3 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 3 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 3 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 3 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 3 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 4 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 4 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 4 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 4 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 4 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 4 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 4 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 4 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 4 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 4 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 4 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 4 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 4 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 4 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 4 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 4 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 5 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 5 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 5 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 5 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 5 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 5 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 5 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 5 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 5 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 5 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 5 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 5 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 5 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 5 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 5 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 5 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 6 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 6 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 6 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 6 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 6 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 6 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 6 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 6 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 6 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 6 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 6 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 6 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 6 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 6 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 6 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 6 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 7 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 7 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 7 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 7 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 7 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 7 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 7 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 7 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 7 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 7 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 7 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 7 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 7 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 7 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 7 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 7 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 8 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 8 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 8 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 8 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 8 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 8 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 8 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 8 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 8 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 8 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 8 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 8 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 8 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 8 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 8 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 8 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 9 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 9 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 9 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 9 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 9 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 9 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 9 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 9 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 9 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 9 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 9 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 9 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 9 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 9 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 9 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 9 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 10 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 10 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 10 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 10 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 10 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 10 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 10 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 10 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 10 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 10 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 10 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 10 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 10 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 10 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 10 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 10 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 11 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 11 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 11 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 11 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 11 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 11 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 11 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 11 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 11 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 11 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 11 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 11 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 11 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 11 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 11 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 11 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 12 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 12 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 12 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 12 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 12 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 12 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 12 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 12 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 12 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 12 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 12 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 12 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 12 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 12 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 12 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 12 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 13 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 13 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 13 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 13 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 13 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 13 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 13 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 13 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 13 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 13 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 13 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 13 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 13 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 13 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 13 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 13 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 14 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 14 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 14 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 14 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 14 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 14 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 14 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 14 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 14 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 14 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 14 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 14 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 14 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 14 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 14 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 14 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 15 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 15 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 15 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 15 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 15 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 15 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 15 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 15 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 15 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 15 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 15 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 15 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 15 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 15 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 15 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 15 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 16 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 16 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 16 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 16 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 16 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 16 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 16 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 16 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 16 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 16 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 16 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 16 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 16 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 16 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 16 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 16 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 17 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 17 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 17 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 17 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 17 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 17 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 17 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 17 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 17 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 17 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 17 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 17 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 17 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 17 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 17 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 17 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 18 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 18 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 18 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 18 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 18 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 18 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 18 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 18 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 18 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 18 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 18 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 18 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 18 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 18 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 18 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 18 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 19 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 19 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 19 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 19 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 19 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 19 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 19 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 19 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 19 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 19 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 19 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 19 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 19 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 19 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 19 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 19 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 20 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 20 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 20 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 20 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 20 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 20 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 20 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 20 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 20 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 20 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 20 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 20 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 20 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 20 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 20 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 20 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 21 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 21 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 21 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 21 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 21 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 21 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 21 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 21 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 21 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 21 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 21 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 21 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 21 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 21 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 21 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 21 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 22 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 22 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 22 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 22 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 22 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 22 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 22 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 22 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 22 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 22 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 22 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 22 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 22 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 22 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 22 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 22 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 23 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 23 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 23 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 23 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 23 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 23 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 23 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 23 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 23 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 23 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 23 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 23 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 23 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 23 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 23 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 23 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 24 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 24 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 24 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 24 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 24 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 24 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 24 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 24 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 24 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 24 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 24 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 24 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 24 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 24 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 24 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 24 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 25 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 25 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 25 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 25 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 25 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 25 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 25 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 25 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 25 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 25 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 25 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 25 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 25 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 25 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 25 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 25 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 26 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 26 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 26 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 26 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 26 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 26 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 26 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 26 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 26 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 26 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 26 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 26 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 26 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 26 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 26 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 26 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 27 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 27 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 27 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 27 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 27 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 27 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 27 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 27 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 27 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 27 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 27 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 27 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 27 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 27 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 27 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 27 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 28 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 28 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 28 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 28 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 28 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 28 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 28 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 28 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 28 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 28 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 28 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 28 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 28 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 28 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 28 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 28 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 29 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 29 / baseline / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 29 / featcopilot-predictive / catboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 29 / featcopilot-predictive / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 29 / baseline / lightgbm: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 29 / baseline / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 29 / featcopilot-predictive / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 29 / featcopilot-predictive / catboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 29 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 29 / featcopilot-predictive / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 29 / baseline / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 29 / baseline / xgboost: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 29 / baseline / lightgbm: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 29 / baseline / linear: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical
- tabarena-46916 / 2026 / fold 29 / featcopilot-predictive / linear: dataset_error: AdmissionError: avgAge: nonnumeric predictor must be explicitly declared categorical
- tabarena-46937 / 2026 / fold 29 / baseline / xgboost: dataset_error: AdmissionError: coupon: nonnumeric predictor must be explicitly declared categorical

Raw cell metrics and provenance: local `results.json`. Completed runs also write `archive.json`,
which lists the authoritative lossless gzip parts and their uncompressed checksum.
