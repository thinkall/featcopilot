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
    "C:\\code\\featcopilot\\benchmarks\\suites\\tabarena_confirmation_completion_repair_20260921.json"
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
| real_world / confirmation | classification | catboost | baseline | 2/2 / 2/2 | 60/60 | 0/2/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | classification | catboost | featcopilot-predictive | 2/2 / 2/2 | 60/60 | 0/2/0 | +0.01% | [+0.00%, +0.02%] |
| real_world / confirmation | classification | lightgbm | baseline | 2/2 / 2/2 | 60/60 | 0/2/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | classification | lightgbm | featcopilot-predictive | 2/2 / 2/2 | 60/60 | 0/2/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | classification | linear | baseline | 2/2 / 2/2 | 60/60 | 0/2/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | classification | linear | featcopilot-predictive | 2/2 / 2/2 | 60/60 | 0/2/0 | +0.03% | [+0.01%, +0.05%] |
| real_world / confirmation | classification | xgboost | baseline | 2/2 / 2/2 | 60/60 | 0/2/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | classification | xgboost | featcopilot-predictive | 2/2 / 2/2 | 60/60 | 0/2/0 | +0.02% | [-0.00%, +0.04%] |

## Per-dataset means (complete folds only)

| Dataset | Learner | Method | Primary loss | FE seconds (inner + outer) | Pipeline seconds | Features |
|---|---|---|---|---|---|---|
| tabarena-46916 | catboost | baseline | 0.202239 | 0.28 | 51.46 | 85.0 |
| tabarena-46916 | catboost | featcopilot-predictive | 0.202208 | 92.68 | 148.45 | 89.5 |
| tabarena-46916 | lightgbm | baseline | 0.203751 | 0.28 | 2.21 | 85.0 |
| tabarena-46916 | lightgbm | featcopilot-predictive | 0.203751 | 50.80 | 52.26 | 85.2 |
| tabarena-46916 | linear | baseline | 0.206124 | 0.29 | 1.00 | 85.0 |
| tabarena-46916 | linear | featcopilot-predictive | 0.206021 | 90.60 | 91.48 | 86.1 |
| tabarena-46916 | xgboost | baseline | 0.202848 | 0.29 | 1.76 | 85.0 |
| tabarena-46916 | xgboost | featcopilot-predictive | 0.202853 | 51.05 | 52.52 | 85.3 |
| tabarena-46937 | catboost | baseline | 0.502203 | 0.25 | 86.88 | 24.0 |
| tabarena-46937 | catboost | featcopilot-predictive | 0.502193 | 71.93 | 157.63 | 24.0 |
| tabarena-46937 | lightgbm | baseline | 0.506696 | 0.26 | 2.40 | 24.0 |
| tabarena-46937 | lightgbm | featcopilot-predictive | 0.506696 | 22.22 | 23.78 | 24.0 |
| tabarena-46937 | linear | baseline | 0.596617 | 0.26 | 0.99 | 24.0 |
| tabarena-46937 | linear | featcopilot-predictive | 0.596534 | 81.13 | 81.92 | 24.4 |
| tabarena-46937 | xgboost | baseline | 0.506252 | 0.26 | 2.22 | 24.0 |
| tabarena-46937 | xgboost | featcopilot-predictive | 0.506048 | 25.43 | 27.26 | 24.7 |

## Failures / timeouts

None.

Raw cell metrics and provenance: local `results.json`. Completed runs also write `archive.json`,
which lists the authoritative lossless gzip parts and their uncompressed checksum.
