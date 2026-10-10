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
    42
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
  "allow_confirmation": false,
  "detailed_diagnostics": false,
  "live_llm": false,
  "llm_ledger": null,
  "allow_soft_llm_tokens": false,
  "methods": [
    "baseline",
    "featcopilot-predictive",
    "polynomial"
  ],
  "learners": [
    "linear",
    "lightgbm"
  ],
  "suite": [
    "C:\\code\\featcopilot\\benchmarks\\suites\\seoul_development.json"
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
| real_world / development | regression | lightgbm | baseline | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| real_world / development | regression | lightgbm | featcopilot-predictive | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| real_world / development | regression | lightgbm | polynomial | 1/1 / 1/1 | 3/3 | 1/0/0 | +8.43% | n/a |
| real_world / development | regression | linear | baseline | 1/1 / 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| real_world / development | regression | linear | featcopilot-predictive | 1/1 / 1/1 | 3/3 | 1/0/0 | +3.71% | n/a |
| real_world / development | regression | linear | polynomial | 1/1 / 1/1 | 3/3 | 0/1/0 | -0.74% | n/a |

## Per-dataset means (complete folds only)

| Dataset | Learner | Method | Primary loss | FE seconds (inner + outer) | Pipeline seconds | Features |
|---|---|---|---|---|---|---|
| uci-seoul-bike | lightgbm | baseline | 545.977023 | 0.04 | 1.25 | 11.0 |
| uci-seoul-bike | lightgbm | featcopilot-predictive | 545.977023 | 20.94 | 21.66 | 11.0 |
| uci-seoul-bike | lightgbm | polynomial | 499.935186 | 0.07 | 1.73 | 56.0 |
| uci-seoul-bike | linear | baseline | 578.661396 | 0.04 | 0.13 | 11.0 |
| uci-seoul-bike | linear | featcopilot-predictive | 557.173675 | 3.58 | 3.72 | 11.7 |
| uci-seoul-bike | linear | polynomial | 582.923389 | 0.07 | 0.24 | 56.0 |

## Failures / timeouts

None.

Raw cell metrics and provenance: local `results.json`. Completed runs also write `archive.json`,
which lists the authoritative lossless gzip parts and their uncompressed checksum.
