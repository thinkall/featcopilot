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
  "datasets": [
    "uci-online-shoppers",
    "uci-concrete-strength"
  ],
  "folds": 3,
  "seeds": [
    42,
    137
  ],
  "max_rows": 5000,
  "max_added_features": 50,
  "max_candidates": 1000,
  "threads": 8,
  "memory_limit_mb": 4096,
  "case_timeout": 300.0,
  "synthetic": false,
  "stability_repeats": 3,
  "screening": "auto",
  "allow_confirmation": false,
  "detailed_diagnostics": false,
  "live_llm": true,
  "llm_ledger": "C:\\code\\featcopilot\\benchmarks\\.cache\\llm-evaluation-budget-100",
  "allow_soft_llm_tokens": true,
  "methods": [
    "baseline",
    "featcopilot-predictive",
    "llm-plan",
    "llm-hybrid"
  ],
  "learners": [
    "linear",
    "lightgbm"
  ],
  "suite": [
    "C:\\code\\featcopilot\\benchmarks\\suites\\uci_development.json"
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
| real_world / development | classification | lightgbm | baseline | 1/1 / 1/1 | 6/6 | 0/1/0 | +0.00% | n/a |
| real_world / development | classification | lightgbm | featcopilot-predictive | 1/1 / 1/1 | 6/6 | 0/1/0 | +0.00% | n/a |
| real_world / development | classification | lightgbm | llm-hybrid | 0/1 / 0/1 | 5/6 | 0/0/0 | n/a | n/a |
| real_world / development | classification | lightgbm | llm-plan | 1/1 / 1/1 | 6/6 | 0/1/0 | -0.03% | n/a |
| real_world / development | classification | linear | baseline | 1/1 / 1/1 | 6/6 | 0/1/0 | +0.00% | n/a |
| real_world / development | classification | linear | featcopilot-predictive | 1/1 / 1/1 | 6/6 | 1/0/0 | +2.21% | n/a |
| real_world / development | classification | linear | llm-hybrid | 0/1 / 0/1 | 5/6 | 0/0/0 | n/a | n/a |
| real_world / development | classification | linear | llm-plan | 1/1 / 1/1 | 6/6 | 1/0/0 | +2.77% | n/a |
| real_world / development | regression | lightgbm | baseline | 1/1 / 1/1 | 6/6 | 0/1/0 | +0.00% | n/a |
| real_world / development | regression | lightgbm | featcopilot-predictive | 1/1 / 1/1 | 6/6 | 0/1/0 | -0.46% | n/a |
| real_world / development | regression | lightgbm | llm-hybrid | 1/1 / 1/1 | 6/6 | 0/0/1 | -1.31% | n/a |
| real_world / development | regression | lightgbm | llm-plan | 1/1 / 1/1 | 6/6 | 0/1/0 | -0.68% | n/a |
| real_world / development | regression | linear | baseline | 1/1 / 1/1 | 6/6 | 0/1/0 | +0.00% | n/a |
| real_world / development | regression | linear | featcopilot-predictive | 1/1 / 1/1 | 6/6 | 1/0/0 | +10.75% | n/a |
| real_world / development | regression | linear | llm-hybrid | 1/1 / 1/1 | 6/6 | 1/0/0 | +33.67% | n/a |
| real_world / development | regression | linear | llm-plan | 1/1 / 1/1 | 6/6 | 1/0/0 | +34.55% | n/a |

## Per-dataset means (complete folds only)

| Dataset | Learner | Method | Primary loss | FE seconds (inner + outer) | Pipeline seconds | Features |
|---|---|---|---|---|---|---|
| uci-concrete-strength | lightgbm | baseline | 4.553248 | 0.02 | 1.07 | 8.0 |
| uci-concrete-strength | lightgbm | featcopilot-predictive | 4.574290 | 12.17 | 12.65 | 16.3 |
| uci-concrete-strength | lightgbm | llm-hybrid | 4.612807 | 57.15 | 57.61 | 24.3 |
| uci-concrete-strength | lightgbm | llm-plan | 4.584298 | 46.17 | 47.13 | 16.0 |
| uci-concrete-strength | linear | baseline | 10.536313 | 0.02 | 0.07 | 8.0 |
| uci-concrete-strength | linear | featcopilot-predictive | 9.403466 | 2.81 | 2.86 | 35.0 |
| uci-concrete-strength | linear | llm-hybrid | 6.989135 | 51.50 | 51.56 | 43.0 |
| uci-concrete-strength | linear | llm-plan | 6.895543 | 47.40 | 47.45 | 16.0 |
| uci-online-shoppers | lightgbm | baseline | 0.379942 | 0.06 | 1.41 | 11.0 |
| uci-online-shoppers | lightgbm | featcopilot-predictive | 0.379942 | 16.95 | 17.72 | 11.0 |
| uci-online-shoppers | lightgbm | llm-plan | 0.380054 | 47.18 | 48.49 | 19.0 |
| uci-online-shoppers | linear | baseline | 0.394934 | 0.05 | 0.32 | 11.0 |
| uci-online-shoppers | linear | featcopilot-predictive | 0.386222 | 65.86 | 66.14 | 12.7 |
| uci-online-shoppers | linear | llm-plan | 0.383985 | 45.70 | 45.99 | 19.0 |

## Failures / timeouts

- uci-online-shoppers / 42 / fold 0 / llm-hybrid / lightgbm: error: ValueError: Invalid fitted rule plan; response content is omitted from diagnostics
- uci-online-shoppers / 42 / fold 2 / llm-hybrid / linear: error: ValueError: Deterministic and plan feature names collide; explicit renaming is required

Raw cell metrics and provenance: local `results.json`. Completed runs also write `archive.json`,
which lists the authoritative lossless gzip parts and their uncompressed checksum.
