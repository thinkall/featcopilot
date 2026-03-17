# Use-Case Benchmarks

Targeted benchmarks for realistic feature-engineering scenarios.

## Auto Feature Engineering

This benchmark compares:
- plain baseline
- FeatCopilot
- Featuretools (if installed)
- autofeat (if installed)

on an interaction-heavy tabular classification task where automatic feature engineering should matter.

```bash
python -m benchmarks.use_cases.run_auto_feature_engineering_benchmark
```

Outputs:
- `AUTO_FEATURE_ENGINEERING_USE_CASE.md`
