# AutoML Benchmark Report — AUTOGLUON

**Generated:** 2026-02-26 12:42:20
**Framework:** AUTOGLUON
**Time Budget:** 37s per model
**LLM Enabled:** False
**Datasets:** 10

## Summary

| Metric | Value |
|--------|-------|
| Total Datasets | 10 |
| Datasets Improved | 9/10 (90%) |
| Datasets Hurt | 1/10 (10%) |
| **Avg Improvement** | **+1.55%** |
| Max Improvement | +7.62% |
| Min Improvement | -1.53% |
| Avg FE Time | 2.9s |

## Classification Results

| Dataset | Baseline Accuracy | FeatCopilot Accuracy | Improvement | Features |
|---------|---------------|---------------|-------------|----------|
| titanic | 0.8212 | 0.8324 | +1.36% | 7→8 |
| credit_risk | 0.8525 | 0.8650 | +1.47% | 10→17 |
| complex_classification | 0.7550 | 0.8125 | +7.62% 🔥 | 15→23 |
| interaction_classification | 0.8000 | 0.8025 | +0.31% | 12→17 |
| xor_classification | 0.8280 | 0.8480 | +2.42% 🔥 | 20→24 |

## Regression Results

| Dataset | Baseline R² | FeatCopilot R² | Improvement | Features |
|---------|---------------|---------------|-------------|----------|
| house_prices | 0.9969 | 0.9975 | +0.06% | 14→16 |
| wine_quality | 0.5112 | 0.5034 | -1.53% ⚠️ | 11→13 |
| complex_regression | 0.9166 | 0.9401 | +2.57% 🔥 | 15→20 |
| polynomial_regression | 0.9404 | 0.9492 | +0.94% | 12→19 |
| sqrt_log_regression | 0.9458 | 0.9482 | +0.26% | 15→25 |
