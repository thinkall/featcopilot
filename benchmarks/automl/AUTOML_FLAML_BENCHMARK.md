# AutoML Benchmark Report — FLAML

**Generated:** 2026-02-26 12:42:20
**Framework:** FLAML
**Time Budget:** 120s per model
**LLM Enabled:** False
**Datasets:** 10

## Summary

| Metric | Value |
|--------|-------|
| Total Datasets | 10 |
| Datasets Improved | 9/10 (90%) |
| Datasets Hurt | 1/10 (10%) |
| **Avg Improvement** | **+1.85%** |
| Max Improvement | +6.67% |
| Min Improvement | -0.92% |
| Avg FE Time | 2.9s |

## Classification Results

| Dataset | Baseline Accuracy | FeatCopilot Accuracy | Improvement | Features |
|---------|---------------|---------------|-------------|----------|
| titanic | 0.8156 | 0.8268 | +1.37% | 7→8 |
| credit_risk | 0.8500 | 0.8550 | +0.59% | 10→17 |
| complex_classification | 0.7875 | 0.8400 | +6.67% 🔥 | 15→23 |
| interaction_classification | 0.8025 | 0.8100 | +0.93% | 12→17 |
| xor_classification | 0.8180 | 0.8640 | +5.62% 🔥 | 20→24 |

## Regression Results

| Dataset | Baseline R² | FeatCopilot R² | Improvement | Features |
|---------|---------------|---------------|-------------|----------|
| house_prices | 0.9963 | 0.9972 | +0.09% | 14→16 |
| wine_quality | 0.5100 | 0.5110 | +0.20% | 11→13 |
| complex_regression | 0.8704 | 0.8790 | +0.99% | 15→20 |
| polynomial_regression | 0.9103 | 0.9375 | +2.99% 🔥 | 12→19 |
| sqrt_log_regression | 0.9322 | 0.9237 | -0.92% | 15→25 |
