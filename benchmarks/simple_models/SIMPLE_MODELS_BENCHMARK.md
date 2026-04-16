# Simple Models Benchmark Report

**Generated:** 2026-04-16 18:17:01
**Models:** RandomForest, LogisticRegression/Ridge
**Cross-Validation:** 5-fold CV × 1 seed(s)
**LLM Enabled:** False
**Datasets:** 5 (3 real-world, 2 synthetic)

## Summary — Real-World Datasets (Primary)

| Metric | Value |
|--------|-------|
| Total Datasets | 3 |
| Win / Tie / Loss | 1 / 1 / 1 |
| Significant Wins (p<0.05) | 0 |
| Mean Improvement | +0.84% |
| Median Improvement | +0.02% |
| Max Regression | -1.14% |

## Summary — Synthetic Datasets (Supplementary)

| Metric | Value |
|--------|-------|
| Total Datasets | 2 |
| Win / Tie / Loss | 2 / 0 / 0 |
| Mean Improvement | +19.24% |

## Summary — All Datasets

| Metric | Value |
|--------|-------|
| Total Datasets | 5 |
| Win / Tie / Loss | 3 / 1 / 1 |
| Significant Wins (p<0.05) | 0 |
| Mean Improvement | +8.20% |
| Median Improvement | +1.57% |

## Real-World Classification

| Dataset | Baseline Score | FeatCopilot Score | Δ% | p-value | Sig | Features |
|---------|----------------|----------------|-----|---------|-----|----------|
| eye_movements | 0.6442±0.0136 | 0.6676±0.0168 | +3.63% | 0.062 |  | 23→30 |
| bank_marketing | 0.8012±0.0090 | 0.8014±0.0086 | +0.02% | 1.000 |  | 7→7 |
| covertype_cat | 0.8734±0.0030 | 0.8634±0.0032 | -1.14% 🔴 | 0.062 |  | 54→58 |

## Synthetic Regression (Supplementary)

| Dataset | Baseline R² | FeatCopilot R² | Δ% | p-value | Sig | Features |
|---------|----------------|----------------|-----|---------|-----|----------|
| complex_regression | 0.6437±0.0265 | 0.8813±0.0181 | +36.91% | 0.062 |  | 15→20 |
| house_prices | 0.9802±0.0030 | 0.9956±0.0004 | +1.57% | 0.062 |  | 14→16 |

