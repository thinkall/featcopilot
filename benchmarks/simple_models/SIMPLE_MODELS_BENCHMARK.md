# Simple Models Benchmark Report

**Generated:** 2026-04-16 15:01:44
**Models:** RandomForest, LogisticRegression/Ridge
**Cross-Validation:** 5-fold CV × 1 seed(s)
**LLM Enabled:** False
**Datasets:** 4 (2 real-world, 2 synthetic)

## Summary — Real-World Datasets (Primary)

| Metric | Value |
|--------|-------|
| Total Datasets | 2 |
| Win / Tie / Loss | 0 / 1 / 1 |
| Significant Wins (p<0.05) | 0 |
| Mean Improvement | -1.86% |
| Median Improvement | -1.86% |
| Max Regression | -3.24% |

## Summary — Synthetic Datasets (Supplementary)

| Metric | Value |
|--------|-------|
| Total Datasets | 2 |
| Win / Tie / Loss | 1 / 1 / 0 |
| Mean Improvement | +0.92% |

## Summary — All Datasets

| Metric | Value |
|--------|-------|
| Total Datasets | 4 |
| Win / Tie / Loss | 1 / 2 / 1 |
| Significant Wins (p<0.05) | 0 |
| Mean Improvement | -0.47% |
| Median Improvement | -0.10% |

## Real-World Classification

| Dataset | Baseline Score | FeatCopilot Score | Δ% | p-value | Sig | Features |
|---------|----------------|----------------|-----|---------|-----|----------|
| credit | 0.7730±0.0055 | 0.7479±0.0109 | -3.24% 🔴 | 0.062 |  | 10→11 |

## Real-World Regression

| Dataset | Baseline R² | FeatCopilot R² | Δ% | p-value | Sig | Features |
|---------|----------------|----------------|-----|---------|-----|----------|
| diamonds | 0.9439±0.0010 | 0.9394±0.0008 | -0.48% | 0.062 |  | 6→4 |

## Synthetic Classification (Supplementary)

| Dataset | Baseline Score | FeatCopilot Score | Δ% | p-value | Sig | Features |
|---------|----------------|----------------|-----|---------|-----|----------|
| titanic | 0.8193±0.0116 | 0.8215±0.0132 | +0.27% | 0.750 |  | 7→8 |

## Synthetic Regression (Supplementary)

| Dataset | Baseline R² | FeatCopilot R² | Δ% | p-value | Sig | Features |
|---------|----------------|----------------|-----|---------|-----|----------|
| house_prices | 0.9802±0.0030 | 0.9956±0.0004 | +1.57% | 0.062 |  | 14→16 |
