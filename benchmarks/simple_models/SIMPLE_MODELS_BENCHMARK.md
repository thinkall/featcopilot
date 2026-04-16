# Simple Models Benchmark Report

**Generated:** 2026-04-16 16:14:06
**Models:** RandomForest, LogisticRegression/Ridge
**Cross-Validation:** 5-fold CV × 1 seed(s)
**LLM Enabled:** False
**Datasets:** 7 (5 real-world, 2 synthetic)

## Summary — Real-World Datasets (Primary)

| Metric | Value |
|--------|-------|
| Total Datasets | 5 |
| Win / Tie / Loss | 0 / 5 / 0 |
| Significant Wins (p<0.05) | 0 |
| Mean Improvement | -0.11% |
| Median Improvement | -0.01% |
| Max Regression | -0.32% |

## Summary — Synthetic Datasets (Supplementary)

| Metric | Value |
|--------|-------|
| Total Datasets | 2 |
| Win / Tie / Loss | 1 / 1 / 0 |
| Mean Improvement | +0.78% |

## Summary — All Datasets

| Metric | Value |
|--------|-------|
| Total Datasets | 7 |
| Win / Tie / Loss | 1 / 6 / 0 |
| Significant Wins (p<0.05) | 0 |
| Mean Improvement | +0.15% |
| Median Improvement | -0.00% |

## Real-World Classification

| Dataset | Baseline Score | FeatCopilot Score | Δ% | p-value | Sig | Features |
|---------|----------------|----------------|-----|---------|-----|----------|
| covertype | 0.8596±0.0044 | 0.8605±0.0046 | +0.11% | 0.438 |  | 10→10 |
| diabetes | 0.6016±0.0027 | 0.6016±0.0028 | -0.01% | 1.000 |  | 7→7 |
| credit | 0.7730±0.0055 | 0.7706±0.0073 | -0.31% | 0.188 |  | 10→10 |
| electricity | 0.8977±0.0018 | 0.8948±0.0022 | -0.32% | 0.062 |  | 8→10 |

## Real-World Regression

| Dataset | Baseline R² | FeatCopilot R² | Δ% | p-value | Sig | Features |
|---------|----------------|----------------|-----|---------|-----|----------|
| diamonds | 0.9439±0.0010 | 0.9439±0.0010 | -0.00% | 1.000 |  | 6→6 |

## Synthetic Classification (Supplementary)

| Dataset | Baseline Score | FeatCopilot Score | Δ% | p-value | Sig | Features |
|---------|----------------|----------------|-----|---------|-----|----------|
| titanic | 0.8193±0.0116 | 0.8193±0.0116 | +0.00% | 1.000 |  | 7→7 |

## Synthetic Regression (Supplementary)

| Dataset | Baseline R² | FeatCopilot R² | Δ% | p-value | Sig | Features |
|---------|----------------|----------------|-----|---------|-----|----------|
| house_prices | 0.9802±0.0030 | 0.9956±0.0004 | +1.57% | 0.062 |  | 14→16 |
