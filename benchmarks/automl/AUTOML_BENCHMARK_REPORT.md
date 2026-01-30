# AutoML Integration Benchmark Report
## Overview
This benchmark evaluates FeatCopilot's impact on AutoML framework performance.
## Summary
- **Datasets tested**: 9
- **Frameworks tested**: 1
- **Average improvement**: -1.07%
- **Positive improvements**: 1/9 (11.1%)

## Results by Framework

### FLAML
| Dataset | Task | Baseline | +FeatCopilot | Improvement |
|---------|------|----------|--------------|-------------|
| Titanic (Kaggle-style) | classification | 0.9665 | 0.9665 | +0.00% |
| House Prices (Kaggle-style) | regression | 0.9195 | 0.9136 | -0.65% |
| Credit Card Fraud (Kaggle-style) | classification | 0.9860 | 0.9860 | +0.00% |
| Bike Sharing (Kaggle-style) | regression | 0.8426 | 0.8359 | -0.79% |
| Employee Attrition (IBM HR) | classification | 0.9796 | 0.9762 | -0.35% |
| Credit Risk (synthetic) | classification | 0.7000 | 0.6450 | -7.86% |
| Medical Diagnosis (synthetic) | classification | 0.8567 | 0.8600 | +0.39% |
| Complex Regression (synthetic) | regression | 0.9797 | 0.9757 | -0.41% |
| Complex Classification (synthetic) | classification | 0.9450 | 0.9450 | +0.00% |

**Average improvement with flaml**: -1.07%

## Detailed Results
| Dataset | Framework | Baseline | +FeatCopilot | Improvement | Features | Time (s) |
|---------|-----------|----------|--------------|-------------|----------|----------|
| Titanic (Kaggle-style) | flaml | 0.9665 | 0.9665 | +0.00% | 7->27 | 30.7 |
| House Prices (Kaggle-style) | flaml | 0.9195 | 0.9136 | -0.65% | 14->50 | 31.1 |
| Credit Card Fraud (Kaggle-style) | flaml | 0.9860 | 0.9860 | +0.00% | 30->50 | 32.0 |
| Bike Sharing (Kaggle-style) | flaml | 0.8426 | 0.8359 | -0.79% | 10->41 | 32.0 |
| Employee Attrition (IBM HR) | flaml | 0.9796 | 0.9762 | -0.35% | 11->41 | 31.7 |
| Credit Risk (synthetic) | flaml | 0.7000 | 0.6450 | -7.86% | 10->50 | 31.1 |
| Medical Diagnosis (synthetic) | flaml | 0.8567 | 0.8600 | +0.39% | 12->48 | 31.2 |
| Complex Regression (synthetic) | flaml | 0.9797 | 0.9757 | -0.41% | 15->50 | 35.3 |
| Complex Classification (synthetic) | flaml | 0.9450 | 0.9450 | +0.00% | 15->50 | 33.6 |
