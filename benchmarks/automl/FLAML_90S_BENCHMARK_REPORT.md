# AutoML Integration Benchmark Report
## Overview
This benchmark evaluates FeatCopilot's impact on AutoML framework performance.
Time budget: 90s per AutoML run (excludes FeatCopilot preprocessing time)

## Summary
- **Datasets tested**: 15
- **Frameworks tested**: 1
- **Average improvement**: -0.12%
- **Positive improvements**: 2/15 (13.3%)

## Results by Framework

### FLAML
| Dataset | Task | Baseline | +FeatCopilot | Improvement | Train Time (B/E) | Predict Time (B/E) |
|---------|------|----------|--------------|-------------|------------------|--------------------|
| Titanic (Kaggle-style) | classification | 0.9665 | 0.9665 | +0.00% | 90.0s / 90.2s | 0.02s / 0.05s |
| House Prices (Kaggle-style) | regression | 0.9242 | 0.9187 | -0.60% | 90.4s / 91.6s | 0.01s / 0.02s |
| Credit Card Fraud (Kaggle-style) | classification | 0.9860 | 0.9860 | +0.00% | 90.9s / 90.1s | 0.05s / 0.04s |
| Bike Sharing (Kaggle-style) | regression | 0.8455 | 0.8330 | -1.48% | 94.3s / 91.1s | 0.01s / 0.01s |
| Employee Attrition (IBM HR) | classification | 0.9762 | 0.9762 | +0.00% | 90.1s / 90.1s | 0.02s / 0.04s |
| Credit Risk (synthetic) | classification | 0.6975 | 0.6925 | -0.72% | 90.0s / 90.6s | 0.02s / 0.04s |
| Medical Diagnosis (synthetic) | classification | 0.8533 | 0.8533 | +0.00% | 91.0s / 90.5s | 0.01s / 0.05s |
| Complex Regression (synthetic) | regression | 0.9795 | 0.9757 | -0.39% | 94.5s / 93.9s | 0.01s / 0.01s |
| Complex Classification (synthetic) | classification | 0.9450 | 0.9425 | -0.26% | 91.4s / 99.9s | 0.01s / 0.02s |
| Product Reviews (text) | text_classification | 0.9425 | 0.9425 | +0.00% | 90.0s / 90.6s | 0.02s / 0.04s |
| Job Postings (text) | text_regression | 0.8949 | 0.8934 | -0.17% | 90.0s / 90.4s | 0.01s / 0.02s |
| News Headlines (text) | text_classification | 0.9800 | 0.9960 | +1.63% | 90.2s / 90.9s | 0.02s / 0.03s |
| Customer Support Tickets (text) | text_classification | 1.0000 | 1.0000 | +0.00% | 373.7s / 557.1s | 0.02s / 0.03s |
| Medical Notes (text) | text_classification | 0.9933 | 0.9933 | +0.00% | 90.3s / 90.2s | 0.01s / 0.03s |
| E-commerce Products (text) | text_regression | 0.4649 | 0.4658 | +0.20% | 90.4s / 90.1s | 0.01s / 0.01s |

**Average improvement with flaml**: -0.12%

## Detailed Results
| Dataset | Framework | Baseline | +FeatCopilot | Improvement | Features | FE Time | Train Time | Predict Time |
|---------|-----------|----------|--------------|-------------|----------|---------|------------|---------------|
| Titanic (Kaggle-style) | flaml | 0.9665 | 0.9665 | +0.00% | 7->27 | 0.5s | 90.2s | 0.05s |
| House Prices (Kaggle-style) | flaml | 0.9242 | 0.9187 | -0.60% | 14->50 | 1.1s | 91.6s | 0.02s |
| Credit Card Fraud (Kaggle-style) | flaml | 0.9860 | 0.9860 | +0.00% | 30->50 | 1.7s | 90.1s | 0.04s |
| Bike Sharing (Kaggle-style) | flaml | 0.8455 | 0.8330 | -1.48% | 10->41 | 1.2s | 91.1s | 0.01s |
| Employee Attrition (IBM HR) | flaml | 0.9762 | 0.9762 | +0.00% | 11->41 | 0.6s | 90.1s | 0.04s |
| Credit Risk (synthetic) | flaml | 0.6975 | 0.6925 | -0.72% | 10->50 | 0.7s | 90.6s | 0.04s |
| Medical Diagnosis (synthetic) | flaml | 0.8533 | 0.8533 | +0.00% | 12->48 | 0.6s | 90.5s | 0.05s |
| Complex Regression (synthetic) | flaml | 0.9795 | 0.9757 | -0.39% | 15->50 | 1.4s | 93.9s | 0.01s |
| Complex Classification (synthetic) | flaml | 0.9450 | 0.9425 | -0.26% | 15->50 | 0.8s | 99.9s | 0.02s |
| Product Reviews (text) | flaml | 0.9425 | 0.9425 | +0.00% | 6->38 | 0.7s | 90.6s | 0.04s |
| Job Postings (text) | flaml | 0.8949 | 0.8934 | -0.17% | 5->34 | 0.8s | 90.4s | 0.02s |
| News Headlines (text) | flaml | 0.9800 | 0.9960 | +1.63% | 5->19 | 0.7s | 90.9s | 0.03s |
| Customer Support Tickets (text) | flaml | 1.0000 | 1.0000 | +0.00% | 6->26 | 0.6s | 557.1s | 0.03s |
| Medical Notes (text) | flaml | 0.9933 | 0.9933 | +0.00% | 5->29 | 0.7s | 90.2s | 0.03s |
| E-commerce Products (text) | flaml | 0.4649 | 0.4658 | +0.20% | 5->26 | 1.3s | 90.1s | 0.01s |
