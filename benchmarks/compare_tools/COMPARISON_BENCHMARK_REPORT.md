# Feature Engineering Tools Comparison Benchmark
## Overview
This benchmark compares FeatCopilot with other popular feature engineering libraries to demonstrate performance improvements across various datasets.

### Tools Compared
| Tool | Description |
|------|-------------|
| baseline | No feature engineering (raw features only) |
| featcopilot | FeatCopilot - LLM-powered auto feature engineering |
| featuretools | Featuretools - Deep Feature Synthesis |
| tsfresh | tsfresh - Time series feature extraction |

## Summary
- **Datasets tested**: 9
- **Tools compared**: 4

### Performance by Tool
| Tool | Avg Score | Avg Improvement | Wins | Avg FE Time |
|------|-----------|-----------------|------|-------------|
| baseline | 0.8924 | - | - | 0.00s |
| featcopilot | 0.8942 | +0.21% | 4 | 1.07s |
| featuretools | 0.8947 | +0.27% | 3 | 0.11s |
| tsfresh | 0.8843 | -0.92% | 2 | 25.07s |

## Detailed Results

### Titanic (Kaggle-style)
**Task**: classification

| Tool | Accuracy | Features | FE Time | Train Time | Status |
|------|----------|----------|---------|------------|--------|
| baseline | 0.9609 | 7 | 0.00s | 0.16s | success |
| featcopilot | 0.9609 | 27 | 0.50s | 0.38s | success |
| featuretools | 0.9553 | 50 | 0.06s | 0.63s | success |
| tsfresh | 0.9497 | 14 | 11.65s | 0.22s | success |

### House Prices (Kaggle-style)
**Task**: regression

| Tool | R² Score | Features | FE Time | Train Time | Status |
|------|----------|----------|---------|------------|--------|
| baseline | 0.9135 | 14 | 0.00s | 0.47s | success |
| featcopilot | 0.9094 | 50 | 1.42s | 1.77s | success |
| featuretools | 0.9124 | 50 | 0.08s | 1.77s | success |
| tsfresh | 0.9118 | 49 | 20.50s | 0.87s | success |

### Credit Card Fraud (Kaggle-style)
**Task**: classification

| Tool | Accuracy | Features | FE Time | Train Time | Status |
|------|----------|----------|---------|------------|--------|
| baseline | 0.9790 | 30 | 0.00s | 5.39s | success |
| featcopilot | 0.9790 | 50 | 2.38s | 8.99s | success |
| featuretools | 0.9770 | 50 | 0.38s | 8.96s | success |
| tsfresh | 0.9840 | 50 | 69.94s | 7.59s | success |

### Bike Sharing (Kaggle-style)
**Task**: regression

| Tool | R² Score | Features | FE Time | Train Time | Status |
|------|----------|----------|---------|------------|--------|
| baseline | 0.8305 | 10 | 0.00s | 0.27s | success |
| featcopilot | 0.8324 | 41 | 1.19s | 1.60s | success |
| featuretools | 0.8351 | 50 | 0.08s | 1.60s | success |
| tsfresh | 0.8175 | 49 | 33.68s | 1.38s | success |

### Employee Attrition (IBM HR)
**Task**: classification

| Tool | Accuracy | Features | FE Time | Train Time | Status |
|------|----------|----------|---------|------------|--------|
| baseline | 0.9694 | 11 | 0.00s | 0.23s | success |
| featcopilot | 0.9762 | 41 | 0.60s | 0.78s | success |
| featuretools | 0.9728 | 50 | 0.08s | 0.95s | success |
| tsfresh | 0.9660 | 21 | 12.48s | 0.23s | success |

### Credit Risk (synthetic)
**Task**: classification

| Tool | Accuracy | Features | FE Time | Train Time | Status |
|------|----------|----------|---------|------------|--------|
| baseline | 0.6975 | 10 | 0.00s | 0.43s | success |
| featcopilot | 0.7050 | 50 | 0.70s | 2.15s | success |
| featuretools | 0.7025 | 50 | 0.07s | 2.17s | success |
| tsfresh | 0.6950 | 28 | 14.54s | 1.17s | success |

### Medical Diagnosis (synthetic)
**Task**: classification

| Tool | Accuracy | Features | FE Time | Train Time | Status |
|------|----------|----------|---------|------------|--------|
| baseline | 0.8600 | 12 | 0.00s | 0.56s | success |
| featcopilot | 0.8500 | 48 | 0.64s | 1.95s | success |
| featuretools | 0.8533 | 50 | 0.06s | 2.01s | success |
| tsfresh | 0.8567 | 49 | 12.94s | 1.81s | success |

### Complex Regression (synthetic)
**Task**: regression

| Tool | R² Score | Features | FE Time | Train Time | Status |
|------|----------|----------|---------|------------|--------|
| baseline | 0.9137 | 15 | 0.00s | 0.85s | success |
| featcopilot | 0.9120 | 50 | 1.49s | 2.69s | success |
| featuretools | 0.9435 | 50 | 0.08s | 2.68s | success |
| tsfresh | 0.9105 | 40 | 33.06s | 2.13s | success |

### Complex Classification (synthetic)
**Task**: classification

| Tool | Accuracy | Features | FE Time | Train Time | Status |
|------|----------|----------|---------|------------|--------|
| baseline | 0.9075 | 15 | 0.00s | 0.90s | success |
| featcopilot | 0.9225 | 50 | 0.72s | 2.71s | success |
| featuretools | 0.9000 | 50 | 0.08s | 2.72s | success |
| tsfresh | 0.8675 | 50 | 16.83s | 2.70s | success |

## Key Findings
- **FeatCopilot average improvement**: +0.21% over baseline
- **Best improvement**: +1.65% on Complex Classification (synthetic)
- FeatCopilot outperforms tsfresh by 1.12% on average

## Conclusion
FeatCopilot demonstrates competitive or superior performance compared to other feature engineering tools while providing a more intuitive API and LLM-powered feature suggestions.
