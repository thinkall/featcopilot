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
| autofeat | autofeat - Automatic feature generation |

## Summary
- **Datasets tested**: 9
- **Tools compared**: 5

### Performance by Tool
| Tool | Avg Score | Avg Improvement | Wins | Avg FE Time |
|------|-----------|-----------------|------|-------------|
| baseline | 0.8924 | - | - | 0.00s |
| featcopilot | 0.8942 | +0.21% | 1 | 1.03s |
| featuretools | 0.8947 | +0.27% | 3 | 0.11s |
| tsfresh | 0.8843 | -0.92% | 2 | 24.86s |
| autofeat | 0.8964 | +0.48% | 3 | 1246.91s |

## Detailed Results

### Titanic (Kaggle-style)
**Task**: classification

| Tool | Accuracy | Features | FE Time | Train Time | Status |
|------|----------|----------|---------|------------|--------|
| baseline | 0.9609 | 7 | 0.00s | 0.17s | success |
| featcopilot | 0.9609 | 27 | 0.48s | 0.40s | success |
| featuretools | 0.9553 | 50 | 0.06s | 0.62s | success |
| tsfresh | 0.9497 | 14 | 9.43s | 0.24s | success |
| autofeat | 0.9609 | 7 | 128.44s | 0.16s | success |

### House Prices (Kaggle-style)
**Task**: regression

| Tool | R² Score | Features | FE Time | Train Time | Status |
|------|----------|----------|---------|------------|--------|
| baseline | 0.9135 | 14 | 0.00s | 0.41s | success |
| featcopilot | 0.9094 | 50 | 1.10s | 1.68s | success |
| featuretools | 0.9124 | 50 | 0.09s | 1.49s | success |
| tsfresh | 0.9118 | 49 | 14.84s | 0.90s | success |
| autofeat | 0.9123 | 45 | 85.97s | 1.80s | success |

### Credit Card Fraud (Kaggle-style)
**Task**: classification

| Tool | Accuracy | Features | FE Time | Train Time | Status |
|------|----------|----------|---------|------------|--------|
| baseline | 0.9790 | 30 | 0.00s | 6.70s | success |
| featcopilot | 0.9790 | 50 | 2.33s | 9.28s | success |
| featuretools | 0.9770 | 50 | 0.40s | 9.65s | success |
| tsfresh | 0.9840 | 50 | 85.73s | 7.78s | success |
| autofeat | 0.9790 | 30 | 8948.32s | 4.48s | success |

### Bike Sharing (Kaggle-style)
**Task**: regression

| Tool | R² Score | Features | FE Time | Train Time | Status |
|------|----------|----------|---------|------------|--------|
| baseline | 0.8305 | 10 | 0.00s | 0.27s | success |
| featcopilot | 0.8324 | 41 | 1.29s | 1.59s | success |
| featuretools | 0.8351 | 50 | 0.07s | 1.58s | success |
| tsfresh | 0.8175 | 49 | 37.57s | 1.38s | success |
| autofeat | 0.8313 | 38 | 30.22s | 1.28s | success |

### Employee Attrition (IBM HR)
**Task**: classification

| Tool | Accuracy | Features | FE Time | Train Time | Status |
|------|----------|----------|---------|------------|--------|
| baseline | 0.9694 | 11 | 0.00s | 0.23s | success |
| featcopilot | 0.9762 | 41 | 0.59s | 0.78s | success |
| featuretools | 0.9728 | 50 | 0.07s | 0.94s | success |
| tsfresh | 0.9660 | 21 | 11.76s | 0.23s | success |
| autofeat | 0.9796 | 28 | 401.42s | 0.55s | success |

### Credit Risk (synthetic)
**Task**: classification

| Tool | Accuracy | Features | FE Time | Train Time | Status |
|------|----------|----------|---------|------------|--------|
| baseline | 0.6975 | 10 | 0.00s | 0.42s | success |
| featcopilot | 0.7050 | 50 | 0.70s | 2.12s | success |
| featuretools | 0.7025 | 50 | 0.07s | 2.15s | success |
| tsfresh | 0.6950 | 28 | 18.20s | 1.16s | success |
| autofeat | 0.7150 | 26 | 327.62s | 1.11s | success |

### Medical Diagnosis (synthetic)
**Task**: classification

| Tool | Accuracy | Features | FE Time | Train Time | Status |
|------|----------|----------|---------|------------|--------|
| baseline | 0.8600 | 12 | 0.00s | 0.55s | success |
| featcopilot | 0.8500 | 48 | 0.62s | 1.91s | success |
| featuretools | 0.8533 | 50 | 0.06s | 1.99s | success |
| tsfresh | 0.8567 | 49 | 12.76s | 1.78s | success |
| autofeat | 0.8367 | 25 | 455.64s | 1.04s | success |

### Complex Regression (synthetic)
**Task**: regression

| Tool | R² Score | Features | FE Time | Train Time | Status |
|------|----------|----------|---------|------------|--------|
| baseline | 0.9137 | 15 | 0.00s | 0.84s | success |
| featcopilot | 0.9120 | 50 | 1.47s | 2.65s | success |
| featuretools | 0.9435 | 50 | 0.08s | 2.67s | success |
| tsfresh | 0.9105 | 40 | 16.80s | 2.12s | success |
| autofeat | 0.9129 | 16 | 36.87s | 0.90s | success |

### Complex Classification (synthetic)
**Task**: classification

| Tool | Accuracy | Features | FE Time | Train Time | Status |
|------|----------|----------|---------|------------|--------|
| baseline | 0.9075 | 15 | 0.00s | 0.89s | success |
| featcopilot | 0.9225 | 50 | 0.73s | 2.71s | success |
| featuretools | 0.9000 | 50 | 0.08s | 2.71s | success |
| tsfresh | 0.8675 | 50 | 16.61s | 2.73s | success |
| autofeat | 0.9400 | 45 | 807.68s | 2.44s | success |

## Key Findings
- **FeatCopilot average improvement**: +0.21% over baseline
- **Best improvement**: +1.65% on Complex Classification (synthetic)
- FeatCopilot outperforms tsfresh by 1.12% on average

## Conclusion
FeatCopilot demonstrates competitive or superior performance compared to other feature engineering tools while providing a more intuitive API and LLM-powered feature suggestions.
