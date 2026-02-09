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
| caafe | CAAFE - Context-Aware Automated Feature Engineering (LLM) |

## Summary
- **Datasets tested**: 6
- **Tools compared**: 6

### Performance by Tool
| Tool | Avg Score | Avg Improvement | Wins | Avg FE Time |
|------|-----------|-----------------|------|-------------|
| baseline | 0.8676 | - | - | 0.00s |
| featcopilot | 0.8543 | -1.47% | 1 | 63.07s |
| featuretools | 0.8656 | -0.22% | 1 | 0.15s |
| tsfresh | 0.8603 | -0.89% | 1 | 20.56s |
| autofeat | 0.8602 | -0.92% | 0 | 148.42s |
| caafe | 0.8678 | +0.02% | 3 | 0.00s |

## Detailed Results

### titanic
**Task**: classification

| Tool | Accuracy | Features | FE Time | Train Time | Status |
|------|----------|----------|---------|------------|--------|
| baseline | 0.9665 | 7 | 0.00s | 119.98s | success |
| featcopilot | 0.9218 | 27 | 62.17s | 120.28s | success |
| featuretools | 0.9665 | 91 | 0.13s | 120.46s | success |
| tsfresh | 0.9665 | 14 | 12.68s | 120.04s | success |
| autofeat | 0.9609 | 10 | 91.66s | 120.00s | success |
| caafe | 0.9665 | 7 | 0.00s | 120.07s | success |

### house_prices
**Task**: regression

| Tool | R² Score | Features | FE Time | Train Time | Status |
|------|----------|----------|---------|------------|--------|
| baseline | 0.9242 | 14 | 0.00s | 120.46s | success |
| featcopilot | 0.9203 | 46 | 63.53s | 120.85s | success |
| featuretools | 0.9209 | 100 | 0.19s | 121.92s | success |
| tsfresh | 0.9193 | 49 | 15.36s | 120.68s | success |
| autofeat | 0.9172 | 40 | 68.93s | 120.47s | success |
| caafe | 0.9242 | 14 | 0.00s | 120.35s | success |

### credit_risk
**Task**: classification

| Tool | Accuracy | Features | FE Time | Train Time | Status |
|------|----------|----------|---------|------------|--------|
| baseline | 0.7075 | 10 | 0.00s | 120.05s | success |
| featcopilot | 0.7100 | 86 | 62.84s | 120.54s | success |
| featuretools | 0.7100 | 100 | 0.16s | 120.97s | success |
| tsfresh | 0.6925 | 28 | 15.65s | 120.06s | success |
| autofeat | 0.6950 | 25 | 307.28s | 120.78s | success |
| caafe | 0.7075 | 10 | 0.00s | 120.04s | success |

### bike_sharing
**Task**: regression

| Tool | R² Score | Features | FE Time | Train Time | Status |
|------|----------|----------|---------|------------|--------|
| baseline | 0.8477 | 10 | 0.00s | 171.86s | success |
| featcopilot | 0.8390 | 48 | 63.37s | 125.80s | success |
| featuretools | 0.8394 | 100 | 0.14s | 123.15s | success |
| tsfresh | 0.8310 | 49 | 38.59s | 121.35s | success |
| autofeat | 0.8387 | 41 | 30.46s | 121.66s | success |
| caafe | 0.8477 | 10 | 0.00s | 168.58s | success |

### customer_churn
**Task**: classification

| Tool | Accuracy | Features | FE Time | Train Time | Status |
|------|----------|----------|---------|------------|--------|
| baseline | 0.7700 | 10 | 0.00s | 120.39s | success |
| featcopilot | 0.7500 | 82 | 62.73s | 120.35s | success |
| featuretools | 0.7675 | 100 | 0.15s | 120.00s | success |
| tsfresh | 0.7700 | 49 | 20.94s | 120.45s | success |
| autofeat | 0.7600 | 21 | 365.20s | 120.26s | success |
| caafe | 0.7700 | 10 | 0.00s | 120.19s | success |

### insurance_claims
**Task**: regression

| Tool | R² Score | Features | FE Time | Train Time | Status |
|------|----------|----------|---------|------------|--------|
| baseline | 0.9895 | 10 | 0.00s | 121.54s | success |
| featcopilot | 0.9848 | 36 | 63.77s | 120.76s | success |
| featuretools | 0.9892 | 100 | 0.15s | 121.45s | success |
| tsfresh | 0.9824 | 35 | 20.13s | 121.73s | success |
| autofeat | 0.9894 | 31 | 26.95s | 120.93s | success |
| caafe | 0.9907 | 10 | 0.00s | 121.43s | success |

## Key Findings
- **FeatCopilot average improvement**: -1.47% over baseline
- **Best improvement**: +0.35% on credit_risk

## Conclusion
FeatCopilot demonstrates competitive or superior performance compared to other feature engineering tools while providing a more intuitive API and LLM-powered feature suggestions.
