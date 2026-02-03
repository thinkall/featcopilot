# FeatCopilot Basic Models Benchmark Report
**Generated:** 2026-02-03 16:39:29
**Engines:** tabular

## Summary

| Dataset | Task | Samples | Features | Best Baseline | Best +FC | Improvement |
|---------|------|---------|----------|---------------|----------|-------------|
| titanic | class | 891 | 7→27 | 0.8815 | 0.8815 | +0.00% |
| house_prices | regre | 1,460 | 14→95 | 0.9306 | 0.8889 | +2.20% |
| medical_cost | regre | 1,300 | 6→13 | 0.8970 | 0.8980 | +0.70% |
| wine_quality | regre | 5,000 | 11→85 | 0.7527 | 0.7466 | +0.27% |
| employee_attrition | class | 1,470 | 11→81 | 0.9342 | 0.9342 | +0.00% |
| bike_sharing | regre | 2,000 | 10→52 | 0.8091 | 0.8050 | +7.34% |
| credit_risk | class | 2,000 | 10→114 | 0.7153 | 0.7043 | +0.00% |

## Overall Statistics

- **Datasets tested:** 7
- **FeatCopilot improved:** 4 datasets (57%)
- **Average improvement:** +1.50%
- **Max improvement:** +7.34%
- **Min improvement:** +0.00%

## Detailed Results

### titanic - Titanic survival prediction
- **Task:** classification
- **Samples:** 891
- **Features:** 7 → 27
- **FE Time:** 0.5s

| Model | Baseline Acc | Baseline F1 | +FC Acc | +FC F1 | Improvement |
|-------|-------------|-------------|---------|--------|-------------|
| RandomForest | 0.9162 | 0.8815 | 0.9162 | 0.8815 | +0.00% |
| LogisticRegression | 0.9162 | 0.8815 | 0.8380 | 0.8405 | -4.64% |

### house_prices - House price prediction
- **Task:** regression
- **Samples:** 1,460
- **Features:** 14 → 95
- **FE Time:** 2.8s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.8697 | 27970.8358 | 0.8889 | 25834.0175 | +2.20% |
| Ridge | 0.9306 | 20411.5708 | 0.3954 | 60254.3950 | -57.51% |

### medical_cost - Medical Cost Prediction
- **Task:** regression
- **Samples:** 1,300
- **Features:** 6 → 13
- **FE Time:** 0.4s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.8786 | 3420.4200 | 0.8848 | 3332.1291 | +0.70% |
| Ridge | 0.8970 | 3149.9687 | 0.8980 | 3134.5446 | +0.11% |

### wine_quality - Wine Quality
- **Task:** regression
- **Samples:** 5,000
- **Features:** 11 → 85
- **FE Time:** 10.2s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.7371 | 0.6056 | 0.7391 | 0.6033 | +0.27% |
| Ridge | 0.7527 | 0.5873 | 0.7466 | 0.5945 | -0.81% |

### employee_attrition - IBM HR Attrition
- **Task:** classification
- **Samples:** 1,470
- **Features:** 11 → 81
- **FE Time:** 1.3s

| Model | Baseline Acc | Baseline F1 | +FC Acc | +FC F1 | Improvement |
|-------|-------------|-------------|---------|--------|-------------|
| RandomForest | 0.9558 | 0.9342 | 0.9558 | 0.9342 | +0.00% |
| LogisticRegression | 0.9558 | 0.9342 | 0.9524 | 0.9325 | -0.18% |

### bike_sharing - Bike Sharing Demand
- **Task:** regression
- **Samples:** 2,000
- **Features:** 10 → 52
- **FE Time:** 1.8s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.8091 | 33.0774 | 0.8050 | 33.4318 | -0.51% |
| Ridge | 0.7211 | 39.9808 | 0.7740 | 35.9865 | +7.34% |

### credit_risk - Credit risk classification
- **Task:** classification
- **Samples:** 2,000
- **Features:** 10 → 114
- **FE Time:** 1.7s

| Model | Baseline Acc | Baseline F1 | +FC Acc | +FC F1 | Improvement |
|-------|-------------|-------------|---------|--------|-------------|
| RandomForest | 0.7075 | 0.7043 | 0.7075 | 0.7043 | +0.00% |
| LogisticRegression | 0.7175 | 0.7153 | 0.6525 | 0.6526 | -8.76% |
