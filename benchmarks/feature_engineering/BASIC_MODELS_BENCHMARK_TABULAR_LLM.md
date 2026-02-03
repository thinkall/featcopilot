# FeatCopilot Basic Models Benchmark Report
**Generated:** 2026-02-03 16:42:45
**Engines:** tabular, llm

## Summary

| Dataset | Task | Samples | Features | Best Baseline | Best +FC | Improvement |
|---------|------|---------|----------|---------------|----------|-------------|
| titanic | class | 891 | 7→49 | 0.8815 | 0.8815 | +0.00% |
| medical_cost | regre | 1,300 | 6→31 | 0.8970 | 0.8956 | +0.66% |
| wine_quality | regre | 5,000 | 11→95 | 0.7527 | 0.7456 | -0.05% |
| house_prices | regre | 1,460 | 14→113 | 0.9306 | 0.8989 | +3.35% |

## Overall Statistics

- **Datasets tested:** 4
- **FeatCopilot improved:** 2 datasets (50%)
- **Average improvement:** +0.99%
- **Max improvement:** +3.35%
- **Min improvement:** -0.05%

## Detailed Results

### titanic - Titanic survival prediction
- **Task:** classification
- **Samples:** 891
- **Features:** 7 → 49
- **FE Time:** 36.3s

| Model | Baseline Acc | Baseline F1 | +FC Acc | +FC F1 | Improvement |
|-------|-------------|-------------|---------|--------|-------------|
| RandomForest | 0.9162 | 0.8815 | 0.9162 | 0.8815 | +0.00% |
| LogisticRegression | 0.9162 | 0.8815 | 0.8380 | 0.8405 | -4.64% |

### medical_cost - Medical Cost Prediction
- **Task:** regression
- **Samples:** 1,300
- **Features:** 6 → 31
- **FE Time:** 34.1s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.8786 | 3420.4200 | 0.8844 | 3337.1500 | +0.66% |
| Ridge | 0.8970 | 3149.9687 | 0.8956 | 3171.6215 | -0.16% |

### wine_quality - Wine Quality
- **Task:** regression
- **Samples:** 5,000
- **Features:** 11 → 95
- **FE Time:** 45.7s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.7371 | 0.6056 | 0.7367 | 0.6060 | -0.05% |
| Ridge | 0.7527 | 0.5873 | 0.7456 | 0.5957 | -0.95% |

### house_prices - House price prediction
- **Task:** regression
- **Samples:** 1,460
- **Features:** 14 → 113
- **FE Time:** 39.0s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.8697 | 27970.8358 | 0.8989 | 24644.7903 | +3.35% |
| Ridge | 0.9306 | 20411.5708 | 0.4328 | 58360.9826 | -53.49% |
