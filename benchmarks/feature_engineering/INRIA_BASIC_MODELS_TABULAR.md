# INRIA Basic Models Benchmark Report
**Generated:** 2026-02-03 17:49:00
**Engines:** tabular
**Max Samples:** 30,000
**Models:** RandomForest, Ridge (regression) / LogisticRegression (classification)

## Summary

| Dataset | Task | Samples | Features | Best Baseline | Best +FC | Improvement |
|---------|------|---------|----------|---------------|----------|-------------|
| abalone | regre | 4,177 | 7→30 | 0.5287 | 0.5723 | +8.25% |
| wine_quality | regre | 6,497 | 11→72 | 0.4596 | 0.4700 | +2.28% |
| diamonds | regre | 30,000 | 6→15 | 0.9464 | 0.9470 | +0.06% |
| credit | class | 16,714 | 10→91 | 0.7765 | 0.7807 | +0.54% |
| eye_movements | class | 7,608 | 23→143 | 0.6275 | 0.5990 | -4.54% |
| cpu_act | regre | 8,192 | 21→136 | 0.9792 | 0.9796 | +0.04% |
| houses | regre | 20,640 | 8→42 | 0.8234 | 0.8146 | -1.08% |
| bike_sharing | regre | 17,379 | 6→37 | 0.6891 | 0.6929 | +0.56% |
| jannis | class | 30,000 | 54→148 | 0.7782 | 0.7804 | +0.29% |
| bioresponse | class | 3,434 | 419→413 | 0.7813 | 0.7623 | -2.42% |

## Overall Statistics

- **Datasets tested:** 10
- **FeatCopilot improved:** 7 (70%)
- **Average improvement:** +0.40%
- **Max improvement:** +8.25%
- **Min improvement:** -4.54%

## Detailed Results

### abalone - Abalone age prediction
- **Task:** regression
- **Samples:** 4,177
- **Features:** 7 → 30
- **FE Time:** 5.6s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.5273 | 2.2620 | 0.5768 | 2.1403 | +9.39% |
| Ridge | 0.5287 | 2.2587 | 0.5723 | 2.1516 | +8.25% |

### wine_quality - Wine quality score
- **Task:** regression
- **Samples:** 6,497
- **Features:** 11 → 72
- **FE Time:** 13.6s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.4596 | 0.6318 | 0.4700 | 0.6256 | +2.28% |
| Ridge | 0.2594 | 0.7396 | 0.3284 | 0.7043 | +26.64% |

### diamonds - Diamond price prediction
- **Task:** regression
- **Samples:** 30,000
- **Features:** 6 → 15
- **FE Time:** 28.9s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.9464 | 0.2339 | 0.9470 | 0.2326 | +0.06% |
| Ridge | 0.9269 | 0.2732 | 0.9380 | 0.2515 | +1.20% |

### credit - Credit approval
- **Task:** classification
- **Samples:** 16,714
- **Features:** 10 → 91
- **FE Time:** 11.3s

| Model | Baseline Acc | Baseline F1 | +FC Acc | +FC F1 | Improvement |
|-------|-------------|-------------|---------|--------|-------------|
| RandomForest | 0.7765 | 0.7765 | 0.7807 | 0.7807 | +0.54% |
| LogisticRegression | 0.7236 | 0.7197 | 0.6820 | 0.6753 | -6.16% |

### eye_movements - Eye movement classification
- **Task:** classification
- **Samples:** 7,608
- **Features:** 23 → 143
- **FE Time:** 6.1s

| Model | Baseline Acc | Baseline F1 | +FC Acc | +FC F1 | Improvement |
|-------|-------------|-------------|---------|--------|-------------|
| RandomForest | 0.6275 | 0.6275 | 0.5992 | 0.5990 | -4.54% |
| LogisticRegression | 0.5736 | 0.5734 | 0.5539 | 0.5532 | -3.52% |

### cpu_act - CPU activity prediction
- **Task:** regression
- **Samples:** 8,192
- **Features:** 21 → 136
- **FE Time:** 17.2s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.9792 | 2.4965 | 0.9796 | 2.4711 | +0.04% |
| Ridge | 0.7371 | 8.8765 | 0.8574 | 6.5363 | +16.33% |

### houses - House value prediction
- **Task:** regression
- **Samples:** 20,640
- **Features:** 8 → 42
- **FE Time:** 40.0s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.8234 | 0.2394 | 0.8146 | 0.2453 | -1.08% |
| Ridge | 0.6301 | 0.3465 | 0.7014 | 0.3113 | +11.32% |

### bike_sharing - Bike rental demand
- **Task:** regression
- **Samples:** 17,379
- **Features:** 6 → 37
- **FE Time:** 14.3s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.6891 | 99.2282 | 0.6929 | 98.6134 | +0.56% |
| Ridge | 0.3429 | 144.2518 | 0.4249 | 134.9495 | +23.92% |

### jannis - Multi-class classification
- **Task:** classification
- **Samples:** 30,000
- **Features:** 54 → 148
- **FE Time:** 28.9s

| Model | Baseline Acc | Baseline F1 | +FC Acc | +FC F1 | Improvement |
|-------|-------------|-------------|---------|--------|-------------|
| RandomForest | 0.7785 | 0.7782 | 0.7807 | 0.7804 | +0.29% |
| LogisticRegression | 0.7368 | 0.7368 | 0.7472 | 0.7470 | +1.38% |

### bioresponse - Biological response
- **Task:** classification
- **Samples:** 3,434
- **Features:** 419 → 413
- **FE Time:** 11.7s

| Model | Baseline Acc | Baseline F1 | +FC Acc | +FC F1 | Improvement |
|-------|-------------|-------------|---------|--------|-------------|
| RandomForest | 0.7817 | 0.7813 | 0.7627 | 0.7623 | -2.42% |
| LogisticRegression | 0.7336 | 0.7336 | 0.7322 | 0.7322 | -0.19% |
