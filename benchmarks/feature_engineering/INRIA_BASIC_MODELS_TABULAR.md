# INRIA Basic Models Benchmark Report
**Generated:** 2026-02-03 16:48:28
**Engines:** tabular
**Max Samples:** 30,000

## Summary

| Dataset | Task | Samples | Features | Best Baseline | Best +FC | Improvement |
|---------|------|---------|----------|---------------|----------|-------------|
| abalone | regre | 4,177 | 7→30 | 0.5287 | 0.5768 | +9.39% |
| wine_quality | regre | 6,497 | 11→80 | 0.4596 | 0.4599 | +27.20% |
| diamonds | regre | 30,000 | 6→15 | 0.9464 | 0.9470 | +1.20% |
| credit | class | 16,714 | 10→95 | 0.7765 | 0.7828 | +0.81% |
| eye_movements | class | 7,608 | 23→193 | 0.6275 | 0.6103 | -2.73% |
| cpu_act | regre | 8,192 | 21→169 | 0.9792 | 0.9795 | +18.86% |
| houses | regre | 20,640 | 8→42 | 0.8234 | 0.8146 | +11.32% |
| bike_sharing | regre | 17,379 | 6→37 | 0.6891 | 0.6929 | +23.92% |
| jannis | class | 30,000 | 54→193 | 0.7782 | 0.7773 | +1.25% |
| bioresponse | class | 3,434 | 419→415 | 0.7813 | 0.7668 | +0.80% |

## Overall Statistics

- **Datasets tested:** 10
- **FeatCopilot improved:** 9 (90%)
- **Average improvement:** +9.20%
- **Max improvement:** +27.20%
- **Min improvement:** -2.73%

## Detailed Results

### abalone - Abalone age prediction
- **Task:** regression
- **Samples:** 4,177
- **Features:** 7 → 30
- **FE Time:** 4.8s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.5273 | 2.2620 | 0.5768 | 2.1403 | +9.39% |
| Ridge | 0.5287 | 2.2587 | 0.5723 | 2.1516 | +8.25% |

### wine_quality - Wine quality score
- **Task:** regression
- **Samples:** 6,497
- **Features:** 11 → 80
- **FE Time:** 15.4s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.4596 | 0.6318 | 0.4599 | 0.6316 | +0.08% |
| Ridge | 0.2594 | 0.7396 | 0.3299 | 0.7035 | +27.20% |

### diamonds - Diamond price prediction
- **Task:** regression
- **Samples:** 30,000
- **Features:** 6 → 15
- **FE Time:** 25.8s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.9464 | 0.2339 | 0.9470 | 0.2326 | +0.06% |
| Ridge | 0.9269 | 0.2732 | 0.9380 | 0.2515 | +1.20% |

### credit - Credit approval
- **Task:** classification
- **Samples:** 16,714
- **Features:** 10 → 95
- **FE Time:** 12.4s

| Model | Baseline Acc | Baseline F1 | +FC Acc | +FC F1 | Improvement |
|-------|-------------|-------------|---------|--------|-------------|
| RandomForest | 0.7765 | 0.7765 | 0.7828 | 0.7828 | +0.81% |
| LogisticRegression | 0.7236 | 0.7197 | 0.6835 | 0.6732 | -6.46% |

### eye_movements - Eye movement classification
- **Task:** classification
- **Samples:** 7,608
- **Features:** 23 → 193
- **FE Time:** 6.9s

| Model | Baseline Acc | Baseline F1 | +FC Acc | +FC F1 | Improvement |
|-------|-------------|-------------|---------|--------|-------------|
| RandomForest | 0.6275 | 0.6275 | 0.6104 | 0.6103 | -2.73% |
| LogisticRegression | 0.5736 | 0.5734 | 0.5414 | 0.5410 | -5.64% |

### cpu_act - CPU activity prediction
- **Task:** regression
- **Samples:** 8,192
- **Features:** 21 → 169
- **FE Time:** 20.2s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.9792 | 2.4965 | 0.9795 | 2.4770 | +0.03% |
| Ridge | 0.7371 | 8.8765 | 0.8761 | 6.0940 | +18.86% |

### houses - House value prediction
- **Task:** regression
- **Samples:** 20,640
- **Features:** 8 → 42
- **FE Time:** 35.2s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.8234 | 0.2394 | 0.8146 | 0.2453 | -1.08% |
| Ridge | 0.6301 | 0.3465 | 0.7014 | 0.3113 | +11.32% |

### bike_sharing - Bike rental demand
- **Task:** regression
- **Samples:** 17,379
- **Features:** 6 → 37
- **FE Time:** 14.2s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.6891 | 99.2282 | 0.6929 | 98.6134 | +0.56% |
| Ridge | 0.3429 | 144.2518 | 0.4249 | 134.9495 | +23.92% |

### jannis - Multi-class classification
- **Task:** classification
- **Samples:** 30,000
- **Features:** 54 → 193
- **FE Time:** 32.8s

| Model | Baseline Acc | Baseline F1 | +FC Acc | +FC F1 | Improvement |
|-------|-------------|-------------|---------|--------|-------------|
| RandomForest | 0.7785 | 0.7782 | 0.7777 | 0.7773 | -0.11% |
| LogisticRegression | 0.7368 | 0.7368 | 0.7462 | 0.7460 | +1.25% |

### bioresponse - Biological response
- **Task:** classification
- **Samples:** 3,434
- **Features:** 419 → 415
- **FE Time:** 11.5s

| Model | Baseline Acc | Baseline F1 | +FC Acc | +FC F1 | Improvement |
|-------|-------------|-------------|---------|--------|-------------|
| RandomForest | 0.7817 | 0.7813 | 0.7671 | 0.7668 | -1.85% |
| LogisticRegression | 0.7336 | 0.7336 | 0.7394 | 0.7394 | +0.80% |
