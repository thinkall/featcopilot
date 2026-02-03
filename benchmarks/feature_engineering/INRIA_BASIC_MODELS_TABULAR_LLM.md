# INRIA Basic Models Benchmark Report
**Generated:** 2026-02-03 18:00:06
**Engines:** tabular, llm
**Max Samples:** 30,000
**Models:** RandomForest, Ridge (regression) / LogisticRegression (classification)

## Summary

| Dataset | Task | Samples | Features | Best Baseline | Best +FC | Improvement |
|---------|------|---------|----------|---------------|----------|-------------|
| abalone | regre | 4,177 | 7→44 | 0.5287 | 0.5418 | +2.46% |
| wine_quality | regre | 6,497 | 11→72 | 0.4596 | 0.4700 | +2.28% |
| diamonds | regre | 30,000 | 6→25 | 0.9464 | 0.9472 | +0.08% |
| credit | class | 16,714 | 10→91 | 0.7765 | 0.7807 | +0.54% |
| eye_movements | class | 7,608 | 23→150 | 0.6275 | 0.6162 | -1.79% |
| cpu_act | regre | 8,192 | 21→136 | 0.9792 | 0.9796 | +0.04% |
| houses | regre | 20,640 | 8→50 | 0.8234 | 0.8153 | -0.99% |
| bike_sharing | regre | 17,379 | 6→54 | 0.6891 | 0.6970 | +1.15% |
| jannis | class | 30,000 | 54→148 | 0.7782 | 0.7804 | +0.29% |
| bioresponse | class | 3,434 | 419→417 | 0.7813 | 0.7551 | -3.35% |

## Overall Statistics

- **Datasets tested:** 10
- **FeatCopilot improved:** 7 (70%)
- **Average improvement:** +0.07%
- **Max improvement:** +2.46%
- **Min improvement:** -3.35%

## Detailed Results

### abalone - Abalone age prediction
- **Task:** regression
- **Samples:** 4,177
- **Features:** 7 → 44
- **FE Time:** 43.9s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.5273 | 2.2620 | 0.5848 | 2.1200 | +10.91% |
| Ridge | 0.5287 | 2.2587 | 0.5418 | 2.2272 | +2.46% |

### wine_quality - Wine quality score
- **Task:** regression
- **Samples:** 6,497
- **Features:** 11 → 72
- **FE Time:** 16.2s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.4596 | 0.6318 | 0.4700 | 0.6256 | +2.28% |
| Ridge | 0.2594 | 0.7396 | 0.3284 | 0.7043 | +26.64% |

### diamonds - Diamond price prediction
- **Task:** regression
- **Samples:** 30,000
- **Features:** 6 → 25
- **FE Time:** 79.2s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.9464 | 0.2339 | 0.9472 | 0.2322 | +0.08% |
| Ridge | 0.9269 | 0.2732 | 0.9400 | 0.2476 | +1.41% |

### credit - Credit approval
- **Task:** classification
- **Samples:** 16,714
- **Features:** 10 → 91
- **FE Time:** 16.4s

| Model | Baseline Acc | Baseline F1 | +FC Acc | +FC F1 | Improvement |
|-------|-------------|-------------|---------|--------|-------------|
| RandomForest | 0.7765 | 0.7765 | 0.7807 | 0.7807 | +0.54% |
| LogisticRegression | 0.7236 | 0.7197 | 0.6820 | 0.6753 | -6.16% |

### eye_movements - Eye movement classification
- **Task:** classification
- **Samples:** 7,608
- **Features:** 23 → 150
- **FE Time:** 43.6s

| Model | Baseline Acc | Baseline F1 | +FC Acc | +FC F1 | Improvement |
|-------|-------------|-------------|---------|--------|-------------|
| RandomForest | 0.6275 | 0.6275 | 0.6163 | 0.6162 | -1.79% |
| LogisticRegression | 0.5736 | 0.5734 | 0.5434 | 0.5427 | -5.35% |

### cpu_act - CPU activity prediction
- **Task:** regression
- **Samples:** 8,192
- **Features:** 21 → 136
- **FE Time:** 40.3s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.9792 | 2.4965 | 0.9796 | 2.4711 | +0.04% |
| Ridge | 0.7371 | 8.8765 | 0.8574 | 6.5363 | +16.33% |

### houses - House value prediction
- **Task:** regression
- **Samples:** 20,640
- **Features:** 8 → 50
- **FE Time:** 86.5s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.8234 | 0.2394 | 0.8153 | 0.2449 | -0.99% |
| Ridge | 0.6301 | 0.3465 | 0.7082 | 0.3077 | +12.39% |

### bike_sharing - Bike rental demand
- **Task:** regression
- **Samples:** 17,379
- **Features:** 6 → 54
- **FE Time:** 79.6s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.6891 | 99.2282 | 0.6970 | 97.9599 | +1.15% |
| Ridge | 0.3429 | 144.2518 | 0.6165 | 110.1971 | +79.81% |

### jannis - Multi-class classification
- **Task:** classification
- **Samples:** 30,000
- **Features:** 54 → 148
- **FE Time:** 91.4s

| Model | Baseline Acc | Baseline F1 | +FC Acc | +FC F1 | Improvement |
|-------|-------------|-------------|---------|--------|-------------|
| RandomForest | 0.7785 | 0.7782 | 0.7807 | 0.7804 | +0.29% |
| LogisticRegression | 0.7368 | 0.7368 | 0.7472 | 0.7470 | +1.38% |

### bioresponse - Biological response
- **Task:** classification
- **Samples:** 3,434
- **Features:** 419 → 417
- **FE Time:** 57.0s

| Model | Baseline Acc | Baseline F1 | +FC Acc | +FC F1 | Improvement |
|-------|-------------|-------------|---------|--------|-------------|
| RandomForest | 0.7817 | 0.7813 | 0.7555 | 0.7551 | -3.35% |
| LogisticRegression | 0.7336 | 0.7336 | 0.7307 | 0.7307 | -0.39% |
