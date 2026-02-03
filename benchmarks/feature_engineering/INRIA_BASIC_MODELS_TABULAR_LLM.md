# INRIA Basic Models Benchmark Report
**Generated:** 2026-02-03 16:54:18
**Engines:** tabular, llm
**Max Samples:** 30,000

## Summary

| Dataset | Task | Samples | Features | Best Baseline | Best +FC | Improvement |
|---------|------|---------|----------|---------------|----------|-------------|
| abalone | regre | 4,177 | 7→41 | 0.5287 | 0.5870 | +11.31% |
| wine_quality | regre | 6,497 | 11→87 | 0.4596 | 0.4671 | +28.15% |
| cpu_act | regre | 8,192 | 21→193 | 0.9792 | 0.9797 | +28.17% |
| houses | regre | 20,640 | 8→49 | 0.8234 | 0.8153 | +12.20% |
| bike_sharing | regre | 17,379 | 6→54 | 0.6891 | 0.6918 | +82.88% |

## Overall Statistics

- **Datasets tested:** 5
- **FeatCopilot improved:** 5 (100%)
- **Average improvement:** +32.54%
- **Max improvement:** +82.88%
- **Min improvement:** +11.31%

## Detailed Results

### abalone - Abalone age prediction
- **Task:** regression
- **Samples:** 4,177
- **Features:** 7 → 41
- **FE Time:** 42.6s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.5273 | 2.2620 | 0.5870 | 2.1145 | +11.31% |
| Ridge | 0.5287 | 2.2587 | 0.5542 | 2.1969 | +4.81% |

### wine_quality - Wine quality score
- **Task:** regression
- **Samples:** 6,497
- **Features:** 11 → 87
- **FE Time:** 55.6s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.4596 | 0.6318 | 0.4671 | 0.6274 | +1.64% |
| Ridge | 0.2594 | 0.7396 | 0.3324 | 0.7022 | +28.15% |

### cpu_act - CPU activity prediction
- **Task:** regression
- **Samples:** 8,192
- **Features:** 21 → 193
- **FE Time:** 56.7s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.9792 | 2.4965 | 0.9797 | 2.4659 | +0.05% |
| Ridge | 0.7371 | 8.8765 | 0.9447 | 4.0720 | +28.17% |

### houses - House value prediction
- **Task:** regression
- **Samples:** 20,640
- **Features:** 8 → 49
- **FE Time:** 78.9s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.8234 | 0.2394 | 0.8153 | 0.2448 | -0.99% |
| Ridge | 0.6301 | 0.3465 | 0.7070 | 0.3084 | +12.20% |

### bike_sharing - Bike rental demand
- **Task:** regression
- **Samples:** 17,379
- **Features:** 6 → 54
- **FE Time:** 52.7s

| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |
|-------|------------|---------------|--------|----------|-------------|
| RandomForest | 0.6891 | 99.2282 | 0.6918 | 98.7839 | +0.40% |
| Ridge | 0.3429 | 144.2518 | 0.6270 | 108.6780 | +82.88% |
