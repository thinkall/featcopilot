# FLAML Real-World Datasets Benchmark Report
**Generated:** 2026-02-03 10:48:22
**Time Budget:** 60s per FLAML run

## Summary

| Dataset | Task | Samples | Features | Baseline | +FeatCopilot | Improvement | FE Time |
|---------|------|---------|----------|----------|--------------|-------------|--------|
| house_prices | regre | 1,460 | 14→69 | 0.9242 | 0.9122 | -1.29% | 1.9s |
| employee_attrition | class | 1,470 | 11→74 | 0.9342 | 0.9342 | +0.00% | 1.0s |
| telco_churn | class | 4,225 | 51→100 | 1.0000 | 1.0000 | +0.00% | 2.4s |
| adult_census | class | 32,561 | 14→100 | 0.8662 | 0.8678 | +0.19% | 13.6s |
| spaceship_titanic | class | 8,000 | 11→70 | 0.5201 | 0.4025 | -22.62% | 3.0s |
| home_credit | class | 5,000 | 10→88 | 0.7370 | 0.7279 | -1.24% | 2.2s |
| bike_sharing | regre | 2,000 | 10→52 | 0.8455 | 0.8366 | -1.06% | 1.5s |
| medical_cost | regre | 1,300 | 6→22 | 0.8918 | 0.8949 | +0.35% | 0.5s |
| wine_quality | regre | 5,000 | 11→51 | 0.7447 | 0.7373 | -1.00% | 2.8s |
| life_expectancy | regre | 2,500 | 11→67 | 0.5860 | 0.5837 | -0.40% | 3.2s |

## Overall Statistics

- **Datasets tested:** 10
- **FeatCopilot improved:** 2 datasets
- **Baseline better:** 6 datasets
- **Average improvement:** -2.71%
- **Max improvement:** +0.35%
- **Min improvement:** -22.62%

## Detailed Results

### House Prices (Kaggle-style)
- **Task:** regression
- **Samples:** 1,460
- **Original features:** 14
- **Engineered features:** 69
- **Feature engineering time:** 1.9s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| RMSE | 21339.0142 | 22957.4430 |
| MAE | 16842.6400 | 18024.4358 |
| R² | 0.9242 | 0.9122 |
| Train Time | 60.8s | 60.1s |
| Best Model | xgboost | lgbm |

### Employee Attrition (IBM HR)
- **Task:** classification
- **Samples:** 1,470
- **Original features:** 11
- **Engineered features:** 74
- **Feature engineering time:** 1.0s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| Accuracy | 0.9558 | 0.9558 |
| F1 (macro) | 0.4887 | 0.4887 |
| F1 (weighted) | 0.9342 | 0.9342 |
| ROC-AUC | 0.7410 | 0.7328 |
| Train Time | 60.2s | 60.6s |
| Best Model | lgbm | lgbm |

### Telco Customer Churn (Kaggle)
- **Task:** classification
- **Samples:** 4,225
- **Original features:** 51
- **Engineered features:** 100
- **Feature engineering time:** 2.4s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| Accuracy | 1.0000 | 1.0000 |
| F1 (macro) | 1.0000 | 1.0000 |
| F1 (weighted) | 1.0000 | 1.0000 |
| ROC-AUC | 1.0000 | 1.0000 |
| Train Time | 60.7s | 61.4s |
| Best Model | lgbm | lgbm |

### Adult Census Income (OpenML)
- **Task:** classification
- **Samples:** 32,561
- **Original features:** 14
- **Engineered features:** 100
- **Feature engineering time:** 13.6s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| Accuracy | 0.8695 | 0.8716 |
| F1 (macro) | 0.8128 | 0.8143 |
| F1 (weighted) | 0.8662 | 0.8678 |
| ROC-AUC | 0.9228 | 0.9247 |
| Train Time | 60.9s | 63.5s |
| Best Model | lgbm | lgbm |

### Spaceship Titanic (synthetic)
- **Task:** classification
- **Samples:** 8,000
- **Original features:** 11
- **Engineered features:** 70
- **Feature engineering time:** 3.0s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| Accuracy | 0.5719 | 0.5600 |
| F1 (macro) | 0.4965 | 0.3590 |
| F1 (weighted) | 0.5201 | 0.4025 |
| ROC-AUC | 0.6015 | 0.5960 |
| Train Time | 60.1s | 60.1s |
| Best Model | lgbm | extra_tree |

### Credit Risk (synthetic)
- **Task:** classification
- **Samples:** 5,000
- **Original features:** 10
- **Engineered features:** 88
- **Feature engineering time:** 2.2s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| Accuracy | 0.7370 | 0.7280 |
| F1 (macro) | 0.7370 | 0.7277 |
| F1 (weighted) | 0.7370 | 0.7279 |
| ROC-AUC | 0.8036 | 0.8017 |
| Train Time | 60.6s | 60.0s |
| Best Model | catboost | xgb_limitdepth |

### Bike Sharing (Kaggle-style)
- **Task:** regression
- **Samples:** 2,000
- **Original features:** 10
- **Engineered features:** 52
- **Feature engineering time:** 1.5s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| RMSE | 29.7510 | 30.5998 |
| MAE | 23.6922 | 24.2514 |
| R² | 0.8455 | 0.8366 |
| Train Time | 68.5s | 62.0s |
| Best Model | catboost | catboost |

### Medical Cost (synthetic)
- **Task:** regression
- **Samples:** 1,300
- **Original features:** 6
- **Engineered features:** 22
- **Feature engineering time:** 0.5s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| RMSE | 3228.3671 | 3181.7073 |
| MAE | 2574.0268 | 2534.3749 |
| R² | 0.8918 | 0.8949 |
| Train Time | 60.0s | 63.7s |
| Best Model | lgbm | catboost |

### Wine Quality (synthetic)
- **Task:** regression
- **Samples:** 5,000
- **Original features:** 11
- **Engineered features:** 51
- **Feature engineering time:** 2.8s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| RMSE | 0.5967 | 0.6054 |
| MAE | 0.4874 | 0.4927 |
| R² | 0.7447 | 0.7373 |
| Train Time | 60.0s | 60.1s |
| Best Model | extra_tree | lgbm |

### Life Expectancy (synthetic)
- **Task:** regression
- **Samples:** 2,500
- **Original features:** 11
- **Engineered features:** 67
- **Feature engineering time:** 3.2s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| RMSE | 3.0720 | 3.0807 |
| MAE | 2.5044 | 2.5021 |
| R² | 0.5860 | 0.5837 |
| Train Time | 60.4s | 60.1s |
| Best Model | catboost | lgbm |
