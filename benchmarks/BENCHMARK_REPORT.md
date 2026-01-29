# FeatCopilot Benchmark Report

## Summary

**Classification Tasks:**
- Average Accuracy Improvement: **+0.50%**
- Max Improvement: +4.93%
- Improvements > 0: 6/15

**Regression Tasks:**
- Average R² Improvement: **-1.44%**
- Max Improvement: +0.98%
- Improvements > 0: 4/12

## Detailed Results

### Classification Datasets

| Dataset | Model | Baseline Acc | FeatCopilot Acc | Improvement | Baseline F1 | FeatCopilot F1 |
|---------|-------|--------------|-----------------|-------------|-------------|----------------|
| Breast Cancer (sklearn) | LogisticRegression | 0.9737 | 0.9825 | +0.90% | 0.9736 | 0.9824 |
| Breast Cancer (sklearn) | RandomForest | 0.9649 | 0.9649 | +0.00% | 0.9647 | 0.9649 |
| Breast Cancer (sklearn) | GradientBoosting | 0.9561 | 0.9561 | +0.00% | 0.9560 | 0.9560 |
| Credit Risk (synthetic) | LogisticRegression | 0.7025 | 0.7025 | +0.00% | 0.7024 | 0.7022 |
| Credit Risk (synthetic) | RandomForest | 0.7050 | 0.7225 | +2.48% | 0.7038 | 0.7218 |
| Credit Risk (synthetic) | GradientBoosting | 0.6975 | 0.7050 | +1.08% | 0.6967 | 0.7041 |
| Customer Churn (synthetic) | LogisticRegression | 0.8600 | 0.8625 | +0.29% | 0.7953 | 0.8057 |
| Customer Churn (synthetic) | RandomForest | 0.8575 | 0.8575 | +0.00% | 0.7986 | 0.7986 |
| Customer Churn (synthetic) | GradientBoosting | 0.8550 | 0.8450 | -1.17% | 0.8054 | 0.7919 |
| Medical Diagnosis (synthetic) | LogisticRegression | 0.8533 | 0.8500 | -0.39% | 0.8074 | 0.8135 |
| Medical Diagnosis (synthetic) | RandomForest | 0.8600 | 0.8600 | +0.00% | 0.8014 | 0.8067 |
| Medical Diagnosis (synthetic) | GradientBoosting | 0.8600 | 0.8500 | -1.16% | 0.8161 | 0.8053 |
| Complex Classification (synthetic) | LogisticRegression | 0.8625 | 0.9050 | +4.93% | 0.8582 | 0.9045 |
| Complex Classification (synthetic) | RandomForest | 0.9125 | 0.9075 | -0.55% | 0.9103 | 0.9039 |
| Complex Classification (synthetic) | GradientBoosting | 0.9075 | 0.9175 | +1.10% | 0.9058 | 0.9163 |

### Regression Datasets

| Dataset | Model | Baseline R² | FeatCopilot R² | Improvement | Baseline RMSE | FeatCopilot RMSE |
|---------|-------|-------------|----------------|-------------|---------------|------------------|
| Diabetes (sklearn) | Ridge | 0.4541 | 0.4170 | -8.18% | 53.78 | 55.58 |
| Diabetes (sklearn) | RandomForest | 0.4415 | 0.4458 | +0.98% | 54.40 | 54.19 |
| Diabetes (sklearn) | GradientBoosting | 0.4530 | 0.4202 | -7.24% | 53.83 | 55.43 |
| California Housing (synthetic fallback) | Ridge | 0.8900 | 0.8888 | -0.14% | 0.54 | 0.54 |
| California Housing (synthetic fallback) | RandomForest | 0.7974 | 0.7846 | -1.61% | 0.73 | 0.75 |
| California Housing (synthetic fallback) | GradientBoosting | 0.8313 | 0.8320 | +0.08% | 0.66 | 0.66 |
| Housing Price (synthetic) | Ridge | 0.9754 | 0.9807 | +0.54% | 37880.85 | 33568.09 |
| Housing Price (synthetic) | RandomForest | 0.9617 | 0.9650 | +0.35% | 47251.47 | 45157.81 |
| Housing Price (synthetic) | GradientBoosting | 0.9746 | 0.9745 | -0.01% | 38498.85 | 38573.89 |
| Complex Regression (synthetic) | Ridge | 0.9967 | 0.9952 | -0.15% | 10.43 | 12.54 |
| Complex Regression (synthetic) | RandomForest | 0.8117 | 0.7971 | -1.79% | 78.85 | 81.84 |
| Complex Regression (synthetic) | GradientBoosting | 0.9137 | 0.9126 | -0.11% | 53.39 | 53.71 |

## Feature Engineering Statistics

| Dataset | Original Features | Engineered Features | FE Time (s) |
|---------|-------------------|---------------------|-------------|
| Diabetes (sklearn) | 10 | 50 | 0.26 |
| Breast Cancer (sklearn) | 30 | 48 | 0.31 |
| California Housing (synthetic fallback) | 8 | 28 | 0.33 |
| Credit Risk (synthetic) | 10 | 50 | 0.33 |
| Housing Price (synthetic) | 10 | 35 | 0.42 |
| Customer Churn (synthetic) | 10 | 48 | 0.33 |
| Medical Diagnosis (synthetic) | 12 | 48 | 0.33 |
| Complex Regression (synthetic) | 15 | 50 | 0.63 |
| Complex Classification (synthetic) | 15 | 50 | 0.34 |

## Methodology

- **Train/Test Split**: 80/20 with random_state=42
- **Feature Engineering**: FeatCopilot TabularEngine with importance-based selection (max 50 features)
- **Preprocessing**: StandardScaler applied to all features
- **Models**: LogisticRegression/Ridge, RandomForest, GradientBoosting
- **Metrics**: Accuracy/R², F1-score/RMSE, ROC-AUC/MAE
