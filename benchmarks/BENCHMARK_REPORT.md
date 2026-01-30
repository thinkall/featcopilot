# FeatCopilot Benchmark Report

## Summary

**Classification Tasks:**
- Average Accuracy Improvement: **+0.47%**
- Max Improvement: +4.35%
- Improvements > 0: 8/21 (38%)

**Regression Tasks:**
- Average R² Improvement: **-0.97%**
- Max Improvement: +7.52%
- Improvements > 0: 3/12 (25%)

**Time Series Tasks:**
- Classification Avg Improvement: **-6.28%** (0/3 wins)
- Regression Avg R² Improvement: **+0.04%** (3/6 wins)

## Detailed Results

### Classification Datasets

| Dataset | Model | Baseline Acc | FeatCopilot Acc | Improvement | Baseline F1 | FeatCopilot F1 |
|---------|-------|--------------|-----------------|-------------|-------------|----------------|
| Breast Cancer (sklearn) | LogisticRegression | 0.9737 | 0.9737 | +0.00% | 0.9736 | 0.9736 |
| Breast Cancer (sklearn) | RandomForest | 0.9649 | 0.9649 | +0.00% | 0.9647 | 0.9647 |
| Breast Cancer (sklearn) | GradientBoosting | 0.9561 | 0.9561 | +0.00% | 0.9560 | 0.9560 |
| Titanic (Kaggle-style) | LogisticRegression | 0.9665 | 0.9665 | +0.00% | 0.9500 | 0.9500 |
| Titanic (Kaggle-style) | RandomForest | 0.9609 | 0.9553 | -0.58% | 0.9472 | 0.9444 |
| Titanic (Kaggle-style) | GradientBoosting | 0.9609 | 0.9497 | -1.16% | 0.9472 | 0.9416 |
| Credit Card Fraud (Kaggle-style) | LogisticRegression | 0.9860 | 0.9860 | +0.00% | 0.9790 | 0.9790 |
| Credit Card Fraud (Kaggle-style) | RandomForest | 0.9860 | 0.9860 | +0.00% | 0.9790 | 0.9790 |
| Credit Card Fraud (Kaggle-style) | GradientBoosting | 0.9790 | 0.9790 | +0.00% | 0.9755 | 0.9755 |
| Employee Attrition (IBM HR) | LogisticRegression | 0.9762 | 0.9762 | +0.00% | 0.9644 | 0.9697 |
| Employee Attrition (IBM HR) | RandomForest | 0.9762 | 0.9762 | +0.00% | 0.9644 | 0.9644 |
| Employee Attrition (IBM HR) | GradientBoosting | 0.9694 | 0.9728 | +0.35% | 0.9610 | 0.9627 |
| Credit Risk (synthetic) | LogisticRegression | 0.7025 | 0.7275 | +3.56% | 0.7024 | 0.7269 |
| Credit Risk (synthetic) | RandomForest | 0.7050 | 0.7075 | +0.35% | 0.7038 | 0.7069 |
| Credit Risk (synthetic) | GradientBoosting | 0.6975 | 0.7075 | +1.43% | 0.6967 | 0.7067 |
| Medical Diagnosis (synthetic) | LogisticRegression | 0.8533 | 0.8567 | +0.39% | 0.8074 | 0.8218 |
| Medical Diagnosis (synthetic) | RandomForest | 0.8600 | 0.8567 | -0.39% | 0.8014 | 0.8047 |
| Medical Diagnosis (synthetic) | GradientBoosting | 0.8600 | 0.8633 | +0.39% | 0.8161 | 0.8226 |
| Complex Classification (synthetic) | LogisticRegression | 0.8625 | 0.9000 | +4.35% | 0.8582 | 0.8995 |
| Complex Classification (synthetic) | RandomForest | 0.9125 | 0.9075 | -0.55% | 0.9103 | 0.9046 |
| Complex Classification (synthetic) | GradientBoosting | 0.9075 | 0.9225 | +1.65% | 0.9058 | 0.9216 |
| Stock Price Direction (time series) | LogisticRegression | 0.5633 | 0.5433 | -3.55% | 0.5051 | 0.4905 |
| Stock Price Direction (time series) | RandomForest | 0.5500 | 0.5100 | -7.27% | 0.5475 | 0.5084 |
| Stock Price Direction (time series) | GradientBoosting | 0.5400 | 0.4967 | -8.02% | 0.5326 | 0.4909 |

### Regression Datasets

| Dataset | Model | Baseline R² | FeatCopilot R² | Improvement | Baseline RMSE | FeatCopilot RMSE |
|---------|-------|-------------|----------------|-------------|---------------|------------------|
| Diabetes (sklearn) | Ridge | 0.4541 | 0.4421 | -2.66% | 53.78 | 54.37 |
| Diabetes (sklearn) | RandomForest | 0.4415 | 0.4399 | -0.36% | 54.40 | 54.48 |
| Diabetes (sklearn) | GradientBoosting | 0.4530 | 0.3792 | -16.28% | 53.83 | 57.35 |
| House Prices (Kaggle-style) | Ridge | 0.9306 | 0.9297 | -0.10% | 20407.82 | 20548.55 |
| House Prices (Kaggle-style) | RandomForest | 0.8700 | 0.8942 | +2.78% | 27940.29 | 25205.06 |
| House Prices (Kaggle-style) | GradientBoosting | 0.9135 | 0.9090 | -0.49% | 22785.85 | 23371.12 |
| Bike Sharing (Kaggle-style) | Ridge | 0.7211 | 0.7753 | +7.52% | 39.98 | 35.88 |
| Bike Sharing (Kaggle-style) | RandomForest | 0.8074 | 0.8019 | -0.69% | 33.22 | 33.69 |
| Bike Sharing (Kaggle-style) | GradientBoosting | 0.8305 | 0.8321 | +0.20% | 31.17 | 31.01 |
| Complex Regression (synthetic) | Ridge | 0.9967 | 0.9966 | -0.01% | 10.43 | 10.53 |
| Complex Regression (synthetic) | RandomForest | 0.8117 | 0.7995 | -1.50% | 78.85 | 81.37 |
| Complex Regression (synthetic) | GradientBoosting | 0.9137 | 0.9132 | -0.05% | 53.39 | 53.54 |
| Energy Consumption (time series) | Ridge | 0.3895 | 0.3824 | -1.83% | 10.24 | 10.30 |
| Energy Consumption (time series) | RandomForest | 0.7974 | 0.8099 | +1.57% | 5.90 | 5.71 |
| Energy Consumption (time series) | GradientBoosting | 0.8357 | 0.8338 | -0.23% | 5.31 | 5.34 |
| Website Traffic (time series) | Ridge | 0.7179 | 0.7153 | -0.35% | 262.39 | 263.57 |
| Website Traffic (time series) | RandomForest | 0.8447 | 0.8485 | +0.46% | 194.70 | 192.26 |
| Website Traffic (time series) | GradientBoosting | 0.9076 | 0.9132 | +0.61% | 150.14 | 145.54 |

## Feature Engineering Statistics

| Dataset | Original Features | Engineered Features | FE Time (s) |
|---------|-------------------|---------------------|-------------|
| Diabetes (sklearn) | 10 | 31 | 0.16 |
| Breast Cancer (sklearn) | 30 | 48 | 0.54 |
| Titanic (Kaggle-style) | 7 | 25 | 0.41 |
| House Prices (Kaggle-style) | 14 | 37 | 0.40 |
| Credit Card Fraud (Kaggle-style) | 30 | 50 | 1.68 |
| Bike Sharing (Kaggle-style) | 10 | 27 | 0.37 |
| Employee Attrition (IBM HR) | 11 | 40 | 0.58 |
| Credit Risk (synthetic) | 10 | 40 | 0.64 |
| Medical Diagnosis (synthetic) | 12 | 47 | 0.62 |
| Complex Regression (synthetic) | 15 | 40 | 0.54 |
| Complex Classification (synthetic) | 15 | 50 | 0.75 |
| Energy Consumption (time series) | 9 | 21 | 0.34 |
| Stock Price Direction (time series) | 10 | 10 | 0.56 |
| Website Traffic (time series) | 9 | 23 | 0.34 |

## Methodology

- **Train/Test Split**: 80/20 with random_state=42
- **Feature Engineering**: FeatCopilot TabularEngine
  - Classification: importance + mutual_info selection, max 50 features
  - Regression: mutual_info selection, max 30 features, correlation_threshold=0.90
- **Preprocessing**: StandardScaler applied to all features
- **Models**: LogisticRegression/Ridge, RandomForest, GradientBoosting
- **Metrics**: Accuracy/R², F1-score/RMSE, ROC-AUC/MAE

## Datasets

### Real-world Datasets
- Diabetes (sklearn) - Medical regression
- Breast Cancer (sklearn) - Medical classification
- Titanic (Kaggle-style) - Survival classification
- House Prices (Kaggle-style) - Price regression
- Credit Card Fraud (Kaggle-style) - Imbalanced classification
- Bike Sharing (Kaggle-style) - Demand regression
- Employee Attrition (IBM HR) - HR classification

### Synthetic Datasets
- Credit Risk - Financial classification
- Medical Diagnosis - Healthcare classification
- Complex Regression - Non-linear regression
- Complex Classification - Imbalanced classification

### Time Series Datasets
- Energy Consumption - Forecasting regression
- Stock Price Direction - Movement classification
- Website Traffic - Traffic regression
