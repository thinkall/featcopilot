# FeatCopilot Benchmark Report

## Summary

**Classification Tasks:**
- Average Accuracy Improvement: **+0.54%**
- Max Improvement: +4.35%
- Improvements > 0: 8/18 (44%)

**Regression Tasks:**
- Average R� Improvement: **+0.65%**
- Max Improvement: +5.57%
- Improvements > 0: 3/9 (33%)

**Time Series Tasks:**
- Regression Avg R� Improvement: **+1.51%** (7/9 wins)

**Text/Semantic Tasks (with Text Engine):**
- Classification Avg Improvement: **+12.44%** (12/12 wins)
- Max Improvement: +49.02%
- Regression Avg R� Improvement: **+1.44%** (3/6 wins)
- Max Improvement: +7.58%

## Detailed Results

### Classification Datasets

| Dataset | Model | Baseline Acc | FeatCopilot Acc | Improvement | Baseline F1 | FeatCopilot F1 |
|---------|-------|--------------|-----------------|-------------|-------------|----------------|
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
| Product Reviews (text) | LogisticRegression | 0.9175 | 0.9650 | +5.18% | 0.9174 | 0.9649 |
| Product Reviews (text) | RandomForest | 0.9225 | 0.9625 | +4.34% | 0.9222 | 0.9623 |
| Product Reviews (text) | GradientBoosting | 0.9200 | 0.9575 | +4.08% | 0.9200 | 0.9574 |
| News Headlines (text) | LogisticRegression | 0.4080 | 0.6080 | +49.02% | 0.3750 | 0.6075 |
| News Headlines (text) | RandomForest | 0.6440 | 0.8580 | +33.23% | 0.6448 | 0.8581 |
| News Headlines (text) | GradientBoosting | 0.6700 | 0.8560 | +27.76% | 0.6678 | 0.8558 |
| Customer Support Tickets (text) | LogisticRegression | 0.8600 | 0.9000 | +4.65% | 0.8547 | 0.8951 |
| Customer Support Tickets (text) | RandomForest | 0.9350 | 0.9925 | +6.15% | 0.9338 | 0.9925 |
| Customer Support Tickets (text) | GradientBoosting | 0.9475 | 0.9975 | +5.28% | 0.9474 | 0.9975 |
| Medical Notes (text) | LogisticRegression | 0.9233 | 0.9833 | +6.50% | 0.9220 | 0.9833 |
| Medical Notes (text) | RandomForest | 0.9733 | 0.9967 | +2.40% | 0.9730 | 0.9967 |
| Medical Notes (text) | GradientBoosting | 0.9800 | 0.9867 | +0.68% | 0.9799 | 0.9868 |

### Regression Datasets

| Dataset | Model | Baseline R� | FeatCopilot R� | Improvement | Baseline RMSE | FeatCopilot RMSE |
|---------|-------|-------------|----------------|-------------|---------------|------------------|
| House Prices (Kaggle-style) | Ridge | 0.9306 | 0.9297 | -0.10% | 20407.82 | 20546.02 |
| House Prices (Kaggle-style) | RandomForest | 0.8700 | 0.8941 | +2.77% | 27940.29 | 25219.52 |
| House Prices (Kaggle-style) | GradientBoosting | 0.9135 | 0.9099 | -0.40% | 22785.85 | 23261.81 |
| Bike Sharing (Kaggle-style) | Ridge | 0.7211 | 0.7613 | +5.57% | 39.98 | 36.99 |
| Bike Sharing (Kaggle-style) | RandomForest | 0.8074 | 0.8025 | -0.61% | 33.22 | 33.64 |
| Bike Sharing (Kaggle-style) | GradientBoosting | 0.8305 | 0.8323 | +0.22% | 31.17 | 31.00 |
| Complex Regression (synthetic) | Ridge | 0.9967 | 0.9966 | -0.01% | 10.43 | 10.53 |
| Complex Regression (synthetic) | RandomForest | 0.8117 | 0.7995 | -1.50% | 78.85 | 81.37 |
| Complex Regression (synthetic) | GradientBoosting | 0.9137 | 0.9132 | -0.05% | 53.39 | 53.54 |
| Sensor Efficiency (time series) | Ridge | 0.0880 | 0.0824 | -6.34% | 2.32 | 2.33 |
| Sensor Efficiency (time series) | RandomForest | 0.2751 | 0.2843 | +3.34% | 2.07 | 2.05 |
| Sensor Efficiency (time series) | GradientBoosting | 0.2596 | 0.2728 | +5.10% | 2.09 | 2.07 |
| Retail Demand (time series) | Ridge | 0.7145 | 0.8011 | +12.12% | 22.34 | 18.64 |
| Retail Demand (time series) | RandomForest | 0.8734 | 0.8670 | -0.74% | 14.87 | 15.25 |
| Retail Demand (time series) | GradientBoosting | 0.9166 | 0.9172 | +0.06% | 12.07 | 12.03 |
| Server Latency (time series) | Ridge | 0.9724 | 0.9727 | +0.03% | 16.08 | 16.01 |
| Server Latency (time series) | RandomForest | 0.9926 | 0.9927 | +0.01% | 8.30 | 8.25 |
| Server Latency (time series) | GradientBoosting | 0.9949 | 0.9949 | +0.00% | 6.94 | 6.92 |
| Job Postings (text) | Ridge | 0.3885 | 0.4179 | +7.58% | 49094.19 | 47898.14 |
| Job Postings (text) | RandomForest | 0.8627 | 0.8408 | -2.53% | 23265.97 | 25049.07 |
| Job Postings (text) | GradientBoosting | 0.8835 | 0.8595 | -2.72% | 21432.15 | 23535.03 |
| E-commerce Products (text) | Ridge | 0.4626 | 0.4626 | -0.00% | 20.41 | 20.41 |
| E-commerce Products (text) | RandomForest | 0.3890 | 0.4024 | +3.45% | 21.76 | 21.52 |
| E-commerce Products (text) | GradientBoosting | 0.4539 | 0.4669 | +2.86% | 20.57 | 20.32 |

## Feature Engineering Statistics

| Dataset | Original Features | Engineered Features | FE Time (s) |
|---------|-------------------|---------------------|-------------|
| Titanic (Kaggle-style) | 7 | 25 | 0.44 |
| House Prices (Kaggle-style) | 14 | 35 | 0.42 |
| Credit Card Fraud (Kaggle-style) | 30 | 50 | 1.70 |
| Bike Sharing (Kaggle-style) | 10 | 24 | 0.38 |
| Employee Attrition (IBM HR) | 11 | 40 | 0.57 |
| Credit Risk (synthetic) | 10 | 40 | 0.66 |
| Medical Diagnosis (synthetic) | 12 | 47 | 0.63 |
| Complex Regression (synthetic) | 15 | 40 | 0.54 |
| Complex Classification (synthetic) | 15 | 50 | 0.76 |
| Sensor Efficiency (time series) | 8 | 15 | 0.31 |
| Retail Demand (time series) | 10 | 33 | 0.40 |
| Server Latency (time series) | 8 | 18 | 0.30 |
| Product Reviews (text) | 6 | 32 | 0.62 |
| Job Postings (text) | 5 | 22 | 0.55 |
| News Headlines (text) | 5 | 23 | 0.79 |
| Customer Support Tickets (text) | 6 | 27 | 0.69 |
| Medical Notes (text) | 5 | 26 | 0.61 |
| E-commerce Products (text) | 5 | 27 | 0.71 |

## Methodology

- **Train/Test Split**: 80/20 with random_state=42
- **Feature Engineering**: FeatCopilot TabularEngine
  - Classification: importance + mutual_info selection, max 50 features
  - Regression: mutual_info selection, max 30 features, correlation_threshold=0.90
- **Preprocessing**: StandardScaler applied to all features
- **Models**: LogisticRegression/Ridge, RandomForest, GradientBoosting
- **Metrics**: Accuracy/R�, F1-score/RMSE, ROC-AUC/MAE

## Datasets

### Kaggle-style Datasets
- Titanic - Survival classification
- House Prices - Price regression
- Credit Card Fraud - Imbalanced classification
- Bike Sharing - Demand regression
- Employee Attrition (IBM HR) - HR classification

### Synthetic Datasets
- Credit Risk - Financial classification
- Medical Diagnosis - Healthcare classification
- Complex Regression - Non-linear regression
- Complex Classification - Multi-class classification

### Time Series Datasets
- Sensor Efficiency - Industrial IoT regression
- Retail Demand - Demand forecasting regression
- Server Latency - Performance prediction regression

### Text/Semantic Datasets
- Product Reviews - Sentiment classification
- News Headlines - Category classification
- Customer Support Tickets - Priority classification
- Medical Notes - Diagnosis classification
- Job Postings - Salary regression
- E-commerce Products - Rating regression
