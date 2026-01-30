# AutoML Integration Benchmark Report
## Overview
This benchmark evaluates FeatCopilot's impact on AutoML framework performance.
Time budget: 30s per AutoML run (excludes FeatCopilot preprocessing time)

## Summary
- **Datasets tested**: 18
- **Frameworks tested**: 3
- **Average improvement**: -0.94%
- **Positive improvements**: 8/51 (15.7%)

## Results by Framework

### FLAML
| Dataset | Task | Baseline | +FeatCopilot | Improvement | Train Time (B/E) | Predict Time (B/E) |
|---------|------|----------|--------------|-------------|------------------|--------------------|
| Titanic (Kaggle-style) | classification | 0.9665 | 0.9665 | +0.00% | 30.1s / 30.3s | 0.01s / 0.05s |
| House Prices (Kaggle-style) | regression | 0.9242 | 0.9136 | -1.15% | 30.4s / 30.1s | 0.01s / 0.02s |
| Credit Card Fraud (Kaggle-style) | classification | 0.9860 | 0.9860 | +0.00% | 30.0s / 30.2s | 0.03s / 0.05s |
| Bike Sharing (Kaggle-style) | regression | 0.8426 | 0.8359 | -0.79% | 30.8s / 30.8s | 0.01s / 0.01s |
| Employee Attrition (IBM HR) | classification | 0.9796 | 0.9762 | -0.35% | 30.2s / 30.5s | 0.02s / 0.02s |
| Credit Risk (synthetic) | classification | 0.7000 | 0.6450 | -7.86% | 30.1s / 30.2s | 0.02s / 0.04s |
| Medical Diagnosis (synthetic) | classification | 0.8567 | 0.8600 | +0.39% | 30.7s / 30.6s | 0.01s / 0.03s |
| Complex Regression (synthetic) | regression | 0.9807 | 0.9757 | -0.50% | 31.9s / 34.0s | 0.01s / 0.01s |
| Complex Classification (synthetic) | classification | 0.9450 | 0.9450 | +0.00% | 31.6s / 34.5s | 0.01s / 0.03s |
| Product Reviews (text) | text_classification | 0.9400 | 0.9375 | -0.27% | 31.4s / 30.2s | 0.02s / 0.03s |
| Job Postings (text) | text_regression | 0.8942 | 0.8913 | -0.32% | 30.1s / 30.5s | 0.01s / 0.01s |
| News Headlines (text) | text_classification | 0.9800 | 0.9820 | +0.20% | 34.7s / 54.0s | 0.01s / 0.03s |
| Customer Support Tickets (text) | text_classification | 1.0000 | 1.0000 | +0.00% | 412.9s / 608.6s | 0.02s / 0.03s |
| Medical Notes (text) | text_classification | 0.9933 | 0.9900 | -0.34% | 30.2s / 30.5s | 0.02s / 0.03s |
| E-commerce Products (text) | text_regression | 0.4626 | 0.4658 | +0.69% | 30.3s / 30.2s | 0.01s / 0.01s |

**Average improvement with flaml**: -0.69%

### AUTOGLUON
| Dataset | Task | Baseline | +FeatCopilot | Improvement | Train Time (B/E) | Predict Time (B/E) |
|---------|------|----------|--------------|-------------|------------------|--------------------|
| Titanic (Kaggle-style) | classification | 0.9665 | 0.9497 | -1.73% | 14.7s / 11.4s | 0.04s / 0.02s |
| House Prices (Kaggle-style) | regression | 0.9202 | 0.9163 | -0.42% | 21.5s / 20.4s | 0.04s / 0.26s |
| Credit Card Fraud (Kaggle-style) | classification | 0.9860 | 0.9860 | +0.00% | 16.1s / 17.4s | 0.04s / 0.04s |
| Bike Sharing (Kaggle-style) | regression | 0.8421 | 0.8340 | -0.97% | 28.9s / 22.0s | 0.06s / 0.05s |
| Employee Attrition (IBM HR) | classification | 0.9762 | 0.9694 | -0.70% | 12.6s / 13.2s | 0.02s / 0.05s |
| Credit Risk (synthetic) | classification | 0.7200 | 0.7075 | -1.74% | 23.3s / 17.0s | 0.05s / 0.01s |
| Medical Diagnosis (synthetic) | classification | 0.8700 | 0.8567 | -1.53% | 13.7s / 14.3s | 0.03s / 0.32s |
| Complex Regression (synthetic) | regression | 0.9771 | 0.9568 | -2.08% | 31.1s / 38.0s | 0.02s / 0.15s |
| Complex Classification (synthetic) | classification | 0.9500 | 0.9125 | -3.95% | 31.1s / 24.6s | 0.02s / 0.10s |
| Sensor Efficiency (time series) | timeseries_regression | 0.2567 | 0.2387 | -6.99% | 13.9s / 16.6s | 0.04s / 0.05s |
| Retail Demand (time series) | timeseries_regression | 0.9261 | 0.9232 | -0.31% | 37.9s / 31.3s | 0.04s / 0.19s |
| Server Latency (time series) | timeseries_regression | 0.9956 | 0.9957 | +0.01% | 31.1s / 31.2s | 0.22s / 0.31s |
| Product Reviews (text) | text_classification | 0.9300 | 0.9250 | -0.54% | 15.8s / 20.2s | 0.05s / 0.03s |
| Job Postings (text) | text_regression | 0.8902 | 0.8821 | -0.90% | 31.1s / 13.3s | 0.17s / 0.03s |
| News Headlines (text) | text_classification | 0.9840 | 0.9960 | +1.22% | 31.2s / 31.2s | 0.34s / 0.24s |
| Customer Support Tickets (text) | text_classification | 1.0000 | 1.0000 | +0.00% | 38.5s / 31.4s | 0.01s / 0.10s |
| Medical Notes (text) | text_classification | 0.9933 | 0.9900 | -0.34% | 31.6s / 31.4s | 0.05s / 0.05s |
| E-commerce Products (text) | text_regression | 0.4671 | 0.4546 | -2.67% | 13.8s / 16.2s | 0.04s / 0.06s |

**Average improvement with autogluon**: -1.31%

### H2O
| Dataset | Task | Baseline | +FeatCopilot | Improvement | Train Time (B/E) | Predict Time (B/E) |
|---------|------|----------|--------------|-------------|------------------|--------------------|
| Titanic (Kaggle-style) | classification | 0.8939 | 0.8603 | -3.75% | 32.2s / 91.3s | 0.47s / 40.89s |
| House Prices (Kaggle-style) | regression | 0.9113 | 0.9104 | -0.10% | 85.2s / 84.5s | 20.49s / 20.49s |
| Credit Card Fraud (Kaggle-style) | classification | 0.9850 | 0.9830 | -0.20% | 85.9s / 85.0s | 41.07s / 41.18s |
| Bike Sharing (Kaggle-style) | regression | 0.8364 | 0.8282 | -0.98% | 84.3s / 83.7s | 20.83s / 20.49s |
| Employee Attrition (IBM HR) | classification | 0.9728 | 0.9728 | +0.00% | 85.7s / 86.6s | 41.05s / 41.05s |
| Credit Risk (synthetic) | classification | 0.7075 | 0.7100 | +0.35% | 84.4s / 87.8s | 40.88s / 41.62s |
| Medical Diagnosis (synthetic) | classification | 0.8567 | 0.8167 | -4.67% | 87.0s / 86.1s | 40.98s / 40.97s |
| Complex Regression (synthetic) | regression | 0.9967 | 0.9967 | -0.00% | 83.8s / 83.6s | 20.42s / 20.51s |
| Complex Classification (synthetic) | classification | 0.9425 | 0.9400 | -0.27% | 86.0s / 86.6s | 41.02s / 41.61s |
| Sensor Efficiency (time series) | timeseries_regression | 0.3023 | 0.2904 | -3.93% | 85.0s / 84.7s | 20.45s / 20.47s |
| Retail Demand (time series) | timeseries_regression | 0.9171 | 0.9138 | -0.37% | 83.9s / 83.1s | 20.50s / 20.53s |
| Server Latency (time series) | timeseries_regression | 0.9740 | 0.9950 | +2.16% | 83.1s / 83.9s | 20.49s / 20.52s |
| Product Reviews (text) | text_classification | 0.9475 | 0.9450 | -0.26% | 85.3s / 85.3s | 41.00s / 41.00s |
| Job Postings (text) | text_regression | 0.8970 | 0.8919 | -0.57% | 85.5s / 83.7s | 20.46s / 20.46s |
| News Headlines (text) | text_classification | 0.9980 | 1.0000 | +0.20% | 85.5s / 85.6s | 40.95s / 40.99s |
| Customer Support Tickets (text) | text_classification | 1.0000 | 1.0000 | +0.00% | 85.2s / 85.4s | 40.95s / 41.03s |
| Medical Notes (text) | text_classification | 0.9967 | 0.9933 | -0.33% | 86.0s / 86.4s | 41.03s / 40.97s |
| E-commerce Products (text) | text_regression | 0.4627 | 0.4573 | -1.17% | 82.7s / 85.9s | 20.44s / 20.50s |

**Average improvement with h2o**: -0.77%

## Detailed Results
| Dataset | Framework | Baseline | +FeatCopilot | Improvement | Features | FE Time | Train Time | Predict Time |
|---------|-----------|----------|--------------|-------------|----------|---------|------------|---------------|
| Titanic (Kaggle-style) | flaml | 0.9665 | 0.9665 | +0.00% | 7->27 | 0.5s | 30.3s | 0.05s |
| Titanic (Kaggle-style) | autogluon | 0.9665 | 0.9497 | -1.73% | 7->27 | 0.5s | 11.4s | 0.02s |
| Titanic (Kaggle-style) | h2o | 0.8939 | 0.8603 | -3.75% | 7->27 | 0.5s | 91.3s | 40.89s |
| House Prices (Kaggle-style) | flaml | 0.9242 | 0.9136 | -1.15% | 14->50 | 1.1s | 30.1s | 0.02s |
| House Prices (Kaggle-style) | autogluon | 0.9202 | 0.9163 | -0.42% | 14->50 | 1.1s | 20.4s | 0.26s |
| House Prices (Kaggle-style) | h2o | 0.9113 | 0.9104 | -0.10% | 14->50 | 1.1s | 84.5s | 20.49s |
| Credit Card Fraud (Kaggle-style) | flaml | 0.9860 | 0.9860 | +0.00% | 30->50 | 1.7s | 30.2s | 0.05s |
| Credit Card Fraud (Kaggle-style) | autogluon | 0.9860 | 0.9860 | +0.00% | 30->50 | 1.7s | 17.4s | 0.04s |
| Credit Card Fraud (Kaggle-style) | h2o | 0.9850 | 0.9830 | -0.20% | 30->50 | 1.7s | 85.0s | 41.18s |
| Bike Sharing (Kaggle-style) | flaml | 0.8426 | 0.8359 | -0.79% | 10->41 | 1.2s | 30.8s | 0.01s |
| Bike Sharing (Kaggle-style) | autogluon | 0.8421 | 0.8340 | -0.97% | 10->41 | 1.2s | 22.0s | 0.05s |
| Bike Sharing (Kaggle-style) | h2o | 0.8364 | 0.8282 | -0.98% | 10->41 | 1.2s | 83.7s | 20.49s |
| Employee Attrition (IBM HR) | flaml | 0.9796 | 0.9762 | -0.35% | 11->41 | 0.6s | 30.5s | 0.02s |
| Employee Attrition (IBM HR) | autogluon | 0.9762 | 0.9694 | -0.70% | 11->41 | 0.6s | 13.2s | 0.05s |
| Employee Attrition (IBM HR) | h2o | 0.9728 | 0.9728 | +0.00% | 11->41 | 0.6s | 86.6s | 41.05s |
| Credit Risk (synthetic) | flaml | 0.7000 | 0.6450 | -7.86% | 10->50 | 0.7s | 30.2s | 0.04s |
| Credit Risk (synthetic) | autogluon | 0.7200 | 0.7075 | -1.74% | 10->50 | 0.7s | 17.0s | 0.01s |
| Credit Risk (synthetic) | h2o | 0.7075 | 0.7100 | +0.35% | 10->50 | 0.7s | 87.8s | 41.62s |
| Medical Diagnosis (synthetic) | flaml | 0.8567 | 0.8600 | +0.39% | 12->48 | 0.6s | 30.6s | 0.03s |
| Medical Diagnosis (synthetic) | autogluon | 0.8700 | 0.8567 | -1.53% | 12->48 | 0.6s | 14.3s | 0.32s |
| Medical Diagnosis (synthetic) | h2o | 0.8567 | 0.8167 | -4.67% | 12->48 | 0.6s | 86.1s | 40.97s |
| Complex Regression (synthetic) | flaml | 0.9807 | 0.9757 | -0.50% | 15->50 | 1.4s | 34.0s | 0.01s |
| Complex Regression (synthetic) | autogluon | 0.9771 | 0.9568 | -2.08% | 15->50 | 1.4s | 38.0s | 0.15s |
| Complex Regression (synthetic) | h2o | 0.9967 | 0.9967 | -0.00% | 15->50 | 1.4s | 83.6s | 20.51s |
| Complex Classification (synthetic) | flaml | 0.9450 | 0.9450 | +0.00% | 15->50 | 0.8s | 34.5s | 0.03s |
| Complex Classification (synthetic) | autogluon | 0.9500 | 0.9125 | -3.95% | 15->50 | 0.8s | 24.6s | 0.10s |
| Complex Classification (synthetic) | h2o | 0.9425 | 0.9400 | -0.27% | 15->50 | 0.8s | 86.6s | 41.61s |
| Sensor Efficiency (time series) | autogluon | 0.2567 | 0.2387 | -6.99% | 8->50 | 4.5s | 16.6s | 0.05s |
| Sensor Efficiency (time series) | h2o | 0.3023 | 0.2904 | -3.93% | 8->50 | 4.5s | 84.7s | 20.47s |
| Retail Demand (time series) | autogluon | 0.9261 | 0.9232 | -0.31% | 10->50 | 5.1s | 31.3s | 0.19s |
| Retail Demand (time series) | h2o | 0.9171 | 0.9138 | -0.37% | 10->50 | 5.1s | 83.1s | 20.53s |
| Server Latency (time series) | autogluon | 0.9956 | 0.9957 | +0.01% | 8->50 | 4.2s | 31.2s | 0.31s |
| Server Latency (time series) | h2o | 0.9740 | 0.9950 | +2.16% | 8->50 | 4.2s | 83.9s | 20.52s |
| Product Reviews (text) | flaml | 0.9400 | 0.9375 | -0.27% | 6->38 | 0.7s | 30.2s | 0.03s |
| Product Reviews (text) | autogluon | 0.9300 | 0.9250 | -0.54% | 6->38 | 0.7s | 20.2s | 0.03s |
| Product Reviews (text) | h2o | 0.9475 | 0.9450 | -0.26% | 6->38 | 0.7s | 85.3s | 41.00s |
| Job Postings (text) | flaml | 0.8942 | 0.8913 | -0.32% | 5->34 | 0.8s | 30.5s | 0.01s |
| Job Postings (text) | autogluon | 0.8902 | 0.8821 | -0.90% | 5->34 | 0.8s | 13.3s | 0.03s |
| Job Postings (text) | h2o | 0.8970 | 0.8919 | -0.57% | 5->34 | 0.8s | 83.7s | 20.46s |
| News Headlines (text) | flaml | 0.9800 | 0.9820 | +0.20% | 5->19 | 0.7s | 54.0s | 0.03s |
| News Headlines (text) | autogluon | 0.9840 | 0.9960 | +1.22% | 5->19 | 0.7s | 31.2s | 0.24s |
| News Headlines (text) | h2o | 0.9980 | 1.0000 | +0.20% | 5->19 | 0.7s | 85.6s | 40.99s |
| Customer Support Tickets (text) | flaml | 1.0000 | 1.0000 | +0.00% | 6->26 | 0.6s | 608.6s | 0.03s |
| Customer Support Tickets (text) | autogluon | 1.0000 | 1.0000 | +0.00% | 6->26 | 0.6s | 31.4s | 0.10s |
| Customer Support Tickets (text) | h2o | 1.0000 | 1.0000 | +0.00% | 6->26 | 0.6s | 85.4s | 41.03s |
| Medical Notes (text) | flaml | 0.9933 | 0.9900 | -0.34% | 5->29 | 0.7s | 30.5s | 0.03s |
| Medical Notes (text) | autogluon | 0.9933 | 0.9900 | -0.34% | 5->29 | 0.7s | 31.4s | 0.05s |
| Medical Notes (text) | h2o | 0.9967 | 0.9933 | -0.33% | 5->29 | 0.7s | 86.4s | 40.97s |
| E-commerce Products (text) | flaml | 0.4626 | 0.4658 | +0.69% | 5->26 | 1.3s | 30.2s | 0.01s |
| E-commerce Products (text) | autogluon | 0.4671 | 0.4546 | -2.67% | 5->26 | 1.3s | 16.2s | 0.06s |
| E-commerce Products (text) | h2o | 0.4627 | 0.4573 | -1.17% | 5->26 | 1.3s | 85.9s | 20.50s |
