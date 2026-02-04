# Simple Models Benchmark Report

**Generated:** 2026-02-04 17:38:59
**Models:** RandomForest, LogisticRegression/Ridge
**LLM Enabled:** False
**Datasets:** 52

## Summary

| Metric | Value |
|--------|-------|
| Total Datasets | 52 |
| Classification | 22 |
| Regression | 20 |
| Forecasting | 3 |
| Text Classification | 5 |
| Text Regression | 2 |
| Improved (Tabular) | 25 |
| Avg Improvement | 3.63% |

## Classification Results

| Dataset | Baseline | Tabular | Improvement | Features |
|---------|----------|---------|-------------|----------|
| titanic | 0.9218 | 0.9162 | -0.61% | 7→27 |
| credit_card_fraud | 0.9840 | 0.9840 | +0.00% | 30→100 |
| employee_attrition | 0.9558 | 0.9558 | +0.00% | 11→74 |
| credit_risk | 0.7125 | 0.7075 | -0.70% | 10→86 |
| medical_diagnosis | 0.8667 | 0.8600 | -0.77% | 12→72 |
| complex_classification | 0.8800 | 0.9300 | +5.68% | 15→100 |
| customer_churn | 0.7575 | 0.7650 | +0.99% | 10→82 |
| higgs | 0.7129 | 0.7081 | -0.67% | 24→100 |
| covertype | 0.8675 | 0.8481 | -2.24% | 10→54 |
| jannis | 0.7863 | 0.7850 | -0.17% | 54→100 |
| miniboone | 0.9307 | 0.9279 | -0.30% | 50→57 |
| california | 0.8861 | 0.8585 | -3.12% | 8→43 |
| credit | 0.7774 | 0.7757 | -0.23% | 10→67 |
| bank_marketing | 0.8043 | 0.7935 | -1.35% | 7→54 |
| diabetes | 0.6074 | 0.5990 | -1.38% | 7→66 |
| bioresponse | 0.7700 | 0.7729 | +0.38% | 419→419 |
| magic_telescope | 0.8509 | 0.8528 | +0.22% | 10→65 |
| electricity | 0.8984 | 0.8986 | +0.03% | 8→50 |
| covertype_cat | 0.8747 | 0.8749 | +0.02% | 54→100 |
| eye_movements | 0.6373 | 0.6222 | -2.37% | 23→97 |
| road_safety | 0.7815 | 0.7895 | +1.02% | 32→89 |
| albert | 0.6558 | 0.6591 | +0.50% | 31→100 |

## Regression Results

| Dataset | Baseline R² | Tabular R² | Improvement | Features |
|---------|-------------|------------|-------------|----------|
| house_prices | 0.9306 | 0.9308 | +0.02% | 14→46 |
| bike_sharing | 0.8080 | 0.8082 | +0.02% | 10→48 |
| complex_regression | 0.9967 | 0.9965 | -0.02% | 15→79 |
| insurance_claims | 0.9639 | 0.9574 | -0.67% | 10→36 |
| spotify_tracks | 0.3898 | 0.3735 | -4.18% | 14→85 |
| diamonds | 0.9456 | 0.9461 | +0.05% | 6→19 |
| house_sales | 0.8785 | 0.8731 | -0.61% | 15→48 |
| houses | 0.8364 | 0.8246 | -1.41% | 8→40 |
| wine_quality | 0.4972 | 0.5027 | +1.12% | 11→45 |
| abalone | 0.5287 | 0.5762 | +8.98% | 7→30 |
| superconduct | 0.9300 | 0.9301 | +0.01% | 79→100 |
| cpu_act | 0.9798 | 0.9800 | +0.02% | 21→100 |
| elevators | 0.8318 | 0.7908 | -4.92% | 16→68 |
| miami_housing | 0.9146 | 0.9201 | +0.61% | 13→70 |
| bike_sharing_inria | 0.6788 | 0.6861 | +1.07% | 6→37 |
| delays_zurich | 0.0051 | 0.0153 | +197.48% | 11→57 |
| allstate_claims | 0.5013 | 0.5012 | -0.02% | 124→124 |
| mercedes_benz | 0.5572 | 0.5572 | +0.00% | 359→359 |
| nyc_taxi | 0.6391 | 0.6253 | -2.17% | 16→44 |
| brazilian_houses | 0.9960 | 0.9964 | +0.04% | 11→66 |

## Forecasting Results

| Dataset | Baseline R² | Tabular R² | Improvement | Features |
|---------|-------------|------------|-------------|----------|
| sensor_anomaly | 0.2773 | 0.2653 | -4.33% | 8→45 |
| retail_demand | 0.8738 | 0.8772 | +0.39% | 10→65 |
| server_latency | 0.9926 | 0.9928 | +0.02% | 8→23 |

## Text Classification Results

| Dataset | Baseline | Tabular | Improvement | Features |
|---------|----------|---------|-------------|----------|
| product_reviews | 0.9350 | 0.9375 | +0.27% | 6→40 |
| news_classification | 0.9300 | 0.9940 | +6.88% | 5→17 |
| customer_support | 1.0000 | 1.0000 | +0.00% | 6→18 |
| medical_notes | 0.9933 | 0.9767 | -1.68% | 5→26 |
| fake_news | 0.9597 | 0.9633 | +0.36% | 2→7 |

## Text Regression Results

| Dataset | Baseline R² | Tabular R² | Improvement | Features |
|---------|-------------|------------|-------------|----------|
| job_postings | 0.8699 | 0.8505 | -2.24% | 5→35 |
| ecommerce_product | 0.4584 | 0.4533 | -1.11% | 5→20 |
