# Simple Models Benchmark Report

**Generated:** 2026-02-26 14:10:12
**Models:** RandomForest, LogisticRegression/Ridge
**LLM Enabled:** False
**Datasets:** 63

## Summary

| Metric | Value |
|--------|-------|
| Total Datasets | 63 |
| Classification | 26 |
| Regression | 30 |
| Forecasting | 3 |
| Text Classification | 4 |
| Text Regression | 0 |
| Improved (Tabular) | 31 |
| Avg Improvement | 7.52% |

## Classification Results

| Dataset | Baseline | Tabular | Improvement | Features |
|---------|----------|---------|-------------|----------|
| titanic | 0.8268 | 0.8101 | -2.03% | 7→8 |
| credit_card_fraud | 0.9840 | 0.9840 | +0.00% | 30→40 |
| employee_attrition | 0.9252 | 0.9252 | +0.00% | 11→16 |
| credit_risk | 0.8525 | 0.8675 | +1.76% | 10→17 |
| medical_diagnosis | 0.8500 | 0.8367 | -1.57% | 12→21 |
| complex_classification | 0.7125 | 0.8300 | +16.49% | 15→23 |
| interaction_classification | 0.7650 | 0.8075 | +5.56% | 12→17 |
| customer_churn | 0.7750 | 0.7600 | -1.94% | 10→15 |
| xor_classification | 0.6960 | 0.8120 | +16.67% | 20→24 |
| polynomial_classification | 0.7875 | 0.8675 | +10.16% | 15→21 |
| customer_support | 0.8900 | 0.8825 | -0.84% | 10→13 |
| higgs | 0.7129 | 0.7003 | -1.77% | 24→27 |
| covertype | 0.8675 | 0.8414 | -3.01% | 10→13 |
| jannis | 0.7863 | 0.7853 | -0.13% | 54→57 |
| miniboone | 0.9307 | 0.9305 | -0.02% | 50→52 |
| california | 0.8861 | 0.8653 | -2.35% | 8→8 |
| credit | 0.7774 | 0.7559 | -2.77% | 10→11 |
| bank_marketing | 0.8043 | 0.7873 | -2.12% | 7→11 |
| diabetes | 0.6074 | 0.5807 | -4.40% | 7→10 |
| bioresponse | 0.7700 | 0.7802 | +1.32% | 419→419 |
| magic_telescope | 0.8509 | 0.8572 | +0.75% | 10→12 |
| electricity | 0.8984 | 0.8738 | -2.73% | 8→10 |
| covertype_cat | 0.8747 | 0.8819 | +0.82% | 54→55 |
| eye_movements | 0.6373 | 0.6393 | +0.31% | 23→42 |
| road_safety | 0.7815 | 0.7723 | -1.18% | 32→27 |
| albert | 0.6558 | 0.6522 | -0.55% | 31→38 |

## Regression Results

| Dataset | Baseline R² | Tabular R² | Improvement | Features |
|---------|-------------|------------|-------------|----------|
| house_prices | 0.9798 | 0.9953 | +1.58% | 14→16 |
| bike_sharing | 0.9534 | 0.9697 | +1.71% | 10→12 |
| complex_regression | 0.6339 | 0.8725 | +37.63% | 15→20 |
| polynomial_regression | 0.7321 | 0.8692 | +18.72% | 12→19 |
| ratio_regression | 0.9689 | 0.9784 | +0.98% | 12→19 |
| nonlinear_regression | 0.6086 | 0.8756 | +43.87% | 12→18 |
| insurance_claims | 0.9621 | 0.9644 | +0.24% | 10→10 |
| xor_regression | 0.3330 | 0.6801 | +104.23% | 20→24 |
| quadratic_heavy_regression | 0.7134 | 0.9341 | +30.94% | 18→25 |
| pairwise_product_regression | 0.5132 | 0.8698 | +69.48% | 16→23 |
| sqrt_log_regression | 0.8725 | 0.8997 | +3.12% | 15→25 |
| triple_interaction_regression | 0.3542 | 0.8649 | +144.18% | 18→23 |
| job_postings | 0.9685 | 0.9735 | +0.52% | 10→14 |
| ecommerce_product | 0.9462 | 0.9564 | +1.08% | 10→11 |
| spotify_tracks | 0.9529 | 0.9648 | +1.25% | 13→17 |
| diamonds | 0.9456 | 0.9404 | -0.56% | 6→4 |
| house_sales | 0.8785 | 0.8752 | -0.37% | 15→11 |
| houses | 0.8364 | 0.8381 | +0.20% | 8→9 |
| wine_quality | 0.4972 | 0.4914 | -1.15% | 11→13 |
| abalone | 0.5287 | 0.5319 | +0.61% | 7→8 |
| superconduct | 0.9300 | 0.9302 | +0.02% | 79→79 |
| cpu_act | 0.9798 | 0.9783 | -0.15% | 21→13 |
| elevators | 0.8318 | 0.8288 | -0.36% | 16→20 |
| miami_housing | 0.9146 | 0.9193 | +0.52% | 13→15 |
| bike_sharing_inria | 0.6788 | 0.6530 | -3.80% | 6→7 |
| delays_zurich | 0.0051 | 0.0051 | -0.00% | 11→11 |
| allstate_claims | 0.5013 | 0.5013 | -0.01% | 124→124 |
| mercedes_benz | 0.5572 | 0.5572 | -0.00% | 359→359 |
| nyc_taxi | 0.6391 | 0.6381 | -0.17% | 16→13 |
| brazilian_houses | 0.9960 | 0.9964 | +0.04% | 11→13 |

## Forecasting Results

| Dataset | Baseline R² | Tabular R² | Improvement | Features |
|---------|-------------|------------|-------------|----------|
| sensor_anomaly | 0.8709 | 0.8720 | +0.12% | 8→8 |
| retail_demand | 0.8738 | 0.8615 | -1.41% | 10→13 |
| server_latency | 0.9926 | 0.9925 | -0.02% | 8→8 |

## Text Classification Results

| Dataset | Baseline | Tabular | Improvement | Features |
|---------|----------|---------|-------------|----------|
| product_reviews | 0.9350 | 0.9075 | -2.94% | 6→7 |
| news_classification | 0.8720 | 0.8480 | -2.75% | 7→13 |
| medical_notes | 0.7400 | 0.7367 | -0.45% | 5→5 |
| fake_news | 0.9597 | 0.9635 | +0.39% | 2→3 |
