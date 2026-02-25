# Simple Models Benchmark Report

**Generated:** 2026-02-25 19:25:00
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
| Improved (Tabular) | 34 |
| Avg Improvement | 8.14% |

## Classification Results

| Dataset | Baseline | Tabular | Improvement | Features |
|---------|----------|---------|-------------|----------|
| titanic | 0.8268 | 0.8212 | -0.68% | 7→8 |
| credit_card_fraud | 0.9840 | 0.9840 | +0.00% | 30→42 |
| employee_attrition | 0.9252 | 0.9252 | +0.00% | 11→20 |
| credit_risk | 0.8525 | 0.8650 | +1.47% | 10→27 |
| medical_diagnosis | 0.8500 | 0.8367 | -1.57% | 12→26 |
| complex_classification | 0.7125 | 0.8200 | +15.09% | 15→45 |
| interaction_classification | 0.7650 | 0.8275 | +8.17% | 12→34 |
| customer_churn | 0.7750 | 0.7575 | -2.26% | 10→27 |
| xor_classification | 0.6960 | 0.8040 | +15.52% | 20→49 |
| polynomial_classification | 0.7875 | 0.8700 | +10.48% | 15→45 |
| customer_support | 0.8900 | 0.8850 | -0.56% | 10→29 |
| higgs | 0.7129 | 0.6984 | -2.03% | 24→30 |
| covertype | 0.8675 | 0.8584 | -1.05% | 10→24 |
| jannis | 0.7863 | 0.7821 | -0.53% | 54→82 |
| miniboone | 0.9307 | 0.9292 | -0.16% | 50→53 |
| california | 0.8861 | 0.8692 | -1.91% | 8→18 |
| credit | 0.7774 | 0.7667 | -1.39% | 10→22 |
| bank_marketing | 0.8043 | 0.7916 | -1.59% | 7→18 |
| diabetes | 0.6074 | 0.5852 | -3.65% | 7→18 |
| bioresponse | 0.7700 | 0.7758 | +0.76% | 419→419 |
| magic_telescope | 0.8509 | 0.8516 | +0.09% | 10→18 |
| electricity | 0.8984 | 0.8945 | -0.43% | 8→20 |
| covertype_cat | 0.8747 | 0.8711 | -0.41% | 54→85 |
| eye_movements | 0.6373 | 0.6445 | +1.13% | 23→48 |
| road_safety | 0.7815 | 0.7865 | +0.64% | 32→53 |
| albert | 0.6558 | 0.6606 | +0.73% | 31→57 |

## Regression Results

| Dataset | Baseline R² | Tabular R² | Improvement | Features |
|---------|-------------|------------|-------------|----------|
| house_prices | 0.9798 | 0.9954 | +1.59% | 14→35 |
| bike_sharing | 0.9534 | 0.9685 | +1.58% | 10→29 |
| complex_regression | 0.6339 | 0.9559 | +50.79% | 15→45 |
| polynomial_regression | 0.7321 | 0.9135 | +24.78% | 12→36 |
| ratio_regression | 0.9689 | 0.9821 | +1.35% | 12→36 |
| nonlinear_regression | 0.6086 | 0.9152 | +50.39% | 12→36 |
| insurance_claims | 0.9621 | 0.9559 | -0.64% | 10→22 |
| xor_regression | 0.3330 | 0.6795 | +104.03% | 20→60 |
| quadratic_heavy_regression | 0.7134 | 0.9377 | +31.44% | 18→54 |
| pairwise_product_regression | 0.5132 | 0.8628 | +68.12% | 16→48 |
| sqrt_log_regression | 0.8725 | 0.9164 | +5.02% | 15→45 |
| triple_interaction_regression | 0.3542 | 0.8574 | +142.08% | 18→54 |
| job_postings | 0.9685 | 0.9804 | +1.23% | 10→29 |
| ecommerce_product | 0.9462 | 0.9609 | +1.56% | 10→27 |
| spotify_tracks | 0.9529 | 0.9662 | +1.40% | 13→37 |
| diamonds | 0.9456 | 0.9460 | +0.04% | 6→12 |
| house_sales | 0.8785 | 0.8764 | -0.23% | 15→15 |
| houses | 0.8364 | 0.8380 | +0.20% | 8→9 |
| wine_quality | 0.4972 | 0.4921 | -1.02% | 11→13 |
| abalone | 0.5287 | 0.5319 | +0.61% | 7→8 |
| superconduct | 0.9300 | 0.9299 | -0.00% | 79→79 |
| cpu_act | 0.9798 | 0.9798 | +0.00% | 21→29 |
| elevators | 0.8318 | 0.8269 | -0.59% | 16→22 |
| miami_housing | 0.9146 | 0.9192 | +0.50% | 13→15 |
| bike_sharing_inria | 0.6788 | 0.6657 | -1.92% | 6→15 |
| delays_zurich | 0.0051 | 0.0051 | -0.00% | 11→11 |
| allstate_claims | 0.5013 | 0.5008 | -0.09% | 124→124 |
| mercedes_benz | 0.5572 | 0.5572 | -0.00% | 359→359 |
| nyc_taxi | 0.6391 | 0.6255 | -2.13% | 16→33 |
| brazilian_houses | 0.9960 | 0.9963 | +0.03% | 11→25 |

## Forecasting Results

| Dataset | Baseline R² | Tabular R² | Improvement | Features |
|---------|-------------|------------|-------------|----------|
| sensor_anomaly | 0.8709 | 0.8726 | +0.20% | 8→19 |
| retail_demand | 0.8738 | 0.8779 | +0.47% | 10→28 |
| server_latency | 0.9926 | 0.9927 | +0.01% | 8→23 |

## Text Classification Results

| Dataset | Baseline | Tabular | Improvement | Features |
|---------|----------|---------|-------------|----------|
| product_reviews | 0.9350 | 0.9200 | -1.60% | 6→16 |
| news_classification | 0.8720 | 0.8740 | +0.23% | 7→17 |
| medical_notes | 0.7400 | 0.7300 | -1.35% | 5→6 |
| fake_news | 0.9597 | 0.9497 | -1.04% | 2→8 |
