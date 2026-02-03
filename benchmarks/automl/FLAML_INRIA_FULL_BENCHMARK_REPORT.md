# FLAML INRIA-SODA Tabular Benchmark Report
**Generated:** 2026-02-03 15:39:28
**Time Budget:** 60s per FLAML run
**Max Samples:** 50,000

## Summary

| Dataset | Task | Samples | Features | Baseline | +FeatCopilot | Improvement | FE Time |
|---------|------|---------|----------|----------|--------------|-------------|--------|
| higgs | class | 50,000 | 24→100 | 0.7209 | 0.7228 | +0.26% | 33.4s |
| covertype | class | 50,000 | 10→54 | 0.8884 | 0.8848 | -0.40% | 31.7s |
| jannis | class | 50,000 | 54→100 | 0.7898 | 0.7932 | +0.43% | 35.7s |
| miniboone | class | 50,000 | 50→18 | 0.9427 | 0.9053 | -3.96% | 154.6s |
| california | class | 20,634 | 8→43 | 0.9009 | 0.8820 | -2.10% | 10.2s |
| credit | class | 16,714 | 10→67 | 0.7840 | 0.7822 | -0.23% | 8.1s |
| bank_marketing | class | 10,578 | 7→54 | 0.8099 | 0.8010 | -1.11% | 4.5s |
| diabetes | class | 50,000 | 7→66 | 0.6085 | 0.6106 | +0.34% | 23.0s |
| bioresponse | class | 3,434 | 419→418 | 0.7816 | 0.8020 | +2.62% | 11.2s |
| electricity | class | 38,474 | 8→50 | 0.9120 | 0.9037 | -0.91% | 20.2s |
| covertype_cat | class | 50,000 | 54→100 | 0.9152 | 0.9024 | -1.40% | 32.7s |
| eye_movements | class | 7,608 | 23→97 | 0.6649 | 0.6682 | +0.50% | 3.7s |
| road_safety | class | 50,000 | 32→93 | 0.7937 | 0.7956 | +0.24% | 26.4s |
| albert | class | 50,000 | 31→100 | 0.6627 | 0.6617 | -0.15% | 26.4s |
| diamonds | regre | 50,000 | 6→19 | 0.9475 | 0.9487 | +0.13% | 41.3s |
| house_sales | regre | 21,613 | 15→50 | 0.8942 | 0.8904 | -0.42% | 25.9s |
| houses | regre | 20,640 | 8→40 | 0.8342 | 0.8363 | +0.24% | 27.3s |
| wine_quality | regre | 6,497 | 11→45 | 0.4664 | 0.5001 | +7.24% | 7.1s |
| abalone | regre | 4,177 | 7→30 | 0.5395 | 0.5844 | +8.34% | 4.1s |
| superconduct | regre | 21,263 | 79→100 | 0.9312 | 0.9322 | +0.10% | 39.8s |
| cpu_act | regre | 8,192 | 21→100 | 0.9803 | 0.9831 | +0.29% | 10.0s |
| elevators | regre | 16,599 | 16→76 | 0.8991 | 0.8883 | -1.20% | 18.5s |
| miami_housing | regre | 13,932 | 13→70 | 0.9303 | 0.9310 | +0.08% | 17.7s |
| bike_sharing | regre | 17,379 | 6→37 | 0.7026 | 0.7054 | +0.40% | 11.8s |
| delays_zurich | regre | 50,000 | 11→58 | 0.0810 | 0.0853 | +5.32% | 55.8s |
| allstate_claims | regre | 50,000 | 124→122 | 0.5456 | 0.5452 | -0.07% | 101.5s |
| mercedes_benz | regre | 4,209 | 359→274 | 0.5830 | 0.5998 | +2.89% | 9.8s |
| nyc_taxi | regre | 50,000 | 16→62 | 0.6375 | 0.6194 | -2.85% | 50.9s |
| brazilian_houses | regre | 10,692 | 11→66 | 0.9973 | 0.9876 | -0.97% | 9.7s |

## Overall Statistics

- **Datasets tested:** 29
- **FeatCopilot improved:** 16 datasets
- **Baseline better:** 13 datasets
- **Average improvement:** +0.47%
- **Max improvement:** +8.34%
- **Min improvement:** -3.96%

## Detailed Results

### higgs - Higgs boson detection (physics)
- **Task:** classification
- **Samples:** 50,000
- **Features:** 24 → 100
- **FE Time:** 33.4s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| Accuracy | 0.7209 | 0.7228 |
| F1 (weighted) | 0.7209 | 0.7228 |
| Best Model | lgbm | xgboost |

### covertype - Forest cover type prediction
- **Task:** classification
- **Samples:** 50,000
- **Features:** 10 → 54
- **FE Time:** 31.7s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| Accuracy | 0.8884 | 0.8848 |
| F1 (weighted) | 0.8884 | 0.8848 |
| Best Model | xgboost | xgboost |

### jannis - Multi-class classification
- **Task:** classification
- **Samples:** 50,000
- **Features:** 54 → 100
- **FE Time:** 35.7s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| Accuracy | 0.7900 | 0.7934 |
| F1 (weighted) | 0.7898 | 0.7932 |
| Best Model | catboost | catboost |

### miniboone - Particle physics classification
- **Task:** classification
- **Samples:** 50,000
- **Features:** 50 → 18
- **FE Time:** 154.6s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| Accuracy | 0.9427 | 0.9054 |
| F1 (weighted) | 0.9427 | 0.9053 |
| Best Model | lgbm | catboost |

### california - California housing (binned)
- **Task:** classification
- **Samples:** 20,634
- **Features:** 8 → 43
- **FE Time:** 10.2s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| Accuracy | 0.9009 | 0.8820 |
| F1 (weighted) | 0.9009 | 0.8820 |
| Best Model | catboost | xgboost |

### credit - Credit approval prediction
- **Task:** classification
- **Samples:** 16,714
- **Features:** 10 → 67
- **FE Time:** 8.1s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| Accuracy | 0.7840 | 0.7822 |
| F1 (weighted) | 0.7840 | 0.7822 |
| Best Model | catboost | xgb_limitdepth |

### bank_marketing - Bank marketing response
- **Task:** classification
- **Samples:** 10,578
- **Features:** 7 → 54
- **FE Time:** 4.5s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| Accuracy | 0.8100 | 0.8010 |
| F1 (weighted) | 0.8099 | 0.8010 |
| Best Model | xgboost | lgbm |

### diabetes - Diabetes readmission
- **Task:** classification
- **Samples:** 50,000
- **Features:** 7 → 66
- **FE Time:** 23.0s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| Accuracy | 0.6125 | 0.6130 |
| F1 (weighted) | 0.6085 | 0.6106 |
| Best Model | extra_tree | extra_tree |

### bioresponse - Biological response prediction
- **Task:** classification
- **Samples:** 3,434
- **Features:** 419 → 418
- **FE Time:** 11.2s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| Accuracy | 0.7817 | 0.8020 |
| F1 (weighted) | 0.7816 | 0.8020 |
| Best Model | lgbm | xgboost |

### electricity - Electricity price direction
- **Task:** classification
- **Samples:** 38,474
- **Features:** 8 → 50
- **FE Time:** 20.2s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| Accuracy | 0.9120 | 0.9037 |
| F1 (weighted) | 0.9120 | 0.9037 |
| Best Model | xgb_limitdepth | lgbm |

### covertype_cat - Forest cover (categorical)
- **Task:** classification
- **Samples:** 50,000
- **Features:** 54 → 100
- **FE Time:** 32.7s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| Accuracy | 0.9152 | 0.9024 |
| F1 (weighted) | 0.9152 | 0.9024 |
| Best Model | lgbm | lgbm |

### eye_movements - Eye movement classification
- **Task:** classification
- **Samples:** 7,608
- **Features:** 23 → 97
- **FE Time:** 3.7s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| Accuracy | 0.6649 | 0.6682 |
| F1 (weighted) | 0.6649 | 0.6682 |
| Best Model | lgbm | lgbm |

### road_safety - Road safety prediction
- **Task:** classification
- **Samples:** 50,000
- **Features:** 32 → 93
- **FE Time:** 26.4s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| Accuracy | 0.7940 | 0.7958 |
| F1 (weighted) | 0.7937 | 0.7956 |
| Best Model | xgboost | xgboost |

### albert - Albert dataset
- **Task:** classification
- **Samples:** 50,000
- **Features:** 31 → 100
- **FE Time:** 26.4s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| Accuracy | 0.6630 | 0.6619 |
| F1 (weighted) | 0.6627 | 0.6617 |
| Best Model | catboost | lgbm |

### diamonds - Diamond price prediction
- **Task:** regression
- **Samples:** 50,000
- **Features:** 6 → 19
- **FE Time:** 41.3s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| RMSE | 0.2331 | 0.2305 |
| R² | 0.9475 | 0.9487 |
| Best Model | xgb_limitdepth | xgboost |

### house_sales - House sale price prediction
- **Task:** regression
- **Samples:** 21,613
- **Features:** 15 → 50
- **FE Time:** 25.9s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| RMSE | 0.1737 | 0.1767 |
| R² | 0.8942 | 0.8904 |
| Best Model | xgb_limitdepth | xgb_limitdepth |

### houses - House value prediction
- **Task:** regression
- **Samples:** 20,640
- **Features:** 8 → 40
- **FE Time:** 27.3s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| RMSE | 0.2320 | 0.2305 |
| R² | 0.8342 | 0.8363 |
| Best Model | catboost | catboost |

### wine_quality - Wine quality score
- **Task:** regression
- **Samples:** 6,497
- **Features:** 11 → 45
- **FE Time:** 7.1s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| RMSE | 0.6278 | 0.6076 |
| R² | 0.4664 | 0.5001 |
| Best Model | lgbm | lgbm |

### abalone - Abalone age prediction
- **Task:** regression
- **Samples:** 4,177
- **Features:** 7 → 30
- **FE Time:** 4.1s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| RMSE | 2.2328 | 2.1210 |
| R² | 0.5395 | 0.5844 |
| Best Model | lgbm | catboost |

### superconduct - Superconductor temperature
- **Task:** regression
- **Samples:** 21,263
- **Features:** 79 → 100
- **FE Time:** 39.8s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| RMSE | 8.8960 | 8.8366 |
| R² | 0.9312 | 0.9322 |
| Best Model | lgbm | lgbm |

### cpu_act - CPU activity prediction
- **Task:** regression
- **Samples:** 8,192
- **Features:** 21 → 100
- **FE Time:** 10.0s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| RMSE | 2.4318 | 2.2521 |
| R² | 0.9803 | 0.9831 |
| Best Model | catboost | lgbm |

### elevators - Elevator control
- **Task:** regression
- **Samples:** 16,599
- **Features:** 16 → 76
- **FE Time:** 18.5s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| RMSE | 0.0021 | 0.0022 |
| R² | 0.8991 | 0.8883 |
| Best Model | catboost | xgboost |

### miami_housing - Miami housing prices
- **Task:** regression
- **Samples:** 13,932
- **Features:** 13 → 70
- **FE Time:** 17.7s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| RMSE | 0.1498 | 0.1490 |
| R² | 0.9303 | 0.9310 |
| Best Model | catboost | xgboost |

### bike_sharing - Bike rental demand
- **Task:** regression
- **Samples:** 17,379
- **Features:** 6 → 37
- **FE Time:** 11.8s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| RMSE | 97.0385 | 96.5789 |
| R² | 0.7026 | 0.7054 |
| Best Model | catboost | catboost |

### delays_zurich - Zurich transport delays
- **Task:** regression
- **Samples:** 50,000
- **Features:** 11 → 58
- **FE Time:** 55.8s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| RMSE | 2.9966 | 2.9896 |
| R² | 0.0810 | 0.0853 |
| Best Model | extra_tree | lgbm |

### allstate_claims - Insurance claim severity
- **Task:** regression
- **Samples:** 50,000
- **Features:** 124 → 122
- **FE Time:** 101.5s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| RMSE | 0.5402 | 0.5404 |
| R² | 0.5456 | 0.5452 |
| Best Model | xgboost | xgboost |

### mercedes_benz - Manufacturing time
- **Task:** regression
- **Samples:** 4,209
- **Features:** 359 → 274
- **FE Time:** 9.8s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| RMSE | 8.0565 | 7.8923 |
| R² | 0.5830 | 0.5998 |
| Best Model | rf | extra_tree |

### nyc_taxi - NYC taxi trip duration
- **Task:** regression
- **Samples:** 50,000
- **Features:** 16 → 62
- **FE Time:** 50.9s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| RMSE | 0.3621 | 0.3711 |
| R² | 0.6375 | 0.6194 |
| Best Model | xgb_limitdepth | xgboost |

### brazilian_houses - Brazilian house prices
- **Task:** regression
- **Samples:** 10,692
- **Features:** 11 → 66
- **FE Time:** 9.7s

| Metric | Baseline | FeatCopilot |
|--------|----------|-------------|
| RMSE | 0.0408 | 0.0870 |
| R² | 0.9973 | 0.9876 |
| Best Model | xgboost | xgb_limitdepth |
