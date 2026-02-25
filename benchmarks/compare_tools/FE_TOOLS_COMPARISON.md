# Feature Engineering Tools Comparison Benchmark
## Overview
This benchmark compares FeatCopilot with other popular feature engineering libraries across multiple datasets using FLAML AutoML for model training.

### Tools Compared
| Tool | Description |
|------|-------------|
| baseline | No feature engineering (raw features only) |
| featcopilot | FeatCopilot - Multi-engine auto feature engineering |
| featuretools | Featuretools - Deep Feature Synthesis |
| autofeat | autofeat - Automatic feature generation with L1 selection |

## 1. Win Rate (Best Score Per Dataset)
| Tool | Wins | Datasets Tested | Win Rate |
|------|------|-----------------|----------|
| featcopilot | 8 | 10 | 80% 🏆 |
| autofeat | 2 | 5 | 40% |
| featuretools | 0 | 10 | 0% |

## 2. Average Improvement Over Baseline
| Tool | Avg Improvement | Min Improvement | Max Improvement | Positive % |
|------|-----------------|-----------------|-----------------|------------|
| featcopilot | +1.89% | -0.02% | +5.78% | 80% |
| autofeat | +1.46% | -0.54% | +4.38% | 60% |
| featuretools | -2.71% | -16.72% | +2.72% | 20% |

## 3. Feature Engineering Speed
| Tool | Avg FE Time | Median FE Time | Speedup vs Slowest |
|------|-------------|----------------|--------------------|
| featuretools | 0.11s ⚡ | 0.11s | 433x |
| featcopilot | 1.90s | 1.81s | 25x |
| autofeat | 48.09s | 40.52s | 1x |

## 4. Dataset Coverage
| Tool | Successful | Errored | Timed Out | Coverage |
|------|-----------|---------|-----------|----------|
| featcopilot | 10 | 0 | 0 | 100% 🏆 |
| featuretools | 10 | 0 | 0 | 100% 🏆 |
| autofeat | 5 | 5 | 0 | 50% |

## 5. Composite Score (Overall Ranking)
Composite score combines: accuracy improvement (40%), win rate (30%), speed (15%), coverage (15%).

| Rank | Tool | Accuracy (40%) | Win Rate (30%) | Speed (15%) | Coverage (15%) | Composite |
|------|------|----------------|----------------|-------------|----------------|----------|
| 🥇 1 | **featcopilot** | +1.89% | 8/10 | 1.9s | 10/10 | **0.606** |
| 🥈 2 | **featuretools** | -2.71% | 0/10 | 0.1s | 10/10 | **0.397** |
| 🥉 3 | **autofeat** | +1.46% | 2/5 | 48.1s | 5/10 | **0.351** |

## Detailed Results

### complex_regression
**Task**: regression

| Tool | R² Score | Features | FE Time | Status |
|------|----------|----------|---------|--------|
| baseline | 0.8691 | 15.0 | 0.00s | ✅ |
| featcopilot | 0.8825 | 20.0 | 3.17s | ✅ |
| featuretools | 0.8490 | 100.0 | 0.10s | ✅ |
| autofeat | **0.8988** 🏆 | 31.0 | 40.52s | ✅ |

### polynomial_regression
**Task**: regression

| Tool | R² Score | Features | FE Time | Status |
|------|----------|----------|---------|--------|
| baseline | 0.9026 | 12.0 | 0.00s | ✅ |
| featcopilot | 0.9342 | 19.0 | 2.90s | ✅ |
| featuretools | 0.8843 | 100.0 | 0.09s | ✅ |
| autofeat | **0.9421** 🏆 | 28.0 | 27.63s | ✅ |

### xor_classification
**Task**: classification

| Tool | Accuracy | Features | FE Time | Status |
|------|----------|----------|---------|--------|
| baseline | 0.8480 | 20.0 | 0.00s | ✅ |
| featcopilot | **0.8540** 🏆 | 24.0 | 2.27s | ✅ |
| featuretools | 0.8140 | 100.0 | 0.13s | ✅ |
| autofeat | nan | nan | 1290.00s | ❌ FE timeout (>120s) |

### complex_classification
**Task**: classification

| Tool | Accuracy | Features | FE Time | Status |
|------|----------|----------|---------|--------|
| baseline | 0.7350 | 15.0 | 0.00s | ✅ |
| featcopilot | **0.7775** 🏆 | 21.0 | 1.69s | ✅ |
| featuretools | 0.7550 | 100.0 | 0.10s | ✅ |
| autofeat | nan | nan | 624.00s | ❌ FE timeout (>120s) |

### interaction_classification
**Task**: classification

| Tool | Accuracy | Features | FE Time | Status |
|------|----------|----------|---------|--------|
| baseline | 0.7925 | 12.0 | 0.00s | ✅ |
| featcopilot | **0.8100** 🏆 | 17.0 | 1.69s | ✅ |
| featuretools | 0.7725 | 100.0 | 0.11s | ✅ |
| autofeat | nan | nan | 360.00s | ❌ FE timeout (>120s) |

### titanic
**Task**: classification

| Tool | Accuracy | Features | FE Time | Status |
|------|----------|----------|---------|--------|
| baseline | 0.8603 | 7.0 | 0.00s | ✅ |
| featcopilot | **0.8603** 🏆 | 10.0 | 0.56s | ✅ |
| featuretools | **0.8603** 🏆 | 91.0 | 0.11s | ✅ |
| autofeat | **0.8603** 🏆 | 17.0 | 88.36s | ✅ |

### house_prices
**Task**: regression

| Tool | R² Score | Features | FE Time | Status |
|------|----------|----------|---------|--------|
| baseline | 0.9958 | 14.0 | 0.00s | ✅ |
| featcopilot | **0.9972** 🏆 | 16.0 | 1.98s | ✅ |
| featuretools | 0.9964 | 100.0 | 0.15s | ✅ |
| autofeat | 0.9965 | 37.0 | 54.83s | ✅ |

### credit_risk
**Task**: classification

| Tool | Accuracy | Features | FE Time | Status |
|------|----------|----------|---------|--------|
| baseline | 0.8750 | 10.0 | 0.00s | ✅ |
| featcopilot | **0.8925** 🏆 | 16.0 | 1.44s | ✅ |
| featuretools | 0.8575 | 100.0 | 0.09s | ✅ |
| autofeat | nan | nan | 411.00s | ❌ FE timeout (>120s) |

### bike_sharing
**Task**: regression

| Tool | R² Score | Features | FE Time | Status |
|------|----------|----------|---------|--------|
| baseline | 0.9795 | 10.0 | 0.00s | ✅ |
| featcopilot | **0.9793** 🏆 | 12.0 | 1.94s | ✅ |
| featuretools | 0.9770 | 100.0 | 0.14s | ✅ |
| autofeat | 0.9742 | 40.0 | 29.09s | ✅ |

### customer_churn
**Task**: classification

| Tool | Accuracy | Features | FE Time | Status |
|------|----------|----------|---------|--------|
| baseline | 0.7325 | 10.0 | 0.00s | ✅ |
| featcopilot | **0.7550** 🏆 | 12.0 | 1.35s | ✅ |
| featuretools | 0.6100 | 100.0 | 0.09s | ✅ |
| autofeat | nan | nan | 300.00s | ❌ FE timeout (>120s) |

## Conclusion
**Overall Winner: featcopilot** (composite score: 0.606)

FeatCopilot achieves the best composite score with: highest win rate, broadest dataset coverage.
