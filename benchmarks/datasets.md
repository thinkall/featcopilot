# Benchmark Datasets for FeatCopilot

This document catalogs recommended datasets for benchmarking FeatCopilot's feature engineering capabilities. These datasets are selected because they are known to benefit significantly from feature engineering techniques.

## Quick Reference

| # | Dataset | Task | Size | Features | Source | Loader Function |
|---|---------|------|------|----------|--------|-----------------|
| 1 | House Prices | Regression | 1,460 | 79 | Kaggle | `load_kaggle_house_prices()` |
| 2 | IBM HR Attrition | Classification | 1,470 | 35 | Kaggle | `load_kaggle_employee_attrition()` |
| 3 | Telco Customer Churn | Classification | 7,043 | 21 | Kaggle | `load_kaggle_telco_churn()` |
| 4 | Adult Census Income | Classification | 48,842 | 14 | OpenML | `load_openml_adult_census()` |
| 5 | Credit Card Fraud | Classification | 284,807 | 30 | Kaggle | `load_kaggle_credit_card_fraud()` |
| 6 | Spaceship Titanic | Classification | 8,693 | 13 | Kaggle | `load_kaggle_spaceship_titanic()` |
| 7 | Bike Sharing Demand | Regression | 10,886 | 12 | Kaggle | `load_kaggle_bike_sharing()` |
| 8 | Medical Cost | Regression | 1,338 | 7 | Kaggle | `load_kaggle_medical_cost()` |
| 9 | Wine Quality | Regression/Classification | 6,497 | 11 | OpenML | `load_openml_wine_quality()` |
| 10 | Life Expectancy | Regression | 2,938 | 22 | Kaggle | `load_kaggle_life_expectancy()` |
| 11 | Home Credit Default | Classification | 307,511 | 122 | Kaggle | `load_kaggle_home_credit()` |
| 12 | Store Sales | Regression | 3M+ | Multi-table | Kaggle | `load_kaggle_store_sales()` |

---

## Classification Datasets

### 1. House Prices - Advanced Regression Techniques
**Source:** [Kaggle Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

**Description:** Predict sale prices of homes in Ames, Iowa based on 79 explanatory variables describing almost every aspect of residential homes.

**Why Good for FeatCopilot:**
- Rich mix of numeric and categorical features
- Known to benefit heavily from feature interactions (e.g., TotalSF = 1stFlrSF + 2ndFlrSF + TotalBsmtSF)
- Quality score ratios and polynomial features improve performance
- Missing value patterns carry information

**Key Feature Engineering Opportunities:**
- Area combinations and ratios
- Age-related features (YearBuilt, YearRemodAdd)
- Quality × quantity interactions
- Bathroom/bedroom ratios

---

### 2. IBM HR Employee Attrition
**Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

**Description:** Predict employee attrition based on HR metrics including satisfaction scores, work-life balance, overtime, and compensation.

**Why Good for FeatCopilot:**
- HR domain with interpretable features
- Ordinal satisfaction scales benefit from interaction features
- Tenure-based features (YearsAtCompany, YearsSinceLastPromotion)
- Work-life indicators can be combined meaningfully

**Key Feature Engineering Opportunities:**
- Satisfaction × workload interactions
- Tenure ratios (YearsSinceLastPromotion / YearsAtCompany)
- Income-to-age ratios
- Overtime × satisfaction interactions

---

### 3. Telco Customer Churn
**Source:** [Kaggle Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)

**Description:** Predict whether a customer will churn based on their service usage, contract type, and billing information.

**Why Good for FeatCopilot:**
- Industry-relevant churn prediction task
- Service combination features (multiple lines, internet, streaming)
- Contract and payment interactions
- Tenure-based patterns

**Key Feature Engineering Opportunities:**
- Service bundle combinations
- Monthly charges / tenure ratios
- Contract × payment method interactions
- Total charges vs expected charges

---

### 4. Adult Census Income
**Source:** [OpenML](https://www.openml.org/d/1590) / [UCI](https://archive.ics.uci.edu/ml/datasets/Adult)

**Description:** Classic benchmark dataset predicting whether income exceeds $50K/year based on census data.

**Why Good for FeatCopilot:**
- Standard ML benchmark with known baselines
- Mix of categorical and numeric features
- Education and occupation interactions are valuable
- Demographics combinations improve predictions

**Key Feature Engineering Opportunities:**
- Education × occupation interactions
- Age × hours-per-week
- Capital gain/loss ratios
- Marital status × relationship combinations

---

### 5. Credit Card Fraud Detection
**Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Description:** Detect fraudulent credit card transactions from anonymized PCA-transformed features.

**Why Good for FeatCopilot:**
- Highly imbalanced dataset (0.17% fraud)
- PCA features can still benefit from interactions
- Amount and time features are interpretable
- Anomaly detection patterns

**Key Feature Engineering Opportunities:**
- V-feature interactions and ratios
- Amount transformations (log, bins)
- Time-based patterns (hour of day)
- Statistical aggregations across V features

---

### 6. Spaceship Titanic
**Source:** [Kaggle Competition](https://www.kaggle.com/c/spaceship-titanic)

**Description:** Modern Titanic-style competition predicting which passengers were transported to an alternate dimension.

**Why Good for FeatCopilot:**
- Modern beginner-friendly competition
- Cabin information extraction (deck, num, side)
- Expenditure patterns across amenities
- Group/family features from PassengerId

**Key Feature Engineering Opportunities:**
- Cabin parsing (Deck/Num/Side)
- Total spending and spending ratios
- Group size from PassengerId
- Age × spending interactions

---

## Regression Datasets

### 7. Bike Sharing Demand
**Source:** [Kaggle Competition](https://www.kaggle.com/c/bike-sharing-demand)

**Description:** Predict hourly bike rental demand based on weather, time, and seasonal factors.

**Why Good for FeatCopilot:**
- Time-based features (hour, day, month) benefit from cyclical encoding
- Weather interactions are valuable
- Rush hour patterns need feature engineering
- Known to improve ~65% with proper FE

**Key Feature Engineering Opportunities:**
- Cyclical encoding (sin/cos for hour, month)
- Rush hour indicators
- Weather × workday interactions
- Temperature comfort zones

---

### 8. Medical Cost Personal
**Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)

**Description:** Predict individual medical costs based on demographic and lifestyle factors.

**Why Good for FeatCopilot:**
- Simple but effective for demonstrating FE value
- BMI × smoker interaction is highly predictive
- Age bands and BMI categories help
- Regional cost variations

**Key Feature Engineering Opportunities:**
- BMI × smoker interaction (critical!)
- Age × smoker interaction
- BMI categories (underweight, normal, overweight, obese)
- Age bands

---

### 9. Wine Quality
**Source:** [OpenML](https://www.openml.org/d/187)

**Description:** Predict wine quality score (0-10) based on physicochemical properties.

**Why Good for FeatCopilot:**
- Chemical property ratios are meaningful
- Acidity balance features
- Alcohol × other property interactions
- Can be used as regression or classification

**Key Feature Engineering Opportunities:**
- Fixed acidity / volatile acidity ratio
- Total sulfur dioxide / free sulfur dioxide
- Alcohol × density interaction
- pH × acidity interactions

---

### 10. Life Expectancy (WHO)
**Source:** [Kaggle Dataset](https://www.kaggle.com/kumarajarshi/life-expectancy-who)

**Description:** Predict life expectancy based on health, economic, and social factors from WHO data.

**Why Good for FeatCopilot:**
- Socioeconomic feature interactions
- Health indicator ratios
- Development status interactions
- Missing data patterns

**Key Feature Engineering Opportunities:**
- GDP per capita interactions
- Health expenditure ratios
- Immunization coverage combinations
- Mortality rate ratios

---

## Advanced / Competition-Level Datasets

### 11. Home Credit Default Risk
**Source:** [Kaggle Competition](https://www.kaggle.com/c/home-credit-default-risk)

**Description:** Predict loan default probability using alternative data including transactional and bureau data.

**Why Good for FeatCopilot:**
- Multi-table structure requires aggregations
- Winning solutions relied heavily on feature engineering
- 100+ features with complex relationships
- Time-based transaction patterns

**Key Feature Engineering Opportunities:**
- Aggregations from related tables (mean, max, count)
- Payment behavior features
- Credit utilization ratios
- Trend features over time

---

### 12. Store Sales - Time Series Forecasting
**Source:** [Kaggle Competition](https://www.kaggle.com/c/store-sales-time-series-forecasting)

**Description:** Forecast daily sales for multiple stores and product families.

**Why Good for FeatCopilot:**
- Lag features and rolling statistics
- Holiday and promotion effects
- Store × product interactions
- External factors (oil prices)

**Key Feature Engineering Opportunities:**
- Lag features (1, 7, 14, 28 days)
- Rolling means and std
- Holiday proximity features
- Promotion carry-over effects

---

## Usage

### Loading Real-World Datasets

```python
from benchmarks.datasets import (
    load_kaggle_house_prices,
    load_kaggle_employee_attrition,
    load_openml_adult_census,
    # ... etc
)

# Load a dataset
X, y, task_type, dataset_name = load_kaggle_house_prices()

# Apply FeatCopilot
from featcopilot import AutoFeatureEngineer

engineer = AutoFeatureEngineer(engines=["tabular"], max_features=100)
X_enhanced = engineer.fit_transform(X, y)
```

### Loading Synthetic Datasets (for controlled benchmarking)

```python
from benchmarks.datasets import (
    load_titanic_dataset,           # Synthetic Titanic-like
    load_house_prices_dataset,      # Synthetic house prices
    create_credit_risk_dataset,     # Synthetic credit risk
    # ... etc
)
```

---

## Dataset Selection Guide

### For Quick Demos
- **Medical Cost** - Small, simple, clear FE wins
- **Bike Sharing** - Medium size, obvious temporal patterns

### For Comprehensive Benchmarking
- **House Prices** - Gold standard for tabular FE
- **Adult Census** - Classic ML benchmark
- **Telco Churn** - Industry-relevant

### For Advanced Testing
- **Home Credit** - Multi-table, complex FE
- **Store Sales** - Time series, external factors

### For Imbalanced Data
- **Credit Card Fraud** - Extreme imbalance
- **Employee Attrition** - Moderate imbalance

---

## INRIA-SODA Tabular Benchmark Datasets

The [inria-soda/tabular-benchmark](https://huggingface.co/datasets/inria-soda/tabular-benchmark) on HuggingFace contains 57 curated real-world datasets from OpenML, specifically designed for benchmarking tabular ML models. These are challenging, competition-level datasets.

### Available Datasets (30 tested)

| Dataset | Task | Samples | Features | Description |
|---------|------|---------|----------|-------------|
| higgs | Classification | 940K | 24 | Higgs boson detection (physics) |
| covertype | Classification | 566K | 10 | Forest cover type prediction |
| jannis | Classification | 57K | 54 | Multi-class classification |
| miniboone | Classification | 73K | 50 | Particle physics classification |
| california | Classification | 20K | 8 | California housing (binned) |
| credit | Classification | 16K | 10 | Credit approval prediction |
| bank_marketing | Classification | 10K | 7 | Bank marketing response |
| diabetes | Classification | 50K+ | 7 | Diabetes readmission |
| bioresponse | Classification | 3.4K | 419 | Biological response prediction |
| electricity | Classification | 38K | 8 | Electricity price direction |
| eye_movements | Classification | 7.6K | 23 | Eye movement classification |
| road_safety | Classification | 50K+ | 32 | Road safety prediction |
| albert | Classification | 50K+ | 31 | Albert dataset |
| diamonds | Regression | 54K | 6 | Diamond price prediction |
| house_sales | Regression | 21K | 15 | House sale price prediction |
| houses | Regression | 20K | 8 | House value prediction |
| wine_quality | Regression | 6.5K | 11 | Wine quality score |
| abalone | Regression | 4.2K | 7 | Abalone age prediction |
| superconduct | Regression | 21K | 79 | Superconductor temperature |
| cpu_act | Regression | 8K | 21 | CPU activity prediction |
| elevators | Regression | 16K | 16 | Elevator control |
| miami_housing | Regression | 14K | 13 | Miami housing prices |
| bike_sharing | Regression | 17K | 6 | Bike rental demand |
| delays_zurich | Regression | 50K+ | 11 | Zurich transport delays |
| allstate_claims | Regression | 50K+ | 124 | Insurance claim severity |
| mercedes_benz | Regression | 4.2K | 359 | Manufacturing time |
| nyc_taxi | Regression | 50K+ | 16 | NYC taxi trip duration |
| brazilian_houses | Regression | 10K | 11 | Brazilian house prices |

### Loading INRIA Datasets

```python
from datasets import load_dataset

# Load classification dataset
ds = load_dataset("inria-soda/tabular-benchmark", "clf_num_Higgs", split="train")
df = ds.to_pandas()
target_col = df.columns[-1]  # Target is always last column
X, y = df.drop(columns=[target_col]), df[target_col]

# Load regression dataset
ds = load_dataset("inria-soda/tabular-benchmark", "reg_num_diamonds", split="train")
```

### Benchmark Results Summary

Tested with FLAML AutoML (60s budget), FeatCopilot improved **16/29 datasets (55%)** with:
- **Average improvement:** +0.47%
- **Max improvement:** +8.34% (abalone R² 0.5395 → 0.5844)
- **Top improvements:** abalone (+8.34%), wine_quality (+7.24%), delays_zurich (+5.32%), mercedes_benz (+2.89%), bioresponse (+2.62%)

See `benchmarks/automl/FLAML_INRIA_FULL_BENCHMARK_REPORT.md` for detailed results.

---

## References

- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [OpenML Benchmark Suites](https://www.openml.org/search?type=benchmark)
- [Hugging Face Datasets](https://huggingface.co/datasets?task_categories=tabular-classification)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [INRIA-SODA Tabular Benchmark](https://huggingface.co/datasets/inria-soda/tabular-benchmark)
