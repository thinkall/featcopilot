# Benchmarks

FeatCopilot has been extensively benchmarked to demonstrate its effectiveness in automated feature engineering. This page presents comprehensive results across **63 datasets** spanning classification, regression, forecasting, and text tasks.

## Executive Summary

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Simple Models Benchmark__

    ---

    **+7.52%** average improvement

    **+144%** max improvement (triple_interaction)

-   :material-trophy:{ .lg .middle } __FE Tools Comparison__

    ---

    **#1 overall** — 80% win rate

    Beats autofeat and featuretools

-   :material-chart-line:{ .lg .middle } __AutoML Benchmark__

    ---

    **+1.85%** avg improvement (FLAML)

    **90%** datasets improved

-   :material-lightning-bolt:{ .lg .middle } __Feature Generation__

    ---

    **7→24** to **54→57** feature expansion

    GBM-based importance selection

</div>

---

## Simple Models Benchmark

Testing FeatCopilot with RandomForest (n_estimators=200, max_depth=20) and LogisticRegression/Ridge across 63 datasets to measure feature engineering impact.

### Summary Results

| Configuration | Datasets | Improved | Avg Improvement | Best Improvement |
|---------------|----------|----------|-----------------|------------------|
| **Tabular Engine** | 63 | 31 (49%) | **+7.52%** | +144% (triple_interaction) |

### Classification Results (26 Datasets)

| Dataset | Baseline | +FeatCopilot | Improvement | Features |
|---------|----------|--------------|-------------|----------|
| xor_classification | 0.6960 | **0.8120** | **+16.67%** | 20→24 |
| complex_classification | 0.7125 | **0.8300** | **+16.49%** | 15→23 |
| polynomial_classification | 0.7875 | **0.8675** | **+10.16%** | 15→21 |
| interaction_classification | 0.7650 | **0.8075** | **+5.56%** | 12→17 |
| credit_risk | 0.8525 | **0.8675** | **+1.76%** | 10→17 |
| bioresponse | 0.7700 | **0.7802** | **+1.32%** | 419→419 |
| covertype_cat | 0.8747 | **0.8819** | **+0.82%** | 54→55 |
| magic_telescope | 0.8509 | **0.8572** | **+0.75%** | 10→12 |
| credit_card_fraud | 0.9840 | 0.9840 | +0.00% | 30→40 |
| employee_attrition | 0.9252 | 0.9252 | +0.00% | 11→16 |

### Regression Results (30 Datasets)

| Dataset | Baseline R² | +FeatCopilot R² | Improvement | Features |
|---------|-------------|-----------------|-------------|----------|
| triple_interaction_regression | 0.3542 | **0.8649** | **+144%** | 18→23 |
| xor_regression | 0.3330 | **0.6801** | **+104%** | 20→24 |
| pairwise_product_regression | 0.5132 | **0.8698** | **+69.5%** | 16→23 |
| nonlinear_regression | 0.6086 | **0.8756** | **+43.9%** | 12→18 |
| complex_regression | 0.6339 | **0.8725** | **+37.6%** | 15→20 |
| quadratic_heavy_regression | 0.7134 | **0.9341** | **+30.9%** | 18→25 |
| polynomial_regression | 0.7321 | **0.8692** | **+18.7%** | 12→19 |
| sqrt_log_regression | 0.8725 | **0.8997** | **+3.12%** | 15→25 |
| bike_sharing | 0.9534 | **0.9697** | **+1.71%** | 10→12 |
| house_prices | 0.9798 | **0.9953** | **+1.58%** | 14→16 |
| spotify_tracks | 0.9529 | **0.9648** | **+1.25%** | 13→17 |
| ecommerce_product | 0.9462 | **0.9564** | **+1.08%** | 10→11 |

!!! success "Key Insight"
    FeatCopilot provides the **largest improvements** on datasets with complex feature interactions (XOR, polynomial, triple interactions) where simple models struggle with raw features. The GBM-based feature selection ensures only high-quality derived features are kept.

---

## FE Tools Comparison

Comparing FeatCopilot against autofeat and featuretools across 10 datasets using FLAML (30s budget).

### Overall Ranking

| Rank | Tool | Win Rate | Avg Improvement | Speed | Coverage | Composite |
|------|------|----------|-----------------|-------|----------|-----------|
| 🥇 | **FeatCopilot** | **80%** (8/10) | **+1.89%** | 1.9s | **100%** | **0.606** |
| 🥈 | featuretools | 0% (0/10) | -2.71% | 0.1s | 100% | 0.397 |
| 🥉 | autofeat | 40% (2/5) | +1.46% | 48.1s | 50% | 0.351 |

### Key Advantages

- **Highest win rate**: FeatCopilot wins 8 out of 10 datasets
- **Best coverage**: Works on all datasets (autofeat times out on 50%)
- **Fast**: ~25x faster than autofeat, with intelligent feature selection
- **No harm**: Never significantly hurts performance (min improvement -0.02%)

!!! note "Why autofeat has low coverage"
    autofeat uses L1 regularization which is computationally expensive for classification tasks. It timed out (>120s) on 5 of 10 datasets. FeatCopilot uses GBM-based selection which is much faster.

---

## AutoML Benchmark

Testing FeatCopilot with FLAML and AutoGluon (120s time budget) across 10 datasets to evaluate feature engineering benefits with AutoML optimization.

### Cross-Framework Summary

| Framework | Datasets | Improved | Avg Improvement | Max Improvement |
|-----------|----------|----------|-----------------|-----------------|
| **FLAML** | 10 | 9 (90%) | **+1.85%** | +6.67% |
| **AutoGluon** | 10 | 9 (90%) | **+1.55%** | +7.62% |
| **Combined** | 20 | 18 (90%) | **+1.70%** | — |

### Per-Dataset Results

| Dataset | FLAML Baseline | FLAML +FE | Δ | AutoGluon Base | AutoGluon +FE | Δ |
|---------|---------------|-----------|---|---------------|---------------|---|
| complex_classification | 0.7875 | 0.8400 | **+6.67%** 🔥 | 0.7550 | 0.8125 | **+7.62%** 🔥 |
| xor_classification | 0.8180 | 0.8640 | **+5.62%** 🔥 | 0.8280 | 0.8480 | **+2.42%** 🔥 |
| polynomial_regression | 0.9103 | 0.9375 | **+2.99%** 🔥 | 0.9404 | 0.9492 | +0.94% |
| titanic | 0.8156 | 0.8268 | +1.37% | 0.8212 | 0.8324 | +1.36% |
| complex_regression | 0.8704 | 0.8790 | +0.99% | 0.9166 | 0.9401 | **+2.57%** 🔥 |
| credit_risk | 0.8500 | 0.8550 | +0.59% | 0.8525 | 0.8650 | +1.47% |
| house_prices | 0.9963 | 0.9972 | +0.09% | 0.9969 | 0.9975 | +0.06% |

!!! note "AutoML Observations"
    With AutoML (FLAML/AutoGluon), improvements are more modest because these frameworks already use powerful gradient boosting models. FeatCopilot still provides consistent value by generating meaningful derived features that help AutoML find better models faster.

---

## When FeatCopilot Excels

Based on comprehensive benchmarking across 63 datasets, FeatCopilot provides the most value in these scenarios:

| Scenario | Expected Benefit | Evidence |
|----------|------------------|----------|
| **Complex interactions** | **Very High** (+30-144%) | triple_interaction: +144%, xor: +104% |
| **Small feature sets** | **High** (+5-20%) | polynomial_classification: +10%, complex_classification: +16.5% |
| **AutoML enhancement** | **Medium** (+1-8%) | complex_classification: +6.67% (FLAML), +7.62% (AutoGluon) |
| **Already high-performing** | **Low** (0-1%) | credit_card_fraud: +0.00% (baseline 0.984) |
| **FE tool comparison** | **Winner** (80% win rate) | Composite score 0.606 (#1 of 3 tools) |

!!! info "Key Insight"
    FeatCopilot provides the **largest improvements** on datasets where:

    1. **Feature interactions matter** - XOR, polynomial, and multi-way interactions
    2. **Feature set is small** - More potential for derived features
    3. **Baseline performance has room** - Near-perfect baselines show minimal improvement

    FeatCopilot's GBM-based feature selection ensures it **rarely hurts** performance, even when improvements are small.

---

## Running Benchmarks

### Quick Start

```bash
# Clone and install
git clone https://github.com/thinkall/featcopilot.git
cd featcopilot
pip install -e ".[benchmark]"

# Run simple models benchmark (63 datasets)
python -m benchmarks.simple_models.run_simple_models_benchmark --all

# Run with LLM engine
python -m benchmarks.simple_models.run_simple_models_benchmark --all --with-llm

# Run AutoML benchmark (FLAML + AutoGluon)
python -m benchmarks.automl.run_automl_benchmark --framework all

# Run FE tools comparison
python -m benchmarks.compare_tools.run_fe_tools_comparison
```

### Available Benchmarks

| Benchmark | Command | Description |
|-----------|---------|-------------|
| **Simple Models** | `python -m benchmarks.simple_models.run_simple_models_benchmark --all` | RF/Ridge on 63 datasets |
| **Simple Models + LLM** | `python -m benchmarks.simple_models.run_simple_models_benchmark --all --with-llm` | With LLM-generated features |
| **AutoML (FLAML)** | `python -m benchmarks.automl.run_automl_benchmark --framework flaml` | FLAML on 10 datasets |
| **AutoML (AutoGluon)** | `python -m benchmarks.automl.run_automl_benchmark --framework autogluon` | AutoGluon on 10 datasets |
| **AutoML (All)** | `python -m benchmarks.automl.run_automl_benchmark --framework all` | Both frameworks |
| **Tool Comparison** | `python -m benchmarks.compare_tools.run_fe_tools_comparison` | vs autofeat, featuretools |

### Benchmark Options

```bash
# Specific datasets
python -m benchmarks.simple_models.run_simple_models_benchmark --datasets titanic,house_prices

# By category
python -m benchmarks.automl.run_automl_benchmark --category classification

# Use AutoGluon instead of FLAML
python -m benchmarks.automl.run_automl_benchmark --framework autogluon

# Custom time budget
python -m benchmarks.automl.run_automl_benchmark --time-budget 300
```

### Benchmark Reports

```
benchmarks/
├── simple_models/
│   ├── SIMPLE_MODELS_BENCHMARK.md      # Tabular only
│   └── SIMPLE_MODELS_BENCHMARK_LLM.md  # Tabular + LLM
├── automl/
│   ├── AUTOML_BENCHMARK.md             # Combined (FLAML + AutoGluon)
│   ├── AUTOML_FLAML_BENCHMARK.md       # FLAML only
│   └── AUTOML_AUTOGLUON_BENCHMARK.md   # AutoGluon only
└── compare_tools/
    └── FE_TOOLS_COMPARISON.md          # FeatCopilot vs competitors
```

---

## Methodology

### Evaluation Protocol

- **Train/Test Split**: 80/20 with `random_state=42` for reproducibility
- **Metrics**: Accuracy for classification, R² for regression
- **Models**:
  - Simple: RandomForest (n_estimators=200, max_depth=20), LogisticRegression/Ridge
  - AutoML: FLAML / AutoGluon with 120s time budget
  - FE Comparison: FLAML with 30s time budget
- **Feature Selection**: GBM-based importance with median threshold + GBM refinement

### FeatCopilot Configuration

```python
from featcopilot import AutoFeatureEngineer

# Standard configuration
engineer = AutoFeatureEngineer(
    engines=["tabular", "relational"],
    max_features=100,
    verbose=False,
)

X_transformed = engineer.fit_transform(X_train, y_train)
X_test_transformed = engineer.transform(X_test)
```

---

## Dataset Coverage

63 datasets across 4 categories:

| Category | Count | Examples |
|----------|-------|----------|
| **Classification** | 26 | titanic, xor_classification, complex_classification, higgs, covertype |
| **Regression** | 30 | house_prices, complex_regression, polynomial_regression, diamonds |
| **Forecasting** | 3 | sensor_anomaly, retail_demand, server_latency |
| **Text** | 4 | product_reviews, fake_news, news_classification, medical_notes |

Dataset sources include:
- **OpenML** (INRIA benchmark suite)
- **Kaggle** (popular ML datasets)
- **Synthetic** (interaction-heavy datasets for FE evaluation)

---

## Conclusion

FeatCopilot demonstrates **consistent improvements** across diverse datasets and frameworks:

1. **Simple Models**: +7.52% average, up to +144% maximum improvement across 63 datasets
2. **AutoML**: +1.70% average across FLAML and AutoGluon, 90% improvement rate
3. **vs Competitors**: #1 ranked FE tool with 80% win rate, beating autofeat and featuretools

**Best use cases**:
- Datasets with complex feature interactions (XOR, polynomial, multi-way)
- Small feature sets that benefit from derived features
- Any ML pipeline where consistent, safe feature engineering is needed

**Key strengths**:
- GBM-based selection ensures features rarely hurt performance
- Fast (~2s) compared to alternatives (autofeat: ~48s)
- Works on all dataset types (100% coverage vs autofeat's 50%)
