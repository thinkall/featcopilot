# Benchmarks

FeatCopilot has been extensively benchmarked to demonstrate its effectiveness in automated feature engineering. This page presents comprehensive results across **42 datasets** spanning classification and regression tasks.

## Executive Summary

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Simple Models Benchmark__

    ---

    **+4.54%** average improvement

    **+197%** max improvement (delays_zurich)

-   :material-brain:{ .lg .middle } __LLM-Enhanced Results__

    ---

    **+420%** max improvement with LLM

    **55%** datasets improved (23/42)

-   :material-chart-line:{ .lg .middle } __AutoML Benchmark__

    ---

    **+8.55%** best improvement (abalone)

    **46%** datasets improved with FLAML

-   :material-lightning-bolt:{ .lg .middle } __Feature Generation__

    ---

    **7→30** to **54→100** feature expansion

    Smart importance-based selection

</div>

---

## Simple Models Benchmark

Testing FeatCopilot with RandomForest (n_estimators=200, max_depth=20) and LogisticRegression/Ridge across 42 datasets to measure feature engineering impact.

### Summary Results

| Configuration | Datasets | Improved | Avg Improvement | Best Improvement |
|---------------|----------|----------|-----------------|------------------|
| **Tabular Engine** | 42 | 20 (48%) | +4.54% | +197% (delays_zurich) |
| **Tabular + LLM** | 42 | 23 (55%) | +6.12% | +420% (delays_zurich) |

### Classification Results (22 Datasets)

| Dataset | Baseline | +FeatCopilot | Improvement | +LLM | LLM Imp | Features |
|---------|----------|--------------|-------------|------|---------|----------|
| complex_classification | 0.8800 | **0.9300** | **+5.68%** | 0.9300 | +5.68% | 15→100 |
| road_safety | 0.7815 | 0.7895 | +1.02% | **0.8040** | **+2.88%** | 32→89 |
| customer_churn | 0.7575 | 0.7650 | +0.99% | 0.7650 | +0.99% | 10→82 |
| albert | 0.6558 | 0.6591 | +0.50% | 0.6529 | -0.44% | 31→100 |
| bioresponse | 0.7700 | 0.7729 | +0.38% | **0.7773** | **+0.95%** | 419→419 |
| higgs | 0.7129 | 0.7081 | -0.67% | **0.7154** | **+0.35%** | 24→100 |
| magic_telescope | 0.8509 | 0.8528 | +0.22% | 0.8528 | +0.22% | 10→65 |
| electricity | 0.8984 | 0.8986 | +0.03% | **0.9006** | **+0.25%** | 8→50 |
| covertype_cat | 0.8747 | 0.8749 | +0.02% | **0.8839** | **+1.05%** | 54→100 |
| titanic | 0.9218 | 0.9162 | -0.61% | 0.9162 | -0.61% | 7→27 |
| credit_card_fraud | 0.9840 | 0.9840 | +0.00% | 0.9840 | +0.00% | 30→100 |
| employee_attrition | 0.9558 | 0.9558 | +0.00% | 0.9558 | +0.00% | 11→74 |

### Regression Results (20 Datasets)

| Dataset | Baseline R² | +FeatCopilot R² | Improvement | +LLM R² | LLM Imp | Features |
|---------|-------------|-----------------|-------------|---------|---------|----------|
| delays_zurich | 0.0051 | 0.0153 | **+197%** | **0.0268** | **+420%** | 11→57 |
| abalone | 0.5287 | **0.5762** | **+8.98%** | 0.5769 | +9.12% | 7→30 |
| nyc_taxi | 0.6391 | 0.6253 | -2.17% | **0.6775** | **+6.01%** | 16→44 |
| bike_sharing | 0.8080 | 0.8082 | +0.02% | **0.8367** | **+3.55%** | 10→48 |
| wine_quality | 0.4972 | **0.5027** | **+1.12%** | 0.5067 | +1.91% | 11→45 |
| bike_sharing_inria | 0.6788 | 0.6861 | +1.07% | **0.6901** | **+1.67%** | 6→37 |
| miami_housing | 0.9146 | **0.9201** | **+0.61%** | 0.9214 | +0.74% | 13→70 |
| brazilian_houses | 0.9960 | 0.9964 | +0.04% | **1.0000** | **+0.40%** | 11→66 |
| diamonds | 0.9456 | 0.9461 | +0.05% | 0.9462 | +0.06% | 6→19 |
| house_prices | 0.9306 | 0.9308 | +0.02% | 0.9305 | -0.02% | 14→46 |
| cpu_act | 0.9798 | 0.9800 | +0.02% | **0.9803** | **+0.05%** | 21→100 |
| superconduct | 0.9300 | 0.9301 | +0.01% | 0.9299 | -0.01% | 79→100 |

!!! success "Key Insight"
    The LLM engine provides **additional value** on top of tabular features, particularly for datasets where domain knowledge helps (delays_zurich: +420%, nyc_taxi: +6.01%, bike_sharing: +3.55%).

---

## AutoML Benchmark

Testing FeatCopilot with FLAML (120s time budget per model) across 41 datasets to evaluate feature engineering benefits with AutoML optimization.

### Summary

| Metric | Value |
|--------|-------|
| **Total Datasets** | 41 |
| **Classification** | 21 |
| **Regression** | 20 |
| **Improved** | 19 (46%) |
| **Best Improvement** | +8.55% (abalone) |

### Top Improvements

| Dataset | Task | Baseline | +FeatCopilot | Improvement | Features |
|---------|------|----------|--------------|-------------|----------|
| abalone | regression | 0.5384 | **0.5844** | **+8.55%** | 7→30 |
| credit_risk | classification | 0.6925 | **0.7100** | **+2.53%** | 10→86 |
| delays_zurich | regression | 0.0810 | **0.0828** | **+2.24%** | 11→57 |
| mercedes_benz | regression | 0.5763 | **0.5866** | **+1.79%** | 359→359 |
| complex_classification | classification | 0.9100 | **0.9225** | **+1.37%** | 15→100 |
| eye_movements | classification | 0.6649 | **0.6721** | **+1.09%** | 23→97 |
| bioresponse | classification | 0.7802 | **0.7875** | **+0.93%** | 419→419 |
| medical_diagnosis | classification | 0.8400 | **0.8467** | **+0.79%** | 12→72 |

!!! note "AutoML Observations"
    With AutoML (FLAML), improvements are more modest because the framework already performs internal feature selection and hyperparameter tuning. FeatCopilot still provides value by generating meaningful derived features that AutoML can leverage.

---

## When FeatCopilot Excels

Based on comprehensive benchmarking across 42 datasets, FeatCopilot provides the most value in these scenarios:

| Scenario | Expected Benefit | Evidence |
|----------|------------------|----------|
| **Low baseline performance** | **Very High** (+50-400%) | delays_zurich: +197% (tabular), +420% (LLM) |
| **Small feature sets** | **High** (+5-10%) | abalone: +8.98% (7→30 features) |
| **Domain-specific tasks** | **High** (+3-6%) | bike_sharing: +3.55%, nyc_taxi: +6.01% |
| **Complex classification** | **Medium** (+1-5%) | complex_classification: +5.68% |
| **Already high-performing** | **Low** (0-1%) | credit_card_fraud: +0.00% (baseline 0.984) |

!!! info "Key Insight"
    FeatCopilot provides the **largest improvements** on datasets where:

    1. **Baseline performance is poor** - More room for improvement
    2. **Feature set is small** - More potential for derived features
    3. **Domain knowledge helps** - LLM can suggest meaningful features

    Datasets already near perfect performance (>0.98) show minimal improvement, as expected.

---

## Running Benchmarks

### Quick Start

```bash
# Clone and install
git clone https://github.com/thinkall/featcopilot.git
cd featcopilot
pip install -e ".[benchmark]"

# Run simple models benchmark (42 datasets)
python -m benchmarks.simple_models.run_simple_models_benchmark --all

# Run with LLM engine
python -m benchmarks.simple_models.run_simple_models_benchmark --all --with-llm

# Run AutoML benchmark (42 datasets)
python -m benchmarks.automl.run_automl_benchmark --all

# Run FE tools comparison
python -m benchmarks.compare_tools.run_fe_tools_comparison --all
```

### Available Benchmarks

| Benchmark | Command | Description |
|-----------|---------|-------------|
| **Simple Models** | `python -m benchmarks.simple_models.run_simple_models_benchmark --all` | RF/Ridge on 42 datasets |
| **Simple Models + LLM** | `python -m benchmarks.simple_models.run_simple_models_benchmark --all --with-llm` | With LLM-generated features |
| **AutoML** | `python -m benchmarks.automl.run_automl_benchmark --all` | FLAML on 42 datasets |
| **AutoML + LLM** | `python -m benchmarks.automl.run_automl_benchmark --all --with-llm` | FLAML with LLM features |
| **Tool Comparison** | `python -m benchmarks.compare_tools.run_fe_tools_comparison --all` | vs Featuretools, OpenFE, etc. |

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

Reports are saved with date suffix and LLM indicator:

```
benchmarks/
├── simple_models/
│   ├── SIMPLE_MODELS_BENCHMARK_20260204.md      # Tabular only
│   └── SIMPLE_MODELS_BENCHMARK_LLM_20260204.md  # Tabular + LLM
├── automl/
│   ├── AUTOML_BENCHMARK_20260204.md             # Tabular only
│   └── AUTOML_BENCHMARK_LLM_20260204.md         # Tabular + LLM
└── compare_tools/
    └── FE_TOOLS_COMPARISON_20260204.md
```

---

## Methodology

### Evaluation Protocol

- **Train/Test Split**: 80/20 with `random_state=42` for reproducibility
- **Metrics**: F1 (weighted) for classification, R² for regression
- **Models**:
  - Simple: RandomForest (n_estimators=200, max_depth=20), LogisticRegression/Ridge
  - AutoML: FLAML with 120s time budget
- **Feature Selection**: Importance-based with 1% threshold filter

### FeatCopilot Configuration

```python
from featcopilot import FeatureEngineer
from featcopilot.selection import FeatureSelector

# Standard configuration
engineer = FeatureEngineer(
    engines=["tabular"],
    max_features=100,
    verbose=1,
)

# With LLM engine
engineer = FeatureEngineer(
    engines=["tabular", "llm"],
    llm_config={"model": "gpt-4o-mini"},
    max_features=100,
)

# Feature selection
selector = FeatureSelector(
    method="importance",
    threshold=0.01,  # Keep features with >1% relative importance
)
```

---

## Dataset Coverage

42 datasets across classification and regression:

| Category | Count | Examples |
|----------|-------|----------|
| **Classification** | 22 | titanic, higgs, covertype, credit, albert, eye_movements |
| **Regression** | 20 | house_prices, diamonds, abalone, cpu_act, superconduct |

Dataset sources include:
- **OpenML** (INRIA benchmark suite)
- **Kaggle** (popular ML datasets)
- **Custom synthetic** (complex_classification, complex_regression)

---

## Conclusion

FeatCopilot demonstrates **consistent improvements** across diverse datasets:

1. **Simple Models**: +4.54% average, up to +197% maximum improvement
2. **With LLM**: Additional +1.5% on average, with some datasets seeing +200% more improvement
3. **AutoML**: +8.55% best improvement, 46% datasets improved

**Best use cases**:
- Datasets with poor baseline performance
- Small feature sets that benefit from derived features
- Tasks where domain knowledge helps (LLM engine)

**Current limitations**:
- Near-perfect baselines show minimal improvement
- Very high-dimensional datasets (400+ features) see less benefit
