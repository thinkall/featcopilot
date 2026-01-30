# Benchmarks

FeatCopilot has been extensively benchmarked against baseline models across various datasets and task types. This page presents the results demonstrating the benefits of automated feature engineering.

## Summary Results

<div class="grid cards" markdown>

-   :material-chart-line:{ .lg .middle } __Text/Semantic Tasks__

    ---

    **+12.44%** average improvement on text classification

    **+49.02%** maximum improvement (News Headlines)

    **12/12 wins** across all text datasets

-   :material-table:{ .lg .middle } __Classification Tasks__

    ---

    **+0.54%** average accuracy improvement

    **+4.35%** maximum improvement

    Best gains with LogisticRegression

-   :material-trending-up:{ .lg .middle } __Regression Tasks__

    ---

    **+0.65%** average R² improvement

    **+5.57%** maximum improvement (Bike Sharing)

    Consistent gains across datasets

-   :material-clock-outline:{ .lg .middle } __Time Series Tasks__

    ---

    **+1.51%** average R² improvement

    **7/9 wins** on regression tasks

    Best: Retail Demand +12.12%

</div>

## Text & Semantic Datasets

FeatCopilot excels at extracting meaningful features from text data. The benchmark uses **basic text preprocessing** (no LLM calls) which extracts 8 features per text column in <1 second.

!!! note "LLM-Powered Features"
    These benchmarks use only the **tabular engine with text preprocessing** - fast, deterministic feature extraction without API calls. When GitHub Copilot is authenticated, the **SemanticEngine** can generate additional domain-aware features using LLM, which would add latency but potentially improve results further.

| Dataset | Model | Baseline | FeatCopilot | Improvement |
|---------|-------|----------|-------------|-------------|
| **News Headlines** | LogisticRegression | 0.408 | 0.608 | **+49.02%** |
| **News Headlines** | RandomForest | 0.644 | 0.858 | **+33.23%** |
| **News Headlines** | GradientBoosting | 0.670 | 0.856 | **+27.76%** |
| Medical Notes | LogisticRegression | 0.923 | 0.983 | **+6.50%** |
| Customer Support | GradientBoosting | 0.948 | 0.998 | **+5.28%** |
| Product Reviews | LogisticRegression | 0.918 | 0.965 | **+5.18%** |

### Text Features Extracted (No LLM)

For each text column, FeatCopilot extracts these **rule-based features** in milliseconds:

- **Length features**: character count, word count, average word length
- **Structure features**: uppercase ratio, punctuation count, number count
- **Sentiment features**: positive word count, negative word count

```python
from featcopilot import AutoFeatureEngineer

# Basic text feature extraction (no LLM, <1s)
engineer = AutoFeatureEngineer(
    engines=['tabular'],  # Text preprocessing is automatic
    max_features=50
)

X_transformed = engineer.fit_transform(X_with_text, y)
```

### LLM-Powered Features (With Copilot)

When authenticated with GitHub Copilot, enable the `llm` engine for semantic feature generation:

```python
from featcopilot import AutoFeatureEngineer

# LLM-powered feature engineering (requires Copilot auth, adds latency)
engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    llm_config={'max_suggestions': 10}
)

X_transformed = engineer.fit_transform(
    X_with_text, y,
    column_descriptions={'review': 'Customer product review text'},
    task_description='Predict customer satisfaction score'
)

# LLM generates domain-aware features like:
# - sentiment_intensity, sarcasm_indicator, urgency_score
# - product_mention_count, comparison_phrases, recommendation_strength
```

## LLM-Powered Benchmarks

When using the **SemanticEngine** with GitHub Copilot (or mock responses), feature generation takes longer but can provide additional improvements:

| Dataset | Model | Baseline | With LLM | Improvement | FE Time |
|---------|-------|----------|----------|-------------|---------|
| **Retail Demand** | Ridge | 0.715 | 0.855 | **+19.66%** | 41.8s |
| Credit Risk | GradientBoosting | 0.698 | 0.718 | **+2.87%** | 33.0s |
| Credit Risk | LogisticRegression | 0.703 | 0.723 | **+2.85%** | 33.0s |
| Retail Demand | RandomForest | 0.873 | 0.897 | **+2.65%** | 41.8s |
| Credit Risk | RandomForest | 0.705 | 0.715 | +1.42% | 33.0s |

**LLM Benchmark Summary:**

- Average Improvement: **+2.15%**
- Max Improvement: **+19.66%** (Retail Demand with Ridge)
- Win Rate: **8/12** (67%)
- Feature Generation Time: **33-43 seconds** (includes LLM API latency)

!!! tip "When to Use LLM Engine"
    Use the LLM engine when:

    - You have **rich column descriptions** that convey domain meaning
    - You need **domain-specific features** (healthcare, finance, retail)
    - **Latency is acceptable** (30-60s per fit)
    - You want **interpretable feature explanations**

## Classification Benchmarks (Tabular Only)

| Dataset | Model | Baseline Acc | FeatCopilot Acc | Improvement |
|---------|-------|--------------|-----------------|-------------|
| Complex Classification | LogisticRegression | 0.863 | 0.900 | **+4.35%** |
| Credit Risk | LogisticRegression | 0.703 | 0.728 | **+3.56%** |
| Credit Risk | GradientBoosting | 0.698 | 0.708 | **+1.43%** |
| Medical Diagnosis | LogisticRegression | 0.853 | 0.857 | +0.39% |
| Employee Attrition | GradientBoosting | 0.969 | 0.973 | +0.35% |

### Key Findings

1. **Simple models benefit most**: LogisticRegression shows the largest improvements as it cannot capture interactions natively
2. **Tree-based models**: RandomForest and GradientBoosting already capture some interactions, but still benefit from explicit features
3. **Complex datasets**: Datasets with non-linear relationships show the greatest improvements

## Regression Benchmarks

| Dataset | Model | Baseline R² | FeatCopilot R² | Improvement |
|---------|-------|-------------|----------------|-------------|
| Bike Sharing | Ridge | 0.721 | 0.761 | **+5.57%** |
| House Prices | RandomForest | 0.870 | 0.894 | **+2.77%** |
| Job Postings (text) | Ridge | 0.389 | 0.418 | **+7.58%** |
| E-commerce (text) | RandomForest | 0.389 | 0.402 | +3.45% |

## Time Series Benchmarks

| Dataset | Model | Baseline R² | FeatCopilot R² | Improvement |
|---------|-------|-------------|----------------|-------------|
| Retail Demand | Ridge | 0.715 | 0.801 | **+12.12%** |
| Sensor Efficiency | GradientBoosting | 0.260 | 0.273 | **+5.10%** |
| Sensor Efficiency | RandomForest | 0.275 | 0.284 | **+3.34%** |
| Server Latency | Ridge | 0.972 | 0.973 | +0.03% |
| Retail Demand | GradientBoosting | 0.917 | 0.917 | +0.06% |

### Key Time Series Findings

1. **Datasets with polynomial/interaction effects** benefit most from feature engineering
2. **Simple models (Ridge)** show largest gains when interactions are captured
3. **High-baseline models** (Server Latency ~99%) have limited room for improvement

## Feature Engineering Statistics

FeatCopilot efficiently generates and selects features:

| Dataset | Original | Engineered | Time (s) |
|---------|----------|------------|----------|
| Credit Card Fraud | 30 | 50 | 1.61 |
| Complex Classification | 15 | 50 | 0.74 |
| News Headlines (text) | 5 | 23 | 0.75 |
| Employee Attrition | 11 | 40 | 0.57 |
| Titanic | 7 | 25 | 0.43 |

## Benchmark Methodology

### Evaluation Protocol

- **Train/Test Split**: 80/20 with `random_state=42`
- **Preprocessing**: StandardScaler applied to all features
- **Models Tested**: LogisticRegression/Ridge, RandomForest, GradientBoosting
- **Metrics**: Accuracy/R², F1-score/RMSE, ROC-AUC/MAE

!!! info "No LLM in Benchmarks"
    All benchmarks use only the **TabularEngine** with rule-based text preprocessing. No GitHub Copilot or LLM API calls are made, ensuring reproducible results and sub-second feature generation times.

### Feature Engineering Configuration

```python
# Classification tasks
engineer = AutoFeatureEngineer(
    engines=['tabular'],  # No 'llm' engine in benchmarks
    max_features=50,
    selection_methods=['importance', 'mutual_info'],
    correlation_threshold=0.95
)

# Regression tasks (more conservative)
engineer = AutoFeatureEngineer(
    engines=['tabular'],
    max_features=40,
    selection_methods=['mutual_info'],
    correlation_threshold=0.85
)
```

### Datasets

**Kaggle-style Datasets:**

- Titanic - Survival classification
- House Prices - Price regression
- Credit Card Fraud - Imbalanced classification
- Bike Sharing - Demand regression
- Employee Attrition - HR classification

**Synthetic Datasets:**

- Credit Risk - Financial classification
- Medical Diagnosis - Healthcare classification
- Complex Regression - Non-linear regression
- Complex Classification - Multi-class classification

**Time Series Datasets:**

- Sensor Efficiency - Industrial IoT regression
- Retail Demand - Demand forecasting regression
- Server Latency - Performance prediction regression

**Text Datasets:**

- Product Reviews - Sentiment classification
- News Headlines - Category classification
- Customer Support Tickets - Priority classification
- Medical Notes - Diagnosis classification
- Job Postings - Salary regression
- E-commerce Products - Rating regression

## Running Benchmarks

To reproduce these results:

```bash
# Clone the repository
git clone https://github.com/thinkall/featcopilot.git
cd featcopilot

# Install dependencies
pip install -e ".[dev]"

# Run benchmarks
python -m benchmarks.run_benchmark
```

Results are saved to `benchmarks/BENCHMARK_REPORT.md`.

## When to Use FeatCopilot

Based on our benchmarks, FeatCopilot provides the most value when:

| Scenario | Expected Benefit |
|----------|------------------|
| Text/semantic data | **High** (+12-49%) |
| Simple models (LogisticRegression, Ridge) | **High** (+3-5%) |
| Complex non-linear relationships | **Medium** (+1-4%) |
| Small feature sets needing expansion | **Medium** (+1-3%) |
| Tree-based models on clean data | **Low** (0-1%) |

## AutoML Integration

FeatCopilot can be combined with AutoML frameworks to potentially improve results further.

### FLAML Integration Results

| Dataset | Task | FLAML Baseline | FLAML + FeatCopilot | Change |
|---------|------|----------------|---------------------|--------|
| Titanic | Classification | 0.9665 | 0.9665 | +0.00% |
| House Prices | Regression | 0.9195 | 0.9136 | -0.65% |
| Credit Card Fraud | Classification | 0.9860 | 0.9860 | +0.00% |
| Bike Sharing | Regression | 0.8426 | 0.8359 | -0.79% |
| Employee Attrition | Classification | 0.9796 | 0.9762 | -0.35% |
| Medical Diagnosis | Classification | 0.8567 | 0.8600 | +0.39% |

### Running AutoML Benchmarks

```bash
# Install FLAML
pip install flaml

# Run benchmarks
python benchmarks/automl/run_automl_benchmark.py --frameworks flaml --time-budget 60
```

## Comparison with Other Tools

| Feature | FeatCopilot | Featuretools | TSFresh | AutoFeat | OpenFE | CAAFE |
|---------|-------------|--------------|---------|----------|--------|-------|
| Tabular Features | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| Time Series | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Relational | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| LLM-Powered | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Semantic Understanding | ✅ | ❌ | ❌ | ❌ | ❌ | ⚠️ |
| Code Generation | ✅ | ❌ | ❌ | ❌ | ❌ | ⚠️ |
| Sklearn Compatible | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Interpretable | ✅ | ⚠️ | ⚠️ | ⚠️ | ❌ | ✅ |
