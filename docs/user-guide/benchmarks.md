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

## Available Benchmark Reports

All benchmark reports are available in the repository under `benchmarks/`:

| Report | Description | Location |
|--------|-------------|----------|
| Feature Engineering | Core FeatCopilot vs baseline models | `benchmarks/feature_engineering/BENCHMARK_REPORT.md` |
| LLM-Powered | SemanticEngine with GitHub Copilot | `benchmarks/feature_engineering/LLM_BENCHMARK_REPORT.md` |
| Tool Comparison | FeatCopilot vs Featuretools, TSFresh, AutoFeat | `benchmarks/compare_tools/COMPARISON_BENCHMARK_REPORT.md` |
| AutoML (30s) | FLAML, AutoGluon, H2O integration | `benchmarks/automl/AUTOML_BENCHMARK_REPORT.md` |
| FLAML (90s) | FLAML with extended time budget | `benchmarks/automl/FLAML_90S_BENCHMARK_REPORT.md` |

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

# Install dependencies (for basic benchmarks)
pip install -e ".[dev]"

# Install dependencies (for AutoML benchmarks)
pip install -e ".[benchmark]"

# Run feature engineering benchmarks
python -m benchmarks.feature_engineering.run_benchmark

# Run tool comparison benchmarks
python -m benchmarks.compare_tools.run_comparison_benchmark

# Run AutoML benchmarks
python -m benchmarks.automl.run_automl_benchmark --frameworks flaml autogluon h2o --time-budget 30
```

Reports are saved to:

- `benchmarks/feature_engineering/BENCHMARK_REPORT.md`
- `benchmarks/feature_engineering/LLM_BENCHMARK_REPORT.md`
- `benchmarks/compare_tools/COMPARISON_BENCHMARK_REPORT.md`
- `benchmarks/automl/AUTOML_BENCHMARK_REPORT.md`

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

FeatCopilot can be combined with AutoML frameworks. However, modern AutoML frameworks already perform extensive feature engineering and model selection, which may overlap with FeatCopilot's capabilities.

### Summary (30s Time Budget)

| Framework | Datasets | Avg Improvement | Positive Improvements |
|-----------|----------|-----------------|----------------------|
| FLAML | 15 | -0.69% | 3/15 (20%) |
| AutoGluon | 18 | -1.31% | 3/18 (17%) |
| H2O | 18 | -0.77% | 4/18 (22%) |
| **Overall** | **51 runs** | **-0.94%** | **8/51 (16%)** |

### FLAML Integration Results (30s)

| Dataset | Task | Baseline | +FeatCopilot | Change | Train Time (B/E) |
|---------|------|----------|--------------|--------|------------------|
| Titanic | Classification | 0.9665 | 0.9665 | +0.00% | 30.1s / 30.3s |
| House Prices | Regression | 0.9242 | 0.9136 | -1.15% | 30.4s / 30.1s |
| Credit Card Fraud | Classification | 0.9860 | 0.9860 | +0.00% | 30.0s / 30.2s |
| Medical Diagnosis | Classification | 0.8567 | 0.8600 | **+0.39%** | 30.7s / 30.6s |
| News Headlines | Text Classification | 0.9800 | 0.9820 | **+0.20%** | 34.7s / 54.0s |
| E-commerce Products | Text Regression | 0.4626 | 0.4658 | **+0.69%** | 30.3s / 30.2s |

**Average improvement with FLAML (30s)**: -0.69%

### FLAML Integration Results (90s)

With extended time budget, FLAML shows improved results:

| Dataset | Task | Baseline | +FeatCopilot | Change | Train Time (B/E) |
|---------|------|----------|--------------|--------|------------------|
| Titanic | Classification | 0.9665 | 0.9665 | +0.00% | 90.0s / 90.2s |
| House Prices | Regression | 0.9242 | 0.9187 | -0.60% | 90.4s / 91.6s |
| Credit Card Fraud | Classification | 0.9860 | 0.9860 | +0.00% | 90.9s / 90.1s |
| Medical Diagnosis | Classification | 0.8533 | 0.8533 | +0.00% | 91.0s / 90.5s |
| News Headlines | Text Classification | 0.9800 | 0.9960 | **+1.63%** | 90.2s / 90.9s |
| E-commerce Products | Text Regression | 0.4649 | 0.4658 | **+0.20%** | 90.4s / 90.1s |

**Average improvement with FLAML (90s)**: -0.12%

### AutoGluon Integration Results

| Dataset | Task | Baseline | +FeatCopilot | Change | Train Time (B/E) |
|---------|------|----------|--------------|--------|------------------|
| Titanic | Classification | 0.9665 | 0.9497 | -1.73% | 14.7s / 11.4s |
| Credit Card Fraud | Classification | 0.9860 | 0.9860 | +0.00% | 16.1s / 17.4s |
| Server Latency | Time Series | 0.9956 | 0.9957 | **+0.01%** | 31.1s / 31.2s |
| News Headlines | Text Classification | 0.9840 | 0.9960 | **+1.22%** | 31.2s / 31.2s |
| Customer Support | Text Classification | 1.0000 | 1.0000 | +0.00% | 38.5s / 31.4s |

**Average improvement with AutoGluon**: -1.31%

### H2O Integration Results

| Dataset | Task | Baseline | +FeatCopilot | Change | Train Time (B/E) |
|---------|------|----------|--------------|--------|------------------|
| Credit Risk | Classification | 0.7075 | 0.7100 | **+0.35%** | 84.4s / 87.8s |
| Server Latency | Time Series | 0.9740 | 0.9950 | **+2.16%** | 83.1s / 83.9s |
| News Headlines | Text Classification | 0.9980 | 1.0000 | **+0.20%** | 85.5s / 85.6s |
| Employee Attrition | Classification | 0.9728 | 0.9728 | +0.00% | 85.7s / 86.6s |

**Average improvement with H2O**: -0.77%

### Running AutoML Benchmarks

```bash
# Install benchmark dependencies
pip install featcopilot[benchmark]

# Run benchmarks (30s time budget per task)
python -m benchmarks.automl.run_automl_benchmark --frameworks flaml autogluon h2o --time-budget 30

# Run FLAML only with 90s time budget
python -m benchmarks.automl.run_automl_benchmark --frameworks flaml --time-budget 90 --output benchmarks/automl/FLAML_90S_BENCHMARK_REPORT.md
```

## Comparison with Other Tools

FeatCopilot has been benchmarked against other popular feature engineering libraries.

### Performance Comparison

| Tool | Avg Score | Avg Improvement | Wins | Avg FE Time |
|------|-----------|-----------------|------|-------------|
| baseline | 0.8924 | - | - | 0.00s |
| **featcopilot** | 0.8942 | **+0.21%** | 1 | 1.03s |
| featuretools | 0.8947 | +0.27% | 3 | 0.11s |
| tsfresh | 0.8843 | -0.92% | 2 | 24.86s |
| autofeat | 0.8964 | +0.48% | 3 | 1246.91s |

### Dataset-by-Dataset Results

| Dataset | FeatCopilot | Featuretools | TSFresh | AutoFeat | Best Tool |
|---------|-------------|--------------|---------|----------|-----------|
| Titanic | 0.9609 | 0.9553 | 0.9497 | 0.9609 | **FeatCopilot/AutoFeat** |
| House Prices | 0.9094 | 0.9124 | 0.9118 | 0.9123 | Featuretools |
| Credit Card Fraud | 0.9790 | 0.9770 | **0.9840** | 0.9790 | TSFresh |
| Bike Sharing | 0.8324 | **0.8351** | 0.8175 | 0.8313 | Featuretools |
| Employee Attrition | 0.9762 | 0.9728 | 0.9660 | **0.9796** | AutoFeat |
| Credit Risk | 0.7050 | 0.7025 | 0.6950 | **0.7150** | AutoFeat |
| Medical Diagnosis | 0.8500 | 0.8533 | **0.8567** | 0.8367 | TSFresh |
| Complex Regression | 0.9120 | **0.9435** | 0.9105 | 0.9129 | Featuretools |
| Complex Classification | 0.9225 | 0.9000 | 0.8675 | **0.9400** | AutoFeat |

## Hugging Face Datasets Benchmark

FeatCopilot was benchmarked on publicly available Hugging Face datasets to demonstrate its performance on real-world data with various characteristics, including **text-to-numerical feature engineering**.

### Datasets Overview

| Dataset | Source | Rows | Features | Task | Description |
|---------|--------|------|----------|------|-------------|
| Spotify Tracks | `maharshipandya/spotify-tracks-dataset` | 114,000 | 13 numerical + 114 genres | Regression | Predict track popularity from audio features |
| Fake News | `GonzaloA/fake_news` | 24,353 | 2 text | Classification | Classify news as real or fake |

### Key Results

#### Spotify Tracks (Regression - Numerical + Genre Features)

!!! warning "Limited Predictability"
    Track popularity depends heavily on artist fame, marketing, and release timing - factors not captured in audio features. Audio features have near-zero correlation with popularity (~0.01-0.05). This is a dataset limitation, not a model limitation.

| Model | R² | MAE | RMSE | Notes |
|-------|-----|-----|------|-------|
| Ridge (baseline) | 0.026 | 17.2 | 21.7 | Audio + genre features |
| Ridge (+FeatCopilot) | 0.142 | 15.8 | 20.3 | +447% R², -8% MAE |
| GradientBoosting (baseline) | 0.193 | 15.1 | 19.7 | Audio + genre features |
| GradientBoosting (+FeatCopilot) | **0.215** | **14.9** | **19.4** | +11% R², -1% MAE |

Target: `popularity` (0-100 scale, mean=33, std=22)

Features: 13 numerical + 114 genre categories → 177 engineered | FE Time: 51s

!!! tip "Interpreting MAE"
    MAE of ~15 means predictions are off by ~15 popularity points on average (on a 0-100 scale). This is more interpretable than R²=0.2.

#### Fake News (Classification - Text Features)

Using **SemanticEngine** to convert text columns to numerical features:

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| LogisticRegression | 0.9625 | 0.9911 |
| GradientBoosting | 0.9760 | **0.9979** |

Features: 2 text → 22 numerical | FE Time: 6s

!!! success "Text Feature Engineering"
    The SemanticEngine automatically extracts 11 numerical features per text column without any LLM API calls, including word count, character length, sentence count, uppercase ratio, and more.

##### Advanced Text Features (Transformers + Spacy)

The **TextEngine** provides advanced text feature extraction using local transformers and spacy models:

| Method | Features | ROC-AUC | FE Time | Description |
|--------|----------|---------|---------|-------------|
| Basic (SemanticEngine) | 22 | 0.9969 | 5s | Fast rule-based text features |
| Advanced (sentiment+NER+POS) | 64 | **0.9984** | 1754s | Deep NLP features |
| Embeddings (sentence-transformers) | 128 | 0.9541 | 147s | Dense vector representations |

!!! tip "When to Use Advanced Text Features"
    - **Basic features** are sufficient for most tasks and run in seconds
    - **Advanced features** (sentiment, NER, POS) provide marginal improvement (+0.15%) but require ~30 min on CPU
    - **Embeddings** are useful when semantic similarity matters, but may not outperform explicit features for classification

### Text Feature Engineering (SemanticEngine)

FeatCopilot's SemanticEngine converts text columns into ML-ready numerical features:

```python
from featcopilot.llm.semantic_engine import SemanticEngine

# Convert text to numerical features
engine = SemanticEngine(enable_text_features=True, verbose=True)
X_numerical = engine.fit_transform(
    X_text,
    y,
    column_descriptions={"title": "News headline", "text": "Article content"},
    task_description="Classify news as real or fake"
)
```

**Features extracted per text column (11 total):**

| Feature | Description |
|---------|-------------|
| `{col}_char_length` | Character count |
| `{col}_word_count` | Word count |
| `{col}_avg_word_length` | Average word length |
| `{col}_sentence_count` | Sentence count (approximate) |
| `{col}_uppercase_ratio` | Ratio of uppercase characters |
| `{col}_digit_count` | Count of digits |
| `{col}_special_char_count` | Count of special characters |
| `{col}_unique_word_ratio` | Unique words / total words |
| `{col}_exclamation_count` | Exclamation marks (emphasis) |
| `{col}_question_count` | Question marks |
| `{col}_caps_word_ratio` | All-caps words ratio |

### Advanced Text Features (TextEngine)

For advanced NLP-based features, use the **TextEngine** with transformers and spacy:

```python
from featcopilot.engines.text import TextEngine

# Advanced text features (requires transformers, spacy, sentence-transformers)
engine = TextEngine(
    features=['sentiment', 'ner', 'pos', 'embeddings'],
    config={
        'embedding_dim': 32,  # PCA-reduced embedding dimensions
        'sentiment_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
        'spacy_model': 'en_core_web_sm',
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
    }
)
X_advanced = engine.fit_transform(X_text)
```

**Advanced feature types:**

| Feature Type | Model | Features per Column | Description |
|--------------|-------|---------------------|-------------|
| `sentiment` | cardiffnlp/twitter-roberta-base-sentiment-latest | 3 | Positive, negative, neutral scores |
| `ner` | spacy en_core_web_sm | 8 | Entity counts: PERSON, ORG, GPE, DATE, MONEY, PRODUCT, EVENT, LOC |
| `pos` | spacy en_core_web_sm | 10 | POS ratios: NOUN, VERB, ADJ, ADV, PROPN, etc. |
| `embeddings` | sentence-transformers/all-MiniLM-L6-v2 | 32 (configurable) | PCA-reduced sentence embeddings |

### Enhanced Time Series Features (tsfresh-inspired)

The TimeSeriesEngine now includes comprehensive features inspired by tsfresh:

| Feature Group | Features |
|---------------|----------|
| **entropy** | binned_entropy, sample_entropy, approximate_entropy |
| **energy** | abs_energy, mean_abs_change, mean_second_deriv_central, rms, crest_factor |
| **complexity** | cid_ce, c3, ratio_unique_values, has_duplicate, sum_reoccurring_values |
| **counts** | count_above_mean, count_below_mean, first_loc_max/min, last_loc_max/min, longest_strike_above/below_mean, number_crossings_mean, abs_sum_changes |

```python
from featcopilot.engines.timeseries import TimeSeriesEngine

# Use enhanced time series features
engine = TimeSeriesEngine(
    features=["basic_stats", "distribution", "autocorrelation", "entropy", "energy", "complexity", "counts"]
)
X_ts_features = engine.fit_transform(time_series_data)
```

### Running Hugging Face Benchmarks

```bash
# Install Hugging Face datasets
pip install datasets

# Run benchmarks
python benchmarks/huggingface/run_hf_benchmark.py
```

### Key Findings

- **FeatCopilot** provides competitive performance with fast feature engineering time (~1s)
- **AutoFeat** achieves best overall accuracy but with extremely long FE time (~1247s)
- **TSFresh** excels at fraud detection but is slow (~25s)
- **Featuretools** is fastest (~0.1s) but doesn't always improve over baseline

### Feature Comparison

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
