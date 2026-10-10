# Feature Engineering Engines

FeatCopilot provides multiple specialized engines for different types of data and feature generation strategies.

## TabularEngine

Generates features from numeric tabular data through mathematical transformations.

### Features Generated

- **Polynomial features**: x², x³, etc.
- **Interaction features**: x₁ × x₂
- **Mathematical transforms**: log, sqrt, exp, sin, cos
- **Ratio features**: x₁ / x₂
- **Difference features**: x₁ - x₂
- **Categorical encoding**: one-hot for low cardinality and cross-fitted target encoding for medium cardinality

### Usage

```python
from featcopilot.engines import TabularEngine

engine = TabularEngine(
    polynomial_degree=2,      # Max polynomial degree
    interaction_only=False,   # Include powers, not just interactions
    include_transforms=['log', 'sqrt', 'square'],
    keep_original_categorical=False,
    max_features=50,
    verbose=True
)

X_transformed = engine.fit_transform(X, y)
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `polynomial_degree` | int | 2 | Maximum polynomial degree (1-4) |
| `interaction_only` | bool | False | Only interactions, no powers |
| `include_transforms` | list | ['log', 'sqrt', 'square'] | Math transforms to apply |
| `max_features` | int | None | Maximum features to generate |
| `min_unique_values` | int | 5 | Minimum unique values for continuous |
| `encode_categorical` | bool | True | Encode supported categorical columns |
| `keep_original_categorical` | bool | True | Keep original categorical columns alongside encoded features |
| `onehot_ratio_threshold` | float | 0.05 | Maximum cardinality ratio for one-hot encoding |
| `target_encode_ratio_threshold` | float | 0.5 | Maximum cardinality ratio for target encoding |
| `target_encode_cv` | int | 5 | Cross-fitting folds used for training-row target encodings |
| `min_samples_per_category` | int | 3 | Minimum samples required for an encoded category |

When a target is provided, `fit_transform` uses out-of-fold target encodings for
the training rows. A later `transform` call uses category means learned from the
full training set. This distinction prevents each training row's own target
from leaking into its encoded feature.

### Available Transforms

```python
TRANSFORMS = [
    'log',      # log(1 + |x|)
    'log10',    # log10(|x| + 1)
    'sqrt',     # sqrt(|x|)
    'square',   # x²
    'cube',     # x³
    'reciprocal', # 1/x
    'exp',      # e^x (clipped)
    'tanh',     # tanh(x)
    'sin',      # sin(x)
    'cos',      # cos(x)
]
```

---

## PredictiveTabularEngine (opt-in)

The predictive engine uses a bounded numeric candidate pool and internal
learner-based validation, while preserving original categorical columns:

```python
from sklearn.linear_model import Ridge
from featcopilot.engines.predictive import PredictiveTabularEngine

engine = PredictiveTabularEngine(
    task="regression",
    estimator=Ridge(alpha=1.0),
    max_candidates=1000,
    max_features=50,
    random_state=42,
)
X_train_fe = engine.fit_transform(X_train, y_train)
X_test_fe = engine.transform(X_test)
```

For the main API, use
`AutoFeatureEngineer(selection_strategy="predictive", task="regression")`.
For a single-engine sklearn wrapper, use
`FeatureEngineerTransformer(engine="predictive", task="regression")`.
See [predictive selection](selection.md#predictive-tabular-strategy-opt-in)
for selection semantics and limitations.

---

## TimeSeriesEngine

Extracts statistical and frequency-domain features from time series data.

### Features Generated

- **Basic statistics**: mean, std, min, max, median
- **Distribution**: skewness, kurtosis, quantiles
- **Autocorrelation**: lag correlations
- **Trends**: slope, change rate
- **Frequency**: FFT coefficients
- **Peaks**: count, locations

### Usage

```python
from featcopilot.engines import TimeSeriesEngine

engine = TimeSeriesEngine(
    features=['basic_stats', 'distribution', 'autocorrelation', 'trends'],
    window_sizes=[5, 10, 20],
    n_fft_coefficients=10,
    verbose=True
)

X_features = engine.fit_transform(time_series_df)
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `features` | list | ['basic_stats', 'distribution', 'autocorrelation'] | Feature groups |
| `window_sizes` | list | [5, 10, 20] | Rolling window sizes |
| `n_fft_coefficients` | int | 10 | FFT coefficients to extract |
| `n_autocorr_lags` | int | 10 | Autocorrelation lags |

### Feature Groups

| Group | Features |
|-------|----------|
| `basic_stats` | mean, std, min, max, range, median, sum, var, cv |
| `distribution` | skewness, kurtosis, q10, q25, q75, q90, iqr |
| `autocorrelation` | autocorr_lag1, autocorr_lag2, ... |
| `peaks` | n_peaks, n_troughs, peak_mean, trough_mean |
| `trends` | slope, intercept, change, mean_abs_change |
| `rolling` | rolling window statistics |
| `fft` | fft_coeff_1, fft_coeff_2, ..., spectral_energy |

---

## RelationalEngine

Generates aggregation features from related tables, inspired by Featuretools.

For leakage-controlled temporal histories, use
`PointInTimeRelationalEngine` with declared entity keys, prediction timestamps
and related-table availability timestamps. It preserves a frozen history
snapshot and excludes equal-time/future events for each query.
See the [point-in-time example](../examples/relational-feature-engineering.md#prediction-time-history-features).
The legacy engine below does not automatically enforce those boundaries.

### Features Generated

- **Aggregations**: mean, sum, count, min, max per group
- **Self-aggregations**: statistics by categorical columns

### Usage

```python
from featcopilot.engines import RelationalEngine

engine = RelationalEngine(
    aggregation_functions=['mean', 'sum', 'count', 'max', 'min'],
    max_depth=2
)

# Define relationships
engine.add_relationship(
    child_table='orders',
    parent_table='customers',
    key_column='customer_id'
)

# Transform with related tables
X_features = engine.fit_transform(
    orders_df,
    related_tables={'customers': customers_df}
)
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `aggregation_functions` | list | ['mean', 'sum', 'count', 'max', 'min'] | Aggregations |
| `max_depth` | int | 2 | Depth for feature synthesis |

### Aggregation Functions

```python
AGGREGATIONS = [
    'mean', 'sum', 'min', 'max', 'count',
    'std', 'median', 'first', 'last', 'nunique'
]
```

---

## TextEngine

Extracts features from text columns.

### Features Generated

- **Length features**: character count, word count
- **Character statistics**: uppercase ratio, digit ratio
- **Word statistics**: average word length, unique word ratio
- **TF-IDF**: reduced dimensionality text embeddings

### Usage

```python
from featcopilot.engines import TextEngine

engine = TextEngine(
    features=['length', 'word_count', 'char_stats', 'tfidf'],
    max_vocab_size=5000,
    n_components=50
)

X_features = engine.fit_transform(
    text_df,
    text_columns=['description', 'title']
)
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `features` | list | ['length', 'word_count', 'char_stats'] | Feature types |
| `max_vocab_size` | int | 5000 | TF-IDF vocabulary size |
| `n_components` | int | 50 | SVD components for TF-IDF |

---

## Combining Engines

Use multiple engines together:

```python
from featcopilot import AutoFeatureEngineer

engineer = AutoFeatureEngineer(
    engines=['tabular', 'timeseries', 'text'],
    max_features=100
)

# All engines run and features are combined
X_transformed = engineer.fit_transform(X, y)
```

Features from all engines are:

1. Generated independently
2. Combined into a single DataFrame
3. Selected based on importance
4. Deduplicated by correlation
