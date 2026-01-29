# Engines API

Documentation for feature engineering engines.

## TabularEngine

```python
from featcopilot.engines import TabularEngine
```

### Constructor

```python
TabularEngine(
    polynomial_degree: int = 2,
    interaction_only: bool = False,
    include_transforms: List[str] = ['log', 'sqrt', 'square'],
    max_features: Optional[int] = None,
    verbose: bool = False
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `polynomial_degree` | int | 2 | Max polynomial degree (1-4) |
| `interaction_only` | bool | False | Only interactions, no powers |
| `include_transforms` | list | `['log', 'sqrt', 'square']` | Math transforms |
| `max_features` | int | None | Maximum features |
| `verbose` | bool | False | Verbose output |

### Methods

- `fit(X, y=None)` - Fit to data
- `transform(X)` - Generate features
- `fit_transform(X, y=None)` - Fit and transform
- `get_feature_names()` - Get feature names
- `get_feature_set()` - Get FeatureSet with metadata

### Example

```python
engine = TabularEngine(
    polynomial_degree=2,
    include_transforms=['log', 'sqrt']
)
X_fe = engine.fit_transform(X)
```

---

## TimeSeriesEngine

```python
from featcopilot.engines import TimeSeriesEngine
```

### Constructor

```python
TimeSeriesEngine(
    features: List[str] = ['basic_stats', 'distribution', 'autocorrelation'],
    window_sizes: List[int] = [5, 10, 20],
    n_fft_coefficients: int = 10,
    n_autocorr_lags: int = 10,
    max_features: Optional[int] = None,
    verbose: bool = False
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `features` | list | `['basic_stats', ...]` | Feature groups to extract |
| `window_sizes` | list | `[5, 10, 20]` | Rolling window sizes |
| `n_fft_coefficients` | int | 10 | FFT coefficients |
| `n_autocorr_lags` | int | 10 | Autocorrelation lags |

### Feature Groups

- `basic_stats` - mean, std, min, max, etc.
- `distribution` - skewness, kurtosis, quantiles
- `autocorrelation` - lag correlations
- `peaks` - peak/trough detection
- `trends` - slope, change rate
- `rolling` - rolling window stats
- `fft` - frequency domain features

### Example

```python
engine = TimeSeriesEngine(
    features=['basic_stats', 'trends', 'fft']
)
X_fe = engine.fit_transform(time_series_df)
```

---

## RelationalEngine

```python
from featcopilot.engines import RelationalEngine
```

### Constructor

```python
RelationalEngine(
    aggregation_functions: List[str] = ['mean', 'sum', 'count', 'max', 'min'],
    max_depth: int = 2,
    max_features: Optional[int] = None,
    verbose: bool = False
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `aggregation_functions` | list | `['mean', 'sum', ...]` | Aggregation functions |
| `max_depth` | int | 2 | Feature synthesis depth |

### Methods

#### add_relationship

```python
def add_relationship(
    self,
    child_table: str,
    parent_table: str,
    key_column: str,
    parent_key: Optional[str] = None
) -> "RelationalEngine"
```

Define a relationship between tables.

### Example

```python
engine = RelationalEngine()
engine.add_relationship('orders', 'customers', 'customer_id')

X_fe = engine.fit_transform(
    orders_df,
    related_tables={'customers': customers_df}
)
```

---

## TextEngine

```python
from featcopilot.engines import TextEngine
```

### Constructor

```python
TextEngine(
    features: List[str] = ['length', 'word_count', 'char_stats'],
    max_vocab_size: int = 5000,
    n_components: int = 50,
    max_features: Optional[int] = None,
    verbose: bool = False
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `features` | list | `['length', ...]` | Feature types |
| `max_vocab_size` | int | 5000 | TF-IDF vocabulary |
| `n_components` | int | 50 | SVD components |

### Feature Types

- `length` - character and word counts
- `word_count` - word-level statistics
- `char_stats` - character composition
- `tfidf` - TF-IDF with dimensionality reduction

### Example

```python
engine = TextEngine(
    features=['length', 'char_stats', 'tfidf']
)
X_fe = engine.fit_transform(text_df, text_columns=['description'])
```

---

## Base Classes

### BaseEngine

```python
from featcopilot.core import BaseEngine
```

Abstract base class for all engines.

```python
class BaseEngine(ABC):
    def fit(self, X, y=None, **kwargs) -> "BaseEngine": ...
    def transform(self, X, **kwargs) -> pd.DataFrame: ...
    def fit_transform(self, X, y=None, **kwargs) -> pd.DataFrame: ...
    def get_feature_names(self) -> List[str]: ...
    def get_feature_metadata(self) -> Dict[str, Any]: ...
```

### Feature

```python
from featcopilot.core import Feature, FeatureType, FeatureOrigin
```

Feature representation with metadata.

```python
@dataclass
class Feature:
    name: str
    dtype: FeatureType
    origin: FeatureOrigin
    source_columns: List[str]
    transformation: str
    explanation: Optional[str]
    code: Optional[str]
    importance: Optional[float]
    metadata: Dict[str, Any]
```

### FeatureSet

```python
from featcopilot.core import FeatureSet
```

Collection of features with operations.

```python
class FeatureSet:
    def add(self, feature: Feature) -> None: ...
    def remove(self, name: str) -> Optional[Feature]: ...
    def get(self, name: str) -> Optional[Feature]: ...
    def filter_by_origin(self, origin: FeatureOrigin) -> "FeatureSet": ...
    def filter_by_importance(self, min_importance: float) -> "FeatureSet": ...
    def get_explanations(self) -> Dict[str, str]: ...
```
