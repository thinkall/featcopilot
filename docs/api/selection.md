# Selection API

Documentation for feature selection methods.

## FeatureSelector

```python
from featcopilot.selection import FeatureSelector
```

Unified selector combining multiple methods.

### Constructor

```python
FeatureSelector(
    methods: List[str] = ['mutual_info', 'importance'],
    max_features: Optional[int] = None,
    correlation_threshold: float = 0.95,
    combination: str = 'union',
    verbose: bool = False
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `methods` | list | `['mutual_info', 'importance']` | Selection methods |
| `max_features` | int | None | Maximum features to select |
| `correlation_threshold` | float | 0.95 | Redundancy threshold |
| `combination` | str | `'union'` | How to combine methods |
| `verbose` | bool | False | Verbose output |

### Available Methods

- `'mutual_info'` - Mutual information
- `'f_test'` - F-test (ANOVA)
- `'chi2'` - Chi-square test
- `'correlation'` - Correlation with target
- `'importance'` - Random Forest importance
- `'xgboost'` - XGBoost importance

### Methods

#### fit

```python
def fit(
    self,
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray]
) -> "FeatureSelector"
```

#### transform

```python
def transform(self, X: pd.DataFrame) -> pd.DataFrame
```

#### get_selected_features

```python
def get_selected_features(self) -> List[str]
```

#### get_feature_scores

```python
def get_feature_scores(self) -> Dict[str, float]
```

Get combined normalized scores.

#### get_method_scores

```python
def get_method_scores(self) -> Dict[str, Dict[str, float]]
```

Get scores from each individual method.

#### get_ranking

```python
def get_ranking(self) -> List[tuple]
```

Get sorted list of (feature, score) tuples.

### Example

```python
selector = FeatureSelector(
    methods=['mutual_info', 'importance'],
    max_features=30,
    correlation_threshold=0.95
)

X_selected = selector.fit_transform(X, y)

print(f"Selected: {selector.get_selected_features()}")
print(f"Top 5: {selector.get_ranking()[:5]}")
```

---

## StatisticalSelector

```python
from featcopilot.selection import StatisticalSelector
```

Feature selection based on statistical tests.

### Constructor

```python
StatisticalSelector(
    method: str = 'mutual_info',
    max_features: Optional[int] = None,
    threshold: Optional[float] = None,
    verbose: bool = False
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | `'mutual_info'` | Selection method |
| `max_features` | int | None | Max features |
| `threshold` | float | None | Minimum score threshold |

### Methods

- `'mutual_info'` - Mutual information (handles non-linear)
- `'f_test'` - F-test/ANOVA (linear relationships)
- `'chi2'` - Chi-square (categorical targets)
- `'correlation'` - Absolute correlation with target

### Example

```python
# Mutual information
selector = StatisticalSelector(
    method='mutual_info',
    max_features=20
)

# F-test
selector = StatisticalSelector(
    method='f_test',
    threshold=0.05  # p-value threshold
)
```

---

## ImportanceSelector

```python
from featcopilot.selection import ImportanceSelector
```

Feature selection based on model importance.

### Constructor

```python
ImportanceSelector(
    model: str = 'random_forest',
    max_features: Optional[int] = None,
    threshold: Optional[float] = None,
    n_estimators: int = 100,
    verbose: bool = False
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `'random_forest'` | Model type |
| `max_features` | int | None | Max features |
| `threshold` | float | None | Min importance |
| `n_estimators` | int | 100 | Number of trees |

### Model Types

- `'random_forest'` - Random Forest (default)
- `'gradient_boosting'` - Gradient Boosting
- `'xgboost'` - XGBoost (if installed)

### Example

```python
selector = ImportanceSelector(
    model='random_forest',
    max_features=30,
    n_estimators=200
)

X_selected = selector.fit_transform(X, y)
importances = selector.get_feature_scores()
```

---

## RedundancyEliminator

```python
from featcopilot.selection import RedundancyEliminator
```

Remove redundant (highly correlated) features.

### Constructor

```python
RedundancyEliminator(
    correlation_threshold: float = 0.95,
    method: str = 'pearson',
    importance_scores: Optional[Dict[str, float]] = None,
    verbose: bool = False
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `correlation_threshold` | float | 0.95 | Correlation threshold |
| `method` | str | `'pearson'` | Correlation method |
| `importance_scores` | dict | None | Pre-computed importance |

### Correlation Methods

- `'pearson'` - Pearson correlation
- `'spearman'` - Spearman rank correlation
- `'kendall'` - Kendall tau correlation

### Methods

#### get_removed_features

```python
def get_removed_features(self) -> List[str]
```

Get list of removed features.

#### get_correlation_matrix

```python
def get_correlation_matrix(self) -> pd.DataFrame
```

Get computed correlation matrix.

### Example

```python
eliminator = RedundancyEliminator(
    correlation_threshold=0.95,
    importance_scores={'feat1': 0.8, 'feat2': 0.3}
)

X_reduced = eliminator.fit_transform(X)

print(f"Removed: {eliminator.get_removed_features()}")
```

---

## Base Classes

### BaseSelector

```python
from featcopilot.core import BaseSelector
```

Abstract base class for selectors.

```python
class BaseSelector(ABC):
    def fit(self, X, y, **kwargs) -> "BaseSelector": ...
    def transform(self, X, **kwargs) -> pd.DataFrame: ...
    def fit_transform(self, X, y, **kwargs) -> pd.DataFrame: ...
    def get_selected_features(self) -> List[str]: ...
    def get_feature_scores(self) -> Dict[str, float]: ...
```

### SelectorConfig

```python
from featcopilot.core import SelectorConfig

config = SelectorConfig(
    max_features=30,
    min_importance=0.01,
    correlation_threshold=0.95
)
```
