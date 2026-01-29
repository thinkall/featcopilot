# Feature Selection

After generating features, FeatCopilot automatically selects the most important ones to prevent overfitting and reduce dimensionality.

## Overview

The selection pipeline:

1. **Statistical Selection**: Filter by statistical significance
2. **Model-based Selection**: Rank by ML model importance
3. **Redundancy Elimination**: Remove highly correlated features
4. **Final Selection**: Keep top N features

## FeatureSelector

The unified selector combines multiple methods:

```python
from featcopilot.selection import FeatureSelector

selector = FeatureSelector(
    methods=['mutual_info', 'importance'],
    max_features=50,
    correlation_threshold=0.95,
    verbose=True
)

X_selected = selector.fit_transform(X, y)

# Get results
print(f"Selected: {len(selector.get_selected_features())} features")
print(f"Top features: {selector.get_ranking()[:5]}")
```

## Selection Methods

### Statistical Selection

```python
from featcopilot.selection import StatisticalSelector

# Mutual Information
selector = StatisticalSelector(
    method='mutual_info',
    max_features=30
)

# F-test (ANOVA)
selector = StatisticalSelector(
    method='f_test',
    max_features=30
)

# Chi-square (for categorical targets)
selector = StatisticalSelector(
    method='chi2',
    max_features=30
)

# Correlation with target
selector = StatisticalSelector(
    method='correlation',
    threshold=0.1  # Minimum correlation
)
```

### Model-based Selection

```python
from featcopilot.selection import ImportanceSelector

# Random Forest importance
selector = ImportanceSelector(
    model='random_forest',
    max_features=30,
    n_estimators=100
)

# Gradient Boosting
selector = ImportanceSelector(
    model='gradient_boosting',
    max_features=30
)

# XGBoost (if installed)
selector = ImportanceSelector(
    model='xgboost',
    max_features=30
)
```

### Redundancy Elimination

```python
from featcopilot.selection import RedundancyEliminator

eliminator = RedundancyEliminator(
    correlation_threshold=0.95,  # Remove if corr > 0.95
    method='pearson',            # pearson, spearman, kendall
    importance_scores=None       # Optional: keep more important feature
)

X_reduced = eliminator.fit_transform(X)

# See what was removed
print(f"Removed: {eliminator.get_removed_features()}")
```

## Combined Selection

### In AutoFeatureEngineer

```python
from featcopilot import AutoFeatureEngineer

engineer = AutoFeatureEngineer(
    engines=['tabular'],
    max_features=50,
    selection_methods=['mutual_info', 'importance'],
    correlation_threshold=0.95
)

# Selection happens automatically in fit_transform
X_selected = engineer.fit_transform(X, y)
```

### Manual Pipeline

```python
from featcopilot.engines import TabularEngine
from featcopilot.selection import FeatureSelector

# Generate features
engine = TabularEngine(polynomial_degree=2)
X_features = engine.fit_transform(X)

# Select best features
selector = FeatureSelector(
    methods=['mutual_info', 'f_test', 'importance'],
    max_features=30,
    correlation_threshold=0.95
)
X_selected = selector.fit_transform(X_features, y)
```

## Configuration

### Method Comparison

| Method | Best For | Speed | Handles Non-linear |
|--------|----------|-------|-------------------|
| `mutual_info` | General | Medium | ✅ Yes |
| `f_test` | Linear relationships | Fast | ❌ No |
| `chi2` | Categorical features | Fast | ❌ No |
| `correlation` | Quick filtering | Fast | ❌ No |
| `importance` | Complex patterns | Slow | ✅ Yes |

### Combining Methods

```python
# Union: feature selected by ANY method
selector = FeatureSelector(
    methods=['mutual_info', 'importance'],
    combination='union'
)

# Intersection: feature selected by ALL methods
selector = FeatureSelector(
    methods=['mutual_info', 'importance'],
    combination='intersection'
)
```

## Accessing Results

### Feature Scores

```python
# Combined scores (normalized 0-1)
scores = selector.get_feature_scores()

# Per-method scores
method_scores = selector.get_method_scores()
print(method_scores['mutual_info'])
print(method_scores['importance'])
```

### Feature Ranking

```python
# Sorted list of (feature, score) tuples
ranking = selector.get_ranking()

for i, (feature, score) in enumerate(ranking[:10], 1):
    print(f"{i}. {feature}: {score:.4f}")
```

### Selected Features

```python
selected = selector.get_selected_features()
print(f"Selected {len(selected)} features:")
print(selected)
```

## Best Practices

### 1. Use Multiple Methods

```python
# More robust selection
selector = FeatureSelector(
    methods=['mutual_info', 'f_test', 'importance']
)
```

### 2. Set Appropriate Thresholds

```python
# Conservative: fewer, stronger features
selector = FeatureSelector(
    max_features=20,
    correlation_threshold=0.90
)

# Liberal: more features
selector = FeatureSelector(
    max_features=100,
    correlation_threshold=0.99
)
```

### 3. Consider Task Type

```python
# Classification
selector = StatisticalSelector(method='mutual_info')

# Regression
selector = StatisticalSelector(method='f_test')
```

### 4. Handle Imbalanced Data

```python
# For imbalanced classification, importance-based selection
# often works better than statistical methods
selector = ImportanceSelector(
    model='random_forest',
    n_estimators=200
)
```

## Correlation Matrix

Visualize feature correlations:

```python
eliminator = RedundancyEliminator(correlation_threshold=0.95)
eliminator.fit(X)

# Get correlation matrix
corr_matrix = eliminator.get_correlation_matrix()

# Plot (requires matplotlib/seaborn)
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.savefig('correlation_matrix.png')
```
