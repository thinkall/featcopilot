# AutoFeatureEngineer API

The main entry point for FeatCopilot.

## Class Definition

```python
class AutoFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Main auto feature engineering class.
    
    Combines multiple engines and selection methods for comprehensive
    automated feature engineering with LLM capabilities.
    """
```

## Constructor

```python
AutoFeatureEngineer(
    engines: List[str] = ['tabular'],
    max_features: Optional[int] = None,
    selection_methods: List[str] = ['mutual_info', 'importance'],
    correlation_threshold: float = 0.95,
    llm_config: Optional[Dict] = None,
    verbose: bool = False
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engines` | list | `['tabular']` | Engines to use: `'tabular'`, `'timeseries'`, `'text'`, `'llm'` |
| `max_features` | int | None | Maximum features to generate/select |
| `selection_methods` | list | `['mutual_info', 'importance']` | Selection methods |
| `correlation_threshold` | float | 0.95 | Threshold for redundancy elimination |
| `llm_config` | dict | None | Configuration for LLM engine |
| `verbose` | bool | False | Enable verbose output |

### LLM Config Options

```python
llm_config = {
    'model': 'gpt-5',           # Model to use
    'max_suggestions': 20,       # Max features to suggest
    'domain': 'healthcare',      # Domain context
    'validate_features': True,   # Validate generated code
    'temperature': 0.3,          # Generation temperature
}
```

## Methods

### fit

```python
def fit(
    self,
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    column_descriptions: Optional[Dict[str, str]] = None,
    task_description: str = "prediction task",
    **fit_params
) -> "AutoFeatureEngineer"
```

Fit the feature engineer to the data.

**Parameters:**

- `X`: Input features (DataFrame or array)
- `y`: Target variable (optional)
- `column_descriptions`: Human-readable column descriptions (for LLM)
- `task_description`: Description of ML task (for LLM)
- `**fit_params`: Additional parameters passed to engines

**Returns:** self

### transform

```python
def transform(
    self,
    X: Union[pd.DataFrame, np.ndarray],
    **transform_params
) -> pd.DataFrame
```

Generate features from input data.

**Parameters:**

- `X`: Input features
- `**transform_params`: Additional parameters

**Returns:** DataFrame with generated features

### fit_transform

```python
def fit_transform(
    self,
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    column_descriptions: Optional[Dict[str, str]] = None,
    task_description: str = "prediction task",
    apply_selection: bool = True,
    **fit_params
) -> pd.DataFrame
```

Fit and transform in one step.

**Parameters:**

- `X`: Input features
- `y`: Target variable
- `column_descriptions`: Column descriptions for LLM
- `task_description`: Task description for LLM
- `apply_selection`: Whether to apply feature selection
- `**fit_params`: Additional parameters

**Returns:** DataFrame with generated and selected features

### get_feature_names

```python
def get_feature_names(self) -> List[str]
```

Get names of all generated features.

**Returns:** List of feature names

### explain_features

```python
def explain_features(self) -> Dict[str, str]
```

Get human-readable explanations for features.

**Returns:** Dictionary mapping feature names to explanations

### get_feature_code

```python
def get_feature_code(self) -> Dict[str, str]
```

Get Python code for generated features.

**Returns:** Dictionary mapping feature names to code strings

### generate_custom_features

```python
def generate_custom_features(
    self,
    prompt: str,
    n_features: int = 5
) -> List[Dict[str, Any]]
```

Generate custom features via LLM prompt.

**Parameters:**

- `prompt`: Natural language description
- `n_features`: Number of features to generate

**Returns:** List of feature definitions

## Properties

### feature_importances_

```python
@property
def feature_importances_(self) -> Optional[Dict[str, float]]
```

Feature importance scores (if selection was applied).

**Returns:** Dictionary mapping feature names to importance scores

## Examples

### Basic Usage

```python
from featcopilot import AutoFeatureEngineer

engineer = AutoFeatureEngineer(
    engines=['tabular'],
    max_features=50
)

X_fe = engineer.fit_transform(X, y)
```

### With LLM

```python
engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    llm_config={'model': 'gpt-5', 'domain': 'finance'}
)

X_fe = engineer.fit_transform(
    X, y,
    column_descriptions={
        'income': 'Annual income USD',
        'debt': 'Total debt'
    },
    task_description='Predict loan default'
)

# Get explanations
for feat, expl in engineer.explain_features().items():
    print(f"{feat}: {expl}")
```

### In Sklearn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('features', AutoFeatureEngineer(engines=['tabular'])),
    ('model', LogisticRegression())
])
```

## See Also

- [Engines](engines.md) - Individual engine documentation
- [LLM Module](llm.md) - LLM capabilities
- [Selection](selection.md) - Feature selection methods
