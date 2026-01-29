# LLM Module API

Documentation for LLM-powered feature engineering.

## SemanticEngine

```python
from featcopilot.llm import SemanticEngine
```

LLM-powered semantic feature engineering.

### Constructor

```python
SemanticEngine(
    model: str = 'gpt-5',
    max_suggestions: int = 20,
    validate_features: bool = True,
    domain: Optional[str] = None,
    verbose: bool = False
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `'gpt-5'` | LLM model to use |
| `max_suggestions` | int | 20 | Max features to suggest |
| `validate_features` | bool | True | Validate generated code |
| `domain` | str | None | Domain context |
| `verbose` | bool | False | Verbose output |

### Methods

#### fit

```python
def fit(
    self,
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    column_descriptions: Optional[Dict[str, str]] = None,
    task_description: str = "classification/regression task"
) -> "SemanticEngine"
```

#### get_feature_explanations

```python
def get_feature_explanations(self) -> Dict[str, str]
```

Get explanations for all generated features.

#### get_feature_code

```python
def get_feature_code(self) -> Dict[str, str]
```

Get Python code for all generated features.

#### suggest_more_features

```python
def suggest_more_features(
    self,
    focus_area: str,
    n_features: int = 5
) -> List[Dict[str, Any]]
```

Request additional feature suggestions in a specific area.

#### generate_custom_feature

```python
def generate_custom_feature(
    self,
    description: str,
    constraints: Optional[List[str]] = None
) -> Dict[str, Any]
```

Generate a specific feature from natural language.

### Example

```python
engine = SemanticEngine(
    model='gpt-5',
    domain='healthcare',
    max_suggestions=15
)

X_fe = engine.fit_transform(
    X, y,
    column_descriptions={'age': 'Patient age'},
    task_description='Predict diabetes'
)

# Get explanations
explanations = engine.get_feature_explanations()
```

---

## CopilotFeatureClient

```python
from featcopilot.llm import CopilotFeatureClient
```

GitHub Copilot SDK wrapper for feature engineering.

### Constructor

```python
CopilotFeatureClient(
    model: str = 'gpt-5',
    temperature: float = 0.3,
    timeout: float = 60.0
)
```

### Methods

#### start / stop

```python
async def start(self) -> "CopilotFeatureClient"
async def stop(self) -> None
```

Start/stop the Copilot client.

#### suggest_features

```python
async def suggest_features(
    self,
    column_info: Dict[str, str],
    task_description: str,
    column_descriptions: Optional[Dict[str, str]] = None,
    domain: Optional[str] = None,
    max_suggestions: int = 10
) -> List[Dict[str, Any]]
```

Get LLM suggestions for new features.

**Returns:** List of feature suggestions:
```python
[
    {
        'name': 'feature_name',
        'code': 'result = df["col1"] * df["col2"]',
        'explanation': 'Why this feature is useful',
        'source_columns': ['col1', 'col2']
    }
]
```

#### explain_feature

```python
async def explain_feature(
    self,
    feature_name: str,
    feature_code: str,
    column_descriptions: Optional[Dict[str, str]] = None,
    task_description: Optional[str] = None
) -> str
```

Get human-readable explanation of a feature.

#### generate_feature_code

```python
async def generate_feature_code(
    self,
    description: str,
    column_info: Dict[str, str],
    constraints: Optional[List[str]] = None
) -> str
```

Generate Python code for a described feature.

#### validate_feature_code

```python
async def validate_feature_code(
    self,
    code: str,
    sample_data: Optional[Dict[str, List]] = None
) -> Dict[str, Any]
```

Validate generated feature code.

**Returns:**
```python
{
    'valid': True/False,
    'error': 'Error message if invalid',
    'warnings': ['List of warnings']
}
```

### Synchronous Wrapper

```python
from featcopilot.llm import SyncCopilotFeatureClient

# Synchronous version for non-async contexts
client = SyncCopilotFeatureClient(model='gpt-5')
client.start()
suggestions = client.suggest_features(...)
client.stop()
```

---

## FeatureExplainer

```python
from featcopilot.llm import FeatureExplainer
```

Generate human-readable explanations.

### Constructor

```python
FeatureExplainer(model: str = 'gpt-5', verbose: bool = False)
```

### Methods

#### explain_feature

```python
def explain_feature(
    self,
    feature: Feature,
    column_descriptions: Optional[Dict[str, str]] = None,
    task_description: Optional[str] = None
) -> str
```

#### explain_features

```python
def explain_features(
    self,
    features: FeatureSet,
    column_descriptions: Optional[Dict[str, str]] = None,
    task_description: Optional[str] = None
) -> Dict[str, str]
```

#### generate_feature_report

```python
def generate_feature_report(
    self,
    features: FeatureSet,
    X: pd.DataFrame,
    column_descriptions: Optional[Dict[str, str]] = None,
    task_description: Optional[str] = None
) -> str
```

Generate comprehensive markdown report.

---

## FeatureCodeGenerator

```python
from featcopilot.llm import FeatureCodeGenerator
```

Generate feature code from natural language.

### Constructor

```python
FeatureCodeGenerator(
    model: str = 'gpt-5',
    validate: bool = True,
    verbose: bool = False
)
```

### Methods

#### generate

```python
def generate(
    self,
    description: str,
    columns: Dict[str, str],
    constraints: Optional[List[str]] = None,
    sample_data: Optional[pd.DataFrame] = None
) -> Feature
```

Generate a feature from description.

#### generate_batch

```python
def generate_batch(
    self,
    descriptions: List[str],
    columns: Dict[str, str],
    sample_data: Optional[pd.DataFrame] = None
) -> List[Feature]
```

Generate multiple features.

#### generate_domain_features

```python
def generate_domain_features(
    self,
    domain: str,
    columns: Dict[str, str],
    n_features: int = 5
) -> List[Feature]
```

Generate domain-specific features.

**Supported domains:** `'healthcare'`, `'finance'`, `'retail'`, `'telecom'`

### Example

```python
generator = FeatureCodeGenerator(model='gpt-5')

feature = generator.generate(
    description="Calculate BMI from height and weight",
    columns={'height_m': 'float', 'weight_kg': 'float'}
)

print(feature.code)
# result = df['weight_kg'] / (df['height_m'] ** 2)
```
