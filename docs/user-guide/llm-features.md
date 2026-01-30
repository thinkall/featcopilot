# LLM-Powered Features

FeatCopilot's LLM integration via GitHub Copilot SDK is its key differentiator, enabling semantic understanding and intelligent feature generation.

## Overview

The LLM engine provides:

- **Semantic Feature Discovery**: Understands column meanings from names/descriptions
- **Domain-Aware Generation**: Tailored features for specific domains
- **Feature Explanations**: Human-readable descriptions for stakeholders
- **Code Generation**: Automatic Python code for custom features
- **Iterative Refinement**: Improve features based on feedback

## SemanticEngine

The core LLM-powered engine:

```python
from featcopilot.llm import SemanticEngine

engine = SemanticEngine(
    model='gpt-5.2',
    max_suggestions=20,
    validate_features=True,
    domain='healthcare',
    verbose=True
)

X_features = engine.fit_transform(
    X, y,
    column_descriptions={
        'age': 'Patient age in years',
        'bmi': 'Body Mass Index',
        'glucose': 'Fasting blood glucose mg/dL',
        'hba1c': 'Hemoglobin A1c percentage'
    },
    task_description='Predict Type 2 diabetes diagnosis'
)
```

## Configuration

### LLM Config Options

```python
llm_config = {
    'model': 'gpt-5.2',           # Model: gpt-5.2, gpt-4.1, etc.
    'max_suggestions': 20,       # Max features to suggest
    'domain': 'healthcare',      # Domain context
    'validate_features': True,   # Validate generated code
    'temperature': 0.3,          # Generation temperature
}

engineer = AutoFeatureEngineer(
    engines=['llm'],
    llm_config=llm_config
)
```

### Listing Available Models

Use the model utilities to discover supported models directly from the Copilot client:

```python
from featcopilot.utils import list_models, get_model_info, get_model_names, fetch_models

# Fetch and list all available models from Copilot
models = list_models()
for m in models:
    model_id = m.get('id') or m.get('name')
    print(f"{model_id}: {m.get('description', '')}")

# List with verbose output to logger
list_models(verbose=True)
# Output:
# Available models from Copilot:
# ------------------------------------------------------------
#   gpt-5.2 (OpenAI) - Latest gpt-5.2 model...
#   claude-sonnet-4 (Anthropic) - Claude Sonnet 4...
#   ...
# ------------------------------------------------------------
# Total: N models

# Get just the model names/identifiers
model_names = get_model_names()
print(model_names)  # ['gpt-5.2', 'claude-sonnet-4', ...]

# Filter by provider
openai_models = list_models(provider='OpenAI')
claude_models = list_models(provider='Anthropic')

# Get info about a specific model
info = get_model_info('gpt-5.2')
if info:
    print(info)

# Force refresh the cached model list
models = list_models(force_refresh=True)

# Check if a model is valid
from featcopilot.utils import is_valid_model
if is_valid_model('gpt-5.2'):
    print("Model is available")
```

### Supported Models

The available models are retrieved dynamically from the Copilot client. Common models include:

| Model | Provider | Description |
|-------|----------|-------------|
| `gpt-5.2` | OpenAI | Latest gpt-5.2 model (default) |
| `gpt-5.2-mini` | OpenAI | Smaller, faster gpt-5.2 variant |
| `gpt-5.1-codex` | OpenAI | Optimized for code generation |
| `gpt-4.1` | OpenAI | Fast and efficient |
| `claude-sonnet-4` | Anthropic | Balanced performance |
| `claude-sonnet-4.5` | Anthropic | Improved reasoning |
| `claude-haiku-4.5` | Anthropic | Fast and efficient |
| `claude-opus-4.5` | Anthropic | Premium quality |
| `gemini-3-pro-preview` | Google | Gemini 3 Pro Preview |

> **Note:** Run `list_models(verbose=True)` to see the current list of available models from your Copilot client.

### Choosing a Model

```python
from featcopilot.utils import get_default_model, get_model_names

# Use the default model (gpt-5.2)
llm_config = {'model': get_default_model()}

# Or choose from available models
available = get_model_names()
print(f"Available models: {available}")

# Pick based on your needs:
# - gpt-5.2: Best all-around choice (default)
# - gpt-5.1-codex: Best for code generation tasks
# - claude-sonnet-4: Alternative with strong reasoning
# - gpt-4.1 / claude-haiku-4.5: Faster, lower cost
# - claude-opus-4.5: Premium quality (slower)
```

## Providing Context

### Column Descriptions

Help the LLM understand your data:

```python
column_descriptions = {
    'age': 'Customer age in years (18-100)',
    'income': 'Annual household income in USD',
    'tenure_months': 'Number of months as customer',
    'monthly_charges': 'Monthly subscription fee',
    'total_charges': 'Cumulative charges to date',
    'contract_type': 'Month-to-month, One year, or Two year',
    'support_tickets': 'Number of support tickets filed'
}
```

### Task Description

Describe what you're trying to predict:

```python
task_description = """
Predict customer churn for a telecommunications company.
Churn is defined as canceling service within the next 30 days.
Key business goals: reduce churn rate, identify at-risk customers early.
"""
```

### Domain Context

Specify the domain for relevant features:

```python
# Supported domains
domains = [
    'healthcare',    # Medical features, risk scores
    'finance',       # Financial ratios, risk metrics
    'retail',        # RFM, customer lifetime value
    'telecom',       # Usage patterns, churn indicators
    'manufacturing', # Quality metrics, efficiency
]

engineer = AutoFeatureEngineer(
    engines=['llm'],
    llm_config={'domain': 'finance'}
)
```

## Feature Explanations

Get human-readable explanations:

```python
# After fit_transform
explanations = engineer.explain_features()

for feature, explanation in explanations.items():
    print(f"📊 {feature}")
    print(f"   {explanation}\n")
```

**Example Output:**
```
📊 age_bmi_ratio
   Ratio of age to BMI, may indicate metabolic age vs chronological age

📊 glucose_hba1c_interaction
   Interaction between fasting glucose and HbA1c captures glucose control

📊 cholesterol_risk_score
   Composite score from total, HDL, and LDL cholesterol levels
```

## Generated Code

Access the Python code for features:

```python
feature_code = engineer.get_feature_code()

for name, code in feature_code.items():
    print(f"# {name}")
    print(code)
    print()
```

**Example Output:**
```python
# age_bmi_ratio
result = df['age'] / (df['bmi'] + 1e-8)

# glucose_hba1c_interaction
result = df['glucose'] * df['hba1c']

# cholesterol_risk_score
result = (df['cholesterol_total'] - df['cholesterol_hdl']) / (df['cholesterol_ldl'] + 1)
```

## Custom Feature Generation

Request specific features:

```python
# Generate features for a specific focus area
custom_features = engineer.generate_custom_features(
    prompt="Create risk stratification features for cardiac patients",
    n_features=5
)

for feature in custom_features:
    print(f"Name: {feature['name']}")
    print(f"Code: {feature['code']}")
    print(f"Explanation: {feature['explanation']}\n")
```

## FeatureCodeGenerator

Generate features from natural language:

```python
from featcopilot.llm import FeatureCodeGenerator

generator = FeatureCodeGenerator(model='gpt-5.2')

# Generate single feature
feature = generator.generate(
    description="Calculate BMI from height in meters and weight in kg",
    columns={'height_m': 'float', 'weight_kg': 'float'}
)

print(feature.code)
# result = df['weight_kg'] / (df['height_m'] ** 2)

# Generate domain-specific features
healthcare_features = generator.generate_domain_features(
    domain='healthcare',
    columns={'age': 'int', 'bmi': 'float', 'glucose': 'float'},
    n_features=5
)
```

## FeatureExplainer

Generate reports:

```python
from featcopilot.llm import FeatureExplainer

explainer = FeatureExplainer(model='gpt-5.2')

# Generate comprehensive report
report = explainer.generate_feature_report(
    features=engineer._feature_set,
    X=X_transformed,
    column_descriptions=column_descriptions,
    task_description=task_description
)

# Save as markdown
with open('feature_report.md', 'w') as f:
    f.write(report)
```

## Best Practices

### 1. Provide Rich Context

```python
# ❌ Minimal context
engineer.fit_transform(X, y)

# ✅ Rich context
engineer.fit_transform(
    X, y,
    column_descriptions={...},
    task_description="...",
)
```

### 2. Validate Generated Features

```python
llm_config = {
    'validate_features': True  # Default
}
```

### 3. Review Generated Code

```python
# Always review LLM-generated code before production
for name, code in engineer.get_feature_code().items():
    print(f"Review: {name}")
    print(code)
```

### 4. Iterate and Refine

```python
# Generate more features in specific areas
more_features = engineer.generate_custom_features(
    prompt="Focus on interaction effects between age and metabolic markers"
)
```

## Fallback Behavior

When Copilot is unavailable:

```python
# Without authentication, mock responses are used
engineer = AutoFeatureEngineer(engines=['llm'])
# Warning: copilot-sdk not installed. Using mock LLM responses.

# Mock generates context-aware features based on column names
# but without true semantic understanding
```
