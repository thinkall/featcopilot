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
    model='gpt-5',
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
    'model': 'gpt-5',           # Model: gpt-5, gpt-4.1, etc.
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

### Supported Models

| Model | Description | Speed | Quality |
|-------|-------------|-------|---------|
| `gpt-5` | Latest GPT model | Fast | Excellent |
| `gpt-4.1` | GPT-4.1 | Fast | Very Good |
| `claude-sonnet-4` | Claude Sonnet | Medium | Excellent |

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

generator = FeatureCodeGenerator(model='gpt-5')

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

explainer = FeatureExplainer(model='gpt-5')

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
