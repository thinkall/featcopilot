# FeatCopilot

<div align="center">
  <h2>🚀 Next-Generation LLM-Powered Auto Feature Engineering</h2>
  <p><strong>The first unified feature engineering framework with native GitHub Copilot SDK integration</strong></p>
</div>

---

## Benchmark Highlights

<div class="grid cards" markdown>

-   :material-chart-areaspline:{ .lg .middle } __+49% on Text Data__

    ---

    News headline classification improved from 40.8% to 60.8% accuracy

-   :material-check-all:{ .lg .middle } __12/12 Text Wins__

    ---

    100% improvement rate on text/semantic classification tasks

-   :material-speedometer:{ .lg .middle } __<1s Processing__

    ---

    Feature engineering completes in under 1 second for most datasets

</div>

[:octicons-arrow-right-24: View Full Benchmark Results](user-guide/benchmarks.md)

---

## What is FeatCopilot?

FeatCopilot is a comprehensive Python library for automated feature engineering that combines traditional approaches with cutting-edge LLM-powered capabilities via GitHub Copilot SDK.

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } __Multi-Engine Architecture__

    ---

    Tabular, time series, relational, and text feature engines in one unified API

-   :material-robot:{ .lg .middle } __LLM-Powered Intelligence__

    ---

    Semantic feature discovery, domain-aware generation, and automatic code synthesis

-   :material-chart-bar:{ .lg .middle } __Intelligent Selection__

    ---

    Statistical testing, importance ranking, and redundancy elimination

-   :material-puzzle:{ .lg .middle } __Sklearn Compatible__

    ---

    Drop-in replacement for scikit-learn transformers in your ML pipelines

</div>

## Why FeatCopilot?

| Feature | FeatCopilot | Featuretools | TSFresh | AutoFeat | OpenFE | CAAFE |
|---------|-------------|--------------|---------|----------|--------|-------|
| Tabular Features | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| Time Series | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Relational | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **LLM-Powered** | ✅ | ❌ | ❌ | ❌ | ❌ | ⚠️ |
| **Semantic Understanding** | ✅ | ❌ | ❌ | ❌ | ❌ | ⚠️ |
| **Code Generation** | ✅ | ❌ | ❌ | ❌ | ❌ | ⚠️ |
| Sklearn Compatible | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Interpretable | ✅ | ⚠️ | ⚠️ | ⚠️ | ❌ | ✅ |

## Quick Example

```python
from featcopilot import AutoFeatureEngineer

# Initialize with LLM capabilities
engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    max_features=50,
    llm_config={'model': 'gpt-5', 'domain': 'healthcare'}
)

# Fit and transform with semantic understanding
X_transformed = engineer.fit_transform(
    X, y,
    column_descriptions={
        'age': 'Patient age in years',
        'bmi': 'Body Mass Index',
        'glucose': 'Fasting blood glucose level'
    },
    task_description='Predict diabetes risk'
)

# Get human-readable explanations
for feature, explanation in engineer.explain_features().items():
    print(f"{feature}: {explanation}")
```

## Installation

```bash
# Basic installation
pip install featcopilot

# With LLM capabilities
pip install featcopilot[llm]

# Full installation with all extras
pip install featcopilot[full]
```

## Getting Started

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } __Installation__

    ---

    Install FeatCopilot and set up your environment

    [:octicons-arrow-right-24: Installation Guide](getting-started/installation.md)

-   :material-rocket-launch:{ .lg .middle } __Quick Start__

    ---

    Get up and running in 5 minutes

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

-   :material-key:{ .lg .middle } __Authentication__

    ---

    Set up GitHub Copilot for LLM features

    [:octicons-arrow-right-24: Authentication](getting-started/authentication.md)

</div>

## License

FeatCopilot is released under the MIT License.
