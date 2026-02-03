# FeatCopilot

<div align="center">
  <h2>🚀 Next-Generation LLM-Powered Auto Feature Engineering Framework</h2>
  <p><strong>Automatically generate, select, and explain predictive features using semantic understanding of your data</strong></p>
</div>

---

## Benchmark Highlights

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __+12.37% F1 on Spotify__

    ---

    Genre classification with LLM + Text + Tabular engines (flagship benchmark)

-   :material-brain:{ .lg .middle } __+11.03% with LLM Engine__

    ---

    Max improvement on INRIA benchmark (abalone dataset)

-   :material-scale-balance:{ .lg .middle } __Competitive & Fast__

    ---

    Matches other tools at 1000x speed (1s vs 1247s for AutoFeat)

-   :material-speedometer:{ .lg .middle } __Flexible Speed__

    ---

    <1s (tabular) or 30-60s (with LLM) based on your needs

</div>

[:octicons-arrow-right-24: View Full Benchmark Results](user-guide/benchmarks.md)

---

## Two Modes of Operation

=== "⚡ Fast Mode (Tabular Only)"

    Sub-second feature engineering using rule-based transformations:

    ```python
    from featcopilot import AutoFeatureEngineer

    # Fast, deterministic feature engineering
    engineer = AutoFeatureEngineer(
        engines=['tabular'],
        max_features=50
    )
    X_transformed = engineer.fit_transform(X, y)  # <1 second
    ```

    **Best for:** Production pipelines, real-time inference, reproducible results

=== "🧠 LLM Mode (With LiteLLM)"

    Domain-aware semantic feature generation with any LLM provider:

    ```python
    from featcopilot import AutoFeatureEngineer

    # LLM-powered semantic features
    engineer = AutoFeatureEngineer(
        engines=['tabular', 'llm'],
        max_features=50
    )
    X_transformed = engineer.fit_transform(
        X, y,
        column_descriptions={'age': 'Patient age in years'},
        task_description='Predict heart disease risk'
    )  # 30-60 seconds
    ```

    **Best for:** Exploratory analysis, domain-specific features, maximum accuracy

---

## What is FeatCopilot?

FeatCopilot is a Python library for automated feature engineering powered by large language models. It analyzes column meanings and descriptions to generate domain-aware features, applies intelligent selection to keep only the most predictive ones, and provides human-readable explanations for every feature it creates.

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

    Set up LLM providers for AI features

    [:octicons-arrow-right-24: Authentication](getting-started/authentication.md)

-   :material-chart-line:{ .lg .middle } __Benchmarks__

    ---

    See performance improvements across datasets

    [:octicons-arrow-right-24: Benchmark Results](user-guide/benchmarks.md)

</div>

## License

FeatCopilot is released under the MIT License.
