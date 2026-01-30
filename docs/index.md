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

-   :material-robot:{ .lg .middle } __+19.7% with LLM Engine__

    ---

    Retail demand forecasting with semantic feature generation

-   :material-check-all:{ .lg .middle } __12/12 Text Wins__

    ---

    100% improvement rate on text/semantic classification tasks

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

=== "🧠 LLM Mode (With Copilot)"

    Domain-aware semantic feature generation with GitHub Copilot:

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

-   :material-chart-line:{ .lg .middle } __Benchmarks__

    ---

    See performance improvements across datasets

    [:octicons-arrow-right-24: Benchmark Results](user-guide/benchmarks.md)

</div>

## License

FeatCopilot is released under the MIT License.
