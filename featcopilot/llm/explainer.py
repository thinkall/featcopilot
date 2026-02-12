"""Feature explanation generator using LLM.

Generates human-readable explanations for features.
"""

from typing import Any, Literal, Optional

import pandas as pd

from featcopilot.core.feature import Feature, FeatureSet
from featcopilot.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureExplainer:
    """
    Generate human-readable explanations for features.

    Uses LLM to create interpretable explanations that can be
    understood by non-technical stakeholders.

    Parameters
    ----------
    model : str, default='gpt-5.2'
        LLM model to use
    backend : str, default='copilot'
        LLM backend to use: 'copilot', 'openai', or 'litellm'
    api_key : str, optional
        API key for litellm backend
    api_base : str, optional
        Custom API base URL for litellm backend

    Examples
    --------
    >>> explainer = FeatureExplainer()
    >>> explanations = explainer.explain_features(feature_set, task='predict churn')
    """

    def __init__(
        self,
        model: str = "gpt-5.2",
        verbose: bool = False,
        backend: Literal["copilot", "litellm", "openai"] = "copilot",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        self.model = model
        self.verbose = verbose
        self.backend = backend
        self.api_key = api_key
        self.api_base = api_base
        self._client: Optional[Any] = None

    def _ensure_client(self) -> None:
        """Ensure client is initialized."""
        if self._client is None:
            if self.backend == "openai":
                from featcopilot.llm.openai_client import SyncOpenAIFeatureClient

                self._client = SyncOpenAIFeatureClient(model=self.model, api_key=self.api_key, api_base=self.api_base)
            elif self.backend == "litellm":
                from featcopilot.llm.litellm_client import SyncLiteLLMFeatureClient

                self._client = SyncLiteLLMFeatureClient(model=self.model, api_key=self.api_key, api_base=self.api_base)
            else:
                from featcopilot.llm.copilot_client import SyncCopilotFeatureClient

                self._client = SyncCopilotFeatureClient(model=self.model)
            self._client.start()

    def explain_feature(
        self,
        feature: Feature,
        column_descriptions: Optional[dict[str, str]] = None,
        task_description: Optional[str] = None,
    ) -> str:
        """
        Generate explanation for a single feature.

        Parameters
        ----------
        feature : Feature
            Feature to explain
        column_descriptions : dict, optional
            Descriptions of source columns
        task_description : str, optional
            ML task description

        Returns
        -------
        explanation : str
            Human-readable explanation
        """
        self._ensure_client()

        explanation = self._client.explain_feature(
            feature_name=feature.name,
            feature_code=feature.code or feature.transformation,
            column_descriptions=column_descriptions,
            task_description=task_description,
        )

        return explanation

    def explain_features(
        self,
        features: FeatureSet,
        column_descriptions: Optional[dict[str, str]] = None,
        task_description: Optional[str] = None,
        batch_size: int = 5,
    ) -> dict[str, str]:
        """
        Generate explanations for multiple features.

        Parameters
        ----------
        features : FeatureSet
            Features to explain
        column_descriptions : dict, optional
            Descriptions of source columns
        task_description : str, optional
            ML task description
        batch_size : int, default=5
            Number of features to explain in each LLM call

        Returns
        -------
        explanations : dict
            Mapping of feature names to explanations
        """
        explanations = {}

        for feature in features:
            # Skip if already has explanation
            if feature.explanation:
                explanations[feature.name] = feature.explanation
                continue

            try:
                explanation = self.explain_feature(feature, column_descriptions, task_description)
                explanations[feature.name] = explanation
                feature.explanation = explanation

            except Exception as e:
                if self.verbose:
                    logger.error(f"Could not explain {feature.name}: {e}")
                explanations[feature.name] = f"Feature based on: {', '.join(feature.source_columns)}"

        return explanations

    def generate_feature_report(
        self,
        features: FeatureSet,
        X: pd.DataFrame,
        column_descriptions: Optional[dict[str, str]] = None,
        task_description: Optional[str] = None,
    ) -> str:
        """
        Generate a comprehensive report about features.

        Parameters
        ----------
        features : FeatureSet
            Features to report on
        X : DataFrame
            Data with features
        column_descriptions : dict, optional
            Descriptions of source columns
        task_description : str, optional
            ML task description

        Returns
        -------
        report : str
            Markdown-formatted report
        """
        explanations = self.explain_features(features, column_descriptions, task_description)

        report = "# Feature Engineering Report\n\n"

        if task_description:
            report += f"**Task:** {task_description}\n\n"

        report += f"**Total Features Generated:** {len(features)}\n\n"

        # Summary by origin
        report += "## Features by Origin\n\n"
        origins = {}
        for feature in features:
            origin = feature.origin.value
            origins[origin] = origins.get(origin, 0) + 1

        for origin, count in sorted(origins.items()):
            report += f"- {origin}: {count}\n"

        # Feature details
        report += "\n## Feature Details\n\n"

        for feature in features:
            report += f"### {feature.name}\n\n"
            report += f"- **Type:** {feature.dtype.value}\n"
            report += f"- **Origin:** {feature.origin.value}\n"
            report += f"- **Source Columns:** {', '.join(feature.source_columns)}\n"

            if feature.name in X.columns:
                report += f"- **Non-null Values:** {X[feature.name].notna().sum()}\n"
                if X[feature.name].dtype in ["float64", "int64"]:
                    report += f"- **Mean:** {X[feature.name].mean():.4f}\n"
                    report += f"- **Std:** {X[feature.name].std():.4f}\n"

            explanation = explanations.get(feature.name, "")
            if explanation:
                report += f"\n**Explanation:** {explanation}\n"

            if feature.code:
                report += f"\n**Code:**\n```python\n{feature.code}\n```\n"

            report += "\n"

        return report

    def __del__(self):
        """Clean up client."""
        if self._client:
            try:
                self._client.stop()
            except Exception:
                pass
