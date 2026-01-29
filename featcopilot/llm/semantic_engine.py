"""LLM-powered semantic feature engineering engine.

Uses contextual understanding of data to generate meaningful features.
"""

from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from pydantic import Field

from featcopilot.core.base import BaseEngine, EngineConfig
from featcopilot.core.feature import Feature, FeatureOrigin, FeatureSet, FeatureType
from featcopilot.llm.copilot_client import SyncCopilotFeatureClient


class SemanticEngineConfig(EngineConfig):
    """Configuration for semantic feature engine."""

    name: str = "SemanticEngine"
    model: str = Field(default="gpt-5", description="LLM model to use")
    max_suggestions: int = Field(default=20, description="Max features to suggest")
    validate_features: bool = Field(default=True, description="Validate generated code")
    domain: Optional[str] = Field(default=None, description="Domain context")
    temperature: float = Field(default=0.3, description="LLM temperature")


class SemanticEngine(BaseEngine):
    """
    LLM-powered semantic feature engineering engine.

    Uses GitHub Copilot SDK to:
    - Understand column semantics from names and descriptions
    - Generate domain-aware features
    - Create interpretable features with explanations
    - Generate custom Python code for complex transformations

    This is the KEY DIFFERENTIATOR from existing libraries like CAAFE.

    Parameters
    ----------
    model : str, default='gpt-5'
        LLM model to use
    max_suggestions : int, default=20
        Maximum number of features to suggest
    validate_features : bool, default=True
        Whether to validate generated feature code
    domain : str, optional
        Domain context (e.g., 'healthcare', 'finance', 'retail')

    Examples
    --------
    >>> engine = SemanticEngine(model='gpt-5', domain='healthcare')
    >>> X_features = engine.fit_transform(
    ...     X, y,
    ...     column_descriptions={'age': 'Patient age', 'bmi': 'Body mass index'},
    ...     task_description='Predict diabetes risk'
    ... )
    """

    def __init__(
        self,
        model: str = "gpt-5",
        max_suggestions: int = 20,
        validate_features: bool = True,
        domain: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
    ):
        config = SemanticEngineConfig(
            model=model,
            max_suggestions=max_suggestions,
            validate_features=validate_features,
            domain=domain,
            verbose=verbose,
            **kwargs,
        )
        super().__init__(config=config)
        self.config: SemanticEngineConfig = config
        self._client: Optional[SyncCopilotFeatureClient] = None
        self._suggested_features: list[dict[str, Any]] = []
        self._feature_set = FeatureSet()
        self._column_info: dict[str, str] = {}
        self._column_descriptions: dict[str, str] = {}
        self._task_description: str = ""

    def _ensure_client(self) -> None:
        """Ensure Copilot client is initialized."""
        if self._client is None:
            self._client = SyncCopilotFeatureClient(model=self.config.model)
            self._client.start()

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        column_descriptions: Optional[dict[str, str]] = None,
        task_description: str = "classification/regression task",
        **kwargs,
    ) -> "SemanticEngine":
        """
        Fit the engine by analyzing data and generating feature suggestions.

        Parameters
        ----------
        X : DataFrame
            Input data
        y : Series, optional
            Target variable
        column_descriptions : dict, optional
            Human-readable descriptions of columns
        task_description : str
            Description of the ML task

        Returns
        -------
        self : SemanticEngine
        """
        X = self._validate_input(X)
        self._ensure_client()

        # Store metadata
        self._column_descriptions = column_descriptions or {}
        self._task_description = task_description

        # Build column info
        self._column_info = {}
        for col in X.columns:
            dtype = str(X[col].dtype)
            if X[col].dtype == "object":
                dtype = "string"
            elif np.issubdtype(X[col].dtype, np.integer):
                dtype = "integer"
            elif np.issubdtype(X[col].dtype, np.floating):
                dtype = "float"
            self._column_info[col] = dtype

        # Get LLM suggestions
        if self.config.verbose:
            print("SemanticEngine: Requesting feature suggestions from LLM...")

        self._suggested_features = self._client.suggest_features(
            column_info=self._column_info,
            task_description=task_description,
            column_descriptions=column_descriptions,
            domain=self.config.domain,
            max_suggestions=self.config.max_suggestions,
        )

        if self.config.verbose:
            print(f"SemanticEngine: Received {len(self._suggested_features)} suggestions")

        # Validate features if enabled
        if self.config.validate_features:
            self._validate_suggestions(X)

        # Build feature set
        self._build_feature_set()

        self._is_fitted = True
        return self

    def _validate_suggestions(self, X: pd.DataFrame) -> None:
        """Validate suggested feature code."""
        valid_features = []
        sample_data = {col: X[col].head(100).tolist() for col in X.columns}

        for feature in self._suggested_features:
            code = feature.get("code", "")
            if not code:
                continue

            result = self._client.validate_feature_code(code, sample_data)

            if result["valid"]:
                valid_features.append(feature)
            elif self.config.verbose:
                print(
                    f"SemanticEngine: Invalid feature '{feature.get('name', 'unknown')}': {result.get('error', 'unknown error')}"
                )

        self._suggested_features = valid_features

        if self.config.verbose:
            print(f"SemanticEngine: {len(valid_features)} valid features after validation")

    def _build_feature_set(self) -> None:
        """Build FeatureSet from suggestions."""
        self._feature_set = FeatureSet()

        for suggestion in self._suggested_features:
            feature = Feature(
                name=suggestion.get("name", f"llm_feature_{len(self._feature_set)}"),
                dtype=FeatureType.NUMERIC,
                origin=FeatureOrigin.LLM_GENERATED,
                source_columns=suggestion.get("source_columns", []),
                transformation="llm_generated",
                explanation=suggestion.get("explanation", ""),
                code=suggestion.get("code", ""),
            )
            self._feature_set.add(feature)

    def transform(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> pd.DataFrame:
        """
        Generate LLM-suggested features.

        Parameters
        ----------
        X : DataFrame
            Input data

        Returns
        -------
        X_features : DataFrame
            Data with generated features
        """
        if not self._is_fitted:
            raise RuntimeError("Engine must be fitted before transform")

        X = self._validate_input(X)
        result = X.copy()

        successful_features = []

        for suggestion in self._suggested_features:
            name = suggestion.get("name", "")
            code = suggestion.get("code", "")

            if not code:
                continue

            try:
                # Execute feature code
                local_vars = {"df": result, "np": np, "pd": pd}
                exec(
                    code,
                    {
                        "__builtins__": {
                            "len": len,
                            "sum": sum,
                            "max": max,
                            "min": min,
                            "abs": abs,
                            "round": round,
                            "int": int,
                            "float": float,
                            "str": str,
                            "list": list,
                            "dict": dict,
                            "set": set,
                        }
                    },
                    local_vars,
                )

                if "result" in local_vars:
                    feature_values = local_vars["result"]

                    # Ensure it's a Series with correct index
                    if isinstance(feature_values, pd.Series):
                        result[name] = feature_values.values
                    else:
                        result[name] = feature_values

                    successful_features.append(name)

            except Exception as e:
                if self.config.verbose:
                    print(f"SemanticEngine: Error computing '{name}': {e}")

        # Handle infinities and NaNs
        result = result.replace([np.inf, -np.inf], np.nan)

        self._feature_names = successful_features

        if self.config.verbose:
            print(f"SemanticEngine: Successfully generated {len(successful_features)} features")

        return result

    def get_feature_explanations(self) -> dict[str, str]:
        """
        Get explanations for all generated features.

        Returns
        -------
        explanations : dict
            Mapping of feature names to explanations
        """
        return {s.get("name", ""): s.get("explanation", "") for s in self._suggested_features if s.get("name")}

    def get_feature_code(self) -> dict[str, str]:
        """
        Get code for all generated features.

        Returns
        -------
        code : dict
            Mapping of feature names to Python code
        """
        return {s.get("name", ""): s.get("code", "") for s in self._suggested_features if s.get("name")}

    def suggest_more_features(self, focus_area: str, n_features: int = 5) -> list[dict[str, Any]]:
        """
        Request additional feature suggestions in a specific area.

        Parameters
        ----------
        focus_area : str
            Area to focus on (e.g., 'interactions', 'ratios', 'time-based')
        n_features : int, default=5
            Number of additional features to suggest

        Returns
        -------
        suggestions : list
            New feature suggestions
        """
        self._ensure_client()

        # Build focused prompt
        enhanced_task = f"{self._task_description}\n\nFocus specifically on: {focus_area}"

        new_suggestions = self._client.suggest_features(
            column_info=self._column_info,
            task_description=enhanced_task,
            column_descriptions=self._column_descriptions,
            domain=self.config.domain,
            max_suggestions=n_features,
        )

        return new_suggestions

    def generate_custom_feature(self, description: str, constraints: Optional[list[str]] = None) -> dict[str, Any]:
        """
        Generate a specific feature from natural language description.

        Parameters
        ----------
        description : str
            Natural language description of desired feature
        constraints : list, optional
            Constraints on the generated code

        Returns
        -------
        feature : dict
            Generated feature with name, code, and explanation
        """
        self._ensure_client()

        code = self._client.generate_feature_code(
            description=description,
            column_info=self._column_info,
            constraints=constraints,
        )

        # Generate name from description
        name = "_".join(description.lower().split()[:4]).replace("-", "_")
        name = "".join(c if c.isalnum() or c == "_" else "" for c in name)

        return {
            "name": name,
            "code": code,
            "description": description,
            "explanation": f"Custom feature: {description}",
        }

    def get_feature_set(self) -> FeatureSet:
        """Get the feature set with metadata."""
        return self._feature_set

    def __del__(self):
        """Clean up client on deletion."""
        if self._client:
            try:
                self._client.stop()
            except Exception:
                pass
