"""LLM-powered semantic feature engineering engine.

Uses contextual understanding of data to generate meaningful features.
"""

from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import Field

from featcopilot.core.base import BaseEngine, EngineConfig
from featcopilot.core.feature import Feature, FeatureOrigin, FeatureSet, FeatureType
from featcopilot.utils.logger import get_logger

logger = get_logger(__name__)


class SemanticEngineConfig(EngineConfig):
    """Configuration for semantic feature engine."""

    name: str = "SemanticEngine"
    model: str = Field(default="gpt-5.2", description="LLM model to use")
    max_suggestions: int = Field(default=20, description="Max features to suggest")
    validate_features: bool = Field(default=True, description="Validate generated code")
    domain: Optional[str] = Field(default=None, description="Domain context")
    temperature: float = Field(default=0.3, description="LLM temperature")
    backend: Literal["copilot", "litellm"] = Field(default="copilot", description="LLM backend to use")
    api_key: Optional[str] = Field(default=None, description="API key for litellm backend")
    api_base: Optional[str] = Field(default=None, description="Custom API base URL for litellm")
    enable_text_features: bool = Field(default=True, description="Generate ML features from text columns")
    keep_text_columns: bool = Field(
        default=True, description="Keep original text columns (for models that handle them natively)"
    )
    text_feature_types: list[str] = Field(
        default_factory=lambda: ["sentiment", "readability", "linguistic", "semantic"],
        description="Types of text features to generate",
    )


class SemanticEngine(BaseEngine):
    """
    LLM-powered semantic feature engineering engine.

    Uses GitHub Copilot SDK or LiteLLM to:
    - Understand column semantics from names and descriptions
    - Generate domain-aware features
    - Create interpretable features with explanations
    - Generate custom Python code for complex transformations

    This is the KEY DIFFERENTIATOR from existing libraries like CAAFE.

    Parameters
    ----------
    model : str, default='gpt-5.2'
        LLM model to use
    max_suggestions : int, default=20
        Maximum number of features to suggest
    validate_features : bool, default=True
        Whether to validate generated feature code
    domain : str, optional
        Domain context (e.g., 'healthcare', 'finance', 'retail')
    backend : str, default='copilot'
        LLM backend to use: 'copilot' or 'litellm'
    api_key : str, optional
        API key for litellm backend (uses environment variable if not provided)
    api_base : str, optional
        Custom API base URL for litellm backend (for self-hosted models)

    Examples
    --------
    Using GitHub Copilot SDK (default):
    >>> engine = SemanticEngine(model='gpt-5.2', domain='healthcare')
    >>> X_features = engine.fit_transform(
    ...     X, y,
    ...     column_descriptions={'age': 'Patient age', 'bmi': 'Body mass index'},
    ...     task_description='Predict diabetes risk'
    ... )

    Using LiteLLM with OpenAI:
    >>> engine = SemanticEngine(
    ...     model='gpt-4o',
    ...     backend='litellm',
    ...     api_key='your-api-key'  # or set OPENAI_API_KEY env var
    ... )

    Using LiteLLM with Anthropic:
    >>> engine = SemanticEngine(
    ...     model='claude-3-opus',
    ...     backend='litellm'
    ... )

    Using LiteLLM with local Ollama:
    >>> engine = SemanticEngine(
    ...     model='ollama/llama2',
    ...     backend='litellm',
    ...     api_base='http://localhost:11434'
    ... )
    """

    def __init__(
        self,
        model: str = "gpt-5.2",
        max_suggestions: int = 20,
        validate_features: bool = True,
        domain: Optional[str] = None,
        verbose: bool = False,
        backend: Literal["copilot", "litellm"] = "copilot",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        enable_text_features: bool = True,
        text_feature_types: Optional[list[str]] = None,
        **kwargs,
    ):
        config = SemanticEngineConfig(
            model=model,
            max_suggestions=max_suggestions,
            validate_features=validate_features,
            domain=domain,
            verbose=verbose,
            backend=backend,
            api_key=api_key,
            api_base=api_base,
            enable_text_features=enable_text_features,
            text_feature_types=text_feature_types or ["sentiment", "readability", "linguistic", "semantic"],
            **kwargs,
        )
        super().__init__(config=config)
        self.config: SemanticEngineConfig = config
        self._client: Optional[Any] = None
        self._suggested_features: list[dict[str, Any]] = []
        self._text_features: list[dict[str, Any]] = []
        self._feature_set = FeatureSet()
        self._column_info: dict[str, str] = {}
        self._column_descriptions: dict[str, str] = {}
        self._task_description: str = ""
        self._text_columns: list[str] = []

    def _ensure_client(self) -> None:
        """Ensure LLM client is initialized."""
        if self._client is None:
            if self.config.backend == "litellm":
                from featcopilot.llm.litellm_client import SyncLiteLLMFeatureClient

                self._client = SyncLiteLLMFeatureClient(
                    model=self.config.model,
                    api_key=self.config.api_key,
                    api_base=self.config.api_base,
                )
            else:
                from featcopilot.llm.copilot_client import SyncCopilotFeatureClient

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

        # Build column info and detect text columns
        self._column_info = {}
        self._text_columns = []
        for col in X.columns:
            dtype = str(X[col].dtype)
            if X[col].dtype == "object":
                dtype = "string"
                # Detect if it's a text column (long strings with high variance)
                if X[col].str.len().mean() > 20 and X[col].nunique() > 10:
                    self._text_columns.append(col)
            elif np.issubdtype(X[col].dtype, np.integer):
                dtype = "integer"
            elif np.issubdtype(X[col].dtype, np.floating):
                dtype = "float"
            self._column_info[col] = dtype

        if self.config.verbose:
            logger.info(f"SemanticEngine: Detected {len(self._text_columns)} text columns: {self._text_columns}")

        # Generate text-specific features if enabled
        if self.config.enable_text_features and self._text_columns:
            self._text_features = self._generate_text_features(X)
            if self.config.verbose:
                logger.info(f"SemanticEngine: Generated {len(self._text_features)} text features")

        # Get LLM suggestions for general features (excluding text columns)
        if self.config.verbose:
            logger.info("SemanticEngine: Requesting feature suggestions from LLM...")

        # Filter out text columns for general feature suggestions
        non_text_column_info = {k: v for k, v in self._column_info.items() if k not in self._text_columns}

        if non_text_column_info:
            try:
                self._suggested_features = self._client.suggest_features(
                    column_info=non_text_column_info,
                    task_description=task_description,
                    column_descriptions=column_descriptions,
                    domain=self.config.domain,
                    max_suggestions=self.config.max_suggestions,
                )
            except Exception as e:
                if self.config.verbose:
                    logger.warning(f"SemanticEngine: Could not get LLM suggestions: {e}")
                self._suggested_features = []
        else:
            self._suggested_features = []

        if self.config.verbose:
            logger.info(f"SemanticEngine: Received {len(self._suggested_features)} suggestions")

        # Validate features if enabled
        if self.config.validate_features:
            self._validate_suggestions(X)

        # Build feature set
        self._build_feature_set()

        self._is_fitted = True
        return self

    def _generate_text_features(self, X: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Generate ML-ready numerical features from text columns using LLM suggestions.

        This is the key differentiator - LLM suggests Python code to transform text
        into numerical features that can be used by ML models.
        """
        text_features = []

        for col in self._text_columns:
            # Always add fallback features first (don't require LLM)
            fallback_features = self._get_fallback_text_features(col)
            text_features.extend(fallback_features)

            # Try to get LLM-suggested features (optional)
            try:
                col_desc = self._column_descriptions.get(col, f"Text column: {col}")

                # Use suggest_features instead of send_prompt for better compatibility
                response = self._client.suggest_features(
                    column_info={col: "string"},
                    task_description=f"Extract numerical features from text column '{col}' for {self._task_description}",
                    column_descriptions={col: col_desc},
                    domain=self.config.domain,
                    max_suggestions=5,
                )

                # Response is already parsed as list of features
                for f in response:
                    f["source_columns"] = [col]
                    f["is_text_feature"] = True
                    text_features.append(f)

            except Exception as e:
                if self.config.verbose:
                    logger.warning(f"SemanticEngine: Could not get LLM suggestions for '{col}': {e}")

        return text_features

    def _build_text_feature_prompt(self, col: str, samples: list[str], description: str) -> str:
        """Build prompt for text feature generation."""
        return f"""You are an expert data scientist. Generate Python code to extract NUMERICAL features from text data.

## Text Column
Name: {col}
Description: {description}

## Sample Values
{chr(10).join([f'- "{s[:200]}..."' if len(str(s)) > 200 else f'- "{s}"' for s in samples[:5]])}

## Task
{self._task_description}

## Requirements
Generate features that transform text into NUMERICAL values suitable for ML models:
1. Sentiment scores (positive/negative/neutral)
2. Readability metrics (Flesch score, word complexity)
3. Linguistic features (noun ratio, verb ratio, sentence count)
4. Pattern detection (contains numbers, URLs, emails)
5. Domain-specific indicators

## Output Format
Return JSON with "features" array:
{{
  "features": [
    {{
      "name": "{col}_sentiment_score",
      "code": "result = df['{col}'].apply(lambda x: len([w for w in str(x).lower().split() if w in ['good','great','excellent','best']]) - len([w for w in str(x).lower().split() if w in ['bad','poor','worst','terrible']]))",
      "explanation": "Simple sentiment score based on positive/negative word counts"
    }}
  ]
}}

Return ONLY the JSON object, no other text. Generate 5-10 useful features."""

    def _parse_text_features(self, response: str, col: str) -> list[dict[str, Any]]:
        """Parse text features from LLM response."""
        import json
        import re

        try:
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])

            data = json.loads(response)
            features = data.get("features", [])

            # Add source column info
            for f in features:
                f["source_columns"] = [col]
                f["is_text_feature"] = True

            return features

        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    features = data.get("features", [])
                    for f in features:
                        f["source_columns"] = [col]
                        f["is_text_feature"] = True
                    return features
                except json.JSONDecodeError:
                    pass
            return []

    def _get_fallback_text_features(self, col: str) -> list[dict[str, Any]]:
        """Generate fallback text features that don't require LLM."""
        return [
            {
                "name": f"{col}_char_length",
                "code": f"result = df['{col}'].fillna('').astype(str).str.len()",
                "explanation": "Character length of text",
                "source_columns": [col],
                "is_text_feature": True,
            },
            {
                "name": f"{col}_word_count",
                "code": f"result = df['{col}'].fillna('').astype(str).str.split().str.len()",
                "explanation": "Word count in text",
                "source_columns": [col],
                "is_text_feature": True,
            },
            {
                "name": f"{col}_avg_word_length",
                "code": f"result = df['{col}'].fillna('').astype(str).apply(lambda x: np.mean([len(w) for w in x.split()] or [0]))",
                "explanation": "Average word length",
                "source_columns": [col],
                "is_text_feature": True,
            },
            {
                "name": f"{col}_sentence_count",
                "code": f"result = df['{col}'].fillna('').astype(str).str.count(r'[.!?]+')",
                "explanation": "Number of sentences (approximate)",
                "source_columns": [col],
                "is_text_feature": True,
            },
            {
                "name": f"{col}_uppercase_ratio",
                "code": f"result = df['{col}'].fillna('').astype(str).apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))",
                "explanation": "Ratio of uppercase characters",
                "source_columns": [col],
                "is_text_feature": True,
            },
            {
                "name": f"{col}_digit_count",
                "code": f"result = df['{col}'].fillna('').astype(str).str.count(r'\\d')",
                "explanation": "Count of digits in text",
                "source_columns": [col],
                "is_text_feature": True,
            },
            {
                "name": f"{col}_special_char_count",
                "code": f"result = df['{col}'].fillna('').astype(str).str.count(r'[^a-zA-Z0-9\\s]')",
                "explanation": "Count of special characters",
                "source_columns": [col],
                "is_text_feature": True,
            },
            {
                "name": f"{col}_unique_word_ratio",
                "code": f"result = df['{col}'].fillna('').astype(str).apply(lambda x: len(set(x.lower().split())) / max(len(x.split()), 1))",
                "explanation": "Ratio of unique words to total words",
                "source_columns": [col],
                "is_text_feature": True,
            },
            {
                "name": f"{col}_exclamation_count",
                "code": f"result = df['{col}'].fillna('').astype(str).str.count('!')",
                "explanation": "Count of exclamation marks (indicates emphasis/emotion)",
                "source_columns": [col],
                "is_text_feature": True,
            },
            {
                "name": f"{col}_question_count",
                "code": f"result = df['{col}'].fillna('').astype(str).str.count(r'\\?')",
                "explanation": "Count of question marks",
                "source_columns": [col],
                "is_text_feature": True,
            },
            {
                "name": f"{col}_caps_word_ratio",
                "code": f"result = df['{col}'].fillna('').astype(str).apply(lambda x: sum(1 for w in x.split() if w.isupper()) / max(len(x.split()), 1))",
                "explanation": "Ratio of all-caps words (indicates shouting/emphasis)",
                "source_columns": [col],
                "is_text_feature": True,
            },
        ]

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
                logger.warning(
                    f"SemanticEngine: Invalid feature '{feature.get('name', 'unknown')}': {result.get('error', 'unknown error')}"
                )

        self._suggested_features = valid_features

        if self.config.verbose:
            logger.info(f"SemanticEngine: {len(valid_features)} valid features after validation")

    def _build_feature_set(self) -> None:
        """Build FeatureSet from suggestions."""
        self._feature_set = FeatureSet()

        # Add text features
        for suggestion in self._text_features:
            feature = Feature(
                name=suggestion.get("name", f"text_feature_{len(self._feature_set)}"),
                dtype=FeatureType.NUMERIC,
                origin=FeatureOrigin.LLM_GENERATED,
                source_columns=suggestion.get("source_columns", []),
                transformation="text_to_numeric",
                explanation=suggestion.get("explanation", ""),
                code=suggestion.get("code", ""),
            )
            self._feature_set.add(feature)

        # Add general features
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
            Data with generated features (numerical only, text columns dropped)
        """
        if not self._is_fitted:
            raise RuntimeError("Engine must be fitted before transform")

        X = self._validate_input(X)
        result = X.copy()

        successful_features = []

        # Apply text features first
        for suggestion in self._text_features:
            name = suggestion.get("name", "")
            code = suggestion.get("code", "")

            if not code:
                continue

            try:
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
                        },
                        "np": np,
                        "pd": pd,
                    },
                    local_vars,
                )

                if "result" in local_vars:
                    feature_values = local_vars["result"]
                    if isinstance(feature_values, pd.Series):
                        result[name] = feature_values.values
                    else:
                        result[name] = feature_values
                    successful_features.append(name)

            except Exception as e:
                if self.config.verbose:
                    logger.error(f"SemanticEngine: Error computing text feature '{name}': {e}")

        # Apply general features
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
                    logger.error(f"SemanticEngine: Error computing '{name}': {e}")

        # Handle infinities and NaNs
        result = result.replace([np.inf, -np.inf], np.nan)

        # Optionally drop original text columns (only if not keeping them for downstream models)
        if not self.config.keep_text_columns:
            cols_to_drop = [col for col in self._text_columns if col in result.columns]
            if cols_to_drop:
                result = result.drop(columns=cols_to_drop)
                if self.config.verbose:
                    logger.info(f"SemanticEngine: Dropped {len(cols_to_drop)} text columns, keeping numerical features")

        self._feature_names = successful_features

        if self.config.verbose:
            logger.info(f"SemanticEngine: Successfully generated {len(successful_features)} features")

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

    def standardize_categories(
        self,
        df: pd.DataFrame,
        column: str,
        target_categories: Optional[list[str]] = None,
        similarity_threshold: float = 0.8,
        max_categories: int = 50,
        context: Optional[str] = None,
    ) -> dict[str, str]:
        """
        Use LLM to standardize similar category values in a column.

        Identifies semantically similar values (e.g., "software engineer", "Software Engineer",
        "SDE") and maps them to a canonical form.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame containing the column to standardize
        column : str
            Name of the categorical column to standardize
        target_categories : list[str], optional
            If provided, map values to these specific categories.
            If None, LLM will infer appropriate canonical forms.
        similarity_threshold : float, default=0.8
            Minimum similarity for grouping (hint for LLM, not strictly enforced)
        max_categories : int, default=50
            Maximum number of unique values to process (for efficiency)
        context : str, optional
            Additional context about the data domain (e.g., "job titles in tech industry")

        Returns
        -------
        mapping : dict[str, str]
            Dictionary mapping original values to standardized values.
            Only includes values that need transformation.

        Examples
        --------
        >>> engine = SemanticEngine()
        >>> mapping = engine.standardize_categories(
        ...     df,
        ...     column="job_title",
        ...     context="job titles in software industry"
        ... )
        >>> print(mapping)
        {'software engineer': 'Software Engineer', 'SDE': 'Software Engineer',
         'Sr. SWE': 'Senior Software Engineer', 'data scientist': 'Data Scientist'}

        >>> # Apply the mapping
        >>> df_clean = engine.apply_category_mapping(df, "job_title", mapping)
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        self._ensure_client()

        # Get unique values (excluding NaN)
        unique_values = df[column].dropna().unique().tolist()

        # Convert to strings and filter
        unique_values = [str(v) for v in unique_values if v is not None and str(v).strip()]
        unique_values = list(set(unique_values))  # Remove duplicates after string conversion

        if len(unique_values) == 0:
            if self.config.verbose:
                logger.info(f"SemanticEngine: No valid values found in column '{column}'")
            return {}

        if len(unique_values) > max_categories:
            if self.config.verbose:
                logger.warning(
                    f"SemanticEngine: Column '{column}' has {len(unique_values)} unique values, "
                    f"truncating to {max_categories} most frequent"
                )
            # Get most frequent values
            value_counts = df[column].value_counts().head(max_categories)
            unique_values = [str(v) for v in value_counts.index.tolist()]

        # Build and send prompt
        prompt = self._build_category_standardization_prompt(
            column=column,
            unique_values=unique_values,
            target_categories=target_categories,
            context=context,
            similarity_threshold=similarity_threshold,
        )

        try:
            # Use the client's send_prompt method if available, otherwise use suggest_features
            if hasattr(self._client, "send_prompt"):
                response = self._client.send_prompt(prompt)
            else:
                # Fallback: use suggest_features with a specialized task
                response_list = self._client.suggest_features(
                    column_info={column: "categorical"},
                    task_description=prompt,
                    column_descriptions={column: context or "Categorical column to standardize"},
                    domain=self.config.domain,
                    max_suggestions=1,
                )
                # Extract mapping from response if possible
                if response_list and isinstance(response_list, list) and len(response_list) > 0:
                    first = response_list[0]
                    if isinstance(first, dict) and "mapping" in first:
                        return first["mapping"]
                    response = str(first)
                else:
                    response = str(response_list)

            mapping = self._parse_category_mapping(response, unique_values)

            if self.config.verbose:
                logger.info(f"SemanticEngine: Created mapping for {len(mapping)} values in column '{column}'")

            return mapping

        except Exception as e:
            if self.config.verbose:
                logger.error(f"SemanticEngine: Error standardizing categories: {e}")
            return {}

    def _build_category_standardization_prompt(
        self,
        column: str,
        unique_values: list[str],
        target_categories: Optional[list[str]] = None,
        context: Optional[str] = None,
        similarity_threshold: float = 0.8,
    ) -> str:
        """Build prompt for category standardization."""
        values_str = "\n".join([f'- "{v}"' for v in unique_values[:100]])

        target_str = ""
        if target_categories:
            target_str = f"""
## Target Categories (map values to these)
{chr(10).join([f'- "{c}"' for c in target_categories])}
"""

        context_str = f"\n## Context\n{context}" if context else ""

        return f"""You are an expert data scientist specializing in data cleaning and standardization.

## Task
Analyze the following categorical values from column "{column}" and identify semantically similar values that should be standardized to a common form.

## Unique Values in Column
{values_str}
{target_str}{context_str}

## Requirements
1. Identify values that represent the same concept (case variations, abbreviations, typos, synonyms)
2. Map similar values to a single canonical/standardized form
3. Use proper capitalization for the standardized form (e.g., "Software Engineer" not "software engineer")
4. Common patterns to look for:
   - Case variations: "Software Engineer" vs "software engineer" vs "SOFTWARE ENGINEER"
   - Abbreviations: "SDE" vs "Software Development Engineer", "Sr." vs "Senior"
   - Typos: "Enginer" vs "Engineer"
   - Synonyms: "Developer" vs "Programmer" vs "Software Engineer"
   - Formatting: "Data-Scientist" vs "Data Scientist" vs "DataScientist"
5. Only include values that need mapping (exclude already-standardized values)
6. Preserve values that are already properly formatted or don't have similar alternatives

## Output Format
Return ONLY a valid JSON object with this structure:
{{
  "mapping": {{
    "original_value_1": "Standardized Value",
    "original_value_2": "Standardized Value",
    "typo_value": "Corrected Value"
  }},
  "groups": [
    {{
      "canonical": "Software Engineer",
      "members": ["software engineer", "SDE", "Software Dev", "SW Engineer"]
    }}
  ]
}}

Return ONLY the JSON object, no markdown formatting, no explanation text."""

    def _parse_category_mapping(
        self,
        response: str,
        original_values: list[str],
    ) -> dict[str, str]:
        """Parse category mapping from LLM response."""
        import json
        import re

        try:
            # Clean response
            response = response.strip()

            # Remove markdown code blocks if present
            if response.startswith("```"):
                lines = response.split("\n")
                # Find the JSON content between ``` markers
                start_idx = 1 if lines[0].startswith("```") else 0
                end_idx = len(lines)
                for i, line in enumerate(lines[1:], 1):
                    if line.strip() == "```":
                        end_idx = i
                        break
                response = "\n".join(lines[start_idx:end_idx])

            # Try to parse as JSON
            data = json.loads(response)

            # Extract mapping from response
            if isinstance(data, dict):
                if "mapping" in data:
                    mapping = data["mapping"]
                elif "groups" in data:
                    # Build mapping from groups
                    mapping = {}
                    for group in data["groups"]:
                        canonical = group.get("canonical", "")
                        members = group.get("members", [])
                        for member in members:
                            if member != canonical:
                                mapping[member] = canonical
                else:
                    # Assume the entire dict is the mapping
                    mapping = data
            else:
                mapping = {}

            # Validate mapping - only keep mappings for values that exist
            original_set = set(original_values)
            original_lower = {v.lower(): v for v in original_values}

            validated_mapping = {}
            for orig, standardized in mapping.items():
                # Check exact match or case-insensitive match
                if orig in original_set:
                    validated_mapping[orig] = standardized
                elif orig.lower() in original_lower:
                    actual_orig = original_lower[orig.lower()]
                    validated_mapping[actual_orig] = standardized

            return validated_mapping

        except json.JSONDecodeError:
            # Try to extract JSON from response
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                try:
                    return self._parse_category_mapping(json_match.group(), original_values)
                except Exception:
                    pass

            if self.config.verbose:
                logger.warning("SemanticEngine: Could not parse category mapping response")
            return {}

    def apply_category_mapping(
        self,
        df: pd.DataFrame,
        column: str,
        mapping: dict[str, str],
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Apply a category mapping to standardize values in a DataFrame column.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame
        column : str
            Column to transform
        mapping : dict[str, str]
            Mapping from original values to standardized values
        inplace : bool, default=False
            If True, modify DataFrame in place

        Returns
        -------
        DataFrame
            DataFrame with standardized column values
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        if not inplace:
            df = df.copy()

        # Apply mapping, keeping original values for unmapped entries
        df[column] = df[column].apply(lambda x: mapping.get(str(x), x) if pd.notna(x) else x)

        if self.config.verbose:
            logger.info(f"SemanticEngine: Applied mapping to column '{column}'")

        return df

    def standardize_multiple_columns(
        self,
        df: pd.DataFrame,
        columns: list[str],
        contexts: Optional[dict[str, str]] = None,
        **kwargs,
    ) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
        """
        Standardize multiple categorical columns at once.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame
        columns : list[str]
            List of column names to standardize
        contexts : dict[str, str], optional
            Context descriptions for each column
        **kwargs
            Additional arguments passed to standardize_categories

        Returns
        -------
        df_clean : DataFrame
            DataFrame with standardized columns
        all_mappings : dict[str, dict[str, str]]
            Dictionary of mappings for each column
        """
        contexts = contexts or {}
        all_mappings = {}
        result_df = df.copy()

        for col in columns:
            if col not in df.columns:
                if self.config.verbose:
                    logger.warning(f"SemanticEngine: Column '{col}' not found, skipping")
                continue

            context = contexts.get(col)
            mapping = self.standardize_categories(result_df, col, context=context, **kwargs)
            all_mappings[col] = mapping

            if mapping:
                result_df = self.apply_category_mapping(result_df, col, mapping)

        return result_df, all_mappings

    def __del__(self):
        """Clean up client on deletion."""
        if self._client:
            try:
                self._client.stop()
            except Exception:
                pass
