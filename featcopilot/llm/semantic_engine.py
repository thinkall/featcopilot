"""LLM-powered semantic feature engineering engine.

Uses contextual understanding of data to generate meaningful features.
"""

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field, JsonValue

from featcopilot.core.base import BaseEngine, EngineConfig
from featcopilot.core.feature import Feature, FeatureOrigin, FeatureSet, FeatureType
from featcopilot.core.fitted_rule import FittedRulePlan, FittedRuleSpec
from featcopilot.engines.fitted_rules import FittedRuleEngine
from featcopilot.llm._execution import execute_feature_code
from featcopilot.llm._types import FeatureClient, feature_suggestions
from featcopilot.llm.plan_generator import request_fitted_plan
from featcopilot.utils.logger import get_logger
from featcopilot.utils.models import DEFAULT_MODEL

logger = get_logger(__name__)


class SemanticEngineConfig(EngineConfig):
    """Configuration for semantic feature engine."""

    name: str = "SemanticEngine"
    model: str = Field(default=DEFAULT_MODEL, description="LLM model to use")
    max_suggestions: int = Field(default=20, ge=1, description="Max features to suggest")
    validate_features: bool = Field(default=True, description="Validate generated code")
    domain: str | None = Field(default=None, description="Domain context")
    temperature: float = Field(default=0.3, ge=0, le=2, description="LLM temperature")
    timeout: float = Field(
        default=60.0, gt=0, strict=True, allow_inf_nan=False, description="Provider request timeout in seconds"
    )
    max_tokens: int = Field(default=4096, gt=0, description="Maximum response tokens for chat backends")
    backend: Literal["copilot", "litellm", "openai"] = Field(default="copilot", description="LLM backend to use")
    api_key: str | None = Field(default=None, description="API key for litellm/openai backend")
    api_base: str | None = Field(default=None, description="Custom API base URL for litellm/openai")
    api_version: str | None = Field(default=None, description="API version for Azure OpenAI")
    feature_format: Literal["code", "plan"] = Field(default="code", description="Trusted Python or fitted typed rules")
    task: Literal["auto", "classification", "regression"] = "auto"
    random_state: int = 42
    target_cv: int = Field(default=5, ge=2)
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
    model : str, optional
        LLM model to use. Defaults to
        :data:`featcopilot.utils.models.DEFAULT_MODEL`.
    max_suggestions : int, default=20
        Maximum number of features to suggest
    validate_features : bool, default=True
        Whether to validate generated feature code
    domain : str, optional
        Domain context (e.g., 'healthcare', 'finance', 'retail')
    backend : str, default='copilot'
        LLM backend to use: 'copilot', 'openai', or 'litellm'
    api_key : str, optional
        API key for openai/litellm backend (uses environment variable if not provided)
    api_base : str, optional
        Custom API base URL for openai/litellm backend (for internal/self-hosted endpoints)
    api_version : str, optional
        Azure API version for the openai backend
    enable_text_features : bool, default=True
        Generate built-in and optional LLM-suggested text features
    text_feature_types : list[str], optional
        Text feature categories to request. Defaults to sentiment, readability,
        linguistic, and semantic. This guides suggestions, not built-in features.
    keep_text_columns : bool, default=True
        Retain original text columns in transformed data
    temperature : float, default=0.3
        Generation temperature for openai/litellm; not exposed by the Copilot SDK
    timeout : float, default=60.0
        Provider request timeout in seconds
    max_tokens : int, default=4096
        Response token limit for openai/litellm; not exposed by the Copilot SDK
    feature_format : {'code', 'plan'}, default='code'
        Code mode preserves trusted Python generation. Plan mode requests only
        allowlisted fitted tabular rules, never executes generated Python, and
        skips built-in/generated text features while retaining original columns.
        Copilot plans use a fresh isolated session per request, without tools,
        MCPs, skills, custom agents or workspace instructions. An existing CLI
        login and a supported isolation contract are required; no permissive
        session fallback is allowed. Chat backends remain tool-free completions.
        Code-generation helpers suggest_more_features/generate_custom_feature
        are unavailable in plan mode; supply new rules and refit instead.
    rules : list[FittedRuleSpec or dict], optional
        Predeclared rules for plan mode, bypassing the provider entirely.
    task : {'auto', 'classification', 'regression'}, default='auto'
        Target interpretation for plan-mode target means. Multiclass target
        means are unsupported. Pass groups or timestamps to fit for grouped
        or chronological cross-fitting; label_end_times additionally purges
        labels unavailable at temporal validation cutoffs.
    random_state : int, default=42
        Seed for plan-mode target cross-fitting.
    target_cv : int, default=5
        Cross-fitting folds for plan-mode target means.

    Examples
    --------
    Using GitHub Copilot (default):
    >>> engine = SemanticEngine(model='gpt-4o', domain='healthcare')
    >>> X_features = engine.fit_transform(
    ...     X, y,
    ...     column_descriptions={'age': 'Patient age', 'bmi': 'Body mass index'},
    ...     task_description='Predict diabetes risk'
    ... )

    Using OpenAI SDK with custom endpoint:
    >>> engine = SemanticEngine(
    ...     model='gpt-4o',
    ...     backend='openai',
    ...     api_base='https://internal.endpoint/v1',
    ...     api_key='your-api-key'
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
        model: str = DEFAULT_MODEL,
        max_suggestions: int = 20,
        validate_features: bool = True,
        domain: str | None = None,
        verbose: bool = False,
        backend: Literal["copilot", "litellm", "openai"] = "copilot",
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        enable_text_features: bool = True,
        text_feature_types: list[str] | None = None,
        keep_text_columns: bool = True,
        temperature: float = 0.3,
        timeout: float = 60.0,
        max_tokens: int = 4096,
        feature_format: Literal["code", "plan"] = "code",
        rules: list[FittedRuleSpec | dict[str, JsonValue]] | None = None,
        task: Literal["auto", "classification", "regression"] = "auto",
        random_state: int = 42,
        target_cv: int = 5,
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
            api_version=api_version,
            enable_text_features=enable_text_features,
            keep_text_columns=keep_text_columns,
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens,
            feature_format=feature_format,
            task=task,
            random_state=random_state,
            target_cv=target_cv,
            text_feature_types=(
                text_feature_types
                if text_feature_types is not None
                else ["sentiment", "readability", "linguistic", "semantic"]
            ),
            **kwargs,
        )
        super().__init__(config=config)
        self.config: SemanticEngineConfig = config
        self._client: FeatureClient | None = None
        self._suggested_features: list[dict[str, Any]] = []
        self._text_features: list[dict[str, Any]] = []
        self._feature_set = FeatureSet()
        self._column_info: dict[str, str] = {}
        self._column_descriptions: dict[str, str] = {}
        self._task_description: str = ""
        self._text_columns: list[str] = []
        if rules is not None and feature_format != "plan":
            raise ValueError("Predeclared rules require feature_format='plan'")
        self._configured_rules = (
            [rule.model_copy(deep=True) for rule in FittedRulePlan.model_validate({"rules": rules}).rules]
            if rules is not None
            else None
        )
        self._plan_engine: FittedRuleEngine | None = None

    def _ensure_client(self) -> FeatureClient:
        """Ensure LLM client is initialized."""
        if self._client is None:
            ignored_options = []
            if self.config.backend == "copilot":
                ignored_options = [
                    name for name in ("api_key", "api_base", "api_version") if getattr(self.config, name) is not None
                ]
                if self.config.temperature != 0.3:
                    ignored_options.append("temperature")
                if self.config.max_tokens != 4096:
                    ignored_options.append("max_tokens")
            elif self.config.backend == "litellm" and self.config.api_version is not None:
                ignored_options.append("api_version")
            if ignored_options:
                logger.warning(
                    f"SemanticEngine: Backend '{self.config.backend}' does not support "
                    f"{', '.join(ignored_options)}; these options are ignored"
                )
            if self.config.backend == "openai":
                from featcopilot.llm.openai_client import SyncOpenAIFeatureClient

                self._client = SyncOpenAIFeatureClient(
                    model=self.config.model,
                    api_key=self.config.api_key,
                    api_base=self.config.api_base,
                    api_version=self.config.api_version,
                    temperature=self.config.temperature,
                    timeout=self.config.timeout,
                    max_tokens=self.config.max_tokens,
                )
            elif self.config.backend == "litellm":
                from featcopilot.llm.litellm_client import SyncLiteLLMFeatureClient

                self._client = SyncLiteLLMFeatureClient(
                    model=self.config.model,
                    api_key=self.config.api_key,
                    api_base=self.config.api_base,
                    temperature=self.config.temperature,
                    timeout=self.config.timeout,
                    max_tokens=self.config.max_tokens,
                )
            else:
                from featcopilot.llm.copilot_client import SyncCopilotFeatureClient

                self._client = SyncCopilotFeatureClient(
                    model=self.config.model,
                    timeout=self.config.timeout,
                    **({"isolated_plan": True} if self.config.feature_format == "plan" else {}),
                )
            try:
                self._client.start()
            except Exception:
                self._client = None
                raise
        return self._client

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        column_descriptions: dict[str, str] | None = None,
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
        **kwargs : dict
            In plan mode, groups, timestamps and label_end_times are forwarded
            to FittedRuleEngine.fit. Temporal prefix rows receive a fixed-zero
            prior; get_fitted_rules()["cross_fit_report"] records their positions.

        Returns
        -------
        self : SemanticEngine
        """
        self._is_fitted = False
        self._suggested_features = []
        self._text_features = []
        self._feature_names = []
        self._feature_set = FeatureSet()
        self._column_info = {}
        self._text_columns = []
        self._feature_metadata = {}
        self._plan_engine = None
        X = self._validate_input(X)
        if not X.columns.is_unique:
            raise ValueError("SemanticEngine requires unique input column names")
        # Store metadata
        self._column_descriptions = column_descriptions or {}
        self._task_description = task_description
        if self.config.feature_format == "plan":
            return self._fit_plan(X, y, **kwargs)
        self._ensure_client()

        # Build column info and detect text columns
        self._column_info = {}
        self._text_columns = []
        for col in X.columns:
            dtype = str(X[col].dtype)
            if pd.api.types.is_string_dtype(X[col]):
                dtype = "string"
                # Detect if it's a text column (long strings with high variance)
                if X[col].str.len().mean() > 20 and X[col].nunique() > 10:
                    self._text_columns.append(col)
            elif pd.api.types.is_integer_dtype(X[col]):
                dtype = "integer"
            elif pd.api.types.is_float_dtype(X[col]):
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
            self._suggested_features = self._ensure_client().suggest_features(
                column_info=non_text_column_info,
                task_description=task_description,
                column_descriptions=column_descriptions,
                domain=self.config.domain,
                max_suggestions=self.config.max_suggestions,
            )[: self.config.max_suggestions]
        else:
            self._suggested_features = []

        if self.config.verbose:
            logger.info(f"SemanticEngine: Received {len(self._suggested_features)} suggestions")

        self._validate_suggestions(X)

        # Build feature set
        self._build_feature_set()
        self._feature_names = [suggestion["name"] for suggestion in self._text_features + self._suggested_features]
        if not self._feature_names:
            logger.warning("SemanticEngine: No valid features were generated; returning input columns only")

        self._is_fitted = True
        return self

    def _fit_plan(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None, **kwargs: object) -> "SemanticEngine":
        if any(not isinstance(column, str) for column in X.columns):
            raise ValueError("Plan mode requires string input column names")
        if not len(X):
            raise ValueError("Plan mode requires non-empty training data")
        self._column_info = {column: str(X[column].dtype) for column in X.columns}
        if self.config.enable_text_features or not self.config.keep_text_columns:
            logger.info(
                "SemanticEngine: plan mode uses only fitted tabular rules; text feature generation and "
                "keep_text_columns=False are ignored, preserving all original columns"
            )
        rules = self._configured_rules
        if rules is None:
            rules = request_fitted_plan(
                self._ensure_client(),
                self._column_info,
                self._column_descriptions,
                self._task_description,
                self.config.domain,
                self.config.max_suggestions,
                has_target=y is not None,
            )
        if len(rules) > self.config.max_suggestions:
            raise ValueError("Predeclared plan exceeds max_suggestions")
        engine = FittedRuleEngine(
            rules=list(rules),
            task=self.config.task,
            random_state=self.config.random_state,
            target_cv=self.config.target_cv,
        )
        engine.fit(X, y, **kwargs)
        self._plan_engine = engine
        self._feature_names = engine.get_feature_names()
        self._feature_metadata = engine.get_feature_metadata()
        for rule in rules:
            self._feature_set.add(
                Feature(
                    name=rule.name,
                    dtype=FeatureType.CATEGORICAL if rule.operator == "rare_category" else FeatureType.NUMERIC,
                    origin=FeatureOrigin.CUSTOM if self._configured_rules is not None else FeatureOrigin.LLM_GENERATED,
                    source_columns=rule.columns.copy(),
                    transformation=rule.operator,
                    explanation=rule.explanation,
                    metadata={"feature_format": "plan", "rule": rule.model_dump(mode="json")},
                )
            )
        if not rules:
            logger.warning("SemanticEngine: No fitted rules were proposed; returning input columns only")
        self._is_fitted = True
        return self

    def _transform_fit_data(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None, **kwargs: object
    ) -> pd.DataFrame:
        if self.config.feature_format == "plan":
            if self._plan_engine is None or not self._is_fitted:
                raise RuntimeError("Engine must be fitted before transform")
            return self._plan_engine._transform_fit_data(X, y, **kwargs)
        return self.transform(X, **kwargs)

    def get_fitted_rules(self) -> dict[str, JsonValue]:
        """
        Export plan-mode inference state without provider connections.

        Returns
        -------
        dict
            JSON-compatible state accepted by FittedRuleEngine.from_dict,
            including cross_fit_report with temporal fixed-zero prior positions.

        Raises
        ------
        RuntimeError
            If no plan has been fitted.
        """
        if self._plan_engine is None or not self._is_fitted:
            raise RuntimeError("A plan must be fitted before exporting fitted rules")
        return self._plan_engine.to_dict()

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
            for feature in fallback_features:
                feature["is_builtin"] = True
            text_features.extend(fallback_features)

            # Try to get LLM-suggested features (optional)
            try:
                col_desc = self._column_descriptions.get(col, f"Text column: {col}")

                # Use suggest_features instead of send_prompt for better compatibility
                response = self._ensure_client().suggest_features(
                    column_info={col: "string"},
                    task_description=(
                        f"Extract numerical {', '.join(self.config.text_feature_types)} features "
                        f"from text column '{col}' for {self._task_description}"
                    ),
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
                logger.warning(
                    f"SemanticEngine: Could not get LLM suggestions for '{col}': {e}; using built-in text features"
                )

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
            features = feature_suggestions(data)

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
                    features = feature_suggestions(data)
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
        sample_data = {col: X[col].head(100).tolist() for col in X.columns}
        used_names = set(X.columns)
        for attribute in ("_text_features", "_suggested_features"):
            valid_features = []
            for feature in getattr(self, attribute):
                name = feature.get("name")
                code = feature.get("code")
                if not isinstance(name, str) or not name.strip() or name in used_names:
                    logger.warning(f"SemanticEngine: Ignoring invalid or duplicate feature name {name!r}")
                    continue
                if not isinstance(code, str) or not code.strip():
                    logger.warning(f"SemanticEngine: Feature '{name}' has no code")
                    continue
                if self.config.validate_features:
                    validation = self._ensure_client().validate_feature_code(code, sample_data)
                    if not validation["valid"]:
                        logger.warning(f"SemanticEngine: Invalid feature '{name}': {validation.get('error')}")
                        continue
                    try:
                        execute_feature_code(code, X)
                    except Exception as error:
                        logger.warning(f"SemanticEngine: Invalid feature '{name}': {error}")
                        continue
                used_names.add(name)
                valid_features.append(feature.copy())
            setattr(self, attribute, valid_features)

    def _build_feature_set(self) -> None:
        """Build FeatureSet from suggestions."""
        self._feature_set = FeatureSet()

        # Add text features
        for suggestion in self._text_features:
            feature = Feature(
                name=suggestion.get("name", f"text_feature_{len(self._feature_set)}"),
                dtype=FeatureType.NUMERIC,
                origin=FeatureOrigin.CUSTOM if suggestion.get("is_builtin") else FeatureOrigin.LLM_GENERATED,
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

    def transform(self, X: pd.DataFrame | np.ndarray, **kwargs) -> pd.DataFrame:
        """
        Generate LLM-suggested features.

        Parameters
        ----------
        X : DataFrame
            Input data

        Returns
        -------
        X_features : DataFrame
            Data with the fitted generated features. Original text columns are
            retained unless keep_text_columns=False.
        """
        if not self._is_fitted:
            raise RuntimeError("Engine must be fitted before transform")
        if self.config.feature_format == "plan":
            if self._plan_engine is None:
                raise RuntimeError("Plan engine is not fitted")
            return self._plan_engine.transform(X, **kwargs)

        X = self._validate_input(X)
        missing = [column for column in self._column_info if column not in X.columns]
        if missing:
            raise ValueError(f"SemanticEngine input is missing fitted columns: {missing}")
        result = X.copy()

        for suggestion in self._text_features + self._suggested_features:
            name = suggestion.get("name", "")
            code = suggestion.get("code", "")
            try:
                if name in result.columns:
                    raise ValueError("Generated feature name collides with an input column")
                result[name] = execute_feature_code(code, X, allow_all_missing=True)
            except Exception as error:
                raise ValueError(f"SemanticEngine: Error computing fitted feature '{name}': {error}") from error

        # Optionally drop original text columns (only if not keeping them for downstream models)
        if not self.config.keep_text_columns:
            cols_to_drop = [col for col in self._text_columns if col in result.columns]
            if cols_to_drop:
                result = result.drop(columns=cols_to_drop)
                if self.config.verbose:
                    logger.info(f"SemanticEngine: Dropped {len(cols_to_drop)} text columns, keeping numerical features")

        if self.config.verbose:
            logger.info(f"SemanticEngine: Successfully generated {len(self._feature_names)} features")

        return result

    def get_feature_explanations(self) -> dict[str, str]:
        """
        Get explanations for all generated features.

        Returns
        -------
        explanations : dict
            Mapping of feature names to explanations
        """
        if self.config.feature_format == "plan":
            return {feature.name: feature.explanation or "" for feature in self._feature_set}
        return {
            s["name"]: s.get("explanation", "") for s in self._text_features + self._suggested_features if s.get("name")
        }

    def get_feature_code(self) -> dict[str, str]:
        """
        Get code for all generated features.

        Returns
        -------
        code : dict
            Mapping of feature names to Python code
        """
        return {s["name"]: s.get("code", "") for s in self._text_features + self._suggested_features if s.get("name")}

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
        if self.config.feature_format == "plan":
            raise ValueError("suggest_more_features generates trusted code; refit with a new plan instead")
        self._ensure_client()

        # Build focused prompt
        enhanced_task = f"{self._task_description}\n\nFocus specifically on: {focus_area}"

        new_suggestions = self._ensure_client().suggest_features(
            column_info=self._column_info,
            task_description=enhanced_task,
            column_descriptions=self._column_descriptions,
            domain=self.config.domain,
            max_suggestions=n_features,
        )

        return new_suggestions

    def generate_custom_feature(self, description: str, constraints: list[str] | None = None) -> dict[str, Any]:
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
        if self.config.feature_format == "plan":
            raise ValueError("generate_custom_feature generates trusted code; supply a typed rule and refit instead")
        self._ensure_client()

        code = self._ensure_client().generate_feature_code(
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

    def __getstate__(self) -> dict[str, Any]:
        """Persist generated rules, not live provider connections."""
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def standardize_categories(
        self,
        df: pd.DataFrame,
        column: str,
        target_categories: list[str] | None = None,
        similarity_threshold: float = 0.8,
        max_categories: int = 50,
        context: str | None = None,
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
            client = self._ensure_client()
            if hasattr(client, "send_prompt"):
                response = client.send_prompt(prompt)
            else:
                # Fallback: use suggest_features with a specialized task
                response_list = client.suggest_features(
                    column_info={column: "categorical"},
                    task_description=prompt,
                    column_descriptions={column: context or "Categorical column to standardize"},
                    domain=self.config.domain,
                    max_suggestions=1,
                )
                # Extract mapping from response if possible
                if response_list and isinstance(response_list, list) and len(response_list) > 0:
                    first = response_list[0]
                    if isinstance(first, dict) and isinstance(first.get("mapping"), dict):
                        return {str(key): str(value) for key, value in first["mapping"].items()}
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
        target_categories: list[str] | None = None,
        context: str | None = None,
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
        contexts: dict[str, str] | None = None,
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
        client = getattr(self, "_client", None)
        if client:
            try:
                client.stop()
            except Exception:
                pass
