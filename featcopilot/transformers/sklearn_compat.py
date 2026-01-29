"""Scikit-learn compatible feature engineering transformers.

Provides drop-in sklearn transformers for feature engineering pipelines.
"""

from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from featcopilot.core.feature import FeatureSet
from featcopilot.engines.relational import RelationalEngine
from featcopilot.engines.tabular import TabularEngine
from featcopilot.engines.text import TextEngine
from featcopilot.engines.timeseries import TimeSeriesEngine
from featcopilot.selection.unified import FeatureSelector


class FeatureEngineerTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible feature engineering transformer.

    Wraps individual engines for use in sklearn pipelines.

    Parameters
    ----------
    engine : str, default='tabular'
        Engine type ('tabular', 'timeseries', 'relational', 'text')
    **engine_kwargs : dict
        Arguments passed to the engine

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> pipe = Pipeline([
    ...     ('features', FeatureEngineerTransformer(engine='tabular')),
    ...     ('model', LogisticRegression())
    ... ])
    """

    def __init__(self, engine: str = "tabular", **engine_kwargs):
        self.engine = engine
        self.engine_kwargs = engine_kwargs
        self._engine_instance = None

    def _create_engine(self):
        """Create the appropriate engine instance."""
        engines = {
            "tabular": TabularEngine,
            "timeseries": TimeSeriesEngine,
            "relational": RelationalEngine,
            "text": TextEngine,
        }

        if self.engine not in engines:
            raise ValueError(f"Unknown engine: {self.engine}")

        return engines[self.engine](**self.engine_kwargs)

    def fit(self, X, y=None, **fit_params):
        """Fit the transformer."""
        self._engine_instance = self._create_engine()
        self._engine_instance.fit(X, y, **fit_params)
        return self

    def transform(self, X, **transform_params):
        """Transform data to generate features."""
        if self._engine_instance is None:
            raise RuntimeError("Transformer must be fitted before transform")
        return self._engine_instance.transform(X, **transform_params)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if self._engine_instance is None:
            return []
        return self._engine_instance.get_feature_names()


class AutoFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Main auto feature engineering class.

    Combines multiple engines and selection methods for comprehensive
    automated feature engineering with LLM capabilities.

    Parameters
    ----------
    engines : list, default=['tabular']
        Engines to use ('tabular', 'timeseries', 'text', 'llm')
    max_features : int, optional
        Maximum features to generate/select
    selection_methods : list, default=['mutual_info', 'importance']
        Feature selection methods
    llm_config : dict, optional
        Configuration for LLM engine
    verbose : bool, default=False
        Verbose output

    Examples
    --------
    >>> engineer = AutoFeatureEngineer(
    ...     engines=['tabular', 'llm'],
    ...     max_features=100,
    ...     llm_config={'model': 'gpt-5', 'enable_semantic': True}
    ... )
    >>> X_transformed = engineer.fit_transform(X, y)
    """

    def __init__(
        self,
        engines: Optional[list[str]] = None,
        max_features: Optional[int] = None,
        selection_methods: Optional[list[str]] = None,
        correlation_threshold: float = 0.95,
        llm_config: Optional[dict[str, Any]] = None,
        verbose: bool = False,
    ):
        self.engines = engines or ["tabular"]
        self.max_features = max_features
        self.selection_methods = selection_methods or ["mutual_info", "importance"]
        self.correlation_threshold = correlation_threshold
        self.llm_config = llm_config or {}
        self.verbose = verbose

        self._engine_instances: dict[str, Any] = {}
        self._selector: Optional[FeatureSelector] = None
        self._feature_set = FeatureSet()
        self._is_fitted = False
        self._column_descriptions: dict[str, str] = {}
        self._task_description: str = ""

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        column_descriptions: Optional[dict[str, str]] = None,
        task_description: str = "prediction task",
        **fit_params,
    ) -> "AutoFeatureEngineer":
        """
        Fit the auto feature engineer.

        Parameters
        ----------
        X : DataFrame or ndarray
            Input data
        y : Series or ndarray, optional
            Target variable
        column_descriptions : dict, optional
            Human-readable descriptions of columns (for LLM)
        task_description : str
            Description of the ML task (for LLM)
        **fit_params : dict
            Additional parameters

        Returns
        -------
        self : AutoFeatureEngineer
        """
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        self._column_descriptions = column_descriptions or {}
        self._task_description = task_description

        # Fit each engine
        for engine_name in self.engines:
            engine = self._create_engine(engine_name)

            if engine_name == "llm":
                engine.fit(
                    X,
                    y,
                    column_descriptions=column_descriptions,
                    task_description=task_description,
                    **fit_params,
                )
            else:
                engine.fit(X, y, **fit_params)

            self._engine_instances[engine_name] = engine

            if self.verbose:
                print(f"Fitted {engine_name} engine")

        self._is_fitted = True
        return self

    def _create_engine(self, engine_name: str):
        """Create an engine instance."""
        if engine_name == "tabular":
            return TabularEngine(max_features=self.max_features, verbose=self.verbose)
        elif engine_name == "timeseries":
            return TimeSeriesEngine(max_features=self.max_features, verbose=self.verbose)
        elif engine_name == "text":
            return TextEngine(max_features=self.max_features, verbose=self.verbose)
        elif engine_name == "llm":
            from featcopilot.llm.semantic_engine import SemanticEngine

            return SemanticEngine(
                model=self.llm_config.get("model", "gpt-5"),
                max_suggestions=self.llm_config.get("max_suggestions", 20),
                domain=self.llm_config.get("domain"),
                verbose=self.verbose,
            )
        else:
            raise ValueError(f"Unknown engine: {engine_name}")

    def transform(self, X: Union[pd.DataFrame, np.ndarray], **transform_params) -> pd.DataFrame:
        """
        Transform data using fitted engines.

        Parameters
        ----------
        X : DataFrame or ndarray
            Input data
        **transform_params : dict
            Additional parameters

        Returns
        -------
        X_transformed : DataFrame
            Data with generated features
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit before transform")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        result = X.copy()

        # Transform with each engine
        for engine_name, engine in self._engine_instances.items():
            transformed = engine.transform(X, **transform_params)

            # Add new features to result
            new_cols = [c for c in transformed.columns if c not in result.columns]
            for col in new_cols:
                result[col] = transformed[col]

            if self.verbose:
                print(f"{engine_name}: Added {len(new_cols)} features")

        # Handle infinities and NaNs
        result = result.replace([np.inf, -np.inf], np.nan)

        # Apply selection if selector was fitted
        if self._selector is not None:
            selected_features = self._selector.get_selected_features()
            # Keep only selected features that exist in result
            available = [f for f in selected_features if f in result.columns]
            result = result[available]

        return result

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        column_descriptions: Optional[dict[str, str]] = None,
        task_description: str = "prediction task",
        apply_selection: bool = True,
        **fit_params,
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Parameters
        ----------
        X : DataFrame or ndarray
            Input data
        y : Series or ndarray, optional
            Target variable
        column_descriptions : dict, optional
            Human-readable column descriptions
        task_description : str
            ML task description
        apply_selection : bool, default=True
            Whether to apply feature selection
        **fit_params : dict
            Additional parameters

        Returns
        -------
        X_transformed : DataFrame
            Transformed data with generated features
        """
        self.fit(X, y, column_descriptions, task_description, **fit_params)
        result = self.transform(X)

        # Apply feature selection if enabled and y is provided
        if apply_selection and y is not None and self.max_features:
            self._selector = FeatureSelector(
                methods=self.selection_methods,
                max_features=self.max_features,
                correlation_threshold=self.correlation_threshold,
                verbose=self.verbose,
            )
            result = self._selector.fit_transform(result, y)

            if self.verbose:
                print(f"Selected {len(self._selector.get_selected_features())} features")

        return result

    def get_feature_names(self) -> list[str]:
        """Get names of all generated features."""
        names = []
        for engine in self._engine_instances.values():
            names.extend(engine.get_feature_names())
        return names

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Sklearn-compatible method for feature names."""
        return self.get_feature_names()

    def explain_features(self) -> dict[str, str]:
        """
        Get explanations for all features.

        Returns
        -------
        explanations : dict
            Mapping of feature names to explanations
        """
        explanations = {}

        for _, engine in self._engine_instances.items():
            if hasattr(engine, "get_feature_explanations"):
                explanations.update(engine.get_feature_explanations())
            elif hasattr(engine, "get_feature_set"):
                feature_set = engine.get_feature_set()
                explanations.update(feature_set.get_explanations())

        return explanations

    def get_feature_code(self) -> dict[str, str]:
        """
        Get code for generated features.

        Returns
        -------
        code : dict
            Mapping of feature names to Python code
        """
        code = {}

        for _, engine in self._engine_instances.items():
            if hasattr(engine, "get_feature_code"):
                code.update(engine.get_feature_code())

        return code

    def generate_custom_features(self, prompt: str, n_features: int = 5) -> list[dict[str, Any]]:
        """
        Generate custom features via LLM prompt.

        Parameters
        ----------
        prompt : str
            Natural language description of desired features
        n_features : int, default=5
            Number of features to generate

        Returns
        -------
        features : list
            List of generated feature definitions
        """
        if "llm" not in self._engine_instances:
            raise RuntimeError("LLM engine not enabled. Add 'llm' to engines list.")

        llm_engine = self._engine_instances["llm"]
        return llm_engine.suggest_more_features(prompt, n_features)

    @property
    def feature_importances_(self) -> Optional[dict[str, float]]:
        """Get feature importance scores if selection was applied."""
        if self._selector is not None:
            return self._selector.get_feature_scores()
        return None

    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            "engines": self.engines,
            "max_features": self.max_features,
            "selection_methods": self.selection_methods,
            "correlation_threshold": self.correlation_threshold,
            "llm_config": self.llm_config,
            "verbose": self.verbose,
        }

    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
