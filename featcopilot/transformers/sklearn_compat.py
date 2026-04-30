"""Scikit-learn compatible feature engineering transformers.

Provides drop-in sklearn transformers for feature engineering pipelines.
"""

import warnings
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
from featcopilot.utils.logger import get_logger
from featcopilot.utils.validation import find_potential_leakage_columns

logger = get_logger(__name__)


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
        Engines to use ('tabular', 'timeseries', 'relational', 'text', 'llm')
    max_features : int, optional
        Maximum features to generate/select
    selection_methods : list, default=['mutual_info', 'importance']
        Feature selection methods
    correlation_threshold : float, default=0.85
        Maximum pairwise correlation allowed during correlation-based selection
    llm_config : dict, optional
        Configuration for LLM engine
    verbose : bool, default=False
        Verbose output
    leakage_guard : {'off', 'warn', 'raise'}, default='warn'
        How to handle columns whose names suggest target, label, or future-information leakage

    Other Parameters
    ----------------
    target_name : str, optional
        Fit-time parameter accepted by :meth:`fit` and :meth:`fit_transform`.
        When provided, the leakage guard cross-references column names against
        the target so derived variants (e.g. ``target_encoded``) are flagged.

    Examples
    --------
    >>> engineer = AutoFeatureEngineer(
    ...     engines=['tabular', 'llm'],
    ...     max_features=100,
    ...     llm_config={'model': 'gpt-5.2', 'enable_semantic': True}
    ... )
    >>> X_transformed = engineer.fit_transform(X, y, target_name='label')
    """

    SUPPORTED_ENGINES = {"tabular", "timeseries", "relational", "text", "llm"}
    SUPPORTED_SELECTION_METHODS = {"mutual_info", "importance", "f_test", "chi2", "correlation", "xgboost"}
    SUPPORTED_LEAKAGE_GUARDS = {"off", "warn", "raise"}

    def __init__(
        self,
        engines: Optional[list[str]] = None,
        max_features: Optional[int] = None,
        selection_methods: Optional[list[str]] = None,
        correlation_threshold: float = 0.85,
        llm_config: Optional[dict[str, Any]] = None,
        verbose: bool = False,
        leakage_guard: str = "warn",
    ):
        # Use ``is not None`` defaulting (rather than ``or``) so that explicit
        # empty containers and identity-bearing arguments are preserved. This
        # also keeps ``self.<param> is param`` for any non-None argument, which
        # is required for sklearn's ``clone`` round-trip identity check.
        self.engines = engines if engines is not None else ["tabular"]
        self.max_features = max_features
        self.selection_methods = selection_methods if selection_methods is not None else ["mutual_info", "importance"]
        self.correlation_threshold = correlation_threshold
        self.llm_config = llm_config if llm_config is not None else {}
        self.verbose = verbose
        self.leakage_guard = leakage_guard

        self._validate_configuration()

        self._engine_instances: dict[str, Any] = {}
        self._selector: Optional[FeatureSelector] = None
        self._feature_set = FeatureSet()
        self._is_fitted = False
        self._column_descriptions: dict[str, str] = {}
        self._task_description: str = ""

    def _validate_configuration(self) -> None:
        """Validate user-facing configuration early."""
        unknown_engines = sorted(set(self.engines) - self.SUPPORTED_ENGINES)
        if unknown_engines:
            raise ValueError(f"Unknown engines: {unknown_engines}. Supported engines: {sorted(self.SUPPORTED_ENGINES)}")

        unknown_methods = sorted(set(self.selection_methods) - self.SUPPORTED_SELECTION_METHODS)
        if unknown_methods:
            raise ValueError(
                "Unknown selection methods: "
                f"{unknown_methods}. Supported methods: {sorted(self.SUPPORTED_SELECTION_METHODS)}"
            )

        if self.leakage_guard not in self.SUPPORTED_LEAKAGE_GUARDS:
            raise ValueError(
                f"leakage_guard must be one of {sorted(self.SUPPORTED_LEAKAGE_GUARDS)}, got {self.leakage_guard!r}"
            )

        if self.max_features is not None and self.max_features <= 0:
            raise ValueError("max_features must be positive when provided")

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        column_descriptions: Optional[dict[str, str]] = None,
        task_description: str = "prediction task",
        target_name: Optional[str] = None,
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
        target_name : str, optional
            Target column name used by leakage checks to identify related feature columns
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

        suspicious_columns = find_potential_leakage_columns(X.columns.tolist(), target_name=target_name)
        if suspicious_columns and self.leakage_guard != "off":
            message = (
                "Potential leakage-prone columns detected: "
                f"{suspicious_columns}. Review time/label leakage before fitting, "
                "or set leakage_guard='off' to disable this check."
            )
            if self.leakage_guard == "raise":
                raise ValueError(message)
            warnings.warn(message, UserWarning, stacklevel=2)

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
                logger.info(f"Fitted {engine_name} engine")

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
        elif engine_name == "relational":
            return RelationalEngine(max_features=self.max_features, verbose=self.verbose)
        elif engine_name == "llm":
            from featcopilot.llm.semantic_engine import SemanticEngine

            return SemanticEngine(
                model=self.llm_config.get("model", "gpt-5.2"),
                max_suggestions=self.llm_config.get("max_suggestions", 20),
                domain=self.llm_config.get("domain"),
                verbose=self.verbose,
                backend=self.llm_config.get("backend", "copilot"),
                api_key=self.llm_config.get("api_key"),
                api_base=self.llm_config.get("api_base"),
                api_version=self.llm_config.get("api_version"),
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
                logger.info(f"{engine_name}: Added {len(new_cols)} features")

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
        target_name: Optional[str] = None,
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
        target_name : str, optional
            Target column name used by leakage checks to identify related feature columns
        apply_selection : bool, default=True
            Whether to apply feature selection
        **fit_params : dict
            Additional parameters

        Returns
        -------
        X_transformed : DataFrame
            Transformed data with generated features
        """
        self.fit(X, y, column_descriptions, task_description, target_name=target_name, **fit_params)
        # Reuse transform-relevant kwargs (e.g. text_columns, related_tables) during fit_transform.
        result = self.transform(X, **fit_params)

        # Track original features (input columns) vs derived features
        if isinstance(X, np.ndarray):
            original_features = {f"feature_{i}" for i in range(X.shape[1])}
        else:
            original_features = set(X.columns)

        # Apply feature selection if enabled and y is provided
        if apply_selection and y is not None and self.max_features:
            self._selector = FeatureSelector(
                methods=self.selection_methods,
                max_features=self.max_features,
                correlation_threshold=self.correlation_threshold,
                original_features=original_features,
                verbose=self.verbose,
            )
            result = self._selector.fit_transform(result, y)

            if self.verbose:
                logger.info(f"Selected {len(self._selector.get_selected_features())} features")

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
            "leakage_guard": self.leakage_guard,
        }

    def set_params(self, **params):
        """
        Set parameters for sklearn compatibility.

        Mirrors the defaulting performed in ``__init__`` so callers (e.g. sklearn
        cloning, ``GridSearchCV`` parameter grids) can pass ``None`` for
        collection-valued parameters and have it normalized back to the default
        rather than raising during validation.
        """
        for key, value in params.items():
            setattr(self, key, value)
        if self.engines is None:
            self.engines = ["tabular"]
        if self.selection_methods is None:
            self.selection_methods = ["mutual_info", "importance"]
        if self.llm_config is None:
            self.llm_config = {}
        self._validate_configuration()
        return self
