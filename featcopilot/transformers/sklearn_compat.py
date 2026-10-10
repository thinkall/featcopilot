"""Scikit-learn compatible feature engineering transformers.

Provides drop-in sklearn transformers for feature engineering pipelines.
"""

import warnings
from copy import deepcopy
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone

from featcopilot.core.feature import FeatureSet
from featcopilot.engines.relational import RelationalEngine
from featcopilot.engines.tabular import TabularEngine
from featcopilot.engines.text import TextEngine
from featcopilot.engines.timeseries import TimeSeriesEngine
from featcopilot.selection.unified import FeatureSelector
from featcopilot.utils.logger import get_logger
from featcopilot.utils.models import DEFAULT_MODEL
from featcopilot.utils.task import resolve_task, validate_task
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
        if self.engine == "point_in_time":
            from featcopilot.engines.point_in_time import PointInTimeRelationalEngine

            return PointInTimeRelationalEngine(**self.engine_kwargs)
        if self.engine == "fitted_rules":
            from featcopilot.engines.fitted_rules import FittedRuleEngine

            return FittedRuleEngine(**self.engine_kwargs)
        if self.engine == "predictive":
            from featcopilot.engines.predictive import PredictiveTabularEngine

            return PredictiveTabularEngine(**self.engine_kwargs)
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

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and transform while preserving engine-specific training semantics."""
        self._engine_instance = self._create_engine()
        return self._engine_instance.fit_transform(X, y, **fit_params)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if self._engine_instance is None:
            return []
        return self._engine_instance.get_feature_names()

    def get_params(self, deep=True):
        """Return cloneable sklearn parameters, including engine options."""
        params = {"engine": self.engine, **self.engine_kwargs}
        if deep:
            for name, value in self.engine_kwargs.items():
                if hasattr(value, "get_params"):
                    params.update({f"{name}__{key}": item for key, item in value.get_params(deep=True).items()})
        return params

    def set_params(self, **params):
        """Set the engine name or engine-specific options."""
        if not params:
            return self
        engine = params.pop("engine", self.engine)
        options = self.engine_kwargs.copy()
        nested: dict[str, dict[str, Any]] = {}
        for name, value in params.items():
            parameter, separator, sub_parameter = name.partition("__")
            if separator:
                nested.setdefault(parameter, {})[sub_parameter] = value
            else:
                options[name] = value
        for parameter, updates in nested.items():
            estimator = options.get(parameter)
            if not hasattr(estimator, "set_params"):
                raise ValueError(f"{parameter!r} is not a configured sklearn estimator")
            options[parameter] = clone(estimator).set_params(**updates)
        self.engine = engine
        self.engine_kwargs = options
        self._engine_instance = None
        return self


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
    gate_n_jobs : int, default=1
        Number of parallel jobs for the do-no-harm gate's internal validation
        models (RandomForest). Defaults to ``1`` (sequential) to avoid CPU
        oversubscription when ``AutoFeatureEngineer`` is wrapped in an outer
        parallel CV / grid-search. Set to ``-1`` to use all cores when running
        standalone.
    task : {'auto', 'classification', 'regression'}, default='auto'
        Explicit task override for selection and gate models. Set 'regression'
        for integer-valued regression targets. 'auto' preserves legacy heuristics.
    selection_strategy : {'legacy', 'predictive'}, default='legacy'
        Opt-in predictive tabular selection uses an internal training holdout
        and the configured estimator instead of the legacy selector/gate.
        Currently requires engines=['tabular'] and a target.
    selection_estimator : sklearn estimator, optional
        Bare downstream learner used by the predictive strategy. It is cloned
        and receives internally preprocessed numeric data. Defaults to sklearn
        histogram gradient boosting. Ignored estimators are rejected in legacy mode.
    max_candidates : int, default=1000
        Maximum candidate additions considered by the predictive strategy.
    random_state : int, default=42
        Seed for the predictive strategy's internal splitting and selection.
    stability_repeats : int, default=3
        Internal training-only validation repeats for predictive subset selection.
    screening : {'auto', 'linear', 'tree', 'importance'}, default='auto'
        Residual-screening learner. Automatic selection uses linear screening
        for supported bare linear estimators and tree screening otherwise.
        Importance mode requires an explicitly supplied compatible learner.
    selection_rule : {'one_se', 'paired_gain'}, default='one_se'
        Predictive subset rule. The experimental paired-gain rule uses loss
        differences on matching internal repeats, not absolute loss variation.
    candidate_set : {'standard', 'extended'}, default='standard'
        Predictive candidate vocabulary. Extended adds sums/minima/maxima and
        frozen training-observed categorical-pair indicators with balanced coverage.
    include_frequency : bool, default=False
        Include bounded training-fitted unary frequency candidates. Requires
        predictive selection with explicit importance screening and a compatible learner.
    numeric_clip_quantiles : tuple[float, float], optional
        Train-fitted numeric clipping for predictive importance screening.
        Feature outputs remain unchanged; apply the same clipper in the downstream model.
    importance_resamples : {1, 5}, default=1
        Joint-importance ranking fits. Five enables empirical mean-rank
        aggregation over training-only IID subsamples; one preserves existing behavior.
    categorical_encoding : {"one_hot", "target"}, default="one_hot"
        Model-input encoding for predictive importance screening. Target mode
        requires IID classification and a matching downstream encoding policy.

    Other Parameters
    ----------------
    target_name : hashable, optional
        Fit-time parameter accepted by :meth:`fit` and :meth:`fit_transform`.
        When provided, the leakage guard cross-references column labels
        against the target so derived variants (e.g. ``target_encoded``) are
        flagged. Accepts any column-label type DataFrames support
        (typically ``str``, but also ``int`` or other hashables).

    Examples
    --------
    >>> engineer = AutoFeatureEngineer(
    ...     engines=['tabular', 'llm'],
    ...     max_features=100,
    ...     llm_config={'model': 'gpt-5.6-sol', 'validate_features': True}
    ... )
    >>> X_transformed = engineer.fit_transform(X, y, target_name='label')
    """

    SUPPORTED_ENGINES = {"tabular", "timeseries", "relational", "text", "llm"}
    SUPPORTED_SELECTION_METHODS = {"mutual_info", "importance", "f_test", "chi2", "correlation", "xgboost"}
    SUPPORTED_LEAKAGE_GUARDS = {"off", "warn", "raise"}
    SUPPORTED_TASKS = {"auto", "classification", "regression"}
    SUPPORTED_SELECTION_STRATEGIES = {"legacy", "predictive"}
    SUPPORTED_LLM_CONFIG_KEYS = {
        "model",
        "max_suggestions",
        "validate_features",
        "domain",
        "verbose",
        "backend",
        "api_key",
        "api_base",
        "api_version",
        "enable_text_features",
        "text_feature_types",
        "keep_text_columns",
        "temperature",
        "timeout",
        "max_tokens",
        "feature_format",
        "rules",
        "target_cv",
    }

    def __init__(
        self,
        engines: list[str] | None = None,
        max_features: int | None = None,
        selection_methods: list[str] | None = None,
        correlation_threshold: float = 0.85,
        llm_config: dict[str, Any] | None = None,
        verbose: bool = False,
        leakage_guard: str = "warn",
        gate_n_jobs: int = 1,
        task: str = "auto",
        selection_strategy: str = "legacy",
        selection_estimator: BaseEstimator | None = None,
        max_candidates: int = 1000,
        random_state: int = 42,
        stability_repeats: int = 3,
        screening: str = "auto",
        selection_rule: str = "one_se",
        candidate_set: str = "standard",
        include_frequency: bool = False,
        numeric_clip_quantiles: tuple[float, float] | None = None,
        importance_resamples: int = 1,
        categorical_encoding: str = "one_hot",
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
        self.gate_n_jobs = gate_n_jobs
        self.task = task
        self.selection_strategy = selection_strategy
        self.selection_estimator = selection_estimator
        self.max_candidates = max_candidates
        self.random_state = random_state
        self.stability_repeats = stability_repeats
        self.screening = screening
        self.selection_rule = selection_rule
        self.candidate_set = candidate_set
        self.include_frequency = include_frequency
        self.numeric_clip_quantiles = numeric_clip_quantiles
        self.importance_resamples = importance_resamples
        self.categorical_encoding = categorical_encoding

        self._validate_configuration()

        self._engine_instances: dict[str, Any] = {}
        self._selector: FeatureSelector | None = None
        self._feature_set = FeatureSet()
        self._is_fitted = False
        self._column_descriptions: dict[str, str] = {}
        self._task_description: str = ""
        self.selection_report_: dict[str, Any] | None = None

    def _validate_configuration(self) -> None:
        """Validate user-facing configuration early."""
        from featcopilot.utils.numeric_clipping import validate_clip_quantiles

        if self.categorical_encoding not in ("one_hot", "target"):
            raise ValueError("categorical_encoding must be 'one_hot' or 'target'")
        if self.categorical_encoding == "target":
            if self.selection_strategy != "predictive" or self.screening != "importance":
                raise ValueError("categorical_encoding='target' requires predictive importance screening")
            if self.task == "regression":
                raise ValueError("categorical_encoding='target' supports classification only")
        validate_clip_quantiles(self.numeric_clip_quantiles)
        if self.numeric_clip_quantiles is not None and self.selection_strategy != "predictive":
            raise ValueError("numeric_clip_quantiles requires selection_strategy='predictive'")
        if self.numeric_clip_quantiles is not None and self.screening != "importance":
            raise ValueError("numeric_clip_quantiles requires screening='importance'")
        if (
            isinstance(self.importance_resamples, bool)
            or not isinstance(self.importance_resamples, (int, np.integer))
            or self.importance_resamples not in (1, 5)
        ):
            raise ValueError("importance_resamples must be 1 or 5")
        if self.importance_resamples > 1 and (
            self.selection_strategy != "predictive" or self.screening != "importance"
        ):
            raise ValueError("importance_resamples=5 requires predictive selection with screening='importance'")
        validate_task(self.task)
        if (
            not isinstance(self.selection_strategy, str)
            or self.selection_strategy not in self.SUPPORTED_SELECTION_STRATEGIES
        ):
            raise ValueError(f"selection_strategy must be one of {sorted(self.SUPPORTED_SELECTION_STRATEGIES)}")
        if self.selection_strategy == "predictive" and self.engines not in (["tabular"], ("tabular",)):
            raise ValueError("The predictive strategy currently requires engines=['tabular']")
        if self.selection_estimator is not None:
            if self.selection_strategy != "predictive":
                raise ValueError("selection_estimator requires selection_strategy='predictive'")
            if not all(
                callable(getattr(self.selection_estimator, name, None)) for name in ("get_params", "fit", "predict")
            ):
                raise ValueError("selection_estimator must be a cloneable sklearn estimator")
            try:
                clone(self.selection_estimator)
            except (TypeError, RuntimeError) as exc:
                raise ValueError("selection_estimator must be a cloneable sklearn estimator") from exc
        for name, value in (
            ("max_candidates", self.max_candidates),
            ("random_state", self.random_state),
            ("stability_repeats", self.stability_repeats),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if self.stability_repeats < 1:
            raise ValueError("stability_repeats must be positive")
        if self.selection_strategy == "predictive" and (self.stability_repeats > 10 or self.max_candidates > 1000):
            raise ValueError(
                "Predictive selection supports at most 10 stability repeats and 1000 candidates per repeat"
            )
        if self.screening not in ("auto", "linear", "tree", "importance"):
            raise ValueError("screening must be 'auto', 'linear', 'tree' or 'importance'")
        if not isinstance(self.include_frequency, bool):
            raise ValueError("include_frequency must be a boolean")
        if self.include_frequency and (self.selection_strategy != "predictive" or self.screening != "importance"):
            raise ValueError("include_frequency requires predictive selection with screening='importance'")
        if (
            self.selection_strategy == "predictive"
            and self.screening == "importance"
            and self.selection_estimator is None
        ):
            raise ValueError("Importance screening requires an explicit compatible selection_estimator")
        if self.selection_rule not in ("one_se", "paired_gain"):
            raise ValueError("selection_rule must be 'one_se' or 'paired_gain'")
        if self.candidate_set not in ("standard", "extended"):
            raise ValueError("candidate_set must be 'standard' or 'extended'")
        if not 0 <= self.random_state < 2**32:
            raise ValueError("random_state must be between 0 and 2**32 - 1")
        if not isinstance(self.llm_config, dict):
            raise ValueError("llm_config must be a dictionary")
        unknown_llm_keys = set(self.llm_config) - self.SUPPORTED_LLM_CONFIG_KEYS
        if unknown_llm_keys:
            raise ValueError(f"Unknown llm_config keys: {sorted(map(str, unknown_llm_keys))}")
        # Reject non-sequence containers (and ``str``/``bytes``, which are
        # technically iterable but would be iterated character-by-character)
        # before any iteration so that the downstream non-string-entry,
        # empty, and set-diff checks all run on a real list/tuple. Without
        # this guard, ``engines="tabular"`` would silently expand into
        # individual characters and produce a confusing "Unknown engines"
        # error, and ``engines=5`` would raise an unrelated ``TypeError``
        # from ``set(self.engines)``.
        if not isinstance(self.engines, (list, tuple)):
            raise ValueError(
                "engines must be a list or tuple of strings; got "
                f"{type(self.engines).__name__}={self.engines!r}. "
                f"Supported engines: {sorted(self.SUPPORTED_ENGINES)}"
            )

        if not isinstance(self.selection_methods, (list, tuple)):
            raise ValueError(
                "selection_methods must be a list or tuple of strings; got "
                f"{type(self.selection_methods).__name__}={self.selection_methods!r}. "
                f"Supported methods: {sorted(self.SUPPORTED_SELECTION_METHODS)}"
            )

        # Reject non-string entries up front so that the diff against the
        # supported-name sets (and the ``sorted(...)`` used to build the error
        # message) cannot raise an unrelated ``TypeError`` for mixed-type
        # inputs (e.g. ``engines=[None, "spaceship"]``).
        non_string_engines = [e for e in self.engines if not isinstance(e, str)]
        if non_string_engines:
            raise ValueError(
                "engines must contain only strings; got non-string entries: "
                f"{non_string_engines!r}. Supported engines: {sorted(self.SUPPORTED_ENGINES)}"
            )

        # Reject empty collections explicitly. ``engines=None`` is normalized to
        # the default in ``__init__`` / ``set_params``; an explicit empty list
        # would otherwise leave ``fit()`` running zero engines and ``transform()``
        # silently returning the input (modulo NaN/inf cleanup), which is a
        # surprising silent no-op rather than a misconfiguration.
        if not self.engines:
            raise ValueError(
                "engines must contain at least one engine; got an empty sequence. "
                f"Pass ``engines=None`` for the default ['tabular'] or pick from {sorted(self.SUPPORTED_ENGINES)}."
            )

        non_string_methods = [m for m in self.selection_methods if not isinstance(m, str)]
        if non_string_methods:
            raise ValueError(
                "selection_methods must contain only strings; got non-string entries: "
                f"{non_string_methods!r}. Supported methods: {sorted(self.SUPPORTED_SELECTION_METHODS)}"
            )

        if not self.selection_methods:
            raise ValueError(
                "selection_methods must contain at least one method; got an empty sequence. "
                "Pass ``selection_methods=None`` for the default "
                f"['mutual_info', 'importance'] or pick from {sorted(self.SUPPORTED_SELECTION_METHODS)}."
            )

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

        if not isinstance(self.gate_n_jobs, (int, np.integer)) or isinstance(self.gate_n_jobs, bool):
            raise ValueError(f"gate_n_jobs must be an int, got {type(self.gate_n_jobs).__name__}")
        if self.gate_n_jobs == 0 or self.gate_n_jobs < -1:
            raise ValueError(f"gate_n_jobs must be -1 or a positive int, got {self.gate_n_jobs}")

    def _reset_fit_state(self) -> None:
        """Reset all attributes populated during ``fit``/``fit_transform``.

        Called at the start of ``fit`` so that re-fitting (e.g. after changing
        ``engines`` via ``set_params``) cannot leave stale fitted engines, a
        stale selector, or fit-time metadata in place. Mirrors the fit-derived
        attribute initialization in ``__init__``.
        """
        self._engine_instances = {}
        self._selector = None
        self._feature_set = FeatureSet()
        self._is_fitted = False
        self._column_descriptions = {}
        self._task_description = ""
        self.selection_report_ = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        column_descriptions: dict[str, str] | None = None,
        task_description: str = "prediction task",
        target_name: Any | None = None,
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
        target_name : hashable, optional
            Target column label used by leakage checks to identify related
            feature columns. Accepts any column-label type DataFrames support
            (typically ``str``, but also ``int`` or other hashables); the
            leakage helper normalizes labels via ``str(...)`` before matching.
        **fit_params : dict
            Additional parameters

        Returns
        -------
        self : AutoFeatureEngineer
        """
        # Reset all fit-derived state so a refit (e.g. after changing ``engines``
        # or after a previous ``fit_transform`` that built a selector) cannot leak
        # stale engines, a stale selector, or the previous ``_is_fitted`` flag
        # into a subsequent ``transform`` call. Any early exit below (validation
        # error, leakage_guard='raise', engine fit failure) leaves the estimator
        # in a clean, unfitted state rather than a partially-fitted one.
        self._reset_fit_state()
        self._validate_configuration()
        if self.selection_strategy == "predictive" and y is None:
            raise ValueError("The predictive strategy requires a target y")
        if y is not None and self.task != "auto":
            resolve_task(y, self.task)
        split_metadata = any(fit_params.get(name) is not None for name in ("groups", "timestamps", "label_end_times"))
        if split_metadata and self.selection_strategy != "predictive":
            if self.engines != ["llm"] or self.llm_config.get("feature_format") != "plan":
                raise ValueError("Group/time metadata requires predictive selection or the fitted LLM plan path")

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
            if self.selection_strategy == "predictive":
                self.selection_report_ = deepcopy(engine.selection_report_)

            if self.verbose:
                logger.info(f"Fitted {engine_name} engine")

        self._is_fitted = True
        return self

    def _create_engine(self, engine_name: str):
        """Create an engine instance."""
        if engine_name == "tabular":
            if self.selection_strategy == "predictive":
                from featcopilot.engines.predictive import PredictiveTabularEngine

                return PredictiveTabularEngine(
                    max_features=self.max_features,
                    max_candidates=self.max_candidates,
                    task=self.task,
                    estimator=self.selection_estimator,
                    random_state=self.random_state,
                    n_jobs=self.gate_n_jobs,
                    verbose=self.verbose,
                    stability_repeats=self.stability_repeats,
                    screening=self.screening,
                    selection_rule=self.selection_rule,
                    candidate_set=self.candidate_set,
                    include_frequency=self.include_frequency,
                    numeric_clip_quantiles=self.numeric_clip_quantiles,
                    importance_resamples=self.importance_resamples,
                    categorical_encoding=self.categorical_encoding,
                )
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
                model=self.llm_config.get("model", DEFAULT_MODEL),
                max_suggestions=self.llm_config.get("max_suggestions", 20),
                domain=self.llm_config.get("domain"),
                verbose=self.llm_config.get("verbose", self.verbose),
                backend=self.llm_config.get("backend", "copilot"),
                api_key=self.llm_config.get("api_key"),
                api_base=self.llm_config.get("api_base"),
                api_version=self.llm_config.get("api_version"),
                validate_features=self.llm_config.get("validate_features", True),
                temperature=self.llm_config.get("temperature", 0.3),
                timeout=self.llm_config.get("timeout", 60.0),
                max_tokens=self.llm_config.get("max_tokens", 4096),
                enable_text_features=self.llm_config.get("enable_text_features", True),
                keep_text_columns=self.llm_config.get("keep_text_columns", True),
                text_feature_types=self.llm_config.get("text_feature_types"),
                feature_format=self.llm_config.get("feature_format", "code"),
                rules=self.llm_config.get("rules"),
                task=cast(Literal["auto", "classification", "regression"], self.task),
                random_state=self.random_state,
                target_cv=self.llm_config.get("target_cv", 5),
            )
        else:
            raise ValueError(f"Unknown engine: {engine_name}")

    def _transform_fitted_engines(
        self,
        X: pd.DataFrame | np.ndarray,
        transform_params: dict[str, Any],
        training_target: pd.Series | np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Transform with fitted engines, optionally using training-safe semantics."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        result = X.copy()
        dropped_originals: set[Any] = set()

        for engine_name, engine in self._engine_instances.items():
            if training_target is not None and self.selection_strategy == "predictive":
                inference_params = {
                    key: value
                    for key, value in transform_params.items()
                    if key not in {"groups", "timestamps", "label_end_times"}
                }
                transformed = engine.transform(X, **inference_params)
            elif training_target is None:
                transformed = engine.transform(X, **transform_params)
            else:
                transformed = engine._transform_fit_data(X, training_target, **transform_params)
            if engine_name == "llm" and not self.llm_config.get("keep_text_columns", True):
                dropped_originals.update(col for col in X.columns if col not in transformed.columns)

            new_cols = [c for c in transformed.columns if c not in result.columns]
            for col in new_cols:
                result[col] = transformed[col]

            if self.verbose:
                logger.info(f"{engine_name}: Added {len(new_cols)} features")

        if dropped_originals:
            result = result.drop(columns=list(dropped_originals))
        result = result.replace([np.inf, -np.inf], np.nan)

        if self._selector is not None:
            selected_features = self._selector.get_selected_features()
            available = [f for f in selected_features if f in result.columns]
            result = result[available]

        return result

    def transform(self, X: pd.DataFrame | np.ndarray, **transform_params) -> pd.DataFrame:
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

        return self._transform_fitted_engines(X, transform_params)

    def fit_transform(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        column_descriptions: dict[str, str] | None = None,
        task_description: str = "prediction task",
        target_name: Any | None = None,
        apply_selection: bool = True,
        apply_gate: bool = True,
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
        target_name : hashable, optional
            Target column label used by leakage checks to identify related
            feature columns. Accepts any column-label type DataFrames support
            (typically ``str``, but also ``int`` or other hashables); the
            leakage helper normalizes labels via ``str(...)`` before matching.
        apply_selection : bool, default=True
            Whether to apply feature selection
        apply_gate : bool, default=True
            Apply the legacy post-selection validation gate. Disable explicitly
            for selection-only ablation; predictive selection has its own rule.
        **fit_params : dict
            Additional parameters

        Returns
        -------
        X_transformed : DataFrame
            Transformed data with generated features
        """
        if self.selection_strategy == "predictive" and not apply_selection:
            raise ValueError("apply_selection=False requires selection_strategy='legacy'")
        if self.selection_strategy == "predictive" and not apply_gate:
            raise ValueError("apply_gate=False is a legacy-only ablation, not a predictive selection override")
        if (
            self.selection_strategy == "legacy"
            and apply_selection
            and y is not None
            and self.max_features
            and any(fit_params.get(name) is not None for name in ("groups", "timestamps", "label_end_times"))
        ):
            raise ValueError(
                "Legacy selection is not group/time-aware; use apply_selection=False for fitted plans with metadata"
            )
        self.fit(X, y, column_descriptions, task_description, target_name=target_name, **fit_params)
        # Reuse transform-relevant kwargs (e.g. text_columns, related_tables)
        # while allowing supervised engines to cross-fit training-only features.
        result = self._transform_fitted_engines(X, fit_params, training_target=y)

        # Track original features (input columns) vs derived features
        if isinstance(X, np.ndarray):
            original_features = {f"feature_{i}" for i in range(X.shape[1])}
        else:
            original_features = set(X.columns).intersection(result.columns)

        # Apply feature selection if enabled and y is provided
        if self.selection_strategy == "legacy" and apply_selection and y is not None and self.max_features:
            self._selector = FeatureSelector(
                methods=self.selection_methods,
                max_features=self.max_features,
                correlation_threshold=self.correlation_threshold,
                original_features=original_features,
                verbose=self.verbose,
                task=self.task,
            )
            result = self._selector.fit_transform(result, y)

            if self.verbose:
                logger.info(f"Selected {len(self._selector.get_selected_features())} features")

        # Do-no-harm gate: validate derived features help via held-out validation.
        # Only run when selection actually fitted a selector — otherwise this is a
        # no-op for users who explicitly opt out of selection (e.g., max_features=None).
        if apply_gate and apply_selection and y is not None and self._selector is not None:
            original_for_gate = X
            if isinstance(X, pd.DataFrame) and len(original_features) != X.shape[1]:
                original_for_gate = X.loc[:, [col for col in X.columns if col in original_features]]
            result = self._do_no_harm_gate(result, original_for_gate, y, original_features)

        return result

    @staticmethod
    def _encode_for_gate(df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a numeric matrix for the do-no-harm gate.

        Numeric columns pass through unchanged. Scalar non-numeric columns
        (``bool``, ``category``, scalar ``object``) are ordinal-encoded via
        ``pd.Categorical(...).codes``, which uses ``-1`` as the missing-value
        sentinel — preserving missingness through the encoding instead of
        collapsing NaN into a real category. Datetime columns are converted to
        int64 nanoseconds. Object columns whose values are not scalar (e.g.,
        embeddings stored as numpy arrays / lists / dicts) are skipped —
        encoding them would create row-id-like codes that fool the model
        rather than carrying real signal.

        Parameters
        ----------
        df : DataFrame
            Input frame, possibly mixing numeric and non-numeric columns.

        Returns
        -------
        DataFrame
            Numeric matrix with the original index. Empty if no usable columns
            remain.
        """
        numeric_part = df.select_dtypes(include=[np.number, "bool", "boolean"]).copy()
        # Bool dtypes carry signal but may contain pandas NA (nullable BooleanDtype).
        # Cast accordingly so missing values become NaN downstream where the gate's
        # ``.fillna(0)`` can neutralise them, instead of raising or silently dropping.
        for col in numeric_part.columns:
            series = numeric_part[col]
            if isinstance(series.dtype, pd.BooleanDtype):
                # Nullable boolean -> Float64 -> float64 so NA -> NaN.
                numeric_part[col] = series.astype("Float64").astype("float64")
            elif pd.api.types.is_bool_dtype(series):
                # Plain numpy bool: cast to int8 for downstream consistency.
                numeric_part[col] = series.astype(np.int8)

        def _is_scalar_like(value: Any) -> bool:
            """Treat strings/bytes/numbers/booleans/None as scalar-like for encoding."""
            if value is None:
                return True
            return np.isscalar(value) or isinstance(value, (str, bytes))

        encoded_cols: dict[str, pd.Series] = {}
        for col in df.columns:
            if col in numeric_part.columns:
                continue
            series = df[col]
            dtype = series.dtype

            if pd.api.types.is_datetime64_any_dtype(dtype):
                # Cast to int64 nanoseconds, but preserve NaT as NaN so the
                # gate's `.fillna(0)` neutralises missing datetimes instead
                # of letting numpy's int64 NaT sentinel (-9223372036854775808)
                # leak through as a giant artificial signal.
                nan_mask = series.isna()
                # tz-aware datetimes (DatetimeTZDtype) cannot be cast directly to
                # int64; drop the timezone first (convert to UTC, then strip tz)
                # so the gate doesn't always hit the exception path on tz-aware
                # datasets.
                if isinstance(dtype, pd.DatetimeTZDtype):
                    series_naive = series.dt.tz_convert("UTC").dt.tz_localize(None)
                else:
                    series_naive = series
                encoded_cols[col] = series_naive.astype("int64").astype("float64").mask(nan_mask, np.nan)
                continue

            if isinstance(dtype, pd.CategoricalDtype):
                # ``cat.codes`` already uses -1 for missing values — much better
                # than ``astype(str)`` which would coerce NaN into the literal
                # string "nan" and assign it a real code, fabricating signal.
                encoded_cols[col] = series.cat.codes.astype("int64").rename(col)
                continue

            if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
                # Skip object columns containing non-scalar values (arrays, lists,
                # dicts, ...) — ordinal-encoding by str() would produce row-id-like
                # codes that fool the gate model rather than reflecting real signal.
                # Sample up to 50 non-null values rather than just the first one
                # so mixed columns (mostly strings + a few embeddings) still
                # trigger the skip.
                non_null = series.dropna()
                if len(non_null) == 0:
                    continue
                sample = non_null.head(50)
                if not all(_is_scalar_like(v) for v in sample):
                    continue
                # Build a Categorical from the object values directly so NaN is
                # preserved as code -1 (instead of being str()-cast to "nan").
                encoded_cols[col] = pd.Series(
                    pd.Categorical(series).codes.astype("int64"), index=series.index, name=col
                )

        if encoded_cols:
            extra = pd.DataFrame(encoded_cols, index=df.index)
            return pd.concat([numeric_part, extra], axis=1)
        return numeric_part

    def _do_no_harm_gate(
        self,
        X_engineered: pd.DataFrame,
        X_original: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        original_features: set[str],
    ) -> pd.DataFrame:
        """
        Validate that engineered features help using held-out validation.

        Runs 5 stratified (classification) or random (regression) shuffle
        splits at ``test_size=0.2``, fits a fresh RandomForest on each
        train fold, scores on each held-out fold, and compares the
        AVERAGE original-only score to the AVERAGE original+derived score.
        Repeated splits damp single-split variance — the reported
        ``orig_mean`` / ``full_mean`` and the ``threshold`` decision are
        all based on these 5-split means. Cost-wise this trains
        ``2 * 5 = 10`` RandomForests per ``fit_transform()`` call, each
        capped at ``n_estimators=50`` and ``max_depth=10``.

        The original-only baseline is built from ``X_original`` (the truly
        original input), not from a column-subset of ``X_engineered``, so the
        comparison stays meaningful even if selection or downstream steps drop
        or rename original columns in the engineered frame. Non-numeric original
        columns (categorical / string / datetime / bool) are ordinal-encoded so
        signal carried by them is reflected in the baseline (see
        :meth:`_encode_for_gate`).

        Falls back to original features if the mean derived improvement
        doesn't clear the (feature-ratio-scaled) threshold, or — fail-closed —
        if validation raises an exception.

        Parameters
        ----------
        X_engineered : DataFrame
            Data with engineered features (selected).
        X_original : DataFrame or ndarray
            The unmodified original input. Used as the baseline for the
            engineered-vs-original comparison.
        y : Series or ndarray
            Target variable.
        original_features : set[str]
            Names of original (non-derived) features.

        Returns
        -------
        DataFrame
            Either X_engineered if features help, or original-only subset.
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
        from sklearn.utils.multiclass import type_of_target

        y_arr = np.array(y)

        # Normalize X_original to a DataFrame so the baseline matrix is built
        # from the truly original input, independent of the engineered frame.
        if isinstance(X_original, np.ndarray):
            X_original_df = pd.DataFrame(X_original, columns=[f"feature_{i}" for i in range(X_original.shape[1])])
        else:
            X_original_df = X_original

        derived_cols = [c for c in X_engineered.columns if c not in original_features]

        if len(derived_cols) == 0:
            return X_engineered

        orig_cols_in_engineered = [c for c in X_engineered.columns if c in original_features]

        def _conservative_fallback() -> pd.DataFrame:
            """Drop derived features and pin selector to original-only columns.

            Reconstruct the fallback frame from ``X_original_df`` so that any
            originals dropped or renamed by the engineered frame are still
            restored — otherwise ``orig_cols_in_engineered`` could be a
            strict subset (or empty), producing a degenerate frame and pinning
            the selector to too few columns. If ``X_original_df`` doesn't
            carry the named original features (e.g. when the caller passed
            an unnamed ndarray with synthetic ``feature_i`` column names),
            fall back to whatever originals survived in ``X_engineered``.
            The selector is always pinned to the columns actually emitted so
            subsequent ``transform()`` calls stay consistent.
            """
            orig_cols_from_input = [c for c in X_original_df.columns if c in original_features]
            if orig_cols_from_input:
                fallback_df = X_original_df[orig_cols_from_input].copy()
                # Align rows to engineered's index in case engineering changed it.
                if len(fallback_df) == len(X_engineered):
                    fallback_df.index = X_engineered.index
                emitted_cols = orig_cols_from_input
            else:
                fallback_df = X_engineered[orig_cols_in_engineered].copy()
                emitted_cols = orig_cols_in_engineered

            if self._selector is not None:
                self._selector.set_selected_features(emitted_cols)
            return fallback_df

        try:
            # Build numeric-encoded matrices that include ordinal-encoded categorical /
            # string / datetime / bool columns so the baseline isn't artificially weak
            # when signal lives in non-numeric original columns. Performed inside the
            # try block so any encoding/alignment exception (e.g. an unsupported dtype
            # or pandas API surprise) triggers the same conservative fallback as a
            # validation-model failure, instead of bubbling up to fit_transform().
            X_orig_for_gate = self._encode_for_gate(X_original_df)
            X_full_for_gate = self._encode_for_gate(X_engineered)

            if X_orig_for_gate.shape[1] == 0 or X_full_for_gate.shape[1] == 0:
                # Nothing to compare — leave engineered frame untouched.
                return X_engineered

            # Align indices so positional iloc() lines up between the two matrices.
            if not X_orig_for_gate.index.equals(X_full_for_gate.index):
                X_orig_for_gate = X_orig_for_gate.reset_index(drop=True)
                X_full_for_gate = X_full_for_gate.reset_index(drop=True)

            X_orig_for_gate = X_orig_for_gate.replace([np.inf, -np.inf], np.nan).fillna(0)
            X_full_for_gate = X_full_for_gate.replace([np.inf, -np.inf], np.nan).fillna(0)

            # Use sklearn's task inference so bool / nullable Int64 / float-encoded
            # binary labels (e.g. {0.0, 1.0}) are correctly detected as classification.
            try:
                target_type = type_of_target(y_arr)
            except (ValueError, TypeError):
                target_type = "unknown"
            is_classification = target_type in {"binary", "multiclass"}
            if self.task != "auto":
                is_classification = resolve_task(y_arr, self.task) == "classification"
            if is_classification:
                model_cls = RandomForestClassifier
                splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
                split_target = y_arr
            else:
                model_cls = RandomForestRegressor
                splitter = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
                split_target = y_arr

            model_params = {
                "n_estimators": 50,
                "max_depth": 10,
                "random_state": 42,
                # Default to single-threaded to avoid nested-parallelism /
                # CPU oversubscription when AutoFeatureEngineer is wrapped in
                # an outer parallel CV / grid-search. Configurable via
                # ``gate_n_jobs`` on the constructor.
                "n_jobs": self.gate_n_jobs,
            }

            orig_scores = []
            full_scores = []

            for train_idx, val_idx in splitter.split(X_orig_for_gate, split_target):
                # Fit and score on held-out data
                m_orig = model_cls(**model_params)
                m_orig.fit(X_orig_for_gate.iloc[train_idx], y_arr[train_idx])
                orig_scores.append(m_orig.score(X_orig_for_gate.iloc[val_idx], y_arr[val_idx]))

                m_full = model_cls(**model_params)
                m_full.fit(X_full_for_gate.iloc[train_idx], y_arr[train_idx])
                full_scores.append(m_full.score(X_full_for_gate.iloc[val_idx], y_arr[val_idx]))

            orig_mean = np.mean(orig_scores)
            full_mean = np.mean(full_scores)
            improvement = full_mean - orig_mean

            # Scale threshold by feature ratio — more added features = higher bar.
            # Use post-encoding column counts so derived columns dropped by
            # ``_encode_for_gate`` (e.g. non-scalar object columns) don't inflate
            # the ratio and make the gate stricter than intended.
            n_orig = max(X_orig_for_gate.shape[1], 1)
            encoded_derived_cols = [c for c in X_full_for_gate.columns if c not in X_orig_for_gate.columns]
            feature_ratio = len(encoded_derived_cols) / n_orig
            threshold = 0.001 + 0.001 * feature_ratio

            if self.verbose:
                logger.info(
                    f"Do-no-harm gate: orig={orig_mean:.4f}, full={full_mean:.4f}, "
                    f"delta={improvement:+.4f}, threshold={threshold:.4f} "
                    f"({len(encoded_derived_cols)} derived features)"
                )

            # Require clear positive benefit to keep derived features
            if improvement < threshold:
                # Build the fallback first so we can log the actual emitted
                # feature count (which may be larger than
                # ``orig_cols_in_engineered`` when ``_conservative_fallback``
                # restores originals dropped by the engineered output).
                fallback_result = _conservative_fallback()
                if self.verbose:
                    logger.warning(
                        f"Do-no-harm: Derived features not beneficial ({improvement:+.4f}). "
                        f"Falling back to {fallback_result.shape[1]} original features."
                    )
                # Keep the selector but constrain it to original-only columns so
                # subsequent transform() calls (e.g. inside a sklearn Pipeline)
                # emit the same feature set as fit_transform. Use the public
                # setter on BaseSelector instead of mutating private state.
                return fallback_result

        except Exception as e:
            # Fail-closed: a "do-no-harm" gate that silently keeps engineered
            # features on validation failure isn't doing its job. Always log,
            # not just when verbose, and conservatively fall back to original
            # columns + pin the selector to that set.
            fallback_result = _conservative_fallback()
            logger.warning(
                f"Do-no-harm gate failed ({type(e).__name__}: {e}); "
                f"conservatively falling back to {fallback_result.shape[1]} original features."
            )
            return fallback_result

        return X_engineered

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

    def get_fitted_rules(self) -> dict[str, Any] | None:
        """Get serialized fitted LLM rules when plan mode is enabled."""
        engine = self._engine_instances.get("llm")
        if engine is None or self.llm_config.get("feature_format", "code") != "plan":
            return None
        rules = engine.get_fitted_rules()
        if rules is None:
            return None
        if not isinstance(rules, dict):
            raise TypeError("The fitted rule engine returned an invalid serialized plan")
        return rules

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
        return cast(list[dict[str, Any]], llm_engine.suggest_more_features(prompt, n_features))

    @property
    def feature_importances_(self) -> dict[str, float] | None:
        """Get feature importance scores if selection was applied."""
        if self._selector is not None:
            return self._selector.get_feature_scores()
        return None

    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return super().get_params(deep=deep)

    def set_params(self, **params):
        """
        Set parameters for sklearn compatibility.

        Validates parameter keys against the estimator's known parameters
        (raising :class:`ValueError` on unknown keys, matching scikit-learn
        ``BaseEstimator.set_params`` behavior) and then mirrors the defaulting
        performed in ``__init__`` so callers (e.g. sklearn cloning,
        ``GridSearchCV`` parameter grids) can pass ``None`` for
        collection-valued parameters and have it normalized back to the default
        rather than raising during validation.

        The update is atomic: if any provided value fails configuration
        validation, all in-flight mutations are rolled back so the estimator
        is left in its pre-call state rather than a partially-mutated invalid
        one.

        Parameters
        ----------
        **params
            Estimator parameters to update. Each key must already be a
            top-level parameter accepted by ``__init__``.

        Returns
        -------
        AutoFeatureEngineer
            ``self``, to support fluent chaining.

        Raises
        ------
        ValueError
            If ``params`` contains a key that is not a known estimator
            parameter, or if any provided value fails configuration
            validation (see :meth:`_validate_configuration`). On validation
            failure the estimator's parameters are restored to the values
            they held before the call.
        """
        estimator_params = {
            key.removeprefix("selection_estimator__"): value
            for key, value in params.items()
            if key.startswith("selection_estimator__")
        }
        if estimator_params:
            estimator = params.get("selection_estimator", self.selection_estimator)
            if estimator is None:
                raise ValueError("Set selection_estimator before updating its nested parameters")
            updated_estimator = clone(estimator).set_params(**estimator_params)
            params = {key: value for key, value in params.items() if not key.startswith("selection_estimator__")}
            params["selection_estimator"] = updated_estimator
        valid_params = self.get_params(deep=False)
        invalid_keys = sorted(set(params) - set(valid_params))
        if invalid_keys:
            raise ValueError(
                f"Invalid parameter(s) {invalid_keys} for estimator {type(self).__name__}. "
                f"Valid parameters are: {sorted(valid_params)}."
            )

        # Snapshot the current values for every parameter we are about to
        # change (including any whose final value will come from None
        # normalization below) so that a validation failure can roll back to
        # a fully consistent pre-call state.
        snapshot = {key: getattr(self, key) for key in params}

        try:
            for key, value in params.items():
                setattr(self, key, value)
            if self.engines is None:
                self.engines = ["tabular"]
            if self.selection_methods is None:
                self.selection_methods = ["mutual_info", "importance"]
            if self.llm_config is None:
                self.llm_config = {}
            self._validate_configuration()
        except Exception:
            for key, value in snapshot.items():
                setattr(self, key, value)
            raise

        return self
