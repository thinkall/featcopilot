"""Scikit-learn compatible feature engineering transformers.

Provides drop-in sklearn transformers for feature engineering pipelines.
"""

import warnings
from typing import Any

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
    gate_n_jobs : int, default=1
        Number of parallel jobs for the do-no-harm gate's internal validation
        models (RandomForest). Defaults to ``1`` (sequential) to avoid CPU
        oversubscription when ``AutoFeatureEngineer`` is wrapped in an outer
        parallel CV / grid-search. Set to ``-1`` to use all cores when running
        standalone.

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
    ...     llm_config={'model': 'gpt-5.2', 'enable_semantic': True}
    ... )
    >>> X_transformed = engineer.fit_transform(X, y, target_name='label')
    """

    SUPPORTED_ENGINES = {"tabular", "timeseries", "relational", "text", "llm"}
    SUPPORTED_SELECTION_METHODS = {"mutual_info", "importance", "f_test", "chi2", "correlation", "xgboost"}
    SUPPORTED_LEAKAGE_GUARDS = {"off", "warn", "raise"}

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

        self._validate_configuration()

        self._engine_instances: dict[str, Any] = {}
        self._selector: FeatureSelector | None = None
        self._feature_set = FeatureSet()
        self._is_fitted = False
        self._column_descriptions: dict[str, str] = {}
        self._task_description: str = ""

    def _validate_configuration(self) -> None:
        """Validate user-facing configuration early."""
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
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        column_descriptions: dict[str, str] | None = None,
        task_description: str = "prediction task",
        target_name: Any | None = None,
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
        target_name : hashable, optional
            Target column label used by leakage checks to identify related
            feature columns. Accepts any column-label type DataFrames support
            (typically ``str``, but also ``int`` or other hashables); the
            leakage helper normalizes labels via ``str(...)`` before matching.
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

        # Do-no-harm gate: validate derived features help via held-out validation.
        # Only run when selection actually fitted a selector — otherwise this is a
        # no-op for users who explicitly opt out of selection (e.g., max_features=None).
        if apply_selection and y is not None and self._selector is not None:
            result = self._do_no_harm_gate(result, X, y, original_features)

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

        Holds out 20% of the data, fits a fresh model on the remaining 80%,
        and compares performance with and without derived features. This avoids
        the bias from features being selected on the same data.

        The original-only baseline is built from ``X_original`` (the truly
        original input), not from a column-subset of ``X_engineered``, so the
        comparison stays meaningful even if selection or downstream steps drop
        or rename original columns in the engineered frame. Non-numeric original
        columns (categorical / string / datetime / bool) are ordinal-encoded so
        signal carried by them is reflected in the baseline (see
        :meth:`_encode_for_gate`).

        Falls back to original features if derived features don't show
        clear benefit on the held-out set, or — fail-closed — if validation
        raises an exception.

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
                if self.verbose:
                    logger.warning(
                        f"Do-no-harm: Derived features not beneficial ({improvement:+.4f}). "
                        f"Falling back to {len(orig_cols_in_engineered)} original features."
                    )
                # Keep the selector but constrain it to original-only columns so
                # subsequent transform() calls (e.g. inside a sklearn Pipeline)
                # emit the same feature set as fit_transform. Use the public
                # setter on BaseSelector instead of mutating private state.
                return _conservative_fallback()

        except Exception as e:
            # Fail-closed: a "do-no-harm" gate that silently keeps engineered
            # features on validation failure isn't doing its job. Always log,
            # not just when verbose, and conservatively fall back to original
            # columns + pin the selector to that set.
            logger.warning(
                f"Do-no-harm gate failed ({type(e).__name__}: {e}); "
                f"conservatively falling back to {len(orig_cols_in_engineered)} original features."
            )
            return _conservative_fallback()

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
    def feature_importances_(self) -> dict[str, float] | None:
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
            "gate_n_jobs": self.gate_n_jobs,
        }

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
        valid_params = self.get_params(deep=True)
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
