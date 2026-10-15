"""Train-fitted numeric model-input clipping; never a feature-generation rule."""

from numbers import Real
from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from featcopilot.utils.logger import get_logger

logger = get_logger(__name__)

NUMERIC_GUARD_LIMIT = 1e30
NUMERIC_GUARD_POLICY = "finite_clip_1e30_nonfinite_to_nan_v1"


class NumericClipReport(TypedDict):
    """Model-input clipping diagnostics excluding numerical bound values."""

    policy: str
    quantiles: list[float]
    numeric_guard: str
    guard_limit: float
    fit_rows: int
    numeric_columns: list[str]
    boolean_columns: list[str]
    constant_columns: list[str]
    all_missing_columns: list[str]
    missing_policy: str
    all_missing_policy: str


def validate_clip_quantiles(value: object) -> tuple[float, float] | None:
    """
    Validate an optional, canonical numeric model-input clipping policy.

    Parameters
    ----------
    value : object
        None disables the policy. Otherwise supply a two-element tuple or list
        of finite real quantiles with 0 <= low < high <= 1. Booleans and strings
        are rejected rather than coerced.

    Returns
    -------
    tuple[float, float] or None
        Canonical quantiles, without modifying the supplied object.

    Raises
    ------
    ValueError
        If the policy is not a valid quantile pair.
    """
    if value is None:
        return None
    return _validate_pair(value, allow_equal=False)


def _validate_pair(value: object, *, allow_equal: bool) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("numeric clip quantiles must be a pair of finite real numbers")
    if any(isinstance(item, (bool, np.bool_)) or not isinstance(item, Real) for item in value):
        raise ValueError("numeric clip quantiles must be real numbers, not booleans or strings")
    lower, upper = float(value[0]), float(value[1])
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("numeric clip quantiles must be finite")
    if not 0 <= lower <= upper <= 1 or (not allow_equal and lower == upper):
        relation = "<=" if allow_equal else "<"
        raise ValueError(f"numeric clip quantiles must satisfy 0 <= lower {relation} upper <= 1")
    return lower, upper


def guard_numeric_values(values: np.ndarray) -> np.ndarray:
    """
    Apply the existing finite-value guard without learning statistics.

    Parameters
    ----------
    values : ndarray
        Real numeric values.

    Returns
    -------
    ndarray
        A new array: nonfinite values become NaN and finite values are clipped
        to +/-1e30. Missingness is not imputed.
    """
    return np.clip(np.where(np.isfinite(values), values, np.nan), -NUMERIC_GUARD_LIMIT, NUMERIC_GUARD_LIMIT)


def finite_quantile_bounds(values: np.ndarray, quantiles: tuple[float, float]) -> tuple[float, float] | None:
    """
    Estimate finite-data quantile bounds using overflow-resistant scaling.

    Parameters
    ----------
    values : ndarray
        One-dimensional real numeric training values. This kernel deliberately
        does NOT impose the model-input guard, preserving fitted-rule clip
        semantics. Call ``guard_numeric_values`` first for model-input clipping.
    quantiles : tuple[float, float]
        Ordered quantiles in [0, 1]. Equal bounds remain supported here for
        existing fitted-rule clips, unlike the model-input policy validator.
        Endpoints zero/one use exact finite min/max; when both are endpoints,
        no quantile interpolation or magnitude normalization is performed.

    Returns
    -------
    tuple[float, float] or None
        Finite lower/upper bounds, or None when no finite training values exist.
        No fabricated zero statistics are returned.

    Raises
    ------
    ValueError
        If values or quantiles are invalid.
    """
    lower, upper = _validate_pair(quantiles, allow_equal=True)
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "biuf":
        raise ValueError("Finite quantile bounds require one-dimensional real numeric values")
    array = np.asarray(array, dtype=float)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return None
    if lower in (0.0, 1.0) and upper in (0.0, 1.0):
        minimum, maximum = float(finite.min()), float(finite.max())
        return minimum if lower == 0.0 else maximum, minimum if upper == 0.0 else maximum
    magnitude = float(np.max(np.abs(finite))) or 1.0
    bounds = np.quantile(finite / magnitude, [lower, upper]) * magnitude
    if lower == 0.0:
        bounds[0] = finite.min()
    if upper == 1.0:
        bounds[1] = finite.max()
    return float(bounds[0]), float(bounds[1])


class NumericQuantileClipper(TransformerMixin, BaseEstimator):
    """
    Clip model-input numeric columns using frozen training quantiles.

    Parameters
    ----------
    quantiles : tuple[float, float], default=(0.0, 1.0)
        Finite pair 0 <= low < high <= 1. The default uses training min/max
        in native units, not MinMaxScaler. Experiments should pass their
        declared pair explicitly. Constructor parameters retain their identity
        for sklearn cloning; validated float quantiles are stored at fit time.

    Attributes
    ----------
    bounds_ : dict[str, tuple[float, float] or None]
        Frozen bounds for fit-time numeric, non-boolean columns. None explicitly
        marks columns with no finite training values.
    numeric_columns_ : tuple[str, ...]
        Columns selected by dtype during fit; booleans are excluded.
    all_missing_columns_ : tuple[str, ...]
        Columns with no finite training observations after the finite guard.
    constant_columns_ : tuple[str, ...]
        Columns constant on their finite guarded training observations.
    quantiles_ : tuple[float, float]
        Canonical fitted quantile pair.

    Notes
    -----
    At both fit and transform, nonfinite numeric inputs become NaN and finite
    numeric values are guarded at +/-1e30 BEFORE quantile fitting/application.
    Quantiles are fitted only on finite training observations. NaNs remain for
    a downstream imputer or native missing-value learner. All-missing training
    columns have guard-only behavior at inference, not invented quantile bounds.
    With (0, 1), finite float64 training values within the guard are unchanged
    bitwise, including signed zeros: only values strictly outside fitted bounds
    are assigned a boundary value. Values beyond +/-1e30 are still changed by
    the shared guard, independently of the quantile policy.

    Input must be a DataFrame with unique string columns. Transform requires
    the fitted column names/order, but permits new categorical levels/dtypes.
    It returns a copy with the same index/schema and untouched nonnumeric and
    fit-time boolean columns. Numeric dtypes may become float. This transformer
    belongs after candidate generation and before model imputation/scaling;
    it does not mutate raw user data or generated feature definitions.
    """

    def __init__(self, quantiles: tuple[float, float] = (0.0, 1.0)):
        validated = validate_clip_quantiles(quantiles)
        if validated is None:
            raise ValueError("NumericQuantileClipper requires an enabled quantile pair")
        self.quantiles = quantiles
        self._is_fitted = False

    @staticmethod
    def _validate_frame(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("NumericQuantileClipper requires a pandas DataFrame")
        if not X.columns.is_unique or not all(isinstance(column, str) for column in X.columns):
            raise ValueError("NumericQuantileClipper requires unique string column names")
        if any(pd.api.types.is_complex_dtype(dtype) for dtype in X.dtypes):
            raise ValueError("NumericQuantileClipper does not support complex-valued columns")

    @staticmethod
    def _column_values(column: pd.Series) -> np.ndarray:
        return guard_numeric_values(column.to_numpy(dtype=float, na_value=np.nan))

    def fit(self, X: pd.DataFrame, y: object = None) -> "NumericQuantileClipper":
        """
        Fit bounds using only the supplied training rows.

        Parameters
        ----------
        X : DataFrame
            Nonempty training frame; no target information is used.
        y : object, optional
            Ignored, for sklearn pipeline compatibility.

        Returns
        -------
        NumericQuantileClipper
            This fitted transformer. Failed refits leave it unfitted.
        """
        self._is_fitted = False
        for name in (
            "feature_names_in_",
            "n_features_in_",
            "numeric_columns_",
            "boolean_columns_",
            "all_missing_columns_",
            "constant_columns_",
            "quantiles_",
            "bounds_",
            "n_samples_seen_",
        ):
            self.__dict__.pop(name, None)
        self._validate_frame(X)
        if not len(X):
            raise ValueError("NumericQuantileClipper requires nonempty training rows")
        quantiles = validate_clip_quantiles(self.quantiles)
        if quantiles is None:
            raise ValueError("NumericQuantileClipper requires an enabled quantile pair")
        numeric = tuple(
            column
            for column in X.columns
            if pd.api.types.is_numeric_dtype(X[column].dtype) and not pd.api.types.is_bool_dtype(X[column].dtype)
        )
        bounds: dict[str, tuple[float, float] | None] = {}
        missing: list[str] = []
        constant: list[str] = []
        for column in numeric:
            values = self._column_values(X[column])
            bounds[column] = finite_quantile_bounds(values, quantiles)
            finite = values[np.isfinite(values)]
            if not len(finite):
                missing.append(column)
            elif np.all(finite == finite[0]):
                constant.append(column)
        self.feature_names_in_: np.ndarray = np.asarray(X.columns, dtype=object).copy()
        self.n_features_in_ = X.shape[1]
        self.numeric_columns_ = numeric
        self.boolean_columns_ = tuple(column for column in X.columns if pd.api.types.is_bool_dtype(X[column].dtype))
        self.all_missing_columns_ = tuple(missing)
        self.constant_columns_ = tuple(constant)
        self.quantiles_ = quantiles
        self.bounds_ = bounds
        self.n_samples_seen_ = len(X)
        self._is_fitted = True
        if missing:
            logger.warning(
                "NumericQuantileClipper: %d columns have no finite training observations; using guard-only inference",
                len(missing),
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the frozen guard/bounds without fitting on this batch.

        Parameters
        ----------
        X : DataFrame
            Input rows in the fitted column order; empty batches are supported.

        Returns
        -------
        DataFrame
            A new frame with unchanged index, column names/order and passthrough
            categorical/boolean values. Numeric NaNs remain NaN.
        """
        if not self._is_fitted:
            raise RuntimeError("NumericQuantileClipper must be fitted before transform")
        self._validate_frame(X)
        if list(X.columns) != self.feature_names_in_.tolist():
            raise ValueError("NumericQuantileClipper input schema differs from fit")
        result = X.copy()
        for column in self.numeric_columns_:
            if len(X) and (
                not pd.api.types.is_numeric_dtype(X[column].dtype) or pd.api.types.is_bool_dtype(X[column].dtype)
            ):
                raise TypeError(f"Numeric model-input column {column!r} must remain non-boolean numeric")
            values = self._column_values(X[column])
            bounds = self.bounds_[column]
            if bounds is not None:
                values[values < bounds[0]] = bounds[0]
                values[values > bounds[1]] = bounds[1]
            result[column] = values
        return result

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return unchanged fitted feature names, validating any supplied schema."""
        if not self._is_fitted:
            raise RuntimeError("NumericQuantileClipper must be fitted before reporting feature names")
        if input_features is not None and not np.array_equal(np.asarray(input_features), self.feature_names_in_):
            raise ValueError("NumericQuantileClipper input feature names differ from fit")
        return np.array(self.feature_names_in_, dtype=object, copy=True)

    def get_clip_report(self) -> NumericClipReport:
        """Return fitted policy/schema diagnostics without numerical bound values."""
        if not self._is_fitted:
            raise RuntimeError("NumericQuantileClipper must be fitted before reporting its policy")
        return {
            "policy": "train_quantile_clip_v1",
            "quantiles": list(self.quantiles_),
            "numeric_guard": NUMERIC_GUARD_POLICY,
            "guard_limit": NUMERIC_GUARD_LIMIT,
            "fit_rows": self.n_samples_seen_,
            "numeric_columns": list(self.numeric_columns_),
            "boolean_columns": list(self.boolean_columns_),
            "constant_columns": list(self.constant_columns_),
            "all_missing_columns": list(self.all_missing_columns_),
            "missing_policy": "preserve_nan",
            "all_missing_policy": "guard_only_no_fitted_bounds",
        }
