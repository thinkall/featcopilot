"""Stateless input normalization and train-fitted classification preprocessing."""

from __future__ import annotations

import copy
import re
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from featcopilot.utils.numeric_clipping import NumericQuantileClipper

CategoricalEncoding = Literal["one_hot", "target"]


def validate_categorical_encoding(value: object) -> CategoricalEncoding:
    """Require one of the two explicit model-input categorical policies."""
    if isinstance(value, str):
        if value == "one_hot":
            return "one_hot"
        if value == "target":
            return "target"
    raise ValueError("categorical_encoding must be 'one_hot' or 'target'")


def _category_token(value: object) -> str | float:
    if not pd.api.types.is_scalar(value):
        raise TypeError("Target-encoding categories must be scalar values")
    if pd.isna(value):
        return np.nan
    if isinstance(value, np.generic):
        value = value.item()
    return f"{type(value).__module__}.{type(value).__qualname__}:{value}"


def normalize_categorical_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Normalize scalar category types and missingness without learning statistics."""
    X = validate_classification_frame(X)
    return X.apply(lambda column: column.astype(object).map(_category_token))


def _make_target_encoder(target_type: Literal["binary", "multiclass"], random_state: int) -> BaseEstimator:
    import sklearn

    release = re.match(r"^(\d+)\.(\d+)", sklearn.__version__)
    if release is None or tuple(map(int, release.groups())) < (1, 4):
        raise ImportError(
            "Active categorical_encoding='target' requires scikit-learn >=1.4; "
            f"found {sklearn.__version__}. Use categorical_encoding='one_hot' on older installations."
        )
    from sklearn.preprocessing import TargetEncoder

    if tuple(map(int, release.groups())) >= (1, 9):
        from sklearn.model_selection import StratifiedKFold

        return TargetEncoder(
            target_type=target_type,
            smooth="auto",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        )
    return TargetEncoder(target_type=target_type, smooth="auto", cv=5, shuffle=True, random_state=random_state)


class CrossFittedTargetEncoder(TransformerMixin, BaseEstimator):
    """Fixed, classification-only sklearn target encoding for actual TRAIN rows.

    Parameters
    ----------
    random_state : int, default=42
        Five-fold shuffled stratification seed. Smoothing is fixed to ``"auto"``;
        target type is explicitly binary or multiclass from the supplied labels.

    Notes
    -----
    Each actual categorical fit requires every class at least five times.
    Missing values remain a category, distinct from literal marker-like strings;
    other scalar values are type-prefixed without fitting a vocabulary.
    ``fit_transform`` returns sklearn's cross-fitted matrix directly. ``transform``
    uses the full-TRAIN fitted mappings, so ``fit(X, y).transform(X)`` is NOT a
    replacement for the training matrix. Encoded training matrices are not cached.
    Active use requires sklearn >=1.4, imported lazily. No group/time metadata,
    sample weights, external validation, or regression targets are supported.
    """

    encoder_: BaseEstimator
    n_features_in_: int
    feature_names_in_: np.ndarray
    encoding_report_: dict[str, object]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray, **fit_params: object) -> CrossFittedTargetEncoder:
        """Fit encodings; use fit_transform, not a later transform, to train a learner."""
        self.fit_transform(X, y, **fit_params)
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | np.ndarray, **fit_params: object) -> np.ndarray:
        """Return cross-fitted TRAIN encodings and retain only fitted inference state."""
        for name in list(vars(self)):
            if name.endswith("_"):
                delattr(self, name)
        if fit_params:
            raise TypeError(f"Target encoding supports IID classification without fit metadata: {sorted(fit_params)}")
        if type(self.random_state) is not int or not 0 <= self.random_state < 2**32:
            raise ValueError("Target encoder random_state must be an integer between zero and 2**32 - 1")
        categories = normalize_categorical_frame(X)
        target = np.asarray(y)
        if target.ndim != 1 or len(target) != len(X) or pd.isna(target).any():
            raise ValueError("Target encoding requires nonmissing, one-dimensional, row-aligned classification labels")
        check_classification_targets(target)
        classes, counts = np.unique(target, return_counts=True)
        if len(classes) < 2 or np.min(counts) < 5:
            raise ValueError("Target encoding requires at least five rows of every class in each actual TRAIN fit")
        target_type: Literal["binary", "multiclass"] = "binary" if len(classes) == 2 else "multiclass"
        encoder = _make_target_encoder(target_type, self.random_state)
        output = np.asarray(encoder.fit_transform(categories, target))
        width = X.shape[1] * (len(classes) if target_type == "multiclass" else 1)
        if output.shape != (len(X), width) or output.dtype.kind != "f" or not np.isfinite(output).all():
            raise ValueError("Target encoder returned invalid cross-fitted output shape, dtype, or finite values")
        report: dict[str, object] = {
            "requested": "target",
            "active": True,
            "target_type": target_type,
            "cv": 5,
            "smooth": "auto",
            "shuffle": True,
            "random_state": self.random_state,
            "n_train_rows": len(X),
            "class_counts": counts.tolist(),
            "n_input_features": X.shape[1],
            "output_width": width,
            "output_dtype": str(output.dtype),
            "training_output_bytes": int(output.nbytes),
            "training_representation": "cross_fitted",
        }
        self.encoder_ = encoder
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.encoding_report_ = report
        return output

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply only full-TRAIN fitted maps; unknown categories use their fitted prior."""
        check_is_fitted(self, ["encoder_", "feature_names_in_"])
        categories = normalize_categorical_frame(X)
        if list(categories.columns) != list(self.feature_names_in_):
            raise ValueError("Target encoder input schema/order differs from fit")
        output = np.asarray(self.encoder_.transform(categories))
        if output.shape != (len(X), self.encoding_report_["output_width"]) or not np.isfinite(output).all():
            raise ValueError("Target encoder returned invalid inference output")
        return output

    def get_feature_names_out(self, input_features: np.ndarray | None = None) -> np.ndarray:
        """Return model-input names, not generated-feature definitions."""
        check_is_fitted(self, ["encoder_"])
        return np.asarray(self.encoder_.get_feature_names_out(input_features), dtype=object)

    def get_encoding_report(self) -> dict[str, object]:
        """Return counts/policy/actual output size, never maps, priors, or row encodings."""
        check_is_fitted(self, ["encoding_report_"])
        return copy.deepcopy(self.encoding_report_)


def validate_classification_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Require nonempty DataFrames with unique string feature names."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Classification requires a pandas DataFrame, not positional arrays")
    if X.empty or not X.columns.is_unique or not all(isinstance(column, str) for column in X.columns):
        raise ValueError("Feature schema must contain nonempty rows and unique string columns")
    if any(pd.api.types.is_complex_dtype(dtype) for dtype in X.dtypes):
        raise ValueError("Classification does not support complex-valued features")
    return X


def normalize_model_frame(
    schema: pd.DataFrame, X: pd.DataFrame, *, preserve_categorical_types: bool = False
) -> pd.DataFrame:
    """Normalize rows using only raw schema, without fitting any statistics.

    Parameters
    ----------
    schema : pandas.DataFrame
        Empty, unmodified raw training schema. Column order and numeric/boolean
        roles are authoritative; categorical levels are not used for encoding.
    X : pandas.DataFrame
        Training or inference rows. Numeric infinities become missing; missing
        categories remain distinct from literal strings. Boolean dtype persists.
    preserve_categorical_types : bool, default=False
        Target-only policy: preserve raw categorical scalars and missing values
        for the shared typed encoder and feature generation. Retain a supplied
        categorical dtype, otherwise use object dtype so numeric-looking category
        values cannot become numeric features. Never cast to fitted category
        levels, which would erase unseen values. False preserves legacy OHE
        string normalization exactly.

    Returns
    -------
    pandas.DataFrame
        Independent normalized rows, with no imputation or learned vocabulary.
    """
    X = validate_classification_frame(X)
    if not schema.columns.equals(X.columns):
        raise ValueError("Input feature schema/order differs from fit")
    normalized = X.copy()
    for column in schema:
        if preserve_categorical_types and (
            isinstance(schema[column].dtype, pd.CategoricalDtype)
            or not (pd.api.types.is_bool_dtype(schema[column]) or pd.api.types.is_numeric_dtype(schema[column]))
        ):
            normalized[column] = (
                X[column].copy() if isinstance(X[column].dtype, pd.CategoricalDtype) else X[column].astype(object)
            )
        elif pd.api.types.is_bool_dtype(schema[column]):
            if not pd.api.types.is_bool_dtype(X[column]):
                raise TypeError(f"Boolean feature {column!r} must remain boolean")
        elif pd.api.types.is_numeric_dtype(schema[column]):
            if not pd.api.types.is_numeric_dtype(X[column]) or pd.api.types.is_bool_dtype(X[column]):
                raise TypeError(f"Numeric feature {column!r} must remain numeric")
            normalized[column] = X[column].astype(float).replace([np.inf, -np.inf], np.nan)
        else:
            normalized[column] = (
                X[column].astype(object).map(lambda value: "missing:" if pd.isna(value) else f"value:{value}")
            )
    return normalized


def make_classification_pipeline(
    train: pd.DataFrame,
    estimator: BaseEstimator,
    *,
    numeric_clip_quantiles: tuple[float, float] = (0.0, 1.0),
    categorical_encoding: CategoricalEncoding = "one_hot",
    random_state: int = 42,
) -> Pipeline:
    """Build matched numeric/OHE preprocessing around an owned classifier clone.

    Parameters
    ----------
    train : pandas.DataFrame
        Post-generation training schema. No statistics are read here.
    estimator : sklearn estimator
        Unfitted classifier owned by the caller, not an external fitted model.
    numeric_clip_quantiles : tuple[float, float], default=(0.0, 1.0)
        Numeric bounds fitted by the pipeline, after feature generation.
    categorical_encoding : {"one_hot", "target"}, default="one_hot"
        Target encoding is fixed cross-fitted IID classification. Empty categorical
        blocks retain the numeric-only path, with no target-encoder import or fit.
    random_state : int, default=42
        Target-encoder fold seed; ignored by the unchanged one-hot path.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Numeric clipping, median imputation/scaling and categorical OHE, then
        the supplied classifier. This is not a native-categorical representation.
    """
    encoding = validate_categorical_encoding(categorical_encoding)
    numeric = train.select_dtypes(include=np.number).columns.tolist()
    numeric = [
        column
        for column in train
        if column in numeric
        or (
            pd.api.types.is_bool_dtype(train[column])
            and not (encoding == "target" and isinstance(train[column].dtype, pd.CategoricalDtype))
        )
    ]
    categorical = [column for column in train if column not in numeric]
    categorical_transformer = (
        CrossFittedTargetEncoder(random_state=random_state)
        if encoding == "target" and categorical
        else OneHotEncoder(handle_unknown="ignore")
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            ("categorical", categorical_transformer, categorical),
        ],
        sparse_threshold=0,
    )
    return make_pipeline(NumericQuantileClipper(quantiles=numeric_clip_quantiles), preprocessor, estimator)
