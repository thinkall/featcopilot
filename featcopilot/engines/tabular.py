"""Tabular feature engineering engine.

Generates polynomial features, interaction terms, and mathematical transformations.
"""

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field
from sklearn.preprocessing import LabelEncoder

from featcopilot.core.base import BaseEngine, EngineConfig
from featcopilot.core.feature import Feature, FeatureOrigin, FeatureSet, FeatureType
from featcopilot.utils.logger import get_logger

logger = get_logger(__name__)


class TabularEngineConfig(EngineConfig):
    """Configuration for tabular feature engine."""

    name: str = "TabularEngine"
    polynomial_degree: int = Field(default=2, ge=1, le=4, description="Max polynomial degree")
    interaction_only: bool = Field(default=False, description="Only interaction terms, no powers")
    include_bias: bool = Field(default=False, description="Include bias/intercept term")
    include_transforms: list[str] = Field(
        default_factory=lambda: ["log", "sqrt", "square"],
        description="Mathematical transformations to apply",
    )
    numeric_only: bool = Field(default=True, description="Only process numeric columns")
    min_unique_values: int = Field(default=5, description="Min unique values for continuous")
    # Categorical encoding settings
    encode_categorical: bool = Field(default=True, description="Auto-encode categorical columns")
    keep_original_categorical: bool = Field(
        default=True, description="Keep original categorical columns (for models that handle them natively)"
    )
    onehot_ratio_threshold: float = Field(default=0.05, description="Max n_unique/n_rows ratio for one-hot encoding")
    target_encode_ratio_threshold: float = Field(
        default=0.5, description="Max n_unique/n_rows ratio for target encoding"
    )
    target_encode_cv: int = Field(default=5, ge=2, description="Cross-fitting folds for target encoding")
    min_samples_per_category: int = Field(default=3, description="Min samples per category to include")


class TabularEngine(BaseEngine):
    """
    Tabular feature engineering engine.

    Generates:
    - Polynomial features (x^2, x^3, etc.)
    - Interaction features (x1 * x2)
    - Mathematical transformations (log, sqrt, etc.)
    - Ratio features (x1 / x2)
    - Difference features (x1 - x2)

    Parameters
    ----------
    polynomial_degree : int, default=2
        Maximum degree for polynomial features
    interaction_only : bool, default=False
        If True, only generate interaction terms, not polynomial powers
    include_transforms : list, default=['log', 'sqrt', 'square']
        Mathematical transformations to apply
    target_encode_cv : int, default=5
        Number of folds used to create leakage-safe training target encodings
    max_features : int, optional
        Maximum number of features to generate

    Examples
    --------
    >>> engine = TabularEngine(polynomial_degree=2, include_transforms=['log', 'sqrt'])
    >>> X_transformed = engine.fit_transform(X)
    """

    # Available transformations
    TRANSFORMATIONS = {
        "log": ("log1p", lambda x: np.log1p(np.abs(x))),
        "log10": ("log10", lambda x: np.log10(np.abs(x) + 1)),
        "sqrt": ("sqrt", lambda x: np.sqrt(np.abs(x))),
        "square": ("sq", lambda x: x**2),
        "cube": ("cb", lambda x: x**3),
        "reciprocal": ("recip", lambda x: 1 / (x + 1e-8)),
        "exp": ("exp", lambda x: np.exp(np.clip(x, -50, 50))),
        "tanh": ("tanh", lambda x: np.tanh(x)),
        "sin": ("sin", lambda x: np.sin(x)),
        "cos": ("cos", lambda x: np.cos(x)),
    }

    def __init__(
        self,
        polynomial_degree: int = 2,
        interaction_only: bool = False,
        include_transforms: list[str] | None = None,
        max_features: int | None = None,
        verbose: bool = False,
        encode_categorical: bool = True,
        onehot_ratio_threshold: float = 0.05,
        target_encode_ratio_threshold: float = 0.5,
        min_samples_per_category: int = 3,
        *,
        target_encode_cv: int = 5,
        **kwargs,
    ):
        config = TabularEngineConfig(
            polynomial_degree=polynomial_degree,
            interaction_only=interaction_only,
            include_transforms=include_transforms if include_transforms is not None else ["log", "sqrt", "square"],
            max_features=max_features,
            verbose=verbose,
            encode_categorical=encode_categorical,
            onehot_ratio_threshold=onehot_ratio_threshold,
            target_encode_ratio_threshold=target_encode_ratio_threshold,
            target_encode_cv=target_encode_cv,
            min_samples_per_category=min_samples_per_category,
            **kwargs,
        )
        super().__init__(config=config)
        self.config: TabularEngineConfig = config
        self._reset()

    def _reset(self) -> None:
        self._is_fitted = False
        self._feature_names = []
        self._feature_metadata = {}
        self._numeric_columns: list[str] = []
        self._feature_set = FeatureSet()
        # Categorical encoding state
        self._categorical_columns: list[str] = []
        self._onehot_columns: list[str] = []
        self._target_encode_columns: list[str] = []
        self._onehot_categories: dict[str, list] = {}
        self._target_encode_maps: dict[str, dict] = {}
        self._target_encode_global_mean: float = 0.0
        self._target_label_encoder: LabelEncoder | None = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        **kwargs,
    ) -> "TabularEngine":
        """
        Fit the engine to identify numeric columns and plan features.

        Parameters
        ----------
        X : DataFrame or ndarray
            Input features
        y : Series or ndarray, optional
            Target variable (used for target encoding of categorical columns)

        Returns
        -------
        self : TabularEngine
        """
        self._reset()
        X = self._validate_input(X)

        # Identify numeric columns
        self._numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()

        # Filter by unique values
        self._numeric_columns = [
            col for col in self._numeric_columns if X[col].nunique() >= self.config.min_unique_values
        ]

        if self.config.verbose:
            logger.info(f"TabularEngine: Found {len(self._numeric_columns)} numeric columns")

        # Handle categorical columns
        if self.config.encode_categorical:
            self._fit_categorical_encoding(X, y)
        if self._target_encode_columns and len(X) < 2:
            raise ValueError("Target encoding requires at least two rows for own-target exclusion")

        # Plan features to generate
        self._plan_features(X)
        self._is_fitted = True

        return self

    def _fit_categorical_encoding(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> None:
        """Fit categorical encoding based on cardinality ratio."""
        self._categorical_columns = []
        self._onehot_columns = []
        self._target_encode_columns = []
        self._onehot_categories = {}
        self._target_encode_maps = {}
        self._target_label_encoder = None  # For string targets

        # Find categorical columns (object or category dtype)
        cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        self._categorical_columns = cat_cols

        if not cat_cols:
            return

        n_rows = len(X)
        y_encoded = None
        if y is not None:
            y_encoded = self._encode_target(y, X.index, fit_label_encoder=True)
            self._target_encode_global_mean = float(y_encoded.mean())

        for col in cat_cols:
            n_unique = X[col].nunique()
            ratio = n_unique / n_rows

            # Count samples per category
            value_counts = X[col].value_counts()
            # Filter categories with enough samples
            valid_categories = value_counts[value_counts >= self.config.min_samples_per_category].index.tolist()

            if len(valid_categories) == 0:
                if self.config.verbose:
                    logger.info(f"TabularEngine: Skipping '{col}' - no categories with enough samples")
                continue

            if ratio <= self.config.onehot_ratio_threshold:
                # One-hot encoding for low cardinality
                self._onehot_columns.append(col)
                self._onehot_categories[col] = valid_categories
                if self.config.verbose:
                    logger.info(
                        f"TabularEngine: One-hot encoding '{col}' "
                        f"({len(valid_categories)} categories, ratio={ratio:.4f})"
                    )

            elif ratio <= self.config.target_encode_ratio_threshold and y_encoded is not None:
                # Target encoding for medium cardinality
                self._target_encode_columns.append(col)
                self._target_encode_maps[col] = self._build_target_encoding_map(
                    X[col],
                    y_encoded,
                    valid_categories=set(valid_categories),
                )
                if self.config.verbose:
                    logger.info(
                        f"TabularEngine: Target encoding '{col}' "
                        f"({len(self._target_encode_maps[col])} categories, ratio={ratio:.4f})"
                    )

            else:
                # High cardinality - likely ID column, skip
                if self.config.verbose:
                    logger.info(
                        f"TabularEngine: Skipping '{col}' - high cardinality " f"({n_unique} unique, ratio={ratio:.4f})"
                    )

    def _encode_target(
        self,
        y: pd.Series | np.ndarray,
        index: pd.Index,
        *,
        fit_label_encoder: bool,
    ) -> pd.Series:
        """Convert a target to a positionally aligned numeric Series."""
        y_array = np.asarray(y)
        if y_array.ndim != 1:
            raise ValueError(f"y must be one-dimensional, got shape {y_array.shape}")
        if len(y_array) != len(index):
            raise ValueError(f"X and y must contain the same number of rows, got {len(index)} and {len(y_array)}")

        y_series = pd.Series(y.array if isinstance(y, pd.Series) else y_array, index=index)
        is_categorical = pd.api.types.is_string_dtype(y_series) or isinstance(y_series.dtype, pd.CategoricalDtype)
        if is_categorical:
            labels = y_series.astype(str)
            if fit_label_encoder or self._target_label_encoder is None:
                self._target_label_encoder = LabelEncoder().fit(labels)
            assert self._target_label_encoder is not None
            encoded = self._target_label_encoder.transform(labels)
            return pd.Series(encoded, index=index, dtype=float)

        return pd.to_numeric(y_series, errors="raise").astype(float)

    def _build_target_encoding_map(
        self,
        categories: pd.Series,
        target: pd.Series,
        valid_categories: set[Any] | None = None,
    ) -> dict[Any, float]:
        """Build category means from positionally aligned category/target values."""
        frame = pd.DataFrame(
            {
                "category": categories.to_numpy(),
                "target": target.to_numpy(),
            }
        )
        grouped = frame.groupby("category", observed=True)["target"].agg(["mean", "count"])
        grouped = grouped[grouped["count"] >= self.config.min_samples_per_category]
        if valid_categories is not None:
            grouped = grouped[grouped.index.isin(valid_categories)]
        return {category: float(mean) for category, mean in grouped["mean"].items()}

    def _cross_fit_target_encodings(self, X: pd.DataFrame, y_encoded: pd.Series) -> dict[str, pd.Series]:
        """Encode training rows from folds that exclude each encoded row."""
        from sklearn.model_selection import KFold

        if not self._target_encode_columns:
            return {}

        n_splits = min(self.config.target_encode_cv, len(X))
        if n_splits < 2:
            raise ValueError("Target encoding requires at least two rows for own-target exclusion")

        splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X))
        encoded_columns: dict[str, pd.Series] = {}

        for col in self._target_encode_columns:
            if col not in X.columns:
                continue

            encoded = pd.Series(index=X.index, dtype=float)
            for train_idx, validation_idx in splits:
                fold_target = y_encoded.iloc[train_idx]
                fold_global_mean = float(fold_target.mean())
                fold_map = self._build_target_encoding_map(X[col].iloc[train_idx], fold_target)
                fold_values = X[col].iloc[validation_idx].map(fold_map).astype(float).fillna(fold_global_mean)
                encoded.iloc[validation_idx] = fold_values.to_numpy()

            encoded_columns[col] = encoded

        return encoded_columns

    def _plan_features(self, X: pd.DataFrame) -> None:
        """Plan which features to generate."""
        self._feature_set = FeatureSet()
        cols = self._numeric_columns

        # 1. Polynomial features (powers)
        if not self.config.interaction_only:
            for col in cols:
                for degree in range(2, self.config.polynomial_degree + 1):
                    feature = Feature(
                        name=f"{col}_pow{degree}",
                        dtype=FeatureType.NUMERIC,
                        origin=FeatureOrigin.POLYNOMIAL,
                        source_columns=[col],
                        transformation=f"power_{degree}",
                        explanation=f"{col} raised to power {degree}",
                        code=f"result = df['{col}'] ** {degree}",
                    )
                    self._feature_set.add(feature)

        # 2. Interaction features (pairwise products)
        for col1, col2 in combinations(cols, 2):
            feature = Feature(
                name=f"{col1}_x_{col2}",
                dtype=FeatureType.NUMERIC,
                origin=FeatureOrigin.INTERACTION,
                source_columns=[col1, col2],
                transformation="multiply",
                explanation=f"Product of {col1} and {col2}",
                code=f"result = df['{col1}'] * df['{col2}']",
            )
            self._feature_set.add(feature)

        # 3. Mathematical transformations
        for col in cols:
            for transform_name in self.config.include_transforms:
                if transform_name in self.TRANSFORMATIONS:
                    suffix, func = self.TRANSFORMATIONS[transform_name]
                    feature = Feature(
                        name=f"{col}_{suffix}",
                        dtype=FeatureType.NUMERIC,
                        origin=FeatureOrigin.POLYNOMIAL,
                        source_columns=[col],
                        transformation=transform_name,
                        explanation=f"{transform_name} transformation of {col}",
                    )
                    self._feature_set.add(feature)

        # 4. Ratio features (for positive columns)
        for col1, col2 in combinations(cols, 2):
            feature = Feature(
                name=f"{col1}_div_{col2}",
                dtype=FeatureType.NUMERIC,
                origin=FeatureOrigin.INTERACTION,
                source_columns=[col1, col2],
                transformation="divide",
                explanation=f"Ratio of {col1} to {col2}",
                code=f"result = df['{col1}'] / (df['{col2}'] + 1e-8)",
            )
            self._feature_set.add(feature)

        # 5. Difference features
        for col1, col2 in combinations(cols, 2):
            feature = Feature(
                name=f"{col1}_minus_{col2}",
                dtype=FeatureType.NUMERIC,
                origin=FeatureOrigin.INTERACTION,
                source_columns=[col1, col2],
                transformation="subtract",
                explanation=f"Difference between {col1} and {col2}",
                code=f"result = df['{col1}'] - df['{col2}']",
            )
            self._feature_set.add(feature)

        if self.config.verbose:
            logger.info(f"TabularEngine: Planned {len(self._feature_set)} features")

    def transform(self, X: pd.DataFrame | np.ndarray, **kwargs) -> pd.DataFrame:
        """
        Generate new features from input data.

        Parameters
        ----------
        X : DataFrame or ndarray
            Input features

        Returns
        -------
        X_transformed : DataFrame
            DataFrame with original and generated features
        """
        if not self._is_fitted:
            raise RuntimeError("Engine must be fitted before transform")

        X = self._validate_input(X)
        return self._transform_validated(X)

    def _transform_fit_data(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Transform training rows with out-of-fold target encodings."""
        if not self._is_fitted:
            raise RuntimeError("Engine must be fitted before transform")

        X = self._validate_input(X)
        if y is None or not self._target_encode_columns:
            return self._transform_validated(X)

        y_encoded = self._encode_target(y, X.index, fit_label_encoder=False)
        target_encoded_values = self._cross_fit_target_encodings(X, y_encoded)
        return self._transform_validated(X, target_encoded_values=target_encoded_values)

    def _transform_validated(
        self,
        X: pd.DataFrame,
        target_encoded_values: dict[str, pd.Series] | None = None,
    ) -> pd.DataFrame:
        """Transform a validated DataFrame."""
        result = X.copy()
        original_columns = set(X.columns)

        # Apply categorical encoding first
        if self.config.encode_categorical:
            result = self._transform_categorical(result, target_encoded_values=target_encoded_values)

        cols = self._numeric_columns
        max_features = self.config.max_features

        # Count categorical features generated so far against max_features
        categorical_features = [c for c in result.columns if c not in original_columns]
        feature_count = len(categorical_features)

        # Generate polynomial features
        if not self.config.interaction_only:
            for col in cols:
                if max_features and feature_count >= max_features:
                    break
                for degree in range(2, self.config.polynomial_degree + 1):
                    name = f"{col}_pow{degree}"
                    result[name] = X[col] ** degree
                    feature_count += 1
                    if max_features and feature_count >= max_features:
                        break

        # Generate interactions
        for col1, col2 in combinations(cols, 2):
            if max_features and feature_count >= max_features:
                break
            result[f"{col1}_x_{col2}"] = X[col1] * X[col2]
            feature_count += 1

        # Apply transformations
        for col in cols:
            if max_features and feature_count >= max_features:
                break
            for transform_name in self.config.include_transforms:
                if transform_name in self.TRANSFORMATIONS:
                    if max_features and feature_count >= max_features:
                        break
                    suffix, func = self.TRANSFORMATIONS[transform_name]
                    result[f"{col}_{suffix}"] = func(X[col])
                    feature_count += 1

        # Generate ratios
        for col1, col2 in combinations(cols, 2):
            if max_features and feature_count >= max_features:
                break
            result[f"{col1}_div_{col2}"] = X[col1] / (X[col2] + 1e-8)
            feature_count += 1

        # Generate differences
        for col1, col2 in combinations(cols, 2):
            if max_features and feature_count >= max_features:
                break
            result[f"{col1}_minus_{col2}"] = X[col1] - X[col2]
            feature_count += 1

        # Handle infinities and NaNs
        result = result.replace([np.inf, -np.inf], np.nan)

        self._feature_names = [c for c in result.columns if c not in X.columns]

        if self.config.verbose:
            logger.info(f"TabularEngine: Generated {len(self._feature_names)} features")

        return result

    def _transform_categorical(
        self,
        X: pd.DataFrame,
        target_encoded_values: dict[str, pd.Series] | None = None,
    ) -> pd.DataFrame:
        """Apply categorical encoding to DataFrame."""
        result = X.copy()
        target_encoded_values = target_encoded_values or {}

        # One-hot encoding
        for col in self._onehot_columns:
            if col not in result.columns:
                continue
            categories = self._onehot_categories.get(col, [])
            for cat in categories:
                col_name = f"{col}_{cat}"
                result[col_name] = (result[col] == cat).astype(int)
            # Add "other" column for rare categories
            col_other = f"{col}_other"
            result[col_other] = (~result[col].isin(categories)).astype(int)

        # Target encoding
        for col in self._target_encode_columns:
            if col not in result.columns:
                continue
            col_name = f"{col}_target_encoded"
            if col in target_encoded_values:
                result[col_name] = target_encoded_values[col]
            else:
                encode_map = self._target_encode_maps.get(col, {})
                result[col_name] = result[col].map(encode_map).astype(float).fillna(self._target_encode_global_mean)

        if not self.config.keep_original_categorical:
            columns_to_drop = [col for col in self._categorical_columns if col in result.columns]
            result = result.drop(columns=columns_to_drop)

        return result

    def get_feature_set(self) -> FeatureSet:
        """Get the feature set with metadata."""
        return self._feature_set
