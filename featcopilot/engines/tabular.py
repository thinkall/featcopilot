"""Tabular feature engineering engine.

Generates polynomial features, interaction terms, and mathematical transformations.
"""

from itertools import combinations
from typing import Optional, Union

import numpy as np
import pandas as pd
from pydantic import Field

from featcopilot.core.base import BaseEngine, EngineConfig
from featcopilot.core.feature import Feature, FeatureOrigin, FeatureSet, FeatureType


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
        include_transforms: Optional[list[str]] = None,
        max_features: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ):
        config = TabularEngineConfig(
            polynomial_degree=polynomial_degree,
            interaction_only=interaction_only,
            include_transforms=include_transforms or ["log", "sqrt", "square"],
            max_features=max_features,
            verbose=verbose,
            **kwargs,
        )
        super().__init__(config=config)
        self.config: TabularEngineConfig = config
        self._numeric_columns: list[str] = []
        self._feature_set = FeatureSet()

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs,
    ) -> "TabularEngine":
        """
        Fit the engine to identify numeric columns and plan features.

        Parameters
        ----------
        X : DataFrame or ndarray
            Input features
        y : Series or ndarray, optional
            Target variable (unused, for API compatibility)

        Returns
        -------
        self : TabularEngine
        """
        X = self._validate_input(X)

        # Identify numeric columns
        self._numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()

        # Filter by unique values
        self._numeric_columns = [
            col for col in self._numeric_columns if X[col].nunique() >= self.config.min_unique_values
        ]

        if self.config.verbose:
            print(f"TabularEngine: Found {len(self._numeric_columns)} numeric columns")

        # Plan features to generate
        self._plan_features(X)
        self._is_fitted = True

        return self

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
            print(f"TabularEngine: Planned {len(self._feature_set)} features")

    def transform(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> pd.DataFrame:
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
        result = X.copy()

        cols = self._numeric_columns
        feature_count = 0
        max_features = self.config.max_features

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
            print(f"TabularEngine: Generated {len(self._feature_names)} features")

        return result

    def get_feature_set(self) -> FeatureSet:
        """Get the feature set with metadata."""
        return self._feature_set
