"""Relational feature engineering engine.

Generates aggregation features from related tables (inspired by Featuretools).
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
from pydantic import Field

from featcopilot.core.base import BaseEngine, EngineConfig
from featcopilot.core.feature import FeatureSet


class RelationalEngineConfig(EngineConfig):
    """Configuration for relational feature engine."""

    name: str = "RelationalEngine"
    aggregation_functions: list[str] = Field(
        default_factory=lambda: ["mean", "sum", "min", "max", "count", "std"],
        description="Aggregation functions to apply",
    )
    max_depth: int = Field(default=2, ge=1, le=4, description="Max depth for feature synthesis")
    include_time_based: bool = Field(default=True, description="Include time-based aggregations")


class RelationalEngine(BaseEngine):
    """
    Relational feature engineering engine.

    Generates features from related tables using aggregation operations,
    similar to Featuretools' Deep Feature Synthesis but with:
    - Simpler API
    - LLM integration capabilities
    - Better performance

    Parameters
    ----------
    aggregation_functions : list
        Aggregation functions to use (mean, sum, min, max, count, std, etc.)
    max_depth : int, default=2
        Maximum depth for feature synthesis

    Examples
    --------
    >>> engine = RelationalEngine()
    >>> engine.add_relationship('orders', 'customers', 'customer_id')
    >>> X_features = engine.fit_transform(orders_df, related_tables={'customers': customers_df})
    """

    AGGREGATION_FUNCTIONS = {
        "mean": np.mean,
        "sum": np.sum,
        "min": np.min,
        "max": np.max,
        "count": len,
        "std": np.std,
        "median": np.median,
        "first": lambda x: x.iloc[0] if len(x) > 0 else np.nan,
        "last": lambda x: x.iloc[-1] if len(x) > 0 else np.nan,
        "nunique": lambda x: len(set(x)),
    }

    def __init__(
        self,
        aggregation_functions: Optional[list[str]] = None,
        max_depth: int = 2,
        max_features: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ):
        config = RelationalEngineConfig(
            aggregation_functions=aggregation_functions or ["mean", "sum", "count", "max", "min"],
            max_depth=max_depth,
            max_features=max_features,
            verbose=verbose,
            **kwargs,
        )
        super().__init__(config=config)
        self.config: RelationalEngineConfig = config
        self._relationships: list[dict[str, str]] = []
        self._feature_set = FeatureSet()

    def add_relationship(
        self, child_table: str, parent_table: str, key_column: str, parent_key: Optional[str] = None
    ) -> "RelationalEngine":
        """
        Define a relationship between tables.

        Parameters
        ----------
        child_table : str
            Name of child table (many side)
        parent_table : str
            Name of parent table (one side)
        key_column : str
            Foreign key column in child table
        parent_key : str, optional
            Primary key column in parent table (defaults to key_column)

        Returns
        -------
        self : RelationalEngine
        """
        self._relationships.append(
            {
                "child": child_table,
                "parent": parent_table,
                "child_key": key_column,
                "parent_key": parent_key or key_column,
            }
        )
        return self

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        related_tables: Optional[dict[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> "RelationalEngine":
        """
        Fit the engine to the data.

        Parameters
        ----------
        X : DataFrame
            Primary table
        y : Series, optional
            Target variable
        related_tables : dict, optional
            Dictionary of related tables {name: DataFrame}

        Returns
        -------
        self : RelationalEngine
        """
        X = self._validate_input(X)
        self._related_tables = related_tables or {}
        self._primary_columns = X.columns.tolist()

        if self.config.verbose:
            print(f"RelationalEngine: {len(self._relationships)} relationships defined")

        self._is_fitted = True
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        related_tables: Optional[dict[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate aggregation features.

        Parameters
        ----------
        X : DataFrame
            Primary table
        related_tables : dict, optional
            Dictionary of related tables

        Returns
        -------
        X_features : DataFrame
            DataFrame with aggregated features
        """
        if not self._is_fitted:
            raise RuntimeError("Engine must be fitted before transform")

        X = self._validate_input(X)
        related_tables = related_tables or self._related_tables
        result = X.copy()

        # Generate features from relationships
        for rel in self._relationships:
            if rel["parent"] not in related_tables:
                continue

            parent_df = related_tables[rel["parent"]]
            features = self._aggregate_from_relationship(
                X, parent_df, rel["child_key"], rel["parent_key"], rel["parent"]
            )
            result = pd.concat([result, features], axis=1)

        # Generate group-by aggregations within the primary table
        result = self._add_self_aggregations(result)

        self._feature_names = [c for c in result.columns if c not in X.columns]

        if self.config.verbose:
            print(f"RelationalEngine: Generated {len(self._feature_names)} features")

        return result

    def _aggregate_from_relationship(
        self,
        child_df: pd.DataFrame,
        parent_df: pd.DataFrame,
        child_key: str,
        parent_key: str,
        parent_name: str,
    ) -> pd.DataFrame:
        """Generate aggregation features from a parent table."""
        features = pd.DataFrame(index=child_df.index)

        # Get numeric columns from parent
        numeric_cols = parent_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != parent_key]

        # Merge and aggregate
        for col in numeric_cols:
            for agg_name in self.config.aggregation_functions:
                if agg_name not in self.AGGREGATION_FUNCTIONS:
                    continue

                feature_name = f"{parent_name}_{col}_{agg_name}"

                # Group by parent key and aggregate
                agg_values = parent_df.groupby(parent_key)[col].agg(agg_name).to_dict()

                # Map to child table
                if child_key in child_df.columns:
                    features[feature_name] = child_df[child_key].map(agg_values)

        return features

    def _add_self_aggregations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add aggregations within the same table (e.g., by category columns)."""
        result = df.copy()

        # Find categorical columns that could be used for grouping
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Limit to avoid explosion
        cat_cols = cat_cols[:3]
        num_cols = num_cols[:5]

        for cat_col in cat_cols:
            for num_col in num_cols:
                for agg_name in ["mean", "count"]:  # Limited aggregations for self
                    if agg_name not in self.AGGREGATION_FUNCTIONS:
                        continue

                    feature_name = f"{num_col}_by_{cat_col}_{agg_name}"
                    agg_values = df.groupby(cat_col)[num_col].transform(agg_name)
                    result[feature_name] = agg_values

        return result

    def get_feature_set(self) -> FeatureSet:
        """Get the feature set with metadata."""
        return self._feature_set
