"""Base classes for feature store integrations."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, Field

from featcopilot.core.feature import FeatureSet


class FeatureStoreConfig(BaseModel):
    """Base configuration for feature stores."""

    name: str = Field(description="Feature store name")
    entity_columns: list[str] = Field(default_factory=list, description="Entity/key columns")
    timestamp_column: Optional[str] = Field(default=None, description="Event timestamp column")
    feature_prefix: str = Field(default="", description="Prefix for feature names")
    tags: dict[str, str] = Field(default_factory=dict, description="Tags/labels for features")


class BaseFeatureStore(ABC):
    """
    Abstract base class for feature store integrations.

    Provides a unified interface for saving and retrieving
    engineered features from various feature stores.

    Parameters
    ----------
    config : FeatureStoreConfig
        Configuration for the feature store

    Examples
    --------
    >>> store = ConcreteFeatureStore(config)
    >>> store.save_features(X_transformed, feature_set, feature_view_name='my_features')
    >>> features = store.get_features(entity_df, feature_names=['feat1', 'feat2'])
    """

    def __init__(self, config: FeatureStoreConfig):
        self.config = config
        self._is_initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize connection to the feature store.

        This should be called before any other operations.
        """
        pass

    @abstractmethod
    def save_features(
        self,
        df: pd.DataFrame,
        feature_set: Optional[FeatureSet] = None,
        feature_view_name: str = "featcopilot_features",
        description: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Save features to the feature store.

        Parameters
        ----------
        df : DataFrame
            DataFrame containing features to save
        feature_set : FeatureSet, optional
            FeatCopilot FeatureSet with metadata
        feature_view_name : str
            Name for the feature view/table
        description : str, optional
            Description of the feature view
        **kwargs
            Additional store-specific options
        """
        pass

    @abstractmethod
    def get_features(
        self,
        entity_df: pd.DataFrame,
        feature_names: list[str],
        feature_view_name: str = "featcopilot_features",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Retrieve features from the feature store.

        Parameters
        ----------
        entity_df : DataFrame
            DataFrame with entity keys and timestamps
        feature_names : list
            Names of features to retrieve
        feature_view_name : str
            Name of the feature view/table
        **kwargs
            Additional store-specific options

        Returns
        -------
        DataFrame
            DataFrame with requested features
        """
        pass

    @abstractmethod
    def list_feature_views(self) -> list[str]:
        """
        List all feature views in the store.

        Returns
        -------
        list
            Names of feature views
        """
        pass

    @abstractmethod
    def get_feature_view_schema(self, feature_view_name: str) -> dict[str, Any]:
        """
        Get schema/metadata for a feature view.

        Parameters
        ----------
        feature_view_name : str
            Name of the feature view

        Returns
        -------
        dict
            Schema information
        """
        pass

    @abstractmethod
    def delete_feature_view(self, feature_view_name: str) -> bool:
        """
        Delete a feature view.

        Parameters
        ----------
        feature_view_name : str
            Name of the feature view to delete

        Returns
        -------
        bool
            Whether deletion was successful
        """
        pass

    def close(self) -> None:
        """Close connection to the feature store."""
        self._is_initialized = False

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
