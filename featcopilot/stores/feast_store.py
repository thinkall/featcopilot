"""Feast feature store integration.

Provides integration with Feast (https://feast.dev) for saving
and retrieving engineered features.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from pydantic import Field

from featcopilot.core.feature import FeatureSet, FeatureType
from featcopilot.stores.base import BaseFeatureStore, FeatureStoreConfig
from featcopilot.utils.logger import get_logger

logger = get_logger(__name__)


class FeastConfig(FeatureStoreConfig):
    """Configuration for Feast feature store."""

    name: str = "feast"
    repo_path: Optional[str] = Field(default=None, description="Path to Feast repo directory")
    project_name: str = Field(default="featcopilot", description="Feast project name")
    provider: str = Field(default="local", description="Feast provider (local, gcp, aws)")
    online_store_type: str = Field(default="sqlite", description="Online store type")
    offline_store_type: str = Field(default="file", description="Offline store type")
    ttl_days: int = Field(default=365, description="Feature TTL in days")
    auto_materialize: bool = Field(default=True, description="Auto-materialize to online store")


class FeastFeatureStore(BaseFeatureStore):
    """
    Feast feature store integration.

    Enables saving FeatCopilot-generated features to Feast for:
    - Historical feature retrieval (training)
    - Online feature serving (inference)
    - Feature discovery and reuse

    Parameters
    ----------
    repo_path : str, optional
        Path to Feast repository. If None, creates a temporary repo.
    project_name : str, default='featcopilot'
        Name of the Feast project
    entity_columns : list, optional
        Columns to use as entity keys
    timestamp_column : str, optional
        Column containing event timestamps
    provider : str, default='local'
        Feast provider (local, gcp, aws)
    auto_materialize : bool, default=True
        Whether to automatically materialize features to online store

    Examples
    --------
    Basic usage with FeatCopilot:

    >>> from featcopilot import AutoFeatureEngineer
    >>> from featcopilot.stores import FeastFeatureStore
    >>>
    >>> # Generate features
    >>> engineer = AutoFeatureEngineer(engines=['tabular'])
    >>> X_transformed = engineer.fit_transform(X, y)
    >>>
    >>> # Save to Feast
    >>> store = FeastFeatureStore(
    ...     repo_path='./feature_repo',
    ...     entity_columns=['customer_id'],
    ...     timestamp_column='event_timestamp'
    ... )
    >>> store.initialize()
    >>> store.save_features(
    ...     X_transformed,
    ...     feature_view_name='customer_features',
    ...     description='Customer churn prediction features'
    ... )

    Retrieve features for training:

    >>> entity_df = pd.DataFrame({
    ...     'customer_id': [1, 2, 3],
    ...     'event_timestamp': [datetime.now()] * 3
    ... })
    >>> features = store.get_features(
    ...     entity_df,
    ...     feature_names=['age_income_ratio', 'tenure_months'],
    ...     feature_view_name='customer_features'
    ... )
    """

    def __init__(
        self,
        repo_path: Optional[str] = None,
        project_name: str = "featcopilot",
        entity_columns: Optional[list[str]] = None,
        timestamp_column: Optional[str] = None,
        provider: str = "local",
        online_store_type: str = "sqlite",
        offline_store_type: str = "file",
        ttl_days: int = 365,
        auto_materialize: bool = True,
        **kwargs,
    ):
        config = FeastConfig(
            repo_path=repo_path,
            project_name=project_name,
            entity_columns=entity_columns or [],
            timestamp_column=timestamp_column,
            provider=provider,
            online_store_type=online_store_type,
            offline_store_type=offline_store_type,
            ttl_days=ttl_days,
            auto_materialize=auto_materialize,
            **kwargs,
        )
        super().__init__(config)
        self.config: FeastConfig = config
        self._feast_store = None
        self._repo_path: Optional[Path] = None
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self._feature_views: dict[str, Any] = {}
        self._entities: dict[str, Any] = {}

    def initialize(self) -> None:
        """
        Initialize the Feast feature store.

        Creates the Feast repo if it doesn't exist and initializes
        the FeatureStore object.
        """
        try:
            from feast import FeatureStore
        except ImportError as err:
            raise ImportError(
                "Feast is not installed. Install with: pip install feast\n"
                "Or install FeatCopilot with Feast support: pip install featcopilot[feast]"
            ) from err

        # Set up repo path
        if self.config.repo_path:
            self._repo_path = Path(self.config.repo_path)
            self._repo_path.mkdir(parents=True, exist_ok=True)
        else:
            self._temp_dir = tempfile.TemporaryDirectory()
            self._repo_path = Path(self._temp_dir.name)

        # Create feature_store.yaml if not exists
        config_path = self._repo_path / "feature_store.yaml"
        if not config_path.exists():
            self._create_feast_config(config_path)

        # Initialize Feast store
        self._feast_store = FeatureStore(repo_path=str(self._repo_path))
        self._is_initialized = True

        logger.info(f"Feast feature store initialized at {self._repo_path}")

    def _create_feast_config(self, config_path: Path) -> None:
        """Create Feast feature_store.yaml configuration."""
        online_store_config = self._get_online_store_config()
        offline_store_config = self._get_offline_store_config()

        config_content = f"""project: {self.config.project_name}
registry: {self._repo_path}/registry.db
provider: {self.config.provider}

online_store:
{online_store_config}

offline_store:
{offline_store_config}

entity_key_serialization_version: 2
"""
        config_path.write_text(config_content)

    def _get_online_store_config(self) -> str:
        """Get online store configuration."""
        if self.config.online_store_type == "sqlite":
            return f"  type: sqlite\n  path: {self._repo_path}/online_store.db"
        elif self.config.online_store_type == "redis":
            return "  type: redis\n  connection_string: localhost:6379"
        else:
            return f"  type: {self.config.online_store_type}"

    def _get_offline_store_config(self) -> str:
        """Get offline store configuration."""
        if self.config.offline_store_type == "file":
            return "  type: file"
        elif self.config.offline_store_type == "bigquery":
            return "  type: bigquery"
        elif self.config.offline_store_type == "redshift":
            return "  type: redshift"
        else:
            return f"  type: {self.config.offline_store_type}"

    def _infer_feast_dtype(self, pandas_dtype: str, feat_type: Optional[FeatureType] = None) -> str:
        """Infer Feast data type from pandas dtype."""
        from feast import ValueType

        dtype_str = str(pandas_dtype).lower()

        if feat_type == FeatureType.BOOLEAN or "bool" in dtype_str:
            return ValueType.BOOL
        elif "int64" in dtype_str or "int32" in dtype_str:
            return ValueType.INT64
        elif "float" in dtype_str or "double" in dtype_str:
            return ValueType.DOUBLE
        elif "object" in dtype_str or "string" in dtype_str:
            return ValueType.STRING
        elif "datetime" in dtype_str:
            return ValueType.UNIX_TIMESTAMP
        else:
            return ValueType.DOUBLE  # Default to double for numeric

    def save_features(
        self,
        df: pd.DataFrame,
        feature_set: Optional[FeatureSet] = None,
        feature_view_name: str = "featcopilot_features",
        description: Optional[str] = None,
        entity_columns: Optional[list[str]] = None,
        timestamp_column: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Save features to Feast.

        Parameters
        ----------
        df : DataFrame
            DataFrame containing features to save
        feature_set : FeatureSet, optional
            FeatCopilot FeatureSet with metadata
        feature_view_name : str
            Name for the Feast feature view
        description : str, optional
            Description of the feature view
        entity_columns : list, optional
            Override entity columns from config
        timestamp_column : str, optional
            Override timestamp column from config
        """
        # Determine entity and timestamp columns (validate before imports)
        entity_cols = entity_columns or self.config.entity_columns
        ts_col = timestamp_column or self.config.timestamp_column

        # Validate columns exist
        if not entity_cols:
            raise ValueError(
                "entity_columns must be specified either in config or save_features(). "
                "These are the key columns that identify each row (e.g., 'customer_id')."
            )

        for col in entity_cols:
            if col not in df.columns:
                raise ValueError(f"Entity column '{col}' not found in DataFrame")

        if not self._is_initialized:
            self.initialize()

        from feast import Entity, FeatureView, Field, FileSource
        from feast.types import Float64, Int64, String

        # Add timestamp column if not present
        if ts_col and ts_col not in df.columns:
            df = df.copy()
            df[ts_col] = datetime.now()
        elif not ts_col:
            ts_col = "event_timestamp"
            df = df.copy()
            df[ts_col] = datetime.now()

        # Save DataFrame to parquet
        data_path = self._repo_path / f"{feature_view_name}.parquet"
        df.to_parquet(data_path, index=False)

        # Create entities
        entities = []
        for entity_col in entity_cols:
            entity_name = entity_col.replace(" ", "_").lower()
            if entity_name not in self._entities:
                # Infer value type from dataframe
                col_dtype = str(df[entity_col].dtype)
                if "int" in col_dtype:
                    from feast import ValueType

                    value_type = ValueType.INT64
                elif "float" in col_dtype:
                    from feast import ValueType

                    value_type = ValueType.DOUBLE
                else:
                    from feast import ValueType

                    value_type = ValueType.STRING

                entity = Entity(
                    name=entity_name,
                    value_type=value_type,
                    description=f"Entity key: {entity_col}",
                )
                self._entities[entity_name] = entity
            entities.append(self._entities[entity_name])

        # Determine feature columns (exclude entity and timestamp)
        exclude_cols = set(entity_cols) | {ts_col}
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        # Create schema
        schema = []
        for col in feature_cols:
            dtype = str(df[col].dtype)
            if "int" in dtype:
                schema.append(Field(name=col, dtype=Int64))
            elif "float" in dtype or "double" in dtype:
                schema.append(Field(name=col, dtype=Float64))
            elif "object" in dtype or "string" in dtype:
                schema.append(Field(name=col, dtype=String))
            else:
                schema.append(Field(name=col, dtype=Float64))

        # Create file source
        source = FileSource(
            path=str(data_path),
            timestamp_field=ts_col,
        )

        # Create feature view
        feature_view = FeatureView(
            name=feature_view_name,
            entities=entities,  # Pass Entity objects, not strings
            ttl=timedelta(days=self.config.ttl_days),
            schema=schema,
            source=source,
            description=description or "Features generated by FeatCopilot",
            tags=self.config.tags,
        )

        self._feature_views[feature_view_name] = feature_view

        # Apply to Feast
        self._feast_store.apply([*entities, feature_view])

        logger.info(f"Saved {len(feature_cols)} features to Feast view '{feature_view_name}'")

        # Materialize to online store if enabled
        if self.config.auto_materialize:
            self._materialize(feature_view_name)

    def _materialize(self, feature_view_name: str) -> None:
        """Materialize features to online store."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config.ttl_days)

            self._feast_store.materialize(
                start_date=start_date,
                end_date=end_date,
                feature_views=[feature_view_name],
            )
            logger.info(f"Materialized '{feature_view_name}' to online store")
        except Exception as e:
            logger.warning(f"Could not materialize to online store: {e}")

    def get_features(
        self,
        entity_df: pd.DataFrame,
        feature_names: list[str],
        feature_view_name: str = "featcopilot_features",
        online: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Retrieve features from Feast.

        Parameters
        ----------
        entity_df : DataFrame
            DataFrame with entity keys and timestamps
        feature_names : list
            Names of features to retrieve
        feature_view_name : str
            Name of the feature view
        online : bool, default=False
            If True, use online store; otherwise use offline store

        Returns
        -------
        DataFrame
            DataFrame with requested features
        """
        if not self._is_initialized:
            self.initialize()

        # Format feature references
        feature_refs = [f"{feature_view_name}:{name}" for name in feature_names]

        if online:
            # Get from online store
            entity_rows = entity_df.to_dict("records")
            result = self._feast_store.get_online_features(
                features=feature_refs,
                entity_rows=entity_rows,
            )
            return pd.DataFrame(result.to_dict())
        else:
            # Get from offline store (historical)
            result = self._feast_store.get_historical_features(
                entity_df=entity_df,
                features=feature_refs,
            )
            return result.to_df()

    def get_online_features(
        self,
        entity_dict: Union[dict[str, list], pd.DataFrame],
        feature_names: list[str],
        feature_view_name: str = "featcopilot_features",
    ) -> dict[str, Any]:
        """
        Get features from online store for real-time inference.

        Parameters
        ----------
        entity_dict : dict or DataFrame
            Entity keys as dict or DataFrame
        feature_names : list
            Names of features to retrieve
        feature_view_name : str
            Name of the feature view

        Returns
        -------
        dict
            Features as dictionary
        """
        if not self._is_initialized:
            self.initialize()

        if isinstance(entity_dict, pd.DataFrame):
            entity_rows = entity_dict.to_dict("records")
        else:
            # Convert dict of lists to list of dicts
            keys = list(entity_dict.keys())
            n_rows = len(entity_dict[keys[0]])
            entity_rows = [{k: entity_dict[k][i] for k in keys} for i in range(n_rows)]

        feature_refs = [f"{feature_view_name}:{name}" for name in feature_names]

        result = self._feast_store.get_online_features(
            features=feature_refs,
            entity_rows=entity_rows,
        )
        return result.to_dict()

    def push_features(
        self,
        df: pd.DataFrame,
        feature_view_name: str = "featcopilot_features",
    ) -> None:
        """
        Push features to online store (streaming/real-time update).

        Parameters
        ----------
        df : DataFrame
            DataFrame with entity keys and feature values
        feature_view_name : str
            Name of the feature view
        """
        if not self._is_initialized:
            self.initialize()

        self._feast_store.push(feature_view_name, df)
        logger.info(f"Pushed {len(df)} rows to '{feature_view_name}'")

    def list_feature_views(self) -> list[str]:
        """List all feature views in the store."""
        if not self._is_initialized:
            self.initialize()

        views = self._feast_store.list_feature_views()
        return [v.name for v in views]

    def get_feature_view_schema(self, feature_view_name: str) -> dict[str, Any]:
        """Get schema/metadata for a feature view."""
        if not self._is_initialized:
            self.initialize()

        try:
            fv = self._feast_store.get_feature_view(feature_view_name)
            return {
                "name": fv.name,
                "entities": list(fv.entities),
                "features": [{"name": f.name, "dtype": str(f.dtype)} for f in fv.schema],
                "ttl": str(fv.ttl),
                "description": fv.description,
                "tags": fv.tags,
            }
        except Exception as e:
            logger.error(f"Could not get schema for '{feature_view_name}': {e}")
            return {}

    def delete_feature_view(self, feature_view_name: str) -> bool:
        """Delete a feature view."""
        if not self._is_initialized:
            self.initialize()

        try:
            self._feast_store.get_feature_view(feature_view_name)  # Verify it exists
            self._feast_store.delete_feature_view(feature_view_name)
            self._feature_views.pop(feature_view_name, None)

            # Clean up data file
            data_path = self._repo_path / f"{feature_view_name}.parquet"
            if data_path.exists():
                data_path.unlink()

            logger.info(f"Deleted feature view '{feature_view_name}'")
            return True
        except Exception as e:
            logger.error(f"Could not delete '{feature_view_name}': {e}")
            return False

    def close(self) -> None:
        """Close the Feast store and clean up resources."""
        self._feast_store = None
        self._is_initialized = False

        if self._temp_dir:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def __repr__(self) -> str:
        return f"FeastFeatureStore(repo_path='{self._repo_path}', project='{self.config.project_name}')"
