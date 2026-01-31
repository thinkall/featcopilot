"""Tests for feature store integrations."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from featcopilot.core.feature import Feature, FeatureOrigin, FeatureSet, FeatureType
from featcopilot.stores.base import FeatureStoreConfig
from featcopilot.stores.feast_store import FeastConfig, FeastFeatureStore


class TestFeatureStoreConfig:
    """Tests for FeatureStoreConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = FeatureStoreConfig(name="test")

        assert config.name == "test"
        assert config.entity_columns == []
        assert config.timestamp_column is None
        assert config.feature_prefix == ""
        assert config.tags == {}

    def test_custom_config(self):
        """Test custom configuration."""
        config = FeatureStoreConfig(
            name="custom",
            entity_columns=["user_id", "item_id"],
            timestamp_column="event_time",
            feature_prefix="feat_",
            tags={"team": "ml"},
        )

        assert config.name == "custom"
        assert config.entity_columns == ["user_id", "item_id"]
        assert config.timestamp_column == "event_time"
        assert config.feature_prefix == "feat_"
        assert config.tags == {"team": "ml"}


class TestFeastConfig:
    """Tests for FeastConfig."""

    def test_default_feast_config(self):
        """Test default Feast configuration."""
        config = FeastConfig()

        assert config.name == "feast"
        assert config.project_name == "featcopilot"
        assert config.provider == "local"
        assert config.online_store_type == "sqlite"
        assert config.offline_store_type == "file"
        assert config.ttl_days == 365
        assert config.auto_materialize is True

    def test_custom_feast_config(self):
        """Test custom Feast configuration."""
        config = FeastConfig(
            repo_path="/tmp/feast_repo",
            project_name="my_project",
            provider="gcp",
            online_store_type="redis",
            offline_store_type="bigquery",
            ttl_days=30,
            auto_materialize=False,
        )

        assert config.repo_path == "/tmp/feast_repo"
        assert config.project_name == "my_project"
        assert config.provider == "gcp"
        assert config.online_store_type == "redis"
        assert config.offline_store_type == "bigquery"
        assert config.ttl_days == 30
        assert config.auto_materialize is False


class TestFeastFeatureStoreInit:
    """Tests for FeastFeatureStore initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        store = FeastFeatureStore()

        assert store.config.project_name == "featcopilot"
        assert store.config.provider == "local"
        assert store._is_initialized is False

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        store = FeastFeatureStore(
            repo_path="/tmp/test_repo",
            project_name="test_project",
            entity_columns=["customer_id"],
            timestamp_column="ts",
            provider="local",
            ttl_days=7,
        )

        assert store.config.repo_path == "/tmp/test_repo"
        assert store.config.project_name == "test_project"
        assert store.config.entity_columns == ["customer_id"]
        assert store.config.timestamp_column == "ts"
        assert store.config.ttl_days == 7

    def test_repr(self):
        """Test string representation."""
        store = FeastFeatureStore(repo_path="/tmp/repo", project_name="test")
        # Before initialization, repo_path is None
        assert "FeastFeatureStore" in repr(store)


class TestFeastFeatureStoreWithMocks:
    """Tests for FeastFeatureStore with mocked Feast."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "customer_id": range(1, 101),
                "event_timestamp": [datetime.now()] * 100,
                "feature_a": np.random.randn(100),
                "feature_b": np.random.randn(100),
                "feature_c": np.random.randint(0, 10, 100),
            }
        )

    @pytest.fixture
    def feature_set(self):
        """Create sample FeatureSet."""
        fs = FeatureSet()
        fs.add(
            Feature(
                name="feature_a",
                dtype=FeatureType.NUMERIC,
                origin=FeatureOrigin.POLYNOMIAL,
                source_columns=["col1"],
            )
        )
        fs.add(
            Feature(
                name="feature_b",
                dtype=FeatureType.NUMERIC,
                origin=FeatureOrigin.INTERACTION,
                source_columns=["col1", "col2"],
            )
        )
        return fs

    def test_save_features_validates_entity_columns(self, sample_df):
        """Test that save_features validates entity columns."""
        store = FeastFeatureStore(
            entity_columns=[],  # Empty entity columns
        )

        with pytest.raises(ValueError, match="entity_columns must be specified"):
            store.save_features(sample_df, feature_view_name="test")

    def test_save_features_validates_missing_columns(self, sample_df):
        """Test that save_features validates missing entity columns."""
        store = FeastFeatureStore(
            entity_columns=["nonexistent_column"],
        )

        with pytest.raises(ValueError, match="Entity column 'nonexistent_column' not found"):
            store.save_features(sample_df, feature_view_name="test")

    @patch("featcopilot.stores.feast_store.FeastFeatureStore.initialize")
    def test_context_manager(self, mock_init):
        """Test context manager usage."""
        store = FeastFeatureStore()

        with store as s:
            assert s is store
            mock_init.assert_called_once()

    def test_close(self):
        """Test close method."""
        store = FeastFeatureStore()
        store._is_initialized = True
        store._feast_store = MagicMock()

        store.close()

        assert store._is_initialized is False
        assert store._feast_store is None


class TestFeastConfigGeneration:
    """Tests for Feast config file generation."""

    def test_online_store_config_sqlite(self):
        """Test SQLite online store config."""
        store = FeastFeatureStore(online_store_type="sqlite")
        store._repo_path = "/tmp/test"

        config = store._get_online_store_config()

        assert "sqlite" in config
        assert "path" in config

    def test_online_store_config_redis(self):
        """Test Redis online store config."""
        store = FeastFeatureStore(online_store_type="redis")

        config = store._get_online_store_config()

        assert "redis" in config

    def test_offline_store_config_file(self):
        """Test file offline store config."""
        store = FeastFeatureStore(offline_store_type="file")

        config = store._get_offline_store_config()

        assert "file" in config

    def test_offline_store_config_bigquery(self):
        """Test BigQuery offline store config."""
        store = FeastFeatureStore(offline_store_type="bigquery")

        config = store._get_offline_store_config()

        assert "bigquery" in config


class TestFeatureTypeInference:
    """Tests for Feast data type inference."""

    def test_infer_int_dtype(self):
        """Test integer dtype inference."""
        store = FeastFeatureStore()

        # We need to mock the feast import
        with patch.dict("sys.modules", {"feast": MagicMock()}):
            from feast import ValueType

            with patch.object(ValueType, "INT64", "INT64"):
                dtype = store._infer_feast_dtype("int64")
                # Without actual Feast, this will use the mocked value

    def test_infer_float_dtype(self):
        """Test float dtype inference."""
        store = FeastFeatureStore()

        with patch.dict("sys.modules", {"feast": MagicMock()}):
            dtype = store._infer_feast_dtype("float64")
            # Verifies the method handles float types

    def test_infer_string_dtype(self):
        """Test string dtype inference."""
        store = FeastFeatureStore()

        with patch.dict("sys.modules", {"feast": MagicMock()}):
            dtype = store._infer_feast_dtype("object")
            # Verifies the method handles object/string types
