"""Tests for feature store integrations."""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from featcopilot.core.feature import Feature, FeatureOrigin, FeatureSet, FeatureType
from featcopilot.core.transform_rule import TransformRule
from featcopilot.stores.base import BaseFeatureStore, FeatureStoreConfig
from featcopilot.stores.feast_store import FeastConfig, FeastFeatureStore
from featcopilot.stores.rule_store import TransformRuleStore


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


# ---------------------------------------------------------------------------
# Additional tests appended for extended coverage
# ---------------------------------------------------------------------------


class _ConcreteFeatureStore(BaseFeatureStore):
    """Minimal concrete subclass used to test BaseFeatureStore."""

    def initialize(self) -> None:
        self._is_initialized = True

    def save_features(self, df, feature_set=None, feature_view_name="fv", description=None, **kw):
        pass

    def get_features(self, entity_df, feature_names, feature_view_name="fv", **kw):
        return entity_df

    def list_feature_views(self):
        return []

    def get_feature_view_schema(self, feature_view_name):
        return {}

    def delete_feature_view(self, feature_view_name):
        return True


class TestBaseFeatureStoreConcreteSubclass:
    """Tests for BaseFeatureStore using a concrete subclass."""

    def test_initialize_sets_flag(self):
        """Test that initialize sets _is_initialized."""
        store = _ConcreteFeatureStore(config=FeatureStoreConfig(name="test"))
        assert store._is_initialized is False
        store.initialize()
        assert store._is_initialized is True

    def test_close_resets_flag(self):
        """Test that close resets _is_initialized."""
        store = _ConcreteFeatureStore(config=FeatureStoreConfig(name="test"))
        store.initialize()
        assert store._is_initialized is True
        store.close()
        assert store._is_initialized is False

    def test_context_manager_enter_exit(self):
        """Test __enter__ calls initialize and __exit__ calls close."""
        store = _ConcreteFeatureStore(config=FeatureStoreConfig(name="ctx"))
        with store as s:
            assert s is store
            assert store._is_initialized is True
        assert store._is_initialized is False

    def test_context_manager_exit_returns_false(self):
        """Test that __exit__ returns False (does not suppress exceptions)."""
        store = _ConcreteFeatureStore(config=FeatureStoreConfig(name="ctx"))
        result = store.__exit__(None, None, None)
        assert result is False

    def test_abstract_methods_callable(self):
        """Test that concrete implementations of abstract methods work."""
        cfg = FeatureStoreConfig(name="abs")
        store = _ConcreteFeatureStore(config=cfg)
        df = pd.DataFrame({"a": [1]})
        store.save_features(df)
        result = store.get_features(df, ["a"])
        assert isinstance(result, pd.DataFrame)
        assert store.list_feature_views() == []
        assert store.get_feature_view_schema("v") == {}
        assert store.delete_feature_view("v") is True


class TestFeastConfigModel:
    """Additional FeastConfig model tests."""

    def test_feast_config_entity_columns_default(self):
        """Test that entity_columns defaults to empty list."""
        config = FeastConfig()
        assert config.entity_columns == []

    def test_feast_config_custom_tags(self):
        """Test FeastConfig with custom tags."""
        config = FeastConfig(tags={"env": "prod", "team": "ds"})
        assert config.tags == {"env": "prod", "team": "ds"}

    def test_feast_config_feature_prefix(self):
        """Test FeastConfig inherits feature_prefix."""
        config = FeastConfig(feature_prefix="f_")
        assert config.feature_prefix == "f_"


class TestFeastFeatureStoreOnlineOfflineConfig:
    """Tests for online/offline store config edge cases."""

    def test_online_store_config_unknown_type(self):
        """Test unknown online store type returns generic config."""
        store = FeastFeatureStore(online_store_type="dynamodb")
        config = store._get_online_store_config()
        assert "dynamodb" in config

    def test_offline_store_config_redshift(self):
        """Test redshift offline store config."""
        store = FeastFeatureStore(offline_store_type="redshift")
        config = store._get_offline_store_config()
        assert "redshift" in config

    def test_offline_store_config_unknown_type(self):
        """Test unknown offline store type returns generic config."""
        store = FeastFeatureStore(offline_store_type="snowflake")
        config = store._get_offline_store_config()
        assert "snowflake" in config


class TestFeastFeatureStoreInitialize:
    """Tests for FeastFeatureStore.initialize with mocked feast."""

    def test_initialize_feast_not_installed(self):
        """Test initialize raises ImportError when feast is missing."""
        store = FeastFeatureStore()
        with patch.dict("sys.modules", {"feast": None}):
            with pytest.raises(ImportError, match="Feast is not installed"):
                store.initialize()

    def test_initialize_with_temp_dir(self):
        """Test initialize creates temp directory when repo_path is None."""
        mock_feast_mod = MagicMock()
        mock_feast_store_cls = mock_feast_mod.FeatureStore

        store = FeastFeatureStore()
        with patch.dict("sys.modules", {"feast": mock_feast_mod}):
            store.initialize()

        assert store._is_initialized is True
        assert store._repo_path is not None
        mock_feast_store_cls.assert_called_once()
        store.close()

    def test_initialize_with_explicit_repo_path(self):
        """Test initialize uses given repo_path."""
        mock_feast_mod = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            repo = os.path.join(tmpdir, "feast_repo")
            store = FeastFeatureStore(repo_path=repo)
            with patch.dict("sys.modules", {"feast": mock_feast_mod}):
                store.initialize()

            assert store._is_initialized is True
            assert store._repo_path == Path(repo)
            assert store._repo_path.exists()
            store.close()


class TestFeastFeatureStoreMethods:
    """Tests for FeastFeatureStore methods with fully mocked feast store."""

    @pytest.fixture
    def mock_feast_store(self):
        """Create a FeastFeatureStore with mocked internals."""
        store = FeastFeatureStore(
            entity_columns=["user_id"],
            timestamp_column="ts",
        )
        store._is_initialized = True
        store._feast_store = MagicMock()
        store._repo_path = Path(tempfile.mkdtemp())
        return store

    def test_get_features_offline(self, mock_feast_store):
        """Test get_features in offline mode."""
        mock_result = MagicMock()
        mock_result.to_df.return_value = pd.DataFrame({"feat": [1, 2]})
        mock_feast_store._feast_store.get_historical_features.return_value = mock_result

        entity_df = pd.DataFrame({"user_id": [1, 2], "ts": [datetime.now()] * 2})
        result = mock_feast_store.get_features(entity_df, ["feat"], feature_view_name="fv")

        assert isinstance(result, pd.DataFrame)
        mock_feast_store._feast_store.get_historical_features.assert_called_once()

    def test_get_features_online(self, mock_feast_store):
        """Test get_features in online mode."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"feat": [10]}
        mock_feast_store._feast_store.get_online_features.return_value = mock_result

        entity_df = pd.DataFrame({"user_id": [1]})
        result = mock_feast_store.get_features(entity_df, ["feat"], feature_view_name="fv", online=True)

        assert isinstance(result, pd.DataFrame)
        mock_feast_store._feast_store.get_online_features.assert_called_once()

    def test_get_features_auto_initializes(self):
        """Test get_features calls initialize when not initialized."""
        store = FeastFeatureStore(entity_columns=["id"])
        store._is_initialized = False
        with patch.object(store, "initialize") as mock_init:
            store._feast_store = MagicMock()
            mock_result = MagicMock()
            mock_result.to_df.return_value = pd.DataFrame()
            store._feast_store.get_historical_features.return_value = mock_result
            store.get_features(pd.DataFrame({"id": [1]}), ["f"])
            mock_init.assert_called_once()

    def test_get_online_features_with_dict(self, mock_feast_store):
        """Test get_online_features with dict input."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"feat": [5]}
        mock_feast_store._feast_store.get_online_features.return_value = mock_result

        result = mock_feast_store.get_online_features(
            entity_dict={"user_id": [1, 2]},
            feature_names=["feat"],
            feature_view_name="fv",
        )

        assert result == {"feat": [5]}
        call_args = mock_feast_store._feast_store.get_online_features.call_args
        assert len(call_args.kwargs["entity_rows"]) == 2

    def test_get_online_features_with_dataframe(self, mock_feast_store):
        """Test get_online_features with DataFrame input."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"feat": [5]}
        mock_feast_store._feast_store.get_online_features.return_value = mock_result

        result = mock_feast_store.get_online_features(
            entity_dict=pd.DataFrame({"user_id": [1]}),
            feature_names=["feat"],
        )
        assert result == {"feat": [5]}

    def test_get_online_features_auto_initializes(self):
        """Test get_online_features calls initialize when not initialized."""
        store = FeastFeatureStore(entity_columns=["id"])
        store._is_initialized = False
        with patch.object(store, "initialize") as mock_init:
            store._feast_store = MagicMock()
            mock_result = MagicMock()
            mock_result.to_dict.return_value = {}
            store._feast_store.get_online_features.return_value = mock_result
            store.get_online_features({"id": [1]}, ["f"])
            mock_init.assert_called_once()

    def test_push_features(self, mock_feast_store):
        """Test push_features calls feast push."""
        df = pd.DataFrame({"user_id": [1], "feat": [0.5]})
        mock_feast_store.push_features(df, feature_view_name="fv")
        mock_feast_store._feast_store.push.assert_called_once_with("fv", df)

    def test_push_features_auto_initializes(self):
        """Test push_features calls initialize when not initialized."""
        store = FeastFeatureStore(entity_columns=["id"])
        store._is_initialized = False
        with patch.object(store, "initialize") as mock_init:
            store._feast_store = MagicMock()
            store.push_features(pd.DataFrame({"id": [1]}))
            mock_init.assert_called_once()

    def test_list_feature_views(self, mock_feast_store):
        """Test list_feature_views returns names."""
        mock_fv1, mock_fv2 = MagicMock(), MagicMock()
        mock_fv1.name = "view_a"
        mock_fv2.name = "view_b"
        mock_feast_store._feast_store.list_feature_views.return_value = [mock_fv1, mock_fv2]

        result = mock_feast_store.list_feature_views()
        assert result == ["view_a", "view_b"]

    def test_list_feature_views_auto_initializes(self):
        """Test list_feature_views calls initialize when not initialized."""
        store = FeastFeatureStore()
        store._is_initialized = False
        with patch.object(store, "initialize") as mock_init:
            store._feast_store = MagicMock()
            store._feast_store.list_feature_views.return_value = []
            store.list_feature_views()
            mock_init.assert_called_once()

    def test_get_feature_view_schema_success(self, mock_feast_store):
        """Test get_feature_view_schema returns schema dict."""
        mock_fv = MagicMock()
        mock_fv.name = "fv"
        mock_fv.entities = ["user_id"]
        mock_field = MagicMock()
        mock_field.name = "feat"
        mock_field.dtype = "DOUBLE"
        mock_fv.schema = [mock_field]
        mock_fv.ttl = "365 days"
        mock_fv.description = "test"
        mock_fv.tags = {}
        mock_feast_store._feast_store.get_feature_view.return_value = mock_fv

        schema = mock_feast_store.get_feature_view_schema("fv")
        assert schema["name"] == "fv"
        assert schema["features"] == [{"name": "feat", "dtype": "DOUBLE"}]

    def test_get_feature_view_schema_not_found(self, mock_feast_store):
        """Test get_feature_view_schema returns empty dict on error."""
        mock_feast_store._feast_store.get_feature_view.side_effect = Exception("not found")

        result = mock_feast_store.get_feature_view_schema("missing")
        assert result == {}

    def test_delete_feature_view_success(self, mock_feast_store):
        """Test delete_feature_view succeeds."""
        # Create a dummy parquet file
        parquet_path = mock_feast_store._repo_path / "fv.parquet"
        parquet_path.write_text("dummy")

        result = mock_feast_store.delete_feature_view("fv")
        assert result is True
        mock_feast_store._feast_store.delete_feature_view.assert_called_once_with("fv")
        assert not parquet_path.exists()

    def test_delete_feature_view_failure(self, mock_feast_store):
        """Test delete_feature_view returns False on error."""
        mock_feast_store._feast_store.get_feature_view.side_effect = Exception("nope")

        result = mock_feast_store.delete_feature_view("bad")
        assert result is False

    def test_materialize_success(self, mock_feast_store):
        """Test _materialize calls feast materialize."""
        mock_feast_store._materialize("fv")
        mock_feast_store._feast_store.materialize.assert_called_once()

    def test_materialize_logs_warning_on_failure(self, mock_feast_store):
        """Test _materialize logs warning instead of raising."""
        mock_feast_store._feast_store.materialize.side_effect = Exception("fail")
        # Should not raise
        mock_feast_store._materialize("fv")

    def test_close_cleans_temp_dir(self):
        """Test close cleans up temp directory."""
        store = FeastFeatureStore()
        mock_temp_dir = MagicMock()
        store._temp_dir = mock_temp_dir
        store._feast_store = MagicMock()
        store._is_initialized = True

        store.close()

        assert store._feast_store is None
        assert store._is_initialized is False
        assert store._temp_dir is None
        mock_temp_dir.cleanup.assert_called_once()


class TestFeastDtypeInference:
    """Additional dtype inference tests for _infer_feast_dtype."""

    def _make_store_with_mock(self):
        """Helper to create store and mock feast ValueType."""
        store = FeastFeatureStore()
        mock_value_type = MagicMock()
        mock_value_type.BOOL = "BOOL"
        mock_value_type.INT64 = "INT64"
        mock_value_type.DOUBLE = "DOUBLE"
        mock_value_type.STRING = "STRING"
        mock_value_type.UNIX_TIMESTAMP = "UNIX_TIMESTAMP"
        return store, mock_value_type

    def test_infer_bool_dtype_from_feat_type(self):
        """Test boolean inference via FeatureType.BOOLEAN."""
        store, mock_vt = self._make_store_with_mock()
        with patch.dict("sys.modules", {"feast": MagicMock(ValueType=mock_vt)}):
            result = store._infer_feast_dtype("float64", feat_type=FeatureType.BOOLEAN)
            assert result == "BOOL"

    def test_infer_bool_dtype_from_string(self):
        """Test boolean inference from dtype string."""
        store, mock_vt = self._make_store_with_mock()
        with patch.dict("sys.modules", {"feast": MagicMock(ValueType=mock_vt)}):
            result = store._infer_feast_dtype("bool")
            assert result == "BOOL"

    def test_infer_int32_dtype(self):
        """Test int32 maps to INT64."""
        store, mock_vt = self._make_store_with_mock()
        with patch.dict("sys.modules", {"feast": MagicMock(ValueType=mock_vt)}):
            result = store._infer_feast_dtype("int32")
            assert result == "INT64"

    def test_infer_datetime_dtype(self):
        """Test datetime maps to UNIX_TIMESTAMP."""
        store, mock_vt = self._make_store_with_mock()
        with patch.dict("sys.modules", {"feast": MagicMock(ValueType=mock_vt)}):
            result = store._infer_feast_dtype("datetime64[ns]")
            assert result == "UNIX_TIMESTAMP"

    def test_infer_unknown_dtype_defaults_to_double(self):
        """Test unknown dtype defaults to DOUBLE."""
        store, mock_vt = self._make_store_with_mock()
        with patch.dict("sys.modules", {"feast": MagicMock(ValueType=mock_vt)}):
            result = store._infer_feast_dtype("complex128")
            assert result == "DOUBLE"


class TestRuleStoreEdgeCases:
    """Edge-case tests for TransformRuleStore."""

    @pytest.fixture
    def tmp_store(self, tmp_path):
        """Create a TransformRuleStore backed by a temp file."""
        store_path = str(tmp_path / "rules.json")
        return TransformRuleStore(path=store_path)

    @pytest.fixture
    def sample_rule(self):
        """Create a sample TransformRule."""
        return TransformRule(
            name="ratio",
            description="Calculate ratio of two columns",
            code="result = df['{col1}'] / (df['{col2}'] + 1e-8)",
            input_columns=["col1", "col2"],
            column_patterns=["price.*", "qty.*"],
            tags=["ratio", "numeric"],
        )

    def test_get_rule_by_name_not_found(self, tmp_store):
        """Test get_rule_by_name returns None when name doesn't exist."""
        assert tmp_store.get_rule_by_name("nonexistent") is None

    def test_delete_rule_not_found(self, tmp_store):
        """Test delete_rule returns False for missing rule."""
        assert tmp_store.delete_rule("no-such-id") is False

    def test_find_matching_rules_min_usage_filter(self, tmp_store, sample_rule):
        """Test find_matching_rules filters by min_usage."""
        sample_rule.usage_count = 2
        tmp_store.save_rule(sample_rule)

        results = tmp_store.find_matching_rules(min_usage=5)
        assert len(results) == 0

        results = tmp_store.find_matching_rules(min_usage=1)
        assert len(results) == 1

    def test_find_matching_rules_tag_filter(self, tmp_store, sample_rule):
        """Test find_matching_rules filters by tags."""
        tmp_store.save_rule(sample_rule)

        results = tmp_store.find_matching_rules(tags=["ratio"])
        assert len(results) == 1

        results = tmp_store.find_matching_rules(tags=["missing_tag"])
        assert len(results) == 0

    def test_find_matching_rules_description_filter(self, tmp_store, sample_rule):
        """Test find_matching_rules filters by description keywords."""
        tmp_store.save_rule(sample_rule)

        results = tmp_store.find_matching_rules(description="ratio")
        assert len(results) == 1

        results = tmp_store.find_matching_rules(description="zzz_no_match")
        assert len(results) == 0

    def test_find_matching_rules_column_filter(self, tmp_store, sample_rule):
        """Test find_matching_rules filters by column compatibility."""
        tmp_store.save_rule(sample_rule)

        results = tmp_store.find_matching_rules(columns=["col1", "col2"])
        assert len(results) == 1

        results = tmp_store.find_matching_rules(columns=["unrelated"])
        assert len(results) == 0

    def test_search_by_description_empty_store(self, tmp_store):
        """Test search_by_description on empty store."""
        results = tmp_store.search_by_description("anything")
        assert results == []

    def test_search_by_description_scoring(self, tmp_store):
        """Test search_by_description ranks by word overlap."""
        rule1 = TransformRule(name="add", description="add two columns together", code="result = 1", input_columns=[])
        rule2 = TransformRule(
            name="multiply", description="multiply columns for product", code="result = 1", input_columns=[]
        )
        tmp_store.save_rule(rule1)
        tmp_store.save_rule(rule2)

        results = tmp_store.search_by_description("add columns")
        assert len(results) >= 1
        assert results[0].name == "add"

    def test_search_by_description_limit(self, tmp_store):
        """Test search_by_description respects limit."""
        for i in range(5):
            r = TransformRule(name=f"rule_{i}", description="common word", code="result = 1", input_columns=[])
            tmp_store.save_rule(r)

        results = tmp_store.search_by_description("common", limit=2)
        assert len(results) == 2

    def test_import_rules_file_not_found(self, tmp_store):
        """Test import_rules raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Import file not found"):
            tmp_store.import_rules("/nonexistent/path.json")

    def test_import_rules_merge(self, tmp_store, sample_rule, tmp_path):
        """Test import_rules merges with existing rules."""
        tmp_store.save_rule(sample_rule)
        assert len(tmp_store) == 1

        import_rule = TransformRule(name="imported", description="imported rule", code="result = 1", input_columns=[])
        import_path = str(tmp_path / "import.json")
        with open(import_path, "w", encoding="utf-8") as f:
            json.dump({import_rule.id: import_rule.to_dict()}, f)

        count = tmp_store.import_rules(import_path, merge=True)
        assert count == 1
        assert len(tmp_store) == 2

    def test_import_rules_replace(self, tmp_store, sample_rule, tmp_path):
        """Test import_rules replaces when merge=False."""
        tmp_store.save_rule(sample_rule)
        assert len(tmp_store) == 1

        import_rule = TransformRule(name="replacement", description="replaces all", code="result = 1", input_columns=[])
        import_path = str(tmp_path / "import.json")
        with open(import_path, "w", encoding="utf-8") as f:
            json.dump({import_rule.id: import_rule.to_dict()}, f)

        count = tmp_store.import_rules(import_path, merge=False)
        assert count == 1
        assert len(tmp_store) == 1

    def test_export_rules_with_tag_filter(self, tmp_store, tmp_path):
        """Test export_rules filters by tags."""
        r1 = TransformRule(name="a", description="d", code="result = 1", input_columns=[], tags=["keep"])
        r2 = TransformRule(name="b", description="d", code="result = 1", input_columns=[], tags=["skip"])
        tmp_store.save_rule(r1)
        tmp_store.save_rule(r2)

        export_path = str(tmp_path / "out.json")
        count = tmp_store.export_rules(export_path, tags=["keep"])
        assert count == 1

        with open(export_path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1

    def test_clear(self, tmp_store, sample_rule):
        """Test clear removes all rules."""
        tmp_store.save_rule(sample_rule)
        assert len(tmp_store) >= 1
        tmp_store.clear()
        assert len(tmp_store) == 0

    def test_contains(self, tmp_store, sample_rule):
        """Test __contains__ check."""
        tmp_store.save_rule(sample_rule)
        assert sample_rule.id in tmp_store
        assert "nonexistent" not in tmp_store

    def test_iter(self, tmp_store, sample_rule):
        """Test __iter__ yields rules."""
        tmp_store.save_rule(sample_rule)
        rules = list(tmp_store)
        assert len(rules) == 1
        assert rules[0].name == sample_rule.name

    def test_load_corrupt_file(self, tmp_path):
        """Test loading from a corrupt JSON file falls back to empty."""
        store_path = tmp_path / "rules.json"
        store_path.write_text("NOT VALID JSON!!!")
        store = TransformRuleStore(path=str(store_path))
        assert len(store) == 0

    def test_persistence_across_instances(self, tmp_path, sample_rule):
        """Test rules persist across store instances."""
        store_path = str(tmp_path / "rules.json")
        store1 = TransformRuleStore(path=store_path)
        store1.save_rule(sample_rule)

        store2 = TransformRuleStore(path=store_path)
        assert len(store2) == 1
        assert store2.get_rule(sample_rule.id) is not None
