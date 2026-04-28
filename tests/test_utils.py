"""Tests for featcopilot.utils modules."""

import asyncio
import logging
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd

from featcopilot.utils.cache import FeatureCache
from featcopilot.utils.logger import get_logger, set_level
from featcopilot.utils.models import (
    DEFAULT_MODEL,
    _get_event_loop,
    _print_models,
    fetch_models,
    get_default_model,
    get_model_info,
    get_model_names,
    is_valid_model,
    list_models,
)
from featcopilot.utils.parallel import parallel_apply, parallel_transform
from featcopilot.utils.validation import find_potential_leakage_columns

# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_leakage_detection_non_string_columns():
    """Test leakage detection accepts non-string column labels."""
    assert find_potential_leakage_columns([], target_name="label") == []
    assert find_potential_leakage_columns([123, "future_label"], target_name="label") == ["future_label"]
    assert find_potential_leakage_columns([123, "target"], target_name="other") == ["target"]
    assert find_potential_leakage_columns([123, 456], target_name="label") == []
    assert find_potential_leakage_columns(["Churn Label"], target_name="churn_label") == ["Churn Label"]
    assert find_potential_leakage_columns(["future-target!"], target_name="target") == ["future-target!"]


# ---------------------------------------------------------------------------
# FeatureCache tests
# ---------------------------------------------------------------------------


class TestFeatureCacheMemory:
    """Tests for FeatureCache with memory-only storage."""

    def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = FeatureCache()
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_nonexistent_key(self):
        """Test getting a key that does not exist returns None."""
        cache = FeatureCache()
        assert cache.get("missing") is None

    def test_has_existing_key(self):
        """Test has returns True for existing key."""
        cache = FeatureCache()
        cache.set("key1", 42)
        assert cache.has("key1") is True

    def test_has_missing_key(self):
        """Test has returns False for missing key."""
        cache = FeatureCache()
        assert cache.has("no_such_key") is False

    def test_delete_existing_key(self):
        """Test deleting an existing key returns True and removes it."""
        cache = FeatureCache()
        cache.set("key1", "val")
        assert cache.delete("key1") is True
        assert cache.get("key1") is None

    def test_delete_nonexistent_key(self):
        """Test deleting a non-existent key returns False."""
        cache = FeatureCache()
        assert cache.delete("ghost") is False

    def test_clear(self):
        """Test clearing the cache removes all entries."""
        cache = FeatureCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.list_keys() == []

    def test_list_keys(self):
        """Test listing all keys in the cache."""
        cache = FeatureCache()
        cache.set("x", 10)
        cache.set("y", 20)
        assert sorted(cache.list_keys()) == ["x", "y"]


class TestFeatureCacheDisk:
    """Tests for FeatureCache with disk-backed storage."""

    def test_disk_persistence(self):
        """Test that values persist to disk and can be retrieved after memory clear."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(cache_dir=tmpdir)
            cache.set("persist_key", [1, 2, 3])
            # Clear memory, value should still be retrievable from disk
            cache._memory_cache.clear()
            assert cache.get("persist_key") == [1, 2, 3]

    def test_metadata_storage_and_retrieval(self):
        """Test storing and retrieving metadata alongside cached values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(cache_dir=tmpdir)
            meta = {"source": "tabular", "version": 2}
            cache.set("feat", "data", metadata=meta)
            assert cache.get_metadata("feat") == meta

    def test_metadata_from_disk(self):
        """Test metadata retrieval from disk when not in memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(cache_dir=tmpdir)
            meta = {"engine": "openfe"}
            cache.set("feat2", "data2", metadata=meta)
            # Clear in-memory metadata
            cache._metadata.clear()
            assert cache.get_metadata("feat2") == meta

    def test_metadata_nonexistent_key(self):
        """Test metadata retrieval for a non-existent key returns None."""
        cache = FeatureCache()
        assert cache.get_metadata("nope") is None

    def test_disk_delete(self):
        """Test that delete removes both memory and disk entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(cache_dir=tmpdir)
            cache.set("del_key", "to_delete")
            assert cache.delete("del_key") is True
            assert cache.get("del_key") is None

    def test_disk_clear(self):
        """Test that clear removes all disk entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(cache_dir=tmpdir)
            cache.set("a", 1)
            cache.set("b", 2)
            cache.clear()
            assert cache.list_keys() == []

    def test_list_keys_includes_disk(self):
        """Test that list_keys includes keys only on disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(cache_dir=tmpdir)
            cache.set("disk_key", "val")
            cache._memory_cache.clear()
            assert "disk_key" in cache.list_keys()


class TestFeatureCacheDataHash:
    """Tests for FeatureCache with data_hash parameter."""

    def test_set_and_get_with_hash(self):
        """Test set/get using data_hash produces a combined cache key."""
        cache = FeatureCache()
        cache.set("feat", "v1", data_hash="abc123")
        assert cache.get("feat", data_hash="abc123") == "v1"
        # Without hash should not find it
        assert cache.get("feat") is None

    def test_compute_data_hash(self):
        """Test _compute_data_hash returns a deterministic hex string."""
        cache = FeatureCache()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        h1 = cache._compute_data_hash(df)
        h2 = cache._compute_data_hash(df)
        assert h1 == h2
        assert len(h1) == 16
        assert isinstance(h1, str)

    def test_compute_data_hash_different_data(self):
        """Test _compute_data_hash returns different hashes for different DataFrames."""
        cache = FeatureCache()
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})
        assert cache._compute_data_hash(df1) != cache._compute_data_hash(df2)


class TestFeatureCacheEviction:
    """Tests for FeatureCache eviction behaviour."""

    def test_eviction_when_over_limit(self):
        """Test that oldest item is evicted when max_memory_items is exceeded."""
        cache = FeatureCache(max_memory_items=3)
        cache.set("k1", 1)
        cache.set("k2", 2)
        cache.set("k3", 3)
        # Adding a 4th item should evict the oldest (k1)
        cache.set("k4", 4)
        assert cache.get("k1") is None
        assert cache.get("k4") == 4
        assert len(cache._memory_cache) == 3


# ---------------------------------------------------------------------------
# Logger tests
# ---------------------------------------------------------------------------


class TestLogger:
    """Tests for the logger module."""

    def test_get_logger_with_name(self):
        """Test get_logger returns a logger with the correct name."""
        log = get_logger("mymodule")
        assert log.name == "featcopilot.mymodule"

    def test_get_logger_none(self):
        """Test get_logger with None returns the root featcopilot logger."""
        log = get_logger(None)
        assert log.name == "featcopilot"

    def test_get_logger_strips_prefix(self):
        """Test get_logger strips 'featcopilot.' prefix to avoid duplication."""
        log = get_logger("featcopilot.utils.cache")
        assert log.name == "featcopilot.utils.cache"

    def test_set_level_with_string(self):
        """Test set_level accepts a string level name."""
        set_level("DEBUG")
        root = get_logger(None)
        assert root.level == logging.DEBUG
        # Reset
        set_level("INFO")

    def test_set_level_with_int(self):
        """Test set_level accepts an integer level."""
        set_level(logging.WARNING)
        root = get_logger(None)
        assert root.level == logging.WARNING
        # Reset
        set_level(logging.INFO)


# ---------------------------------------------------------------------------
# Parallel tests
# ---------------------------------------------------------------------------


class TestParallelApply:
    """Tests for parallel_apply function."""

    def test_parallel_apply_simple(self):
        """Test parallel_apply with a simple function."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        results = parallel_apply(lambda row: row["a"] + row["b"], df, n_jobs=1)
        assert results == [5, 7, 9]

    def test_parallel_apply_verbose(self):
        """Test parallel_apply with verbose=True does not raise."""
        df = pd.DataFrame({"x": [10, 20]})
        results = parallel_apply(lambda row: row["x"] * 2, df, n_jobs=1, verbose=True)
        assert results == [20, 40]


class TestParallelTransform:
    """Tests for parallel_transform function."""

    def test_parallel_transform_with_mock_transformers(self):
        """Test parallel_transform combines transformer outputs."""
        df = pd.DataFrame({"a": [1, 2, 3]})

        t1 = MagicMock()
        t1.transform.return_value = pd.DataFrame({"a": [1, 2, 3], "feat1": [10, 20, 30]})

        t2 = MagicMock()
        t2.transform.return_value = pd.DataFrame({"a": [1, 2, 3], "feat2": [100, 200, 300]})

        result = parallel_transform([("t1", t1), ("t2", t2)], df, n_jobs=1)

        assert "feat1" in result.columns
        assert "feat2" in result.columns
        assert list(result["feat1"]) == [10, 20, 30]
        assert list(result["feat2"]) == [100, 200, 300]
        t1.transform.assert_called_once()
        t2.transform.assert_called_once()


# ---------------------------------------------------------------------------
# Models tests
# ---------------------------------------------------------------------------

FAKE_MODELS = [
    {"id": "gpt-5.2", "name": "GPT 5.2", "provider": "OpenAI", "description": "Latest GPT model"},
    {"id": "claude-4", "name": "Claude 4", "provider": "Anthropic", "description": "Latest Claude model"},
    {"id": "gemini-pro", "name": "Gemini Pro", "provider": "Google", "description": "Google Gemini Pro"},
]


class TestModelsDefault:
    """Tests for model default and constants."""

    def test_get_default_model(self):
        """Test get_default_model returns the expected default."""
        assert get_default_model() == "gpt-5.2"

    def test_default_model_constant(self):
        """Test DEFAULT_MODEL constant value."""
        assert DEFAULT_MODEL == "gpt-5.2"


class TestFetchModels:
    """Tests for fetch_models function."""

    def setup_method(self):
        """Reset the cached models before each test."""
        import featcopilot.utils.models as models_mod

        models_mod._cached_models = None

    @patch("featcopilot.utils.models._fetch_models_from_copilot", new_callable=AsyncMock)
    def test_fetch_models(self, mock_fetch):
        """Test fetch_models returns models from the async fetcher."""
        mock_fetch.return_value = FAKE_MODELS
        result = fetch_models(force_refresh=True)
        assert len(result) == 3
        assert result[0]["id"] == "gpt-5.2"

    @patch("featcopilot.utils.models._fetch_models_from_copilot", new_callable=AsyncMock)
    def test_fetch_models_caching(self, mock_fetch):
        """Test fetch_models uses cache on subsequent calls."""
        mock_fetch.return_value = FAKE_MODELS
        first = fetch_models(force_refresh=True)
        second = fetch_models(force_refresh=False)
        assert first == second
        # The async function should only be called once (cached on second call)
        assert mock_fetch.call_count == 1

    @patch("featcopilot.utils.models._fetch_models_from_copilot", new_callable=AsyncMock)
    def test_fetch_models_force_refresh(self, mock_fetch):
        """Test fetch_models with force_refresh bypasses cache."""
        mock_fetch.return_value = FAKE_MODELS
        fetch_models(force_refresh=True)
        fetch_models(force_refresh=True)
        assert mock_fetch.call_count == 2


class TestListModels:
    """Tests for list_models function."""

    def setup_method(self):
        """Reset the cached models before each test."""
        import featcopilot.utils.models as models_mod

        models_mod._cached_models = None

    @patch("featcopilot.utils.models._fetch_models_from_copilot", new_callable=AsyncMock)
    def test_list_models_no_filter(self, mock_fetch):
        """Test list_models returns all models without a filter."""
        mock_fetch.return_value = FAKE_MODELS
        result = list_models(force_refresh=True)
        assert len(result) == 3

    @patch("featcopilot.utils.models._fetch_models_from_copilot", new_callable=AsyncMock)
    def test_list_models_provider_filter(self, mock_fetch):
        """Test list_models filters by provider."""
        mock_fetch.return_value = FAKE_MODELS
        result = list_models(provider="Anthropic", force_refresh=True)
        assert len(result) == 1
        assert result[0]["id"] == "claude-4"

    @patch("featcopilot.utils.models._fetch_models_from_copilot", new_callable=AsyncMock)
    def test_list_models_provider_case_insensitive(self, mock_fetch):
        """Test list_models provider filter is case-insensitive."""
        mock_fetch.return_value = FAKE_MODELS
        result = list_models(provider="openai", force_refresh=True)
        assert len(result) == 1
        assert result[0]["id"] == "gpt-5.2"

    @patch("featcopilot.utils.models._fetch_models_from_copilot", new_callable=AsyncMock)
    def test_list_models_verbose(self, mock_fetch):
        """Test list_models with verbose=True invokes _print_models."""
        mock_fetch.return_value = FAKE_MODELS
        with patch("featcopilot.utils.models._print_models") as mock_print:
            list_models(verbose=True, force_refresh=True)
            mock_print.assert_called_once()


class TestGetModelInfo:
    """Tests for get_model_info function."""

    def setup_method(self):
        """Reset the cached models before each test."""
        import featcopilot.utils.models as models_mod

        models_mod._cached_models = None

    @patch("featcopilot.utils.models._fetch_models_from_copilot", new_callable=AsyncMock)
    def test_get_model_info_existing(self, mock_fetch):
        """Test get_model_info returns info for an existing model."""
        mock_fetch.return_value = FAKE_MODELS
        info = get_model_info("claude-4", force_refresh=True)
        assert info is not None
        assert info["id"] == "claude-4"
        assert info["provider"] == "Anthropic"

    @patch("featcopilot.utils.models._fetch_models_from_copilot", new_callable=AsyncMock)
    def test_get_model_info_nonexistent(self, mock_fetch):
        """Test get_model_info returns None for an unknown model."""
        mock_fetch.return_value = FAKE_MODELS
        info = get_model_info("nonexistent-model", force_refresh=True)
        assert info is None


class TestGetModelNames:
    """Tests for get_model_names function."""

    def setup_method(self):
        """Reset the cached models before each test."""
        import featcopilot.utils.models as models_mod

        models_mod._cached_models = None

    @patch("featcopilot.utils.models._fetch_models_from_copilot", new_callable=AsyncMock)
    def test_get_model_names(self, mock_fetch):
        """Test get_model_names returns a list of model id strings."""
        mock_fetch.return_value = FAKE_MODELS
        names = get_model_names(force_refresh=True)
        assert sorted(names) == ["claude-4", "gemini-pro", "gpt-5.2"]


class TestIsValidModel:
    """Tests for is_valid_model function."""

    def setup_method(self):
        """Reset the cached models before each test."""
        import featcopilot.utils.models as models_mod

        models_mod._cached_models = None

    @patch("featcopilot.utils.models._fetch_models_from_copilot", new_callable=AsyncMock)
    def test_is_valid_model_true(self, mock_fetch):
        """Test is_valid_model returns True for a known model."""
        mock_fetch.return_value = FAKE_MODELS
        assert is_valid_model("gpt-5.2", force_refresh=True) is True

    @patch("featcopilot.utils.models._fetch_models_from_copilot", new_callable=AsyncMock)
    def test_is_valid_model_false(self, mock_fetch):
        """Test is_valid_model returns False for an unknown model."""
        mock_fetch.return_value = FAKE_MODELS
        assert is_valid_model("unknown-model", force_refresh=True) is False

    @patch("featcopilot.utils.models._fetch_models_from_copilot", new_callable=AsyncMock)
    def test_is_valid_model_empty_list_returns_true(self, mock_fetch):
        """Test is_valid_model returns True when no models are fetched (fallback)."""
        mock_fetch.return_value = []
        assert is_valid_model("anything", force_refresh=True) is True


class TestPrintModels:
    """Tests for _print_models helper."""

    def test_print_models_empty(self):
        """Test _print_models with an empty list logs 'No models found'."""
        with patch("featcopilot.utils.models.logger") as mock_logger:
            _print_models([])
            mock_logger.info.assert_called_once()
            assert "No models found" in mock_logger.info.call_args[0][0]

    def test_print_models_non_empty(self):
        """Test _print_models with models logs model information."""
        with patch("featcopilot.utils.models.logger") as mock_logger:
            _print_models(FAKE_MODELS)
            # Should log header, separator, each model, separator, total
            assert mock_logger.info.call_count >= len(FAKE_MODELS) + 3


class TestGetEventLoop:
    """Tests for _get_event_loop helper."""

    def test_get_event_loop_returns_loop(self):
        """Test _get_event_loop returns an asyncio event loop."""
        loop = _get_event_loop()
        assert isinstance(loop, asyncio.AbstractEventLoop)
        assert not loop.is_closed()
