"""Tests for scikit-learn compatible feature engineering transformers."""

import importlib
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from featcopilot.transformers.sklearn_compat import AutoFeatureEngineer, FeatureEngineerTransformer


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "num1": np.random.randn(100),
            "num2": np.random.randn(100) * 10,
            "num3": np.random.randint(1, 100, 100),
            "cat1": pd.array(np.random.choice(["A", "B", "C"], 100), dtype="object"),
        }
    )


@pytest.fixture
def sample_target():
    """Create sample target variable."""
    np.random.seed(42)
    return pd.Series(np.random.randint(0, 2, 100), name="target")


@pytest.fixture
def sample_ndarray():
    """Create sample ndarray input."""
    np.random.seed(42)
    return np.random.randn(100, 3)


class TestFeatureEngineerTransformer:
    """Tests for FeatureEngineerTransformer."""

    def test_init_defaults(self):
        """Test default initialization."""
        transformer = FeatureEngineerTransformer()
        assert transformer.engine == "tabular"
        assert transformer.engine_kwargs == {}
        assert transformer._engine_instance is None

    def test_init_with_engine(self):
        """Test initialization with specific engine."""
        transformer = FeatureEngineerTransformer(engine="timeseries", window_size=5)
        assert transformer.engine == "timeseries"
        assert transformer.engine_kwargs == {"window_size": 5}

    def test_fit_tabular(self, sample_df):
        """Test fitting with tabular engine."""
        transformer = FeatureEngineerTransformer(engine="tabular")
        result = transformer.fit(sample_df)

        assert result is transformer
        assert transformer._engine_instance is not None

    def test_transform_tabular(self, sample_df):
        """Test transforming with tabular engine."""
        transformer = FeatureEngineerTransformer(engine="tabular")
        transformer.fit(sample_df)
        result = transformer.transform(sample_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100
        assert len(result.columns) >= len(sample_df.columns)

    def test_get_feature_names_out_after_fit(self, sample_df):
        """Test getting feature names after fit."""
        transformer = FeatureEngineerTransformer(engine="tabular")
        transformer.fit(sample_df)
        names = transformer.get_feature_names_out()

        assert isinstance(names, list)

    def test_transform_before_fit_raises(self, sample_df):
        """Test that transform before fit raises RuntimeError."""
        transformer = FeatureEngineerTransformer(engine="tabular")

        with pytest.raises(RuntimeError, match="Transformer must be fitted before transform"):
            transformer.transform(sample_df)

    def test_get_feature_names_out_before_fit(self):
        """Test that get_feature_names_out before fit returns empty list."""
        transformer = FeatureEngineerTransformer(engine="tabular")
        result = transformer.get_feature_names_out()

        assert result == []

    def test_unknown_engine_raises(self, sample_df):
        """Test that unknown engine raises ValueError."""
        transformer = FeatureEngineerTransformer(engine="nonexistent")

        with pytest.raises(ValueError, match="Unknown engine: nonexistent"):
            transformer.fit(sample_df)

    def test_create_engine_types(self):
        """Test that _create_engine creates correct engine types."""
        from featcopilot.engines.tabular import TabularEngine

        transformer = FeatureEngineerTransformer(engine="tabular")
        engine = transformer._create_engine()
        assert isinstance(engine, TabularEngine)


class TestAutoFeatureEngineer:
    """Tests for AutoFeatureEngineer."""

    def test_init_defaults(self):
        """Test default initialization."""
        afe = AutoFeatureEngineer()
        assert afe.engines == ["tabular"]
        assert afe.max_features is None
        assert afe.selection_methods == ["mutual_info", "importance"]
        assert afe.correlation_threshold == 0.85
        assert afe.llm_config == {}
        assert afe.verbose is False
        assert afe._is_fitted is False

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        afe = AutoFeatureEngineer(
            engines=["tabular", "timeseries"],
            max_features=50,
            selection_methods=["importance"],
            correlation_threshold=0.9,
            llm_config={"model": "gpt-5.2"},
            verbose=True,
        )
        assert afe.engines == ["tabular", "timeseries"]
        assert afe.max_features == 50
        assert afe.correlation_threshold == 0.9
        assert afe.llm_config == {"model": "gpt-5.2"}
        assert afe.verbose is True

    def test_fit_tabular(self, sample_df, sample_target):
        """Test fitting with tabular engine."""
        afe = AutoFeatureEngineer(engines=["tabular"])
        result = afe.fit(sample_df, sample_target)

        assert result is afe
        assert afe._is_fitted is True
        assert "tabular" in afe._engine_instances

    def test_transform_tabular(self, sample_df, sample_target):
        """Test transforming with tabular engine."""
        afe = AutoFeatureEngineer(engines=["tabular"])
        afe.fit(sample_df, sample_target)
        result = afe.transform(sample_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100

    def test_transform_before_fit_raises(self, sample_df):
        """Test that transform before fit raises RuntimeError."""
        afe = AutoFeatureEngineer()

        with pytest.raises(RuntimeError, match="Must call fit before transform"):
            afe.transform(sample_df)

    def test_fit_with_ndarray(self, sample_ndarray, sample_target):
        """Test fitting with ndarray input."""
        afe = AutoFeatureEngineer(engines=["tabular"])
        afe.fit(sample_ndarray, sample_target)

        assert afe._is_fitted is True

    def test_transform_with_ndarray(self, sample_ndarray, sample_target):
        """Test transforming with ndarray input."""
        afe = AutoFeatureEngineer(engines=["tabular"])
        afe.fit(sample_ndarray, sample_target)
        result = afe.transform(sample_ndarray)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100

    def test_fit_transform_no_selection(self, sample_df, sample_target):
        """Test fit_transform without selection."""
        afe = AutoFeatureEngineer(engines=["tabular"])
        result = afe.fit_transform(sample_df, sample_target, apply_selection=False)

        assert isinstance(result, pd.DataFrame)
        assert afe._is_fitted is True
        assert afe._selector is None

    def test_fit_transform_with_selection(self, sample_df, sample_target):
        """Test fit_transform with selection enabled, y provided, and max_features set."""
        afe = AutoFeatureEngineer(engines=["tabular"], max_features=5)
        result = afe.fit_transform(sample_df, sample_target, apply_selection=True)

        assert isinstance(result, pd.DataFrame)
        assert afe._selector is not None

    def test_fit_transform_selection_without_y(self, sample_df):
        """Test fit_transform with selection but no y - selection should not apply."""
        afe = AutoFeatureEngineer(engines=["tabular"], max_features=5)
        result = afe.fit_transform(sample_df, y=None, apply_selection=True)

        assert isinstance(result, pd.DataFrame)
        assert afe._selector is None

    def test_fit_transform_selection_without_max_features(self, sample_df, sample_target):
        """Test fit_transform with selection but no max_features - selection should not apply."""
        afe = AutoFeatureEngineer(engines=["tabular"], max_features=None)
        result = afe.fit_transform(sample_df, sample_target, apply_selection=True)

        assert isinstance(result, pd.DataFrame)
        assert afe._selector is None

    def test_get_feature_names(self, sample_df, sample_target):
        """Test get_feature_names returns list of feature names."""
        afe = AutoFeatureEngineer(engines=["tabular"])
        afe.fit(sample_df, sample_target)
        names = afe.get_feature_names()

        assert isinstance(names, list)

    def test_get_feature_names_out(self, sample_df, sample_target):
        """Test sklearn-compatible get_feature_names_out."""
        afe = AutoFeatureEngineer(engines=["tabular"])
        afe.fit(sample_df, sample_target)
        names = afe.get_feature_names_out()

        assert names == afe.get_feature_names()

    def test_get_params(self):
        """Test get_params returns correct parameters."""
        afe = AutoFeatureEngineer(
            engines=["tabular"],
            max_features=10,
            correlation_threshold=0.9,
            verbose=True,
        )
        params = afe.get_params()

        assert params["engines"] == ["tabular"]
        assert params["max_features"] == 10
        assert params["correlation_threshold"] == 0.9
        assert params["verbose"] is True
        assert "selection_methods" in params
        assert "llm_config" in params

    def test_get_params_deep(self):
        """Test get_params with deep=True."""
        afe = AutoFeatureEngineer()
        params = afe.get_params(deep=True)

        assert isinstance(params, dict)

    def test_set_params(self):
        """Test set_params updates parameters."""
        afe = AutoFeatureEngineer()
        result = afe.set_params(max_features=20, verbose=True)

        assert result is afe
        assert afe.max_features == 20
        assert afe.verbose is True

    def test_set_params_none_normalizes_to_defaults(self):
        """set_params should accept None for collection-valued params (sklearn compat).

        Sklearn's ``clone`` and ``GridSearchCV`` may pass ``None`` for parameters
        whose default in ``__init__`` is also ``None``. Validation should not raise
        in that case; ``None`` should be normalized to the same defaults
        ``__init__`` applies.
        """
        afe = AutoFeatureEngineer(engines=["tabular", "timeseries"])
        afe.set_params(engines=None, selection_methods=None, llm_config=None)

        assert afe.engines == ["tabular"]
        assert afe.selection_methods == ["mutual_info", "importance"]
        assert afe.llm_config == {}

    def test_set_params_invalid_engine_still_raises(self):
        """set_params should still validate non-None values."""
        afe = AutoFeatureEngineer()
        with pytest.raises(ValueError, match="Unknown engines"):
            afe.set_params(engines=["not_a_real_engine"])

    def test_sklearn_clone_round_trip(self):
        """A cloned estimator must be configurable identically to the original."""
        from sklearn.base import clone

        afe = AutoFeatureEngineer(engines=["tabular"], max_features=7)
        cloned = clone(afe)

        assert cloned.engines == ["tabular"]
        assert cloned.max_features == 7
        assert cloned.selection_methods == ["mutual_info", "importance"]


class TestPackageImport:
    """Tests for top-level package import behavior."""

    def test_import_without_installed_metadata_falls_back(self):
        """Test source import works even when distribution metadata is unavailable."""
        import importlib.metadata as importlib_metadata

        import featcopilot

        original_version = importlib_metadata.version

        def fake_version(name):
            if name == "featcopilot":
                raise importlib_metadata.PackageNotFoundError
            return original_version(name)

        with patch("importlib.metadata.version", side_effect=fake_version):
            reloaded = importlib.reload(featcopilot)
            assert reloaded.__version__ == "0+unknown"

        importlib.reload(featcopilot)

    def test_verbose_logging(self, sample_df, sample_target):
        """Test that verbose=True does not error."""
        afe = AutoFeatureEngineer(engines=["tabular"], verbose=True)
        afe.fit(sample_df, sample_target)
        result = afe.transform(sample_df)

        assert isinstance(result, pd.DataFrame)

    def test_verbose_fit_transform_with_selection(self, sample_df, sample_target):
        """Test verbose logging during fit_transform with selection."""
        afe = AutoFeatureEngineer(engines=["tabular"], max_features=5, verbose=True)
        result = afe.fit_transform(sample_df, sample_target, apply_selection=True)

        assert isinstance(result, pd.DataFrame)

    def test_feature_importances_none_without_selector(self, sample_df, sample_target):
        """Test feature_importances_ is None when no selector fitted."""
        afe = AutoFeatureEngineer(engines=["tabular"])
        afe.fit(sample_df, sample_target)

        assert afe.feature_importances_ is None

    def test_feature_importances_with_selector(self, sample_df, sample_target):
        """Test feature_importances_ returns dict when selector is fitted."""
        afe = AutoFeatureEngineer(engines=["tabular"], max_features=5)
        afe.fit_transform(sample_df, sample_target, apply_selection=True)

        importances = afe.feature_importances_
        assert importances is not None

    def test_explain_features_with_get_feature_explanations(self, sample_df, sample_target):
        """Test explain_features when engine has get_feature_explanations."""
        afe = AutoFeatureEngineer(engines=["tabular"])
        afe.fit(sample_df, sample_target)

        mock_engine = MagicMock()
        mock_engine.get_feature_explanations.return_value = {"feat1": "explanation1"}
        afe._engine_instances["mock"] = mock_engine

        explanations = afe.explain_features()
        assert "feat1" in explanations
        assert explanations["feat1"] == "explanation1"

    def test_explain_features_with_get_feature_set(self, sample_df, sample_target):
        """Test explain_features when engine only has get_feature_set."""
        afe = AutoFeatureEngineer(engines=["tabular"])
        afe.fit(sample_df, sample_target)

        mock_engine = MagicMock(spec=[])
        mock_feature_set = MagicMock()
        mock_feature_set.get_explanations.return_value = {"feat2": "explanation2"}
        mock_engine.get_feature_set = MagicMock(return_value=mock_feature_set)

        afe._engine_instances["mock"] = mock_engine

        explanations = afe.explain_features()
        assert "feat2" in explanations

    def test_explain_features_no_methods(self, sample_df, sample_target):
        """Test explain_features when engine has neither method."""
        afe = AutoFeatureEngineer(engines=["tabular"])
        afe.fit(sample_df, sample_target)

        mock_engine = MagicMock(spec=[])
        afe._engine_instances = {"mock": mock_engine}

        explanations = afe.explain_features()
        assert explanations == {}

    def test_get_feature_code(self, sample_df, sample_target):
        """Test get_feature_code collects code from engines."""
        afe = AutoFeatureEngineer(engines=["tabular"])
        afe.fit(sample_df, sample_target)

        mock_engine = MagicMock()
        mock_engine.get_feature_code.return_value = {"feat1": "df['a'] + df['b']"}
        afe._engine_instances["mock"] = mock_engine

        code = afe.get_feature_code()
        assert "feat1" in code
        assert code["feat1"] == "df['a'] + df['b']"

    def test_get_feature_code_no_method(self):
        """Test get_feature_code when engine lacks get_feature_code."""
        afe = AutoFeatureEngineer()
        mock_engine = MagicMock(spec=[])
        afe._engine_instances = {"mock": mock_engine}

        code = afe.get_feature_code()
        assert code == {}

    def test_generate_custom_features_without_llm(self, sample_df, sample_target):
        """Test generate_custom_features raises error without LLM engine."""
        afe = AutoFeatureEngineer(engines=["tabular"])
        afe.fit(sample_df, sample_target)

        with pytest.raises(RuntimeError, match="LLM engine not enabled"):
            afe.generate_custom_features("create ratio features")

    def test_generate_custom_features_with_llm(self):
        """Test generate_custom_features delegates to LLM engine."""
        afe = AutoFeatureEngineer()
        mock_llm = MagicMock()
        mock_llm.suggest_more_features.return_value = [{"name": "feat1", "code": "x+y"}]
        afe._engine_instances["llm"] = mock_llm

        result = afe.generate_custom_features("create ratio features", n_features=3)

        mock_llm.suggest_more_features.assert_called_once_with("create ratio features", 3)
        assert len(result) == 1
        assert result[0]["name"] == "feat1"

    def test_create_engine_unknown_raises(self):
        """Test _create_engine raises ValueError for unknown engine."""
        afe = AutoFeatureEngineer()

        with pytest.raises(ValueError, match="Unknown engine: nonexistent"):
            afe._create_engine("nonexistent")

    @patch("featcopilot.transformers.sklearn_compat.TabularEngine")
    def test_create_engine_tabular(self, mock_tabular_cls):
        """Test _create_engine creates TabularEngine correctly."""
        afe = AutoFeatureEngineer(max_features=10, verbose=True)
        afe._create_engine("tabular")

        mock_tabular_cls.assert_called_once_with(max_features=10, verbose=True)

    def test_create_engine_llm(self):
        """Test _create_engine creates SemanticEngine for 'llm'."""
        mock_semantic = MagicMock()
        with patch(
            "featcopilot.llm.semantic_engine.SemanticEngine",
            mock_semantic,
        ):
            afe = AutoFeatureEngineer(llm_config={"model": "gpt-5.2", "max_suggestions": 10})
            engine = afe._create_engine("llm")

            mock_semantic.assert_called_once()
            call_kwargs = mock_semantic.call_args[1]
            assert call_kwargs["model"] == "gpt-5.2"
            assert call_kwargs["max_suggestions"] == 10

    def test_fit_with_llm_engine(self, sample_df, sample_target):
        """Test fit passes extra args to LLM engine."""
        mock_semantic_cls = MagicMock()
        mock_engine_instance = MagicMock()
        mock_semantic_cls.return_value = mock_engine_instance

        with patch(
            "featcopilot.llm.semantic_engine.SemanticEngine",
            mock_semantic_cls,
        ):
            afe = AutoFeatureEngineer(engines=["llm"])
            afe.fit(
                sample_df,
                sample_target,
                column_descriptions={"num1": "first number"},
                task_description="classification",
            )

            mock_engine_instance.fit.assert_called_once()
            call_kwargs = mock_engine_instance.fit.call_args[1]
            assert call_kwargs["column_descriptions"] == {"num1": "first number"}
            assert call_kwargs["task_description"] == "classification"

    def test_fit_stores_descriptions(self, sample_df, sample_target):
        """Test fit stores column descriptions and task description."""
        afe = AutoFeatureEngineer(engines=["tabular"])
        afe.fit(
            sample_df,
            sample_target,
            column_descriptions={"num1": "first column"},
            task_description="binary classification",
        )

        assert afe._column_descriptions == {"num1": "first column"}
        assert afe._task_description == "binary classification"

    def test_transform_with_fitted_selector(self, sample_df, sample_target):
        """Test transform applies selector when fitted."""
        afe = AutoFeatureEngineer(engines=["tabular"], max_features=3)
        afe.fit_transform(sample_df, sample_target, apply_selection=True)

        result = afe.transform(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_fit_transform_column_descriptions(self, sample_df, sample_target):
        """Test fit_transform accepts column_descriptions and task_description."""
        afe = AutoFeatureEngineer(engines=["tabular"])
        result = afe.fit_transform(
            sample_df,
            sample_target,
            column_descriptions={"num1": "numeric column"},
            task_description="regression",
            apply_selection=False,
        )

        assert isinstance(result, pd.DataFrame)
