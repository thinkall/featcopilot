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

    def test_set_params_unknown_key_raises(self):
        """set_params should reject unknown parameter names (sklearn convention)."""
        afe = AutoFeatureEngineer()
        with pytest.raises(ValueError, match="Invalid parameter"):
            afe.set_params(not_a_real_param=42)

    def test_set_params_unknown_key_does_not_mutate_state(self):
        """A failing set_params call must leave the estimator unchanged."""
        afe = AutoFeatureEngineer(engines=["tabular"], max_features=5)
        with pytest.raises(ValueError):
            afe.set_params(typo_param=99)

        assert afe.engines == ["tabular"]
        assert afe.max_features == 5
        assert not hasattr(afe, "typo_param")

    def test_sklearn_clone_round_trip(self):
        """A cloned estimator must be configurable identically to the original."""
        from sklearn.base import clone

        afe = AutoFeatureEngineer(engines=["tabular"], max_features=7)
        cloned = clone(afe)

        assert cloned.engines == ["tabular"]
        assert cloned.max_features == 7
        assert cloned.selection_methods == ["mutual_info", "importance"]

    def test_set_params_invalid_value_rolls_back_state(self, sample_df):
        """A failing set_params call must leave every parameter at its pre-call value."""
        afe = AutoFeatureEngineer(
            engines=["tabular"],
            max_features=5,
            selection_methods=["mutual_info"],
            correlation_threshold=0.9,
            llm_config={"model": "gpt-5.2"},
            verbose=False,
            leakage_guard="warn",
        )

        with pytest.raises(ValueError):
            afe.set_params(
                max_features=10,
                engines=["bogus_engine"],
                leakage_guard="not_a_mode",
            )

        # Every parameter that was part of the failing call must be restored.
        assert afe.engines == ["tabular"]
        assert afe.max_features == 5
        assert afe.leakage_guard == "warn"
        # Untouched parameters are obviously unchanged but assert anyway to
        # guard against unrelated mutations.
        assert afe.selection_methods == ["mutual_info"]
        assert afe.correlation_threshold == 0.9
        assert afe.llm_config == {"model": "gpt-5.2"}
        assert afe.verbose is False

    def test_set_params_invalid_value_after_none_normalization_rolls_back(self):
        """Rollback must capture the pre-call value, not the None-normalized one."""
        afe = AutoFeatureEngineer(engines=["tabular", "timeseries"])

        with pytest.raises(ValueError):
            afe.set_params(engines=None, max_features=-1)

        # engines was None-normalized to ["tabular"] mid-call; rollback must
        # restore the original ["tabular", "timeseries"], not ["tabular"].
        assert afe.engines == ["tabular", "timeseries"]
        assert afe.max_features is None

    def test_validate_engines_rejects_non_string_entries(self):
        """Mixed-type engine lists must raise ValueError, not TypeError from sorted()."""
        # A bare ``sorted(set(...))`` over a mix of None/str would raise
        # ``TypeError: '<' not supported between instances of 'str' and 'NoneType'``.
        # The validator must surface a clear ValueError instead.
        with pytest.raises(ValueError, match="engines must contain only strings"):
            AutoFeatureEngineer(engines=[None, "tabular"])
        with pytest.raises(ValueError, match="engines must contain only strings"):
            AutoFeatureEngineer(engines=["tabular", 42])

    def test_validate_selection_methods_rejects_non_string_entries(self):
        """Mixed-type selection_methods lists must raise ValueError, not TypeError from sorted()."""
        with pytest.raises(ValueError, match="selection_methods must contain only strings"):
            AutoFeatureEngineer(selection_methods=[None, "mutual_info"])
        with pytest.raises(ValueError, match="selection_methods must contain only strings"):
            AutoFeatureEngineer(selection_methods=["mutual_info", 0])

    def test_set_params_rejects_non_string_engine_entries_and_rolls_back(self):
        """set_params must surface the same ValueError and roll back state."""
        afe = AutoFeatureEngineer(engines=["tabular"])
        with pytest.raises(ValueError, match="engines must contain only strings"):
            afe.set_params(engines=[None, "spaceship"])
        assert afe.engines == ["tabular"]

    def test_init_rejects_empty_engines_list(self):
        """An explicitly empty ``engines=[]`` must raise rather than silently no-op."""
        # ``engines=None`` defaults to ['tabular']; an explicit empty list is a
        # different intent and would otherwise let ``fit()`` mark the estimator
        # fitted with zero engines so that ``transform()`` becomes a silent no-op.
        with pytest.raises(ValueError, match="engines must contain at least one engine"):
            AutoFeatureEngineer(engines=[])

    def test_init_rejects_empty_selection_methods_list(self):
        """An explicitly empty ``selection_methods=[]`` must raise."""
        with pytest.raises(ValueError, match="selection_methods must contain at least one method"):
            AutoFeatureEngineer(selection_methods=[])

    def test_set_params_rejects_empty_engines_and_rolls_back(self):
        """set_params must reject empty ``engines=[]`` and leave state untouched."""
        afe = AutoFeatureEngineer(engines=["tabular"], max_features=5)
        with pytest.raises(ValueError, match="engines must contain at least one engine"):
            afe.set_params(engines=[])
        assert afe.engines == ["tabular"]
        assert afe.max_features == 5

    def test_init_engines_none_still_defaults_to_tabular(self):
        """``engines=None`` continues to normalize to the default ['tabular']."""
        afe = AutoFeatureEngineer(engines=None)
        assert afe.engines == ["tabular"]

    def test_init_rejects_string_engines_argument(self):
        """A bare ``str`` for ``engines`` must raise instead of iterating char-by-char."""
        # Without the container-type guard, ``engines="tabular"`` would expand
        # into individual characters and produce a confusing "Unknown engines"
        # error such as ``Unknown engines: ['a', 'b', 'l', 'r', 't', 'u']``.
        with pytest.raises(ValueError, match="engines must be a list or tuple of strings"):
            AutoFeatureEngineer(engines="tabular")

    def test_init_rejects_non_sequence_engines_argument(self):
        """Non-sequence ``engines`` (e.g. ``int``) must raise a clear ValueError."""
        # Without the guard, ``set(self.engines)`` would raise a bare
        # ``TypeError: 'int' object is not iterable``.
        with pytest.raises(ValueError, match="engines must be a list or tuple of strings"):
            AutoFeatureEngineer(engines=5)
        with pytest.raises(ValueError, match="engines must be a list or tuple of strings"):
            AutoFeatureEngineer(engines={"tabular": True})

    def test_init_rejects_string_selection_methods_argument(self):
        """A bare ``str`` for ``selection_methods`` must raise."""
        with pytest.raises(ValueError, match="selection_methods must be a list or tuple of strings"):
            AutoFeatureEngineer(selection_methods="mutual_info")

    def test_init_rejects_non_sequence_selection_methods_argument(self):
        """Non-sequence ``selection_methods`` must raise a clear ValueError."""
        with pytest.raises(ValueError, match="selection_methods must be a list or tuple of strings"):
            AutoFeatureEngineer(selection_methods=42)

    def test_init_accepts_tuple_engines(self):
        """Tuples of strings are an acceptable container for ``engines``."""
        afe = AutoFeatureEngineer(engines=("tabular",))
        assert afe.engines == ("tabular",)

    def test_set_params_rejects_string_engines_and_rolls_back(self):
        """``set_params`` inherits the container-type check and rolls back on failure."""
        afe = AutoFeatureEngineer(engines=["tabular"], max_features=5)
        with pytest.raises(ValueError, match="engines must be a list or tuple of strings"):
            afe.set_params(engines="tabular")
        assert afe.engines == ["tabular"]
        assert afe.max_features == 5

    def test_fit_accepts_non_string_target_name(self, sample_df, sample_target):
        """``target_name`` is typed Optional[Any]; integer column labels must work."""
        # Build a DataFrame with an integer column name that overlaps the target.
        df = sample_df.copy()
        df.columns = [0, 1, 2, 3]
        afe = AutoFeatureEngineer(engines=["tabular"], leakage_guard="raise")
        # Integer target_name=0 must be honored (and would raise here because
        # leakage_guard="raise" + a column named 0). This pins the type-hint
        # contract: non-string target labels are accepted at runtime.
        with pytest.raises(ValueError, match="leakage-prone"):
            afe.fit(df, sample_target, target_name=0)

    def test_fit_resets_engine_instances_when_engines_change(self, sample_df, sample_target):
        """Refitting after removing an engine must drop the previously fitted engine."""
        afe = AutoFeatureEngineer(engines=["tabular", "timeseries"], verbose=False)
        afe.fit(sample_df, sample_target)
        assert set(afe._engine_instances) == {"tabular", "timeseries"}

        afe.set_params(engines=["tabular"])
        afe.fit(sample_df, sample_target)

        # The previously fitted "timeseries" engine must not survive into the
        # new fit, otherwise transform() would invoke a stale engine.
        assert set(afe._engine_instances) == {"tabular"}

    def test_fit_resets_selector_after_prior_fit_transform(self, sample_df, sample_target):
        """A plain fit() following fit_transform() must clear the selector."""
        afe = AutoFeatureEngineer(engines=["tabular"], max_features=3, verbose=False)
        afe.fit_transform(sample_df, sample_target)
        assert afe._selector is not None

        afe.fit(sample_df, sample_target)

        # Without a selector reset, transform() would still apply the stale
        # selection from the previous fit_transform call.
        assert afe._selector is None
        result = afe.transform(sample_df)
        # Every input column must survive transform when no selector is active.
        for col in sample_df.columns:
            assert col in result.columns

    def test_fit_resets_state_when_called_after_failed_fit(self, sample_df, sample_target, monkeypatch):
        """If fit raises mid-flight, _is_fitted must be False so transform errors out."""
        afe = AutoFeatureEngineer(engines=["tabular"], verbose=False)
        afe.fit(sample_df, sample_target)
        assert afe._is_fitted is True

        # Force the next fit to fail partway through engine fitting.
        from featcopilot.engines.tabular import TabularEngine

        def _boom(self, X, y=None, **kwargs):
            raise RuntimeError("simulated engine failure")

        monkeypatch.setattr(TabularEngine, "fit", _boom)
        with pytest.raises(RuntimeError, match="simulated engine failure"):
            afe.fit(sample_df, sample_target)

        # The failed fit must not leave the estimator in a "fitted" state
        # that points at stale engines from the previous successful fit.
        assert afe._is_fitted is False
        assert afe._engine_instances == {}
        with pytest.raises(RuntimeError, match="Must call fit"):
            afe.transform(sample_df)


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


class TestDoNoHarmGate:
    """Tests for AutoFeatureEngineer._do_no_harm_gate."""

    def _make_engineered_frame(self, original_cols, derived_cols, n_rows=200, seed=0):
        """Build a (X_engineered, original_features) pair with the given columns."""
        rng = np.random.default_rng(seed)
        data = {col: rng.standard_normal(n_rows) for col in original_cols + derived_cols}
        return pd.DataFrame(data), set(original_cols)

    def _make_fitted_engineer(self):
        """Create an AutoFeatureEngineer with a stub selector so the gate gating allows entry."""
        afe = AutoFeatureEngineer(engines=["tabular"], max_features=5, verbose=False)
        # The do-no-harm gate doesn't query the selector contents, only its presence,
        # so a lightweight stub is sufficient for unit-testing the gate in isolation.
        afe._selector = MagicMock()
        afe._selector._selected_features = []
        return afe

    def test_gate_keeps_features_when_derived_help(self):
        """Derived features that perfectly predict y must be kept."""
        rng = np.random.default_rng(0)
        n = 200
        # Original features are pure noise, derived feature equals y -> obvious win
        y = rng.integers(0, 2, size=n)
        X_engineered = pd.DataFrame(
            {
                "orig1": rng.standard_normal(n),
                "orig2": rng.standard_normal(n),
                "derived_signal": y.astype(float),
            }
        )
        original_features = {"orig1", "orig2"}

        afe = self._make_fitted_engineer()
        selector_before = afe._selector

        result = afe._do_no_harm_gate(X_engineered, X_engineered[list(original_features)], y, original_features)

        # All engineered columns survive when derived features clearly help.
        assert "derived_signal" in result.columns
        assert set(result.columns) == set(X_engineered.columns)
        # Selector instance is preserved (not cleared).
        assert afe._selector is selector_before

    def test_gate_falls_back_when_derived_do_not_help(self):
        """When derived features add no signal, gate falls back to original-only."""
        rng = np.random.default_rng(1)
        n = 200
        # Original feature predicts y; derived features are pure noise (no benefit).
        signal = rng.standard_normal(n)
        y = (signal > 0).astype(int)
        X_engineered = pd.DataFrame(
            {
                "orig_signal": signal,
                "derived_noise1": rng.standard_normal(n),
                "derived_noise2": rng.standard_normal(n),
                "derived_noise3": rng.standard_normal(n),
            }
        )
        original_features = {"orig_signal"}

        afe = self._make_fitted_engineer()
        selector_before = afe._selector

        result = afe._do_no_harm_gate(X_engineered, X_engineered[list(original_features)], y, original_features)

        # Only original columns remain in the returned frame.
        assert list(result.columns) == ["orig_signal"]
        # Selector is preserved (not cleared) and rewritten to original-only,
        # so subsequent transform() calls stay consistent with this fall-back.
        assert afe._selector is selector_before
        assert afe._selector._selected_features == ["orig_signal"]

    def test_gate_handles_float_encoded_binary_target(self):
        """Float-encoded {0.0, 1.0} targets must be detected as classification, not regression."""
        rng = np.random.default_rng(2)
        n = 200
        y_float = rng.integers(0, 2, size=n).astype(float)  # {0.0, 1.0}
        X_engineered = pd.DataFrame(
            {
                "orig1": rng.standard_normal(n),
                "derived1": y_float + rng.standard_normal(n) * 0.01,  # near-perfect predictor
            }
        )
        original_features = {"orig1"}

        afe = self._make_fitted_engineer()

        # Capture which model class the gate uses by patching both.
        with (
            patch("sklearn.ensemble.RandomForestClassifier") as mock_clf,
            patch("sklearn.ensemble.RandomForestRegressor") as mock_reg,
        ):
            # Make the mocks behave like a real estimator so the gate can finish.
            inst = MagicMock()
            inst.score.return_value = 0.5
            mock_clf.return_value = inst
            mock_reg.return_value = inst

            afe._do_no_harm_gate(X_engineered, X_engineered[list(original_features)], y_float, original_features)

        # type_of_target() on float {0.0, 1.0} returns "binary" -> classifier path.
        assert mock_clf.called
        assert not mock_reg.called

    def test_gate_skipped_when_no_selector(self, sample_df, sample_target):
        """When max_features=None (no selection), gate must not run."""
        afe = AutoFeatureEngineer(engines=["tabular"], max_features=None, verbose=False)

        with patch.object(AutoFeatureEngineer, "_do_no_harm_gate") as mock_gate:
            afe.fit_transform(sample_df, sample_target, apply_selection=True)

            # Selector was never created, so the gate must be bypassed.
            assert afe._selector is None
            assert not mock_gate.called

    def test_gate_runs_when_selector_fitted(self, sample_df, sample_target):
        """When selection actually runs, the gate is invoked."""
        afe = AutoFeatureEngineer(engines=["tabular"], max_features=5, verbose=False)

        with patch.object(
            AutoFeatureEngineer, "_do_no_harm_gate", side_effect=lambda result, *a, **k: result
        ) as mock_gate:
            afe.fit_transform(sample_df, sample_target, apply_selection=True)

            assert afe._selector is not None
            assert mock_gate.called

    def test_transform_after_fallback_returns_original_only(self):
        """After gate fall-back, transform() must emit the same original-only column set."""
        rng = np.random.default_rng(3)
        n = 200
        df = pd.DataFrame(
            {
                "num1": rng.standard_normal(n),
                "num2": rng.standard_normal(n),
                "num3": rng.standard_normal(n),
            }
        )
        # Random noise target so derived features cannot help -> gate should fall back.
        y = pd.Series(rng.integers(0, 2, size=n))

        afe = AutoFeatureEngineer(engines=["tabular"], max_features=10, verbose=False)
        result_fit = afe.fit_transform(df, y, apply_selection=True)

        # If the gate triggered fall-back, the selector should remain (not be None)
        # and transform() should produce a column set consistent with fit_transform.
        result_transform = afe.transform(df)
        assert list(result_transform.columns) == list(result_fit.columns)
