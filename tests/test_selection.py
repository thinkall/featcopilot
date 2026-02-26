"""Tests for feature selection."""

import numpy as np
import pandas as pd
import pytest

from featcopilot.selection import (
    FeatureSelector,
    ImportanceSelector,
    RedundancyEliminator,
    StatisticalSelector,
)


class TestStatisticalSelector:
    """Tests for StatisticalSelector."""

    @pytest.fixture
    def classification_data(self):
        """Create classification data."""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame(
            {
                "relevant1": np.random.randn(n),
                "relevant2": np.random.randn(n),
                "noise1": np.random.randn(n),
                "noise2": np.random.randn(n),
            }
        )
        # Make target depend on relevant features
        y = (X["relevant1"] + X["relevant2"] > 0).astype(int)
        return X, y

    def test_mutual_info_selection(self, classification_data):
        """Test mutual information selection."""
        X, y = classification_data
        selector = StatisticalSelector(method="mutual_info", max_features=2)
        X_selected = selector.fit_transform(X, y)

        assert len(selector.get_selected_features()) == 2
        scores = selector.get_feature_scores()
        assert len(scores) == 4

    def test_f_test_selection(self, classification_data):
        """Test F-test selection."""
        X, y = classification_data
        selector = StatisticalSelector(method="f_test", max_features=3)
        X_selected = selector.fit_transform(X, y)

        assert len(selector.get_selected_features()) == 3

    def test_correlation_selection(self, classification_data):
        """Test correlation-based selection."""
        X, y = classification_data
        selector = StatisticalSelector(method="correlation")
        selector.fit(X, y)

        scores = selector.get_feature_scores()
        # Relevant features should have higher correlation
        assert scores["relevant1"] > scores["noise1"] or scores["relevant2"] > scores["noise2"]


class TestImportanceSelector:
    """Tests for ImportanceSelector."""

    @pytest.fixture
    def regression_data(self):
        """Create regression data."""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame(
            {
                "important1": np.random.randn(n),
                "important2": np.random.randn(n),
                "unimportant": np.random.randn(n) * 0.01,
            }
        )
        y = X["important1"] * 2 + X["important2"] + np.random.randn(n) * 0.1
        return X, y

    def test_random_forest_importance(self, regression_data):
        """Test random forest importance selection."""
        X, y = regression_data
        selector = ImportanceSelector(model="random_forest", max_features=2)
        X_selected = selector.fit_transform(X, y)

        selected = selector.get_selected_features()
        assert len(selected) == 2
        # Important features should be selected
        assert "important1" in selected or "important2" in selected


class TestRedundancyEliminator:
    """Tests for RedundancyEliminator."""

    @pytest.fixture
    def correlated_data(self):
        """Create data with correlated features."""
        np.random.seed(42)
        n = 200
        base = np.random.randn(n)
        X = pd.DataFrame(
            {
                "original": base,
                "duplicate": base + np.random.randn(n) * 0.01,  # Nearly identical
                "independent": np.random.randn(n),
            }
        )
        return X

    def test_redundancy_elimination(self, correlated_data):
        """Test eliminating redundant features."""
        eliminator = RedundancyEliminator(correlation_threshold=0.95)
        X_reduced = eliminator.fit_transform(correlated_data)

        # Should keep 2 features (original OR duplicate, plus independent)
        assert len(X_reduced.columns) == 2
        assert "independent" in X_reduced.columns

    def test_redundancy_with_importance(self, correlated_data):
        """Test keeping more important feature."""
        eliminator = RedundancyEliminator(
            correlation_threshold=0.95,
            importance_scores={"original": 0.8, "duplicate": 0.2, "independent": 0.5},
        )
        X_reduced = eliminator.fit_transform(correlated_data)

        # Should keep 'original' over 'duplicate'
        assert "original" in X_reduced.columns
        assert "duplicate" not in X_reduced.columns


class TestFeatureSelector:
    """Tests for unified FeatureSelector."""

    @pytest.fixture
    def mixed_data(self):
        """Create data for testing."""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame({f"feature_{i}": np.random.randn(n) for i in range(10)})
        y = X["feature_0"] + X["feature_1"] + np.random.randn(n) * 0.1
        return X, y

    def test_unified_selector(self, mixed_data):
        """Test unified feature selection."""
        X, y = mixed_data
        selector = FeatureSelector(methods=["mutual_info", "importance"], max_features=5)
        X_selected = selector.fit_transform(X, y)

        assert len(selector.get_selected_features()) <= 5
        assert len(selector.get_selected_features()) >= 1

    def test_method_scores(self, mixed_data):
        """Test getting scores from each method."""
        X, y = mixed_data
        selector = FeatureSelector(methods=["mutual_info", "f_test"])
        selector.fit(X, y)

        method_scores = selector.get_method_scores()
        assert "mutual_info" in method_scores
        assert "f_test" in method_scores


class TestImportanceSelectorCoverage:
    """Additional coverage tests for ImportanceSelector."""

    def test_string_target_label_encoding(self):
        """Test label encoding for string targets."""
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = np.array(["cat", "dog"] * 50)
        selector = ImportanceSelector(model="random_forest", max_features=2)
        X_selected = selector.fit_transform(X, y)
        assert len(X_selected.columns) <= 2

    def test_no_numeric_columns(self):
        """Test with no numeric columns returns zero scores."""
        np.random.seed(42)
        X = pd.DataFrame({"cat1": ["a", "b"] * 50, "cat2": ["x", "y"] * 50})
        y = np.array([0, 1] * 50)
        selector = ImportanceSelector(model="random_forest")
        selector.fit(X, y)
        scores = selector.get_feature_scores()
        assert all(v == 0.0 for v in scores.values())
        assert selector._is_fitted

    def test_zero_importance_non_numeric_cols(self):
        """Test non-numeric columns get zero importance."""
        np.random.seed(42)
        X = pd.DataFrame({"num": np.random.randn(100), "cat": ["a", "b"] * 50})
        y = np.array([0, 1] * 50)
        selector = ImportanceSelector(model="random_forest")
        selector.fit(X, y)
        scores = selector.get_feature_scores()
        assert scores["cat"] == 0.0
        assert scores["num"] >= 0.0

    def test_gradient_boosting_classifier(self):
        """Test GradientBoosting classifier model type."""
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = np.array([0, 1] * 50)
        selector = ImportanceSelector(model="gradient_boosting", max_features=2)
        X_selected = selector.fit_transform(X, y)
        assert len(X_selected.columns) <= 2

    def test_gradient_boosting_regressor(self):
        """Test GradientBoosting regressor model type."""
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = X["a"] * 2 + np.random.randn(100) * 0.1
        selector = ImportanceSelector(model="gradient_boosting", max_features=2)
        X_selected = selector.fit_transform(X, y)
        assert len(X_selected.columns) <= 2

    def test_xgboost_classifier(self):
        """Test XGBoost classifier model type."""
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = np.array([0, 1] * 50)
        selector = ImportanceSelector(model="xgboost", max_features=2)
        X_selected = selector.fit_transform(X, y)
        assert len(X_selected.columns) <= 2

    def test_xgboost_regressor(self):
        """Test XGBoost regressor model type."""
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = X["a"] * 2 + np.random.randn(100) * 0.1
        selector = ImportanceSelector(model="xgboost", max_features=2)
        X_selected = selector.fit_transform(X, y)
        assert len(X_selected.columns) <= 2

    def test_unknown_model_type_raises(self):
        """Test unknown model type raises ValueError."""
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100)})
        y = np.array([0, 1] * 50)
        selector = ImportanceSelector(model="unknown_model")
        with pytest.raises(ValueError, match="Unknown model type"):
            selector.fit(X, y)

    def test_xgboost_fallback_classifier(self):
        """Test fallback to RandomForest classifier when XGBoost unavailable."""
        import sys
        from unittest.mock import patch

        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = np.array([0, 1] * 50)
        selector = ImportanceSelector(model="xgboost", verbose=True)
        with patch.dict(sys.modules, {"xgboost": None}):
            selector.fit(X, y)
        assert selector._is_fitted

    def test_xgboost_fallback_regressor(self):
        """Test fallback to RandomForest regressor when XGBoost unavailable."""
        import sys
        from unittest.mock import patch

        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = X["a"] * 2 + np.random.randn(100) * 0.1
        selector = ImportanceSelector(model="xgboost", verbose=True)
        with patch.dict(sys.modules, {"xgboost": None}):
            selector.fit(X, y)
        assert selector._is_fitted

    def test_threshold_filtering(self):
        """Test threshold-based feature filtering."""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame({"important": np.random.randn(n), "noise": np.random.randn(n) * 0.001})
        y = X["important"] * 2 + np.random.randn(n) * 0.1
        selector = ImportanceSelector(model="random_forest", threshold=0.1)
        selector.fit(X, y)
        scores = selector.get_feature_scores()
        for f in selector.get_selected_features():
            assert scores[f] >= 0.1

    def test_verbose_logging(self):
        """Test verbose logging during selection."""
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = np.array([0, 1] * 50)
        selector = ImportanceSelector(model="random_forest", verbose=True)
        selector.fit(X, y)
        assert selector._is_fitted

    def test_transform_before_fit_raises(self):
        """Test RuntimeError when transform called before fit."""
        selector = ImportanceSelector()
        with pytest.raises(RuntimeError, match="Selector must be fitted"):
            selector.transform(pd.DataFrame({"a": [1, 2, 3]}))


class TestStatisticalSelectorCoverage:
    """Additional coverage tests for StatisticalSelector."""

    def test_invalid_method_raises(self):
        """Test ValueError for invalid method in constructor."""
        with pytest.raises(ValueError, match="Method must be one of"):
            StatisticalSelector(method="invalid")

    def test_chi2_method(self):
        """Test chi2 method branch in fit."""
        np.random.seed(42)
        X = pd.DataFrame({"a": np.abs(np.random.randn(100)), "b": np.abs(np.random.randn(100))})
        y = np.array([0, 1] * 50)
        selector = StatisticalSelector(method="chi2")
        selector.fit(X, y)
        assert selector._is_fitted
        scores = selector.get_feature_scores()
        assert len(scores) == 2

    def test_fit_else_branch(self):
        """Test else branch in fit for unknown method."""
        selector = StatisticalSelector(method="mutual_info")
        selector.method = "invalid"
        with pytest.raises(ValueError, match="Unknown method"):
            selector.fit(pd.DataFrame({"a": [1, 2, 3]}), np.array([0, 1, 0]))

    def test_mutual_info_string_target(self):
        """Test label encoding for string targets in mutual_info."""
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = np.array(["cat", "dog"] * 50)
        selector = StatisticalSelector(method="mutual_info")
        selector.fit(X, y)
        assert selector._is_fitted

    def test_f_test_string_target(self):
        """Test label encoding for string targets in f_test."""
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = np.array(["cat", "dog"] * 50)
        selector = StatisticalSelector(method="f_test")
        selector.fit(X, y)
        assert selector._is_fitted

    def test_correlation_string_target(self):
        """Test label encoding for string targets in correlation."""
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = np.array(["cat", "dog"] * 50)
        selector = StatisticalSelector(method="correlation")
        selector.fit(X, y)
        assert selector._is_fitted

    def test_chi2_string_target(self):
        """Test chi2 with string target for label encoding."""
        np.random.seed(42)
        X = pd.DataFrame({"a": np.abs(np.random.randn(100)), "b": np.abs(np.random.randn(100))})
        y = np.array(["cat", "dog"] * 50)
        selector = StatisticalSelector(method="chi2")
        selector.fit(X, y)
        assert selector._is_fitted

    def test_chi2_computation_with_integers(self):
        """Test chi-square computation with integer features."""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randint(0, 5, 100).astype(float),
                "b": np.random.randint(0, 3, 100).astype(float),
            }
        )
        y = np.array([0, 1] * 50)
        selector = StatisticalSelector(method="chi2")
        selector.fit(X, y)
        scores = selector.get_feature_scores()
        assert all(v >= 0 for v in scores.values())

    def test_correlation_exception_handling(self):
        """Test exception handling in correlation computation."""
        from unittest.mock import patch

        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = np.array([0, 1] * 50)
        selector = StatisticalSelector(method="correlation")
        with patch.object(np, "corrcoef", side_effect=ValueError("test error")):
            selector.fit(X, y)
        scores = selector.get_feature_scores()
        assert all(v == 0.0 for v in scores.values())

    def test_threshold_filtering(self):
        """Test threshold-based filtering in selection."""
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = np.array([0, 1] * 50)
        selector = StatisticalSelector(method="mutual_info", threshold=0.01)
        selector.fit(X, y)
        scores = selector.get_feature_scores()
        for f in selector.get_selected_features():
            assert scores[f] >= 0.01

    def test_verbose_logging(self):
        """Test verbose logging during selection."""
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = np.array([0, 1] * 50)
        selector = StatisticalSelector(method="mutual_info", verbose=True)
        selector.fit(X, y)
        assert selector._is_fitted

    def test_transform_before_fit_raises(self):
        """Test RuntimeError when transform called before fit."""
        selector = StatisticalSelector(method="mutual_info")
        with pytest.raises(RuntimeError, match="Selector must be fitted"):
            selector.transform(pd.DataFrame({"a": [1, 2, 3]}))


class TestFeatureSelectorCoverage:
    """Additional coverage tests for FeatureSelector."""

    def test_categorical_baseline_scoring(self):
        """Test categorical column baseline scoring for original features."""
        np.random.seed(42)
        n = 100
        X = pd.DataFrame(
            {
                "num1": np.random.randn(n),
                "num2": np.random.randn(n),
                "cat1": pd.array(np.random.choice(["a", "b", "c"], n), dtype="object"),
            }
        )
        y = np.array([0, 1] * (n // 2))
        selector = FeatureSelector(
            methods=["mutual_info"],
            original_features={"num1", "num2", "cat1"},
        )
        selector.fit(X, y)
        scores = selector.get_feature_scores()
        assert scores.get("cat1", 0) > 0

    def test_chi2_selector_creation(self):
        """Test chi2 selector creation via FeatureSelector."""
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = np.array([0, 1] * 50)
        selector = FeatureSelector(methods=["chi2"])
        selector.fit(X, y)
        assert "chi2" in selector.get_method_scores()

    def test_correlation_selector_creation(self):
        """Test correlation selector creation via FeatureSelector."""
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = np.array([0, 1] * 50)
        selector = FeatureSelector(methods=["correlation"])
        selector.fit(X, y)
        assert "correlation" in selector.get_method_scores()

    def test_xgboost_selector_creation(self):
        """Test xgboost selector creation via FeatureSelector."""
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = np.array([0, 1] * 50)
        selector = FeatureSelector(methods=["xgboost"])
        selector.fit(X, y)
        assert "xgboost" in selector.get_method_scores()

    def test_unknown_method_raises(self):
        """Test unknown selection method raises ValueError."""
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100)})
        y = np.array([0, 1] * 50)
        selector = FeatureSelector(methods=["unknown_method"])
        with pytest.raises(ValueError, match="Unknown selection method"):
            selector.fit(X, y)

    def test_zero_max_score_normalization(self):
        """Test normalization when all method scores are zero."""
        np.random.seed(42)
        X = pd.DataFrame({"a": np.zeros(100), "b": np.zeros(100)})
        y = np.array([0, 1] * 50)
        selector = FeatureSelector(methods=["correlation"])
        selector.fit(X, y)
        scores = selector.get_feature_scores()
        assert all(v == 0 for v in scores.values())

    def test_combine_scores_missing_column(self):
        """Test combine scores when column not in any method scores."""
        selector = FeatureSelector(methods=["mutual_info"])
        selector._method_scores = {"mutual_info": {"a": 0.5}}
        selector._combine_scores(["a", "missing_col"])
        assert selector._feature_scores["missing_col"] == 0

    def test_l1_refine_empty_candidates(self):
        """Test L1 refinement with empty candidates list."""
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = np.array([0, 1] * 50)
        selector = FeatureSelector()
        result = selector._l1_refine(X, y, [])
        assert result == []

    def test_l1_refine_fallback_top_k(self):
        """Test L1 refinement fallback when no features pass threshold."""
        from unittest.mock import MagicMock, patch

        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = np.array([0, 1] * 50)
        selector = FeatureSelector()
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([-1.0, -1.0])
        with patch("sklearn.ensemble.GradientBoostingClassifier", return_value=mock_model):
            result = selector._l1_refine(X, y, ["a", "b"])
        assert len(result) > 0

    def test_l1_refine_exception_fallback(self):
        """Test L1 refinement returns candidates on exception."""
        from unittest.mock import MagicMock, patch

        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = np.array([0, 1] * 50)
        selector = FeatureSelector()
        mock_cls = MagicMock()
        mock_cls.return_value.fit.side_effect = Exception("model fit failed")
        with patch("sklearn.ensemble.GradientBoostingClassifier", mock_cls):
            result = selector._l1_refine(X, y, ["a", "b"])
        assert result == ["a", "b"]

    def test_verbose_excludes_low_importance(self):
        """Test debug logging for excluded low-importance derived features."""
        np.random.seed(42)
        n = 100
        X = pd.DataFrame(
            {
                "orig": np.random.randn(n),
                "derived_noise": np.random.randn(n) * 0.001,
            }
        )
        y = X["orig"] * 2 + np.random.randn(n) * 0.1
        selector = FeatureSelector(
            methods=["mutual_info"],
            original_features={"orig"},
            verbose=True,
        )
        selector.fit(X, y)
        assert selector._is_fitted

    def test_safety_check_adds_missing_original_features(self):
        """Test safety check restores missing original features."""
        np.random.seed(42)
        n = 100
        base = np.random.randn(n)
        X = pd.DataFrame(
            {
                "orig1": base,
                "orig2": base + np.random.randn(n) * 0.001,
                "orig3": np.random.randn(n),
            }
        )
        y = np.array([0, 1] * (n // 2))
        selector = FeatureSelector(
            methods=["mutual_info"],
            original_features={"orig1", "orig2", "orig3"},
            correlation_threshold=0.5,
        )
        selector.fit(X, y)
        selected = selector.get_selected_features()
        for f in ["orig1", "orig2", "orig3"]:
            assert f in selected

    def test_verbose_final_selection(self):
        """Test verbose logging in final selection."""
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = np.array([0, 1] * 50)
        selector = FeatureSelector(methods=["mutual_info"], verbose=True)
        selector.fit(X, y)
        assert selector._is_fitted

    def test_transform_before_fit_raises(self):
        """Test RuntimeError when transform called before fit."""
        selector = FeatureSelector()
        with pytest.raises(RuntimeError, match="Selector must be fitted"):
            selector.transform(pd.DataFrame({"a": [1, 2, 3]}))

    def test_get_ranking(self):
        """Test get_ranking returns sorted tuples."""
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = np.array([0, 1] * 50)
        selector = FeatureSelector(methods=["mutual_info"])
        selector.fit(X, y)
        ranking = selector.get_ranking()
        assert isinstance(ranking, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in ranking)
        scores = [score for _, score in ranking]
        assert scores == sorted(scores, reverse=True)


class TestRedundancyEliminatorCoverage:
    """Additional coverage tests for RedundancyEliminator."""

    def test_importance_scores_via_fit(self):
        """Test passing importance_scores through fit method parameter."""
        np.random.seed(42)
        n = 100
        base = np.random.randn(n)
        X = pd.DataFrame(
            {
                "a": base,
                "b": base + np.random.randn(n) * 0.001,
                "c": np.random.randn(n),
            }
        )
        eliminator = RedundancyEliminator(correlation_threshold=0.95)
        eliminator.fit(X, importance_scores={"a": 0.8, "b": 0.2, "c": 0.5})
        selected = eliminator.get_selected_features()
        assert "a" in selected
        assert "b" not in selected

    def test_original_preference_col2_is_original(self):
        """Test original preference bonus when col2 is original."""
        np.random.seed(42)
        n = 100
        base = np.random.randn(n)
        X = pd.DataFrame(
            {
                "derived": base,
                "original": base + np.random.randn(n) * 0.001,
            }
        )
        eliminator = RedundancyEliminator(
            correlation_threshold=0.95,
            importance_scores={"derived": 0.3, "original": 0.3},
            original_features={"original"},
        )
        eliminator.fit(X)
        selected = eliminator.get_selected_features()
        assert "original" in selected

    def test_verbose_remove_derived_col2(self):
        """Test verbose logging when removing derived col2."""
        np.random.seed(42)
        n = 100
        base = np.random.randn(n)
        X = pd.DataFrame(
            {
                "original": base,
                "derived": base + np.random.randn(n) * 0.001,
            }
        )
        eliminator = RedundancyEliminator(
            correlation_threshold=0.95,
            importance_scores={"original": 0.5, "derived": 0.2},
            original_features={"original"},
            verbose=True,
        )
        eliminator.fit(X)
        removed = eliminator.get_removed_features()
        assert "derived" in removed

    def test_verbose_remove_derived_col1(self):
        """Test verbose logging when removing derived col1."""
        np.random.seed(42)
        n = 100
        base = np.random.randn(n)
        X = pd.DataFrame(
            {
                "derived": base,
                "original": base + np.random.randn(n) * 0.001,
            }
        )
        eliminator = RedundancyEliminator(
            correlation_threshold=0.95,
            importance_scores={"derived": 0.2, "original": 0.5},
            original_features={"original"},
            verbose=True,
        )
        eliminator.fit(X)
        removed = eliminator.get_removed_features()
        assert "derived" in removed

    def test_transform_before_fit_raises(self):
        """Test RuntimeError when transform called before fit."""
        eliminator = RedundancyEliminator()
        with pytest.raises(RuntimeError, match="Eliminator must be fitted"):
            eliminator.transform(pd.DataFrame({"a": [1, 2, 3]}))

    def test_get_removed_features(self):
        """Test get_removed_features returns correct list."""
        np.random.seed(42)
        n = 100
        base = np.random.randn(n)
        X = pd.DataFrame(
            {
                "a": base,
                "b": base + np.random.randn(n) * 0.001,
                "c": np.random.randn(n),
            }
        )
        eliminator = RedundancyEliminator(
            correlation_threshold=0.95,
            importance_scores={"a": 0.8, "b": 0.2, "c": 0.5},
        )
        eliminator.fit(X)
        removed = eliminator.get_removed_features()
        assert isinstance(removed, list)
        assert "b" in removed

    def test_get_correlation_matrix(self):
        """Test get_correlation_matrix returns DataFrame."""
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        eliminator = RedundancyEliminator()
        eliminator.fit(X)
        corr_matrix = eliminator.get_correlation_matrix()
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape == (2, 2)
