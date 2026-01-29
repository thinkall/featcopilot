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

        assert len(selector.get_selected_features()) == 5

    def test_method_scores(self, mixed_data):
        """Test getting scores from each method."""
        X, y = mixed_data
        selector = FeatureSelector(methods=["mutual_info", "f_test"])
        selector.fit(X, y)

        method_scores = selector.get_method_scores()
        assert "mutual_info" in method_scores
        assert "f_test" in method_scores
