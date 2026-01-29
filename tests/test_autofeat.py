"""Tests for main AutoFeatureEngineer."""

import numpy as np
import pandas as pd
import pytest

from featcopilot import AutoFeatureEngineer


class TestAutoFeatureEngineer:
    """Tests for AutoFeatureEngineer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, n),
                "income": np.random.exponential(50000, n),
                "tenure": np.random.randint(1, 120, n),
                "balance": np.random.randn(n) * 10000,
            }
        )
        y = (X["income"] > 50000).astype(int)
        return X, y

    def test_basic_fit_transform(self, sample_data):
        """Test basic fit_transform."""
        X, y = sample_data
        engineer = AutoFeatureEngineer(engines=["tabular"])
        result = engineer.fit_transform(X, y, apply_selection=False)

        assert len(result.columns) > len(X.columns)
        assert engineer._is_fitted

    def test_with_max_features(self, sample_data):
        """Test with max_features limit."""
        X, y = sample_data
        engineer = AutoFeatureEngineer(engines=["tabular"], max_features=20)
        result = engineer.fit_transform(X, y)

        # Should have at most 20 features total
        assert len(result.columns) <= 20

    def test_get_feature_names(self, sample_data):
        """Test getting feature names."""
        X, y = sample_data
        engineer = AutoFeatureEngineer(engines=["tabular"])
        engineer.fit_transform(X, y, apply_selection=False)

        names = engineer.get_feature_names()
        assert len(names) > 0

    def test_sklearn_compatibility(self, sample_data):
        """Test sklearn pipeline compatibility."""

        X, y = sample_data

        # This should work in a pipeline (simplified test)
        engineer = AutoFeatureEngineer(engines=["tabular"], max_features=10)
        engineer.fit(X, y)

        # Test transform separately
        X_transformed = engineer.transform(X)
        assert X_transformed is not None

    def test_multiple_engines(self, sample_data):
        """Test with multiple engines."""
        X, y = sample_data
        engineer = AutoFeatureEngineer(engines=["tabular"], max_features=30)  # Only tabular for non-LLM test
        result = engineer.fit_transform(X, y)

        assert result is not None
        assert len(result.columns) > 0


class TestAutoFeatureEngineerParams:
    """Test parameter handling."""

    def test_get_params(self):
        """Test get_params method."""
        engineer = AutoFeatureEngineer(engines=["tabular"], max_features=50, verbose=True)
        params = engineer.get_params()

        assert params["engines"] == ["tabular"]
        assert params["max_features"] == 50
        assert params["verbose"] is True

    def test_set_params(self):
        """Test set_params method."""
        engineer = AutoFeatureEngineer()
        engineer.set_params(max_features=100, verbose=True)

        assert engineer.max_features == 100
        assert engineer.verbose is True
