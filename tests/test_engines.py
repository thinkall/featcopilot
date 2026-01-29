"""Tests for feature engineering engines."""

import numpy as np
import pandas as pd
import pytest

from featcopilot.engines.tabular import TabularEngine
from featcopilot.engines.timeseries import TimeSeriesEngine


class TestTabularEngine:
    """Tests for TabularEngine."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "num1": np.random.randn(100),
                "num2": np.random.randn(100) * 10,
                "num3": np.random.randint(1, 100, 100),
                "cat1": np.random.choice(["A", "B", "C"], 100),
            }
        )

    def test_tabular_engine_fit(self, sample_data):
        """Test fitting tabular engine."""
        engine = TabularEngine(polynomial_degree=2)
        engine.fit(sample_data)

        assert engine.is_fitted
        assert len(engine._numeric_columns) == 3  # num1, num2, num3

    def test_tabular_engine_transform(self, sample_data):
        """Test transforming with tabular engine."""
        engine = TabularEngine(polynomial_degree=2, include_transforms=["log", "sqrt"])
        result = engine.fit_transform(sample_data)

        # Should have original columns + generated features
        assert len(result.columns) > len(sample_data.columns)

        # Check polynomial features exist
        assert "num1_pow2" in result.columns

        # Check interaction features exist
        assert "num1_x_num2" in result.columns

        # Check transform features exist
        assert "num1_log1p" in result.columns

    def test_tabular_engine_max_features(self, sample_data):
        """Test max_features limit."""
        engine = TabularEngine(polynomial_degree=2, max_features=10)
        result = engine.fit_transform(sample_data)

        # New features should be limited
        new_features = [c for c in result.columns if c not in sample_data.columns]
        assert len(new_features) <= 10

    def test_tabular_engine_interaction_only(self, sample_data):
        """Test interaction-only mode."""
        engine = TabularEngine(polynomial_degree=2, interaction_only=True, include_transforms=[])
        result = engine.fit_transform(sample_data)

        # Should not have power features
        power_features = [c for c in result.columns if "_pow" in c]
        assert len(power_features) == 0


class TestTimeSeriesEngine:
    """Tests for TimeSeriesEngine."""

    @pytest.fixture
    def ts_data(self):
        """Create time series data."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "series1": np.cumsum(np.random.randn(100)),
                "series2": np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.randn(100) * 0.1,
            }
        )

    def test_timeseries_engine_fit(self, ts_data):
        """Test fitting time series engine."""
        engine = TimeSeriesEngine(features=["basic_stats"])
        engine.fit(ts_data)

        assert engine.is_fitted
        assert len(engine._time_columns) == 2

    def test_timeseries_engine_basic_stats(self, ts_data):
        """Test basic statistics extraction."""
        engine = TimeSeriesEngine(features=["basic_stats"])
        result = engine.fit_transform(ts_data)

        # Should have mean, std, min, max features
        assert "series1_mean" in result.columns
        assert "series1_std" in result.columns
        assert "series1_min" in result.columns
        assert "series1_max" in result.columns

    def test_timeseries_engine_distribution(self, ts_data):
        """Test distribution features extraction."""
        engine = TimeSeriesEngine(features=["distribution"])
        result = engine.fit_transform(ts_data)

        # Should have skewness, kurtosis features
        assert "series1_skewness" in result.columns
        assert "series1_kurtosis" in result.columns

    def test_timeseries_engine_trends(self, ts_data):
        """Test trend features extraction."""
        engine = TimeSeriesEngine(features=["trends"])
        result = engine.fit_transform(ts_data)

        # Should have trend slope
        assert "series1_trend_slope" in result.columns
        assert "series1_change" in result.columns
