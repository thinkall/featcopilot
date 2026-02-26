"""Tests for feature engineering engines."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from featcopilot.engines.relational import RelationalEngine
from featcopilot.engines.tabular import TabularEngine
from featcopilot.engines.text import TextEngine
from featcopilot.engines.timeseries import TimeSeriesEngine

try:
    import spacy

    spacy.load("en_core_web_sm")
    HAS_SPACY = True
except Exception:
    HAS_SPACY = False


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


# ---------------------------------------------------------------------------
# Additional TabularEngine tests
# ---------------------------------------------------------------------------


class TestTabularEngineExtended:
    """Extended tests for TabularEngine covering uncovered lines."""

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

    def test_fit_transform_ndarray_input(self):
        """Test fit_transform with ndarray input (line 154 - _validate_input)."""
        np.random.seed(42)
        arr = np.random.randn(50, 3)
        engine = TabularEngine(polynomial_degree=2, include_transforms=["log"])
        result = engine.fit_transform(arr)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 50
        assert len(result.columns) > 3

    def test_fit_transform_single_transform(self, sample_data):
        """Test fit_transform with a single transform type."""
        engine = TabularEngine(polynomial_degree=2, include_transforms=["log"])
        result = engine.fit_transform(sample_data)

        # Should have log transforms but not sqrt or square
        assert "num1_log1p" in result.columns
        sqrt_cols = [c for c in result.columns if "_sqrt" in c]
        assert len(sqrt_cols) == 0
        assert "num1_pow2" in result.columns

    def test_fit_transform_with_all_transforms(self, sample_data):
        """Test with many transform types (lines 183-194 for cat encoding)."""
        engine = TabularEngine(
            polynomial_degree=2,
            include_transforms=["log", "sqrt", "square", "reciprocal", "tanh"],
        )
        result = engine.fit_transform(sample_data)
        assert "num1_log1p" in result.columns
        assert "num1_sqrt" in result.columns
        assert "num1_sq" in result.columns
        assert "num1_recip" in result.columns
        assert "num1_tanh" in result.columns

    def test_categorical_encoding_onehot(self):
        """Test one-hot encoding of categorical columns (lines 206-237)."""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame(
            {
                "num1": np.random.randn(n),
                "num2": np.random.randn(n) * 5,
                "num3": np.random.randint(1, 50, n),
                "num4": np.random.randn(n),
                "num5": np.random.randn(n),
                "cat1": np.random.choice(["X", "Y", "Z"], n),
            }
        )
        engine = TabularEngine(
            polynomial_degree=2,
            include_transforms=[],
            encode_categorical=True,
            onehot_ratio_threshold=0.1,
            verbose=True,
        )
        result = engine.fit_transform(df)

        # One-hot columns should be generated for cat1 (3 unique / 200 rows = 0.015 < 0.1)
        assert "cat1_X" in result.columns
        assert "cat1_Y" in result.columns
        assert "cat1_Z" in result.columns
        assert "cat1_other" in result.columns

    def test_categorical_target_encoding(self):
        """Test target encoding of categorical columns (lines 220-232)."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "num1": np.random.randn(n),
                "num2": np.random.randn(n) * 5,
                "num3": np.random.randint(1, 50, n),
                "num4": np.random.randn(n),
                "num5": np.random.randn(n),
                # 10 unique values / 100 rows = 0.1, above onehot but below target threshold
                "cat_medium": np.random.choice([f"cat_{i}" for i in range(10)], n),
            }
        )
        y = np.random.randn(n)
        engine = TabularEngine(
            polynomial_degree=2,
            include_transforms=[],
            encode_categorical=True,
            onehot_ratio_threshold=0.05,
            target_encode_ratio_threshold=0.5,
            verbose=True,
        )
        result = engine.fit_transform(df, y=y)

        assert "cat_medium_target_encoded" in result.columns

    def test_categorical_string_target_encoding(self):
        """Test target encoding with string target (lines 183-194 LabelEncoder path)."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "num1": np.random.randn(n),
                "num2": np.random.randn(n) * 5,
                "num3": np.random.randint(1, 50, n),
                "num4": np.random.randn(n),
                "num5": np.random.randn(n),
                "cat_medium": np.random.choice([f"cat_{i}" for i in range(10)], n),
            }
        )
        # String target triggers LabelEncoder path
        y = pd.Series(np.random.choice(["pos", "neg"], n))
        engine = TabularEngine(
            polynomial_degree=2,
            include_transforms=[],
            encode_categorical=True,
            onehot_ratio_threshold=0.05,
            target_encode_ratio_threshold=0.5,
            verbose=True,
        )
        result = engine.fit_transform(df, y=y)
        assert "cat_medium_target_encoded" in result.columns

    def test_categorical_high_cardinality_skipped(self):
        """Test high cardinality categorical column is skipped (lines 234-239)."""
        np.random.seed(42)
        n = 50
        df = pd.DataFrame(
            {
                "num1": np.random.randn(n),
                "num2": np.random.randn(n) * 5,
                "num3": np.random.randint(1, 50, n),
                "num4": np.random.randn(n),
                "num5": np.random.randn(n),
                # High cardinality: 50 unique / 50 rows = 1.0
                "id_col": [f"id_{i}" for i in range(n)],
            }
        )
        engine = TabularEngine(
            polynomial_degree=2,
            include_transforms=[],
            encode_categorical=True,
            verbose=True,
        )
        result = engine.fit_transform(df)

        # High cardinality column should not be encoded
        target_encoded_cols = [c for c in result.columns if "id_col_target_encoded" in c]
        assert len(target_encoded_cols) == 0

    def test_categorical_no_valid_categories(self):
        """Test categorical column where no category has enough samples (lines 205-208)."""
        np.random.seed(42)
        n = 10
        df = pd.DataFrame(
            {
                "num1": np.random.randn(n),
                "num2": np.random.randn(n) * 5,
                "num3": np.random.randint(1, 50, n),
                "num4": np.random.randn(n),
                "num5": np.random.randn(n),
                # Each value appears only once or twice, below min_samples_per_category=5
                "rare_cat": [f"rare_{i}" for i in range(n)],
            }
        )
        engine = TabularEngine(
            polynomial_degree=2,
            include_transforms=[],
            encode_categorical=True,
            min_samples_per_category=5,
            verbose=True,
        )
        result = engine.fit_transform(df)

        onehot_cols = [c for c in result.columns if "rare_cat_" in c]
        assert len(onehot_cols) == 0

    def test_verbose_mode(self, sample_data):
        """Test verbose=True triggers logging (lines 154, 316, 401)."""
        engine = TabularEngine(polynomial_degree=2, include_transforms=["log"], verbose=True)
        result = engine.fit_transform(sample_data)

        assert engine.is_fitted
        assert len(result.columns) > len(sample_data.columns)

    def test_max_features_strict_limit(self, sample_data):
        """Test max_features stops generation early (lines 354, 360, 365)."""
        engine = TabularEngine(polynomial_degree=3, include_transforms=["log", "sqrt", "square"], max_features=5)
        result = engine.fit_transform(sample_data)

        new_features = [c for c in result.columns if c not in sample_data.columns]
        assert len(new_features) <= 5

    def test_get_feature_set(self, sample_data):
        """Test get_feature_set returns planned features (line 439)."""
        engine = TabularEngine(polynomial_degree=2, include_transforms=["log"])
        engine.fit(sample_data)

        feature_set = engine.get_feature_set()
        assert len(feature_set) > 0

    def test_get_feature_metadata(self, sample_data):
        """Test get_feature_metadata returns metadata dict."""
        engine = TabularEngine(polynomial_degree=2, include_transforms=["log"])
        engine.fit_transform(sample_data)

        metadata = engine.get_feature_metadata()
        assert isinstance(metadata, dict)

    def test_transform_before_fit_raises(self, sample_data):
        """Test transform before fit raises RuntimeError (line 333)."""
        engine = TabularEngine(polynomial_degree=2)
        with pytest.raises(RuntimeError, match="Engine must be fitted"):
            engine.transform(sample_data)

    def test_keep_original_categorical_false(self):
        """Test dropping original categorical columns (lines 421-422, 431-433)."""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame(
            {
                "num1": np.random.randn(n),
                "num2": np.random.randn(n) * 5,
                "num3": np.random.randint(1, 50, n),
                "num4": np.random.randn(n),
                "num5": np.random.randn(n),
                "cat1": np.random.choice(["A", "B", "C"], n),
            }
        )
        engine = TabularEngine(
            polynomial_degree=2,
            include_transforms=[],
            encode_categorical=True,
            keep_original_categorical=False,
            onehot_ratio_threshold=0.1,
        )
        engine.config.keep_original_categorical = False
        result = engine.fit_transform(df)

        # Original categorical column should be dropped
        assert "cat1" not in result.columns
        assert "cat1_A" in result.columns

    def test_categorical_column_missing_in_transform(self):
        """Test transform when a categorical column is missing (lines 411-412, 426-427)."""
        np.random.seed(42)
        n = 200
        df_fit = pd.DataFrame(
            {
                "num1": np.random.randn(n),
                "num2": np.random.randn(n) * 5,
                "num3": np.random.randint(1, 50, n),
                "num4": np.random.randn(n),
                "num5": np.random.randn(n),
                "cat1": np.random.choice(["A", "B", "C"], n),
            }
        )
        engine = TabularEngine(
            polynomial_degree=2,
            include_transforms=[],
            encode_categorical=True,
            onehot_ratio_threshold=0.1,
        )
        engine.fit(df_fit)

        # Transform with data missing the categorical column
        df_transform = df_fit.drop(columns=["cat1"])
        result = engine.transform(df_transform)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Extended TimeSeriesEngine tests
# ---------------------------------------------------------------------------


class TestTimeSeriesEngineExtended:
    """Extended tests for TimeSeriesEngine covering uncovered lines."""

    @pytest.fixture
    def ts_data(self):
        """Create time series data with numeric columns."""
        np.random.seed(42)
        n = 200
        return pd.DataFrame(
            {
                "value1": np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.randn(n) * 0.1,
                "value2": np.random.randn(n).cumsum(),
                "value3": np.random.randint(0, 100, n).astype(float),
            }
        )

    def test_full_fit_transform(self, ts_data):
        """Test full fit_transform with all default feature groups."""
        engine = TimeSeriesEngine(
            features=[
                "basic_stats",
                "distribution",
                "autocorrelation",
                "peaks",
                "trends",
                "entropy",
                "energy",
                "complexity",
                "counts",
            ],
        )
        result = engine.fit_transform(ts_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1  # Single row of extracted features
        assert "value1_mean" in result.columns
        assert "value1_skewness" in result.columns
        assert "value1_n_peaks" in result.columns

    def test_verbose_mode(self, ts_data):
        """Test verbose=True triggers logging (lines 138, 192)."""
        engine = TimeSeriesEngine(features=["basic_stats"], verbose=True)
        result = engine.fit_transform(ts_data)

        assert engine.is_fitted
        assert "value1_mean" in result.columns

    def test_get_feature_set(self, ts_data):
        """Test get_feature_set returns FeatureSet."""
        engine = TimeSeriesEngine(features=["basic_stats"])
        engine.fit_transform(ts_data)

        feature_set = engine.get_feature_set()
        assert feature_set is not None

    def test_transform_before_fit_raises(self, ts_data):
        """Test transform before fit raises RuntimeError (line 158)."""
        engine = TimeSeriesEngine(features=["basic_stats"])
        with pytest.raises(RuntimeError, match="Engine must be fitted"):
            engine.transform(ts_data)

    def test_with_window_sizes(self, ts_data):
        """Test with rolling window features."""
        engine = TimeSeriesEngine(features=["rolling"], window_sizes=[5, 10])
        result = engine.fit_transform(ts_data)

        assert "value1_rolling5_mean_of_means" in result.columns
        assert "value1_rolling10_mean_of_means" in result.columns

    def test_extract_basic_stats(self, ts_data):
        """Test _extract_basic_stats method directly."""
        engine = TimeSeriesEngine(features=["basic_stats"])
        engine.fit(ts_data)

        series = ts_data["value1"].values
        features = engine._extract_basic_stats(series, "value1")

        assert "value1_mean" in features
        assert "value1_std" in features
        assert "value1_min" in features
        assert "value1_max" in features
        assert "value1_range" in features
        assert "value1_median" in features
        assert "value1_sum" in features
        assert "value1_length" in features
        assert "value1_var" in features
        assert "value1_cv" in features
        assert features["value1_length"] == len(series)

    def test_extract_basic_stats_empty_series(self):
        """Test _extract_basic_stats with empty series."""
        engine = TimeSeriesEngine(features=["basic_stats"])
        features = engine._extract_basic_stats(np.array([]), "test")
        assert features == {}

    def test_extract_basic_stats_zero_mean(self):
        """Test _extract_basic_stats with zero mean (cv edge case, line 244)."""
        engine = TimeSeriesEngine(features=["basic_stats"])
        # Series that sums to approximately zero
        series = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        features = engine._extract_basic_stats(series, "test")
        assert "test_cv" in features

    def test_extract_distribution(self, ts_data):
        """Test _extract_distribution method."""
        engine = TimeSeriesEngine(features=["distribution"])
        engine.fit(ts_data)

        series = ts_data["value1"].values
        features = engine._extract_distribution(series, "value1")

        assert "value1_skewness" in features
        assert "value1_kurtosis" in features
        assert "value1_q25" in features
        assert "value1_q75" in features
        assert "value1_iqr" in features

    def test_extract_distribution_short_series(self):
        """Test _extract_distribution with series too short."""
        engine = TimeSeriesEngine(features=["distribution"])
        features = engine._extract_distribution(np.array([1.0, 2.0]), "test")
        assert features == {}

    def test_extract_autocorrelation(self, ts_data):
        """Test _extract_autocorrelation method."""
        engine = TimeSeriesEngine(features=["autocorrelation"])
        engine.fit(ts_data)

        series = ts_data["value1"].values
        features = engine._extract_autocorrelation(series, "value1")

        assert "value1_autocorr_lag1" in features
        assert len(features) > 0

    def test_extract_autocorrelation_constant_series(self):
        """Test _extract_autocorrelation with constant series (var==0, line 291)."""
        engine = TimeSeriesEngine(features=["autocorrelation"])
        constant_series = np.ones(50)
        features = engine._extract_autocorrelation(constant_series, "test")
        assert features == {}

    def test_extract_autocorrelation_short_series(self):
        """Test _extract_autocorrelation with series too short."""
        engine = TimeSeriesEngine(features=["autocorrelation"], n_autocorr_lags=10)
        features = engine._extract_autocorrelation(np.array([1.0, 2.0]), "test")
        assert features == {}

    def test_extract_peaks(self, ts_data):
        """Test _extract_peaks method."""
        engine = TimeSeriesEngine(features=["peaks"])
        engine.fit(ts_data)

        series = ts_data["value1"].values
        features = engine._extract_peaks(series, "value1")

        assert "value1_n_peaks" in features
        assert "value1_n_troughs" in features
        assert features["value1_n_peaks"] > 0

    def test_extract_peaks_short_series(self):
        """Test _extract_peaks with very short series."""
        engine = TimeSeriesEngine(features=["peaks"])
        features = engine._extract_peaks(np.array([1.0, 2.0]), "test")
        assert features == {}

    def test_extract_trends(self, ts_data):
        """Test _extract_trends method."""
        engine = TimeSeriesEngine(features=["trends"])
        engine.fit(ts_data)

        series = ts_data["value1"].values
        features = engine._extract_trends(series, "value1")

        assert "value1_trend_slope" in features
        assert "value1_trend_intercept" in features
        assert "value1_first_value" in features
        assert "value1_last_value" in features
        assert "value1_change" in features
        assert "value1_mean_abs_change" in features
        assert "value1_mean_change" in features

    def test_extract_fft(self, ts_data):
        """Test _extract_fft method."""
        engine = TimeSeriesEngine(features=["fft"])
        engine.fit(ts_data)

        series = ts_data["value1"].values
        features = engine._extract_fft(series, "value1")

        assert "value1_fft_coeff_1" in features
        assert "value1_spectral_energy" in features
        assert "value1_dominant_freq_idx" in features

    def test_extract_fft_short_series(self):
        """Test _extract_fft with very short series."""
        engine = TimeSeriesEngine(features=["fft"])
        features = engine._extract_fft(np.array([1.0, 2.0]), "test")
        assert features == {}

    def test_extract_entropy(self, ts_data):
        """Test _extract_entropy method."""
        engine = TimeSeriesEngine(features=["entropy"])
        engine.fit(ts_data)

        series = ts_data["value1"].values
        features = engine._extract_entropy(series, "value1")

        assert "value1_binned_entropy" in features
        assert "value1_sample_entropy" in features
        assert "value1_approximate_entropy" in features

    def test_extract_energy(self, ts_data):
        """Test _extract_energy method."""
        engine = TimeSeriesEngine(features=["energy"])
        engine.fit(ts_data)

        series = ts_data["value1"].values
        features = engine._extract_energy(series, "value1")

        assert "value1_abs_energy" in features
        assert "value1_rms" in features
        assert "value1_crest_factor" in features
        assert "value1_mean_second_deriv_central" in features

    def test_extract_complexity(self, ts_data):
        """Test _extract_complexity method."""
        engine = TimeSeriesEngine(features=["complexity"])
        engine.fit(ts_data)

        series = ts_data["value1"].values
        features = engine._extract_complexity(series, "value1")

        assert "value1_cid_ce" in features
        assert "value1_c3" in features
        assert "value1_ratio_unique_values" in features
        assert "value1_has_duplicate" in features
        assert "value1_has_duplicate_max" in features
        assert "value1_has_duplicate_min" in features

    def test_extract_counts(self, ts_data):
        """Test _extract_counts method."""
        engine = TimeSeriesEngine(features=["counts"])
        engine.fit(ts_data)

        series = ts_data["value1"].values
        features = engine._extract_counts(series, "value1")

        assert "value1_count_above_mean" in features
        assert "value1_count_below_mean" in features
        assert "value1_first_loc_max" in features
        assert "value1_first_loc_min" in features
        assert "value1_last_loc_max" in features
        assert "value1_last_loc_min" in features
        assert "value1_longest_strike_above_mean" in features
        assert "value1_longest_strike_below_mean" in features
        assert "value1_number_crossings_mean" in features
        assert "value1_number_zero_crossings" in features
        assert "value1_abs_sum_changes" in features

    def test_sample_entropy(self):
        """Test _sample_entropy helper (line 444+)."""
        engine = TimeSeriesEngine(features=["entropy"])
        np.random.seed(42)
        series = np.random.randn(100)
        result = engine._sample_entropy(series, m=2, r=0.2)
        assert isinstance(result, float)

    def test_sample_entropy_short_series(self):
        """Test _sample_entropy with too-short series."""
        engine = TimeSeriesEngine(features=["entropy"])
        result = engine._sample_entropy(np.array([1.0, 2.0]), m=2, r=0.2)
        assert result == 0

    def test_sample_entropy_constant_series(self):
        """Test _sample_entropy with constant series (r==0 path)."""
        engine = TimeSeriesEngine(features=["entropy"])
        result = engine._sample_entropy(np.ones(50), m=2, r=0.2)
        assert result == 0

    def test_approximate_entropy(self):
        """Test _approximate_entropy helper (line 472+)."""
        engine = TimeSeriesEngine(features=["entropy"])
        np.random.seed(42)
        series = np.random.randn(100)
        result = engine._approximate_entropy(series, m=2, r=0.2)
        assert isinstance(result, float)

    def test_approximate_entropy_short_series(self):
        """Test _approximate_entropy with too-short series."""
        engine = TimeSeriesEngine(features=["entropy"])
        result = engine._approximate_entropy(np.array([1.0, 2.0]), m=2, r=0.2)
        assert result == 0

    def test_approximate_entropy_constant_series(self):
        """Test _approximate_entropy with constant series (r==0 path)."""
        engine = TimeSeriesEngine(features=["entropy"])
        result = engine._approximate_entropy(np.ones(50), m=2, r=0.2)
        assert result == 0

    def test_longest_consecutive(self):
        """Test _longest_consecutive helper."""
        engine = TimeSeriesEngine(features=["counts"])
        result = engine._longest_consecutive(np.array([True, True, False, True, True, True]))
        assert result == 3

    def test_longest_consecutive_all_false(self):
        """Test _longest_consecutive with all False."""
        engine = TimeSeriesEngine(features=["counts"])
        result = engine._longest_consecutive(np.array([False, False, False]))
        assert result == 0

    def test_extract_per_row_single_values(self, ts_data):
        """Test _extract_per_row with scalar values (lines 198-220)."""
        engine = TimeSeriesEngine(features=["basic_stats"])
        engine.fit(ts_data)

        # When columns contain scalar values per row, _extract_per_row creates {col}_value
        result = engine._extract_per_row(ts_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(ts_data)
        # Scalar values should produce {col}_value columns
        assert "value1_value" in result.columns

    def test_extract_per_row_array_values(self):
        """Test _extract_per_row with array values in cells (lines 204-216)."""
        engine = TimeSeriesEngine(features=["basic_stats"])
        df = pd.DataFrame(
            {
                "series": [np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([5.0, 4.0, 3.0, 2.0, 1.0])],
            }
        )
        engine._time_columns = ["series"]
        engine._is_fitted = True

        result = engine._extract_per_row(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "series_mean" in result.columns

    def test_rolling_window_larger_than_series(self):
        """Test rolling features when window is larger than series."""
        engine = TimeSeriesEngine(features=["rolling"], window_sizes=[100])
        short_data = pd.DataFrame({"value": np.random.randn(10)})
        result = engine.fit_transform(short_data)

        # Window=100 > series length=10, so no rolling features for that window
        rolling_cols = [c for c in result.columns if "rolling100" in c]
        assert len(rolling_cols) == 0

    def test_ndarray_input(self):
        """Test fit_transform with ndarray input."""
        np.random.seed(42)
        arr = np.random.randn(100, 2)
        engine = TimeSeriesEngine(features=["basic_stats"])
        result = engine.fit_transform(arr)

        assert isinstance(result, pd.DataFrame)
        assert "feature_0_mean" in result.columns
        assert "feature_1_mean" in result.columns


# ---------------------------------------------------------------------------
# RelationalEngine tests
# ---------------------------------------------------------------------------


class TestRelationalEngine:
    """Tests for RelationalEngine covering lines 76-259."""

    @pytest.fixture
    def orders_data(self):
        """Create orders (child) DataFrame."""
        return pd.DataFrame(
            {
                "order_id": [1, 2, 3, 4, 5],
                "customer_id": [1, 1, 2, 2, 3],
                "amount": [100, 200, 150, 300, 50],
                "category": ["A", "B", "A", "A", "B"],
            }
        )

    @pytest.fixture
    def customers_data(self):
        """Create customers (parent) DataFrame."""
        return pd.DataFrame(
            {
                "customer_id": [1, 2, 3],
                "age": [25, 35, 45],
                "income": [50000, 70000, 60000],
            }
        )

    def test_create_engine_default_config(self):
        """Test creating RelationalEngine with default config (lines 76-86)."""
        engine = RelationalEngine()
        assert not engine.is_fitted
        assert engine.config.name == "RelationalEngine"
        assert "mean" in engine.config.aggregation_functions
        assert "sum" in engine.config.aggregation_functions

    def test_add_relationship(self):
        """Test add_relationship method (lines 88-117)."""
        engine = RelationalEngine()
        result = engine.add_relationship("orders", "customers", "customer_id")
        assert result is engine  # Returns self for chaining
        assert len(engine._relationships) == 1
        assert engine._relationships[0]["child"] == "orders"
        assert engine._relationships[0]["parent"] == "customers"
        assert engine._relationships[0]["child_key"] == "customer_id"
        assert engine._relationships[0]["parent_key"] == "customer_id"

    def test_add_relationship_with_parent_key(self):
        """Test add_relationship with explicit parent_key."""
        engine = RelationalEngine()
        engine.add_relationship("orders", "customers", "cust_id", parent_key="customer_id")
        assert engine._relationships[0]["child_key"] == "cust_id"
        assert engine._relationships[0]["parent_key"] == "customer_id"

    def test_chain_relationships(self):
        """Test chaining multiple add_relationship calls."""
        engine = RelationalEngine()
        engine.add_relationship("orders", "customers", "customer_id").add_relationship(
            "order_items", "orders", "order_id"
        )
        assert len(engine._relationships) == 2

    def test_fit(self, orders_data, customers_data):
        """Test fit method (lines 119-150)."""
        engine = RelationalEngine()
        engine.add_relationship("orders", "customers", "customer_id")
        engine.fit(orders_data, related_tables={"customers": customers_data})

        assert engine.is_fitted

    def test_fit_verbose(self, orders_data, customers_data):
        """Test fit with verbose mode (line 147)."""
        engine = RelationalEngine(verbose=True)
        engine.add_relationship("orders", "customers", "customer_id")
        engine.fit(orders_data, related_tables={"customers": customers_data})
        assert engine.is_fitted

    def test_transform_before_fit_raises(self, orders_data):
        """Test transform before fit raises RuntimeError (line 173-174)."""
        engine = RelationalEngine()
        with pytest.raises(RuntimeError, match="Engine must be fitted"):
            engine.transform(orders_data)

    def test_full_workflow(self, orders_data, customers_data):
        """Test full workflow: create, add relationship, fit, transform (lines 180-199)."""
        engine = RelationalEngine(aggregation_functions=["mean", "sum", "count", "max", "min"])
        engine.add_relationship("orders", "customers", "customer_id")
        result = engine.fit_transform(orders_data, related_tables={"customers": customers_data})

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5  # Same number of rows as orders

        # Should have aggregated features from customers
        assert "customers_age_mean" in result.columns
        assert "customers_income_mean" in result.columns

    def test_aggregate_from_relationship(self, orders_data, customers_data):
        """Test _aggregate_from_relationship method (lines 201-231)."""
        engine = RelationalEngine(aggregation_functions=["mean", "sum", "count"])
        engine.add_relationship("orders", "customers", "customer_id")
        engine.fit(orders_data, related_tables={"customers": customers_data})

        features = engine._aggregate_from_relationship(
            orders_data, customers_data, "customer_id", "customer_id", "customers"
        )
        assert isinstance(features, pd.DataFrame)
        assert "customers_age_mean" in features.columns

    def test_self_aggregations(self, orders_data, customers_data):
        """Test _add_self_aggregations with categorical + numeric columns (lines 233-255)."""
        engine = RelationalEngine()
        engine.add_relationship("orders", "customers", "customer_id")
        engine.fit(orders_data, related_tables={"customers": customers_data})

        result = engine._add_self_aggregations(orders_data)

        # Should have group-by aggregations for 'category' column
        assert "amount_by_category_mean" in result.columns
        assert "amount_by_category_count" in result.columns

    def test_get_feature_set(self, orders_data, customers_data):
        """Test get_feature_set method (lines 257-259)."""
        engine = RelationalEngine()
        engine.add_relationship("orders", "customers", "customer_id")
        engine.fit_transform(orders_data, related_tables={"customers": customers_data})

        feature_set = engine.get_feature_set()
        assert feature_set is not None

    def test_transform_verbose(self, orders_data, customers_data):
        """Test transform with verbose mode (line 197)."""
        engine = RelationalEngine(verbose=True)
        engine.add_relationship("orders", "customers", "customer_id")
        result = engine.fit_transform(orders_data, related_tables={"customers": customers_data})

        assert isinstance(result, pd.DataFrame)

    def test_missing_parent_table_skipped(self, orders_data, customers_data):
        """Test that missing parent table in related_tables is skipped (line 182-183)."""
        engine = RelationalEngine()
        engine.add_relationship("orders", "nonexistent_table", "customer_id")
        engine.fit(orders_data, related_tables={"customers": customers_data})

        result = engine.transform(orders_data, related_tables={"customers": customers_data})
        # No features from nonexistent table
        new_cols = [c for c in result.columns if c not in orders_data.columns]
        # Only self-aggregation features should be present
        agg_from_relationship = [c for c in new_cols if "nonexistent" in c]
        assert len(agg_from_relationship) == 0

    def test_transform_with_different_related_tables(self, orders_data, customers_data):
        """Test transform with related_tables passed at transform time."""
        engine = RelationalEngine()
        engine.add_relationship("orders", "customers", "customer_id")
        engine.fit(orders_data, related_tables={"customers": customers_data})

        # Pass different related_tables at transform time
        result = engine.transform(orders_data, related_tables={"customers": customers_data})
        assert isinstance(result, pd.DataFrame)
        assert "customers_age_mean" in result.columns

    def test_no_relationships(self, orders_data):
        """Test engine with no relationships defined."""
        engine = RelationalEngine()
        result = engine.fit_transform(orders_data)

        assert isinstance(result, pd.DataFrame)
        # Should still have self-aggregation features
        assert "amount_by_category_mean" in result.columns

    def test_ndarray_input(self):
        """Test fit_transform with ndarray input."""
        np.random.seed(42)
        arr = np.random.randn(20, 3)
        engine = RelationalEngine()
        result = engine.fit_transform(arr)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 20

    def test_custom_aggregation_functions(self, orders_data, customers_data):
        """Test with custom aggregation functions."""
        engine = RelationalEngine(aggregation_functions=["mean", "max", "min", "std"])
        engine.add_relationship("orders", "customers", "customer_id")
        result = engine.fit_transform(orders_data, related_tables={"customers": customers_data})

        assert "customers_age_max" in result.columns
        assert "customers_age_min" in result.columns

    def test_max_features_param(self, orders_data, customers_data):
        """Test engine with max_features parameter."""
        engine = RelationalEngine(max_features=5)
        engine.add_relationship("orders", "customers", "customer_id")
        result = engine.fit_transform(orders_data, related_tables={"customers": customers_data})

        assert isinstance(result, pd.DataFrame)


class TestTextEngine:
    """Tests for TextEngine."""

    SAMPLE_TEXTS = [
        "The quick brown fox jumps over the lazy dog near the river bank",
        "Machine learning is transforming how we analyze large datasets",
        "John Smith works at Google headquarters in Mountain View California",
        "The stock market crashed by twenty percent in March 2020",
        "Python programming language is widely used for data science tasks",
        "Natural language processing enables computers to understand human speech",
        "Apple released their new iPhone product line in September last year",
        "Deep neural networks have achieved remarkable results in computer vision",
        "The United Nations held a meeting in Geneva Switzerland yesterday morning",
        "Quantum computing promises to revolutionize cryptography and drug discovery",
        "Amazon Web Services dominates the cloud computing infrastructure market today",
        "Climate change is causing rising sea levels across coastal cities worldwide",
        "The basketball championship game attracted over fifty thousand spectators live",
        "Researchers at MIT developed a new algorithm for protein folding analysis",
        "Electric vehicles are becoming increasingly popular in European countries now",
        "The CEO announced record quarterly earnings of five billion dollars revenue",
        "Artificial intelligence is being applied to medical diagnosis and treatment plans",
        "SpaceX launched another batch of Starlink satellites into low Earth orbit",
        "The symphony orchestra performed Beethoven classics at Carnegie Hall last night",
        "Blockchain technology is disrupting traditional financial services and banking sector",
        "Scientists discovered a new species of marine life in the Pacific Ocean",
        "The Olympic Games will be hosted in Paris France during summer season",
        "Virtual reality headsets are transforming gaming entertainment and education fields",
        "Renewable energy sources like solar and wind are growing rapidly worldwide",
        "Harvard University published groundbreaking research on gene therapy treatments today",
        "Self driving cars are being tested on public roads in multiple states",
        "The prime minister addressed parliament about new economic reform proposals today",
        "Cybersecurity threats continue to evolve posing risks to businesses and governments",
        "The documentary film won several awards at the Cannes Film Festival event",
        "Global supply chain disruptions have affected manufacturing industries around the world",
    ]

    @pytest.fixture
    def text_df(self):
        """Create sample text DataFrame."""
        np.random.seed(42)
        n = 30
        return pd.DataFrame({"text": self.SAMPLE_TEXTS[:n], "num1": np.random.randn(n)})

    def test_text_engine_basic_features(self, text_df):
        """Test basic text feature extraction: length, word_count, char_stats."""
        engine = TextEngine(features=["length", "word_count", "char_stats"])
        result = engine.fit_transform(text_df, text_columns=["text"])

        assert "text_char_length" in result.columns
        assert "text_word_count" in result.columns
        assert "text_uppercase_ratio" in result.columns
        assert "text_digit_ratio" in result.columns
        assert "text_space_ratio" in result.columns
        assert "text_special_char_count" in result.columns
        assert "text_avg_word_length" in result.columns
        assert "text_unique_word_ratio" in result.columns
        assert len(result) == len(text_df)
        # text column should be replaced with encoded version
        assert "text" not in result.columns
        assert "text_encoded" in result.columns

    def test_text_engine_auto_detect_text_columns(self, text_df):
        """Test auto-detection of text columns based on length and cardinality."""
        engine = TextEngine(features=["length"])
        result = engine.fit_transform(text_df)

        # Auto-detect should find 'text' column (avg len > 10 and nunique > 10)
        assert "text_char_length" in result.columns

    def test_text_engine_explicit_text_columns(self, text_df):
        """Test passing text_columns explicitly."""
        engine = TextEngine(features=["length"])
        engine.fit(text_df, text_columns=["text"])
        result = engine.transform(text_df)

        assert "text_char_length" in result.columns

    def test_text_engine_tfidf_features(self, text_df):
        """Test TF-IDF + SVD feature extraction."""
        engine = TextEngine(features=["tfidf"], n_components=5)
        result = engine.fit_transform(text_df, text_columns=["text"])

        tfidf_cols = [c for c in result.columns if "tfidf" in c]
        assert len(tfidf_cols) > 0
        assert "text_tfidf_0" in result.columns

    @pytest.mark.skipif(not HAS_SPACY, reason="spacy en_core_web_sm not available")
    def test_text_engine_ner_features(self, text_df):
        """Test NER feature extraction with spacy."""
        engine = TextEngine(features=["ner"])
        result = engine.fit_transform(text_df, text_columns=["text"])

        assert "text_ner_person" in result.columns
        assert "text_ner_org" in result.columns
        assert "text_ner_gpe" in result.columns
        assert "text_ner_total" in result.columns

    @pytest.mark.skipif(not HAS_SPACY, reason="spacy en_core_web_sm not available")
    def test_text_engine_pos_features(self, text_df):
        """Test POS tag feature extraction with spacy."""
        engine = TextEngine(features=["pos"])
        result = engine.fit_transform(text_df, text_columns=["text"])

        assert "text_pos_noun_ratio" in result.columns
        assert "text_pos_verb_ratio" in result.columns
        assert "text_pos_adj_ratio" in result.columns
        assert "text_pos_noun_verb_ratio" in result.columns
        assert "text_pos_content_ratio" in result.columns

    @patch.object(TextEngine, "_load_sentiment")
    def test_text_engine_sentiment_features(self, mock_load, text_df):
        """Test sentiment feature extraction with mocked pipeline."""
        engine = TextEngine(features=["sentiment", "length"])
        engine._sentiment_pipeline = MagicMock()
        engine._sentiment_pipeline.side_effect = lambda batch: [{"label": "positive", "score": 0.9}] * len(batch)
        engine.fit(text_df, text_columns=["text"])
        result = engine.transform(text_df)

        assert "text_sentiment_positive" in result.columns
        assert "text_sentiment_negative" in result.columns
        assert "text_sentiment_neutral" in result.columns
        assert "text_sentiment_score" in result.columns
        assert "text_char_length" in result.columns

    @patch.object(TextEngine, "_load_embedding_model")
    def test_text_engine_embeddings(self, mock_load_model, text_df):
        """Test embedding feature extraction with mocked model."""
        n = len(text_df)
        engine = TextEngine(features=["embeddings"], embedding_dim=8)
        engine._embedding_model = MagicMock()
        np.random.seed(42)
        engine._embedding_model.encode.return_value = np.random.randn(n, 384)
        engine.fit(text_df, text_columns=["text"])
        result = engine.transform(text_df)

        emb_cols = [c for c in result.columns if "emb_" in c]
        assert len(emb_cols) > 0

    def test_text_engine_verbose(self, text_df):
        """Test verbose mode produces output without error."""
        engine = TextEngine(features=["length"], verbose=True)
        result = engine.fit_transform(text_df, text_columns=["text"])

        assert "text_char_length" in result.columns

    def test_text_engine_transform_before_fit(self, text_df):
        """Test that transform before fit raises RuntimeError."""
        engine = TextEngine(features=["length"])
        with pytest.raises(RuntimeError, match="fitted"):
            engine.transform(text_df)

    def test_text_engine_get_feature_set(self, text_df):
        """Test get_feature_set returns a FeatureSet."""
        from featcopilot.core.feature import FeatureSet

        engine = TextEngine(features=["length"])
        engine.fit(text_df, text_columns=["text"])
        fs = engine.get_feature_set()

        assert isinstance(fs, FeatureSet)

    def test_text_engine_label_encoding(self, text_df):
        """Test that text columns are replaced with label-encoded versions."""
        engine = TextEngine(features=["length"])
        result = engine.fit_transform(text_df, text_columns=["text"])

        assert "text" not in result.columns
        assert "text_encoded" in result.columns
        assert result["text_encoded"].dtype in [np.int64, np.int32, np.float64, int]

    def test_text_engine_fit_transform(self, text_df):
        """Test full fit_transform pipeline."""
        engine = TextEngine(features=["length", "word_count", "char_stats"])
        result = engine.fit_transform(text_df, text_columns=["text"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(text_df)
        assert len(result.columns) > len(text_df.columns)
        assert engine.is_fitted

    def test_text_engine_no_text_columns(self):
        """Test engine with all-numeric data produces no text features."""
        np.random.seed(42)
        numeric_df = pd.DataFrame({"a": np.random.randn(20), "b": np.random.randn(20)})
        engine = TextEngine(features=["length"])
        result = engine.fit_transform(numeric_df)

        # No text columns detected, so only original columns remain
        assert set(result.columns) == set(numeric_df.columns)
