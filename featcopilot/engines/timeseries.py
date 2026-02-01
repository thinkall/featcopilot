"""Time series feature engineering engine.

Extracts statistical, frequency, and temporal features from time series data.
Inspired by TSFresh but with better integration and LLM capabilities.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
from pydantic import Field

from featcopilot.core.base import BaseEngine, EngineConfig
from featcopilot.core.feature import FeatureSet
from featcopilot.utils.logger import get_logger

logger = get_logger(__name__)


class TimeSeriesEngineConfig(EngineConfig):
    """Configuration for time series feature engine."""

    name: str = "TimeSeriesEngine"
    features: list[str] = Field(
        default_factory=lambda: [
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
        description="Feature groups to extract",
    )
    window_sizes: list[int] = Field(
        default_factory=lambda: [5, 10, 20], description="Window sizes for rolling features"
    )
    n_fft_coefficients: int = Field(default=10, description="Number of FFT coefficients")
    n_autocorr_lags: int = Field(default=10, description="Number of autocorrelation lags")
    entropy_bins: int = Field(default=10, description="Number of bins for binned entropy")


class TimeSeriesEngine(BaseEngine):
    """
    Time series feature engineering engine.

    Extracts comprehensive features from time series data including:
    - Basic statistics (mean, std, min, max, etc.)
    - Distribution features (skewness, kurtosis, quantiles)
    - Autocorrelation features
    - Frequency domain features (FFT)
    - Peak and trough features
    - Trend features
    - Rolling window statistics

    Parameters
    ----------
    features : list, default=['basic_stats', 'distribution', 'autocorrelation']
        Feature groups to extract
    window_sizes : list, default=[5, 10, 20]
        Window sizes for rolling features
    max_features : int, optional
        Maximum number of features to generate

    Examples
    --------
    >>> engine = TimeSeriesEngine(features=['basic_stats', 'autocorrelation'])
    >>> X_features = engine.fit_transform(time_series_df)
    """

    # Feature extraction functions (tsfresh-inspired)
    FEATURE_EXTRACTORS = {
        "basic_stats": "_extract_basic_stats",
        "distribution": "_extract_distribution",
        "autocorrelation": "_extract_autocorrelation",
        "peaks": "_extract_peaks",
        "trends": "_extract_trends",
        "rolling": "_extract_rolling",
        "fft": "_extract_fft",
        "entropy": "_extract_entropy",
        "energy": "_extract_energy",
        "complexity": "_extract_complexity",
        "counts": "_extract_counts",
    }

    def __init__(
        self,
        features: Optional[list[str]] = None,
        window_sizes: Optional[list[int]] = None,
        max_features: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ):
        config = TimeSeriesEngineConfig(
            features=features or ["basic_stats", "distribution", "autocorrelation"],
            window_sizes=window_sizes or [5, 10, 20],
            max_features=max_features,
            verbose=verbose,
            **kwargs,
        )
        super().__init__(config=config)
        self.config: TimeSeriesEngineConfig = config
        self._time_columns: list[str] = []
        self._feature_set = FeatureSet()

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        time_column: Optional[str] = None,
        **kwargs,
    ) -> "TimeSeriesEngine":
        """
        Fit the engine to identify time series columns.

        Parameters
        ----------
        X : DataFrame or ndarray
            Input data (each row is a time series or time-indexed data)
        y : Series or ndarray, optional
            Target variable
        time_column : str, optional
            Column containing timestamps

        Returns
        -------
        self : TimeSeriesEngine
        """
        X = self._validate_input(X)

        # Identify numeric columns for time series analysis
        self._time_columns = X.select_dtypes(include=[np.number]).columns.tolist()

        if self.config.verbose:
            logger.info(f"TimeSeriesEngine: Found {len(self._time_columns)} numeric columns")

        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> pd.DataFrame:
        """
        Extract time series features from input data.

        Parameters
        ----------
        X : DataFrame or ndarray
            Input data

        Returns
        -------
        X_features : DataFrame
            Extracted features
        """
        if not self._is_fitted:
            raise RuntimeError("Engine must be fitted before transform")

        X = self._validate_input(X)
        features_dict = {}

        for col in self._time_columns:
            series = X[col].values

            for feature_group in self.config.features:
                if feature_group in self.FEATURE_EXTRACTORS:
                    method_name = self.FEATURE_EXTRACTORS[feature_group]
                    method = getattr(self, method_name)
                    extracted = method(series, col)
                    features_dict.update(extracted)

        # For DataFrames with multiple rows, extract features across the entire column
        if len(X) > 1:
            # Each column is treated as a single time series
            features_dict = {}
            for col in self._time_columns:
                series = X[col].values

                for feature_group in self.config.features:
                    if feature_group in self.FEATURE_EXTRACTORS:
                        method_name = self.FEATURE_EXTRACTORS[feature_group]
                        method = getattr(self, method_name)
                        extracted = method(series, col)
                        features_dict.update(extracted)

        result = pd.DataFrame([features_dict])

        self._feature_names = list(result.columns)

        if self.config.verbose:
            logger.info(f"TimeSeriesEngine: Extracted {len(self._feature_names)} features")

        return result

    def _extract_per_row(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract features for each row (multiple time series)."""
        all_features = []

        for idx in range(len(X)):
            row_features = {}
            for col in self._time_columns:
                value = X[col].iloc[idx]
                if isinstance(value, (list, np.ndarray)):
                    series = np.array(value)
                else:
                    # Single value - create minimal features
                    row_features[f"{col}_value"] = value
                    continue

                for feature_group in self.config.features:
                    if feature_group in self.FEATURE_EXTRACTORS:
                        method_name = self.FEATURE_EXTRACTORS[feature_group]
                        method = getattr(self, method_name)
                        extracted = method(series, col)
                        row_features.update(extracted)

            all_features.append(row_features)

        return pd.DataFrame(all_features)

    def _extract_basic_stats(self, series: np.ndarray, col: str) -> dict[str, float]:
        """Extract basic statistical features."""
        features = {}
        prefix = col

        if len(series) == 0:
            return features

        features[f"{prefix}_mean"] = np.nanmean(series)
        features[f"{prefix}_std"] = np.nanstd(series)
        features[f"{prefix}_min"] = np.nanmin(series)
        features[f"{prefix}_max"] = np.nanmax(series)
        features[f"{prefix}_range"] = features[f"{prefix}_max"] - features[f"{prefix}_min"]
        features[f"{prefix}_median"] = np.nanmedian(series)
        features[f"{prefix}_sum"] = np.nansum(series)
        features[f"{prefix}_length"] = len(series)
        features[f"{prefix}_var"] = np.nanvar(series)

        # Coefficient of variation
        if features[f"{prefix}_mean"] != 0:
            features[f"{prefix}_cv"] = features[f"{prefix}_std"] / abs(features[f"{prefix}_mean"])
        else:
            features[f"{prefix}_cv"] = 0

        return features

    def _extract_distribution(self, series: np.ndarray, col: str) -> dict[str, float]:
        """Extract distribution-based features."""
        from scipy import stats

        features = {}
        prefix = col

        if len(series) < 4:
            return features

        # Remove NaN values
        series_clean = series[~np.isnan(series)]
        if len(series_clean) < 4:
            return features

        features[f"{prefix}_skewness"] = stats.skew(series_clean)
        features[f"{prefix}_kurtosis"] = stats.kurtosis(series_clean)

        # Quantiles
        for q in [0.1, 0.25, 0.75, 0.9]:
            features[f"{prefix}_q{int(q*100)}"] = np.quantile(series_clean, q)

        # IQR
        q75, q25 = np.quantile(series_clean, [0.75, 0.25])
        features[f"{prefix}_iqr"] = q75 - q25

        return features

    def _extract_autocorrelation(self, series: np.ndarray, col: str) -> dict[str, float]:
        """Extract autocorrelation features."""
        features = {}
        prefix = col

        if len(series) < self.config.n_autocorr_lags + 1:
            return features

        series_clean = series[~np.isnan(series)]
        if len(series_clean) < self.config.n_autocorr_lags + 1:
            return features

        # Compute autocorrelation for different lags
        var = np.var(series_clean)

        if var == 0:
            return features

        for lag in range(1, min(self.config.n_autocorr_lags + 1, len(series_clean))):
            autocorr = np.corrcoef(series_clean[:-lag], series_clean[lag:])[0, 1]
            if not np.isnan(autocorr):
                features[f"{prefix}_autocorr_lag{lag}"] = autocorr

        return features

    def _extract_peaks(self, series: np.ndarray, col: str) -> dict[str, float]:
        """Extract peak and trough related features."""
        from scipy.signal import find_peaks

        features = {}
        prefix = col

        if len(series) < 3:
            return features

        series_clean = series[~np.isnan(series)]
        if len(series_clean) < 3:
            return features

        # Find peaks
        peaks, _ = find_peaks(series_clean)
        troughs, _ = find_peaks(-series_clean)

        features[f"{prefix}_n_peaks"] = len(peaks)
        features[f"{prefix}_n_troughs"] = len(troughs)

        if len(peaks) > 0:
            features[f"{prefix}_peak_mean"] = np.mean(series_clean[peaks])
            features[f"{prefix}_peak_max"] = np.max(series_clean[peaks])

        if len(troughs) > 0:
            features[f"{prefix}_trough_mean"] = np.mean(series_clean[troughs])
            features[f"{prefix}_trough_min"] = np.min(series_clean[troughs])

        return features

    def _extract_trends(self, series: np.ndarray, col: str) -> dict[str, float]:
        """Extract trend-related features."""
        features = {}
        prefix = col

        if len(series) < 2:
            return features

        series_clean = series[~np.isnan(series)]
        if len(series_clean) < 2:
            return features

        # Linear trend (slope)
        x = np.arange(len(series_clean))
        slope, intercept = np.polyfit(x, series_clean, 1)
        features[f"{prefix}_trend_slope"] = slope
        features[f"{prefix}_trend_intercept"] = intercept

        # First and last differences
        features[f"{prefix}_first_value"] = series_clean[0]
        features[f"{prefix}_last_value"] = series_clean[-1]
        features[f"{prefix}_change"] = series_clean[-1] - series_clean[0]

        # Mean absolute change
        features[f"{prefix}_mean_abs_change"] = np.mean(np.abs(np.diff(series_clean)))

        # Mean change
        features[f"{prefix}_mean_change"] = np.mean(np.diff(series_clean))

        return features

    def _extract_rolling(self, series: np.ndarray, col: str) -> dict[str, float]:
        """Extract rolling window features."""
        features = {}
        prefix = col

        series_clean = series[~np.isnan(series)]

        for window in self.config.window_sizes:
            if len(series_clean) < window:
                continue

            # Convert to pandas for rolling operations
            s = pd.Series(series_clean)

            rolling = s.rolling(window=window)
            features[f"{prefix}_rolling{window}_mean_of_means"] = rolling.mean().mean()
            features[f"{prefix}_rolling{window}_max_of_means"] = rolling.mean().max()
            features[f"{prefix}_rolling{window}_std_of_stds"] = rolling.std().std()

        return features

    def _extract_fft(self, series: np.ndarray, col: str) -> dict[str, float]:
        """Extract FFT (frequency domain) features."""
        features = {}
        prefix = col

        series_clean = series[~np.isnan(series)]
        if len(series_clean) < 4:
            return features

        # Compute FFT
        fft_vals = np.fft.fft(series_clean)
        fft_abs = np.abs(fft_vals)

        # Get first N coefficients (excluding DC component)
        n_coeffs = min(self.config.n_fft_coefficients, len(fft_abs) // 2)

        for i in range(1, n_coeffs + 1):
            features[f"{prefix}_fft_coeff_{i}"] = fft_abs[i]

        # Spectral energy
        features[f"{prefix}_spectral_energy"] = np.sum(fft_abs**2)

        # Dominant frequency
        dominant_idx = np.argmax(fft_abs[1 : len(fft_abs) // 2]) + 1
        features[f"{prefix}_dominant_freq_idx"] = dominant_idx

        return features

    def _extract_entropy(self, series: np.ndarray, col: str) -> dict[str, float]:
        """Extract entropy-based features (tsfresh-inspired)."""
        features = {}
        prefix = col

        series_clean = series[~np.isnan(series)]
        if len(series_clean) < 4:
            return features

        # Binned entropy
        try:
            hist, _ = np.histogram(series_clean, bins=self.config.entropy_bins)
            hist = hist[hist > 0]
            probs = hist / hist.sum()
            features[f"{prefix}_binned_entropy"] = -np.sum(probs * np.log(probs + 1e-10))
        except Exception:
            features[f"{prefix}_binned_entropy"] = 0

        # Sample entropy (simplified implementation)
        try:
            features[f"{prefix}_sample_entropy"] = self._sample_entropy(series_clean, m=2, r=0.2)
        except Exception:
            features[f"{prefix}_sample_entropy"] = 0

        # Approximate entropy
        try:
            features[f"{prefix}_approximate_entropy"] = self._approximate_entropy(series_clean, m=2, r=0.2)
        except Exception:
            features[f"{prefix}_approximate_entropy"] = 0

        return features

    def _sample_entropy(self, series: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Compute sample entropy of a time series."""
        n = len(series)
        if n < m + 2:
            return 0

        # Normalize r by std
        r = r * np.std(series)
        if r == 0:
            return 0

        def _count_matches(template_length):
            count = 0
            templates = np.array([series[i : i + template_length] for i in range(n - template_length)])
            for i in range(len(templates)):
                for j in range(i + 1, len(templates)):
                    if np.max(np.abs(templates[i] - templates[j])) < r:
                        count += 1
            return count

        a = _count_matches(m)
        b = _count_matches(m + 1)

        if a == 0 or b == 0:
            return 0

        return -np.log(b / a)

    def _approximate_entropy(self, series: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Compute approximate entropy of a time series."""
        n = len(series)
        if n < m + 2:
            return 0

        r = r * np.std(series)
        if r == 0:
            return 0

        def _phi(m_val):
            patterns = np.array([series[i : i + m_val] for i in range(n - m_val + 1)])
            counts = np.zeros(len(patterns))
            for i, pattern in enumerate(patterns):
                for other in patterns:
                    if np.max(np.abs(pattern - other)) < r:
                        counts[i] += 1
            counts = counts / len(patterns)
            return np.sum(np.log(counts + 1e-10)) / len(patterns)

        return _phi(m) - _phi(m + 1)

    def _extract_energy(self, series: np.ndarray, col: str) -> dict[str, float]:
        """Extract energy-based features (tsfresh-inspired)."""
        features = {}
        prefix = col

        series_clean = series[~np.isnan(series)]
        if len(series_clean) < 2:
            return features

        # Absolute energy: sum of squared values
        features[f"{prefix}_abs_energy"] = np.sum(series_clean**2)

        # Mean absolute change
        features[f"{prefix}_mean_abs_change"] = np.mean(np.abs(np.diff(series_clean)))

        # Mean second derivative central
        if len(series_clean) >= 3:
            second_deriv = series_clean[2:] - 2 * series_clean[1:-1] + series_clean[:-2]
            features[f"{prefix}_mean_second_deriv_central"] = np.mean(second_deriv)

        # Root mean square
        features[f"{prefix}_rms"] = np.sqrt(np.mean(series_clean**2))

        # Crest factor (peak/rms)
        rms = features[f"{prefix}_rms"]
        if rms > 0:
            features[f"{prefix}_crest_factor"] = np.max(np.abs(series_clean)) / rms

        return features

    def _extract_complexity(self, series: np.ndarray, col: str) -> dict[str, float]:
        """Extract complexity features (tsfresh-inspired)."""
        features = {}
        prefix = col

        series_clean = series[~np.isnan(series)]
        if len(series_clean) < 3:
            return features

        # CID_CE: Complexity-invariant distance
        diff = np.diff(series_clean)
        features[f"{prefix}_cid_ce"] = np.sqrt(np.sum(diff**2))

        # C3: Time series complexity (lag 1)
        if len(series_clean) >= 3:
            n = len(series_clean)
            c3 = np.sum(series_clean[2:n] * series_clean[1 : n - 1] * series_clean[0 : n - 2]) / (n - 2)
            features[f"{prefix}_c3"] = c3

        # Ratio of unique values to length
        features[f"{prefix}_ratio_unique_values"] = len(np.unique(series_clean)) / len(series_clean)

        # Has duplicate
        features[f"{prefix}_has_duplicate"] = 1 if len(np.unique(series_clean)) < len(series_clean) else 0

        # Has duplicate max
        max_val = np.max(series_clean)
        features[f"{prefix}_has_duplicate_max"] = 1 if np.sum(series_clean == max_val) > 1 else 0

        # Has duplicate min
        min_val = np.min(series_clean)
        features[f"{prefix}_has_duplicate_min"] = 1 if np.sum(series_clean == min_val) > 1 else 0

        # Sum of reoccurring values
        unique, counts = np.unique(series_clean, return_counts=True)
        reoccurring_mask = counts > 1
        features[f"{prefix}_sum_reoccurring_values"] = np.sum(unique[reoccurring_mask] * counts[reoccurring_mask])

        # Sum of reoccurring data points
        features[f"{prefix}_sum_reoccurring_data_points"] = np.sum(counts[reoccurring_mask])

        # Percentage of reoccurring data points
        features[f"{prefix}_pct_reoccurring_data_points"] = np.sum(counts[reoccurring_mask]) / len(series_clean)

        return features

    def _extract_counts(self, series: np.ndarray, col: str) -> dict[str, float]:
        """Extract count-based features (tsfresh-inspired)."""
        features = {}
        prefix = col

        series_clean = series[~np.isnan(series)]
        if len(series_clean) < 2:
            return features

        mean_val = np.mean(series_clean)

        # Count above mean
        features[f"{prefix}_count_above_mean"] = np.sum(series_clean > mean_val)

        # Count below mean
        features[f"{prefix}_count_below_mean"] = np.sum(series_clean < mean_val)

        # First location of maximum
        features[f"{prefix}_first_loc_max"] = np.argmax(series_clean) / len(series_clean)

        # First location of minimum
        features[f"{prefix}_first_loc_min"] = np.argmin(series_clean) / len(series_clean)

        # Last location of maximum
        features[f"{prefix}_last_loc_max"] = (len(series_clean) - 1 - np.argmax(series_clean[::-1])) / len(series_clean)

        # Last location of minimum
        features[f"{prefix}_last_loc_min"] = (len(series_clean) - 1 - np.argmin(series_clean[::-1])) / len(series_clean)

        # Longest strike above mean
        above_mean = series_clean > mean_val
        features[f"{prefix}_longest_strike_above_mean"] = self._longest_consecutive(above_mean)

        # Longest strike below mean
        below_mean = series_clean < mean_val
        features[f"{prefix}_longest_strike_below_mean"] = self._longest_consecutive(below_mean)

        # Number of crossings (mean)
        crossings = np.sum(np.diff(np.sign(series_clean - mean_val)) != 0)
        features[f"{prefix}_number_crossings_mean"] = crossings

        # Number of zero crossings
        zero_crossings = np.sum(np.diff(np.sign(series_clean)) != 0)
        features[f"{prefix}_number_zero_crossings"] = zero_crossings

        # Absolute sum of changes
        features[f"{prefix}_abs_sum_changes"] = np.sum(np.abs(np.diff(series_clean)))

        return features

    def _longest_consecutive(self, bool_array: np.ndarray) -> int:
        """Find longest consecutive True values in boolean array."""
        max_len = 0
        current_len = 0
        for val in bool_array:
            if val:
                current_len += 1
                max_len = max(max_len, current_len)
            else:
                current_len = 0
        return max_len

    def get_feature_set(self) -> FeatureSet:
        """Get the feature set with metadata."""
        return self._feature_set
