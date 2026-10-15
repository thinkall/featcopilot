"""Statistical feature selection methods."""

import numpy as np
import pandas as pd

from featcopilot.core.base import BaseSelector
from featcopilot.utils.logger import get_logger
from featcopilot.utils.task import prepare_target, validate_task

logger = get_logger(__name__)


class StatisticalSelector(BaseSelector):
    """
    Feature selector based on statistical tests.

    Uses statistical tests to evaluate feature relevance:
    - Mutual information
    - Chi-square test (categorical)
    - F-test (ANOVA)
    - Correlation with target

    Parameters
    ----------
    method : str, default='mutual_info'
        Selection method ('mutual_info', 'f_test', 'chi2', 'correlation')
    max_features : int, optional
        Maximum features to select
    threshold : float, optional
        Minimum score threshold
    task : {'auto', 'classification', 'regression'}, default='auto'
        Supervised task. Auto preserves the historical target heuristic.
        Explicit regression requires finite numeric targets; classification
        requires discrete labels. Chi-square does not support explicit regression.

    Examples
    --------
    >>> selector = StatisticalSelector(method='mutual_info', max_features=50)
    >>> X_selected = selector.fit_transform(X, y)
    """

    METHODS = ["mutual_info", "f_test", "chi2", "correlation"]

    def __init__(
        self,
        method: str = "mutual_info",
        max_features: int | None = None,
        threshold: float | None = None,
        verbose: bool = False,
        task: str = "auto",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if method not in self.METHODS:
            raise ValueError(f"Method must be one of {self.METHODS}")
        validate_task(task)

        self.method = method
        self.max_features = max_features
        self.threshold = threshold
        self.verbose = verbose
        self.task = task

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, **kwargs) -> "StatisticalSelector":
        """
        Fit selector to compute feature scores.

        Parameters
        ----------
        X : DataFrame or ndarray
            Input features
        y : Series or ndarray
            Target variable

        Returns
        -------
        self : StatisticalSelector
        """
        self._is_fitted = False
        self._feature_scores = {}
        self._selected_features = []
        X = self._validate_input(X)
        y = prepare_target(y, self.task)

        # Compute scores based on method
        if self.method == "mutual_info":
            scores = self._compute_mutual_info(X, y)
        elif self.method == "f_test":
            scores = self._compute_f_test(X, y)
        elif self.method == "chi2":
            scores = self._compute_chi2(X, y)
        elif self.method == "correlation":
            scores = self._compute_correlation(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self._feature_scores = dict(zip(X.columns, scores, strict=True))

        # Select features
        self._select_features()

        self._is_fitted = True
        return self

    def _compute_mutual_info(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Compute mutual information scores."""
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        from sklearn.preprocessing import LabelEncoder

        # Encode string labels if needed
        y_encoded = y
        if pd.api.types.is_string_dtype(y) or y.dtype.kind in ("U", "S"):
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

        # Determine if classification or regression
        # Check if target is categorical (object type) or has discrete integer values
        unique_y = len(np.unique(y_encoded))
        is_classification = (
            pd.api.types.is_string_dtype(y)
            or y.dtype.kind in ("U", "S")
            or (pd.api.types.is_integer_dtype(y_encoded) and unique_y <= len(y_encoded) * 0.1)
        )
        if self.task != "auto":
            is_classification = self.task == "classification"

        # Filter to numeric columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        scores = np.zeros(len(X.columns))

        if numeric_cols:
            X_numeric = X[numeric_cols].fillna(0).values
            numeric_indices = [X.columns.get_loc(c) for c in numeric_cols]

            if is_classification:
                numeric_scores = mutual_info_classif(X_numeric, y_encoded, random_state=42)
            else:
                numeric_scores = mutual_info_regression(X_numeric, y_encoded, random_state=42)

            for i, idx in enumerate(numeric_indices):
                scores[idx] = numeric_scores[i]

        return scores

    def _compute_f_test(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Compute F-test scores."""
        from sklearn.feature_selection import f_classif, f_regression
        from sklearn.preprocessing import LabelEncoder

        # Encode string labels if needed
        y_encoded = y
        if pd.api.types.is_string_dtype(y) or y.dtype.kind in ("U", "S"):
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

        # Determine if classification or regression
        unique_y = len(np.unique(y_encoded))
        is_classification = (
            pd.api.types.is_string_dtype(y)
            or y.dtype.kind in ("U", "S")
            or (pd.api.types.is_integer_dtype(y_encoded) and unique_y <= len(y_encoded) * 0.1)
        )
        if self.task != "auto":
            is_classification = self.task == "classification"

        # Filter to numeric columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        scores = np.zeros(len(X.columns))

        if numeric_cols:
            X_numeric = X[numeric_cols].fillna(0).values
            numeric_indices = [X.columns.get_loc(c) for c in numeric_cols]

            if is_classification:
                numeric_scores, _ = f_classif(X_numeric, y_encoded)
            else:
                numeric_scores, _ = f_regression(X_numeric, y_encoded)

            # Handle NaN scores
            numeric_scores = np.nan_to_num(numeric_scores, nan=0.0)

            for i, idx in enumerate(numeric_indices):
                scores[idx] = numeric_scores[i]

        return scores

    def _compute_chi2(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Compute chi-square scores (for non-negative features)."""
        from sklearn.feature_selection import chi2
        from sklearn.preprocessing import LabelEncoder

        if self.task == "regression":
            raise ValueError("method='chi2' requires classification; use 'mutual_info' or 'f_test' for regression")

        # Encode string labels if needed
        y_encoded = y
        if pd.api.types.is_string_dtype(y) or y.dtype.kind in ("U", "S"):
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

        # Filter to numeric columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        scores = np.zeros(len(X.columns))

        if numeric_cols:
            X_numeric = X[numeric_cols].fillna(0).values
            numeric_indices = [X.columns.get_loc(c) for c in numeric_cols]

            # Chi2 requires non-negative values
            X_positive = X_numeric - X_numeric.min(axis=0) + 1e-8

            try:
                numeric_scores, _ = chi2(X_positive, y_encoded)
                numeric_scores = np.nan_to_num(numeric_scores, nan=0.0)
            except Exception as exc:
                if self.task != "auto":
                    raise
                logger.warning(f"Chi-square scoring failed; falling back to mutual information: {exc}")
                # Fallback to mutual information
                return self._compute_mutual_info(X, y)

            for i, idx in enumerate(numeric_indices):
                scores[idx] = numeric_scores[i]

        return scores

    def _compute_correlation(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Compute absolute correlation with target."""
        from sklearn.preprocessing import LabelEncoder

        # Encode string labels if needed
        y_encoded = y
        if pd.api.types.is_string_dtype(y) or y.dtype.kind in ("U", "S"):
            le = LabelEncoder()
            y_encoded = le.fit_transform(y).astype(float)

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        scores = np.zeros(len(X.columns))
        target_values = np.asarray(y_encoded, dtype=float)

        for col in numeric_cols:
            try:
                idx = X.columns.get_loc(col)
                feature_values = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(feature_values) & np.isfinite(target_values)
                if valid.sum() < 2:
                    continue

                feature_valid = feature_values[valid]
                target_valid = target_values[valid]
                if np.ptp(feature_valid) == 0 or np.ptp(target_valid) == 0:
                    continue

                corr = float(np.abs(np.corrcoef(feature_valid, target_valid)[0, 1]))
                scores[idx] = corr if np.isfinite(corr) else 0.0
            except (TypeError, ValueError, FloatingPointError) as exc:
                logger.warning(f"Could not compute correlation for feature {col!r}: {exc}")

        return scores

    def _select_features(self) -> None:
        """Select features based on scores."""
        # Sort features by score
        sorted_features = sorted(self._feature_scores.items(), key=lambda x: x[1], reverse=True)

        # Apply threshold
        if self.threshold is not None:
            sorted_features = [(name, score) for name, score in sorted_features if score >= self.threshold]

        # Apply max_features limit
        if self.max_features is not None:
            sorted_features = sorted_features[: self.max_features]

        self._selected_features = [name for name, _ in sorted_features]

        if self.verbose:
            logger.info(f"StatisticalSelector: Selected {len(self._selected_features)} features")

    def transform(self, X: pd.DataFrame | np.ndarray, **kwargs) -> pd.DataFrame:
        """
        Select features from data.

        Parameters
        ----------
        X : DataFrame or ndarray
            Input features

        Returns
        -------
        X_selected : DataFrame
            Data with only selected features
        """
        if not self._is_fitted:
            raise RuntimeError("Selector must be fitted before transform")

        X = self._validate_input(X)

        # Keep only selected features that exist in X
        available = [f for f in self._selected_features if f in X.columns]
        return X[available]
