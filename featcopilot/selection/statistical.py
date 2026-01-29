"""Statistical feature selection methods."""

from typing import Optional, Union

import numpy as np
import pandas as pd

from featcopilot.core.base import BaseSelector


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

    Examples
    --------
    >>> selector = StatisticalSelector(method='mutual_info', max_features=50)
    >>> X_selected = selector.fit_transform(X, y)
    """

    METHODS = ["mutual_info", "f_test", "chi2", "correlation"]

    def __init__(
        self,
        method: str = "mutual_info",
        max_features: Optional[int] = None,
        threshold: Optional[float] = None,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if method not in self.METHODS:
            raise ValueError(f"Method must be one of {self.METHODS}")

        self.method = method
        self.max_features = max_features
        self.threshold = threshold
        self.verbose = verbose

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **kwargs
    ) -> "StatisticalSelector":
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
        X = self._validate_input(X)
        y = np.array(y)

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

        self._feature_scores = dict(zip(X.columns, scores))

        # Select features
        self._select_features()

        self._is_fitted = True
        return self

    def _compute_mutual_info(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Compute mutual information scores."""
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

        # Determine if classification or regression
        unique_y = len(np.unique(y))
        is_classification = unique_y < 20 and y.dtype in [np.int32, np.int64, "object"]

        X_array = X.fillna(0).values

        if is_classification:
            scores = mutual_info_classif(X_array, y, random_state=42)
        else:
            scores = mutual_info_regression(X_array, y, random_state=42)

        return scores

    def _compute_f_test(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Compute F-test scores."""
        from sklearn.feature_selection import f_classif, f_regression

        unique_y = len(np.unique(y))
        is_classification = unique_y < 20

        X_array = X.fillna(0).values

        if is_classification:
            scores, _ = f_classif(X_array, y)
        else:
            scores, _ = f_regression(X_array, y)

        # Handle NaN scores
        scores = np.nan_to_num(scores, 0)
        return scores

    def _compute_chi2(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Compute chi-square scores (for non-negative features)."""
        from sklearn.feature_selection import chi2

        X_array = X.fillna(0).values

        # Chi2 requires non-negative values
        X_positive = X_array - X_array.min(axis=0) + 1e-8

        try:
            scores, _ = chi2(X_positive, y)
            scores = np.nan_to_num(scores, 0)
        except Exception:
            # Fallback to mutual information
            scores = self._compute_mutual_info(X, y)

        return scores

    def _compute_correlation(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Compute absolute correlation with target."""
        scores = []
        for col in X.columns:
            try:
                corr = np.abs(np.corrcoef(X[col].fillna(0).values, y)[0, 1])
                scores.append(corr if not np.isnan(corr) else 0)
            except Exception:
                scores.append(0)

        return np.array(scores)

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
            print(f"StatisticalSelector: Selected {len(self._selected_features)} features")

    def transform(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> pd.DataFrame:
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
