"""Unified feature selector combining multiple methods."""

from typing import Optional, Union

import numpy as np
import pandas as pd

from featcopilot.core.base import BaseSelector
from featcopilot.selection.importance import ImportanceSelector
from featcopilot.selection.redundancy import RedundancyEliminator
from featcopilot.selection.statistical import StatisticalSelector


class FeatureSelector(BaseSelector):
    """
    Unified feature selector combining multiple selection methods.

    Combines statistical tests, model importance, and redundancy
    elimination for comprehensive feature selection.

    Parameters
    ----------
    methods : list, default=['mutual_info', 'importance']
        Selection methods to use
    max_features : int, optional
        Maximum features to select
    correlation_threshold : float, default=0.95
        Threshold for redundancy elimination

    Examples
    --------
    >>> selector = FeatureSelector(
    ...     methods=['mutual_info', 'importance', 'correlation'],
    ...     max_features=50,
    ...     correlation_threshold=0.95
    ... )
    >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        methods: Optional[list[str]] = None,
        max_features: Optional[int] = None,
        correlation_threshold: float = 0.95,
        combination: str = "union",
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.methods = methods or ["mutual_info", "importance"]
        self.max_features = max_features
        self.correlation_threshold = correlation_threshold
        self.combination = combination  # 'union' or 'intersection'
        self.verbose = verbose
        self._selectors: dict[str, BaseSelector] = {}
        self._method_scores: dict[str, dict[str, float]] = {}

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **kwargs) -> "FeatureSelector":
        """
        Fit all selection methods.

        Parameters
        ----------
        X : DataFrame or ndarray
            Input features
        y : Series or ndarray
            Target variable

        Returns
        -------
        self : FeatureSelector
        """
        X = self._validate_input(X)
        y = np.array(y)

        # Initialize and fit each selector
        for method in self.methods:
            selector = self._create_selector(method)
            selector.fit(X, y)
            self._selectors[method] = selector
            self._method_scores[method] = selector.get_feature_scores()

        # Combine scores from all methods
        self._combine_scores(X.columns.tolist())

        # Apply redundancy elimination
        if self.correlation_threshold < 1.0:
            eliminator = RedundancyEliminator(
                correlation_threshold=self.correlation_threshold,
                importance_scores=self._feature_scores,
                verbose=self.verbose,
            )
            eliminator.fit(X)
            non_redundant = set(eliminator.get_selected_features())
            self._feature_scores = {k: v for k, v in self._feature_scores.items() if k in non_redundant}

        # Final selection
        self._final_selection()

        self._is_fitted = True
        return self

    def _create_selector(self, method: str) -> BaseSelector:
        """Create selector for a given method."""
        if method == "mutual_info":
            return StatisticalSelector(method="mutual_info", verbose=self.verbose)
        elif method == "f_test":
            return StatisticalSelector(method="f_test", verbose=self.verbose)
        elif method == "chi2":
            return StatisticalSelector(method="chi2", verbose=self.verbose)
        elif method == "correlation":
            return StatisticalSelector(method="correlation", verbose=self.verbose)
        elif method == "importance":
            return ImportanceSelector(model="random_forest", verbose=self.verbose)
        elif method == "xgboost":
            return ImportanceSelector(model="xgboost", verbose=self.verbose)
        else:
            raise ValueError(f"Unknown selection method: {method}")

    def _combine_scores(self, columns: list[str]) -> None:
        """Combine scores from multiple methods."""
        combined = {}

        for col in columns:
            scores = []
            for _, method_scores in self._method_scores.items():
                if col in method_scores:
                    # Normalize score to 0-1 range
                    all_scores = list(method_scores.values())
                    max_score = max(all_scores) if all_scores else 1
                    if max_score > 0:
                        normalized = method_scores[col] / max_score
                    else:
                        normalized = 0
                    scores.append(normalized)

            # Average normalized scores
            if scores:
                combined[col] = np.mean(scores)
            else:
                combined[col] = 0

        self._feature_scores = combined

    def _final_selection(self) -> None:
        """Make final feature selection."""
        sorted_features = sorted(self._feature_scores.items(), key=lambda x: x[1], reverse=True)

        if self.max_features is not None:
            sorted_features = sorted_features[: self.max_features]

        self._selected_features = [name for name, _ in sorted_features]

        if self.verbose:
            print(f"FeatureSelector: Selected {len(self._selected_features)} features")

    def transform(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> pd.DataFrame:
        """Select features from data."""
        if not self._is_fitted:
            raise RuntimeError("Selector must be fitted before transform")

        X = self._validate_input(X)
        available = [f for f in self._selected_features if f in X.columns]
        return X[available]

    def get_method_scores(self) -> dict[str, dict[str, float]]:
        """Get scores from each individual method."""
        return self._method_scores

    def get_ranking(self) -> list[tuple]:
        """Get feature ranking as list of (name, score) tuples."""
        return sorted(self._feature_scores.items(), key=lambda x: x[1], reverse=True)
