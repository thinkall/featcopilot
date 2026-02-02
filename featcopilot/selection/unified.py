"""Unified feature selector combining multiple methods."""

from typing import Optional, Union

import numpy as np
import pandas as pd

from featcopilot.core.base import BaseSelector
from featcopilot.selection.importance import ImportanceSelector
from featcopilot.selection.redundancy import RedundancyEliminator
from featcopilot.selection.statistical import StatisticalSelector
from featcopilot.utils.logger import get_logger

logger = get_logger(__name__)


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
        original_features: Optional[set[str]] = None,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.methods = methods or ["mutual_info", "importance"]
        self.max_features = max_features
        self.correlation_threshold = correlation_threshold
        self.combination = combination  # 'union' or 'intersection'
        self.original_features = original_features or set()
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

        # Identify categorical/text columns (can't be scored by numeric methods)
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        # Initialize and fit each selector
        for method in self.methods:
            selector = self._create_selector(method)
            selector.fit(X, y)
            self._selectors[method] = selector
            self._method_scores[method] = selector.get_feature_scores()

        # Combine scores from all methods
        self._combine_scores(X.columns.tolist())

        # Give categorical columns a minimum score so they're not filtered out
        # Original categorical columns are important for models that can handle them
        if categorical_cols:
            # Get the median score of numeric features to use as baseline for categorical
            numeric_scores = [v for k, v in self._feature_scores.items() if k in numeric_cols and v > 0]
            if numeric_scores:
                baseline_score = np.median(numeric_scores)
            else:
                baseline_score = 0.5  # Default if no numeric scores

            for col in categorical_cols:
                if col in self.original_features:
                    # Original categorical columns get a baseline score
                    self._feature_scores[col] = max(self._feature_scores.get(col, 0), baseline_score)

        # Apply redundancy elimination
        if self.correlation_threshold < 1.0:
            eliminator = RedundancyEliminator(
                correlation_threshold=self.correlation_threshold,
                importance_scores=self._feature_scores,
                original_features=self.original_features,
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

        # Always include original features first
        original_selected = []
        derived_selected = []

        for name, _score in sorted_features:
            if name in self.original_features:
                original_selected.append(name)
            else:
                derived_selected.append(name)

        # Apply max_features limit only to derived features
        if self.max_features is not None:
            # Reserve slots for original features, then fill with top derived
            n_derived = max(0, self.max_features - len(original_selected))
            derived_selected = derived_selected[:n_derived]

        self._selected_features = original_selected + derived_selected

        if self.verbose:
            logger.info(
                f"FeatureSelector: Selected {len(self._selected_features)} features "
                f"({len(original_selected)} original + {len(derived_selected)} derived)"
            )

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
