"""Redundancy elimination through correlation analysis."""

from typing import Optional, Union

import numpy as np
import pandas as pd

from featcopilot.core.base import BaseSelector


class RedundancyEliminator(BaseSelector):
    """
    Eliminate redundant features based on correlation.

    Removes highly correlated features, keeping the one with
    higher importance (if provided) or the first one.

    Parameters
    ----------
    correlation_threshold : float, default=0.95
        Correlation threshold for redundancy
    method : str, default='pearson'
        Correlation method ('pearson', 'spearman', 'kendall')

    Examples
    --------
    >>> eliminator = RedundancyEliminator(correlation_threshold=0.95)
    >>> X_reduced = eliminator.fit_transform(X, y)
    """

    def __init__(
        self,
        correlation_threshold: float = 0.95,
        method: str = "pearson",
        importance_scores: Optional[dict[str, float]] = None,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.correlation_threshold = correlation_threshold
        self.method = method
        self.importance_scores = importance_scores or {}
        self.verbose = verbose
        self._correlation_matrix: Optional[pd.DataFrame] = None

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Fit and transform in one step (y is optional for this selector)."""
        return self.fit(X, y, **kwargs).transform(X, **kwargs)

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        importance_scores: Optional[dict[str, float]] = None,
        **kwargs,
    ) -> "RedundancyEliminator":
        """
        Fit eliminator by computing correlations.

        Parameters
        ----------
        X : DataFrame or ndarray
            Input features
        y : Series or ndarray, optional
            Target variable (unused)
        importance_scores : dict, optional
            Pre-computed importance scores

        Returns
        -------
        self : RedundancyEliminator
        """
        X = self._validate_input(X)

        if importance_scores:
            self.importance_scores = importance_scores

        # Compute correlation matrix
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        self._correlation_matrix = X[numeric_cols].corr(method=self.method)

        # Find redundant features
        self._find_redundant_features(numeric_cols)

        self._is_fitted = True
        return self

    def _find_redundant_features(self, columns: list[str]) -> None:
        """Identify and mark redundant features for removal."""
        to_remove: set[str] = set()
        checked_pairs: set[tuple] = set()

        for i, col1 in enumerate(columns):
            if col1 in to_remove:
                continue

            for col2 in columns[i + 1 :]:
                if col2 in to_remove:
                    continue

                pair = tuple(sorted([col1, col2]))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)

                # Get correlation
                corr = abs(self._correlation_matrix.loc[col1, col2])

                if corr >= self.correlation_threshold:
                    # Decide which to remove based on importance
                    imp1 = self.importance_scores.get(col1, 0)
                    imp2 = self.importance_scores.get(col2, 0)

                    if imp1 >= imp2:
                        to_remove.add(col2)
                        if self.verbose:
                            print(f"Removing {col2} (corr={corr:.3f} with {col1})")
                    else:
                        to_remove.add(col1)
                        if self.verbose:
                            print(f"Removing {col1} (corr={corr:.3f} with {col2})")
                        break  # col1 is removed, move to next

        # Selected features are those not removed
        self._selected_features = [c for c in columns if c not in to_remove]
        self._removed_features = list(to_remove)

        if self.verbose:
            print(f"RedundancyEliminator: Removed {len(to_remove)} redundant features")

    def transform(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> pd.DataFrame:
        """Remove redundant features."""
        if not self._is_fitted:
            raise RuntimeError("Eliminator must be fitted before transform")

        X = self._validate_input(X)

        # Keep selected features plus any non-numeric columns
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        keep_cols = [c for c in self._selected_features if c in X.columns]
        keep_cols.extend([c for c in non_numeric if c not in keep_cols])

        return X[keep_cols]

    def get_removed_features(self) -> list[str]:
        """Get list of removed redundant features."""
        return getattr(self, "_removed_features", [])

    def get_correlation_matrix(self) -> Optional[pd.DataFrame]:
        """Get the computed correlation matrix."""
        return self._correlation_matrix
