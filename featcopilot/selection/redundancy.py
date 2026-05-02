"""Redundancy elimination through correlation analysis."""

import warnings

import numpy as np
import pandas as pd

from featcopilot.core.base import BaseSelector
from featcopilot.utils.logger import get_logger

logger = get_logger(__name__)


class RedundancyEliminator(BaseSelector):
    """
    Eliminate redundant features based on pairwise correlation.

    For every pair of numeric columns whose ``|correlation|`` reaches
    ``correlation_threshold``, exactly one column is removed. The choice
    follows three rules so original input columns are never silently
    dropped:

    * **original vs derived** — the **derived** column is always removed;
      the original is kept regardless of importance.
    * **original vs original** — neither is removed; original input
      columns are categorically protected from this selector.
    * **derived vs derived** — the column with **lower importance** (per
      ``importance_scores``) is removed; ties keep the first column seen.

    Non-numeric columns (categorical / string / datetime) are not part of
    the correlation analysis and pass through unchanged.

    Parameters
    ----------
    correlation_threshold : float, default=0.95
        Correlation threshold for redundancy
    method : str, default='pearson'
        Correlation method ('pearson', 'spearman', 'kendall')
    importance_scores : dict[str, float], optional
        Per-column importance used to break derived-vs-derived ties.
    original_features : set[str], optional
        Set of original feature names. Originals are categorically
        protected from removal (see rules above).
    original_preference : float, optional
        **Deprecated, accepted for backward compatibility only.** Has no
        effect; passing any non-``None`` value raises a ``FutureWarning``.
        Originals are now categorically protected regardless of importance,
        so there is no tunable trade-off to express. Kept in its original
        positional slot (between ``original_features`` and ``verbose``) so
        existing positional callers don't silently rebind values.
    verbose : bool, default=False
        Emit per-pair logging while pruning correlated pairs.

    Examples
    --------
    >>> eliminator = RedundancyEliminator(correlation_threshold=0.95)
    >>> X_reduced = eliminator.fit_transform(X, y)
    """

    def __init__(
        self,
        correlation_threshold: float = 0.95,
        method: str = "pearson",
        importance_scores: dict[str, float] | None = None,
        original_features: set[str] | None = None,
        original_preference: float | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        if original_preference is not None:
            warnings.warn(
                "`original_preference` is deprecated and has no effect. "
                "Original input columns are now categorically protected "
                "from removal regardless of importance — there is no "
                "tunable trade-off to express.",
                FutureWarning,
                stacklevel=2,
            )
        super().__init__(**kwargs)
        self.correlation_threshold = correlation_threshold
        self.method = method
        self.importance_scores = importance_scores or {}
        self.original_features = original_features or set()
        # Persist the deprecated ``original_preference`` parameter so legacy
        # callers that read ``eliminator.original_preference`` after
        # construction don't crash with ``AttributeError``. The value has
        # no effect on behavior — originals are categorically protected
        # regardless — but the attribute is preserved for read-only
        # backward compatibility.
        self.original_preference = original_preference
        self.verbose = verbose
        self._correlation_matrix: pd.DataFrame | None = None

    def fit_transform(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Fit and transform in one step (y is optional for this selector)."""
        return self.fit(X, y, **kwargs).transform(X, **kwargs)

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        importance_scores: dict[str, float] | None = None,
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

        # Compute correlation matrix (only for numeric columns)
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        self._correlation_matrix = X[numeric_cols].corr(method=self.method)

        # Find redundant features among numeric columns
        self._find_redundant_features(numeric_cols, non_numeric_cols)

        self._is_fitted = True
        return self

    def _find_redundant_features(self, columns: list[str], non_numeric_cols: list[str]) -> None:
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
                    is_orig1 = col1 in self.original_features
                    is_orig2 = col2 in self.original_features

                    # Never remove an original feature if the other is derived
                    if is_orig1 and not is_orig2:
                        to_remove.add(col2)
                        if self.verbose:
                            logger.info(f"Removing {col2} (derived, corr={corr:.3f} with original {col1})")
                        continue
                    elif is_orig2 and not is_orig1:
                        to_remove.add(col1)
                        if self.verbose:
                            logger.info(f"Removing {col1} (derived, corr={corr:.3f} with original {col2})")
                        break

                    # Both are original — never remove either
                    if is_orig1 and is_orig2:
                        continue

                    # Both are derived — remove the one with lower importance
                    imp1 = self.importance_scores.get(col1, 0)
                    imp2 = self.importance_scores.get(col2, 0)

                    if imp1 >= imp2:
                        to_remove.add(col2)
                        if self.verbose:
                            logger.info(f"Removing {col2} (derived, corr={corr:.3f} with {col1})")
                    else:
                        to_remove.add(col1)
                        if self.verbose:
                            logger.info(f"Removing {col1} (derived, corr={corr:.3f} with {col2})")
                        break  # col1 is removed, move to next

        # Selected features are those not removed (numeric) plus all non-numeric columns
        # Non-numeric columns (categorical/text) are always preserved
        self._selected_features = [c for c in columns if c not in to_remove]
        self._selected_features.extend(non_numeric_cols)  # Always include non-numeric
        self._removed_features = list(to_remove)

        if self.verbose:
            logger.info(f"RedundancyEliminator: Removed {len(to_remove)} redundant features")

    def transform(self, X: pd.DataFrame | np.ndarray, **kwargs) -> pd.DataFrame:
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

    def get_correlation_matrix(self) -> pd.DataFrame | None:
        """Get the computed correlation matrix."""
        return self._correlation_matrix
