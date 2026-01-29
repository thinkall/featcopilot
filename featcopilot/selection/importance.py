"""Model-based feature importance selection."""

from typing import Optional, Union

import numpy as np
import pandas as pd

from featcopilot.core.base import BaseSelector


class ImportanceSelector(BaseSelector):
    """
    Feature selector based on model importance scores.

    Uses tree-based models to evaluate feature importance.

    Parameters
    ----------
    model : str, default='random_forest'
        Model to use ('random_forest', 'gradient_boosting', 'xgboost')
    max_features : int, optional
        Maximum features to select
    threshold : float, optional
        Minimum importance threshold

    Examples
    --------
    >>> selector = ImportanceSelector(model='random_forest', max_features=50)
    >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        model: str = "random_forest",
        max_features: Optional[int] = None,
        threshold: Optional[float] = None,
        n_estimators: int = 100,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_type = model
        self.max_features = max_features
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.verbose = verbose
        self._model = None

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **kwargs
    ) -> "ImportanceSelector":
        """
        Fit selector using a tree model.

        Parameters
        ----------
        X : DataFrame or ndarray
            Input features
        y : Series or ndarray
            Target variable

        Returns
        -------
        self : ImportanceSelector
        """
        X = self._validate_input(X)
        y = np.array(y)

        # Determine task type
        unique_y = len(np.unique(y))
        is_classification = unique_y < 20 and not np.issubdtype(y.dtype, np.floating)

        # Create model
        self._model = self._create_model(is_classification)

        # Fit model
        X_array = X.fillna(0).values
        self._model.fit(X_array, y)

        # Get importances
        importances = self._model.feature_importances_
        self._feature_scores = dict(zip(X.columns, importances))

        # Select features
        self._select_features()

        self._is_fitted = True
        return self

    def _create_model(self, is_classification: bool):
        """Create the appropriate model."""
        if self.model_type == "random_forest":
            if is_classification:
                from sklearn.ensemble import RandomForestClassifier

                return RandomForestClassifier(n_estimators=self.n_estimators, random_state=42, n_jobs=-1)
            else:
                from sklearn.ensemble import RandomForestRegressor

                return RandomForestRegressor(n_estimators=self.n_estimators, random_state=42, n_jobs=-1)

        elif self.model_type == "gradient_boosting":
            if is_classification:
                from sklearn.ensemble import GradientBoostingClassifier

                return GradientBoostingClassifier(n_estimators=self.n_estimators, random_state=42)
            else:
                from sklearn.ensemble import GradientBoostingRegressor

                return GradientBoostingRegressor(n_estimators=self.n_estimators, random_state=42)

        elif self.model_type == "xgboost":
            try:
                import xgboost as xgb

                if is_classification:
                    return xgb.XGBClassifier(n_estimators=self.n_estimators, random_state=42, n_jobs=-1)
                else:
                    return xgb.XGBRegressor(n_estimators=self.n_estimators, random_state=42, n_jobs=-1)
            except ImportError:
                if self.verbose:
                    print("XGBoost not available, falling back to RandomForest")
                return self._create_model_fallback(is_classification)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _create_model_fallback(self, is_classification: bool):
        """Fallback to RandomForest."""
        if is_classification:
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(n_estimators=self.n_estimators, random_state=42)
        else:
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(n_estimators=self.n_estimators, random_state=42)

    def _select_features(self) -> None:
        """Select features based on importance."""
        sorted_features = sorted(self._feature_scores.items(), key=lambda x: x[1], reverse=True)

        if self.threshold is not None:
            sorted_features = [(name, score) for name, score in sorted_features if score >= self.threshold]

        if self.max_features is not None:
            sorted_features = sorted_features[: self.max_features]

        self._selected_features = [name for name, _ in sorted_features]

        if self.verbose:
            print(f"ImportanceSelector: Selected {len(self._selected_features)} features")

    def transform(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> pd.DataFrame:
        """Select features from data."""
        if not self._is_fitted:
            raise RuntimeError("Selector must be fitted before transform")

        X = self._validate_input(X)
        available = [f for f in self._selected_features if f in X.columns]
        return X[available]
