"""Model-based feature importance selection."""

from typing import Protocol

import numpy as np
import pandas as pd

from featcopilot.core.base import BaseSelector
from featcopilot.utils.logger import get_logger
from featcopilot.utils.task import prepare_target, validate_task

logger = get_logger(__name__)


class _ImportanceModel(Protocol):
    @property
    def feature_importances_(self) -> np.ndarray: ...

    def fit(self, X: np.ndarray, y: np.ndarray) -> object: ...


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
    task : {'auto', 'classification', 'regression'}, default='auto'
        Supervised task. Auto preserves the historical target heuristic.
        Explicit regression requires finite numeric targets; classification
        requires discrete labels.

    Examples
    --------
    >>> selector = ImportanceSelector(model='random_forest', max_features=50)
    >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        model: str = "random_forest",
        max_features: int | None = None,
        threshold: float | None = None,
        n_estimators: int = 100,
        verbose: bool = False,
        task: str = "auto",
        **kwargs,
    ):
        super().__init__(**kwargs)
        validate_task(task)
        self.model_type = model
        self.max_features = max_features
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.verbose = verbose
        self.task = task
        self._model: _ImportanceModel | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, **kwargs) -> "ImportanceSelector":
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
        from sklearn.preprocessing import LabelEncoder

        self._is_fitted = False
        self._model = None
        self._feature_scores = {}
        self._selected_features = []
        X = self._validate_input(X)
        y = prepare_target(y, self.task)

        # Encode string labels if needed
        y_encoded = y
        if pd.api.types.is_string_dtype(y) or y.dtype.kind in ("U", "S"):
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

        # Determine task type
        unique_y = len(np.unique(y_encoded))
        is_classification = (
            pd.api.types.is_string_dtype(y)
            or y.dtype.kind in ("U", "S")
            or (pd.api.types.is_integer_dtype(y_encoded) and unique_y <= len(y_encoded) * 0.1)
        )
        if self.task != "auto":
            is_classification = self.task == "classification"

        # Create model
        self._model = self._create_model(is_classification)

        # Filter to numeric columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self._numeric_cols = numeric_cols

        if not numeric_cols:
            # No numeric columns, return empty selection
            self._feature_scores = dict.fromkeys(X.columns, 0.0)
            self._select_features()
            self._is_fitted = True
            return self

        X_numeric = X[numeric_cols].fillna(0).values
        self._model.fit(X_numeric, y_encoded)

        # Get importances for numeric columns
        importances = self._model.feature_importances_
        self._feature_scores = {}
        for col in X.columns:
            if col in numeric_cols:
                idx = numeric_cols.index(col)
                self._feature_scores[col] = importances[idx]
            else:
                self._feature_scores[col] = 0.0

        # Select features
        self._select_features()

        self._is_fitted = True
        return self

    def _create_model(self, is_classification: bool) -> _ImportanceModel:
        """Create the appropriate model."""
        model: _ImportanceModel
        if self.model_type == "random_forest":
            if is_classification:
                from sklearn.ensemble import RandomForestClassifier

                model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=42, n_jobs=-1)
            else:
                from sklearn.ensemble import RandomForestRegressor

                model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=42, n_jobs=-1)

        elif self.model_type == "gradient_boosting":
            if is_classification:
                from sklearn.ensemble import GradientBoostingClassifier

                model = GradientBoostingClassifier(n_estimators=self.n_estimators, random_state=42)
            else:
                from sklearn.ensemble import GradientBoostingRegressor

                model = GradientBoostingRegressor(n_estimators=self.n_estimators, random_state=42)

        elif self.model_type == "xgboost":
            try:
                import xgboost as xgb

                if is_classification:
                    model = xgb.XGBClassifier(n_estimators=self.n_estimators, random_state=42, n_jobs=-1)
                else:
                    model = xgb.XGBRegressor(n_estimators=self.n_estimators, random_state=42, n_jobs=-1)
            except ImportError:
                if self.verbose:
                    logger.warning("XGBoost not available, falling back to RandomForest")
                return self._create_model_fallback(is_classification)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return model

    def _create_model_fallback(self, is_classification: bool) -> _ImportanceModel:
        """Fallback to RandomForest."""
        model: _ImportanceModel
        if is_classification:
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=42)
        else:
            from sklearn.ensemble import RandomForestRegressor

            model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=42)
        return model

    def _select_features(self) -> None:
        """Select features based on importance."""
        sorted_features = sorted(self._feature_scores.items(), key=lambda x: x[1], reverse=True)

        if self.threshold is not None:
            sorted_features = [(name, score) for name, score in sorted_features if score >= self.threshold]

        if self.max_features is not None:
            sorted_features = sorted_features[: self.max_features]

        self._selected_features = [name for name, _ in sorted_features]

        if self.verbose:
            logger.info(f"ImportanceSelector: Selected {len(self._selected_features)} features")

    def transform(self, X: pd.DataFrame | np.ndarray, **kwargs) -> pd.DataFrame:
        """Select features from data."""
        if not self._is_fitted:
            raise RuntimeError("Selector must be fitted before transform")

        X = self._validate_input(X)
        available = [f for f in self._selected_features if f in X.columns]
        return X[available]
