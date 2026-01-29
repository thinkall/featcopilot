"""Base classes for feature engineering engines and selectors."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class EngineConfig(BaseModel):
    """Configuration for feature engineering engines."""

    name: str = Field(description="Engine name")
    enabled: bool = Field(default=True, description="Whether engine is enabled")
    max_features: Optional[int] = Field(default=None, description="Max features to generate")
    verbose: bool = Field(default=False, description="Verbose output")


class BaseEngine(ABC):
    """
    Abstract base class for feature engineering engines.

    All engines (tabular, timeseries, relational, llm) inherit from this class.
    """

    def __init__(self, config: Optional[EngineConfig] = None, **kwargs):
        self.config = config or EngineConfig(name=self.__class__.__name__, **kwargs)
        self._is_fitted = False
        self._feature_names: list[str] = []
        self._feature_metadata: dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        """Check if engine has been fitted."""
        return self._is_fitted

    @abstractmethod
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs,
    ) -> "BaseEngine":
        """
        Fit the engine to the data.

        Parameters
        ----------
        X : DataFrame or ndarray
            Input features
        y : Series or ndarray, optional
            Target variable
        **kwargs : dict
            Additional parameters

        Returns
        -------
        self : BaseEngine
            Fitted engine
        """
        pass

    @abstractmethod
    def transform(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> pd.DataFrame:
        """
        Transform data to generate new features.

        Parameters
        ----------
        X : DataFrame or ndarray
            Input features
        **kwargs : dict
            Additional parameters

        Returns
        -------
        X_transformed : DataFrame
            Transformed features
        """
        pass

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y, **kwargs).transform(X, **kwargs)

    def get_feature_names(self) -> list[str]:
        """Get names of generated features."""
        return self._feature_names.copy()

    def get_feature_metadata(self) -> dict[str, Any]:
        """Get metadata for generated features."""
        return self._feature_metadata.copy()

    def _validate_input(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Convert input to DataFrame and validate."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        elif not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected DataFrame or ndarray, got {type(X)}")
        return X


class SelectorConfig(BaseModel):
    """Configuration for feature selectors."""

    max_features: Optional[int] = Field(default=None, description="Max features to select")
    min_importance: float = Field(default=0.0, description="Minimum importance threshold")
    correlation_threshold: float = Field(default=0.95, description="Threshold for correlation-based elimination")


class BaseSelector(ABC):
    """
    Abstract base class for feature selection.

    Handles selection of most important/relevant features from generated set.
    """

    def __init__(self, config: Optional[SelectorConfig] = None, **kwargs):
        self.config = config or SelectorConfig(**kwargs)
        self._is_fitted = False
        self._selected_features: list[str] = []
        self._feature_scores: dict[str, float] = {}

    @property
    def is_fitted(self) -> bool:
        """Check if selector has been fitted."""
        return self._is_fitted

    @abstractmethod
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **kwargs) -> "BaseSelector":
        """
        Fit the selector to determine feature importance.

        Parameters
        ----------
        X : DataFrame or ndarray
            Input features
        y : Series or ndarray
            Target variable
        **kwargs : dict
            Additional parameters

        Returns
        -------
        self : BaseSelector
            Fitted selector
        """
        pass

    @abstractmethod
    def transform(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> pd.DataFrame:
        """
        Transform data to keep only selected features.

        Parameters
        ----------
        X : DataFrame or ndarray
            Input features
        **kwargs : dict
            Additional parameters

        Returns
        -------
        X_selected : DataFrame
            Data with only selected features
        """
        pass

    def fit_transform(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **kwargs
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y, **kwargs).transform(X, **kwargs)

    def get_selected_features(self) -> list[str]:
        """Get names of selected features."""
        return self._selected_features.copy()

    def get_feature_scores(self) -> dict[str, float]:
        """Get importance scores for all features."""
        return self._feature_scores.copy()

    def _validate_input(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Convert input to DataFrame and validate."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        elif not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected DataFrame or ndarray, got {type(X)}")
        return X
