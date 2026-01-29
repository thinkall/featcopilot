"""Text feature engineering engine.

Generates features from text data using embeddings and NLP techniques.
"""

from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from pydantic import Field

from featcopilot.core.base import BaseEngine, EngineConfig
from featcopilot.core.feature import FeatureSet


class TextEngineConfig(EngineConfig):
    """Configuration for text feature engine."""

    name: str = "TextEngine"
    features: list[str] = Field(
        default_factory=lambda: ["length", "word_count", "char_stats"],
        description="Feature types to extract",
    )
    max_vocab_size: int = Field(default=5000, description="Max vocabulary size for TF-IDF")
    n_components: int = Field(default=50, description="Components for dimensionality reduction")


class TextEngine(BaseEngine):
    """
    Text feature engineering engine.

    Extracts features from text columns including:
    - Length and character statistics
    - Word count features
    - TF-IDF features (optional)
    - Sentiment features (optional)
    - Embedding features (with LLM integration)

    Parameters
    ----------
    features : list
        Feature types to extract
    max_vocab_size : int, default=5000
        Maximum vocabulary size for TF-IDF

    Examples
    --------
    >>> engine = TextEngine(features=['length', 'word_count', 'tfidf'])
    >>> X_features = engine.fit_transform(text_df)
    """

    def __init__(
        self,
        features: Optional[list[str]] = None,
        max_vocab_size: int = 5000,
        max_features: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ):
        config = TextEngineConfig(
            features=features or ["length", "word_count", "char_stats"],
            max_vocab_size=max_vocab_size,
            max_features=max_features,
            verbose=verbose,
            **kwargs,
        )
        super().__init__(config=config)
        self.config: TextEngineConfig = config
        self._text_columns: list[str] = []
        self._vectorizers: dict[str, Any] = {}
        self._feature_set = FeatureSet()

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        text_columns: Optional[list[str]] = None,
        **kwargs,
    ) -> "TextEngine":
        """
        Fit the engine to identify and process text columns.

        Parameters
        ----------
        X : DataFrame
            Input data
        y : Series, optional
            Target variable
        text_columns : list, optional
            Specific columns to treat as text

        Returns
        -------
        self : TextEngine
        """
        X = self._validate_input(X)

        # Identify text columns
        if text_columns:
            self._text_columns = text_columns
        else:
            self._text_columns = X.select_dtypes(include=["object"]).columns.tolist()
            # Filter to likely text columns (not IDs, not low cardinality)
            self._text_columns = [
                col for col in self._text_columns if X[col].str.len().mean() > 10 and X[col].nunique() > 10
            ]

        if self.config.verbose:
            print(f"TextEngine: Found {len(self._text_columns)} text columns")

        # Fit TF-IDF vectorizers if needed
        if "tfidf" in self.config.features:
            self._fit_tfidf(X)

        self._is_fitted = True
        return self

    def _fit_tfidf(self, X: pd.DataFrame) -> None:
        """Fit TF-IDF vectorizers for text columns."""
        try:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer

            for col in self._text_columns:
                texts = X[col].fillna("").astype(str)
                vectorizer = TfidfVectorizer(max_features=self.config.max_vocab_size, stop_words="english")
                tfidf_matrix = vectorizer.fit_transform(texts)

                # Reduce dimensions with SVD
                n_components = min(self.config.n_components, tfidf_matrix.shape[1])
                if n_components > 0:
                    svd = TruncatedSVD(n_components=n_components)
                    svd.fit(tfidf_matrix)
                    self._vectorizers[col] = {"vectorizer": vectorizer, "svd": svd}

        except ImportError:
            if self.config.verbose:
                print("TextEngine: sklearn not available for TF-IDF, skipping")

    def transform(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> pd.DataFrame:
        """
        Extract text features.

        Parameters
        ----------
        X : DataFrame
            Input data

        Returns
        -------
        X_features : DataFrame
            Extracted features
        """
        if not self._is_fitted:
            raise RuntimeError("Engine must be fitted before transform")

        X = self._validate_input(X)
        result = X.copy()

        for col in self._text_columns:
            texts = X[col].fillna("").astype(str)

            # Length features
            if "length" in self.config.features:
                result[f"{col}_char_length"] = texts.str.len()
                result[f"{col}_word_count"] = texts.str.split().str.len()

            # Character statistics
            if "char_stats" in self.config.features:
                result[f"{col}_uppercase_ratio"] = texts.apply(
                    lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
                )
                result[f"{col}_digit_ratio"] = texts.apply(lambda x: sum(1 for c in x if c.isdigit()) / max(len(x), 1))
                result[f"{col}_space_ratio"] = texts.apply(lambda x: sum(1 for c in x if c.isspace()) / max(len(x), 1))
                result[f"{col}_special_char_count"] = texts.apply(
                    lambda x: sum(1 for c in x if not c.isalnum() and not c.isspace())
                )

            # Word count features
            if "word_count" in self.config.features:
                result[f"{col}_avg_word_length"] = texts.apply(lambda x: np.mean([len(w) for w in x.split()] or [0]))
                result[f"{col}_unique_word_ratio"] = texts.apply(
                    lambda x: len(set(x.lower().split())) / max(len(x.split()), 1)
                )

            # TF-IDF features
            if "tfidf" in self.config.features and col in self._vectorizers:
                tfidf_features = self._transform_tfidf(texts, col)
                result = pd.concat([result, tfidf_features], axis=1)

        self._feature_names = [c for c in result.columns if c not in X.columns]

        if self.config.verbose:
            print(f"TextEngine: Extracted {len(self._feature_names)} features")

        return result

    def _transform_tfidf(self, texts: pd.Series, col: str) -> pd.DataFrame:
        """Transform texts using fitted TF-IDF + SVD."""
        vectorizer = self._vectorizers[col]["vectorizer"]
        svd = self._vectorizers[col]["svd"]

        tfidf_matrix = vectorizer.transform(texts)
        reduced = svd.transform(tfidf_matrix)

        feature_names = [f"{col}_tfidf_{i}" for i in range(reduced.shape[1])]
        return pd.DataFrame(reduced, columns=feature_names, index=texts.index)

    def get_feature_set(self) -> FeatureSet:
        """Get the feature set with metadata."""
        return self._feature_set
