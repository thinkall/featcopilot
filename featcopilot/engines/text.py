"""Text feature engineering engine.

Generates features from text data using embeddings and NLP techniques.
Supports local offline processing with transformers and spacy.
"""

from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from pydantic import Field

from featcopilot.core.base import BaseEngine, EngineConfig
from featcopilot.core.feature import FeatureSet
from featcopilot.utils.logger import get_logger

logger = get_logger(__name__)


class TextEngineConfig(EngineConfig):
    """Configuration for text feature engine."""

    name: str = "TextEngine"
    features: list[str] = Field(
        default_factory=lambda: ["length", "word_count", "char_stats"],
        description="Feature types to extract: length, word_count, char_stats, tfidf, sentiment, ner, pos, embeddings",
    )
    max_vocab_size: int = Field(default=5000, description="Max vocabulary size for TF-IDF")
    n_components: int = Field(default=50, description="Components for dimensionality reduction")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )
    embedding_dim: int = Field(default=32, description="Reduced embedding dimensions (PCA)")
    spacy_model: str = Field(default="en_core_web_sm", description="Spacy model for NER/POS")
    sentiment_model: str = Field(
        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
        description="HuggingFace sentiment model",
    )


class TextEngine(BaseEngine):
    """
    Text feature engineering engine with advanced NLP capabilities.

    Extracts features from text columns including:
    - Length and character statistics
    - Word count features
    - TF-IDF features (optional)
    - Sentiment analysis using transformers (local, offline)
    - Named Entity Recognition (NER) using spacy
    - Part-of-speech (POS) tag distributions
    - Sentence embeddings using sentence-transformers

    Parameters
    ----------
    features : list
        Feature types to extract. Options:
        - 'length': character and word counts
        - 'word_count': word-level statistics
        - 'char_stats': character-level statistics
        - 'tfidf': TF-IDF with SVD reduction
        - 'sentiment': transformer-based sentiment scores
        - 'ner': named entity counts by type
        - 'pos': part-of-speech tag distributions
        - 'embeddings': sentence embeddings (reduced via PCA)
    max_vocab_size : int, default=5000
        Maximum vocabulary size for TF-IDF
    embedding_model : str
        Sentence transformer model name
    spacy_model : str
        Spacy model for NER/POS tagging

    Examples
    --------
    >>> # Basic features (fast, no dependencies)
    >>> engine = TextEngine(features=['length', 'word_count', 'char_stats'])
    >>> X_features = engine.fit_transform(text_df)

    >>> # Advanced features with transformers/spacy
    >>> engine = TextEngine(features=['sentiment', 'ner', 'pos', 'embeddings'])
    >>> X_features = engine.fit_transform(text_df)
    """

    def __init__(
        self,
        features: Optional[list[str]] = None,
        max_vocab_size: int = 5000,
        n_components: int = 50,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 32,
        spacy_model: str = "en_core_web_sm",
        sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        max_features: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ):
        config = TextEngineConfig(
            features=features or ["length", "word_count", "char_stats"],
            max_vocab_size=max_vocab_size,
            n_components=n_components,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            spacy_model=spacy_model,
            sentiment_model=sentiment_model,
            max_features=max_features,
            verbose=verbose,
            **kwargs,
        )
        super().__init__(config=config)
        self.config: TextEngineConfig = config
        self._text_columns: list[str] = []
        self._vectorizers: dict[str, Any] = {}
        self._feature_set = FeatureSet()

        # Lazy-loaded models
        self._nlp = None  # spacy
        self._sentiment_pipeline = None  # transformers
        self._embedding_model = None  # sentence-transformers
        self._pca_models: dict[str, Any] = {}  # PCA for embeddings

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
            logger.info(f"TextEngine: Found {len(self._text_columns)} text columns")

        # Fit TF-IDF vectorizers if needed
        if "tfidf" in self.config.features:
            self._fit_tfidf(X)

        # Fit embedding PCA if needed
        if "embeddings" in self.config.features:
            self._fit_embeddings(X)

        # Load spacy model if needed
        if "ner" in self.config.features or "pos" in self.config.features:
            self._load_spacy()

        # Load sentiment model if needed
        if "sentiment" in self.config.features:
            self._load_sentiment()

        self._is_fitted = True
        return self

    def _load_spacy(self) -> None:
        """Load spacy model for NER/POS tagging."""
        if self._nlp is not None:
            return

        try:
            import spacy

            try:
                self._nlp = spacy.load(self.config.spacy_model)
                if self.config.verbose:
                    logger.info(f"TextEngine: Loaded spacy model '{self.config.spacy_model}'")
            except OSError:
                # Try to download the model
                if self.config.verbose:
                    logger.info(f"TextEngine: Downloading spacy model '{self.config.spacy_model}'...")
                spacy.cli.download(self.config.spacy_model)
                self._nlp = spacy.load(self.config.spacy_model)

        except ImportError:
            logger.warning("TextEngine: spacy not installed. Install with: pip install spacy")
            self._nlp = None

    def _load_sentiment(self) -> None:
        """Load sentiment analysis pipeline."""
        if self._sentiment_pipeline is not None:
            return

        try:
            from transformers import pipeline

            self._sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.config.sentiment_model,
                truncation=True,
                max_length=512,
            )
            if self.config.verbose:
                logger.info(f"TextEngine: Loaded sentiment model '{self.config.sentiment_model}'")

        except ImportError:
            logger.warning("TextEngine: transformers not installed. Install with: pip install transformers")
            self._sentiment_pipeline = None
        except Exception as e:
            logger.warning(f"TextEngine: Could not load sentiment model: {e}")
            self._sentiment_pipeline = None

    def _load_embedding_model(self) -> None:
        """Load sentence transformer model."""
        if self._embedding_model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self.config.embedding_model)
            if self.config.verbose:
                logger.info(f"TextEngine: Loaded embedding model '{self.config.embedding_model}'")

        except ImportError:
            logger.warning(
                "TextEngine: sentence-transformers not installed. Install with: pip install sentence-transformers"
            )
            self._embedding_model = None

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
                logger.warning("TextEngine: sklearn not available for TF-IDF, skipping")

    def _fit_embeddings(self, X: pd.DataFrame) -> None:
        """Fit PCA for embedding dimensionality reduction."""
        self._load_embedding_model()
        if self._embedding_model is None:
            return

        try:
            from sklearn.decomposition import PCA

            for col in self._text_columns:
                texts = X[col].fillna("").astype(str).tolist()
                # Sample for fitting PCA (limit to 1000 for speed)
                sample_texts = texts[: min(1000, len(texts))]
                embeddings = self._embedding_model.encode(sample_texts, show_progress_bar=False)

                # Fit PCA
                n_components = min(self.config.embedding_dim, embeddings.shape[1], len(sample_texts))
                if n_components > 0:
                    pca = PCA(n_components=n_components)
                    pca.fit(embeddings)
                    self._pca_models[col] = pca

                if self.config.verbose:
                    logger.info(f"TextEngine: Fitted embedding PCA for '{col}' ({n_components} components)")

        except Exception as e:
            logger.warning(f"TextEngine: Could not fit embeddings: {e}")

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
            Extracted features (numerical only, text columns dropped)
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

            # Sentiment features (transformers)
            if "sentiment" in self.config.features:
                sentiment_features = self._extract_sentiment(texts, col)
                for feat_name, feat_values in sentiment_features.items():
                    result[feat_name] = feat_values

            # NER features (spacy)
            if "ner" in self.config.features:
                ner_features = self._extract_ner(texts, col)
                for feat_name, feat_values in ner_features.items():
                    result[feat_name] = feat_values

            # POS features (spacy)
            if "pos" in self.config.features:
                pos_features = self._extract_pos(texts, col)
                for feat_name, feat_values in pos_features.items():
                    result[feat_name] = feat_values

            # Embedding features (sentence-transformers)
            if "embeddings" in self.config.features:
                emb_features = self._extract_embeddings(texts, col)
                if emb_features is not None:
                    result = pd.concat([result, emb_features], axis=1)

        # Drop original text columns
        cols_to_drop = [col for col in self._text_columns if col in result.columns]
        result = result.drop(columns=cols_to_drop)

        self._feature_names = [c for c in result.columns if c not in X.columns or c in cols_to_drop]

        if self.config.verbose:
            logger.info(f"TextEngine: Extracted {len(self._feature_names)} features")

        return result

    def _transform_tfidf(self, texts: pd.Series, col: str) -> pd.DataFrame:
        """Transform texts using fitted TF-IDF + SVD."""
        vectorizer = self._vectorizers[col]["vectorizer"]
        svd = self._vectorizers[col]["svd"]

        tfidf_matrix = vectorizer.transform(texts)
        reduced = svd.transform(tfidf_matrix)

        feature_names = [f"{col}_tfidf_{i}" for i in range(reduced.shape[1])]
        return pd.DataFrame(reduced, columns=feature_names, index=texts.index)

    def _extract_sentiment(self, texts: pd.Series, col: str) -> dict[str, list]:
        """Extract sentiment scores using transformers."""
        if self._sentiment_pipeline is None:
            self._load_sentiment()
            if self._sentiment_pipeline is None:
                return {}

        features = {
            f"{col}_sentiment_positive": [],
            f"{col}_sentiment_negative": [],
            f"{col}_sentiment_neutral": [],
            f"{col}_sentiment_score": [],
        }

        # Process in batches for efficiency
        batch_size = 32
        text_list = texts.tolist()

        for i in range(0, len(text_list), batch_size):
            batch = text_list[i : i + batch_size]
            # Truncate very long texts
            batch = [t[:512] if len(t) > 512 else t for t in batch]

            try:
                results = self._sentiment_pipeline(batch)
                for res in results:
                    label = res["label"].lower()
                    score = res["score"]

                    # Map to standard sentiment scores
                    if "positive" in label or label == "pos":
                        features[f"{col}_sentiment_positive"].append(score)
                        features[f"{col}_sentiment_negative"].append(0)
                        features[f"{col}_sentiment_neutral"].append(0)
                        features[f"{col}_sentiment_score"].append(score)
                    elif "negative" in label or label == "neg":
                        features[f"{col}_sentiment_positive"].append(0)
                        features[f"{col}_sentiment_negative"].append(score)
                        features[f"{col}_sentiment_neutral"].append(0)
                        features[f"{col}_sentiment_score"].append(-score)
                    else:  # neutral
                        features[f"{col}_sentiment_positive"].append(0)
                        features[f"{col}_sentiment_negative"].append(0)
                        features[f"{col}_sentiment_neutral"].append(score)
                        features[f"{col}_sentiment_score"].append(0)

            except Exception as e:
                # Fill with zeros on error
                for _ in batch:
                    features[f"{col}_sentiment_positive"].append(0)
                    features[f"{col}_sentiment_negative"].append(0)
                    features[f"{col}_sentiment_neutral"].append(0)
                    features[f"{col}_sentiment_score"].append(0)
                if self.config.verbose:
                    logger.warning(f"TextEngine: Sentiment error: {e}")

        return features

    def _extract_ner(self, texts: pd.Series, col: str) -> dict[str, list]:
        """Extract NER counts using spacy."""
        if self._nlp is None:
            return {}

        # Entity types to count
        entity_types = ["PERSON", "ORG", "GPE", "DATE", "MONEY", "PRODUCT", "EVENT", "LOC"]
        features = {f"{col}_ner_{ent.lower()}": [] for ent in entity_types}
        features[f"{col}_ner_total"] = []

        for text in texts:
            try:
                doc = self._nlp(text[:10000])  # Limit text length
                ent_counts = {ent: 0 for ent in entity_types}

                for ent in doc.ents:
                    if ent.label_ in ent_counts:
                        ent_counts[ent.label_] += 1

                for ent_type in entity_types:
                    features[f"{col}_ner_{ent_type.lower()}"].append(ent_counts[ent_type])
                features[f"{col}_ner_total"].append(len(doc.ents))

            except Exception:
                for ent_type in entity_types:
                    features[f"{col}_ner_{ent_type.lower()}"].append(0)
                features[f"{col}_ner_total"].append(0)

        return features

    def _extract_pos(self, texts: pd.Series, col: str) -> dict[str, list]:
        """Extract POS tag distributions using spacy."""
        if self._nlp is None:
            return {}

        # POS tags to track (ratios)
        pos_tags = ["NOUN", "VERB", "ADJ", "ADV", "PROPN", "PRON", "DET", "ADP", "PUNCT"]
        features = {f"{col}_pos_{tag.lower()}_ratio": [] for tag in pos_tags}
        features[f"{col}_pos_noun_verb_ratio"] = []
        features[f"{col}_pos_content_ratio"] = []  # nouns + verbs + adj

        for text in texts:
            try:
                doc = self._nlp(text[:10000])
                total_tokens = len(doc)

                if total_tokens == 0:
                    for tag in pos_tags:
                        features[f"{col}_pos_{tag.lower()}_ratio"].append(0)
                    features[f"{col}_pos_noun_verb_ratio"].append(0)
                    features[f"{col}_pos_content_ratio"].append(0)
                    continue

                pos_counts = {tag: 0 for tag in pos_tags}
                for token in doc:
                    if token.pos_ in pos_counts:
                        pos_counts[token.pos_] += 1

                for tag in pos_tags:
                    features[f"{col}_pos_{tag.lower()}_ratio"].append(pos_counts[tag] / total_tokens)

                # Noun to verb ratio
                verb_count = pos_counts["VERB"]
                noun_count = pos_counts["NOUN"]
                features[f"{col}_pos_noun_verb_ratio"].append(noun_count / max(verb_count, 1))

                # Content word ratio (nouns + verbs + adjectives)
                content_count = noun_count + verb_count + pos_counts["ADJ"]
                features[f"{col}_pos_content_ratio"].append(content_count / total_tokens)

            except Exception:
                for tag in pos_tags:
                    features[f"{col}_pos_{tag.lower()}_ratio"].append(0)
                features[f"{col}_pos_noun_verb_ratio"].append(0)
                features[f"{col}_pos_content_ratio"].append(0)

        return features

    def _extract_embeddings(self, texts: pd.Series, col: str) -> Optional[pd.DataFrame]:
        """Extract sentence embeddings using sentence-transformers."""
        if self._embedding_model is None:
            self._load_embedding_model()
            if self._embedding_model is None:
                return None

        try:
            text_list = texts.tolist()
            embeddings = self._embedding_model.encode(text_list, show_progress_bar=False)

            # Apply PCA if fitted
            if col in self._pca_models:
                embeddings = self._pca_models[col].transform(embeddings)

            feature_names = [f"{col}_emb_{i}" for i in range(embeddings.shape[1])]
            return pd.DataFrame(embeddings, columns=feature_names, index=texts.index)

        except Exception as e:
            if self.config.verbose:
                logger.warning(f"TextEngine: Embedding error: {e}")
            return None

    def get_feature_set(self) -> FeatureSet:
        """Get the feature set with metadata."""
        return self._feature_set
