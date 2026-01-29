"""Feature caching utilities."""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Optional

import pandas as pd


class FeatureCache:
    """
    Cache for computed features.

    Stores computed features to avoid recomputation.
    Supports both in-memory and disk-based caching.

    Parameters
    ----------
    cache_dir : str, optional
        Directory for disk cache
    max_memory_items : int, default=100
        Maximum items in memory cache

    Examples
    --------
    >>> cache = FeatureCache(cache_dir='.feature_cache')
    >>> cache.set('my_feature', feature_data, metadata={'source': 'tabular'})
    >>> data = cache.get('my_feature')
    """

    def __init__(self, cache_dir: Optional[str] = None, max_memory_items: int = 100):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_memory_items = max_memory_items
        self._memory_cache: dict[str, Any] = {}
        self._metadata: dict[str, dict] = {}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, key: str, data_hash: Optional[str] = None) -> str:
        """Generate cache key."""
        if data_hash:
            return f"{key}_{data_hash}"
        return key

    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of DataFrame for cache invalidation."""
        # Hash based on shape and sample of data
        shape_str = f"{df.shape}"
        sample = df.head(10).to_string()
        combined = f"{shape_str}_{sample}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    def get(self, key: str, data_hash: Optional[str] = None) -> Optional[Any]:
        """
        Get cached item.

        Parameters
        ----------
        key : str
            Cache key
        data_hash : str, optional
            Data hash for validation

        Returns
        -------
        value : Any or None
            Cached value or None if not found
        """
        cache_key = self._get_cache_key(key, data_hash)

        # Check memory cache first
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        # Check disk cache
        if self.cache_dir:
            cache_path = self.cache_dir / f"{cache_key}.pkl"
            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        value = pickle.load(f)
                    # Store in memory cache
                    self._memory_cache[cache_key] = value
                    return value
                except Exception:
                    pass

        return None

    def set(
        self,
        key: str,
        value: Any,
        data_hash: Optional[str] = None,
        metadata: Optional[dict] = None,
        persist: bool = True,
    ) -> None:
        """
        Set cached item.

        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache
        data_hash : str, optional
            Data hash for validation
        metadata : dict, optional
            Additional metadata
        persist : bool, default=True
            Whether to persist to disk
        """
        cache_key = self._get_cache_key(key, data_hash)

        # Add to memory cache
        self._memory_cache[cache_key] = value
        self._metadata[cache_key] = metadata or {}

        # Evict if over limit
        if len(self._memory_cache) > self.max_memory_items:
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]
            self._metadata.pop(oldest_key, None)

        # Persist to disk
        if persist and self.cache_dir:
            cache_path = self.cache_dir / f"{cache_key}.pkl"
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(value, f)

                # Save metadata
                meta_path = self.cache_dir / f"{cache_key}.meta.json"
                with open(meta_path, "w") as f:
                    json.dump(metadata or {}, f)
            except Exception:
                pass

    def has(self, key: str, data_hash: Optional[str] = None) -> bool:
        """Check if key exists in cache."""
        return self.get(key, data_hash) is not None

    def delete(self, key: str, data_hash: Optional[str] = None) -> bool:
        """
        Delete cached item.

        Parameters
        ----------
        key : str
            Cache key
        data_hash : str, optional
            Data hash

        Returns
        -------
        deleted : bool
            Whether item was deleted
        """
        cache_key = self._get_cache_key(key, data_hash)
        deleted = False

        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]
            self._metadata.pop(cache_key, None)
            deleted = True

        if self.cache_dir:
            cache_path = self.cache_dir / f"{cache_key}.pkl"
            meta_path = self.cache_dir / f"{cache_key}.meta.json"

            if cache_path.exists():
                cache_path.unlink()
                deleted = True

            if meta_path.exists():
                meta_path.unlink()

        return deleted

    def clear(self) -> None:
        """Clear all cached items."""
        self._memory_cache.clear()
        self._metadata.clear()

        if self.cache_dir:
            for f in self.cache_dir.glob("*.pkl"):
                f.unlink()
            for f in self.cache_dir.glob("*.meta.json"):
                f.unlink()

    def get_metadata(self, key: str, data_hash: Optional[str] = None) -> Optional[dict]:
        """Get metadata for cached item."""
        cache_key = self._get_cache_key(key, data_hash)

        if cache_key in self._metadata:
            return self._metadata[cache_key]

        if self.cache_dir:
            meta_path = self.cache_dir / f"{cache_key}.meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        return json.load(f)
                except Exception:
                    pass

        return None

    def list_keys(self) -> list:
        """List all cached keys."""
        keys = set(self._memory_cache.keys())

        if self.cache_dir:
            for f in self.cache_dir.glob("*.pkl"):
                keys.add(f.stem)

        return list(keys)
