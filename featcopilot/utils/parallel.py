"""Parallel processing utilities."""

from typing import Any, Callable, Optional

import pandas as pd


def parallel_apply(
    func: Callable,
    data: pd.DataFrame,
    n_jobs: int = -1,
    batch_size: Optional[int] = None,
    verbose: bool = False,
) -> list[Any]:
    """
    Apply a function in parallel across DataFrame rows.

    Parameters
    ----------
    func : callable
        Function to apply to each row
    data : DataFrame
        Input data
    n_jobs : int, default=-1
        Number of parallel jobs (-1 for all CPUs)
    batch_size : int, optional
        Batch size for processing
    verbose : bool, default=False
        Show progress

    Returns
    -------
    results : list
        Results from applying function to each row
    """
    try:
        from joblib import Parallel, delayed

        if n_jobs == -1:
            import os

            n_jobs = os.cpu_count() or 1

        if batch_size is None:
            batch_size = max(1, len(data) // (n_jobs * 4))

        results = Parallel(n_jobs=n_jobs, verbose=1 if verbose else 0)(delayed(func)(row) for _, row in data.iterrows())

        return results

    except ImportError:
        # Fallback to sequential processing
        if verbose:
            print("joblib not available, using sequential processing")

        return [func(row) for _, row in data.iterrows()]


def parallel_transform(
    transformers: list[tuple], X: pd.DataFrame, n_jobs: int = -1, verbose: bool = False
) -> pd.DataFrame:
    """
    Apply multiple transformers in parallel.

    Parameters
    ----------
    transformers : list
        List of (name, transformer) tuples
    X : DataFrame
        Input data
    n_jobs : int, default=-1
        Number of parallel jobs
    verbose : bool, default=False
        Show progress

    Returns
    -------
    X_combined : DataFrame
        Combined transformed data
    """
    try:
        from joblib import Parallel, delayed

        def apply_transformer(name, transformer, X):
            return name, transformer.transform(X)

        results = Parallel(n_jobs=n_jobs, verbose=1 if verbose else 0)(
            delayed(apply_transformer)(name, t, X) for name, t in transformers
        )

        # Combine results
        combined = X.copy()
        for _, transformed in results:
            new_cols = [c for c in transformed.columns if c not in combined.columns]
            for col in new_cols:
                combined[col] = transformed[col]

        return combined

    except ImportError:
        # Sequential fallback
        combined = X.copy()
        for _, transformer in transformers:
            transformed = transformer.transform(X)
            new_cols = [c for c in transformed.columns if c not in combined.columns]
            for col in new_cols:
                combined[col] = transformed[col]

        return combined
