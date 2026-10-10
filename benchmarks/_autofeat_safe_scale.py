"""Private AutoFeat 2.1.1 selection compatibility, never an upstream replacement.

The five copied routines are AutoFeatModel.fit_transform (autofeat.py:264-416),
_add_noise_features, _noise_filtering, _select_features_1run and select_features
(featsel.py:16-284). Generation, transforms, constructors and estimators remain
upstream. Computational changes are limited to feature/noise scaling calls;
target scaling still calls the original nb_standard_scale. Logging is adapted
to the repository logger; unsupported task values raise rather than print.

``safe-scale-v1`` changes a denominator only when the entire column is finite,
all values equal its first value, AND upstream std is exactly zero. No epsilon,
imputation, feature removal, category merging, or underflow repair is applied.
All nonconstant arithmetic and random draws remain upstream. Source pins are
raw-file SHA-256; any dependency/source mismatch fails before fitting.

Upstream: https://github.com/cod3licious/autofeat, release 2.1.1.
Original fit/selection author: Franziska Horn <cod3licious@gmail.com>.
Original scaling helper author: Jeethu Rao <jboloor@acm.org>.

MIT License

Copyright (c) 2016

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import hashlib
import warnings
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from autofeat import autofeat as _upstream
from autofeat.autofeat import AutoFeatClassifier, AutoFeatRegressor, _parse_units
from autofeat.feateng import engineer_features, n_cols_generated
from autofeat.nb_utils import nb_nanmean, nb_nanstd, nb_standard_scale
from joblib import Parallel, delayed
from sklearn.utils.validation import check_X_y

from featcopilot.utils.logger import get_logger

logger = get_logger(__name__)

PINNED_VERSIONS = {
    "autofeat": "2.1.1",
    "numpy": "2.4.4",
    "pandas": "2.3.3",
    "scikit-learn": "1.7.2",
    "scipy": "1.17.1",
    "numba": "0.65.1",
    "sympy": "1.14.0",
    "joblib": "1.3.2",
}
SOURCE_SHA256 = {
    "autofeat.py": "b05ebbc4c4adb184817f46b9c3b217f24da0aa21db93e0ce11bd35fe6c092741",
    "featsel.py": "cc96ca82b3a2392c21fd5256b526770d6b681dac1c56c9a3b478c905a8ad2122",
    "nb_utils.py": "20f6f76884284c5e0031dd9df55b57bef2a463f1c99ef77bb5d9f591b0cb0204",
    "feateng.py": "7f536fc86b6a29aa3b4b112b072929eeb2e583f4c7aa9285604d6f9d99c5fb31",
}


class AutoFeatCompatibilityError(ImportError):
    """The private compatibility implementation does not match its pinned runtime."""


def _source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_runtime() -> None:
    """Fail closed on dependency versions or upstream source mismatches."""
    for package, expected in PINNED_VERSIONS.items():
        try:
            actual = version(package)
        except PackageNotFoundError as error:
            raise AutoFeatCompatibilityError(f"safe-scale-v1 dependency missing: {package}") from error
        if actual != expected:
            raise AutoFeatCompatibilityError(f"safe-scale-v1 requires {package}=={expected}; found {actual}")
    for filename, expected in SOURCE_SHA256.items():
        path = Path(_upstream.__file__).with_name(filename)
        try:
            actual = _source_digest(path)
        except OSError as error:
            raise AutoFeatCompatibilityError(f"safe-scale-v1 cannot verify source: {filename}") from error
        if actual != expected:
            raise AutoFeatCompatibilityError(f"safe-scale-v1 source mismatch: {filename}; SHA-256 {actual}")


def provenance() -> dict[str, Any]:
    """Return verified, JSON-serializable compatibility policy and source identity."""
    verify_runtime()
    return {
        "name": "safe-scale-v1",
        "versions": dict(PINNED_VERSIONS),
        "upstream_source_sha256": dict(SOURCE_SHA256),
        "compatibility_source_sha256": _source_digest(Path(__file__)),
        "feature_scaling": "denominator=1 only for wholly finite, exactly constant columns with upstream std==0",
        "underflow_policy": "nonconstant columns with std==0 are not repaired",
        "target_scaling": "unchanged",
        "scope": ["selection_features", "shuffled_noise_features"],
        "upstream_generation": "unchanged",
        "license": "MIT",
    }


def scale_features(array: np.ndarray) -> np.ndarray:
    """Scale features with native arithmetic except verified finite constants.

    Parameters
    ----------
    array : ndarray
        Two-dimensional numeric feature/noise matrix, not a target array.

    Returns
    -------
    ndarray
        Original shape with constants centered to zero. Other nonfinite outcomes
        remain unchanged, including nonconstant standard-deviation underflow.
    """
    mean = nb_nanmean(array, 0)
    std = nb_nanstd(array, 0)
    if len(array):
        constant = np.isfinite(array).all(axis=0) & (array == array[0]).all(axis=0) & (std == 0)
        std[constant] = 1.0
    result: np.ndarray = (array - mean) / std
    return result


def _add_noise_features(X: np.ndarray) -> np.ndarray:
    n_feat = X.shape[1]
    if X.shape[0] > 50 and n_feat > 1:
        rand_noise = scale_features(np.random.permutation(X.flatten()).reshape(X.shape))
        X = np.hstack([X, rand_noise])
    rand_noise = np.random.randn(X.shape[0], max(3, int(0.5 * n_feat)))
    X = np.hstack([X, rand_noise])
    return X


def _noise_filtering(
    X: np.ndarray, target: np.ndarray, good_cols: list | None = None, problem_type: str = "regression"
) -> list:
    n_feat = X.shape[1]
    if good_cols is None or not len(good_cols):
        good_cols = list(range(n_feat))
    assert len(good_cols) == n_feat, "fewer column names provided than features in X."
    model: Any
    if problem_type == "regression":
        model = lm.LassoLarsCV(cv=5, eps=1e-8)
    elif problem_type == "classification":
        model = lm.LogisticRegressionCV(cv=5, penalty="l1", solver="saga", class_weight="balanced")
    else:
        raise ValueError(f"Unsupported AutoFeat problem_type: {problem_type}")
    X = _add_noise_features(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model.fit(X, target)
        except ValueError:
            rand_idx = np.random.permutation(X.shape[0])
            model.fit(X[rand_idx], target[rand_idx])
    if problem_type == "regression":
        coefs = np.abs(model.coef_)
    else:
        coefs = np.max(np.abs(model.coef_), axis=0)
    weights = dict(zip(good_cols, coefs[: len(good_cols)], strict=False))
    noise_w_thr = np.max(coefs[n_feat:])
    good_cols = [c for c in good_cols if weights[c] > noise_w_thr]
    return good_cols


def _select_features_1run(
    df: pd.DataFrame, target: np.ndarray, problem_type: str = "regression", verbose: int = 0
) -> list:
    if df.shape[0] <= 1:
        raise ValueError(f"n_samples = {df.shape[0]}")
    model: Any
    if problem_type == "regression":
        model = lm.LassoLarsCV(cv=5, eps=1e-8)
    elif problem_type == "classification":
        model = lm.LogisticRegressionCV(cv=5, penalty="l1", solver="saga", class_weight="balanced")
    else:
        raise ValueError(f"Unsupported AutoFeat problem_type: {problem_type}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model.fit(df, target)
        except ValueError:
            rand_idx = np.random.permutation(df.shape[0])
            model.fit(df.iloc[rand_idx], target[rand_idx])
    if problem_type == "regression":
        coefs = np.abs(model.coef_)
    else:
        coefs = np.max(np.abs(model.coef_), axis=0)
    thr = sorted(coefs, reverse=True)[min(df.shape[1] - 1, df.shape[0] // 5)]
    initial_cols = list(df.columns[coefs > thr])
    initial_cols = _noise_filtering(df[initial_cols].to_numpy(), target, initial_cols, problem_type)
    good_cols_set = set(initial_cols)
    if verbose > 0:
        logger.info(f"[featsel] {len(initial_cols)} initial features.")
    X_w_noise = _add_noise_features(df[initial_cols].to_numpy())
    other_cols = list(np.random.permutation(list(set(df.columns).difference(initial_cols))))
    if other_cols:
        n_splits = int(np.ceil(len(other_cols) / max(10, 0.5 * df.shape[0] - len(initial_cols))))
        split_size = int(np.ceil(len(other_cols) / n_splits))
        for i in range(n_splits):
            current_cols = other_cols[i * split_size : min(len(other_cols), (i + 1) * split_size)]
            X = np.hstack([df[current_cols].to_numpy(), X_w_noise])
            if problem_type == "regression":
                model = lm.LassoLarsCV(cv=5, eps=1e-8)
            else:
                model = lm.LogisticRegressionCV(cv=5, penalty="l1", solver="saga", class_weight="balanced")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model.fit(X, target)
                except ValueError:
                    rand_idx = np.random.permutation(X.shape[0])
                    model.fit(X[rand_idx], target[rand_idx])
            current_cols.extend(initial_cols)
            if problem_type == "regression":
                coefs = np.abs(model.coef_)
            else:
                coefs = np.max(np.abs(model.coef_), axis=0)
            weights = dict(zip(current_cols, coefs[: len(current_cols)], strict=False))
            noise_w_thr = np.max(coefs[len(current_cols) :])
            good_cols_set.update([c for c in weights if abs(weights[c]) > noise_w_thr])
            if verbose > 0:
                logger.info(
                    f"[featsel] Split {i + 1:2}/{n_splits}: {len(good_cols_set):3} candidate features identified."
                )
    good_cols = list(good_cols_set)
    good_cols = _noise_filtering(df[good_cols].to_numpy(), target, good_cols, problem_type)
    if verbose > 0:
        logger.info(f"[featsel] Selected {len(good_cols):3} features after noise filtering.")
    return good_cols


def _select_features(
    df: pd.DataFrame,
    target: np.ndarray,
    featsel_runs: int = 5,
    keep: list | None = None,
    problem_type: str = "regression",
    n_jobs: int = 1,
    verbose: int = 0,
) -> list:
    if not (len(df) == len(target)):
        raise ValueError("[featsel] df and target dimension mismatch.")
    if keep is None:
        keep = []
    keep = [c for c in keep if c in df.columns and str(c) not in df.columns] + [
        str(c) for c in keep if str(c) in df.columns
    ]
    if verbose > 0:
        if featsel_runs > df.shape[0]:
            logger.warning("[featsel] Less data points than featsel runs!")
        logger.info("[featsel] Scaling data...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_scaled = pd.DataFrame(scale_features(df.to_numpy()), columns=df.columns, dtype=np.float32)
        if problem_type == "regression":
            target_scaled = nb_standard_scale(target.reshape(-1, 1)).ravel()
        else:
            target_scaled = target
    good_cols = list(df.columns)

    def run_select_features(i: int) -> list:
        if verbose > 0:
            logger.info(f"[featsel] Feature selection run {i + 1}/{featsel_runs}")
        np.random.seed(i)
        rand_idx = np.random.permutation(df_scaled.index)[: max(10, int(0.85 * len(df_scaled)))]
        return _select_features_1run(
            df_scaled.iloc[rand_idx], target_scaled[rand_idx], problem_type, verbose=verbose - 1
        )

    if featsel_runs >= 1 and problem_type in ("regression", "classification"):
        if n_jobs == 1 or featsel_runs == 1:
            selected_columns = []
            for i in range(featsel_runs):
                selected_columns.extend(run_select_features(i))
        else:

            def flatten_lists(lists: list) -> list:
                return [item for sublist in lists for item in sublist]

            selected_columns = flatten_lists(
                Parallel(n_jobs=n_jobs, verbose=100 * verbose)(
                    delayed(run_select_features)(i) for i in range(featsel_runs)
                )
            )
        if selected_columns:
            counts = Counter(selected_columns)
            selected_columns = sorted(counts, key=lambda x: counts[x] - 0.000001 * len(str(x)), reverse=True)
            if verbose > 0:
                logger.info(f"[featsel] {len(selected_columns)} features after {featsel_runs} feature selection runs")
            selected_columns = keep + [c for c in selected_columns if c not in keep]
            if not keep:
                good_cols = [selected_columns[0]]
                k = 1
            else:
                good_cols = keep
                k = len(keep)
            if len(selected_columns) > k:
                correlations = df_scaled[selected_columns].corr()
                for i, c in enumerate(selected_columns[k:], k):
                    if np.max(np.abs(correlations[c].ravel()[:i])) < 0.9:
                        good_cols.append(c)
            if verbose > 0:
                logger.info(f"[featsel] {len(good_cols)} features after correlation filtering")
    good_cols = _noise_filtering(df_scaled[good_cols].to_numpy(), target_scaled, good_cols, problem_type)
    if verbose > 0:
        logger.info(f"[featsel] {len(good_cols)} features after noise filtering")
        if not good_cols:
            logger.warning("[featsel] Not a single good feature was found.")
    good_cols = keep + [c for c in good_cols if c not in keep]
    if verbose > 0 and keep:
        logger.info(
            f"[featsel] {len(good_cols)} final features selected (including {len(keep)} original keep features)."
        )
    return good_cols


def _fit_transform(self: Any, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
    verify_runtime()
    cols = [str(c) for c in X.columns] if isinstance(X, pd.DataFrame) else []
    X, target = check_X_y(X, y, y_numeric=self.problem_type == "regression", dtype=None)
    if not cols:
        cols = [f"x{i:03}" for i in range(X.shape[1])]
    self.original_columns_ = cols
    df = pd.DataFrame(X, columns=cols)
    df = self._transform_categorical_cols(df)
    if self.feateng_cols:
        fcols = []
        for c in self.feateng_cols:
            if c not in self.original_columns_:
                raise ValueError(f"[AutoFeat] feateng_col {c} not in df.columns")
            if c in self.categorical_cols_map_:
                fcols.extend(self.categorical_cols_map_[c])
            else:
                fcols.append(c)
        self.feateng_cols_ = fcols
    else:
        self.feateng_cols_ = list(df.columns)
    if self.units:
        self.units = {c: self.units[c] if c in self.units else "" for c in self.feateng_cols_}
        df = self._apply_pi_theorem(df)
    n_cols = n_cols_generated(len(self.feateng_cols_), self.feateng_steps, len(self.transformations))
    n_gb = (len(df) * n_cols) / 250000000
    if self.verbose:
        logger.info(
            f"[AutoFeat] The {self.feateng_steps} step feature engineering process could generate up to {n_cols} features."
        )
        logger.info(f"[AutoFeat] With {len(df)} data points this new feature matrix would use about {n_gb:.2f} gb.")
    if self.max_gb and n_gb > self.max_gb:
        n_rows = int(self.max_gb * 250000000 / n_cols)
        if self.verbose:
            logger.info(f"[AutoFeat] max_gb={self.max_gb:.1f}; subsampling to {n_rows} rows.")
        subsample_idx = np.random.permutation(list(df.index))[:n_rows]
        df_subs = df.iloc[subsample_idx]
        df_subs.reset_index(drop=True, inplace=True)
        target_sub = target[subsample_idx]
    else:
        df_subs = df.copy()
        target_sub = target.copy()
    df_subs, self.feature_formulas_ = engineer_features(
        df_subs,
        self.feateng_cols_,
        _parse_units(self.units, verbose=self.verbose),
        self.feateng_steps,
        self.transformations,
        self.verbose,
    )
    if self.featsel_runs <= 0:
        if self.verbose:
            logger.warning("[AutoFeat] Not performing feature selection.")
        good_cols = df_subs.columns
    else:
        if self.problem_type in ("regression", "classification"):
            good_cols = _select_features(
                df_subs, target_sub, self.featsel_runs, None, self.problem_type, self.n_jobs, self.verbose
            )
            if not good_cols:
                good_cols = list(df.columns)
        else:
            raise ValueError(f"Unsupported AutoFeat problem_type: {self.problem_type}")
    self.new_feat_cols_ = [c for c in good_cols if c not in list(df.columns)]
    self.good_cols_ = good_cols
    self.feature_functions_ = {}
    df = self._generate_features(df, self.new_feat_cols_)
    df.columns = [str(c) for c in df.columns]
    self.feature_formulas_ = {f: self.feature_formulas_[f] for f in self.new_feat_cols_ + self.feateng_cols_}
    self.feature_functions_ = {f: self.feature_functions_[f] for f in self.new_feat_cols_}
    self.all_columns_ = list(df.columns)
    if self.verbose:
        logger.info(
            f"[AutoFeat] Final dataframe with {len(df.columns)} feature columns ({len(df.columns) - len(self.original_columns_)} new)."
        )
    model: Any
    if self.problem_type == "regression":
        model = lm.LassoLarsCV(cv=5)
    elif self.problem_type == "classification":
        model = lm.LogisticRegressionCV(cv=5, class_weight="balanced")
    else:
        raise ValueError(f"Unsupported AutoFeat problem_type: {self.problem_type}")
    X = df[self.good_cols_].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, target)
    self.prediction_model_ = model
    if self.problem_type == "classification":
        self.classes_ = model.classes_
    if self.verbose:
        if self.problem_type == "regression":
            coefs = model.coef_
        else:
            coefs = np.max(np.abs(model.coef_), axis=0)
        weights = dict(zip(self.good_cols_, coefs, strict=False))
        logger.info(f"[AutoFeat] Trained model intercept: {model.intercept_}")
        for c in sorted(weights, key=lambda x: abs(weights[x]), reverse=True):
            if abs(weights[c]) < 1e-5:
                break
            logger.info(f"{weights[c]:.6f} * {c}")
        logger.info(f"[AutoFeat] Final score: {model.score(X, target):.4f}")
    if self.always_return_numpy:
        return df.to_numpy()
    return df


class SafeScaleAutoFeatClassifier(AutoFeatClassifier):
    """AutoFeatClassifier defaults with the private safe-scale-v1 fit chain."""

    def fit_transform(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """Fit using pinned upstream generation and guarded feature/noise scaling."""
        return _fit_transform(self, X, y)


class SafeScaleAutoFeatRegressor(AutoFeatRegressor):
    """AutoFeatRegressor defaults with unchanged target standardization."""

    def fit_transform(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """Fit using pinned upstream generation and guarded feature/noise scaling."""
        return _fit_transform(self, X, y)
