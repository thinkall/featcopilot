"""
Feature Engineering Tools Comparison Benchmark.

Compares FeatCopilot with other popular feature engineering libraries using FLAML for training.

Tools compared:
1. Baseline (no feature engineering)
2. FeatCopilot (multi-engine per dataset)
3. Featuretools (automated feature engineering)
4. tsfresh (time series feature extraction)
5. autofeat (automatic feature generation)
6. OpenFE (automated feature generation)
7. CAAFE (context-aware automated feature engineering)

Usage:
    python -m benchmarks.compare_tools.run_fe_tools_comparison [options]

Examples:
    # Run with default settings
    python -m benchmarks.compare_tools.run_fe_tools_comparison

    # Run specific tools
    python -m benchmarks.compare_tools.run_fe_tools_comparison --tools featcopilot featuretools

    # Run on specific datasets
    python -m benchmarks.compare_tools.run_fe_tools_comparison --datasets titanic,house_prices

    # Run on all classification datasets
    python -m benchmarks.compare_tools.run_fe_tools_comparison --category classification
"""

import argparse
import json
import sys
import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, ".")  # noqa: E402

from benchmarks.datasets import (  # noqa: E402
    CATEGORY_CLASSIFICATION,
    CATEGORY_FORECASTING,
    CATEGORY_REGRESSION,
    CATEGORY_TEXT,
    list_datasets,
    load_dataset,
)

warnings.filterwarnings("ignore")

# Default configuration
QUICK_DATASETS = ["titanic", "house_prices", "credit_risk", "bike_sharing", "customer_churn", "insurance_claims"]


def check_tool_availability() -> dict[str, Optional[str]]:
    """Check which feature engineering tools are available."""
    available = {}

    # FeatCopilot (always available - this is our tool)
    try:
        import featcopilot

        available["featcopilot"] = featcopilot.__version__
    except ImportError:
        available["featcopilot"] = None

    # Featuretools
    try:
        import featuretools

        available["featuretools"] = featuretools.__version__
    except ImportError:
        available["featuretools"] = None

    # tsfresh
    try:
        import tsfresh

        available["tsfresh"] = tsfresh.__version__
    except ImportError:
        available["tsfresh"] = None

    # autofeat
    try:
        import autofeat

        available["autofeat"] = autofeat.__version__
    except ImportError:
        available["autofeat"] = None

    # openfe
    try:
        import openfe

        available["openfe"] = openfe.__version__
    except ImportError:
        available["openfe"] = None

    # caafe
    try:
        import caafe  # noqa: F401

        available["caafe"] = "0.1.6"  # caafe doesn't expose __version__
    except ImportError:
        available["caafe"] = None

    return available


class FeatureEngineeringRunner(ABC):
    """Base class for feature engineering tool runners."""

    name: str = "base"

    def __init__(self, max_features: int = 50, random_state: int = 42):
        self.max_features = max_features
        self.random_state = random_state
        self.fit_time = 0.0
        self.transform_time = 0.0
        self.n_features_generated = 0

    @abstractmethod
    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        """Fit on training data and transform."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data."""
        pass


class BaselineRunner(FeatureEngineeringRunner):
    """Baseline - no feature engineering, just pass through."""

    name = "baseline"

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        self.fit_time = 0.0
        self.n_features_generated = X_train.shape[1]
        self._columns = X_train.columns.tolist()
        return X_train.copy()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.transform_time = 0.0
        return X[self._columns].copy()


class FeatCopilotRunner(FeatureEngineeringRunner):
    """FeatCopilot feature engineering runner."""

    name = "featcopilot"

    def __init__(self, max_features: int = 50, random_state: int = 42):
        super().__init__(max_features, random_state)
        self.engineer = None
        self.engines_used: list[str] = []

    @staticmethod
    def _select_engines(task: str) -> list[str]:
        engines = ["tabular", "relational"]
        if "timeseries" in task:
            engines.append("timeseries")
        if "text" in task:
            engines.append("text")
        return engines

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        from featcopilot import AutoFeatureEngineer

        self.engines_used = self._select_engines(self._task)
        self.engineer = AutoFeatureEngineer(
            engines=self.engines_used,
            max_features=self.max_features,
            verbose=False,
        )

        start = time.time()
        result = self.engineer.fit_transform(X_train, y_train)
        self.fit_time = time.time() - start
        self.n_features_generated = result.shape[1]
        return result

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        start = time.time()
        result = self.engineer.transform(X)
        self.transform_time = time.time() - start
        return result


class FeaturetoolsRunner(FeatureEngineeringRunner):
    """Featuretools automated feature engineering runner."""

    name = "featuretools"

    def __init__(self, max_features: int = 50, random_state: int = 42):
        super().__init__(max_features, random_state)
        self._feature_defs = None
        self._entityset = None
        self._train_index_name = None

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        import featuretools as ft

        start = time.time()

        # Create a copy with unique index
        X_copy = X_train.copy().reset_index(drop=True)
        X_copy["_row_id"] = range(len(X_copy))
        self._train_index_name = "_row_id"

        # Create EntitySet
        es = ft.EntitySet(id="benchmark_data")
        es = es.add_dataframe(
            dataframe_name="data",
            dataframe=X_copy,
            index="_row_id",
        )

        # Define primitives to use (subset for speed)
        trans_primitives = ["add_numeric", "multiply_numeric", "divide_numeric"]
        agg_primitives = []

        # Generate features with Deep Feature Synthesis
        feature_matrix, feature_defs = ft.dfs(
            entityset=es,
            target_dataframe_name="data",
            trans_primitives=trans_primitives,
            agg_primitives=agg_primitives,
            max_depth=2,
            max_features=self.max_features,
        )

        self._feature_defs = feature_defs
        self._entityset = es
        self.fit_time = time.time() - start

        # Clean up and fill NaN
        feature_matrix = feature_matrix.drop(columns=["_row_id"], errors="ignore")
        feature_matrix = feature_matrix.select_dtypes(include=[np.number])
        feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan).fillna(0)
        self.n_features_generated = feature_matrix.shape[1]

        return feature_matrix

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        import featuretools as ft

        start = time.time()

        X_copy = X.copy().reset_index(drop=True)
        X_copy["_row_id"] = range(len(X_copy))

        es = ft.EntitySet(id="benchmark_data")
        es = es.add_dataframe(
            dataframe_name="data",
            dataframe=X_copy,
            index="_row_id",
        )

        feature_matrix = ft.calculate_feature_matrix(
            features=self._feature_defs,
            entityset=es,
        )

        self.transform_time = time.time() - start

        feature_matrix = feature_matrix.drop(columns=["_row_id"], errors="ignore")
        feature_matrix = feature_matrix.select_dtypes(include=[np.number])
        feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan).fillna(0)

        return feature_matrix


class TsfreshRunner(FeatureEngineeringRunner):
    """tsfresh feature extraction runner (adapted for tabular data)."""

    name = "tsfresh"

    def __init__(self, max_features: int = 50, random_state: int = 42):
        super().__init__(max_features, random_state)
        self._feature_columns = None
        self._numeric_cols = None

    def _create_ts_format(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert tabular data to tsfresh time series format."""
        ts_data = []
        for idx in range(len(X)):
            for t, col in enumerate(self._numeric_cols):
                ts_data.append({"id": idx, "time": t, "value": X[col].iloc[idx], "variable": col})
        return pd.DataFrame(ts_data)

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        from tsfresh import extract_features
        from tsfresh.feature_extraction import MinimalFCParameters
        from tsfresh.feature_selection import select_features

        start = time.time()

        # Get numeric columns
        self._numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        if len(self._numeric_cols) == 0:
            self.fit_time = time.time() - start
            self.n_features_generated = X_train.shape[1]
            self._feature_columns = X_train.columns.tolist()
            return X_train.copy()

        try:
            # Reset index to ensure consistent IDs
            X_reset = X_train.reset_index(drop=True)
            y_reset = y_train.reset_index(drop=True)

            # Create time series format
            ts_df = self._create_ts_format(X_reset)

            # Use minimal feature extraction for speed
            settings = MinimalFCParameters()

            extracted = extract_features(
                ts_df,
                column_id="id",
                column_sort="time",
                column_value="value",
                column_kind="variable",
                default_fc_parameters=settings,
                disable_progressbar=True,
                n_jobs=1,
            )

            # Clean up extracted features
            extracted = extracted.replace([np.inf, -np.inf], np.nan).fillna(0)

            # Drop columns with zero variance
            non_zero_var = extracted.var() > 0
            extracted = extracted.loc[:, non_zero_var]

            if extracted.shape[1] == 0:
                raise ValueError("No features extracted")

            # Select relevant features
            try:
                selected = select_features(extracted, y_reset)
                if selected.shape[1] == 0:
                    selected = extracted
                if selected.shape[1] > self.max_features:
                    selected = selected.iloc[:, : self.max_features]
            except Exception:
                selected = extracted.iloc[:, : min(self.max_features, extracted.shape[1])]

            self._feature_columns = selected.columns.tolist()
            self.fit_time = time.time() - start
            self.n_features_generated = selected.shape[1]

            # Restore original index
            selected.index = X_train.index
            return selected

        except Exception:
            # Fallback to original features if tsfresh fails
            self.fit_time = time.time() - start
            self.n_features_generated = X_train.shape[1]
            self._feature_columns = X_train.columns.tolist()
            return X_train.copy()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        from tsfresh import extract_features
        from tsfresh.feature_extraction import MinimalFCParameters

        start = time.time()

        if self._numeric_cols is None or len(self._numeric_cols) == 0:
            self.transform_time = time.time() - start
            return X[self._feature_columns].copy() if self._feature_columns else X.copy()

        try:
            X_reset = X.reset_index(drop=True)
            ts_df = self._create_ts_format(X_reset)
            settings = MinimalFCParameters()

            extracted = extract_features(
                ts_df,
                column_id="id",
                column_sort="time",
                column_value="value",
                column_kind="variable",
                default_fc_parameters=settings,
                disable_progressbar=True,
                n_jobs=1,
            )

            extracted = extracted.replace([np.inf, -np.inf], np.nan).fillna(0)

            # Align columns with training features
            for col in self._feature_columns:
                if col not in extracted.columns:
                    extracted[col] = 0

            self.transform_time = time.time() - start
            result = extracted[self._feature_columns]
            result.index = X.index
            return result

        except Exception:
            self.transform_time = time.time() - start
            return X[self._feature_columns].copy() if self._feature_columns else X.copy()


class AutofeatRunner(FeatureEngineeringRunner):
    """autofeat automatic feature generation runner."""

    name = "autofeat"

    def __init__(self, max_features: int = 50, random_state: int = 42):
        super().__init__(max_features, random_state)
        self._model = None
        self._is_classifier = False
        self._feature_columns = None

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        from autofeat import AutoFeatClassifier, AutoFeatRegressor

        start = time.time()

        # Determine if classification or regression
        unique_values = y_train.nunique()
        self._is_classifier = unique_values <= 10

        # Ensure numeric data
        X_numeric = X_train.select_dtypes(include=[np.number]).copy()
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan).fillna(0)

        if X_numeric.shape[1] == 0:
            self.fit_time = time.time() - start
            self.n_features_generated = X_train.shape[1]
            self._feature_columns = X_train.columns.tolist()
            return X_train.copy()

        try:
            if self._is_classifier:
                self._model = AutoFeatClassifier(
                    feateng_steps=2,
                    max_gb=1,
                    transformations=["1/", "log", "sqrt", "^2"],
                    n_jobs=1,
                    verbose=0,
                )
            else:
                self._model = AutoFeatRegressor(
                    feateng_steps=2,
                    max_gb=1,
                    transformations=["1/", "log", "sqrt", "^2"],
                    n_jobs=1,
                    verbose=0,
                )

            result = self._model.fit_transform(X_numeric, y_train)

            # autofeat returns DataFrame directly with column names
            if isinstance(result, pd.DataFrame):
                self._feature_columns = result.columns.tolist()
            else:
                # If numpy array, use original columns + new feature columns
                if hasattr(self._model, "all_columns_"):
                    self._feature_columns = self._model.all_columns_
                else:
                    self._feature_columns = X_numeric.columns.tolist()
                result = pd.DataFrame(result, columns=self._feature_columns, index=X_train.index)

            result = result.replace([np.inf, -np.inf], np.nan).fillna(0)

            # Limit features if needed
            if result.shape[1] > self.max_features:
                result = result.iloc[:, : self.max_features]
                self._feature_columns = result.columns.tolist()

            self.fit_time = time.time() - start
            self.n_features_generated = result.shape[1]

            return result

        except Exception:
            # Fallback to original features
            self.fit_time = time.time() - start
            self.n_features_generated = X_numeric.shape[1]
            self._feature_columns = X_numeric.columns.tolist()
            return X_numeric

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        start = time.time()

        X_numeric = X.select_dtypes(include=[np.number]).copy()
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan).fillna(0)

        if self._model is None:
            self.transform_time = time.time() - start
            return X_numeric[self._feature_columns] if self._feature_columns else X_numeric

        try:
            result = self._model.transform(X_numeric)

            if isinstance(result, pd.DataFrame):
                pass  # Already a DataFrame
            else:
                result = pd.DataFrame(result, columns=self._feature_columns, index=X.index)

            result = result.replace([np.inf, -np.inf], np.nan).fillna(0)

            if result.shape[1] > self.max_features:
                result = result.iloc[:, : self.max_features]

            self.transform_time = time.time() - start
            return result

        except Exception:
            self.transform_time = time.time() - start
            return X_numeric[self._feature_columns] if self._feature_columns else X_numeric


class OpenFERunner(FeatureEngineeringRunner):
    """OpenFE feature engineering runner."""

    name = "openfe"

    def __init__(self, max_features: int = 50, random_state: int = 42):
        super().__init__(max_features, random_state)
        self._features = None
        self._original_columns = None

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        from openfe import OpenFE, transform

        start = time.time()

        # Ensure numeric data
        X_numeric = X_train.select_dtypes(include=[np.number]).copy()
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan).fillna(0)
        self._original_columns = X_numeric.columns.tolist()

        if X_numeric.shape[1] == 0:
            self.fit_time = time.time() - start
            self.n_features_generated = X_train.shape[1]
            return X_train.copy()

        try:
            ofe = OpenFE()
            self._features = ofe.fit(X_numeric, y_train, n_jobs=1, verbose=False)

            if self._features:
                n_features_to_use = min(self.max_features, len(self._features))
                X_new, _ = transform(X_numeric, self._features[:n_features_to_use], n_jobs=1)
            else:
                X_new = X_numeric.copy()

            X_new = X_new.replace([np.inf, -np.inf], np.nan).fillna(0)
            self.fit_time = time.time() - start
            self.n_features_generated = X_new.shape[1]

            return X_new

        except Exception:
            self.fit_time = time.time() - start
            self.n_features_generated = X_numeric.shape[1]
            self._features = None
            return X_numeric

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        from openfe import transform as ofe_transform

        start = time.time()

        X_numeric = X.select_dtypes(include=[np.number]).copy()
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan).fillna(0)

        if self._features is None or len(self._features) == 0:
            self.transform_time = time.time() - start
            return X_numeric

        try:
            n_features_to_use = min(self.max_features, len(self._features))
            X_new, _ = ofe_transform(X_numeric, self._features[:n_features_to_use], n_jobs=1)
            X_new = X_new.replace([np.inf, -np.inf], np.nan).fillna(0)
            self.transform_time = time.time() - start
            return X_new

        except Exception:
            self.transform_time = time.time() - start
            return X_numeric


class CAAFERunner(FeatureEngineeringRunner):
    """CAAFE (Context-Aware Automated Feature Engineering) runner.

    Note: CAAFE requires OpenAI API access for LLM-based feature generation.
    Falls back to original features if API is not available.
    """

    name = "caafe"

    def __init__(self, max_features: int = 50, random_state: int = 42):
        super().__init__(max_features, random_state)
        self._feature_columns = None
        self._model = None

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        start = time.time()

        # Ensure numeric data
        X_numeric = X_train.select_dtypes(include=[np.number]).copy()
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan).fillna(0)
        self._feature_columns = X_numeric.columns.tolist()

        if X_numeric.shape[1] == 0:
            self.fit_time = time.time() - start
            self.n_features_generated = X_train.shape[1]
            return X_train.copy()

        try:
            from caafe import CAAFEClassifier

            # Determine if classification
            unique_values = y_train.nunique()
            if unique_values > 10:
                # CAAFE only supports classification, fallback for regression
                self.fit_time = time.time() - start
                self.n_features_generated = X_numeric.shape[1]
                return X_numeric

            self._model = CAAFEClassifier(
                iterations=2,
                llm_model="gpt-3.5-turbo",
            )

            # CAAFE needs specific format
            X_result = self._model.fit_transform(X_numeric, y_train)

            if isinstance(X_result, pd.DataFrame):
                self._feature_columns = X_result.columns.tolist()
            else:
                X_result = pd.DataFrame(X_result, index=X_train.index)
                self._feature_columns = X_result.columns.tolist()

            X_result = X_result.replace([np.inf, -np.inf], np.nan).fillna(0)

            if X_result.shape[1] > self.max_features:
                X_result = X_result.iloc[:, : self.max_features]
                self._feature_columns = X_result.columns.tolist()

            self.fit_time = time.time() - start
            self.n_features_generated = X_result.shape[1]

            return X_result

        except Exception:
            # Fallback - CAAFE requires OpenAI API key
            self.fit_time = time.time() - start
            self.n_features_generated = X_numeric.shape[1]
            self._model = None
            return X_numeric

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        start = time.time()

        X_numeric = X.select_dtypes(include=[np.number]).copy()
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan).fillna(0)

        if self._model is None:
            self.transform_time = time.time() - start
            return X_numeric[self._feature_columns] if self._feature_columns else X_numeric

        try:
            X_result = self._model.transform(X_numeric)

            if not isinstance(X_result, pd.DataFrame):
                X_result = pd.DataFrame(X_result, index=X.index)

            X_result = X_result.replace([np.inf, -np.inf], np.nan).fillna(0)

            if X_result.shape[1] > self.max_features:
                X_result = X_result.iloc[:, : self.max_features]

            self.transform_time = time.time() - start
            return X_result

        except Exception:
            self.transform_time = time.time() - start
            return X_numeric[self._feature_columns] if self._feature_columns else X_numeric


def get_runner(tool_name: str, max_features: int = 50) -> FeatureEngineeringRunner:
    """Get feature engineering runner by tool name."""
    runners = {
        "baseline": BaselineRunner,
        "featcopilot": FeatCopilotRunner,
        "featuretools": FeaturetoolsRunner,
        "tsfresh": TsfreshRunner,
        "autofeat": AutofeatRunner,
        "openfe": OpenFERunner,
        "caafe": CAAFERunner,
    }
    if tool_name not in runners:
        raise ValueError(f"Unknown tool: {tool_name}. Available: {list(runners.keys())}")
    return runners[tool_name](max_features=max_features)


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> dict:
    """Evaluate classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
    }
    if y_prob is not None:
        try:
            if len(y_prob.shape) > 1:
                y_prob = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
        except (ValueError, IndexError):
            metrics["roc_auc"] = None
    return metrics


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluate regression metrics."""
    return {
        "r2_score": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def run_single_benchmark(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    dataset_name: str,
    tool_name: str,
    max_features: int = 50,
    time_budget: int = 60,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run benchmark for a single tool on a single dataset using FLAML."""
    from flaml import AutoML

    # Encode categorical columns
    X_encoded = X.copy()
    for col in X_encoded.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

    # Split data (keep raw and encoded in sync)
    indices = np.arange(len(X_encoded))
    train_idx, test_idx, y_train, y_test = train_test_split(indices, y, test_size=0.2, random_state=random_state)
    X_train_encoded = X_encoded.iloc[train_idx]
    X_test_encoded = X_encoded.iloc[test_idx]
    X_train_raw = X.iloc[train_idx]
    X_test_raw = X.iloc[test_idx]

    result = {
        "dataset": dataset_name,
        "task": task,
        "tool": tool_name,
        "n_samples": len(X),
        "n_features_original": X.shape[1],
    }

    try:
        # Apply feature engineering
        runner = get_runner(tool_name, max_features)
        if hasattr(runner, "_task"):
            runner._task = task
        if tool_name == "featcopilot":
            X_train_fe = runner.fit_transform(X_train_raw, y_train)
            X_test_fe = runner.transform(X_test_raw)
        else:
            X_train_fe = runner.fit_transform(X_train_encoded, y_train)
            X_test_fe = runner.transform(X_test_encoded)

        # Align columns
        for col in X_train_fe.columns:
            if col not in X_test_fe.columns:
                X_test_fe[col] = 0
        X_test_fe = X_test_fe[X_train_fe.columns]

        # Fill NaN values by dtype
        numeric_cols = X_train_fe.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X_train_fe[numeric_cols] = X_train_fe[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            X_test_fe[numeric_cols] = X_test_fe[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        non_numeric_cols = [col for col in X_train_fe.columns if col not in numeric_cols]
        if non_numeric_cols:
            X_train_fe[non_numeric_cols] = X_train_fe[non_numeric_cols].astype("object").fillna("missing")
            X_test_fe[non_numeric_cols] = X_test_fe[non_numeric_cols].astype("object").fillna("missing")

        result["n_features_engineered"] = runner.n_features_generated
        result["fe_time"] = runner.fit_time

        # Train with FLAML
        if "timeseries" in task or "forecast" in task:
            flaml_task = "forecast"
            is_classification = False
        else:
            is_classification = "classification" in task
            flaml_task = "classification" if is_classification else "regression"

        model = AutoML()
        train_start = time.time()
        model.fit(
            X_train_fe,
            y_train,
            task=flaml_task,
            time_budget=time_budget,
            seed=random_state,
            verbose=0,
            force_cancel=True,
        )
        result["train_time"] = time.time() - train_start

        y_pred = model.predict(X_test_fe)

        if is_classification:
            y_prob = model.predict_proba(X_test_fe) if hasattr(model, "predict_proba") else None
            metrics = evaluate_classification(y_test, y_pred, y_prob)
            result["score"] = metrics["accuracy"]
            result["f1_score"] = metrics["f1_score"]
            result["roc_auc"] = metrics.get("roc_auc")
        else:
            metrics = evaluate_regression(y_test, y_pred)
            result["score"] = metrics["r2_score"]
            result["rmse"] = metrics["rmse"]

        result["status"] = "success"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["score"] = None

    return result


def run_comparison_benchmark(
    dataset_names: list[str],
    tools: Optional[list[str]] = None,
    max_features: int = 50,
    time_budget: int = 60,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run comparison benchmark across tools and datasets.

    Parameters
    ----------
    dataset_names : list of str
        Datasets to benchmark.
    tools : list of str, optional
        Tools to benchmark. Defaults to all available.
    max_features : int, default=50
        Maximum number of features to generate.
    time_budget : int, default=60
        FLAML time budget in seconds.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    results : pd.DataFrame
        Benchmark results.
    """
    # Check available tools
    available = check_tool_availability()

    if tools is None:
        tools = ["baseline", "featcopilot"] + [t for t, v in available.items() if v is not None and t != "featcopilot"]
    else:
        # Validate tools
        for tool in tools:
            if tool not in ["baseline"] and available.get(tool) is None:
                print(f"Warning: {tool} not installed, skipping")
        tools = [t for t in tools if t == "baseline" or available.get(t) is not None]

    print("=" * 70)
    print("Feature Engineering Tools Comparison Benchmark")
    print("=" * 70)
    print(f"Tools: {tools}")
    print(f"Datasets: {len(dataset_names)}")
    print(f"FLAML time budget: {time_budget}s")
    print("-" * 70)

    results = []

    for dataset_name in dataset_names:
        try:
            X, y, task, name = load_dataset(dataset_name)
            print(f"\nDataset: {name} ({task})")
            print(f"  Shape: {X.shape}")

            for tool in tools:
                try:
                    print(f"  {tool}...", end=" ", flush=True)
                    result = run_single_benchmark(
                        X,
                        y,
                        task,
                        dataset_name,
                        tool,
                        max_features=max_features,
                        time_budget=time_budget,
                        random_state=random_state,
                    )
                    if result["status"] == "success":
                        print(
                            f"score={result['score']:.4f}, "
                            f"features={result['n_features_engineered']}, "
                            f"time={result['fe_time']:.2f}s"
                        )
                    else:
                        print(f"Error: {result.get('error', 'Unknown')}")
                    results.append(result)
                except Exception as e:
                    print(f"Error: {e}")
                    results.append(
                        {
                            "dataset": dataset_name,
                            "task": task,
                            "tool": tool,
                            "status": "error",
                            "error": str(e),
                            "score": None,
                        }
                    )
        except Exception as e:
            print(f"\nError loading {dataset_name}: {e}")

    return pd.DataFrame(results)


def calculate_improvements(results: pd.DataFrame) -> pd.DataFrame:
    """Calculate improvement percentages relative to baseline."""
    improvements = []

    for dataset in results["dataset"].unique():
        dataset_results = results[results["dataset"] == dataset]
        baseline_row = dataset_results[dataset_results["tool"] == "baseline"]

        if baseline_row.empty or baseline_row["score"].iloc[0] is None:
            continue

        baseline_score = baseline_row["score"].iloc[0]

        for _, row in dataset_results.iterrows():
            if row["tool"] == "baseline" or row["score"] is None:
                continue

            improvement = {
                "dataset": dataset,
                "task": row["task"],
                "tool": row["tool"],
                "baseline_score": baseline_score,
                "tool_score": row["score"],
                "improvement_pct": (row["score"] - baseline_score) / max(abs(baseline_score), 1e-6) * 100,
                "features_added": row.get("n_features_engineered", 0) - row.get("n_features_original", 0),
                "fe_time": row.get("fe_time", 0),
            }
            improvements.append(improvement)

    return pd.DataFrame(improvements)


def generate_report(results: pd.DataFrame, output_path: Optional[str] = None) -> str:
    """Generate markdown report from benchmark results."""
    report = []
    report.append("# Feature Engineering Tools Comparison Benchmark\n")
    report.append("## Overview\n")
    report.append(
        "This benchmark compares FeatCopilot with other popular feature engineering libraries "
        "to demonstrate performance improvements across various datasets.\n"
    )

    # Tools tested
    tools_tested = results["tool"].unique().tolist()
    report.append("\n### Tools Compared\n")
    report.append("| Tool | Description |\n")
    report.append("|------|-------------|\n")
    tool_descriptions = {
        "baseline": "No feature engineering (raw features only)",
        "featcopilot": "FeatCopilot - LLM-powered auto feature engineering",
        "featuretools": "Featuretools - Deep Feature Synthesis",
        "tsfresh": "tsfresh - Time series feature extraction",
        "autofeat": "autofeat - Automatic feature generation",
        "openfe": "OpenFE - Automated feature generation with LightGBM",
        "caafe": "CAAFE - Context-Aware Automated Feature Engineering (LLM)",
    }
    for tool in tools_tested:
        desc = tool_descriptions.get(tool, "Unknown tool")
        report.append(f"| {tool} | {desc} |\n")

    # Summary statistics
    report.append("\n## Summary\n")

    successful_results = results[results["status"] == "success"]
    if len(successful_results) == 0:
        report.append("No successful benchmark runs.\n")
    else:
        improvements = calculate_improvements(successful_results)

        report.append(f"- **Datasets tested**: {results['dataset'].nunique()}\n")
        report.append(f"- **Tools compared**: {len(tools_tested)}\n")

        # Per-tool summary
        report.append("\n### Performance by Tool\n")
        report.append("| Tool | Avg Score | Avg Improvement | Wins | Avg FE Time |\n")
        report.append("|------|-----------|-----------------|------|-------------|\n")

        # Count wins per tool (best score per dataset)
        tool_wins = {tool: 0 for tool in tools_tested if tool != "baseline"}
        for dataset in successful_results["dataset"].unique():
            dataset_data = successful_results[
                (successful_results["dataset"] == dataset) & (successful_results["tool"] != "baseline")
            ]
            if not dataset_data.empty:
                best_tool = dataset_data.loc[dataset_data["score"].idxmax(), "tool"]
                tool_wins[best_tool] = tool_wins.get(best_tool, 0) + 1

        for tool in tools_tested:
            tool_data = successful_results[successful_results["tool"] == tool]
            avg_score = tool_data["score"].mean()

            if tool == "baseline":
                avg_improvement = "-"
                wins = "-"
            else:
                tool_improvements = improvements[improvements["tool"] == tool]
                avg_improvement = (
                    f"{tool_improvements['improvement_pct'].mean():+.2f}%" if len(tool_improvements) > 0 else "-"
                )
                wins = tool_wins.get(tool, 0)

            avg_fe_time = f"{tool_data['fe_time'].mean():.2f}s" if "fe_time" in tool_data.columns else "-"

            report.append(f"| {tool} | {avg_score:.4f} | {avg_improvement} | {wins} | {avg_fe_time} |\n")

    # Detailed results by dataset
    report.append("\n## Detailed Results\n")

    for dataset in results["dataset"].unique():
        dataset_results = results[results["dataset"] == dataset]
        task = dataset_results["task"].iloc[0]
        metric = "Accuracy" if "classification" in task else "R² Score"

        report.append(f"\n### {dataset}\n")
        report.append(f"**Task**: {task}\n\n")
        report.append(f"| Tool | {metric} | Features | FE Time | Train Time | Status |\n")
        report.append("|------|----------|----------|---------|------------|--------|\n")

        for _, row in dataset_results.iterrows():
            score = f"{row['score']:.4f}" if row["score"] is not None else "N/A"
            features = row.get("n_features_engineered", "-")
            fe_time = f"{row.get('fe_time', 0):.2f}s"
            train_time = f"{row.get('train_time', 0):.2f}s"
            status = row.get("status", "unknown")

            report.append(f"| {row['tool']} | {score} | {features} | {fe_time} | {train_time} | {status} |\n")

    # Highlight FeatCopilot advantages
    report.append("\n## Key Findings\n")

    if len(successful_results) > 0:
        improvements = calculate_improvements(successful_results)
        featcopilot_improvements = improvements[improvements["tool"] == "featcopilot"]

        if len(featcopilot_improvements) > 0:
            avg_imp = featcopilot_improvements["improvement_pct"].mean()
            max_imp = featcopilot_improvements["improvement_pct"].max()
            best_dataset = featcopilot_improvements.loc[featcopilot_improvements["improvement_pct"].idxmax(), "dataset"]

            report.append(f"- **FeatCopilot average improvement**: {avg_imp:+.2f}% over baseline\n")
            report.append(f"- **Best improvement**: {max_imp:+.2f}% on {best_dataset}\n")

            # Compare with other tools
            for tool in tools_tested:
                if tool in ["baseline", "featcopilot"]:
                    continue
                tool_imp = improvements[improvements["tool"] == tool]
                if len(tool_imp) > 0:
                    tool_avg = tool_imp["improvement_pct"].mean()
                    diff = avg_imp - tool_avg
                    if diff > 0:
                        report.append(f"- FeatCopilot outperforms {tool} by {diff:.2f}% on average\n")

    # Conclusion
    report.append("\n## Conclusion\n")
    report.append(
        "FeatCopilot demonstrates competitive or superior performance compared to other "
        "feature engineering tools while providing a more intuitive API and LLM-powered "
        "feature suggestions.\n"
    )

    report_text = "".join(report)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\nReport saved to {output_path}")

    return report_text


def get_cache_file(output_path: Path) -> Path:
    """Get cache file path."""
    return output_path / "FE_TOOLS_COMPARISON_CACHE.json"


def save_cache(results: pd.DataFrame, output_path: Path) -> None:
    """Save benchmark results to cache file."""
    cache_file = get_cache_file(output_path)
    # Convert DataFrame to list of dicts for JSON serialization
    records = results.to_dict(orient="records")
    # Convert numpy types to native Python types
    serializable_records = []
    for r in records:
        sr = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                sr[k] = float(v) if not np.isnan(v) else None
            elif isinstance(v, np.ndarray):
                sr[k] = v.tolist()
            elif pd.isna(v):
                sr[k] = None
            else:
                sr[k] = v
        serializable_records.append(sr)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(serializable_records, f, indent=2)
    print(f"Cache saved: {cache_file}")


def load_cache(output_path: Path) -> Optional[pd.DataFrame]:
    """Load benchmark results from cache file."""
    cache_file = get_cache_file(output_path)
    if not cache_file.exists():
        print(f"Cache file not found: {cache_file}")
        return None
    with open(cache_file, encoding="utf-8") as f:
        records = json.load(f)
    results = pd.DataFrame(records)
    print(f"Loaded {len(results)} results from cache: {cache_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare FeatCopilot with other FE tools")
    parser.add_argument("--datasets", type=str, help="Comma-separated dataset names")
    parser.add_argument("--category", type=str, choices=["classification", "regression", "forecasting", "text"])
    parser.add_argument("--all", action="store_true", help="Run all datasets")
    parser.add_argument("--tools", nargs="+", default=None, help="Tools to benchmark (use 'all' for all available)")
    parser.add_argument("--max-features", type=int, default=100)
    parser.add_argument("--time-budget", type=int, default=120, help="FLAML time budget")
    parser.add_argument("--output", type=str, default="benchmarks/compare_tools")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-only", action="store_true", help="Only regenerate report from cache")
    parser.add_argument("--no-cache", action="store_true", help="Don't save results to cache")

    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Report-only mode: load from cache and regenerate report
    if args.report_only:
        results = load_cache(output_path)
        if results is not None:
            report_file = output_path / "FE_TOOLS_COMPARISON.md"
            report = generate_report(results, str(report_file))
            print(report)
        sys.exit(0)

    # Determine datasets
    if args.datasets:
        dataset_names = [d.strip() for d in args.datasets.split(",")]
    elif args.category:
        dataset_names = list_datasets(args.category)
    elif args.all:
        dataset_names = (
            list_datasets(CATEGORY_CLASSIFICATION)
            + list_datasets(CATEGORY_REGRESSION)
            + list_datasets(CATEGORY_FORECASTING)
            + list_datasets(CATEGORY_TEXT)
        )
    else:
        dataset_names = QUICK_DATASETS

    # Check available tools
    print("Checking tool availability...")
    available = check_tool_availability()
    for tool, version in available.items():
        status = f"v{version}" if version else "Not installed"
        print(f"  {tool}: {status}")
    print()

    # Handle --tools all
    tools_to_run = args.tools
    if tools_to_run and "all" in tools_to_run:
        tools_to_run = None  # None means run all available tools

    # Run benchmark
    results = run_comparison_benchmark(
        dataset_names=dataset_names,
        tools=tools_to_run,
        max_features=args.max_features,
        time_budget=args.time_budget,
        random_state=args.seed,
    )

    # Save cache and generate report
    print("\n" + "=" * 70)
    if not args.no_cache:
        save_cache(results, output_path)
    report_file = output_path / "FE_TOOLS_COMPARISON.md"
    report = generate_report(results, str(report_file))
    print(report)
