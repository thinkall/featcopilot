"""
AutoML Integration Benchmark for FeatCopilot.

Compares AutoML framework performance:
1. Baseline AutoML (no feature engineering)
2. AutoML + FeatCopilot feature engineering

Workflow:
1. Prepare all datasets (baseline and FeatCopilot-enhanced)
2. Cache prepared data to ensure identical inputs across frameworks
3. Run benchmarks on cached data with 30s time budget

Supported AutoML frameworks:
- FLAML (Microsoft)
- AutoGluon (Amazon)
- Auto-sklearn (Linux only)
- H2O AutoML
"""

# ruff: noqa: E402

import hashlib
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, ".")

from benchmarks.datasets import get_all_datasets, get_text_datasets, get_timeseries_datasets

warnings.filterwarnings("ignore")

# Default time budget for each AutoML run
DEFAULT_TIME_BUDGET = 30

# Cache directory for prepared datasets
CACHE_DIR = Path("benchmarks/automl/.cache")


def check_automl_availability():
    """Check which AutoML frameworks are available."""
    available = {}

    try:
        import flaml

        available["flaml"] = flaml.__version__
    except ImportError:
        available["flaml"] = None

    try:
        import autogluon.tabular

        available["autogluon"] = autogluon.tabular.__version__
    except ImportError:
        available["autogluon"] = None

    try:
        import autosklearn

        available["autosklearn"] = autosklearn.__version__
    except ImportError:
        available["autosklearn"] = None

    try:
        import h2o

        available["h2o"] = h2o.__version__
    except ImportError:
        available["h2o"] = None

    return available


class AutoMLRunner:
    """Base class for AutoML runners."""

    def __init__(self, time_budget: int = DEFAULT_TIME_BUDGET, random_state: int = 42):
        self.time_budget = time_budget
        self.random_state = random_state
        self.model = None
        self.actual_train_time = 0

    def fit(self, X: pd.DataFrame, y: pd.Series, task: str) -> float:
        """Fit model and return actual training time."""
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class FLAMLRunner(AutoMLRunner):
    """FLAML AutoML runner."""

    def fit(self, X: pd.DataFrame, y: pd.Series, task: str) -> float:
        from flaml import AutoML

        self.model = AutoML()

        # Map task to FLAML task type
        if "classification" in task:
            flaml_task = "classification"
        elif "timeseries" in task:
            flaml_task = "ts_forecast" if "regression" in task else "ts_forecast_classification"
        else:
            flaml_task = "regression"

        # Configure fit parameters
        fit_kwargs = {
            "task": flaml_task,
            "time_budget": self.time_budget,
            "seed": self.random_state,
            "verbose": 0,
            "force_cancel": True,  # Force cancel when time budget is reached
        }

        # Time series specific configuration
        if flaml_task.startswith("ts_forecast"):
            fit_kwargs["period"] = min(12, len(y) // 10)

        start_time = time.time()
        self.model.fit(X, y, **fit_kwargs)
        self.actual_train_time = time.time() - start_time

        return self.actual_train_time

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None


class AutoGluonRunner(AutoMLRunner):
    """AutoGluon AutoML runner."""

    def __init__(self, time_budget: int = DEFAULT_TIME_BUDGET, random_state: int = 42):
        super().__init__(time_budget, random_state)
        self._label_col = "__target__"

    def fit(self, X: pd.DataFrame, y: pd.Series, task: str) -> float:
        from autogluon.tabular import TabularPredictor

        train_data = X.copy()
        train_data[self._label_col] = y

        problem_type = "binary" if "classification" in task else "regression"
        if "classification" in task and y.nunique() > 2:
            problem_type = "multiclass"

        self.model = TabularPredictor(
            label=self._label_col,
            problem_type=problem_type,
            verbosity=0,
        )

        start_time = time.time()
        self.model.fit(
            train_data,
            time_limit=self.time_budget,
            presets="medium_quality",
        )
        self.actual_train_time = time.time() - start_time

        return self.actual_train_time

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X).values

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            if isinstance(proba, pd.DataFrame):
                return proba.values
            return proba
        return None


class AutoSklearnRunner(AutoMLRunner):
    """Auto-sklearn AutoML runner."""

    def fit(self, X: pd.DataFrame, y: pd.Series, task: str) -> float:
        if "classification" in task:
            from autosklearn.classification import AutoSklearnClassifier

            self.model = AutoSklearnClassifier(
                time_left_for_this_task=self.time_budget,
                seed=self.random_state,
                n_jobs=-1,
            )
        else:
            from autosklearn.regression import AutoSklearnRegressor

            self.model = AutoSklearnRegressor(
                time_left_for_this_task=self.time_budget,
                seed=self.random_state,
                n_jobs=-1,
            )

        start_time = time.time()
        self.model.fit(X, y)
        self.actual_train_time = time.time() - start_time

        return self.actual_train_time

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None


class H2ORunner(AutoMLRunner):
    """H2O AutoML runner."""

    def __init__(self, time_budget: int = DEFAULT_TIME_BUDGET, random_state: int = 42):
        super().__init__(time_budget, random_state)
        self._label_col = "__target__"

    def fit(self, X: pd.DataFrame, y: pd.Series, task: str) -> float:
        import h2o
        from h2o.automl import H2OAutoML

        h2o.init(verbose=False)

        train_data = X.copy()
        train_data[self._label_col] = y
        h2o_frame = h2o.H2OFrame(train_data)

        if "classification" in task:
            h2o_frame[self._label_col] = h2o_frame[self._label_col].asfactor()

        self.model = H2OAutoML(
            max_runtime_secs=self.time_budget,
            seed=self.random_state,
            verbosity="warn",
        )

        start_time = time.time()
        self.model.train(
            y=self._label_col,
            training_frame=h2o_frame,
        )
        self.actual_train_time = time.time() - start_time

        return self.actual_train_time

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import h2o

        h2o_frame = h2o.H2OFrame(X)
        preds = self.model.leader.predict(h2o_frame)
        return preds.as_data_frame()["predict"].values

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import h2o

        h2o_frame = h2o.H2OFrame(X)
        preds = self.model.leader.predict(h2o_frame)
        df = preds.as_data_frame()
        if "p1" in df.columns:
            return df["p1"].values
        return None


def get_automl_runner(framework: str, time_budget: int = DEFAULT_TIME_BUDGET) -> AutoMLRunner:
    """Get AutoML runner by framework name."""
    runners = {
        "flaml": FLAMLRunner,
        "autogluon": AutoGluonRunner,
        "autosklearn": AutoSklearnRunner,
        "h2o": H2ORunner,
    }
    if framework not in runners:
        raise ValueError(f"Unknown framework: {framework}. Available: {list(runners.keys())}")
    return runners[framework](time_budget=time_budget)


def evaluate_classification(y_true, y_pred, y_prob=None) -> dict[str, float]:
    """Evaluate classification metrics."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    """Evaluate regression metrics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "r2_score": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def get_dataset_hash(X: pd.DataFrame, y: pd.Series) -> str:
    """Generate a hash for dataset caching."""
    data_str = f"{X.shape}_{y.shape}_{X.columns.tolist()}_{X.dtypes.tolist()}"
    return hashlib.md5(data_str.encode()).hexdigest()[:8]


def prepare_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    name: str,
    engines: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 50,
    enable_llm: bool = False,
    use_cache: bool = True,
) -> dict[str, Any]:
    """
    Prepare dataset with baseline and FeatCopilot-enhanced versions.

    Returns a dict with train/test splits for both baseline and enhanced data.
    """
    from featcopilot import AutoFeatureEngineer

    # Create cache directory
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Generate cache key
    cache_key = f"{name}_{get_dataset_hash(X, y)}_{'-'.join(sorted(engines))}"
    cache_file = CACHE_DIR / f"{cache_key}.pkl"

    # Try to load from cache
    if use_cache and cache_file.exists():
        print(f"    Loading cached data for {name}...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # Encode categorical columns
    X_encoded = X.copy()
    label_encoders = {}
    for col in X_encoded.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        label_encoders[col] = le

    # Split data (same split for baseline and enhanced)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size, random_state=random_state)

    # Prepare baseline data
    baseline_data = {
        "X_train": X_train.copy(),
        "X_test": X_test.copy(),
        "y_train": y_train.copy(),
        "y_test": y_test.copy(),
    }

    # Apply FeatCopilot
    print(f"    Applying FeatCopilot with engines: {engines}...")
    llm_config = None
    if enable_llm:
        engines = list(engines)
        if "llm" not in engines:
            engines.append("llm")
        llm_config = {"model": "gpt-5.2", "max_suggestions": 10}

    fe_start = time.time()
    engineer = AutoFeatureEngineer(
        engines=engines,
        max_features=max_features,
        llm_config=llm_config,
        verbose=False,
    )
    X_train_fe = engineer.fit_transform(X_train, y_train)
    X_test_fe = engineer.transform(X_test)
    fe_time = time.time() - fe_start

    # Handle missing columns
    for col in X_train_fe.columns:
        if col not in X_test_fe.columns:
            X_test_fe[col] = 0
    X_test_fe = X_test_fe[X_train_fe.columns]

    # Fill NaN values
    X_train_fe = X_train_fe.fillna(0)
    X_test_fe = X_test_fe.fillna(0)

    enhanced_data = {
        "X_train": X_train_fe,
        "X_test": X_test_fe,
        "y_train": y_train.copy(),
        "y_test": y_test.copy(),
    }

    result = {
        "name": name,
        "task": task,
        "engines": engines,
        "baseline": baseline_data,
        "enhanced": enhanced_data,
        "fe_time": fe_time,
        "n_features_original": X_encoded.shape[1],
        "n_features_engineered": X_train_fe.shape[1],
    }

    # Save to cache
    if use_cache:
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        print(f"    Cached data saved to {cache_file}")

    return result


def run_benchmark_on_prepared_data(
    prepared_data: dict[str, Any],
    framework: str,
    time_budget: int = DEFAULT_TIME_BUDGET,
) -> dict[str, Any]:
    """
    Run benchmark on prepared data for a single framework.

    Returns results for both baseline and enhanced runs.
    """
    task = prepared_data["task"]
    baseline = prepared_data["baseline"]
    enhanced = prepared_data["enhanced"]

    results = {
        "dataset": prepared_data["name"],
        "task": task,
        "framework": framework,
        "n_features_original": prepared_data["n_features_original"],
        "n_features_engineered": prepared_data["n_features_engineered"],
        "fe_time": prepared_data["fe_time"],
    }

    # Run baseline
    runner = get_automl_runner(framework, time_budget)
    train_time = runner.fit(baseline["X_train"], baseline["y_train"], task)

    predict_start = time.time()
    y_pred = runner.predict(baseline["X_test"])
    y_prob = runner.predict_proba(baseline["X_test"]) if "classification" in task else None
    predict_time = time.time() - predict_start

    if "classification" in task:
        baseline_metrics = evaluate_classification(baseline["y_test"], y_pred, y_prob)
    else:
        baseline_metrics = evaluate_regression(baseline["y_test"], y_pred)

    results["baseline_score"] = baseline_metrics.get("accuracy" if "classification" in task else "r2_score", 0)
    results["baseline_train_time"] = train_time
    results["baseline_predict_time"] = predict_time

    # Run with enhanced features
    runner = get_automl_runner(framework, time_budget)
    train_time = runner.fit(enhanced["X_train"], enhanced["y_train"], task)

    predict_start = time.time()
    y_pred = runner.predict(enhanced["X_test"])
    y_prob = runner.predict_proba(enhanced["X_test"]) if "classification" in task else None
    predict_time = time.time() - predict_start

    if "classification" in task:
        enhanced_metrics = evaluate_classification(enhanced["y_test"], y_pred, y_prob)
    else:
        enhanced_metrics = evaluate_regression(enhanced["y_test"], y_pred)

    results["featcopilot_score"] = enhanced_metrics.get("accuracy" if "classification" in task else "r2_score", 0)
    results["featcopilot_train_time"] = train_time
    results["featcopilot_predict_time"] = predict_time

    # Calculate improvement
    if results["baseline_score"] > 0:
        results["improvement_pct"] = (
            (results["featcopilot_score"] - results["baseline_score"]) / results["baseline_score"] * 100
        )
    else:
        results["improvement_pct"] = 0

    return results


def prepare_all_datasets(
    max_features: int = 50,
    enable_llm: bool = False,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Prepare all datasets with baseline and FeatCopilot-enhanced versions."""
    print("=" * 60)
    print("Preparing datasets...")
    print("=" * 60)

    # Get all datasets with their engines
    datasets = []
    for func in get_all_datasets():
        datasets.append((func, ["tabular"]))
    for func in get_timeseries_datasets():
        datasets.append((func, ["tabular", "timeseries"]))
    for func in get_text_datasets():
        datasets.append((func, ["tabular", "text"]))

    prepared = []
    for dataset_func, engines in datasets:
        X, y, task, name = dataset_func()
        print(f"\n{name} ({task})")
        print(f"  Original shape: {X.shape}")

        data = prepare_dataset(
            X,
            y,
            task,
            name,
            engines,
            max_features=max_features,
            enable_llm=enable_llm,
            use_cache=use_cache,
        )
        print(f"  Enhanced shape: {data['n_features_engineered']} features")
        print(f"  FeatCopilot time: {data['fe_time']:.2f}s")
        prepared.append(data)

    return prepared


def run_all_automl_benchmarks(
    prepared_datasets: list[dict[str, Any]],
    frameworks: list[str] = None,
    time_budget: int = DEFAULT_TIME_BUDGET,
) -> pd.DataFrame:
    """
    Run AutoML benchmarks on prepared datasets.

    Parameters
    ----------
    prepared_datasets : list
        List of prepared datasets from prepare_all_datasets()
    frameworks : list of str, optional
        AutoML frameworks to test. Defaults to all available.
    time_budget : int, default=30
        Time budget in seconds for each AutoML run.

    Returns
    -------
    results : pd.DataFrame
        Benchmark results.
    """
    # Check available frameworks
    available = check_automl_availability()
    if frameworks is None:
        frameworks = [f for f, v in available.items() if v is not None]

    if not frameworks:
        raise RuntimeError("No AutoML frameworks available. Install flaml, autogluon, autosklearn, or h2o.")

    print("\n" + "=" * 60)
    print("Running AutoML benchmarks")
    print("=" * 60)
    print(f"Frameworks: {frameworks}")
    print(f"Time budget: {time_budget}s per run (excludes FeatCopilot time)")
    print("-" * 60)

    results = []

    for data in prepared_datasets:
        print(f"\nDataset: {data['name']} ({data['task']})")

        for framework in frameworks:
            if available.get(framework) is None:
                print(f"  {framework}: Not available, skipping")
                continue

            try:
                print(f"  {framework}...", end=" ", flush=True)
                result = run_benchmark_on_prepared_data(data, framework, time_budget)
                print(
                    f"done (train: {result['baseline_train_time']:.1f}s/{result['featcopilot_train_time']:.1f}s, "
                    f"predict: {result['baseline_predict_time']:.2f}s/{result['featcopilot_predict_time']:.2f}s, "
                    f"improvement: {result['improvement_pct']:+.2f}%)"
                )
                results.append(result)

            except Exception as e:
                print(f"Error - {e}")
                continue

    return pd.DataFrame(results)


def generate_report(results: pd.DataFrame, output_path: str = None, time_budget: int = None) -> str:
    """Generate markdown report from benchmark results."""
    if time_budget is None:
        time_budget = DEFAULT_TIME_BUDGET
    report = []
    report.append("# AutoML Integration Benchmark Report\n")
    report.append("## Overview\n")
    report.append("This benchmark evaluates FeatCopilot's impact on AutoML framework performance.\n")
    report.append(f"Time budget: {time_budget}s per AutoML run (excludes FeatCopilot preprocessing time)\n")

    # Summary statistics
    report.append("\n## Summary\n")
    report.append(f"- **Datasets tested**: {results['dataset'].nunique()}\n")
    report.append(f"- **Frameworks tested**: {results['framework'].nunique()}\n")

    avg_improvement = results["improvement_pct"].mean()
    positive_improvements = (results["improvement_pct"] > 0).sum()
    total_runs = len(results)

    report.append(f"- **Average improvement**: {avg_improvement:.2f}%\n")
    report.append(
        f"- **Positive improvements**: {positive_improvements}/{total_runs} ({100*positive_improvements/total_runs:.1f}%)\n"
    )

    # Results by framework
    report.append("\n## Results by Framework\n")
    for framework in results["framework"].unique():
        fw_results = results[results["framework"] == framework]
        report.append(f"\n### {framework.upper()}\n")
        report.append(
            "| Dataset | Task | Baseline | +FeatCopilot | Improvement | Train Time (B/E) | Predict Time (B/E) |\n"
        )
        report.append(
            "|---------|------|----------|--------------|-------------|------------------|--------------------|\n"
        )

        for _, row in fw_results.iterrows():
            report.append(
                f"| {row['dataset']} | {row['task']} | "
                f"{row['baseline_score']:.4f} | {row['featcopilot_score']:.4f} | "
                f"{row['improvement_pct']:+.2f}% | "
                f"{row['baseline_train_time']:.1f}s / {row['featcopilot_train_time']:.1f}s | "
                f"{row['baseline_predict_time']:.2f}s / {row['featcopilot_predict_time']:.2f}s |\n"
            )

        avg = fw_results["improvement_pct"].mean()
        report.append(f"\n**Average improvement with {framework}**: {avg:+.2f}%\n")

    # Detailed results table
    report.append("\n## Detailed Results\n")
    report.append(
        "| Dataset | Framework | Baseline | +FeatCopilot | Improvement | Features | FE Time | Train Time | Predict Time |\n"
    )
    report.append(
        "|---------|-----------|----------|--------------|-------------|----------|---------|------------|---------------|\n"
    )

    for _, row in results.iterrows():
        report.append(
            f"| {row['dataset']} | {row['framework']} | "
            f"{row['baseline_score']:.4f} | {row['featcopilot_score']:.4f} | "
            f"{row['improvement_pct']:+.2f}% | "
            f"{row['n_features_original']}->{row['n_features_engineered']} | "
            f"{row['fe_time']:.1f}s | {row['featcopilot_train_time']:.1f}s | {row['featcopilot_predict_time']:.2f}s |\n"
        )

    report_text = "".join(report)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\nReport saved to {output_path}")

    return report_text


def clear_cache():
    """Clear the dataset cache."""
    import shutil

    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        print(f"Cache cleared: {CACHE_DIR}")
    else:
        print("No cache to clear")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AutoML integration benchmarks")
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=None,
        help="AutoML frameworks to test (default: all available)",
    )
    parser.add_argument(
        "--time-budget",
        type=int,
        default=DEFAULT_TIME_BUDGET,
        help=f"Time budget in seconds for each AutoML run (default: {DEFAULT_TIME_BUDGET})",
    )
    parser.add_argument(
        "--enable-llm",
        action="store_true",
        help="Enable LLM-powered feature engineering",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=50,
        help="Maximum number of features to generate (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/automl/AUTOML_BENCHMARK_REPORT.md",
        help="Output path for report",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable dataset caching (re-run FeatCopilot for all datasets)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the dataset cache and exit",
    )

    args = parser.parse_args()

    if args.clear_cache:
        clear_cache()
        sys.exit(0)

    # Check available frameworks
    print("Checking AutoML framework availability...")
    available = check_automl_availability()
    for framework, version in available.items():
        status = f"v{version}" if version else "Not installed"
        print(f"  {framework}: {status}")
    print()

    # Step 1: Prepare all datasets
    prepared = prepare_all_datasets(
        max_features=args.max_features,
        enable_llm=args.enable_llm,
        use_cache=not args.no_cache,
    )

    # Step 2: Run benchmarks on prepared data
    results = run_all_automl_benchmarks(
        prepared,
        frameworks=args.frameworks,
        time_budget=args.time_budget,
    )

    # Step 3: Generate report
    print("\n" + "=" * 60)
    report = generate_report(results, args.output, args.time_budget)
    print(report)
