"""
AutoML Integration Benchmark for FeatCopilot.

Compares AutoML framework performance:
1. Baseline AutoML (no feature engineering)
2. AutoML + FeatCopilot feature engineering

Supported AutoML frameworks:
- FLAML (Microsoft)
- AutoGluon (Amazon)
- Auto-sklearn
- H2O AutoML
"""

# ruff: noqa: E402

import sys
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, ".")

from benchmarks.datasets import get_all_datasets, get_text_datasets, get_timeseries_datasets

warnings.filterwarnings("ignore")


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

    def __init__(self, time_budget: int = 60, random_state: int = 42):
        self.time_budget = time_budget
        self.random_state = random_state
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series, task: str) -> None:
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class FLAMLRunner(AutoMLRunner):
    """FLAML AutoML runner."""

    def fit(self, X: pd.DataFrame, y: pd.Series, task: str) -> None:
        from flaml import AutoML

        self.model = AutoML()

        # Map task to FLAML task type
        if "classification" in task:
            flaml_task = "classification"
        elif "timeseries" in task:
            # FLAML supports ts_forecast for time series
            flaml_task = "ts_forecast" if "regression" in task else "ts_forecast_classification"
        else:
            flaml_task = "regression"

        # Configure fit parameters
        fit_kwargs = {
            "task": flaml_task,
            "time_budget": self.time_budget,
            "seed": self.random_state,
            "verbose": 0,
        }

        # Time series specific configuration
        if flaml_task.startswith("ts_forecast"):
            # For time series, FLAML expects period parameter
            fit_kwargs["period"] = min(12, len(y) // 10)  # Use reasonable period

        self.model.fit(X, y, **fit_kwargs)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None


class AutoGluonRunner(AutoMLRunner):
    """AutoGluon AutoML runner."""

    def __init__(self, time_budget: int = 60, random_state: int = 42):
        super().__init__(time_budget, random_state)
        self._label_col = "__target__"

    def fit(self, X: pd.DataFrame, y: pd.Series, task: str) -> None:
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
        self.model.fit(
            train_data,
            time_limit=self.time_budget,
            presets="medium_quality",
        )

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

    def fit(self, X: pd.DataFrame, y: pd.Series, task: str) -> None:
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
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None


class H2ORunner(AutoMLRunner):
    """H2O AutoML runner."""

    def __init__(self, time_budget: int = 60, random_state: int = 42):
        super().__init__(time_budget, random_state)
        self._label_col = "__target__"

    def fit(self, X: pd.DataFrame, y: pd.Series, task: str) -> None:
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
        self.model.train(
            y=self._label_col,
            training_frame=h2o_frame,
        )

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


def get_automl_runner(framework: str, time_budget: int = 60) -> AutoMLRunner:
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


def run_automl_benchmark(
    framework: str,
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    time_budget: int = 60,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run a single AutoML benchmark."""
    # Encode categorical columns
    X_encoded = X.copy()
    label_encoders = {}
    for col in X_encoded.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        label_encoders[col] = le

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size, random_state=random_state)

    # Run AutoML
    runner = get_automl_runner(framework, time_budget)

    start_time = time.time()
    runner.fit(X_train, y_train, task)
    train_time = time.time() - start_time

    # Predict
    y_pred = runner.predict(X_test)
    y_prob = None
    if "classification" in task:
        y_prob = runner.predict_proba(X_test)

    # Evaluate
    if "classification" in task:
        metrics = evaluate_classification(y_test, y_pred, y_prob)
    else:
        metrics = evaluate_regression(y_test, y_pred)

    return {
        "framework": framework,
        "train_time": train_time,
        "metrics": metrics,
    }


def run_automl_with_featcopilot(
    framework: str,
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    time_budget: int = 60,
    test_size: float = 0.2,
    random_state: int = 42,
    enable_llm: bool = False,
    max_features: int = 50,
    engines: list[str] = None,
) -> dict[str, Any]:
    """Run AutoML with FeatCopilot feature engineering."""
    from featcopilot import AutoFeatureEngineer

    # Configure engines
    if engines is None:
        engines = ["tabular"]
    else:
        engines = list(engines)  # Make a copy

    llm_config = None
    if enable_llm:
        if "llm" not in engines:
            engines.append("llm")
        llm_config = {"model": "gpt-5.2", "max_suggestions": 10}

    # Encode categorical columns first
    X_encoded = X.copy()
    label_encoders = {}
    for col in X_encoded.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        label_encoders[col] = le

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size, random_state=random_state)

    # Apply FeatCopilot
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

    # Handle any missing columns
    for col in X_train_fe.columns:
        if col not in X_test_fe.columns:
            X_test_fe[col] = 0
    X_test_fe = X_test_fe[X_train_fe.columns]

    # Fill NaN values
    X_train_fe = X_train_fe.fillna(0)
    X_test_fe = X_test_fe.fillna(0)

    # Run AutoML
    runner = get_automl_runner(framework, time_budget)

    automl_start = time.time()
    runner.fit(X_train_fe, y_train, task)
    automl_time = time.time() - automl_start

    # Predict
    y_pred = runner.predict(X_test_fe)
    y_prob = None
    if "classification" in task:
        y_prob = runner.predict_proba(X_test_fe)

    # Evaluate
    if "classification" in task:
        metrics = evaluate_classification(y_test, y_pred, y_prob)
    else:
        metrics = evaluate_regression(y_test, y_pred)

    return {
        "framework": framework,
        "fe_time": fe_time,
        "automl_time": automl_time,
        "total_time": fe_time + automl_time,
        "n_features_original": X_encoded.shape[1],
        "n_features_engineered": X_train_fe.shape[1],
        "metrics": metrics,
    }


def run_all_automl_benchmarks(
    frameworks: list[str] = None,
    time_budget: int = 60,
    enable_llm: bool = False,
    max_features: int = 50,
) -> pd.DataFrame:
    """
    Run AutoML benchmarks on all datasets.

    Parameters
    ----------
    frameworks : list of str, optional
        AutoML frameworks to test. Defaults to all available.
    time_budget : int, default=60
        Time budget in seconds for each AutoML run.
    enable_llm : bool, default=False
        Whether to enable LLM-powered features.
    max_features : int, default=50
        Maximum number of features to generate.

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

    print(f"Running benchmarks with frameworks: {frameworks}")
    print(f"Time budget: {time_budget}s per run")
    print(f"LLM enabled: {enable_llm}")
    print("-" * 60)

    # Get all datasets
    datasets = []
    for func in get_all_datasets():
        datasets.append((func, ["tabular"]))
    for func in get_timeseries_datasets():
        datasets.append((func, ["tabular", "timeseries"]))
    for func in get_text_datasets():
        datasets.append((func, ["tabular", "text"]))

    results = []

    for dataset_func, engines in datasets:
        X, y, task, name = dataset_func()
        print(f"\nDataset: {name} ({task})")
        print(f"  Shape: {X.shape}, Engines: {engines}")

        for framework in frameworks:
            if available.get(framework) is None:
                print(f"  {framework}: Not available, skipping")
                continue

            try:
                # Baseline
                print(f"  {framework} baseline...", end=" ", flush=True)
                baseline = run_automl_benchmark(framework, X, y, task, time_budget=time_budget)
                print(f"done ({baseline['train_time']:.1f}s)")

                # With FeatCopilot
                print(f"  {framework} + FeatCopilot...", end=" ", flush=True)
                with_fe = run_automl_with_featcopilot(
                    framework,
                    X,
                    y,
                    task,
                    time_budget=time_budget,
                    enable_llm=enable_llm,
                    max_features=max_features,
                    engines=engines,
                )
                print(f"done ({with_fe['total_time']:.1f}s)")

                # Calculate improvement
                primary_metric = "accuracy" if "classification" in task else "r2_score"
                baseline_score = baseline["metrics"].get(primary_metric, 0)
                fe_score = with_fe["metrics"].get(primary_metric, 0)

                if baseline_score > 0:
                    improvement = (fe_score - baseline_score) / baseline_score * 100
                else:
                    improvement = 0

                results.append(
                    {
                        "dataset": name,
                        "task": task,
                        "framework": framework,
                        "baseline_score": baseline_score,
                        "featcopilot_score": fe_score,
                        "improvement_pct": improvement,
                        "baseline_time": baseline["train_time"],
                        "featcopilot_time": with_fe["total_time"],
                        "n_features_original": with_fe["n_features_original"],
                        "n_features_engineered": with_fe["n_features_engineered"],
                        "primary_metric": primary_metric,
                    }
                )

            except Exception as e:
                print(f"  {framework}: Error - {e}")
                continue

    return pd.DataFrame(results)


def generate_report(results: pd.DataFrame, output_path: str = None) -> str:
    """Generate markdown report from benchmark results."""
    report = []
    report.append("# AutoML Integration Benchmark Report\n")
    report.append("## Overview\n")
    report.append("This benchmark evaluates FeatCopilot's impact on AutoML framework performance.\n")

    # Summary statistics
    report.append("## Summary\n")
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
        report.append("| Dataset | Task | Baseline | +FeatCopilot | Improvement |\n")
        report.append("|---------|------|----------|--------------|-------------|\n")

        for _, row in fw_results.iterrows():
            report.append(
                f"| {row['dataset']} | {row['task']} | "
                f"{row['baseline_score']:.4f} | {row['featcopilot_score']:.4f} | "
                f"{row['improvement_pct']:+.2f}% |\n"
            )

        avg = fw_results["improvement_pct"].mean()
        report.append(f"\n**Average improvement with {framework}**: {avg:+.2f}%\n")

    # Detailed results table
    report.append("\n## Detailed Results\n")
    report.append("| Dataset | Framework | Baseline | +FeatCopilot | Improvement | Features | Time (s) |\n")
    report.append("|---------|-----------|----------|--------------|-------------|----------|----------|\n")

    for _, row in results.iterrows():
        report.append(
            f"| {row['dataset']} | {row['framework']} | "
            f"{row['baseline_score']:.4f} | {row['featcopilot_score']:.4f} | "
            f"{row['improvement_pct']:+.2f}% | "
            f"{row['n_features_original']}->{row['n_features_engineered']} | "
            f"{row['featcopilot_time']:.1f} |\n"
        )

    report_text = "".join(report)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"Report saved to {output_path}")

    return report_text


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
        default=60,
        help="Time budget in seconds for each AutoML run (default: 60)",
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

    args = parser.parse_args()

    # Check available frameworks
    print("Checking AutoML framework availability...")
    available = check_automl_availability()
    for framework, version in available.items():
        status = f"v{version}" if version else "Not installed"
        print(f"  {framework}: {status}")
    print()

    # Run benchmarks
    results = run_all_automl_benchmarks(
        frameworks=args.frameworks,
        time_budget=args.time_budget,
        enable_llm=args.enable_llm,
        max_features=args.max_features,
    )

    # Generate report
    print("\n" + "=" * 60)
    report = generate_report(results, args.output)
    print(report)
