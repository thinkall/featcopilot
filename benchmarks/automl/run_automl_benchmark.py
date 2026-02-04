"""
AutoML Benchmark for FeatCopilot.

Compares AutoML framework performance with and without FeatCopilot feature engineering.

Comparison modes:
1. Baseline (no feature engineering)
2. FeatCopilot (tabular engine only)
3. FeatCopilot + LLM (if --with-llm enabled)

Supported AutoML frameworks:
- FLAML (Microsoft) - default
- AutoGluon (Amazon)

Usage:
    python -m benchmarks.automl.run_automl_benchmark [options]

Examples:
    # Quick benchmark with default settings
    python -m benchmarks.automl.run_automl_benchmark

    # Run on specific datasets
    python -m benchmarks.automl.run_automl_benchmark --datasets titanic,house_prices

    # Run on all classification datasets
    python -m benchmarks.automl.run_automl_benchmark --category classification

    # Run with LLM engine enabled
    python -m benchmarks.automl.run_automl_benchmark --with-llm

    # Use AutoGluon instead of FLAML
    python -m benchmarks.automl.run_automl_benchmark --framework autogluon
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
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
DEFAULT_TIME_BUDGET = 120
DEFAULT_MAX_FEATURES = 100
QUICK_DATASETS = ["titanic", "house_prices", "credit_risk", "bike_sharing", "customer_churn", "insurance_claims"]


# =============================================================================
# AutoML Runners
# =============================================================================


class AutoMLRunner:
    """Base class for AutoML runners."""

    def __init__(self, time_budget: int, random_state: int = 42):
        self.time_budget = time_budget
        self.random_state = random_state
        self.model = None

    def fit(self, X: pd.DataFrame, y, task: str) -> float:
        """Fit model and return training time."""
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict on new data."""
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Predict probabilities (classification only)."""
        return None


class FLAMLRunner(AutoMLRunner):
    """FLAML AutoML runner."""

    def __init__(self, time_budget: int, random_state: int = 42):
        super().__init__(time_budget, random_state)
        self._task = None

    def fit(self, X: pd.DataFrame, y, task: str) -> float:
        from flaml import AutoML

        self.model = AutoML()
        self._task = task
        flaml_task = "classification" if "classification" in task else "regression"

        start = time.time()
        self.model.fit(
            X,
            y,
            task=flaml_task,
            time_budget=self.time_budget,
            seed=self.random_state,
            verbose=0,
            force_cancel=True,
        )
        return time.time() - start

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if self._task and "classification" in self._task and hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None


class AutoGluonRunner(AutoMLRunner):
    """AutoGluon AutoML runner."""

    def __init__(self, time_budget: int, random_state: int = 42):
        super().__init__(time_budget, random_state)
        self._label = "__target__"
        self._task = None

    def fit(self, X: pd.DataFrame, y, task: str) -> float:
        from autogluon.tabular import TabularPredictor

        train_data = X.copy()
        train_data[self._label] = y
        self._task = task

        problem_type = "binary" if "classification" in task else "regression"
        if "classification" in task and pd.Series(y).nunique() > 2:
            problem_type = "multiclass"

        self.model = TabularPredictor(label=self._label, problem_type=problem_type, verbosity=0)

        start = time.time()
        self.model.fit(train_data, time_limit=self.time_budget, presets="medium_quality")
        return time.time() - start

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X).values

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if self._task and "classification" in self._task and hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            return proba.values if isinstance(proba, pd.DataFrame) else proba
        return None


def get_runner(framework: str, time_budget: int) -> AutoMLRunner:
    """Get AutoML runner by framework name."""
    runners = {
        "flaml": FLAMLRunner,
        "autogluon": AutoGluonRunner,
    }
    if framework not in runners:
        raise ValueError(f"Unknown framework: {framework}. Available: {list(runners.keys())}")
    return runners[framework](time_budget)


# =============================================================================
# Evaluation
# =============================================================================


def evaluate(y_true, y_pred, y_prob, task: str) -> dict[str, float]:
    """Evaluate predictions."""
    if "classification" in task:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        }
        if y_prob is not None:
            try:
                if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
                    y_prob = y_prob[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
            except (ValueError, IndexError):
                pass
        return metrics
    else:
        return {
            "r2": r2_score(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
        }


def get_primary_metric(task: str) -> str:
    """Get primary metric name for task."""
    return "accuracy" if "classification" in task else "r2"


# =============================================================================
# Data Preparation
# =============================================================================


def preprocess_data(X: pd.DataFrame, y, task: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Preprocess data for modeling."""
    X_processed = X.copy()

    # Encode categorical columns
    for col in X_processed.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X_processed[col].astype(str).fillna("missing"))

    # Fill numeric NaN
    for col in X_processed.select_dtypes(include=[np.number]).columns:
        X_processed[col] = X_processed[col].fillna(X_processed[col].median())

    # Process target
    if "classification" in task:
        if hasattr(y, "dtype") and (y.dtype == "object" or y.dtype.name == "category"):
            le = LabelEncoder()
            y_processed = le.fit_transform(y.astype(str))
        else:
            y_processed = np.array(y)
    else:
        y_processed = np.array(y).astype(float)

    return X_processed, y_processed


def apply_featcopilot(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train,
    max_features: int,
    with_llm: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Apply FeatCopilot feature engineering."""
    from featcopilot import AutoFeatureEngineer

    engines = ["tabular"]
    llm_config = None

    if with_llm:
        engines.append("llm")
        llm_config = {"model": "gpt-4o-mini", "max_suggestions": 10, "backend": "copilot"}

    engineer = AutoFeatureEngineer(
        engines=engines,
        max_features=max_features,
        llm_config=llm_config,
        verbose=False,
    )

    start = time.time()
    X_train_fe = engineer.fit_transform(X_train, y_train)
    X_test_fe = engineer.transform(X_test)
    fe_time = time.time() - start

    # Align columns
    for col in X_train_fe.columns:
        if col not in X_test_fe.columns:
            X_test_fe[col] = 0
    X_test_fe = X_test_fe[X_train_fe.columns]

    # Fill NaN
    X_train_fe = X_train_fe.fillna(0).replace([np.inf, -np.inf], 0)
    X_test_fe = X_test_fe.fillna(0).replace([np.inf, -np.inf], 0)

    return X_train_fe, X_test_fe, fe_time


# =============================================================================
# Benchmark Runner
# =============================================================================


def run_single_benchmark(
    dataset_name: str,
    framework: str,
    time_budget: int,
    max_features: int,
    with_llm: bool = False,
) -> Optional[dict[str, Any]]:
    """Run benchmark on a single dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    try:
        # Load dataset
        X, y, task, name = load_dataset(dataset_name)
        print(f"Task: {task}, Shape: {X.shape}")

        # Preprocess
        X_processed, y_processed = preprocess_data(X, y, task)

        # Split
        stratify = y_processed if "classification" in task and len(np.unique(y_processed)) < 50 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=0.2, random_state=42, stratify=stratify
        )

        results = {
            "dataset": dataset_name,
            "task": task,
            "framework": framework,
            "n_samples": len(X),
            "n_features_original": X.shape[1],
            "with_llm": with_llm,
        }
        primary_metric = get_primary_metric(task)

        # --- Baseline ---
        print("\n[1/3] Baseline (no FE)...")
        runner = get_runner(framework, time_budget)
        train_time = runner.fit(X_train, y_train, task)
        y_pred = runner.predict(X_test)
        y_prob = runner.predict_proba(X_test)
        baseline_metrics = evaluate(y_test, y_pred, y_prob, task)

        results["baseline_score"] = baseline_metrics[primary_metric]
        results["baseline_train_time"] = train_time
        print(f"   {primary_metric}: {baseline_metrics[primary_metric]:.4f}, time: {train_time:.1f}s")

        # --- FeatCopilot (tabular only) ---
        print("\n[2/3] FeatCopilot (tabular)...")
        X_train_fe, X_test_fe, fe_time = apply_featcopilot(X_train, X_test, y_train, max_features, with_llm=False)
        results["n_features_tabular"] = X_train_fe.shape[1]
        results["fe_time_tabular"] = fe_time

        runner = get_runner(framework, time_budget)
        train_time = runner.fit(X_train_fe, y_train, task)
        y_pred = runner.predict(X_test_fe)
        y_prob = runner.predict_proba(X_test_fe)
        tabular_metrics = evaluate(y_test, y_pred, y_prob, task)

        results["tabular_score"] = tabular_metrics[primary_metric]
        results["tabular_train_time"] = train_time
        improvement = (tabular_metrics[primary_metric] - baseline_metrics[primary_metric]) / max(
            baseline_metrics[primary_metric], 0.001
        )
        results["tabular_improvement_pct"] = improvement * 100
        print(
            f"   {primary_metric}: {tabular_metrics[primary_metric]:.4f} "
            f"({improvement*100:+.2f}%), features: {X_train_fe.shape[1]}"
        )

        # --- FeatCopilot + LLM (if enabled) ---
        if with_llm:
            print("\n[3/3] FeatCopilot (tabular + LLM)...")
            X_train_llm, X_test_llm, fe_time_llm = apply_featcopilot(
                X_train, X_test, y_train, max_features, with_llm=True
            )
            results["n_features_llm"] = X_train_llm.shape[1]
            results["fe_time_llm"] = fe_time_llm

            runner = get_runner(framework, time_budget)
            train_time = runner.fit(X_train_llm, y_train, task)
            y_pred = runner.predict(X_test_llm)
            y_prob = runner.predict_proba(X_test_llm)
            llm_metrics = evaluate(y_test, y_pred, y_prob, task)

            results["llm_score"] = llm_metrics[primary_metric]
            results["llm_train_time"] = train_time
            improvement = (llm_metrics[primary_metric] - baseline_metrics[primary_metric]) / max(
                baseline_metrics[primary_metric], 0.001
            )
            results["llm_improvement_pct"] = improvement * 100
            print(
                f"   {primary_metric}: {llm_metrics[primary_metric]:.4f} "
                f"({improvement*100:+.2f}%), features: {X_train_llm.shape[1]}"
            )
        else:
            print("\n[3/3] Skipped (--with-llm not enabled)")

        return results

    except Exception as e:
        print(f"Error: {e}")
        return None


def generate_report(results: list[dict], framework: str, with_llm: bool, output_path: Path) -> None:
    """Generate markdown report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Separate by task category
    clf_results = [r for r in results if r["task"] == "classification"]
    reg_results = [r for r in results if r["task"] == "regression"]
    ts_results = [r for r in results if r["task"] == "timeseries_regression"]
    text_clf_results = [r for r in results if r["task"] == "text_classification"]
    text_reg_results = [r for r in results if r["task"] == "text_regression"]

    report = f"""# AutoML Benchmark Report

**Generated:** {timestamp}
**Framework:** {framework.upper()}
**LLM Enabled:** {with_llm}
**Datasets:** {len(results)}

## Summary

| Metric | Value |
|--------|-------|
| Total Datasets | {len(results)} |
| Classification | {len(clf_results)} |
| Regression | {len(reg_results)} |
| Forecasting | {len(ts_results)} |
| Text Classification | {len(text_clf_results)} |
| Text Regression | {len(text_reg_results)} |
| Improved ({"LLM" if with_llm else "Tabular"}) | {sum(1 for r in results if r.get('llm_improvement_pct' if with_llm else 'tabular_improvement_pct', 0) > 0)} |
| Avg Improvement | {np.mean([r.get('llm_improvement_pct' if with_llm else 'tabular_improvement_pct', 0) for r in results]):.2f}% |

"""

    def add_classification_table(section_results: list[dict], title: str) -> str:
        """Generate classification results table."""
        if not section_results:
            return ""
        section = f"## {title}\n\n"
        section += "| Dataset | Baseline | Tabular | Improvement |"
        if with_llm:
            section += " LLM | LLM Imp |"
        section += " Features |\n"
        section += "|---------|----------|---------|-------------|"
        if with_llm:
            section += "------|---------|"
        section += "----------|\n"

        for r in section_results:
            section += f"| {r['dataset']} | {r['baseline_score']:.4f} | {r['tabular_score']:.4f} | {r['tabular_improvement_pct']:+.2f}% |"
            if with_llm and "llm_score" in r:
                section += f" {r['llm_score']:.4f} | {r['llm_improvement_pct']:+.2f}% |"
            elif with_llm:
                section += " - | - |"
            section += f" {r['n_features_original']}→{r['n_features_tabular']} |\n"
        return section + "\n"

    def add_regression_table(section_results: list[dict], title: str) -> str:
        """Generate regression results table."""
        if not section_results:
            return ""
        section = f"## {title}\n\n"
        section += "| Dataset | Baseline R² | Tabular R² | Improvement |"
        if with_llm:
            section += " LLM R² | LLM Imp |"
        section += " Features |\n"
        section += "|---------|-------------|------------|-------------|"
        if with_llm:
            section += "--------|---------|"
        section += "----------|\n"

        for r in section_results:
            section += f"| {r['dataset']} | {r['baseline_score']:.4f} | {r['tabular_score']:.4f} | {r['tabular_improvement_pct']:+.2f}% |"
            if with_llm and "llm_score" in r:
                section += f" {r['llm_score']:.4f} | {r['llm_improvement_pct']:+.2f}% |"
            elif with_llm:
                section += " - | - |"
            section += f" {r['n_features_original']}→{r['n_features_tabular']} |\n"
        return section + "\n"

    # Add all category sections
    report += add_classification_table(clf_results, "Classification Results")
    report += add_regression_table(reg_results, "Regression Results")
    report += add_regression_table(ts_results, "Forecasting Results")
    report += add_classification_table(text_clf_results, "Text Classification Results")
    report += add_regression_table(text_reg_results, "Text Regression Results")

    # Write report
    llm_suffix = "_LLM" if with_llm else ""
    report_file = output_path / f"AUTOML_{framework.upper()}_BENCHMARK{llm_suffix}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved: {report_file}")


def get_cache_file(output_path: Path, framework: str, with_llm: bool) -> Path:
    """Get cache file path."""
    llm_suffix = "_LLM" if with_llm else ""
    return output_path / f"AUTOML_{framework.upper()}_CACHE{llm_suffix}.json"


def save_cache(results: list[dict], output_path: Path, framework: str, with_llm: bool) -> None:
    """Save benchmark results to cache file."""
    cache_file = get_cache_file(output_path, framework, with_llm)
    # Convert numpy types to native Python types for JSON serialization
    serializable_results = []
    for r in results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                sr[k] = float(v)
            elif isinstance(v, np.ndarray):
                sr[k] = v.tolist()
            elif isinstance(v, dict):
                sr[k] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv for kk, vv in v.items()}
            else:
                sr[k] = v
        serializable_results.append(sr)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Cache saved: {cache_file}")


def load_cache(output_path: Path, framework: str, with_llm: bool) -> Optional[list[dict]]:
    """Load benchmark results from cache file."""
    cache_file = get_cache_file(output_path, framework, with_llm)
    if not cache_file.exists():
        print(f"Cache file not found: {cache_file}")
        return None
    with open(cache_file, encoding="utf-8") as f:
        results = json.load(f)
    print(f"Loaded {len(results)} results from cache: {cache_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="AutoML Benchmark for FeatCopilot")
    parser.add_argument("--datasets", type=str, help="Comma-separated dataset names")
    parser.add_argument("--category", type=str, choices=["classification", "regression", "forecasting", "text"])
    parser.add_argument("--all", action="store_true", help="Run all datasets")
    parser.add_argument("--framework", type=str, default="flaml", choices=["flaml", "autogluon"])
    parser.add_argument("--with-llm", action="store_true", help="Enable LLM engine")
    parser.add_argument("--time-budget", type=int, default=DEFAULT_TIME_BUDGET)
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--output", type=str, default="benchmarks/automl")
    parser.add_argument("--report-only", action="store_true", help="Only regenerate report from cache")
    parser.add_argument("--no-cache", action="store_true", help="Don't save results to cache")

    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Report-only mode: load from cache and regenerate report
    if args.report_only:
        results = load_cache(output_path, args.framework, args.with_llm)
        if results:
            generate_report(results, args.framework, args.with_llm, output_path)
        return

    # Determine datasets to run
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

    print("AutoML Benchmark")
    print("================")
    print(f"Framework: {args.framework}")
    print(f"Time budget: {args.time_budget}s")
    print(f"LLM enabled: {args.with_llm}")
    print(f"Datasets: {len(dataset_names)}")

    # Run benchmarks
    results = []
    for name in dataset_names:
        result = run_single_benchmark(
            name,
            args.framework,
            args.time_budget,
            args.max_features,
            args.with_llm,
        )
        if result:
            results.append(result)

    # Save cache and generate report
    if results:
        if not args.no_cache:
            save_cache(results, output_path, args.framework, args.with_llm)
        generate_report(results, args.framework, args.with_llm, output_path)


if __name__ == "__main__":
    main()
