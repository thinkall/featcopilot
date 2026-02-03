# Benchmark module
from benchmarks.datasets import (
    # New unified API
    CATEGORY_CLASSIFICATION,
    CATEGORY_FORECASTING,
    CATEGORY_REGRESSION,
    CATEGORY_TEXT,
    # Legacy API
    get_all_datasets,
    get_category_summary,
    get_dataset_info,
    get_text_datasets,
    get_timeseries_datasets,
    list_datasets,
    load_all_datasets,
    load_dataset,
    load_datasets,
)
from benchmarks.feature_engineering.run_benchmark import generate_report, run_all_benchmarks, run_benchmark_single

__all__ = [
    # New unified API
    "list_datasets",
    "load_dataset",
    "load_datasets",
    "load_all_datasets",
    "get_dataset_info",
    "get_category_summary",
    "CATEGORY_CLASSIFICATION",
    "CATEGORY_REGRESSION",
    "CATEGORY_FORECASTING",
    "CATEGORY_TEXT",
    # Legacy API
    "get_all_datasets",
    "get_timeseries_datasets",
    "get_text_datasets",
    # Benchmark functions
    "run_all_benchmarks",
    "run_benchmark_single",
    "generate_report",
]
