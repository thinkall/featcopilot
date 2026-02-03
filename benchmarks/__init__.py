# Benchmark module
from benchmarks.datasets import (
    CATEGORY_CLASSIFICATION,
    CATEGORY_FORECASTING,
    CATEGORY_REGRESSION,
    CATEGORY_TEXT,
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

__all__ = [
    # Dataset API
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
    # Legacy
    "get_all_datasets",
    "get_timeseries_datasets",
    "get_text_datasets",
]
