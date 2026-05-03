# Benchmark module
from benchmarks.datasets import (
    CATEGORY_CLASSIFICATION,
    CATEGORY_FORECASTING,
    CATEGORY_REGRESSION,
    CATEGORY_TEXT,
    SOURCE_REAL_WORLD,
    SOURCE_SYNTHETIC,
    get_all_datasets,
    get_category_summary,
    get_dataset_info,
    get_text_datasets,
    get_timeseries_datasets,
    is_real_world,
    list_datasets,
    list_real_world_datasets,
    list_synthetic_datasets,
    load_all_datasets,
    load_dataset,
    load_datasets,
)

__all__ = [
    # Dataset API
    "list_datasets",
    "list_real_world_datasets",
    "list_synthetic_datasets",
    "is_real_world",
    "load_dataset",
    "load_datasets",
    "load_all_datasets",
    "get_dataset_info",
    "get_category_summary",
    "CATEGORY_CLASSIFICATION",
    "CATEGORY_REGRESSION",
    "CATEGORY_FORECASTING",
    "CATEGORY_TEXT",
    "SOURCE_REAL_WORLD",
    "SOURCE_SYNTHETIC",
    # Legacy
    "get_all_datasets",
    "get_timeseries_datasets",
    "get_text_datasets",
]
