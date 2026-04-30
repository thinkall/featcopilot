"""Tests for the shared benchmarks split helper and its wiring."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.splits import split_benchmark_data

# ---------------------------------------------------------------------------
# Behavioral tests for split_benchmark_data
# ---------------------------------------------------------------------------


def test_classification_uses_stratified_split():
    """Classification tasks should produce a stratified split when class counts allow."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f": rng.normal(size=200)})
    y = pd.Series(([0] * 160) + ([1] * 40))

    train_idx, test_idx, y_train, y_test = split_benchmark_data(X, y, "classification", random_state=42)

    assert len(train_idx) + len(test_idx) == len(X)
    assert set(train_idx).isdisjoint(set(test_idx))
    train_pos_ratio = (y_train == 1).mean()
    test_pos_ratio = (y_test == 1).mean()
    expected_ratio = (y == 1).mean()
    assert abs(train_pos_ratio - expected_ratio) < 0.02
    assert abs(test_pos_ratio - expected_ratio) < 0.02


def test_classification_falls_back_when_class_too_small():
    """Singleton classes should not raise; the split falls back to non-stratified."""
    X = pd.DataFrame({"f": np.arange(10.0)})
    y = pd.Series([0] * 9 + [1])

    train_idx, test_idx, _, _ = split_benchmark_data(X, y, "classification", random_state=0)

    assert len(train_idx) + len(test_idx) == len(X)


def test_forecasting_uses_chronological_split():
    """Forecasting/timeseries tasks should preserve temporal order."""
    X = pd.DataFrame({"t": np.arange(100)})
    y = pd.Series(np.arange(100, dtype=float))

    train_idx, test_idx, y_train, y_test = split_benchmark_data(X, y, "forecasting", random_state=123)

    assert list(train_idx) == list(range(80))
    assert list(test_idx) == list(range(80, 100))
    assert y_train.iloc[-1] < y_test.iloc[0]


def test_timeseries_keyword_also_chronological():
    """The 'timeseries' substring should also trigger chronological split."""
    X = pd.DataFrame({"t": np.arange(50)})
    y = pd.Series(np.arange(50, dtype=float))

    train_idx, test_idx, _, _ = split_benchmark_data(X, y, "timeseries_regression", random_state=0)

    assert list(train_idx) == list(range(40))
    assert list(test_idx) == list(range(40, 50))


def test_regression_uses_random_split_no_stratify():
    """Non-classification, non-forecasting tasks should use a plain random split."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"f": rng.normal(size=100)})
    y = pd.Series(rng.normal(size=100))

    train_idx, test_idx, _, _ = split_benchmark_data(X, y, "regression", random_state=42)

    assert len(train_idx) == 80
    assert len(test_idx) == 20
    assert set(train_idx).isdisjoint(set(test_idx))


def test_custom_test_size_respected():
    """``test_size`` should override the default 0.2."""
    X = pd.DataFrame({"f": np.arange(100.0)})
    y = pd.Series(np.arange(100, dtype=float))

    train_idx, test_idx, _, _ = split_benchmark_data(X, y, "regression", random_state=0, test_size=0.4)

    assert len(test_idx) == 40
    assert len(train_idx) == 60


@pytest.mark.parametrize("bad_test_size", [0.0, 1.0, -0.1, 1.5, 2])
def test_split_benchmark_data_rejects_out_of_range_test_size(bad_test_size):
    """``test_size`` must be strictly between 0 and 1 for both branches."""
    X = pd.DataFrame({"f": np.arange(100.0)})
    y = pd.Series(np.arange(100, dtype=float))

    # Random branch.
    with pytest.raises(ValueError, match="test_size must be a float strictly between 0 and 1"):
        split_benchmark_data(X, y, "regression", random_state=0, test_size=bad_test_size)

    # Chronological branch -- previously silently produced empty/overlapping splits.
    with pytest.raises(ValueError, match="test_size must be a float strictly between 0 and 1"):
        split_benchmark_data(X, y, "forecasting", random_state=0, test_size=bad_test_size)


def test_split_benchmark_data_chronological_rejects_empty_train_split():
    """Tiny datasets with extreme ``test_size`` must raise instead of producing an empty train set."""
    X = pd.DataFrame({"t": np.arange(2)})
    y = pd.Series(np.arange(2, dtype=float))

    # len=2, test_size=0.9 -> split_idx = int(2 * 0.1) = 0 -> empty train.
    with pytest.raises(ValueError, match="Chronological split would leave one side empty"):
        split_benchmark_data(X, y, "forecasting", random_state=0, test_size=0.9)


def test_split_benchmark_data_chronological_single_row_dataset_raises():
    """A single-row dataset cannot be chronologically split for any valid ``test_size``."""
    X = pd.DataFrame({"t": [0]})
    y = pd.Series([0.0])

    with pytest.raises(ValueError, match="Chronological split would leave one side empty"):
        split_benchmark_data(X, y, "forecasting", random_state=0, test_size=0.5)


# ---------------------------------------------------------------------------
# Wiring tests: ensure benchmark scripts actually use the shared helper.
# These guard against the regression flagged on PR #2 where the helper was
# introduced but never wired into the call sites.
# ---------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "module_path",
    [
        "benchmarks.compare_tools.run_fe_tools_comparison",
        "benchmarks.use_cases.run_auto_feature_engineering_benchmark",
    ],
)
def test_benchmark_scripts_import_split_helper(module_path):
    """Both in-scope benchmark scripts must import ``split_benchmark_data``."""
    module = importlib.import_module(module_path)
    assert (
        getattr(module, "split_benchmark_data", None) is split_benchmark_data
    ), f"{module_path} should import split_benchmark_data from benchmarks.splits"


@pytest.mark.parametrize(
    "relative_path",
    [
        "benchmarks/compare_tools/run_fe_tools_comparison.py",
        "benchmarks/use_cases/run_auto_feature_engineering_benchmark.py",
    ],
)
def test_benchmark_scripts_do_not_call_train_test_split_directly(relative_path):
    """Benchmark scripts should route through ``split_benchmark_data`` instead of ``train_test_split``."""
    source = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")
    assert (
        "train_test_split(" not in source
    ), f"{relative_path} should not call train_test_split directly; use split_benchmark_data."
    assert (
        "split_benchmark_data(" in source
    ), f"{relative_path} should call split_benchmark_data from benchmarks.splits."


def test_split_benchmark_data_signature_stable():
    """Pin the public signature so downstream benchmark scripts keep working."""
    sig = inspect.signature(split_benchmark_data)
    params = list(sig.parameters)
    assert params == ["X", "y", "task", "random_state", "test_size"]
    assert sig.parameters["test_size"].default == 0.2
