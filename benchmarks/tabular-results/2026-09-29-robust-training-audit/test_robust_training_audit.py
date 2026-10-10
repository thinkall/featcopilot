"""Synthetic safety checks for the final bounded robust replay."""

import numpy as np
import pandas as pd
import pytest
import robust_training_audit as replay
import training_selection_audit as helper
from test_aligned_training_audit import previous_fixture


def test_exact_prior_splits_and_excluded_outer_rows_remain_isolated():
    data, previous, partitions = previous_fixture()
    for fold in range(3):
        copied = {"X": data["X"].copy(), "y": data["y"].copy()}
        excluded = partitions[fold][1]
        copied["X"].loc[excluded, "x"] = np.inf
        copied["y"].loc[excluded] = np.nan
        X, y, train, audit, checks = replay.isolate_verified_partition(copied, "synthetic", fold, previous, helper)
        payload = replay.make_payload(X, y, train, audit, checks, "featcopilot-frequency-aligned-robust")
        assert not set(X.index) & set(excluded)
        assert np.isfinite(X.to_numpy()).all() and y.notna().all()
        assert payload["split_sha256"] == previous["rows"][fold]["audit_split_sha256"]
        assert payload["config"]["max_candidates"] == 1000
        assert payload["config"]["max_added_features"] == 50
        assert payload["config"]["threads"] == 8
        assert payload["config"]["memory_limit_mb"] == 4096


def test_raw_numeric_diagnostics_use_only_training_bounds_and_publish_no_rows():
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 1000.0], "category": ["private"] * 5})
    before = X.copy(deep=True)
    result = replay.numeric_support_diagnostics(X, np.arange(4), np.array([4]))
    item = result["features"][0]
    assert item["audit_standardized_max_abs_before"] > item["train_standardized_max_abs"]
    assert item["audit_standardized_max_abs_after"] == item["train_standardized_max_abs"]
    assert item["audit_above_training_bound"] == 1
    assert "category" not in str(result)
    assert "private" not in str(result)
    pd.testing.assert_frame_equal(X, before)


def test_fixed_methods_and_budget_reserve():
    assert replay.METHODS == (
        "baseline",
        "baseline-robust",
        "featcopilot-frequency-aligned",
        "featcopilot-frequency-aligned-robust",
    )
    assert len(replay.FAMILIES) * 3 * len(replay.METHODS) == 24
    assert replay.SECONDS_CEILING == 1500
    assert replay.REPORT_RESERVE_SECONDS == 60
    assert replay.CELL_SECONDS == 600


def test_robust_alignment_and_matched_control_contract():
    losses = [3.0, 1.0, 2.0]
    trials = [
        {"parameters": {"regularization": alpha}, "loss": loss, "selection_report": {"trial": index}}
        for index, (alpha, loss) in enumerate(zip(replay.ALPHAS, losses, strict=True))
    ]
    row = {
        "status": "success",
        "method": "featcopilot-frequency-aligned-robust",
        "split_sha256": "audit",
        "n_train": 30,
        "n_test": 10,
        "parameters": {"regularization": 1.0},
        "selection_model_parameters": {"regularization": 1.0},
        "inner_losses": losses,
        "inner_pipeline_trials": trials,
        "inner_selection_report": trials[1]["selection_report"],
        "numeric_clip_quantiles": [0.0, 1.0],
        "matched_baseline_method": "baseline-robust",
    }
    payload = {"split_sha256": "audit", "train_idx": np.arange(30), "test_idx": np.arange(30, 40)}
    replay.validate_record(row, payload, {})
    assert row["aligned_trial_parameter_validation_passed"]
    row["numeric_clip_quantiles"] = [0.01, 0.99]
    with pytest.raises(ValueError, match="Robust worker policy"):
        replay.validate_record(row, payload, {})
