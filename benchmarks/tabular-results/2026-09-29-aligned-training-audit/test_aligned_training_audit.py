"""Synthetic-only validation of exact-partition benchmark replay wiring."""

import json

import aligned_training_audit as replay
import numpy as np
import pandas as pd
import pytest
import training_selection_audit as helper
from sklearn.model_selection import KFold


def previous_fixture():
    data = {"X": pd.DataFrame({"x": np.arange(60, dtype=float)}), "y": pd.Series(np.arange(60, dtype=float))}
    partitions = list(KFold(3, shuffle=True, random_state=2027).split(np.empty((60, 0))))
    splits = [helper.split_hash(train, test) for train, test in partitions]
    previous = {"protocol": {"data": [{"family": "synthetic", "outer_split_hashes": splits}]}, "rows": []}
    for fold in range(3):
        X, y, identity = helper.isolate_outer_train(data, fold, splits)
        train, audit, inner_train, inner_valid = helper.nested_partitions(len(X))
        previous["rows"].append(
            {
                **identity,
                "family": "synthetic",
                "outer_fold": fold,
                "audit_split_sha256": helper.split_hash(train, audit),
                "inner_split_sha256": helper.split_hash(inner_train, inner_valid),
                "isolated_outer_train_sha256": replay.sha(
                    pd.util.hash_pandas_object(pd.concat([X, y], axis=1)).to_numpy().tobytes()
                ),
            }
        )
    return data, previous, partitions


@pytest.mark.parametrize("fold", [0, 1, 2])
def test_only_corresponding_original_outer_train_enters_benchmark_payload(fold):
    data, previous, partitions = previous_fixture()
    excluded = partitions[fold][1]
    data["X"].loc[excluded, "x"] = np.inf
    data["y"].loc[excluded] = np.nan
    X, y, train, audit, checks = replay.isolate_verified_partition(data, "synthetic", fold, previous, helper)
    payload = replay.make_payload(X, y, train, audit, checks, "featcopilot-frequency-aligned")
    assert set(payload["X"].index).isdisjoint(excluded)
    assert set(payload["y"].index).isdisjoint(excluded)
    assert np.isfinite(payload["X"].to_numpy()).all()
    assert payload["y"].notna().all()
    assert helper.split_hash(payload["train_idx"], payload["test_idx"]) == previous["rows"][fold]["audit_split_sha256"]
    assert payload["config"]["seed"] == 2027
    assert payload["config"]["threads"] == 8
    assert payload["config"]["learners"] == ["linear"]
    assert payload["config"]["stability_repeats"] == 3
    assert payload["config"]["max_candidates"] == 1000
    assert payload["config"]["max_added_features"] == 50
    assert payload["config"]["memory_limit_mb"] == 4096


def test_altered_audit_hash_is_rejected_before_any_benchmark_call():
    data, previous, _ = previous_fixture()
    previous["rows"][0]["audit_split_sha256"] = "wrong"
    with pytest.raises(ValueError, match="audit_split_sha256"):
        replay.isolate_verified_partition(data, "synthetic", 0, previous, helper)


def test_aligned_records_must_pair_trial_selection_and_learner_parameters():
    losses = [3.0, 1.0, 2.0]
    trials = [
        {"parameters": {"regularization": alpha}, "loss": loss, "selection_report": {"chosen": index}}
        for index, (alpha, loss) in enumerate(zip(replay.ALPHAS, losses, strict=True))
    ]
    row = {
        "status": "success",
        "method": "featcopilot-frequency-aligned",
        "split_sha256": "audit",
        "n_train": 30,
        "n_test": 10,
        "parameters": {"regularization": 1.0},
        "selection_model_parameters": {"regularization": 1.0},
        "inner_losses": losses,
        "inner_pipeline_trials": trials,
        "inner_selection_report": trials[1]["selection_report"],
    }
    payload = {"split_sha256": "audit", "train_idx": np.arange(30), "test_idx": np.arange(30, 40)}
    replay.validate_record(row, payload, {})
    assert row["aligned_trial_parameter_validation_passed"]
    changed = json.loads(json.dumps(row))
    changed["selection_model_parameters"] = {"regularization": 10.0}
    with pytest.raises(ValueError, match="parameters"):
        replay.validate_record(changed, payload, {})
