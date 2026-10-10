"""Bounded synthetic tests for canonical regression prediction averaging."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks import regression_averaging as averaging
from benchmarks import tabular
from benchmarks.execution import split_fingerprint


def _payload(learner: str = "linear") -> dict:
    rng = np.random.default_rng(913)
    X = pd.DataFrame(
        {"a": rng.normal(size=80), "b": rng.normal(size=80)},
        index=np.arange(80) * 3 + 11,
    )
    X["category"] = ["PRIVATE-ONE", "PRIVATE-TWO", None, "missing:"] * 20
    y = pd.Series(2 * X.a - X.b + rng.normal(scale=0.1, size=len(X)), index=X.index)
    train, test = np.arange(60), np.arange(60, 80)
    return {
        "X": X,
        "y": y,
        "config": {
            "learners": [learner],
            "threads": 1,
            "seed": 42,
            "max_added_features": 4,
            "max_candidates": 12,
            "stability_repeats": 1,
            "categorical_encoding": "one_hot",
            "numeric_clip_quantiles": [0.0, 1.0],
        },
        "dataset": "synthetic-regression",
        "family_id": "synthetic-family",
        "source": "synthetic",
        "task": "regression",
        "method": averaging.METHOD,
        "seed": 42,
        "fold": 1,
        "role": "development",
        "expected_cells": 3,
        "train_idx": train,
        "test_idx": test,
        "split_sha256": split_fingerprint(train, test),
        "groups": None,
        "timestamps": None,
        "label_end_times": None,
    }


def _row(case: dict, prediction: np.ndarray) -> dict:
    learner = case["config"]["learners"][0]
    fe = case["method"] == "featcopilot-frequency-aligned-robust"
    width = case["X"].shape[1] + int(fe)
    parameters = tabular.candidate_parameters(learner)
    losses = [3.0, 1.0, 2.0]
    stages = dict.fromkeys(averaging._STAGES, 0.2 if fe else 0.1)
    row = {
        **{key: case[key] for key in averaging._IDENTITY},
        "method": case["method"],
        "family_id": case.get("family_id", case["dataset"]),
        "role": case.get("role", "development"),
        "expected_cells": case.get("expected_cells", 0),
        "learner": learner,
        "status": "success",
        **tabular.regression_metrics(prediction, case["y"].iloc[case["test_idx"]]),
        "parameters": parameters[1],
        "inner_losses": losses,
        "n_train": len(case["train_idx"]),
        "n_test": len(case["test_idx"]),
        "n_input_features": case["X"].shape[1],
        "n_output_features": width,
        "outer_prediction_sha256": tabular.digest(np.asarray(prediction, dtype="<f8").tobytes()),
        "outer_feature_names_sha256": tabular.digest(
            json.dumps(list(case["X"].columns) + (["abs_a"] if fe else [])).encode()
        ),
        "outer_selection_report": {"selected_features": ["abs_a"]} if fe else None,
        "inner_selection_report": {"selected_features": ["abs_a"]} if fe else None,
        "numeric_clip_quantiles": [0.0, 1.0],
        "matched_baseline_method": "baseline-robust",
        **stages,
        "pipeline_seconds": sum(stages.values()),
        "case_seconds": sum(stages.values()) + 0.1,
    }
    if "stage" in case:
        row["stage"] = case["stage"]
    if fe:
        row["selection_model_parameters"] = parameters[1]
        row["inner_pipeline_trials"] = [
            {
                "parameters": candidate,
                "loss": loss,
                "n_output_features": width,
                "selection_report": {"selected_features": ["abs_a"]},
            }
            for candidate, loss in zip(parameters, losses, strict=True)
        ]
    return row


@pytest.fixture
def controls(monkeypatch: pytest.MonkeyPatch) -> dict:
    calls: list[dict] = []
    returned: list[dict] = []
    captures: list[dict[str, np.ndarray]] = []

    def run(case: dict, workspace: Path, *, outer_predictions: dict[str, np.ndarray] | None = None) -> list[dict]:
        assert outer_predictions is not None
        calls.append(case)
        target = case["y"].iloc[case["test_idx"]].to_numpy()
        prediction = target + (3.0 if case["method"] == "baseline-robust" else -3.0)
        outer_predictions[case["config"]["learners"][0]] = prediction
        captures.append(outer_predictions)
        row = _row(case, prediction)
        returned.append(row)
        return [row]

    monkeypatch.setattr(tabular, "run_case", run)
    return {"calls": calls, "rows": returned, "captures": captures, "run": run}


@pytest.mark.parametrize("learner", ["linear", "lightgbm"])
def test_predictions_not_scores_are_averaged_once_and_controls_unchanged(
    learner: str, controls: dict, tmp_path: Path
) -> None:
    case = _payload(learner)
    original = copy.deepcopy(case)
    case["data_path"] = str(tmp_path / "must-not-reread-prepared-data.pkl")
    rows = averaging.run_case(case, tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["method"] == averaging.METHOD
    assert row["learner"] == learner and row["status"] == "success"
    assert [call["method"] for call in controls["calls"]] == [method for _, method in averaging.COMPONENTS]
    assert len(controls["calls"]) == 2
    assert controls["captures"][0] is not controls["captures"][1]
    for call in controls["calls"]:
        assert call["config"] == original["config"]
        assert call["config"] is not case["config"]
        assert call["X"] is case["X"] and call["y"] is case["y"]
        assert call["train_idx"] is case["train_idx"] and call["test_idx"] is case["test_idx"]
        assert "data_path" not in call
    expected = 0.5 * controls["captures"][0][learner] + 0.5 * controls["captures"][1][learner]
    expected_metrics = tabular.regression_metrics(expected, case["y"].iloc[case["test_idx"]])
    assert {key: row[key] for key in expected_metrics} == expected_metrics
    assert row["loss"] < 1e-14
    assert np.mean([component["loss"] for component in controls["rows"]]) == pytest.approx(3)
    report = row["regression_average"]
    assert report["weights"] == {"raw": 0.5, "fe": 0.5}
    assert report["weight_optimized"] is False
    assert report["complete_procedure_count"] == 2
    assert report["components"]["raw"] == controls["rows"][0]
    assert report["components"]["fe"] == controls["rows"][1]
    assert row["outer_prediction_sha256"] == tabular.digest(np.asarray(expected, dtype="<f8").tobytes())
    pd.testing.assert_frame_equal(case["X"], original["X"])
    pd.testing.assert_series_equal(case["y"], original["y"])
    assert case["config"] == original["config"]


def test_widths_provenance_and_aggregate_costs_are_explicit(controls: dict, tmp_path: Path) -> None:
    row = averaging.run_case(_payload(), tmp_path)[0]
    assert row["component_output_features"] == {"raw": 3, "fe": 4}
    assert row["n_output_features"] == 4
    assert row["n_output_features_semantics"] == "maximum_separate_component_plan_width_not_union"
    assert row["outer_feature_names_sha256_semantics"] == "role_keyed_component_feature_name_hashes_not_union"
    assert row["regression_average"]["component_output_features"] == {"raw": 3, "fe": 4}
    for key in averaging._STAGES[:-1]:
        assert row[key] == pytest.approx(0.3)
    overhead = row["averaging_seconds"] + row["average_scoring_seconds"]
    assert row["predict_seconds"] == pytest.approx(0.3 + overhead)
    assert row["pipeline_seconds"] == pytest.approx(1.5 + overhead)
    assert row["pipeline_seconds"] == sum(row[key] for key in averaging._STAGES)
    assert row["case_seconds"] > 0
    serialized = json.dumps(row, allow_nan=False)
    assert "PRIVATE-ONE" not in serialized and "PRIVATE-TWO" not in serialized
    assert "no_causal_variance_reduction_or_no_harm_guarantee" in serialized
    assert "abs_a" in serialized
    controls["rows"][0]["parameters"]["regularization"] = 999
    assert row["regression_average"]["components"]["raw"]["parameters"] != controls["rows"][0]["parameters"]


@pytest.mark.parametrize(
    "scope, key, value",
    [
        ("payload", "task", "classification"),
        ("payload", "method", "baseline-robust"),
        ("payload", "groups", np.arange(80)),
        ("payload", "timestamps", np.arange(80)),
        ("payload", "label_end_times", np.arange(80)),
        ("config", "groups", [1]),
        ("config", "metadata", {"timestamps": [1]}),
        ("payload", "metadata", {"metadata": {"label_end_times": [1]}}),
        ("config", "categorical_encoding", "target"),
        ("payload", "target_transform", "log1p"),
        ("config", "target_transform", "standardize"),
        ("config", "split_kind", "temporal"),
        ("config", "split_strategy", "grouped"),
        ("config", "numeric_clip_quantiles", [0.01, 0.99]),
        ("config", "learners", ["linear", "lightgbm"]),
        ("config", "learners", ["catboost"]),
        ("config", "learners", ["random_forest"]),
        ("config", "seed", 137),
        ("config", "seed", True),
        ("payload", "seed", 42.0),
        ("payload", "fold", -1),
        ("config", "task", "classification"),
        ("config", "threads", 0),
        ("config", "threads", True),
        ("config", "max_added_features", -1),
        ("config", "max_added_features", 4.0),
        ("config", "max_candidates", 0),
        ("config", "stability_repeats", 0),
        ("attrs", "target_transform", "log1p"),
        ("attrs", "groups", [1]),
    ],
)
def test_unsupported_configuration_fails_before_components(
    scope: str, key: str, value: object, controls: dict, tmp_path: Path
) -> None:
    case = _payload()
    destination = case if scope == "payload" else (case["config"] if scope == "config" else case["X"].attrs)
    destination[key] = value
    with pytest.raises((ValueError, TypeError)):
        averaging.run_case(case, tmp_path)
    assert controls["calls"] == []


@pytest.mark.parametrize(
    "kind",
    [
        "overlap",
        "duplicate",
        "negative",
        "out_of_range",
        "float",
        "empty",
        "hash",
        "target_index",
        "target_nan",
        "target_2d",
        "frame_array",
        "columns",
        "tiny_test",
    ],
)
def test_invalid_splits_and_row_alignment_fail_before_components(kind: str, controls: dict, tmp_path: Path) -> None:
    case = _payload()
    if kind == "overlap":
        case["test_idx"][0] = 0
    elif kind == "duplicate":
        case["train_idx"][0] = 1
    elif kind == "negative":
        case["train_idx"][0] = -1
    elif kind == "out_of_range":
        case["test_idx"][-1] = 80
    elif kind == "float":
        case["train_idx"] = case["train_idx"].astype(float)
    elif kind == "empty":
        case["train_idx"] = np.array([], dtype=int)
    elif kind == "hash":
        case["split_sha256"] = "0" * 64
    elif kind == "target_index":
        case["y"] = case["y"].reset_index(drop=True)
    elif kind == "target_nan":
        case["y"].iloc[0] = np.nan
    elif kind == "target_2d":
        case["y"] = case["y"].to_frame()
    elif kind == "frame_array":
        case["X"] = case["X"].to_numpy()
    elif kind == "columns":
        case["X"].columns = ["a", "a", "c"]
    else:
        case["test_idx"] = case["test_idx"][:1]
    with pytest.raises((ValueError, TypeError)):
        averaging.run_case(case, tmp_path)
    assert controls["calls"] == []


@pytest.mark.parametrize("role", ["raw", "fe"])
@pytest.mark.parametrize(
    "kind", ["missing", "wrong_learner", "short", "two_dim", "nan", "inf", "complex", "strings", "hash"]
)
def test_bad_prediction_captures_never_become_average_success(
    role: str, kind: str, controls: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original = controls["run"]
    method = dict(averaging.COMPONENTS)[role]

    def run(case, workspace, *, outer_predictions):
        rows = original(case, workspace, outer_predictions=outer_predictions)
        if case["method"] != method:
            return rows
        key = case["config"]["learners"][0]
        prediction = outer_predictions[key]
        if kind == "missing":
            outer_predictions.clear()
        elif kind == "wrong_learner":
            outer_predictions["other"] = prediction
        elif kind == "short":
            outer_predictions[key] = prediction[:-1]
        elif kind == "two_dim":
            outer_predictions[key] = prediction[:, None]
        elif kind in ("nan", "inf"):
            prediction[0] = np.nan if kind == "nan" else np.inf
        elif kind == "complex":
            outer_predictions[key] = prediction.astype(complex)
        elif kind == "strings":
            outer_predictions[key] = prediction.astype(str)
        else:
            rows[0]["outer_prediction_sha256"] = "0" * 64
        return rows

    monkeypatch.setattr(tabular, "run_case", run)
    with pytest.raises(ValueError):
        averaging.run_case(_payload(), tmp_path)
    assert len(controls["calls"]) == (1 if role == "raw" else 2)


@pytest.mark.parametrize(
    "key, value",
    [
        ("method", averaging.METHOD),
        ("learner", "lightgbm"),
        ("seed", 99),
        ("split_sha256", "0" * 64),
        ("role", "confirmation"),
        ("family_id", "another-family"),
        ("task", "classification"),
        ("expected_cells", 999),
        ("stage", "unexpected-stage"),
        ("n_test", 19),
        ("n_train", 59),
        ("n_input_features", 2),
        ("n_output_features", 100),
        ("numeric_clip_quantiles", [0.01, 0.99]),
        ("matched_baseline_method", "baseline"),
        ("parameters", {"regularization": 999}),
        ("inner_losses", [1, 2]),
        ("inner_losses", [1, np.nan, 3]),
        ("loss", np.nan),
        ("mae", -1),
        ("pipeline_seconds", 100),
        ("tuning_seconds", -1),
        ("outer_feature_names_sha256", "not-a-hash"),
    ],
)
def test_component_identity_metrics_search_and_timing_failures(
    key: str, value: object, controls: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original = controls["run"]

    def run(case, workspace, *, outer_predictions):
        rows = original(case, workspace, outer_predictions=outer_predictions)
        rows[0][key] = value
        return rows

    monkeypatch.setattr(tabular, "run_case", run)
    with pytest.raises(ValueError):
        averaging.run_case(_payload(), tmp_path)
    assert len(controls["calls"]) == 1


@pytest.mark.parametrize("failure", ["raise", "failure_row", "empty", "duplicate", "fe_trials"])
def test_component_failure_propagates_without_fallback(
    failure: str, controls: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original = controls["run"]
    error = RuntimeError("canonical component failed")

    def run(case, workspace, *, outer_predictions):
        rows = original(case, workspace, outer_predictions=outer_predictions)
        if case["method"] == "baseline-robust":
            return rows
        if failure == "raise":
            raise error
        if failure == "failure_row":
            rows[0]["status"] = "error"
        elif failure == "empty":
            return []
        elif failure == "duplicate":
            return rows * 2
        else:
            rows[0]["inner_pipeline_trials"] = []
        return rows

    monkeypatch.setattr(tabular, "run_case", run)
    with pytest.raises((RuntimeError, ValueError)) as caught:
        averaging.run_case(_payload(), tmp_path)
    if failure == "raise":
        assert caught.value is error
    assert len(controls["calls"]) == 2


def test_stage_identity_must_match(controls: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    case = _payload()
    case["stage"] = "locked-replication"
    assert averaging.run_case(case, tmp_path)[0]["stage"] == "locked-replication"
    original = controls["run"]

    def run(case, workspace, *, outer_predictions):
        rows = original(case, workspace, outer_predictions=outer_predictions)
        rows[0]["stage"] = "different"
        return rows

    monkeypatch.setattr(tabular, "run_case", run)
    with pytest.raises(ValueError, match="stage"):
        averaging.run_case(case, tmp_path)


@pytest.fixture
def small_native(monkeypatch: pytest.MonkeyPatch) -> None:
    original = tabular.make_model

    def small(X, task, learner, parameters, config):
        result = original(X, task, learner, parameters, config)
        if learner == "lightgbm":
            result.steps[-1][1].set_params(n_estimators=3, min_child_samples=5)
        return result

    monkeypatch.setattr(tabular, "make_model", small)


@pytest.mark.parametrize("learner", ["linear", "lightgbm"])
def test_canonical_dispatch_real_synthetic_models_match_direct_controls(
    learner: str, small_native: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if learner == "lightgbm":
        pytest.importorskip("lightgbm")
    case = _payload(learner)
    original = tabular.run_case
    calls = []

    def counted(payload, workspace, *, outer_predictions=None):
        calls.append(payload["method"])
        return original(payload, workspace, outer_predictions=outer_predictions)

    monkeypatch.setattr(tabular, "run_case", counted)
    rows = tabular.run_case(case, tmp_path)
    assert calls == [averaging.METHOD, "baseline-robust", "featcopilot-frequency-aligned-robust"]
    row = rows[0]
    assert len(rows) == 1
    direct = {}
    for role, method in averaging.COMPONENTS:
        capture: dict[str, np.ndarray] = {}
        component = original({**case, "method": method}, tmp_path, outer_predictions=capture)[0]
        direct[role] = capture[learner]
        nested = row["regression_average"]["components"][role]
        for key in (
            "loss",
            "mae",
            "r2",
            "parameters",
            "inner_losses",
            "n_output_features",
            "outer_prediction_sha256",
            "outer_feature_names_sha256",
            "inner_selection_report",
            "outer_selection_report",
        ):
            assert nested[key] == component[key]
        if role == "fe":
            assert nested["inner_pipeline_trials"] == component["inner_pipeline_trials"]
    expected = tabular.regression_metrics(0.5 * direct["raw"] + 0.5 * direct["fe"], case["y"].iloc[case["test_idx"]])
    assert {key: row[key] for key in expected} == expected
    assert row["case_seconds"] >= row["pipeline_seconds"]
    assert averaging.METHOD not in tabular.DEFAULT_METHODS
    serialized = json.dumps(row, allow_nan=False)
    assert "PRIVATE-ONE" not in serialized and "PRIVATE-TWO" not in serialized


def test_outer_test_labels_do_not_change_canonical_search_or_plans(
    small_native: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = _payload()
    original = tabular.generate_features
    seen = []

    def features(train, test, target, *args):
        assert train.index.equals(target.index)
        assert set(train.index) <= set(case["X"].iloc[case["train_idx"]].index)
        assert set(train.index).isdisjoint(test.index)
        seen.append(train.index.tolist())
        return original(train, test, target, *args)

    monkeypatch.setattr(tabular, "generate_features", features)
    before = averaging.run_case(case, tmp_path)[0]
    case["y"].iloc[case["test_idx"]] += 1000
    after = averaging.run_case(case, tmp_path)[0]
    assert len(seen) == 12
    assert before["outer_prediction_sha256"] == after["outer_prediction_sha256"]
    assert before["loss"] != after["loss"]
    for role, _ in averaging.COMPONENTS:
        first = before["regression_average"]["components"][role]
        second = after["regression_average"]["components"][role]
        for key in (
            "parameters",
            "inner_losses",
            "outer_feature_names_sha256",
            "inner_selection_report",
            "outer_selection_report",
        ):
            assert first[key] == second[key]
        if role == "fe":
            assert first["inner_pipeline_trials"] == second["inner_pipeline_trials"]
