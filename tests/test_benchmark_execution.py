"""Integration contracts for per-learner scheduling and exact resume."""

import copy
import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from benchmarks import execution, tabular

REAL_SPLITS = execution._splits
REAL_PREPARE = execution._prepare_dataset


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    manifest = tmp_path / "suite.json"
    suite = {
        "name": "test-suite",
        "datasets": [{"name": "test-data", "family_id": "test-family", "role": "development", "task": "regression"}],
    }
    manifest.write_text(json.dumps(suite))
    monkeypatch.setattr(execution, "_load_suite", lambda *a, **k: (suite, False))
    monkeypatch.setattr(tabular, "ROOT", tmp_path)
    monkeypatch.setattr(tabular, "source_fingerprint", lambda: "fixed-source")
    monkeypatch.setattr(tabular, "source_files", lambda: [])
    monkeypatch.setattr(execution, "environment_versions", lambda: {"numpy": "fixed"})
    data = {
        "X": pd.DataFrame({"x0": np.arange(20.0)}),
        "y": pd.Series(np.arange(20.0)),
        "groups": None,
        "timestamps": None,
    }
    monkeypatch.setattr(
        execution, "_prepare_dataset", lambda *a: (data, {"sha256": "fixed-data", "source": "real_world"}, None)
    )
    monkeypatch.setattr(
        execution,
        "_splits",
        lambda *a: [(42, 0, np.arange(10), np.arange(10, 20)), (42, 1, np.arange(10, 20), np.arange(10))],
    )
    output = tmp_path / "run"
    arguments = [
        "--suite",
        str(manifest),
        "--output",
        str(output),
        "--methods",
        "baseline",
        "--learners",
        "linear",
        "lightgbm",
        "--folds",
        "2",
        "--seeds",
        "42",
        "--max-rows",
        "0",
    ]
    return output, arguments


def successful_result(payload, directory, timeout):
    directory.mkdir(parents=True)
    assert len(payload["config"]["learners"]) == 1
    row = tabular.failure_rows(payload, "success", "", 0.1)[0]
    row.pop("error")
    row.update(
        {
            "loss": 1.0,
            "inner_fe_seconds": 0.01,
            "outer_fe_seconds": 0.01,
            "pipeline_seconds": 0.1,
            "n_output_features": 1,
            "n_input_features": 1,
        }
    )
    return [row]


def test_worker_protocol_accepts_zero_loss_and_integral_feature_counts(experiment, monkeypatch):
    output, arguments = experiment

    def run(payload, directory, timeout):
        rows = successful_result(payload, directory, timeout)
        rows[0]["loss"] = 0.0
        return rows

    monkeypatch.setattr(tabular, "run_isolated", run)
    assert tabular.main(arguments) == 0
    rows = json.loads((output / "results.json").read_text())["rows"]
    assert all(row["loss"] == 0.0 and type(row["n_output_features"]) is int for row in rows)


@pytest.fixture
def classification_preparation(tmp_path, monkeypatch):
    from benchmarks import catalog, datasets

    def prepare(codes, metadata, provider):
        X = pd.DataFrame({"x0": np.arange(len(codes), dtype=float)})
        y = pd.Series(codes)
        spec = {"name": "label-fixture", "task": "classification"}
        pairs = [(np.arange(len(codes) // 2), np.arange(len(codes) // 2, len(codes)))]
        if provider == "legacy":
            spec["path"] = "fixture.csv"
            cached = tmp_path / "benchmarks" / ".cache" / ("a" * 40) / spec["path"]
            cached.parent.mkdir(parents=True)
            cached.write_bytes(b"")
            monkeypatch.setattr(tabular, "load_real_dataset", lambda *a: (X, y, metadata))
            pairs = None
        elif provider == "diagnostic":
            spec.update(metadata, provider="diagnostic")
            monkeypatch.setattr(datasets, "load_dataset", lambda *a: (X, y, "classification", {}))
            pairs = None
        else:
            loaded = SimpleNamespace(X=X, y=y, metadata=metadata, groups=None, timestamps=None, official_splits=pairs)
            monkeypatch.setattr(catalog, "load_dataset", lambda *a, **k: loaded)
        result = REAL_PREPARE(
            spec, "synthetic", {"revision": "a" * 40}, provider == "legacy", {"max_rows": 0}, tmp_path
        )
        return result, pairs

    return prepare


@pytest.mark.parametrize("provider", ["catalog", "legacy"])
@pytest.mark.parametrize("labels", [["first", "middle", "last"], [-7, 13, 91]])
@pytest.mark.parametrize("codes", [[2, 0, 1, 2, 1, 0], [2, 0, 2, 0], [2, 1, 2, 1], [2, 2, 2, 2]])
def test_preparation_metadata_maps_only_observed_encoded_classes(classification_preparation, provider, labels, codes):
    original = {"class_labels": labels, "sha256": "fixture-data"}
    before = copy.deepcopy(original)
    (data, metadata, official), pairs = classification_preparation(codes, original, provider)
    encoder = execution.LabelEncoder()
    expected_y = encoder.fit_transform(codes)
    pd.testing.assert_series_equal(data["y"], pd.Series(expected_y, name="target"))
    assert metadata["evaluation_classes"] == [str(labels[code]) for code in encoder.classes_]
    assert metadata["class_labels"] == labels
    assert original == before
    assert official is pairs
    assert set(metadata) == {"class_labels", "sha256", "evaluation_classes", "family_id", "role", "source"}
    assert set(data) == {
        "X",
        "y",
        "groups",
        "timestamps",
        "label_end_times",
        "column_descriptions",
        "target_transform",
    }


@pytest.mark.parametrize("provider", ["catalog", "legacy", "diagnostic"])
@pytest.mark.parametrize("labels", [["yes", "no", "yes", "no"], [20, 10, 20, 10]])
def test_preparation_metadata_without_vocabulary_keeps_raw_label_encoding(classification_preparation, provider, labels):
    (data, metadata, _), _ = classification_preparation(labels, {}, provider)
    encoder = execution.LabelEncoder()
    expected_y = encoder.fit_transform(labels)
    pd.testing.assert_series_equal(data["y"], pd.Series(expected_y, name="target"))
    assert metadata["evaluation_classes"] == [str(label) for label in encoder.classes_]
    assert "class_labels" not in metadata


def test_diagnostic_preparation_metadata_does_not_treat_raw_labels_as_codes(classification_preparation):
    (data, metadata, _), _ = classification_preparation(
        ["yes", "no", "yes", "no"], {"class_labels": ["unused-a", "unused-b"]}, "diagnostic"
    )
    assert data["y"].tolist() == [1, 0, 1, 0]
    assert metadata["evaluation_classes"] == ["no", "yes"]
    assert metadata["class_labels"] == ["unused-a", "unused-b"]


@pytest.mark.parametrize("provider", ["catalog", "legacy"])
@pytest.mark.parametrize(
    "labels",
    [
        None,
        [],
        "no,yes",
        {"0": "no", "1": "yes"},
        ["same", "same"],
        [None, "yes"],
        [{"label": "no"}, "yes"],
        [0, np.nan],
    ],
)
def test_preparation_metadata_rejects_invalid_class_vocabulary(classification_preparation, provider, labels):
    with pytest.raises(ValueError, match="class_labels"):
        classification_preparation([0, 1, 0, 1], {"class_labels": labels}, provider)


@pytest.mark.parametrize("provider", ["catalog", "legacy"])
@pytest.mark.parametrize("codes", [[-1, 0], [0, 2], [0.0, 1.0], [0.5, 1.5], ["0", "1"], [False, True]])
def test_preparation_metadata_rejects_invalid_encoded_class_indices(classification_preparation, provider, codes):
    with pytest.raises(ValueError, match="class_labels"):
        classification_preparation(codes, {"class_labels": ["no", "yes"]}, provider)


def test_each_learner_is_independent_and_completed_cells_are_reused(experiment, monkeypatch):
    output, arguments = experiment
    calls = []

    def run(payload, directory, timeout):
        calls.append(payload["config"]["learners"])
        return successful_result(payload, directory, timeout)

    monkeypatch.setattr(tabular, "run_isolated", run)
    assert tabular.main(arguments) == 0
    assert len(calls) == 4
    assert all(len(learners) == 1 for learners in calls)
    assert tabular.main(["--output", str(output), "--resume"]) == 0
    assert len(calls) == 4
    assert len(json.loads((output / "results.json").read_text())["rows"]) == 4


def test_all_robust_comparison_uses_its_declared_matched_control(experiment, monkeypatch):
    output, arguments = experiment
    start, end = arguments.index("--methods") + 1, arguments.index("--learners")
    arguments[start:end] = [
        "baseline-robust",
        "featcopilot-frequency-aligned-robust",
        "featcopilot-bagged-aligned-robust",
    ]
    monkeypatch.setattr(tabular, "run_isolated", successful_result)
    assert tabular.main(arguments) == 0
    result = json.loads((output / "results.json").read_text())
    assert len(result["rows"]) == 12
    assert all(item["reference_method"] == "baseline-robust" for item in result["summary"])
    assert all(item["complete_paired_families"] == 1 for item in result["summary"])


def test_fixed_plan_comparison_accepts_discovery_matched_raw_control(experiment, monkeypatch):
    output, arguments = experiment
    start, end = arguments.index("--methods") + 1, arguments.index("--learners")
    arguments[start:end] = ["baseline-discovery-robust", "featcopilot-plan-gated-robust"]
    monkeypatch.setattr(tabular, "run_isolated", successful_result)
    assert tabular.main(arguments) == 0
    result = json.loads((output / "results.json").read_text())
    assert len(result["rows"]) == 8
    assert all(item["reference_method"] == "baseline-discovery-robust" for item in result["summary"])
    assert all(item["complete_paired_families"] == 1 for item in result["matched_plan_summary"])


@pytest.mark.parametrize(
    "methods",
    [
        ["featcopilot-bagged-aligned-robust"],
        ["baseline-robust", "featcopilot-frequency-aligned"],
        ["baseline-robust", "openfe-inductive"],
        ["baseline-discovery-robust", "featcopilot-frequency-aligned-robust"],
        ["featcopilot-plan-gated-robust"],
    ],
)
def test_missing_or_mismatched_paired_control_is_rejected_before_acquisition(experiment, monkeypatch, methods):
    output, arguments = experiment
    start, end = arguments.index("--methods") + 1, arguments.index("--learners")
    arguments[start:end] = methods

    def unexpected_acquisition(*args, **kwargs):
        raise AssertionError("Invalid controls must fail before acquisition")

    monkeypatch.setattr(execution, "_prepare_dataset", unexpected_acquisition)
    assert tabular.main(arguments) == 2
    assert not output.exists()


def test_retry_failed_keeps_successful_other_learners(experiment, monkeypatch):
    output, arguments = experiment
    attempts = []

    def first_run(payload, directory, timeout):
        if payload["config"]["learners"] == ["lightgbm"]:
            directory.mkdir(parents=True)
            return tabular.failure_rows(payload, "timeout", "timeout", timeout)
        return successful_result(payload, directory, timeout)

    monkeypatch.setattr(tabular, "run_isolated", first_run)
    assert tabular.main(arguments) == 1

    def retry(payload, directory, timeout):
        attempts.append(payload["config"]["learners"][0])
        return successful_result(payload, directory, timeout)

    monkeypatch.setattr(tabular, "run_isolated", retry)
    assert tabular.main(["--output", str(output), "--resume", "--retry-failed"]) == 0
    assert attempts == ["lightgbm", "lightgbm"]


def test_changed_source_cannot_resume(experiment, monkeypatch):
    output, arguments = experiment
    monkeypatch.setattr(tabular, "run_isolated", successful_result)
    assert tabular.main(arguments) == 0
    monkeypatch.setattr(tabular, "source_fingerprint", lambda: "new-source")
    assert tabular.main(["--output", str(output), "--resume"]) == 2


def test_changed_source_archive_cannot_resume(experiment, monkeypatch):
    output, arguments = experiment
    monkeypatch.setattr(tabular, "run_isolated", successful_result)
    assert tabular.main(arguments) == 0
    (output / "source.zip").write_bytes(b"corrupt")
    assert tabular.main(["--output", str(output), "--resume"]) == 2


def test_changed_budgeted_configuration_cannot_resume(experiment, monkeypatch):
    output, arguments = experiment
    monkeypatch.setattr(tabular, "run_isolated", successful_result)
    assert tabular.main(arguments) == 0
    assert tabular.main(["--output", str(output), "--resume", "--case-timeout", "999"]) == 2


def test_changed_dataset_aborts_before_any_resumed_worker(experiment, monkeypatch):
    output, arguments = experiment
    monkeypatch.setattr(tabular, "run_isolated", successful_result)
    assert tabular.main(arguments) == 0
    original_manifest = (output / "manifest.json").read_bytes()
    original = execution._prepare_dataset

    def changed(*args):
        data, metadata, splits = original(*args)
        return data, {**metadata, "sha256": "different-data"}, splits

    monkeypatch.setattr(execution, "_prepare_dataset", changed)
    monkeypatch.setattr(tabular, "run_isolated", lambda *a: pytest.fail("mismatched dataset reached worker"))
    assert tabular.main(["--output", str(output), "--resume"]) == 2
    assert (output / "manifest.json").read_bytes() == original_manifest


def test_source_change_during_run_stops_new_workers(experiment, monkeypatch):
    output, arguments = experiment
    calls = []

    def changed(payload, directory, timeout):
        calls.append(payload)
        result = successful_result(payload, directory, timeout)
        monkeypatch.setattr(tabular, "source_fingerprint", lambda: "changed")
        return result

    monkeypatch.setattr(tabular, "run_isolated", changed)
    assert tabular.main(arguments) == 1
    assert len(calls) == 1
    assert json.loads((output / "manifest.json").read_text())["source_changed"] is True
    assert tabular.main(["--output", str(output), "--resume"]) == 2


def test_prepare_only_does_not_fit_or_expose_confirmation(experiment, monkeypatch):
    output, arguments = experiment
    monkeypatch.setattr(tabular, "run_isolated", lambda *a: pytest.fail("prepare-only started a model"))
    assert tabular.main([*arguments, "--prepare-only"]) == 0
    assert not (output / "results.json").exists()
    assert json.loads((output / "manifest.json").read_text())["datasets"][0]["sha256"] == "fixed-data"


@pytest.fixture
def controller_clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(execution.time, "perf_counter", lambda: now[0])
    return now


@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("missing_manifest", [False, True])
@pytest.mark.parametrize("prepare_only", [False, True])
def test_source_initialization_failure_is_discoverable_and_resumes_exactly(
    experiment, monkeypatch, controller_clock, interrupted, missing_manifest, prepare_only
):
    output, arguments = experiment
    archive = execution.archive_source

    def fail(*args):
        controller_clock[0] += 60
        raise KeyboardInterrupt if interrupted else OSError("Synthetic source archive failure")

    with monkeypatch.context() as patch:
        patch.setattr(execution, "archive_source", fail)
        patch.setattr(execution, "_prepare_dataset", Mock(side_effect=AssertionError("initialization opened data")))
        assert tabular.main([*arguments, "--total-seconds", "100", *(["--prepare-only"] if prepare_only else [])]) == 1
    state = json.loads((output / "state.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert state["elapsed_seconds"] == manifest["active_seconds"] == 60
    assert manifest["initialization_status"] == ("interrupted" if interrupted else "error")
    assert manifest["source_archive_sha256"] is None
    assert state["records"] == state["attempts"] == state["datasets"] == {}
    if missing_manifest:
        (output / "manifest.json").unlink()
    before = (output / "state.json").read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(tabular, "source_fingerprint", lambda: "changed-source")
        patch.setattr(execution, "archive_source", Mock(side_effect=AssertionError("changed identity rebuilt source")))
        assert tabular.main(["--output", str(output), "--resume"]) == 2
    assert (output / "state.json").read_bytes() == before
    timeouts = []

    def run(payload, directory, timeout):
        timeouts.append(timeout)
        controller_clock[0] += 1
        return successful_result(payload, directory, timeout)

    monkeypatch.setattr(tabular, "run_isolated", run)
    assert tabular.main(["--output", str(output), "--resume"]) == 0
    assert timeouts == [35, 34, 33, 32]
    assert json.loads((output / "state.json").read_text())["elapsed_seconds"] == 64
    assert json.loads((output / "manifest.json").read_text())["initialization_status"] == "ready"


@pytest.mark.parametrize("evidence", ["records", "attempts", "datasets", "artifact"])
def test_source_initialization_recovery_rejects_any_result_evidence(
    experiment, monkeypatch, controller_clock, evidence
):
    output, arguments = experiment
    with monkeypatch.context() as patch:
        patch.setattr(execution, "archive_source", Mock(side_effect=OSError("Synthetic initialization failure")))
        assert tabular.main(arguments) == 1
    path = output / "state.json"
    state = json.loads(path.read_text())
    if evidence == "artifact":
        (output / "results.json").write_text("{}")
    elif evidence == "records":
        (output / "records").mkdir()
    else:
        state[evidence]["test-data"] = 1 if evidence == "attempts" else {"sha256": "measured"}
        path.write_text(json.dumps(state))
    before = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
    monkeypatch.setattr(execution, "archive_source", Mock(side_effect=AssertionError("populated run rebuilt source")))
    assert tabular.main(["--output", str(output), "--resume"]) == 2
    assert {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()} == before


def test_source_initialization_retry_cannot_recover_consumed_budget(experiment, monkeypatch, controller_clock):
    output, arguments = experiment

    def fail(*args):
        controller_clock[0] += 100
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(execution, "archive_source", fail)
        assert tabular.main([*arguments, "--total-seconds", "100"]) == 1
    monkeypatch.setattr(execution, "archive_source", Mock(side_effect=AssertionError("exhausted retry rebuilt source")))
    monkeypatch.setattr(execution, "_prepare_dataset", Mock(side_effect=AssertionError("exhausted retry opened data")))
    assert tabular.main(["--output", str(output), "--resume"]) == 1
    assert json.loads((output / "state.json").read_text())["elapsed_seconds"] == 100
    assert json.loads((output / "manifest.json").read_text())["initialization_status"] == "budget_exhausted"


def assert_protocol_resume_rejected(output, monkeypatch):
    before = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    with monkeypatch.context() as patch:
        patch.setattr(execution, "RunState", Mock(side_effect=AssertionError("terminal resume opened state")))
        patch.setattr(
            execution.time, "perf_counter", Mock(side_effect=AssertionError("terminal resume started timing"))
        )
        patch.setattr(execution, "_prepare_dataset", Mock(side_effect=AssertionError("terminal resume acquired data")))
        for flags in ([], ["--retry-failed"], ["--prepare-only"]):
            assert tabular.main(["--output", str(output), "--resume", *flags]) == 2
    assert {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("document", [b"{", b"null", b"[42]", b"[null]", b"[]", b"\xff"])
def test_real_launcher_malformed_output_is_terminal(experiment, monkeypatch, controller_clock, document):
    import psutil

    output, arguments = experiment
    process = Mock(pid=424242, returncode=0)
    process.poll.return_value = 0

    def launch(command, **kwargs):
        controller_clock[0] += 7
        Path(command[-1]).write_bytes(document)
        return process

    monkeypatch.setattr(tabular.subprocess, "Popen", launch)
    monkeypatch.setattr(psutil, "Process", Mock(side_effect=psutil.NoSuchProcess(process.pid)))
    assert tabular.main([*arguments, "--total-seconds", "100"]) == 2
    manifest = json.loads((output / "manifest.json").read_text())
    state = json.loads((output / "state.json").read_text())
    assert manifest["run_status"] == "protocol_error"
    assert manifest["protocol_error"].startswith("Worker result file")
    assert manifest["active_seconds"] == state["elapsed_seconds"] == 7
    assert state["records"] == {}
    assert not list(output.rglob("input.pkl"))
    assert_protocol_resume_rejected(output, monkeypatch)


@pytest.mark.parametrize("failure", ["monitor", "launch"])
def test_real_supervisor_operational_failure_is_checkpointed_and_retryable(
    experiment, monkeypatch, controller_clock, failure
):
    import psutil

    output, arguments = experiment
    processes = []
    parent = Mock()
    parent.children.return_value = []
    parent.memory_info.side_effect = psutil.AccessDenied(424242)

    def launch(*args, **kwargs):
        controller_clock[0] += 3
        if failure == "launch":
            raise OSError("Synthetic launch failure")
        process = Mock(pid=424242, returncode=None)
        process.poll.side_effect = lambda: process.returncode
        process.kill.side_effect = lambda: setattr(process, "returncode", -9)
        process.wait.side_effect = lambda timeout: process.returncode
        processes.append(process)
        return process

    monkeypatch.setattr(tabular.subprocess, "Popen", launch)
    monkeypatch.setattr(psutil, "Process", Mock(return_value=parent))
    assert tabular.main([*arguments, "--total-seconds", "100"]) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    rows = json.loads((output / "results.json").read_text())["rows"]
    assert manifest["run_status"] == "failed" and manifest["protocol_error"] is None
    assert manifest["active_seconds"] == 12
    assert len(rows) == 4 and all(row["status"] == "error" and "loss" not in row for row in rows)
    assert all(process.kill.call_count == 1 for process in processes)
    assert not list(output.rglob("input.pkl"))
    monkeypatch.setattr(tabular, "run_isolated", successful_result)
    assert tabular.main(["--output", str(output), "--resume", "--retry-failed"]) == 0


@pytest.mark.parametrize("prior_success", [False, True])
@pytest.mark.parametrize(
    "malformed",
    [
        "empty",
        "duplicate",
        "nonlist",
        "nondict",
        "missing_loss",
        "missing_fe_time",
        "missing_feature_count",
        "missing_status",
        "missing_error",
        "nonfinite_loss",
        "nonfinite_diagnostics",
        "negative_runtime",
        "negative_loss",
        "fractional_feature_count",
        "boolean_feature_count",
        "wrong_cell",
    ],
)
def test_worker_protocol_failure_is_terminal_and_preserves_completed_cells(
    experiment, monkeypatch, controller_clock, caplog, prior_success, malformed
):
    output, arguments = experiment
    calls = []
    previous_records = {}

    def run(payload, directory, timeout):
        calls.append(payload)
        controller_clock[0] += 7
        rows = successful_result(payload, directory, timeout)
        if prior_success and len(calls) == 1:
            return rows
        if prior_success:
            saved = json.loads((output / "state.json").read_text())
            previous_records.update(
                {
                    record["file"]: (output / "records" / record["file"]).read_bytes()
                    for record in saved["records"].values()
                }
            )
            assert not (output / "results.json").exists()
        if malformed == "empty":
            return []
        if malformed == "duplicate":
            return rows * 2
        if malformed == "nonlist":
            return rows[0]
        if malformed == "nondict":
            return ["success"]
        if malformed == "missing_loss":
            rows[0].pop("loss")
        elif malformed == "missing_fe_time":
            rows[0].pop("inner_fe_seconds")
        elif malformed == "missing_feature_count":
            rows[0].pop("n_output_features")
        elif malformed == "missing_status":
            rows[0].pop("status")
        elif malformed == "missing_error":
            rows[0]["status"] = "error"
        elif malformed == "nonfinite_loss":
            rows[0]["loss"] = float("nan")
        elif malformed == "nonfinite_diagnostics":
            rows[0]["diagnostics"] = {"value": float("nan")}
        elif malformed == "negative_runtime":
            rows[0]["pipeline_seconds"] = -1
        elif malformed == "negative_loss":
            rows[0]["loss"] = -1.0
        elif malformed == "fractional_feature_count":
            rows[0]["n_output_features"] = 1.5
        elif malformed == "boolean_feature_count":
            rows[0]["n_output_features"] = True
        else:
            rows[0]["learner"] = "unrequested-learner"
        return rows

    monkeypatch.setattr(tabular, "run_isolated", run)
    assert tabular.main([*arguments, "--total-seconds", "100"]) == 2
    state = json.loads((output / "state.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert state["elapsed_seconds"] == manifest["active_seconds"] == 7 * (1 + prior_success)
    assert manifest["run_status"] == "protocol_error"
    assert manifest["protocol_error"]
    assert "protocol failure" in caplog.text
    assert len(state["records"]) == int(prior_success)
    if prior_success:
        assert all((output / "records" / name).read_bytes() == content for name, content in previous_records.items())
        record = next(iter(state["records"].values()))
        row = json.loads((output / "records" / record["file"]).read_text())
        assert row["status"] == "success" and row["loss"] == 1.0
    assert_protocol_resume_rejected(output, monkeypatch)


def test_protocol_failure_on_resume_charges_time_without_replacing_prior_evidence(
    experiment, monkeypatch, controller_clock
):
    output, arguments = experiment

    def failed(payload, directory, timeout):
        directory.mkdir(parents=True)
        controller_clock[0] += 2
        return tabular.failure_rows(payload, "error", "Synthetic worker failure", 2.0)

    monkeypatch.setattr(tabular, "run_isolated", failed)
    assert tabular.main([*arguments, "--total-seconds", "100"]) == 1
    original_state = json.loads((output / "state.json").read_text())
    names = ["results.json", "REPORT.md", "archive.json"]
    names += json.loads((output / "archive.json").read_text())["parts"]
    evidence = {name: (output / name).read_bytes() for name in names}

    def malformed(*args):
        controller_clock[0] += 7
        return []

    monkeypatch.setattr(tabular, "run_isolated", malformed)
    assert tabular.main(["--output", str(output), "--resume", "--retry-failed"]) == 2
    state = json.loads((output / "state.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert state["elapsed_seconds"] == manifest["active_seconds"] == original_state["elapsed_seconds"] + 7
    assert state["records"] == original_state["records"]
    assert all((output / name).read_bytes() == content for name, content in evidence.items())
    assert manifest["run_status"] == "protocol_error"
    assert_protocol_resume_rejected(output, monkeypatch)


def test_protocol_failure_stays_terminal_when_the_budget_also_overruns(experiment, monkeypatch, controller_clock):
    output, arguments = experiment

    def malformed(*args):
        controller_clock[0] += 107
        return []

    monkeypatch.setattr(tabular, "run_isolated", malformed)
    assert tabular.main([*arguments, "--total-seconds", "100"]) == 2
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["run_status"] == "protocol_error"
    assert manifest["budget_overrun_seconds"] == 7
    assert manifest["active_seconds"] == json.loads((output / "state.json").read_text())["elapsed_seconds"] == 107
    assert_protocol_resume_rejected(output, monkeypatch)


@pytest.mark.parametrize("status", ["error", "timeout", "memory_limit"])
def test_legitimate_worker_failures_remain_retryable(experiment, monkeypatch, controller_clock, status):
    output, arguments = experiment

    def failed(payload, directory, timeout):
        directory.mkdir(parents=True)
        return tabular.failure_rows(payload, status, "Synthetic worker failure", 0.0)

    monkeypatch.setattr(tabular, "run_isolated", failed)
    assert tabular.main(arguments) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["run_status"] == "failed"
    assert manifest.get("protocol_error") is None
    monkeypatch.setattr(tabular, "run_isolated", successful_result)
    assert tabular.main(["--output", str(output), "--resume", "--retry-failed"]) == 0


@pytest.mark.parametrize("folds", [2, 5])
def test_cells_checkpoint_independently_but_publish_full_report_only_once(
    experiment, monkeypatch, controller_clock, folds
):
    output, arguments = experiment
    monkeypatch.setattr(execution, "_splits", REAL_SPLITS)
    calls, publications = [], []
    report, archive = tabular.write_report, execution.archive_results
    expected = folds * 2

    def run(payload, directory, timeout):
        state = json.loads((output / "state.json").read_text())
        assert len(state["records"]) == len(calls)
        assert not publications and not (output / "results.json").exists()
        calls.append(payload)
        controller_clock[0] += 1
        return successful_result(payload, directory, timeout)

    def render(directory, manifest, rows):
        assert len(rows) == len(json.loads((output / "state.json").read_text())["records"]) == expected
        publications.append("report")
        report(directory, manifest, rows)
        controller_clock[0] += 2

    def compress(path):
        publications.append("archive")
        archive(path)
        controller_clock[0] += 3

    monkeypatch.setattr(tabular, "run_isolated", run)
    monkeypatch.setattr(tabular, "write_report", render)
    monkeypatch.setattr(execution, "archive_results", compress)
    assert tabular.main([*arguments, "--folds", str(folds), "--total-seconds", "100"]) == 0
    assert len(calls) == expected
    assert publications == ["report", "archive"]
    assert json.loads((output / "state.json").read_text())["elapsed_seconds"] == expected + 5
    assert json.loads((output / "manifest.json").read_text())["active_seconds"] == expected + 5


def test_resume_uses_cell_checkpoint_when_no_report_was_published(experiment, monkeypatch, controller_clock):
    output, arguments = experiment
    calls, reports = [], []
    report = tabular.write_report

    def interrupted_run(payload, directory, timeout):
        calls.append(payload)
        controller_clock[0] += 1
        if len(calls) == 2:
            raise SystemExit("Synthetic controller termination")
        return successful_result(payload, directory, timeout)

    def render(*args):
        reports.append(None)
        report(*args)
        controller_clock[0] += 2

    monkeypatch.setattr(tabular, "run_isolated", interrupted_run)
    monkeypatch.setattr(tabular, "write_report", render)
    with pytest.raises(SystemExit, match="Synthetic controller termination"):
        tabular.main([*arguments, "--total-seconds", "100"])
    checkpoint = json.loads((output / "state.json").read_text())
    assert checkpoint["elapsed_seconds"] == 2
    assert len(checkpoint["records"]) == 1
    assert not reports and not (output / "results.json").exists()
    cell_id, record = next(iter(checkpoint["records"].items()))
    original = (output / "records" / record["file"]).read_bytes()
    resumed = []

    def run(payload, directory, timeout):
        resumed.append(payload)
        controller_clock[0] += 1
        return successful_result(payload, directory, timeout)

    monkeypatch.setattr(tabular, "run_isolated", run)
    assert tabular.main(["--output", str(output), "--resume"]) == 0
    state = json.loads((output / "state.json").read_text())
    assert len(resumed) == 3 and len(reports) == 1
    assert state["records"][cell_id] == record
    assert (output / "records" / record["file"]).read_bytes() == original
    assert state["elapsed_seconds"] == controller_clock[0] == 7


@pytest.mark.parametrize("budget", [15, 100])
def test_final_publication_is_charged_and_overruns_are_explicit(experiment, monkeypatch, controller_clock, budget):
    output, arguments = experiment
    arguments[arguments.index("--learners") + 1 : arguments.index("--folds")] = ["linear"]
    prepare, report, archive = execution._prepare_dataset, tabular.write_report, execution.archive_results
    timeouts = []
    publications = []

    def prepare_with_time(*args):
        controller_clock[0] += 1
        return prepare(*args)

    def run(payload, directory, timeout):
        timeouts.append(timeout)
        controller_clock[0] += 1
        return successful_result(payload, directory, timeout)

    def render(*args):
        publications.append("report")
        report(*args)
        controller_clock[0] += 10

    def compress(*args):
        publications.append("archive")
        archive(*args)
        controller_clock[0] += 7

    monkeypatch.setattr(execution, "_prepare_dataset", prepare_with_time)
    monkeypatch.setattr(tabular, "run_isolated", run)
    monkeypatch.setattr(tabular, "write_report", render)
    monkeypatch.setattr(execution, "archive_results", compress)
    assert tabular.main([*arguments, "--total-seconds", str(budget)]) == int(budget == 15)
    state = json.loads((output / "state.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert state["elapsed_seconds"] == manifest["active_seconds"] == controller_clock[0] == 20
    assert publications == ["report", "archive"]
    assert 0 < manifest["publication_reserve_seconds"] <= budget
    assert timeouts[0] < budget - 1
    assert manifest["budget_overrun_seconds"] == max(0, controller_clock[0] - budget)
    assert manifest["run_status"] == ("budget_overrun" if budget == 15 else "completed")
    monkeypatch.setattr(tabular, "write_report", lambda *a: pytest.fail("unchanged evidence was republished"))
    monkeypatch.setattr(execution, "archive_results", lambda *a: pytest.fail("unchanged archive was replaced"))
    assert tabular.main(["--output", str(output), "--resume"]) == int(budget == 15)


@pytest.mark.parametrize("phase", ["report", "archive"])
def test_interrupted_publication_is_charged_and_can_finish_without_rerunning_cells(
    experiment, monkeypatch, controller_clock, phase
):
    output, arguments = experiment
    report, archive = tabular.write_report, execution.archive_results
    report_calls = []

    def render(*args):
        report_calls.append(None)
        assert len(args[2]) == 4
        report(*args)
        if phase == "report":
            controller_clock[0] += 7
            raise KeyboardInterrupt

    def compress(*args):
        archive(*args)
        if phase == "archive":
            controller_clock[0] += 7
            raise KeyboardInterrupt

    monkeypatch.setattr(tabular, "run_isolated", successful_result)
    with monkeypatch.context() as patch:
        patch.setattr(tabular, "write_report", render)
        patch.setattr(execution, "archive_results", compress)
        assert tabular.main([*arguments, "--total-seconds", "100"]) == 1
    state = json.loads((output / "state.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert state["elapsed_seconds"] == manifest["active_seconds"] == 7
    assert manifest["publication_pending"] is True
    assert manifest["run_status"] == "interrupted"
    assert len(report_calls) == 1
    monkeypatch.setattr(tabular, "run_isolated", lambda *a: pytest.fail("publication retry reran a completed model"))
    assert tabular.main(["--output", str(output), "--resume"]) == 0
    assert json.loads((output / "state.json").read_text())["records"] == state["records"]
    assert json.loads((output / "manifest.json").read_text())["publication_pending"] is False


def test_failed_publication_does_not_claim_completion(experiment, monkeypatch, controller_clock):
    output, arguments = experiment
    monkeypatch.setattr(tabular, "run_isolated", successful_result)

    def failed(*args):
        controller_clock[0] += 7
        raise ValueError("Synthetic archive failure")

    monkeypatch.setattr(execution, "archive_results", failed)
    assert tabular.main(arguments) == 2
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["active_seconds"] == 7
    assert manifest["run_status"] == "publication_error"
    assert manifest["publication_pending"] is True
    assert manifest["publication_error"] == "ValueError: Synthetic archive failure"


@pytest.mark.parametrize("budget", [30, 100])
def test_prepare_only_finalization_is_part_of_the_budget(experiment, monkeypatch, controller_clock, budget):
    output, arguments = experiment
    prepare = execution._prepare_dataset
    fingerprints = []

    def prepare_with_time(*args):
        controller_clock[0] += 20
        return prepare(*args)

    def fingerprint():
        if fingerprints:
            controller_clock[0] += 15
        fingerprints.append(None)
        return "fixed-source"

    monkeypatch.setattr(execution, "_prepare_dataset", prepare_with_time)
    monkeypatch.setattr(tabular, "source_fingerprint", fingerprint)
    monkeypatch.setattr(tabular, "run_isolated", lambda *a: pytest.fail("prepare-only ran a model"))
    assert tabular.main([*arguments, "--prepare-only", "--total-seconds", str(budget)]) == int(budget == 30)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["active_seconds"] == json.loads((output / "state.json").read_text())["elapsed_seconds"] == 35
    assert manifest["run_status"] == ("budget_overrun" if budget == 30 else "prepared")
    assert not (output / "results.json").exists()


@pytest.mark.parametrize("prepare_only", [False, True])
@pytest.mark.parametrize("outcome", ["failed", "interrupted", "exhausted"])
def test_official_development_fallback_uses_declared_dimensions(
    experiment, monkeypatch, controller_clock, outcome, prepare_only
):
    output, arguments = experiment
    suite, _ = execution._load_suite(None, False)
    suite["datasets"][0].update(
        split_policy="official", source={"official_split_dimensions": {"repeats": 2, "folds": 3, "samples": 1}}
    )

    def unavailable(*args):
        if outcome == "interrupted":
            controller_clock[0] += 10
            raise KeyboardInterrupt
        assert outcome == "failed", "An exhausted run must not acquire data"
        raise ValueError("Synthetic preparation failure")

    if outcome == "exhausted":
        initialize = execution.ExposureLedger.__init__

        def exhaust(*args):
            initialize(*args)
            controller_clock[0] += 100

        monkeypatch.setattr(execution.ExposureLedger, "__init__", exhaust)
    monkeypatch.setattr(execution, "_prepare_dataset", unavailable)
    monkeypatch.setattr(tabular, "run_isolated", lambda *a: pytest.fail("unavailable data reached a model"))
    assert (
        tabular.main(
            [
                *arguments,
                "--total-seconds",
                "100",
                "--methods",
                "baseline",
                "featcopilot-predictive",
                "--seeds",
                "42",
                "137",
                *(["--prepare-only"] if prepare_only else []),
            ]
        )
        == 1
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["planned_cells"] == 48
    assert len(manifest["splits"]) == 12
    state = json.loads((output / "state.json").read_text())
    rows = [json.loads((output / "records" / record["file"]).read_text()) for record in state["records"].values()]
    assert len(rows) == (0 if prepare_only and outcome == "failed" else 48)
    assert all(row["expected_cells"] == 12 and "loss" not in row for row in rows)


def test_unknown_official_dimensions_do_not_claim_an_iid_grid(experiment, monkeypatch, controller_clock):
    output, arguments = experiment
    suite, _ = execution._load_suite(None, False)
    suite["datasets"][0].update(split_policy="official", source={})
    prepare = execution._prepare_dataset
    monkeypatch.setattr(execution, "_prepare_dataset", Mock(side_effect=ValueError("Unavailable official payload")))
    assert tabular.main(arguments) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["planned_cells"] is None
    assert manifest["datasets"][0]["split_count_known"] is False
    assert manifest["splits"] == []

    def available(*args):
        data, metadata, _ = prepare(*args)
        return data, metadata, [(np.arange(10), np.arange(10, 20))] * 6

    monkeypatch.setattr(execution, "_prepare_dataset", available)
    monkeypatch.setattr(execution, "_splits", REAL_SPLITS)
    monkeypatch.setattr(tabular, "run_isolated", successful_result)
    assert tabular.main(["--output", str(output), "--resume"]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["planned_cells"] == 12
    assert len(manifest["splits"]) == 6


@pytest.mark.parametrize("prepare_only", [False, True])
@pytest.mark.parametrize("outcome", ["unchanged", "interrupted", "exhausted"])
def test_resume_without_new_evidence_preserves_prepared_provenance_and_archives(
    experiment, monkeypatch, controller_clock, prepare_only, outcome
):
    output, arguments = experiment
    monkeypatch.setattr(tabular, "run_isolated", successful_result)
    if outcome == "exhausted":
        archive = execution.archive_results

        def consume_budget(*args):
            archive(*args)
            controller_clock[0] += 100

        monkeypatch.setattr(execution, "archive_results", consume_budget)
    assert tabular.main([*arguments, "--total-seconds", "100"]) == 0
    original_manifest = json.loads((output / "manifest.json").read_text())
    original_state = json.loads((output / "state.json").read_text())
    names = ["results.json", "REPORT.md", "archive.json"]
    names += json.loads((output / "archive.json").read_text())["parts"]
    evidence = {name: (output / name).read_bytes() for name in names}
    monkeypatch.setattr(tabular, "run_isolated", lambda *a: pytest.fail("completed cells were rerun"))
    monkeypatch.setattr(tabular, "write_report", lambda *a: pytest.fail("unchanged evidence was republished"))
    monkeypatch.setattr(execution, "archive_results", lambda *a: pytest.fail("unchanged archive was replaced"))
    if outcome == "interrupted":

        def interrupt(*args):
            controller_clock[0] += 7
            raise KeyboardInterrupt

        monkeypatch.setattr(execution, "_prepare_dataset", interrupt)
    if outcome == "exhausted":
        monkeypatch.setattr(execution, "_prepare_dataset", lambda *a: pytest.fail("exhausted run acquired data"))
    assert tabular.main(["--output", str(output), "--resume", *(["--prepare-only"] if prepare_only else [])]) == int(
        outcome != "unchanged"
    )
    manifest = json.loads((output / "manifest.json").read_text())
    state = json.loads((output / "state.json").read_text())
    assert manifest["datasets"] == original_manifest["datasets"]
    assert manifest["splits"] == original_manifest["splits"]
    assert state["records"] == original_state["records"]
    assert state["datasets"] == original_state["datasets"]
    assert all((output / name).read_bytes() == content for name, content in evidence.items())
    assert (
        manifest["run_status"]
        == {
            "unchanged": "prepared" if prepare_only else "completed",
            "interrupted": "interrupted",
            "exhausted": "budget_exhausted",
        }[outcome]
    )


@pytest.mark.parametrize("prepare_only", [False, True])
@pytest.mark.parametrize("stop", ["exhausted", "interrupted"])
@pytest.mark.parametrize("mutation", ["metadata", "split", "missing_split", "missing_dataset", "source_archive"])
def test_retained_input_drift_is_rejected_before_any_resume_write(
    experiment, monkeypatch, controller_clock, caplog, prepare_only, stop, mutation
):
    output, arguments = experiment
    monkeypatch.setattr(tabular, "run_isolated", successful_result)
    assert tabular.main([*arguments, "--total-seconds", "100"]) == 0
    state = json.loads((output / "state.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    if mutation == "metadata":
        state["datasets"]["test-data"]["sha256"] = "tampered-dataset"
    elif mutation == "split":
        manifest["splits"][0]["sha256"] = "tampered-split"
    elif mutation == "missing_split":
        manifest["splits"].pop()
    elif mutation == "missing_dataset":
        state["datasets"].pop("test-data")
    else:
        (output / "source.zip").write_bytes(b"corrupt")
    if stop == "exhausted":
        state["elapsed_seconds"] = 100
    (output / "state.json").write_text(json.dumps(state))
    (output / "manifest.json").write_text(json.dumps(manifest))
    before = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    prepare = Mock(side_effect=KeyboardInterrupt if stop == "interrupted" else AssertionError("No budget for data"))
    monkeypatch.setattr(execution, "_prepare_dataset", prepare)
    monkeypatch.setattr(execution.RunState, "save", Mock(side_effect=AssertionError("Rejected resume rewrote state")))
    monkeypatch.setattr(tabular, "write_report", Mock(side_effect=AssertionError("Rejected resume rewrote evidence")))
    monkeypatch.setattr(
        execution, "archive_results", Mock(side_effect=AssertionError("Rejected resume archived scores"))
    )
    monkeypatch.setattr(execution, "archive_source", Mock(side_effect=AssertionError("Rejected resume rebuilt source")))
    assert tabular.main(["--output", str(output), "--resume", *(["--prepare-only"] if prepare_only else [])]) == 2
    assert "Cannot resume:" in caplog.text
    prepare.assert_not_called()
    assert {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("mutation", ["metadata", "split"])
def test_retained_unfinished_input_drift_is_rejected_even_with_retry_flag(
    experiment, monkeypatch, controller_clock, mutation
):
    output, arguments = experiment
    monkeypatch.setattr(tabular, "run_isolated", Mock(side_effect=KeyboardInterrupt))
    assert tabular.main([*arguments, "--total-seconds", "100"]) == 1
    state = json.loads((output / "state.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    state["elapsed_seconds"] = 100
    if mutation == "metadata":
        state["datasets"]["test-data"]["sha256"] = "tampered"
    else:
        manifest["splits"][0]["sha256"] = "tampered"
    (output / "state.json").write_text(json.dumps(state))
    (output / "manifest.json").write_text(json.dumps(manifest))
    before = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    monkeypatch.setattr(execution.RunState, "save", Mock(side_effect=AssertionError("Rejected resume rewrote state")))
    monkeypatch.setattr(execution, "_prepare_dataset", Mock(side_effect=AssertionError("Exhausted resume opened data")))
    assert tabular.main(["--output", str(output), "--resume", "--retry-failed"]) == 2
    assert {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()} == before


def test_interrupt_after_success_checkpoint_never_downgrades_it(experiment, monkeypatch, controller_clock):
    output, arguments = experiment
    record = execution.RunState.record
    completed = {}

    def checkpoint_then_interrupt(state, cell_id, cell_hash, row):
        record(state, cell_id, cell_hash, row)
        if row["status"] == "success" and not completed:
            completed.update(cell_id=cell_id, record=state.records[cell_id], row=state.rows[cell_id].copy())
            controller_clock[0] += 10
            raise KeyboardInterrupt

    monkeypatch.setattr(tabular, "run_isolated", successful_result)
    with monkeypatch.context() as patch:
        patch.setattr(execution.RunState, "record", checkpoint_then_interrupt)
        assert tabular.main(arguments) == 1
    state = json.loads((output / "state.json").read_text())
    assert state["records"][completed["cell_id"]] == completed["record"]
    rows = json.loads((output / "results.json").read_text())["rows"]
    assert next(row for row in rows if row["cell_id"] == completed["cell_id"]) == completed["row"]
    assert sum(row["status"] == "success" for row in rows) == 1
    assert tabular.main(["--output", str(output), "--resume"]) == 0


@pytest.mark.parametrize("prepare_only", [False, True])
@pytest.mark.parametrize("change", ["dataset", "split"])
def test_rejected_resume_leaves_durable_checkpoint_untouched(
    experiment, monkeypatch, controller_clock, prepare_only, change
):
    output, arguments = experiment
    monkeypatch.setattr(tabular, "run_isolated", successful_result)
    assert tabular.main(arguments) == 0
    original = {name: (output / name).read_bytes() for name in ("manifest.json", "state.json", "results.json")}
    prepare = execution._prepare_dataset
    splits = execution._splits

    def changed_prepare(*args):
        controller_clock[0] += 7
        data, metadata, official = prepare(*args)
        return data, {**metadata, "sha256": "changed"} if change == "dataset" else metadata, official

    monkeypatch.setattr(execution, "_prepare_dataset", changed_prepare)
    if change == "split":
        monkeypatch.setattr(
            execution, "_splits", lambda *args: [(seed, fold, test, train) for seed, fold, train, test in splits(*args)]
        )
    monkeypatch.setattr(execution.RunState, "save", lambda *a: pytest.fail("rejected resume rewrote state"))
    monkeypatch.setattr(tabular, "run_isolated", lambda *a: pytest.fail("rejected inputs reached a model"))
    assert tabular.main(["--output", str(output), "--resume", *(["--prepare-only"] if prepare_only else [])]) == 2
    assert all((output / name).read_bytes() == content for name, content in original.items())


@pytest.mark.parametrize("prepare_only", [False, True])
def test_nonidentity_regression_average_is_rejected_before_acquisition(experiment, monkeypatch, prepare_only):
    output, arguments = experiment
    suite, _ = execution._load_suite(None, False)
    suite["datasets"][0]["target_transform"] = "log1p"
    monkeypatch.setattr(execution, "_prepare_dataset", lambda *a: pytest.fail("invalid target policy acquired data"))
    monkeypatch.setattr(tabular, "run_isolated", lambda *a: pytest.fail("invalid target policy ran models"))
    assert (
        tabular.main(
            [
                *arguments,
                "--methods",
                "baseline-robust",
                "featcopilot-regression-average-robust",
                *(["--prepare-only"] if prepare_only else []),
            ]
        )
        == 2
    )
    assert not output.exists()


def test_catalog_target_transform_survives_hydration_and_worker_validation(experiment, monkeypatch):
    from benchmarks import catalog

    output, arguments = experiment
    data_path = output.parent / "synthetic.csv"
    pd.DataFrame({"measurement": np.arange(20.0), "y": np.arange(20.0)}).to_csv(data_path, index=False)
    family = {
        "family_id": "test-family",
        "aliases": ["fixture-alias"],
        "source_ids": [],
        "upstream_origin": "Synthetic unit-test fixture",
        "prior_exposure": "metadata_only",
        "role": "development",
    }
    spec = {
        "name": "test-data",
        "family_id": "test-family",
        "provider": "local_csv",
        "task": "regression",
        "role": "development",
        "prior_exposure": "metadata_only",
        "origin_kind": "synthetic_diagnostic",
        "target": "y",
        "target_transform": "log1p",
        "categorical": [],
        "drop_columns": [],
        "split_policy": "iid",
        "prediction_time": "Synthetic measurements before fixture outcome",
        "rights": {
            "status": "approved",
            "license": "CC0-1.0",
            "access": "local_only",
            "approval_basis": "Locally generated unit-test fixture",
            "evidence": ["tests/test_benchmark_execution.py"],
        },
        "source": {
            "path": str(data_path),
            "sha256": tabular.digest(data_path.read_bytes()),
            "schema_reviewed": True,
            "columns": ["measurement", "y"],
            "target_columns": ["y"],
        },
    }
    suite = catalog.validate_manifest({"name": "local-test", "families": [family], "datasets": [spec]})
    monkeypatch.setattr(execution, "_load_suite", lambda *a, **k: (suite, False))
    monkeypatch.setattr(execution, "_prepare_dataset", REAL_PREPARE)
    monkeypatch.setattr(catalog, "_fetch_bytes", lambda *a, **k: pytest.fail("local test attempted a download"))
    calls = []

    def run(payload, directory, timeout):
        assert payload["target_transform"] == "log1p"
        # Hydration must restore the authoritative policy even if the dispatch field is missing.
        candidate = {**payload, "method": "featcopilot-regression-average-robust"}
        candidate.pop("target_transform")
        with pytest.raises(ValueError, match="identity target transform"):
            tabular.run_case(candidate, directory)
        calls.append(payload)
        return successful_result(payload, directory, timeout)

    monkeypatch.setattr(tabular, "run_isolated", run)
    assert tabular.main(arguments) == 0
    assert len(calls) == 4


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("prepare_only", [False, True])
@pytest.mark.parametrize("phase", ["controller_start", "before", "acquisition", "splitting", "serialization"])
def test_preparation_interrupt_is_durable_and_resumes_with_remaining_budget(
    experiment, monkeypatch, controller_clock, resume, prepare_only, phase
):
    output, arguments = experiment
    arguments += ["--total-seconds", "100"]
    prepare = execution._prepare_dataset
    if resume:

        def initial_prepare(*args):
            controller_clock[0] += 10
            return prepare(*args)

        with monkeypatch.context() as patch:
            patch.setattr(execution, "_prepare_dataset", initial_prepare)
            assert tabular.main([*arguments, "--prepare-only"]) == 0
        arguments = ["--output", str(output), "--resume"]
    prior = controller_clock[0]

    def interrupt(*args, **kwargs):
        controller_clock[0] += 60
        raise KeyboardInterrupt

    target, attribute = {
        "controller_start": (execution.ExposureLedger, "__init__"),
        "before": (execution.ExposureLedger, "check"),
        "acquisition": (execution, "_prepare_dataset"),
        "splitting": (execution, "_splits"),
        "serialization": (execution.pickle, "dumps"),
    }[phase]
    with monkeypatch.context() as patch:
        patch.setattr(target, attribute, interrupt)
        patch.setattr(tabular, "run_isolated", lambda *a: pytest.fail("interruption started a worker"))
        assert tabular.main([*arguments, *(["--prepare-only"] if prepare_only else [])]) == 1
    state = json.loads((output / "state.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    rows = [json.loads((output / "records" / record["file"]).read_text()) for record in state["records"].values()]
    assert state["elapsed_seconds"] == manifest["active_seconds"] == prior + 60
    assert manifest["planned_cells"] == len(rows) == 4
    assert len(manifest["datasets"]) == 1
    assert len(manifest["splits"]) == 2
    if phase in {"splitting", "serialization"} or resume:
        assert manifest["datasets"][0]["sha256"] == "fixed-data"
    assert all(row["status"] == "interrupted" and "loss" not in row for row in rows)
    assert {(row["learner"], row["fold"]) for row in rows} == {
        (learner, fold) for learner in ("linear", "lightgbm") for fold in range(2)
    }
    if prepare_only:
        assert not (output / "results.json").exists()
    timeouts = []

    def run(payload, directory, timeout):
        timeouts.append(timeout)
        controller_clock[0] += 1
        return successful_result(payload, directory, timeout)

    monkeypatch.setattr(tabular, "run_isolated", run)
    assert tabular.main(["--output", str(output), "--resume"]) == 0
    assert timeouts == [35 - prior - index for index in range(4)]
    assert json.loads((output / "state.json").read_text())["elapsed_seconds"] == prior + 64


def test_interrupted_resume_preserves_prior_successes_and_prepared_identity(experiment, monkeypatch, controller_clock):
    output, arguments = experiment
    calls = []

    def first_run(payload, directory, timeout):
        calls.append(payload)
        controller_clock[0] += 5
        if len(calls) == 2:
            raise KeyboardInterrupt
        return successful_result(payload, directory, timeout)

    monkeypatch.setattr(tabular, "run_isolated", first_run)
    assert tabular.main([*arguments, "--total-seconds", "100"]) == 1
    original_state = json.loads((output / "state.json").read_text())
    original_rows = json.loads((output / "results.json").read_text())["rows"]
    success = next(row for row in original_rows if row["status"] == "success")
    prepare = execution._prepare_dataset

    def interrupt(*args):
        controller_clock[0] += 60
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(execution, "_prepare_dataset", interrupt)
        assert tabular.main(["--output", str(output), "--resume"]) == 1
    state = json.loads((output / "state.json").read_text())
    rows = json.loads((output / "results.json").read_text())["rows"]
    assert state["elapsed_seconds"] == 70
    assert state["datasets"] == original_state["datasets"]
    assert next(row for row in rows if row["status"] == "success") == success
    assert state["records"][success["cell_id"]] == original_state["records"][success["cell_id"]]
    assert len(calls) == 2

    def changed(*args):
        data, metadata, splits = prepare(*args)
        return data, {**metadata, "sha256": "different-data"}, splits

    with monkeypatch.context() as patch:
        patch.setattr(execution, "_prepare_dataset", changed)
        patch.setattr(tabular, "run_isolated", lambda *a: pytest.fail("changed data reached a worker"))
        assert tabular.main(["--output", str(output), "--resume"]) == 2
    timeouts = []

    def continuation(payload, directory, timeout):
        timeouts.append(timeout)
        return successful_result(payload, directory, timeout)

    monkeypatch.setattr(tabular, "run_isolated", continuation)
    assert tabular.main(["--output", str(output), "--resume"]) == 0
    assert timeouts == [25] * 3
    assert (
        next(
            row
            for row in json.loads((output / "results.json").read_text())["rows"]
            if row["cell_id"] == success["cell_id"]
        )
        == success
    )


def test_interrupted_retry_remains_unfinished_without_another_retry_flag(experiment, monkeypatch, controller_clock):
    output, arguments = experiment

    def failed(payload, directory, timeout):
        directory.mkdir(parents=True)
        return tabular.failure_rows(payload, "timeout", "worker timeout", 0.0)

    monkeypatch.setattr(tabular, "run_isolated", failed)
    assert tabular.main([*arguments, "--total-seconds", "100"]) == 1

    def interrupt(*args):
        controller_clock[0] += 60
        raise KeyboardInterrupt

    monkeypatch.setattr(tabular, "run_isolated", interrupt)
    assert tabular.main(["--output", str(output), "--resume", "--retry-failed"]) == 1
    rows = json.loads((output / "results.json").read_text())["rows"]
    assert sum(row["status"] == "interrupted" for row in rows) == 1
    calls = []

    def run(payload, directory, timeout):
        calls.append(timeout)
        return successful_result(payload, directory, timeout)

    monkeypatch.setattr(tabular, "run_isolated", run)
    assert tabular.main(["--output", str(output), "--resume"]) == 1
    assert calls == [35]


def test_interrupt_between_preparation_and_dispatch_is_resumable(experiment, monkeypatch, controller_clock):
    output, arguments = experiment

    def interrupt(*args):
        controller_clock[0] += 60
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(execution.RunState, "next_attempt", interrupt)
        patch.setattr(tabular, "run_isolated", lambda *a: pytest.fail("interruption started a worker"))
        assert tabular.main([*arguments, "--total-seconds", "100"]) == 1
    state = json.loads((output / "state.json").read_text())
    rows = json.loads((output / "results.json").read_text())["rows"]
    assert state["elapsed_seconds"] == 60
    assert len(rows) == 4
    assert all(row["status"] == "interrupted" and row["split_sha256"] != "unavailable" for row in rows)
    splits = execution._splits
    with monkeypatch.context() as patch:
        patch.setattr(
            execution, "_splits", lambda *args: [(seed, fold, test, train) for seed, fold, train, test in splits(*args)]
        )
        patch.setattr(tabular, "run_isolated", lambda *a: pytest.fail("changed split reached a worker"))
        assert tabular.main(["--output", str(output), "--resume"]) == 2
    monkeypatch.setattr(tabular, "run_isolated", successful_result)
    assert tabular.main(["--output", str(output), "--resume"]) == 0


@pytest.mark.parametrize("prepare_only", [False, True])
def test_preparation_budget_exhaustion_is_unfinished_and_never_reacquires_without_time(
    experiment, monkeypatch, controller_clock, prepare_only
):
    output, arguments = experiment
    suite, _ = execution._load_suite(None, False)
    suite["datasets"].append(
        {"name": "second-data", "family_id": "second-family", "role": "development", "task": "regression"}
    )
    prepare = execution._prepare_dataset
    calls = []

    def slow_prepare(spec, *args):
        calls.append(spec["name"])
        controller_clock[0] += 100
        return prepare(spec, *args)

    monkeypatch.setattr(execution, "_prepare_dataset", slow_prepare)
    monkeypatch.setattr(tabular, "run_isolated", lambda *a: pytest.fail("exhausted budget started a worker"))
    assert tabular.main([*arguments, "--total-seconds", "100", *(["--prepare-only"] if prepare_only else [])]) == 1
    state = json.loads((output / "state.json").read_text())
    rows = [json.loads((output / "records" / record["file"]).read_text()) for record in state["records"].values()]
    assert state["elapsed_seconds"] == 100
    assert len(rows) == 8
    assert all(row["status"] == "budget_exhausted" and "loss" not in row for row in rows)
    assert all(row["preparation_state"] == "unprepared" for row in rows if row["dataset"] == "second-data")
    assert tabular.main(["--output", str(output), "--resume"]) == 1
    assert calls == ["test-data"]
    assert json.loads((output / "state.json").read_text())["elapsed_seconds"] == 100


def test_summary_averages_variants_within_family_and_keeps_roles_separate():
    rows = []
    for name, family, role, loss in (
        ("variant-a", "shared", "development", 0.8),
        ("variant-b", "shared", "development", 1.0),
        ("heldout", "fresh", "confirmation", 0.5),
    ):
        for fold in range(2):
            for method in ("baseline", "new"):
                rows.append(
                    {
                        "dataset": name,
                        "family_id": family,
                        "role": role,
                        "source": "real_world",
                        "task": "regression",
                        "learner": "linear",
                        "method": method,
                        "seed": 42,
                        "fold": fold,
                        "split_sha256": str(fold),
                        "status": "success",
                        "loss": 1.0 if method == "baseline" else loss,
                        "pipeline_seconds": 1.0,
                        "expected_cells": 2,
                    }
                )
    summary = tabular.summarize(rows, 2, 1)
    dev = next(row for row in summary if row["role"] == "development" and row["method"] == "new")
    confirm = next(row for row in summary if row["role"] == "confirmation" and row["method"] == "new")
    assert dev["complete_paired_datasets"] == 2
    assert dev["complete_paired_families"] == 1
    assert dev["mean_loss_reduction"] == pytest.approx(0.1)
    assert confirm["mean_loss_reduction"] == pytest.approx(0.5)


def test_split_fingerprint_includes_train_test_boundary():
    first = execution.split_fingerprint(np.array([1, 2]), np.array([3, 4]))
    second = execution.split_fingerprint(np.array([1]), np.array([2, 3, 4]))
    assert first != second


def registration():
    return {
        "status": "locked_before_confirmation_model_execution",
        "family_ids": ["fresh"],
        "selection_parameters": {
            "stability_repeats": 3,
            "screening": "auto",
            "max_candidates": 1000,
            "max_added_features": 50,
        },
        "methods": ["baseline", "featcopilot-predictive"],
        "learners": ["linear"],
        "learner_search": {"linear": {"regularization": [0.1, 1.0, 10.0], "selection_estimator_regularization": 1.0}},
        "seeds": [2026],
        "sample_seed": 2026,
        "max_rows": 0,
        "threads": 8,
        "case_timeout_seconds": 600,
        "total_active_seconds_ceiling": 129600,
        "expected_official_split_dimensions_per_task": {"repeats": 1, "folds": 2, "samples": 1},
        "classification_admission_margin": {"min_total": 4, "min_outer_train": 2, "min_outer_test": 2},
    }


def test_locked_confirmation_rejects_configuration_or_cohort_changes():
    registered = registration()
    suite = {"manifests": [{"preregistration": registered}]}
    specs = [{"family_id": "fresh", "role": "confirmation"}]
    config = {
        **registered["selection_parameters"],
        "methods": registered["methods"],
        "learners": registered["learners"],
        "seeds": [2026],
        "max_rows": 0,
        "threads": 8,
        "case_timeout": 600,
        "synthetic": False,
    }
    assert execution.bind_preregistration(suite, specs, config, 100) is registered
    with pytest.raises(ValueError, match="setting"):
        execution.bind_preregistration(suite, specs, {**config, "stability_repeats": 1}, 100)
    with pytest.raises(ValueError, match="families"):
        execution.bind_preregistration(suite, [{"family_id": "different", "role": "confirmation"}], config, 100)
    with pytest.raises(ValueError, match="ceiling"):
        execution.bind_preregistration(suite, specs, config, 129601)


def test_confirmation_checks_official_split_identity_and_class_support():
    registered = registration()
    data = {"y": pd.Series([0, 0, 1, 1, 0, 0, 1, 1])}
    pairs = [(np.arange(4), np.arange(4, 8)), (np.arange(4, 8), np.arange(4))]
    meta = {"official_split_ids": [{"repeat": 0, "fold": fold, "sample": 0} for fold in range(2)]}
    execution.validate_confirmation_payload({"task": "classification"}, data, meta, pairs, registered)
    with pytest.raises(ValueError, match="identifiers"):
        execution.validate_confirmation_payload(
            {"task": "classification"}, data, {"official_split_ids": []}, pairs, registered
        )
    data["y"] = pd.Series([0, 0, 0, 1, 0, 0, 0, 1])
    with pytest.raises(ValueError, match="class support"):
        execution.validate_confirmation_payload({"task": "classification"}, data, meta, pairs, registered)


def test_official_splits_with_unavailable_labels_are_rejected_not_rewritten():
    data = {
        "X": pd.DataFrame({"value": np.arange(12)}),
        "y": pd.Series(np.arange(12.0)),
        "groups": None,
        "timestamps": pd.Series(np.arange(12)),
        "label_end_times": pd.Series(np.arange(12) + 10),
    }
    with pytest.raises(ValueError):
        execution._splits(data, "regression", {"seeds": [42], "folds": 2}, [(np.arange(6), np.arange(6, 12))])


def split_plan_registration():
    registered = {
        **registration(),
        "family_ids": ["fresh", "uci-fresh"],
        "memory_limit_mb": 4096,
        "expected_cells": 8,
        "split_plan": {
            "fresh": {"policy": "official", "official_split_ids": [{"repeat": 0, "fold": 0, "sample": 0}]},
            "uci-fresh": {"policy": "iid", "folds": 3},
        },
    }
    specs = [
        {
            "name": "official-data",
            "family_id": "fresh",
            "role": "confirmation",
            "task": "classification",
            "provider": "openml_task",
            "split_policy": "official",
            "source": {"official_split_dimensions": {"repeats": 1, "folds": 2, "samples": 1}},
        },
        {
            "name": "iid-data",
            "family_id": "uci-fresh",
            "role": "confirmation",
            "task": "classification",
            "provider": "uci",
            "split_policy": "iid",
        },
    ]
    config = {
        **execution.DEFAULTS,
        **registered["selection_parameters"],
        "methods": registered["methods"],
        "learners": registered["learners"],
        "seeds": registered["seeds"],
        "max_rows": registered["max_rows"],
        "threads": registered["threads"],
        "case_timeout": registered["case_timeout_seconds"],
        "allow_confirmation": True,
    }
    return registered, specs, config


def confirmation_data():
    return {
        "X": pd.DataFrame({"x0": np.arange(12.0)}),
        "y": pd.Series([0, 1] * 6),
        "groups": None,
        "timestamps": None,
    }


def official_payload():
    # The selected identifier is second in the source, not the first pair.
    pairs = [(np.arange(8, 12), np.arange(8)), (np.arange(8), np.arange(8, 12))]
    metadata = {
        "official_split_ids": [
            {"repeat": 0, "fold": 1, "sample": 0},
            {"repeat": 0, "fold": 0, "sample": 0},
        ]
    }
    return confirmation_data(), metadata, pairs


def test_mixed_split_plan_binds_exact_counts_without_mutating_sources():
    registered, specs, config = split_plan_registration()
    before = copy.deepcopy((registered, specs, config))
    suite = {"manifests": [{"preregistration": registered}]}
    assert execution.bind_preregistration(suite, specs, config, 100) is registered
    assert (registered, specs, config) == before


@pytest.mark.parametrize(
    "plan",
    [
        None,
        [],
        {},
        {"fresh": {"policy": "official", "official_split_ids": [{"repeat": 0, "fold": 0, "sample": 0}]}},
        {"fresh": {"policy": "iid", "folds": 3}, "uci-fresh": {"policy": "iid", "folds": 3}},
        {"fresh": {"policy": "official", "official_split_ids": []}, "uci-fresh": {"policy": "iid", "folds": 3}},
    ],
)
def test_split_plan_rejects_invalid_schema_or_family_coverage(plan):
    registered, specs, config = split_plan_registration()
    registered["split_plan"] = plan
    with pytest.raises(ValueError, match="split_plan|policy|identifiers"):
        execution.bind_preregistration({"manifests": [{"preregistration": registered}]}, specs, config, 100)


@pytest.mark.parametrize(
    "field,value",
    [
        ("official_split_ids", None),
        ("official_split_ids", [{"repeat": 0, "fold": 0}]),
        ("official_split_ids", [{"repeat": 0, "fold": 0, "sample": 0, "extra": 0}]),
        ("official_split_ids", [{"repeat": False, "fold": 0, "sample": 0}]),
        ("official_split_ids", [{"repeat": 0, "fold": 0.0, "sample": 0}]),
        ("official_split_ids", [{"repeat": 0, "fold": 0, "sample": -1}]),
        ("official_split_ids", [{"repeat": 0, "fold": 0, "sample": 0}] * 2),
        ("official_split_ids", [{"repeat": 0, "fold": 2, "sample": 0}]),
        ("folds", 3),
        ("policy", "grouped"),
    ],
)
def test_split_plan_rejects_malformed_official_entries(field, value):
    registered, specs, config = split_plan_registration()
    registered["split_plan"]["fresh"][field] = value
    with pytest.raises(ValueError, match="split_plan|policy|identifiers"):
        execution.bind_preregistration({"manifests": [{"preregistration": registered}]}, specs, config, 100)


@pytest.mark.parametrize("folds", [True, 3.0, "3", None, 0, 1, 2, 4])
def test_split_plan_rejects_invalid_or_unbound_iid_folds(folds):
    registered, specs, config = split_plan_registration()
    registered["split_plan"]["uci-fresh"]["folds"] = folds
    with pytest.raises(ValueError, match="folds"):
        execution.bind_preregistration({"manifests": [{"preregistration": registered}]}, specs, config, 100)


@pytest.mark.parametrize(
    "key,value",
    [
        ("expected_cells", 12),
        ("expected_cells", True),
        ("expected_cells", 8.0),
        ("family_ids", ["fresh", "uci-fresh", "fresh"]),
        ("methods", ["baseline", "baseline", "featcopilot-predictive"]),
        ("learners", ["linear", "linear"]),
        ("seeds", [2026, 2026]),
        ("seeds", [True]),
    ],
)
def test_split_plan_rejects_ambiguous_registration_counts(key, value):
    registered, specs, config = split_plan_registration()
    registered[key] = value
    with pytest.raises(ValueError):
        execution.bind_preregistration({"manifests": [{"preregistration": registered}]}, specs, config, 100)


@pytest.mark.parametrize(
    "key,value",
    [
        ("folds", 2),
        ("folds", 3.0),
        ("memory_limit_mb", 2048),
        ("allow_confirmation", False),
        ("max_candidates", 999),
        ("max_added_features", 49),
        ("max_rows", 100),
        ("threads", 4),
        ("case_timeout", 900),
        ("seeds", [42]),
        ("methods", ["baseline"]),
        ("learners", ["lightgbm"]),
    ],
)
def test_split_plan_rejects_configuration_changes(key, value):
    registered, specs, config = split_plan_registration()
    with pytest.raises(ValueError):
        execution.bind_preregistration(
            {"manifests": [{"preregistration": registered}]}, specs, {**config, key: value}, 100
        )


def test_split_plan_rejects_boolean_config_seed_even_when_equal_to_registration():
    registered, specs, config = split_plan_registration()
    registered["seeds"] = [1]
    config["seeds"] = [True]
    with pytest.raises(ValueError, match="seeds"):
        execution.bind_preregistration({"manifests": [{"preregistration": registered}]}, specs, config, 100)


@pytest.mark.parametrize("field", ["regularization", "selection_estimator_regularization"])
def test_split_plan_still_binds_model_search(field):
    registered, specs, config = split_plan_registration()
    registered["learner_search"]["linear"][field] = [1, 2, 3] if field == "regularization" else 2.0
    with pytest.raises(ValueError, match="learner search|selection estimator"):
        execution.bind_preregistration({"manifests": [{"preregistration": registered}]}, specs, config, 100)


def test_split_plan_still_enforces_budget_ceiling():
    registered, specs, config = split_plan_registration()
    with pytest.raises(ValueError, match="ceiling"):
        execution.bind_preregistration(
            {"manifests": [{"preregistration": registered}]},
            specs,
            config,
            registered["total_active_seconds_ceiling"] + 1,
        )


def test_split_plan_rejects_extra_family_iid_fields_and_missing_cell_count():
    registered, specs, config = split_plan_registration()
    suite = {"manifests": [{"preregistration": registered}]}
    registered["split_plan"]["extra"] = {"policy": "iid", "folds": 3}
    with pytest.raises(ValueError, match="split_plan"):
        execution.bind_preregistration(suite, specs, config, 100)
    del registered["split_plan"]["extra"]
    registered["split_plan"]["uci-fresh"]["official_split_ids"] = []
    with pytest.raises(ValueError, match="split_plan"):
        execution.bind_preregistration(suite, specs, config, 100)
    del registered["split_plan"]["uci-fresh"]["official_split_ids"]
    del registered["expected_cells"]
    with pytest.raises(ValueError, match="expected_cells"):
        execution.bind_preregistration(suite, specs, config, 100)


def test_split_plan_public_validation_checks_only_selected_official_class_support():
    registered, specs, _ = split_plan_registration()
    data, metadata, pairs = official_payload()
    pairs[0] = (np.array([0, 2, 4, 6]), np.array([1, 3, 5, 7, 8, 9, 10, 11]))
    assert execution.validate_confirmation_payload(specs[0], data, metadata, pairs, registered) is None
    pairs[1] = pairs[0]
    with pytest.raises(ValueError, match="class support"):
        execution.validate_confirmation_payload(specs[0], data, metadata, pairs, registered)


@pytest.mark.parametrize("problem", ["unavailable", "missing", "duplicate", "unpaired", "boolean", "source-missing"])
def test_split_plan_rejects_missing_or_ambiguous_source_identifiers(problem):
    registered, specs, _ = split_plan_registration()
    data, metadata, pairs = official_payload()
    if problem == "unavailable":
        pairs = None
    elif problem == "missing":
        metadata["official_split_ids"][1]["fold"] = 2
    elif problem == "duplicate":
        metadata["official_split_ids"][1] = metadata["official_split_ids"][0]
    elif problem == "unpaired":
        pairs.pop()
    elif problem == "boolean":
        metadata["official_split_ids"][1]["repeat"] = False
    else:
        metadata["official_split_ids"].pop(0)
        pairs.pop(0)
    with pytest.raises(ValueError, match="official|Official"):
        execution.validate_confirmation_payload(specs[0], data, metadata, pairs, registered)


def test_split_plan_public_validation_supports_iid_and_rejects_official_fallback():
    registered, specs, _ = split_plan_registration()
    data = confirmation_data()
    assert execution.validate_confirmation_payload(specs[1], data, {}, None, registered) is None
    with pytest.raises(ValueError, match="IID|iid"):
        execution.validate_confirmation_payload(specs[1], data, {}, [(np.arange(8), np.arange(8, 12))], registered)


@pytest.mark.parametrize("margin", ["min_total", "min_outer_train", "min_outer_test"])
def test_split_plan_iid_checks_class_support_on_requested_splits(margin):
    registered, specs, _ = split_plan_registration()
    registered["classification_admission_margin"][margin] = 7
    with pytest.raises(ValueError, match="class support"):
        execution.validate_confirmation_payload(specs[1], confirmation_data(), {}, None, registered)


def test_split_plan_iid_does_not_reduce_infeasible_fold_count():
    registered, specs, _ = split_plan_registration()
    data = confirmation_data()
    data["y"] = pd.Series([0] * 10 + [1] * 2)
    with pytest.raises(ValueError, match="Requested 3 outer folds"):
        execution.validate_confirmation_payload(specs[1], data, {}, None, registered)


@pytest.fixture
def mixed_confirmation(experiment, monkeypatch):
    output, arguments = experiment
    registered, specs, config = split_plan_registration()
    suite = {"name": "mixed-confirmation", "datasets": specs, "preregistration": registered}
    monkeypatch.setattr(execution, "_load_suite", lambda *a, **k: (suite, False))
    # Undo the general experiment fixture's fake splitter: exercise the real safeguards.
    monkeypatch.setattr(execution, "_splits", REAL_SPLITS)
    payloads = {"official-data": official_payload(), "iid-data": (confirmation_data(), {}, None)}
    monkeypatch.setattr(execution, "_prepare_dataset", lambda spec, *a: copy.deepcopy(payloads[spec["name"]]))
    arguments += [
        "--allow-confirmation",
        "--folds",
        "3",
        "--seeds",
        "2026",
        "--threads",
        "8",
        "--case-timeout",
        "600",
        "--methods",
        *registered["methods"],
        "--learners",
        "linear",
    ]
    return output, arguments, registered, specs, config, payloads


@pytest.mark.parametrize("failed_preparation", [False, True])
def test_concurrent_confirmation_controllers_reserve_before_acquisition(
    mixed_confirmation, monkeypatch, failed_preparation
):
    output, arguments, registered, specs, config, payloads = mixed_confirmation
    specs[:] = specs[:1]
    registered["family_ids"] = ["fresh"]
    registered["split_plan"] = {"fresh": registered["split_plan"]["fresh"]}
    registered["expected_cells"] = 2
    prepare = execution._prepare_dataset
    admit = execution.ExposureLedger.admit
    gate = Barrier(2)
    opened = []
    outputs = [output, output.with_name("competing-run")]

    def acquire(*args):
        opened.append(args[0]["name"])
        if failed_preparation:
            raise ValueError("Synthetic failure after reservation")
        return prepare(*args)

    def reserve(ledger, family_id, run_id, role):
        gate.wait(timeout=10)
        return admit(ledger, family_id, run_id, role)

    monkeypatch.setattr(execution, "_prepare_dataset", acquire)
    monkeypatch.setattr(tabular, "run_isolated", successful_result)
    with monkeypatch.context() as patch:
        patch.setattr(execution.ExposureLedger, "admit", reserve)
        with ThreadPoolExecutor(max_workers=2) as pool:
            runs = [pool.submit(tabular.main, [*arguments, "--output", str(directory)]) for directory in outputs]
            statuses = [run.result(timeout=30) for run in runs]
    assert len(opened) == 1
    assert sorted(statuses) == ([1, 1] if failed_preparation else [0, 1])
    ledger_path = output.parent / "benchmarks" / ".cache" / "exposure.sqlite"
    with sqlite3.connect(ledger_path) as connection:
        reservations = connection.execute("SELECT family_id, run_id FROM exposure").fetchall()
    assert len(reservations) == 1 and reservations[0][0] == "fresh"
    winner = next(
        directory
        for directory in outputs
        if json.loads((directory / "manifest.json").read_text())["run_id"] == reservations[0][1]
    )
    loser = next(directory for directory in outputs if directory != winner)
    resumed_prepare = Mock(wraps=prepare)
    monkeypatch.setattr(execution, "_prepare_dataset", resumed_prepare)
    assert tabular.main(["--output", str(winner), "--resume", "--retry-failed"]) == 0
    assert resumed_prepare.call_count == 1
    loser_archive = (loser / "archive.json").read_bytes()
    assert tabular.main(["--output", str(loser), "--resume", "--retry-failed"]) == 1
    assert resumed_prepare.call_count == 1
    assert (loser / "archive.json").read_bytes() == loser_archive


@pytest.mark.parametrize(
    "error", [ValueError("Reserved by another run"), sqlite3.OperationalError("database is locked")]
)
def test_failed_confirmation_reservation_never_opens_data(mixed_confirmation, monkeypatch, error):
    output, arguments, registered, specs, config, payloads = mixed_confirmation
    acquire = Mock(side_effect=AssertionError("failed reservation opened protected data"))
    monkeypatch.setattr(execution, "_prepare_dataset", acquire)
    monkeypatch.setattr(execution.ExposureLedger, "admit", Mock(side_effect=error))
    monkeypatch.setattr(tabular, "run_isolated", Mock(side_effect=AssertionError("failed reservation ran a model")))
    assert tabular.main(arguments) == 1
    acquire.assert_not_called()
    rows = json.loads((output / "results.json").read_text())["rows"]
    assert all(row["status"] == "dataset_error" and "loss" not in row for row in rows)


@pytest.mark.parametrize("prepare_only", [False, True])
@pytest.mark.parametrize("interrupted", [False, True])
def test_locked_count_protocol_failure_is_terminal(
    mixed_confirmation, monkeypatch, controller_clock, caplog, prepare_only, interrupted
):
    output, arguments, registered, specs, config, payloads = mixed_confirmation
    specs[:] = specs[:1]
    registered["family_ids"] = ["fresh"]
    registered.pop("split_plan")
    registered["expected_cells"] = 5
    prepare = execution._prepare_dataset

    def timed_prepare(*args):
        controller_clock[0] += 7
        if interrupted:
            raise KeyboardInterrupt
        return prepare(*args)

    monkeypatch.setattr(execution, "_prepare_dataset", timed_prepare)
    monkeypatch.setattr(tabular, "run_isolated", Mock(side_effect=AssertionError("invalid plan started a worker")))
    assert tabular.main([*arguments, *(["--prepare-only"] if prepare_only else [])]) == 2
    state = json.loads((output / "state.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert state["elapsed_seconds"] == manifest["active_seconds"] == 7
    assert manifest["run_status"] == "protocol_error"
    assert "Planned confirmation cell count" in manifest["protocol_error"]
    assert manifest["planned_cells"] == 4
    assert "protocol failure" in caplog.text
    assert state["records"] == {}
    assert_protocol_resume_rejected(output, monkeypatch)


def test_mixed_confirmation_executes_exact_locked_ids_and_seeded_iid_splits(mixed_confirmation, monkeypatch):
    output, arguments, registered, specs, config, payloads = mixed_confirmation
    calls = []

    def run(payload, directory, timeout):
        calls.append(payload)
        return successful_result(payload, directory, timeout)

    monkeypatch.setattr(tabular, "run_isolated", run)
    assert tabular.main(arguments) == 0
    assert len(calls) == 8
    for payload in calls:
        if payload["family_id"] == "fresh":
            expected = payloads["official-data"][2][1]
            assert payload["fold"] == 0
            assert payload["expected_cells"] == 1
        else:
            iid = REAL_SPLITS(payloads["iid-data"][0], "classification", config, None)
            expected = iid[payload["fold"]][2:]
            assert payload["expected_cells"] == 3
        assert payload["seed"] == 2026
        np.testing.assert_array_equal(payload["train_idx"], expected[0])
        np.testing.assert_array_equal(payload["test_idx"], expected[1])
    manifest = json.loads((output / "manifest.json").read_text())
    official, iid = manifest["datasets"]
    assert official["official_split_ids"] == payloads["official-data"][1]["official_split_ids"]
    assert official["selected_official_split_ids"] == registered["split_plan"]["fresh"]["official_split_ids"]
    assert official["selected_split_count"] == official["outer_split_count"] == 1
    assert iid["selected_split_count"] == iid["outer_split_count"] == 3
    assert manifest["splits"][0]["official_split_id"] == {"repeat": 0, "fold": 0, "sample": 0}
    assert manifest["planned_cells"] == registered["expected_cells"]
    assert tabular.main(["--output", str(output), "--resume"]) == 0
    assert len(calls) == 8


@pytest.mark.parametrize("mutation", ["metadata", "split"])
def test_confirmation_exhausted_resume_rejects_retained_input_drift(
    mixed_confirmation, monkeypatch, controller_clock, mutation
):
    output, arguments, registered, specs, config, payloads = mixed_confirmation
    monkeypatch.setattr(tabular, "run_isolated", successful_result)
    assert tabular.main(arguments) == 0
    state = json.loads((output / "state.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    state["elapsed_seconds"] = manifest["budget_seconds"]
    if mutation == "metadata":
        state["datasets"]["official-data"]["sha256"] = "tampered"
    else:
        manifest["splits"][0]["sha256"] = "tampered"
    (output / "state.json").write_text(json.dumps(state))
    (output / "manifest.json").write_text(json.dumps(manifest))
    before = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    monkeypatch.setattr(
        execution.ExposureLedger, "__init__", Mock(side_effect=AssertionError("Opened exposure ledger"))
    )
    monkeypatch.setattr(execution.RunState, "save", Mock(side_effect=AssertionError("Rejected resume rewrote state")))
    assert tabular.main(["--output", str(output), "--resume"]) == 2
    assert {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("prepare_only", [False, True])
def test_interrupted_confirmation_retains_all_seeds_and_continues_without_retry(
    mixed_confirmation, monkeypatch, controller_clock, prepare_only
):
    output, arguments, registered, specs, config, payloads = mixed_confirmation
    registered["seeds"] = [2026, 2027]
    registered["expected_cells"] *= 2
    prepare = execution._prepare_dataset

    def interrupt(spec, *args):
        if spec["name"] == "iid-data":
            controller_clock[0] += 60
            raise KeyboardInterrupt
        return prepare(spec, *args)

    with monkeypatch.context() as patch:
        patch.setattr(execution, "_prepare_dataset", interrupt)
        patch.setattr(tabular, "run_isolated", lambda *a: pytest.fail("interruption started a worker"))
        assert (
            tabular.main(
                [
                    *arguments,
                    "--seeds",
                    "2026",
                    "2027",
                    "--total-seconds",
                    "100",
                    *(["--prepare-only"] if prepare_only else []),
                ]
            )
            == 1
        )
    state = json.loads((output / "state.json").read_text())
    rows = [json.loads((output / "records" / record["file"]).read_text()) for record in state["records"].values()]
    assert len(rows) == 16
    assert {(row["dataset"], row["seed"], row["fold"], row["method"]) for row in rows} == {
        (name, seed, fold, method)
        for name, folds in (("official-data", 1), ("iid-data", 3))
        for seed in (2026, 2027)
        for fold in range(folds)
        for method in registered["methods"]
    }
    assert all(row["status"] == "interrupted" and "loss" not in row for row in rows)
    timeouts = []

    def run(payload, directory, timeout):
        timeouts.append(timeout)
        return successful_result(payload, directory, timeout)

    monkeypatch.setattr(tabular, "run_isolated", run)
    assert tabular.main(["--output", str(output), "--resume"]) == 0
    assert timeouts == [35] * 16


@pytest.mark.parametrize("interrupted", [False, True])
def test_split_plan_prepares_exact_12_family_128_cell_cohort(
    mixed_confirmation, monkeypatch, controller_clock, interrupted
):
    output, arguments, registered, specs, config, payloads = mixed_confirmation
    official_template, iid_template = copy.deepcopy(specs)
    specs[:] = [
        {
            **copy.deepcopy(official_template if index < 10 else iid_template),
            "name": f"data-{index}",
            "family_id": f"family-{index}",
        }
        for index in range(12)
    ]
    registered["family_ids"] = [spec["family_id"] for spec in specs]
    registered["split_plan"] = {
        spec["family_id"]: copy.deepcopy(registered["split_plan"]["fresh" if index < 10 else "uci-fresh"])
        for index, spec in enumerate(specs)
    }
    registered["methods"] = ["baseline", "featcopilot-importance-paired", "openfe-inductive-native", "autofeat-native"]
    registered["learners"] = ["linear", "lightgbm"]
    registered["learner_search"]["lightgbm"] = {"num_leaves": [7, 15, 31], "selection_estimator_num_leaves": 15}
    registered["case_timeout_seconds"] = 900
    registered["total_active_seconds_ceiling"] = 38 * 3600
    registered["expected_cells"] = 128
    del registered["expected_official_split_dimensions_per_task"]
    all_ids = [{"repeat": repeat, "fold": fold, "sample": 0} for repeat in range(10) for fold in range(3)][::-1]
    for index, spec in enumerate(specs):
        if index < 10:
            spec["source"]["official_split_dimensions"] = {"repeats": 10, "folds": 3, "samples": 1}
            payloads[spec["name"]] = (
                confirmation_data(),
                {"official_split_ids": all_ids},
                [(np.arange(8), np.arange(8, 12)) for _ in all_ids],
            )
        else:
            payloads[spec["name"]] = (confirmation_data(), {}, None)
    source_specs = copy.deepcopy(specs)
    if interrupted:
        prepare = execution._prepare_dataset

        def interrupt(spec, *args):
            if spec["name"] == specs[1]["name"]:
                controller_clock[0] += 60
                raise KeyboardInterrupt
            return prepare(spec, *args)

        monkeypatch.setattr(execution, "_prepare_dataset", interrupt)
    monkeypatch.setattr(tabular, "run_isolated", lambda *a: pytest.fail("planning started a model"))
    assert tabular.main(
        [
            *arguments,
            "--prepare-only",
            "--case-timeout",
            "900",
            "--total-seconds",
            str(38 * 3600),
            "--methods",
            *registered["methods"],
            "--learners",
            *registered["learners"],
        ]
    ) == int(interrupted)
    manifest = json.loads((output / "manifest.json").read_text())
    assert specs == source_specs
    assert len(manifest["datasets"]) == 12
    assert len(manifest["splits"]) == 16
    assert manifest["planned_cells"] == 128
    assert not (output / "results.json").exists()
    for dataset in manifest["datasets"][: (1 if interrupted else 10)]:
        assert len(dataset["official_split_ids"]) == 30
        assert dataset["selected_official_split_ids"] == [{"repeat": 0, "fold": 0, "sample": 0}]
        assert dataset["selected_split_count"] == dataset["outer_split_count"] == 1
    if interrupted:
        state = json.loads((output / "state.json").read_text())
        rows = [json.loads((output / "records" / record["file"]).read_text()) for record in state["records"].values()]
        assert state["elapsed_seconds"] == manifest["active_seconds"] == 60
        assert len(rows) == 128
        assert all(row["status"] == "interrupted" and "loss" not in row for row in rows)
        assert all(row["expected_cells"] == (3 if row["dataset"] in {"data-10", "data-11"} else 1) for row in rows)


def test_mixed_confirmation_supports_different_official_counts_and_multiple_seeds(mixed_confirmation, monkeypatch):
    output, arguments, registered, specs, config, payloads = mixed_confirmation
    second_official = {**copy.deepcopy(specs[0]), "name": "official-two", "family_id": "fresh-two"}
    specs.append(second_official)
    registered["family_ids"].append("fresh-two")
    registered["split_plan"]["fresh-two"] = {
        "policy": "official",
        "official_split_ids": [{"repeat": 0, "fold": fold, "sample": 0} for fold in range(2)],
    }
    payloads["official-two"] = official_payload()
    registered["seeds"] = [2026, 2027]
    registered["expected_cells"] = (1 + 3 + 2) * 2 * 2
    calls = []

    def run(payload, directory, timeout):
        calls.append(payload)
        return successful_result(payload, directory, timeout)

    monkeypatch.setattr(tabular, "run_isolated", run)
    assert tabular.main([*arguments, "--seeds", "2026", "2027"]) == 0
    assert len(calls) == 24
    for payload in calls:
        if payload["family_id"] == "fresh-two":
            expected = payloads["official-two"][2][1 - payload["fold"]]
            np.testing.assert_array_equal(payload["train_idx"], expected[0])
            np.testing.assert_array_equal(payload["test_idx"], expected[1])
            assert payload["expected_cells"] == 4
    manifest = json.loads((output / "manifest.json").read_text())
    assert [dataset["selected_split_count"] for dataset in manifest["datasets"]] == [1, 3, 2]
    assert [dataset["outer_split_count"] for dataset in manifest["datasets"]] == [2, 6, 4]


@pytest.mark.parametrize("violation", ["cell_count", "permission", "memory", "policy"])
def test_invalid_locked_plan_blocks_before_opening_data(mixed_confirmation, monkeypatch, violation):
    output, arguments, registered, specs, config, payloads = mixed_confirmation
    if violation == "cell_count":
        registered["expected_cells"] += 1
    elif violation == "permission":
        arguments.remove("--allow-confirmation")
    elif violation == "memory":
        registered["memory_limit_mb"] = 2048
    else:
        specs[1]["split_policy"] = "temporal"
    monkeypatch.setattr(execution, "_prepare_dataset", lambda *a: pytest.fail("invalid plan opened data"))
    monkeypatch.setattr(tabular, "run_isolated", lambda *a: pytest.fail("invalid plan started a model"))
    assert tabular.main(arguments) == 2
    assert not output.exists()


@pytest.mark.parametrize("seeds", [[2026], [2026, 2027]])
@pytest.mark.parametrize("failure", ["acquisition", "identity", "class_support", "boundary"])
def test_mixed_confirmation_failure_preserves_per_family_expected_cells(
    mixed_confirmation, monkeypatch, failure, seeds
):
    output, arguments, registered, specs, config, payloads = mixed_confirmation
    registered["seeds"] = seeds
    registered["expected_cells"] *= len(seeds)
    arguments += ["--seeds", *map(str, seeds)]
    if failure == "acquisition":

        def failed(*args):
            raise ValueError("Unavailable fixture payload")

        monkeypatch.setattr(execution, "_prepare_dataset", failed)
    elif failure == "identity":
        payloads["official-data"][1]["official_split_ids"] *= 2
        payloads["iid-data"] = (confirmation_data(), {}, payloads["official-data"][2])
    elif failure == "class_support":
        registered["classification_admission_margin"]["min_outer_test"] = 7
    else:
        payloads["official-data"][2][1] = (np.arange(8), np.arange(7, 12))
        from featcopilot.utils import splitting

        monkeypatch.setattr(splitting, "make_cv_splits", lambda *a, **k: [(np.arange(8), np.arange(7, 12))] * 3)
    monkeypatch.setattr(tabular, "run_isolated", lambda *a: pytest.fail("invalid preparation reached a model"))
    assert tabular.main(arguments) == 1
    rows = json.loads((output / "results.json").read_text())["rows"]
    assert len(rows) == registered["expected_cells"] == 8 * len(seeds)
    for family_id, expected in (("fresh", len(seeds)), ("uci-fresh", 3 * len(seeds))):
        family_rows = [row for row in rows if row["family_id"] == family_id]
        assert len(family_rows) == expected * 2
        assert all(row["expected_cells"] == expected and row["status"] == "dataset_error" for row in family_rows)
    if failure != "acquisition":
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["datasets"][0]["official_split_ids"] == payloads["official-data"][1]["official_split_ids"]


def test_mixed_confirmation_prepare_only_and_exposure_gate(mixed_confirmation, monkeypatch):
    output, arguments, registered, specs, config, payloads = mixed_confirmation
    monkeypatch.setattr(tabular, "run_isolated", lambda *a: pytest.fail("prepare-only started a model"))
    assert tabular.main([*arguments, "--prepare-only"]) == 0
    assert not (output / "results.json").exists()
    ledger = execution.ExposureLedger(output.parent / "benchmarks" / ".cache" / "exposure.sqlite")
    for spec in specs:
        ledger.check(spec["family_id"], "independent-run", "confirmation")
        ledger.admit(spec["family_id"], "earlier-run", "development")
    monkeypatch.setattr(execution, "_prepare_dataset", lambda *a: pytest.fail("exposed family data opened"))
    assert tabular.main(["--output", str(output), "--resume"]) == 1


def test_historical_confirmation_keeps_all_repeats_and_optional_memory_binding():
    registered, specs, config = split_plan_registration()
    del registered["split_plan"]
    registered["family_ids"] = ["fresh"]
    del registered["expected_cells"]
    del registered["memory_limit_mb"]
    suite = {"manifests": [{"preregistration": registered}]}
    assert execution.bind_preregistration(suite, specs[:1], {**config, "memory_limit_mb": 1024}, 100) is registered
    registered["memory_limit_mb"] = 4096
    with pytest.raises(ValueError, match="memory_limit_mb"):
        execution.bind_preregistration(suite, specs[:1], {**config, "memory_limit_mb": 1024}, 100)
    data, metadata, pairs = official_payload()
    assert execution.validate_confirmation_payload(specs[0], data, metadata, pairs, registered) is None
    with pytest.raises(ValueError, match="every preregistered official split"):
        execution.validate_confirmation_payload(specs[0], data, metadata, pairs[:1], registered)
    pairs[0] = (np.array([0, 2, 4, 6]), np.array([1, 3, 5, 7, 8, 9, 10, 11]))
    with pytest.raises(ValueError, match="class support"):
        execution.validate_confirmation_payload(specs[0], data, metadata, pairs, registered)


@pytest.mark.parametrize("outcome", ["prepared", "failed", "interrupted"])
def test_historical_execution_uses_all_official_repeats_not_config_folds(
    mixed_confirmation, monkeypatch, controller_clock, outcome
):
    output, arguments, registered, specs, config, payloads = mixed_confirmation
    specs[:] = specs[:1]
    registered["family_ids"] = ["fresh"]
    del registered["split_plan"]
    registered["expected_official_split_dimensions_per_task"]["repeats"] = 2
    registered["expected_cells"] = 4 * 2
    data, metadata, pairs = payloads["official-data"]
    pairs *= 2
    metadata["official_split_ids"] = [
        {"repeat": repeat, "fold": fold, "sample": 0} for repeat in (1, 0) for fold in (1, 0)
    ]
    calls = []

    def run(payload, directory, timeout):
        calls.append(payload)
        return successful_result(payload, directory, timeout)

    if outcome != "prepared":

        def unavailable(*args):
            if outcome == "interrupted":
                controller_clock[0] += 60
                raise KeyboardInterrupt
            raise ValueError("Unavailable historical fixture")

        monkeypatch.setattr(execution, "_prepare_dataset", unavailable)
    monkeypatch.setattr(tabular, "run_isolated", run)
    assert tabular.main(arguments) == int(outcome != "prepared")
    rows = json.loads((output / "results.json").read_text())["rows"]
    assert len(rows) == 8
    assert all(row["expected_cells"] == 4 for row in rows)
    assert {row["fold"] for row in rows} == {0, 1, 2, 3}
    assert len(calls) == (8 if outcome == "prepared" else 0)
    manifest = json.loads((output / "manifest.json").read_text())
    assert "selected_split_count" not in manifest["datasets"][0]
    if outcome == "interrupted":
        assert manifest["active_seconds"] == 60
        assert all(row["status"] == "interrupted" and "loss" not in row for row in rows)


@pytest.fixture
def resplit_confirmation(mixed_confirmation):
    output, arguments, registered, specs, config, payloads = mixed_confirmation
    registered["resplit_official_as_iid"] = True
    registered["split_plan"] = {spec["family_id"]: {"policy": "iid", "folds": 3} for spec in specs}
    registered["seeds"] = config["seeds"] = [2027]
    registered["expected_cells"] = 12
    registered["classification_admission_margin"] = {"min_total": 30, "min_outer_train": 20, "min_outer_test": 10}
    data = {
        "X": pd.DataFrame({"x0": np.arange(120.0)}),
        "y": pd.Series([0] * 60 + [1] * 60),
        "groups": None,
        "timestamps": None,
    }
    pairs = []
    identifiers = []
    for fold in reversed(range(10)):
        test = np.concatenate([np.arange(fold * 6, (fold + 1) * 6), np.arange(60 + fold * 6, 60 + (fold + 1) * 6)])
        pairs.append((np.setdiff1d(np.arange(120), test), test))
        identifiers.append({"repeat": 0, "fold": fold, "sample": 0})
    specs[0]["source"]["official_split_dimensions"] = {"repeats": 1, "folds": 10, "samples": 1}
    payloads["official-data"] = (
        data,
        {"sha256": "raw-source-hash", "official_splits_sha256": "raw-official-hash", "official_split_ids": identifiers},
        pairs,
    )
    payloads["iid-data"] = (copy.deepcopy(data), {"sha256": "raw-iid-hash"}, None)
    arguments += ["--seeds", "2027"]
    return output, arguments, registered, specs, config, payloads


@pytest.mark.parametrize(
    "violation",
    [
        "integer-flag",
        "string-flag",
        "null-flag",
        "false-flag",
        "absent-flag",
        "missing-plan",
        "mixed-policy",
        "folds-two",
        "boolean-folds",
        "float-folds",
        "config-folds",
        "extra-entry-field",
        "row-cap",
        "boolean-row-cap",
        "grouped-source",
        "time-key",
        "unlocked",
        "permission",
        "seed",
        "memory",
        "cohort",
    ],
)
def test_resplit_protocol_rejects_invalid_binding_before_acquisition(resplit_confirmation, monkeypatch, violation):
    output, arguments, registered, specs, config, payloads = resplit_confirmation
    if violation in ("integer-flag", "string-flag", "null-flag", "false-flag"):
        registered["resplit_official_as_iid"] = {
            "integer-flag": 1,
            "string-flag": "true",
            "null-flag": None,
            "false-flag": False,
        }[violation]
    elif violation == "absent-flag":
        del registered["resplit_official_as_iid"]
    elif violation == "missing-plan":
        del registered["split_plan"]
    elif violation == "mixed-policy":
        registered["split_plan"]["fresh"] = {
            "policy": "official",
            "official_split_ids": [{"repeat": 0, "fold": 0, "sample": 0}],
        }
        registered["expected_cells"] = 8
    elif violation in ("folds-two", "boolean-folds", "float-folds"):
        value = {"folds-two": 2, "boolean-folds": True, "float-folds": 3.0}[violation]
        for entry in registered["split_plan"].values():
            entry["folds"] = value
        if violation == "folds-two":
            registered["expected_cells"] = 8
            arguments += ["--folds", "2"]
    elif violation == "config-folds":
        arguments += ["--folds", "2"]
    elif violation == "extra-entry-field":
        registered["split_plan"]["fresh"]["source_policy"] = "official"
    elif violation in ("row-cap", "boolean-row-cap"):
        registered["max_rows"] = 100 if violation == "row-cap" else False
        arguments += ["--max-rows", "100" if violation == "row-cap" else "0"]
    elif violation == "grouped-source":
        specs[1]["split_policy"] = "grouped"
    elif violation == "time-key":
        specs[0]["time_column"] = "event-time"
    elif violation == "unlocked":
        registered["status"] = "draft"
    elif violation == "permission":
        arguments.remove("--allow-confirmation")
    elif violation == "seed":
        arguments += ["--seeds", "2026"]
    elif violation == "memory":
        arguments += ["--memory-limit-mb", "2048"]
    else:
        registered["family_ids"] = ["fresh"]
    monkeypatch.setattr(execution, "_prepare_dataset", lambda *a: pytest.fail("invalid resplitting opened data"))
    monkeypatch.setattr(tabular, "run_isolated", lambda *a: pytest.fail("invalid resplitting started a model"))
    assert tabular.main(arguments) == 2
    assert not output.exists()


def test_resplit_protocol_uses_seeded_iid_preserves_full_source_and_can_resume(resplit_confirmation, monkeypatch):
    output, arguments, registered, specs, config, payloads = resplit_confirmation
    original_specs = copy.deepcopy(specs)
    calls = []
    original_prepare = execution._prepare_dataset

    def prepare(spec, source, suite, legacy, runtime_config, root):
        assert runtime_config["max_rows"] == 0
        assert spec == next(item for item in original_specs if item["name"] == spec["name"])
        return original_prepare(spec, source, suite, legacy, runtime_config, root)

    def run(payload, directory, timeout):
        calls.append(payload)
        return successful_result(payload, directory, timeout)

    monkeypatch.setattr(execution, "_prepare_dataset", prepare)
    monkeypatch.setattr(tabular, "run_isolated", run)
    assert tabular.main(arguments) == 0
    assert len(calls) == 12
    assert specs == original_specs
    for payload in calls:
        data = payloads[payload["dataset"]][0]
        expected = REAL_SPLITS(data, "classification", config, None)[payload["fold"]]
        assert payload["seed"] == 2027
        assert payload["expected_cells"] == 3
        np.testing.assert_array_equal(payload["train_idx"], expected[2])
        np.testing.assert_array_equal(payload["test_idx"], expected[3])
        assert len(np.union1d(payload["train_idx"], payload["test_idx"])) == 120
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["planned_cells"] == 12
    assert all("official_split_id" not in item for item in manifest["splits"])
    for dataset in manifest["datasets"]:
        assert dataset["used_split_policy"] == "iid"
        assert dataset["selected_official_split_ids"] == []
        assert dataset["selected_split_count"] == dataset["outer_split_count"] == 3
    official = manifest["datasets"][0]
    assert official["sha256"] == "raw-source-hash"
    assert official["official_splits_sha256"] == "raw-official-hash"
    assert official["official_split_ids"] == payloads["official-data"][1]["official_split_ids"]
    assert len(official["official_split_ids"]) == 10
    assert tabular.main(["--output", str(output), "--resume"]) == 0
    assert len(calls) == 12
    payloads["official-data"][1]["official_splits_sha256"] = "changed-source-splits"
    assert tabular.main(["--output", str(output), "--resume"]) == 2
    assert len(calls) == 12


@pytest.mark.parametrize("problem", ["missing-pairs", "missing-id", "duplicate-id", "unpaired-id", "group-metadata"])
def test_resplit_protocol_does_not_ignore_source_validation(resplit_confirmation, problem):
    output, arguments, registered, specs, config, payloads = resplit_confirmation
    data, metadata, pairs = copy.deepcopy(payloads["official-data"])
    if problem == "missing-pairs":
        pairs = None
    elif problem == "missing-id":
        pairs.pop()
        metadata["official_split_ids"].pop()
    elif problem == "duplicate-id":
        metadata["official_split_ids"][1] = metadata["official_split_ids"][0]
    elif problem == "unpaired-id":
        metadata["official_split_ids"].pop()
    else:
        data["groups"] = pd.Series(np.repeat(np.arange(12), 10))
    with pytest.raises(ValueError, match="Official|IID"):
        execution.validate_confirmation_payload(specs[0], data, metadata, pairs, registered)


def test_resplit_protocol_still_rejects_official_pairs_for_native_iid_source(resplit_confirmation):
    output, arguments, registered, specs, config, payloads = resplit_confirmation
    data, metadata, pairs = payloads["official-data"]
    with pytest.raises(ValueError, match="IID confirmation cannot use official"):
        execution.validate_confirmation_payload(specs[1], data, metadata, pairs, registered)


def test_resplit_flag_false_preserves_locked_mixed_and_historical_validation():
    registered, specs, config = split_plan_registration()
    registered["resplit_official_as_iid"] = False
    suite = {"manifests": [{"preregistration": registered}]}
    assert execution.bind_preregistration(suite, specs, config, 100) is registered
    data, metadata, pairs = official_payload()
    assert execution.validate_confirmation_payload(specs[0], data, metadata, pairs, registered) is None
    del registered["split_plan"]
    registered["family_ids"] = ["fresh"]
    assert execution.bind_preregistration(suite, specs[:1], config, 100) is registered
    assert execution.validate_confirmation_payload(specs[0], data, metadata, pairs, registered) is None
    with pytest.raises(ValueError, match="every preregistered official split"):
        execution.validate_confirmation_payload(specs[0], data, metadata, pairs[:1], registered)
    registered["resplit_official_as_iid"] = True
    with pytest.raises(ValueError, match="every split_plan entry"):
        execution.validate_confirmation_payload(specs[0], data, metadata, pairs, registered)


@pytest.mark.parametrize("failure", [None, "min_total", "min_outer_train", "min_outer_test"])
def test_resplit_protocol_enforces_unchanged_margins_on_every_iid_fold(resplit_confirmation, monkeypatch, failure):
    from featcopilot.utils import splitting

    output, arguments, registered, specs, config, payloads = resplit_confirmation
    data = {
        "X": pd.DataFrame({"x0": np.arange(60.0)}),
        "y": pd.Series([0] * 30 + [1] * 30),
        "groups": None,
        "timestamps": None,
    }
    if failure == "min_total":
        data["y"].iloc[-1] = 0
    elif failure is not None:
        pairs = [(train.copy(), test.copy()) for _, _, train, test in REAL_SPLITS(data, "classification", config, None)]
        train, test = pairs[-1]
        if failure == "min_outer_train":
            moved = train[-1]
            pairs[-1] = (train[train != moved], np.append(test, moved))
        else:
            moved = test[-1]
            pairs[-1] = (np.append(train, moved), test[test != moved])
        monkeypatch.setattr(splitting, "make_cv_splits", lambda *a, **k: pairs)
    if failure is None:
        assert execution.validate_confirmation_payload(specs[1], data, {}, None, registered) is None
    else:
        with pytest.raises(ValueError, match="overall" if failure == "min_total" else failure):
            execution.validate_confirmation_payload(specs[1], data, {}, None, registered)
    assert registered["classification_admission_margin"] == {
        "min_total": 30,
        "min_outer_train": 20,
        "min_outer_test": 10,
    }


@pytest.mark.parametrize("outcome", ["prepared", "failed", "interrupted"])
def test_resplit_protocol_plans_exact_480_cells_even_on_failed_preparation(
    resplit_confirmation, monkeypatch, controller_clock, outcome
):
    output, arguments, registered, specs, config, payloads = resplit_confirmation
    template = copy.deepcopy(specs[0])
    specs[:] = [
        {**copy.deepcopy(template), "name": f"resplit-data-{index}", "family_id": f"resplit-family-{index}"}
        for index in range(10)
    ]
    registered["family_ids"] = [spec["family_id"] for spec in specs]
    registered["split_plan"] = {spec["family_id"]: {"policy": "iid", "folds": 3} for spec in specs}
    registered["methods"] = ["baseline", "featcopilot-importance-paired", "openfe-inductive-native", "autofeat-native"]
    registered["learners"] = ["linear", "lightgbm", "catboost", "xgboost"]
    for learner in registered["learners"][1:]:
        parameter = "num_leaves" if learner == "lightgbm" else "depth"
        registered["learner_search"][learner] = {
            parameter: [candidate[parameter] for candidate in tabular.candidate_parameters(learner)],
            f"selection_estimator_{parameter}": tabular.candidate_parameters(learner)[1][parameter],
        }
    registered["expected_cells"] = 480
    for spec in specs:
        payloads[spec["name"]] = copy.deepcopy(payloads["official-data"])
    if outcome != "prepared":

        def unavailable(*args):
            if outcome == "interrupted":
                controller_clock[0] += 60
                raise KeyboardInterrupt
            raise ValueError("Synthetic acquisition failure")

        monkeypatch.setattr(execution, "_prepare_dataset", unavailable)
        # Inspect persisted cell records instead of rendering 480 intermediate reports.
        monkeypatch.setattr(tabular, "write_report", lambda *a: None)
        monkeypatch.setattr(execution, "archive_results", lambda *a: None)
    monkeypatch.setattr(tabular, "run_isolated", lambda *a: pytest.fail("planning/failure started a model"))
    assert tabular.main(
        [
            *arguments,
            "--methods",
            *registered["methods"],
            "--learners",
            *registered["learners"],
            *(["--prepare-only"] if outcome == "prepared" else []),
        ]
    ) == int(outcome != "prepared")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["planned_cells"] == 480
    assert len(manifest["splits"]) == 30
    assert all(item["selected_split_count"] == 3 for item in manifest["datasets"])
    if outcome != "prepared":
        state = json.loads((output / "state.json").read_text())
        rows = [json.loads((output / "records" / record["file"]).read_text()) for record in state["records"].values()]
        assert len(rows) == 480
        for spec in specs:
            family_rows = [row for row in rows if row["family_id"] == spec["family_id"]]
            assert len(family_rows) == 48
            status = "interrupted" if outcome == "interrupted" else "dataset_error"
            assert all(row["expected_cells"] == 3 and row["status"] == status for row in family_rows)
        if outcome == "interrupted":
            assert state["elapsed_seconds"] == manifest["active_seconds"] == 60
            assert all(row["preparation_state"] == "unprepared" and "loss" not in row for row in rows)


def test_resplit_protocol_cannot_bypass_exposure_ledger(resplit_confirmation, monkeypatch):
    output, arguments, registered, specs, config, payloads = resplit_confirmation
    ledger = execution.ExposureLedger(output.parent / "benchmarks" / ".cache" / "exposure.sqlite")
    for spec in specs:
        ledger.admit(spec["family_id"], "prior-run", "development")
    monkeypatch.setattr(execution, "_prepare_dataset", lambda *a: pytest.fail("exposed source was opened"))
    monkeypatch.setattr(tabular, "run_isolated", lambda *a: pytest.fail("exposed source reached a model"))
    assert tabular.main([*arguments, "--prepare-only"]) == 1


def test_resplit_flag_does_not_change_nonconfirmation_splits(experiment, monkeypatch):
    output, arguments = experiment
    suite = {
        "name": "development-with-inert-registration",
        "datasets": [{"name": "test-data", "family_id": "test-family", "role": "development", "task": "regression"}],
        "preregistration": {"resplit_official_as_iid": True},
    }
    monkeypatch.setattr(execution, "_load_suite", lambda *a, **k: (suite, False))
    monkeypatch.setattr(tabular, "run_isolated", successful_result)
    assert tabular.main(arguments) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["preregistration"] is None
    assert manifest["planned_cells"] == 4
    assert "used_split_policy" not in manifest["datasets"][0]


def test_llm_preflight_blocks_before_data_opening_without_explicit_soft_policy(experiment, monkeypatch):
    from benchmarks import llm_evaluation

    output, arguments = experiment
    provider = Mock(
        spec=llm_evaluation.PlanProvider,
        capabilities=llm_evaluation.ProviderCapabilities(
            single_attempt=True,
            configured=True,
            attempt_scope="logical_api_call",
            evidence="Local capability fixture: no verified hard token cap and no provider calls.",
        ),
    )
    monkeypatch.setattr(llm_evaluation, "CopilotPlanProvider", lambda: provider)
    monkeypatch.setattr(execution, "_prepare_dataset", lambda *a: pytest.fail("blocked preflight opened data"))
    assert (
        tabular.main(
            [
                *arguments,
                "--methods",
                "baseline",
                "llm-plan",
                "--live-llm",
                "--llm-ledger",
                str(output.parent / "shared-budget"),
            ]
        )
        == 2
    )
    provider.send_plan.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize("soft_policy", [False, True])
def test_missing_copilot_sdk_exits_before_data_ledger_or_provider(experiment, monkeypatch, caplog, soft_policy):
    from benchmarks import llm_evaluation

    output, arguments = experiment
    ledger_path = output.parent / "shared-budget"
    monkeypatch.setitem(sys.modules, "copilot", None)
    prepare = Mock(side_effect=AssertionError("missing SDK opened data"))
    ledger = Mock(side_effect=AssertionError("missing SDK created a ledger"))
    send = Mock(side_effect=AssertionError("missing SDK started the provider"))
    monkeypatch.setattr(execution, "_prepare_dataset", prepare)
    monkeypatch.setattr(llm_evaluation, "EvaluationLedger", ledger)
    monkeypatch.setattr(llm_evaluation.CopilotPlanProvider, "send_plan", send)
    assert (
        tabular.main(
            [
                *arguments,
                "--methods",
                "baseline",
                "llm-plan",
                "--live-llm",
                "--llm-ledger",
                str(ledger_path),
                *(["--allow-soft-llm-tokens"] if soft_policy else []),
            ]
        )
        == 2
    )
    assert "featcopilot[llm]" in caplog.text
    prepare.assert_not_called()
    ledger.assert_not_called()
    send.assert_not_called()
    assert not output.exists()
    assert not ledger_path.exists()
