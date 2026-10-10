"""Synthetic/mocked contracts for the fixed classification calibration study."""

import copy
import gzip
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from benchmarks import calibration_evaluation as study
from benchmarks import execution, tabular
from benchmarks.run_state import fingerprint

REAL_RUN_ISOLATED = tabular.run_isolated
REAL_SPLITS = execution._splits
REPOSITORY_ROOT = tabular.ROOT


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    spec = {
        "name": "test-classifier",
        "family_id": "test-family",
        "role": "development",
        "task": "classification",
        "provider": "uci",
        "origin_kind": "observational",
        "split_policy": "iid",
        "rights": {"status": "approved", "evidence": ["mock-only"]},
        "source": {"schema_reviewed": True},
        "_family": {"family_id": "test-family"},
    }
    suite = {"name": "mock-classification", "schema_version": 1, "datasets": [spec]}
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite))
    monkeypatch.setattr(execution, "_load_suite", lambda *args: (copy.deepcopy(suite), False))
    canonical = {
        spec["name"]: copy.deepcopy(spec),
        "second": {**copy.deepcopy(spec), "name": "second", "family_id": "second-family"},
    }
    monkeypatch.setattr(study, "_canonical_specs", lambda: copy.deepcopy(canonical))
    monkeypatch.setattr(execution, "environment_versions", lambda: {"scikit-learn": "mock"})
    monkeypatch.setattr(tabular, "ROOT", tmp_path)
    monkeypatch.setattr(tabular, "source_files", lambda: [])
    monkeypatch.setattr(tabular, "source_fingerprint", lambda: "fixed-source")
    data = {
        "X": pd.DataFrame({"x0": np.arange(24.0)}),
        "y": pd.Series(["zeta", "alpha"] * 12),
        "groups": None,
        "timestamps": None,
        "label_end_times": None,
    }
    acquisitions, admissions, calls = [], [], []

    def prepare(*args):
        acquisitions.append(args[0]["name"])
        return copy.deepcopy(data), {"sha256": "fixed-data", "review_notes": "retain even on failure"}, None

    monkeypatch.setattr(execution, "_prepare_dataset", prepare)
    monkeypatch.setattr(
        execution,
        "_splits",
        lambda *args: [
            (42, 0, np.arange(12), np.arange(12, 24)),
            (42, 1, np.arange(12, 24), np.arange(12)),
        ],
    )

    class Ledger:
        def __init__(self, path):
            admissions.append(("created", str(path)))

        def admit(self, *args):
            admissions.append(args)

    monkeypatch.setattr(study, "ExposureLedger", Ledger)

    def run(payload, directory, timeout, **kwargs):
        calls.append((copy.deepcopy(payload), kwargs))
        return success(payload)

    monkeypatch.setattr(tabular, "run_isolated", run)
    output = tmp_path / "run"
    argv = [
        "--output",
        str(output),
        "--suite",
        str(suite_path),
        "--datasets",
        spec["name"],
        "--folds",
        "2",
        "--learners",
        "linear",
    ]
    return SimpleNamespace(
        output=output,
        argv=argv,
        suite=suite,
        spec=spec,
        canonical=canonical,
        data=data,
        calls=calls,
        acquisitions=acquisitions,
        admissions=admissions,
        resume=["--output", str(output), "--resume"],
    )


def success(payload):
    rows = study._outcomes(payload, "success", "")
    for row in rows:
        row.pop("error")
        row.update(
            dict.fromkeys(study.METRICS, 0.5),
            n_train=len(payload["train_idx"]),
            n_test=len(payload["test_idx"]),
            n_input_features=1,
            n_output_features=2,
            pipeline_seconds=100.0,
            worker_wall_seconds=0.1,
        )
        if payload["job_kind"].startswith("bundle"):
            row["shared_bundle_diagnostics"] = bundle_report(payload["config"]["ensemble_sampling"])["diagnostics"]
    return rows


def results(experiment):
    return json.loads((experiment.output / "results.json").read_text())


def state_rows(experiment):
    state = json.loads((experiment.output / "state.json").read_text())
    return [
        json.loads((experiment.output / "records" / record["file"]).read_text()) for record in state["records"].values()
    ]


def test_four_physical_jobs_six_prediction_rows_and_exact_archive(experiment):
    e = experiment
    assert study.main(e.argv) == 0
    assert len(e.calls) == 8
    result = results(e)
    assert result["expected_physical_jobs"] == 8
    assert result["planned_cells"] == len(result["rows"]) == 12
    assert {row["method"] for row in result["rows"]} == set(study.METHODS)
    jobs = state_rows(e)
    assert len(jobs) == 8
    assert sorted(len(job["outcomes"]) for job in jobs) == [1, 1, 1, 1, 2, 2, 2, 2]
    for payload, kwargs in e.calls:
        expected = (
            {"worker_module": "benchmarks.calibration_evaluation"} if payload["job_kind"].startswith("bundle") else {}
        )
        assert kwargs == expected
        assert len(payload["config"]["learners"]) == 1
        matching = [row for row in result["rows"] if row["physical_job_id"] == payload["physical_job_id"]]
        assert {row["method"] for row in matching} == set(study.JOBS[payload["job_kind"]])
        assert all(row["dataset_metadata"]["review_notes"] == "retain even on failure" for row in matching)
    assert result["physical_cost"]["measured_physical_job_seconds_including_retries"] < 100
    assert sum(row["pipeline_seconds"] for row in result["rows"]) == 1200
    assert all(item["complete_paired_families"] == 1 for item in result["summary"])
    archive = json.loads((e.output / "archive.json").read_text())
    content = b"".join((e.output / name).read_bytes() for name in archive["parts"])
    assert gzip.decompress(content) == (e.output / "results.json").read_bytes()
    assert all((e.output / name).stat().st_size <= 480 * 1024 for name in archive["parts"])
    assert study.main(e.resume) == 0
    assert len(e.calls) == 8
    assert e.acquisitions == ["test-classifier", "test-classifier"]


def test_default_cohort_and_budget_denominators():
    assert study.DEFAULTS["ensemble_sampling"] == "cv3"
    assert study.SAMPLING_PROTOCOLS["cv3"] == study.PROTOCOL
    assert len(study.DEFAULT_DATASETS) == 9
    assert study.DEFAULTS["learners"] == ["linear", "lightgbm"]
    assert len(study.DEFAULT_DATASETS) * study.DEFAULTS["folds"] * 2 * len(study.JOBS) == 216
    assert len(study.DEFAULT_DATASETS) * study.DEFAULTS["folds"] * 2 * len(study.METHODS) == 324
    assert study.DEFAULTS["case_timeout"] == 900
    assert study.DEFAULTS["threads"] == 8
    assert study.DEFAULTS["memory_limit_mb"] == 4096
    assert len(set(study.DEFAULT_DATASETS)) == 9


def test_default_catalog_metadata_is_eligible_without_opening_payloads():
    suite, legacy = execution._load_suite(Path(study.DEFAULTS["suite"]), False)
    specs = study._selected_specs(suite, legacy, study.DEFAULT_DATASETS)
    assert len(specs) == 9 and all(spec["role"] == "development" for spec in specs)


@pytest.mark.parametrize(
    "changes",
    [
        {"name": "outside-classifier"},
        {"family_id": "outside-family"},
        {"source": {"schema_reviewed": True, "sha256": "different-version"}},
        {"source": {"schema_reviewed": True, "columns": ["replacement-schema"]}},
        {"source": {"schema_reviewed": True, "uci_id": 99999}},
        {"source": {"schema_reviewed": True, "data_url": "https://example.invalid/replacement.csv"}},
        {"target": "replacement-target"},
        {"target_transform": "replacement-transform"},
        {"drop_columns": ["replacement-predictor"]},
    ],
)
def test_changed_canonical_source_or_new_cohort_is_rejected_before_acquisition(experiment, changes):
    e = experiment
    e.spec.update(changes)
    argv = [*e.argv, "--datasets", e.spec["name"]]
    assert study.main(argv) == 2
    assert not e.acquisitions and not e.calls and not e.admissions
    assert not e.output.exists()


def test_canonical_definitions_can_be_filtered_for_subset_smoke(experiment):
    e = experiment
    assert set(e.canonical) == {"test-classifier", "second"}
    assert study.main([*e.argv, "--prepare-only"]) == 0
    manifest = json.loads((e.output / "manifest.json").read_text())
    assert manifest["cohort_scope"] == "canonical_subset_smoke"
    assert manifest["config"]["suite"] == str(Path(e.argv[e.argv.index("--suite") + 1]).resolve())
    assert manifest["config"]["datasets"] == ["test-classifier"]
    assert not e.calls and not e.admissions


@pytest.mark.parametrize("status", ["success", "timeout"])
@pytest.mark.parametrize("recorded_seed", [42, 43, True, "42"])
def test_worker_hash_provenance_survives_normalization_and_rejects_mismatch(experiment, status, recorded_seed):
    assert study.main(experiment.argv) == 0
    payload = next(payload for payload, _ in experiment.calls if payload["job_kind"] == "bundle-raw")
    rows = success(payload) if status == "success" else tabular.failure_rows(payload, status, "limit", 1.0)
    for row in rows:
        row["worker_python_hash_seed"] = recorded_seed
    normalized = study.normalize_outcomes(payload, rows)
    if type(recorded_seed) is int and recorded_seed == payload["seed"]:
        assert all(row["status"] == status and row["worker_python_hash_seed"] == recorded_seed for row in normalized)
    else:
        assert all(row["status"] == "error" and "loss" not in row for row in normalized)
        assert all("hash seed" in row["error"] for row in normalized)


@pytest.mark.parametrize("failure_status", ["timeout", "memory_limit", "error"])
def test_bundle_failure_is_explicit_for_both_arms_without_erasing_single(experiment, monkeypatch, failure_status):
    e = experiment

    def failed(payload, *args, **kwargs):
        if payload["job_kind"] == "bundle-raw":
            return tabular.failure_rows(payload, failure_status, "mock failure", 1.2)
        return success(payload)

    monkeypatch.setattr(tabular, "run_isolated", failed)
    assert study.main(e.argv) == 1
    rows = results(e)["rows"]
    assert len(rows) == 12
    unscored = [row for row in rows if row["status"] != "success"]
    assert len(unscored) == 4
    assert all(row["method"] in study.JOBS["bundle-raw"] and "loss" not in row for row in unscored)
    assert all(row["error"] == "mock failure" and row["case_seconds"] == 1.2 for row in unscored)
    retry_calls = []

    def retry(payload, *args, **kwargs):
        retry_calls.append(payload["job_kind"])
        return success(payload)

    monkeypatch.setattr(tabular, "run_isolated", retry)
    assert study.main(e.resume) == 1
    assert retry_calls == []
    assert study.main([*e.resume, "--retry-failed"]) == 0
    assert retry_calls == ["bundle-raw", "bundle-raw"]
    assert all(job["attempt"] == (2 if job["job_kind"] == "bundle-raw" else 1) for job in state_rows(e))


def test_learner_specific_bundle_failure_never_repeats_other_learner(experiment, monkeypatch):
    e = experiment

    def run(payload, *args, **kwargs):
        if payload["job_kind"] == "bundle-fe" and payload["config"]["learners"] == ["lightgbm"]:
            return study._outcomes(payload, "error", "one learner's bundle failed")
        return success(payload)

    monkeypatch.setattr(tabular, "run_isolated", run)
    assert study.main([*e.argv, "--learners", "linear", "lightgbm"]) == 1
    result = results(e)
    assert result["expected_physical_jobs"] == 16 and result["planned_cells"] == len(result["rows"]) == 24
    assert sum(row["status"] == "success" for row in result["rows"]) == 20
    retried = []

    def retry(payload, *args, **kwargs):
        retried.append((payload["job_kind"], payload["config"]["learners"]))
        return success(payload)

    monkeypatch.setattr(tabular, "run_isolated", retry)
    assert study.main([*e.resume, "--retry-failed"]) == 0
    assert retried == [("bundle-fe", ["lightgbm"]), ("bundle-fe", ["lightgbm"])]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "partial",
        "wrong_method",
        "wrong_split",
        "wrong_physical_job",
        "nonfinite",
        "missing_metric",
        "wrong_size",
    ],
)
def test_malformed_bundle_never_keeps_partial_scores(experiment, monkeypatch, mutation):
    def run(payload, *args, **kwargs):
        rows = success(payload)
        if payload["job_kind"] != "bundle-fe":
            return rows
        if mutation == "missing":
            rows.pop()
        elif mutation == "duplicate":
            rows[1] = dict(rows[0])
        elif mutation == "partial":
            rows[1] = {**rows[1], "status": "error", "error": "second failed"}
        elif mutation == "wrong_method":
            rows[0]["method"] = "baseline-robust"
        elif mutation == "wrong_split":
            rows[0]["split_sha256"] = "wrong"
        elif mutation == "wrong_physical_job":
            rows[0]["physical_job_id"] = "wrong"
        elif mutation == "nonfinite":
            rows[0]["loss"] = float("nan")
        elif mutation == "missing_metric":
            rows[0].pop("brier_sum")
        else:
            rows[0]["n_train"] += 1
        return rows

    monkeypatch.setattr(tabular, "run_isolated", run)
    assert study.main(experiment.argv) == 1
    rows = [row for row in results(experiment)["rows"] if row["job_kind"] == "bundle-fe"]
    assert len(rows) == 4
    assert all(row["status"] == "error" and "loss" not in row and "Invalid worker" in row["error"] for row in rows)
    if mutation == "partial":
        assert all(any(item.get("error") == "second failed" for item in row["reported_outcomes"]) for row in rows)


@pytest.mark.parametrize(
    "kind", ["source", "versions", "config", "suite_bytes", "source_zip", "data", "split", "python", "platform"]
)
def test_resume_identity_corruption_is_rejected(experiment, monkeypatch, kind):
    e = experiment
    assert study.main(e.argv) == 0
    argv = list(e.resume)
    if kind == "source":
        monkeypatch.setattr(tabular, "source_fingerprint", lambda: "changed-source")
    elif kind == "versions":
        monkeypatch.setattr(execution, "environment_versions", lambda: {"scikit-learn": "changed"})
    elif kind == "config":
        argv += ["--threads", "7"]
    elif kind == "suite_bytes":
        suite = Path(e.argv[e.argv.index("--suite") + 1])
        suite.write_text(suite.read_text() + "\n")
    elif kind == "source_zip":
        (e.output / "source.zip").write_bytes(b"corrupt")
    elif kind == "data":
        (e.output / "prepared" / "test-classifier.pkl").write_bytes(b"corrupt")
    elif kind == "split":
        monkeypatch.setattr(
            execution,
            "_splits",
            lambda *args: [(42, 0, np.arange(12)[::-1], np.arange(12, 24)), (42, 1, np.arange(12, 24), np.arange(12))],
        )
    elif kind == "python":
        monkeypatch.setattr(study.sys, "version", "changed")
    else:
        monkeypatch.setattr(study.platform, "platform", lambda: "changed")
    assert study.main(argv) == 2
    assert len(e.calls) == 8


def test_budget_only_may_increase_and_successes_are_not_repeated(experiment):
    e = experiment
    assert study.main(e.argv) == 0
    assert study.main([*e.resume, "--total-seconds", "50000"]) == 0
    assert study.main([*e.resume, "--total-seconds", "49000"]) == 2
    assert len(e.calls) == 8


def test_budget_exhausted_during_preparation_resumes_with_larger_ceiling(experiment):
    e = experiment
    assert study.main([*e.argv, "--total-seconds", "1"]) == 1
    assert not e.acquisitions
    assert study.main([*e.resume, "--total-seconds", "43200"]) == 0
    assert len(e.calls) == 8 and len(results(e)["rows"]) == 12


@pytest.mark.parametrize("registered", [False, True])
def test_interrupted_dataset_registration_reuses_same_prepared_file(experiment, monkeypatch, registered):
    e = experiment
    original = study.RunState.validate_dataset
    interrupted = False

    def register(state, name, metadata):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            if registered:
                original(state, name, metadata)
            raise KeyboardInterrupt
        original(state, name, metadata)

    monkeypatch.setattr(study.RunState, "validate_dataset", register)
    assert study.main(e.argv) == 1
    data_path = e.output / "prepared" / "test-classifier.pkl"
    content, modified = data_path.read_bytes(), data_path.stat().st_mtime_ns
    assert not e.calls
    unfinished = state_rows(e)
    assert all(job["preparation_state"] == "unfinished" and job["attempt"] == 0 for job in unfinished)
    assert all(
        row["prepared_sha256"] == tabular.digest(content)
        and row["split_sha256"] == "unavailable"
        and row["status"] == "interrupted"
        for job in unfinished
        for row in job["outcomes"]
    )
    assert study.main(e.resume) == 0
    assert len(e.calls) == 8
    assert data_path.read_bytes() == content
    assert data_path.stat().st_mtime_ns == modified
    assert e.acquisitions == ["test-classifier", "test-classifier"]
    assert all(job["preparation_state"] == "validated" for job in state_rows(e))


@pytest.mark.parametrize("failure", ["raw_checksum", "pin_identity", "metadata", "prepared_values"])
@pytest.mark.parametrize("prepare_only", [False, True])
def test_resume_revalidates_raw_source_before_reuse_or_model_work(experiment, monkeypatch, failure, prepare_only):
    from benchmarks.catalog import AcquisitionError

    e = experiment
    assert study.main([*e.argv, *(["--prepare-only"] if prepare_only else [])]) == 0
    before = (e.output / "results.json").read_bytes()
    original = execution._prepare_dataset
    checks = []

    def revalidate(*args):
        checks.append(args[0]["name"])
        if failure == "raw_checksum":
            raise AcquisitionError("Raw payload SHA-256 drift")
        if failure == "pin_identity":
            raise AcquisitionError("Checksum-pin source identity drift")
        data, metadata, official = original(*args)
        if failure == "metadata":
            metadata["sha256"] = "different-original-raw-payload"
        else:
            data["X"].iloc[0, 0] = -123.0
        return data, metadata, official

    monkeypatch.setattr(execution, "_prepare_dataset", revalidate)
    assert study.main(e.resume) == 2
    assert checks == ["test-classifier"]
    assert len(e.calls) == (0 if prepare_only else 8)
    assert (e.output / "results.json").read_bytes() == before


def test_interrupted_revalidation_never_relaxes_completed_job_identity(experiment, monkeypatch):
    e = experiment
    assert study.main(e.argv) == 0
    before = state_rows(e)

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(study.RunState, "validate_dataset", interrupt)
    assert study.main(e.resume) == 2
    assert len(e.calls) == 8
    assert state_rows(e) == before
    assert all(job["status"] == "success" and job["preparation_state"] == "validated" for job in before)


def test_bundle_checkpoint_survives_controller_crash_as_one_atomic_record(experiment, monkeypatch):
    original = study.RunState.record
    crashed = False

    def record(state, job_id, cell_hash, row):
        nonlocal crashed
        original(state, job_id, cell_hash, row)
        if row["job_kind"] == "bundle-raw" and not crashed:
            crashed = True
            raise RuntimeError("simulated post-checkpoint crash")

    monkeypatch.setattr(study.RunState, "record", record)
    with pytest.raises(RuntimeError, match="post-checkpoint"):
        study.main(experiment.argv)
    stored = state_rows(experiment)
    assert len(stored) == 3
    bundle = next(job for job in stored if job["job_kind"] == "bundle-raw")
    assert bundle["status"] == "success" and len(bundle["outcomes"]) == 2
    assert study.main(experiment.resume) == 0
    assert len(experiment.calls) == 8
    assert len(results(experiment)["rows"]) == 12


@pytest.mark.parametrize(
    "changes",
    [
        {"role": "confirmation"},
        {"role": "inventory"},
        {"role": "stress"},
        {"task": "regression"},
        {"split_policy": "group"},
        {"split_policy": "temporal"},
        {"group_column": "patient"},
        {"time_column": "timestamp"},
        {"label_end_column": "known_at"},
        {"origin_kind": "simulation"},
        {"provider": "diagnostic"},
        {"rights": {"status": "pending"}},
        {"_family": {"reservation": {"status": "active"}}},
    ],
)
def test_ineligible_selected_source_rejected_before_any_acquisition(experiment, changes):
    e = experiment
    e.spec.update(changes)
    assert study.main(e.argv) == 2
    assert not e.acquisitions and not e.calls and not e.admissions
    assert not e.output.exists()


def test_later_invalid_spec_is_checked_before_first_acquisition(experiment):
    e = experiment
    second = {**e.spec, "name": "second", "family_id": "second", "split_policy": "group"}
    e.suite["datasets"].append(second)
    assert study.main([*e.argv, "--datasets", e.spec["name"], "second"]) == 2
    assert not e.acquisitions


def test_duplicate_family_and_legacy_rejected(experiment, monkeypatch):
    e = experiment
    e.suite["datasets"].append({**e.spec, "name": "second"})
    assert study.main([*e.argv, "--datasets", e.spec["name"], "second"]) == 2
    monkeypatch.setattr(execution, "_load_suite", lambda *args: (e.suite, True))
    assert study.main(e.argv) == 2
    assert not e.acquisitions


def test_prepare_only_has_no_worker_or_exposure_writes_and_can_resume(experiment):
    e = experiment
    assert study.main([*e.argv, "--prepare-only"]) == 0
    assert not e.calls and not e.admissions
    assert (e.output / "prepared" / "test-classifier.pkl").is_file()
    state = json.loads((e.output / "state.json").read_text())
    assert state["records"] == {}
    assert state["datasets"]["test-classifier"]["prepared_sha256"]
    assert len(results(e)["rows"]) == 12
    assert all(row["status"] == "prepared" for row in results(e)["rows"])
    assert study.main(e.resume) == 0
    assert len(e.calls) == 8
    assert e.acquisitions == ["test-classifier", "test-classifier"]
    assert all(admission[-1] == "development" for admission in e.admissions[1:])


def test_preparation_failure_keeps_complete_denominator_and_metadata(experiment, monkeypatch):
    def fail(*args):
        raise ValueError("reviewed payload missing")

    monkeypatch.setattr(execution, "_prepare_dataset", fail)
    assert study.main(experiment.argv) == 1
    rows = results(experiment)["rows"]
    assert len(rows) == 12
    assert all(row["status"] == "dataset_error" and "reviewed payload missing" in row["error"] for row in rows)
    assert all(row["dataset_metadata"]["rights"]["evidence"] == ["mock-only"] for row in rows)
    assert not experiment.calls and not experiment.admissions
    assert all(item["complete_paired_families"] == 0 for item in results(experiment)["summary"])


def test_failed_family_is_never_removed_from_report_coverage(experiment, monkeypatch):
    e = experiment
    e.suite["datasets"].append({**e.spec, "name": "second", "family_id": "second-family"})
    original = execution._prepare_dataset

    def prepare(spec, *args):
        if spec["name"] == "second":
            raise ValueError("second unavailable")
        return original(spec, *args)

    monkeypatch.setattr(execution, "_prepare_dataset", prepare)
    assert study.main([*e.argv, "--datasets", e.spec["name"], "second"]) == 1
    result = results(e)
    assert result["expected_physical_jobs"] == 16 and len(result["rows"]) == 24
    assert all(item["requested_families"] == 2 for item in result["summary"])
    assert all(item["complete_paired_families"] == 1 for item in result["summary"])
    assert all(item["requested_families"] == 2 for item in result["calibration_operation_summary"])
    assert all(item["complete_paired_families"] == 1 for item in result["calibration_operation_summary"])


def test_retry_failed_preparation_replaces_all_physical_records(experiment, monkeypatch):
    prepare = execution._prepare_dataset
    monkeypatch.setattr(execution, "_prepare_dataset", lambda *args: (_ for _ in ()).throw(ValueError("missing")))
    assert study.main(experiment.argv) == 1
    monkeypatch.setattr(execution, "_prepare_dataset", prepare)
    assert study.main([*experiment.resume, "--retry-failed"]) == 0
    assert len(experiment.calls) == 8
    assert len(results(experiment)["rows"]) == 12


def test_budget_reserve_preserves_all_denominators_and_nonzero_exit(experiment):
    assert study.main([*experiment.argv, "--total-seconds", "1"]) == 1
    assert not experiment.calls and not experiment.acquisitions and not experiment.admissions
    rows = results(experiment)["rows"]
    assert len(rows) == 12
    assert all(row["status"] == "budget_exhausted" for row in rows)
    manifest = json.loads((experiment.output / "manifest.json").read_text())
    assert manifest["active_seconds"] >= 0
    assert manifest["budget_overshoot_seconds"] >= 0


def test_interruption_in_bundle_preserves_singles_and_remaining_cells(experiment, monkeypatch):
    calls = []

    def run(payload, *args, **kwargs):
        calls.append(payload["job_kind"])
        if payload["job_kind"] == "bundle-raw":
            raise KeyboardInterrupt
        return success(payload)

    monkeypatch.setattr(tabular, "run_isolated", run)
    assert study.main(experiment.argv) == 1
    assert calls == ["single-raw", "single-fe", "bundle-raw"]
    rows = results(experiment)["rows"]
    assert len(rows) == 12
    assert sum(row["status"] == "success" for row in rows) == 2
    assert sum(row["status"] == "interrupted" for row in rows) == 10


def test_interruption_during_preparation_is_recorded(experiment, monkeypatch):
    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(execution, "_prepare_dataset", interrupt)
    assert study.main(experiment.argv) == 1
    assert len(results(experiment)["rows"]) == 12
    assert all(row["status"] == "interrupted" for row in results(experiment)["rows"])


def test_source_change_stops_dispatch_invalidates_current_job_and_cannot_resume(experiment, monkeypatch):
    changed = False

    def run(payload, *args, **kwargs):
        nonlocal changed
        changed = True
        return success(payload)

    monkeypatch.setattr(tabular, "run_isolated", run)
    monkeypatch.setattr(tabular, "source_fingerprint", lambda: "changed" if changed else "fixed-source")
    assert study.main(experiment.argv) == 1
    assert len(results(experiment)["rows"]) == 12
    assert all(row["status"] == "source_changed" and "loss" not in row for row in results(experiment)["rows"])
    assert study.main(experiment.resume) == 2


def test_controller_elapsed_includes_report_work_and_records_overshoot(experiment, monkeypatch):
    ticks = [0.0]
    real_report = study.write_report

    def report(*args):
        real_report(*args)
        ticks[0] += 15.0

    monkeypatch.setattr(study.time, "perf_counter", lambda: ticks[0])
    monkeypatch.setattr(study, "write_report", report)
    assert study.main([*experiment.argv, "--total-seconds", "25"]) == 1
    manifest = json.loads((experiment.output / "manifest.json").read_text())
    assert manifest["active_seconds"] == 30
    assert manifest["budget_overshoot_seconds"] == 5
    assert len(experiment.calls) == 1
    assert len(results(experiment)["rows"]) == 12
    assert results(experiment)["physical_cost"]["measured_physical_job_seconds_including_retries"] == 0


@pytest.mark.parametrize(
    "args",
    [
        ["--folds", "1"],
        ["--folds", "100"],
        ["--threads", "0"],
        ["--max-rows", "0"],
        ["--max-added-features", "0"],
        ["--max-added-features", "25"],
        ["--max-candidates", "500"],
        ["--max-candidates", "1001"],
        ["--stability-repeats", "0"],
        ["--memory-limit-mb", "0"],
        ["--case-timeout", "nan"],
        ["--case-timeout", "inf"],
        ["--total-seconds", "0"],
        ["--total-seconds", "nan"],
        ["--total-seconds", "999999999"],
        ["--seeds", "-1"],
        ["--seeds", "42", "42"],
        ["--learners", "linear", "linear"],
        ["--datasets", "test-classifier", "test-classifier"],
        ["--retry-failed"],
    ],
)
def test_invalid_cli_configuration_before_acquisition(experiment, args):
    assert study.main([*experiment.argv, *args]) == 2
    assert not experiment.acquisitions


def test_cli_has_one_suite_no_method_or_calibration_grid(experiment):
    for args in (
        ["--methods", "baseline-robust"],
        ["--calibration", "isotonic"],
        ["--suite", "one", "two"],
        ["--learners", "random_forest"],
    ):
        with pytest.raises(SystemExit) as error:
            study.main([*experiment.argv, *args])
        assert error.value.code == 2


def test_method_list_cannot_omit_controls():
    config = copy.deepcopy(study.DEFAULTS)
    config["methods"] = list(study.METHODS[1:])
    with pytest.raises(ValueError, match="six methods"):
        study.validate_config(config, 43200)


def worker_payload(experiment):
    data = experiment.data
    train, test = np.arange(12), np.arange(12, 24)
    return {
        **data,
        "dataset": "test-classifier",
        "family_id": "test-family",
        "role": "development",
        "task": "classification",
        "source": "real_world",
        "method": "bundle-fe",
        "job_kind": "bundle-fe",
        "physical_job_id": "mock-job-id",
        "dataset_metadata": experiment.spec,
        "seed": 42,
        "fold": 0,
        "split_sha256": execution.split_fingerprint(train, test),
        "train_idx": train,
        "test_idx": test,
        "expected_cells": 2,
        "prepared_sha256": None,
        "config": {**copy.deepcopy(study.DEFAULTS), "learners": ["linear"], "seed": 42},
    }


def bundle_report(mode="cv3"):
    report = {
        "plain_metrics": dict.fromkeys(study.METRICS, 0.5),
        "calibrated_metrics": dict.fromkeys(study.METRICS, 0.4),
        "timing": {
            "fit_seconds": 12.0,
            "base_fit_seconds": 10.0,
            "calibration_and_orchestration_seconds": 2.0,
            "plain_predict_seconds": 0.5,
            "calibrated_predict_seconds": 0.75,
            "base_fe_seconds": 4.0,
        },
        "diagnostics": {
            "ensemble_sampling": mode,
            "protocol": study.CORE_PROTOCOLS[mode],
            "n_input_features": 1,
            "max_member_output_features": 4,
            "member_output_feature_counts": [2, 3, 4],
            "members": [{"id": member} for member in range(3)],
        },
    }
    if mode == "subsample85":
        report["diagnostics"].update(
            sampling_design="repeated_stratified_holdout",
            requested_train_fraction=0.85,
            requested_calibration_fraction=0.15,
            calibration_coverage_histogram=[8, 2, 2, 0],
            pairwise_overlap=[
                {"members": [0, 1], "base_train": 9, "calibration_validation": 1},
                {"members": [0, 2], "base_train": 8, "calibration_validation": 0},
                {"members": [1, 2], "base_train": 9, "calibration_validation": 1},
            ],
            distinct_train_bag_count=3,
            members=[
                {
                    "member": member,
                    "sampling_seed": 42 + 32452843 + member,
                    "base_train": {"size": 10, "fraction": 10 / 12, "mask_sha256": "a" * 64},
                    "calibration_validation": {"size": 2, "fraction": 2 / 12, "mask_sha256": "b" * 64},
                }
                for member in range(3)
            ],
        )
    return report


@pytest.mark.parametrize("mode", ["cv3", "subsample85"])
def test_worker_dispatch_is_hash_checked_preserves_class_alignment_and_shared_diagnostics(
    experiment, monkeypatch, mode
):
    from benchmarks import prediction_ensemble

    calls = []

    def fit(train_X, train_y, test_X, test_y, **kwargs):
        assert train_X.index.equals(train_y.index)
        assert test_X.index.equals(test_y.index)
        assert list(train_y[:2]) == ["zeta", "alpha"]
        assert kwargs["task"] == "classification" and kwargs["engineered"]
        calls.append(kwargs)
        assert kwargs["config"]["ensemble_sampling"] == mode
        report = bundle_report(mode)
        return prediction_ensemble.PredictionBundleResult(
            {"plain": report["plain_metrics"], "sigmoid": report["calibrated_metrics"]},
            report["diagnostics"],
            report["timing"],
            source=object(),
        )

    monkeypatch.setattr(prediction_ensemble, "fit_prediction_bundle_result", fit)
    payload = worker_payload(experiment)
    payload["config"]["ensemble_sampling"] = mode
    payload.update(study._sampling_identity(payload["config"]))
    data_path = experiment.output.parent / "worker-data.pkl"
    content = pickle.dumps(experiment.data)
    data_path.write_bytes(content)
    payload.update(data_path=str(data_path), prepared_sha256=tabular.digest(content))
    rows = study.run_case(payload, experiment.output.parent)
    assert len(calls) == 1
    assert [row["method"] for row in rows] == list(study.JOBS["bundle-fe"])
    assert [row["pipeline_seconds"] for row in rows] == [10.5, 12.75]
    assert rows[0]["shared_bundle_diagnostics"] == rows[1]["shared_bundle_diagnostics"]
    assert all(row["ensemble_sampling"] == mode and row["protocol"] == study.SAMPLING_PROTOCOLS[mode] for row in rows)
    assert [row["diagnostic_only"] for row in rows] == [False, mode == "subsample85"]
    assert all(row["n_output_features"] == 4 and row["member_output_feature_counts"] == [2, 3, 4] for row in rows)
    fingerprint(rows)
    data_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Prepared dataset changed"):
        study.run_case(payload, experiment.output.parent)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "kind", ["overlap", "range", "dtype", "duplicate", "unaligned", "class", "group", "task", "split_hash"]
)
def test_worker_rejects_bad_payload_before_fitting(experiment, monkeypatch, kind):
    from benchmarks import prediction_ensemble

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid payload must never reach model fitting")

    monkeypatch.setattr(prediction_ensemble, "fit_prediction_bundle", unexpected)
    payload = worker_payload(experiment)
    if kind == "overlap":
        payload["test_idx"] = np.arange(12)
    elif kind == "range":
        payload["test_idx"] = np.array([24])
    elif kind == "dtype":
        payload["test_idx"] = np.array([12.0])
    elif kind == "duplicate":
        payload["test_idx"] = np.array([12, 12])
    elif kind == "unaligned":
        payload["y"] = payload["y"].set_axis(np.arange(1, 25))
    elif kind == "class":
        payload["y"] = payload["y"].copy()
        payload["y"].iloc[12:] = "unseen"
    elif kind == "group":
        payload["groups"] = np.arange(24)
    elif kind == "task":
        payload["task"] = "regression"
    else:
        payload["split_sha256"] = "bad"
    with pytest.raises(ValueError):
        study.run_case(payload, experiment.output.parent)


def test_worker_top_level_logs_errors_and_writes_both_unscored_rows(experiment, monkeypatch, caplog):
    payload = worker_payload(experiment)
    input_path = experiment.output.parent / "input.pkl"
    output_path = experiment.output.parent / "output.json"
    input_path.write_bytes(pickle.dumps(payload))

    def fail(*args):
        raise RuntimeError("mock core failure")

    monkeypatch.setattr(study, "run_case", fail)
    assert study.main(["--worker", str(input_path), "--worker-output", str(output_path)]) == 0
    rows = json.loads(output_path.read_text())
    assert len(rows) == 2 and all(row["status"] == "error" for row in rows)
    assert all("mock core failure" in row["error"] and "loss" not in row for row in rows)
    assert "Calibration bundle worker failed" in caplog.text


def test_isolated_worker_protocol_rejects_corrupt_synthetic_payload_without_fitting(experiment, monkeypatch):
    monkeypatch.setattr(tabular, "ROOT", REPOSITORY_ROOT)
    payload = worker_payload(experiment)
    data_path = experiment.output.parent / "corrupt-worker-data.pkl"
    data_path.write_bytes(pickle.dumps(experiment.data))
    payload.update(data_path=str(data_path), prepared_sha256="not-the-prepared-hash")
    directory = experiment.output.parent / "isolated-worker"
    # Exercise payload validation, not interpreter/import speed under coverage.
    rows = REAL_RUN_ISOLATED(payload, directory, 120.0, worker_module="benchmarks.calibration_evaluation")
    assert len(rows) == 2
    assert all(row["status"] == "error" and "Prepared dataset changed" in row["error"] for row in rows)
    assert all(row["worker_wall_seconds"] > 0 and "loss" not in row for row in rows)
    assert all(row["worker_python_hash_seed"] == payload["seed"] for row in rows)
    assert "Calibration bundle worker failed" in (directory / "worker.log").read_text()


def test_calibration_contrast_uses_complete_family_means_not_independent_rows(experiment):
    rows = []
    for family, plain_losses, calibrated_losses in (
        ("a", [2.0, 4.0], [1.0, 2.0]),
        ("b", [4.0, 8.0], [4.0, 8.0]),
        ("c", [2.0, 2.0], [1.0, 1.0]),
    ):
        for fold in range(2):
            for method, values in ((study.METHODS[2], plain_losses), (study.METHODS[3], calibrated_losses)):
                rows.append(
                    {
                        "dataset": family,
                        "family_id": family,
                        "learner": "linear",
                        "method": method,
                        "seed": 42,
                        "fold": fold,
                        "split_sha256": f"split-{fold}",
                        "loss": values[fold],
                        "brier_sum": values[fold] / 10,
                        "status": "error" if family == "c" and fold == 1 and method == study.METHODS[3] else "success",
                    }
                )
    summary = study._calibration_contrasts(rows, {"learners": ["linear"], "seeds": [42], "folds": 2})[0]
    assert summary["requested_families"] == 3
    assert summary["complete_paired_families"] == 2
    assert summary["requested_prediction_rows"] == 12
    assert summary["mean_loss_difference"] == pytest.approx(-0.75)
    assert summary["mean_brier_sum_difference"] == pytest.approx(-0.075)
    assert summary["mean_loss_reduction"] == pytest.approx(0.25)


def test_report_matches_fe_to_raw_inside_each_architecture(experiment, monkeypatch):
    losses = dict(zip(study.METHODS, [1.0, 0.8, 0.8, 0.6, 0.4, 0.3], strict=True))

    def run(payload, *args, **kwargs):
        return [{**row, "loss": losses[row["method"]]} for row in success(payload)]

    monkeypatch.setattr(tabular, "run_isolated", run)
    assert study.main(experiment.argv) == 0
    report = results(experiment)
    matched = {row["architecture"]: row for row in report["matched_architecture_summary"]}
    assert matched["single"]["mean_loss_reduction"] == pytest.approx(0.2)
    assert matched["plain-ensemble"]["mean_loss_reduction"] == pytest.approx(0.5)
    assert matched["sigmoid-ensemble"]["mean_loss_reduction"] == pytest.approx(0.5)
    assert all(row["reference_method"] == study.METHODS[0] for row in report["summary"])
    assert all(row["mean_loss_reduction"] == pytest.approx(0.25) for row in report["calibration_operation_summary"])


@pytest.mark.parametrize("mode", ["cv3", "subsample85"])
def test_sampling_mode_is_bound_to_every_surface_and_report(experiment, mode):
    e = experiment
    argv = e.argv if mode == "cv3" else [*e.argv, "--ensemble-sampling", mode]
    assert study.main(argv) == 0
    manifest = json.loads((e.output / "manifest.json").read_text())
    state = json.loads((e.output / "state.json").read_text())
    report = results(e)
    identity = study._sampling_identity({"ensemble_sampling": mode})
    assert "cycle_budget_context" not in manifest
    assert "parent-ledger authoritative" in manifest["budget_scope"]
    for surface in [manifest, state["identity"], report, *report["rows"], *state_rows(e)]:
        assert all(surface[key] == value for key, value in identity.items())
    assert len(report["rows"]) == 12 and len(state_rows(e)) == 8
    for payload, kwargs in e.calls:
        assert all(payload[key] == value for key, value in identity.items())
        assert payload["config"]["ensemble_sampling"] == mode
        if payload["job_kind"].startswith("single"):
            assert kwargs == {}
    for row in report["rows"]:
        assert row["diagnostic_only"] == (
            mode == "subsample85" and row["method"] in (study.METHODS[3], study.METHODS[5])
        )
    for item in [*report["summary"], *report["matched_architecture_summary"], *report["calibration_operation_summary"]]:
        assert all(item[key] == value for key, value in identity.items())
        assert item["diagnostic_only"] == (
            mode == "subsample85" and item["method"] in (study.METHODS[3], study.METHODS[5])
        )
    text = (e.output / "REPORT.md").read_text()
    assert len(report["sampling_diagnostics"]) == 4
    if mode == "cv3":
        assert "Three-fold sigmoid calibration" in text
        assert "requested_train_fraction" not in report["sampling_diagnostics"][0]["diagnostics"]
        assert "diagnostic-only" not in text
    else:
        assert "repeated stratified training bags" in text
        assert "not a three-fold partition or pooled OOF sample" in text
        assert "diagnostic-only" in text
        assert "[8, 2, 2, 0]" in text
        assert str(10 / 12) in text and str(2 / 12) in text
        diagnostics = report["sampling_diagnostics"][0]["diagnostics"]
        assert diagnostics["pairwise_overlap"] == bundle_report(mode)["diagnostics"]["pairwise_overlap"]
        assert diagnostics["distinct_train_bag_count"] == 3
    assert study.main(e.resume) == 0
    assert len(e.calls) == 8


@pytest.mark.parametrize("original,changed", [("cv3", "subsample85"), ("subsample85", "cv3")])
def test_resume_cannot_change_sampling_mode(experiment, original, changed):
    e = experiment
    assert study.main([*e.argv, "--ensemble-sampling", original]) == 0
    acquisitions = len(e.acquisitions)
    assert study.main([*e.resume, "--ensemble-sampling", changed]) == 2
    assert len(e.acquisitions) == acquisitions
    assert len(e.calls) == 8
    assert results(e)["ensemble_sampling"] == original


@pytest.mark.parametrize("mode", ["unknown", "", None, True, ["cv3", "subsample85"]])
def test_invalid_sampling_config_fails_before_acquisition(experiment, monkeypatch, mode):
    monkeypatch.setitem(study.DEFAULTS, "ensemble_sampling", mode)
    assert study.main(experiment.argv) == 2
    assert not experiment.acquisitions and not experiment.output.exists()


def test_invalid_sampling_cli_has_no_acquisition(experiment):
    with pytest.raises(SystemExit) as error:
        study.main([*experiment.argv, "--ensemble-sampling", "grid"])
    assert error.value.code == 2
    assert not experiment.acquisitions and not experiment.output.exists()


@pytest.mark.parametrize("mode", ["cv3", "subsample85"])
@pytest.mark.parametrize("mismatch", ["mode", "protocol", "missing_mode", "row_mode", "row_protocol"])
def test_bundle_mode_mismatch_never_checkpoints_scores(experiment, monkeypatch, mode, mismatch):
    def run(payload, *args, **kwargs):
        rows = success(payload)
        if payload["job_kind"].startswith("bundle"):
            other = "subsample85" if mode == "cv3" else "cv3"
            if mismatch == "mode":
                rows[0]["shared_bundle_diagnostics"]["ensemble_sampling"] = other
            elif mismatch == "protocol":
                rows[0]["shared_bundle_diagnostics"]["protocol"] = study.CORE_PROTOCOLS[other]
            elif mismatch == "missing_mode":
                rows[0]["shared_bundle_diagnostics"].pop("ensemble_sampling")
            elif mismatch == "row_mode":
                rows[0]["ensemble_sampling"] = other
            else:
                rows[0]["protocol"] = study.SAMPLING_PROTOCOLS[other]
        return rows

    monkeypatch.setattr(tabular, "run_isolated", run)
    assert study.main([*experiment.argv, "--ensemble-sampling", mode]) == 1
    rows = results(experiment)["rows"]
    bundles = [row for row in rows if row["job_kind"].startswith("bundle")]
    assert len(rows) == 12 and len(bundles) == 8
    assert all(row["status"] == "error" and "loss" not in row and row["ensemble_sampling"] == mode for row in bundles)
    assert all(row["status"] == "success" for row in rows if row["job_kind"].startswith("single"))


@pytest.mark.parametrize("mode", ["cv3", "subsample85"])
def test_worker_rejects_core_report_in_wrong_sampling_mode(experiment, monkeypatch, mode):
    from benchmarks import prediction_ensemble

    other = "subsample85" if mode == "cv3" else "cv3"
    monkeypatch.setattr(prediction_ensemble, "fit_prediction_bundle", lambda *args, **kwargs: bundle_report(other))
    payload = worker_payload(experiment)
    payload["config"]["ensemble_sampling"] = mode
    rows = study.run_case(payload, experiment.output.parent)
    assert len(rows) == 2
    assert all(row["status"] == "error" and "loss" not in row and row["ensemble_sampling"] == mode for row in rows)


@pytest.mark.parametrize("failure", ["dataset_error", "budget_exhausted", "interrupted", "timeout", "memory_limit"])
def test_unscored_rows_always_retain_sampling_identity(experiment, monkeypatch, failure):
    if failure in ("dataset_error", "interrupted"):

        def prepare(*args):
            if failure == "interrupted":
                raise KeyboardInterrupt
            raise ValueError("mock preparation failure")

        monkeypatch.setattr(execution, "_prepare_dataset", prepare)
    elif failure in ("timeout", "memory_limit"):
        monkeypatch.setattr(
            tabular,
            "run_isolated",
            lambda payload, *args, **kwargs: tabular.failure_rows(payload, failure, "mock limit", 0.1),
        )
    args = [*experiment.argv, "--ensemble-sampling", "subsample85"]
    if failure == "budget_exhausted":
        args += ["--total-seconds", "1"]
    assert study.main(args) == 1
    rows = results(experiment)["rows"]
    assert len(rows) == 12
    assert all(row["status"] == failure and row["ensemble_sampling"] == "subsample85" for row in rows)
    assert all(row["protocol"] == study.SAMPLING_PROTOCOLS["subsample85"] and "loss" not in row for row in rows)


@pytest.mark.parametrize("mutation", ["mixed_mode", "mixed_protocol", "unlabelled"])
def test_report_refuses_silent_cross_mode_aggregation(experiment, mutation):
    e = experiment
    assert study.main(e.argv) == 0
    rows = results(e)["rows"]
    if mutation == "mixed_mode":
        rows[-1]["ensemble_sampling"] = "subsample85"
    elif mutation == "mixed_protocol":
        rows[-1]["protocol"] = study.SAMPLING_PROTOCOLS["subsample85"]
    else:
        rows[-1].pop("ensemble_sampling")
    manifest = json.loads((e.output / "manifest.json").read_text())
    before = (e.output / "results.json").read_bytes()
    with pytest.raises(ValueError, match="one explicit sampling mode"):
        study.write_report(e.output, manifest, rows, state_rows(e))
    assert (e.output / "results.json").read_bytes() == before


def test_requested_subsample85_plan_has_324_jobs_486_outcomes_without_models(experiment, monkeypatch):
    e = experiment
    specs = [
        {**copy.deepcopy(e.spec), "name": name, "family_id": name.removeprefix("uci-")}
        for name in study.DEFAULT_DATASETS
    ]
    e.suite["datasets"] = specs
    e.canonical.clear()
    e.canonical.update({spec["name"]: spec for spec in specs})
    monkeypatch.setattr(
        execution,
        "_splits",
        lambda data, task, config, official: [
            (seed, fold, np.arange(12), np.arange(12, 24))
            for seed in config["seeds"]
            for fold in range(config["folds"])
        ],
    )
    assert (
        study.main(
            [
                *e.argv,
                "--datasets",
                *study.DEFAULT_DATASETS,
                "--ensemble-sampling",
                "subsample85",
                "--learners",
                "lightgbm",
                "--folds",
                "3",
                "--seeds",
                "42",
                "137",
                "2029",
                "--total-seconds",
                "57600",
                "--prepare-only",
            ]
        )
        == 0
    )
    manifest = json.loads((e.output / "manifest.json").read_text())
    assert manifest["expected_physical_jobs"] == 324 and manifest["planned_cells"] == 486
    assert manifest["budget_seconds"] == 57600
    assert manifest["cohort_scope"] == "canonical_nine"
    assert len(results(e)["rows"]) == 486
    assert all(row["ensemble_sampling"] == "subsample85" and row["status"] == "prepared" for row in results(e)["rows"])
    assert not e.calls and not e.admissions


def pin_confirmation_specs(experiment):
    experiment.suite["preregistration"]["dataset_spec_sha256"] = {
        spec["name"]: fingerprint({key: value for key, value in spec.items() if not key.startswith("_")})
        for spec in experiment.suite["datasets"]
    }


@pytest.fixture
def confirmation(experiment, monkeypatch):
    e = experiment
    boundary = {
        "evaluation_tier": study.CONFIRMATION_TIER,
        "claim_scope": "Row-level standard public IID only; no patient/entity-independent or clinical claims.",
        **study.CLAIM_FLAGS,
    }
    e.suite["evaluation_tier"] = study.CONFIRMATION_TIER
    e.suite["claim_boundary"] = boundary
    e.suite["datasets"] = [
        {
            **copy.deepcopy(e.spec),
            "name": family,
            "family_id": family,
            "role": "confirmation",
            "evaluation_tier": study.CONFIRMATION_TIER,
            "claim_boundary": boundary,
            "_family": {
                "family_id": family,
                "role": "confirmation",
                "evaluation_tier": study.CONFIRMATION_TIER,
                "reservation": {"status": "released"},
            },
        }
        for family in study.CONFIRMATION_FAMILIES
    ]
    e.suite["preregistration"] = {
        "status": "locked_before_confirmation_model_execution",
        "controller_protocol": study.CONFIRMATION_PROTOCOL,
        "evaluation_tier": study.CONFIRMATION_TIER,
        "claim_boundary": boundary,
        "dataset_names": list(study.CONFIRMATION_FAMILIES),
        "family_ids": list(study.CONFIRMATION_FAMILIES),
        "methods": list(study.CONFIRMATION_METHODS),
        "learners": ["linear", "lightgbm"],
        "learner_search": {
            "linear": {"regularization": [0.1, 1.0, 10.0], "selection_estimator_regularization": 1.0},
            "lightgbm": {"num_leaves": [7, 15, 31], "selection_estimator_num_leaves": 15},
        },
        "selection_parameters": {
            "max_candidates": 1000,
            "max_added_features": 50,
            "stability_repeats": 3,
            "screening": "auto",
            "ensemble_sampling": "subsample85",
            "numeric_clip_quantiles": [0.0, 1.0],
        },
        "resplit_official_as_iid": True,
        "split_plan": {family: {"policy": "iid", "folds": 3} for family in study.CONFIRMATION_FAMILIES},
        "seeds": [2030],
        "sample_seed": 2026,
        "folds": 3,
        "max_rows": 0,
        "threads": 8,
        "memory_limit_mb": 4096,
        "case_timeout_seconds": 900,
        "total_active_seconds_ceiling": 36000,
        "expected_cells": 96,
        "expected_physical_jobs": 72,
        "classification_admission_margin": {"min_total": 30, "min_outer_train": 20, "min_outer_test": 10},
        "primary_method_by_learner": dict(study.PRIMARY_METHODS),
        "calibrated_outcomes_diagnostic_only": True,
        "reporting_and_acceptance": {
            "evaluation_tier": study.CONFIRMATION_TIER,
            "frozen_primary_method_by_learner": dict(study.PRIMARY_METHODS),
            "primary_loss": "log_loss",
            "primary_raw_comparators": [study.METHODS[0], study.METHODS[2]],
            "native_fe_comparators": list(study.CONFIRMATION_METHODS[6:]),
            "acceptance": dict(study.CONFIRMATION_ACCEPTANCE),
        },
    }
    pin_confirmation_specs(e)
    e.data.update(X=pd.DataFrame({"x0": np.arange(90.0)}), y=pd.Series(["zeta", "alpha"] * 45))
    monkeypatch.setattr(execution, "_splits", REAL_SPLITS)
    e.checks = []

    class Ledger:
        def __init__(self, path):
            e.admissions.append(("created", str(path)))

        def check(self, *args):
            assert not e.acquisitions
            e.checks.append(args)

        def admit(self, *args):
            assert len(e.acquisitions) == 2
            e.admissions.append(args)

    monkeypatch.setattr(study, "ExposureLedger", Ledger)
    e.argv = [
        "--output",
        str(e.output),
        "--suite",
        str(e.output.parent / "suite.json"),
        "--allow-confirmation",
    ]
    return e


def test_locked_confirmation_has_72_jobs_96_tiered_outcomes_and_native_dispatch(confirmation):
    e = confirmation
    assert study.main(e.argv) == 0
    result = results(e)
    assert len(e.calls) == result["expected_physical_jobs"] == 72
    assert len(result["rows"]) == result["planned_cells"] == 96
    assert len(e.checks) == 2
    assert len(e.admissions) == 3
    assert all(entry[-1] == "confirmation" for entry in e.admissions[1:])
    native_calls = [
        (payload, kwargs) for payload, kwargs in e.calls if payload["method"] in study.CONFIRMATION_METHODS[6:]
    ]
    assert len(native_calls) == 24
    assert all(kwargs == {} for payload, kwargs in native_calls)
    manifest = json.loads((e.output / "manifest.json").read_text())
    assert manifest["budget_seconds"] == 36000
    assert manifest["config"]["numeric_clip_quantiles"] == [0.0, 1.0]
    assert manifest["cohort_scope"] == "locked_qualified_confirmation"
    for row in [
        *result["rows"],
        *result["summary"],
        *result["matched_raw85_summary"],
        *result["matched_architecture_summary"],
        *result["calibration_operation_summary"],
    ]:
        assert row["evaluation_tier"] == study.CONFIRMATION_TIER
        assert row["claim_boundary"] == e.suite["claim_boundary"]
        assert row["protocol"] == study.CONFIRMATION_PROTOCOL
    assert all(row["reference_method"] == study.METHODS[0] for row in result["summary"])
    assert all(row["reference_method"] == study.METHODS[2] for row in result["matched_raw85_summary"])
    assert all(row["role"] == "confirmation" for row in result["rows"])
    assert sum(row["primary_predeclared"] for row in result["rows"]) == 12
    assert sum(row["diagnostic_only"] for row in result["rows"]) == 24
    assert all(
        row["primary_predeclared"] == (row["method"] == study.PRIMARY_METHODS[row["learner"]]) for row in result["rows"]
    )
    text = (e.output / "REPORT.md").read_text()
    assert "No patient/entity-independent" in text and "No robust guarantee" in text
    assert "native-width single-model" in text


@pytest.mark.parametrize(
    "key,value",
    [
        ("status", "draft"),
        ("controller_protocol", study.PROTOCOL),
        ("evaluation_tier", "verified_entity"),
        ("family_ids", ["haberman-survival"]),
        ("dataset_names", ["haberman-survival"]),
        ("methods", list(study.METHODS)),
        ("learners", ["linear"]),
        ("seeds", [2031]),
        ("max_rows", 5000),
        ("folds", 2),
        ("threads", 4),
        ("memory_limit_mb", 2048),
        ("sample_seed", 42),
        ("case_timeout_seconds", 600),
        ("total_active_seconds_ceiling", 43200),
        ("expected_cells", 72),
        ("expected_physical_jobs", 96),
        ("resplit_official_as_iid", False),
        ("classification_admission_margin", {"min_total": 29, "min_outer_train": 20, "min_outer_test": 10}),
        ("primary_method_by_learner", {"linear": study.METHODS[4], "lightgbm": study.METHODS[1]}),
        ("calibrated_outcomes_diagnostic_only", False),
        ("selection_parameters", {}),
        ("learner_search", {}),
    ],
)
def test_confirmation_registration_mismatch_is_pre_acquisition(confirmation, key, value):
    e = confirmation
    e.suite["preregistration"][key] = value
    assert study.main(e.argv) == 2
    assert not e.acquisitions and not e.calls and not e.admissions
    assert not e.output.exists()


@pytest.mark.parametrize(
    "args",
    [
        ["--datasets", "haberman-survival"],
        ["--learners", "linear"],
        ["--ensemble-sampling", "cv3"],
        ["--seeds", "42"],
        ["--folds", "2"],
        ["--max-rows", "5000"],
        ["--threads", "4"],
        ["--memory-limit-mb", "2048"],
        ["--case-timeout", "899"],
        ["--total-seconds", "36001"],
    ],
)
def test_confirmation_cli_cannot_narrow_or_override_locked_design(confirmation, args):
    assert study.main([*confirmation.argv, *args]) == 2
    assert not confirmation.acquisitions and not confirmation.output.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "no_opt_in",
        "no_lock",
        "missing_family",
        "new_family",
        "role",
        "tier",
        "schema",
        "active",
        "group",
        "time",
        "claim",
    ],
)
def test_qualified_cohort_source_tier_and_reservation_gates(confirmation, mutation):
    e = confirmation
    args = list(e.argv)
    spec = e.suite["datasets"][1]
    if mutation == "no_opt_in":
        args.remove("--allow-confirmation")
    elif mutation == "no_lock":
        e.suite.pop("preregistration")
    elif mutation == "missing_family":
        e.suite["datasets"].pop()
    elif mutation == "new_family":
        spec.update(name="new-source", family_id="new-source")
    elif mutation == "role":
        spec["role"] = "development"
    elif mutation == "tier":
        spec["evaluation_tier"] = "verified_entity"
    elif mutation == "schema":
        spec["source"]["columns"] = ["new-predictor"]
    elif mutation == "active":
        spec["_family"]["reservation"]["status"] = "active"
    elif mutation in ("group", "time"):
        spec[f"{mutation}_column"] = "known-boundary"
    else:
        spec["claim_boundary"]["clinical_generalization_claims_allowed"] = True
    assert study.main(args) == 2
    assert not e.acquisitions and not e.calls and not e.admissions


def test_confirmation_preparation_failures_retain_96_tiered_rows(confirmation, monkeypatch):
    monkeypatch.setattr(
        execution, "_prepare_dataset", lambda *args: (_ for _ in ()).throw(ValueError("mock source failed"))
    )
    assert study.main(confirmation.argv) == 1
    result = results(confirmation)
    assert len(state_rows(confirmation)) == 72 and len(result["rows"]) == 96
    assert all(
        row["status"] == "dataset_error" and row["evaluation_tier"] == study.CONFIRMATION_TIER for row in result["rows"]
    )
    assert all(row["requested_families"] == 2 and row["complete_paired_families"] == 0 for row in result["summary"])
    assert len(confirmation.admissions) == 1 and not confirmation.calls


@pytest.mark.parametrize("boundary", ["groups", "timestamps", "label_end_times"])
def test_confirmation_rejects_actual_known_boundary_metadata(confirmation, boundary):
    confirmation.data[boundary] = pd.Series(np.arange(90))
    assert study.main(confirmation.argv) == 1
    assert not confirmation.calls and len(results(confirmation)["rows"]) == 96


@pytest.mark.parametrize("counts,passes", [([30, 30], True), ([29, 60], False), ([90], False)])
def test_confirmation_actual_class_margins_not_metadata_claims(confirmation, counts, passes):
    y = np.repeat(np.arange(len(counts)), counts)
    confirmation.data.update(X=pd.DataFrame({"x0": np.arange(len(y))}), y=pd.Series(y))
    assert study.main([*confirmation.argv, "--prepare-only"]) == (0 if passes else 1)
    assert not confirmation.calls and not confirmation.admissions


@pytest.mark.parametrize("train,test", [(np.arange(40), np.arange(40, 58)), (np.arange(38), np.arange(38, 60))])
def test_confirmation_actual_outer_train_test_class_margins(confirmation, monkeypatch, train, test):
    confirmation.data.update(X=pd.DataFrame({"x0": np.arange(60)}), y=pd.Series([0, 1] * 30))
    monkeypatch.setattr(execution, "_splits", lambda *args: [(2030, fold, train, test) for fold in range(3)])
    assert study.main([*confirmation.argv, "--prepare-only"]) == 1
    assert not confirmation.calls
    assert all("class support" in row["error"] for row in results(confirmation)["rows"])


def test_confirmation_exposure_check_blocks_entire_cohort_before_payload(confirmation, monkeypatch):
    class BlockedLedger:
        def __init__(self, path):
            pass

        def check(self, family, *args):
            if family == study.CONFIRMATION_FAMILIES[1]:
                raise ValueError("Already exposed")

    monkeypatch.setattr(study, "ExposureLedger", BlockedLedger)
    assert study.main(confirmation.argv) == 2
    assert not confirmation.acquisitions and not confirmation.calls


def test_confirmation_prepare_only_read_only_exposure_check(confirmation):
    import sqlite3

    path = tabular.ROOT / "benchmarks" / ".cache" / "exposure.sqlite"
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE exposure(family_id TEXT, run_id TEXT)")
        connection.execute("INSERT INTO exposure VALUES (?, ?)", (study.CONFIRMATION_FAMILIES[0], "other-run"))
    before = path.read_bytes()
    assert study.main([*confirmation.argv, "--prepare-only"]) == 2
    assert path.read_bytes() == before
    assert not confirmation.acquisitions and not confirmation.calls and not confirmation.admissions


@pytest.mark.parametrize("mutation", ["role", "tier", "source", "config"])
def test_confirmation_resume_rejects_changed_role_tier_source_or_config(confirmation, mutation):
    e = confirmation
    assert study.main([*e.argv, "--prepare-only"]) == 0
    args = list(e.resume)
    if mutation == "config":
        args += ["--seeds", "2031"]
    else:
        spec = e.suite["datasets"][0]
        if mutation == "source":
            spec["source"]["sha256"] = "changed"
        elif mutation == "tier":
            spec["evaluation_tier"] = "verified_entity"
        else:
            spec["role"] = "development"
    acquired = len(e.acquisitions)
    assert study.main(args) == 2
    assert len(e.acquisitions) == acquired and not e.calls


def test_confirmation_official_ids_survive_locked_iid_resplitting(confirmation, monkeypatch):
    e = confirmation
    for spec in e.suite["datasets"]:
        spec["split_policy"] = "official"
        spec["source"]["official_split_dimensions"] = {"repeats": 1, "folds": 2, "samples": 1}
    pin_confirmation_specs(e)
    original = execution._prepare_dataset
    ids = [{"repeat": 0, "fold": fold, "sample": 0} for fold in range(2)]

    def prepare(*args):
        data, metadata, _ = original(*args)
        metadata["official_split_ids"] = ids
        metadata["official_splits_sha256"] = "fixed-official-splits"
        return data, metadata, [(np.arange(45), np.arange(45, 90)), (np.arange(45, 90), np.arange(45))]

    monkeypatch.setattr(execution, "_prepare_dataset", prepare)
    assert study.main([*e.argv, "--prepare-only"]) == 0
    manifest = json.loads((e.output / "manifest.json").read_text())
    assert len(manifest["splits"]) == 6
    assert all(row["seed"] == 2030 for row in manifest["splits"])
    assert all(
        spec["official_split_ids"] == ids
        and spec["official_splits_sha256"] == "fixed-official-splits"
        and spec["used_split_policy"] == "iid"
        and spec["selected_official_split_ids"] == []
        for spec in manifest["datasets"]
    )
    assert not e.calls and not e.admissions


@pytest.mark.parametrize("mutation", ["tier", "claim", "role"])
def test_confirmation_reports_cannot_pool_verified_tier_or_relax_claims(confirmation, mutation):
    e = confirmation
    assert study.main([*e.argv, "--prepare-only"]) == 0
    manifest = json.loads((e.output / "manifest.json").read_text())
    rows = results(e)["rows"]
    if mutation == "tier":
        rows[0]["evaluation_tier"] = "verified_entity"
    elif mutation == "claim":
        rows[0]["claim_boundary"]["patient_entity_disjoint_claims_allowed"] = True
    else:
        rows[0]["role"] = "development"
    with pytest.raises(ValueError, match="sampling mode|pooled"):
        study.write_report(e.output, manifest, rows, [])


def test_confirmation_resume_reuses_successes_and_rechecks_raw_source(confirmation, monkeypatch):
    e = confirmation
    assert study.main(e.argv) == 0
    original = execution._prepare_dataset
    checks = []

    def prepare(*args):
        checks.append(args[0]["name"])
        return original(*args)

    class Ledger:
        def __init__(self, path):
            pass

        def check(self, family, run_id, role):
            assert role == "confirmation"

        def admit(self, *args):
            pytest.fail("Completed physical jobs must not be admitted/evaluated again")

    monkeypatch.setattr(execution, "_prepare_dataset", prepare)
    monkeypatch.setattr(study, "ExposureLedger", Ledger)
    assert study.main(e.resume) == 0
    assert checks == list(study.CONFIRMATION_FAMILIES)
    assert len(e.calls) == 72 and len(results(e)["rows"]) == 96


@pytest.fixture
def decision_probe(confirmation):
    e = confirmation
    config = {
        **copy.deepcopy(study.DEFAULTS),
        **copy.deepcopy(study.CONFIRMATION_SETTINGS),
        "allow_confirmation": True,
        "claim_boundary": copy.deepcopy(e.suite["claim_boundary"]),
    }
    manifest = {
        **study._sampling_identity(config),
        "config": config,
        "preregistration": copy.deepcopy(e.suite["preregistration"]),
        "run_id": "synthetic-decision-probe",
        "expected_physical_jobs": 72,
        "planned_cells": 96,
        "cohort_scope": "locked_qualified_confirmation",
        "splits": [],
    }
    rows, jobs = [], []
    for family in study.CONFIRMATION_FAMILIES:
        for fold in range(3):
            split_hash = f"{family}-2030-{fold}"
            manifest["splits"].append({"dataset": family, "seed": 2030, "fold": fold, "sha256": split_hash})
            for learner in ("linear", "lightgbm"):
                for kind, methods in study.CONFIRMATION_JOBS.items():
                    job_id = study.RunState.cell_id(family, 2030, fold, kind, learner)
                    payload = {
                        **study._sampling_identity(config),
                        "dataset": family,
                        "family_id": family,
                        "role": "confirmation",
                        "source": "real_world",
                        "task": "classification",
                        "method": methods[0] if len(methods) == 1 else kind,
                        "job_kind": kind,
                        "physical_job_id": job_id,
                        "seed": 2030,
                        "fold": fold,
                        "split_sha256": split_hash,
                        "train_idx": np.arange(60),
                        "test_idx": np.arange(60, 90),
                        "expected_cells": 3,
                        "dataset_metadata": {"name": family},
                        "prepared_sha256": "synthetic-prepared-hash",
                        "config": {**config, "learners": [learner]},
                    }
                    outcomes = success(payload)
                    for row in outcomes:
                        row["loss"] = 0.5 if row["method"] == study.PRIMARY_METHODS[learner] else 1.0
                    rows.extend(outcomes)
                    jobs.append(
                        {"physical_job_id": job_id, "job_kind": kind, "status": "success", "outcomes": outcomes}
                    )
    return manifest, rows, jobs


def test_decision_rules_match_actual_immutable_released_preregistration():
    path = REPOSITORY_ROOT / "benchmarks" / "suites" / "standard_iid_policy_confirmation_20261003.json"
    registration = json.loads(path.read_text())["preregistration"]
    rules = study._confirmation_rules(registration)
    assert rules["acceptance"] == study.CONFIRMATION_ACCEPTANCE
    assert rules["primary_raw_comparators"] + rules["native_fe_comparators"] == [
        study.METHODS[0],
        study.METHODS[2],
        *study.CONFIRMATION_METHODS[6:],
    ]


def test_locked_decision_rejects_harm_and_native_losses_despite_positive_raw_mean(decision_probe, tmp_path):
    manifest, rows, jobs = decision_probe
    for row in rows:
        if row["method"] == study.PRIMARY_METHODS[row["learner"]]:
            row["loss"] = 1.1 if row["family_id"] == study.CONFIRMATION_FAMILIES[0] else 0.5
        elif row["method"] in study.CONFIRMATION_METHODS[6:]:
            row["loss"] = 0.4
        elif row["method"] in (study.METHODS[3], study.METHODS[5]):
            row["loss"] = 0.001
    study.write_report(tmp_path, manifest, rows, jobs)
    decision = json.loads((tmp_path / "results.json").read_text())["confirmation_decision"]
    assert decision["status"] == "fail" and decision["valid_prediction_rows"] == 96
    assert decision["complete_physical_jobs"] == 72 and decision["requested_families"] == 2
    assert decision["automatic_sota"] is False and decision["automatic_default_promotion"] is False
    assert decision["primary_method_by_learner"] == study.PRIMARY_METHODS
    assert len(decision["comparisons"]) == 8 and len(decision["family_losses"]) == 32
    for comparison in decision["comparisons"]:
        assert comparison["requested_families"] == comparison["defined_relative_families"] == 2
        assert comparison["policy_method"] == study.PRIMARY_METHODS[comparison["learner"]]
        assert comparison["status"] == "fail"
        if comparison["reference_kind"] == "raw_control":
            assert comparison["mean_relative_gain"] == pytest.approx(0.2)
            assert comparison["families"][0]["relative_gain"] == pytest.approx(-0.1)
            assert comparison["families"][0]["raw_family_harm_passed"] is False
            assert comparison["raw_family_harm_gate_passed"] is False
        else:
            assert comparison["mean_relative_gain"] == pytest.approx(-1.0)
            assert comparison["mean_gain_gate_passed"] is False
    text = (tmp_path / "REPORT.md").read_text()
    assert "**FAIL**" in text and "raw family harm exceeds 5%" in text
    assert "96/96" in text and "72/72" in text
    assert "automatic SOTA claims or default promotion" in text
    assert all(
        row["method"] not in (study.METHODS[3], study.METHODS[5])
        for row in [{"method": comparison["policy_method"]} for comparison in decision["comparisons"]]
    )


def test_zero_raw_reference_blocks_without_reducing_the_family_denominator(decision_probe, tmp_path):
    manifest, rows, jobs = decision_probe
    for row in rows:
        if row["family_id"] == study.CONFIRMATION_FAMILIES[0] and row["method"] == study.METHODS[0]:
            row["loss"] = 0.0
    study.write_report(tmp_path, manifest, rows, jobs)
    decision = json.loads((tmp_path / "results.json").read_text())["confirmation_decision"]
    assert decision["status"] == "blocked" and decision["locked_criteria_passed"] is False
    assert decision["valid_prediction_rows"] == 96 and decision["complete_physical_jobs"] == 72
    for comparison in decision["comparisons"]:
        if comparison["reference_method"] != study.METHODS[0]:
            continue
        assert comparison["requested_families"] == comparison["complete_paired_families"] == 2
        assert comparison["defined_relative_families"] == 1
        assert comparison["mean_relative_gain"] is None and comparison["median_relative_gain"] is None
        assert comparison["status"] == "blocked"
        first, second = comparison["families"]
        assert first["reference_mean_loss"] == 0.0 and first["policy_mean_loss"] == 0.5
        assert first["relative_gain"] is None and first["undefined_reason"] == "undefined_zero_reference_loss"
        assert second["relative_gain"] == 0.5
    text = (tmp_path / "REPORT.md").read_text()
    assert "**BLOCKED**" in text and "all 2 registered families retained" in text
    assert "undefined_zero_reference_loss" in text


@pytest.mark.parametrize(
    "failure",
    [
        "missing_fold",
        "failed_reference",
        "nonfinite_reference",
        "nonfinite_metric",
        "diagnostic_failed",
        "missing_job",
        "failed_job",
        "duplicate",
        "split",
        "job_group",
    ],
)
def test_locked_decision_blocks_every_incomplete_invalid_or_mismatched_outcome(decision_probe, failure):
    manifest, rows, jobs = decision_probe
    target = next(row for row in rows if row["method"] == study.CONFIRMATION_METHODS[6] and row["fold"] == 0)
    if failure == "missing_fold":
        rows.remove(target)
    elif failure == "failed_reference":
        target.update(status="timeout", error="mock timeout")
    elif failure == "nonfinite_reference":
        target["loss"] = float("nan")
    elif failure == "nonfinite_metric":
        target["brier_sum"] = float("inf")
    elif failure == "diagnostic_failed":
        next(row for row in rows if row["method"] == study.METHODS[5])["status"] = "error"
    elif failure == "missing_job":
        jobs.pop()
    elif failure == "failed_job":
        jobs[-1]["status"] = "error"
    elif failure == "duplicate":
        rows.append(dict(target))
    elif failure == "split":
        target["split_sha256"] = "different"
    else:
        target["physical_job_id"] = "wrong-physical-job"
    decision = study._confirmation_decision(manifest, rows, jobs)
    assert decision["status"] == "blocked" and decision["locked_criteria_passed"] is False
    assert decision["requested_families"] == 2
    assert all(
        comparison["requested_families"] == 2 and len(comparison["families"]) == 2
        for comparison in decision["comparisons"]
    )
    assert decision["blockers"]
    fingerprint(decision)
    if failure in ("missing_fold", "failed_reference", "nonfinite_reference", "nonfinite_metric", "duplicate", "split"):
        comparison = next(
            item
            for item in decision["comparisons"]
            if item["learner"] == target["learner"] and item["reference_method"] == target["method"]
        )
        assert comparison["mean_relative_gain"] is None
        assert comparison["families"][0]["reference_mean_loss"] is None


def test_locked_decision_pass_is_scope_limited_without_automatic_promotion(decision_probe):
    decision = study._confirmation_decision(*decision_probe)
    assert decision["status"] == "pass" and decision["locked_criteria_passed"] is True
    assert decision["valid_prediction_rows"] == 96 and decision["complete_physical_jobs"] == 72
    assert decision["automatic_sota"] is False and decision["automatic_default_promotion"] is False
    assert decision["evaluation_tier"] == study.CONFIRMATION_TIER
    assert decision["claim_boundary"]["patient_entity_disjoint_claims_allowed"] is False
    assert all(comparison["mean_relative_gain"] == 0.5 for comparison in decision["comparisons"])


@pytest.mark.parametrize("harm,passes", [(1.05, True), (1.050001, False)])
def test_locked_family_harm_threshold_is_exactly_five_percent(decision_probe, harm, passes):
    manifest, rows, jobs = decision_probe
    for row in rows:
        if row["method"] == study.PRIMARY_METHODS[row["learner"]]:
            row["loss"] = harm if row["family_id"] == study.CONFIRMATION_FAMILIES[0] else 0.5
        elif row["method"] in study.CONFIRMATION_METHODS[6:]:
            row["loss"] = 2.0
    decision = study._confirmation_decision(manifest, rows, jobs)
    assert decision["status"] == ("pass" if passes else "fail")


def test_raw_mean_gate_is_strict_but_native_mean_gate_allows_equality(decision_probe):
    manifest, rows, jobs = decision_probe
    for row in rows:
        if row["method"] in study.CONFIRMATION_METHODS[6:]:
            row["loss"] = 0.5
    decision = study._confirmation_decision(manifest, rows, jobs)
    assert decision["status"] == "pass"
    assert all(
        comparison["mean_relative_gain"] == 0.0 and comparison["mean_gain_gate_passed"] is True
        for comparison in decision["comparisons"]
        if comparison["reference_kind"] == "native_fe"
    )
    for row in rows:
        if row["method"] in (study.METHODS[0], study.METHODS[2]):
            row["loss"] = 0.5
    decision = study._confirmation_decision(manifest, rows, jobs)
    assert decision["status"] == "fail"
    assert all(
        comparison["mean_relative_gain"] == 0.0
        and comparison["mean_gain_gate_passed"] is False
        and comparison["raw_family_harm_gate_passed"] is True
        for comparison in decision["comparisons"]
        if comparison["reference_kind"] == "raw_control"
    )


@pytest.mark.parametrize("rule", ["family_harm", "native_reference_gate", "automatic_default_promotion"])
def test_changed_locked_acceptance_is_rejected_before_acquisition(confirmation, rule):
    confirmation.suite["preregistration"]["reporting_and_acceptance"]["acceptance"][rule] = "altered"
    assert study.main(confirmation.argv) == 2
    assert not confirmation.acquisitions and not confirmation.output.exists()
