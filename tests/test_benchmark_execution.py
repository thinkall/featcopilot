"""Integration contracts for per-learner scheduling and exact resume."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from benchmarks import execution, tabular

REAL_SPLITS = execution._splits


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


def test_split_plan_prepares_exact_12_family_128_cell_cohort(mixed_confirmation, monkeypatch):
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
    monkeypatch.setattr(tabular, "run_isolated", lambda *a: pytest.fail("planning started a model"))
    assert (
        tabular.main(
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
        )
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert specs == source_specs
    assert len(manifest["datasets"]) == 12
    assert len(manifest["splits"]) == 16
    assert manifest["planned_cells"] == 128
    assert not (output / "results.json").exists()
    for dataset in manifest["datasets"][:10]:
        assert len(dataset["official_split_ids"]) == 30
        assert dataset["selected_official_split_ids"] == [{"repeat": 0, "fold": 0, "sample": 0}]
        assert dataset["selected_split_count"] == dataset["outer_split_count"] == 1


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


@pytest.mark.parametrize("failed", [False, True])
def test_historical_execution_uses_all_official_repeats_not_config_folds(mixed_confirmation, monkeypatch, failed):
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

    if failed:

        def unavailable(*args):
            raise ValueError("Unavailable historical fixture")

        monkeypatch.setattr(execution, "_prepare_dataset", unavailable)
    monkeypatch.setattr(tabular, "run_isolated", run)
    assert tabular.main(arguments) == int(failed)
    rows = json.loads((output / "results.json").read_text())["rows"]
    assert len(rows) == 8
    assert all(row["expected_cells"] == 4 for row in rows)
    assert {row["fold"] for row in rows} == {0, 1, 2, 3}
    assert len(calls) == (0 if failed else 8)
    manifest = json.loads((output / "manifest.json").read_text())
    assert "selected_split_count" not in manifest["datasets"][0]


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


@pytest.mark.parametrize("failed", [False, True])
def test_resplit_protocol_plans_exact_480_cells_even_on_failed_preparation(resplit_confirmation, monkeypatch, failed):
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
    if failed:

        def unavailable(*args):
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
            *([] if failed else ["--prepare-only"]),
        ]
    ) == int(failed)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["planned_cells"] == 480
    assert len(manifest["splits"]) == 30
    assert all(item["selected_split_count"] == 3 for item in manifest["datasets"])
    if failed:
        state = json.loads((output / "state.json").read_text())
        rows = [json.loads((output / "records" / record["file"]).read_text()) for record in state["records"].values()]
        assert len(rows) == 480
        for spec in specs:
            family_rows = [row for row in rows if row["family_id"] == spec["family_id"]]
            assert len(family_rows) == 48
            assert all(row["expected_cells"] == 3 and row["status"] == "dataset_error" for row in family_rows)


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
    output, arguments = experiment
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
