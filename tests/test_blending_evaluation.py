"""Mocked-only lifecycle and aggregate-output contracts for the blending study."""

from __future__ import annotations

import copy
import gzip
import json
import pickle
import sqlite3
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from benchmarks import blending_evaluation as study
from benchmarks import execution, tabular
from benchmarks.run_state import ExposureLedger, fingerprint


@pytest.fixture
def experiment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    spec = {
        "name": "mock-classifier",
        "family_id": "mock-family",
        "role": "development",
        "task": "classification",
        "provider": "uci",
        "origin_kind": "observational",
        "split_policy": "iid",
        "target_transform": "identity",
        "rights": {"status": "approved"},
        "source": {"sha256": "original-raw-pin", "schema_reviewed": True},
    }
    suite = {"schema_version": 1, "name": "mock-only", "datasets": [spec]}
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    canonical = {spec["name"]: copy.deepcopy(spec)}
    monkeypatch.setattr(study, "_canonical_specs", lambda: copy.deepcopy(canonical))
    monkeypatch.setattr(execution, "_load_suite", lambda *args: (copy.deepcopy(suite), False))
    monkeypatch.setattr(execution, "environment_versions", lambda: {"lightgbm": "mock-no-model"})
    monkeypatch.setattr(tabular, "ROOT", tmp_path)
    monkeypatch.setattr(tabular, "source_files", lambda: [])
    monkeypatch.setattr(tabular, "source_fingerprint", lambda: "frozen-source")
    data = {
        "X": pd.DataFrame({"x0": np.arange(36.0)}),
        "y": pd.Series([0, 1] * 18),
        "groups": None,
        "timestamps": None,
        "label_end_times": None,
    }
    calls: list = []
    acquisitions: list = []
    exposures: list = []

    def prepare(*args: Any) -> tuple[dict, dict, None]:
        acquisitions.append(args[0]["name"])
        return copy.deepcopy(data), {"sha256": "raw-pin", "review_notes": "retain-metadata"}, None

    def splits(data: dict, task: str, config: dict, official: Any) -> list:
        indices = np.arange(len(data["X"]))
        return [
            (seed, fold, np.setdiff1d(indices, test), test)
            for seed in config["seeds"]
            for fold, test in enumerate(np.array_split(indices, 3))
        ]

    monkeypatch.setattr(execution, "_prepare_dataset", prepare)
    monkeypatch.setattr(execution, "_splits", splits)

    class Ledger:
        def __init__(self, path: Path):
            exposures.append(("created", str(path)))

        def check(self, *args: Any) -> None:
            exposures.append(("check", *args))

        def admit(self, *args: Any) -> None:
            exposures.append(("admit", *args))

    monkeypatch.setattr(study, "ExposureLedger", Ledger)

    def run(payload: dict, directory: Path, timeout: float, **kwargs: Any) -> list[dict]:
        calls.append((copy.deepcopy(payload), directory, timeout, kwargs))
        return success(payload)

    monkeypatch.setattr(tabular, "run_isolated", run)
    output = tmp_path / "run"
    return SimpleNamespace(
        root=tmp_path,
        output=output,
        argv=["--output", str(output), "--suite", str(suite_path), "--datasets", spec["name"]],
        resume=["--output", str(output), "--resume"],
        suite=suite,
        spec=spec,
        canonical=canonical,
        data=data,
        calls=calls,
        acquisitions=acquisitions,
        exposures=exposures,
        prepare=prepare,
        splits=splits,
    )


@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("missing_manifest", [False, True])
def test_source_initialization_failure_resumes_without_repeating_time(
    experiment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, interrupted: bool, missing_manifest: bool
) -> None:
    e = experiment
    clock = [0.0]
    monkeypatch.setattr(study.time, "perf_counter", lambda: clock[0])

    def fail(*args: Any) -> Any:
        clock[0] += 60
        raise KeyboardInterrupt if interrupted else OSError("Synthetic source publication failure")

    with monkeypatch.context() as patch:
        patch.setattr(study, "archive_source", fail)
        assert study.main([*e.argv, "--total-seconds", "100"]) == 1
    assert state(e)["elapsed_seconds"] == manifest(e)["active_seconds"] == 60
    assert not records(e) and not e.calls and not e.acquisitions and not e.exposures
    if missing_manifest:
        (e.output / "manifest.json").unlink()
    before = (e.output / "state.json").read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(tabular, "source_fingerprint", lambda: "changed")
        assert study.main(e.resume) == 2
    assert (e.output / "state.json").read_bytes() == before
    run = tabular.run_isolated

    def timed(*args: Any, **kwargs: Any) -> list[dict]:
        clock[0] += 1
        return run(*args, **kwargs)

    monkeypatch.setattr(tabular, "run_isolated", timed)
    assert study.main(e.resume) == 0
    assert len(e.calls) == 3
    assert state(e)["elapsed_seconds"] == 63
    assert manifest(e)["initialization_status"] == "ready"


def base_report(engineered: bool, rows: int, fit_id: str, classes: int = 2, input_features: int = 1) -> dict:
    names = [*(f"x{index}" for index in range(input_features)), *(["fake_fe"] if engineered else [])]

    def partition(size: int) -> dict:
        return {
            "size": size,
            "class_counts": [size // classes + int(index < size % classes) for index in range(classes)],
            "indices_sha256": fingerprint(size),
        }

    return {
        "fit_identity_sha256": fit_id,
        "feature_names": names,
        "n_output_features": len(names),
        "feature_names_sha256": fingerprint(names),
        "schema_sha256": fingerprint("schema"),
        "formula_sha256": fingerprint(["formula", engineered]),
        "fitted_state_sha256": fingerprint(["fitted", fit_id]),
        "selected_parameters": {"num_leaves": 7},
        "trials": [
            {"parameters": parameters, "loss": 0.5, "n_output_features": len(names)}
            for parameters in tabular.candidate_parameters("lightgbm")
        ],
        "inner_split": {"train": partition(rows - 4), "validation": partition(4)},
    }


def bundle_report() -> dict:
    """Create aggregate-only mock metrics; no estimator fitting occurs."""
    timing = dict.fromkeys(study.TIMING_KEYS, 0.0)
    timing.update(full_raw_fit_seconds=1.0, full_fe_fit_seconds=2.0, oof_fit_seconds=10.0, oof_seconds=10.0)
    for arm, training in zip(study.ARMS, (1.0, 2.0, 3.0, 13.0), strict=True):
        timing[f"{arm}_training_seconds"] = timing[f"{arm}_standalone_seconds"] = training
    return {
        **{f"{arm}_metrics": dict.fromkeys(study.METRICS, 0.5) for arm in study.ARMS},
        "n_input_features": 1,
        "n_output_features": 2,
        "shared_cost_not_additive": True,
        "diagnostics": {
            "protocol": study.CORE_PROTOCOL,
            "task": "classification",
            "learner": "lightgbm",
            "split_kind": "iid",
            "seed": 42,
            "oof_seed": 1042,
            "sklearn_version": "mock",
            "class_counts": [12, 12],
            "n_oof_covered": 24,
            "oof_coverage_histogram": [0, 24],
            "member_count": 5,
            "full_fit_identity_sha256": {
                "raw": ["a" * 64],
                "fe": ["b" * 64],
                "average": ["a" * 64, "b" * 64],
                "blend": ["a" * 64, "b" * 64],
            },
            "full_models": {
                "raw": base_report(False, 24, "a" * 64),
                "fe": base_report(True, 24, "b" * 64),
            },
            "members": [
                {
                    "member": index,
                    "train": {
                        "size": 24 - size,
                        "class_counts": [(24 - size) // 2, 24 - size - (24 - size) // 2],
                        "indices_sha256": fingerprint(["train", index]),
                    },
                    "holdout": {
                        "size": size,
                        "class_counts": [size // 2, size - size // 2],
                        "indices_sha256": fingerprint(["holdout", index]),
                    },
                    "raw": base_report(False, 24 - size, fingerprint(["raw", index])),
                    "fe": base_report(True, 24 - size, fingerprint(["fe", index])),
                }
                for index, size in enumerate((5, 5, 5, 5, 4))
            ],
            "exact_shared_fits": True,
            "full_complete_procedure_fit_count": 2,
            "oof_complete_procedure_fit_count": 10,
            "weight": 0.3,
            "n_input_features": 1,
            "n_output_features": 2,
            "full_output_feature_counts": [1, 2],
            "oof_losses": {"raw": 0.6, "fe": 0.55, "average": 0.52, "blend": 0.51},
            "optimization": {
                "method": "bounded",
                "bounds": [0.0, 1.0],
                "xatol": 1e-5,
                "maxiter": 500,
                "tie_break": "smaller_weight_on_exact_equal_loss",
                "endpoint_candidates": [{"weight": 0.0, "loss": 0.6}, {"weight": 1.0, "loss": 0.55}],
                "epsilon": float(np.finfo(np.float64).eps),
                "success": True,
                "status": "converged",
                "nfev": 4,
                "interior_candidate": {"weight": 0.3, "loss": 0.51},
            },
            "criterion": study.CRITERION,
            "source_policy": copy.deepcopy(study.CORE_POLICY),
        },
        "timing": timing,
    }


def fixed_bundle_report() -> dict:
    """Return the exact three-arm fixed-core shape without invented OOF losses."""
    oof = bundle_report()
    diagnostics = {key: value for key, value in oof["diagnostics"].items() if key in study.FIXED_DIAGNOSTIC_KEYS}
    diagnostics.update(
        protocol=study.FIXED_CORE_PROTOCOL,
        blend_mode=study.FIXED_MODE,
        weight=0.5,
        weight_optimized=False,
        criterion="prespecified_probability_average_weight_0.5",
        source_policy=copy.deepcopy(study.FIXED_CORE_POLICY),
        member_count=0,
        members=[],
        oof_complete_procedure_fit_count=0,
    )
    diagnostics["full_fit_identity_sha256"].pop("blend")
    return {
        **{f"{arm}_metrics": oof[f"{arm}_metrics"] for arm in study.MODE_ARMS[study.FIXED_MODE]},
        "n_input_features": 1,
        "n_output_features": 2,
        "shared_cost_not_additive": True,
        "diagnostics": diagnostics,
        "timing": {key: value for key, value in oof["timing"].items() if key in study.FIXED_TIMING_KEYS},
    }


def success(payload: dict) -> list[dict]:
    if payload["method"] in study.NATIVE_REFERENCES:
        return native_success(payload)
    mode = payload["config"]["blend_mode"]
    fixed = mode == study.FIXED_MODE
    report = fixed_bundle_report() if fixed else bundle_report()
    if payload["config"].get("allow_confirmation"):
        width = len(payload["dataset_metadata"]["column_mapping"])
        count = len(payload["train_idx"])
        report.update(n_input_features=width, n_output_features=width + 1)
        report["diagnostics"].update(
            n_input_features=width,
            n_output_features=width + 1,
            full_output_feature_counts=[width, width + 1],
            class_counts=[count // 2, count - count // 2],
            full_models={
                "raw": base_report(False, count, "a" * 64, input_features=width),
                "fe": base_report(True, count, "b" * 64, input_features=width),
            },
        )
    keys = study.FIXED_DIAGNOSTIC_KEYS if fixed else study.DIAGNOSTIC_KEYS
    report["diagnostics"] = {key: report["diagnostics"][key] for key in keys}
    report["diagnostics"].update(study._provenance(report["diagnostics"], include_members=not fixed))
    report["diagnostics"]["seed"] = payload["seed"]
    if not fixed:
        report["diagnostics"]["oof_seed"] = payload["seed"] + 1000
    rows = study._outcomes(payload, "success", "")
    for arm, row in zip(study.MODE_ARMS[mode], rows, strict=True):
        row.pop("error")
        row.update(
            report[f"{arm}_metrics"],
            n_train=len(payload["train_idx"]),
            n_test=len(payload["test_idx"]),
            n_input_features=report["n_input_features"],
            n_output_features=report["n_input_features"] if arm == "raw" else report["n_output_features"],
            pipeline_seconds=report["timing"][f"{arm}_standalone_seconds"],
            shared_bundle_diagnostics=copy.deepcopy(report["diagnostics"]),
            shared_bundle_timing=copy.deepcopy(report["timing"]),
            worker_wall_seconds=0.001,
        )
    return rows


def native_success(payload: dict) -> list[dict]:
    width = len(payload["dataset_metadata"]["column_mapping"])
    method = payload["method"]
    reference = {
        "status": "success",
        "method": "openfe-inductive" if method == study.NATIVE_REFERENCES[0] else "autofeat-native",
        "task": "classification",
        "seed": payload["seed"],
        "threads": 8,
        "width_policy": "native_output",
        "max_added_features": None,
        "numeric_clip_quantiles": [0.0, 1.0],
        "clipping_scope": "final_matched_learner_only",
        "generation_preprocessing": "existing_training_median_fill",
        "original_numeric_missingness_restored": True,
        "input_features": width,
        "selected_count": 75,
        "added_features": 75,
        "preprocessing": {"one_hot_levels": {"x0": ["private-category"]}},
        "feature_definitions": [{"name": "native_feature", "expression": "x0*x1"}],
    }
    return [
        {
            **{
                key: payload[key]
                for key in (
                    "dataset",
                    "family_id",
                    "source",
                    "task",
                    "role",
                    "method",
                    "seed",
                    "fold",
                    "split_sha256",
                    "expected_cells",
                )
            },
            **dict.fromkeys(study.METRICS, 0.5),
            "learner": "lightgbm",
            "status": "success",
            "n_train": len(payload["train_idx"]),
            "n_test": len(payload["test_idx"]),
            "n_input_features": width,
            "n_output_features": width + 75,
            "pipeline_seconds": 1.0,
            "case_seconds": 1.0,
            "parameters": {"num_leaves": 7},
            "inner_losses": [0.5, 0.6, 0.7],
            "numeric_clip_quantiles": [0.0, 1.0],
            "matched_baseline_method": study.METHODS[0],
            "inner_reference_diagnostics": copy.deepcopy(reference),
            "outer_reference_diagnostics": copy.deepcopy(reference),
        }
    ]


@pytest.fixture
def confirmation(experiment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    e = experiment
    boundary = {
        "evaluation_tier": study.CONFIRMATION_TIER,
        "claim_scope": "mock-qualified-row-level-only",
        **study.calibration_evaluation.CLAIM_FLAGS,
    }
    predictors = [f"V{index}" for index in range(1, 17)]
    e.spec.clear()
    e.spec.update(
        name=study.CONFIRMATION_FAMILY,
        family_id=study.CONFIRMATION_FAMILY,
        provider="openml_task",
        task="classification",
        role="confirmation",
        origin_kind="observational",
        target="Class",
        target_transform="identity",
        categorical=[],
        drop_columns=[],
        split_policy="official",
        rights={"status": "approved"},
        evaluation_tier=study.CONFIRMATION_TIER,
        claim_boundary=copy.deepcopy(boundary),
        source={
            "uci_id": 277,
            "dataset_id": 1506,
            "dataset_version": 1,
            "task_id": 9969,
            "expected_rows": 470,
            "columns": [*predictors, "Class"],
            "target_columns": ["Class"],
            "md5_checksum": "original-raw-pin",
            "official_split_dimensions": {"repeats": 1, "folds": 10, "samples": 1},
        },
    )
    pin = fingerprint(e.spec)
    monkeypatch.setattr(study, "CONFIRMATION_SPEC_SHA256", pin)
    registration = {
        "status": "locked_before_confirmation_model_execution",
        "controller_protocol": study.CONFIRMATION_PROTOCOL,
        "evaluation_tier": study.CONFIRMATION_TIER,
        "claim_boundary": copy.deepcopy(boundary),
        "dataset_names": [study.CONFIRMATION_FAMILY],
        "family_ids": [study.CONFIRMATION_FAMILY],
        "dataset_spec_sha256": {study.CONFIRMATION_FAMILY: pin},
        "methods": list(study.CONFIRMATION_METHODS),
        "learners": ["lightgbm"],
        "seeds": [2031],
        "sample_seed": 2026,
        "folds": 3,
        "max_rows": 0,
        "threads": 8,
        "memory_limit_mb": 4096,
        "case_timeout_seconds": 900,
        "total_active_seconds_ceiling": 14400,
        "expected_cells": 15,
        "expected_physical_jobs": 9,
        "resplit_official_as_iid": True,
        "split_plan": {study.CONFIRMATION_FAMILY: {"policy": "iid", "folds": 3}},
        "classification_admission_margin": {"min_total": 30, "min_outer_train": 20, "min_outer_test": 10},
        "blend_mode": study.FIXED_MODE,
        "primary_method": study.METHODS[2],
        "fixed_weight": 0.5,
        "prediction_rule": {"weight_learning": False, "oof": False, "calibration": False},
        "selection_parameters": {
            "max_candidates": 1000,
            "max_added_features": 50,
            "stability_repeats": 3,
            "screening": "auto",
            "blend_mode": "fixed_average",
            "categorical_encoding": "one_hot",
            "numeric_clip_quantiles": [0.0, 1.0],
        },
        "learner_search": {"lightgbm": {"num_leaves": [7, 15, 31], "selection_estimator_num_leaves": 15}},
        "reporting_and_acceptance": study._confirmation_rules(),
    }
    e.suite.update(
        evaluation_tier=study.CONFIRMATION_TIER, claim_boundary=copy.deepcopy(boundary), preregistration=registration
    )
    suite_path = Path(e.argv[e.argv.index("--suite") + 1])
    suite_path.write_text(json.dumps(e.suite), encoding="utf-8")
    e.argv = ["--output", str(e.output), "--suite", str(suite_path), "--allow-confirmation"]
    e.registration = registration
    e.data["X"] = pd.DataFrame(np.arange(470 * 16, dtype=float).reshape(470, 16), columns=[f"x{i}" for i in range(16)])
    e.data["y"] = pd.Series([0, 1] * 235)
    e.metadata = {
        **copy.deepcopy(e.spec),
        "original_rows": 470,
        "sampled_rows": 470,
        "sample_seed": 2026,
        "columns": predictors,
        "column_mapping": dict(zip(e.data["X"].columns, predictors, strict=True)),
        "official_split_ids": [{"repeat": 0, "fold": fold, "sample": 0} for fold in range(10)],
        "sha256": "fixed-raw-payload",
    }
    indices = np.arange(470)
    e.official = [(np.setdiff1d(indices, test), test) for test in np.array_split(indices, 10)]

    def prepare(*args: Any) -> tuple[dict, dict, list]:
        e.acquisitions.append(args[0]["name"])
        return copy.deepcopy(e.data), copy.deepcopy(e.metadata), copy.deepcopy(e.official)

    e.prepare = prepare
    monkeypatch.setattr(execution, "_prepare_dataset", prepare)

    def run(payload: dict, directory: Path, timeout: float, **kwargs: Any) -> list[dict]:
        e.calls.append((copy.deepcopy(payload), directory, timeout, kwargs))
        rows = success(payload)
        for row in rows:
            row["loss"] = {
                study.METHODS[0]: 0.5,
                study.METHODS[1]: 0.45,
                study.METHODS[2]: 0.4,
                study.NATIVE_REFERENCES[0]: 0.46,
                study.NATIVE_REFERENCES[1]: 0.47,
            }[row["method"]]
        return rows

    e.run = run
    monkeypatch.setattr(tabular, "run_isolated", run)
    return e


def results(e: SimpleNamespace) -> dict:
    return cast(dict, json.loads((e.output / "results.json").read_text(encoding="utf-8")))


def manifest(e: SimpleNamespace) -> dict:
    return cast(dict, json.loads((e.output / "manifest.json").read_text(encoding="utf-8")))


def state(e: SimpleNamespace) -> dict:
    return cast(dict, json.loads((e.output / "state.json").read_text(encoding="utf-8")))


def records(e: SimpleNamespace) -> list[dict]:
    return [
        json.loads((e.output / "records" / row["file"]).read_text(encoding="utf-8"))
        for row in state(e)["records"].values()
    ]


def test_default_design_is_one_frozen_27_bundle_108_outcome_experiment() -> None:
    assert len(study.DEFAULT_DATASETS) == 9
    assert study.DEFAULTS["learners"] == ["lightgbm"]
    assert study.DEFAULTS["folds"] == 3 and study.DEFAULTS["seed"] == 42
    assert 9 * study.DEFAULTS["folds"] == 27
    assert 27 * len(study.METHODS) == 108
    assert study.DEFAULTS["oof_folds"] == 5
    assert study.DEFAULTS["oof_seed_offset"] == 1000
    assert study.DEFAULTS["threads"] == 8
    assert study.DEFAULTS["case_timeout"] == 900
    assert study.DEFAULTS["memory_limit_mb"] == 4096
    assert study.TOTAL_SECONDS == 21600
    assert study.REPORT_RESERVE_SECONDS == 10


def test_canonical_catalog_can_be_validated_without_payload_acquisition(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(execution, "_prepare_dataset", lambda *args: pytest.fail("No acquisition is permitted"))
    suite, legacy = execution._load_suite(study.DEFAULT_SUITE, False)
    selected = study._selected_specs(suite, legacy, study.DEFAULT_DATASETS)
    assert len(selected) == 9
    assert all(spec["role"] == "development" and spec["task"] == "classification" for spec in selected)


@pytest.mark.parametrize(
    "changes",
    [
        {"role": "confirmation"},
        {"role": "inventory"},
        {"task": "regression"},
        {"family_id": "unrequested-family"},
        {"source": {"sha256": "altered-original-pin"}},
        {"target_transform": "log"},
        {"split_policy": "group"},
        {"split_policy": "time"},
        {"group_column": "patient"},
        {"time_column": "date"},
        {"label_end_column": "date"},
        {"rights": {"status": "restricted"}},
        {"evaluation_tier": "standard_iid_unverified_entity"},
    ],
)
def test_reject_source_role_tier_and_boundaries_before_acquisition(experiment: SimpleNamespace, changes: dict) -> None:
    e = experiment
    e.spec.update(changes)
    assert study.main(e.argv) == 2
    assert not e.acquisitions and not e.calls and not e.exposures
    assert not e.output.exists()


@pytest.mark.parametrize(
    "flag,value",
    [
        ("threads", "0"),
        ("threads", "9"),
        ("case-timeout", "0"),
        ("case-timeout", "901"),
        ("case-timeout", "nan"),
        ("total-seconds", "0"),
        ("total-seconds", "21601"),
        ("total-seconds", "inf"),
        ("seed", "-1"),
        ("seed", str(2**32 - 1000)),
        ("folds", "2"),
        ("max-rows", "0"),
        ("max-candidates", "999"),
        ("max-added-features", "51"),
        ("stability-repeats", "1"),
        ("memory-limit-mb", "8192"),
    ],
)
def test_invalid_controls_rejected_before_acquisition(experiment: SimpleNamespace, flag: str, value: str) -> None:
    e = experiment
    assert study.main([*e.argv, f"--{flag}", value]) == 2
    assert not e.acquisitions and not e.output.exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("oof_folds", 3),
        ("oof_seed_offset", 1),
        ("blend_mode", "grid"),
        ("categorical_encoding", "target"),
        ("numeric_clip_quantiles", [0.01, 0.99]),
        ("target_transform", "log"),
        ("learners", ["linear"]),
        ("methods", list(study.METHODS[:2])),
        ("threads", True),
        ("folds", 3.0),
        ("seed", True),
        ("datasets", []),
        ("datasets", ["a", "a"]),
    ],
)
def test_frozen_core_configuration_cannot_be_changed(key: str, value: Any) -> None:
    with pytest.raises(ValueError):
        study.validate_config({**study.DEFAULTS, key: value}, study.TOTAL_SECONDS)


def test_three_bundles_atomic_rows_cost_and_archives(experiment: SimpleNamespace) -> None:
    e = experiment
    assert study.main(e.argv) == 0
    result = results(e)
    assert len(e.calls) == result["expected_physical_jobs"] == 3
    assert result["planned_cells"] == len(result["rows"]) == 12
    assert len(result["physical_jobs"]) == len(records(e)) == 3
    assert all(len(job["outcomes"]) == 4 for job in records(e))
    assert all(row["expected_cells"] == 3 for row in result["rows"])
    assert all(row["dataset_metadata"]["review_notes"] == "retain-metadata" for row in result["rows"])
    assert all(row["prepared_sha256"] and row["split_sha256"] for row in result["rows"])
    assert {row["method"] for row in result["rows"]} == set(study.METHODS)
    assert len(manifest(e)["predeclared_jobs"]) == 3
    assert len(manifest(e)["splits"]) == 3
    for _, _, timeout, kwargs in e.calls:
        assert kwargs == {"worker_module": "benchmarks.blending_evaluation"}
        assert 0 < timeout <= 900
    assert result["physical_cost"]["measured_physical_job_seconds_including_retries"] < 10
    assert sum(row["pipeline_seconds"] for row in result["rows"]) == 57
    assert all(item["complete_paired_families"] == 1 for item in result["summary"])
    assert result["average_vs_learned"]["complete_absolute_paired_families"] == 1
    assert len(result["blend_diagnostics"]) == 3
    archive = json.loads((e.output / "archive.json").read_text(encoding="utf-8"))
    compressed = b"".join((e.output / name).read_bytes() for name in archive["parts"])
    assert gzip.decompress(compressed) == (e.output / "results.json").read_bytes()
    assert all((e.output / name).stat().st_size <= 480 * 1024 for name in archive["parts"])
    assert (e.output / "source.zip").exists()
    report = (e.output / "REPORT.md").read_text(encoding="utf-8")
    assert "not a formal no-harm" in result["claim_note"] or "formal no-harm guarantee" in report
    assert all(method in report for method in study.METHODS)
    assert [value[0] for value in e.exposures] == ["created", "check", "admit"]


def test_whole_canonical_plan_is_persisted_before_any_model(
    experiment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    e = experiment
    all_specs = [
        {**copy.deepcopy(e.spec), "name": name, "family_id": f"family-{index}"}
        for index, name in enumerate(study.DEFAULT_DATASETS)
    ]
    e.suite["datasets"] = all_specs
    e.canonical.clear()
    e.canonical.update({spec["name"]: copy.deepcopy(spec) for spec in all_specs})
    invoked = []

    def run(payload: dict, *args: Any, **kwargs: Any) -> list[dict]:
        declared = manifest(e)
        assert declared["expected_physical_jobs"] == len(declared["predeclared_jobs"]) == 27
        assert declared["planned_cells"] == 108
        assert declared["cohort_scope"] == "canonical_nine"
        assert len(declared["datasets"]) == 9
        invoked.append(payload["physical_job_id"])
        return success(payload)

    monkeypatch.setattr(tabular, "run_isolated", run)
    assert study.main([*e.argv, "--datasets", *study.DEFAULT_DATASETS]) == 0
    assert len(invoked) == 27
    assert len(results(e)["rows"]) == 108


def test_prepare_only_never_admits_exposure_and_resume_does_not_rerun_success(
    experiment: SimpleNamespace,
) -> None:
    e = experiment
    assert study.main([*e.argv, "--prepare-only"]) == 0
    assert not e.calls and not e.exposures
    assert all(row["status"] == "prepared" for row in results(e)["rows"])
    assert manifest(e)["cohort_scope"].endswith("smoke")
    assert study.main(e.resume) == 0
    assert len(e.calls) == 3
    assert study.main(e.resume) == 0
    assert len(e.calls) == 3
    assert len(e.acquisitions) == 3
    assert all(job["attempt"] == 1 for job in records(e))


@pytest.mark.parametrize("failure_status", ["timeout", "memory_limit", "error"])
def test_failed_physical_job_expands_to_all_four_and_explicit_retry_preserves_attempts(
    experiment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, failure_status: str
) -> None:
    e = experiment
    attempted: list[str] = []

    def run(payload: dict, *args: Any, **kwargs: Any) -> list[dict]:
        attempted.append(payload["physical_job_id"])
        if payload["fold"] == 1:
            return tabular.failure_rows(payload, failure_status, "mock shared failure", 0.1)
        return success(payload)

    monkeypatch.setattr(tabular, "run_isolated", run)
    assert study.main(e.argv) == 1
    failed = [row for row in results(e)["rows"] if row["status"] != "success"]
    assert len(failed) == 4
    assert {row["method"] for row in failed} == set(study.METHODS)
    assert all("loss" not in row and "brier_sum" not in row for row in failed)
    assert all(row["error"] == "mock shared failure" for row in failed)
    assert results(e)["average_vs_learned"]["complete_paired_families"] == 0
    assert study.main(e.resume) == 1
    assert len(attempted) == 3
    old_paths = set((e.output / "records").glob("*.json"))
    retried = []

    def retry(payload: dict, *args: Any, **kwargs: Any) -> list[dict]:
        retried.append(payload["fold"])
        return success(payload)

    monkeypatch.setattr(tabular, "run_isolated", retry)
    assert study.main([*e.resume, "--retry-failed"]) == 0
    assert retried == [1]
    assert old_paths.issubset(set((e.output / "records").glob("*.json")))
    assert sorted(job["attempt"] for job in records(e)) == [1, 1, 2]


def test_dataset_failure_keeps_three_jobs_twelve_rows_and_requires_retry(
    experiment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    e = experiment
    acquisitions = []

    def failed(*args: Any) -> Any:
        acquisitions.append(1)
        raise ValueError("raw source pin unavailable")

    monkeypatch.setattr(execution, "_prepare_dataset", failed)
    assert study.main(e.argv) == 1
    assert len(records(e)) == 3 and len(results(e)["rows"]) == 12
    assert all(row["status"] == "dataset_error" for row in results(e)["rows"])
    assert not e.calls and not e.exposures
    assert study.main(e.resume) == 1
    assert len(acquisitions) == 1
    monkeypatch.setattr(execution, "_prepare_dataset", e.prepare)
    assert study.main([*e.resume, "--retry-failed"]) == 0
    assert len(e.calls) == 3


@pytest.mark.parametrize("mutation", ["source", "dependency", "suite", "config", "pickle", "raw", "split", "archive"])
def test_resume_revalidates_original_source_raw_cache_and_exact_identity(
    experiment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    e = experiment
    assert study.main(e.argv) == 0
    resume = list(e.resume)
    if mutation == "source":
        monkeypatch.setattr(tabular, "source_fingerprint", lambda: "changed-source")
    elif mutation == "dependency":
        monkeypatch.setattr(execution, "environment_versions", lambda: {"lightgbm": "changed"})
    elif mutation == "suite":
        Path(e.argv[e.argv.index("--suite") + 1]).write_text("{}\n", encoding="utf-8")
    elif mutation == "config":
        resume.extend(["--threads", "1"])
    elif mutation == "pickle":
        next((e.output / "prepared").glob("*.pkl")).write_bytes(b"corrupt")
    elif mutation == "raw":
        e.data["X"].iloc[0, 0] += 100
    elif mutation == "split":

        def changed(*args: Any) -> list:
            original = e.splits(*args)
            seed, fold, train, test = original[0]
            return [(seed, fold, train[::-1], test), *original[1:]]

        monkeypatch.setattr(execution, "_splits", changed)
    else:
        (e.output / "source.zip").write_bytes(b"corrupt")
    assert study.main(resume) != 0
    assert len(e.calls) == 3
    if mutation in {"raw", "split"}:
        assert len(e.acquisitions) == 2


def test_raw_pin_rechecked_even_when_prepared_metadata_and_jobs_are_cached(
    experiment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    e = experiment
    assert study.main(e.argv) == 0

    def fresh(*args: Any) -> Any:
        raise ValueError("Original raw checksum no longer matches the source pin")

    monkeypatch.setattr(execution, "_prepare_dataset", fresh)
    assert study.main(e.resume) != 0
    assert len(e.calls) == 3
    assert manifest(e)["identity_invalid"]


def test_checkpoint_corruption_rejected_before_worker(experiment: SimpleNamespace) -> None:
    e = experiment
    assert study.main(e.argv) == 0
    record = next(iter(state(e)["records"].values()))
    (e.output / "records" / record["file"]).write_text("{}\n", encoding="utf-8")
    assert study.main(e.resume) == 2
    assert len(e.calls) == 3


def test_preparation_interrupt_marks_all_unavailable_then_can_resume(
    experiment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    e = experiment

    def interrupted(*args: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(execution, "_prepare_dataset", interrupted)
    assert study.main(e.argv) == 1
    assert len(records(e)) == 3
    assert all(study._unfinished_preparation(job) for job in records(e))
    assert all(row["status"] == "interrupted" for row in results(e)["rows"])
    assert not e.calls and not e.exposures
    monkeypatch.setattr(execution, "_prepare_dataset", e.prepare)
    assert study.main(e.resume) == 0
    assert len(e.calls) == 3


def test_only_unattempted_unavailable_interruption_can_relax_cell_identity() -> None:
    row = {
        "status": "interrupted",
        "split_sha256": "unavailable",
        "prepared_sha256": None,
        "preparation_state": "unavailable",
    }
    job = {"status": "interrupted", "attempt": 0, "preparation_state": "unavailable", "outcomes": [row] * 4}
    assert study._unfinished_preparation(job)
    assert not study._unfinished_preparation({**job, "attempt": 1})
    assert not study._unfinished_preparation({**job, "status": "success"})
    assert not study._unfinished_preparation({**job, "outcomes": [{**row, "loss": 0.2}] * 4})


def test_worker_interrupt_stops_future_jobs_and_resume_keeps_success(
    experiment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    e = experiment
    calls = []

    def interrupted(payload: dict, *args: Any, **kwargs: Any) -> list[dict]:
        calls.append(payload["fold"])
        if payload["fold"] == 1:
            raise KeyboardInterrupt
        return success(payload)

    monkeypatch.setattr(tabular, "run_isolated", interrupted)
    assert study.main(e.argv) == 1
    assert calls == [0, 1]
    assert len(results(e)["rows"]) == 12
    assert sum(row["status"] == "success" for row in results(e)["rows"]) == 4
    assert sum(row["status"] == "interrupted" for row in results(e)["rows"]) == 8
    retried = []

    def resumed(payload: dict, *args: Any, **kwargs: Any) -> list[dict]:
        retried.append(payload["fold"])
        return success(payload)

    monkeypatch.setattr(tabular, "run_isolated", resumed)
    assert study.main(e.resume) == 0
    assert retried == [1, 2]


def test_source_change_during_first_worker_stops_future_bundles(
    experiment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    e = experiment
    invoked = []

    def run(payload: dict, *args: Any, **kwargs: Any) -> list[dict]:
        invoked.append(payload["fold"])
        monkeypatch.setattr(tabular, "source_fingerprint", lambda: "new-source")
        return success(payload)

    monkeypatch.setattr(tabular, "run_isolated", run)
    assert study.main(e.argv) == 1
    assert invoked == [0]
    assert manifest(e)["source_changed"]
    assert all(row["status"] == "source_changed" and "loss" not in row for row in results(e)["rows"])
    assert study.main(e.resume) == 2


def test_budget_exhaustion_preserves_denominator_without_exposure(experiment: SimpleNamespace) -> None:
    e = experiment
    assert study.main([*e.argv, "--total-seconds", "1"]) == 1
    assert len(results(e)["rows"]) == 12 and len(records(e)) == 3
    assert all(row["status"] == "budget_exhausted" for row in results(e)["rows"])
    assert not e.acquisitions and not e.calls and not e.exposures
    assert study.main([*e.resume, "--total-seconds", "21600"]) == 0
    assert len(e.calls) == 3


def test_cumulative_budget_charges_preparation_workers_reports_and_archives(
    experiment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    e = experiment
    clock = [0.0]
    monkeypatch.setattr(study.time, "perf_counter", lambda: clock[0])
    original_report = study.write_report

    def prepare(*args: Any) -> tuple[dict, dict, None]:
        clock[0] += 2
        return cast(tuple[dict, dict, None], e.prepare(*args))

    def run(payload: dict, *args: Any, **kwargs: Any) -> list[dict]:
        clock[0] += 3
        return success(payload)

    def report(*args: Any) -> None:
        original_report(*args)
        clock[0] += 1

    monkeypatch.setattr(execution, "_prepare_dataset", prepare)
    monkeypatch.setattr(tabular, "run_isolated", run)
    monkeypatch.setattr(study, "write_report", report)
    assert study.main(e.argv) == 0
    elapsed = manifest(e)["active_seconds"]
    assert elapsed == 15
    assert state(e)["elapsed_seconds"] == elapsed
    assert results(e)["physical_cost"]["measured_physical_job_seconds_including_retries"] == 9
    assert study.main(e.resume) == 0
    assert manifest(e)["active_seconds"] == elapsed + 3


def test_nondefault_seed_is_smoke_and_config_cannot_be_adapted_on_resume(experiment: SimpleNamespace) -> None:
    e = experiment
    assert study.main([*e.argv, "--seed", "7"]) == 0
    assert manifest(e)["cohort_scope"].endswith("smoke")
    assert {row["seed"] for row in results(e)["rows"]} == {7}
    assert study.main([*e.resume, "--seed", "42"]) == 2
    assert len(e.calls) == 3


def test_outer_worker_slices_hash_bound_data_and_calls_core_once(
    experiment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    e = experiment
    assert study.main(e.argv) == 0
    payload = e.calls[0][0]
    invocations = []
    module = ModuleType("benchmarks.oof_blending")

    def fit(train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame, test_y: pd.Series, **kwargs: Any) -> dict:
        invocations.append(kwargs)
        assert train_X.equals(e.data["X"].iloc[payload["train_idx"]])
        assert train_y.equals(e.data["y"].iloc[payload["train_idx"]])
        assert test_X.equals(e.data["X"].iloc[payload["test_idx"]])
        assert test_y.equals(e.data["y"].iloc[payload["test_idx"]])
        assert not set(train_X.index) & set(test_X.index)
        report = bundle_report()
        report["diagnostics"]["full_models"]["raw"]["private_model"] = object()
        report["diagnostics"]["members"][0]["raw"]["oof_predictions"] = np.ones((5, 2))
        report["diagnostics"]["classes"] = ["private-vocabulary"]
        return report

    monkeypatch.setattr(module, "fit_oof_blend_bundle", fit, raising=False)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    rows = study.run_case(payload, e.root)
    assert len(invocations) == 1
    assert invocations[0]["learner"] == "lightgbm"
    assert invocations[0]["config"] == payload["config"]
    assert len(rows) == 4 and all(row["status"] == "success" for row in rows)
    fingerprint(rows)
    encoded = json.dumps(rows)
    for forbidden in (
        "probabilities",
        "oof_predictions",
        "target_rows",
        "vocabulary",
        "estimators",
        "private_model",
    ):
        assert f'"{forbidden}":' not in encoded
    assert rows[0]["pipeline_seconds"] == 1
    assert rows[1]["pipeline_seconds"] == 2
    assert rows[3]["pipeline_seconds"] == 13


@pytest.mark.parametrize("mutation", ["hash", "indices", "overlap", "mode", "role", "groups", "seed"])
def test_worker_rejects_changed_hash_mode_and_boundaries_before_core(
    experiment: SimpleNamespace, mutation: str
) -> None:
    e = experiment
    assert study.main(e.argv) == 0
    payload = copy.deepcopy(e.calls[0][0])
    if mutation == "hash":
        payload["prepared_sha256"] = "different"
    elif mutation == "indices":
        payload["test_idx"] = np.array([-1])
    elif mutation == "overlap":
        payload["test_idx"] = payload["train_idx"]
    elif mutation == "mode":
        payload["blend_mode"] = "adaptive"
    elif mutation == "role":
        payload["role"] = "confirmation"
    elif mutation == "seed":
        payload["config"]["seed"] = 77
    else:
        data = copy.deepcopy(e.data)
        data["groups"] = pd.Series(np.arange(36))
        content = pickle.dumps(data, protocol=5)
        Path(payload["data_path"]).write_bytes(content)
        payload["prepared_sha256"] = tabular.digest(content)
    with pytest.raises(ValueError):
        study._worker_data(payload)


@pytest.mark.parametrize(
    "mutation",
    ["partial", "missing", "duplicate", "nan", "brier", "width", "rowcount", "split", "weight", "oof", "shared"],
)
def test_invalid_or_partial_success_invalidates_all_four_outcomes(experiment: SimpleNamespace, mutation: str) -> None:
    e = experiment
    assert study.main(e.argv) == 0
    payload = e.calls[0][0]
    rows = success(payload)
    if mutation == "partial":
        rows[1]["status"] = "error"
    elif mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[1]["method"] = rows[0]["method"]
    elif mutation == "nan":
        rows[0]["loss"] = float("nan")
    elif mutation == "brier":
        rows[0]["brier_sum"] = -1
    elif mutation == "width":
        rows[0]["n_output_features"] = 999
    elif mutation == "rowcount":
        rows[0]["n_test"] = 0
    elif mutation == "split":
        rows[0]["split_sha256"] = "different"
    elif mutation == "weight":
        rows[0]["shared_bundle_diagnostics"]["weight"] = 1.1
    elif mutation == "oof":
        rows[0]["shared_bundle_diagnostics"]["member_count"] = 3
    else:
        rows[0]["shared_bundle_diagnostics"]["weight"] = 0.4
    output = study.normalize_outcomes(payload, rows)
    assert len(output) == 4
    assert all(row["status"] == "error" and "loss" not in row for row in output)


@pytest.mark.parametrize("status", ["success", "timeout"])
@pytest.mark.parametrize("recorded_seed", [42, 43, True, "42"])
def test_worker_hash_provenance_survives_normalization_and_rejects_mismatch(
    experiment: SimpleNamespace, status: str, recorded_seed: Any
) -> None:
    assert study.main(experiment.argv) == 0
    payload = experiment.calls[0][0]
    rows = success(payload) if status == "success" else tabular.failure_rows(payload, status, "limit", 1.0)
    for row in rows:
        row["worker_python_hash_seed"] = recorded_seed
    normalized = study.normalize_outcomes(payload, rows)
    if type(recorded_seed) is int and recorded_seed == payload["seed"]:
        assert all(row["status"] == status and row["worker_python_hash_seed"] == recorded_seed for row in normalized)
    else:
        assert all(row["status"] == "error" and "loss" not in row for row in normalized)
        assert all("hash seed" in row["error"] for row in normalized)


def test_worker_entry_exports_error_bundle_not_partial_scores(
    experiment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    e = experiment
    assert study.main(e.argv) == 0
    path = e.root / "case.pkl"
    output = e.root / "worker.json"
    path.write_bytes(pickle.dumps(e.calls[0][0], protocol=5))

    def failed(*args: Any) -> Any:
        raise ValueError("mock bundle failed")

    monkeypatch.setattr(study, "run_case", failed)
    assert study.main(["--worker", str(path), "--worker-output", str(output)]) == 0
    rows = json.loads(output.read_text(encoding="utf-8"))
    assert len(rows) == 4 and all(row["status"] == "error" for row in rows)
    assert all("loss" not in row for row in rows)


def test_no_cli_weight_grid_or_confirmation_mode(experiment: SimpleNamespace) -> None:
    for flag in ("--weight", "--oof-folds", "--learners", "--allow-confirmation", "--blend-mode"):
        with pytest.raises(SystemExit):
            study.main([*experiment.argv, flag, "1"])
    assert not experiment.acquisitions


def test_retry_requires_resume(experiment: SimpleNamespace) -> None:
    assert study.main([*experiment.argv, "--retry-failed"]) == 2
    assert not experiment.acquisitions


def test_altered_default_canonical_file_is_rejected_before_loading_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "changed-canonical.json"
    path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(study, "DEFAULT_SUITE", path)
    monkeypatch.setattr(
        study.calibration_evaluation, "_canonical_specs", lambda: pytest.fail("Changed canonical file must fail first")
    )
    with pytest.raises(ValueError, match="canonical replay source manifest"):
        study._canonical_specs()


def test_altered_normalized_canonical_registry_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(study.calibration_evaluation, "_canonical_specs", lambda: {"changed": {}})
    with pytest.raises(ValueError, match="normalized source/family"):
        study._canonical_specs()


@pytest.mark.parametrize("boundary", ["groups", "timestamps", "label_end_times"])
def test_prepared_boundaries_fail_all_jobs_without_models(experiment: SimpleNamespace, boundary: str) -> None:
    e = experiment
    e.data[boundary] = pd.Series(np.arange(36))
    assert study.main(e.argv) == 1
    assert len(records(e)) == 3 and len(results(e)["rows"]) == 12
    assert all(row["status"] == "dataset_error" for row in results(e)["rows"])
    assert not e.calls and not e.exposures


@pytest.mark.parametrize("mutation", ["ids", "private", "cost", "coverage", "policy", "counts", "epsilon", "criterion"])
def test_shared_fit_and_public_diagnostic_contract_rejects_invalid_reports(
    experiment: SimpleNamespace, mutation: str
) -> None:
    e = experiment
    assert study.main(e.argv) == 0
    payload = e.calls[0][0]
    rows = success(payload)
    diagnostic = rows[0]["shared_bundle_diagnostics"]
    if mutation == "ids":
        diagnostic["full_fit_identity_sha256"]["blend"] = ["c" * 64, "d" * 64]
    elif mutation == "private":
        diagnostic["oof_predictions"] = [0.1, 0.9]
    elif mutation == "cost":
        rows[0]["shared_bundle_timing"]["raw_training_seconds"] += 10
    elif mutation == "coverage":
        diagnostic["n_oof_covered"] -= 1
    elif mutation == "policy":
        diagnostic["source_policy"]["categorical_encoding"] = "target"
    elif mutation == "counts":
        diagnostic["class_counts"] = [24, 24]
    elif mutation == "epsilon":
        diagnostic["optimization"]["epsilon"] = [0.1, 0.2]
    else:
        diagnostic["weight"] = 0.6
    output = study.normalize_outcomes(payload, rows)
    assert len(output) == 4 and all(row["status"] == "error" for row in output)
    assert "oof_predictions" not in json.dumps(output)


def test_root_dispatch_exception_preserves_every_requested_row(
    experiment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    e = experiment

    def failed(*args: Any) -> Any:
        raise RuntimeError("Cannot allocate next attempt")

    monkeypatch.setattr(study.RunState, "next_attempt", failed)
    assert study.main(e.argv) == 1
    assert not e.calls
    assert len(records(e)) == 3
    assert len(results(e)["rows"]) == 12
    assert all(row["status"] == "error" for row in results(e)["rows"])
    assert "Cannot allocate next attempt" in manifest(e)["root_error"]


def test_interrupt_keeps_metadata_and_unavailable_splits_for_every_requested_family(
    experiment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    e = experiment
    spec = {**copy.deepcopy(e.spec), "name": "second", "family_id": "second-family"}
    e.suite["datasets"].append(spec)
    e.canonical["second"] = copy.deepcopy(spec)

    def interrupted(*args: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(execution, "_prepare_dataset", interrupted)
    assert study.main([*e.argv, "--datasets", e.spec["name"], "second"]) == 1
    assert len(results(e)["rows"]) == 24
    assert len(manifest(e)["datasets"]) == 2
    assert len(manifest(e)["splits"]) == 6
    assert all(item["sha256"] == "unavailable" for item in manifest(e)["splits"])


def test_source_change_during_preparation_never_dispatches_a_model(
    experiment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    e = experiment

    def prepare(*args: Any) -> tuple[dict, dict, None]:
        result = e.prepare(*args)
        monkeypatch.setattr(tabular, "source_fingerprint", lambda: "source-after-prepare")
        return cast(tuple[dict, dict, None], result)

    monkeypatch.setattr(execution, "_prepare_dataset", prepare)
    assert study.main(e.argv) == 1
    assert not e.calls and not e.exposures
    assert all(row["status"] == "source_changed" for row in results(e)["rows"])
    assert manifest(e)["source_changed"]


def test_exhausted_resume_still_refingerprints_cached_raw_without_rerunning_successes(
    experiment: SimpleNamespace,
) -> None:
    e = experiment
    assert study.main(e.argv) == 0
    saved = state(e)
    saved["elapsed_seconds"] = study.TOTAL_SECONDS
    (e.output / "state.json").write_text(json.dumps(saved), encoding="utf-8")
    assert study.main(e.resume) == 1
    assert len(e.acquisitions) == 2
    assert len(e.calls) == 3
    assert not manifest(e)["identity_invalid"]
    assert manifest(e)["budget_overshoot_seconds"] > 0


def test_failed_prepare_only_is_durable_and_requires_explicit_retry(
    experiment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    e = experiment
    attempts = []

    def failed(*args: Any) -> Any:
        attempts.append(1)
        raise ValueError("No valid raw payload")

    monkeypatch.setattr(execution, "_prepare_dataset", failed)
    assert study.main([*e.argv, "--prepare-only"]) == 1
    assert len(records(e)) == 3
    assert not e.calls and not e.exposures
    monkeypatch.setattr(execution, "_prepare_dataset", e.prepare)
    assert study.main(e.resume) == 1
    assert not e.acquisitions and not e.calls
    assert study.main([*e.resume, "--retry-failed"]) == 0
    assert len(e.calls) == 3 and len(attempts) == 1


def test_interrupted_source_archive_is_unavailable_and_resumable_only_before_preparation(
    experiment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    e = experiment
    original = study.archive_source

    def interrupted(*args: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(study, "archive_source", interrupted)
    assert study.main(e.argv) == 1
    assert manifest(e)["source_archive_sha256"] is None
    assert all(study._unfinished_preparation(job) for job in records(e))
    assert not e.acquisitions and not e.calls
    monkeypatch.setattr(study, "archive_source", original)
    assert study.main(e.resume) == 0
    assert len(e.calls) == 3
    saved = manifest(e)
    saved["source_archive_sha256"] = None
    (e.output / "manifest.json").write_text(json.dumps(saved), encoding="utf-8")
    assert study.main(e.resume) == 2
    assert len(e.calls) == 3


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("mode", [study.MODE, study.FIXED_MODE])
def test_real_core_json_contract_with_mocked_complete_procedures(
    experiment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, n_classes: int, mode: str
) -> None:
    from benchmarks import oof_blending

    e = experiment
    e.data["y"] = pd.Series(np.arange(36) % n_classes)
    assert study.main([*e.argv, "--blend-mode", mode, "--seed", "42"]) == 0
    payload = e.calls[0][0]
    fitted: list[tuple[bool, list[int]]] = []

    class FakeBase:
        def __init__(self, *, engineered: bool, **kwargs: Any):
            self.engineered = engineered

        def fit(self, X: pd.DataFrame, y: np.ndarray) -> FakeBase:
            fitted.append((self.engineered, list(X.index)))
            self.classes_ = np.unique(y)
            self.n_features_in_ = X.shape[1]
            self.feature_names_out_ = [*X.columns, *(["fake_fe"] if self.engineered else [])]
            self.fit_identity_sha256_ = fingerprint([self.engineered, len(fitted)])
            self.fitted_rows_ = len(X)
            return self

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            probability = np.full((len(X), len(self.classes_)), 1.0 / len(self.classes_))
            if n_classes == 3:
                rows = np.arange(len(X))
                true_class = np.asarray(X["x0"], dtype=int) % 3
                probability[rows, true_class] = 0.5
                probability[rows, (true_class + 1) % 3] = 0.1 if self.engineered else 0.25
                probability[rows, (true_class + 2) % 3] = 0.4 if self.engineered else 0.25
            return probability

        def diagnostics(self) -> dict:
            return base_report(self.engineered, self.fitted_rows_, self.fit_identity_sha256_, n_classes)

    monkeypatch.setattr(oof_blending, "CompleteBaseProcedure", FakeBase)
    if mode == study.FIXED_MODE:
        for name in ("fit_oof_blend_bundle", "make_oof_splits", "optimize_blend_weight"):
            monkeypatch.setattr(
                oof_blending, name, lambda *args, **kwargs: pytest.fail("Fixed average must never call OOF")
            )
    rows = study.run_case(payload, e.root)
    fixed = mode == study.FIXED_MODE
    assert len(rows) == (3 if fixed else 4) and all(row["status"] == "success" for row in rows)
    assert len(fitted) == (2 if fixed else 12)
    assert [engineered for engineered, _ in fitted] == [False, True] * (1 if fixed else 6)
    assert all(set(indices).issubset(set(payload["train_idx"])) for _, indices in fitted)
    assert all(not set(indices) & set(payload["test_idx"]) for _, indices in fitted)
    assert all(len(indices) == len(payload["train_idx"]) for _, indices in fitted[-2:])
    diagnostic = rows[0]["shared_bundle_diagnostics"]
    assert diagnostic["full_models"]["fe"]["feature_names"] == ["x0", "fake_fe"]
    if fixed:
        assert diagnostic["weight"] == 0.5 and diagnostic["weight_optimized"] is False
        assert set(diagnostic) == study.FIXED_DIAGNOSTIC_KEYS
        assert not {"oof_losses", "members", "optimization"} & set(diagnostic)
    else:
        assert diagnostic["weight"] == 0.0
        assert diagnostic["optimization"]["status"] == "identical_probabilities"
        assert set(diagnostic) == study.DIAGNOSTIC_KEYS
        assert len(diagnostic["members"]) == 5
        assert all("feature_names" not in member["fe"] for member in diagnostic["members"])
        assert all(len(member["fe"]["trials"]) == 3 for member in diagnostic["members"])
    fingerprint(rows)


def test_compact_provenance_is_persisted_and_invalid_plan_or_private_data_rejects_bundle(
    experiment: SimpleNamespace,
) -> None:
    e = experiment
    assert study.main(e.argv) == 0
    payload = e.calls[0][0]
    report = results(e)["blend_diagnostics"][0]["diagnostics"]
    assert len(report["members"]) == 5
    assert report["full_models"]["fe"]["feature_names"] == ["x0", "fake_fe"]
    assert report["full_models"]["raw"]["selected_parameters"] == {"num_leaves": 7}
    assert all(len(report["full_models"][arm]["trials"]) == 3 for arm in ("raw", "fe"))
    for mutation in ("members", "names", "identity", "loss", "rows", "private", "parameters"):
        rows = success(payload)
        diagnostic = rows[0]["shared_bundle_diagnostics"]
        if mutation == "members":
            diagnostic["members"].pop()
        elif mutation == "names":
            diagnostic["full_models"]["fe"]["feature_names"] = ["wrong-width"]
        elif mutation == "identity":
            diagnostic["full_models"]["fe"]["fit_identity_sha256"] = "c" * 64
        elif mutation == "loss":
            diagnostic["full_models"]["fe"]["trials"][0]["loss"] = float("nan")
        elif mutation == "rows":
            diagnostic["members"][0]["raw"]["inner_split"]["train"]["size"] = 999
        elif mutation == "private":
            diagnostic["members"][0]["fe"]["target_rows"] = [0, 1]
        else:
            diagnostic["full_models"]["raw"]["selected_parameters"]["vocabulary"] = ["secret"]
        normalized = study.normalize_outcomes(payload, rows)
        assert len(normalized) == 4 and all(row["status"] == "error" for row in normalized)
        assert all("shared_bundle_diagnostics" not in row for row in normalized)


def test_fixed_average_default_multiseed_counts_policy_and_resume(experiment: SimpleNamespace) -> None:
    e = experiment
    assert study.main([*e.argv, "--blend-mode", "fixed_average"]) == 0
    result = results(e)
    assert len(e.calls) == result["expected_physical_jobs"] == 9
    assert result["planned_cells"] == len(result["rows"]) == 27
    assert len(records(e)) == 9 and all(len(job["outcomes"]) == 3 for job in records(e))
    assert result["config"]["seeds"] == [42, 137, 2029]
    assert result["config"]["seed"] == 42
    assert not {"oof_folds", "oof_seed_offset"} & set(result["config"])
    assert set(result["coverage"]) == set(study.METHODS[:3])
    assert "average_vs_learned" not in result
    assert result["fixed_average_policy"] == {
        "weight": 0.5,
        "scope": "global_predeclared",
        "optimizer_used": False,
        "complete_procedures_per_job": 2,
        "pipeline_fits_per_job": 8,
    }
    assert len(manifest(e)["datasets"]) == 1 and len(manifest(e)["splits"]) == 9
    assert len(manifest(e)["predeclared_jobs"]) == 9
    assert all(item["complete_paired_families"] == 1 for item in result["summary"])
    assert all(item["blend_mode"] == "fixed_average" for item in result["summary"])
    for payload, _, timeout, kwargs in e.calls:
        assert payload["seed"] == payload["config"]["seed"]
        assert payload["config"]["seeds"] == study.FIXED_SEEDS
        assert payload["expected_cells"] == 9
        assert timeout <= 900 and payload["config"]["threads"] == 8
        assert kwargs == {"worker_module": "benchmarks.blending_evaluation"}
    for row in result["rows"]:
        assert row["expected_cells"] == 9
        diagnostic = row["shared_bundle_diagnostics"]
        assert diagnostic["seed"] == row["seed"] and diagnostic["weight"] == 0.5
        assert set(diagnostic) == study.FIXED_DIAGNOSTIC_KEYS
        assert not {"oof_losses", "oof_seed", "optimization", "members"} & set(diagnostic)
        assert len(diagnostic["full_models"]["fe"]["trials"]) == 3
        assert not any(key.startswith(("oof_", "blend_")) for key in row["shared_bundle_timing"])
    report = (e.output / "REPORT.md").read_text(encoding="utf-8")
    assert "eight pipeline fits" in report and "three-arm coverage" in report
    assert "average versus OOF-learned" not in report
    archive = json.loads((e.output / "archive.json").read_text(encoding="utf-8"))
    assert (
        gzip.decompress(b"".join((e.output / part).read_bytes() for part in archive["parts"]))
        == (e.output / "results.json").read_bytes()
    )
    assert study.main(e.resume) == 0
    assert len(e.calls) == 9 and len(e.acquisitions) == 2


def test_fixed_full_canonical_design_declares_81_bundles_243_outcomes_before_models(
    experiment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    e = experiment
    specs = [
        {**copy.deepcopy(e.spec), "name": name, "family_id": f"family-{index}"}
        for index, name in enumerate(study.DEFAULT_DATASETS)
    ]
    e.suite["datasets"] = specs
    e.canonical.clear()
    e.canonical.update({spec["name"]: copy.deepcopy(spec) for spec in specs})
    calls = []

    def run(payload: dict, *args: Any, **kwargs: Any) -> list[dict]:
        declared = manifest(e)
        assert declared["expected_physical_jobs"] == len(declared["predeclared_jobs"]) == 81
        assert declared["planned_cells"] == 243
        assert len(declared["splits"]) == 81 and len(declared["datasets"]) == 9
        assert declared["cohort_scope"] == "canonical_nine"
        calls.append(payload["physical_job_id"])
        return success(payload)

    monkeypatch.setattr(tabular, "run_isolated", run)
    assert study.main([*e.argv, "--blend-mode", "fixed_average", "--datasets", *study.DEFAULT_DATASETS]) == 0
    assert len(set(calls)) == 81 and len(results(e)["rows"]) == 243
    assert all(item["complete_paired_families"] == 9 for item in results(e)["summary"])


@pytest.mark.parametrize(
    "extra",
    [
        ["--seeds", "42", "137"],
        ["--blend-mode", "fixed_average", "--seeds", "42", "42"],
        ["--blend-mode", "fixed_average", "--seeds", "1", "2", "3", "4"],
        ["--blend-mode", "fixed_average", "--seeds", "-1"],
        ["--blend-mode", "fixed_average", "--seeds", str(2**32 - 1000)],
    ],
)
def test_mode_seed_controls_reject_before_acquisition(experiment: SimpleNamespace, extra: list[str]) -> None:
    assert study.main([*experiment.argv, *extra]) == 2
    assert not experiment.acquisitions and not experiment.output.exists()


def test_seed_and_seeds_cli_are_mutually_exclusive(experiment: SimpleNamespace) -> None:
    with pytest.raises(SystemExit):
        study.main([*experiment.argv, "--blend-mode", "fixed_average", "--seed", "42", "--seeds", "42"])
    assert not experiment.acquisitions


@pytest.mark.parametrize("extra,expected", [(["--seed", "137"], [137]), (["--seeds", "7", "9"], [7, 9])])
def test_fixed_seed_and_seeds_seed_every_physical_job(
    experiment: SimpleNamespace, extra: list[str], expected: list[int]
) -> None:
    e = experiment
    assert study.main([*e.argv, "--blend-mode", "fixed_average", *extra]) == 0
    assert manifest(e)["config"]["seeds"] == expected
    assert len(e.calls) == len(expected) * 3
    assert {payload["seed"] for payload, *_ in e.calls} == set(expected)
    assert all(payload["config"]["seed"] == payload["seed"] for payload, *_ in e.calls)
    assert manifest(e)["cohort_scope"].endswith("smoke")
    assert study.main(e.resume) == 0
    assert len(e.calls) == len(expected) * 3


@pytest.mark.parametrize("failure", ["timeout", "memory_limit", "error"])
def test_fixed_failure_atomic_three_arms_retry_only_failed_seed_fold(
    experiment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    e = experiment

    def run(payload: dict, *args: Any, **kwargs: Any) -> list[dict]:
        if payload["seed"] == 137 and payload["fold"] == 1:
            return tabular.failure_rows(payload, failure, "fixed bundle failed", 0.3)
        return success(payload)

    monkeypatch.setattr(tabular, "run_isolated", run)
    assert study.main([*e.argv, "--blend-mode", "fixed_average"]) == 1
    rows = results(e)["rows"]
    assert len(rows) == 27
    failed = [row for row in rows if row["status"] != "success"]
    assert len(failed) == 3 and {row["method"] for row in failed} == set(study.METHODS[:3])
    assert all("loss" not in row for row in failed)
    assert all(item["complete_paired_families"] == 0 for item in results(e)["summary"])
    previous = set((e.output / "records").glob("*.json"))
    retried = []

    def retry(payload: dict, *args: Any, **kwargs: Any) -> list[dict]:
        retried.append((payload["seed"], payload["fold"]))
        return success(payload)

    monkeypatch.setattr(tabular, "run_isolated", retry)
    assert study.main([*e.resume, "--retry-failed"]) == 0
    assert retried == [(137, 1)]
    assert previous.issubset(set((e.output / "records").glob("*.json")))
    assert sorted(job["attempt"] for job in records(e)) == [1] * 8 + [2]


@pytest.mark.parametrize("kind", ["dataset", "interrupt", "budget"])
def test_fixed_unavailable_preparation_keeps_every_seed_fold_and_resumes(
    experiment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    e = experiment
    extra = []
    if kind == "budget":
        extra = ["--total-seconds", "1"]
    else:

        def failed(*args: Any) -> Any:
            if kind == "interrupt":
                raise KeyboardInterrupt
            raise ValueError("Preparation failed")

        monkeypatch.setattr(execution, "_prepare_dataset", failed)
    assert study.main([*e.argv, "--blend-mode", "fixed_average", *extra]) == 1
    assert len(records(e)) == 9 and len(results(e)["rows"]) == 27
    assert not e.calls and not e.exposures
    if kind != "dataset":
        assert all(study._unfinished_preparation(job, mode=study.FIXED_MODE) for job in records(e))
    monkeypatch.setattr(execution, "_prepare_dataset", e.prepare)
    assert study.main([*e.resume, "--total-seconds", "21600", "--retry-failed"]) == 0
    assert len(e.calls) == 9


def test_fixed_prepare_only_has_three_arm_rows_no_exposure(experiment: SimpleNamespace) -> None:
    e = experiment
    assert study.main([*e.argv, "--blend-mode", "fixed_average", "--prepare-only"]) == 0
    assert not e.calls and not e.exposures
    assert len(results(e)["rows"]) == 27
    assert all(row["status"] == "prepared" for row in results(e)["rows"])
    assert study.main(e.resume) == 0
    assert len(e.calls) == 9 and len(e.acquisitions) == 2


@pytest.mark.parametrize("mutation", ["mode", "seeds", "source", "raw", "pickle"])
def test_fixed_resume_is_bound_to_mode_seed_list_source_and_raw_identity(
    experiment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    e = experiment
    assert study.main([*e.argv, "--blend-mode", "fixed_average"]) == 0
    resume = list(e.resume)
    if mutation == "mode":
        resume.extend(["--blend-mode", "oof_cv5", "--seed", "42"])
    elif mutation == "seeds":
        resume.extend(["--seeds", "42", "2029", "137"])
    elif mutation == "source":
        monkeypatch.setattr(tabular, "source_fingerprint", lambda: "changed-source")
    elif mutation == "raw":
        e.data["X"].iloc[0, 0] += 100
    else:
        next((e.output / "prepared").glob("*.pkl")).write_bytes(b"corrupt")
    assert study.main(resume) != 0
    assert len(e.calls) == 9


def test_fixed_rejects_learned_arm_mode_mixing_and_invented_oof_fields(experiment: SimpleNamespace) -> None:
    e = experiment
    assert study.main([*e.argv, "--blend-mode", "fixed_average", "--seed", "42"]) == 0
    payload = e.calls[0][0]
    for mutation in ("method", "mode", "weight", "optimizer", "oof", "timing", "identity", "count"):
        rows = success(payload)
        if mutation == "method":
            rows[-1]["method"] = study.METHODS[3]
        elif mutation == "mode":
            rows[0]["blend_mode"] = study.MODE
        elif mutation == "weight":
            rows[0]["shared_bundle_diagnostics"]["weight"] = 0.6
        elif mutation == "optimizer":
            rows[0]["shared_bundle_diagnostics"]["weight_optimized"] = True
        elif mutation == "oof":
            rows[0]["shared_bundle_diagnostics"]["oof_losses"] = {"raw": 0.3}
        elif mutation == "timing":
            rows[0]["shared_bundle_timing"]["oof_fit_seconds"] = 10.0
        elif mutation == "identity":
            rows[0]["shared_bundle_diagnostics"]["full_fit_identity_sha256"]["average"] = ["c" * 64, "d" * 64]
        else:
            rows.pop()
        output = study.normalize_outcomes(payload, rows)
        assert len(output) == 3 and all(row["status"] == "error" for row in output)
    result = results(e)
    result["rows"][0]["blend_mode"] = study.MODE
    with pytest.raises(ValueError, match="cannot pool"):
        study.write_report(e.output, manifest(e), result["rows"], records(e))


def test_locked_confirmation_nine_jobs_fifteen_outcomes_tier_and_native_dispatch(confirmation: SimpleNamespace) -> None:
    e = confirmation
    assert study.main(e.argv) == 0
    result, declared = results(e), manifest(e)
    assert len(e.calls) == result["expected_physical_jobs"] == 9
    assert len(result["rows"]) == result["planned_cells"] == 15
    assert sorted(len(job["outcomes"]) for job in records(e)) == [1] * 6 + [3] * 3
    assert len(declared["splits"]) == 3 and len(declared["predeclared_jobs"]) == 9
    assert len(declared["datasets"][0]["official_split_ids"]) == 10
    assert declared["datasets"][0]["used_split_policy"] == "iid"
    assert declared["datasets"][0]["selected_official_split_ids"] == []
    assert declared["cohort_scope"] == "locked_single_family_qualified_confirmation"
    assert declared["budget_seconds"] == 14400
    assert declared["config"]["max_rows"] == 0 and declared["config"]["seeds"] == [2031]
    for payload, _, timeout, kwargs in e.calls:
        assert payload["seed"] == 2031 and payload["expected_cells"] == 3 and timeout <= 900
        assert kwargs == (
            {} if payload["method"] in study.NATIVE_REFERENCES else {"worker_module": "benchmarks.blending_evaluation"}
        )
    for item in [result, declared, declared["config"], *result["rows"], *result["summary"]]:
        assert item["evaluation_tier"] == study.CONFIRMATION_TIER
        assert item["claim_boundary"] == e.registration["claim_boundary"]
    natives = [row for row in result["rows"] if row["method"] in study.NATIVE_REFERENCES]
    assert all(row["n_output_features"] > row["n_input_features"] + 50 for row in natives)
    assert all(row["outer_reference_diagnostics"]["width_policy"] == "native_output" for row in natives)
    assert "private-category" not in json.dumps(result)
    decision = result["confirmation_decision"]
    assert decision["status"] == "pass" and decision["locked_criteria_passed"]
    assert decision["valid_prediction_rows"] == 15 and decision["complete_physical_jobs"] == 9
    assert len(decision["family_losses"]) == 5 and len(decision["comparisons"]) == 4
    assert all(item["requested_families"] == 1 for item in decision["comparisons"])
    assert all(item["dataset_bootstrap_95_ci"] is None for item in result["summary"])
    assert not decision["automatic_default_promotion"] and not decision["automatic_sota"]
    assert (
        not decision["clinical_generalization_claims_allowed"]
        and not decision["patient_entity_disjoint_claims_allowed"]
    )
    assert "Authoritative locked confirmation decision" in (e.output / "REPORT.md").read_text(encoding="utf-8")
    assert [event[0] for event in e.exposures][:2] == ["created", "check"]
    assert sum(event[0] == "admit" for event in e.exposures) == 1
    assert study.main(e.resume) == 0
    assert len(e.calls) == 9 and len(e.acquisitions) == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "inventory",
        "family",
        "active",
        "rights",
        "group",
        "time",
        "schema_and_pin",
        "root_tier",
        "spec_tier",
        "boundary",
        "criteria",
        "primary",
        "weight",
        "learning",
        "automatic_promotion",
        "official",
        "margins",
        "methods",
        "budget",
    ],
)
def test_confirmation_preflight_rejects_altered_release_before_payload(
    confirmation: SimpleNamespace, mutation: str
) -> None:
    e = confirmation
    if mutation == "inventory":
        e.spec["role"] = "inventory"
    elif mutation == "family":
        e.spec["family_id"] = "other-family"
    elif mutation == "active":
        e.spec["_family"] = {"reservation": {"status": "active"}}
    elif mutation == "rights":
        e.spec["rights"]["status"] = "restricted"
    elif mutation in ("group", "time"):
        e.spec[f"{mutation}_column"] = "provided-boundary"
    elif mutation == "schema_and_pin":
        e.spec["drop_columns"] = ["V1"]
        e.registration["dataset_spec_sha256"][study.CONFIRMATION_FAMILY] = fingerprint(e.spec)
    elif mutation == "root_tier":
        e.suite["evaluation_tier"] = "verified_entity"
    elif mutation == "spec_tier":
        e.spec["evaluation_tier"] = "verified_entity"
    elif mutation == "boundary":
        e.registration["claim_boundary"]["clinical_generalization_claims_allowed"] = True
    elif mutation == "criteria":
        e.registration["reporting_and_acceptance"]["acceptance"]["positive_vs_raw"] = False
    elif mutation == "primary":
        e.registration["primary_method"] = study.METHODS[1]
    elif mutation == "weight":
        e.registration["fixed_weight"] = 0.4
    elif mutation == "learning":
        e.registration["prediction_rule"]["weight_learning"] = True
    elif mutation == "automatic_promotion":
        e.registration["reporting_and_acceptance"]["automatic_default_promotion"] = True
    elif mutation == "official":
        e.registration["resplit_official_as_iid"] = False
    elif mutation == "margins":
        e.registration["classification_admission_margin"]["min_outer_test"] = 1
    elif mutation == "methods":
        e.registration["methods"].pop()
    else:
        e.registration["total_active_seconds_ceiling"] = 21600
    assert study.main(e.argv) == 2
    assert not e.acquisitions and not e.calls and not e.exposures
    assert not e.output.exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--seed", "42"],
        ["--seeds", "2031", "2032"],
        ["--threads", "4"],
        ["--max-rows", "470"],
        ["--case-timeout", "899"],
        ["--total-seconds", "14401"],
        ["--blend-mode", "oof_cv5"],
        ["--datasets", "other-family"],
    ],
)
def test_confirmation_settings_are_not_a_cli_grid(confirmation: SimpleNamespace, extra: list[str]) -> None:
    assert study.main([*confirmation.argv, *extra]) == 2
    assert not confirmation.acquisitions and not confirmation.output.exists()


def test_confirmation_requires_explicit_flag_and_released_suite(confirmation: SimpleNamespace) -> None:
    e = confirmation
    assert study.main([value for value in e.argv if value != "--allow-confirmation"]) == 2
    assert study.main(["--output", str(e.output), "--allow-confirmation"]) == 2
    assert not e.acquisitions and not e.calls


def test_confirmation_checks_exposure_before_prepare_and_prepare_only_does_not_admit(
    confirmation: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    e = confirmation

    def prepare(*args: Any) -> tuple[dict, dict, list]:
        assert [event[0] for event in e.exposures] == ["created", "check"]
        return cast(tuple[dict, dict, list], e.prepare(*args))

    monkeypatch.setattr(execution, "_prepare_dataset", prepare)
    assert study.main([*e.argv, "--prepare-only"]) == 0
    assert not e.calls and not any(event[0] == "admit" for event in e.exposures)
    assert len(results(e)["rows"]) == 15
    assert results(e)["confirmation_decision"]["status"] == "blocked"


def test_confirmation_previously_exposed_family_is_blocked_before_acquisition(
    confirmation: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    e = confirmation

    class Ledger:
        def __init__(self, path: Path):
            pass

        def check(self, *args: Any) -> None:
            raise ValueError("Family already exposed in another run")

        def admit(self, *args: Any) -> None:
            pytest.fail("A blocked family cannot be admitted")

    monkeypatch.setattr(study, "ExposureLedger", Ledger)
    assert study.main(e.argv) == 1
    assert not e.acquisitions and not e.calls
    assert len(results(e)["rows"]) == 15 and len(records(e)) == 9
    assert results(e)["confirmation_decision"]["status"] == "blocked"


@pytest.mark.parametrize("failed_preparation", [False, True])
def test_confirmation_contenders_reserve_before_protected_payload_open(
    confirmation: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, failed_preparation: bool
) -> None:
    e = confirmation
    gate = Barrier(2)
    prepare = execution._prepare_dataset
    opened = []

    class RacingLedger(ExposureLedger):
        def check(self, family_id: str, run_id: str, role: str) -> None:
            super().check(family_id, run_id, role)
            gate.wait(timeout=15)

    def acquire(spec: dict, *args: Any) -> Any:
        opened.append(spec["family_id"])
        if failed_preparation:
            raise ValueError("Synthetic failure after protected acquisition began")
        return prepare(spec, *args)

    monkeypatch.setattr(study, "ExposureLedger", RacingLedger)
    monkeypatch.setattr(execution, "_prepare_dataset", acquire)
    outputs = [e.output, e.output.with_name("competing-run")]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(study.main, [*e.argv, "--output", str(directory)]) for directory in outputs]
        statuses = [future.result(timeout=90) for future in futures]
    families = [spec["family_id"] for spec in e.suite["datasets"]]
    assert Counter(opened) == Counter(families)
    assert sorted(statuses) == ([1, 1] if failed_preparation else [0, 1])
    with sqlite3.connect(tabular.ROOT / "benchmarks" / ".cache" / "exposure.sqlite") as connection:
        reserved = connection.execute("SELECT family_id, COUNT(*) FROM exposure GROUP BY family_id").fetchall()
    assert dict(reserved) == dict.fromkeys(families, 1)


@pytest.mark.parametrize(
    "mutation",
    ["ids_missing", "ids_duplicate", "ids_changed", "official_missing", "groups", "times", "tier", "rows", "mapping"],
)
def test_confirmation_validates_original_ids_and_full_payload_before_iid_execution(
    confirmation: SimpleNamespace, mutation: str
) -> None:
    e = confirmation
    if mutation == "ids_missing":
        e.metadata["official_split_ids"].pop()
    elif mutation == "ids_duplicate":
        e.metadata["official_split_ids"][-1] = e.metadata["official_split_ids"][0]
    elif mutation == "ids_changed":
        e.metadata["official_split_ids"][-1]["fold"] = 10
    elif mutation == "official_missing":
        e.official.pop()
    elif mutation in ("groups", "times"):
        e.data["groups" if mutation == "groups" else "timestamps"] = pd.Series(np.arange(470))
    elif mutation == "tier":
        e.metadata["evaluation_tier"] = "verified_entity"
    elif mutation == "rows":
        e.data["X"] = e.data["X"].iloc[:-1]
        e.data["y"] = e.data["y"].iloc[:-1]
    else:
        e.metadata["column_mapping"]["x0"] = "Class"
    assert study.main(e.argv) == 1
    assert not e.calls
    assert sum(event[0] == "admit" for event in e.exposures) == 1
    assert len(records(e)) == 9 and len(results(e)["rows"]) == 15
    assert all(row["status"] == "dataset_error" for row in results(e)["rows"])


@pytest.mark.parametrize("margin", ["total", "train", "test"])
def test_confirmation_applies_actual_30_20_10_class_margins(confirmation: SimpleNamespace, margin: str) -> None:
    e = confirmation
    e.data["y"] = pd.Series(np.zeros(470, dtype=int))
    if margin == "total":
        e.data["y"].iloc[:20] = 1
    elif margin == "train":
        e.data["y"].iloc[:30] = 1
    else:
        e.data["y"].iloc[:16] = 1
        e.data["y"].iloc[157:173] = 1
        e.data["y"].iloc[314:322] = 1
    assert study.main(e.argv) == 1
    assert not e.calls
    assert results(e)["confirmation_decision"]["status"] == "blocked"


@pytest.mark.parametrize(
    "method,failed_rows", [(study.FIXED_MODE, 3), (study.NATIVE_REFERENCES[0], 1), (study.NATIVE_REFERENCES[1], 1)]
)
def test_confirmation_native_failure_is_single_and_bundle_failure_is_atomic(
    confirmation: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, method: str, failed_rows: int
) -> None:
    e = confirmation

    def run(payload: dict, directory: Path, timeout: float, **kwargs: Any) -> list[dict]:
        if payload["fold"] == 1 and payload["method"] == method:
            return tabular.failure_rows(payload, "timeout", "mock physical timeout", 900.0)
        return cast(list[dict], e.run(payload, directory, timeout, **kwargs))

    monkeypatch.setattr(tabular, "run_isolated", run)
    assert study.main(e.argv) == 1
    result = results(e)
    assert len(result["rows"]) == 15
    assert sum(row["status"] == "timeout" for row in result["rows"]) == failed_rows
    assert result["confirmation_decision"]["status"] == "blocked"
    assert result["confirmation_decision"]["valid_prediction_rows"] == 15 - failed_rows
    retried = []

    def retry(payload: dict, directory: Path, timeout: float, **kwargs: Any) -> list[dict]:
        retried.append((payload["method"], payload["fold"]))
        return cast(list[dict], e.run(payload, directory, timeout, **kwargs))

    monkeypatch.setattr(tabular, "run_isolated", retry)
    assert study.main([*e.resume, "--retry-failed"]) == 0
    assert retried == [(method, 1)]
    assert results(e)["confirmation_decision"]["status"] == "pass"


def test_confirmation_native_normalization_checks_caps_identity_and_retains_only_compact_metadata(
    confirmation: SimpleNamespace,
) -> None:
    e = confirmation
    assert study.main(e.argv) == 0
    payload = next(payload for payload, *_ in e.calls if payload["method"] in study.NATIVE_REFERENCES)
    for mutation in ("tier", "caps", "generation", "width", "params"):
        rows = native_success(payload)
        if mutation == "tier":
            rows[0]["evaluation_tier"] = "verified_entity"
        elif mutation == "caps":
            rows[0]["numeric_clip_quantiles"] = [0.01, 0.99]
        elif mutation == "generation":
            rows[0]["outer_reference_diagnostics"]["generation_preprocessing"] = "clipped-generation"
        elif mutation == "width":
            rows[0]["n_output_features"] -= 1
        else:
            rows[0]["parameters"] = {"num_leaves": 999}
        normalized = study.normalize_outcomes(payload, rows)
        assert len(normalized) == 1 and normalized[0]["status"] == "error"
    normalized = study.normalize_outcomes(payload, native_success(payload))
    assert normalized[0]["evaluation_tier"] == study.CONFIRMATION_TIER
    assert "private-category" not in json.dumps(normalized)
    again = study.normalize_outcomes(payload, normalized)
    assert again[0]["outer_reference_diagnostics"] == normalized[0]["outer_reference_diagnostics"]


def confirmation_decision(e: SimpleNamespace, replacements: dict[str, float]) -> dict:
    rows = results(e)["rows"]
    jobs = records(e)
    for row in rows:
        if row["method"] in replacements:
            row["loss"] = replacements[row["method"]]
    for job in jobs:
        job["outcomes"] = [
            next(
                row
                for row in rows
                if row["physical_job_id"] == job["physical_job_id"] and row["method"] == old["method"]
            )
            for old in job["outcomes"]
        ]
    return study._confirmation_decision(manifest(e), rows, jobs)


def test_confirmation_authoritative_gates_use_complete_absolute_family_means(confirmation: SimpleNamespace) -> None:
    e = confirmation
    assert study.main(e.argv) == 0
    assert confirmation_decision(e, {})["status"] == "pass"
    assert confirmation_decision(e, {study.METHODS[0]: 0.4})["status"] == "fail"
    for reference in (study.METHODS[1], *study.NATIVE_REFERENCES):
        assert confirmation_decision(e, {reference: 0.39})["status"] == "fail"
        blocked = confirmation_decision(e, {reference: 0.0})
        comparison = next(item for item in blocked["comparisons"] if item["reference_method"] == reference)
        assert blocked["status"] == "blocked"
        assert comparison["relative_gain"] is None and comparison["absolute_loss_difference"] == 0.4
    assert confirmation_decision(e, {study.METHODS[0]: 0.0})["status"] == "blocked"
    equal_limit = confirmation_decision(e, {study.METHODS[2]: 0.525})
    over_limit = confirmation_decision(e, {study.METHODS[2]: 0.526})
    assert equal_limit["comparisons"][0]["raw_family_harm_gate_passed"] is True
    assert over_limit["comparisons"][0]["raw_family_harm_gate_passed"] is False
    rows = results(e)["rows"]
    rows.pop()
    incomplete = study._confirmation_decision(manifest(e), rows, records(e))
    assert incomplete["status"] == "blocked" and incomplete["expected_prediction_rows"] == 15
    assert any(item["mean_loss"] is None for item in incomplete["family_losses"])
    declared = manifest(e)
    declared["evaluation_tier"] = "verified_entity"
    assert study._confirmation_decision(declared, results(e)["rows"], records(e))["status"] == "blocked"


@pytest.mark.parametrize("interruption", ["budget", "keyboard"])
def test_confirmation_unavailable_preparation_resumes_bundle_and_single_jobs(
    confirmation: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, interruption: str
) -> None:
    e = confirmation
    extra = ["--total-seconds", "1"] if interruption == "budget" else []
    if interruption == "keyboard":

        def interrupted(*args: Any) -> Any:
            raise KeyboardInterrupt

        monkeypatch.setattr(execution, "_prepare_dataset", interrupted)
    assert study.main([*e.argv, *extra]) == 1
    assert len(records(e)) == 9 and len(results(e)["rows"]) == 15
    assert all(study._unfinished_preparation(job, mode=study.FIXED_MODE, method=job["method"]) for job in records(e))
    assert not e.calls
    monkeypatch.setattr(execution, "_prepare_dataset", e.prepare)
    assert study.main([*e.resume, "--total-seconds", "14400"]) == 0
    assert len(e.calls) == 9
    assert results(e)["confirmation_decision"]["status"] == "pass"


def test_confirmation_resume_revalidates_original_official_identity_without_rerunning_models(
    confirmation: SimpleNamespace,
) -> None:
    e = confirmation
    assert study.main(e.argv) == 0
    e.metadata["official_split_ids"][0]["fold"] = 99
    assert study.main(e.resume) == 1
    assert len(e.calls) == 9 and len(e.acquisitions) == 2
    assert manifest(e)["identity_invalid"]
    assert results(e)["confirmation_decision"]["status"] == "blocked"


def test_confirmation_bundle_worker_calls_only_exact_frozen_core_with_bound_claims(
    confirmation: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarks import oof_blending

    e = confirmation
    assert study.main(e.argv) == 0
    payload = next(payload for payload, *_ in e.calls if payload["method"] == study.FIXED_MODE)
    calls = []

    def fit(train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame, test_y: pd.Series, **kwargs: Any) -> dict:
        calls.append(kwargs)
        assert train_X.equals(e.data["X"].iloc[payload["train_idx"]])
        assert train_y.equals(e.data["y"].iloc[payload["train_idx"]])
        assert test_X.equals(e.data["X"].iloc[payload["test_idx"]])
        assert test_y.equals(e.data["y"].iloc[payload["test_idx"]])
        assert kwargs["learner"] == "lightgbm" and kwargs["config"]["seed"] == 2031
        report = fixed_bundle_report()
        compact = success(payload)[0]
        report.update(n_input_features=16, n_output_features=17)
        report["diagnostics"] = {
            **compact["shared_bundle_diagnostics"],
            "member_count": 0,
            "members": [],
            "oof_complete_procedure_fit_count": 0,
        }
        return report

    monkeypatch.setattr(oof_blending, "fit_fixed_average_bundle", fit)
    rows = study.run_case(payload, e.root)
    assert len(calls) == 1 and len(rows) == 3
    assert all(row["status"] == "success" and row["role"] == "confirmation" for row in rows)
    assert all(row["protocol"] == study.CONFIRMATION_PROTOCOL for row in rows)
    assert all(row["claim_boundary"] == e.registration["claim_boundary"] for row in rows)


def test_real_prospective_confirmation_source_pin_is_exact_without_acquisition() -> None:
    reservation = json.loads(
        (study.DEFAULT_SUITE.parent / "qualified_binary_reservation_20261004.json").read_text(encoding="utf-8")
    )
    spec = reservation["datasets"][0]
    assert spec["name"] == study.CONFIRMATION_FAMILY and spec["role"] == "inventory"
    spec.update(role="confirmation", prior_exposure="audited_unexposed")
    assert fingerprint(spec) == study.CONFIRMATION_SPEC_SHA256
    assert spec["source"]["uci_id"] == 277 and spec["source"]["dataset_id"] == 1506
    assert spec["source"]["task_id"] == 9969
    assert spec["source"]["official_split_dimensions"] == {"repeats": 1, "folds": 10, "samples": 1}
    assert spec["claim_boundary"] == reservation["claim_boundary"]


def test_confirmation_accepts_descriptive_registration_notes_without_weakening_criteria(
    confirmation: SimpleNamespace,
) -> None:
    e = confirmation
    e.registration["reporting_and_acceptance"].update(
        family_mean_loss="Arithmetic mean over all three folds",
        automatic_default_promotion=False,
        automatic_sota=False,
    )
    assert study.main(e.argv) == 0
    assert results(e)["confirmation_decision"]["status"] == "pass"
