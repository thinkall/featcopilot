"""Canonical nine-source development study of single, ensemble and calibrated arms.

Four physical jobs produce six prediction cells per dataset/seed/fold/learner.
The two bundle outcomes share fitted bases; their attributed pipeline times are
not additive measurements. Only controller wall time consumes the active budget.
Filtered suites may retain unchanged canonical definitions for subset smoke runs.
The optional subsample85 mode uses three repeated stratified training bags;
its calibrated outcomes are diagnostic-only, not another selectable variant.
Explicit ``--allow-confirmation`` instead requires the locked two-family,
eight-arm standard_iid_unverified_entity registration. Its pinned public
normalized dataset definitions, claim boundary and fixed settings are checked
before exposure checks or acquisition; development source eligibility is unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import platform
import re
import sqlite3
import sys
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarks import execution, tabular
from benchmarks.run_state import (
    ExposureLedger,
    ResumeMismatchError,
    RunState,
    archive_results,
    archive_source,
    atomic_json,
    fingerprint,
    load_run_manifest,
)

logger = logging.getLogger(__name__)
PROTOCOL = "classification-calibration-development-v1"
SAMPLING_PROTOCOLS = {
    "cv3": PROTOCOL,
    "subsample85": "classification-calibration-development-subsample85-v1",
}
CORE_PROTOCOLS = {
    "cv3": "shared_retained_sigmoid_3fold_v1",
    "subsample85": "shared_retained_sigmoid_subsample85_v1",
}
METHODS = (
    "baseline-robust",
    "featcopilot-frequency-aligned-robust",
    "baseline-ensemble-robust",
    "baseline-calibrated-robust",
    "featcopilot-ensemble-robust",
    "featcopilot-calibrated-robust",
)
JOBS = {
    "single-raw": METHODS[:1],
    "single-fe": METHODS[1:2],
    "bundle-raw": METHODS[2:4],
    "bundle-fe": METHODS[4:6],
}
CONFIRMATION_METHODS = (*METHODS, "openfe-inductive-native-robust", "autofeat-native-robust")
CONFIRMATION_JOBS = {
    **JOBS,
    "single-openfe-native": CONFIRMATION_METHODS[6:7],
    "single-autofeat-native": CONFIRMATION_METHODS[7:8],
}
CONFIRMATION_FAMILIES = ("haberman-survival", "mammographic-mass")
CONFIRMATION_TIER = "standard_iid_unverified_entity"
CONFIRMATION_PROTOCOL = "classification-qualified-confirmation-subsample85-v1"
PRIMARY_METHODS = {"linear": METHODS[1], "lightgbm": METHODS[4]}
CONFIRMATION_ACCEPTANCE = {
    "positive_vs_each_raw_control": "For each learner, equal-family mean relative improvement must be strictly positive versus BOTH raw single and plain85 raw ensemble.",
    "family_harm": "No family/learner loss increase above5% versus either raw control.",
    "native_reference_gate": "For each learner, mean paired relative improvement >=0 against EACH completely paired clipped native FE reference. No post-outcome reference selection.",
    "coverage": "All96 outcomes and all72 scheduled physical jobs must be present; any unsuccessful/nonfinite/missing outcome or incomplete pair blocks broad positive claims. Retain failures/timeouts/RSS/budget skips.",
    "zero_reference_loss": "Do not substitute epsilons or delete a family; undefined relative gains block automatic acceptance and require explicit absolute-loss reporting.",
    "automatic_sota": False,
    "automatic_default_promotion": False,
}
CLAIM_FLAGS = {
    "patient_entity_independence_verified": False,
    "patient_entity_disjoint_claims_allowed": False,
    "clinical_generalization_claims_allowed": False,
    "pool_with_verified_entity_cohorts": False,
    "prospective_clinical_decision_use_authorized": False,
    "known_group_time_constraints_must_be_preserved": True,
}
CONFIRMATION_SETTINGS: dict[str, Any] = {
    "datasets": list(CONFIRMATION_FAMILIES),
    "methods": list(CONFIRMATION_METHODS),
    "learners": ["linear", "lightgbm"],
    "ensemble_sampling": "subsample85",
    "max_rows": 0,
    "folds": 3,
    "seeds": [2030],
    "threads": 8,
    "memory_limit_mb": 4096,
    "case_timeout": 900.0,
    "max_added_features": 50,
    "max_candidates": 1000,
    "stability_repeats": 3,
}
LEARNERS = ("linear", "lightgbm", "catboost", "xgboost")
DEFAULT_SUITE = tabular.ROOT / "benchmarks" / "suites" / "frequency_development_replay_20260924.json"
DEFAULT_DATASETS = [
    f"uci-{name}"
    for name in (
        "adult",
        "german-credit",
        "online-shoppers",
        "dry-bean",
        "steel-faults",
        "spambase",
        "banknote-authentication",
        "chronic-kidney-disease",
        "wisconsin-diagnostic",
    )
]
DEFAULTS: dict[str, Any] = {
    "suite": str(DEFAULT_SUITE),
    "datasets": DEFAULT_DATASETS,
    "methods": list(METHODS),
    "learners": ["linear", "lightgbm"],
    "seeds": [42],
    "folds": 3,
    "max_rows": 5000,
    "max_added_features": 50,
    "max_candidates": 1000,
    "stability_repeats": 3,
    "threads": 8,
    "memory_limit_mb": 4096,
    "case_timeout": 900.0,
    "ensemble_sampling": "cv3",
    "allow_confirmation": False,
}
REPORT_RESERVE_SECONDS = 10.0
METRICS = ("loss", "accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "brier_sum")
COST_NOTE = (
    "Bundle plain/calibrated outcomes reuse exactly the same three fitted base procedures. "
    "Row pipeline_seconds are standalone attributions with shared training, not additive measured cost. "
    "Physical-job dispatch wall time is recorded once per physical_job_id, including retries; active_seconds in "
    "manifest.json is cumulative whole-controller wall time, including preparation, reporting and archiving. "
    "The ten-second report reserve is conservative, not a hard bound; actual budget overshoot is recorded."
)


def _sampling_identity(config: dict) -> dict[str, Any]:
    mode = config.get("ensemble_sampling", "cv3")
    if not isinstance(mode, str) or mode not in SAMPLING_PROTOCOLS:
        raise ValueError("ensemble_sampling must be cv3 or subsample85")
    identity: dict[str, Any] = {"ensemble_sampling": mode, "protocol": SAMPLING_PROTOCOLS[mode]}
    if config.get("allow_confirmation"):
        identity.update(
            protocol=CONFIRMATION_PROTOCOL,
            evaluation_tier=CONFIRMATION_TIER,
            claim_boundary=config["claim_boundary"],
        )
    return identity


def _job_methods(config: dict) -> dict[str, tuple[str, ...]]:
    return CONFIRMATION_JOBS if config.get("allow_confirmation") else JOBS


def _diagnostic_only(config: dict, method: str) -> bool:
    return config.get("ensemble_sampling", "cv3") == "subsample85" and method in (METHODS[3], METHODS[5])


def validate_config(config: dict[str, Any], budget: float) -> None:
    """Reject unsupported or unbounded controls before acquiring data."""
    confirmation = config.get("allow_confirmation", False)
    if type(confirmation) is not bool:
        raise ValueError("allow_confirmation must be boolean")
    _sampling_identity(config)
    limits = {
        "folds": (2, 20),
        "max_rows": (0, 0) if confirmation else (1, 10_000_000),
        "max_added_features": (50, 50),
        "max_candidates": (1000, 1000),
        "stability_repeats": (1, 10),
        "threads": (1, 256),
        "memory_limit_mb": (1, 1_048_576),
    }
    for name, (lower, upper) in limits.items():
        value = config[name]
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    for name, value, ceiling in (
        ("total_seconds", budget, 72 * 3600),
        ("case_timeout", config["case_timeout"], 43200),
    ):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= ceiling:
            raise ValueError(f"{name} must be finite and in (0, {ceiling}]")
    if config["methods"] != list(CONFIRMATION_METHODS if confirmation else METHODS):
        raise ValueError(
            "The eight confirmation methods and matched controls are fixed"
            if confirmation
            else "The six methods and their matched controls are fixed for this study"
        )
    for key in ("datasets", "learners", "seeds"):
        values = config[key]
        if not isinstance(values, list) or not values or len(values) > 100:
            raise ValueError(f"{key} must be a nonempty list of at most 100 entries")
        if key == "seeds":
            valid = all(type(value) is int and 0 <= value < 2**32 for value in values)
        else:
            valid = all(
                isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) for value in values
            )
        if not valid or len(values) != len(set(values)):
            raise ValueError(f"{key} must contain unique valid identifiers")
    if not set(config["learners"]).issubset(LEARNERS):
        raise ValueError(f"learners must be selected from {LEARNERS}")
    if confirmation:
        for key, value in CONFIRMATION_SETTINGS.items():
            if config[key] != value:
                raise ValueError(f"Qualified confirmation fixes {key}={value!r}")
        if budget > 36000:
            raise ValueError("Qualified confirmation is capped at 36000 cumulative active seconds")


def _canonical_specs() -> dict[str, dict]:
    suite, legacy = execution._load_suite(DEFAULT_SUITE, False)
    by_name = {spec["name"]: spec for spec in suite["datasets"]}
    if legacy or len(by_name) != len(suite["datasets"]) or set(DEFAULT_DATASETS) - set(by_name):
        raise ValueError("Canonical reviewed suite must contain all nine original classification sources")
    return {name: by_name[name] for name in DEFAULT_DATASETS}


def _selected_specs(suite: dict, legacy: bool, names: list[str]) -> list[dict]:
    if legacy:
        raise ValueError("Legacy replays are outside this reviewed development study")
    by_name = {spec["name"]: spec for spec in suite["datasets"]}
    if len(by_name) != len(suite["datasets"]) or set(names) - set(by_name):
        raise ValueError("Dataset names must be unique and present in the selected suite")
    specs = [by_name[name] for name in names]
    canonical = _canonical_specs()
    if set(names) - set(canonical):
        raise ValueError("Only the nine canonical classification sources or unchanged subset smoke runs are allowed")
    if len({spec["family_id"] for spec in specs}) != len(specs):
        raise ValueError("Selected datasets must have unique origin families")
    for spec in specs:
        if spec.get("role") != "development" or spec.get("task") != "classification":
            raise ValueError("Only development classification datasets are supported")
        if execution.source_kind(spec) != "real_world" or spec.get("provider") == "diagnostic":
            raise ValueError("Simulation and synthetic diagnostics are outside this study")
        if spec.get("split_policy", "iid") != "iid" or any(
            spec.get(key) is not None for key in ("group_column", "time_column", "label_end_column")
        ):
            raise ValueError("Only reviewed IID datasets without group/time boundaries are supported")
        if spec.get("rights", {}).get("status") != "approved":
            raise ValueError("Every selected dataset requires approved catalog rights")
        if spec.get("_family", {}).get("reservation", {}).get("status") == "active":
            raise ValueError("Reserved confirmation candidates cannot enter the development study")
        if fingerprint(spec) != fingerprint(canonical[spec["name"]]):
            raise ValueError(f"{spec['name']}: source/family/schema differs from the canonical reviewed definition")
    return specs


def _bind_confirmation(suite: dict, legacy: bool, config: dict, budget: float) -> tuple[list[dict], dict]:
    registration = suite.get("preregistration")
    if legacy or not isinstance(registration, dict):
        raise ValueError("Qualified confirmation requires a new explicitly locked catalog preregistration")
    specs = suite["datasets"]
    if (
        len(specs) != 2
        or {spec["name"] for spec in specs} != set(CONFIRMATION_FAMILIES)
        or {spec["family_id"] for spec in specs} != set(CONFIRMATION_FAMILIES)
    ):
        raise ValueError("Qualified confirmation requires the entire two-family cohort without substitutions")
    boundary = registration.get("claim_boundary")
    if (
        not isinstance(boundary, dict)
        or boundary.get("evaluation_tier") != CONFIRMATION_TIER
        or any(boundary.get(key) is not value for key, value in CLAIM_FLAGS.items())
        or suite.get("claim_boundary") != boundary
        or suite.get("evaluation_tier") != CONFIRMATION_TIER
    ):
        raise ValueError("Qualified confirmation requires the explicit restricted IID claim boundary")
    expected_registration: dict[str, Any] = {
        "status": "locked_before_confirmation_model_execution",
        "controller_protocol": CONFIRMATION_PROTOCOL,
        "evaluation_tier": CONFIRMATION_TIER,
        "dataset_names": list(CONFIRMATION_FAMILIES),
        "family_ids": list(CONFIRMATION_FAMILIES),
        "methods": list(CONFIRMATION_METHODS),
        "learners": ["linear", "lightgbm"],
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
        "resplit_official_as_iid": True,
        "split_plan": {family: {"policy": "iid", "folds": 3} for family in CONFIRMATION_FAMILIES},
        "classification_admission_margin": {"min_total": 30, "min_outer_train": 20, "min_outer_test": 10},
        "primary_method_by_learner": PRIMARY_METHODS,
        "calibrated_outcomes_diagnostic_only": True,
        "selection_parameters": {
            "max_candidates": 1000,
            "max_added_features": 50,
            "stability_repeats": 3,
            "screening": "auto",
            "ensemble_sampling": "subsample85",
            "numeric_clip_quantiles": [0.0, 1.0],
        },
    }
    expected_search = {}
    for learner in ("linear", "lightgbm"):
        candidates = tabular.candidate_parameters(learner)
        parameter = next(iter(candidates[0]))
        expected_search[learner] = {
            parameter: [candidate[parameter] for candidate in candidates],
            f"selection_estimator_{parameter}": candidates[1][parameter],
        }
    expected_registration["learner_search"] = expected_search
    for key, value in expected_registration.items():
        if fingerprint(registration.get(key)) != fingerprint(value):
            raise ValueError(f"Qualified confirmation registration differs: {key}")
    _confirmation_rules(registration)
    pins = registration.get("dataset_spec_sha256")
    if not isinstance(pins, dict) or set(pins) != set(CONFIRMATION_FAMILIES):
        raise ValueError("Qualified confirmation must pin every normalized source/schema definition")
    for spec in specs:
        if (
            spec["name"] != spec["family_id"]
            or spec.get("role") != "confirmation"
            or spec.get("task") != "classification"
            or spec.get("provider") != "uci"
            or execution.source_kind(spec) != "real_world"
            or spec.get("evaluation_tier") != CONFIRMATION_TIER
            or spec.get("claim_boundary") != boundary
        ):
            raise ValueError("All selected sources must be classification confirmation in the same qualified IID tier")
        if (
            spec.get("split_policy") not in ("iid", "official")
            or any(spec.get(key) is not None for key in ("group_column", "time_column", "label_end_column"))
            or spec.get("rights", {}).get("status") != "approved"
            or spec.get("_family", {}).get("reservation", {}).get("status") == "active"
        ):
            raise ValueError("Qualified confirmation cannot bypass rights, reservations or known group/time boundaries")
        if pins[spec["name"]] != fingerprint({key: value for key, value in spec.items() if not key.startswith("_")}):
            raise ValueError(f"Locked source/schema definition changed: {spec['name']}")
    config.update(
        claim_boundary=boundary,
        evaluation_tier=CONFIRMATION_TIER,
        synthetic=False,
        screening="auto",
        numeric_clip_quantiles=[0.0, 1.0],
    )
    validate_config(config, budget)
    bound = execution.bind_preregistration({"manifests": [suite]}, specs, config, budget)
    if bound is None:
        raise ValueError("Qualified confirmation registration did not bind")
    return specs, bound


def _check_preparation_exposure(path: Path, run_id: str) -> None:
    if not path.exists():
        return
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        for family in CONFIRMATION_FAMILIES:
            previous = connection.execute(
                "SELECT 1 FROM exposure WHERE family_id = ? AND run_id != ? LIMIT 1", (family, run_id)
            ).fetchone()
            if previous is not None:
                raise ValueError(f"Confirmation family {family!r} was already exposed in another run")


def _outcomes(payload: dict, status: str, error: str, seconds: float = 0.0) -> list[dict]:
    rows = []
    for method in _job_methods(payload["config"])[payload["job_kind"]]:
        row = tabular.failure_rows({**payload, "method": method}, status, error, seconds)[0]
        row.update(
            **_sampling_identity(payload["config"]),
            diagnostic_only=_diagnostic_only(payload["config"], method),
            physical_job_id=payload["physical_job_id"],
            job_kind=payload["job_kind"],
            dataset_metadata=payload["dataset_metadata"],
            prepared_sha256=payload.get("prepared_sha256"),
            preparation_state=payload.get("preparation_state", "validated"),
            primary_predeclared=bool(payload["config"].get("allow_confirmation"))
            and PRIMARY_METHODS.get(payload["config"]["learners"][0]) == method,
        )
        rows.append(row)
    return rows


def _validate_bundle_sampling(diagnostics: Any, config: dict) -> None:
    mode = _sampling_identity(config)["ensemble_sampling"]
    if (
        not isinstance(diagnostics, dict)
        or diagnostics.get("ensemble_sampling") != mode
        or diagnostics.get("protocol") != CORE_PROTOCOLS[mode]
    ):
        raise ValueError("Bundle diagnostics sampling mode/protocol differ from the requested mode")
    if mode == "subsample85":
        if (
            diagnostics.get("sampling_design") != "repeated_stratified_holdout"
            or diagnostics.get("requested_train_fraction") != 0.85
            or diagnostics.get("requested_calibration_fraction") != 0.15
        ):
            raise ValueError("subsample85 requires the fixed repeated stratified 85/15 design")
        histogram = diagnostics.get("calibration_coverage_histogram")
        overlap = diagnostics.get("pairwise_overlap")
        members = diagnostics.get("members")
        if (
            not isinstance(histogram, list)
            or len(histogram) != 4
            or any(type(value) is not int or value < 0 for value in histogram)
            or not isinstance(overlap, list)
            or len(overlap) != 3
            or not isinstance(members, list)
            or len(members) != 3
            or type(diagnostics.get("distinct_train_bag_count")) is not int
            or not 1 <= diagnostics["distinct_train_bag_count"] <= 3
        ):
            raise ValueError("subsample85 requires three-member overlap and calibration membership diagnostics")
        for member in members:
            for key in ("base_train", "calibration_validation"):
                fraction = member[key]["fraction"]
                if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
                    raise ValueError("subsample85 requires actual member training/calibration fractions")


def normalize_outcomes(payload: dict, rows: Any) -> list[dict]:
    """Validate a physical job atomically, expanding supervisor failures to both arms.

    Parameters
    ----------
    payload : dict
        One physical job with its fixed outcome methods and original metadata.
    rows : object
        Untrusted worker result, including possible supervisor-only failure rows.

    Returns
    -------
    list[dict]
        Exactly the requested one or two canonical outcomes. Malformed or partial
        successes become explicit errors for the entire job, never copied scores.
    """
    expected = _job_methods(payload["config"])[payload["job_kind"]]
    sampling = _sampling_identity(payload["config"])
    try:
        if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Worker output must be a nonempty list of result objects")
        for row in rows:
            if "worker_python_hash_seed" in row and (
                type(row["worker_python_hash_seed"]) is not int or row["worker_python_hash_seed"] != payload["seed"]
            ):
                raise ValueError("Worker Python hash seed differs from the requested seed")
        failures = [row for row in rows if row.get("status") != "success"]
        if failures:
            if any(row.get("status") == "success" for row in rows):
                raise ValueError("A shared bundle cannot checkpoint partial success")
            status = str(failures[0].get("status", "error"))
            errors = "; ".join(str(row.get("error", "Worker returned an unscored outcome")) for row in failures)
            result = _outcomes(payload, status, errors)
            for row in result:
                row["reported_failures"] = [
                    {key: value for key, value in failure.items() if key not in (*METRICS, "average_precision")}
                    for failure in failures
                ]
                for key in (
                    "case_seconds",
                    "worker_wall_seconds",
                    "peak_tree_rss_mb",
                    "log",
                    "worker_python_hash_seed",
                ):
                    if key in failures[0]:
                        row[key] = failures[0][key]
            fingerprint(result)
            return result
        if len(rows) != len(expected) or {row.get("method") for row in rows} != set(expected):
            raise ValueError("Worker methods/count differ from the complete requested physical job")
        canonical = []
        for method in expected:
            row = next(row for row in rows if row["method"] == method)
            for key in ("dataset", "family_id", "role", "source", "task", "seed", "fold", "split_sha256"):
                if row.get(key) != payload[key]:
                    raise ValueError(f"Worker result identity differs: {key}")
            for key in ("physical_job_id", "job_kind"):
                if key in row and row[key] != payload[key]:
                    raise ValueError(f"Worker result identity differs: {key}")
            for key, expected_value in sampling.items():
                if key in row and row[key] != expected_value:
                    raise ValueError(f"Worker result identity differs: {key}")
            if payload["job_kind"].startswith("bundle"):
                _validate_bundle_sampling(row.get("shared_bundle_diagnostics"), payload["config"])
            if row.get("learner") != payload["config"]["learners"][0]:
                raise ValueError("Worker learner differs from physical job")
            for key in (*METRICS, "pipeline_seconds"):
                value = row.get(key)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"Worker success requires a finite, nonnegative {key}")
            for key in ("n_train", "n_test", "n_input_features", "n_output_features"):
                if type(row.get(key)) is not int or row[key] <= 0:
                    raise ValueError(f"Worker success requires a positive {key}")
            if row["n_train"] != len(payload["train_idx"]) or row["n_test"] != len(payload["test_idx"]):
                raise ValueError("Worker train/test row counts differ from the requested split")
            canonical.append(
                {
                    **row,
                    **sampling,
                    "diagnostic_only": _diagnostic_only(payload["config"], method),
                    "primary_predeclared": bool(payload["config"].get("allow_confirmation"))
                    and PRIMARY_METHODS.get(row["learner"]) == method,
                    "expected_cells": payload["expected_cells"],
                    "physical_job_id": payload["physical_job_id"],
                    "job_kind": payload["job_kind"],
                    "dataset_metadata": payload["dataset_metadata"],
                    "prepared_sha256": payload["prepared_sha256"],
                    "preparation_state": payload.get("preparation_state", "validated"),
                    "cost_note": COST_NOTE,
                }
            )
        fingerprint(canonical)
        return canonical
    except (ValueError, TypeError, KeyError, StopIteration) as exc:
        logger.error("Invalid worker output for %s: %s", payload["physical_job_id"], exc)
        failed = _outcomes(payload, "error", f"Invalid worker output: {exc}")
        if isinstance(rows, list):
            reported = [
                {key: str(row[key]) for key in ("method", "status", "error") if key in row}
                for row in rows
                if isinstance(row, dict)
            ]
            for row in failed:
                row["reported_outcomes"] = reported
        return failed


def _worker_data(payload: dict) -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
    for key, value in _sampling_identity(payload["config"]).items():
        if key in payload and payload[key] != value:
            raise ValueError(f"Worker payload {key} differs from its configuration")
    if payload.get("task") != "classification" or payload.get("job_kind") not in ("bundle-raw", "bundle-fe"):
        raise ValueError("This worker supports classification bundle jobs only")
    if payload.get("data_path") is not None:
        content = Path(payload["data_path"]).read_bytes()
        if tabular.digest(content) != payload.get("prepared_sha256"):
            raise ValueError("Prepared dataset changed before worker execution")
        data = pickle.loads(content)
    else:
        data = payload
    if any(data.get(key) is not None for key in ("groups", "timestamps", "label_end_times")):
        raise ValueError("Prediction bundles require IID data without group/time metadata")
    X, y = data["X"], data["y"]
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series) or not X.index.equals(y.index):
        raise ValueError("Worker X/y must be index-aligned pandas data")
    if len(X) != len(y) or y.isna().any():
        raise ValueError("Worker targets must be row-aligned and nonmissing")
    train, test = np.asarray(payload["train_idx"]), np.asarray(payload["test_idx"])
    for indices in (train, test):
        if (
            indices.ndim != 1
            or indices.dtype.kind not in "iu"
            or not len(indices)
            or len(np.unique(indices)) != len(indices)
            or indices.min() < 0
            or indices.max() >= len(X)
        ):
            raise ValueError("Worker split indices must be nonempty unique in-range integer positions")
    if np.intersect1d(train, test).size or execution.split_fingerprint(train, test) != payload["split_sha256"]:
        raise ValueError("Worker split fingerprint or train/test boundary differs")
    if not np.array_equal(np.unique(y.iloc[train]), np.unique(y.iloc[test])):
        raise ValueError("Every outer test class must align with the complete outer training classes")
    return X, y, train, test


def run_case(payload: dict, workspace: Path) -> list[dict]:
    """Evaluate a bundle using exactly the core's retained three fitted bases.

    Parameters
    ----------
    payload : dict
        Hash-bound prepared data or index-aligned X/y, outer indices and controls.
    workspace : pathlib.Path
        Supervisor-owned attempt directory; retained for the worker interface.

    Returns
    -------
    list[dict]
        Plain and sigmoid-calibrated metrics with shared JSON-only diagnostics.
        No fitted objects, probability matrices or categorical vocabularies.
    """
    from benchmarks.prediction_ensemble import fit_prediction_bundle

    X, y, train, test = _worker_data(payload)
    config = payload["config"]
    if len(config["learners"]) != 1:
        raise ValueError("Each physical job must select exactly one learner")
    started = time.perf_counter()
    result = fit_prediction_bundle(
        X.iloc[train],
        y.iloc[train],
        X.iloc[test],
        y.iloc[test],
        task="classification",
        learner=config["learners"][0],
        config={**config, "seed": payload["seed"]},
        engineered=payload["job_kind"] == "bundle-fe",
    )
    timing, diagnostics = result["timing"], result["diagnostics"]
    rows = _outcomes(payload, "success", "")
    for index, row in enumerate(rows):
        row.pop("error")
        calibrated = index == 1
        prediction_seconds = timing["calibrated_predict_seconds" if calibrated else "plain_predict_seconds"]
        training_seconds = (
            timing["fit_seconds"]
            if calibrated
            else timing.get("plain_training_estimate_seconds", timing["base_fit_seconds"])
        )
        scoring_seconds = timing.get(
            "sigmoid_score_seconds" if calibrated else "plain_score_seconds", prediction_seconds
        )
        row.update(
            result["calibrated_metrics" if calibrated else "plain_metrics"],
            n_train=len(train),
            n_test=len(test),
            n_input_features=diagnostics["n_input_features"],
            n_output_features=diagnostics["max_member_output_features"],
            max_member_output_features=diagnostics["max_member_output_features"],
            member_output_feature_counts=diagnostics["member_output_feature_counts"],
            feature_width_semantics="maximum member width, not a shared feature plan or feature union",
            shared_bundle_timing=timing,
            shared_bundle_diagnostics=diagnostics,
            standalone_training_seconds=training_seconds,
            predict_seconds=prediction_seconds,
            score_seconds=scoring_seconds,
            pipeline_seconds=training_seconds + scoring_seconds,
            case_seconds=time.perf_counter() - started,
            cost_note=COST_NOTE,
        )
    return normalize_outcomes(payload, rows)


def _calibration_contrasts(rows: list[dict], config: dict) -> list[dict]:
    summaries = []
    expected = {(seed, fold) for seed in config["seeds"] for fold in range(config["folds"])}
    for learner in config["learners"]:
        for representation, plain, calibrated in (("raw", METHODS[2], METHODS[3]), ("fe", METHODS[4], METHODS[5])):
            subset = [row for row in rows if row["learner"] == learner and row["method"] in (plain, calibrated)]
            families = sorted({row["family_id"] for row in subset})
            differences, brier_differences, reductions = [], [], []
            for family in families:
                members = [row for row in subset if row["family_id"] == family]
                paired = []
                for method in (plain, calibrated):
                    success = [row for row in members if row["method"] == method and row["status"] == "success"]
                    keys = {(row["seed"], row["fold"]) for row in success}
                    if len(success) != len(expected) or keys != expected:
                        break
                    paired.append({(row["seed"], row["fold"]): row for row in success})
                if len(paired) != 2 or any(
                    paired[0][key]["split_sha256"] != paired[1][key]["split_sha256"] for key in expected
                ):
                    continue
                plain_loss = float(np.mean([paired[0][key]["loss"] for key in expected]))
                calibrated_loss = float(np.mean([paired[1][key]["loss"] for key in expected]))
                differences.append(calibrated_loss - plain_loss)
                brier_differences.append(
                    float(np.mean([paired[1][key]["brier_sum"] - paired[0][key]["brier_sum"] for key in expected]))
                )
                if plain_loss > 0:
                    reductions.append(1.0 - calibrated_loss / plain_loss)
            summaries.append(
                {
                    **_sampling_identity(config),
                    "diagnostic_only": _diagnostic_only(config, calibrated),
                    "learner": learner,
                    "representation": representation,
                    "reference_method": plain,
                    "method": calibrated,
                    "unit": "complete-fold origin family; folds and paired prediction rows are not independent units",
                    "requested_families": len(families),
                    "complete_paired_families": len(differences),
                    "requested_prediction_rows": len(subset),
                    "mean_loss_difference": float(np.mean(differences)) if differences else None,
                    "mean_brier_sum_difference": float(np.mean(brier_differences)) if brier_differences else None,
                    "mean_loss_reduction": float(np.mean(reductions)) if reductions else None,
                    "relative_loss_families": len(reductions),
                }
            )
    return summaries


def _confirmation_rules(registration: dict) -> dict:
    rules = registration.get("reporting_and_acceptance")
    expected = {
        "evaluation_tier": CONFIRMATION_TIER,
        "frozen_primary_method_by_learner": PRIMARY_METHODS,
        "primary_loss": "log_loss",
        "primary_raw_comparators": [METHODS[0], METHODS[2]],
        "native_fe_comparators": list(CONFIRMATION_METHODS[6:]),
        "acceptance": CONFIRMATION_ACCEPTANCE,
    }
    if not isinstance(rules, dict) or any(
        fingerprint(rules.get(key)) != fingerprint(value) for key, value in expected.items()
    ):
        raise ValueError("Qualified confirmation reporting/acceptance differs from the locked decision rules")
    return rules


def _confirmation_decision(manifest: dict, rows: list[dict], physical_jobs: list[dict]) -> dict:
    registration, config = manifest["preregistration"], manifest["config"]
    rules = _confirmation_rules(registration)
    identity = _sampling_identity(config)
    families, learners = registration["family_ids"], registration["learners"]
    schedule = [(seed, fold) for seed in registration["seeds"] for fold in range(registration["folds"])]
    expected = {
        (family, learner, method, seed, fold)
        for family in families
        for learner in learners
        for method in registration["methods"]
        for seed, fold in schedule
    }
    expected_jobs = {
        RunState.cell_id(family, seed, fold, kind, learner): [
            (family, learner, method, seed, fold) for method in methods
        ]
        for family in families
        for learner in learners
        for seed, fold in schedule
        for kind, methods in CONFIRMATION_JOBS.items()
    }
    splits = {(item["dataset"], item["seed"], item["fold"]): item["sha256"] for item in manifest["splits"]}
    by_cell: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in ("family_id", "learner", "method", "seed", "fold"))
        by_cell.setdefault(key, []).append(row)
    by_job: dict[str, list[dict]] = {}
    for job in physical_jobs:
        by_job.setdefault(job.get("physical_job_id", ""), []).append(job)
    blockers = []
    if set(by_cell) != expected or len(rows) != registration["expected_cells"]:
        blockers.append("Prediction-cell coverage differs from the complete locked schedule")
    if set(by_job) != set(expected_jobs) or len(physical_jobs) != registration["expected_physical_jobs"]:
        blockers.append("Physical-job coverage differs from the complete locked schedule")
    if manifest.get("source_changed"):
        blockers.append("Source identity changed; the run is invalidated")
    losses: dict[tuple, float] = {}
    for key in sorted(expected):
        candidates = by_cell.get(key, [])
        if len(candidates) != 1:
            blockers.append(f"{key}: missing or duplicate prediction cell")
            continue
        row = candidates[0]
        metrics_valid = all(
            not isinstance(row.get(metric), bool)
            and isinstance(row.get(metric), (int, float))
            and math.isfinite(row[metric])
            and row[metric] >= 0
            for metric in METRICS
        )
        if (
            row.get("status") != "success"
            or not metrics_valid
            or row.get("dataset") != key[0]
            or row.get("role") != "confirmation"
            or any(row.get(field) != value for field, value in identity.items())
            or row.get("split_sha256") in (None, "unavailable")
            or row.get("split_sha256") != splits.get((key[0], key[3], key[4]))
        ):
            blockers.append(f"{key}: unsuccessful, nonfinite or mismatched prediction cell")
            continue
        losses[key] = float(row["loss"])
    complete_jobs = 0
    for job_id, keys in expected_jobs.items():
        records = by_job.get(job_id, [])
        if (
            len(records) != 1
            or records[0].get("status") != "success"
            or any(key not in losses or by_cell[key][0].get("physical_job_id") != job_id for key in keys)
            or records[0].get("outcomes") != [by_cell[key][0] for key in keys]
        ):
            blockers.append(f"{job_id}: physical job or its atomic outcomes are incomplete/mismatched")
        else:
            complete_jobs += 1
    family_losses = []
    means: dict[tuple[str, str, str], float | None] = {}
    for family in families:
        for learner in learners:
            for method in registration["methods"]:
                values = [
                    losses[(family, learner, method, seed, fold)]
                    for seed, fold in schedule
                    if (family, learner, method, seed, fold) in losses
                ]
                mean_loss = (
                    math.fsum(value / len(schedule) for value in values) if len(values) == len(schedule) else None
                )
                means[family, learner, method] = mean_loss
                family_losses.append(
                    {
                        **identity,
                        "family_id": family,
                        "learner": learner,
                        "method": method,
                        "requested_folds": len(schedule),
                        "successful_folds": len(values),
                        "mean_loss": mean_loss,
                    }
                )
    comparisons, failures = [], []
    for learner in learners:
        policy = rules["frozen_primary_method_by_learner"][learner]
        for reference in [*rules["primary_raw_comparators"], *rules["native_fe_comparators"]]:
            raw_control = reference in rules["primary_raw_comparators"]
            paired_families = []
            for family in families:
                policy_loss, reference_loss = means[family, learner, policy], means[family, learner, reference]
                gain, reason = None, None
                if policy_loss is None or reference_loss is None:
                    reason = "incomplete_family_folds"
                elif reference_loss <= 0:
                    reason = "undefined_zero_reference_loss"
                else:
                    gain = 1.0 - policy_loss / reference_loss
                    if not math.isfinite(gain):
                        gain, reason = None, "undefined_nonfinite_relative_gain"
                paired_families.append(
                    {
                        **identity,
                        "family_id": family,
                        "policy_mean_loss": policy_loss,
                        "reference_mean_loss": reference_loss,
                        "relative_gain": gain,
                        "undefined_reason": reason,
                        "raw_family_harm_passed": (
                            policy_loss <= reference_loss * 1.05
                            if raw_control
                            and gain is not None
                            and policy_loss is not None
                            and reference_loss is not None
                            else None
                        ),
                    }
                )
            gains = [family["relative_gain"] for family in paired_families if family["relative_gain"] is not None]
            complete = len(gains) == len(families)
            mean_gain = math.fsum(value / len(families) for value in gains) if complete else None
            mean_passed = (mean_gain > 0 if raw_control else mean_gain >= 0) if mean_gain is not None else None
            harm_passed = (
                all(family["raw_family_harm_passed"] for family in paired_families)
                if complete and raw_control
                else None
            )
            status = (
                "blocked" if not complete else "pass" if mean_passed and (not raw_control or harm_passed) else "fail"
            )
            if not complete:
                blockers.extend(
                    f"{learner}/{reference}/{family['family_id']}: {family['undefined_reason']}"
                    for family in paired_families
                    if family["undefined_reason"] is not None
                )
            if mean_passed is False:
                failures.append(f"{learner}/{reference}: locked mean-gain gate failed")
            failures.extend(
                f"{learner}/{reference}/{family['family_id']}: raw family harm exceeds 5%"
                for family in paired_families
                if family["raw_family_harm_passed"] is False
            )
            comparisons.append(
                {
                    **identity,
                    "learner": learner,
                    "policy_method": policy,
                    "reference_method": reference,
                    "reference_kind": "raw_control" if raw_control else "native_fe",
                    "requested_families": len(families),
                    "complete_paired_families": sum(
                        family["policy_mean_loss"] is not None and family["reference_mean_loss"] is not None
                        for family in paired_families
                    ),
                    "defined_relative_families": len(gains),
                    "mean_relative_gain": mean_gain,
                    "median_relative_gain": float(np.median(gains)) if complete else None,
                    "mean_gain_gate_passed": mean_passed,
                    "raw_family_harm_gate_passed": harm_passed,
                    "status": status,
                    "families": paired_families,
                }
            )
    status = "blocked" if blockers else "fail" if failures else "pass"
    return {
        **identity,
        "status": status,
        "locked_criteria_passed": status == "pass",
        "automatic_sota": False,
        "automatic_default_promotion": False,
        "scope": "Only the locked two-family standard IID tier with unverified entity mapping; no robust, patient-independent, clinical, general-SOTA or default-promotion claim.",
        "rules": rules,
        "primary_method_by_learner": rules["frozen_primary_method_by_learner"],
        "requested_families": len(families),
        "expected_prediction_rows": registration["expected_cells"],
        "observed_prediction_rows": len(rows),
        "valid_prediction_rows": len(losses),
        "expected_physical_jobs": registration["expected_physical_jobs"],
        "observed_physical_jobs": len(physical_jobs),
        "complete_physical_jobs": complete_jobs,
        "family_losses": family_losses,
        "comparisons": comparisons,
        "blockers": blockers,
        "failures": failures,
    }


def _confirmation_decision_lines(decision: dict) -> list[str]:
    lines = [
        "",
        "## Locked confirmation decision (authoritative; descriptive summaries below are not acceptance gates)",
        "",
        f"**{decision['status'].upper()}**: {decision['valid_prediction_rows']}/{decision['expected_prediction_rows']} valid prediction rows; "
        f"{decision['complete_physical_jobs']}/{decision['expected_physical_jobs']} complete physical jobs; "
        f"all {decision['requested_families']} registered families retained.",
        decision["scope"],
        "Undefined gains remain undefined; no epsilon, successful-family-only mean, or calibrated-policy substitution. "
        "Even PASS never authorizes automatic SOTA claims or default promotion.",
        "",
        "| Learner | Fixed policy | Comparator | Defined / requested families | Mean gain | Median gain | Mean gate | Raw harm gate | Decision |",
        "|---|---|---|---:|---:|---:|---|---|---|",
    ]
    for comparison in decision["comparisons"]:
        lines.append(
            f"| {comparison['learner']} | {comparison['policy_method']} | {comparison['reference_method']} | "
            f"{comparison['defined_relative_families']} / {comparison['requested_families']} | "
            f"{comparison['mean_relative_gain']} | {comparison['median_relative_gain']} | "
            f"{comparison['mean_gain_gate_passed']} | {comparison['raw_family_harm_gate_passed']} | {comparison['status']} |"
        )
    lines.extend(
        [
            "",
            "### Both-family absolute losses and policy gains",
            "",
            "| Learner | Comparator | Family | Policy mean loss | Reference mean loss | Gain | Undefined reason | Raw harm passed |",
            "|---|---|---|---:|---:|---:|---|---|",
        ]
    )
    for comparison in decision["comparisons"]:
        for family in comparison["families"]:
            lines.append(
                f"| {comparison['learner']} | {comparison['reference_method']} | {family['family_id']} | "
                f"{family['policy_mean_loss']} | {family['reference_mean_loss']} | {family['relative_gain']} | "
                f"{family['undefined_reason']} | {family['raw_family_harm_passed']} |"
            )
    if decision["blockers"]:
        lines.extend(["", "Blocking evidence:", *[f"- {reason}" for reason in decision["blockers"]]])
    if decision["failures"]:
        lines.extend(["", "Failed locked criteria:", *[f"- {reason}" for reason in decision["failures"]]])
    return lines


def write_report(output: Path, manifest: dict, rows: list[dict], physical_jobs: list[dict]) -> None:
    """Archive exact rows and family-paired descriptive contrasts without double billing.

    Parameters
    ----------
    output : pathlib.Path
        Existing study directory.
    manifest : dict
        Frozen design and requested cohort, including failed source families.
    rows : list[dict]
        One scored or explicitly unscored row for every prediction cell.
    physical_jobs : list[dict]
        Atomic physical-job records, carrying cumulative measured dispatch time.

    Notes
    -----
    Contrasts use complete-fold origin families, not independent prediction rows.
    Shared training is counted once in physical cost, but attributed to both arms.
    For qualified confirmation, ``confirmation_decision`` is the authoritative
    locked-policy gate; descriptive summaries never supply acceptance decisions.
    """
    config = manifest["config"]
    confirmation = config.get("allow_confirmation", False)
    sampling = _sampling_identity(config)
    subsample = sampling["ensemble_sampling"] == "subsample85"
    if any(manifest.get(key) != value for key, value in sampling.items()) or any(
        row.get(key) != value for row in rows for key, value in sampling.items()
    ):
        raise ValueError("Reports require one explicit sampling mode/protocol; mixed or unlabelled rows are forbidden")
    if confirmation and any(row["role"] != "confirmation" for row in rows):
        raise ValueError("Qualified confirmation cannot be pooled with other roles or evaluation tiers")
    decision = _confirmation_decision(manifest, rows, physical_jobs) if confirmation else None
    summary = tabular.summarize(rows, config["folds"], len(config["seeds"]), baseline_method=METHODS[0])
    raw85_summary = (
        tabular.summarize(rows, config["folds"], len(config["seeds"]), baseline_method=METHODS[2])
        if confirmation
        else []
    )
    matched: list[dict] = []
    for architecture, raw, engineered in (
        ("single", METHODS[0], METHODS[1]),
        ("plain-ensemble", METHODS[2], METHODS[4]),
        ("sigmoid-ensemble", METHODS[3], METHODS[5]),
    ):
        paired = tabular.summarize(
            [row for row in rows if row["method"] in (raw, engineered)],
            config["folds"],
            len(config["seeds"]),
            baseline_method=raw,
        )
        matched.extend({**item, "architecture": architecture} for item in paired if item["method"] == engineered)
    contrasts = _calibration_contrasts(rows, config)
    for item in [*summary, *matched, *raw85_summary]:
        item.update(
            **sampling,
            diagnostic_only=_diagnostic_only(config, item["method"]),
            primary_predeclared=confirmation and PRIMARY_METHODS.get(item["learner"]) == item["method"],
        )
    sampling_diagnostics = []
    for row in rows:
        if row["status"] == "success" and row["method"] in (METHODS[2], METHODS[4]):
            _validate_bundle_sampling(row["shared_bundle_diagnostics"], config)
            sampling_diagnostics.append(
                {
                    **sampling,
                    **{key: row[key] for key in ("physical_job_id", "dataset", "learner", "seed", "fold", "method")},
                    "diagnostics": row["shared_bundle_diagnostics"],
                }
            )
    costs = {
        "note": COST_NOTE,
        "active_seconds_authority": "manifest.json",
        "measured_physical_job_seconds_including_retries": sum(
            job.get("physical_seconds", 0.0) for job in physical_jobs
        ),
        "physical_clock_scope": "dispatch preflight, exposure admission and supervised worker startup/teardown",
        "recorded_physical_jobs": len(physical_jobs),
        "attempted_physical_jobs": sum(job.get("attempt", 0) > 0 for job in physical_jobs),
    }
    atomic_json(
        output / "results.json",
        {
            **sampling,
            "run_id": manifest["run_id"],
            "config": config,
            "expected_physical_jobs": manifest["expected_physical_jobs"],
            "planned_cells": manifest["planned_cells"],
            "rows": rows,
            "summary": summary,
            "matched_architecture_summary": matched,
            "calibration_operation_summary": contrasts,
            "sampling_diagnostics": sampling_diagnostics,
            **(
                {
                    "matched_raw85_summary": raw85_summary,
                    "primary_method_by_learner": PRIMARY_METHODS,
                    "claim_boundary": config["claim_boundary"],
                    "confirmation_decision": decision,
                }
                if confirmation
                else {}
            ),
            "physical_cost": costs,
            "physical_jobs": [{key: value for key, value in job.items() if key != "outcomes"} for job in physical_jobs],
        },
    )
    lines = [
        "# Robust classification calibration development study",
        "",
        (
            "Locked qualified confirmation: standard public row-level IID evidence with unresolved patient/entity mapping."
            if confirmation
            else "Exposed development data only; not fresh confirmation. No source is silently dropped."
        ),
        (
            f"Evaluation tier: `{CONFIRMATION_TIER}`. No patient/entity-independent, clinical-generalization, "
            "prospective clinical-use or verified-entity pooled claim is permitted. Known group/time constraints remain binding."
            if confirmation
            else f"Cohort scope: {manifest['cohort_scope']}; subset smoke results are not the full nine-family study."
        ),
        f"Sampling mode: `{sampling['ensemble_sampling']}`; protocol: `{sampling['protocol']}`.",
        (
            "Three repeated stratified training bags request 85% training / 15% calibration per member. "
            "Returned row order is preserved; overlapping bags are not a three-fold partition or pooled OOF sample. "
            "Calibrated outcomes and every contrast involving them are diagnostic-only, not a variant-selection grid."
            if subsample
            else "Three-fold sigmoid calibration and plain prediction averaging share the exact same fitted bases."
        ),
        "Member feature counts / maximum width describe distinct fold-local plans, not a common feature union.",
        "Brier-sum is mean(sum over classes of squared probability error), including both binary classes.",
        "",
        COST_NOTE,
        "",
        f"Planned: {manifest['expected_physical_jobs']} physical jobs / {manifest['planned_cells']} prediction rows.",
        f"Scored: {sum(row['status'] == 'success' for row in rows)} / {len(rows)} prediction rows.",
        *(_confirmation_decision_lines(decision) if decision is not None else []),
        "",
        "## Relative to raw single-model baseline",
        "",
        "| Learner | Method | Complete families / requested | Mean loss reduction | Diagnostic only |",
        "|---|---|---:|---:|---|",
    ]
    for item in summary:
        value = item["mean_loss_reduction"]
        reduction = "unscored" if value is None else f"{value:.6f}"
        lines.append(
            f"| {item['learner']} | {item['method']} | "
            f"{item['complete_paired_families']} / {item['requested_families']} | {reduction} | {item['diagnostic_only']} |"
        )
    if confirmation:
        lines.extend(
            [
                "",
                "Predeclared policy: linear uses single FE; LightGBM uses plain subsample85 FE. "
                "No post-score learner, dataset or method choice. Calibration remains diagnostic-only.",
                "OpenFE/AutoFeat references use the existing native-width single-model adaptations with matched final "
                "clipping, not an ensemble architecture or a paper-best performance claim.",
                "No robust guarantee follows from this two-family tier.",
                "",
                "Full claim boundary: `" + json.dumps(config["claim_boundary"], sort_keys=True) + "`.",
                "",
                "## All arms and native references versus matched raw subsample85",
                "",
                "| Learner | Method | Complete families / requested | Mean loss reduction | Diagnostic only |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for item in raw85_summary:
            value = item["mean_loss_reduction"]
            reduction = "unscored" if value is None else f"{value:.6f}"
            lines.append(
                f"| {item['learner']} | {item['method']} | "
                f"{item['complete_paired_families']} / {item['requested_families']} | {reduction} | {item['diagnostic_only']} |"
            )
    lines.extend(
        [
            "",
            "## FE versus matched raw representation",
            "",
            "| Learner | Architecture | Complete families / requested | Mean loss reduction | Diagnostic only |",
            "|---|---|---:|---:|---|",
        ]
    )
    for item in matched:
        value = item["mean_loss_reduction"]
        reduction = "unscored" if value is None else f"{value:.6f}"
        lines.append(
            f"| {item['learner']} | {item['architecture']} | "
            f"{item['complete_paired_families']} / {item['requested_families']} | {reduction} | {item['diagnostic_only']} |"
        )
    lines.extend(
        [
            "",
            "## Calibration operation: calibrated minus plain" + (" (diagnostic-only)" if subsample else ""),
            "",
            "Equal-weight complete-fold families; negative loss/Brier differences favor calibration.",
            "",
            "| Learner | Representation | Complete / requested families | Loss difference | Brier-sum difference |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for item in contrasts:
        loss = item["mean_loss_difference"]
        brier = item["mean_brier_sum_difference"]
        lines.append(
            f"| {item['learner']} | {item['representation']} | "
            f"{item['complete_paired_families']} / {item['requested_families']} | "
            f"{'unscored' if loss is None else f'{loss:.6f}'} | "
            f"{'unscored' if brier is None else f'{brier:.6f}'} |"
        )
    if subsample:
        lines.extend(
            [
                "",
                "## Repeated stratified training bags: observed membership",
                "",
                "One row per scored physical bundle (not per prediction arm). Fractions use actual member sizes. "
                "Coverage counts outer-training rows held out 0, 1, 2, or 3 times. Overlap lists pairwise "
                "train/calibration intersections for pairs 0-1, 0-2, 1-2; no pooled OOF claim.",
                "",
                "| Physical job | Actual train fractions | Actual calibration fractions | Coverage 0..3 | Overlap | Distinct bags |",
                "|---|---|---|---|---|---:|",
            ]
        )
        for item in sampling_diagnostics:
            diagnostics = item["diagnostics"]
            train_fractions = [member["base_train"]["fraction"] for member in diagnostics["members"]]
            calibration_fractions = [member["calibration_validation"]["fraction"] for member in diagnostics["members"]]
            lines.append(
                f"| {item['physical_job_id']} | {train_fractions} | {calibration_fractions} | "
                f"{diagnostics['calibration_coverage_histogram']} | {diagnostics['pairwise_overlap']} | "
                f"{diagnostics['distinct_train_bag_count']} |"
            )
    lines.extend(
        [
            "",
            "Matched raw/FE architecture summaries, all errors and original dataset metadata: `results.json`.",
            "Exact lossless gzip parts (each <=480 KiB): `archive.json`. Source allowlist: `manifest.json`.",
            "Final cumulative controller time and any budget overshoot: `manifest.json`.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    archive_results(output / "results.json")


def execute(args: argparse.Namespace) -> int:
    """Prepare or resume this fixed six-arm study, checkpointing physical jobs atomically."""
    started = time.perf_counter()
    output = args.output.resolve()
    previous = load_run_manifest(output) if args.resume else {}
    if previous.get("source_changed"):
        raise ResumeMismatchError("Cannot resume a study invalidated by changed source")
    confirmation = (
        args.allow_confirmation
        if args.allow_confirmation is not None
        else previous.get("config", {}).get("allow_confirmation", False)
    )
    defaults = {**DEFAULTS, **(CONFIRMATION_SETTINGS if confirmation else {})}
    config: dict[str, Any] = {
        key: (
            getattr(args, key, None)
            if getattr(args, key, None) is not None
            else previous.get("config", {}).get(key, default)
        )
        for key, default in defaults.items()
    }
    config["suite"] = str(Path(config["suite"]).resolve())
    budget = (
        args.total_seconds
        if args.total_seconds is not None
        else previous.get("budget_seconds", 36000.0 if confirmation else 43200.0)
    )
    if not confirmation:
        validate_config(config, budget)
    if args.retry_failed and not args.resume:
        raise ValueError("--retry-failed requires --resume")
    if args.resume and budget < previous["budget_seconds"]:
        raise ResumeMismatchError("Only increasing the operational total-time ceiling is allowed on resume")
    suite, legacy = execution._load_suite(Path(config["suite"]), confirmation)
    registration = None
    if confirmation:
        specs, registration = _bind_confirmation(suite, legacy, config, budget)
    else:
        specs = _selected_specs(suite, legacy, config["datasets"])
    job_methods = _job_methods(config)
    identity = {
        **_sampling_identity(config),
        "config": config,
        "suite": suite,
        "suite_file_sha256": tabular.digest(Path(config["suite"]).read_bytes()),
        "source_sha256": tabular.source_fingerprint(),
        "versions": execution.environment_versions(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    paths = tabular.source_files()
    state = RunState(output, identity, resume=args.resume, initialization={"config": config, "budget_seconds": budget})
    prior_elapsed = state.elapsed_seconds
    deadline = started + max(0.0, budget - prior_elapsed)
    source_archive = previous.get("source_archive_sha256")
    count = len(specs) * len(config["seeds"]) * config["folds"] * len(config["learners"])
    manifest = {
        **identity,
        "run_id": state.run_id,
        "source_archive_sha256": source_archive,
        "source_allowlist": [path.relative_to(tabular.ROOT).as_posix() for path in paths],
        "budget_seconds": budget,
        "report_reserve_seconds": REPORT_RESERVE_SECONDS,
        "budget_scope": "This run's cumulative active-time ceiling only; remaining cycle usage is parent-ledger authoritative.",
        "started_utc": previous.get("started_utc", datetime.now(timezone.utc).isoformat()),
        "expected_physical_jobs": count * len(job_methods),
        "planned_cells": count * len(config["methods"]),
        "cohort_scope": (
            "locked_qualified_confirmation"
            if confirmation
            else "canonical_nine" if len(specs) == len(DEFAULT_DATASETS) else "canonical_subset_smoke"
        ),
        **({"preregistration": registration, "primary_method_by_learner": PRIMARY_METHODS} if confirmation else {}),
        "datasets": [],
        "splits": [],
        "prepare_only": args.prepare_only,
        "resumed": args.resume,
        "source_changed": False,
        "cost_note": COST_NOTE,
        "ensemble_protocol": {
            "ensemble_sampling": config["ensemble_sampling"],
            "protocol": CORE_PROTOCOLS[config["ensemble_sampling"]],
            "sampling_design": (
                "repeated_stratified_holdout"
                if config["ensemble_sampling"] == "subsample85"
                else "stratified_three_fold"
            ),
            "calibrated_outcomes_diagnostic_only": config["ensemble_sampling"] == "subsample85",
            "members": 3,
            "calibration": "sigmoid",
            "calibrator_n_jobs": 1,
            "shared_fitted_bases": True,
            "feature_budget": {"max_added_features": 50, "max_candidates": 1000},
        },
    }
    if not state.initialize_source(
        manifest, tabular.ROOT, paths, suite, started=started, archive_writer=archive_source
    ):
        return 1
    prepared_dir = output / "prepared"
    prepared_dir.mkdir(exist_ok=True)
    if not args.resume:
        atomic_json(output / "manifest.json", manifest)
    jobs = []
    interrupted = False
    ledger: ExposureLedger | None = None
    admitted_families: set[str] = set()
    if confirmation:
        ledger_path = tabular.ROOT / "benchmarks" / ".cache" / "exposure.sqlite"
        if args.prepare_only:
            _check_preparation_exposure(ledger_path, state.run_id)
        else:
            ledger = ExposureLedger(ledger_path)
            for spec in specs:
                ledger.check(spec["family_id"], state.run_id, "confirmation")
    previous_datasets = {item["name"]: item for item in previous.get("datasets", [])}
    for spec in specs:
        name, source = spec["name"], execution.source_kind(spec)
        metadata = {**spec, "source": source, "source_specification": spec.get("source")}
        data_path = prepared_dir / f"{name}.pkl"
        prepared_sha256 = None
        error, status = None, "dataset_error"
        cached = state.datasets.get(name)
        preparation_state = "unfinished"
        split_rows: list[tuple[int, int, Any, Any]] = [
            (seed, fold, None, None) for seed in config["seeds"] for fold in range(config["folds"])
        ]
        try:
            old = previous_datasets.get(name, {})
            if cached is None:
                if old.get("error") and old.get("preparation_status") == "dataset_error" and not args.retry_failed:
                    raise ValueError(old["error"])
                if interrupted:
                    raise KeyboardInterrupt
                if time.perf_counter() >= deadline - REPORT_RESERVE_SECONDS:
                    raise TimeoutError("Active-time budget exhausted before preparation (including report reserve)")
            if confirmation and not args.prepare_only:
                assert ledger is not None
                ledger.admit(spec["family_id"], state.run_id, "confirmation")
                admitted_families.add(spec["family_id"])
            reuse_hash = (
                cached["prepared_sha256"]
                if cached is not None
                else old.get("prepared_sha256") if data_path.exists() else None
            )
            prepared_data = None
            if reuse_hash is not None:
                content = data_path.read_bytes()
                if tabular.digest(content) != reuse_hash:
                    raise ResumeMismatchError(f"Cannot resume: prepared data changed for {name}")
                prepared_data = pickle.loads(content)
                prepared_sha256 = reuse_hash
            # Catalog acquisition validates the original raw payload and checksum pin
            # even when a prepared pickle and completed jobs already exist.
            data, acquired, official = execution._prepare_dataset(spec, source, suite, False, config, tabular.ROOT)
            metadata = {**metadata, **acquired}
            if (official is not None and not confirmation) or any(
                data.get(key) is not None for key in ("groups", "timestamps", "label_end_times")
            ):
                raise ValueError("Prepared dataset contains unsupported official/group/time boundaries")
            if confirmation:
                if (
                    metadata.get("role") != "confirmation"
                    or metadata.get("evaluation_tier") != CONFIRMATION_TIER
                    or metadata.get("claim_boundary") != config["claim_boundary"]
                ):
                    raise ValueError("Prepared confirmation metadata changed the role, tier or claim boundary")
                if data["y"].nunique() < 2:
                    raise ValueError("Qualified classification confirmation requires at least two actual classes")
                metadata.update(used_split_policy="iid", selected_official_split_ids=[])
            if prepared_data is not None:
                if (
                    set(data) != set(prepared_data)
                    or not data["X"].equals(prepared_data["X"])
                    or not data["y"].equals(prepared_data["y"])
                    or fingerprint({key: value for key, value in data.items() if key not in ("X", "y")})
                    != fingerprint({key: value for key, value in prepared_data.items() if key not in ("X", "y")})
                ):
                    raise ResumeMismatchError(
                        f"Cannot resume: current raw payload differs from prepared data for {name}"
                    )
                data = prepared_data
            else:
                content = pickle.dumps(data, protocol=5)
                prepared_sha256 = tabular.digest(content)
            split_rows = (
                execution._confirmation_splits(spec, data, metadata, official, registration, config)
                if registration is not None
                else execution._splits(data, "classification", config, None)
            )
            if {(seed, fold) for seed, fold, _, _ in split_rows} != {
                (seed, fold) for seed in config["seeds"] for fold in range(config["folds"])
            } or len(split_rows) != config["folds"] * len(config["seeds"]):
                raise ValueError("Prepared outer splits differ from the requested folds/seeds")
            dataset_identity = {
                "metadata": metadata,
                "prepared_sha256": prepared_sha256,
                "splits": [
                    {"seed": seed, "fold": fold, "sha256": execution.split_fingerprint(train, test)}
                    for seed, fold, train, test in split_rows
                ],
            }
            if prepared_data is None:
                staging = data_path.with_suffix(".pkl.pending")
                staging.write_bytes(content)
                staging.replace(data_path)
            state.validate_dataset(name, dataset_identity)
            state.save()
            preparation_state = "validated"
        except ResumeMismatchError:
            raise
        except KeyboardInterrupt:
            interrupted, error, status = True, "Experiment interrupted during preparation", "interrupted"
        except Exception as exc:
            if cached is not None:
                raise ResumeMismatchError(f"Cannot resume: prepared data/splits invalid for {name}: {exc}") from exc
            logger.exception("Dataset preparation failed for %s", name)
            error = f"{type(exc).__name__}: {exc}"
            status = "budget_exhausted" if isinstance(exc, TimeoutError) else "dataset_error"
            if status == "dataset_error":
                preparation_state = "failed"
        if error is not None:
            metadata = {**metadata, "error": error, "preparation_status": status}
            split_rows = [(seed, fold, None, None) for seed in config["seeds"] for fold in range(config["folds"])]
        manifest["datasets"].append(
            {**metadata, "name": name, "prepared_sha256": prepared_sha256, "preparation_state": preparation_state}
        )
        for seed, fold, train, test in split_rows:
            split_hash = execution.split_fingerprint(train, test) if error is None else "unavailable"
            manifest["splits"].append({"dataset": name, "seed": seed, "fold": fold, "sha256": split_hash})
            for learner in config["learners"]:
                for kind, methods in job_methods.items():
                    job_id = state.cell_id(name, seed, fold, kind, learner)
                    payload = {
                        **_sampling_identity(config),
                        "dataset": name,
                        "family_id": spec["family_id"],
                        "role": spec["role"],
                        "source": source,
                        "task": "classification",
                        "method": methods[0] if len(methods) == 1 else kind,
                        "job_kind": kind,
                        "physical_job_id": job_id,
                        "seed": seed,
                        "fold": fold,
                        "split_sha256": split_hash,
                        "train_idx": train,
                        "test_idx": test,
                        "expected_cells": len(split_rows),
                        "dataset_metadata": metadata,
                        "data_path": str(data_path),
                        "prepared_sha256": prepared_sha256,
                        "preparation_state": preparation_state,
                        "config": {**config, "learners": [learner], "seed": seed},
                    }
                    cell_hash = fingerprint(
                        {
                            **_sampling_identity(config),
                            "identity": state.identity_hash,
                            "prepared": prepared_sha256,
                            "split": split_hash,
                            "physical_job_id": job_id,
                        }
                    )
                    jobs.append((job_id, cell_hash, payload, error, status))
    # Preserve single-arm progress before either bundle, spreading outer-fold coverage.
    jobs.sort(
        key=lambda job: (
            job[2]["fold"],
            job[2]["seed"],
            job[2]["dataset"],
            job[2]["config"]["learners"],
            list(job_methods).index(job[2]["job_kind"]),
        )
    )

    def persist() -> None:
        state.elapsed_seconds = prior_elapsed + time.perf_counter() - started
        state.save()
        rows = []
        for job_id, _, payload, error, status in jobs:
            if job_id in state.rows:
                rows.extend(state.rows[job_id]["outcomes"])
            else:
                rows.extend(
                    _outcomes(
                        payload,
                        status if error else "prepared" if args.prepare_only else "pending",
                        error or "Not evaluated",
                    )
                )
        write_report(output, manifest, rows, list(state.rows.values()))
        state.elapsed_seconds = prior_elapsed + time.perf_counter() - started
        manifest["active_seconds"] = state.elapsed_seconds
        manifest["budget_overshoot_seconds"] = max(0.0, state.elapsed_seconds - budget)
        state.save()
        atomic_json(output / "manifest.json", manifest)

    source_changed = False
    for job_id, cell_hash, payload, error, status in jobs:
        if args.prepare_only:
            break
        try:
            if tabular.source_fingerprint() != identity["source_sha256"]:
                source_changed = manifest["source_changed"] = True
        except KeyboardInterrupt:
            interrupted = True
        previous_job = state.rows.get(job_id, {})
        unfinished_preparation = (
            previous_job.get("status") in {"budget_exhausted", "interrupted"}
            and previous_job.get("preparation_state") == "unfinished"
            and previous_job.get("attempt", 0) == 0
            and bool(previous_job.get("outcomes"))
            and all(
                row["status"] in {"budget_exhausted", "interrupted"}
                and row.get("preparation_state") == "unfinished"
                and row["split_sha256"] == "unavailable"
                and not any(metric in row for metric in METRICS)
                for row in previous_job["outcomes"]
            )
        )
        if not unfinished_preparation and state.reusable(job_id, cell_hash, retry_failed=args.retry_failed):
            continue
        remaining = deadline - time.perf_counter() - REPORT_RESERVE_SECONDS
        seconds = 0.0
        attempted = False
        if source_changed:
            outcomes = _outcomes(payload, "source_changed", "Source changed during the study; run invalidated")
        elif interrupted:
            outcomes = _outcomes(payload, "interrupted", "Experiment interrupted")
        elif error is not None:
            outcomes = _outcomes(payload, status, error)
        elif remaining <= 0:
            outcomes = _outcomes(payload, "budget_exhausted", "Active-time budget exhausted (including report reserve)")
        else:
            attempted = True
            directory = state.next_attempt(job_id)
            attempt_started = time.perf_counter()
            try:
                if tabular.digest(Path(payload["data_path"]).read_bytes()) != payload["prepared_sha256"]:
                    raise ResumeMismatchError("Prepared dataset changed before physical job dispatch")
                if ledger is None:
                    ledger = ExposureLedger(tabular.ROOT / "benchmarks" / ".cache" / "exposure.sqlite")
                if payload["family_id"] not in admitted_families:
                    ledger.admit(payload["family_id"], state.run_id, payload["role"])
                    admitted_families.add(payload["family_id"])
                if payload["job_kind"].startswith("bundle"):
                    raw = tabular.run_isolated(
                        payload,
                        directory,
                        min(config["case_timeout"], remaining),
                        worker_module="benchmarks.calibration_evaluation",
                    )
                else:
                    raw = tabular.run_isolated(payload, directory, min(config["case_timeout"], remaining))
                outcomes = normalize_outcomes(payload, raw)
            except KeyboardInterrupt:
                interrupted = True
                outcomes = _outcomes(payload, "interrupted", "Experiment interrupted")
            except Exception as exc:
                logger.exception("Physical job %s failed", job_id)
                outcomes = _outcomes(payload, "error", f"{type(exc).__name__}: {exc}")
            seconds = time.perf_counter() - attempt_started
            if tabular.source_fingerprint() != identity["source_sha256"]:
                source_changed = manifest["source_changed"] = True
                outcomes = _outcomes(
                    payload, "source_changed", "Source changed during physical job; scores invalidated", seconds
                )
            for row in outcomes:
                row["log"] = str((directory / "worker.log").relative_to(output))
        previous_seconds = state.rows.get(job_id, {}).get("physical_seconds", 0.0)
        state.elapsed_seconds = prior_elapsed + time.perf_counter() - started
        state.record(
            job_id,
            cell_hash,
            {
                **_sampling_identity(config),
                "status": "success" if all(row["status"] == "success" for row in outcomes) else outcomes[0]["status"],
                "job_kind": payload["job_kind"],
                "physical_job_id": job_id,
                "preparation_state": payload["preparation_state"],
                "physical_seconds": previous_seconds + seconds,
                "last_attempt_seconds": seconds,
                "outcomes": outcomes,
            },
        )
        if attempted:
            try:
                persist()
            except KeyboardInterrupt:
                interrupted = True
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["source_sha256_at_finish"] = tabular.source_fingerprint()
    manifest["source_changed"] = source_changed or manifest["source_sha256_at_finish"] != identity["source_sha256"]
    manifest["interrupted"] = interrupted
    persist()
    return int(
        manifest["source_changed"]
        or interrupted
        or state.elapsed_seconds > budget
        or any(item.get("error") for item in manifest["datasets"])
        or (not args.prepare_only and any(row["status"] != "success" for row in state.rows.values()))
    )


def main(argv: list[str] | None = None) -> int:
    """Run development, explicitly locked qualified confirmation, or a private bundle worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=tabular.ROOT / "benchmarks" / "tabular-results" / "calibration-development"
    )
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--allow-confirmation", action="store_true", default=None)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--learners", nargs="+", choices=LEARNERS)
    parser.add_argument(
        "--ensemble-sampling", choices=tuple(SAMPLING_PROTOCOLS), help="Fixed ensemble design; default: cv3"
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    for name in (
        "folds",
        "max_rows",
        "max_added_features",
        "max_candidates",
        "stability_repeats",
        "threads",
        "memory_limit_mb",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    parser.add_argument("--case-timeout", type=float)
    parser.add_argument("--total-seconds", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.worker:
        if args.worker_output is None:
            parser.error("--worker requires --worker-output")
        payload = pickle.loads(args.worker.read_bytes())
        started = time.perf_counter()
        try:
            rows = run_case(payload, args.worker.parent)
        except Exception as exc:
            logger.exception("Calibration bundle worker failed")
            rows = _outcomes(payload, "error", f"{type(exc).__name__}: {exc}", time.perf_counter() - started)
        atomic_json(args.worker_output, rows)
        return 0
    if args.worker_output is not None:
        parser.error("--worker-output requires --worker")
    try:
        return execute(args)
    except (ValueError, OSError) as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
