"""One source-frozen, development-only raw/FE prediction-blending experiment.

The unchanged OOF default produces four inseparable outcomes from shared full-data
fits and ten fold-local OOF fits: nine original classification families, three
outer folds, seed 42 and LightGBM give 27 jobs, not 108 datasets. Explicit
``--blend-mode fixed_average`` instead produces raw/FE/fixed-0.5 outcomes from two
full-data procedures, without OOF fitting or weight optimization. Its default
seeds 42/137/2029 give 81 jobs and 243 outcomes across the same nine families.
``--datasets`` and nondefault seeds are explicitly labelled smoke replays.
No model, probability array, OOF vector, target row or vocabulary is exported.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarks import calibration_evaluation, execution, tabular
from benchmarks.run_state import (
    ExposureLedger,
    ResumeMismatchError,
    RunState,
    archive_results,
    archive_source,
    atomic_json,
    fingerprint,
)

logger = logging.getLogger(__name__)
PROTOCOL = "classification-oof-blending-development-v1"
MODE = "oof_cv5"
FIXED_MODE = "fixed_average"
FIXED_SEEDS = [42, 137, 2029]
PROTOCOLS = {MODE: PROTOCOL, FIXED_MODE: "classification-fixed-average-development-v1"}
CORE_PROTOCOL = "lightgbm_iid_oof_convex_blend_5fold_v1"
FIXED_CORE_PROTOCOL = "lightgbm_iid_fixed_average_v1"
CRITERION = "float64_mean_log_loss_global_convex_weight"
METHODS = (
    "baseline-robust",
    "featcopilot-frequency-aligned-robust",
    "prediction-average-robust",
    "prediction-oof-blend-robust",
)
ARMS = ("raw", "fe", "average", "blend")
MODE_METHODS = {MODE: METHODS, FIXED_MODE: METHODS[:3]}
MODE_ARMS = {MODE: ARMS, FIXED_MODE: ARMS[:3]}
NATIVE_REFERENCES = ("openfe-inductive-native-robust", "autofeat-native-robust")
CONFIRMATION_FAMILY = "thoracic-surgery"
CONFIRMATION_PROTOCOL = "classification-fixed-average-qualified-confirmation-v1"
CONFIRMATION_TIER = "standard_iid_unverified_entity"
CONFIRMATION_METHODS = (*MODE_METHODS[FIXED_MODE], *NATIVE_REFERENCES)
CONFIRMATION_SPEC_SHA256 = "9bff7a76aa57c3d1cd45d5e909b2214c07311b73d6b2c89adc04051c574cde0b"
CONFIRMATION_SETTINGS: dict[str, Any] = {
    "allow_confirmation": True,
    "datasets": [CONFIRMATION_FAMILY],
    "methods": list(CONFIRMATION_METHODS),
    "learners": ["lightgbm"],
    "blend_mode": FIXED_MODE,
    "seed": 2031,
    "seeds": [2031],
    "folds": 3,
    "max_rows": 0,
    "threads": 8,
    "memory_limit_mb": 4096,
    "case_timeout": 900.0,
    "max_candidates": 1000,
    "max_added_features": 50,
    "stability_repeats": 3,
}
CONFIRMATION_ACCEPTANCE = {
    "positive_vs_raw": True,
    "max_raw_family_relative_harm": 0.05,
    "nonnegative_gain_vs_fe": True,
    "nonnegative_gain_vs_each_native": True,
    "complete_prediction_rows": 15,
    "complete_physical_jobs": 9,
    "zero_reference_loss": "block_relative_claim_report_absolute",
    "automatic_sota": False,
    "automatic_default_promotion": False,
}
CONFIRMATION_CLAIM_NOTE = (
    "Locked single-family standard IID confirmation with unverified patient/entity mapping. Three folds are "
    "not three independent datasets and do not justify an independent-family confidence interval. No clinical, "
    "patient/entity-disjoint, verified-tier pooled, automatic default-promotion or SOTA claim is authorized, "
    "even if every registered criterion passes. Known group/time boundaries must never be discarded."
)
DEFAULT_SUITE = calibration_evaluation.DEFAULT_SUITE
DEFAULT_DATASETS = list(calibration_evaluation.DEFAULT_DATASETS)
CANONICAL_SUITE_SHA256 = "1f7d95ba7ada0bbd188d575756dc7ce6a8da8d55d1fb1963a1bcc222df7d0d84"
CANONICAL_SPECS_SHA256 = "a2e5f573635a9269230d9e65f45df22976daa5aba6b1725091106b6af81030d1"
TOTAL_SECONDS = 6 * 3600.0
REPORT_RESERVE_SECONDS = 10.0
METRICS = ("loss", "accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "brier_sum")
DEFAULTS: dict[str, Any] = {
    "suite": str(DEFAULT_SUITE),
    "datasets": DEFAULT_DATASETS,
    "methods": list(METHODS),
    "learners": ["lightgbm"],
    "seed": 42,
    "folds": 3,
    "max_rows": 5000,
    "max_candidates": 1000,
    "max_added_features": 50,
    "stability_repeats": 3,
    "threads": 8,
    "memory_limit_mb": 4096,
    "case_timeout": 900.0,
    "blend_mode": MODE,
    "oof_folds": 5,
    "oof_seed_offset": 1000,
    "categorical_encoding": "one_hot",
    "numeric_clip_quantiles": [0.0, 1.0],
    "target_transform": "identity",
}
CORE_POLICY = {
    "oof": "independent_complete_raw_and_fe_procedures_per_complement_no_whole_T_parameters",
    "inference": "fresh_complete_full_T_refits_after_weight_freeze_no_oof_children",
    "controls": "exact_same_full_T_objects_and_cached_probabilities_for_all_four_arms",
    "interpretation": "OOF_meta_training_loss_not_independent_validation_no_outer_test_no_harm_guarantee",
    "refit_shift": "standard_refit_full_stacking_parameters_and_feature_plans_may_change",
    "categorical_encoding": "one_hot",
    "numeric_clip_quantiles": [0.0, 1.0],
    "max_candidates": 1000,
    "max_added_features": 50,
    "stability_repeats": 3,
    "importance_resamples": 1,
    "selection_rule": "paired_gain",
    "candidate_set": "extended",
    "include_frequency": True,
}
FIXED_CORE_POLICY = {
    **{key: value for key, value in CORE_POLICY.items() if key != "refit_shift"},
    "oof": "not_performed",
    "inference": "complete_full_T_raw_and_fe_procedures_no_oof",
    "controls": "exact_same_full_T_objects_and_cached_probabilities_for_all_three_arms",
    "interpretation": "replication_of_prespecified_50_50_arm_no_weight_search_no_causal_variance_or_no_harm_guarantee",
}
DIAGNOSTIC_KEYS = {
    "protocol",
    "task",
    "learner",
    "split_kind",
    "seed",
    "oof_seed",
    "sklearn_version",
    "class_counts",
    "n_oof_covered",
    "oof_coverage_histogram",
    "member_count",
    "full_fit_identity_sha256",
    "exact_shared_fits",
    "full_complete_procedure_fit_count",
    "oof_complete_procedure_fit_count",
    "n_input_features",
    "n_output_features",
    "full_output_feature_counts",
    "weight",
    "oof_losses",
    "optimization",
    "criterion",
    "source_policy",
    "members",
    "full_models",
}
FIXED_DIAGNOSTIC_KEYS = (
    DIAGNOSTIC_KEYS
    - {
        "oof_seed",
        "n_oof_covered",
        "oof_coverage_histogram",
        "member_count",
        "members",
        "oof_complete_procedure_fit_count",
        "oof_losses",
        "optimization",
    }
) | {"blend_mode", "weight_optimized"}
TIMING_KEYS = {
    "split_seconds",
    "oof_fit_seconds",
    "oof_predict_seconds",
    "oof_seconds",
    "oof_orchestration_seconds",
    "weight_optimization_seconds",
    "full_raw_fit_seconds",
    "full_fe_fit_seconds",
    "fit_seconds",
    "test_score_seconds",
    "physical_bundle_wall_seconds",
    *(
        f"{arm}_{stage}_seconds"
        for arm in ARMS
        for stage in ("predict", "cached_score", "training", "score", "standalone")
    ),
}
FIXED_TIMING_KEYS = {
    "full_raw_fit_seconds",
    "full_fe_fit_seconds",
    "fit_seconds",
    "test_score_seconds",
    "physical_bundle_wall_seconds",
    *(
        f"{arm}_{stage}_seconds"
        for arm in MODE_ARMS[FIXED_MODE]
        for stage in ("predict", "cached_score", "training", "score", "standalone")
    ),
}
COST_NOTE = (
    "Raw and FE standalone costs exclude OOF fits. All four outcomes reuse the same full raw/FE fitted objects; "
    "their attributed pipeline times are shared, not additive physical cost. Each physical job includes all "
    "twelve fits once; physical_seconds includes every attempt. Cumulative active_seconds in manifest.json "
    "includes preparation, workers, reports and final archives. The ten-second cleanup reserve is not a "
    "perfect hard bound; actual overshoot is reported."
)
CLAIM_NOTE = (
    "Exposed IID development evidence only, not fresh confirmation or a formal no-harm guarantee. "
    "Optimized OOF loss is meta-training performance, not independent validation. Full-data refitting changes "
    "training distributions and feature plans. Outer-test labels are used only for scoring, never fitting "
    "or selecting weights. No adaptive weight/configuration changes after outer results."
)
FIXED_COST_NOTE = (
    "All three outcomes reuse the same full raw/FE fitted objects. Two complete procedures perform eight "
    "pipeline fits total (three inner candidates and one full refit each); there are no OOF fits or optimizer. "
    "Standalone pipeline times overlap and are not additive physical cost. physical_seconds counts each "
    "bundle once, including retries. Cumulative active_seconds in manifest.json includes preparation, workers, "
    "reports and archives. The ten-second cleanup reserve is not a hard bound; overshoot is reported."
)
FIXED_CLAIM_NOTE = (
    "Exposed IID development evidence only, not fresh confirmation, a formal no-harm guarantee, or default "
    "promotion. One globally fixed 0.5 raw/FE probability average is predeclared for every dataset, seed and "
    "fold. There is no OOF learning, learned-weight arm or optimizer. Outer-test labels are used only for "
    "scoring. Modes/runs must not be pooled or selected post hoc."
)


def _protocol(config: dict) -> str:
    return CONFIRMATION_PROTOCOL if config.get("allow_confirmation") else PROTOCOLS[config["blend_mode"]]


def _all_methods(config: dict) -> tuple[str, ...]:
    return CONFIRMATION_METHODS if config.get("allow_confirmation") else MODE_METHODS[config["blend_mode"]]


def _claim_fields(config: dict) -> dict:
    if not config.get("allow_confirmation"):
        return {}
    return {"evaluation_tier": config["evaluation_tier"], "claim_boundary": config["claim_boundary"]}


def _outcome_methods(payload: dict) -> tuple[str, ...]:
    if payload["config"].get("allow_confirmation") and payload["method"] in NATIVE_REFERENCES:
        return (payload["method"],)
    return MODE_METHODS[payload["config"]["blend_mode"]]


def validate_config(config: dict[str, Any], budget: float) -> None:
    """Reject unbounded controls or changes to the single frozen study."""
    mode = config.get("blend_mode")
    if not isinstance(mode, str) or mode not in MODE_METHODS:
        raise ValueError(f"blend_mode must be one of {tuple(MODE_METHODS)}")
    confirmation = config.get("allow_confirmation", False)
    if type(confirmation) is not bool:
        raise ValueError("allow_confirmation must be boolean")
    if config.get("methods") != list(_all_methods(config)):
        raise ValueError("Methods must exactly match the explicitly selected blending mode")
    for key in (
        "learners",
        "folds",
        "max_rows",
        "max_candidates",
        "max_added_features",
        "stability_repeats",
        "memory_limit_mb",
        "categorical_encoding",
        "numeric_clip_quantiles",
        "target_transform",
    ):
        expected = CONFIRMATION_SETTINGS.get(key, DEFAULTS[key]) if confirmation else DEFAULTS[key]
        if fingerprint(config.get(key)) != fingerprint(expected):
            raise ValueError(f"The fixed experiment requires {key}={expected!r}")
    for key in ("oof_folds", "oof_seed_offset"):
        if mode == MODE and fingerprint(config.get(key)) != fingerprint(DEFAULTS[key]):
            raise ValueError(f"The OOF experiment requires {key}={DEFAULTS[key]!r}")
        if mode == FIXED_MODE and key in config:
            raise ValueError("fixed_average has no OOF configuration")
    if type(config["threads"]) is not int or not 1 <= config["threads"] <= 8:
        raise ValueError("threads must be an integer in [1, 8]")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32 - 1000:
        raise ValueError("seed must be an integer in [0, 2**32 - 1000)")
    seeds = config.get("seeds", [config["seed"]])
    if (
        not isinstance(seeds, list)
        or not 1 <= len(seeds) <= (1 if mode == MODE else 3)
        or any(type(seed) is not int or not 0 <= seed < 2**32 - 1000 for seed in seeds)
        or len(set(seeds)) != len(seeds)
        or config["seed"] not in seeds
    ):
        raise ValueError("OOF requires one seed; fixed_average permits one to three unique bounded integer seeds")
    for name, value, ceiling in (
        ("case_timeout", config["case_timeout"], 900),
        ("total_seconds", budget, 14400.0 if confirmation else TOTAL_SECONDS),
    ):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= ceiling:
            raise ValueError(f"{name} must be finite and in (0, {ceiling}]")
    names = config["datasets"]
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) for name in names)
        or len(names) != len(set(names))
        or len(names) > 9
    ):
        raise ValueError("datasets must be one to nine unique canonical dataset names")
    if confirmation:
        for key, value in CONFIRMATION_SETTINGS.items():
            if fingerprint(config.get(key)) != fingerprint(value):
                raise ValueError(f"Qualified confirmation fixes {key}={value!r}")


def _confirmation_rules(registration: dict | None = None) -> dict:
    expected = {
        "evaluation_tier": CONFIRMATION_TIER,
        "frozen_primary_method_by_learner": {"lightgbm": METHODS[2]},
        "primary_loss": "log_loss",
        "primary_raw_comparators": [METHODS[0]],
        "paired_fe_comparators": [METHODS[1]],
        "native_fe_comparators": list(NATIVE_REFERENCES),
        "acceptance": dict(CONFIRMATION_ACCEPTANCE),
    }
    if registration is None:
        return expected
    rules = registration.get("reporting_and_acceptance")
    if not isinstance(rules, dict) or any(
        fingerprint(rules.get(key)) != fingerprint(value) for key, value in expected.items()
    ):
        raise ValueError("Qualified confirmation reporting/acceptance differs from the locked decision rules")
    if any(rules.get(key, False) is not False for key in ("automatic_sota", "automatic_default_promotion")):
        raise ValueError("Confirmation must never authorize automatic SOTA or default promotion")
    return rules


def _bind_confirmation(suite: dict, legacy: bool, config: dict, budget: float) -> tuple[list[dict], dict]:
    registration = suite.get("preregistration")
    specs = suite["datasets"]
    if (
        legacy
        or not isinstance(registration, dict)
        or len(specs) != 1
        or specs[0].get("name") != CONFIRMATION_FAMILY
        or specs[0].get("family_id") != CONFIRMATION_FAMILY
    ):
        raise ValueError("Qualified confirmation requires only the released Thoracic Surgery family")
    boundary = registration.get("claim_boundary")
    spec = specs[0]
    if (
        not isinstance(boundary, dict)
        or boundary.get("evaluation_tier") != CONFIRMATION_TIER
        or any(boundary.get(key) is not value for key, value in calibration_evaluation.CLAIM_FLAGS.items())
        or any(
            item.get("claim_boundary") != boundary or item.get("evaluation_tier") != CONFIRMATION_TIER
            for item in (suite, spec)
        )
    ):
        raise ValueError("The exact unverified-entity IID claim boundary must bind root, source and registration")
    if (
        spec.get("role") != "confirmation"
        or spec.get("task") != "classification"
        or spec.get("provider") != "openml_task"
        or execution.source_kind(spec) != "real_world"
        or spec.get("split_policy") != "official"
        or any(spec.get(key) is not None for key in ("group_column", "time_column", "label_end_column"))
        or spec.get("rights", {}).get("status") != "approved"
        or spec.get("_family", {}).get("reservation", {}).get("status") == "active"
    ):
        raise ValueError("Confirmation cannot bypass source identity, rights, reservations or group/time boundaries")
    public_hash = fingerprint({key: value for key, value in spec.items() if not key.startswith("_")})
    if public_hash != CONFIRMATION_SPEC_SHA256 or registration.get("dataset_spec_sha256") != {
        CONFIRMATION_FAMILY: CONFIRMATION_SPEC_SHA256
    }:
        raise ValueError("Released Thoracic source/schema/target-exclusion definition changed")
    expected = {
        "status": "locked_before_confirmation_model_execution",
        "controller_protocol": CONFIRMATION_PROTOCOL,
        "evaluation_tier": CONFIRMATION_TIER,
        "dataset_names": [CONFIRMATION_FAMILY],
        "family_ids": [CONFIRMATION_FAMILY],
        "methods": list(CONFIRMATION_METHODS),
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
        "split_plan": {CONFIRMATION_FAMILY: {"policy": "iid", "folds": 3}},
        "classification_admission_margin": {"min_total": 30, "min_outer_train": 20, "min_outer_test": 10},
        "blend_mode": FIXED_MODE,
        "primary_method": METHODS[2],
        "fixed_weight": 0.5,
        "selection_parameters": {
            "max_candidates": 1000,
            "max_added_features": 50,
            "stability_repeats": 3,
            "screening": "auto",
            "blend_mode": FIXED_MODE,
            "categorical_encoding": "one_hot",
            "numeric_clip_quantiles": [0.0, 1.0],
        },
        "learner_search": {"lightgbm": {"num_leaves": [7, 15, 31], "selection_estimator_num_leaves": 15}},
    }
    for key, value in expected.items():
        if fingerprint(registration.get(key)) != fingerprint(value):
            raise ValueError(f"Qualified confirmation registration differs: {key}")
    _confirmation_rules(registration)
    if "primary_method_by_learner" in registration and registration["primary_method_by_learner"] != {
        "lightgbm": METHODS[2]
    }:
        raise ValueError("Conflicting primary-method declarations")
    if "prediction_rule" in registration and (
        not isinstance(registration["prediction_rule"], dict)
        or any(
            registration["prediction_rule"].get(key) is not False for key in ("weight_learning", "oof", "calibration")
        )
    ):
        raise ValueError("The registered fixed rule forbids weight learning, OOF and calibration")
    config.update(evaluation_tier=CONFIRMATION_TIER, claim_boundary=boundary, synthetic=False, screening="auto")
    validate_config(config, budget)
    bound = execution.bind_preregistration({"manifests": [suite]}, specs, config, budget)
    if bound is None:
        raise ValueError("Qualified confirmation registration did not bind")
    return specs, bound


def _canonical_specs() -> dict[str, dict]:
    if tabular.digest(DEFAULT_SUITE.read_bytes()) != CANONICAL_SUITE_SHA256:
        raise ValueError("The original canonical replay source manifest was altered")
    specs = calibration_evaluation._canonical_specs()
    if fingerprint(specs) != CANONICAL_SPECS_SHA256:
        raise ValueError("The original canonical normalized source/family definitions were altered")
    return specs


def _selected_specs(suite: dict, legacy: bool, names: list[str]) -> list[dict]:
    canonical = _canonical_specs()
    by_name = {spec["name"]: spec for spec in suite["datasets"]}
    if legacy or len(by_name) != len(suite["datasets"]) or set(names) - (set(by_name) & set(canonical)):
        raise ValueError("Only the nine original canonical classification families or subset smoke are allowed")
    specs = [by_name[name] for name in names]
    for spec in specs:
        if (
            spec.get("role") != "development"
            or spec.get("task") != "classification"
            or spec.get("split_policy", "iid") != "iid"
            or any(spec.get(key) is not None for key in ("group_column", "time_column", "label_end_column"))
            or fingerprint(spec) != fingerprint(canonical[spec["name"]])
        ):
            raise ValueError(f"{spec['name']}: requires the unchanged canonical development IID source definition")
    return specs


def _outcomes(payload: dict, status: str, error: str, seconds: float = 0.0) -> list[dict]:
    return [
        {
            **{
                key: payload[key]
                for key in (
                    "dataset",
                    "family_id",
                    "role",
                    "source",
                    "task",
                    "seed",
                    "fold",
                    "split_sha256",
                    "physical_job_id",
                    "expected_cells",
                    "prepared_sha256",
                    "dataset_metadata",
                    "preparation_state",
                )
            },
            "method": method,
            "learner": "lightgbm",
            "protocol": _protocol(payload["config"]),
            "blend_mode": payload["config"]["blend_mode"],
            **_claim_fields(payload["config"]),
            "status": status,
            "error": error,
            "case_seconds": seconds,
        }
        for method in _outcome_methods(payload)
    ]


def _finite(value: Any, name: str, *, upper: float | None = None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (upper is not None and value > upper)
    ):
        raise ValueError(f"{name} must be finite and nonnegative" + (f", at most {upper}" if upper else ""))
    return float(value)


def _sha256(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("Provenance identities must be SHA256 strings")
    return value


def _positions(report: dict) -> dict:
    size, counts = report["size"], report["class_counts"]
    if (
        type(size) is not int
        or size < 1
        or not isinstance(counts, list)
        or len(counts) < 2
        or any(type(count) is not int or count < 1 for count in counts)
        or sum(counts) != size
    ):
        raise ValueError("Partition provenance must contain positive class counts, not rows")
    return {"size": size, "class_counts": counts, "indices_sha256": _sha256(report["indices_sha256"])}


def _base_provenance(report: dict, *, names: bool) -> dict:
    width = report["n_output_features"]
    if type(width) is not int or width < 1:
        raise ValueError("Base plan width must be a positive integer")
    parameters = tabular.candidate_parameters("lightgbm")
    selected = report["selected_parameters"]
    if selected not in parameters or len(report["trials"]) != len(parameters):
        raise ValueError("Base provenance must describe the frozen three-setting LightGBM search")
    trials = []
    for trial, expected in zip(report["trials"], parameters, strict=True):
        if (
            trial["parameters"] != expected
            or type(trial["n_output_features"]) is not int
            or trial["n_output_features"] < 1
        ):
            raise ValueError("Invalid base trial parameters or feature-plan width")
        trials.append(
            {
                "parameters": expected,
                "loss": _finite(trial["loss"], "trial_loss"),
                "n_output_features": trial["n_output_features"],
            }
        )
    result = {
        **{
            key: _sha256(report[key])
            for key in (
                "fit_identity_sha256",
                "feature_names_sha256",
                "schema_sha256",
                "formula_sha256",
                "fitted_state_sha256",
            )
        },
        "selected_parameters": selected,
        "trials": trials,
        "n_output_features": width,
        "inner_split": {key: _positions(report["inner_split"][key]) for key in ("train", "validation")},
    }
    if names:
        feature_names = report["feature_names"]
        if (
            not isinstance(feature_names, list)
            or len(feature_names) != width
            or any(not isinstance(name, str) for name in feature_names)
            or len(set(feature_names)) != width
        ):
            raise ValueError("Full feature-plan names must match the reported width")
        result["feature_names"] = feature_names
    return result


def _provenance(diagnostics: dict, *, include_members: bool = True) -> dict:
    models = {arm: _base_provenance(diagnostics["full_models"][arm], names=True) for arm in ("raw", "fe")}
    if not include_members:
        return {"full_models": models}
    members = []
    if not isinstance(diagnostics["members"], list) or len(diagnostics["members"]) != 5:
        raise ValueError("Exactly five compact OOF member reports are required")
    for index, member in enumerate(diagnostics["members"]):
        if type(member["member"]) is not int or member["member"] != index:
            raise ValueError("OOF member provenance must follow the frozen five-fold order")
        members.append(
            {
                "member": index,
                "train": _positions(member["train"]),
                "holdout": _positions(member["holdout"]),
                **{arm: _base_provenance(member[arm], names=False) for arm in ("raw", "fe")},
            }
        )
    return {"full_models": models, "members": members}


def _worker_data(payload: dict) -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
    confirmation = payload["config"].get("allow_confirmation", False)
    validate_config(payload["config"], 14400.0 if confirmation else TOTAL_SECONDS)
    if (
        payload.get("task") != "classification"
        or payload.get("role") != ("confirmation" if confirmation else "development")
        or payload.get("protocol") != _protocol(payload["config"])
        or payload.get("blend_mode") != payload["config"]["blend_mode"]
        or payload["seed"] != payload["config"]["seed"]
        or payload.get("method") != payload["config"]["blend_mode"]
        or any(payload.get(key) != value for key, value in _claim_fields(payload["config"]).items())
    ):
        raise ValueError("Worker supports only the frozen development classification blending mode")
    content = Path(payload["data_path"]).read_bytes()
    if tabular.digest(content) != payload["prepared_sha256"]:
        raise ValueError("Prepared dataset changed before worker execution")
    data = pickle.loads(content)
    if any(data.get(key) is not None for key in ("groups", "timestamps", "label_end_times")):
        raise ValueError("Worker requires IID data without group/time metadata")
    X, y = data["X"], data["y"]
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series) or not X.index.equals(y.index) or y.isna().any():
        raise ValueError("Worker X/y must be index-aligned pandas data with nonmissing targets")
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
            raise ValueError("Outer split indices must be unique, in-range integer positions")
    if np.intersect1d(train, test).size or execution.split_fingerprint(train, test) != payload["split_sha256"]:
        raise ValueError("Outer split boundary or fingerprint changed")
    if not np.array_equal(np.unique(y.iloc[train]), np.unique(y.iloc[test])):
        raise ValueError("Every outer split must contain the full class vocabulary")
    return X, y, train, test


def run_case(payload: dict, workspace: Path) -> list[dict]:
    """Fit one shared bundle and expose only the selected mode's scoring views.

    Parameters
    ----------
    payload : dict
        Hash-bound prepared dataset, outer indices and the frozen configuration.
    workspace : pathlib.Path
        Supervisor-owned attempt directory.

    Returns
    -------
    list[dict]
        Three or four JSON-only outcomes, all unscored if any arm is invalid.
    """
    X, y, train, test = _worker_data(payload)
    fixed = payload["config"]["blend_mode"] == FIXED_MODE
    if fixed:
        from benchmarks.oof_blending import fit_fixed_average_bundle as fit_bundle
    else:
        from benchmarks.oof_blending import fit_oof_blend_bundle as fit_bundle
    started = time.perf_counter()
    result: dict[str, Any] = dict(
        fit_bundle(
            X.iloc[train],
            y.iloc[train],
            X.iloc[test],
            y.iloc[test],
            learner="lightgbm",
            config=payload["config"],
        )
    )
    if result["shared_cost_not_additive"] is not True or result["n_input_features"] != X.shape[1]:
        raise ValueError("Core result changed the shared-cost or input-width contract")
    if fixed and (
        result["diagnostics"].get("member_count") != 0
        or result["diagnostics"].get("members") != []
        or result["diagnostics"].get("oof_complete_procedure_fit_count") != 0
        or any(key in result["diagnostics"] for key in ("oof_losses", "oof_seed", "optimization"))
        or "blend_metrics" in result
    ):
        raise ValueError("fixed_average must not perform OOF fitting or emit learned-arm diagnostics")
    diagnostic_keys = FIXED_DIAGNOSTIC_KEYS if fixed else DIAGNOSTIC_KEYS
    diagnostics = {key: result["diagnostics"][key] for key in sorted(diagnostic_keys)}
    diagnostics.update(_provenance(diagnostics, include_members=not fixed))
    if result["n_output_features"] != diagnostics["n_output_features"]:
        raise ValueError("Core result changed the full FE feature-plan width")
    rows = _outcomes(payload, "success", "")
    for arm, row in zip(MODE_ARMS[payload["config"]["blend_mode"]], rows, strict=True):
        row.pop("error")
        row.update(
            result[f"{arm}_metrics"],
            n_train=len(train),
            n_test=len(test),
            n_input_features=X.shape[1],
            n_output_features=result["n_input_features"] if arm == "raw" else result["n_output_features"],
            shared_bundle_diagnostics=diagnostics,
            shared_bundle_timing=result["timing"],
            pipeline_seconds=result["timing"][f"{arm}_standalone_seconds"],
            case_seconds=time.perf_counter() - started,
        )
    return normalize_outcomes(payload, rows)


def _native_details(payload: dict, row: dict) -> dict:
    if row.get("numeric_clip_quantiles") != [0.0, 1.0] or row.get("matched_baseline_method") != METHODS[0]:
        raise ValueError("Native references require the existing matched final-learner q0/q1 caps")
    if row.get("parameters") not in tabular.candidate_parameters("lightgbm"):
        raise ValueError("Native reference selected parameters differ from the frozen LightGBM search")
    losses = row["inner_losses"]
    if not isinstance(losses, list) or len(losses) != 3:
        raise ValueError("Native references must retain all three inner-trial losses")
    result = {
        "parameters": row["parameters"],
        "inner_losses": [_finite(value, "inner_loss") for value in losses],
        "numeric_clip_quantiles": [0.0, 1.0],
        "matched_baseline_method": METHODS[0],
    }
    for stage in ("inner", "outer"):
        reference = row[f"{stage}_reference_diagnostics"]
        required = {
            "status": "success",
            "method": "openfe-inductive" if payload["method"] == NATIVE_REFERENCES[0] else "autofeat-native",
            "task": "classification",
            "seed": payload["seed"],
            "threads": payload["config"]["threads"],
            "width_policy": "native_output",
            "max_added_features": None,
            "numeric_clip_quantiles": [0.0, 1.0],
            "clipping_scope": "final_matched_learner_only",
            "generation_preprocessing": "existing_training_median_fill",
            "original_numeric_missingness_restored": True,
        }
        if not isinstance(reference, dict) or any(reference.get(key) != value for key, value in required.items()):
            raise ValueError("Native generation/final-clipping policy differs from the registered reference")
        for key in ("input_features", "selected_count", "added_features"):
            if type(reference.get(key)) is not int or reference[key] < (1 if key == "input_features" else 0):
                raise ValueError("Native reference counts must be nonnegative integers")
        if reference["input_features"] != row["n_input_features"] or (
            stage == "outer" and row["n_output_features"] != reference["input_features"] + reference["added_features"]
        ):
            raise ValueError("Native reference output width differs from its generation provenance")
        result[f"{stage}_reference_diagnostics"] = {
            **required,
            **{key: reference[key] for key in ("input_features", "selected_count", "added_features")},
            "reference_sha256": (
                _sha256(reference["reference_sha256"]) if "reference_sha256" in reference else fingerprint(reference)
            ),
            "feature_definitions_sha256": (
                _sha256(reference["feature_definitions_sha256"])
                if "feature_definitions_sha256" in reference
                else fingerprint(reference.get("feature_definitions"))
            ),
        }
    return result


def normalize_outcomes(payload: dict, rows: Any) -> list[dict]:
    """Validate and checkpoint all mode-specific outcomes together, never a fallback."""
    try:
        methods = _outcome_methods(payload)
        native = payload["config"].get("allow_confirmation") and payload["method"] in NATIVE_REFERENCES
        if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Worker must return a list of outcome objects")
        for row in rows:
            if "worker_python_hash_seed" in row and (
                type(row["worker_python_hash_seed"]) is not int or row["worker_python_hash_seed"] != payload["seed"]
            ):
                raise ValueError("Worker Python hash seed differs from the requested seed")
        failures = [row for row in rows if row.get("status") != "success"]
        if failures:
            if len(failures) != len(rows):
                raise ValueError("Partial shared-bundle success is forbidden")
            result = _outcomes(
                payload,
                str(failures[0].get("status", "error")),
                "; ".join(str(row.get("error", "Unscored worker outcome")) for row in failures),
                _finite(failures[0].get("case_seconds", 0.0), "case_seconds"),
            )
            for row in result:
                if "worker_python_hash_seed" in failures[0]:
                    row["worker_python_hash_seed"] = failures[0]["worker_python_hash_seed"]
                for key in ("worker_wall_seconds", "peak_tree_rss_mb"):
                    if key in failures[0]:
                        row[key] = _finite(failures[0][key], key)
            return result
        if len(rows) != len(methods) or {row.get("method") for row in rows} != set(methods):
            raise ValueError("Worker must return exactly the requested mode-specific arms")
        normalized = []
        shared = None
        for expected in _outcomes(payload, "success", ""):
            row = next(row for row in rows if row["method"] == expected["method"])
            for key in (
                "dataset",
                "family_id",
                "role",
                "source",
                "task",
                "seed",
                "fold",
                "split_sha256",
                "physical_job_id",
                "learner",
                "protocol",
                "blend_mode",
                *_claim_fields(payload["config"]),
            ):
                if (
                    native
                    and key in {"physical_job_id", "protocol", "blend_mode", "evaluation_tier", "claim_boundary"}
                    and key not in row
                ):
                    continue
                if row.get(key) != expected[key]:
                    raise ValueError(f"Worker identity changed: {key}")
            expected.pop("error")
            for key in (*METRICS, "pipeline_seconds", "case_seconds"):
                expected[key] = _finite(row.get(key), key)
            for key in ("n_train", "n_test", "n_input_features", "n_output_features"):
                if type(row.get(key)) is not int or row[key] < 1:
                    raise ValueError(f"{key} must be a positive integer")
                expected[key] = row[key]
            if row["n_train"] != len(payload["train_idx"]) or row["n_test"] != len(payload["test_idx"]):
                raise ValueError("Worker row counts differ from the frozen outer split")
            if "dataset_metadata" in row and fingerprint(row["dataset_metadata"]) != fingerprint(
                payload["dataset_metadata"]
            ):
                raise ValueError("Worker dataset metadata differs from its prepared source identity")
            if native:
                expected.update(_native_details(payload, row))
            else:
                diagnostics, timing = row["shared_bundle_diagnostics"], row["shared_bundle_timing"]
                _validate_diagnostics(diagnostics, timing, payload)
                signature = fingerprint([diagnostics, timing])
                if shared is not None and signature != shared:
                    raise ValueError("All outcome views must reference the same fitted bundle")
                shared = signature
                arm = ARMS[METHODS.index(row["method"])]
                width = diagnostics["n_input_features"] if arm == "raw" else diagnostics["n_output_features"]
                if row["n_input_features"] != diagnostics["n_input_features"] or row["n_output_features"] != width:
                    raise ValueError("Outcome width differs from the shared bundle")
                if row["pipeline_seconds"] != timing[f"{arm}_standalone_seconds"]:
                    raise ValueError("Outcome cost differs from its shared standalone attribution")
                expected.update(shared_bundle_diagnostics=diagnostics, shared_bundle_timing=timing)
            if "worker_python_hash_seed" in row:
                expected["worker_python_hash_seed"] = row["worker_python_hash_seed"]
            for key in ("worker_wall_seconds", "peak_tree_rss_mb"):
                if key in row:
                    expected[key] = _finite(row[key], key)
            normalized.append(expected)
        fingerprint(normalized)
        return normalized
    except (ValueError, TypeError, KeyError, StopIteration) as exc:
        logger.error("Invalid physical bundle %s: %s", payload["physical_job_id"], exc)
        return _outcomes(payload, "error", f"Invalid worker output: {exc}")


def _validate_diagnostics(diagnostics: dict, timing: dict, payload: dict) -> None:
    """Validate the core's aggregate-only public contract before persistence."""
    fixed = payload["config"]["blend_mode"] == FIXED_MODE
    arms = MODE_ARMS[payload["config"]["blend_mode"]]
    if not isinstance(diagnostics, dict) or not isinstance(timing, dict):
        raise ValueError("Diagnostics and timing must be public aggregate objects")
    if set(diagnostics) != (FIXED_DIAGNOSTIC_KEYS if fixed else DIAGNOSTIC_KEYS) or set(timing) != (
        FIXED_TIMING_KEYS if fixed else TIMING_KEYS
    ):
        raise ValueError("Only the allowlisted aggregate diagnostic/timing fields may be exported")
    expected_fields = {
        "protocol": FIXED_CORE_PROTOCOL if fixed else CORE_PROTOCOL,
        "task": "classification",
        "learner": "lightgbm",
        "split_kind": "iid",
        "seed": payload["seed"],
        "full_complete_procedure_fit_count": 2,
        "exact_shared_fits": True,
        "source_policy": FIXED_CORE_POLICY if fixed else CORE_POLICY,
    }
    expected_fields.update(
        {
            "blend_mode": FIXED_MODE,
            "weight": 0.5,
            "weight_optimized": False,
            "criterion": "prespecified_probability_average_weight_0.5",
        }
        if fixed
        else {
            "oof_seed": payload["seed"] + 1000,
            "member_count": 5,
            "oof_complete_procedure_fit_count": 10,
            "criterion": CRITERION,
            "n_oof_covered": len(payload["train_idx"]),
            "oof_coverage_histogram": [0, len(payload["train_idx"])],
        }
    )
    for key, expected in expected_fields.items():
        if fingerprint(diagnostics[key]) != fingerprint(expected):
            raise ValueError(f"Frozen OOF bundle diagnostic changed: {key}")
    widths = diagnostics["full_output_feature_counts"]
    if (
        not isinstance(widths, list)
        or len(widths) != 2
        or any(type(value) is not int or value <= 0 for value in widths)
        or widths != [diagnostics["n_input_features"], diagnostics["n_output_features"]]
    ):
        raise ValueError("The full raw and FE procedure widths must match the shared bundle")
    if not 0 <= widths[1] - widths[0] <= 50:
        raise ValueError("FE output exceeds the frozen fifty-feature budget")
    counts = diagnostics["class_counts"]
    if (
        not isinstance(counts, list)
        or len(counts) < 2
        or any(type(value) is not int or value < (2 if fixed else 5) for value in counts)
        or sum(counts) != len(payload["train_idx"])
        or not isinstance(diagnostics["sklearn_version"], str)
    ):
        raise ValueError("OOF class support does not match outer-training row counts")
    identities = diagnostics["full_fit_identity_sha256"]
    if not isinstance(identities, dict) or set(identities) != set(arms):
        raise ValueError("Every arm requires its shared full-fit identities")
    for arm, count in zip(arms, (1, 1, 2) if fixed else (1, 1, 2, 2), strict=True):
        values = identities[arm]
        if (
            not isinstance(values, list)
            or len(values) != count
            or any(
                not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
                for value in values
            )
        ):
            raise ValueError("Full-fit identities must be SHA256 strings")
    if (
        identities["raw"] == identities["fe"]
        or identities["average"] != identities["raw"] + identities["fe"]
        or (not fixed and identities["blend"] != identities["average"])
    ):
        raise ValueError("All four arms must use exactly the same full raw/FE fitted objects")
    provenance = _provenance(diagnostics, include_members=not fixed)
    if fingerprint(provenance) != fingerprint({key: diagnostics[key] for key in provenance}):
        raise ValueError("Only compact allowlisted member/full-plan provenance may be persisted")
    for arm, width in zip(("raw", "fe"), widths, strict=True):
        base = provenance["full_models"][arm]
        if base["fit_identity_sha256"] != identities[arm][0] or base["n_output_features"] != width:
            raise ValueError("Full-model provenance differs from the shared fitted identities or plan widths")
    for member in provenance.get("members", []):
        if member["train"]["size"] + member["holdout"]["size"] != len(payload["train_idx"]):
            raise ValueError("OOF provenance does not account for every outer-training row")
    for bases, rows in [
        (provenance["full_models"], len(payload["train_idx"])),
        *((member, member["train"]["size"]) for member in provenance.get("members", [])),
    ]:
        for arm in ("raw", "fe"):
            base = bases[arm]
            if sum(part["size"] for part in base["inner_split"].values()) != rows:
                raise ValueError("Base inner partitions differ from the fitted TRAIN row count")
            if not widths[0] <= base["n_output_features"] <= widths[0] + (50 if arm == "fe" else 0):
                raise ValueError("Member provenance violates the raw/FE feature budget")
    _finite(diagnostics["weight"], "weight", upper=1.0)
    if fixed:
        for key, value in timing.items():
            _finite(value, key)
        both_fits = timing["full_raw_fit_seconds"] + timing["full_fe_fit_seconds"]
        for arm in arms:
            training = timing[f"full_{arm}_fit_seconds"] if arm != "average" else both_fits
            scoring = timing[f"{arm}_cached_score_seconds"]
            if arm == "average":
                scoring += timing["raw_predict_seconds"] + timing["fe_predict_seconds"]
            if (
                not math.isclose(timing[f"{arm}_training_seconds"], training)
                or not math.isclose(timing[f"{arm}_score_seconds"], scoring)
                or not math.isclose(timing[f"{arm}_standalone_seconds"], training + scoring)
            ):
                raise ValueError("Fixed-average standalone costs must reuse only the two full fits")
        fingerprint([diagnostics, timing])
        return
    losses = diagnostics["oof_losses"]
    if not isinstance(losses, dict) or set(losses) != set(ARMS):
        raise ValueError("Exactly four OOF meta-training losses are required")
    for key, value in losses.items():
        _finite(value, key)
    optimization = diagnostics["optimization"]
    optimizer_keys = {
        "method",
        "bounds",
        "xatol",
        "maxiter",
        "tie_break",
        "endpoint_candidates",
        "epsilon",
        "success",
        "status",
        "nfev",
        "interior_candidate",
    }
    if not isinstance(optimization, dict) or set(optimization) != optimizer_keys:
        raise ValueError("Only aggregate optimization diagnostics are allowed")
    for key, value in {
        "method": "bounded",
        "bounds": [0.0, 1.0],
        "xatol": 1e-5,
        "maxiter": 500,
        "tie_break": "smaller_weight_on_exact_equal_loss",
        "success": True,
    }.items():
        if fingerprint(optimization[key]) != fingerprint(value):
            raise ValueError("The fixed endpoint/bounded-scalar criterion changed")
    endpoints = optimization["endpoint_candidates"]
    if not isinstance(endpoints, list) or len(endpoints) != 2:
        raise ValueError("Both endpoint candidates must be retained")
    for candidate, weight in zip(endpoints, (0.0, 1.0), strict=True):
        if not isinstance(candidate, dict) or set(candidate) != {"weight", "loss"} or candidate["weight"] != weight:
            raise ValueError("Invalid optimization endpoint")
        _finite(candidate["loss"], "endpoint_loss")
    if endpoints[0]["loss"] != losses["raw"] or endpoints[1]["loss"] != losses["fe"]:
        raise ValueError("Endpoint losses differ from raw/FE OOF diagnostics")
    interior = optimization["interior_candidate"]
    if interior is not None:
        if not isinstance(interior, dict) or set(interior) != {"weight", "loss"}:
            raise ValueError("Invalid scalar optimization candidate")
        _finite(interior["weight"], "interior_weight", upper=1.0)
        _finite(interior["loss"], "interior_loss")
    if (
        optimization["status"] not in ("converged", "identical_probabilities")
        or type(optimization["nfev"]) is not int
        or not 0 <= optimization["nfev"] <= 500
    ):
        raise ValueError("Invalid optimization completion diagnostics")
    _finite(optimization["epsilon"], "epsilon", upper=1.0)
    candidates = [*endpoints, *([] if interior is None else [interior])]
    winner = min(candidates, key=lambda candidate: (candidate["loss"], candidate["weight"]))
    if diagnostics["weight"] != winner["weight"] or losses["blend"] != winner["loss"]:
        raise ValueError("Reported weight/loss differs from the fixed OOF-only selection criterion")
    for key, value in timing.items():
        _finite(value, key)
    both_fits = timing["full_raw_fit_seconds"] + timing["full_fe_fit_seconds"]
    oof = sum(timing[key] for key in ("split_seconds", "oof_seconds", "weight_optimization_seconds"))
    if not math.isclose(
        timing["oof_seconds"],
        timing["oof_fit_seconds"] + timing["oof_predict_seconds"] + timing["oof_orchestration_seconds"],
    ):
        raise ValueError("OOF physical stages do not account for the complete OOF cost")
    training = {
        "raw": timing["full_raw_fit_seconds"],
        "fe": timing["full_fe_fit_seconds"],
        "average": both_fits,
        "blend": both_fits + oof,
    }
    for arm in ARMS:
        if not math.isclose(timing[f"{arm}_training_seconds"], training[arm]) or not math.isclose(
            timing[f"{arm}_standalone_seconds"], training[arm] + timing[f"{arm}_score_seconds"]
        ):
            raise ValueError("Standalone controls must exclude OOF costs and share full fits")
    fingerprint([diagnostics, timing])


def _contrasts(rows: list[dict], config: dict) -> dict:
    paired = tabular.summarize(
        [row for row in rows if row["method"] in METHODS[2:]],
        config["folds"],
        1,
        baseline_method=METHODS[2],
    )
    result = next(item for item in paired if item["method"] == METHODS[3])
    result["unit"] = "complete-fold origin families, not independent outcome rows"
    expected = {(config["seed"], fold) for fold in range(config["folds"])}
    differences, brier = [], []
    for family in sorted({row["family_id"] for row in rows}):
        arms = []
        for method in METHODS[2:]:
            arm = {
                (row["seed"], row["fold"]): row
                for row in rows
                if row["family_id"] == family and row["method"] == method and row["status"] == "success"
            }
            arms.append(arm)
        if all(set(arm) == expected for arm in arms) and all(
            arms[0][key]["split_sha256"] == arms[1][key]["split_sha256"] for key in expected
        ):
            differences.append(float(np.mean([arms[1][key]["loss"] - arms[0][key]["loss"] for key in expected])))
            brier.append(float(np.mean([arms[1][key]["brier_sum"] - arms[0][key]["brier_sum"] for key in expected])))
    result.update(
        complete_absolute_paired_families=len(differences),
        mean_loss_difference=float(np.mean(differences)) if differences else None,
        mean_brier_sum_difference=float(np.mean(brier)) if brier else None,
    )
    return result


def _confirmation_decision(manifest: dict, rows: list[dict], physical_jobs: list[dict]) -> dict:
    config = manifest["config"]
    claims = _claim_fields(config)
    expected = {(method, 2031, fold) for method in CONFIRMATION_METHODS for fold in range(3)}
    cells: dict[tuple, list[dict]] = {}
    jobs: dict[str, list[dict]] = {}
    for row in rows:
        cells.setdefault((row.get("method"), row.get("seed"), row.get("fold")), []).append(row)
    for job in physical_jobs:
        jobs.setdefault(job.get("physical_job_id", ""), []).append(job)
    splits = {(row["seed"], row["fold"]): row["sha256"] for row in manifest["splits"]}
    expected_jobs: dict[str, list[tuple]] = {}
    for method, seed, fold in sorted(expected):
        kind = FIXED_MODE if method in MODE_METHODS[FIXED_MODE] else method
        identifier = RunState.cell_id(CONFIRMATION_FAMILY, seed, fold, kind, "lightgbm")
        expected_jobs.setdefault(identifier, []).append((method, seed, fold))
    blockers, failures = [], []
    if (
        config.get("evaluation_tier") != CONFIRMATION_TIER
        or any(
            config.get("claim_boundary", {}).get(key) is not value
            for key, value in calibration_evaluation.CLAIM_FLAGS.items()
        )
        or any(
            container.get(key) != value
            for container in (manifest, config, manifest["preregistration"], *manifest["datasets"])
            for key, value in claims.items()
        )
    ):
        blockers.append("Root, registration, configuration and dataset must bind the identical qualified claim tier")
    if len(rows) != 15 or set(cells) != expected:
        blockers.append("All 15 registered prediction cells are required without a reduced denominator")
    if len(physical_jobs) != 9 or set(jobs) != set(expected_jobs):
        blockers.append("All nine registered physical jobs are required")
    if len(manifest["splits"]) != 3 or set(splits) != {(2031, fold) for fold in range(3)}:
        blockers.append("Exactly the three locked IID folds are required")
    if (
        manifest.get("source_changed")
        or manifest.get("identity_invalid")
        or manifest.get("interrupted")
        or manifest.get("root_error")
        or manifest.get("source_sha256_at_finish") != manifest["source_sha256"]
    ):
        blockers.append("Final source/run identity verification is incomplete or invalid")
    if len(manifest["datasets"]) != 1 or manifest["datasets"][0].get("preparation_state") != "validated":
        blockers.append("Released full-dataset preparation and locked criteria are required")
    try:
        _confirmation_rules(manifest["preregistration"])
    except ValueError as exc:
        blockers.append(str(exc))
    valid: dict[tuple, dict] = {}
    for key in sorted(expected):
        candidates = cells.get(key, [])
        if len(candidates) != 1:
            blockers.append(f"{key}: missing or duplicate outcome")
            continue
        row = candidates[0]
        if (
            row.get("status") != "success"
            or row.get("expected_cells") != 3
            or row.get("dataset") != CONFIRMATION_FAMILY
            or row.get("family_id") != CONFIRMATION_FAMILY
            or row.get("role") != "confirmation"
            or row.get("learner") != "lightgbm"
            or row.get("protocol") != CONFIRMATION_PROTOCOL
            or row.get("blend_mode") != FIXED_MODE
            or any(
                row.get(field) != value or row.get("dataset_metadata", {}).get(field) != value
                for field, value in claims.items()
            )
            or row.get("split_sha256") in (None, "unavailable")
            or row.get("split_sha256") != splits.get((key[1], key[2]))
            or any(
                isinstance(row.get(metric), bool)
                or not isinstance(row.get(metric), (int, float))
                or not math.isfinite(row[metric])
                or row[metric] < 0
                for metric in METRICS
            )
        ):
            blockers.append(f"{key}: unscored, nonfinite or mismatched source/tier/split identity")
        else:
            valid[key] = row
    complete_jobs = 0
    for identifier, keys in expected_jobs.items():
        records = jobs.get(identifier, [])
        expected_rows = [valid[key] for key in keys if key in valid]
        if (
            len(records) != 1
            or records[0].get("status") != "success"
            or len(expected_rows) != len(keys)
            or any(row.get("physical_job_id") != identifier for row in expected_rows)
            or records[0].get("outcomes") != expected_rows
        ):
            blockers.append(f"{identifier}: incomplete or non-atomic physical job")
        else:
            complete_jobs += 1
    means = {}
    family_losses = []
    for method in CONFIRMATION_METHODS:
        values = [valid[method, 2031, fold]["loss"] for fold in range(3) if (method, 2031, fold) in valid]
        means[method] = math.fsum(value / 3 for value in values) if len(values) == 3 else None
        family_losses.append(
            {
                **claims,
                "family_id": CONFIRMATION_FAMILY,
                "method": method,
                "requested_folds": 3,
                "valid_folds": len(values),
                "mean_loss": means[method],
            }
        )
    comparisons = []
    for reference in (METHODS[0], METHODS[1], *NATIVE_REFERENCES):
        primary, control = means[METHODS[2]], means[reference]
        difference = primary - control if primary is not None and control is not None else None
        reason: str | None = None
        gain: float | None = None
        harm_passed: bool | None = None
        if primary is None or control is None:
            reason = "incomplete_family_folds"
        elif control <= 0:
            reason = "undefined_zero_reference_loss"
        else:
            gain = 1.0 - primary / control
            if reference == METHODS[0]:
                harm_passed = primary <= control * 1.05
        if gain is not None and not math.isfinite(gain):
            gain, reason = None, "undefined_nonfinite_relative_gain"
        mean_passed = (gain > 0 if reference == METHODS[0] else gain >= 0) if gain is not None else None
        status = "blocked" if reason else "pass" if mean_passed and harm_passed is not False else "fail"
        if reason:
            blockers.append(f"{reference}: {reason}")
        elif status == "fail":
            failures.append(f"{reference}: registered mean-gain/raw-harm gate failed")
        comparisons.append(
            {
                **claims,
                "family_id": CONFIRMATION_FAMILY,
                "policy_method": METHODS[2],
                "reference_method": reference,
                "requested_families": 1,
                "complete_paired_families": int(difference is not None),
                "policy_mean_loss": primary,
                "reference_mean_loss": control,
                "absolute_loss_difference": difference,
                "relative_gain": gain,
                "undefined_reason": reason,
                "mean_gain_gate_passed": mean_passed,
                "raw_family_harm_gate_passed": harm_passed,
                "status": status,
            }
        )
    status = "blocked" if blockers else "fail" if failures else "pass"
    return {
        **claims,
        "protocol": CONFIRMATION_PROTOCOL,
        "status": status,
        "locked_criteria_passed": status == "pass",
        "primary_method": METHODS[2],
        "rules": _confirmation_rules(),
        "scope": CONFIRMATION_CLAIM_NOTE,
        "requested_families": 1,
        "unit": "one complete origin family; three folds are not independent datasets",
        "expected_prediction_rows": 15,
        "observed_prediction_rows": len(rows),
        "valid_prediction_rows": len(valid),
        "expected_physical_jobs": 9,
        "observed_physical_jobs": len(physical_jobs),
        "complete_physical_jobs": complete_jobs,
        "family_losses": family_losses,
        "comparisons": comparisons,
        "blockers": blockers,
        "failures": failures,
        "automatic_sota": False,
        "automatic_default_promotion": False,
        "clinical_generalization_claims_allowed": False,
        "patient_entity_disjoint_claims_allowed": False,
    }


def _decision_lines(decision: dict) -> list[str]:
    return [
        "",
        "## Authoritative locked confirmation decision",
        "",
        f"**{decision['status'].upper()}**: {decision['valid_prediction_rows']}/15 valid outcomes; "
        f"{decision['complete_physical_jobs']}/9 complete physical jobs; the single family is never dropped.",
        decision["scope"],
        "",
        "| Comparator | Primary mean loss | Comparator mean loss | Absolute difference | Relative gain | Raw harm gate | Decision |",
        "|---|---:|---:|---:|---:|---|---|",
        *[
            f"| {item['reference_method']} | {item['policy_mean_loss']} | {item['reference_mean_loss']} | "
            f"{item['absolute_loss_difference']} | {item['relative_gain']} | {item['raw_family_harm_gate_passed']} | {item['status']} |"
            for item in decision["comparisons"]
        ],
        "",
        *[f"- Blocked: {reason}" for reason in decision["blockers"]],
        *[f"- Failed: {reason}" for reason in decision["failures"]],
    ]


def write_report(output: Path, manifest: dict, rows: list[dict], physical_jobs: list[dict]) -> None:
    """Write complete coverage, family contrasts and single-counted physical costs."""
    config = manifest["config"]
    mode = config["blend_mode"]
    fixed = mode == FIXED_MODE
    confirmation = config.get("allow_confirmation", False)
    methods, arms = _all_methods(config), MODE_ARMS[mode]
    bundle_methods = MODE_METHODS[mode]
    claims = _claim_fields(config)
    if (
        manifest["blend_mode"] != mode
        or manifest["protocol"] != _protocol(config)
        or any(manifest.get(key) != value for key, value in claims.items())
        or config["methods"] != list(methods)
        or any(
            row.get("blend_mode") != mode
            or row.get("protocol") != _protocol(config)
            or row.get("method") not in methods
            or any(row.get(key) != value for key, value in claims.items())
            for row in rows
        )
    ):
        raise ValueError("Reports cannot pool modes, protocols or unrequested method labels")
    summary = tabular.summarize(rows, config["folds"], len(config["seeds"]), baseline_method=METHODS[0])
    for item in summary:
        item.update(blend_mode=mode, protocol=_protocol(config), **claims)
        if confirmation:
            item["primary_predeclared"] = item["method"] == METHODS[2]
    contrast = None if fixed else _contrasts(rows, config)
    coverage = {
        method: {
            status: sum(row["method"] == method and row["status"] == status for row in rows)
            for status in sorted({row["status"] for row in rows})
        }
        for method in methods
    }
    costs = {
        "note": manifest["cost_note"],
        "active_seconds_authority": "manifest.json",
        "recorded_physical_jobs": len(physical_jobs),
        "attempted_physical_jobs": sum(job.get("attempt", 0) > 0 for job in physical_jobs),
        "measured_physical_job_seconds_including_retries": sum(job["physical_seconds"] for job in physical_jobs),
    }
    diagnostics = [
        {
            **{key: row[key] for key in ("physical_job_id", "dataset", "family_id", "seed", "fold")},
            "diagnostics": row["shared_bundle_diagnostics"],
            "outer_loss": {
                arm: next(
                    item["loss"]
                    for item in rows
                    if item["physical_job_id"] == row["physical_job_id"] and item["method"] == method
                )
                for arm, method in zip(arms, bundle_methods, strict=True)
            },
        }
        for row in rows
        if row["method"] == bundle_methods[-1] and row["status"] == "success"
    ]
    decision = _confirmation_decision(manifest, rows, physical_jobs) if confirmation else None
    atomic_json(
        output / "results.json",
        {
            "protocol": _protocol(config),
            "blend_mode": mode,
            **claims,
            "run_id": manifest["run_id"],
            "config": manifest["config"],
            "cohort_scope": manifest["cohort_scope"],
            "source_changed": manifest["source_changed"],
            "identity_invalid": manifest["identity_invalid"],
            "interrupted": manifest.get("interrupted", False),
            "root_error": manifest.get("root_error"),
            "expected_physical_jobs": manifest["expected_physical_jobs"],
            "planned_cells": manifest["planned_cells"],
            "rows": rows,
            "summary": summary,
            **(
                {
                    "fixed_average_policy": {
                        "weight": 0.5,
                        "scope": "global_predeclared",
                        "optimizer_used": False,
                        "complete_procedures_per_job": 2,
                        "pipeline_fits_per_job": 8,
                        **({"applies_to_physical_method": FIXED_MODE} if confirmation else {}),
                    }
                }
                if fixed
                else {"average_vs_learned": contrast}
            ),
            "coverage": coverage,
            "blend_diagnostics": diagnostics,
            "physical_cost": costs,
            "physical_jobs": [{key: value for key, value in job.items() if key != "outcomes"} for job in physical_jobs],
            "claim_note": manifest["claim_note"],
            **({"confirmation_decision": decision} if decision is not None else {}),
        },
    )
    lines = [
        (
            "# Locked qualified fixed-average confirmation"
            if confirmation
            else (
                "# Fixed-average development experiment"
                if fixed
                else "# Fixed OOF prediction-blending development experiment"
            )
        ),
        "",
        manifest["claim_note"],
        "",
        (
            f"Scope: **{CONFIRMATION_TIER}**. One locked family, nine physical jobs, 15 outcomes; fixed policy `{METHODS[2]}`."
            if confirmation
            else f"Scope: **{manifest['cohort_scope']}**. Mode: **{mode}**. Full design: nine families, "
            f"{81 if fixed else 27} physical bundles, {243 if fixed else 108} outcomes."
        ),
        f"Source changed: {manifest['source_changed']}; identity invalid: {manifest['identity_invalid']}; "
        f"interrupted: {manifest.get('interrupted', False)}. Root error: {manifest.get('root_error')}.",
        "Outcome rows and outer folds are not independent datasets. Subset/nondefault-seed smoke is not the full study.",
        (
            (
                "LightGBM only; seed 2031; full rows, three new IID folds; one fixed 0.5 average, no OOF or optimizer."
                if confirmation
                else "LightGBM only; seeds 42/137/2029; one global fixed 0.5 probability average, no OOF or optimizer."
            )
            if fixed
            else "LightGBM only; CV5 seed+1000 OOF weight; endpoints 0/1 and bounded scalar minimization of OOF log loss."
        ),
        "Onehot categorical encoding, q0/q1 robust clipping, identity targets; candidates/additions 1000/50, repeats 3.",
        "Reported FE/average/blend width is the final full-FE plan width, not an OOF feature union or ensemble width.",
        "Compact provenance retains full-plan names, member/full plan/schema/state hashes, selected parameters, "
        "three inner-trial losses and partition counts/hashes; it contains no models, rows or vocabularies.",
        "",
        manifest["cost_note"],
        "",
        *(_decision_lines(decision) if decision is not None else []),
        (
            "## Requested five-method coverage"
            if confirmation
            else "## Requested three-arm coverage" if fixed else "## Requested four-arm coverage"
        ),
        "",
        "| Method | Status counts |",
        "|---|---|",
        *[f"| {method} | {json.dumps(counts, sort_keys=True)} |" for method, counts in coverage.items()],
        "",
        "## Equal-family primary loss versus raw robust",
        "",
        "| Method | Complete/requested families | Mean relative loss reduction |",
        "|---|---|---|",
        *[
            f"| {item['method']} | {item['complete_paired_families']}/{item['requested_families']} "
            f"| {item['mean_loss_reduction']} |"
            for item in summary
        ],
        "",
        *(
            []
            if fixed
            else ["## Fixed 0.5 average versus OOF-learned blend", "", json.dumps(contrast, sort_keys=True), ""]
        ),
        (
            "## Per-job fixed-average policy, full-model provenance and outer-test diagnostics"
            if fixed
            else "## Per-job weight and OOF/meta-training versus outer-test diagnostics"
        ),
        "",
        *[json.dumps(item, sort_keys=True) for item in diagnostics],
        "",
        "## Unscored outcomes",
        "",
        *[
            f"- {row['dataset']} seed={row['seed']} fold={row['fold']} {row['method']}: "
            f"{row['status']}: {str(row.get('error', '')).replace(chr(10), ' ')}"
            for row in rows
            if row["status"] != "success"
        ],
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    archive_results(output / "results.json")


def _prepare(
    spec: dict, suite: dict, config: dict, state: RunState, data_path: Path, *, registration: dict | None = None
) -> tuple[dict, str, list[tuple[int, int, Any, Any]]]:
    cached = state.datasets.get(spec["name"])
    prepared = None
    if cached is not None:
        content = data_path.read_bytes()
        if tabular.digest(content) != cached["prepared_sha256"]:
            raise ResumeMismatchError("Cannot resume: prepared pickle hash changed")
        prepared = pickle.loads(content)
    # Never trust cached metadata alone: acquisition rechecks original raw pins.
    data, acquired, official = execution._prepare_dataset(
        spec, execution.source_kind(spec), suite, False, config, tabular.ROOT
    )
    if (official is not None and registration is None) or any(
        data.get(key) is not None for key in ("groups", "timestamps", "label_end_times")
    ):
        raise ValueError("Prepared data contains unsupported official/group/time boundaries")
    metadata = {**spec, "source_specification": spec.get("source"), **acquired, "source": execution.source_kind(spec)}
    if any(metadata.get(key) != spec.get(key) for key in ("name", "family_id", "role", "task")):
        raise ValueError("Acquired metadata changed the canonical dataset/family/role/task identity")
    if registration is not None:
        predictors = [
            column
            for column in spec["source"]["columns"]
            if column not in {*spec["source"]["target_columns"], *spec["drop_columns"]}
        ]
        mapping = dict(zip((f"x{index}" for index in range(len(predictors))), predictors, strict=True))
        if (
            any(metadata.get(key) != value for key, value in _claim_fields(config).items())
            or len(data["X"]) != spec["source"]["expected_rows"]
            or metadata.get("original_rows") != spec["source"]["expected_rows"]
            or metadata.get("sampled_rows") != len(data["X"])
            or metadata.get("sample_seed") != 2026
            or metadata.get("column_mapping") != mapping
            or metadata.get("columns") != predictors
            or list(data["X"].columns) != list(mapping)
        ):
            raise ValueError(
                "Confirmation payload changed its full rows, feature/target exclusions, sampling or claim tier"
            )
    if prepared is not None:
        if (
            set(data) != set(prepared)
            or not data["X"].equals(prepared["X"])
            or not data["y"].equals(prepared["y"])
            or fingerprint({key: value for key, value in data.items() if key not in ("X", "y")})
            != fingerprint({key: value for key, value in prepared.items() if key not in ("X", "y")})
        ):
            raise ResumeMismatchError("Cannot resume: current raw payload differs from prepared dataset")
        data = prepared
    content = pickle.dumps(data, protocol=5) if prepared is None else content
    prepared_hash = tabular.digest(content)
    splits = (
        execution._confirmation_splits(spec, data, metadata, official, registration, config)
        if registration is not None
        else execution._splits(data, "classification", config, None)
    )
    if registration is not None:
        metadata.update(used_split_policy="iid", resplit_official_as_iid=True, selected_official_split_ids=[])
    if len(splits) != config["folds"] * len(config["seeds"]) or {(seed, fold) for seed, fold, _, _ in splits} != {
        (seed, fold) for seed in config["seeds"] for fold in range(config["folds"])
    }:
        raise ValueError("Prepared outer splits differ from the requested frozen design")
    identity = {
        "metadata": metadata,
        "prepared_sha256": prepared_hash,
        "splits": [
            {"seed": seed, "fold": fold, "sha256": execution.split_fingerprint(train, test)}
            for seed, fold, train, test in splits
        ],
    }
    if prepared is None:
        staging = data_path.with_suffix(".pkl.pending")
        staging.write_bytes(content)
        staging.replace(data_path)
    state.validate_dataset(spec["name"], identity)
    state.save()
    return metadata, prepared_hash, splits


def _unfinished_preparation(previous: dict, *, mode: str = MODE, method: str | None = None) -> bool:
    return (
        previous.get("status") in {"budget_exhausted", "interrupted"}
        and previous.get("preparation_state") == "unavailable"
        and previous.get("attempt", 0) == 0
        and len(previous.get("outcomes", [])) == (1 if method in NATIVE_REFERENCES else len(MODE_METHODS[mode]))
        and all(
            row.get("preparation_state") == "unavailable"
            and row.get("split_sha256") == "unavailable"
            and row.get("prepared_sha256") is None
            and row.get("status") in {"budget_exhausted", "interrupted"}
            and not any(metric in row for metric in METRICS)
            for row in previous["outcomes"]
        )
    )


def execute(args: argparse.Namespace) -> int:
    """Predeclare and run or resume one bounded, mode-locked atomic experiment."""
    started = time.perf_counter()
    output = args.output.resolve()
    previous = json.loads((output / "manifest.json").read_text(encoding="utf-8")) if args.resume else {}
    if previous.get("source_changed") or previous.get("identity_invalid"):
        raise ResumeMismatchError("Cannot resume an invalidated experiment; use a new directory")
    confirmation = (
        args.allow_confirmation
        if args.allow_confirmation is not None
        else previous.get("config", {}).get("allow_confirmation", False)
    )
    if confirmation and not args.resume and args.suite is None:
        raise ValueError("Qualified confirmation requires an explicit released --suite")
    mode = args.blend_mode or previous.get("config", {}).get("blend_mode", FIXED_MODE if confirmation else MODE)
    if mode not in MODE_METHODS:
        raise ValueError(f"blend_mode must be one of {tuple(MODE_METHODS)}")
    defaults = dict(DEFAULTS)
    if mode == FIXED_MODE:
        defaults.update(blend_mode=mode, methods=list(MODE_METHODS[mode]))
        for key in ("oof_folds", "oof_seed_offset"):
            defaults.pop(key)
    if confirmation:
        defaults.update(CONFIRMATION_SETTINGS)
    config: dict[str, Any] = {
        key: (
            getattr(args, key, None)
            if getattr(args, key, None) is not None
            else previous.get("config", {}).get(key, value)
        )
        for key, value in defaults.items()
    }
    if args.seed is not None:
        config["seeds"] = [args.seed]
    elif args.seeds is not None:
        config["seeds"] = args.seeds
    else:
        config["seeds"] = previous.get("config", {}).get(
            "seeds", [2031] if confirmation else FIXED_SEEDS.copy() if mode == FIXED_MODE else [config["seed"]]
        )
    if config["seeds"]:
        config["seed"] = config["seeds"][0]
    config["suite"] = str(Path(config["suite"]).resolve())
    budget = (
        args.total_seconds
        if args.total_seconds is not None
        else previous.get("budget_seconds", 14400.0 if confirmation else TOTAL_SECONDS)
    )
    validate_config(config, budget)
    if args.retry_failed and not args.resume:
        raise ValueError("--retry-failed requires --resume")
    if args.resume and budget < previous["budget_seconds"]:
        raise ResumeMismatchError("The cumulative operational budget cannot decrease on resume")
    suite, legacy = execution._load_suite(Path(config["suite"]), confirmation)
    registration = None
    if confirmation:
        specs, registration = _bind_confirmation(suite, legacy, config, budget)
    else:
        specs = _selected_specs(suite, legacy, config["datasets"])
    identity = {
        "protocol": _protocol(config),
        "blend_mode": mode,
        **_claim_fields(config),
        "config": config,
        "suite": suite,
        "canonical_source_pin": CANONICAL_SUITE_SHA256,
        "canonical_normalized_pin": CANONICAL_SPECS_SHA256,
        "suite_file_sha256": tabular.digest(Path(config["suite"]).read_bytes()),
        "source_sha256": tabular.source_fingerprint(),
        "versions": execution.environment_versions(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    previous_archive = previous.get("source_archive_sha256")
    if (
        args.resume
        and previous_archive is not None
        and tabular.digest((output / "source.zip").read_bytes()) != previous_archive
    ):
        raise ResumeMismatchError("Cannot resume: source ZIP changed")
    state = RunState(output, identity, resume=args.resume)
    if (
        args.resume
        and previous_archive is None
        and (
            state.datasets
            or any(not _unfinished_preparation(job, mode=mode, method=job.get("method")) for job in state.rows.values())
        )
    ):
        raise ResumeMismatchError("Cannot resume: source archive identity is unavailable for prepared/attempted jobs")
    prior_elapsed = state.elapsed_seconds
    deadline = started + max(0.0, budget - prior_elapsed)
    manifest = {
        **identity,
        "run_id": state.run_id,
        "started_utc": previous.get("started_utc", datetime.now(timezone.utc).isoformat()),
        "budget_seconds": budget,
        "report_reserve_seconds": REPORT_RESERVE_SECONDS,
        "expected_physical_jobs": len(specs) * config["folds"] * len(config["seeds"]) * (3 if confirmation else 1),
        "planned_cells": len(specs) * config["folds"] * len(config["seeds"]) * len(_all_methods(config)),
        "cohort_scope": (
            "locked_single_family_qualified_confirmation"
            if confirmation
            else (
                "canonical_nine"
                if set(config["datasets"]) == set(DEFAULT_DATASETS)
                and config["seeds"] == (FIXED_SEEDS if mode == FIXED_MODE else [42])
                else "canonical_subset_or_nondefault_seed_smoke"
            )
        ),
        "prepare_only": args.prepare_only,
        "source_changed": False,
        "source_archive_sha256": previous_archive,
        "identity_invalid": False,
        "cost_note": (
            "Per fold, the three-arm fixed-average bundle and the two native-reference singles are separate "
            "physical jobs. The following shared-cost attribution applies only to the bundle: " + FIXED_COST_NOTE
            if confirmation
            else FIXED_COST_NOTE if mode == FIXED_MODE else COST_NOTE
        ),
        "claim_note": (
            CONFIRMATION_CLAIM_NOTE if confirmation else FIXED_CLAIM_NOTE if mode == FIXED_MODE else CLAIM_NOTE
        ),
        **({"preregistration": registration} if registration is not None else {}),
        "datasets": [],
        "splits": [],
    }
    jobs: list[dict] = []
    for spec in specs:
        metadata = {**spec, "source_specification": spec.get("source"), "source": execution.source_kind(spec)}
        for seed, fold, method in (
            (seed, fold, method)
            for seed in config["seeds"]
            for fold in range(config["folds"])
            for method in ((mode, *NATIVE_REFERENCES) if confirmation else (mode,))
        ):
            job_id = state.cell_id(spec["name"], seed, fold, method, "lightgbm")
            jobs.append(
                {
                    "dataset": spec["name"],
                    "family_id": spec["family_id"],
                    "role": spec["role"],
                    "source": execution.source_kind(spec),
                    "task": "classification",
                    "protocol": _protocol(config),
                    "blend_mode": mode,
                    **_claim_fields(config),
                    "method": method,
                    "physical_job_id": job_id,
                    "seed": seed,
                    "fold": fold,
                    "split_sha256": "unavailable",
                    "train_idx": None,
                    "test_idx": None,
                    "expected_cells": config["folds"] * len(config["seeds"]),
                    "dataset_metadata": metadata,
                    "data_path": str(output / "prepared" / f"{spec['name']}.pkl"),
                    "prepared_sha256": None,
                    "preparation_state": "unavailable",
                    "config": {**config, "seed": seed},
                    "status": "pending",
                    "error": "Not evaluated",
                }
            )
    manifest["predeclared_jobs"] = [
        {
            key: job[key]
            for key in ("dataset", "family_id", "seed", "fold", "method", "physical_job_id", "expected_cells")
        }
        for job in jobs
    ]
    interrupted = False
    fatal: Exception | None = None
    ledger: ExposureLedger | None = None
    admitted: set[str] = set()

    def cell_hash(job: dict) -> str:
        return fingerprint([state.identity_hash, job["physical_job_id"], job["prepared_sha256"], job["split_sha256"]])

    def record(job: dict, outcomes: list[dict], seconds: float = 0.0) -> None:
        job_id = job["physical_job_id"]
        state.elapsed_seconds = prior_elapsed + time.perf_counter() - started
        state.record(
            job_id,
            cell_hash(job),
            {
                "status": outcomes[0]["status"],
                "blend_mode": mode,
                "protocol": _protocol(config),
                **_claim_fields(config),
                "method": job["method"],
                "physical_job_id": job_id,
                "preparation_state": job["preparation_state"],
                "physical_seconds": state.rows.get(job_id, {}).get("physical_seconds", 0.0) + seconds,
                "last_attempt_seconds": seconds,
                "outcomes": outcomes,
            },
        )

    def persist() -> None:
        snapshots = [
            state.rows.get(job["physical_job_id"])
            or {
                "status": job["status"],
                "physical_job_id": job["physical_job_id"],
                "physical_seconds": 0.0,
                "attempt": 0,
                "outcomes": _outcomes(job, job["status"], job["error"]),
            }
            for job in jobs
        ]
        write_report(output, manifest, [row for job in snapshots for row in job["outcomes"]], snapshots)
        state.elapsed_seconds = prior_elapsed + time.perf_counter() - started
        manifest["active_seconds"] = state.elapsed_seconds
        manifest["budget_overshoot_seconds"] = max(0.0, state.elapsed_seconds - budget)
        state.save()
        atomic_json(output / "manifest.json", manifest)

    try:
        atomic_json(output / "manifest.json", manifest)
        paths = tabular.source_files()
        manifest["source_archive_sha256"] = (
            previous_archive if previous_archive is not None else archive_source(output, tabular.ROOT, paths, suite)
        )
        manifest["source_allowlist"] = [str(path.relative_to(tabular.ROOT)) for path in paths]
        atomic_json(output / "manifest.json", manifest)
        (output / "prepared").mkdir(exist_ok=True)
        if confirmation:
            ledger = ExposureLedger(tabular.ROOT / "benchmarks" / ".cache" / "exposure.sqlite")
            for spec in specs:
                ledger.check(spec["family_id"], state.run_id, "confirmation")
        for spec in specs:
            selected = [job for job in jobs if job["dataset"] == spec["name"]]
            cached = state.datasets.get(spec["name"])
            try:
                if tabular.source_fingerprint() != identity["source_sha256"]:
                    manifest["source_changed"] = True
                    raise RuntimeError("Source changed during preparation; stopping the frozen study")
                if cached is None and time.perf_counter() >= deadline - REPORT_RESERVE_SECONDS:
                    raise TimeoutError("Active-time budget exhausted before preparation (including cleanup reserve)")
                old = state.rows.get(selected[0]["physical_job_id"], {})
                if cached is None and old.get("status") == "dataset_error" and not args.retry_failed:
                    raise ValueError(old["outcomes"][0]["error"])
                metadata, prepared_hash, splits = _prepare(
                    spec, suite, config, state, Path(selected[0]["data_path"]), registration=registration
                )
                for seed, fold, train, test in splits:
                    for job in selected:
                        if job["seed"] == seed and job["fold"] == fold:
                            job.update(
                                dataset_metadata=metadata,
                                prepared_sha256=prepared_hash,
                                train_idx=train,
                                test_idx=test,
                                split_sha256=execution.split_fingerprint(train, test),
                                preparation_state="validated",
                                status="prepared" if args.prepare_only else "pending",
                                error="Prepared only" if args.prepare_only else "Not evaluated",
                            )
            except (KeyboardInterrupt, ResumeMismatchError):
                raise
            except Exception as exc:
                if cached is not None and not isinstance(exc, TimeoutError):
                    raise ResumeMismatchError(f"Cannot resume: fresh dataset validation failed: {exc}") from exc
                for job in selected:
                    job.update(
                        status="budget_exhausted" if isinstance(exc, TimeoutError) else "dataset_error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
            first = selected[0]
            manifest["datasets"].append(
                {
                    **first["dataset_metadata"],
                    "name": spec["name"],
                    "prepared_sha256": first["prepared_sha256"],
                    "preparation_state": first["preparation_state"],
                    "preparation_status": first["status"],
                    "error": first["error"] if first["status"] not in {"pending", "prepared"} else None,
                }
            )
        manifest["splits"] = [
            {**{key: job[key] for key in ("dataset", "seed", "fold")}, "sha256": job["split_sha256"]}
            for job in jobs
            if job["method"] == mode
        ]
        atomic_json(output / "manifest.json", manifest)
        for job in sorted(jobs, key=lambda item: (item["fold"], item["seed"], item["dataset"])):
            if args.prepare_only:
                break
            if tabular.source_fingerprint() != identity["source_sha256"]:
                manifest["source_changed"] = True
            old = state.rows.get(job["physical_job_id"], {})
            if not _unfinished_preparation(old, mode=mode, method=job["method"]) and state.reusable(
                job["physical_job_id"], cell_hash(job), retry_failed=args.retry_failed
            ):
                if old["status"] == "success" and any(
                    row["status"] != "success" for row in normalize_outcomes(job, old["outcomes"])
                ):
                    raise ResumeMismatchError("Cannot resume: completed bundle identity/metrics changed")
                continue
            remaining = deadline - time.perf_counter() - REPORT_RESERVE_SECONDS
            if manifest["source_changed"]:
                outcomes = _outcomes(job, "source_changed", "Source changed; study invalidated")
            elif job["status"] not in {"pending", "prepared"}:
                outcomes = _outcomes(job, job["status"], job["error"])
            elif remaining <= 0:
                outcomes = _outcomes(
                    job, "budget_exhausted", "Active-time budget exhausted (including cleanup reserve)"
                )
            else:
                directory = state.next_attempt(job["physical_job_id"])
                attempt_started = time.perf_counter()
                try:
                    if tabular.digest(Path(job["data_path"]).read_bytes()) != job["prepared_sha256"]:
                        raise ResumeMismatchError("Prepared dataset changed before dispatch")
                    if ledger is None:
                        ledger = ExposureLedger(tabular.ROOT / "benchmarks" / ".cache" / "exposure.sqlite")
                    if job["family_id"] not in admitted:
                        ledger.check(job["family_id"], state.run_id, job["role"])
                        ledger.admit(job["family_id"], state.run_id, job["role"])
                        admitted.add(job["family_id"])
                    raw = tabular.run_isolated(
                        job,
                        directory,
                        min(config["case_timeout"], remaining),
                        **(
                            {}
                            if job["method"] in NATIVE_REFERENCES
                            else {"worker_module": "benchmarks.blending_evaluation"}
                        ),
                    )
                    outcomes = normalize_outcomes(job, raw)
                except KeyboardInterrupt:
                    record(
                        job,
                        _outcomes(job, "interrupted", "Experiment interrupted"),
                        time.perf_counter() - attempt_started,
                    )
                    raise
                except Exception as exc:
                    outcomes = _outcomes(job, "error", f"{type(exc).__name__}: {exc}")
                seconds = time.perf_counter() - attempt_started
                try:
                    if tabular.source_fingerprint() != identity["source_sha256"]:
                        manifest["source_changed"] = True
                        outcomes = _outcomes(
                            job, "source_changed", "Source changed during physical job; scores invalidated"
                        )
                except KeyboardInterrupt:
                    record(job, _outcomes(job, "interrupted", "Experiment interrupted after worker"), seconds)
                    raise
                for row in outcomes:
                    row["log"] = str((directory / "worker.log").relative_to(output))
                record(job, outcomes, seconds)
                persist()
                continue
            record(job, outcomes)
    except KeyboardInterrupt:
        interrupted = True
    except Exception as exc:
        fatal = exc
        manifest["identity_invalid"] = isinstance(exc, ResumeMismatchError)
        logger.exception("Experiment stopped; retaining all requested outcomes")
    finally:
        for job in jobs:
            if job["physical_job_id"] not in state.rows and (
                not args.prepare_only or interrupted or fatal or job["status"] not in {"pending", "prepared"}
            ):
                status = "interrupted" if interrupted else "error" if fatal else job["status"]
                if job["preparation_state"] == "unavailable" and (interrupted or fatal):
                    job.update(status=status, error=str(fatal) if fatal else "Experiment interrupted")
                record(
                    job,
                    _outcomes(
                        job, status, str(fatal) if fatal else "Experiment interrupted" if interrupted else job["error"]
                    ),
                )
        manifest["datasets"] = [
            {
                **job["dataset_metadata"],
                "name": job["dataset"],
                "prepared_sha256": job["prepared_sha256"],
                "preparation_state": job["preparation_state"],
                "preparation_status": job["status"],
                "error": job["error"] if job["status"] not in {"pending", "prepared"} else None,
            }
            for job in jobs
            if job["fold"] == 0 and job["seed"] == config["seeds"][0] and job["method"] == mode
        ]
        manifest["splits"] = [
            {**{key: job[key] for key in ("dataset", "seed", "fold")}, "sha256": job["split_sha256"]}
            for job in jobs
            if job["method"] == mode
        ]
        manifest["interrupted"] = interrupted
        manifest["root_error"] = f"{type(fatal).__name__}: {fatal}" if fatal else None
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            manifest["source_sha256_at_finish"] = tabular.source_fingerprint()
            manifest["source_changed"] |= manifest["source_sha256_at_finish"] != identity["source_sha256"]
            persist()
        except KeyboardInterrupt:
            manifest["interrupted"] = interrupted = True
            persist()
    return int(
        interrupted
        or fatal is not None
        or manifest["source_changed"]
        or state.elapsed_seconds > budget
        or any(job["status"] not in {"pending", "prepared"} for job in jobs)
        or (not args.prepare_only and any(job["status"] != "success" for job in state.rows.values()))
    )


def main(argv: list[str] | None = None) -> int:
    """Run the fixed development experiment, preparation, resume or private worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=tabular.ROOT / "benchmarks" / "tabular-results" / "blending-development"
    )
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--allow-confirmation", action="store_true", default=None)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--blend-mode", choices=tuple(MODE_METHODS))
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int)
    seeds.add_argument("--seeds", type=int, nargs="+")
    for name in (
        "folds",
        "threads",
        "max_rows",
        "max_candidates",
        "max_added_features",
        "stability_repeats",
        "memory_limit_mb",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    parser.add_argument("--case-timeout", type=float)
    parser.add_argument("--total-seconds", type=float)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if args.worker_output is None:
            parser.error("--worker requires --worker-output")
        payload = pickle.loads(args.worker.read_bytes())
        try:
            rows = run_case(payload, args.worker.parent)
        except Exception as exc:
            rows = _outcomes(payload, "error", f"{type(exc).__name__}: {exc}")
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
