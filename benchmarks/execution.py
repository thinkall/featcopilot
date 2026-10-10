"""Catalog-backed, per-learner benchmark execution and exact resume."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import pickle
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
DEFAULTS: dict[str, Any] = {
    "datasets": None,
    "folds": 3,
    "seeds": [42, 137],
    "max_rows": 5000,
    "max_added_features": 50,
    "max_candidates": 1000,
    "threads": 2,
    "memory_limit_mb": 4096,
    "case_timeout": 120.0,
    "synthetic": False,
    "stability_repeats": 3,
    "screening": "auto",
    "allow_confirmation": False,
    "detailed_diagnostics": False,
    "live_llm": False,
    "llm_ledger": None,
    "allow_soft_llm_tokens": False,
}
LEGACY_FAMILIES = {
    "magic": "magic-telescope",
    "bank": "bank-marketing",
    "credit": "give-me-some-credit",
    "heloc": "heloc",
    "credit_default": "default-of-credit-card-clients",
    "eye_movements": "eye-movements",
    "cpu": "cpu-act",
    "wine": "wine-quality",
    "houses": "california-housing",
    "elevators": "elevators",
    "abalone": "abalone",
    "diamonds": "diamonds",
}


def environment_versions() -> dict[str, str | None]:
    """Record all libraries that can affect feature generation or model fitting."""
    names = (
        "featcopilot",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "lightgbm",
        "catboost",
        "xgboost",
        "openfe",
        "autofeat",
        "openml",
        "psutil",
        "joblib",
        "threadpoolctl",
        "pydantic",
        "sympy",
        "numba",
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _load_suite(path: Path, allow_confirmation: bool) -> tuple[dict, bool]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    legacy = (
        "revision" in raw
        and "schema_version" not in raw
        and all("provider" not in row for row in raw.get("datasets", []))
    )
    if legacy:
        if not isinstance(raw["revision"], str) or not re.fullmatch(r"[0-9a-fA-F]{40}", raw["revision"]):
            raise ValueError("Legacy replay requires an immutable 40-character source revision")
        for spec in raw["datasets"]:
            spec.setdefault("family_id", LEGACY_FAMILIES.get(spec["name"], spec["name"]))
            spec.setdefault("role", "development")
            spec["rights_status"] = "legacy-not-reviewed"
        return raw, True
    from benchmarks.catalog import load_suite

    return load_suite(path, allow_confirmation=allow_confirmation), False


def _prepare_dataset(spec: dict, source: str, suite: dict, legacy: bool, config: dict, root: Path):
    from benchmarks import tabular

    groups, timestamps, label_end_times, official_splits = None, None, None, None
    if spec.get("provider") == "diagnostic":
        from benchmarks.datasets import load_dataset

        X, y, _, _ = load_dataset(spec["name"])
        metadata = {
            **spec,
            "source": source,
            "sampled_rows": len(X),
            "sha256": tabular.digest(pd.util.hash_pandas_object(pd.concat([X, y], axis=1)).to_numpy().tobytes()),
        }
    elif legacy or "_legacy_revision" in spec:
        revision = spec.get("_legacy_revision", suite.get("revision"))
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError("Legacy replay requires an immutable source revision")
        cached = root / "benchmarks" / ".cache" / revision / Path(spec["path"])
        if not cached.exists():
            raise ValueError(
                "Legacy manifests support local cached-data replay only; use a rights-reviewed catalog manifest for acquisition"
            )
        X, y, metadata = tabular.load_real_dataset(spec, revision, root / "benchmarks" / ".cache", config["max_rows"])
    else:
        from benchmarks.catalog import load_dataset as load_catalog_dataset

        loaded = load_catalog_dataset(
            spec, root / "benchmarks" / ".cache", max_rows=config["max_rows"], sample_seed=2026
        )
        X, y, metadata = loaded.X, loaded.y, loaded.metadata
        groups, timestamps, official_splits = loaded.groups, loaded.timestamps, loaded.official_splits
        label_end_times = getattr(loaded, "label_end_times", None)
        if spec.get("label_end_column") is not None and label_end_times is None:
            raise ValueError("Declared label_end_column was not returned as split metadata")
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y), name="target")
    if spec["task"] == "classification":
        encoder = LabelEncoder()
        y = pd.Series(encoder.fit_transform(y), name="target")
        metadata = {**metadata, "evaluation_classes": [str(value) for value in encoder.classes_]}
    for name, values in (("groups", groups), ("timestamps", timestamps), ("label_end_times", label_end_times)):
        if values is not None and len(values) != len(X):
            raise ValueError(f"{name} must have one value per dataset row")
    groups = None if groups is None else pd.Series(np.asarray(groups), index=X.index)
    timestamps = None if timestamps is None else pd.Series(np.asarray(timestamps), index=X.index)
    label_end_times = None if label_end_times is None else pd.Series(np.asarray(label_end_times), index=X.index)
    metadata = {
        **metadata,
        "family_id": spec.get("family_id", spec["name"]),
        "role": spec.get("role", "development"),
        "source": source,
    }
    descriptions = metadata.get("column_mapping")
    if descriptions is None:
        descriptions = dict(zip(X.columns, metadata.get("columns", X.columns), strict=True))
    return (
        {
            "X": X,
            "y": y,
            "groups": groups,
            "timestamps": timestamps,
            "label_end_times": label_end_times,
            "column_descriptions": descriptions,
        },
        metadata,
        official_splits,
    )


def _splits(data: dict, task: str, config: dict, official_splits) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    from featcopilot.utils.splitting import make_cv_splits, validate_split_boundaries

    results = []
    for seed in config["seeds"]:
        pairs = official_splits
        if pairs is None:
            pairs = make_cv_splits(
                data["y"].to_numpy(),
                task,
                config["folds"],
                seed,
                groups=None if data["groups"] is None else data["groups"].to_numpy(),
                timestamps=None if data["timestamps"] is None else data["timestamps"].to_numpy(),
                label_end_times=None if data.get("label_end_times") is None else data["label_end_times"].to_numpy(),
            )
            if len(pairs) != config["folds"]:
                raise ValueError(
                    f"Requested {config['folds']} outer folds but only {len(pairs)} are feasible; "
                    "choose a feasible fold count explicitly"
                )
        for fold, (train, test) in enumerate(pairs):
            train, test = np.asarray(train), np.asarray(test)
            if train.dtype.kind not in "iu" or test.dtype.kind not in "iu" or train.ndim != 1 or test.ndim != 1:
                raise ValueError("Outer split indices must be one-dimensional integer arrays")
            if not len(train) or not len(test) or np.intersect1d(train, test).size:
                raise ValueError("Outer splits must be nonempty and disjoint")
            if np.unique(train).size != train.size or np.unique(test).size != test.size:
                raise ValueError("Outer splits cannot contain duplicated row positions")
            if min(train.min(), test.min()) < 0 or max(train.max(), test.max()) >= len(data["X"]):
                raise ValueError("Outer split indices are outside the prepared dataset")
            validate_split_boundaries(
                train,
                test,
                n_rows=len(data["X"]),
                groups=data.get("groups"),
                timestamps=data.get("timestamps"),
                label_end_times=data.get("label_end_times"),
            )
            results.append((seed, fold, train, test))
    return results


def split_fingerprint(train: np.ndarray, test: np.ndarray) -> str:
    """Fingerprint ordered indices including the train/test boundary."""
    from benchmarks.tabular import digest

    lengths = np.asarray([len(train), len(test)], dtype="<i8").tobytes()
    return digest(lengths + train.astype("<i8").tobytes() + test.astype("<i8").tobytes())


def source_kind(spec: dict) -> str:
    """Normalize declared data origin without losing simulation distinctions."""
    kind = spec.get("origin_kind", "observational")
    mapping = {
        "observational": "real_world",
        "real_world": "real_world",
        "simulation": "simulation",
        "domain_simulation": "simulation",
        "synthetic": "synthetic",
        "synthetic_diagnostic": "synthetic",
    }
    if not isinstance(kind, str) or kind not in mapping:
        raise ValueError(f"Unknown dataset origin_kind: {kind!r}")
    return mapping[kind]


def _official_split_ids(value: Any) -> list[tuple[int, int, int]]:
    if not isinstance(value, list) or not value:
        raise ValueError("Official split identifiers must be a nonempty list")
    identifiers = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"repeat", "fold", "sample"}:
            raise ValueError("Official split identifiers require exactly repeat, fold and sample")
        if any(type(item[key]) is not int or item[key] < 0 for key in ("repeat", "fold", "sample")):
            raise ValueError("Official split identifiers must be nonnegative integers, not booleans")
        identifiers.append((item["repeat"], item["fold"], item["sample"]))
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Official split identifiers must not be duplicated")
    return identifiers


def _declared_official_ids(spec: dict) -> set[tuple[int, int, int]] | None:
    dimensions = spec.get("source", {}).get("official_split_dimensions")
    if dimensions is None:
        return None
    if not isinstance(dimensions, dict) or any(
        type(dimensions.get(key)) is not int or dimensions[key] < 1 for key in ("repeats", "folds", "samples")
    ):
        raise ValueError("Official source dimensions must be positive integers")
    return {
        (repeat, fold, sample)
        for repeat in range(dimensions["repeats"])
        for fold in range(dimensions["folds"])
        for sample in range(dimensions["samples"])
    }


def _resplit_official_as_iid(registration: dict) -> bool:
    enabled = registration.get("resplit_official_as_iid", False)
    if type(enabled) is not bool:
        raise ValueError("resplit_official_as_iid must be a boolean")
    if enabled:
        plan = registration.get("split_plan")
        if (
            not isinstance(plan, dict)
            or not plan
            or any(
                not isinstance(entry, dict)
                or entry.get("policy") != "iid"
                or type(entry.get("folds")) is not int
                or entry["folds"] != 3
                for entry in plan.values()
            )
        ):
            raise ValueError("resplit_official_as_iid requires every split_plan entry to declare IID folds=3")
        if type(registration.get("max_rows")) is not int or registration["max_rows"] != 0:
            raise ValueError("resplit_official_as_iid requires uncapped raw rows (max_rows=0)")
    return enabled


def _split_plan_count(spec: dict, entry: Any, config: dict, *, resplit: bool = False) -> int:
    if not isinstance(entry, dict) or entry.get("policy") not in ("official", "iid"):
        raise ValueError("Confirmation split_plan entries require an official or iid policy")
    if resplit:
        if spec.get("split_policy", "iid") not in ("official", "iid") or any(
            spec.get(key) is not None for key in ("group_column", "time_column", "label_end_column")
        ):
            raise ValueError("IID resplitting requires official or IID sources without group/time boundaries")
        if type(config.get("max_rows")) is not int or config["max_rows"] != 0:
            raise ValueError("IID resplitting requires uncapped raw rows (max_rows=0)")
    elif entry["policy"] != spec.get("split_policy", "iid"):
        raise ValueError(f"Confirmation split_plan policy differs for family {spec['family_id']!r}")
    if entry["policy"] == "official":
        if set(entry) != {"policy", "official_split_ids"}:
            raise ValueError("Official split_plan entries require only policy and official_split_ids")
        identifiers = _official_split_ids(entry["official_split_ids"])
        declared = _declared_official_ids(spec)
        if declared is not None and not set(identifiers).issubset(declared):
            raise ValueError("Official split_plan identifiers are outside declared source dimensions")
        return len(identifiers)
    if set(entry) != {"policy", "folds"}:
        raise ValueError("IID split_plan entries require only policy and folds")
    folds = entry["folds"]
    if type(folds) is not int or folds < 2:
        raise ValueError("IID split_plan folds must be an integer >= 2, not a boolean")
    if type(config.get("folds")) is not int or config["folds"] != folds:
        raise ValueError("Confirmation IID folds differ from preregistration")
    return folds


def _bind_split_plan(registration: dict, specs: list[dict], config: dict, *, resplit: bool = False) -> None:
    plan = registration["split_plan"]
    families = registration.get("family_ids")
    if (
        not isinstance(families, list)
        or not families
        or any(not isinstance(family, str) or not family for family in families)
        or len(set(families)) != len(families)
    ):
        raise ValueError("Confirmation split_plan requires nonempty, unique family_ids")
    if (
        not isinstance(plan, dict)
        or set(plan) != set(families)
        or set(plan) != {spec["family_id"] for spec in specs}
        or len(specs) != len(families)
    ):
        raise ValueError("Confirmation split_plan must cover exactly the preregistered families")
    if config.get("allow_confirmation") is not True:
        raise ValueError("Confirmation split_plan requires explicit allow_confirmation")
    for settings in (registration, config):
        for key in ("methods", "learners", "seeds"):
            values = settings.get(key)
            if not isinstance(values, list) or not values:
                raise ValueError(f"Confirmation {key} must be a nonempty list")
            if key == "seeds":
                if any(type(value) is not int or not 0 <= value < 2**32 for value in values):
                    raise ValueError("Confirmation seeds must be uint32 integers, not booleans")
            elif any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"Confirmation {key} must contain nonempty strings")
            if len(values) != len(set(values)):
                raise ValueError(f"Confirmation {key} must be unique")
    split_count = sum(_split_plan_count(spec, plan[spec["family_id"]], config, resplit=resplit) for spec in specs)
    expected = split_count * len(registration["seeds"]) * len(registration["methods"]) * len(registration["learners"])
    if type(registration.get("expected_cells")) is not int or registration["expected_cells"] != expected:
        raise ValueError("Confirmation expected_cells differs from the locked split_plan")


def bind_preregistration(suite: dict, specs: list[dict], config: dict, budget: float) -> dict | None:
    """Bind confirmation to locked settings before opening data.

    Notes
    -----
    Optional ``split_plan`` maps every family ID to either
    ``{"policy": "official", "official_split_ids": [{"repeat": 0, "fold": 0, "sample": 0}]}``
    or ``{"policy": "iid", "folds": 3}``. Entries have no additional fields.
    Official IDs select from the full provider payload in registration order;
    IID folds must match the run configuration. ``expected_cells`` counts the
    selected splits across families times seeds, methods and learners.
    Without ``split_plan``, historical all-official validation is unchanged.
    Explicit ``resplit_official_as_iid: true`` requires every family to declare
    IID folds=3 on uncapped rows. Official source policies/dimensions remain
    unchanged and validated, but execution uses new seeded IID splits.
    """
    from benchmarks.tabular import candidate_parameters

    if not any(spec.get("role") == "confirmation" for spec in specs):
        return None
    if not all(spec.get("role") == "confirmation" for spec in specs):
        raise ValueError("Confirmation must run separately from development/stress cohorts")
    registrations = [item["preregistration"] for item in suite["manifests"] if "preregistration" in item]
    if len(registrations) != 1:
        raise ValueError("A dedicated locked preregistration is required for confirmation")
    registration = registrations[0]
    if not isinstance(registration, dict):
        raise ValueError("Confirmation preregistration must be a JSON object")
    if registration.get("status") != "locked_before_confirmation_model_execution":
        raise ValueError("Confirmation preregistration is not locked")
    resplit = _resplit_official_as_iid(registration)
    if "split_plan" in registration:
        _bind_split_plan(registration, specs, config, resplit=resplit)
    if {spec["family_id"] for spec in specs} != set(registration["family_ids"]):
        raise ValueError("Confirmation cohort differs from the preregistered families")
    if config["synthetic"]:
        raise ValueError("Synthetic diagnostics cannot be added to a locked confirmation run")
    if config.get("live_llm") or config.get("llm_ledger"):
        raise ValueError("Live LLM evaluation cannot be mixed into this locked confirmation run")
    expected = {
        **registration["selection_parameters"],
        "seeds": registration["seeds"],
        "max_rows": registration["max_rows"],
        "threads": registration["threads"],
        "case_timeout": registration["case_timeout_seconds"],
    }
    if "memory_limit_mb" in registration:
        memory = registration["memory_limit_mb"]
        if type(memory) is not int or memory < 1 or type(config.get("memory_limit_mb")) is not int:
            raise ValueError("Confirmation memory_limit_mb must be a positive integer")
        expected["memory_limit_mb"] = memory
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Confirmation setting {key!r} differs from preregistration")
    for key in ("methods", "learners"):
        if set(config[key]) != set(registration[key]):
            raise ValueError(f"Confirmation {key} differ from preregistration")
    if budget > registration["total_active_seconds_ceiling"]:
        raise ValueError("Confirmation budget exceeds its preregistered ceiling")
    if registration.get("sample_seed") != 2026:
        raise ValueError("Confirmation sampling seed differs from the runner's declared policy")
    for learner in config["learners"]:
        for key, values in registration["learner_search"][learner].items():
            if key.startswith("selection_estimator_"):
                parameter = key.removeprefix("selection_estimator_")
                if candidate_parameters(learner)[1].get(parameter) != values:
                    raise ValueError(f"Confirmation selection estimator changed for {learner!r}")
                continue
            if [parameters.get(key) for parameters in candidate_parameters(learner)] != values:
                raise ValueError(f"Confirmation learner search changed for {learner!r}")
    return registration


def _confirmation_splits(
    spec: dict, data: dict, metadata: dict, official_splits, registration: dict, config: dict
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    entry = registration["split_plan"][spec["family_id"]]
    resplit = _resplit_official_as_iid(registration)
    count = _split_plan_count(spec, entry, config, resplit=resplit)
    if resplit and any(data.get(key) is not None for key in ("groups", "timestamps", "label_end_times")):
        raise ValueError("IID resplitting cannot use group/time split metadata")
    selected = None
    if entry["policy"] == "official" or (resplit and spec.get("split_policy") == "official"):
        identifiers = _official_split_ids(metadata.get("official_split_ids"))
        if official_splits is None or len(official_splits) != len(identifiers):
            raise ValueError("Official confirmation splits are unavailable or not paired with their identifiers")
        declared = _declared_official_ids(spec)
        if declared is not None and set(identifiers) != declared:
            raise ValueError("Official confirmation source identifiers differ from declared source dimensions")
        if entry["policy"] == "official":
            by_id = dict(zip(identifiers, official_splits, strict=True))
            requested = _official_split_ids(entry["official_split_ids"])
            if not set(requested).issubset(by_id):
                raise ValueError("Official confirmation payload is missing locked split identifiers")
            selected = [by_id[identifier] for identifier in requested]
    elif official_splits is not None or metadata.get("official_split_ids"):
        raise ValueError("IID confirmation cannot use official splits or identifiers")
    splits = _splits(data, spec["task"], config, selected)
    if len(splits) != count * len(config["seeds"]):
        raise ValueError("Confirmation outer split count differs from split_plan")
    _validate_confirmation_class_support(spec, data, [(train, test) for _, _, train, test in splits], registration)
    return splits


def validate_confirmation_payload(spec: dict, data: dict, metadata: dict, official_splits, registration: dict) -> None:
    """Check selected outer splits and class support without changing the payload.

    Notes
    -----
    With ``split_plan``, IID splits use registered seeds/folds and official
    splits are selected by exact identifiers unless whole-cohort IID
    resplitting is explicitly locked. Source official IDs are still validated.
    The return value remains ``None`` for both new and historical registrations.
    """
    _resplit_official_as_iid(registration)
    if "split_plan" in registration:
        entry = registration["split_plan"][spec["family_id"]]
        config = {"seeds": registration["seeds"], "folds": entry.get("folds"), "max_rows": registration.get("max_rows")}
        _confirmation_splits(spec, data, metadata, official_splits, registration, config)
        return
    dimensions = registration["expected_official_split_dimensions_per_task"]
    expected = dimensions["repeats"] * dimensions["folds"] * dimensions["samples"]
    if official_splits is None or len(official_splits) != expected:
        raise ValueError("Confirmation requires every preregistered official split")
    identifiers = metadata.get("official_split_ids", [])
    observed = {(item["repeat"], item["fold"], item["sample"]) for item in identifiers}
    declared = {
        (repeat, fold, sample)
        for repeat in range(dimensions["repeats"])
        for fold in range(dimensions["folds"])
        for sample in range(dimensions["samples"])
    }
    if observed != declared or len(identifiers) != expected:
        raise ValueError("Official confirmation repeat/fold/sample identifiers differ")
    _validate_confirmation_class_support(spec, data, official_splits, registration)


def _validate_confirmation_class_support(spec: dict, data: dict, splits, registration: dict) -> None:
    if spec["task"] == "classification":
        margins = registration["classification_admission_margin"]
        y = data["y"]
        classes = pd.unique(y)
        if y.value_counts().min() < margins["min_total"]:
            raise ValueError("Confirmation payload fails preregistered overall class support")
        for train, test in splits:
            for indices, key in ((train, "min_outer_train"), (test, "min_outer_test")):
                if y.iloc[indices].value_counts().reindex(classes, fill_value=0).min() < margins[key]:
                    raise ValueError(f"Confirmation payload fails preregistered {key} class support")


def execute(args) -> int:
    """Run new cells or resume fingerprint-identical cells from an existing run."""
    from benchmarks import tabular

    output = args.output.resolve()
    previous = {}
    if args.resume:
        previous = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if previous.get("source_changed"):
            raise ValueError("Cannot resume a run invalidated by source changes")
    old_config = previous.get("config", {})
    defaults = {
        **DEFAULTS,
        "methods": list(tabular.DEFAULT_METHODS),
        "learners": list(tabular.LEARNERS),
        "suite": str(tabular.SUITE_PATH),
    }
    config: dict[str, Any] = {
        key: getattr(args, key, None) if getattr(args, key, None) is not None else old_config.get(key, default)
        for key, default in defaults.items()
    }
    suite_paths = config["suite"] if isinstance(config["suite"], list) else [config["suite"]]
    config["suite"] = [str(Path(path).resolve()) for path in suite_paths]
    if config["llm_ledger"] is not None:
        config["llm_ledger"] = str(Path(config["llm_ledger"]).resolve())
    budget = args.total_seconds if args.total_seconds is not None else previous.get("budget_seconds", 14400.0)
    validate_config(config, budget)
    suites, specs = [], []
    for path in config["suite"]:
        loaded_suite, is_legacy = _load_suite(Path(path), config["allow_confirmation"])
        suites.append(loaded_suite)
        for spec in loaded_suite["datasets"]:
            specs.append({**spec, "_legacy_revision": loaded_suite["revision"]} if is_legacy else spec)
    suite = {"name": "+".join(item["name"] for item in suites), "manifests": suites, "datasets": specs}
    legacy = False
    if not specs or any(
        not isinstance(spec.get("name"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", spec["name"])
        for spec in specs
    ):
        raise ValueError("Suite datasets must have nonempty, path-safe identifiers")
    if config["datasets"]:
        unknown = set(config["datasets"]) - {spec["name"] for spec in specs}
        if unknown:
            raise ValueError(f"Unknown datasets: {sorted(unknown)}")
        specs = [spec for spec in specs if spec["name"] in config["datasets"]]
    if len({spec["name"] for spec in specs}) != len(specs):
        raise ValueError("Combined suites contain duplicated dataset identifiers")
    if len({spec.get("family_id", spec["name"]) for spec in specs}) != len(specs):
        raise ValueError("Combined primary suites contain repeated origin families; run variants separately")
    if args.list_datasets:
        for spec in specs:
            logger.info(
                "%s: family=%s task=%s role=%s provider=%s rights=%s",
                spec["name"],
                spec.get("family_id"),
                spec["task"],
                spec.get("role"),
                spec.get("provider", "legacy-hf"),
                spec.get("rights", spec.get("rights_status")),
            )
        return 0
    preregistration = bind_preregistration(suite, specs, config, budget)
    resplit = preregistration is not None and _resplit_official_as_iid(preregistration)
    if any(method.startswith("llm-") for method in config["methods"]):
        from benchmarks.llm_evaluation import CopilotPlanProvider, EvaluationBudget, preflight

        if not config["live_llm"] or not config["llm_ledger"]:
            raise ValueError("LLM methods require --live-llm and an explicit shared --llm-ledger")
        readiness = preflight(
            CopilotPlanProvider(),
            live_approved=True,
            budget=EvaluationBudget(
                max_calls=100, max_tokens=4096, require_hard_token_cap=not config["allow_soft_llm_tokens"]
            ),
        )
        if not readiness.allowed:
            raise ValueError(f"LLM preflight blocked: {', '.join(readiness.blockers)}")
    identity = {
        "protocol_version": tabular.PROTOCOL_VERSION,
        "source_sha256": tabular.source_fingerprint(),
        "suite": suite,
        "config": config,
        "versions": environment_versions(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    state = RunState(output, identity, resume=args.resume)
    if args.resume:
        if tabular.digest((output / "source.zip").read_bytes()) != previous["source_archive_sha256"]:
            raise ResumeMismatchError("Cannot resume: source archive changed")
        source_archive_hash = previous["source_archive_sha256"]
    else:
        source_archive_hash = archive_source(output, tabular.ROOT, tabular.source_files(), suite)
    started = time.perf_counter()
    prior_elapsed = state.elapsed_seconds
    deadline = started + max(0.0, budget - prior_elapsed)
    manifest = {
        **previous,
        "protocol_version": tabular.PROTOCOL_VERSION,
        "run_id": state.run_id,
        "suite": suite,
        "config": config,
        "budget_seconds": budget,
        "started_utc": previous.get("started_utc", datetime.now(timezone.utc).isoformat()),
        "versions": identity["versions"],
        "python": identity["python"],
        "platform": identity["platform"],
        "source_sha256": identity["source_sha256"],
        "source_archive_sha256": source_archive_hash,
        "datasets": [],
        "splits": [],
        "resumed": bool(args.resume),
        "evaluation_unit": "dataset-split-method-learner",
        "preregistration": preregistration,
    }
    if not args.resume:
        atomic_json(output / "manifest.json", manifest)
    ledger = ExposureLedger(tabular.ROOT / "benchmarks" / ".cache" / "exposure.sqlite")
    requested = [(spec, source_kind(spec)) for spec in specs]
    if config["synthetic"]:
        requested.extend(
            (
                {
                    "name": name,
                    "family_id": f"synthetic:{name}",
                    "role": "development",
                    "task": task,
                    "provider": "diagnostic",
                },
                "synthetic",
            )
            for name, task in (("polynomial_regression", "regression"), ("xor_classification", "classification"))
        )
    jobs = []
    prepared_dir = output / "prepared"
    prepared_dir.mkdir(exist_ok=True)
    for spec, source in requested:
        name = spec["name"]
        error = None
        plan = (
            preregistration["split_plan"][spec["family_id"]]
            if preregistration and "split_plan" in preregistration
            else None
        )
        selection_metadata: dict[str, Any] = {}
        if plan is not None:
            selection_metadata["selected_split_count"] = _split_plan_count(spec, plan, config, resplit=resplit)
            if resplit:
                selection_metadata.update(used_split_policy="iid", selected_official_split_ids=[])
            elif plan["policy"] == "official":
                selection_metadata["selected_official_split_ids"] = [dict(item) for item in plan["official_split_ids"]]
        metadata = {**spec, "source": source, **selection_metadata}
        splits: list[tuple[int, int, np.ndarray | None, np.ndarray | None]]
        logger.info("Preparing %s", name)
        try:
            if time.perf_counter() >= deadline:
                raise TimeoutError("Experiment budget exhausted before preparation")
            ledger.check(spec.get("family_id", name), state.run_id, spec.get("role", "development"))
            data, metadata, official = _prepare_dataset(spec, source, suite, legacy, config, tabular.ROOT)
            metadata = {**metadata, **selection_metadata}
            if preregistration is not None and plan is not None:
                outer_splits = _confirmation_splits(spec, data, metadata, official, preregistration, config)
            else:
                if preregistration is not None:
                    validate_confirmation_payload(spec, data, metadata, official, preregistration)
                outer_splits = _splits(data, spec["task"], config, official)
            splits = [(seed, fold, train, test) for seed, fold, train, test in outer_splits]
            metadata["outer_split_count"] = len(splits)
            state.validate_dataset(name, metadata)
            if not args.prepare_only:
                ledger.admit(spec.get("family_id", name), state.run_id, spec.get("role", "development"))
            data_path = prepared_dir / f"{name}.pkl"
            prepared_bytes = pickle.dumps(data)
            data_path.write_bytes(prepared_bytes)
            prepared_sha256 = tabular.digest(prepared_bytes)
        except ResumeMismatchError:
            raise
        except Exception as exc:
            logger.exception("Dataset preparation failed for %s", name)
            error = f"{type(exc).__name__}: {exc}"
            metadata = {**(metadata if plan is not None else {**spec, "source": source}), "error": error}
            data_path = None
            prepared_sha256 = None
            failed_fold_count = config["folds"]
            if plan is not None:
                failed_fold_count = _split_plan_count(spec, plan, config, resplit=resplit)
            elif preregistration is not None:
                dimensions = preregistration["expected_official_split_dimensions_per_task"]
                failed_fold_count = dimensions["repeats"] * dimensions["folds"] * dimensions["samples"]
            splits = [(seed, fold, None, None) for seed in config["seeds"] for fold in range(failed_fold_count)]
        manifest["datasets"].append(metadata)
        for seed, fold, train, test in splits:
            split_hash = split_fingerprint(train, test) if train is not None and test is not None else "unavailable"
            split_metadata = {"dataset": name, "seed": seed, "fold": fold, "sha256": split_hash}
            if plan is not None and plan["policy"] == "official":
                split_metadata["official_split_id"] = dict(plan["official_split_ids"][fold])
            manifest["splits"].append(split_metadata)
            for method in config["methods"]:
                for learner in config["learners"]:
                    cell_id = state.cell_id(name, seed, fold, method, learner)
                    cell_hash = fingerprint(
                        {"identity": state.identity_hash, "data": metadata, "split": split_hash, "cell": cell_id}
                    )
                    payload = {
                        "data_path": None if data_path is None else str(data_path),
                        "prepared_sha256": prepared_sha256,
                        "dataset": name,
                        "family_id": spec.get("family_id", name),
                        "role": spec.get("role", "development"),
                        "source": metadata.get("source", source),
                        "task": spec["task"],
                        "method": method,
                        "seed": seed,
                        "fold": fold,
                        "train_idx": train,
                        "test_idx": test,
                        "split_sha256": split_hash,
                        "expected_cells": len(splits),
                        "config": {**config, "learners": [learner], "seed": seed},
                    }
                    jobs.append((cell_id, cell_hash, payload, error))
    # Spread early coverage across datasets rather than exhausting one dataset first.
    jobs.sort(key=lambda job: (job[2]["fold"], job[2]["seed"], fingerprint(job[0])))
    manifest["planned_cells"] = len(jobs)
    if preregistration is not None and len(jobs) != preregistration["expected_cells"]:
        raise ValueError("Planned confirmation cell count differs from preregistration")
    interrupted = False
    source_changed = False
    for cell_id, cell_hash, payload, error in jobs:
        if args.prepare_only:
            break
        if state.reusable(cell_id, cell_hash, retry_failed=args.retry_failed):
            continue
        remaining = deadline - time.perf_counter()
        if not source_changed and tabular.source_fingerprint() != manifest["source_sha256"]:
            source_changed = True
            manifest["source_changed"] = True
        if error is not None:
            row = tabular.failure_rows(payload, "dataset_error", error, 0.0)[0]
        elif source_changed:
            row = tabular.failure_rows(payload, "source_changed", "Source code changed during the experiment", 0.0)[0]
        elif interrupted or remaining <= 0:
            row = tabular.failure_rows(
                payload,
                "interrupted" if interrupted else "budget_exhausted",
                "Experiment interrupted" if interrupted else "Experiment active-time budget exhausted",
                0.0,
            )[0]
        else:
            logger.info(
                "%s seed=%s fold=%s %s/%s",
                payload["dataset"],
                payload["seed"],
                payload["fold"],
                payload["method"],
                payload["config"]["learners"][0],
            )
            directory = state.next_attempt(cell_id)
            try:
                result = tabular.run_isolated(payload, directory, min(config["case_timeout"], remaining))
                if len(result) != 1:
                    raise ValueError("An isolated evaluation cell must return exactly one learner result")
                row = result[0]
            except KeyboardInterrupt:
                interrupted = True
                row = tabular.failure_rows(payload, "interrupted", "Experiment interrupted", 0.0)[0]
            row["log"] = str((directory / "worker.log").relative_to(output))
            if tabular.source_fingerprint() != manifest["source_sha256"]:
                source_changed = True
                manifest["source_changed"] = True
        state.elapsed_seconds = prior_elapsed + time.perf_counter() - started
        state.record(cell_id, cell_hash, row)
        manifest["active_seconds"] = state.elapsed_seconds
        atomic_json(output / "manifest.json", manifest)
        tabular.write_report(output, manifest, list(state.rows.values()))
    state.elapsed_seconds = prior_elapsed + time.perf_counter() - started
    state.save()
    manifest["active_seconds"] = state.elapsed_seconds
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["source_sha256_at_finish"] = tabular.source_fingerprint()
    manifest["source_changed"] = source_changed or manifest["source_sha256_at_finish"] != manifest["source_sha256"]
    atomic_json(output / "manifest.json", manifest)
    if not args.prepare_only:
        tabular.write_report(output, manifest, list(state.rows.values()))
        archive_results(output / "results.json")
    logger.info("Saved %s: %d/%d evaluation cells", output, len(state.rows), len(jobs))
    return (
        0
        if not manifest["source_changed"]
        and all(row["status"] == "success" for row in state.rows.values())
        and all("error" not in item for item in manifest["datasets"])
        else 1
    )


def validate_config(config: dict, budget: float) -> None:
    """Reject invalid experiment controls before creating output state."""
    if config["folds"] < 2 or config["threads"] < 1 or config["max_rows"] < 0:
        raise ValueError("folds >= 2, threads >= 1 and max_rows >= 0 are required")
    if (
        min(
            config["max_added_features"],
            config["max_candidates"],
            config["stability_repeats"],
            config["memory_limit_mb"],
        )
        < 1
    ):
        raise ValueError("Feature, repeat and memory limits must be positive")
    if not np.isfinite([budget, config["case_timeout"]]).all() or min(budget, config["case_timeout"]) <= 0:
        raise ValueError("Time limits must be finite and positive")
    for key in ("methods", "learners", "seeds"):
        if not config[key] or len(config[key]) != len(set(config[key])):
            raise ValueError(f"{key} must be nonempty and unique")
    if "baseline" not in config["methods"]:
        from benchmarks.tabular import PLAN_VALIDATION_METHODS, ROBUST_METHODS

        methods = set(config["methods"])
        robust_control = "baseline-robust" in methods and methods.issubset(ROBUST_METHODS)
        plan_control = "baseline-discovery-robust" in methods and methods.issubset(PLAN_VALIDATION_METHODS)
        if not (robust_control or plan_control):
            raise ValueError("A paired baseline is required; specialized controls must match their method family")
    if any(not 0 <= seed < 2**32 for seed in config["seeds"]):
        raise ValueError("Seeds must be in [0, 2**32)")
