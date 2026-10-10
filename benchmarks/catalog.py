"""Fail-closed dataset admission and raw-data providers for tabular benchmarks.

Manifests declare roles, rights and prediction-time schemas before acquisition.
``load_suite`` returns ordinary dictionaries; pass its dataset entries to
``load_dataset``. Inventory entries are valid metadata, never download approval.
Only raw source bytes, source metadata, split files and checksum pins are cached.
No preprocessing statistics or engineered features are fitted here.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib
import io
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

_REGISTRY = Path(__file__).with_name("families.json")
_ROLES = {"development", "confirmation", "diagnostic", "inventory"}
_EXPOSURES = {"exposed", "loader_only", "metadata_only", "unknown", "audited_unexposed"}
_PROVIDERS = {"hf_csv", "uci", "openml_task", "local_csv", "kaggle_local"}
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,99}\Z")
_WINDOWS_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_CSV_PARSER_DEFAULTS: dict[str, Any] = {
    "encoding": "utf-8-sig",
    "separator": ",",
    "na_values": ["?"],
    "keep_default_na": True,
    "skipinitialspace": False,
}


class AdmissionError(ValueError):
    """A manifest, schema or requested acquisition violates admission policy."""


class AcquisitionError(RuntimeError):
    """An admitted source could not be acquired or its checksum changed."""


@dataclass
class LoadedDataset:
    """Raw aligned features, labels, provenance and optional split restrictions."""

    X: pd.DataFrame
    y: pd.Series
    metadata: dict[str, Any]
    official_splits: list[tuple[np.ndarray, np.ndarray]] | None = None
    groups: pd.Series | None = None
    timestamps: pd.Series | None = None
    label_end_times: pd.Series | None = None


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise AdmissionError(message)


def _identifier(value: Any, field: str) -> str:
    _require(
        isinstance(value, str) and _SAFE_ID.fullmatch(value) and value not in _WINDOWS_RESERVED,
        f"{field} must be a safe lowercase identifier (letters, digits, hyphens, underscores; no paths)",
    )
    return str(value)


def _strings(value: Any, field: str, *, nonempty: bool = False) -> list[str]:
    _require(
        isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value),
        f"{field} must be a list of nonempty strings",
    )
    _require(len(value) == len(set(value)), f"{field} contains duplicates")
    _require(not nonempty or value, f"{field} must not be empty")
    return list(value)


def _positive_int(value: Any, field: str) -> None:
    _require(type(value) is int and value > 0, f"{field} must be a positive integer")


def _alias(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _path(value: str, base: Path) -> Path:
    path = Path(value.replace("\\", "/"))
    return (path if path.is_absolute() else base / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AdmissionError(f"Cannot read JSON manifest {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return dict(value)


def _families(extra: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for family in [*_read_json(_REGISTRY)["families"], *extra]:
        _require(isinstance(family, dict), "Each family must be an object")
        fid = _identifier(family.get("family_id"), "family_id")
        _require(
            isinstance(family.get("prior_exposure"), str) and family["prior_exposure"] in _EXPOSURES,
            f"{fid}: invalid prior_exposure",
        )
        _require(isinstance(family.get("role"), str) and family["role"] in _ROLES, f"{fid}: invalid family role")
        _strings(family.get("aliases", []), f"{fid}.aliases")
        _strings(family.get("source_ids", []), f"{fid}.source_ids")
        _require(
            isinstance(family.get("upstream_origin"), str) and family["upstream_origin"].strip(),
            f"{fid}: upstream_origin is required",
        )
        reservation = family.get("reservation")
        if reservation is not None:
            _require(
                isinstance(reservation, dict)
                and isinstance(reservation.get("status"), str)
                and reservation.get("status") in {"active", "released"}
                and reservation.get("role") == "confirmation_candidate",
                f"{fid}: invalid confirmation-candidate reservation",
            )
            _identifier(reservation.get("id"), f"{fid}.reservation.id")
        if family["prior_exposure"] == "audited_unexposed":
            audit = family.get("exposure_audit", {})
            _require(
                isinstance(audit, dict) and audit.get("status") == "complete" and audit.get("evidence"),
                f"{fid}: audited_unexposed requires a completed exposure audit with evidence",
            )
            _strings(audit["evidence"], f"{fid}.exposure_audit.evidence", nonempty=True)
        if family["role"] == "confirmation":
            _require(
                family["prior_exposure"] == "audited_unexposed",
                f"{fid}: confirmation families must be audited_unexposed",
            )
        if fid in registry:
            old = registry[fid]
            covered_identity = (
                family["upstream_origin"] == old["upstream_origin"]
                and set(family.get("aliases", [])).issubset(old.get("aliases", []))
                and set(family.get("source_ids", [])).issubset(old.get("source_ids", []))
            )
            historical_inventory = family["role"] == "inventory" and old["role"] != "inventory"
            later_exposure = old["prior_exposure"] == "exposed"
            if covered_identity and (historical_inventory or later_exposure):
                # Historical manifests cannot erase a later admission or recorded exposure.
                continue
            _require(
                family["prior_exposure"] == old["prior_exposure"] and family["role"] == old["role"],
                f"{fid}: cannot override registry exposure or role",
            )
            _require(
                set(old.get("aliases", [])).issubset(family.get("aliases", []))
                and set(old.get("source_ids", [])).issubset(family.get("source_ids", [])),
                f"{fid}: cannot erase historical aliases or source identities",
            )
            _require(family["upstream_origin"] == old["upstream_origin"], f"{fid}: cannot override registry origin")
            if old.get("reservation") is not None:
                family = {**family, "reservation": copy.deepcopy(old["reservation"])}
        registry[fid] = copy.deepcopy(family)
    identities: dict[str, str] = {}
    for fid, family in registry.items():
        for identity in [fid, *family.get("aliases", []), *family.get("source_ids", [])]:
            key = _alias(identity)
            _require(key not in identities or identities[key] == fid, f"Ambiguous family alias/source: {identity}")
            identities[key] = fid
    return registry


def _validate_rights(rights: Any, name: str, *, runnable: bool) -> None:
    _require(isinstance(rights, dict), f"{name}: declared rights are required")
    _require(
        isinstance(rights.get("status"), str) and rights["status"] in {"approved", "pending_review", "restricted"},
        f"{name}: invalid rights status",
    )
    _require(isinstance(rights.get("license"), str) and rights["license"].strip(), f"{name}: license is required")
    _require(
        isinstance(rights.get("access"), str) and rights["access"] in {"public", "local_only", "restricted"},
        f"{name}: invalid access",
    )
    if runnable:
        _require(rights["status"] == "approved", f"{name}: pending/restricted rights block admission")
        _require(rights["access"] != "restricted", f"{name}: restricted access blocks acquisition")
        _require(
            rights["license"].strip().lower() not in {"unknown", "pending", "unspecified"},
            f"{name}: unknown license",
        )
        _require(
            isinstance(rights.get("approval_basis"), str) and rights["approval_basis"].strip(),
            f"{name}: approved rights need an approval_basis",
        )
        _strings(rights.get("evidence"), f"{name}.rights.evidence", nonempty=True)


def _validate_source(spec: dict[str, Any], *, runnable: bool) -> None:
    name, source, provider = spec["name"], spec["source"], spec["provider"]
    _require(isinstance(source, dict), f"{name}: source must be an object")
    _require(isinstance(spec.get("target"), str) and spec["target"].strip(), f"{name}: explicit target is required")
    _strings(spec["categorical"], f"{name}.categorical")
    _strings(spec["drop_columns"], f"{name}.drop_columns")
    transform = spec.setdefault("target_transform", "identity")
    _require(
        isinstance(transform, str) and transform in {"identity", "log1p", "one_hot"},
        f"{name}: unsupported target_transform",
    )
    if transform == "log1p":
        _require(spec["task"] == "regression", f"{name}: log1p requires regression")
    if not runnable:
        return
    _require(source.get("schema_reviewed") is True, f"{name}: source schema must be explicitly reviewed")
    columns = _strings(source.get("columns"), f"{name}.source.columns", nonempty=True)
    targets = _strings(source.get("target_columns"), f"{name}.source.target_columns", nonempty=True)
    _require(set(targets).issubset(columns), f"{name}: native target columns missing from schema")
    override = source.get("target_role_override")
    if override is not None:
        _require(
            provider == "uci" and isinstance(override, dict) and override.get("status") == "reviewed",
            f"{name}: target_role_override requires an explicitly reviewed UCI mapping",
        )
        metadata_targets = _strings(
            override.get("metadata_target_columns"),
            f"{name}.target_role_override.metadata_target_columns",
            nonempty=True,
        )
        _require(
            override.get("selected_target") == spec["target"]
            and set(targets) == set(metadata_targets) | {spec["target"]},
            f"{name}: override must declare the selected target and preserve all metadata targets for exclusion",
        )
        _require(
            isinstance(override.get("metadata_selected_target_role"), str)
            and override["metadata_selected_target_role"] in {"Feature", "Target"},
            f"{name}: override must pin the original selected-column role",
        )
        _require(
            (spec["target"] in metadata_targets) == (override["metadata_selected_target_role"] == "Target"),
            f"{name}: overridden metadata target list and selected-column role disagree",
        )
        _require(
            isinstance(override.get("rationale"), str) and override["rationale"].strip(),
            f"{name}: override needs an explicit review rationale",
        )
        _strings(override.get("evidence"), f"{name}.target_role_override.evidence", nonempty=True)
    excluded = set(spec["drop_columns"])
    if transform == "one_hot":
        _require(spec["task"] == "classification" and len(targets) > 1, f"{name}: one_hot requires multiple labels")
        _require(spec["target"] not in columns, f"{name}: derived target must not shadow a source column")
        _require(set(targets).issubset(excluded), f"{name}: drop all one-hot target indicators")
    else:
        _require(spec["target"] in targets, f"{name}: target is not a reviewed native target")
        _require(set(targets) - {spec["target"]} <= excluded, f"{name}: exclude all other native target columns")
        _require(spec["target"] not in excluded, f"{name}: selected target must not be in drop_columns")
    _require(excluded.issubset(columns), f"{name}: unknown drop_columns")
    for key in ("group_column", "time_column", "label_end_column"):
        _require(spec.get(key) is None or isinstance(spec[key], str) and spec[key].strip(), f"{name}: invalid {key}")
    _require(
        not spec.get("label_end_column") or spec.get("time_column"),
        f"{name}: label_end_column requires time_column",
    )
    side_columns = {
        spec[key] for key in ("group_column", "time_column", "label_end_column") if spec.get(key) is not None
    }
    _require(side_columns.issubset(columns), f"{name}: unknown group/time/label-end column")
    _require(not side_columns.intersection(targets), f"{name}: target cannot be a group/time/label-end key")
    features = set(columns) - set(targets) - excluded - side_columns
    _require(features, f"{name}: no reviewed feature columns remain")
    _require(set(spec["categorical"]).issubset(features), f"{name}: categorical columns must be predictors")
    if spec.get("time_hour_column") is not None:
        _require(
            isinstance(spec["time_hour_column"], str)
            and spec["time_hour_column"] in features
            and spec.get("time_column"),
            f"{name}: time_hour_column requires time_column and a reviewed predictor column",
        )
    policy = spec.get("split_policy", "iid")
    _require(
        isinstance(policy, str) and policy in {"iid", "official", "grouped", "temporal"},
        f"{name}: invalid split_policy",
    )
    _require(policy != "grouped" or spec.get("group_column"), f"{name}: grouped policy requires group_column")
    _require(policy != "temporal" or spec.get("time_column"), f"{name}: temporal policy requires time_column")
    _require(not spec.get("time_column") or spec.get("time_format"), f"{name}: explicit time_format is required")
    _require(
        not spec.get("group_column") or policy in {"grouped", "temporal", "official"},
        f"{name}: group key needs grouped/temporal policy",
    )
    _require(
        not spec.get("time_column") or policy in {"temporal", "official"}, f"{name}: time key needs temporal policy"
    )
    _require(
        isinstance(spec.get("prediction_time"), str) and spec["prediction_time"].strip(),
        f"{name}: prediction_time availability declaration is required",
    )
    sha = source.get("sha256")
    for checksum_field in ("sha256", "splits_sha256"):
        checksum = source.get(checksum_field)
        if checksum is not None:
            _require(
                isinstance(checksum, str) and _SHA256.fullmatch(checksum),
                f"{name}: invalid source {checksum_field} SHA-256",
            )
    if source.get("expected_rows") is not None:
        _positive_int(source["expected_rows"], f"{name}.expected_rows")
    if provider in {"hf_csv", "uci", "openml_task"}:
        _require(spec["rights"]["access"] == "public", f"{name}: remote providers require public acquisition rights")
    if provider == "hf_csv":
        _require(
            isinstance(source.get("revision"), str) and _REVISION.fullmatch(source["revision"]),
            f"{name}: HF revision must be an immutable 40-character commit",
        )
        _require(sha, f"{name}: HF payload SHA-256 is required, not only a code revision")
        _require(
            isinstance(source.get("repo_id"), str) and re.fullmatch(r"[\w.-]+/[\w.-]+", source["repo_id"]),
            f"{name}: invalid HF repo_id",
        )
        remote_path = source.get("path", "")
        _require(
            isinstance(remote_path, str)
            and remote_path
            and "\\" not in remote_path
            and all(part not in {"", ".", ".."} for part in remote_path.split("/")),
            f"{name}: unsafe HF source path",
        )
    elif provider == "uci":
        _positive_int(source.get("uci_id"), f"{name}.uci_id")
        expected_url = f"https://archive.ics.uci.edu/static/public/{source['uci_id']}/data.csv"
        _require(source.get("data_url") == expected_url, f"{name}: use the reviewed official UCI normalized CSV URL")
    elif provider == "openml_task":
        for field in ("task_id", "dataset_id", "dataset_version"):
            _positive_int(source.get(field), f"{name}.{field}")
        _require(policy == "official", f"{name}: OpenML tasks must preserve official splits")
        _require(transform != "one_hot", f"{name}: OpenML target must match its official task")
        if source.get("md5_checksum") is not None:
            _require(
                isinstance(source["md5_checksum"], str) and re.fullmatch(r"[0-9a-f]{32}", source["md5_checksum"]),
                f"{name}: invalid upstream payload MD5 pin",
            )
        dimensions = source.get("official_split_dimensions")
        if dimensions is not None:
            _require(isinstance(dimensions, dict), f"{name}: official_split_dimensions must be an object")
            for dimension in ("repeats", "folds", "samples"):
                _positive_int(dimensions.get(dimension), f"{name}.official_split_dimensions.{dimension}")
    else:
        _require(sha, f"{name}: local files require source SHA-256")
        _require(isinstance(source.get("path"), str) and source["path"], f"{name}: local path is required")
        _require(spec["rights"]["access"] == "local_only", f"{name}: local files require local_only rights")
    _require(provider == "openml_task" or policy != "official", f"{name}: official splits require an OpenML task")
    support = spec.get("class_support")
    if support is not None:
        _require(
            isinstance(support, dict) and spec["task"] == "classification" and policy == "official",
            f"{name}: class_support requires an official classification task",
        )
        _strings(support.get("labels"), f"{name}.class_support.labels", nonempty=True)
        for field in ("min_total", "min_outer_train", "min_outer_test"):
            _positive_int(support.get(field), f"{name}.class_support.{field}")


def validate_manifest(
    manifest: dict[str, Any],
    *,
    manifest_path: Path | None = None,
    allow_confirmation: bool = False,
    require_admission: bool = True,
) -> dict[str, Any]:
    """Validate and normalize metadata without acquiring any dataset.

    Parameters
    ----------
    manifest : dict
        Suite with name, datasets and optional families/family_registry.
        Dataset defaults may be declared in ``dataset_defaults``.
    manifest_path : Path, optional
        Manifest filename, used to resolve registry and local data paths.
    allow_confirmation : bool, default=False
        Explicitly unlock only audited-unexposed, approved-rights confirmation.
    require_admission : bool, default=True
        Require acquisition approval for non-inventory entries. ``load_suite``
        disables this to list pending/restricted development entries. Approved
        entries still require complete schemas; confirmation never bypasses
        its rights, audit or explicit-unlock gates.

    Returns
    -------
    dict
        Normalized suite with a list of resolved family records. Dataset entries
        contain private admission context for subsequent ``load_dataset`` calls.

    Notes
    -----
    Recorded registry exposure is authoritative. Older development/inventory
    declarations retain their original value as ``declared_prior_exposure`` and
    normalize to the current exposed state. An exposed family cannot be opened
    as fresh confirmation, even with ``allow_confirmation=True``; its historical
    manifest remains an immutable record rather than new acquisition authority.
    Active candidate reservations are authoritative registry state too. Metadata
    remains listable, but acquisition is blocked until an explicit registry
    release following the parent algorithm/protocol freeze.
    """
    _require(isinstance(manifest, dict), "Suite must be an object")
    suite = copy.deepcopy(manifest)
    _identifier(suite.get("name"), "suite.name")
    _require(type(allow_confirmation) is bool, "allow_confirmation must be boolean")
    _require(type(require_admission) is bool, "require_admission must be boolean")
    base = manifest_path.resolve().parent if manifest_path is not None else Path.cwd()
    extra = suite.get("families", [])
    _require(isinstance(extra, list), "families must be a list")
    if suite.get("family_registry"):
        extra = [*_read_json(_path(suite["family_registry"], base))["families"], *extra]
    registry = _families(extra)
    aliases = {
        _alias(identity): fid
        for fid, family in registry.items()
        for identity in [fid, *family.get("aliases", []), *family.get("source_ids", [])]
    }
    datasets = suite.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise AdmissionError("datasets must be a nonempty list")
    defaults = suite.get("dataset_defaults", {})
    _require(isinstance(defaults, dict), "dataset_defaults must be an object")
    seen_names: set[str] = set()
    seen_families: set[str] = set()
    normalized = []
    for entry in datasets:
        _require(isinstance(entry, dict), "dataset entries must be objects")
        spec = {**copy.deepcopy(defaults), **entry}
        name = _identifier(spec.get("name"), "dataset.name")
        fid = _identifier(spec.get("family_id"), f"{name}.family_id")
        _require(name not in seen_names, f"Duplicate dataset name: {name}")
        seen_names.add(name)
        _require(fid in registry, f"{name}: unregistered family {fid}")
        family = registry[fid]
        _require(isinstance(spec.get("role"), str) and spec["role"] in _ROLES, f"{name}: invalid role")
        if spec["role"] == "confirmation":
            _require(
                family["prior_exposure"] != "exposed",
                f"{name}: family is already exposed; historical confirmation cannot be reused as fresh confirmation",
            )
        if (
            family["prior_exposure"] == "exposed"
            and isinstance(spec.get("prior_exposure"), str)
            and spec["prior_exposure"] in _EXPOSURES - {"exposed"}
        ):
            spec["declared_prior_exposure"] = spec["prior_exposure"]
            spec["prior_exposure"] = "exposed"
        if (
            spec["role"] == "inventory"
            and spec.get("prior_exposure") == "metadata_only"
            and family["prior_exposure"] == "audited_unexposed"
        ):
            spec["inventory_snapshot_prior_exposure"] = "metadata_only"
            spec["prior_exposure"] = family["prior_exposure"]
        _require(spec.get("prior_exposure") == family["prior_exposure"], f"{name}: family prior_exposure mismatch")
        _require(
            spec["role"] in {"inventory", family["role"]}
            or spec["role"] == "diagnostic"
            and family["role"] == "development",
            f"{name}: family role mismatch",
        )
        _require(
            isinstance(spec.get("task"), str) and spec["task"] in {"classification", "regression"},
            f"{name}: invalid task",
        )
        _require(
            isinstance(spec.get("provider"), str) and spec["provider"] in _PROVIDERS,
            f"{name}: invalid provider",
        )
        _require(
            isinstance(spec.get("origin_kind"), str)
            and spec["origin_kind"] in {"observational", "simulation", "synthetic_diagnostic", "unknown"},
            f"{name}: invalid origin_kind",
        )
        runnable = spec["role"] != "inventory"
        _validate_rights(spec.get("rights"), name, runnable=False)
        admission_required = runnable and (
            require_admission or spec["rights"]["status"] == "approved" or spec["role"] == "confirmation"
        )
        _validate_rights(spec["rights"], name, runnable=admission_required)
        spec.setdefault("categorical", [])
        spec.setdefault("drop_columns", [])
        _require("source" in spec, f"{name}: source is required")
        _validate_source(spec, runnable=admission_required)
        if runnable:
            _require(
                fid not in seen_families,
                f"Duplicate primary family: {fid}; use separate diagnostic suites for variants",
            )
            seen_families.add(fid)
        if spec["role"] == "confirmation":
            _require(allow_confirmation, f"{name}: confirmation is locked; explicit allow_confirmation required")
            _require(family["prior_exposure"] == "audited_unexposed", f"{name}: confirmation exposure audit required")
        source = spec["source"]
        identities = [name, *_strings(spec.get("aliases", []), f"{name}.aliases")]
        for field, prefix in (("uci_id", "uci"), ("dataset_id", "openml"), ("task_id", "openml-task")):
            if source.get(field) is not None:
                identities.append(f"{prefix}:{source[field]}")
        if source.get("identity"):
            identities.append(source["identity"])
        if spec["provider"] == "hf_csv" and source.get("repo_id"):
            identities.append(f"hf:{source['repo_id']}")
            if source["repo_id"].lower() == "inria-soda/tabular-benchmark" and source.get("path"):
                config = source["path"].removesuffix(".csv").replace("/", "_")
                identities.extend([config, f"inria:{config}"])
        for identity in identities:
            _require(
                _alias(identity) not in aliases or aliases[_alias(identity)] == fid,
                f"{name}: alias/source {identity!r} belongs to another family; exposure cannot be relabeled",
            )
        if spec["provider"] in {"local_csv", "kaggle_local"} and isinstance(source.get("path"), str) and source["path"]:
            source["path"] = str(_path(source["path"], base))
        spec["_family"] = copy.deepcopy(family)
        spec["_allow_confirmation"] = allow_confirmation
        normalized.append(spec)
    suite["datasets"] = normalized
    used = {spec["family_id"] for spec in normalized}
    suite["families"] = [copy.deepcopy(family) for fid, family in registry.items() if fid in used]
    suite["family_count"] = len(used)
    suite["variant_policy"] = "one_per_family"
    suite["manifest_path"] = str(manifest_path.resolve()) if manifest_path is not None else None
    return suite


def load_suite(path: Path, *, allow_confirmation: bool = False) -> dict[str, Any]:
    """Read validated metadata, including pending entries; never acquire data."""
    return validate_manifest(
        _read_json(path),
        manifest_path=path,
        allow_confirmation=allow_confirmation,
        require_admission=False,
    )


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_cache_identity(spec: dict[str, Any]) -> dict[str, Any]:
    source = copy.deepcopy(spec["source"])
    if spec["provider"] != "openml_task":
        source = {**copy.deepcopy(_CSV_PARSER_DEFAULTS), **source}
    for field in ("sha256", "splits_sha256", "md5_checksum"):
        if source.get(field) is None:
            source.pop(field, None)
    return {"provider": spec["provider"], "source": source}


def _cache_directory(spec: dict[str, Any], cache: Path) -> Path:
    name = _identifier(spec.get("name"), "dataset.name")
    identity = {"cache_format": 1, **_source_cache_identity(spec)}
    try:
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AdmissionError(f"{name}: source/parser cache identity must be valid JSON: {exc}") from exc
    return cache / name / "sources" / _digest(encoded)


def _read_cache_pin(path: Path) -> dict[str, Any]:
    try:
        pin = _read_json(path)
    except AdmissionError as exc:
        raise AcquisitionError(f"Invalid raw-cache checksum pin at {path}: {exc}") from exc
    if (
        not isinstance(pin.get("identity"), dict)
        or not isinstance(pin.get("sha256"), str)
        or not _SHA256.fullmatch(pin["sha256"])
    ):
        raise AcquisitionError(f"Invalid raw-cache identity/checksum pin at {path}")
    return pin


def _can_reuse_pin(pin: dict[str, Any], identity: dict[str, Any], expected: str | None) -> bool:
    previous = pin["identity"]
    if expected is not None and pin["sha256"] != expected:
        return False
    if previous == identity:
        return True
    provider = identity.get("provider")
    if not isinstance(provider, str) or provider != previous.get("provider"):
        return False
    if not isinstance(previous.get("source"), dict) or not isinstance(identity.get("source"), dict):
        return False
    if _source_cache_identity(previous) == identity:
        return True
    location_fields = {
        "uci": ("uci_id", "data_url"),
        "hf_csv": ("repo_id", "revision", "path"),
        "local_csv": ("path",),
        "kaggle_local": ("path",),
    }.get(provider)
    # Across different parser/schema declarations, only a declared payload SHA
    # can authorize reusing bytes from the same acquisition location.
    return (
        expected is not None
        and location_fields is not None
        and all(
            field in previous["source"]
            and field in identity["source"]
            and previous["source"][field] == identity["source"][field]
            for field in location_fields
        )
    )


def _reuse_cached_bytes(
    directories: Sequence[Path], filename: str, identity: dict[str, Any], expected: str | None
) -> bytes | None:
    for directory in directories:
        payload, pin_path = directory / filename, directory / f"{filename}.pin.json"
        if payload.exists() != pin_path.exists():
            raise AcquisitionError(f"Incomplete raw cache at {payload}; reuse cannot bypass missing evidence")
        if not payload.exists():
            continue
        pin = _read_cache_pin(pin_path)
        if not _can_reuse_pin(pin, identity, expected):
            continue
        data = payload.read_bytes()
        if _digest(data) != pin["sha256"]:
            raise AcquisitionError(f"Checksum drift or corrupt cache at reuse source: {payload}")
        return data
    return None


def _fetch_bytes(url: str, *, timeout: int = 30, max_bytes: int = 128 * 1024 * 1024) -> bytes:
    _require(urlparse(url).scheme == "https", f"Only HTTPS public acquisition URLs are accepted: {url}")
    try:
        with urlopen(
            Request(url, headers={"User-Agent": "FeatCopilot-dataset-catalog/1"}), timeout=timeout
        ) as response:
            data = bytes(response.read(max_bytes + 1))
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise AcquisitionError(
            f"Could not fetch {url} within {timeout}s: {exc}. No synthetic fallback is permitted."
        ) from exc
    if len(data) > max_bytes:
        raise AcquisitionError(f"{url} exceeds the {max_bytes}-byte acquisition limit")
    return data


def _cached_bytes(
    directory: Path,
    filename: str,
    identity: dict[str, Any],
    fetch: Callable[[], bytes],
    expected: str | None = None,
    *,
    reuse_directories: Sequence[Path] = (),
) -> bytes:
    directory.mkdir(parents=True, exist_ok=True)
    payload, pin_path = directory / filename, directory / f"{filename}.pin.json"
    if payload.exists() != pin_path.exists():
        raise AcquisitionError(f"Incomplete raw cache at {payload}; preserve evidence and use a new cache location")
    if payload.exists():
        data = payload.read_bytes()
        pin = _read_cache_pin(pin_path)
        if pin.get("identity") != identity or pin.get("sha256") != _digest(data):
            raise AcquisitionError(f"Source identity/checksum drift or corrupt cache: {payload}")
    else:
        reused = _reuse_cached_bytes(reuse_directories, filename, identity, expected)
        data = fetch() if reused is None else reused
    checksum = _digest(data)
    if expected is not None and checksum != expected:
        raise AcquisitionError(f"SHA-256 mismatch for {payload}: expected {expected}, received {checksum}")
    if not payload.exists():
        with payload.open("xb") as stream:
            stream.write(data)
        with pin_path.open("x", encoding="utf-8") as stream:
            json.dump({"identity": identity, "sha256": checksum}, stream, sort_keys=True)
    return data


def _csv_frame(data: bytes, source: dict[str, Any]) -> pd.DataFrame:
    source = {**_CSV_PARSER_DEFAULTS, **source}
    encoding = source["encoding"]
    separator = source["separator"]
    _require(isinstance(separator, str) and len(separator) == 1, "CSV separator must be a single character")
    try:
        rows = csv.reader(
            io.StringIO(data.decode(encoding)),
            delimiter=separator,
            skipinitialspace=source["skipinitialspace"],
        )
        header = next(rows)
        _require(len(header) == len(set(header)), "Duplicate source column names are not permitted")
        for row in rows:
            _require(
                not row or len(row) == len(header),
                "CSV row width differs from header; refusing implicit index/column shifts",
            )
        frame = pd.read_csv(
            io.BytesIO(data),
            encoding=encoding,
            sep=separator,
            na_values=source["na_values"],
            keep_default_na=source["keep_default_na"],
            skipinitialspace=source["skipinitialspace"],
        )
    except (
        UnicodeError,
        LookupError,
        csv.Error,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        StopIteration,
    ) as exc:
        raise AcquisitionError(f"Cannot parse reviewed CSV source: {exc}") from exc
    return frame


def _load_csv_provider(
    spec: dict[str, Any], directory: Path, reuse_directories: Sequence[Path] = ()
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source, provider = spec["source"], spec["provider"]
    provenance: dict[str, Any] = {"provider": provider, "source": copy.deepcopy(source)}
    if provider == "uci":
        metadata_url = f"https://archive.ics.uci.edu/api/dataset?id={source['uci_id']}"
        metadata = json.loads(
            _cached_bytes(
                directory,
                "metadata.json",
                {"url": metadata_url},
                lambda: _fetch_bytes(metadata_url),
                reuse_directories=reuse_directories,
            )
        )["data"]
        _require(
            metadata["uci_id"] == source["uci_id"] and metadata["data_url"] == source["data_url"],
            "UCI metadata source identity changed",
        )
        actual_columns = {variable["name"] for variable in metadata["variables"]}
        actual_targets = {variable["name"] for variable in metadata["variables"] if variable["role"] == "Target"}
        _require(actual_columns == set(source["columns"]), "UCI metadata column schema changed; review required")
        override = source.get("target_role_override")
        expected_targets = override["metadata_target_columns"] if override is not None else source["target_columns"]
        _require(
            actual_targets == set(expected_targets),
            "UCI native target-role metadata conflicts with reviewed targets",
        )
        if override is not None:
            roles = {variable["name"]: variable["role"] for variable in metadata["variables"]}
            _require(
                roles[override["selected_target"]] == override["metadata_selected_target_role"],
                "UCI selected-column role changed since the explicit target review",
            )
            provenance["target_role_override"] = copy.deepcopy(override)
        url = source["data_url"]
        provenance["version"] = {"uci_id": source["uci_id"], "representation": "UCI normalized public CSV"}
        provenance["upstream_metadata"] = metadata
    elif provider == "hf_csv":
        url = f"https://huggingface.co/datasets/{source['repo_id']}/resolve/{source['revision']}/{quote(source['path'], safe='/')}"
        provenance["version"] = {"revision": source["revision"]}
    else:
        path = Path(source["path"])
        try:
            local_bytes = path.read_bytes()
        except OSError as exc:
            raise AcquisitionError(
                f"Supply the approved local CSV at {path}: {exc}. Automatic Kaggle downloads are disabled."
            ) from exc
        if _digest(local_bytes) != source["sha256"]:
            raise AcquisitionError(f"Local source checksum drift: {path}")
        url = str(path)
        provenance["version"] = {"sha256": source["sha256"]}
    identity = _source_cache_identity(spec)
    fetch = (lambda: local_bytes) if provider in {"local_csv", "kaggle_local"} else lambda: _fetch_bytes(url)
    data = _cached_bytes(
        directory, "data.csv", identity, fetch, source.get("sha256"), reuse_directories=reuse_directories
    )
    provenance.update({"url": url, "sha256": _digest(data)})
    return _csv_frame(data, source), provenance


def _openml_url(url: str) -> str:
    _require(
        urlparse(url).scheme == "https"
        and urlparse(url).hostname in {"openml.org", "www.openml.org", "data.openml.org"},
        f"Unexpected OpenML acquisition URL: {url}",
    )
    return url


def _split_indices(values: Any, size: int) -> np.ndarray:
    array = np.asarray(values)
    _require(array.ndim == 1 and len(array) > 0, "Official split indices must be nonempty one-dimensional arrays")
    _require(
        np.issubdtype(array.dtype, np.number) and np.isfinite(array).all(),
        "Official split indices must be finite integers",
    )
    _require(
        np.equal(array, np.floor(array)).all() and (array >= 0).all() and (array < size).all(),
        "Official split indices are nonintegral or out of range",
    )
    result = array.astype(np.int64)
    _require(len(np.unique(result)) == len(result), "Official split contains duplicate row indices")
    return result


def _official_splits(
    split_frame: pd.DataFrame, size: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict[str, int]]]:
    _require({"type", "rowid", "repeat", "fold"} <= set(split_frame), "Official split file lacks required columns")
    frame = split_frame.copy()
    if "sample" not in frame:
        frame["sample"] = 0
    _require(
        not frame[["type", "rowid", "repeat", "fold", "sample"]].isna().any().any(), "Official split has missing fields"
    )
    _require(frame["type"].isin(["TRAIN", "TEST"]).all(), "Official split type must be TRAIN or TEST")
    for col in ("repeat", "fold", "sample"):
        values = np.asarray(frame[col], dtype=float)
        _require(
            np.isfinite(values).all() and (values >= 0).all() and np.equal(values, np.floor(values)).all(),
            f"Invalid official {col} IDs",
        )
    result, identifiers = [], []
    for key, part in frame.groupby(["repeat", "fold", "sample"], sort=True):
        train = _split_indices(part.loc[part["type"] == "TRAIN", "rowid"].to_numpy(), size)
        test = _split_indices(part.loc[part["type"] == "TEST", "rowid"].to_numpy(), size)
        _require(not np.intersect1d(train, test).size, "Official train/test rows overlap")
        result.append((train, test))
        identifiers.append(dict(zip(("repeat", "fold", "sample"), map(int, key), strict=True)))
    _require(result, "Official task has no outer splits")
    return result, identifiers


def _load_openml(
    spec: dict[str, Any], directory: Path, reuse_directories: Sequence[Path] = ()
) -> tuple[pd.DataFrame, dict[str, Any], list[tuple[np.ndarray, np.ndarray]]]:
    try:
        openml = importlib.import_module("openml")
        arff = importlib.import_module("arff")
    except ImportError as exc:
        raise AcquisitionError(
            "OpenML tasks require the optional 'openml>=0.15.1' dependency (including liac-arff)"
        ) from exc
    source = spec["source"]
    task_url = f"https://www.openml.org/api/v1/json/task/{source['task_id']}"
    task = json.loads(
        _cached_bytes(
            directory,
            "task.json",
            {"url": task_url},
            lambda: _fetch_bytes(task_url),
            reuse_directories=reuse_directories,
        )
    )["task"]
    _require(int(task["task_id"]) == source["task_id"], "OpenML task ID mismatch")
    _require(
        int(task["task_type_id"]) == (1 if spec["task"] == "classification" else 2),
        "OpenML official task type mismatch",
    )
    inputs = {item["name"]: item for item in task["input"]}
    data_info = inputs["source_data"]["data_set"]
    _require(
        int(data_info["data_set_id"]) == source["dataset_id"] and data_info["target_feature"] == spec["target"],
        "OpenML task dataset/target mismatch",
    )
    description_url = f"https://www.openml.org/api/v1/json/data/{source['dataset_id']}"
    description = json.loads(
        _cached_bytes(
            directory,
            "dataset.json",
            {"url": description_url},
            lambda: _fetch_bytes(description_url),
            reuse_directories=reuse_directories,
        )
    )["data_set_description"]
    _require(
        int(description["id"]) == source["dataset_id"] and int(description["version"]) == source["dataset_version"],
        "OpenML dataset version/ID mismatch",
    )
    if source.get("md5_checksum") is not None:
        _require(
            description.get("md5_checksum") == source["md5_checksum"],
            "OpenML upstream payload checksum changed since metadata admission",
        )
    _require(description["format"].lower() == "arff", "Only reviewed dense OpenML ARFF sources are supported")
    data_url = _openml_url(description["url"])
    data = _cached_bytes(
        directory,
        "data.arff",
        {"url": data_url, "version": source["dataset_version"]},
        lambda: _fetch_bytes(data_url),
        source.get("sha256"),
        reuse_directories=reuse_directories,
    )
    if description.get("md5_checksum") and hashlib.md5(data).hexdigest() != description["md5_checksum"]:
        raise AcquisitionError("OpenML upstream payload MD5 mismatch")
    parsed = arff.loads(data.decode("utf-8"))
    frame = pd.DataFrame(parsed["data"], columns=[item[0] for item in parsed["attributes"]])
    split_url = _openml_url(inputs["estimation_procedure"]["estimation_procedure"]["data_splits_url"])
    split_data = _cached_bytes(
        directory,
        "splits.arff",
        {"task_id": source["task_id"], "url": split_url},
        lambda: _fetch_bytes(split_url),
        source.get("splits_sha256"),
        reuse_directories=reuse_directories,
    )
    parsed_splits = arff.loads(split_data.decode("utf-8"))
    split_frame = pd.DataFrame(parsed_splits["data"], columns=[item[0] for item in parsed_splits["attributes"]])
    splits, ids = _official_splits(split_frame, len(frame))
    dimensions = source.get("official_split_dimensions")
    if dimensions is not None:
        expected_ids = [
            {"repeat": repeat, "fold": fold, "sample": sample}
            for repeat in range(dimensions["repeats"])
            for fold in range(dimensions["folds"])
            for sample in range(dimensions["samples"])
        ]
        _require(ids == expected_ids, "Official split dimensions differ from the locked task protocol")
    # Use the official public split container without its pickle/disk cache.
    nested: dict[int, dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]] = {}
    for pair, split_id in zip(splits, ids, strict=True):
        nested.setdefault(split_id["repeat"], {}).setdefault(split_id["fold"], {})[split_id["sample"]] = pair
    official = openml.tasks.split.OpenMLSplit(source["task_id"], "Pinned task-provided outer splits", nested)
    splits = [official.get(**split_id) for split_id in ids]
    return (
        frame,
        {
            "provider": "openml_task",
            "source": copy.deepcopy(source),
            "url": data_url,
            "sha256": _digest(data),
            "version": {
                "dataset_id": source["dataset_id"],
                "dataset_version": source["dataset_version"],
                "task_id": source["task_id"],
            },
            "official_split_ids": ids,
            "official_splits_sha256": _digest(split_data),
            "official_estimation_procedure": inputs["estimation_procedure"]["estimation_procedure"],
            "openml_version": openml.__version__,
        },
        splits,
    )


def _profile(X: pd.DataFrame, y: pd.Series, task: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rows": len(X),
        "features": X.shape[1],
        "missingness": {str(col): float(X[col].isna().mean()) for col in X},
        "cardinality": {str(col): int(X[col].nunique(dropna=False)) for col in X},
        "duplicate_feature_rows": int(X.duplicated().sum()),
    }
    if task == "classification":
        counts = y.value_counts(dropna=False)
        result["class_counts"] = {str(label): int(count) for label, count in counts.items()}
        result["class_prevalence"] = {str(label): float(count / len(y)) for label, count in counts.items()}
        result["n_classes"] = len(counts)
    return result


def load_dataset(spec: dict[str, Any], cache: Path, max_rows: int = 0, sample_seed: int = 2026) -> LoadedDataset:
    """Acquire an admitted dataset, preserving raw row and split alignment.

    Parameters
    ----------
    spec : dict
        A normalized dataset entry returned by ``load_suite`` or
        ``validate_manifest``. Standalone entries may use registered families.
    cache : Path
        Raw-data cache, namespaced by dataset name and normalized source/parser
        identity. Manifest versions coexist without replacing legacy flat pins.
        A compatible legacy/sibling payload is copied only after verifying its
        actual SHA-256 against its pin and any declared payload checksum.
        Existing namespace corruption always raises; it is never repaired by
        downloading or falling back to another namespace. Returned provenance
        is independent of cache layout, preserving metadata/resume checks.
    max_rows : int, default=0
        Zero preserves every row. Positive IID caps sample without balancing.
        Official, grouped and temporal datasets reject every nonzero cap.
    sample_seed : int, default=2026
        Seed for deterministic, classification-stratified IID sampling.

    Returns
    -------
    LoadedDataset
        Features use x0/x1/... names and classification labels use contiguous
        integers. Original names, label meanings, profiles and source indices
        remain in metadata. No learned preprocessing is performed.
        Label availability is parsed only from an explicit ``label_end_column``
        using the declared ``time_format``; no horizon is inferred.
        An optional ``time_hour_column`` adds validated integer hours to a
        date-only time column, retaining the hour predictor. This produces
        calendar-hour identifiers, not inferred label-availability timestamps.
    """
    _require(type(max_rows) is int and max_rows >= 0, "max_rows must be a nonnegative integer")
    _require(type(sample_seed) is int and 0 <= sample_seed < 2**32, "sample_seed must be a uint32 integer")
    extra = [spec["_family"]] if "_family" in spec else []
    normalized = validate_manifest(
        {"name": "single-dataset-admission", "datasets": [spec], "families": extra},
        allow_confirmation=spec.get("_allow_confirmation", False),
    )["datasets"][0]
    _require(normalized["role"] != "inventory", f"{normalized['name']}: inventory/pending datasets must never download")
    _require(
        normalized["_family"].get("reservation", {}).get("status") != "active",
        f"{normalized['name']}: family is reserved as a confirmation candidate; acquisition is blocked until explicit release after algorithm/protocol freeze",
    )
    policy = normalized.get("split_policy", "iid")
    _require(not max_rows or policy == "iid", "Nonzero row caps are forbidden for official/grouped/temporal splits")
    directory = _cache_directory(normalized, cache)
    reuse_directories = [cache / normalized["name"]]
    if directory.parent.exists():
        reuse_directories.extend(
            path
            for path in sorted(directory.parent.iterdir())
            if path != directory and path.is_dir() and _SHA256.fullmatch(path.name)
        )
    splits = None
    if normalized["provider"] == "openml_task":
        frame, provenance, splits = _load_openml(normalized, directory, reuse_directories)
    else:
        frame, provenance = _load_csv_provider(normalized, directory, reuse_directories)
    source = normalized["source"]
    _require(not frame.empty, "Source has no rows")
    _require(not frame.columns.duplicated().any(), "Duplicate source column names are not permitted")
    _require(set(frame.columns) == set(source["columns"]), "Payload column schema differs from reviewed schema")
    frame = frame.loc[:, source["columns"]].reset_index(drop=True)
    if source.get("expected_rows") is not None:
        _require(len(frame) == source["expected_rows"], "Payload row count differs from reviewed source count")
    targets = source["target_columns"]
    if normalized["target_transform"] == "one_hot":
        labels = frame[targets]
        _require(
            labels.isin([0, 1]).all().all() and (labels.sum(axis=1) == 1).all(),
            "One-hot labels must contain exactly one active binary indicator per row",
        )
        y = labels.idxmax(axis=1).rename(normalized["target"])
    else:
        y = frame[normalized["target"]].copy()
        if source.get("target_value_map"):
            mapping = source["target_value_map"]
            _require(y.isin(mapping).all(), "Target contains values absent from reviewed target_value_map")
            y = y.map(mapping)
    _require(not y.isna().any(), "Missing targets are forbidden; silently dropping rows changes task/split identity")
    groups = frame[normalized["group_column"]].copy() if normalized.get("group_column") else None
    timestamps = None
    if normalized.get("time_column"):
        timestamps = pd.to_datetime(frame[normalized["time_column"]], format=normalized["time_format"], errors="raise")
        _require(not timestamps.isna().any(), "Missing timestamps are forbidden")
        if normalized.get("time_hour_column"):
            _require(
                timestamps.eq(timestamps.dt.normalize()).all(),
                "time_hour_column requires a date-only time column, not a second existing hour offset",
            )
            hours = pd.to_numeric(frame[normalized["time_hour_column"]], errors="raise")
            _require(
                not pd.api.types.is_bool_dtype(hours)
                and np.isfinite(hours).all()
                and np.equal(hours, np.floor(hours)).all()
                and hours.between(0, 23).all(),
                "Calendar-hour offsets must be finite integers from 0 through 23",
            )
            timestamps = timestamps + pd.to_timedelta(hours, unit="h")
    label_end_times = None
    if normalized.get("label_end_column"):
        label_end_times = pd.to_datetime(
            frame[normalized["label_end_column"]], format=normalized["time_format"], errors="raise"
        )
        _require(not label_end_times.isna().any(), "Missing label-end timestamps are forbidden")
    if groups is not None:
        _require(not groups.isna().any(), "Missing groups are forbidden")
    excluded = {
        *targets,
        *normalized["drop_columns"],
        normalized.get("group_column"),
        normalized.get("time_column"),
        normalized.get("label_end_column"),
    }
    X = frame[[col for col in frame if col not in excluded]].copy()
    for column in normalized["categorical"]:
        X[column] = X[column].astype("category")
    for column in X:
        if column not in normalized["categorical"]:
            try:
                X[column] = pd.to_numeric(X[column], errors="raise")
            except (TypeError, ValueError) as exc:
                raise AdmissionError(f"{column}: nonnumeric predictor must be explicitly declared categorical") from exc
    class_labels = None
    if normalized["task"] == "classification":
        support = normalized.get("class_support")
        if support is not None:
            _require(set(y.unique()) == set(support["labels"]), "Target classes differ from the reviewed class schema")
            _require(int(y.value_counts().min()) >= support["min_total"], "Total class support below admission margin")
        encoder = LabelEncoder()
        y = pd.Series(encoder.fit_transform(y), name=normalized["target"])
        class_labels = [item.item() if isinstance(item, np.generic) else item for item in encoder.classes_]
        _require(len(class_labels) >= 2, "Classification requires at least two observed classes")
        if support is not None and splits is not None:
            for train, test in splits:
                for indices, threshold in ((train, support["min_outer_train"]), (test, support["min_outer_test"])):
                    counts = np.bincount(np.asarray(y.iloc[indices], dtype=int), minlength=len(class_labels))
                    _require(int(counts.min()) >= threshold, "Official split class support below admission margin")
    else:
        y = pd.to_numeric(y, errors="raise").astype(float)
        _require(np.isfinite(y).all(), "Regression targets must be finite")
        if normalized["target_transform"] == "log1p":
            _require((y > -1).all(), "log1p targets must be greater than -1")
            y = np.log1p(y)
    original_columns = X.columns.tolist()
    original_profile = _profile(X, y, normalized["task"])
    indices = np.arange(len(X), dtype=np.int64)
    if max_rows and max_rows < len(X):
        try:
            indices, _ = train_test_split(
                indices,
                train_size=max_rows,
                random_state=sample_seed,
                stratify=y if normalized["task"] == "classification" else None,
            )
        except ValueError as exc:
            raise AdmissionError(f"Requested row cap cannot preserve class sampling: {exc}") from exc
        indices = np.sort(indices)
    X, y = X.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(drop=True)
    groups = groups.iloc[indices].reset_index(drop=True) if groups is not None else None
    timestamps = timestamps.iloc[indices].reset_index(drop=True) if timestamps is not None else None
    label_end_times = label_end_times.iloc[indices].reset_index(drop=True) if label_end_times is not None else None
    sampled_profile = _profile(X, y, normalized["task"])
    X.columns = [f"x{i}" for i in range(X.shape[1])]
    metadata = {
        **{key: value for key, value in normalized.items() if not key.startswith("_")},
        **provenance,
        "original_rows": len(frame),
        "sampled_rows": len(X),
        "columns": original_columns,
        "original_columns": original_columns,
        "column_mapping": dict(zip(X.columns, original_columns, strict=True)),
        "dtypes": [str(dtype) for dtype in X.dtypes],
        "class_labels": class_labels,
        "original_profile": original_profile,
        "sampled_profile": sampled_profile,
        "sample_indices": indices.tolist(),
        "sample_indices_sha256": _digest(indices.astype("<i8").tobytes()),
        "sample_seed": sample_seed,
        "official_split_ids": provenance.get("official_split_ids", []),
    }
    return LoadedDataset(X, y, metadata, splits, groups, timestamps, label_end_times)
