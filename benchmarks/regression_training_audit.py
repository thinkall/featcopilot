"""TRAIN-only numerical instrumentation of the unchanged canonical linear fits.

No outer-test data, predictions or metrics are accepted or produced. A one-row
TRAIN probe satisfies generate_features' transform interface for the final
refit; it is never scored. The eight canonical prediction-pipeline fits are
unchanged. Diagnostics run only after tuning and full-TRAIN fitting finish.

Clipping saturation means equality to a fitted nonconstant numeric min/max on
the same training rows, not proof that clipping changed a value. Bound values,
design matrices, coefficients, residual vectors and categorical vocabularies
are never serialized. Trusted local pickle artifacts only; this is not a
general model-loading service or a new feature-selection policy.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import logging
import pickle
import random
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from benchmarks import tabular
from featcopilot.utils.logger import get_logger
from featcopilot.utils.numeric_clipping import NumericQuantileClipper, validate_clip_quantiles

logger = get_logger(__name__)
METHOD = "regression-training-audit"
PROTOCOL = "canonical_linear_train_only_numerical_audit_v1"
COMPONENTS = (("raw", "baseline-robust"), ("fe", "featcopilot-frequency-aligned-robust"))
_PAYLOAD_KEYS = {
    "X",
    "y",
    "config",
    "dataset",
    "source",
    "task",
    "method",
    "seed",
    "fold",
    "split_sha256",
    "family_id",
    "role",
    "expected_cells",
    "data_path",
    "prepared_sha256",
    "provenance",
    "expected_components",
    "original_train_indices_sha256",
    "training_rows",
    "reference_run",
    "package_versions",
    "column_descriptions",
}
_CONFIG_KEYS = {
    "learners",
    "seed",
    "threads",
    "max_candidates",
    "max_added_features",
    "stability_repeats",
    "numeric_clip_quantiles",
    "categorical_encoding",
    "target_transform",
    "split_kind",
    "split_strategy",
    "groups",
    "timestamps",
    "label_end_times",
    "package_versions",
    "detailed_diagnostics",
    "memory_limit_mb",
    "screening",
}
_BOUNDARIES = ("groups", "group", "group_ids", "timestamps", "time", "times", "label_end", "label_end_times")
_REFERENCE_KEYS = {"parameters", "inner_losses", "outer_feature_names_sha256", "n_output_features"}


class AuditSettings(BaseModel):
    """Validated execution settings; bounded synthetic tests may lower FE budgets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = Field(ge=0, lt=2**32, strict=True)
    threads: int = Field(default=8, ge=1, le=8, strict=True)
    max_candidates: int = Field(default=1000, ge=1, le=1000, strict=True)
    max_added_features: int = Field(default=50, ge=0, le=50, strict=True)
    stability_repeats: int = Field(default=3, ge=1, le=3, strict=True)


def _sha256(value: object, scope: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{scope} must be a lowercase SHA256 digest")
    return value


def _policy(values: Mapping[str, object]) -> None:
    for key in _BOUNDARIES:
        if values.get(key) is not None:
            raise ValueError(f"TRAIN-only audit requires IID data; {key} is unsupported")
    for key in ("split_kind", "split_strategy"):
        if key in values and values[key] != "iid":
            raise ValueError(f"TRAIN-only audit requires {key}='iid'")
    if values.get("target_transform", "identity") != "identity":
        raise ValueError("TRAIN-only audit requires the identity target transform")
    if values.get("categorical_encoding", "one_hot") != "one_hot":
        raise ValueError("TRAIN-only audit requires one_hot encoding")


def _metadata(value: object, scope: str) -> None:
    """Admit scalar provenance only, never rows or unknown test-bearing vectors."""
    if isinstance(value, Mapping):
        _policy(value)
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{scope} metadata keys must be strings")
            if any(token in key.lower() for token in ("prediction", "target_values", "labels", "test_x", "test_y")):
                raise ValueError(f"{scope} must not contain data-bearing field {key}")
            _metadata(item, f"{scope}.{key}")
    elif value is None or isinstance(value, (str, bool, int)):
        return
    elif isinstance(value, float) and np.isfinite(value):
        return
    else:
        raise ValueError(f"{scope} must contain only finite scalar metadata, not arrays or rows")


def _pins(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("package_versions must be a mapping")
    observed = {}
    for package, expected in value.items():
        if not isinstance(package, str) or not isinstance(expected, str):
            raise ValueError("Package pins must map distribution names to exact version strings")
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise ValueError(f"Package pin mismatch for {package}: expected {expected}, observed {actual}")
        observed[package] = actual
    return observed


def _load_train(payload: dict) -> tuple[pd.DataFrame, pd.Series, str | None]:
    unknown = set(payload) - _PAYLOAD_KEYS
    if unknown:
        raise ValueError(f"TRAIN-only audit received unsupported payload fields: {sorted(unknown)}")
    path = payload.get("data_path")
    if path is not None:
        if "X" in payload or "y" in payload:
            raise ValueError("Supply either TRAIN X/y or a hashed TRAIN-only artifact, not both")
        expected = _sha256(payload.get("prepared_sha256"), "TRAIN data hash")
        content = Path(path).read_bytes()
        if tabular.digest(content) != expected:
            raise ValueError("TRAIN data SHA256 mismatch")
        # Only parent-created, trusted, explicitly TRAIN-only local artifacts.
        data = pickle.loads(content)
        if not isinstance(data, dict) or set(data) != {"X", "y"}:
            raise ValueError("TRAIN artifact must contain exactly X and y, never outer-test fields")
    else:
        if payload.get("prepared_sha256") is not None:
            raise ValueError("prepared_sha256 requires data_path")
        data = payload
        expected = None
    X, y = data["X"], data["y"]
    if (
        not isinstance(X, pd.DataFrame)
        or X.empty
        or not X.columns.is_unique
        or not all(isinstance(column, str) for column in X)
        or any(pd.api.types.is_complex_dtype(dtype) for dtype in X.dtypes)
    ):
        raise ValueError("TRAIN X must be a nonempty DataFrame with unique string columns and real features")
    if not isinstance(y, pd.Series) or not X.index.equals(y.index) or len(X) != len(y):
        raise ValueError("TRAIN X/y must preserve matching original positional indices")
    if y.to_numpy().dtype.kind not in "iuf" or not np.isfinite(y.to_numpy()).all() or y.nunique() < 2:
        raise ValueError("TRAIN targets must be finite nonconstant single-output real values")
    _metadata(X.attrs, "X.attrs")
    _metadata(y.attrs, "y.attrs")
    return X, y, expected


def _name_hash(names: list[str]) -> str:
    return tabular.digest(json.dumps(names).encode())


def _generated(
    train: pd.DataFrame, valid: pd.DataFrame, target: pd.Series, method: str, config: dict, workspace: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit, other = tabular.generate_features(train, valid, target, method, "regression", config, workspace)
    if (
        not isinstance(fit, pd.DataFrame)
        or not isinstance(other, pd.DataFrame)
        or not fit.index.equals(train.index)
        or not other.index.equals(valid.index)
        or not fit.columns.equals(other.columns)
        or not fit.columns.is_unique
        or not all(isinstance(column, str) for column in fit)
        or list(fit.columns[: train.shape[1]]) != list(train.columns)
        or not train.shape[1] <= fit.shape[1] <= train.shape[1] + config["max_added_features"]
    ):
        raise ValueError("Canonical feature generation changed row alignment or violated its plan schema")
    return fit, other


def _matrix_diagnostics(design: np.ndarray) -> dict[str, object]:
    values = np.asarray(design, dtype=np.float64)
    if values.ndim != 2 or not min(values.shape) or not np.isfinite(values).all():
        raise ValueError("Scaled training design must be a nonempty finite matrix")
    singular = np.linalg.svd(values, compute_uv=False)
    if not np.isfinite(singular).all():
        raise ValueError("Nonfinite scaled training design singular values")
    largest, smallest = float(singular[0]), float(singular[-1])
    tolerance = max(values.shape) * np.finfo(np.float64).eps * largest
    rank = int(np.count_nonzero(singular > tolerance))
    deficient = rank < values.shape[1]
    condition = None if deficient else largest / smallest
    if condition is not None and not np.isfinite(condition):
        raise ValueError("Full-column-rank condition estimate must be finite")
    return {
        "rows": values.shape[0],
        "columns": values.shape[1],
        "rank": rank,
        "rank_tolerance": tolerance,
        "singular_value_min": smallest,
        "singular_value_max": largest,
        "condition_number": condition,
        "condition_status": "rank_deficient" if deficient else "full_column_rank",
        "design_scope": "fitted_pipeline_training_transform_no_added_intercept_column",
    }


def _clipping_diagnostics(clipper: NumericQuantileClipper, frame: pd.DataFrame) -> dict[str, object]:
    if clipper.quantiles_ != (0.0, 1.0) or clipper.n_samples_seen_ != len(frame):
        raise ValueError("Clipping audit requires the canonical training-fitted min/max clipper")
    clipped = clipper.transform(frame)
    columns = []
    total_finite = total_lower = total_upper = 0
    for name in clipper.numeric_columns_:
        bounds = clipper.bounds_[name]
        values = clipped[name].to_numpy(dtype=float, na_value=np.nan)
        finite = np.isfinite(values)
        count = int(finite.sum())
        undefined = bounds is None
        constant = bounds is not None and bounds[0] == bounds[1]
        status = "undefined" if undefined else ("constant" if constant else "eligible")
        lower = upper = None
        if bounds is not None and not constant:
            lower = int(np.count_nonzero(values[finite] == bounds[0]))
            upper = int(np.count_nonzero(values[finite] == bounds[1]))
            total_finite += count
            total_lower += lower
            total_upper += upper
        columns.append(
            {
                "name": name,
                "status": status,
                "finite_count": count,
                "missing_count": len(frame) - count,
                "lower_count": lower,
                "upper_count": upper,
                "saturated_count": None if lower is None or upper is None else lower + upper,
                "saturated_fraction": None if lower is None or upper is None or count == 0 else (lower + upper) / count,
            }
        )
    return {
        "definition": "exact_fitted_min_or_max_after_guard_and_clipping_on_training_values_not_changed_value_count",
        "eligible_finite_count": total_finite,
        "lower_count": total_lower,
        "upper_count": total_upper,
        "saturated_count": total_lower + total_upper,
        "saturated_fraction": (total_lower + total_upper) / total_finite if total_finite else None,
        "undefined_columns": list(clipper.all_missing_columns_),
        "constant_columns": list(clipper.constant_columns_),
        "excluded_boolean_columns": list(clipper.boolean_columns_),
        "columns": columns,
    }


def _fit_diagnostics(
    model: Pipeline, frame: pd.DataFrame, target: pd.Series, originals: list[str]
) -> dict[str, object]:
    clean = frame.replace([np.inf, -np.inf], np.nan)
    if not isinstance(model, Pipeline) or len(model.steps) != 3:
        raise TypeError("Audit requires the canonical clipper/preprocessor/Ridge pipeline")
    clipper, estimator = model.steps[0][1], model.steps[-1][1]
    if not isinstance(clipper, NumericQuantileClipper) or not isinstance(estimator, Ridge):
        raise TypeError("Audit supports canonical training-fitted clipping and linear Ridge only")
    design = np.asarray(model[:-1].transform(clean), dtype=np.float64)
    coefficients = np.asarray(estimator.coef_, dtype=np.float64)
    if coefficients.shape != (design.shape[1],) or not np.isfinite(coefficients).all():
        raise ValueError("Audit requires finite single-output Ridge coefficients")
    norms = {"l2": float(np.linalg.norm(coefficients)), "linf": float(np.max(np.abs(coefficients)))}
    if not all(np.isfinite(value) for value in norms.values()):
        raise ValueError("Nonfinite coefficient norm")
    names = list(frame.columns)
    return {
        "fit_rows": len(frame),
        "feature_names": names,
        "feature_names_sha256": _name_hash(names),
        "n_output_features": len(names),
        "selected_features": names[len(originals) :],
        "scaled_training_design": _matrix_diagnostics(design),
        "coefficient_norms": norms,
        "training_residual_rmse": tabular.regression_metrics(np.asarray(model.predict(clean)), target)["loss"],
        "clipping_saturation": _clipping_diagnostics(clipper, clean),
    }


def _component(X: pd.DataFrame, y: pd.Series, method: str, config: dict, workspace: Path) -> dict[str, object]:
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    train, valid = train_test_split(np.arange(len(X)), test_size=0.25, random_state=config["seed"], stratify=None)
    candidates = tabular.candidate_parameters("linear")
    if candidates != [{"regularization": value} for value in (0.1, 1.0, 10.0)]:
        raise ValueError("Canonical linear parameter grid differs from the declared three-setting replay")
    aligned = method == "featcopilot-frequency-aligned-robust"
    settings = {
        **config,
        "selection_learner": "linear",
        "groups": None,
        "timestamps": None,
        "label_end_times": None,
    }
    first_config = {**settings, "selection_model_parameters": candidates[0]} if aligned else settings
    inner_X, valid_X = _generated(X.iloc[train], X.iloc[valid], y.iloc[train], method, first_config, workspace)
    trials = []
    losses: list[float] = []
    best_model: Pipeline | None = None
    best_frame: pd.DataFrame | None = None
    for index, parameters in enumerate(candidates):
        if aligned and index:
            inner_X, valid_X = _generated(
                X.iloc[train],
                X.iloc[valid],
                y.iloc[train],
                method,
                {**settings, "selection_model_parameters": parameters},
                workspace,
            )
        model = tabular.make_model(inner_X, "regression", "linear", parameters, settings)
        if not isinstance(model, Pipeline):
            raise TypeError("Canonical make_model must return a Pipeline")
        model.fit(inner_X.replace([np.inf, -np.inf], np.nan), y.iloc[train])
        loss = tabular.metrics(model, valid_X.replace([np.inf, -np.inf], np.nan), y.iloc[valid], "regression")["loss"]
        if not np.isfinite(loss):
            raise ValueError("Nonfinite inner RMSE")
        if best_model is None or loss < min(losses):
            best_model, best_frame = model, inner_X
        losses.append(loss)
        trials.append(
            {
                "parameters": copy.deepcopy(parameters),
                "loss": loss,
                "n_output_features": inner_X.shape[1],
                "feature_names_sha256": _name_hash(list(inner_X.columns)),
            }
        )
    best = int(np.argmin(losses))
    parameters = candidates[best]
    outer_config = {**settings, "selection_model_parameters": parameters} if aligned else settings
    final_X, _ = _generated(X, X.iloc[:1], y, method, outer_config, workspace)
    final_model = tabular.make_model(final_X, "regression", "linear", parameters, settings)
    if not isinstance(final_model, Pipeline):
        raise TypeError("Canonical make_model must return a Pipeline")
    final_model.fit(final_X.replace([np.inf, -np.inf], np.nan), y)
    if best_model is None or best_frame is None:
        raise RuntimeError("Canonical inner search did not retain a winning fit")
    # Instrument only after the unchanged canonical fitting sequence is complete.
    diagnostic_started = time.perf_counter()
    originals = list(X.columns)
    inner = _fit_diagnostics(best_model, best_frame, y.iloc[train], originals)
    final = _fit_diagnostics(final_model, final_X, y, originals)
    inner_names, final_names = list(best_frame.columns[len(originals) :]), list(final_X.columns[len(originals) :])
    common = set(inner_names) & set(final_names)
    return {
        "method": method,
        "parameters": copy.deepcopy(parameters),
        "inner_losses": losses,
        "inner_trials": trials,
        "winning_trial": best,
        "inner_train_rows": len(train),
        "inner_validation_rows": len(valid),
        "inner_train_positions_sha256": tabular.digest(np.asarray(train, dtype="<i8").tobytes()),
        "inner_validation_positions_sha256": tabular.digest(np.asarray(valid, dtype="<i8").tobytes()),
        "outer_feature_names_sha256": _name_hash(list(final_X.columns)),
        "n_output_features": final_X.shape[1],
        "winning_inner": inner,
        "full_train": final,
        "selected_name_turnover": {
            "common_count": len(common),
            "removed": [name for name in inner_names if name not in common],
            "added": [name for name in final_names if name not in common],
            "ordered_names_equal": inner_names == final_names,
        },
        "instrumentation_seconds": time.perf_counter() - diagnostic_started,
        "prediction_pipeline_fit_count": 4,
    }


def run_case(payload: dict, workspace: Path) -> list[dict]:
    """Replay raw/FE fits using only outer-TRAIN data and report numerical diagnostics.

    Parameters
    ----------
    payload : dict
        Standard worker identity fields, config, TRAIN-only X/y or trusted
        data_path/prepared_sha256, optional scalar provenance and exact
        package_versions. expected_components may map raw/fe to canonical
        parameters, inner_losses, outer_feature_names_sha256 and n_output_features.
        Optional original_train_indices_sha256/reference_run describe the
        parent's source; training_rows must match the supplied TRAIN frame.
    workspace : pathlib.Path
        Parent-owned worker directory passed to canonical feature generation.

    Returns
    -------
    list of dict
        One success row with an audit object, never a top-level scored loss.
        Reference mismatches and invalid inputs raise; the worker boundary logs
        them and emits ordinary failure_rows rather than a successful audit.
    """
    started = time.perf_counter()
    if payload.get("method") != METHOD or payload.get("task") != "regression":
        raise ValueError("TRAIN-only audit requires its explicit regression method")
    config = payload["config"]
    if not isinstance(config, dict) or set(config) - _CONFIG_KEYS:
        raise ValueError("TRAIN-only audit received unsupported config fields")
    _policy(config)
    if config.get("learners") != ["linear"]:
        raise ValueError("TRAIN-only audit supports exactly the linear learner")
    if config.get("screening", "auto") != "auto":
        raise ValueError("TRAIN-only audit preserves canonical screening='auto' input settings")
    if "memory_limit_mb" in config and (
        type(config["memory_limit_mb"]) is not int or config["memory_limit_mb"] != 4096
    ):
        raise ValueError("TRAIN-only audit supervisor memory_limit_mb must equal 4096")
    if validate_clip_quantiles(config.get("numeric_clip_quantiles", (0.0, 1.0))) != (0.0, 1.0):
        raise ValueError("TRAIN-only audit requires canonical min/max clipping")
    keys = ("seed", "threads", "max_candidates", "max_added_features", "stability_repeats")
    settings = AuditSettings.model_validate({key: config[key] for key in keys if key in config})
    if type(payload.get("seed")) is not int or payload["seed"] != settings.seed:
        raise ValueError("Payload and config seeds must match")
    if type(payload.get("fold")) is not int or payload["fold"] < 0:
        raise ValueError("fold must be a nonnegative integer")
    _sha256(payload.get("split_sha256"), "original outer split hash")
    provenance = copy.deepcopy(payload.get("provenance", {}))
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be a mapping of scalar metadata")
    for key in ("original_train_indices_sha256", "training_rows", "reference_run"):
        if key in payload:
            if key in provenance and provenance[key] != payload[key]:
                raise ValueError(f"Conflicting provenance for {key}")
            provenance[key] = payload[key]
    _metadata(provenance, "provenance")
    if "original_train_indices_sha256" in provenance:
        _sha256(provenance["original_train_indices_sha256"], "parent original TRAIN indices hash")
    if "reference_run" in provenance and not isinstance(provenance["reference_run"], str):
        raise ValueError("reference_run must be a string")
    descriptions = payload.get("column_descriptions", {})
    if not isinstance(descriptions, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in descriptions.items()
    ):
        raise ValueError("column_descriptions must map feature names to descriptions")
    observed_versions = _pins(config.get("package_versions", {}))
    observed_versions.update(_pins(payload.get("package_versions", {})))
    expectations = payload.get("expected_components", {})
    if not isinstance(expectations, dict) or set(expectations) - {"raw", "fe"}:
        raise ValueError("expected_components must map raw/fe to canonical fields")
    for expected in expectations.values():
        if not isinstance(expected, dict) or set(expected) - _REFERENCE_KEYS:
            raise ValueError("Unsupported reference expectation fields")
        if "outer_feature_names_sha256" in expected:
            _sha256(expected["outer_feature_names_sha256"], "reference feature-name hash")
        if "n_output_features" in expected and (
            type(expected["n_output_features"]) is not int or expected["n_output_features"] < 1
        ):
            raise ValueError("Reference n_output_features must be a positive integer")
        if "parameters" in expected:
            parameters = expected["parameters"]
            if (
                not isinstance(parameters, dict)
                or set(parameters) != {"regularization"}
                or type(parameters["regularization"]) not in (int, float)
                or not np.isfinite(parameters["regularization"])
            ):
                raise ValueError("Reference parameters must contain one finite numeric regularization value")
        if "inner_losses" in expected:
            losses = expected["inner_losses"]
            if (
                not isinstance(losses, list)
                or len(losses) != 3
                or any(type(value) not in (int, float) or not np.isfinite(value) or value < 0 for value in losses)
            ):
                raise ValueError("Reference inner_losses must contain exactly three finite nonnegative RMSEs")
    X, y, data_hash = _load_train(payload)
    if "training_rows" in provenance and (
        type(provenance["training_rows"]) is not int or provenance["training_rows"] != len(X)
    ):
        raise ValueError("Parent training_rows metadata differs from the supplied TRAIN frame")
    runtime = {
        **settings.model_dump(),
        "learners": ["linear"],
        "numeric_clip_quantiles": (0.0, 1.0),
        "categorical_encoding": "one_hot",
        "column_descriptions": descriptions,
    }
    components = {}
    with threadpool_limits(limits=settings.threads):
        for role, method in COMPONENTS:
            result = _component(X, y, method, runtime, workspace)
            for key, expected in expectations.get(role, {}).items():
                if result[key] != expected:
                    raise ValueError(f"Canonical replay mismatch for {role}.{key}")
            components[role] = result
    identity = {key: payload[key] for key in ("dataset", "source", "task", "method", "seed", "fold", "split_sha256")}
    report = {
        **identity,
        "family_id": payload.get("family_id", payload["dataset"]),
        "role": payload.get("role", "development"),
        "expected_cells": payload.get("expected_cells", 0),
        "learner": "linear",
        "status": "success",
        "n_train": len(X),
        "n_input_features": X.shape[1],
        "audit": {
            "protocol": PROTOCOL,
            "scope": "original_outer_train_only_no_outer_test_access_or_scoring",
            "settings": settings.model_dump(),
            "package_versions": observed_versions,
            "train_data_sha256": data_hash,
            "provenance": provenance,
            "reference_fields_verified": {role: sorted(expected) for role, expected in expectations.items()},
            "prediction_pipeline_fit_count": 8,
            "components": components,
        },
        "case_seconds": time.perf_counter() - started,
    }
    json.dumps(report, allow_nan=False)
    return [report]


def main(argv: list[str] | None = None) -> int:
    """Run a trusted local worker payload, logging failures through canonical rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--worker-output", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    payload = pickle.loads(args.worker.read_bytes())
    started = time.perf_counter()
    try:
        rows = run_case(payload, args.worker.parent)
    except Exception as error:
        logger.exception("TRAIN-only regression audit failed")
        rows = tabular.failure_rows(payload, "error", f"{type(error).__name__}: {error}", time.perf_counter() - started)
    tabular.write_json(args.worker_output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
