"""Benchmark-only equal-weight regression prediction averaging.

Two independent canonical complete procedures are run unchanged: robust raw,
then frequency/aligned/robust FE. Their final test predictions, not their
metrics, are averaged in float64. There is no OOF fitting, fitted blend weight,
gate, calibration, target transform, provider access or production default.
This experiment makes no causal variance-reduction or no-harm claim.

The canonical dispatcher owns prepared-data loading, learner expansion and
worker failure logging. This module writes no artifacts and returns no models,
rows, vocabularies or prediction vectors. Component plans remain separate.
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Mapping
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from benchmarks import tabular
from benchmarks.execution import split_fingerprint
from featcopilot.utils.splitting import validate_split_boundaries

METHOD = "featcopilot-regression-average-robust"
PROTOCOL = "canonical_regression_fixed_average_robust_v1"
COMPONENTS = (("raw", "baseline-robust"), ("fe", "featcopilot-frequency-aligned-robust"))
_STAGES = ("inner_fe_seconds", "outer_fe_seconds", "tuning_seconds", "model_fit_seconds", "predict_seconds")
_IDENTITY = ("dataset", "source", "task", "seed", "fold", "split_sha256")
_BOUNDARIES = ("groups", "group", "group_ids", "timestamps", "times", "label_end", "label_end_times")
FloatArray = NDArray[np.float64]


def _validate_policy(values: Mapping[str, object]) -> None:
    if values.get("task", "regression") != "regression":
        raise ValueError("Regression averaging supports regression only")
    for key in _BOUNDARIES:
        if values.get(key) is not None:
            raise ValueError(f"Regression averaging requires IID data; {key} is unsupported")
    for key in ("split_kind", "split_strategy"):
        if key in values and values[key] != "iid":
            raise ValueError(f"Regression averaging requires {key}='iid'")
    if values.get("categorical_encoding", "one_hot") != "one_hot":
        raise ValueError("Regression averaging requires one_hot encoding; target encoding is unsupported")
    if values.get("target_transform", "identity") != "identity":
        raise ValueError("Regression averaging requires the identity target transform")
    if "numeric_clip_quantiles" in values:
        quantiles = values["numeric_clip_quantiles"]
        if not isinstance(quantiles, (list, tuple)) or tuple(quantiles) != (0.0, 1.0):
            raise ValueError("Regression averaging fixes numeric_clip_quantiles=(0.0, 1.0)")
    metadata = values.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise TypeError("Regression averaging metadata must be a mapping")
        _validate_policy(metadata)


def _finite_number(value: object, scope: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{scope} must be a finite number")
    result = float(value)
    if not np.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{scope} must be finite" + (" and nonnegative" if nonnegative else ""))
    return result


def _positive_width(value: object, scope: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{scope} must be a positive integer")
    return value


def _prediction(value: object, rows: int) -> FloatArray:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (rows,)
        or value.dtype.kind not in "iuf"
        or not np.isfinite(value).all()
    ):
        raise ValueError("Component predictions must be finite, numeric, one-dimensional and row-aligned")
    prediction = np.asarray(value, dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError("Component predictions must be representable in float64")
    return prediction


def _prediction_hash(prediction: FloatArray) -> str:
    return tabular.digest(np.asarray(prediction, dtype="<f8").tobytes())


def _sha256(value: object, scope: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{scope} must be a lowercase SHA256 digest")
    return value


def _component(
    rows: list[dict],
    capture: dict[str, np.ndarray],
    *,
    role: str,
    method: str,
    learner: str,
    identity: dict[str, object],
    n_train: int,
    n_test: int,
    n_input: int,
    max_added: int,
) -> tuple[dict[str, object], FloatArray]:
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError(f"{role} component must return exactly one canonical row")
    row = rows[0]
    if row.get("status") != "success":
        raise RuntimeError(f"{role} component failed; averaging cannot emit a success")
    for key, expected in {**identity, "method": method, "learner": learner}.items():
        if row.get(key) != expected:
            raise ValueError(f"{role} component {key} identity differs from the requested cell")
    if row.get("stage") != identity.get("stage"):
        raise ValueError(f"{role} component stage identity differs from the requested cell")
    if any(
        row.get(key) != value
        for key, value in (("n_train", n_train), ("n_test", n_test), ("n_input_features", n_input))
    ):
        raise ValueError(f"{role} component row counts or input feature width differ")
    width = _positive_width(row.get("n_output_features"), f"{role} plan width")
    if not n_input <= width <= n_input + max_added or (role == "raw" and width != n_input):
        raise ValueError(f"{role} component plan width violates the canonical feature budget")
    if row.get("numeric_clip_quantiles") != [0.0, 1.0] or row.get("matched_baseline_method") != "baseline-robust":
        raise ValueError(f"{role} component robust preprocessing policy differs")
    _validate_policy(row)
    for name in ("loss", "mae", "r2"):
        _finite_number(row.get(name), f"{role} {name}", nonnegative=name != "r2")
    losses = row.get("inner_losses")
    parameters = row.get("parameters")
    candidates = tabular.candidate_parameters(learner)
    if len(candidates) != 3 or not isinstance(losses, list) or len(losses) != 3:
        raise ValueError(f"{role} component must retain the canonical three-setting inner search")
    for loss in losses:
        _finite_number(loss, f"{role} inner loss", nonnegative=True)
    if not isinstance(parameters, dict) or parameters != candidates[int(np.argmin(losses))]:
        raise ValueError(f"{role} component selected parameters differ from its winning inner trial")
    if role == "fe":
        trials = row.get("inner_pipeline_trials")
        if not isinstance(trials, list) or len(trials) != 3 or row.get("selection_model_parameters") != parameters:
            raise ValueError("FE component must retain its aligned three-setting pipeline trials")
        for trial, candidate, loss in zip(trials, candidates, losses, strict=True):
            if not isinstance(trial, dict) or trial.get("parameters") != candidate or trial.get("loss") != loss:
                raise ValueError("FE pipeline trial parameters or losses differ from the canonical search")
            trial_width = _positive_width(trial.get("n_output_features"), "FE trial width")
            if not n_input <= trial_width <= n_input + max_added:
                raise ValueError("FE trial width violates the canonical feature budget")
    stages = [_finite_number(row.get(key), f"{role} {key}", nonnegative=True) for key in _STAGES]
    pipeline = _finite_number(row.get("pipeline_seconds"), f"{role} pipeline_seconds", nonnegative=True)
    _finite_number(row.get("case_seconds"), f"{role} case_seconds", nonnegative=True)
    if not np.isclose(pipeline, sum(stages), rtol=1e-12, atol=1e-9):
        raise ValueError(f"{role} component pipeline timing does not match its stages")
    if set(capture) != {learner}:
        raise ValueError(f"{role} component is missing the exact learner prediction capture")
    prediction = _prediction(capture[learner], n_test)
    if _sha256(row.get("outer_prediction_sha256"), f"{role} prediction hash") != _prediction_hash(prediction):
        raise ValueError(f"{role} component prediction capture differs from its canonical hash")
    _sha256(row.get("outer_feature_names_sha256"), f"{role} feature-name hash")
    json.dumps(row, allow_nan=False)
    return copy.deepcopy(row), prediction


def run_case(payload: dict, workspace: Path) -> list[dict]:
    """Run two canonical procedures and return one equal-weight regression row.

    Parameters
    ----------
    payload : dict
        Already hydrated canonical cell with aligned DataFrame X/Series y,
        valid positional train_idx/test_idx and their canonical split hash.
        Exactly one learner (linear or lightgbm) is required. Config, metadata
        and frame attributes must not request group/time boundaries, target
        encoding or a nonidentity target transform.
    workspace : pathlib.Path
        Supervisor-owned workspace passed unchanged to both canonical runs.

    Returns
    -------
    list of dict
        One success row for featcopilot-regression-average-robust, containing
        averaged-prediction metrics and nested complete raw/FE provenance.
        n_output_features is the maximum separate component plan width, not a
        union or a shared model-input schema. Both complete procedures are
        charged once. averaging_seconds and average_scoring_seconds are
        already included in predict_seconds/pipeline_seconds, not extra costs.

    Raises
    ------
    ValueError
        Unsupported configuration, invalid splits, identities or component
        predictions. There is no component substitution or fallback average.
    RuntimeError
        A component returns a failure row. Exceptions from canonical fitting
        propagate unchanged to the standard worker logger and failure_rows.
    """
    started = time.perf_counter()
    if payload.get("method") != METHOD or payload.get("task") != "regression":
        raise ValueError("Regression averaging supports only its explicit regression method")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise TypeError("Regression averaging config must be a mapping")
    _validate_policy(payload)
    _validate_policy(config)
    learners = config.get("learners")
    if not isinstance(learners, list) or len(learners) != 1 or learners[0] not in ("linear", "lightgbm"):
        raise ValueError("Regression averaging requires exactly one learner: linear or lightgbm")
    learner = learners[0]
    seed = config.get("seed")
    if (
        type(seed) is not int
        or not 0 <= seed < 2**32
        or type(payload.get("seed")) is not int
        or payload.get("seed") != seed
    ):
        raise ValueError("Payload and config must use the same uint32 integer seed")
    if type(payload.get("fold")) is not int or payload["fold"] < 0:
        raise ValueError("fold must be a nonnegative integer")
    _positive_width(config.get("threads"), "threads")
    for key in ("max_candidates", "stability_repeats"):
        if key in config:
            _positive_width(config[key], key)
    max_added = config.get("max_added_features")
    if type(max_added) is not int or max_added < 0:
        raise ValueError("max_added_features must be a nonnegative integer")
    X, y = payload["X"], payload["y"]
    if (
        not isinstance(X, pd.DataFrame)
        or X.empty
        or not X.columns.is_unique
        or not all(isinstance(column, str) for column in X)
        or any(pd.api.types.is_complex_dtype(dtype) for dtype in X.dtypes)
    ):
        raise ValueError("Regression averaging requires a nonempty DataFrame with unique string feature names")
    if not isinstance(y, pd.Series) or len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("X and y must be positionally and index aligned")
    if y.to_numpy().dtype.kind not in "iuf" or not np.isfinite(y.to_numpy()).all():
        raise ValueError("Regression targets must be finite real numeric values")
    _validate_policy(X.attrs)
    _validate_policy(y.attrs)
    train, test = np.asarray(payload["train_idx"]), np.asarray(payload["test_idx"])
    validate_split_boundaries(train, test, n_rows=len(X))
    if len(test) < 2:
        raise ValueError("Regression evaluation needs at least two test rows for R2")
    if payload.get("split_sha256") != split_fingerprint(train, test):
        raise ValueError("Declared split identity differs from the ordered train/test positions")
    identity = {key: payload[key] for key in _IDENTITY}
    identity.update(
        family_id=payload.get("family_id", payload["dataset"]),
        role=payload.get("role", "development"),
        expected_cells=payload.get("expected_cells", 0),
    )
    if "stage" in payload:
        identity["stage"] = payload["stage"]
    records: dict[str, dict[str, object]] = {}
    predictions: dict[str, FloatArray] = {}
    for role, method in COMPONENTS:
        # The dispatcher already hydrated the data. Never reread prepared files
        # or let a component re-enter this experiment's dispatch branch.
        component_payload = {key: value for key, value in payload.items() if key != "data_path"}
        component_payload.update(method=method, config=copy.deepcopy(config))
        capture: dict[str, np.ndarray] = {}
        rows = tabular.run_case(component_payload, workspace, outer_predictions=capture)
        records[role], predictions[role] = _component(
            rows,
            capture,
            role=role,
            method=method,
            learner=learner,
            identity=identity,
            n_train=len(train),
            n_test=len(test),
            n_input=X.shape[1],
            max_added=max_added,
        )
    average_started = time.perf_counter()
    average = _prediction(0.5 * predictions["raw"] + 0.5 * predictions["fe"], len(test))
    averaging_seconds = time.perf_counter() - average_started
    scoring_started = time.perf_counter()
    scores = tabular.regression_metrics(average, y.iloc[test])
    for key in ("loss", "r2", "mae"):
        _finite_number(scores.get(key), f"average {key}", nonnegative=key != "r2")
    scoring_seconds = time.perf_counter() - scoring_started
    timings = {
        key: sum(_finite_number(record[key], key, nonnegative=True) for record in records.values()) for key in _STAGES
    }
    timings["predict_seconds"] += averaging_seconds + scoring_seconds
    widths = {role: _positive_width(record["n_output_features"], role) for role, record in records.items()}
    feature_hashes = {role: record["outer_feature_names_sha256"] for role, record in records.items()}
    result = {
        **identity,
        "method": METHOD,
        "learner": learner,
        "status": "success",
        **scores,
        "parameters": {role: copy.deepcopy(record["parameters"]) for role, record in records.items()},
        "n_train": len(train),
        "n_test": len(test),
        "n_input_features": X.shape[1],
        "n_output_features": max(widths.values()),
        "n_output_features_semantics": "maximum_separate_component_plan_width_not_union",
        "component_output_features": widths,
        "outer_prediction_sha256": _prediction_hash(average),
        "outer_feature_names_sha256": tabular.digest(json.dumps(feature_hashes, sort_keys=True).encode()),
        "outer_feature_names_sha256_semantics": "role_keyed_component_feature_name_hashes_not_union",
        "numeric_clip_quantiles": [0.0, 1.0],
        "matched_baseline_method": "baseline-robust",
        **timings,
        "averaging_seconds": averaging_seconds,
        "average_scoring_seconds": scoring_seconds,
        "pipeline_seconds": sum(timings.values()),
        "regression_average": {
            "protocol": PROTOCOL,
            "weights": {"raw": 0.5, "fe": 0.5},
            "component_methods": dict(COMPONENTS),
            "complete_procedure_count": 2,
            "weight_optimized": False,
            "target_transform": "identity",
            "component_output_features": widths.copy(),
            "source_policy": "independent_unchanged_canonical_full_train_procedures_then_equal_prediction_mean",
            "interpretation": "experimental_fixed_average_no_causal_variance_reduction_or_no_harm_guarantee",
            "components": records,
        },
        "case_seconds": time.perf_counter() - started,
    }
    json.dumps(result, allow_nan=False)
    return [result]
