"""Experimental IID discovery/plan gate, with a separately tuned matched raw control.

Only a single discovery plan sees the gate. The gate can select that plan or the
raw branch, not change formulas, ranks, hyperparameters, or feature counts.
This module is benchmark-only; the tabular supervisor owns resource limits.
"""

import json
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from featcopilot.core.feature import Feature
from featcopilot.engines.predictive import PredictiveTabularEngine
from featcopilot.utils.frequency import fit_frequency_mapping
from featcopilot.utils.splitting import make_holdout_splits, validate_split_boundaries

_RAW_METHOD = "baseline-discovery-robust"
_PLAN_METHOD = "featcopilot-plan-gated-robust"
_FREQUENCY = "fitted_frequency_v1"
_MARGIN = 0.01
_TIMINGS = ("inner_fe_seconds", "outer_fe_seconds", "tuning_seconds", "model_fit_seconds", "predict_seconds")


def _hash(value: Any) -> str:
    from benchmarks.tabular import digest

    return digest(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8"))


def _indices_hash(indices: np.ndarray) -> str:
    from benchmarks.tabular import digest

    return digest(np.asarray(indices, dtype="<i8").tobytes())


def _schema(frame: pd.DataFrame) -> tuple[tuple[str, str], ...]:
    if not isinstance(frame, pd.DataFrame) or not len(frame.columns):
        raise ValueError("Plan inputs must be a dataframe with at least one column")
    if not frame.columns.is_unique or not all(isinstance(name, str) for name in frame.columns):
        raise ValueError("Plan input schema must have unique string column names")
    if any(pd.api.types.is_complex_dtype(dtype) for dtype in frame.dtypes):
        raise ValueError("Complex feature values are not supported")
    return tuple((column, str(dtype)) for column, dtype in frame.dtypes.items())


@dataclass(frozen=True)
class FrozenPlan:
    """
    Benchmark-internal ordered feature snapshot, not a refittable selector.

    Parameters
    ----------
    input_schema : tuple of (str, str)
        Names and normalized dtypes in their original order.
    features : tuple of Feature
        Trusted features copied from a fitted predictive discovery engine.
        Only registered unary frequency state may subsequently be refitted.
    """

    input_schema: tuple[tuple[str, str], ...]
    features: tuple[Feature, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", deepcopy(self.features))
        inputs = [name for name, _ in self.input_schema]
        names = inputs + self.names
        if not inputs or len(names) != len(set(names)):
            raise ValueError("Frozen plan schema contains missing or colliding feature names")
        for feature in self.features:
            if not feature.code or not set(feature.source_columns).issubset(inputs):
                raise ValueError("Frozen plan requires trusted formulas over original columns")
            if feature.transformation == _FREQUENCY:
                if len(feature.source_columns) != 1:
                    raise ValueError("Only unary frequency state can be refitted")
                mapping = PredictiveTabularEngine._frequency_mapping(feature)
                state = feature.metadata["frequency_state"]
                expected = PredictiveTabularEngine._frequency_feature(
                    feature.source_columns[0], mapping, state["fit_rows"], state["scope"]
                )
                expected.importance = feature.importance
                if feature.to_dict() != expected.to_dict():
                    raise ValueError("Frequency feature does not match the registered formula")

    @classmethod
    def from_engine(cls, engine: PredictiveTabularEngine, frame: pd.DataFrame) -> "FrozenPlan":
        """Snapshot a fitted discovery engine without retaining its fitted selector."""
        return cls(_schema(frame), tuple(engine.get_feature_set()))

    @property
    def names(self) -> list[str]:
        """Get generated feature names in frozen delivery order."""
        return [feature.name for feature in self.features]

    @property
    def identity(self) -> str:
        """Hash formulas, order and schema, excluding only registered frequency state."""
        definitions = []
        for feature in self.features:
            definition = deepcopy(feature.to_dict())
            if feature.transformation == _FREQUENCY:
                definition.pop("code")
                state = definition["metadata"]["frequency_state"]
                # Version is fixed; fit scope/count and the mapping are learned state.
                definition["metadata"]["frequency_state"] = {"version": state["version"]}
            definitions.append(definition)
        return _hash({"input_schema": self.input_schema, "features": definitions})

    @property
    def state_hash(self) -> str:
        """Hash learned state without exposing vocabulary, values, or rows."""
        return _hash(
            [
                {
                    "name": feature.name,
                    "state": feature.metadata["frequency_state"],
                }
                for feature in self.features
                if feature.transformation == _FREQUENCY
            ]
        )

    def _validate_schema(self, frame: pd.DataFrame) -> None:
        if _schema(frame) != self.input_schema:
            raise ValueError("Frozen plan input schema differs from discovery")

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Compute exact formulas without learning, dropping, or repairing features.

        Parameters
        ----------
        frame : pandas.DataFrame
            Already normalized rows with the exact discovery schema.

        Returns
        -------
        pandas.DataFrame
            Original columns followed by every frozen feature in order.
            Missing numeric results remain missing for learner preprocessing.
        """
        self._validate_schema(frame)
        generated = {}
        for feature in self.features:
            values = feature.compute(frame)
            if not isinstance(values, pd.Series) or len(values) != len(frame) or not values.index.equals(frame.index):
                raise ValueError("Frozen feature changed row index or length")
            if not pd.api.types.is_numeric_dtype(values.dtype) or pd.api.types.is_complex_dtype(values.dtype):
                raise ValueError("Frozen generated features must remain real numeric columns")
            values = values.astype(float)
            if np.isinf(values.to_numpy()).any():
                raise ValueError("Frozen feature produced infinite values")
            generated[feature.name] = values
        result = pd.concat([frame.copy(), pd.DataFrame(generated, index=frame.index)], axis=1)
        result.columns.name = frame.columns.name
        if (
            not result.index.equals(frame.index)
            or list(result.columns) != [name for name, _ in self.input_schema] + self.names
        ):
            raise ValueError("Frozen plan changed the delivery schema")
        return result

    def refit_frequency(self, frame: pd.DataFrame) -> "FrozenPlan":
        """
        Refit only unsupervised unary frequency mappings, preserving every column.

        Parameters
        ----------
        frame : pandas.DataFrame
            All allowed outer-training rows, with no targets.

        Returns
        -------
        FrozenPlan
            Independent snapshot with the same identity and order. Constant or
            duplicate columns created by mapping refits are deliberately kept.
        """
        self._validate_schema(frame)
        features = []
        for feature in self.features:
            if feature.transformation == _FREQUENCY:
                column = feature.source_columns[0]
                # Scoped adapter to the engine's shared formula generator; no fitted
                # engine state, candidate planner, or deduplicator is modified.
                replacement = PredictiveTabularEngine._frequency_feature(
                    column, fit_frequency_mapping(frame[column]), len(frame), "final_fit"
                )
                replacement.importance = feature.importance
                features.append(replacement)
            else:
                features.append(feature)
        result = FrozenPlan(self.input_schema, tuple(features))
        if result.identity != self.identity:
            raise ValueError("State-only refit changed the frozen plan identity")
        return result


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    from benchmarks.tabular import prepare_numeric

    _schema(frame)
    return prepare_numeric(frame, frame.iloc[:0], impute=False, preserve_boolean=True)[0]


@dataclass
class _Predictions:
    prediction: np.ndarray
    probability: np.ndarray | None = None
    classes_: np.ndarray | None = None

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.prediction

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.probability is None:
            raise ValueError("No classification probabilities were captured")
        return self.probability


def _additive_losses(predictions: _Predictions, y: pd.Series, task: str) -> np.ndarray:
    target = np.asarray(y)
    prediction = predictions.prediction
    if target.ndim != 1 or prediction.shape != (len(target),) or not len(target) or pd.isna(target).any():
        raise ValueError("Invalid target or prediction shape")
    if task == "regression":
        if not np.isfinite(target.astype(float)).all() or not np.isfinite(prediction).all():
            raise ValueError("Regression targets and predictions must be finite")
        with np.errstate(over="ignore", invalid="ignore"):
            losses = np.square(target.astype(float) - prediction.astype(float))
    elif task == "classification":
        probability, classes = predictions.probability, predictions.classes_
        if (
            classes is None
            or classes.ndim != 1
            or len(classes) < 2
            or pd.isna(classes).any()
            or (np.issubdtype(classes.dtype, np.number) and not np.isfinite(classes).all())
            or not np.array_equal(classes, np.unique(classes))
            or probability is None
            or probability.shape != (len(target), len(classes))
            or not np.issubdtype(probability.dtype, np.floating)
        ):
            raise ValueError("Invalid probability shape, dtype, or classes")
        positions = pd.Index(classes).get_indexer(target)
        if (positions < 0).any() or not np.isin(prediction, classes).all():
            raise ValueError("Targets and predictions must belong to the fitted classes")
        epsilon = np.finfo(probability.dtype).eps
        if (
            not np.isfinite(probability).all()
            or (probability < 0).any()
            or (probability > 1).any()
            or not np.allclose(probability.sum(axis=1), 1.0, rtol=0, atol=max(1e-12, 8 * epsilon))
        ):
            raise ValueError("Invalid probabilities: require finite bounds and normalized sums")
        # Match sklearn's dtype-aware clipping, not arbitrary epsilon or silent
        # renormalization of invalid scores. Convert only after clipping.
        true_probability = probability[np.arange(len(target)), positions]
        losses = -np.log(np.clip(true_probability, epsilon, 1 - epsilon).astype(float))
    else:
        raise ValueError("Plan validation requires an explicit classification or regression task")
    if not np.isfinite(losses).all() or (losses < 0).any():
        raise ValueError("Additive losses must be finite and nonnegative")
    return np.asarray(losses, dtype=float)


def _score(model: Any, frame: pd.DataFrame, target: pd.Series, task: str) -> tuple[dict, np.ndarray]:
    from benchmarks.tabular import metrics

    prediction = np.asarray(model.predict(frame))
    if task == "classification" and prediction.shape == (len(frame), 1):
        prediction = prediction[:, 0]
    predictions = _Predictions(prediction)
    if task == "classification":
        predictions.probability = np.asarray(model.predict_proba(frame))
        predictions.classes_ = np.asarray(model.classes_)
    losses = _additive_losses(predictions, target, task)
    # metrics consumes cached outputs: each fitted model predicts each evaluation
    # frame once, including the single gate and the original outer test.
    return metrics(predictions, frame, target, task), losses


def _gate_decision(raw: np.ndarray, candidate: np.ndarray, proposed_count: int) -> dict:
    raw, candidate = np.asarray(raw, dtype=float), np.asarray(candidate, dtype=float)
    if raw.ndim != 1 or raw.shape != candidate.shape or len(raw) < 2:
        raise ValueError("The gate requires at least two aligned per-example losses")
    if not np.isfinite(raw).all() or not np.isfinite(candidate).all() or (raw < 0).any() or (candidate < 0).any():
        raise ValueError("Gate losses must be finite and nonnegative")
    with np.errstate(over="ignore", invalid="ignore"):
        raw_mean, candidate_mean = float(raw.mean()), float(candidate.mean())
        differences = raw - candidate
        gain = float(differences.mean())
        standard_error = float(np.sqrt(np.var(differences, ddof=1) / len(raw)))
    if not all(np.isfinite(value) for value in (raw_mean, candidate_mean, gain, standard_error)):
        raise ValueError("Gate diagnostics must be finite")
    accepted = proposed_count > 0 and raw_mean > 0 and candidate_mean < (1 - _MARGIN) * raw_mean
    reason = (
        "no_features"
        if proposed_count == 0
        else "zero_raw_loss" if raw_mean == 0 else "accepted" if accepted else "insufficient_gain"
    )
    return {
        "decision_valid": True,
        "accepted": bool(accepted),
        "reason": reason,
        "raw_mean": raw_mean,
        "candidate_mean": candidate_mean,
        "paired_mean_gain": gain,
        "paired_gain_standard_error": standard_error,
        "standard_error_used_for_decision": False,
        "final_branch": "candidate" if accepted else "raw",
    }


@contextmanager
def _timed(timings: dict[str, float], stage: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[stage] += time.perf_counter() - start


def _discover(
    frame: pd.DataFrame, target: pd.Series, learner: str, parameters: dict, task: str, config: dict
) -> tuple[FrozenPlan, dict[str, Any]]:
    from benchmarks.tabular import make_model, selection_diagnostics

    estimator = make_model(frame, task, learner, parameters, config).steps[-1][1]
    if learner == "catboost":
        estimator = type(estimator)(
            **{key: value for key, value in estimator.get_params().items() if key != "cat_features"}
        )
    engine = PredictiveTabularEngine(
        estimator=estimator,
        task=task,
        candidate_set="extended",
        include_frequency=True,
        screening="importance",
        selection_rule="paired_gain",
        importance_resamples=1,
        max_candidates=config.get("max_candidates", 1000),
        max_features=frame.shape[1] + config.get("max_added_features", 50),
        stability_repeats=3,
        random_state=config["seed"],
        n_jobs=config["threads"],
        numeric_clip_quantiles=(0.0, 1.0),
    ).fit(frame, target)
    selection_report: dict[str, Any] = dict(engine.selection_report_)
    report = selection_diagnostics(selection_report, False)
    if report is None:
        raise RuntimeError("Discovery did not produce a selection report")
    return FrozenPlan.from_engine(engine, frame), report


def _tune(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    train_y: pd.Series,
    valid_y: pd.Series,
    learner: str,
    task: str,
    config: dict,
    timings: dict[str, float],
    *,
    engineered: bool,
) -> dict[str, Any]:
    from benchmarks.tabular import candidate_parameters, make_model

    trials = []
    for parameters in candidate_parameters(learner):
        plan, report = None, None
        train_X, valid_X = train, valid
        if engineered:
            with _timed(timings, "inner_fe_seconds"):
                plan, report = _discover(train, train_y, learner, parameters, task, config)
                train_X, valid_X = plan.transform(train), plan.transform(valid)
        with _timed(timings, "tuning_seconds"):
            model = make_model(train_X, task, learner, parameters, config).fit(train_X, train_y)
            scores, _ = _score(model, valid_X, valid_y, task)
            trials.append(
                {
                    "parameters": deepcopy(parameters),
                    "loss": scores["loss"],
                    "n_output_features": train_X.shape[1],
                    "selection_report": report,
                    "plan_identity": None if plan is None else plan.identity,
                    "learned_state_sha256": None if plan is None else plan.state_hash,
                }
            )
    if len(trials) != 3:
        raise ValueError("The plan protocol requires exactly three declared learner settings")
    best = int(np.argmin([trial["loss"] for trial in trials]))
    return {
        "parameters": deepcopy(trials[best]["parameters"]),
        "inner_losses": [trial["loss"] for trial in trials],
        "selection_report": trials[best]["selection_report"],
        "trials": trials,
    }


def run_case(payload: dict, workspace: Path) -> list[dict]:
    """
    Evaluate one hydrated IID benchmark cell with D-only tuning and one plan gate.

    Parameters
    ----------
    payload : dict
        Existing tabular cell metadata plus X, y, train_idx, test_idx and config.
        Config must name exactly one learner; no loading or process spawning
        occurs here. Grouped or chronological metadata is rejected.
    workspace : pathlib.Path
        Supervisor-owned workspace, retained for the shared entry signature.
        This protocol writes no artifacts or private feature state.

    Returns
    -------
    list of dict
        One standard tabular success row with compact discovery trial reports
        and ``plan_gate`` diagnostics. Invalid inputs, splits, or predictions
        raise; they never become successful raw-control scores.
    """
    from benchmarks.tabular import make_model, selection_diagnostics

    started = time.perf_counter()
    X, y = payload["X"], payload["y"]
    config = {**payload["config"], "numeric_clip_quantiles": (0.0, 1.0)}
    task, method = payload["task"], payload["method"]
    if method not in {_RAW_METHOD, _PLAN_METHOD} or task not in {"classification", "regression"}:
        raise ValueError("Unknown plan-validation method or task")
    if len(config["learners"]) != 1:
        raise ValueError("Plan validation requires exactly one learner per cell")
    for name in ("groups", "timestamps", "label_end_times"):
        if payload.get(name) is not None or config.get(name) is not None:
            raise ValueError("Plan validation supports IID only; group/time metadata is not allowed")
    _schema(X)
    if not isinstance(y, pd.Series) or len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("X and y must be positionally and index aligned")
    train_idx, test_idx = np.asarray(payload["train_idx"]), np.asarray(payload["test_idx"])
    validate_split_boundaries(train_idx, test_idx, n_rows=len(X))
    learner = config["learners"][0]
    timings = dict.fromkeys(_TIMINGS, 0.0)
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    with threadpool_limits(limits=config["threads"]):
        with _timed(timings, "outer_fe_seconds"):
            gate_seed = (config["seed"] + 104729) % 2**32
            d, g = make_holdout_splits(y.iloc[train_idx], task, 1, gate_seed, test_size=0.25)[0]
            discovery_idx, gate_idx = train_idx[d], train_idx[g]
            if len(gate_idx) < 2:
                raise ValueError("The gate needs at least two rows")
            discovery_y = y.iloc[discovery_idx]
            inner_train, inner_valid = make_holdout_splits(discovery_y, task, 1, config["seed"], test_size=0.25)[0]
            discovery = _prepare(X.iloc[discovery_idx])
            train, valid = discovery.iloc[inner_train], discovery.iloc[inner_valid]
            train_y, valid_y = discovery_y.iloc[inner_train], discovery_y.iloc[inner_valid]
            gate: dict[str, Any] = {
                "protocol": "discovery_plan_gate_v1",
                "requested": method == _PLAN_METHOD,
                "discovery_rows": len(discovery_idx),
                "gate_rows": len(gate_idx),
                "inner_train_rows": len(inner_train),
                "inner_valid_rows": len(inner_valid),
                "discovery_sha256": _indices_hash(discovery_idx),
                "gate_sha256": _indices_hash(gate_idx),
                "inner_train_sha256": _indices_hash(discovery_idx[inner_train]),
                "inner_valid_sha256": _indices_hash(discovery_idx[inner_valid]),
                "gate_split_seed": gate_seed,
                "inner_split_seed": config["seed"],
                "gate_fraction": 0.25,
                "inner_valid_fraction": 0.25,
                "relative_margin": _MARGIN,
                "additive_loss": "negative_log_true_class_probability" if task == "classification" else "squared_error",
                "tuning_policy": "independent_three_setting_searches_on_discovery_only",
                "source_policy": (
                    "discovery_only_plan_then_train_only_unsupervised_frequency_refit"
                    if method == _PLAN_METHOD
                    else "all_training_raw_with_discovery_tuned_parameters"
                ),
                "gate_evaluations": 0,
                "accepted": False,
                "decision_valid": None,
                "reason": "not_requested",
                "final_branch": "raw",
                "proposed_feature_names": [],
                "proposed_count": 0,
                "delivered_feature_names": [],
                "delivered_count": 0,
                "candidate_parameters": None,
                "plan_identity_before_refit": None,
                "plan_identity_after_refit": None,
                "learned_state_sha256_before_refit": None,
                "learned_state_sha256_after_refit": None,
            }
        raw = _tune(train, valid, train_y, valid_y, learner, task, config, timings, engineered=False)
        candidate: dict[str, Any] | None = None
        plan: FrozenPlan | None = None
        discovery_report: dict[str, Any] | None = None
        gate["raw_parameters"] = deepcopy(raw["parameters"])
        if method == _PLAN_METHOD:
            candidate = _tune(train, valid, train_y, valid_y, learner, task, config, timings, engineered=True)
            with _timed(timings, "outer_fe_seconds"):
                plan, discovery_report = _discover(
                    discovery, discovery_y, learner, candidate["parameters"], task, config
                )
                candidate_D = plan.transform(discovery)
                gate.update(
                    candidate_parameters=deepcopy(candidate["parameters"]),
                    proposed_feature_names=plan.names,
                    proposed_count=len(plan.names),
                    plan_identity_before_refit=plan.identity,
                    learned_state_sha256_before_refit=plan.state_hash,
                )
            with _timed(timings, "model_fit_seconds"):
                raw_model = make_model(discovery, task, learner, raw["parameters"], config).fit(discovery, discovery_y)
                candidate_model = make_model(candidate_D, task, learner, candidate["parameters"], config).fit(
                    candidate_D, discovery_y
                )
            # No G values have been materialized before all discovery decisions
            # and both D-trained models are complete.
            with _timed(timings, "outer_fe_seconds"):
                gate_X, gate_y = _prepare(X.iloc[gate_idx]), y.iloc[gate_idx]
                candidate_G = plan.transform(gate_X)
            with _timed(timings, "predict_seconds"):
                raw_scores, raw_losses = _score(raw_model, gate_X, gate_y, task)
                candidate_scores, candidate_losses = _score(candidate_model, candidate_G, gate_y, task)
                gate.update(_gate_decision(raw_losses, candidate_losses, len(plan.names)))
                gate.update(
                    gate_evaluations=1,
                    raw_primary_loss=raw_scores["loss"],
                    candidate_primary_loss=candidate_scores["loss"],
                )

        chosen = raw
        delivered_plan: FrozenPlan | None = None
        with _timed(timings, "outer_fe_seconds"):
            train_X = _prepare(X.iloc[train_idx])
            if gate["accepted"]:
                if plan is None or candidate is None:
                    raise RuntimeError("Accepted gate requires a discovered plan and candidate parameters")
                chosen = candidate
                delivered_plan = plan.refit_frequency(train_X)
                gate.update(
                    delivered_feature_names=delivered_plan.names,
                    delivered_count=len(delivered_plan.names),
                    plan_identity_after_refit=delivered_plan.identity,
                    learned_state_sha256_after_refit=delivered_plan.state_hash,
                )
                train_X = delivered_plan.transform(train_X)
            elif plan is not None:
                gate["plan_identity_after_refit"] = plan.identity
                gate["learned_state_sha256_after_refit"] = plan.state_hash
            gate["frequency_refit_count"] = (
                sum(feature.transformation == _FREQUENCY for feature in delivered_plan.features)
                if delivered_plan is not None
                else 0
            )
            gate["frequency_state_refitted"] = gate["frequency_refit_count"] > 0
            gate["frequency_refit_rows"] = len(train_idx) if gate["frequency_state_refitted"] else 0
            gate["delivered_source_policy"] = (
                "frozen_discovery_formulas_with_all_training_frequency_state"
                if gate["accepted"]
                else "all_training_raw_with_discovery_tuned_parameters"
            )
        with _timed(timings, "model_fit_seconds"):
            model = make_model(train_X, task, learner, chosen["parameters"], config).fit(train_X, y.iloc[train_idx])
        with _timed(timings, "outer_fe_seconds"):
            test_X = _prepare(X.iloc[test_idx])
            if delivered_plan is not None:
                test_X = delivered_plan.transform(test_X)
            if not train_X.columns.equals(test_X.columns):
                raise ValueError("Final training and test schemas differ")
        with _timed(timings, "predict_seconds"):
            scores, _ = _score(model, test_X, y.iloc[test_idx], task)

    outer_report: dict[str, Any] = {
        "selected_features": gate["delivered_feature_names"],
        "selected_count": gate["delivered_count"],
        "original_count": X.shape[1],
        "selection_scope": "delivered_branch_after_independent_plan_gate",
        "final_branch": gate["final_branch"],
    }
    base = {key: payload[key] for key in ("dataset", "source", "task", "method", "seed", "fold", "split_sha256")}
    return [
        {
            **base,
            "family_id": payload.get("family_id", payload["dataset"]),
            "role": payload.get("role", "development"),
            "expected_cells": payload.get("expected_cells", 0),
            "learner": learner,
            "status": "success",
            **scores,
            "parameters": deepcopy(chosen["parameters"]),
            "inner_losses": chosen["inner_losses"],
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_input_features": X.shape[1],
            "n_output_features": train_X.shape[1],
            **timings,
            "pipeline_seconds": sum(timings.values()),
            "case_seconds": time.perf_counter() - started,
            "inner_selection_report": chosen["selection_report"],
            "outer_selection_report": selection_diagnostics(outer_report, False),
            "candidate_discovery_selection_report": discovery_report,
            "inner_pipeline_trials": [] if candidate is None else candidate["trials"],
            "raw_inner_pipeline_trials": raw["trials"],
            "plan_gate": gate,
            "numeric_clip_quantiles": [0.0, 1.0],
            "matched_baseline_method": _RAW_METHOD,
            **{
                f"{scope}_{field}": None
                for scope in ("inner", "outer")
                for field in ("llm_diagnostics", "llm_plan", "reference_diagnostics")
            },
        }
    ]
