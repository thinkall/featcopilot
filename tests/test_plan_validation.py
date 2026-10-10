"""Synthetic isolation, frozen-state and matched-control checks for the plan gate."""

import json
import pickle
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline

from benchmarks import plan_validation as protocol
from benchmarks import tabular
from featcopilot.core.feature import Feature
from featcopilot.engines.predictive import PredictiveTabularEngine
from featcopilot.utils.frequency import fit_frequency_mapping
from featcopilot.utils.splitting import make_holdout_splits


def _case(task="regression", learner="linear", *, nonlinear=True):
    rng = np.random.default_rng(142)
    X = pd.DataFrame(
        {
            "number": rng.uniform(-3, 3, 160),
            "other": rng.normal(size=160),
            "category": ["private-common", "private-common", "private-common", "private-rare"] * 40,
            "flag": np.arange(160) % 2 == 0,
        },
        index=np.arange(160) + 1000,
    )
    signal = X.number**2 if nonlinear else 3 * X.number
    y = pd.Series((signal > signal.median()).astype(int) if task == "classification" else signal, index=X.index)
    return {
        "X": X,
        "y": y,
        "config": {
            "seed": 42,
            "threads": 1,
            "learners": [learner],
            "max_candidates": 10,
            "max_added_features": 2,
        },
        "dataset": "synthetic-plan-gate",
        "source": "synthetic",
        "task": task,
        "method": "featcopilot-plan-gated-robust",
        "seed": 42,
        "fold": 0,
        "split_sha256": "test-outer-split",
        "train_idx": np.arange(120),
        "test_idx": np.arange(120, 160),
    }


def _partition(case):
    train = case["train_idx"]
    d, g = make_holdout_splits(case["y"].iloc[train], case["task"], 1, 104771, test_size=0.25)[0]
    return train[d], train[g]


def _fixed_discovery(frame, target, learner, parameters, task, config):
    square = PredictiveTabularEngine._feature("square_number", ["number"], "square", "x ** 2")
    frequency = PredictiveTabularEngine._frequency_feature(
        "category", fit_frequency_mapping(frame.category), len(frame), "final_fit"
    )
    plan = protocol.FrozenPlan(protocol._schema(frame), (square, frequency))
    return plan, {"selected_features": plan.names, "selected_count": len(plan.names)}


def _empty_discovery(frame, target, learner, parameters, task, config):
    return protocol.FrozenPlan(protocol._schema(frame), ()), {"selected_features": [], "selected_count": 0}


def test_gate_uses_strict_one_percent_margin_and_se_is_only_diagnostic():
    raw = np.ones(4)
    boundary = protocol._gate_decision(raw, np.full(4, 0.99), 1)
    assert boundary["reason"] == "insufficient_gain"
    assert not boundary["accepted"]
    assert protocol._gate_decision(raw, np.full(4, np.nextafter(0.99, 0.0)), 1)["accepted"]
    # Large uncertainty is deliberately not an additional significance threshold.
    noisy = protocol._gate_decision(np.array([100.0, 0.0]), np.array([0.0, 98.0]), 1)
    assert noisy["accepted"]
    assert noisy["paired_gain_standard_error"] > noisy["paired_mean_gain"]
    assert noisy["standard_error_used_for_decision"] is False
    assert protocol._gate_decision(np.zeros(4), np.zeros(4), 1)["reason"] == "zero_raw_loss"
    assert protocol._gate_decision(raw, np.zeros(4), 0)["reason"] == "no_features"


@pytest.mark.parametrize(
    "raw,candidate",
    [
        ([np.nan, 1], [1, 1]),
        ([1, 1], [np.inf, 1]),
        ([1, -1], [1, 1]),
        ([1], [0]),
        ([1, 1], [1]),
        ([[1, 1]], [[1, 1]]),
        ([1e308, 1e308], [1, 1]),
    ],
)
def test_invalid_or_infeasible_gate_raises(raw, candidate):
    with pytest.raises(ValueError):
        protocol._gate_decision(raw, candidate, 1)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("classes", [2, 3])
def test_additive_log_loss_matches_sklearn_precision_without_probability_repair(dtype, classes):
    probabilities = np.full((6, classes), 1 / classes, dtype=dtype)
    probabilities[0] = 0
    probabilities[0, 0] = 1
    probabilities[1] = 0
    probabilities[1, -1] = 1
    targets = pd.Series(np.arange(6) % classes)
    prediction = protocol._Predictions(np.argmax(probabilities, axis=1), probabilities, np.arange(classes))
    losses = protocol._additive_losses(prediction, targets, "classification")
    assert losses.mean() == pytest.approx(
        log_loss(targets, probabilities, labels=np.arange(classes)), rel=4 * np.finfo(dtype).eps
    )
    original = probabilities.copy()
    probabilities[2, 0] += 2 * np.finfo(dtype).eps
    protocol._additive_losses(prediction, targets, "classification")
    assert probabilities[2, 0] != original[2, 0]
    assert probabilities[2].sum() != 1  # Validation did not silently normalize.
    probabilities[2] *= 0.99
    with pytest.raises(ValueError, match="normalized"):
        protocol._additive_losses(prediction, targets, "classification")


@pytest.mark.parametrize(
    "probability",
    [
        np.array([[1.1, -0.1], [0.5, 0.5]]),
        np.array([[np.nan, 0.5], [0.5, 0.5]]),
        np.array([[np.inf, 0.0], [0.5, 0.5]]),
        np.array([[0.3, 0.3], [0.5, 0.5]]),
        np.array([[1], [0]]),
        np.array([0.5, 0.5]),
        np.array([[1, 0], [0, 1]]),
    ],
)
def test_invalid_classification_predictions_raise(probability):
    prediction = protocol._Predictions(np.array([0, 1]), probability, np.array([0, 1]))
    with pytest.raises(ValueError):
        protocol._additive_losses(prediction, pd.Series([0, 1]), "classification")


@pytest.mark.parametrize("prediction", [np.array([np.nan, 1.0]), np.array([[1.0], [2.0]]), np.array([1e308, 1.0])])
def test_invalid_regression_predictions_raise(prediction):
    with pytest.raises(ValueError):
        protocol._additive_losses(protocol._Predictions(prediction), pd.Series([0.0, 1.0]), "regression")


def test_score_caches_multiclass_column_labels_and_probabilities():
    frame = pd.DataFrame({"number": np.arange(6.0)})
    target = pd.Series([0, 1, 2, 0, 1, 2])
    probabilities = np.full((len(frame), 3), 0.1)
    probabilities[np.arange(len(frame)), target] = 0.8
    model = Mock(classes_=np.arange(3))
    model.predict.return_value = target.to_numpy()[:, None]
    model.predict_proba.return_value = probabilities

    scores, losses = protocol._score(model, frame, target, "classification")

    assert scores["loss"] == pytest.approx(0.2231435513142097)
    np.testing.assert_allclose(losses, -np.log(0.8))
    assert scores["accuracy"] == scores["balanced_accuracy"] == scores["macro_f1"] == 1.0
    model.predict.assert_called_once_with(frame)
    model.predict_proba.assert_called_once_with(frame)
    assert model.predict.return_value.shape == (len(frame), 1)


@pytest.mark.parametrize(
    "task,shape",
    [
        ("classification", (6, 2)),
        ("classification", (6, 1, 1)),
        ("classification", (1, 6)),
        ("classification", (5, 1)),
        ("regression", (6, 1)),
        ("regression", (6, 1, 1)),
    ],
)
def test_score_does_not_flatten_malformed_prediction_shapes(task, shape):
    frame = pd.DataFrame({"number": np.arange(6.0)})
    model = Mock(classes_=np.arange(3))
    model.predict.return_value = np.zeros(shape)
    model.predict_proba.return_value = np.full((len(frame), 3), 1 / 3)
    with pytest.raises(ValueError, match="prediction shape"):
        protocol._score(model, frame, pd.Series([0, 1, 2, 0, 1, 2]), task)


def test_frozen_frequency_refit_preserves_constants_duplicates_order_schema_and_pickle():
    discovery = pd.DataFrame({"number": [1.0, 2.0, 3.0, 4.0], "category": ["a", "a", "a", "b"]})
    frequency = PredictiveTabularEngine._frequency_feature(
        "category", fit_frequency_mapping(discovery.category), len(discovery), "final_fit"
    )
    constant = Feature(
        name="constant", source_columns=["number"], transformation="constant", code="result = df['number'] * 0 + 0.25"
    )
    square = PredictiveTabularEngine._feature("square_number", ["number"], "square", "x ** 2")
    plan = protocol.FrozenPlan(protocol._schema(discovery), (frequency, constant, square))
    original_dicts = [feature.to_dict() for feature in plan.features]
    all_train = pd.DataFrame(
        {
            "number": np.r_[np.arange(4.0), [1e15, -1e15, 4, 5, 6, 7, 8, 9]],
            "category": ["a"] * 3 + ["b"] * 3 + ["gate-only-c"] * 3 + ["gate-only-d"] * 3,
        },
        index=np.arange(12) + 40,
    )
    refitted = plan.refit_frequency(all_train)
    assert refitted.identity == plan.identity
    assert refitted.state_hash != plan.state_hash
    assert refitted.names == plan.names
    assert [feature.to_dict() for feature in plan.features] == original_dicts
    transformed = refitted.transform(all_train)
    assert transformed[frequency.name].nunique() == 1
    pd.testing.assert_series_equal(transformed[frequency.name], transformed.constant, check_names=False)
    assert list(transformed.columns) == ["number", "category", *plan.names]
    assert plan.transform(all_train).loc[46, frequency.name] == 0
    assert transformed.loc[46, frequency.name] == 0.25
    pd.testing.assert_frame_equal(
        transformed,
        pd.concat([refitted.transform(all_train.iloc[:5]), refitted.transform(all_train.iloc[5:])]),
    )
    pd.testing.assert_frame_equal(transformed, pickle.loads(pickle.dumps(refitted)).transform(all_train))
    changed = deepcopy(square)
    changed.code = changed.code.replace("x ** 2", "x ** 3")
    assert protocol.FrozenPlan(plan.input_schema, (frequency, constant, changed)).identity != plan.identity
    assert protocol.FrozenPlan(plan.input_schema, (square, constant, frequency)).identity != plan.identity
    with pytest.raises(ValueError, match="schema"):
        refitted.transform(all_train[["category", "number"]])
    with pytest.raises(ValueError, match="schema"):
        refitted.transform(all_train.assign(number=all_train.number.astype(str)))


@pytest.mark.parametrize(
    "code,error",
    [
        ("result = pd.Series([1.0] * len(df))", "index or length"),
        ("result = df['number'].iloc[:-1]", "index or length"),
        ("result = np.ones(len(df))", "index or length"),
        ("result = df['number'].astype(str)", "numeric"),
        ("result = df['number'] * np.inf", "infinite"),
        ("raise pd.errors.ParserError('broken formula')", "broken formula"),
    ],
)
def test_frozen_transform_never_swallows_bad_features(code, error):
    frame = pd.DataFrame({"number": [1.0, 2.0]}, index=[4, 5])
    plan = protocol.FrozenPlan(protocol._schema(frame), (Feature("broken", source_columns=["number"], code=code),))
    with pytest.raises(ValueError, match=error):
        plan.transform(frame)


def test_frequency_identity_exemption_cannot_hide_rewritten_formulas():
    frame = pd.DataFrame({"category": ["a", "b"]})
    feature = PredictiveTabularEngine._frequency_feature(
        "category", fit_frequency_mapping(frame.category), 2, "final_fit"
    )
    feature.code = "result = df['category'].map(lambda value: 0.0)"
    with pytest.raises(ValueError, match="registered formula"):
        protocol.FrozenPlan(protocol._schema(frame), (feature,))


@pytest.mark.parametrize("source", ["payload", "config"])
@pytest.mark.parametrize("metadata", ["groups", "timestamps", "label_end_times"])
def test_non_iid_is_rejected_without_fitting(monkeypatch, source, metadata):
    case = _case()
    (case if source == "payload" else case["config"])[metadata] = np.arange(len(case["X"]))
    monkeypatch.setattr(protocol, "_tune", lambda *args, **kwargs: pytest.fail("Non-IID inputs reached tuning"))
    with pytest.raises(ValueError, match="IID only"):
        protocol.run_case(case, Path("."))


def test_invalid_outer_boundary_and_infeasible_split_propagate(monkeypatch):
    case = _case()
    case["test_idx"][0] = case["train_idx"][0]
    with pytest.raises(ValueError, match="disjoint"):
        protocol.run_case(case, Path("."))
    case = _case("classification")
    case["y"].iloc[:120] = 0
    case["y"].iloc[0] = 1
    with pytest.raises(ValueError):
        protocol.run_case(case, Path("."))


def test_gate_row_values_and_labels_cannot_reach_any_discovery_fit_or_pre_gate_predictions(monkeypatch):
    case = _case()
    discovery_idx, gate_idx = _partition(case)
    d_labels = set(case["X"].index[discovery_idx])
    t_labels = set(case["X"].index[case["train_idx"]])
    inner, _ = make_holdout_splits(case["y"].iloc[discovery_idx], "regression", 1, 42)[0]
    inner_labels = set(case["X"].index[discovery_idx[inner]])
    sentinel = protocol._prepare(case["X"].iloc[discovery_idx[:7]])
    original_fit, original_discover = Pipeline.fit, protocol._discover
    original_score, original_gate = protocol._score, protocol._gate_decision
    original_engine_fit = PredictiveTabularEngine.fit
    original_frequency = protocol.fit_frequency_mapping
    runs, state = [], {}

    def record_fit(model, frame, target=None, **kwargs):
        if isinstance(frame, pd.DataFrame):
            indices = set(frame.index)
            assert indices <= (t_labels if state["gate_done"] else d_labels)
            if not state["gate_done"]:
                assert indices <= state["allowed_labels"]
            state["fit_indices"].append(tuple(frame.index))
        result = original_fit(model, frame, target, **kwargs)
        if isinstance(frame, pd.DataFrame) and set(frame.index) == d_labels and hasattr(model.steps[-1][1], "predict"):
            state["d_models"].append(model)
        return result

    def record_engine(engine, frame, target=None, **kwargs):
        assert not state["gate_done"], "Selector fit after gate"
        assert set(frame.index) <= d_labels
        return original_engine_fit(engine, frame, target, **kwargs)

    def record_frequency(series):
        assert set(series.index) <= (t_labels if state["gate_done"] else d_labels)
        state["frequency_fit_indices"].append(tuple(series.index))
        return original_frequency(series)

    def record_discover(*args):
        state["allowed_labels"] = set(args[0].index)
        plan, report = original_discover(*args)
        state["plans"].append((plan.identity, plan.state_hash, plan.transform(sentinel)))
        state["latest_plan"] = plan
        return plan, report

    def record_score(model, frame, target, task):
        if set(frame.index) == set(case["X"].index[gate_idx]) and not state["sentinel_predictions"]:
            raw_model, candidate_model = state["d_models"][-2:]
            state["sentinel_predictions"] = [
                raw_model.predict(sentinel),
                candidate_model.predict(state["latest_plan"].transform(sentinel)),
            ]
        return original_score(model, frame, target, task)

    def record_gate(*args):
        assert not state["gate_done"]
        state["gate_done"] = True
        return original_gate(*args)

    monkeypatch.setattr(Pipeline, "fit", record_fit)
    monkeypatch.setattr(PredictiveTabularEngine, "fit", record_engine)
    monkeypatch.setattr(protocol, "_discover", record_discover)
    monkeypatch.setattr(protocol, "_score", record_score)
    monkeypatch.setattr(protocol, "_gate_decision", record_gate)
    monkeypatch.setattr(protocol, "fit_frequency_mapping", record_frequency)
    monkeypatch.setattr("featcopilot.engines.predictive.fit_frequency_mapping", record_frequency)
    for perturb in (False, True):
        state = {
            "gate_done": False,
            "fit_indices": [],
            "frequency_fit_indices": [],
            "d_models": [],
            "plans": [],
            "sentinel_predictions": [],
            "allowed_labels": inner_labels,
        }
        current = deepcopy(case)
        if perturb:
            current["X"].iloc[gate_idx, current["X"].columns.get_loc("number")] = 1e12
            current["X"].iloc[gate_idx, current["X"].columns.get_loc("category")] = "gate-only-secret"
            current["y"].iloc[gate_idx] += 1e6
        row = protocol.run_case(current, Path("."))[0]
        runs.append((row, state))
    first, second = runs[0][0], runs[1][0]
    assert first["plan_gate"]["raw_mean"] != second["plan_gate"]["raw_mean"]
    assert first["raw_inner_pipeline_trials"] == second["raw_inner_pipeline_trials"]
    assert first["inner_pipeline_trials"] == second["inner_pipeline_trials"]
    assert first["candidate_discovery_selection_report"] == second["candidate_discovery_selection_report"]
    for key in (
        "raw_parameters",
        "candidate_parameters",
        "plan_identity_before_refit",
        "learned_state_sha256_before_refit",
    ):
        assert first["plan_gate"][key] == second["plan_gate"][key]
    assert len(runs[0][1]["plans"]) == len(runs[1][1]["plans"]) == 4
    for (identity, mapping, values), (identity2, mapping2, values2) in zip(
        runs[0][1]["plans"], runs[1][1]["plans"], strict=True
    ):
        assert (identity, mapping) == (identity2, mapping2)
        pd.testing.assert_frame_equal(values, values2)
    for before, after in zip(runs[0][1]["sentinel_predictions"], runs[1][1]["sentinel_predictions"], strict=True):
        np.testing.assert_array_equal(before, after)
    assert runs[0][1]["fit_indices"]
    assert runs[0][1]["frequency_fit_indices"]


@pytest.mark.parametrize("real_discovery", [False, True])
def test_outer_test_cannot_affect_plans_gate_tuning_or_accepted_refit(monkeypatch, real_discovery):
    if not real_discovery:
        monkeypatch.setattr(protocol, "_discover", _fixed_discovery)
    case = _case()
    first = protocol.run_case(case, Path("."))[0]
    changed = deepcopy(case)
    changed["X"].iloc[case["test_idx"], 0] = 1e8
    changed["X"].iloc[case["test_idx"], 2] = "outer-test-private"
    changed["y"].iloc[case["test_idx"]] += 100
    second = protocol.run_case(changed, Path("."))[0]
    assert first["plan_gate"] == second["plan_gate"]
    assert first["plan_gate"]["accepted"]
    assert first["inner_pipeline_trials"] == second["inner_pipeline_trials"]
    assert first["raw_inner_pipeline_trials"] == second["raw_inner_pipeline_trials"]
    assert first["parameters"] == second["parameters"]
    assert first["loss"] != second["loss"]


def test_fixed_classification_partition_gate_perturbation_changes_only_decision(monkeypatch):
    case = _case("classification")
    discovery_idx, gate_idx = _partition(case)
    original_holdouts = protocol.make_holdout_splits

    def fixed_gate(target, task, count, seed, **kwargs):
        if seed == 104771:
            return [(discovery_idx.copy(), gate_idx.copy())]
        return original_holdouts(target, task, count, seed, **kwargs)

    monkeypatch.setattr(protocol, "make_holdout_splits", fixed_gate)
    monkeypatch.setattr(protocol, "_discover", _fixed_discovery)
    first = protocol.run_case(case, Path("."))[0]
    case["y"].iloc[gate_idx] = 1 - case["y"].iloc[gate_idx]
    case["X"].iloc[gate_idx, 2] = "unseen-gate-only"
    second = protocol.run_case(case, Path("."))[0]
    assert first["plan_gate"]["accepted"] and not second["plan_gate"]["accepted"]
    for key in (
        "raw_parameters",
        "candidate_parameters",
        "plan_identity_before_refit",
        "learned_state_sha256_before_refit",
    ):
        assert first["plan_gate"][key] == second["plan_gate"][key]
    assert first["raw_inner_pipeline_trials"] == second["raw_inner_pipeline_trials"]
    assert first["inner_pipeline_trials"] == second["inner_pipeline_trials"]
    assert first["candidate_discovery_selection_report"] == second["candidate_discovery_selection_report"]
    for result in (first, second):
        gate = result["plan_gate"]
        assert gate["additive_loss"] == "negative_log_true_class_probability"
        assert gate["raw_mean"] == pytest.approx(gate["raw_primary_loss"], abs=1e-14)
        assert gate["candidate_mean"] == pytest.approx(gate["candidate_primary_loss"], abs=1e-14)


def test_accepted_gate_refits_only_frequency_then_model_and_reports_delivered_features(monkeypatch):
    case = _case()
    _, gate_idx = _partition(case)
    case["X"].iloc[gate_idx[:2], 2] = "gate-only-private"
    case["X"].iloc[gate_idx[:2], 1] = 1e12
    original_gate = protocol._gate_decision
    original_refit = protocol.FrozenPlan.refit_frequency
    done, refits, discoveries, evaluations = False, [], [], []
    original_score = protocol._score

    def discover(*args):
        assert not done, "Discovery after the decision is forbidden"
        result = _fixed_discovery(*args)
        discoveries.append(result[0])
        return result

    def gate(*args):
        nonlocal done
        assert not done
        done = True
        return original_gate(*args)

    def refit(plan, frame):
        assert done
        assert list(frame.index) == list(case["X"].index[case["train_idx"]])
        result = original_refit(plan, frame)
        refits.append((plan, result))
        return result

    def score(model, frame, target, task):
        evaluations.append(tuple(frame.index))
        return original_score(model, frame, target, task)

    monkeypatch.setattr(protocol, "_discover", discover)
    monkeypatch.setattr(protocol, "_gate_decision", gate)
    monkeypatch.setattr(protocol.FrozenPlan, "refit_frequency", refit)
    monkeypatch.setattr(protocol, "_score", score)
    row = protocol.run_case(case, Path("."))[0]
    diag = row["plan_gate"]
    assert len(discoveries) == 4
    assert len(refits) == 1
    assert diag["accepted"] and diag["gate_evaluations"] == 1
    assert diag["plan_identity_before_refit"] == diag["plan_identity_after_refit"]
    assert diag["learned_state_sha256_before_refit"] != diag["learned_state_sha256_after_refit"]
    assert row["outer_selection_report"]["selected_features"] == diag["delivered_feature_names"]
    assert row["n_output_features"] == row["n_input_features"] + diag["delivered_count"]
    assert evaluations.count(tuple(case["X"].index[gate_idx])) == 2  # One prediction per locked model.
    assert evaluations.count(tuple(case["X"].index[case["test_idx"]])) == 1
    assert row["pipeline_seconds"] == sum(row[key] for key in protocol._TIMINGS)
    assert 0 < row["pipeline_seconds"] <= row["case_seconds"]
    serialized = json.dumps(row, allow_nan=False)
    for secret in ("private-common", "private-rare", "gate-only-private", "mapping", "category_pair_values"):
        assert secret not in serialized


@pytest.mark.parametrize("empty", [False, True])
def test_rejected_candidate_exactly_matches_independently_tuned_raw_control(monkeypatch, empty):
    case = _case(nonlinear=False)
    monkeypatch.setattr(protocol, "_discover", _empty_discovery if empty else _fixed_discovery)
    predictions = []
    original_score = protocol._score

    def record(model, frame, target, task):
        if list(frame.index) == list(case["X"].index[case["test_idx"]]):
            predictions.append(model.predict(frame))
        return original_score(model, frame, target, task)

    monkeypatch.setattr(protocol, "_score", record)
    candidate = protocol.run_case(case, Path("."))[0]
    monkeypatch.setattr(protocol, "_discover", lambda *args: pytest.fail("Raw control attempted feature discovery"))
    baseline = protocol.run_case({**case, "method": "baseline-discovery-robust"}, Path("."))[0]
    assert candidate["plan_gate"]["reason"] == ("no_features" if empty else "insufficient_gain")
    assert candidate["plan_gate"]["final_branch"] == "raw"
    assert not candidate["plan_gate"]["frequency_state_refitted"]
    assert candidate["parameters"] == baseline["parameters"] == candidate["plan_gate"]["raw_parameters"]
    assert candidate["inner_losses"] == baseline["inner_losses"]
    assert candidate["raw_inner_pipeline_trials"] == baseline["raw_inner_pipeline_trials"]
    assert candidate["loss"] == baseline["loss"]
    assert candidate["mae"] == baseline["mae"]
    assert candidate["r2"] == baseline["r2"]
    np.testing.assert_array_equal(*predictions)
    assert candidate["outer_selection_report"]["selected_features"] == []
    assert candidate["outer_selection_report"]["selected_count"] == 0
    assert candidate["n_output_features"] == case["X"].shape[1]
    assert baseline["plan_gate"]["reason"] == "not_requested"
    assert baseline["plan_gate"]["gate_evaluations"] == 0


def test_invalid_gate_prediction_is_not_a_success_shaped_raw_fallback(monkeypatch):
    monkeypatch.setattr(protocol, "_discover", _fixed_discovery)
    original = protocol._score
    case = _case()
    _, gate_idx = _partition(case)
    gate_labels = set(case["X"].index[gate_idx])

    def invalid(model, frame, target, task):
        if set(frame.index) == gate_labels:
            return {"loss": 1.0}, np.full(len(frame), np.nan)
        return original(model, frame, target, task)

    monkeypatch.setattr(protocol, "_score", invalid)
    with pytest.raises(ValueError, match="finite"):
        protocol.run_case(case, Path("."))


def test_rejection_never_substitutes_candidate_hyperparameters_for_stronger_raw_control(monkeypatch):
    case = _case(nonlinear=False)
    monkeypatch.setattr(protocol, "_discover", _empty_discovery)
    original_tune = protocol._tune

    def distinct_winners(*args, **kwargs):
        result = original_tune(*args, **kwargs)
        if kwargs["engineered"]:
            # Explicitly exercise different D-selected winners, including a
            # weaker candidate regularization, without changing raw selection.
            result["parameters"] = {"regularization": 10.0}
        return result

    monkeypatch.setattr(protocol, "_tune", distinct_winners)
    candidate = protocol.run_case(case, Path("."))[0]
    baseline = protocol.run_case({**case, "method": "baseline-discovery-robust"}, Path("."))[0]
    assert candidate["plan_gate"]["candidate_parameters"] != candidate["plan_gate"]["raw_parameters"]
    assert candidate["parameters"] == baseline["parameters"] == candidate["plan_gate"]["raw_parameters"]
    assert candidate["loss"] == baseline["loss"]


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("learner", ["linear", "lightgbm", "catboost", "xgboost"])
def test_real_discovery_and_downstream_models_support_all_four_synthetic_learners(task, learner, monkeypatch):
    if learner != "linear":
        pytest.importorskip(learner)
    case = _case(task, learner)
    frame = protocol._prepare(case["X"].iloc[:100])
    target = case["y"].iloc[:100]
    config = {**case["config"], "numeric_clip_quantiles": (0.0, 1.0)}
    original_init = PredictiveTabularEngine.__init__
    received = []

    def capture(engine, *args, **kwargs):
        received.append(kwargs)
        original_init(engine, *args, **kwargs)

    monkeypatch.setattr(PredictiveTabularEngine, "__init__", capture)
    with protocol.threadpool_limits(limits=1):
        plan, report = protocol._discover(
            frame, target, learner, tabular.candidate_parameters(learner)[0], task, config
        )
        transformed = plan.transform(frame)
        test = plan.transform(protocol._prepare(case["X"].iloc[120:]))
        model = tabular.make_model(transformed, task, learner, tabular.candidate_parameters(learner)[0], config)
        model.fit(transformed, target)
        scores, losses = protocol._score(model, test, case["y"].iloc[120:], task)
    assert np.isfinite(losses).all()
    assert np.isfinite(scores["loss"])
    assert received[0]["stability_repeats"] == 3
    assert received[0]["importance_resamples"] == 1
    assert received[0]["candidate_set"] == "extended"
    assert received[0]["selection_rule"] == "paired_gain"
    assert received[0]["screening"] == "importance"
    assert received[0]["numeric_clip_quantiles"] == (0.0, 1.0)
    assert received[0]["include_frequency"] is True
    assert received[0]["task"] == task
    if learner == "catboost":
        assert "cat_features" not in received[0]["estimator"].get_params()
        assert model.steps[-1][1].get_params()["cat_features"] == ["category"]
    assert report["stability_repeats"] == 3
    assert "private-common" not in json.dumps(report)


@pytest.mark.parametrize("method", ["baseline-discovery-robust", "featcopilot-plan-gated-robust"])
def test_real_catboost_multiclass_protocol_end_to_end(method):
    pytest.importorskip("catboost")
    case = _case("classification", "catboost")
    case["method"] = method
    case["X"] = case["X"].iloc[:96].copy()
    case["y"] = pd.Series(np.digitize(case["X"].number, [-1.0, 1.0]), index=case["X"].index)
    case["train_idx"], case["test_idx"] = np.arange(72), np.arange(72, 96)

    row = protocol.run_case(case, Path("."))[0]

    assert row["status"] == "success"
    assert all(np.isfinite(row[metric]) for metric in ("loss", "accuracy", "balanced_accuracy", "macro_f1", "roc_auc"))
    assert len(row["raw_inner_pipeline_trials"]) == 3
    assert row["n_train"] == 72 and row["n_test"] == 24
    gate = row["plan_gate"]
    if method == "featcopilot-plan-gated-robust":
        assert gate["gate_evaluations"] == 1 and gate["decision_valid"]
        assert len(row["inner_pipeline_trials"]) == 3
        assert gate["raw_mean"] == pytest.approx(gate["raw_primary_loss"])
        assert gate["candidate_mean"] == pytest.approx(gate["candidate_primary_loss"])
    else:
        assert gate["gate_evaluations"] == 0 and gate["reason"] == "not_requested"
