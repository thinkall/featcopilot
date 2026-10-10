"""Frozen unary-frequency allocation, shared semantics and fit-boundary tests."""

import json
import pickle
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from featcopilot.core.feature import Feature
from featcopilot.core.fitted_rule import FittedRuleSpec
from featcopilot.core.registry import registry
from featcopilot.engines.fitted_rules import FittedRuleEngine
from featcopilot.engines.predictive import PredictiveTabularEngine
from featcopilot.utils.frequency import apply_frequency_mapping, category_key, category_keys, fit_frequency_mapping

OPERATOR = "fitted_frequency_v1"


@pytest.fixture(autouse=True)
def bounded_threads():
    with threadpool_limits(limits=1):
        yield


class UnitRegressor(RegressorMixin, BaseEstimator):
    """Unit-test learner with deterministic importance and constant predictions."""

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        self.feature_importances_ = np.arange(1, X.shape[1] + 1, dtype=float)
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class UnitClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        self.feature_importances_ = np.arange(1, X.shape[1] + 1, dtype=float)
        self.classes_, counts = np.unique(y, return_counts=True)
        self.prior_ = counts / counts.sum()
        return self

    def predict_proba(self, X):
        return np.tile(self.prior_, (len(X), 1))

    def predict(self, X):
        return np.full(len(X), self.classes_[np.argmax(self.prior_)])


def frequency_engine(**kwargs):
    options = {
        "include_frequency": True,
        "screening": "importance",
        "estimator": UnitRegressor(),
        "task": "regression",
        "stability_repeats": 1,
        "max_candidates": 20,
    }
    options.update(kwargs)
    return PredictiveTabularEngine(**options)


def categorical_frame(n=120):
    assert n % 12 == 0
    return pd.DataFrame(
        {
            "left": ["private_a", "private_a", "private_a", "private_b", "private_b", "private_c"] * (n // 6),
            "right": ["u", "v", "u", "u"] * (n // 4),
        },
        index=pd.Index(np.arange(n) * 7 + 3, name="sample"),
    )


def plan(frame, **kwargs):
    fitted = frequency_engine(**kwargs)
    frame = fitted._validate_frame(frame)
    fitted._input_columns = frame.columns.tolist()
    fitted._numeric_columns = sorted(column for column in frame if pd.api.types.is_numeric_dtype(frame[column].dtype))
    features, _ = fitted._plan_candidates(frame, budget=50)
    return fitted, features


def favor_frequencies(model, frame, target, task, n_classes):
    """Force a stable selection outcome to test state lifecycles, not accuracy."""
    return 10.0 - 2.0 * sum(column.startswith("frequency_") for column in frame.columns)


def test_shared_frequency_semantics_and_fitted_rule_serialization_are_unchanged():
    values = pd.Series([1, "1", True, "__missing__", None, 1.0], dtype=object, index=[9, 9, 2, 1, 3, 8])
    expected = {
        '["number", "1"]': 2 / 6,
        '["string", "1"]': 1 / 6,
        '["boolean", true]': 1 / 6,
        '["string", "__missing__"]': 1 / 6,
    }
    mapping = fit_frequency_mapping(values)
    assert mapping == expected
    np.testing.assert_array_equal(apply_frequency_mapping(values, mapping), [2 / 6, 1 / 6, 1 / 6, 1 / 6, 0, 2 / 6])
    assert category_key(1) == category_key(1.0) == category_key(np.int64(1))
    assert len({category_key(1), category_key("1"), category_key(True)}) == 3
    assert category_keys(pd.Series([None, pd.NA, np.nan, np.inf, -np.inf])) == [None] * 5
    rule = FittedRuleSpec(name="frequency", operator="frequency", columns=["value"])
    X = values.to_frame("value")
    old_interface = FittedRuleEngine([rule]).fit(X)
    state = old_interface.to_dict()
    assert state["rules"][0]["mapping"] == expected
    restored = FittedRuleEngine.from_dict(json.loads(json.dumps(state, allow_nan=False)))
    assert_frame_equal(old_interface.transform(X), restored.transform(X))
    future = pd.Series([1, "new", None, np.inf], index=[8, 8, 1, 7], dtype=object)
    before = mapping.copy()
    np.testing.assert_array_equal(apply_frequency_mapping(future, mapping), [2 / 6, 0, 0, 0])
    assert mapping == before


@pytest.mark.parametrize("value", [[1, 2], {"secret": 1}, 1 + 2j, object()])
def test_shared_unsupported_values_raise_without_coercion_or_contents(value):
    series = pd.Series([value], dtype=object)
    for call in [lambda: fit_frequency_mapping(series), lambda: apply_frequency_mapping(series, {})]:
        with pytest.raises(ValueError, match="scalar strings, booleans, numbers") as error:
            call()
        assert "secret" not in str(error.value)
    with pytest.raises(ValueError, match="scalar strings, booleans, numbers"):
        FittedRuleEngine([FittedRuleSpec(name="f", operator="frequency", columns=["x"])]).fit(series.to_frame("x"))


def test_frequency_fit_requires_rows_and_registered_apply_requires_a_mapping():
    with pytest.raises(ValueError, match="nonempty"):
        fit_frequency_mapping(pd.Series([], dtype=object))
    apply = registry.get_transformation(OPERATOR)
    assert apply is apply_frequency_mapping
    with pytest.raises(TypeError):
        apply(pd.Series(["a"]))


@pytest.mark.parametrize("value", ["true", "false", 0, 1, None, np.bool_(True)])
def test_frequency_flag_is_strictly_boolean(value):
    with pytest.raises(ValueError, match="include_frequency"):
        frequency_engine(include_frequency=value)


@pytest.mark.parametrize("screening", ["auto", "linear", "tree"])
def test_frequency_never_silently_promotes_screening(screening):
    with pytest.raises(ValueError, match="requires explicit screening='importance'"):
        frequency_engine(screening=screening)


@pytest.mark.parametrize("candidate_set", ["standard", "extended"])
def test_disabled_frequency_preserves_pool_and_report_exactly(candidate_set):
    rng = np.random.default_rng(41)
    X = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a", "b", "c"])
    y = rng.normal(size=len(X))
    options = {
        "screening": "importance",
        "estimator": UnitRegressor(),
        "task": "regression",
        "stability_repeats": 1,
        "max_candidates": 30,
        "candidate_set": candidate_set,
    }
    default = PredictiveTabularEngine(**options).fit(X, y)
    disabled = PredictiveTabularEngine(**options, include_frequency=False).fit(X, y)
    assert default.selection_report_ == disabled.selection_report_
    assert "frequency_policy" not in disabled.selection_report_
    assert all("frequency_planning" not in repeat for repeat in disabled.selection_report_["repeats"])
    assert_frame_equal(default.transform(X), disabled.transform(X))


@pytest.mark.parametrize("candidate_set", ["standard", "extended"])
def test_all_categorical_ridge_input_still_plans_and_ranks_frequency_candidates(candidate_set):
    X = categorical_frame()
    X["left"] = pd.Categorical(X["left"])
    X["right"] = X["right"].astype("string")
    y = np.where(X["left"] == "private_a", 3.0, -1.0) + np.where(X["right"] == "u", 0.1, -0.2)
    supplied = Ridge(alpha=0.1)
    fitted = frequency_engine(estimator=supplied, candidate_set=candidate_set, max_features=4)
    assert X.select_dtypes(include=np.number).empty
    with patch.object(fitted, "_fit_learner", wraps=fitted._fit_learner) as learner:
        output = fitted.fit_transform(X, y)
    report = fitted.selection_report_
    assert fitted.include_frequency is fitted.config.include_frequency is True
    assert report["numeric_count"] == 0
    assert report["candidate_family_counts"][OPERATOR] == 2
    assert report["importance_ranking"]["retained_count"] >= 2
    assert report["importance_ranking"]["learner_fits"] == 1
    assert learner.call_count == report["learner_fit_count"]
    assert sum(column.startswith("frequency_") for column in learner.call_args_list[0].args[0]) == 2
    assert not hasattr(supplied, "coef_")
    assert_frame_equal(output[X.columns], X)
    assert_frame_equal(output, fitted.transform(X))


@pytest.mark.parametrize("candidate_set", ["standard", "extended"])
@pytest.mark.parametrize("budget", [0, 1, 9, 10, 20, 1000])
def test_quota_cap_base_prefix_and_unused_slots(candidate_set, budget):
    X = pd.DataFrame({f"x{i:03d}": [0, 0, 0, 1, 1, 2] * 4 for i in range(40)})
    enabled, planned = plan(X, candidate_set=candidate_set, max_candidates=budget)
    disabled, original = plan(X, candidate_set=candidate_set, max_candidates=budget, include_frequency=False)
    frequencies = [feature for feature in planned if feature.transformation == OPERATOR]
    base = [feature for feature in planned if feature.transformation != OPERATOR]
    quota = min(32, budget // 10, len(X.columns))
    assert len(frequencies) == quota
    assert len(planned) == budget
    assert [feature.to_dict() for feature in base] == [feature.to_dict() for feature in original[: budget - quota]]
    assert planned == base + frequencies
    assert len(base) >= budget - budget // 10
    expected_positions = (
        []
        if not quota
        else [len(X.columns) // 2] if quota == 1 else [i * (len(X.columns) - 1) // (quota - 1) for i in range(quota)]
    )
    assert [feature.source_columns[0] for feature in frequencies] == [X.columns[i] for i in expected_positions]
    _, permuted = plan(X[X.columns[::-1]].iloc[::-1], candidate_set=candidate_set, max_candidates=budget)
    assert [feature.to_dict() for feature in planned] == [feature.to_dict() for feature in permuted]
    continuous = pd.DataFrame({column: np.arange(24) + i for i, column in enumerate(X.columns)})
    _, with_empty_quota = plan(continuous, candidate_set=candidate_set, max_candidates=budget)
    _, without = plan(continuous, candidate_set=candidate_set, max_candidates=budget, include_frequency=False)
    assert [feature.to_dict() for feature in with_empty_quota] == [feature.to_dict() for feature in without]


def test_nonconstant_frequency_can_encode_missingness_and_supports_numeric_columns():
    X = pd.DataFrame({"numeric": [1, 1, 2, np.nan] * 10, "category": ["a", "a", "a", None] * 10})
    fitted, planned = plan(X)
    frequencies = [feature for feature in planned if feature.transformation == OPERATOR]
    assert len(frequencies) == 2
    by_source = {feature.source_columns[0]: fitted._compute(feature, X) for feature in frequencies}
    np.testing.assert_array_equal(by_source["numeric"], [0.5, 0.5, 0.25, 0] * 10)
    np.testing.assert_array_equal(by_source["category"], [0.75, 0.75, 0.75, 0] * 10)


def test_registry_export_json_pickle_and_fresh_process_parity():
    X = pd.DataFrame({"source'quoted": ["a'quote", "a'quote", "b", None] * 6})
    fitted, features = plan(X)
    feature = next(feature for feature in features if feature.transformation == OPERATOR)
    future = pd.DataFrame({"source'quoted": ["a'quote", "b", "new", None]}, index=[8, 8, 1, 2])
    expected = np.array([0.5, 0.25, 0, 0])
    np.testing.assert_array_equal(fitted._compute(feature, future), expected)
    np.testing.assert_array_equal(feature.compute(future), expected)
    exported = Feature.from_dict(json.loads(json.dumps(feature.to_dict(), allow_nan=False)))
    np.testing.assert_array_equal(exported.compute(future), expected)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(feature)).compute(future), expected)
    partitioned = pd.concat([exported.compute(future.iloc[:1]), exported.compute(future.iloc[1:])])
    assert_series_equal(partitioned, exported.compute(future))
    payload = {"feature": feature.to_dict(), "values": future.iloc[:, 0].tolist()}
    code = (
        "import json,sys,pandas as pd; from featcopilot.core.feature import Feature; "
        "p=json.load(sys.stdin); f=Feature.from_dict(p['feature']); "
        "v=f.compute(pd.DataFrame({f.source_columns[0]:p['values']})); "
        "assert v.tolist()==[0.5,0.25,0.0,0.0]"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=Path(__file__).resolve().parents[1],
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("repeats", [1, 3])
def test_repeat_states_are_independent_and_final_maps_use_full_fit_rows(task, repeats):
    X = categorical_frame()
    y = np.arange(len(X)) % 2 if task == "classification" else np.linspace(-1, 1, len(X))
    supplied = UnitClassifier() if task == "classification" else UnitRegressor()
    fitted = frequency_engine(task=task, estimator=supplied, stability_repeats=repeats, max_features=4)
    original_rank = fitted._rank_candidates
    captured = []

    def record_rank(candidates, raw, *args, **kwargs):
        ranking = original_rank(candidates, raw, *args, **kwargs)
        captured.append((pickle.dumps(candidates), candidates, raw.copy()))
        return ranking

    with (
        patch.object(fitted, "_rank_candidates", side_effect=record_rank),
        patch.object(fitted, "_loss", side_effect=favor_frequencies),
    ):
        result = fitted.fit_transform(X, y)
    assert fitted.selection_report_["selected_count"] == 2
    assert len(captured) == repeats + (repeats > 1)
    for before, candidates, raw in captured[:repeats]:
        assert pickle.dumps(candidates) == before
        for feature in candidates:
            if feature.transformation == OPERATOR:
                state = feature.metadata["frequency_state"]
                assert state["fit_rows"] == len(raw)
                assert state["mapping"] == fit_frequency_mapping(raw[feature.source_columns[0]])
    for feature in fitted.get_feature_set():
        assert feature.metadata["frequency_state"]["fit_rows"] == len(X)
        assert feature.metadata["frequency_state"]["scope"] == "final_fit"
        expected = apply_frequency_mapping(
            X[feature.source_columns[0]], fit_frequency_mapping(X[feature.source_columns[0]])
        )
        np.testing.assert_array_equal(result[feature.name], expected)
        np.testing.assert_array_equal(feature.compute(X), expected)
    expected_fits = repeats * (1 + 3) + (repeats > 1)
    assert fitted.selection_report_["learner_fit_count"] == expected_fits
    assert fitted.selection_report_["frequency_policy"]["final_refit_count"] == 2
    assert fitted.selection_report_["frequency_policy"]["final_refit_rows"] == len(X)
    assert_frame_equal(result[X.columns], X)
    assert_frame_equal(pickle.loads(pickle.dumps(fitted)).transform(X), result)
    assert_frame_equal(pd.concat([fitted.transform(X.iloc[:37]), fitted.transform(X.iloc[37:])]), result)
    assert "private_a" not in json.dumps(fitted.selection_report_)
    assert "private_a" in json.dumps(fitted.get_feature_set().to_dataframe().to_dict())


@pytest.mark.parametrize("repeats", [1, 3])
def test_final_refit_can_drop_a_now_constant_frequency_without_extra_learner_fits(repeats):
    X = pd.DataFrame({"category": ["a"] * 60 + ["b"] * 60})
    y = np.linspace(0, 1, len(X))
    fitted = frequency_engine(stability_repeats=repeats, verbose=True)
    partitions = []
    for index in range(repeats):
        train = np.r_[np.arange(50 - index), np.arange(60, 100 + index)]
        partitions.append((train, np.setdiff1d(np.arange(len(X)), train)))
    with (
        patch.object(fitted, "_holdouts", return_value=partitions),
        patch.object(fitted, "_loss", side_effect=favor_frequencies),
        patch("featcopilot.engines.predictive.logger") as logger,
    ):
        result = fitted.fit_transform(X, y)
    policy = fitted.selection_report_["frequency_policy"]
    assert policy["validated_selected_count"] == 1
    assert policy["final_refit_count"] == 1
    assert policy["final_discarded"][0]["reason"] == "constant"
    assert fitted.selection_report_["selected_count"] == 0
    assert fitted.selection_report_["ranking_scope"] == "baseline_retained_after_frequency_refit"
    assert fitted.selection_report_["learner_fit_count"] == repeats * 3
    assert_frame_equal(result, X)
    assert logger.info.called


def test_holdout_values_and_targets_never_enter_repeat_frequency_maps():
    X = categorical_frame()
    y = np.linspace(-1, 1, len(X))
    train, validation = train_test_split(np.arange(len(X)), test_size=0.25, random_state=42)
    changed = X.copy()
    changed.iloc[validation, 0] = "validation_only"
    fitted = frequency_engine()
    with patch.object(fitted, "_loss", return_value=10.0):
        fitted.fit(changed, y)
    assert fitted.selection_report_["frequency_policy"]["final_refit_count"] == 0
    _, features = plan(changed.iloc[train])
    assert "validation_only" not in json.dumps([feature.to_dict() for feature in features])
    assert fitted.selection_report_["frequency_policy"]["selected_frequency_count"] == 0
    changed_y = y.copy()
    changed_y[validation] += 1000
    second = frequency_engine()
    with patch.object(second, "_loss", return_value=10.0):
        second.fit(changed, changed_y)
    assert fitted.selection_report_["candidate_scores"] == second.selection_report_["candidate_scores"]
    assert fitted.selection_report_["repeats"] == second.selection_report_["repeats"]


@pytest.mark.parametrize("kind", ["grouped", "temporal"])
def test_frequency_fits_obey_internal_group_time_and_label_availability_boundaries(kind):
    X = categorical_frame(240)
    y = np.linspace(-1, 1, len(X))
    metadata = (
        {"groups": np.repeat(np.arange(12), 20)}
        if kind == "grouped"
        else {"timestamps": np.repeat(np.arange(24), 10), "label_end_times": np.repeat(np.arange(24), 10) + 1}
    )
    fitted = frequency_engine(stability_repeats=3)
    boundaries = fitted._validate_boundaries(
        metadata.get("groups"), metadata.get("timestamps"), len(X), label_end_times=metadata.get("label_end_times")
    )
    partitions = fitted._holdouts(y, "regression", boundaries)
    observed = []
    original = fit_frequency_mapping

    def record_fit(series):
        observed.append(set(series.index))
        return original(series)

    with (
        patch("featcopilot.engines.predictive.fit_frequency_mapping", side_effect=record_fit),
        patch.object(fitted, "_loss", return_value=10.0),
    ):
        fitted.fit(X, y, **metadata)
    assert len(observed) == 6
    for repeat, (train, validation) in enumerate(partitions):
        assert observed[2 * repeat] == observed[2 * repeat + 1] == set(X.index[train])
        assert observed[2 * repeat].isdisjoint(X.index[validation])
    assert fitted.selection_report_["frequency_policy"]["final_refit_count"] == 0


def test_output_budget_remains_truthful_including_the_existing_fifty_cap():
    X = pd.DataFrame({f"x{i:03d}": [0, 0, 0, 1, 1, 2] * 10 for i in range(60)})
    y = np.linspace(-1, 1, len(X))
    for allowance in [0, 1, 5, 50, 100]:
        fitted = frequency_engine(max_features=len(X.columns) + allowance, max_candidates=100)

        # Supply distinct arrays to avoid a correlated fixture shrinking the
        # candidate count; this test isolates the existing count-grid contract.
        def no_dedup(candidates, raw, prepared, fitted=fitted):
            arrays = {feature.name: fitted._compute(feature, raw) for feature in candidates}
            return list(candidates), arrays, []

        with (
            patch.object(fitted, "_deduplicate_candidates", side_effect=no_dedup),
            patch.object(fitted, "_loss", side_effect=lambda model, frame, *args: 1000.0 - frame.shape[1]),
        ):
            output = fitted.fit_transform(X, y)
        expected = min(allowance, 50)
        assert fitted.selection_report_["selected_count"] == expected
        assert output.shape[1] == X.shape[1] + expected
        assert max(score["count"] for score in fitted.selection_report_["validation_scores"]) == expected
        assert fitted.selection_report_["frequency_policy"]["effective_output_cap"] == expected


def test_failed_refit_zero_budget_and_hash_collisions_do_not_keep_stale_frequency_state():
    X = categorical_frame()
    y = np.linspace(-1, 1, len(X))
    fitted = frequency_engine()
    with patch.object(fitted, "_loss", side_effect=favor_frequencies):
        fitted.fit(X, y)
    assert fitted.get_feature_names()
    with pytest.raises(ValueError, match="y is required"):
        fitted.fit(X)
    assert not fitted.is_fitted and fitted.selection_report_ == {}
    assert len(fitted.get_feature_set()) == 0
    zero = frequency_engine(max_candidates=0).fit(X, y)
    assert zero.selection_report_["learner_fit_count"] == 1
    assert zero.selection_report_["repeats"][0]["frequency_planning"]["quota"] == 0
    assert_frame_equal(zero.transform(X), X)
    _, features = plan(X)
    collision = next(feature.name for feature in features if feature.transformation == OPERATOR)
    with pytest.raises(ValueError, match="collision"):
        plan(X.assign(**{collision: 0.0}), max_candidates=100)
