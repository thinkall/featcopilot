"""Leakage, schema and serialization contracts for bounded fitted rules."""

import json
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.model_selection import KFold

from featcopilot.core.fitted_rule import FittedRulePlan, FittedRuleSpec
from featcopilot.engines.fitted_rules import FittedRuleEngine


def rule(operator, columns=None, name=None, **parameters):
    return FittedRuleSpec(
        name=name or operator,
        operator=operator,
        columns=columns or ["category"],
        parameters=parameters,
        explanation=f"Frozen {operator}",
    )


@pytest.mark.parametrize(
    "operator,columns,parameters",
    [
        ("execute", ["x"], {}),
        ("standardize", ["x"], {"ddof": 1}),
        ("frequency", ["x"], {"unknown": "learn"}),
        ("group_mean", ["x"], {}),
        ("group_mean", ["x", "x"], {}),
        ("target_mean", ["x", "y"], {}),
        ("target_mean", ["x"], {"smoothing": -1}),
        ("target_mean", ["x"], {"smoothing": "10"}),
        ("target_mean", ["x"], {"smoothing": True}),
        ("target_mean", ["x"], {"smoothing": np.inf}),
        ("clip", ["x"], {"lower_quantile": -0.1}),
        ("clip", ["x"], {"lower_quantile": 0.9, "upper_quantile": 0.1}),
        ("rare_category", ["x"], {"min_frequency": 2}),
        ("rare_category", ["x"], {"other_label": ""}),
        ("frequency", [""], {}),
    ],
)
def test_rule_parameters_reject_invalid_contracts(operator, columns, parameters):
    with pytest.raises(ValidationError):
        FittedRuleSpec(name="f", operator=operator, columns=columns, parameters=parameters)


def test_spec_defaults_and_json_round_trip():
    spec = rule("clip", ["x"])
    assert spec.parameters == {"lower_quantile": 0.01, "upper_quantile": 0.99}
    assert FittedRuleSpec.model_validate_json(spec.model_dump_json()) == spec
    with pytest.raises(ValidationError, match="Extra inputs"):
        FittedRuleSpec(name="f", operator="frequency", columns=["x"], code="result = 1")
    with pytest.raises(ValidationError, match="blank"):
        rule("frequency", name=" ")


def test_plan_budget_and_unique_names():
    with pytest.raises(ValidationError, match="unique"):
        FittedRulePlan(rules=[rule("frequency"), rule("frequency")])
    with pytest.raises(ValidationError, match="100"):
        FittedRulePlan(rules=[rule("frequency", name=f"f{i}") for i in range(101)])


def test_all_operators_fit_only_on_training_and_preserve_schema_and_index():
    X = pd.DataFrame(
        {"category": pd.Categorical(["a", "a", "b", None]), "value": [0.0, 2.0, 10.0, np.nan]},
        index=[9, 9, 1, 4],
    )
    rules = [
        rule("standardize", ["value"]),
        rule("clip", ["value"], lower_quantile=0.0, upper_quantile=1.0),
        rule("frequency"),
        rule("group_mean", ["category", "value"]),
        rule("target_mean", smoothing=0.0),
        rule("rare_category", min_frequency=0.4),
    ]
    engine = FittedRuleEngine(rules, task="regression", target_cv=2)
    training = engine.fit_transform(X, np.array([0.0, 2.0, 6.0, 8.0]))
    pd.testing.assert_frame_equal(training[X.columns], X)
    assert list(training.columns) == list(X.columns) + engine.get_feature_names()
    test = pd.DataFrame({"category": ["a", "b", "new", None], "value": [100.0, -20.0, 5.0, np.nan]}, index=[7, 7, 3, 0])
    output = engine.transform(test)
    np.testing.assert_allclose(
        output["standardize"],
        [(100 - 4) / np.std([0, 2, 10]), (-20 - 4) / np.std([0, 2, 10]), 1 / np.std([0, 2, 10]), 0],
    )
    assert output["clip"].tolist() == [10.0, 0.0, 5.0, 0.0]
    assert output["frequency"].tolist() == [0.5, 0.25, 0.0, 0.0]
    np.testing.assert_allclose(output["group_mean"], [1.0, 10.0, 4.0, 4.0])
    assert output["target_mean"].tolist() == [1.0, 6.0, 4.0, 4.0]
    assert output["rare_category"].tolist() == ["a", "__other__", "__other__", "__other__"]
    pd.testing.assert_frame_equal(output[X.columns], test)
    before = engine.to_dict()
    pd.testing.assert_frame_equal(output, pd.concat([engine.transform(test.iloc[:1]), engine.transform(test.iloc[1:])]))
    assert engine.to_dict() == before
    assert all(engine.get_feature_metadata()[spec.name]["explanation"] == spec.explanation for spec in rules)


def test_target_encodings_are_fold_local_and_positionally_aligned():
    X = pd.DataFrame({"category": ["a", "a", "a", "b", "b", None]}, index=[3, 3, 0, 2, 8, 9])
    y = pd.Series([1.0, 5.0, 9.0, 20.0, 50.0, 60.0], index=[90, 80, 70, 60, 50, 40])
    engine = FittedRuleEngine([rule("target_mean", smoothing=3.0)], task="regression", target_cv=3)
    output = engine.fit_transform(X, y)
    expected = np.empty(len(X))
    for train, valid in KFold(n_splits=3, shuffle=True, random_state=42).split(X):
        train_y = y.iloc[train].to_numpy()
        prior = train_y.mean()
        for position in valid:
            category = X.iloc[position, 0]
            members = train_y[(X.iloc[train, 0] == category).to_numpy()] if category is not None else np.array([])
            expected[position] = (members.sum() + 3 * prior) / (len(members) + 3)
    np.testing.assert_allclose(output["target_mean"], expected)
    assert output.index.equals(X.index)
    np.testing.assert_allclose(
        output["target_mean"],
        FittedRuleEngine([rule("target_mean", smoothing=3.0)], task="regression", target_cv=3).fit_transform(
            X.reset_index(drop=True), y.to_numpy()
        )["target_mean"],
    )
    assert not np.allclose(output["target_mean"], engine.transform(X)["target_mean"])


def test_changing_own_target_cannot_change_own_training_encoding():
    X = pd.DataFrame({"category": ["a", "a", "b", "b", "singleton", None]})
    y = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
    rules = [rule("target_mean", smoothing=2.0)]
    original = FittedRuleEngine(rules, task="regression", target_cv=3).fit_transform(X, y)
    for row in range(len(X)):
        changed = y.copy()
        changed[row] = 9000
        output = FittedRuleEngine(rules, task="regression", target_cv=3).fit_transform(X, changed)
        assert output["target_mean"].iloc[row] == original["target_mean"].iloc[row]


def test_target_singletons_use_other_fold_global_mean_not_full_training_prior():
    X = pd.DataFrame({"category": ["a", "b", None, "d"]})
    y = np.array([0.0, 2.0, 4.0, 10.0])
    output = FittedRuleEngine([rule("target_mean", smoothing=0.0)], task="regression", target_cv=4).fit_transform(X, y)
    np.testing.assert_allclose(output["target_mean"], (y.sum() - y) / 3)


def test_binary_target_means_are_probabilities_and_record_class_mapping():
    X = pd.DataFrame({"category": ["a", "a", "b", "b"]})
    engine = FittedRuleEngine([rule("target_mean", smoothing=0.0)], task="classification", target_cv=2)
    output = engine.fit_transform(X, np.array(["yes", "no", "yes", "yes"]))
    assert output["target_mean"].between(0, 1).all()
    assert engine.transform(X)["target_mean"].tolist() == [0.5, 0.5, 1.0, 1.0]
    assert engine.to_dict()["target_classes"] == ["no", "yes"]


@pytest.mark.parametrize(
    "target,task,message",
    [
        (None, "auto", "requires a target"),
        (np.array([0, 1]), "auto", "match X"),
        (np.array([[0], [1], [0]]), "auto", "one-dimensional"),
        (np.array([0.0, np.nan, 1.0]), "regression", "missing"),
        (np.array([0.0, np.inf, 1.0]), "regression", "finite"),
        (np.array(["a", "b", "c"]), "classification", "binary classification only"),
        (np.array([0, 1, 2]), "auto", "multiclass"),
        (np.array(["1", "2", "3"]), "regression", "numeric"),
    ],
)
def test_invalid_targets_raise_helpfully(target, task, message):
    engine = FittedRuleEngine([rule("target_mean")], task=task)
    with pytest.raises(ValueError, match=message):
        engine.fit(pd.DataFrame({"category": ["a", "b", "c"]}), target)
    assert not engine.is_fitted


@pytest.mark.parametrize("metadata", ["groups", "timestamps", "label_end_times"])
def test_unsupervised_rules_do_not_consume_split_metadata(metadata):
    X = pd.DataFrame({"category": ["a", "a", "b", "b"]})
    FittedRuleEngine([rule("frequency")]).fit(X, **{metadata: np.arange(4)})


def test_missing_constant_and_one_class_fallbacks_are_finite_and_logged():
    X = pd.DataFrame({"category": ["a", None, "a"], "constant": [4.0] * 3, "missing": [None] * 3})
    rules = [
        rule("standardize", ["constant"]),
        rule("clip", ["missing"]),
        rule("group_mean", ["category", "missing"]),
        rule("target_mean"),
    ]
    engine = FittedRuleEngine(rules, task="classification", target_cv=5)
    with patch("featcopilot.engines.fitted_rules.logger.warning") as warning:
        output = engine.fit_transform(X, np.array(["yes"] * 3))
    assert warning.call_count >= 4
    assert np.isfinite(output[engine.get_feature_names()].to_numpy()).all()
    assert (output[engine.get_feature_names()] == 0).all().all()


def test_numerical_extremes_are_finite_and_batch_invariant():
    extreme = np.finfo(float).max
    X = pd.DataFrame({"x": [-extreme, extreme]})
    engine = FittedRuleEngine(
        [rule("standardize", ["x"]), rule("clip", ["x"], lower_quantile=0.25, upper_quantile=0.75)]
    ).fit(X)
    output = engine.transform(X)
    np.testing.assert_allclose(output["standardize"], [-1, 1])
    np.testing.assert_allclose(output["clip"], [-extreme / 2, extreme / 2])
    assert np.isfinite(output.to_numpy()).all()
    pd.testing.assert_frame_equal(output, pd.concat([engine.transform(X.iloc[:1]), engine.transform(X.iloc[1:])]))
    tiny = pd.DataFrame({"x": [0.0, 1e-300]})
    saturated = FittedRuleEngine([rule("standardize", ["x"])]).fit(tiny).transform(X)
    assert np.isfinite(saturated["standardize"]).all()


def test_categorical_keys_do_not_merge_strings_numbers_booleans_or_missing_sentinels():
    X = pd.DataFrame({"category": pd.Series([1, "1", True, "__missing__", None, 1.0], dtype=object)})
    engine = FittedRuleEngine([rule("frequency")]).fit(X)
    output = engine.transform(X)
    np.testing.assert_allclose(output["frequency"], [2 / 6, 1 / 6, 1 / 6, 1 / 6, 0, 2 / 6])
    restored = FittedRuleEngine.from_dict(json.loads(json.dumps(engine.to_dict(), allow_nan=False)))
    pd.testing.assert_frame_equal(output, restored.transform(X))


def test_rare_categories_validate_strings_and_reserved_label():
    with pytest.raises(ValueError, match="string categories"):
        FittedRuleEngine([rule("rare_category")]).fit(pd.DataFrame({"category": [1, 2]}))
    with pytest.raises(ValueError, match="collides"):
        FittedRuleEngine([rule("rare_category")]).fit(pd.DataFrame({"category": ["a", "__other__"]}))
    engine = FittedRuleEngine([rule("rare_category", min_frequency=0.5)]).fit(
        pd.DataFrame({"category": pd.Categorical(["a", "a", "b"], categories=["a", "b", "unused"])})
    )
    output = engine.transform(pd.DataFrame({"category": ["b", "unused", "a", None]}))
    assert output["rare_category"].tolist() == ["__other__", "__other__", "a", "__other__"]


@pytest.mark.parametrize(
    "frame,message",
    [
        (pd.DataFrame([[1, 2]], columns=["x", "x"]), "unique"),
        (pd.DataFrame({1: [1]}), "string"),
        (pd.DataFrame({"frequency": ["a"]}), "collide"),
        (pd.DataFrame({"other": ["a"]}), "missing source"),
        (pd.DataFrame({"category": []}), "non-empty"),
    ],
)
def test_invalid_training_schema(frame, message):
    with pytest.raises(ValueError, match=message):
        FittedRuleEngine([rule("frequency")]).fit(frame)


def test_inference_schema_errors_and_failed_refit_clear_state():
    engine = FittedRuleEngine([rule("frequency")]).fit(pd.DataFrame({"category": ["a"], "x": [2.0]}))
    for frame in [
        pd.DataFrame({"category": ["a"]}),
        pd.DataFrame({"x": [2.0], "category": ["a"]}),
        pd.DataFrame({"category": ["a"], "x": [2.0], "frequency": [0]}),
    ]:
        with pytest.raises(ValueError, match="schema"):
            engine.transform(frame)
    with pytest.raises(ValueError, match="missing source"):
        engine.fit(pd.DataFrame({"x": [1]}))
    assert not engine.is_fitted
    assert engine.get_feature_names() == []
    with pytest.raises(RuntimeError, match="fitted"):
        engine.transform(pd.DataFrame({"category": ["a"]}))


def test_refit_replaces_statistics_and_does_not_mutate_user_specifications():
    spec = rule("frequency")
    engine = FittedRuleEngine([spec]).fit(pd.DataFrame({"category": ["a", "a"]}))
    spec.columns[0] = "changed"
    engine.rules[0].columns[0] = "also_changed"
    assert engine.transform(pd.DataFrame({"category": ["a"]}))["frequency"].iloc[0] == 1
    engine = FittedRuleEngine([rule("frequency")])
    engine.fit(pd.DataFrame({"category": ["a", "a"]}))
    engine.fit(pd.DataFrame({"category": ["b"]}))
    assert engine.transform(pd.DataFrame({"category": ["a"]}))["frequency"].iloc[0] == 0


def test_empty_plan_and_ndarray_preserve_originals():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    engine = FittedRuleEngine([])
    pd.testing.assert_frame_equal(engine.fit_transform(X), pd.DataFrame(X, columns=["feature_0", "feature_1"]))
    assert engine.get_feature_names() == []
    assert engine.transform(X[:0]).shape == (0, 2)


def test_json_and_pickle_round_trip_all_learned_state():
    X = pd.DataFrame({"category": ["a", "a", "b", None], "x": [1.0, 2.0, 4.0, np.nan]})
    rules = [
        rule("standardize", ["x"]),
        rule("clip", ["x"]),
        rule("frequency"),
        rule("group_mean", ["category", "x"]),
        rule("rare_category", min_frequency=0.5),
        rule("target_mean", smoothing=1.0),
    ]
    engine = FittedRuleEngine(rules, task="regression", target_cv=2).fit(X, np.arange(4, dtype=float))
    exported = json.loads(json.dumps(engine.to_dict(), allow_nan=False))
    json_restored = FittedRuleEngine.from_dict(exported)
    pickle_restored = pickle.loads(pickle.dumps(engine))
    expected = engine.transform(X)
    for restored in [json_restored, pickle_restored]:
        pd.testing.assert_frame_equal(restored.transform(X), expected)
        assert restored.get_feature_names() == engine.get_feature_names()
        assert restored.get_feature_metadata() == engine.get_feature_metadata()
        assert restored.to_dict() == engine.to_dict()
    exported["rules"][0]["mean"] = 999
    assert json_restored.to_dict() == engine.to_dict()
    assert json_restored._training_target_features == {}
    assert "_training_target_features" not in engine.to_dict()


def test_serialized_state_validation():
    state = FittedRuleEngine([rule("clip", ["x"])]).fit(pd.DataFrame({"x": [0, 1]})).to_dict()
    state["rules"][0]["lower"] = 10
    with pytest.raises(ValueError, match="lower bounds"):
        FittedRuleEngine.from_dict(state)
    state["version"] = 2
    with pytest.raises(ValidationError):
        FittedRuleEngine.from_dict(state)


@pytest.mark.parametrize(
    "options", [{"target_cv": 1}, {"target_cv": True}, {"random_state": "42"}, {"task": "ranking"}]
)
def test_invalid_engine_options(options):
    with pytest.raises(ValueError):
        FittedRuleEngine([], **options)


def test_single_row_supervised_and_unsupported_fit_options_rejected():
    with pytest.raises(ValueError, match="at least two"):
        FittedRuleEngine([rule("target_mean")]).fit(pd.DataFrame({"category": ["a"]}), np.array([1]))
    with pytest.raises(TypeError, match="Unsupported"):
        FittedRuleEngine([]).fit(pd.DataFrame({"x": [1]}), sample_weight=np.array([1]))


def test_group_target_means_exclude_entire_entity_and_align_metadata_positionally():
    X = pd.DataFrame({"category": ["shared", "shared", "shared", None, "unique", "shared"]}, index=[9, 9, 5, 1, 1, 3])
    groups = pd.Series(["a", "a", "b", "b", "c", "c"], index=[100, 90, 80, 70, 60, 50])
    y = np.array([1.0, 3.0, 10.0, 20.0, 100.0, 200.0])
    engine = FittedRuleEngine([rule("target_mean", smoothing=2.0)], task="regression", target_cv=5)
    output = engine.fit_transform(X, y, groups=groups)
    expected = np.empty(len(X))
    for group in groups.unique():
        train = groups.to_numpy() != group
        prior = y[train].mean()
        for position in np.flatnonzero(~train):
            key = X["category"].iloc[position]
            members = y[train & (X["category"] == key).to_numpy()] if pd.notna(key) else np.array([])
            expected[position] = (members.sum() + 2 * prior) / (len(members) + 2)
    np.testing.assert_allclose(output["target_mean"], expected)
    pd.testing.assert_frame_equal(output[X.columns], X)
    report = engine.get_cross_fit_report()
    assert report == {
        "split_kind": "grouped",
        "n_splits": 3,
        "n_training_rows": 6,
        "prior_row_positions": [],
        "prior_policy": "fixed_zero",
        "label_end_times_used": False,
    }
    for group in groups.unique():
        own_group = groups.to_numpy() == group
        changed = y.copy()
        changed[own_group] += 9000
        perturbed = FittedRuleEngine([rule("target_mean", smoothing=2.0)], task="regression", target_cv=5)
        changed_output = perturbed.fit_transform(X, changed, groups=groups)
        np.testing.assert_allclose(output.loc[own_group, "target_mean"], changed_output.loc[own_group, "target_mean"])
    future = pd.DataFrame({"category": ["new", None]})
    np.testing.assert_allclose(engine.transform(future)["target_mean"], y.mean())


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_group_folds_allow_constant_training_targets(task):
    X = pd.DataFrame({"category": ["same"] * 4})
    y = np.array([0, 0, 1, 1])
    engine = FittedRuleEngine([rule("target_mean")], task=task, target_cv=2)
    output = engine.fit_transform(X, y, groups=np.array(["a", "a", "b", "b"]))
    np.testing.assert_allclose(output["target_mean"], [1, 1, 0, 0])


def test_two_row_target_cross_fitting_keeps_single_row_training_support():
    X = pd.DataFrame({"category": ["same", "same"]})
    output = FittedRuleEngine([rule("target_mean")], task="regression").fit_transform(X, np.array([4.0, 9.0]))
    np.testing.assert_allclose(output["target_mean"], [9, 4])


@pytest.mark.parametrize("boundary", ["iid", "groups", "timestamps"])
def test_binary_fold_membership_does_not_depend_on_own_label(boundary):
    X = pd.DataFrame({"category": ["same"] * 16})
    y = np.tile([0, 1], 8)
    metadata = {}
    if boundary == "groups":
        metadata["groups"] = np.repeat(np.arange(4), 4)
    elif boundary == "timestamps":
        metadata["timestamps"] = np.repeat(np.arange(8), 2)
    engine = FittedRuleEngine([rule("target_mean")], task="classification", target_cv=2)
    baseline = engine.fit_transform(X, y, **metadata)["target_mean"].to_numpy()
    for position in range(len(X)):
        changed = y.copy()
        changed[position] = 1 - changed[position]
        changed_output = FittedRuleEngine([rule("target_mean")], task="classification", target_cv=2).fit_transform(
            X, changed, **metadata
        )
        assert changed_output["target_mean"].iloc[position] == pytest.approx(baseline[position])


@pytest.mark.parametrize("boundary", ["iid", "groups", "timestamps"])
def test_removing_last_negative_label_cannot_recode_other_positive_targets(boundary):
    X = pd.DataFrame({"category": ["same"] * 16})
    y = np.ones(16)
    y[-1] = 0
    metadata = {}
    if boundary == "groups":
        metadata["groups"] = np.repeat(np.arange(4), 4)
    elif boundary == "timestamps":
        metadata["timestamps"] = np.repeat(np.arange(8), 2)
    before = FittedRuleEngine([rule("target_mean")], task="classification", target_cv=2)
    after = FittedRuleEngine([rule("target_mean")], task="classification", target_cv=2)
    original = before.fit_transform(X, y, **metadata)
    changed = after.fit_transform(X, np.ones(16), **metadata)
    assert changed["target_mean"].iloc[-1] == pytest.approx(original["target_mean"].iloc[-1])
    if boundary == "timestamps":
        np.testing.assert_allclose(changed["target_mean"], original["target_mean"])
    assert after.to_dict()["target_classes"] == [0, 1]


@pytest.mark.parametrize("task,target,expected", [("regression", 7.0, 7.0), ("classification", "yes", 0.0)])
@pytest.mark.parametrize("boundary", ["iid", "groups", "timestamps"])
def test_constant_targets_supported_for_all_split_protocols(task, target, expected, boundary):
    X = pd.DataFrame({"category": ["same"] * 16})
    metadata = {}
    if boundary == "groups":
        metadata["groups"] = np.repeat(np.arange(4), 4)
    elif boundary == "timestamps":
        metadata["timestamps"] = np.repeat(np.arange(8), 2)
    engine = FittedRuleEngine([rule("target_mean")], task=task, target_cv=2)
    output = engine.fit_transform(X, np.full(len(X), target), **metadata)
    values = np.full(len(X), expected)
    if boundary == "timestamps":
        values[:8] = 0.0
    np.testing.assert_allclose(output["target_mean"], values)
    assert np.isfinite(output["target_mean"]).all()


def test_temporal_target_means_are_strictly_historical_with_marked_zero_prefix():
    timestamps = np.array([6, 1, 4, 2, 6, 5, 3, 7, 5, 7])
    X = pd.DataFrame({"category": ["same"] * len(timestamps)}, index=[5, 5, 2, 2, 1, 0, 9, 9, 6, 7])
    y = np.array([60, 10, 40, 20, 64, 50, 30, 70, 54, 74], dtype=float)
    engine = FittedRuleEngine([rule("target_mean", smoothing=0.0)], task="regression", target_cv=2)
    with patch("featcopilot.engines.fitted_rules.logger.warning") as warning:
        output = engine.fit_transform(X, y, timestamps=pd.Series(timestamps, index=np.arange(100, 110)))
    expected = np.where(timestamps <= 4, 0.0, np.where(timestamps < 7, 25.0, y[timestamps < 7].mean()))
    np.testing.assert_allclose(output["target_mean"], expected)
    pd.testing.assert_frame_equal(output[X.columns], X)
    assert any("fixed-zero no-history prior" in call.args[0] for call in warning.call_args_list)
    report = engine.get_cross_fit_report()
    assert report["split_kind"] == "temporal"
    assert report["prior_row_positions"] == [1, 2, 3, 6]
    assert report["n_splits"] == 2
    assert report["label_end_times_used"] is False
    for cutoff in [5, 7]:
        changed = y.copy()
        changed[timestamps >= cutoff] += 10000
        perturbed = FittedRuleEngine([rule("target_mean", smoothing=0.0)], task="regression", target_cv=2)
        after = perturbed.fit_transform(X, changed, timestamps=timestamps)
        stable = timestamps < 7 if cutoff == 5 else np.ones(len(X), dtype=bool)
        np.testing.assert_allclose(output.loc[stable, "target_mean"], after.loc[stable, "target_mean"])
    np.testing.assert_allclose(engine.transform(pd.DataFrame({"category": ["unknown", None]}))["target_mean"], y.mean())


@pytest.mark.parametrize("datetime_metadata", [False, True])
def test_label_end_times_purge_fold_mappings_and_priors(datetime_metadata):
    timestamps = np.repeat(np.arange(1, 9), 2)
    label_ends = timestamps.copy()
    label_ends[[0, 2, 4]] = [5, 7, 100]
    X = pd.DataFrame(
        {"category": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "new", None, "b", "a", "new", None, "b"]},
        index=np.repeat(np.arange(8)[::-1], 2),
    )
    y = np.arange(16, dtype=float)
    y[[0, 2, 4]] = [1000, 2000, 3000]
    if datetime_metadata:
        date_base = pd.Timestamp("2024-01-01", tz="UTC")
        metadata = {
            "timestamps": pd.Series(date_base + pd.to_timedelta(timestamps, unit="D"), index=np.arange(16) + 100),
            "label_end_times": pd.Series(date_base + pd.to_timedelta(label_ends, unit="D"), index=np.arange(16) + 200),
        }
    else:
        metadata = {"timestamps": timestamps, "label_end_times": label_ends}
    rules = [rule("target_mean", smoothing=2.0), rule("frequency")]
    engine = FittedRuleEngine(rules, task="regression", target_cv=2)
    output = engine.fit_transform(X, y, **metadata)
    expected = np.zeros(len(X))
    for cutoff, end in [(5, 7), (7, 9)]:
        train = (timestamps < cutoff) & (label_ends < cutoff)
        prior = y[train].mean()
        for position in np.flatnonzero((timestamps >= cutoff) & (timestamps < end)):
            key = X["category"].iloc[position]
            members = y[train & (X["category"] == key).to_numpy()] if pd.notna(key) else np.array([])
            expected[position] = (members.sum() + 2 * prior) / (len(members) + 2)
    np.testing.assert_allclose(output["target_mean"], expected)
    assert engine.get_cross_fit_report()["label_end_times_used"] is True
    for position, unaffected_before in [(0, 7), (2, 9), (4, 9)]:
        changed = y.copy()
        changed[position] += 99999
        perturbed = FittedRuleEngine(rules, task="regression", target_cv=2)
        changed_output = perturbed.fit_transform(X, changed, **metadata)
        stable = timestamps < unaffected_before
        np.testing.assert_allclose(output.loc[stable, "target_mean"], changed_output.loc[stable, "target_mean"])
    # Non-supervised rules are unchanged by the temporal fitting protocol.
    np.testing.assert_allclose(output["frequency"], engine.transform(X)["frequency"])
    future = pd.DataFrame({"category": ["a", "unknown", None, "b"]}, index=[4, 4, 1, 0])
    before = engine.to_dict()
    expected_future = engine.transform(future)
    restored = FittedRuleEngine.from_dict(json.loads(json.dumps(before, allow_nan=False)))
    for copy in [restored, pickle.loads(pickle.dumps(engine))]:
        pd.testing.assert_frame_equal(copy.transform(future), expected_future)
        assert copy.get_cross_fit_report() == engine.get_cross_fit_report()
        assert copy.to_dict() == engine.to_dict()
    pd.testing.assert_frame_equal(
        expected_future, pd.concat([engine.transform(future.iloc[:2]), engine.transform(future.iloc[2:])])
    )
    assert engine.to_dict() == before


@pytest.mark.parametrize(
    "metadata,message",
    [
        ({"groups": np.arange(8), "timestamps": np.arange(8)}, "groups and timestamps"),
        ({"groups": np.zeros(8)}, "groups"),
        ({"groups": np.arange(7)}, "same number"),
        ({"timestamps": np.array([0, 1, 2, 3, 4, 5, 6, np.nan])}, "missing"),
        ({"timestamps": np.zeros(8)}, "time blocks"),
        ({"label_end_times": np.arange(8)}, "timestamps"),
        ({"timestamps": np.arange(8), "label_end_times": np.arange(7)}, "same number"),
        ({"timestamps": np.arange(8), "label_end_times": np.full(8, 100)}, "nonempty|purging|purged"),
    ],
)
def test_invalid_split_or_availability_metadata_raises_without_fallback(metadata, message):
    engine = FittedRuleEngine([rule("target_mean")], task="regression", target_cv=2)
    X = pd.DataFrame({"category": ["a"] * 8})
    with pytest.raises(ValueError, match=message):
        engine.fit(X, np.arange(8, dtype=float), **metadata)
    assert not engine.is_fitted
    assert engine.get_feature_names() == []
    with pytest.raises(RuntimeError, match="fitted"):
        engine.transform(X)


def test_cross_fit_report_resets_on_refit_and_accepts_older_json_exports():
    X = pd.DataFrame({"category": ["same"] * 8})
    engine = FittedRuleEngine([rule("target_mean")], task="regression", target_cv=2)
    with pytest.raises(RuntimeError, match="fitted"):
        engine.get_cross_fit_report()
    engine.fit(X, np.arange(8), timestamps=np.arange(8))
    report = engine.get_cross_fit_report()
    report["prior_row_positions"].append(999)
    assert 999 not in engine.get_cross_fit_report()["prior_row_positions"]
    engine.fit(X, np.arange(8))
    assert engine.get_cross_fit_report()["split_kind"] == "iid"
    assert engine.get_cross_fit_report()["prior_row_positions"] == []
    old_export = engine.to_dict()
    del old_export["cross_fit_report"]
    restored = FittedRuleEngine.from_dict(old_export)
    assert restored.get_cross_fit_report() == {}
    pd.testing.assert_frame_equal(restored.transform(X), engine.transform(X))
    with pytest.raises(ValueError):
        engine.fit(X, np.arange(8), groups=np.zeros(8))
    assert engine._cross_fit_report is None
    assert FittedRuleEngine([]).fit(X).get_cross_fit_report() == {}


@pytest.mark.parametrize(
    "operator,columns,parameters",
    [
        ("multiply", ["x"], {}),
        ("divide", ["x"], {}),
        ("multiply", ["x", "x"], {}),
        ("divide", ["x", "x"], {}),
        ("multiply", ["x", "y", "z"], {}),
        ("log1p", ["x", "y"], {}),
        ("log1p", [], {}),
        ("multiply", ["x", "y"], {"scale": 2}),
        ("divide", ["x", "y"], {"epsilon": 1e-6}),
        ("log1p", ["x"], {"signed": True}),
    ],
)
def test_rowwise_rules_reject_invalid_arity_and_parameters(operator, columns, parameters):
    with pytest.raises(ValidationError):
        FittedRuleSpec(name="rowwise", operator=operator, columns=columns, parameters=parameters)


def test_rowwise_rules_preserve_schema_learn_no_statistics_and_round_trip():
    rules = [rule("multiply", ["x", "y"]), rule("divide", ["x", "y"]), rule("log1p", ["x"])]
    X = pd.DataFrame({"x": [-6.0, 0, 12, np.nan], "y": [2.0, 0, 4, 5]}, index=[7, 7, 1, 0])
    engine = FittedRuleEngine(rules)
    output = engine.fit_transform(X)
    pd.testing.assert_frame_equal(output[X.columns], X)
    assert list(output.columns) == ["x", "y", "multiply", "divide", "log1p"]
    np.testing.assert_allclose(output["multiply"], [-12, 0, 48, np.nan], equal_nan=True)
    np.testing.assert_allclose(output["divide"], [-3, np.nan, 3, np.nan], equal_nan=True)
    np.testing.assert_allclose(output["log1p"], np.log1p(np.abs(X["x"])), equal_nan=True)
    assert engine.get_cross_fit_report() == {}
    state = engine.to_dict()
    different_training = pd.DataFrame({"x": [1e10, -1e10], "y": [1e-8, 1e8]})
    assert FittedRuleEngine(rules).fit(different_training).to_dict() == state
    future = pd.DataFrame({"x": [-10.0, 20, np.nan], "y": [4.0, 5, 0]}, index=[9, 9, 2])
    expected = engine.transform(future)
    pd.testing.assert_frame_equal(
        expected, pd.concat([engine.transform(future.iloc[:1]), engine.transform(future.iloc[1:])])
    )
    assert engine.to_dict() == state
    json_restored = FittedRuleEngine.from_dict(json.loads(json.dumps(state, allow_nan=False)))
    for restored in [json_restored, pickle.loads(pickle.dumps(engine))]:
        pd.testing.assert_frame_equal(restored.transform(future), expected)
        assert restored.get_feature_metadata() == engine.get_feature_metadata()
        assert restored.to_dict() == state
    empty = engine.transform(future.iloc[:0])
    assert list(empty.columns) == list(expected.columns)
    assert empty.empty


def test_rowwise_missing_nonfinite_and_overflow_policy():
    maximum = np.finfo(float).max
    X = pd.DataFrame(
        {
            "x": [maximum, -maximum, 2.0, 0.0, np.inf, -np.inf, np.nan, 4.0],
            "y": [2.0, 2.0, 0.0, -0.0, 3.0, 4.0, 5.0, np.nan],
        }
    )
    rules = [rule("multiply", ["x", "y"]), rule("divide", ["x", "y"]), rule("log1p", ["x"])]
    with np.errstate(all="raise"):
        output = FittedRuleEngine(rules).fit_transform(X)
    np.testing.assert_allclose(
        output["multiply"], [maximum, -maximum, 0, 0, np.nan, np.nan, np.nan, np.nan], equal_nan=True
    )
    np.testing.assert_allclose(
        output["divide"],
        [maximum / 2, -maximum / 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        output["log1p"],
        [np.log1p(maximum), np.log1p(maximum), np.log1p(2), 0, np.nan, np.nan, np.nan, np.log1p(4)],
        equal_nan=True,
    )
    engine = FittedRuleEngine([rule("divide", ["x", "y"])]).fit(X)
    tiny_denominators = pd.DataFrame({"x": [maximum, -maximum, 1.0], "y": [1e-300, 1e-300, 1e-300]})
    np.testing.assert_allclose(engine.transform(tiny_denominators)["divide"], [maximum, -maximum, 1e300])
    assert not np.isinf(output[[r.name for r in rules]].to_numpy()).any()


@pytest.mark.parametrize("operator,columns", [("multiply", ["x", "y"]), ("divide", ["x", "y"]), ("log1p", ["x"])])
@pytest.mark.parametrize("invalid", [["1", "2"], [1 + 2j, 3 + 4j], pd.date_range("2024-01-01", periods=2)])
def test_rowwise_numeric_validation_at_fit_and_transform(operator, columns, invalid):
    spec = rule(operator, columns)
    for column in columns:
        bad = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        bad[column] = invalid
        with pytest.raises(ValueError, match="numeric"):
            FittedRuleEngine([spec]).fit(bad)
        engine = FittedRuleEngine([spec]).fit(pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}))
        with pytest.raises(ValueError, match="numeric"):
            engine.transform(bad)


def test_rowwise_nullable_numeric_missing_operands_stay_missing():
    X = pd.DataFrame({"x": pd.Series([2, pd.NA, 0], dtype="Int64"), "y": pd.Series([pd.NA, 3, 0], dtype="Float64")})
    engine = FittedRuleEngine([rule("multiply", ["x", "y"]), rule("divide", ["x", "y"]), rule("log1p", ["x"])])
    output = engine.fit_transform(X)
    pd.testing.assert_frame_equal(output[X.columns], X)
    np.testing.assert_allclose(output["multiply"], [np.nan, np.nan, 0], equal_nan=True)
    assert output["divide"].isna().all()
    np.testing.assert_allclose(output["log1p"], [np.log1p(2), np.nan, 0], equal_nan=True)


def test_plan_runtime_schema_exposes_the_nine_allowed_operators():
    operators = FittedRuleSpec.model_json_schema()["properties"]["operator"]["enum"]
    assert set(operators) == {
        "standardize",
        "clip",
        "frequency",
        "group_mean",
        "target_mean",
        "rare_category",
        "multiply",
        "divide",
        "log1p",
    }
