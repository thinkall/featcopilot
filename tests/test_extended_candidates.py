"""Opt-in candidate coverage, frozen vocabulary and small synthetic gain tests."""

import json
import pickle
import re
from collections import Counter
from itertools import combinations, islice
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from featcopilot.core.feature import Feature, FeatureSet
from featcopilot.engines.predictive import PredictiveTabularEngine


@pytest.fixture(autouse=True)
def bounded_threads():
    with threadpool_limits(limits=1):
        yield


def extended_engine(**kwargs):
    options = {
        "candidate_set": "extended",
        "task": "regression",
        "estimator": Ridge(alpha=0.01),
        "stability_repeats": 1,
    }
    options.update(kwargs)
    return PredictiveTabularEngine(**options)


def plan(frame, **kwargs):
    engine = extended_engine(**kwargs)
    frame = engine._validate_frame(frame)
    engine._input_columns = frame.columns.tolist()
    engine._numeric_columns = sorted(column for column in frame if pd.api.types.is_numeric_dtype(frame[column].dtype))
    features, _ = engine._plan_candidates(frame, budget=50)
    return engine, features


def attach_plan(engine, features):
    engine._selected_features = features
    engine._feature_names = [feature.name for feature in features]
    engine._feature_set = FeatureSet(features)
    engine._is_fitted = True


def categorical_signal(n=400, seed=54):
    rng = np.random.default_rng(seed)
    left, right = rng.integers(0, 2, size=(2, n))
    X = pd.DataFrame(
        {
            "switch": np.where(left, "private_alpha", "private_beta"),
            "mode": np.where(right, "private_hot", "private_cold"),
        }
    )
    return X, np.logical_xor(left, right).astype(int)


def test_standard_stream_is_exactly_preserved_and_is_the_default():
    expected = [
        "a_sq",
        "a_x_b",
        "a_minus_b",
        "a_div_b",
        "b_div_a",
        "a_log1p",
        "a_sqrt",
        "a_cb",
        "a_recip",
        "a_tanh",
        "a_sin",
        "a_cos",
        "b_sq",
        "a_x_c",
        "a_minus_c",
        "a_div_c",
        "c_div_a",
        "b_log1p",
        "b_sqrt",
        "b_cb",
        "b_recip",
        "b_tanh",
        "b_sin",
        "b_cos",
        "c_sq",
        "b_x_c",
        "b_minus_c",
        "b_div_c",
        "c_div_b",
        "c_log1p",
        "c_sqrt",
        "c_cb",
        "c_recip",
        "c_tanh",
        "c_sin",
        "c_cos",
    ]
    default = PredictiveTabularEngine()
    standard = PredictiveTabularEngine(candidate_set="standard")
    first = list(default._candidate_stream(["a", "b", "c"], ["a", "b", "c"]))
    second = list(standard._candidate_stream(["a", "b", "c"], ["a", "b", "c"]))
    assert default.config.candidate_set == "standard"
    assert [feature.name for feature in first] == expected
    assert [feature.to_dict() for feature in first] == [feature.to_dict() for feature in second]
    assert all(feature.metadata == {"numeric_policy": "finite_clip_1e30_nonfinite_to_nan_v1"} for feature in first)
    categorical, _ = categorical_signal(80)
    _, categorical_features = plan(categorical, candidate_set="standard")
    assert categorical_features == []


@pytest.mark.parametrize("width", [0, 1, 2, 3, 6, 7, 10])
def test_balanced_pairs_cover_each_unordered_pair_exactly_once(width):
    columns = [f"x{i:02d}" for i in range(width)]
    pairs = list(PredictiveTabularEngine._balanced_pairs(columns))
    assert len(pairs) == width * (width - 1) // 2
    assert len(set(pairs)) == len(pairs)
    assert set(pairs) == set(combinations(columns, 2))


def test_every_extended_numeric_operator_remains_available_for_every_pair():
    X = pd.DataFrame(np.arange(30).reshape(10, 3), columns=["a", "b", "c"])
    _, features = plan(X)
    for left, right in combinations(X.columns, 2):
        names = {feature.name for feature in features}
        assert {
            f"{left}_plus_{right}",
            f"{left}_x_{right}",
            f"{left}_minus_{right}",
            f"{left}_min_{right}",
            f"{left}_max_{right}",
            f"{left}_div_{right}",
            f"{right}_div_{left}",
        } <= names
    assert len(features) == 8 * 3 + 7 * 3
    assert len({feature.name for feature in features}) == len(features)


def test_wide_budget_balances_column_degrees_without_quadratic_materialization():
    columns = [f"x{i:04d}" for i in range(2000)]
    engine = extended_engine(max_candidates=1000)
    with patch.object(PredictiveTabularEngine, "_feature", wraps=PredictiveTabularEngine._feature) as factory:
        features = list(islice(engine._candidate_stream(columns, columns), 1000))
    assert len(features) == factory.call_count == 1000
    used = {column for feature in features for column in feature.source_columns}
    assert len(used) >= 900
    assert "x1999" in used
    pairs = [feature for feature in features if len(feature.source_columns) == 2]
    degree = Counter(column for feature in pairs for column in feature.source_columns)
    assert max(degree.values()) <= 2
    counts = Counter(feature.transformation for feature in features)
    assert {"add", "minimum", "maximum", "multiply", "subtract", "divide"} <= set(counts)
    assert (
        max(counts[operation] for operation in ["add", "minimum", "maximum", "multiply", "subtract"])
        - min(counts[operation] for operation in ["add", "minimum", "maximum", "multiply", "subtract"])
        <= 1
    )


def test_mixed_pool_planning_is_column_and_row_permutation_stable():
    X, _ = categorical_signal(120)
    rng = np.random.default_rng(65)
    X["number_a"], X["number_b"] = rng.normal(size=(2, len(X)))
    _, first = plan(X, max_candidates=40)
    _, permuted = plan(X[X.columns[::-1]].iloc[::-1], max_candidates=40)
    assert [feature.to_dict() for feature in first] == [feature.to_dict() for feature in permuted]
    assert {"add", "minimum", "maximum", "category_pair"} <= {feature.transformation for feature in first}


def test_sum_enables_a_limited_tree_to_learn_an_oblique_boundary():
    rng = np.random.default_rng(214)
    X = pd.DataFrame(rng.uniform(-2, 2, size=(600, 2)), columns=["a", "b"])
    y = (X["a"] + X["b"] > 0).astype(int)
    supplied = DecisionTreeClassifier(max_depth=1, random_state=42)
    engine = extended_engine(task="classification", estimator=supplied, max_features=3, max_candidates=32).fit(X, y)
    assert engine.get_feature_names() == ["a_plus_b"]
    assert not hasattr(supplied, "tree_")
    future_rng = np.random.default_rng(997)
    future = pd.DataFrame(future_rng.uniform(-2, 2, size=(500, 2)), columns=X.columns)
    future_y = (future["a"] + future["b"] > 0).astype(int)
    baseline = DecisionTreeClassifier(max_depth=1, random_state=42).fit(X, y)
    enhanced = DecisionTreeClassifier(max_depth=1, random_state=42).fit(engine.transform(X), y)
    raw_accuracy = accuracy_score(future_y, baseline.predict(future))
    enhanced_accuracy = accuracy_score(future_y, enhanced.predict(engine.transform(future)))
    assert enhanced_accuracy >= 0.97
    assert enhanced_accuracy - raw_accuracy >= 0.12


def test_sum_min_max_numerical_export_and_batch_parity():
    X = pd.DataFrame(
        {
            "a": pd.array([1e308, -1e308, 1e20, 0, pd.NA, np.inf, 2], dtype="Float64"),
            "b": [1e308, 1e20, -1e20, 0, 1, 2, -np.inf],
        },
        index=pd.Index([8, 1, 8, 3, 2, 0, 7], name="rows"),
    )
    engine, features = plan(X)
    selected = [feature for feature in features if feature.transformation in {"add", "minimum", "maximum"}]
    attach_plan(engine, selected)
    result = engine.transform(X)
    assert np.isnan(result["a_plus_b"].iloc[0])
    assert result["a_min_b"].iloc[0] == result["a_max_b"].iloc[0] == 1e30
    assert result["a_min_b"].iloc[1] == -1e30
    assert result["a_max_b"].iloc[1] == 1e20
    assert result[[feature.name for feature in selected]].iloc[4:].isna().all().all()
    augmented = engine._augment(engine._model_frame(X), X, selected)
    for feature in selected:
        np.testing.assert_array_equal(feature.compute(X), result[feature.name])
        np.testing.assert_array_equal(augmented[feature.name], result[feature.name])
    assert_frame_equal(result[X.columns], X)
    restored = pickle.loads(pickle.dumps(engine))
    assert_frame_equal(pd.concat([restored.transform(X.iloc[:2]), restored.transform(X.iloc[2:])]), result)


def test_frozen_categorical_pair_materially_improves_xor_without_target_encoding():
    X, y = categorical_signal()
    supplied = LogisticRegression(max_iter=300)
    with patch("featcopilot.engines.predictive.logger") as logger:
        engine = extended_engine(task="classification", estimator=supplied, max_features=3, verbose=True).fit(X, y)
    assert len(engine.get_feature_names()) == 1
    feature = engine.get_feature_set()[engine.get_feature_names()[0]]
    assert feature.transformation == "category_pair"
    assert re.fullmatch(r"catpair_[0-9a-f]{24}", feature.name)
    assert not hasattr(supplied, "coef_")
    future, future_y = categorical_signal(500, seed=773)
    baseline = engine._fit_learner(engine._model_frame(X), y, LogisticRegression(max_iter=300))
    enhanced = engine._fit_learner(engine._model_frame(engine.transform(X)), y, LogisticRegression(max_iter=300))
    baseline_loss = log_loss(future_y, baseline.predict_proba(engine._model_frame(future)))
    enhanced_loss = log_loss(future_y, enhanced.predict_proba(engine._model_frame(engine.transform(future))))
    assert enhanced_loss < 0.5 * baseline_loss
    assert engine.selection_report_["candidate_set"] == "extended"
    assert engine.selection_report_["candidate_family_counts"] == {"category_pair": 4}
    assert "private_" not in json.dumps(engine.selection_report_, allow_nan=False)
    assert "private_" not in repr(logger.mock_calls)
    assert "private_" not in feature.explanation
    assert "private_" in json.dumps(feature.to_dict())
    assert_frame_equal(pickle.loads(pickle.dumps(engine)).transform(future), engine.transform(future))


def test_typed_category_values_missing_policy_and_export_roundtrip():
    values = [1, np.int64(1), 1.0, np.float32(1), True, np.bool_(True), "1", np.str_("1"), None, np.nan, pd.NA, np.inf]
    X = pd.DataFrame({"mixed": pd.Series(values * 6, dtype=object), "other": ["a"] * (len(values) * 6)})
    engine, _ = plan(X)
    tokens = engine._categorical_levels(X, min_count=3)["mixed"]
    expected_positions = {"int": [0, 1], "float": [2, 3], "bool": [4, 5], "str": [6, 7], "missing": [8, 9, 10, 11]}
    for token in tokens:
        feature = engine._category_feature("mixed", "other", (token, ("str", "a")), 6)
        expected = np.tile(np.isin(np.arange(len(values)), expected_positions[token[0]]).astype(float), 6)
        np.testing.assert_array_equal(engine._compute(feature, X), expected)
        np.testing.assert_array_equal(feature.compute(X), expected)
        exported = Feature.from_dict(json.loads(json.dumps(feature.to_dict(), allow_nan=False)))
        np.testing.assert_array_equal(exported.compute(X), expected)
        np.testing.assert_array_equal(engine._compute(exported, X), expected)


def test_unseen_pairs_missing_values_quotes_and_duplicate_indices_are_batch_invariant():
    first = "a'quoted\\value\nline"
    second = 'b"quoted'
    X = pd.DataFrame(
        {
            "left": pd.Series([first, second, None] * 20, dtype="string"),
            "right": pd.Series(["x", "y", "z"] * 20, dtype="category"),
        }
    )
    engine, features = plan(X)
    assert len(features) == 3
    attach_plan(engine, features)
    future = pd.DataFrame(
        {"left": [first, "unseen", first, pd.NA, None], "right": ["x", "x", "y", "z", "unseen"]},
        index=pd.Index([7, 2, 7, 3, 1], name="custom"),
    )
    output = engine.transform(future)
    columns = engine.get_feature_names()
    assert output[columns].iloc[[1, 2, 4]].eq(0).all().all()
    assert output[columns].iloc[[0, 3]].sum(axis=1).eq(1).all()
    assert_frame_equal(output[future.columns], future)
    assert_frame_equal(pd.concat([engine.transform(future.iloc[:2]), engine.transform(future.iloc[2:])]), output)
    for feature in features:
        np.testing.assert_array_equal(feature.compute(future), output[feature.name])


def test_only_internal_training_values_enter_the_frozen_vocabulary():
    X, _ = categorical_signal(160)
    rng = np.random.default_rng(477)
    y = rng.normal(size=len(X))
    train, validation = train_test_split(np.arange(len(X)), test_size=0.25, random_state=42)
    changed = X.copy()
    changed.loc[validation, "switch"] = "holdout_only_secret"
    first = extended_engine(estimator=DummyRegressor()).fit(X, y)
    second = extended_engine(estimator=DummyRegressor()).fit(changed, y)
    assert first.selection_report_["candidate_ranking"] == second.selection_report_["candidate_ranking"]
    assert first.selection_report_["candidate_scores"] == second.selection_report_["candidate_scores"]
    _, planned = plan(changed.iloc[train])
    assert "holdout_only_secret" not in json.dumps([feature.to_dict() for feature in planned])
    # Declared but unobserved category levels must not enter the vocabulary either.
    categorical = changed.iloc[train].copy()
    categorical["switch"] = pd.Categorical(
        categorical["switch"], categories=["private_alpha", "private_beta", "unused"]
    )
    _, categorical_plans = plan(categorical)
    assert "unused" not in json.dumps([feature.to_dict() for feature in categorical_plans])


def test_categorical_levels_frequency_support_and_pair_scan_are_bounded():
    n = 240
    X = pd.DataFrame(
        {
            "left": [f"level_{i % 20}" for i in range(n)],
            "right": [f"other_{i % 3}" for i in range(n)],
            "identifier": [f"id_{i}" for i in range(n)],
        }
    )
    engine, features = plan(X)
    levels = engine._categorical_levels(X, min_count=3)
    assert len(levels["left"]) == 8
    assert "identifier" not in levels
    assert len(features) <= 4
    assert all(feature.metadata["training_pair_count"] >= 3 for feature in features)
    wide = pd.DataFrame({f"cat{i:03d}": ["a", "b"] * 30 for i in range(100)})
    bounded = extended_engine(max_candidates=7)
    seen = []
    original_pairs = bounded._balanced_pairs

    def counted_pairs(columns):
        for pair in original_pairs(columns):
            seen.append(pair)
            yield pair

    with patch.object(bounded, "_balanced_pairs", side_effect=counted_pairs):
        indicators = list(islice(bounded._categorical_pair_family(wide), 7))
    assert len(indicators) == len(seen) == 7
    assert len({column for feature in indicators for column in feature.source_columns}) == 14


@pytest.mark.parametrize("bad", [["not", "scalar"], {"private_value": "not logged"}, object()])
def test_unsupported_category_values_raise_without_logging_contents(bad):
    X, _ = categorical_signal(40)
    X = X.astype(object)
    X.at[X.index[0], X.columns[0]] = bad
    with pytest.raises(ValueError, match="unsupported categorical scalar") as error:
        plan(X)
    assert "private_value" not in str(error.value)


def test_invalid_category_values_raise_in_both_runtime_and_exported_code():
    X, _ = categorical_signal(40)
    engine, features = plan(X)
    feature = features[0]
    bad = X.iloc[:2].astype(object).copy()
    bad.at[bad.index[0], feature.source_columns[0]] = ["invalid"]
    with pytest.raises(ValueError, match="unsupported categorical scalar"):
        engine._compute(feature, bad)
    with pytest.raises(ValueError, match="unsupported categorical scalar"):
        feature.compute(bad)


def test_collisions_and_duplicate_input_names_never_overwrite_originals():
    X, _ = categorical_signal(60)
    _, features = plan(X)
    with pytest.raises(ValueError, match="collision"):
        plan(X.assign(**{features[0].name: 0.0}))
    with pytest.raises(ValueError, match="collision"):
        plan(pd.DataFrame({"a": range(10), "b": range(10, 20), "a_plus_b": 0}))
    with pytest.raises(ValueError, match="unique"):
        plan(pd.DataFrame([["a", "b"]] * 10, columns=["same", "same"]))
    with patch("featcopilot.engines.predictive.hashlib.sha256") as digest:
        digest.return_value.hexdigest.return_value = "0" * 64
        with pytest.raises(ValueError, match="collision"):
            plan(X)


def test_datetime_and_nullable_boolean_originals_are_retained():
    X = pd.DataFrame(
        {
            "when": pd.date_range("2024-01-01", periods=60),
            "flag": pd.Series([True, False, pd.NA] * 20, dtype="boolean"),
            "category": pd.Categorical(["a", "b", "a"] * 20),
        }
    )
    engine, features = plan(X)
    assert features
    assert all("when" not in feature.source_columns for feature in features)
    attach_plan(engine, features)
    output = engine.transform(X)
    assert_frame_equal(output[X.columns], X)
    for feature in features:
        np.testing.assert_array_equal(feature.compute(X), output[feature.name])


def test_zero_budget_refit_and_parent_zero_count_optimization_are_preserved():
    X, y = categorical_signal()
    engine = extended_engine(task="classification", estimator=LogisticRegression(max_iter=300), max_features=3).fit(
        X, y
    )
    assert engine.get_feature_names()
    constant = pd.DataFrame({"constant": ["value"] * len(X)})
    engine.fit(constant, y)
    assert engine.get_feature_names() == []
    assert len(engine.get_feature_set()) == 0
    assert engine.selection_report_["candidate_family_counts"] == {}
    assert_frame_equal(engine.transform(constant), constant)
    with pytest.raises(ValueError, match="y is required"):
        engine.fit(X)
    assert not engine.is_fitted
    assert engine.selection_report_ == {}
    zero = extended_engine(max_candidates=0, stability_repeats=3).fit(X, y)
    assert zero.selection_report_["candidate_count"] == 0
    assert zero.selection_report_["learner_fit_count"] == 3
    assert zero.selection_report_["ranking_scope"] == "baseline_retained_no_final_refit"
    assert_frame_equal(zero.transform(X), X)
    untouched = extended_engine(estimator=DummyRegressor(), stability_repeats=3)
    original = untouched._rank_candidates
    with patch.object(untouched, "_rank_candidates", wraps=original) as rank:
        untouched.fit(X, y)
    assert rank.call_count == 3
    assert untouched.selection_report_["selected_count"] == 0


def test_parent_paired_gain_rule_and_extended_constructor_validation():
    summarize = PredictiveTabularEngine._summarize_losses
    scores = [summarize(0, [10, 20, 30]), summarize(1, [9, 19, 29])]
    assert PredictiveTabularEngine._choose_count(scores, "one_se") == 0
    assert PredictiveTabularEngine._choose_count(scores, "paired_gain") == 1
    assert extended_engine(selection_rule="paired_gain").config.selection_rule == "paired_gain"
    with pytest.raises(ValueError, match="candidate_set"):
        PredictiveTabularEngine(candidate_set="invalid")
