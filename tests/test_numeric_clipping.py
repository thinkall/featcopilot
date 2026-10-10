"""Train-frozen model-input clipping, dtype/schema parity and engine isolation."""

import json
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from featcopilot.core.fitted_rule import FittedRuleSpec
from featcopilot.engines.fitted_rules import FittedRuleEngine
from featcopilot.engines.predictive import PredictiveTabularEngine
from featcopilot.utils.numeric_clipping import (
    NUMERIC_GUARD_LIMIT,
    NumericQuantileClipper,
    finite_quantile_bounds,
    guard_numeric_values,
    validate_clip_quantiles,
)


@pytest.fixture(autouse=True)
def limited_threads():
    with threadpool_limits(limits=1):
        yield


class IndexedRegressor(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        self.feature_importances_ = np.arange(1, X.shape[1] + 1, dtype=float)
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def engine(**kwargs):
    options = {
        "screening": "importance",
        "numeric_clip_quantiles": (0.25, 0.75),
        "estimator": IndexedRegressor(),
        "task": "regression",
        "stability_repeats": 1,
        "max_candidates": 12,
        "max_features": 4,
    }
    options.update(kwargs)
    return PredictiveTabularEngine(**options)


def sample_data(n=120):
    rng = np.random.default_rng(754)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = 3 * X["a"].to_numpy() + X["b"].to_numpy() ** 2
    return X, y


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        "clip",
        (True, 0.9),
        (0.1, False),
        (np.bool_(True), 1.0),
        ("0.1", 0.9),
        (0.1, np.nan),
        (0.1, np.inf),
        (-0.1, 0.9),
        (0.9, 0.1),
        (0.5, 0.5),
        (0.1, 1.1),
        (0.1,),
        (0, 0.5, 1),
    ],
)
def test_invalid_quantile_policies_raise_without_coercion(value):
    with pytest.raises(ValueError, match="quantile"):
        validate_clip_quantiles(value)
    with pytest.raises(ValueError, match="quantile"):
        NumericQuantileClipper(value)
    with pytest.raises(ValueError, match="quantile"):
        engine(numeric_clip_quantiles=value)


def test_valid_quantiles_are_canonical_but_sklearn_clone_keeps_parameter_identity():
    assert validate_clip_quantiles(None) is None
    assert validate_clip_quantiles([0, np.float64(1)]) == (0.0, 1.0)
    options = [0.1, 0.9]
    original = NumericQuantileClipper(options)
    assert original.get_params()["quantiles"] is options
    copied = clone(original)
    assert copied.get_params() == original.get_params()
    X = pd.DataFrame({"x": np.arange(10, dtype=float)})
    original.fit(X)
    assert original.quantiles_ == (0.1, 0.9)
    assert not clone(original)._is_fitted
    assert engine(numeric_clip_quantiles=[0, 1]).config.numeric_clip_quantiles == (0.0, 1.0)
    with pytest.raises(ValueError, match="enabled quantile"):
        NumericQuantileClipper(None)


def test_endpoint_minmax_path_is_linear_and_preserves_training_float_bits():
    values = np.array(
        [
            -1e30,
            -31.3,
            -0.12345678912345678,
            -np.nextafter(0.0, 1.0),
            -0.0,
            0.0,
            np.nextafter(0.0, 1.0),
            0.1,
            1.2345678901234567,
            1e30,
        ],
        dtype=np.float64,
    )
    X = pd.DataFrame(
        {"numeric": values, "constant": np.full(len(values), 0.1), "zeros": np.tile([-0.0, 0.0], len(values) // 2)}
    )
    before = X.to_numpy().copy()
    with patch("featcopilot.utils.numeric_clipping.np.quantile", side_effect=AssertionError("No interpolation")):
        clipper = NumericQuantileClipper().fit(X)
        output = clipper.transform(X)
        assert clipper.bounds_["numeric"] == (float(values.min()), float(values.max()))
    assert clipper.quantiles_ == (0.0, 1.0)
    np.testing.assert_array_equal(output.to_numpy().view(np.uint64), before.view(np.uint64))
    np.testing.assert_array_equal(X.to_numpy().view(np.uint64), before.view(np.uint64))


def test_endpoint_bounds_do_not_round_the_small_endpoint_through_normalization():
    low = np.nextafter(0.1, 1.0)
    high = 987654321.1234567
    values = np.array([low, 1.0, high])
    bounds = finite_quantile_bounds(values, (0.0, 1.0))
    assert bounds == (low, high)
    assert finite_quantile_bounds(values, (0.0, 0.75))[0] == low
    assert finite_quantile_bounds(values, (0.25, 1.0))[1] == high
    assert finite_quantile_bounds(values, (0.0, 0.0)) == (low, low)
    assert finite_quantile_bounds(values, (1.0, 1.0)) == (high, high)
    X = pd.DataFrame({"x": values})
    clipper = NumericQuantileClipper((0.0, 1.0)).fit(X)
    np.testing.assert_array_equal(clipper.transform(X)["x"].to_numpy().view(np.uint64), values.view(np.uint64))
    future = pd.DataFrame({"x": [low - 1.0, high + 1.0, -np.inf, np.inf, np.nan]})
    np.testing.assert_array_equal(clipper.transform(future)["x"], [low, high, np.nan, np.nan, np.nan])


def test_endpoint_policy_keeps_guard_exception_and_fitted_rule_defaults_explicit():
    X = pd.DataFrame({"x": [-1e300, -0.0, 0.1, 1e300, np.nan, np.inf]})
    clipper = NumericQuantileClipper((0.0, 1.0)).fit(X)
    np.testing.assert_array_equal(clipper.transform(X)["x"], [-1e30, -0.0, 0.1, 1e30, np.nan, np.nan])
    assert np.signbit(clipper.transform(X)["x"].iloc[1])
    default_rule = FittedRuleSpec(name="clipped", operator="clip", columns=["x"])
    assert default_rule.parameters == {"lower_quantile": 0.01, "upper_quantile": 0.99}


@pytest.mark.parametrize("screening", ["auto", "linear", "tree"])
def test_clipping_requires_explicit_importance_until_residual_screens_share_policy(screening):
    with pytest.raises(ValueError, match="explicit screening='importance'"):
        engine(screening=screening)


def test_guard_is_applied_before_quantiles_and_preserves_missingness_and_original_frame():
    X = pd.DataFrame({"x": [-np.inf, np.inf, np.nan, -1e300, 1e300, 0.0]})
    before = X.copy(deep=True)
    clipper = NumericQuantileClipper((0.25, 0.75)).fit(X)
    assert clipper.bounds_["x"] == (-5e29, 5e29)
    future = pd.DataFrame({"x": [1e300, -1e300, np.inf, np.nan, 0.0]})
    actual = clipper.transform(future)
    np.testing.assert_array_equal(actual["x"], [5e29, -5e29, np.nan, np.nan, 0.0])
    assert_frame_equal(X, before)
    assert future["x"].iloc[0] == 1e300
    assert np.isposinf(future["x"].iloc[2])
    np.testing.assert_array_equal(guard_numeric_values(X["x"].to_numpy()), [np.nan, np.nan, np.nan, -1e30, 1e30, 0])
    assert clipper.get_clip_report()["guard_limit"] == NUMERIC_GUARD_LIMIT


def test_dataframe_schema_index_categories_and_boolean_values_are_preserved():
    X = pd.DataFrame(
        {
            "number": pd.array([0, 1, 2, 3, 4, pd.NA], dtype="Float64"),
            "integer": [0, 1, 2, 3, 4, 100],
            "flag": pd.array([True, False, True, pd.NA, False, True], dtype="boolean"),
            "boolean": [False, False, False, False, False, True],
            "category": pd.Categorical(["a", "b", "a", None, "a", "b"]),
            "text": pd.array(["a", None, "a", "b", "c", "d"], dtype="string"),
        },
        index=pd.Index([8, 8, 2, 1, 4, 7], name="rows"),
    )
    X.columns.name = "columns"
    before = X.copy(deep=True)
    clipper = NumericQuantileClipper((0.25, 0.75)).fit(X)
    result = clipper.transform(X)
    assert result is not X
    assert_frame_equal(X, before)
    assert clipper.numeric_columns_ == ("number", "integer")
    for column in ["flag", "boolean", "category", "text"]:
        assert_series_equal(result[column], X[column])
    assert result.index.equals(X.index)
    assert result.columns.equals(X.columns)
    assert pd.isna(result["number"].iloc[-1])
    assert clipper.get_feature_names_out().tolist() == X.columns.tolist()
    names = clipper.get_feature_names_out()
    names[0] = "mutated"
    assert clipper.get_feature_names_out()[0] == "number"
    assert_frame_equal(pd.concat([clipper.transform(X.iloc[:2]), clipper.transform(X.iloc[2:])]), result)
    assert_frame_equal(pickle.loads(pickle.dumps(clipper)).transform(X), result)


def test_categorical_only_empty_batches_constants_and_all_missing_state():
    categories = pd.DataFrame({"category": pd.Categorical(["a", "b"]), "flag": [True, False]})
    categorical = NumericQuantileClipper((0.1, 0.9)).fit(categories)
    assert categorical.bounds_ == {}
    assert_frame_equal(categorical.transform(categories), categories)
    X = pd.DataFrame({"constant": [7.0, np.nan, 7.0], "missing": [np.nan, np.inf, -np.inf]})
    clipper = NumericQuantileClipper((0.1, 0.9)).fit(X)
    assert clipper.constant_columns_ == ("constant",)
    assert clipper.all_missing_columns_ == ("missing",)
    assert clipper.bounds_ == {"constant": (7.0, 7.0), "missing": None}
    future = pd.DataFrame({"constant": [100.0, np.nan], "missing": [1e300, np.nan]})
    result = clipper.transform(future)
    np.testing.assert_array_equal(result["constant"], [7.0, np.nan])
    np.testing.assert_array_equal(result["missing"], [1e30, np.nan])
    assert_frame_equal(clipper.transform(X.iloc[:0]), X.iloc[:0])
    assert clipper.transform(pd.DataFrame(columns=X.columns)).empty
    with pytest.raises(ValueError, match="nonempty"):
        clipper.fit(X.iloc[:0])
    assert not clipper._is_fitted
    with pytest.raises(RuntimeError, match="fitted"):
        clipper.transform(X)


def test_schema_errors_and_successful_refit_do_not_reuse_previous_bounds():
    clipper = NumericQuantileClipper((0, 1)).fit(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))
    for frame in [
        pd.DataFrame({"a": [1.0]}),
        pd.DataFrame({"b": [1.0], "a": [2.0]}),
        pd.DataFrame({"a": [1.0], "b": [2.0], "extra": [3.0]}),
    ]:
        with pytest.raises(ValueError, match="schema"):
            clipper.transform(frame)
    with pytest.raises(TypeError, match="non-boolean numeric"):
        clipper.transform(pd.DataFrame({"a": ["bad"], "b": [2.0]}))
    with pytest.raises(ValueError, match="unique string"):
        clipper.fit(pd.DataFrame([[1, 2]], columns=["a", "a"]))
    with pytest.raises(TypeError, match="DataFrame"):
        clipper.fit(np.ones((2, 2)))
    with pytest.raises(ValueError, match="complex"):
        clipper.fit(pd.DataFrame({"a": [1j, 2j]}))
    clipper.fit(pd.DataFrame({"new": [10.0, 20.0]}))
    assert clipper.bounds_ == {"new": (10.0, 20.0)}
    assert clipper.get_clip_report()["fit_rows"] == 2
    json.dumps(clipper.get_clip_report(), allow_nan=False)


def test_fitted_rule_bound_kernel_preserves_uncapped_serialized_and_missing_behavior():
    maximum = np.finfo(float).max
    values = np.array([-maximum, 0.0, maximum, np.nan])
    finite = values[np.isfinite(values)]
    magnitude = float(np.max(np.abs(finite))) or 1.0
    expected = np.quantile(finite / magnitude, [0.25, 0.75]) * magnitude
    assert finite_quantile_bounds(values, (0.25, 0.75)) == tuple(expected)
    assert finite_quantile_bounds(np.array([np.nan, np.inf]), (0, 1)) is None
    assert finite_quantile_bounds(np.array([1.0, 2.0, 3.0]), (0.5, 0.5)) == (2.0, 2.0)
    X = pd.DataFrame({"x": values})
    rule = FittedRuleSpec(
        name="clipped", operator="clip", columns=["x"], parameters={"lower_quantile": 0.25, "upper_quantile": 0.75}
    )
    engine = FittedRuleEngine([rule]).fit(X)
    state = engine.to_dict()
    assert state["rules"][0]["lower"] == expected[0]
    assert state["rules"][0]["upper"] == expected[1]
    assert abs(expected[1]) > 1e30
    assert engine.transform(X)["clipped"].iloc[-1] == 0.0
    restored = FittedRuleEngine.from_dict(json.loads(json.dumps(state, allow_nan=False)))
    assert_frame_equal(engine.transform(X), restored.transform(X))


def test_clip_bounds_and_imputation_are_fitted_before_transform_only_on_train():
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 100.0, np.nan]})
    validation = pd.DataFrame({"x": [-1e20, 1e20, np.nan]})
    pipeline = Pipeline([("clip", NumericQuantileClipper((0.25, 0.75))), ("imputer", SimpleImputer(strategy="median"))])
    pipeline.fit(train)
    before = pickle.dumps(pipeline)
    transformed = pipeline.transform(validation)
    bounds = pipeline.named_steps["clip"].bounds_["x"]
    assert bounds == pytest.approx((0.75, 26.5))
    assert transformed[0, 0] == bounds[0]
    assert transformed[1, 0] == bounds[1]
    assert pipeline.named_steps["imputer"].statistics_[0] == pytest.approx(1.5)
    assert pickle.dumps(pipeline) == before


def test_disabled_engine_pipeline_reports_and_candidate_definitions_are_unchanged():
    X, y = sample_data()
    options = {
        "screening": "importance",
        "estimator": Ridge(),
        "task": "regression",
        "stability_repeats": 1,
        "max_candidates": 12,
    }
    default = PredictiveTabularEngine(**options).fit(X, y)
    disabled = PredictiveTabularEngine(**options, numeric_clip_quantiles=None).fit(X, y)
    assert default.selection_report_ == disabled.selection_report_
    assert "numeric_preprocessing" not in disabled.selection_report_
    assert_frame_equal(default.transform(X), disabled.transform(X))
    ordinary = disabled._fit_learner(disabled._model_frame(X), y, Ridge())
    assert list(ordinary.named_steps) == ["preprocess", "learner"]
    enabled = engine()
    for instance in [default, enabled]:
        instance._input_columns = list(X.columns)
        instance._numeric_columns = list(X.columns)
    first, _ = default._plan_candidates(X, 50)
    second, _ = enabled._plan_candidates(X, 50)
    for features in [first, second]:
        for feature in features:
            feature.importance = None
    assert [feature.to_dict() for feature in first] == [feature.to_dict() for feature in second]


def test_engine_clips_original_and_generated_model_inputs_not_public_feature_outputs():
    X, y = sample_data()
    X["flag"] = pd.array([True, False, pd.NA] * 40, dtype="boolean")
    fitted = engine(max_features=5)
    models = []
    original_fit = fitted._fit_learner

    def record_fit(frame, target, template, **kwargs):
        model = original_fit(frame, target, template, **kwargs)
        models.append(model)
        return model

    with patch.object(fitted, "_fit_learner", side_effect=record_fit):
        output = fitted.fit_transform(X, y)
    assert models
    assert all(list(model.named_steps) == ["numeric_clip", "preprocess", "learner"] for model in models)
    assert all("flag" not in model.named_steps["numeric_clip"].numeric_columns_ for model in models)
    assert all("flag" in model.named_steps["numeric_clip"].boolean_columns_ for model in models)
    assert_frame_equal(output[X.columns], X)
    assert fitted.selection_report_["numeric_preprocessing"]["guard_limit"] == 1e30
    assert fitted.selection_report_["numeric_preprocessing"]["quantiles"] == [0.25, 0.75]
    assert fitted.selection_report_["learner_fit_count"] == len(models)
    clipper = models[0].named_steps["numeric_clip"]
    assert any(column not in X for column in clipper.numeric_columns_)
    for feature in fitted.get_feature_set():
        np.testing.assert_array_equal(feature.compute(X), fitted.transform(X)[feature.name])


def test_selected_feature_output_and_export_are_not_quantile_clipped():
    X, y = sample_data()
    fitted = engine(max_candidates=1)
    original_fit = fitted._fit_learner
    models = []

    def record_fit(frame, target, template, **kwargs):
        model = original_fit(frame, target, template, **kwargs)
        models.append(model)
        return model

    with (
        patch.object(fitted, "_fit_learner", side_effect=record_fit),
        patch.object(fitted, "_choose_count", return_value=1),
    ):
        fitted.fit(X, y)
    feature = fitted.get_feature_set()[fitted.get_feature_names()[0]]
    assert feature.name == "a_sq"
    future = X.iloc[:2].copy()
    future["a"] = [1e4, -1e4]
    output = fitted.transform(future)
    np.testing.assert_array_equal(output[feature.name], [1e8, 1e8])
    np.testing.assert_array_equal(feature.compute(future), output[feature.name])
    clipper = models[-1].named_steps["numeric_clip"]
    model_input = fitted._augment(fitted._model_frame(future), future, [feature])
    capped = clipper.transform(model_input)
    assert capped[feature.name].eq(clipper.bounds_[feature.name][1]).all()
    assert_frame_equal(output[X.columns], future)
    assert_frame_equal(pickle.loads(pickle.dumps(fitted)).transform(future), output)


def test_enabled_policy_does_not_add_learner_fits_or_change_caller_parameters():
    X, y = sample_data()
    supplied = IndexedRegressor()
    before = pickle.dumps(supplied)
    plain = engine(estimator=supplied, numeric_clip_quantiles=None, stability_repeats=3).fit(X, y)
    robust = engine(estimator=supplied, stability_repeats=3).fit(X, y)
    assert plain.selection_report_["learner_fit_count"] == robust.selection_report_["learner_fit_count"]
    assert plain.get_feature_names() == robust.get_feature_names() == []
    assert pickle.dumps(supplied) == before
    assert not hasattr(supplied, "feature_importances_")


def test_engine_all_missing_numeric_column_does_not_hide_usable_categories():
    X = pd.DataFrame({"missing": np.full(24, np.nan), "category": ["a", "b"] * 12})
    y = np.arange(len(X), dtype=float)
    fitted = engine(max_candidates=0).fit(X, y)
    assert_frame_equal(fitted.transform(X), X)
    prepared = fitted._model_frame(X)
    model = fitted._fit_selection_learner(prepared, y, IndexedRegressor())
    assert model.named_steps["numeric_clip"].bounds_ == {"missing": None}
    assert model.predict(prepared).shape == (len(X),)


@pytest.mark.parametrize("kind", ["iid", "grouped", "temporal"])
def test_every_internal_clipper_fit_respects_train_boundaries(kind):
    X, y = sample_data(240)
    groups = np.repeat(np.arange(12), 20) if kind == "grouped" else None
    times = np.repeat(np.arange(24), 10) if kind == "temporal" else None
    ends = times + 1 if times is not None else None
    fitted = engine(stability_repeats=3)
    boundaries = fitted._validate_boundaries(groups, times, len(X), label_end_times=ends)
    partitions = fitted._holdouts(y, "regression", boundaries)
    expected = [set(X.index[train]) for train, _ in partitions]
    observed = []
    original_fit = NumericQuantileClipper.fit

    def record_fit(clipper, frame, target=None):
        observed.append(set(frame.index))
        assert set(frame.index) in expected
        return original_fit(clipper, frame, target)

    with patch.object(NumericQuantileClipper, "fit", record_fit):
        fitted.fit(X, y, groups=groups, timestamps=times, label_end_times=ends)
    assert observed
    assert fitted.selection_report_["selected_count"] == 0
    assert len(observed) == fitted.selection_report_["learner_fit_count"]
    for train, validation in partitions:
        if kind == "grouped":
            assert set(groups[train]).isdisjoint(groups[validation])
        if kind == "temporal":
            assert ends[train].max() < times[validation].min()


def test_perturbed_holdout_statistics_do_not_change_joint_ranking_or_train_caps():
    from sklearn.model_selection import train_test_split

    X, y = sample_data()
    train, validation = train_test_split(np.arange(len(X)), test_size=0.25, random_state=42)
    first = engine(estimator=Ridge()).fit(X, y)
    changed = X.copy()
    changed.loc[validation, "a"] = 1e20
    changed_y = y.copy()
    changed_y[validation] += 1e10
    second = engine(estimator=Ridge()).fit(changed, changed_y)
    assert first.selection_report_["candidate_scores"] == second.selection_report_["candidate_scores"]
    assert first.selection_report_["importance_ranking"] == second.selection_report_["importance_ranking"]
    first_caps = NumericQuantileClipper((0.25, 0.75)).fit(X.iloc[train]).bounds_
    second_caps = NumericQuantileClipper((0.25, 0.75)).fit(changed.iloc[train]).bounds_
    assert first_caps == second_caps


def test_named_preprocessor_keeps_encoded_importance_attribution_correct():
    X = pd.DataFrame({"numeric": np.linspace(-2, 2, 60), "category": ["a", "b", "c"] * 20})
    y = np.arange(len(X), dtype=float)
    fitted = engine()
    prepared = fitted._model_frame(X).assign(generated=np.sin(np.arange(len(X))))
    model = fitted._fit_learner(prepared, y, IndexedRegressor(), numeric_clip_quantiles=(0.25, 0.75))
    weights, _ = fitted._importance_values(model)
    from featcopilot.core.feature import Feature

    positions = fitted._candidate_output_indices(model, [Feature(name="generated")], len(weights))
    assert positions == {"generated": 1}
    assert weights[positions["generated"]] == 2
    assert model.steps[0][0] == "numeric_clip"
    assert model.named_steps["preprocess"].output_indices_["numeric"] == slice(0, 2)


def test_raw_and_engineered_original_bounds_match_with_boolean_and_missing_columns():
    X, _ = sample_data()
    X.loc[3, "a"] = np.nan
    X["flag"] = pd.array([True, False, pd.NA] * 40, dtype="boolean")
    X["category"] = pd.Categorical(["a", "b", None] * 40)
    raw = NumericQuantileClipper((0.25, 0.75)).fit(X)
    extended = NumericQuantileClipper((0.25, 0.75)).fit(X.assign(derived=X["b"] ** 2))
    assert {column: extended.bounds_[column] for column in raw.bounds_} == raw.bounds_
    assert_frame_equal(raw.transform(X), extended.transform(X.assign(derived=X["b"] ** 2))[X.columns])
    fitted = engine()
    prepared = fitted._model_frame(X)
    selection = NumericQuantileClipper((0.25, 0.75)).fit(prepared)
    assert selection.bounds_ == raw.bounds_
    assert selection.boolean_columns_ == raw.boolean_columns_
