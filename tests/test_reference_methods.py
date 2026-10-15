"""Small reference-contract tests; never run OpenFE's feature search."""

import importlib
import json
import pickle
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from packaging.requirements import Requirement

from benchmarks import reference_methods as reference


def test_benchmark_extra_matches_native_reference_and_calibration_version_contracts():
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = {
        item.name: item for item in map(Requirement, project["project"]["optional-dependencies"]["benchmark-tabular"])
    }
    autofeat = requirements["autofeat"].specifier
    sklearn = requirements["scikit-learn"].specifier
    assert "2.1.1" in autofeat
    assert all(version not in autofeat for version in ("2.1.0", "2.1.2", "2.1.3", "3.0.0"))
    assert "1.7.2" in sklearn
    assert all(version not in sklearn for version in ("1.6.0", "1.6.1", "1.7.0", "1.7.1", "1.8.0"))


def leaf(name):
    return SimpleNamespace(name=name)


def node(name, *children):
    return SimpleNamespace(name=name, children=list(children))


@pytest.fixture
def reference_versions(monkeypatch):
    versions = {"openfe": "0.0.12", "autofeat": "2.1.1", "pandas": "2.3.3", "scikit-learn": "1.7.2", "numpy": "2.4.4"}
    monkeypatch.setattr(reference, "version", versions.__getitem__)


@pytest.fixture
def finite_frame():
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 2.0, 4.0, 5.0, 8.0],
            "z": [2.0, 4.0, 3.0, 2.0, 4.0, 1.0],
            "a": ["red", "red", "red", "blue", "blue", "blue"],
            "b": ["u", "v", "v", "u", "u", "w"],
        },
        index=pd.Index([91, 7, 44, 3, 51, 20], name="row"),
    )


UNARY = ["abs", "log", "sqrt", "square", "sigmoid", "round", "residual", "freq"]
BINARY = ["min", "max", "+", "-", "*", "/"]
GROUP = [f"GroupByThen{s}" for s in ("Min", "Max", "Mean", "Median", "Std", "Rank", "Freq", "NUnique")]


@pytest.mark.parametrize(
    ("operation", "columns"),
    [(op, ["x"]) for op in UNARY]
    + [(op, ["x", "z"]) for op in BINARY]
    + [(op, ["x", "a"]) for op in GROUP]
    + [(op, ["a", "b"]) for op in ("Combine", "CombineThenFreq")],
)
def test_all_operators_match_official_training_values(operation, columns, finite_frame):
    fg = pytest.importorskip("openfe.FeatureGenerator")
    feature = fg.Node(operation, [fg.FNode(c) for c in columns])
    expected = feature.calculate(finite_frame)
    if isinstance(expected.index, pd.MultiIndex):
        expected = expected.droplevel(0).reindex(finite_frame.index)
    dag = reference.FittedOpenFEDAG([feature]).fit(finite_frame)
    actual = dag.transform(finite_frame).iloc[:, 0]
    np.testing.assert_allclose(actual.astype(float), expected.astype(float), equal_nan=True)


def test_published_candidate_universe_is_supported(finite_frame):
    openfe = pytest.importorskip("openfe")
    candidates = openfe.get_candidate_features(numerical_features=["x", "z"], categorical_features=["a", "b"], order=1)
    dag = reference.FittedOpenFEDAG(candidates).fit(finite_frame)
    result = dag.transform(finite_frame)
    assert result.shape == (len(finite_frame), len(candidates))
    assert set(UNARY + BINARY + GROUP + ["Combine", "CombineThenFreq"]) == reference.OPENFE_OPERATORS


def test_composed_dag_freezes_shared_statistics_and_pickles(finite_frame):
    group = node("GroupByThenMean", leaf("x"), leaf("a"))
    combined = node("Combine", leaf("a"), leaf("b"))
    features = [
        node("-", leaf("x"), group),
        node("freq", group),
        node("freq", combined),
        node("GroupByThenRank", node("square", leaf("x")), combined),
    ]
    dag = reference.FittedOpenFEDAG(features).fit(finite_frame)
    probe = pd.DataFrame({"x": [2.0, 999.0, np.nan], "z": [1, 1, 1], "a": ["red", "new", None], "b": ["v", "u", "w"]})
    before = pickle.dumps(dag)
    whole = dag.transform(probe)
    parts = pd.concat([dag.transform(probe.iloc[:1]), dag.transform(probe.iloc[1:])])
    pd.testing.assert_frame_equal(whole, parts)
    pd.testing.assert_frame_equal(whole, pickle.loads(before).transform(probe))
    assert pickle.dumps(dag) == before
    assert whole.iloc[0, 0] == pytest.approx(2 - 5 / 3)
    assert whole.iloc[0, 1] == 3
    assert whole.iloc[0, 2] == 2
    assert whole.iloc[1, 2] == 0
    assert whole.iloc[0, 3] == 0.75
    assert len(dag.states_) < 12  # The repeated mean/Combine subtrees are cached.
    assert dag.transform(probe.iloc[:0]).empty


def test_rank_policy_ties_unseen_values_and_missing_groups():
    train = pd.DataFrame({"x": [1.0, 2.0, 2.0, np.nan, 5.0], "a": ["a", "a", "a", "a", "b"]})
    dag = reference.FittedOpenFEDAG([node("GroupByThenRank", leaf("x"), leaf("a"))]).fit(train)
    probe = pd.DataFrame({"x": [2.0, 1.5, 0.0, 10.0, 2.0, np.nan], "a": ["a", "a", "a", "a", "new", "a"]})
    np.testing.assert_allclose(dag.transform(probe).iloc[:, 0], [5 / 6, 1 / 3, 0, 1, np.nan, np.nan], equal_nan=True)


def test_typed_pairs_avoid_string_delimiter_and_type_collisions():
    train = pd.DataFrame({"a": ["a_b", "a", 1, "1", None], "b": ["c", "b_c", "x", "x", "x"]})
    dag = reference.FittedOpenFEDAG(
        [node("Combine", leaf("a"), leaf("b")), node("CombineThenFreq", leaf("a"), leaf("b"))]
    ).fit(train)
    generated = dag.transform(train)
    assert generated.iloc[:, 0].astype(float).tolist() == [0, 1, 2, 3, -1]
    np.testing.assert_allclose(generated.iloc[:, 1], [1, 1, 1, 1, np.nan], equal_nan=True)
    unknown = dag.transform(pd.DataFrame({"a": ["new", None], "b": ["x", "x"]}))
    assert unknown.iloc[:, 0].astype(float).tolist() == [-2, -1]
    assert unknown.iloc[0, 1] == 0
    assert unknown.iloc[:, 0].dtype == generated.iloc[:, 0].dtype


def test_missing_and_unknown_statistics_are_explicit():
    train = pd.DataFrame({"x": [1.0, 1.0, np.nan, 4.0], "a": ["a", "a", "a", None]})
    features = [
        node("freq", leaf("x")),
        node("GroupByThenFreq", leaf("x"), leaf("a")),
        node("GroupByThenMean", leaf("x"), leaf("a")),
        node("GroupByThenNUnique", leaf("x"), leaf("a")),
    ]
    dag = reference.FittedOpenFEDAG(features).fit(train)
    probe = pd.DataFrame({"x": [1.0, 99.0, np.nan, 1.0], "a": ["a", "new", "a", None]})
    np.testing.assert_allclose(
        dag.transform(probe).to_numpy(),
        [[2, 2, 1, 1], [0, 0, np.nan, np.nan], [np.nan, np.nan, 1, 1], [2, np.nan, np.nan, np.nan]],
        equal_nan=True,
    )


def test_names_are_candidate_derived_collision_safe_and_stable(finite_frame):
    feature = node("square", leaf("x"))
    first = reference.FittedOpenFEDAG([feature]).fit(finite_frame)
    name = first.transform(finite_frame).columns[0]
    frame = finite_frame.assign(**{name: 0})
    second = reference.FittedOpenFEDAG([node("log", leaf("x")), feature]).fit(frame)
    result = second.transform(frame)
    assert result.columns[1] != name
    assert result.columns.is_unique
    assert not set(result.columns) & set(frame.columns)
    third = reference.FittedOpenFEDAG([feature, node("log", leaf("x"))]).fit(frame)
    assert third.transform(frame).columns[0] == result.columns[1]


@pytest.mark.parametrize("feature", [node("unsupported", leaf("x")), node("square"), leaf("absent")])
def test_unknown_or_malformed_operator_fails(feature, finite_frame):
    with pytest.raises(reference.ReferenceOperatorError):
        reference.FittedOpenFEDAG([feature]).fit(finite_frame)


def call_reference(train, test, tmp_path, **kwargs):
    options = {
        "method": "openfe-inductive",
        "task": "regression",
        "max_added_features": None,
        "threads": 2,
        "seed": 73,
        "workspace": tmp_path,
    }
    y = kwargs.pop("y", pd.Series(np.arange(len(train), dtype=float), index=train.index))
    options.update(kwargs)
    return reference.generate_reference(train, test, y, **options)


def test_openfe_api_uses_full_metadata_only_training_and_native_rank(
    monkeypatch, finite_frame, tmp_path, reference_versions
):
    observed = {}

    def candidates(**kwargs):
        observed["metadata"] = kwargs
        return [node("freq", leaf("x000002")), node("square", leaf("x000000"))]

    class Engine:
        def fit(self, data, label, **kwargs):
            observed["fit"] = kwargs
            assert data.index.equals(label.index)
            assert label.iloc[:, 0].tolist() == [0, 1, 2, 3, 4, 5]
            assert len(data) == len(finite_frame)
            assert set(kwargs["train_index"]).isdisjoint(kwargs["val_index"])
            assert set(kwargs["train_index"]) | set(kwargs["val_index"]) == set(data.index)
            assert kwargs["task"] == "regression"
            assert kwargs["categorical_features"] == ["x000002", "x000003"]
            return kwargs["candidate_features_list"]

    monkeypatch.setattr(reference, "_load_openfe", lambda: (Engine, candidates))
    train = finite_frame
    test = finite_frame.iloc[:2].assign(x=[10000.0, 20000.0], a=["unseen", "red"])
    y = pd.Series(range(6), index=train.index).iloc[::-1]
    result_train, result_test, diagnostics = call_reference(train, test, tmp_path, y=y, max_added_features=1)
    pd.testing.assert_frame_equal(result_train[train.columns], train)
    pd.testing.assert_frame_equal(result_test[test.columns], test)
    assert observed["metadata"] == {
        "numerical_features": ["x000000", "x000001"],
        "categorical_features": ["x000002", "x000003"],
        "order": 1,
    }
    assert result_test.iloc[:, -1].tolist() == [0, 3]
    assert diagnostics["candidate_count"] == 2
    assert diagnostics["selected_count"] == 2
    assert diagnostics["added_features"] == 1
    assert diagnostics["width_policy"] == "native_importance_order"
    assert diagnostics["status"] == "success"
    assert observed["fit"]["n_jobs"] == 2
    assert observed["fit"]["seed"] == 73
    assert not list(tmp_path.glob("*.feather"))


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize(("method", "max_gb"), [("autofeat-default", 1), ("autofeat-native", None)])
def test_autofeat_native_defaults_all_selected_and_frozen_categories(
    monkeypatch, finite_frame, tmp_path, task, reference_versions, method, max_gb
):
    observed = {}

    class Engine:
        def __init__(self, **kwargs):
            observed["kwargs"] = kwargs

        def fit_transform(self, data, label):
            observed["train"] = data.copy()
            assert np.isfinite(data.to_numpy(dtype=float)).all()
            result = data.copy()
            for i in range(55):
                result[f"selected_{i}"] = np.arange(len(data)) + i
            return result.reset_index(drop=True)

        def transform(self, data):
            observed["test"] = data.copy()
            result = data.copy()
            for i in range(55):
                result[f"selected_{i}"] = np.arange(len(data)) + i
            return result.reset_index(drop=True)

    def load(actual_task):
        assert actual_task == task
        return Engine

    monkeypatch.setattr(reference, "_load_autofeat", load)
    train = finite_frame.copy()
    train.iloc[1, 0] = np.nan
    test = train.iloc[:2].assign(x=[np.inf, 100000.0], a=["new", "red"])
    result_train, result_test, diagnostics = call_reference(train, test, tmp_path, method=method, task=task)
    assert observed["kwargs"] == {"max_gb": max_gb, "n_jobs": 2, "verbose": 0}
    assert observed["train"].columns.equals(observed["test"].columns)
    assert observed["test"].iloc[0, 0] == train["x"].median()
    cat_columns = [c for c in observed["test"] if c.startswith("cat_x000002_")]
    assert observed["test"].loc[test.index[0], cat_columns].sum() == 0
    pd.testing.assert_frame_equal(result_train[train.columns], train)
    pd.testing.assert_frame_equal(result_test[test.columns], test)
    assert diagnostics["added_features"] == 55 + len(observed["train"].columns) - 2
    assert diagnostics["method"] == method
    assert diagnostics["max_added_features"] is None
    assert diagnostics["width_policy"] == "native_output"
    assert diagnostics["resources"]["max_gb"] == max_gb
    assert diagnostics["resources"]["hard_memory_limit"] is False
    assert diagnostics["resources"]["hard_deadline"] == "caller"
    assert diagnostics["search"]["feateng_steps"] == 2
    assert diagnostics["search"]["featsel_runs"] == 5
    assert diagnostics["search"]["transformations"] == ["1/", "exp", "log", "abs", "sqrt", "^2", "^3"]
    assert len(diagnostics["feature_definitions"]) == diagnostics["added_features"]
    assert set(diagnostics["preprocessing"]["one_hot_levels"]) == set(observed["train"]) - {"x000000", "x000001"}
    assert set(diagnostics) == {
        "status",
        "method",
        "task",
        "seed",
        "threads",
        "dependency_versions",
        "input_features",
        "input_columns",
        "candidate_count",
        "selected_count",
        "added_features",
        "max_added_features",
        "width_policy",
        "feature_names",
        "feature_definitions",
        "preprocessing",
        "operator_policy",
        "search",
        "resources",
        "seed_policy",
    }
    json.dumps(diagnostics, allow_nan=False)


@pytest.mark.parametrize("cap", [0, 1, 50])
@pytest.mark.parametrize("method", ["autofeat-default", "autofeat-native"])
def test_autofeat_rejects_unranked_width_cap_before_import(cap, finite_frame, tmp_path, monkeypatch, method):
    monkeypatch.setattr(reference, "_load_autofeat", lambda task: pytest.fail("must reject before optional import"))
    with pytest.raises(reference.ReferenceWidthError, match=method):
        call_reference(finite_frame, finite_frame, tmp_path, method=method, max_added_features=cap)


@pytest.mark.parametrize(
    "kwargs",
    [{"task": "auto"}, {"method": "openfe"}, {"threads": 0}, {"max_added_features": -1}, {"seed": -1}],
)
def test_invalid_metadata_raises_specific_errors(kwargs, finite_frame, tmp_path):
    with pytest.raises(reference.ReferenceInputError):
        call_reference(finite_frame, finite_frame, tmp_path, **kwargs)


def test_label_alignment_is_validated_before_dependency_loading(finite_frame, tmp_path):
    for y in (pd.Series(range(6)), pd.Series([np.nan] * 6, index=finite_frame.index)):
        with pytest.raises(reference.ReferenceInputError, match="label"):
            call_reference(finite_frame, finite_frame, tmp_path, y=y)


def test_dependency_and_engine_failures_are_not_baseline_fallback(monkeypatch, finite_frame, tmp_path):
    original_import = importlib.import_module

    def missing(name):
        if name == "openfe":
            raise ImportError("deliberately missing")
        return original_import(name)

    monkeypatch.setattr(reference, "import_module", missing)
    with pytest.raises(reference.ReferenceDependencyError, match="openfe"):
        call_reference(finite_frame, finite_frame, tmp_path)

    class Broken:
        def fit(self, *args, **kwargs):
            raise RuntimeError("search failed")

    monkeypatch.setattr(reference, "_load_openfe", lambda: (Broken, lambda **kwargs: [node("square", leaf("x000000"))]))
    with pytest.raises(RuntimeError, match="search failed"):
        call_reference(finite_frame, finite_frame, tmp_path)
    assert not list(tmp_path.glob("*.feather"))


def test_rng_state_is_restored_on_failure(monkeypatch, finite_frame, tmp_path):
    class Broken:
        def fit(self, *args, **kwargs):
            random.random()
            np.random.rand()
            raise RuntimeError("search failed")

    monkeypatch.setattr(reference, "_load_openfe", lambda: (Broken, lambda **kwargs: [node("square", leaf("x000000"))]))
    numpy_state = pickle.dumps(np.random.get_state())
    python_state = random.getstate()
    with pytest.raises(RuntimeError):
        call_reference(finite_frame, finite_frame, tmp_path)
    assert pickle.dumps(np.random.get_state()) == numpy_state
    assert random.getstate() == python_state


def test_numeric_signed_zero_uses_equal_frequency_keys():
    frame = pd.DataFrame({"x": [0.0, -0.0, 1.0]})
    result = reference.FittedOpenFEDAG([node("freq", leaf("x"))]).fit(frame).transform(frame)
    assert result.iloc[:, 0].tolist() == [2, 2, 1]


def test_every_operator_is_batch_invariant_with_unseen_and_missing(finite_frame):
    features = (
        [node(op, leaf("x")) for op in UNARY]
        + [node(op, leaf("x"), leaf("z")) for op in BINARY]
        + [node(op, leaf("x"), leaf("a")) for op in GROUP]
        + [node(op, leaf("a"), leaf("b")) for op in ("Combine", "CombineThenFreq")]
    )
    probe = pd.DataFrame(
        {
            "x": [-2.0, 0, 100, np.nan],
            "z": [0.0, 2, 1, np.nan],
            "a": ["red", "new", "blue", None],
            "b": ["v", "u", "new", None],
        },
        index=pd.Index(["last", "first", "middle", "missing"], name="custom"),
    )
    dag = reference.FittedOpenFEDAG(features).fit(finite_frame)
    whole = dag.transform(probe)
    parts = pd.concat([dag.transform(probe.iloc[i : i + 1]) for i in range(len(probe))])
    pd.testing.assert_frame_equal(whole, parts)
    pd.testing.assert_frame_equal(whole, pickle.loads(pickle.dumps(dag)).transform(probe))
    pd.testing.assert_frame_equal(whole.iloc[[1, 0]], dag.transform(probe.iloc[[1, 0]]))


def test_composed_official_nodes_match_training_values(finite_frame):
    fg = pytest.importorskip("openfe.FeatureGenerator")
    combined = fg.Node("Combine", [fg.FNode("a"), fg.FNode("b")])
    features = [
        fg.Node("GroupByThenMean", [fg.Node("square", [fg.FNode("x")]), combined]),
        fg.Node("freq", [combined]),
        fg.Node("GroupByThenRank", [fg.FNode("x"), combined]),
    ]
    dag = reference.FittedOpenFEDAG(features).fit(finite_frame)
    for i, feature in enumerate(features):
        np.testing.assert_allclose(dag.transform(finite_frame).iloc[:, i], feature.calculate(finite_frame))


@pytest.mark.parametrize("cap", [None, 0, 1, 10])
def test_openfe_width_policy_does_not_change_search(monkeypatch, finite_frame, tmp_path, reference_versions, cap):
    candidates = [node("freq", leaf("x000002")), node("square", leaf("x000000"))]

    class Engine:
        def fit(self, data, label, **kwargs):
            assert kwargs["candidate_features_list"] == candidates
            return candidates

    monkeypatch.setattr(reference, "_load_openfe", lambda: (Engine, lambda **kwargs: candidates))
    result, heldout, diagnostics = call_reference(finite_frame, finite_frame.iloc[:0], tmp_path, max_added_features=cap)
    expected = 2 if cap is None else min(cap, 2)
    assert result.shape[1] == len(finite_frame.columns) + expected
    assert heldout.empty
    assert result.columns.equals(heldout.columns)
    assert diagnostics["selected_count"] == 2
    assert diagnostics["added_features"] == expected


def test_openfe_classification_routes_and_stratifies_training_labels(monkeypatch, tmp_path, reference_versions):
    train = pd.DataFrame({"x": np.arange(20, dtype=float)}, index=[f"train{i}" for i in range(20)])
    y = pd.Series(["class0", "class1"] * 10, index=train.index).iloc[::-1]
    features = [node("square", leaf("x000000"))]

    class Engine:
        def fit(self, data, label, **kwargs):
            assert kwargs["task"] == "classification"
            assert label.iloc[:, 0].tolist() == ["class0", "class1"] * 10
            assert set(label.loc[kwargs["train_index"], "target"]) == {"class0", "class1"}
            assert set(label.loc[kwargs["val_index"], "target"]) == {"class0", "class1"}
            assert not set(data.x000000) & {900.0, 999.0}
            return features

    monkeypatch.setattr(reference, "_load_openfe", lambda: (Engine, lambda **kwargs: features))
    _, _, diagnostics = call_reference(train, pd.DataFrame({"x": [900.0, 999.0]}), tmp_path, task="classification", y=y)
    assert diagnostics["search"]["fit_rows"] == 16
    assert diagnostics["search"]["validation_rows"] == 4


@pytest.mark.parametrize("mode", ["array", "rows", "duplicate", "schema"])
@pytest.mark.parametrize("method", ["autofeat-default", "autofeat-native"])
def test_autofeat_invalid_output_raises(monkeypatch, finite_frame, tmp_path, mode, method):
    class Engine:
        def __init__(self, **kwargs):
            pass

        def fit_transform(self, data, label):
            if mode == "array":
                return data.to_numpy()
            if mode == "rows":
                return data.iloc[:-1]
            if mode == "duplicate":
                return pd.concat([data, data], axis=1)
            return data

        def transform(self, data):
            return data.assign(unexpected=1)

    monkeypatch.setattr(reference, "_load_autofeat", lambda task: Engine)
    with pytest.raises(reference.ReferenceOutputError):
        call_reference(finite_frame, finite_frame, tmp_path, method=method)


@pytest.mark.parametrize(("method", "max_gb"), [("autofeat-default", 1), ("autofeat-native", None)])
@pytest.mark.parametrize("failure_stage", ["fit", "transform"])
def test_autofeat_upstream_failure_propagates(monkeypatch, finite_frame, tmp_path, method, max_gb, failure_stage):
    failure = ValueError("Input X contains NaN.")

    class Engine:
        def __init__(self, **kwargs):
            assert kwargs == {"max_gb": max_gb, "n_jobs": 2, "verbose": 0}

        def fit_transform(self, data, label):
            assert data["x000000"].eq(0).all()
            assert data["cat_x000002_0"].eq(1).all()
            assert np.isfinite(data.to_numpy(dtype=float)).all()
            if failure_stage == "fit":
                raise failure
            return data

        def transform(self, data):
            assert data["x000000"].eq(0).all()
            raise failure

    monkeypatch.setattr(reference, "_load_autofeat", lambda task: Engine)
    train = finite_frame.assign(x=0.0, a="constant")
    with pytest.raises(ValueError, match="Input X contains NaN") as caught:
        call_reference(train, train, tmp_path, method=method)
    assert caught.value is failure


def test_openfe_unknown_candidate_is_not_filtered(monkeypatch, finite_frame, tmp_path):
    def forbidden():
        pytest.fail("Unknown operators must fail before search")

    monkeypatch.setattr(
        reference,
        "_load_openfe",
        lambda: (forbidden, lambda **kwargs: [node("square", leaf("x000000")), node("future_op", leaf("x000000"))]),
    )
    with pytest.raises(reference.ReferenceOperatorError, match="future_op"):
        call_reference(finite_frame, finite_frame, tmp_path)


@pytest.mark.parametrize(
    "mutation",
    ["duplicate_columns", "mismatched_columns", "duplicate_index", "nonstring_columns", "complex"],
)
def test_invalid_frame_metadata_is_rejected(finite_frame, tmp_path, mutation, monkeypatch):
    monkeypatch.setattr(reference, "_load_openfe", lambda: pytest.fail("Reject metadata before optional imports"))
    train = finite_frame.copy()
    test = finite_frame.copy()
    if mutation == "duplicate_columns":
        train.columns = ["x", "x", "a", "b"]
    elif mutation == "mismatched_columns":
        test = test[test.columns[::-1]]
    elif mutation == "duplicate_index":
        train.index = [0] * len(train)
    elif mutation == "nonstring_columns":
        train.columns = test.columns = [0, 1, 2, 3]
    else:
        train["x"] = train.x.astype(complex) + 1j
    with pytest.raises(reference.ReferenceInputError):
        call_reference(train, test, tmp_path)


def test_preparation_handles_nullable_numeric_missing_and_typed_categories():
    train = pd.DataFrame(
        {
            "n": pd.Series([1, pd.NA, 3], dtype="Int64"),
            "empty": pd.Series([pd.NA] * 3, dtype="Float64"),
            "cat": pd.Categorical(["a", None, "missing:"]),
            "flag": pd.Series([True, False, pd.NA], dtype="boolean"),
        }
    )
    prepared = reference._PreparedInput(train)
    data = prepared.transform(train)
    assert data.x000000.tolist() == [1, 2, 3]
    assert data.x000001.tolist() == [0, 0, 0]
    assert prepared.all_missing == ["x000001"]
    assert prepared.categorical == ["x000002", "x000003"]
    assert pd.isna(data.x000002.iloc[1])
    assert data.x000002.iloc[2] != "missing:"
    assert pd.isna(data.x000003.iloc[2])


def test_compatibility_versions_fail_clearly(monkeypatch, reference_versions):
    original = reference.version
    monkeypatch.setattr(reference, "version", lambda name: "1.8.0" if name == "scikit-learn" else original(name))
    with pytest.raises(reference.ReferenceDependencyError, match="scikit-learn"):
        reference._require("autofeat", "2.1.1")
    with pytest.raises(reference.ReferenceDependencyError, match="requires 0.0.99"):
        reference._require("openfe", "0.0.99")


@pytest.mark.parametrize("max_gb", [1, None])
def test_real_autofeat_constructor_defaults_and_task_routing(reference_versions, max_gb):
    pytest.importorskip("autofeat")
    for task, class_name in (("classification", "AutoFeatClassifier"), ("regression", "AutoFeatRegressor")):
        engine = reference._load_autofeat(task)(max_gb=max_gb, n_jobs=1, verbose=0)
        assert type(engine).__name__ == class_name
        assert engine.max_gb == max_gb
        assert engine.feateng_steps == 2
        assert engine.featsel_runs == 5
        assert engine.transformations == ("1/", "exp", "log", "abs", "sqrt", "^2", "^3")


def test_import_does_not_load_optional_reference_packages():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import benchmarks.reference_methods; "
            "assert 'openfe' not in sys.modules; assert 'autofeat' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
