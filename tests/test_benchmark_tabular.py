"""Leakage, pairing and failure-contract tests for the canonical tabular benchmark."""

import json
import os
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from benchmarks import tabular


def config():
    return {"learners": ["linear"], "threads": 1, "max_added_features": 4, "seed": 42}


def payload():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"x0": rng.normal(size=60), "x1": rng.normal(size=60)})
    y = pd.Series(X.x0 * 2 + rng.normal(size=60))
    return {
        "X": X,
        "y": y,
        "config": config(),
        "dataset": "test",
        "source": "synthetic",
        "task": "regression",
        "method": "baseline",
        "seed": 42,
        "fold": 0,
        "train_idx": np.arange(40),
        "test_idx": np.arange(40, 60),
        "split_sha256": "split",
    }


def test_source_helpers_cover_actual_sources_and_ignore_generated_evidence(tmp_path, monkeypatch):
    included = [
        "featcopilot/core.py",
        "featcopilot/llm/provider.py",
        "benchmarks/runner.py",
        "pyproject.toml",
        "benchmarks/tabular_suite.json",
        "benchmarks/families.json",
        "benchmarks/suites/test.json",
    ]
    excluded = [
        "featcopilot/__pycache__/cached.py",
        "benchmarks/.cache/generated.py",
        "benchmarks/tabular-results/run/runner.py",
        "reviews/review.py",
        "submissions/script.py",
    ]
    for name in included + excluded:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"source bytes\n")
    monkeypatch.setattr(tabular, "ROOT", tmp_path)
    monkeypatch.setattr(tabular, "SUITE_PATH", tmp_path / "benchmarks" / "tabular_suite.json")
    selected = tabular.source_files()
    assert {path.relative_to(tmp_path).as_posix() for path in selected} == set(included)
    expected = b"".join(path.relative_to(tmp_path).as_posix().encode() + b"\0" + path.read_bytes() for path in selected)
    assert tabular.source_fingerprint() == tabular.digest(expected)
    before = tabular.source_fingerprint()
    (tmp_path / "featcopilot" / "core.py").write_bytes(b"changed source\n")
    assert tabular.source_fingerprint() != before


def test_source_fingerprint_uses_zip_names_even_for_windows_relative_paths(monkeypatch):
    relative = PureWindowsPath("pkg", "module.py")
    assert str(relative) == "pkg\\module.py"
    source = SimpleNamespace(relative_to=lambda root: relative, read_bytes=lambda: b"module bytes")
    monkeypatch.setattr(tabular, "source_files", lambda: [source])
    assert tabular.source_fingerprint() == tabular.digest(b"pkg/module.py\0module bytes")


def test_source_walk_prunes_excluded_trees_before_descent_without_changing_allowlist(tmp_path, monkeypatch):
    excluded = {"tabular-results", ".cache", "__pycache__"}
    sources = [
        "featcopilot/z.py",
        "featcopilot/nested/a.py",
        "featcopilot/upper.PY",
        "benchmarks/z.py",
        "benchmarks/nested/a.py",
        "pyproject.toml",
        "benchmarks/tabular_suite.json",
        "benchmarks/families.json",
        "benchmarks/suites/z.json",
        "benchmarks/suites/a.json",
    ]
    for name in sources:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    for tree in ("featcopilot", "benchmarks", "benchmarks/nested"):
        for ignored in excluded:
            directory = tmp_path / tree / ignored / "deep" / "generated"
            directory.mkdir(parents=True)
            for index in range(20):
                (directory / f"file-{index}.py").write_bytes(b"excluded")
    legacy = sorted((tmp_path / "featcopilot").rglob("*.py")) + sorted((tmp_path / "benchmarks").rglob("*.py"))
    expected = [path for path in legacy if not excluded.intersection(path.relative_to(tmp_path).parts)]
    expected += [tmp_path / name for name in sources[5:8]]
    expected += sorted((tmp_path / "benchmarks" / "suites").glob("*.json"))
    expected_hash = tabular.digest(
        b"".join(path.relative_to(tmp_path).as_posix().encode() + b"\0" + path.read_bytes() for path in expected)
    )
    scandir = os.scandir
    scanned = []

    def guarded_scandir(directory):
        path = Path(directory)
        assert not excluded.intersection(path.relative_to(tmp_path).parts), f"Descended into excluded tree: {path}"
        scanned.append(path)
        return scandir(directory)

    monkeypatch.setattr(tabular, "ROOT", tmp_path)
    monkeypatch.setattr(tabular, "SUITE_PATH", tmp_path / "benchmarks" / "tabular_suite.json")
    monkeypatch.setattr(os, "scandir", guarded_scandir)
    assert tabular.source_files() == expected
    assert tabular.source_fingerprint() == expected_hash
    (tmp_path / "benchmarks" / "tabular-results" / "deep" / "generated" / "file-0.py").write_bytes(b"changed")
    assert tabular.source_fingerprint() == expected_hash
    assert tmp_path / "featcopilot" / "nested" in scanned
    assert tmp_path / "benchmarks" / "nested" in scanned


def test_source_fingerprint_detects_same_size_same_mtime_edits(tmp_path, monkeypatch):
    source = tmp_path / "featcopilot" / "module.py"
    source.parent.mkdir()
    source.write_bytes(b"VALUE = 1\n")
    monkeypatch.setattr(tabular, "ROOT", tmp_path)
    monkeypatch.setattr(tabular, "source_files", lambda: [source])
    before = tabular.source_fingerprint()
    original = source.stat()
    source.write_bytes(b"VALUE = 2\n")
    os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert source.stat().st_size == original.st_size
    assert source.stat().st_mtime_ns == original.st_mtime_ns
    assert tabular.source_fingerprint() != before


def test_source_walk_preserves_glob_candidates_instead_of_silently_dropping_them(tmp_path, monkeypatch):
    directory = tmp_path / "benchmarks" / "unexpected.py"
    directory.mkdir(parents=True)
    monkeypatch.setattr(tabular, "ROOT", tmp_path)
    monkeypatch.setattr(tabular, "SUITE_PATH", tmp_path / "benchmarks" / "tabular_suite.json")
    assert directory in tabular.source_files()
    with pytest.raises(OSError):
        tabular.source_fingerprint()


def test_regression_prediction_capture_preserves_complete_procedure(tmp_path):
    case = payload()
    case["method"] = "baseline-robust"
    ordinary = tabular.run_case(case, tmp_path)[0]
    captured = {}
    repeated = tabular.run_case(case, tmp_path, outer_predictions=captured)[0]
    assert set(captured) == {"linear"}
    assert captured["linear"].shape == (20,)
    assert tabular.regression_metrics(captured["linear"], case["y"].iloc[case["test_idx"]])["loss"] == ordinary["loss"]
    for key in (
        "loss",
        "r2",
        "mae",
        "parameters",
        "inner_losses",
        "n_output_features",
        "outer_prediction_sha256",
        "outer_feature_names_sha256",
    ):
        assert repeated[key] == ordinary[key]
    assert repeated["outer_prediction_sha256"] == tabular.digest(np.asarray(captured["linear"], dtype="<f8").tobytes())


@pytest.mark.parametrize(
    "prediction",
    [
        np.zeros((3, 1)),
        np.zeros(2),
        np.array([0.0, np.nan, 2.0]),
        np.array([0.0, np.inf, 2.0]),
        np.array([0j, 1j, 2j]),
        np.array(["0", "1", "2"]),
        np.array([True, False, True]),
    ],
)
def test_regression_metrics_reject_invalid_prediction_shapes_and_values(prediction):
    with pytest.raises(ValueError, match="Regression predictions"):
        tabular.regression_metrics(prediction, pd.Series([0.0, 1.0, 2.0]))


def test_outer_prediction_capture_rejects_classification(tmp_path):
    case = payload()
    case["task"] = "classification"
    with pytest.raises(ValueError, match="capture supports regression only"):
        tabular.run_case(case, tmp_path, outer_predictions={})


def test_preprocessing_uses_training_medians_and_preserves_missing_category():
    train = pd.DataFrame({"numeric": [1.0, np.nan, 3.0], "cat": pd.Categorical(["a", None, "missing:"])})
    test = pd.DataFrame({"numeric": [np.nan, 1000.0], "cat": ["new", None]})
    prepared_train, prepared_test = tabular.prepare_numeric(train, test)
    assert prepared_train.numeric.tolist() == [1, 2, 3]
    assert prepared_test.numeric.tolist() == [2, 1000]
    assert prepared_train.cat.tolist() == ["value:a", "missing:", "value:missing:"]
    assert prepared_test.cat.tolist() == ["value:new", "missing:"]


def test_preprocessing_can_preserve_missingness_without_fitting_a_median():
    train = pd.DataFrame({"numeric": [1.0, np.nan, np.inf], "cat": ["a", None, "missing:"]})
    test = pd.DataFrame({"numeric": [-np.inf, np.nan, 1000.0], "cat": ["new", None, "a"]})
    original = train.copy(deep=True)
    with patch.object(pd.Series, "median", side_effect=AssertionError("Premature median fit")):
        prepared_train, prepared_test = tabular.prepare_numeric(train, test, impute=False)
    assert prepared_train.numeric.isna().tolist() == [False, True, True]
    assert prepared_test.numeric.isna().tolist() == [True, True, False]
    assert prepared_train.cat.tolist() == ["value:a", "missing:", "value:missing:"]
    pd.testing.assert_frame_equal(train, original)


@pytest.mark.parametrize("method", sorted(tabular.PREDICTIVE_METHODS))
def test_predictive_engine_receives_unimputed_values_for_internal_training_splits(method, tmp_path):
    train = pd.DataFrame({"numeric": [1.0, np.nan, 1000.0], "cat": ["a", "b", "c"]})
    test = pd.DataFrame({"numeric": [np.nan, np.inf], "cat": ["a", "new"]})
    y = pd.Series([0.0, 1.0, 2.0])
    with patch("featcopilot.AutoFeatureEngineer") as constructor:
        constructor.return_value.fit_transform.side_effect = lambda X, *args, **kwargs: X
        constructor.return_value.transform.side_effect = lambda X: X
        constructor.return_value.selection_report_ = None
        tabular.generate_features(
            train,
            test,
            y,
            method,
            "classification" if method in tabular.TARGET_ENCODING_METHODS else "regression",
            {**config(), "selection_learner": "linear"},
            tmp_path,
        )
    fitted = constructor.return_value.fit_transform.call_args.args[0]
    transformed = constructor.return_value.transform.call_args.args[0]
    assert fitted.numeric.tolist()[0] == 1.0
    assert fitted.numeric.isna().tolist() == [False, True, False]
    assert transformed.numeric.isna().all()


@pytest.mark.parametrize("method", ["llm-plan", "llm-hybrid", "llm-hybrid-namespaced"])
def test_plan_and_hybrid_entry_preserves_numeric_missingness_without_live_calls(method, tmp_path):
    train = pd.DataFrame({"numeric": [1.0, np.nan, 3.0]})
    test = pd.DataFrame({"numeric": [np.nan]})
    y = pd.Series([1.0, 2.0, 3.0])
    seen = []

    def stop_before_provider(fit_X, fit_y, test_X, **kwargs):
        seen.append((fit_X, test_X))
        raise RuntimeError("Stop before any provider or ledger call")

    with (
        patch("benchmarks.llm_evaluation.CopilotPlanProvider") as provider,
        patch("benchmarks.llm_evaluation.EvaluationLedger") as ledger,
        patch("benchmarks.llm_evaluation.evaluate_plan_fit", side_effect=stop_before_provider),
        pytest.raises(RuntimeError, match="Stop before any provider"),
    ):
        tabular.generate_features(
            train,
            test,
            y,
            method,
            "regression",
            {**config(), "selection_learner": "linear", "live_llm": True, "llm_ledger": str(tmp_path)},
            tmp_path,
        )
    assert seen[0][0].numeric.isna().tolist() == [False, True, False]
    assert seen[0][1].numeric.isna().all()
    provider.return_value.generate.assert_not_called()
    ledger.return_value.reserve.assert_not_called()


def test_internal_preprocessor_median_excludes_validation_values(tmp_path):
    from sklearn.impute import SimpleImputer

    train = pd.DataFrame({"numeric": np.where(np.arange(80) % 2, np.nan, np.arange(80) + 1.0)})
    y = pd.Series(np.sin(np.arange(80)) + np.arange(80) * 0.1)
    fitted_subsets = []
    original_fit = SimpleImputer.fit

    def inspect_fit(imputer, X, y=None):
        result = original_fit(imputer, X, y)
        if isinstance(X, pd.DataFrame) and "numeric" in X.columns:
            expected = train.loc[X.index, "numeric"]
            pd.testing.assert_series_equal(X["numeric"], expected, check_dtype=False)
            assert imputer.statistics_[X.columns.get_loc("numeric")] == expected.median()
            fitted_subsets.append((len(X), bool(X.numeric.isna().any())))
        return result

    with patch.object(SimpleImputer, "fit", inspect_fit):
        tabular.generate_features(
            train,
            pd.DataFrame({"numeric": [np.nan, 1e12]}, index=[100, 101]),
            y,
            "featcopilot-frequency-paired",
            "regression",
            {**config(), "selection_learner": "linear", "max_candidates": 20, "stability_repeats": 1},
            tmp_path,
        )
    assert any(size < len(train) and missing for size, missing in fitted_subsets)


@pytest.mark.parametrize(
    "method,expected_calls",
    [("baseline", 4), ("featcopilot-frequency-aligned", 8), ("featcopilot-frequency-aligned-robust", 8)],
)
def test_inner_fit_and_outer_refit_never_receive_outer_test_targets(method, expected_calls, tmp_path):
    case = payload()
    case["method"] = method
    case["config"].update(max_candidates=20, stability_repeats=1)
    original = tabular.generate_features
    calls = []

    def record(train, test, y, *args):
        calls.append((train.index.tolist(), test.index.tolist(), y.index.tolist()))
        assert train.index.equals(y.index)
        assert set(train.index).isdisjoint(test.index)
        return original(train, test, y, *args)

    with patch.object(tabular, "generate_features", side_effect=record):
        first = tabular.run_case(case, tmp_path)
        case["y"].iloc[40:] += 10000
        second = tabular.run_case(case, tmp_path)
    assert len(calls) == expected_calls
    assert all(set(train).issubset(set(range(40))) for train, _, _ in calls)
    assert all(set(target).issubset(set(range(40))) for _, _, target in calls)
    assert first[0]["inner_losses"] == second[0]["inner_losses"]
    assert first[0]["parameters"] == second[0]["parameters"]
    assert first[0]["loss"] != second[0]["loss"]
    if method in tabular.ALIGNED_METHODS:
        assert first[0]["selection_model_parameters"] == first[0]["parameters"]
        assert len(first[0]["inner_pipeline_trials"]) == 3


def test_robust_raw_baseline_defers_imputation_until_its_learner(tmp_path):
    train = pd.DataFrame({"number": [1.0, np.nan, 1000.0], "flag": [True, True, True]})
    test = pd.DataFrame({"number": [np.nan], "flag": [False]})
    with patch.object(pd.Series, "median", side_effect=AssertionError("Premature median fit")):
        prepared_train, prepared_test = tabular.generate_features(
            train, test, pd.Series([1.0, 2.0, 3.0]), "baseline-robust", "regression", config(), tmp_path
        )
    assert prepared_train.number.isna().tolist() == [False, True, False]
    assert prepared_test.number.isna().all()
    assert prepared_train.flag.dtype == train.flag.dtype
    assert prepared_test.flag.tolist() == [False]
    assert "baseline-robust" not in tabular.DEFAULT_METHODS


@pytest.mark.parametrize("method", sorted(tabular.PLAN_VALIDATION_METHODS))
def test_fixed_plan_methods_cannot_bypass_the_complete_protocol(method, tmp_path):
    case = payload()
    with patch.object(tabular, "prepare_numeric") as prepare:
        with pytest.raises(ValueError, match="complete discovery/gate/refit protocol"):
            tabular.generate_features(case["X"], case["X"], case["y"], method, "regression", config(), tmp_path)
    prepare.assert_not_called()
    assert method not in tabular.DEFAULT_METHODS


@pytest.mark.parametrize("method", sorted(tabular.PLAN_VALIDATION_METHODS))
def test_fixed_plan_dispatch_preserves_raw_input_and_independent_learner_cells(method, tmp_path):
    case = payload()
    case["X"].loc[0, "x0"] = np.nan
    case["method"] = method
    case["config"] = {**case["config"], "learners": ["linear", "lightgbm"]}
    seen = []

    def run_plan(current, workspace):
        assert current["X"].isna().equals(case["X"].isna())
        assert len(current["config"]["learners"]) == 1
        assert current["config"]["numeric_clip_quantiles"] == (0.0, 1.0)
        seen.append(current["config"]["learners"][0])
        return [{"status": "success", "learner": seen[-1]}]

    with patch("benchmarks.plan_validation.run_case", side_effect=run_plan):
        rows = tabular.run_case(case, tmp_path)
    assert seen == ["linear", "lightgbm"]
    assert len(rows) == 2


@pytest.mark.parametrize("method", ["baseline", "polynomial", "featcopilot", "featcopilot-task"])
def test_methods_preserve_rows_originals_and_schema(method, tmp_path):
    case = payload()
    train, test = case["X"].iloc[:40], case["X"].iloc[40:]
    result_train, result_test = tabular.generate_features(
        train, test, case["y"].iloc[:40], method, "regression", config(), tmp_path
    )
    assert result_train.index.equals(train.index)
    assert result_test.index.equals(test.index)
    assert result_train.columns.equals(result_test.columns)
    pd.testing.assert_frame_equal(result_train[train.columns], train)
    assert result_train.shape[1] <= train.shape[1] + config()["max_added_features"]


def test_predictive_method_fits_each_learner_representation_without_test_targets(tmp_path):
    case = payload()
    case["method"] = "featcopilot-predictive"
    case["config"] = {**case["config"], "learners": ["linear", "random_forest"]}
    observed = []

    def generate(train, test, y, method, task, settings, workspace):
        assert set(y.index).issubset(range(40))
        assert train.index.equals(y.index)
        learner = settings["selection_learner"]
        observed.append(learner)
        train, test = tabular.prepare_numeric(train, test)
        train.attrs["selection_report"] = {"estimator": learner}
        return train, test

    with patch.object(tabular, "generate_features", side_effect=generate):
        rows = tabular.run_case(case, tmp_path)
    assert observed == ["linear", "linear", "random_forest", "random_forest"]
    assert [row["learner"] for row in rows] == ["linear", "random_forest"]
    assert all(row["outer_selection_report"]["estimator"] == row["learner"] for row in rows)


def test_task_ablation_changes_only_explicit_routing(tmp_path):
    case = payload()
    with patch("featcopilot.AutoFeatureEngineer") as cls:
        cls.return_value.fit_transform.return_value = case["X"]
        cls.return_value.transform.return_value = case["X"]
        cls.return_value.selection_report_ = None
        tabular.generate_features(case["X"], case["X"], case["y"], "featcopilot-task", "regression", config(), tmp_path)
    assert cls.call_args.kwargs["task"] == "regression"
    assert cls.call_args.kwargs["selection_strategy"] == "legacy"
    assert cls.call_args.kwargs["selection_estimator"] is None


def test_paired_variant_is_explicit_and_not_a_default(tmp_path):
    case = payload()
    settings = {**config(), "selection_learner": "linear"}
    with patch("featcopilot.AutoFeatureEngineer") as cls:
        cls.return_value.fit_transform.return_value = case["X"]
        cls.return_value.transform.return_value = case["X"]
        cls.return_value.selection_report_ = None
        tabular.generate_features(
            case["X"], case["X"], case["y"], "featcopilot-paired", "regression", settings, tmp_path
        )
    assert cls.call_args.kwargs["selection_rule"] == "paired_gain"
    assert "featcopilot-paired" not in tabular.DEFAULT_METHODS


@pytest.mark.parametrize(
    "method,rule", [("featcopilot-extended", "one_se"), ("featcopilot-extended-paired", "paired_gain")]
)
def test_extended_variants_preserve_explicit_candidate_and_rule_identity(method, rule, tmp_path):
    case = payload()
    with patch("featcopilot.AutoFeatureEngineer") as cls:
        cls.return_value.fit_transform.return_value = case["X"]
        cls.return_value.transform.return_value = case["X"]
        cls.return_value.selection_report_ = None
        tabular.generate_features(
            case["X"],
            case["X"],
            case["y"],
            method,
            "regression",
            {**config(), "selection_learner": "linear"},
            tmp_path,
        )
    assert cls.call_args.kwargs["candidate_set"] == "extended"
    assert cls.call_args.kwargs["selection_rule"] == rule
    assert method not in tabular.DEFAULT_METHODS


@pytest.mark.parametrize(
    "method,rule",
    [
        ("featcopilot-importance", "one_se"),
        ("featcopilot-importance-paired", "paired_gain"),
        ("featcopilot-frequency-paired", "paired_gain"),
        ("featcopilot-frequency-aligned", "paired_gain"),
        ("featcopilot-frequency-aligned-robust", "paired_gain"),
        ("featcopilot-bagged-aligned-robust", "paired_gain"),
    ],
)
def test_importance_variants_preserve_explicit_screening_identity(method, rule, tmp_path):
    case = payload()
    with patch("featcopilot.AutoFeatureEngineer") as cls:
        cls.return_value.fit_transform.return_value = case["X"]
        cls.return_value.transform.return_value = case["X"]
        cls.return_value.selection_report_ = None
        tabular.generate_features(
            case["X"],
            case["X"],
            case["y"],
            method,
            "regression",
            {**config(), "selection_learner": "linear"},
            tmp_path,
        )
    assert cls.call_args.kwargs["screening"] == "importance"
    assert cls.call_args.kwargs["candidate_set"] == "extended"
    assert cls.call_args.kwargs["selection_rule"] == rule
    assert cls.call_args.kwargs["include_frequency"] is (
        method in {"featcopilot-frequency-paired"} | tabular.ALIGNED_METHODS
    )
    assert cls.call_args.kwargs["numeric_clip_quantiles"] == (
        tabular.ROBUST_CLIP_QUANTILES if method in tabular.ROBUST_METHODS else None
    )
    assert cls.call_args.kwargs["importance_resamples"] == (5 if method == "featcopilot-bagged-aligned-robust" else 1)
    assert method not in tabular.DEFAULT_METHODS


@pytest.mark.parametrize(
    "method,expected_alpha",
    [
        ("featcopilot-frequency-aligned", 0.1),
        ("featcopilot-frequency-aligned-robust", 0.1),
        ("featcopilot-bagged-aligned-robust", 0.1),
        ("featcopilot-frequency-paired", 1.0),
    ],
)
def test_only_aligned_variant_uses_trial_selection_parameters(method, expected_alpha, tmp_path):
    case = payload()
    with patch("featcopilot.AutoFeatureEngineer") as constructor:
        constructor.return_value.fit_transform.return_value = case["X"]
        constructor.return_value.transform.return_value = case["X"]
        constructor.return_value.selection_report_ = None
        tabular.generate_features(
            case["X"],
            case["X"],
            case["y"],
            method,
            "regression",
            {
                **config(),
                "selection_learner": "linear",
                "selection_model_parameters": {"regularization": 0.1},
            },
            tmp_path,
        )
    assert constructor.call_args.kwargs["selection_estimator"].alpha == expected_alpha


def test_aligned_variant_rejects_undeclared_search_values(tmp_path):
    case = payload()
    with pytest.raises(ValueError, match="declared learner search"):
        tabular.generate_features(
            case["X"],
            case["X"],
            case["y"],
            "featcopilot-frequency-aligned",
            "regression",
            {**config(), "selection_learner": "linear", "selection_model_parameters": {"regularization": 2.0}},
            tmp_path,
        )


@pytest.mark.parametrize(
    "method,expected_fit_alphas",
    [
        ("featcopilot-frequency-aligned", [0.1, 1.0, 10.0, 0.1]),
        ("featcopilot-frequency-aligned-robust", [0.1, 1.0, 10.0, 0.1]),
        ("featcopilot-bagged-aligned-robust", [0.1, 1.0, 10.0, 0.1]),
        ("featcopilot-frequency-paired", [1.0, 1.0]),
    ],
)
def test_aligned_tuning_refits_each_trial_and_deploys_the_selected_configuration(method, expected_fit_alphas, tmp_path):
    case = payload()
    case["method"] = method
    observed = []

    def generate(train, test, target, name, task, settings, workspace):
        assert train.index.equals(target.index)
        assert set(train.index).issubset(set(case["train_idx"]))
        assert set(train.index).isdisjoint(test.index)
        assert settings.get("numeric_clip_quantiles") == (
            tabular.ROBUST_CLIP_QUANTILES if method in tabular.ROBUST_METHODS else None
        )
        alpha = settings.get("selection_model_parameters", {"regularization": 1.0})["regularization"]
        observed.append((alpha, train.index.tolist(), test.index.tolist()))
        transformed_train = train.assign(engineered=train.x0**2)
        transformed_test = test.assign(engineered=test.x0**2)
        transformed_train.attrs["selection_report"] = {"selected_features": [f"alpha_{alpha}"]}
        return transformed_train, transformed_test

    def score(model, X, y, task):
        return {"loss": {0.1: 1.0, 1.0: 2.0, 10.0: 3.0}[model.steps[-1][1].alpha]}

    with (
        patch.object(tabular, "generate_features", side_effect=generate),
        patch.object(tabular, "metrics", side_effect=score),
    ):
        row = tabular.run_case(case, tmp_path)[0]
    assert [item[0] for item in observed] == expected_fit_alphas
    assert row["parameters"] == {"regularization": 0.1}
    assert row["inner_losses"] == [1.0, 2.0, 3.0]
    assert row["inner_selection_report"]["selected_features"] == [f"alpha_{expected_fit_alphas[0]}"]
    assert row["outer_selection_report"]["selected_features"] == [f"alpha_{expected_fit_alphas[-1]}"]
    assert observed[-1][1:] == (case["train_idx"].tolist(), case["test_idx"].tolist())
    assert row["inner_fe_seconds"] > 0
    assert row["tuning_seconds"] >= 0
    if method in tabular.ALIGNED_METHODS:
        assert len({tuple(item[1]) for item in observed[:-1]}) == 1
        assert row["selection_model_parameters"] == row["parameters"]
        assert [trial["parameters"] for trial in row["inner_pipeline_trials"]] == tabular.candidate_parameters("linear")
    else:
        assert "selection_model_parameters" not in row
        assert "inner_pipeline_trials" not in row
    if method in tabular.ROBUST_METHODS:
        assert row["matched_baseline_method"] == "baseline-robust"
        assert row["numeric_clip_quantiles"] == list(tabular.ROBUST_CLIP_QUANTILES)


@pytest.mark.parametrize(
    "method,reference,width",
    [
        ("openfe-inductive", "openfe-inductive", 4),
        ("openfe-inductive-native", "openfe-inductive", None),
        ("openfe-inductive-native-robust", "openfe-inductive", None),
        ("autofeat-default", "autofeat-default", None),
        ("autofeat-native", "autofeat-native", None),
        ("autofeat-native-robust", "autofeat-native", None),
        ("autofeat-native-safe-scale-v1-robust", "autofeat-native-safe-scale-v1", None),
    ],
)
def test_full_reference_variants_have_explicit_width_policy(method, reference, width, tmp_path):
    case = payload()
    diagnostics = {"preprocessing": {"one_hot_levels": {"private_category": ["value"]}}, "width_policy": "declared"}
    with patch(
        "benchmarks.reference_methods.generate_reference",
        return_value=(case["X"].copy(), case["X"].copy(), diagnostics),
    ) as generate:
        train, _ = tabular.generate_features(case["X"], case["X"], case["y"], method, "regression", config(), tmp_path)
    assert generate.call_args.kwargs["method"] == reference
    assert generate.call_args.kwargs["max_added_features"] == width
    assert "one_hot_levels" not in train.attrs["reference_diagnostics"]["preprocessing"]
    assert method not in tabular.DEFAULT_METHODS


@pytest.mark.parametrize(
    "method,reference",
    [
        ("openfe-inductive-native-robust", "openfe-inductive"),
        ("autofeat-native-robust", "autofeat-native"),
        ("autofeat-native-safe-scale-v1-robust", "autofeat-native-safe-scale-v1"),
    ],
)
def test_robust_references_preserve_generation_and_restore_original_missingness(method, reference, tmp_path):
    train = pd.DataFrame({"value": [1.0, np.nan, 3.0], "cat": ["a", None, "b"], "flag": [True, False, True]})
    test = pd.DataFrame({"value": [np.nan, 100.0], "cat": ["new", "a"], "flag": [False, True]})
    expected_train, expected_test = tabular.prepare_numeric(train, test)
    expected_originals, expected_heldout = tabular.prepare_numeric(train, test, impute=False, preserve_boolean=True)

    def generate(fit_X, test_X, target, **kwargs):
        pd.testing.assert_frame_equal(fit_X, expected_train)
        pd.testing.assert_frame_equal(test_X, expected_test)
        assert kwargs["method"] == reference
        assert kwargs["max_added_features"] is None
        return fit_X.assign(derived=fit_X.value * 2), test_X.assign(derived=test_X.value * 2), {}

    with patch("benchmarks.reference_methods.generate_reference", side_effect=generate):
        output_train, output_test = tabular.generate_features(
            train, test, pd.Series([1.0, 2.0, 3.0]), method, "regression", config(), tmp_path
        )
    pd.testing.assert_frame_equal(output_train[train.columns], expected_originals)
    pd.testing.assert_frame_equal(output_test[test.columns], expected_heldout)
    assert output_train.derived.tolist() == [2.0, 4.0, 6.0]
    assert output_test.derived.tolist() == [4.0, 200.0]
    assert output_train.attrs["reference_diagnostics"]["clipping_scope"] == "final_matched_learner_only"


def test_dependency_failure_is_not_a_baseline_fallback(tmp_path):
    case = payload()
    with patch.object(tabular, "make_openfe", side_effect=ImportError("not installed")):
        # Candidate import is lazy too, so this test does not require OpenFE.
        with patch.dict("sys.modules", {"openfe": type("OpenFEStub", (), {"get_candidate_features": lambda **kw: []})}):
            with pytest.raises(ImportError, match="not installed"):
                tabular.generate_features(
                    case["X"], case["X"], case["y"], "openfe-stateless", "regression", config(), tmp_path
                )


def test_openfe_compatibility_metric_and_pickling():
    pytest.importorskip("openfe")
    import pickle

    engineer = tabular.make_openfe()
    engineer.metric = "rmse"
    assert engineer.get_init_metric([1, 3], [1, 1]) == pytest.approx(np.sqrt(2))
    assert pickle.loads(pickle.dumps(engineer)).metric == "rmse"


def test_openfe_rejects_global_candidates_and_never_concatenates_test(tmp_path):
    class Feature:
        def __init__(self, name):
            self.name = name

        def calculate(self, frame):
            return frame["x0"] ** 2

    class Engine:
        def fit(self, data, label, **kwargs):
            assert len(data) == 40
            assert label.index.equals(data.index)
            assert [node.name for node in kwargs["candidate_features_list"]] == ["square"]
            return kwargs["candidate_features_list"]

    stub = type(
        "OpenFEStub",
        (),
        {"get_candidate_features": lambda **kw: [Feature("square"), Feature("freq"), Feature("GroupByThenMean")]},
    )
    case = payload()
    with patch.dict("sys.modules", {"openfe": stub}), patch.object(tabular, "make_openfe", return_value=Engine()):
        train, test = tabular.generate_features(
            case["X"].iloc[:40],
            case["X"].iloc[40:],
            case["y"].iloc[:40],
            "openfe-stateless",
            "regression",
            config(),
            tmp_path,
        )
    assert np.allclose(train["derived_0"], case["X"].iloc[:40].x0 ** 2)
    assert np.allclose(test["derived_0"], case["X"].iloc[40:].x0 ** 2)


def fold_row(dataset, method, fold, loss, source="real_world"):
    return {
        "dataset": dataset,
        "method": method,
        "fold": fold,
        "seed": 42,
        "split_sha256": f"split{fold}",
        "source": source,
        "task": "regression",
        "learner": "linear",
        "status": "success",
        "loss": loss,
        "pipeline_seconds": 2.0,
    }


def test_summary_uses_dataset_means_and_excludes_incomplete_pairs():
    rows = [
        fold_row(dataset, method, fold, loss)
        for dataset, loss in (("a", 10.0), ("b", 100.0))
        for fold in range(2)
        for method in ("baseline", "featcopilot")
    ]
    for row in rows:
        if row["method"] == "featcopilot":
            row["loss"] *= 0.8 if row["dataset"] == "a" else 1.1
    result = next(row for row in tabular.summarize(rows, 2, 1) if row["method"] == "featcopilot")
    assert result["mean_loss_reduction"] == pytest.approx(0.05)
    assert (result["wins"], result["ties"], result["losses"]) == (1, 0, 1)
    rows[-1]["status"] = "timeout"
    del rows[-1]["loss"]
    result = next(row for row in tabular.summarize(rows, 2, 1) if row["method"] == "featcopilot")
    assert result["complete_paired_datasets"] == 1
    assert result["requested_datasets"] == 2
    assert result["successful_folds"] == 3
    assert result["dataset_bootstrap_95_ci"] is None


def test_summary_never_pairs_different_splits_or_sources():
    rows = [fold_row("a", method, fold, 1.0) for method in ("baseline", "featcopilot") for fold in range(2)]
    rows[-1]["split_sha256"] = "different"
    rows += [
        fold_row("diagnostic", method, fold, 0.5, source="synthetic")
        for method in ("baseline", "featcopilot")
        for fold in range(2)
    ]
    results = tabular.summarize(rows, 2, 1)
    real = next(row for row in results if row["method"] == "featcopilot" and row["source"] == "real_world")
    synthetic = next(row for row in results if row["method"] == "featcopilot" and row["source"] == "synthetic")
    assert real["complete_paired_datasets"] == 0
    assert synthetic["complete_paired_datasets"] == 1


def test_robust_summary_separates_clipping_from_incremental_feature_gain(tmp_path):
    method = "featcopilot-frequency-aligned-robust"
    rows = [
        fold_row("a", name, fold, loss)
        for fold in range(2)
        for name, loss in [("baseline", 10.0), ("baseline-robust", 5.0), (method, 4.0)]
    ]
    for row in rows:
        row.update(inner_fe_seconds=0.1, outer_fe_seconds=0.1, n_output_features=3)
    manifest = {"config": {"folds": 2, "seeds": [42]}}
    tabular.write_report(tmp_path, manifest, rows)
    result = json.loads((tmp_path / "results.json").read_text())
    original = next(item for item in result["summary"] if item["method"] == method)
    matched = next(item for item in result["matched_robust_summary"] if item["method"] == method)
    assert original["reference_method"] == "baseline"
    assert original["mean_loss_reduction"] == pytest.approx(0.6)
    assert matched["reference_method"] == "baseline-robust"
    assert matched["mean_loss_reduction"] == pytest.approx(0.2)
    assert "Clipping gains alone are not counted" in (tmp_path / "REPORT.md").read_text()
    failed = next(row for row in rows if row["method"] == "baseline-robust" and row["fold"] == 1)
    failed.update(status="error", error="clip failed")
    del failed["loss"]
    matched = next(
        item for item in tabular.summarize(rows, 2, 1, baseline_method="baseline-robust") if item["method"] == method
    )
    assert matched["complete_paired_families"] == 0
    assert matched["mean_loss_reduction"] is None


def test_report_uses_declared_robust_reference_but_never_falls_back_after_failure(tmp_path):
    method = "featcopilot-bagged-aligned-robust"
    rows = [
        fold_row("a", name, fold, loss) for fold in range(2) for name, loss in [("baseline-robust", 5.0), (method, 4.0)]
    ]
    for row in rows:
        row.update(inner_fe_seconds=0.1, outer_fe_seconds=0.1, n_output_features=3)
    manifest = {"config": {"folds": 2, "seeds": [42], "methods": ["baseline-robust", method]}}
    tabular.write_report(tmp_path, manifest, rows)
    result = json.loads((tmp_path / "results.json").read_text())
    summary = next(item for item in result["summary"] if item["method"] == method)
    assert summary["reference_method"] == "baseline-robust"
    assert summary["mean_loss_reduction"] == pytest.approx(0.2)
    assert "Reference method: `baseline-robust`" in (tmp_path / "REPORT.md").read_text()

    manifest["config"]["methods"].append("baseline")
    failures = [
        {**fold_row("a", "baseline", fold, 10.0), "status": "error", "error": "raw control failed"} for fold in range(2)
    ]
    for failure in failures:
        del failure["loss"]
    tabular.write_report(tmp_path, manifest, rows + failures)
    summary = next(
        item for item in json.loads((tmp_path / "results.json").read_text())["summary"] if item["method"] == method
    )
    assert summary["reference_method"] == "baseline"
    assert summary["complete_paired_families"] == 0
    assert summary["mean_loss_reduction"] is None


def test_compact_resampling_diagnostics_retain_counts_and_masks_not_full_rank_arrays():
    report = {
        "importance_resampling": {
            "completed_bags": 5,
            "eligible_count": 1,
            "never_positive_count": 1,
            "score_kind": "negative_mean_candidate_rank",
            "positive_counts": {"chosen": 5, "unused": 0},
            "mean_ranks": {"chosen": 1.0, "unused": 2.0},
            "never_positive_features": ["unused"],
            "bags": [{"position_mask_sha256": "mask", "learner_fits": 1}],
        },
        "selected_features": ["chosen"],
    }
    compact = tabular.selection_diagnostics(report, False)
    assert compact["selected_features"] == ["chosen"]
    details = compact["importance_resampling"]
    assert details["eligible_count"] == details["never_positive_count"] == 1
    assert details["completed_bags"] == 5
    assert details["bags"][0]["position_mask_sha256"] == "mask"
    assert not {"positive_counts", "mean_ranks", "never_positive_features"}.intersection(details)
    assert tabular.selection_diagnostics(report, True) is report
    assert report["importance_resampling"]["mean_ranks"]["chosen"] == 1.0


def test_fixed_plan_summary_preserves_discovery_matched_control(tmp_path):
    method = "featcopilot-plan-gated-robust"
    rows = [
        fold_row("a", name, fold, loss)
        for fold in range(2)
        for name, loss in [
            ("baseline-robust", 4.0),
            ("baseline-discovery-robust", 5.0),
            (method, 4.5),
        ]
    ]
    for row in rows:
        row.update(inner_fe_seconds=0.1, outer_fe_seconds=0.1, n_output_features=3)
    manifest = {"config": {"folds": 2, "seeds": [42], "methods": [row["method"] for row in rows[:3]]}}
    tabular.write_report(tmp_path, manifest, rows)
    result = json.loads((tmp_path / "results.json").read_text())
    ordinary = next(item for item in result["summary"] if item["method"] == method)
    matched = next(item for item in result["matched_plan_summary"] if item["method"] == method)
    assert ordinary["reference_method"] == "baseline-robust"
    assert ordinary["mean_loss_reduction"] == pytest.approx(-0.125)
    assert matched["reference_method"] == "baseline-discovery-robust"
    assert matched["mean_loss_reduction"] == pytest.approx(0.1)
    assert all(item["method"] not in tabular.PLAN_VALIDATION_METHODS for item in result["matched_robust_summary"])


@pytest.mark.parametrize("method", sorted(tabular.TARGET_ENCODING_METHODS))
def test_target_methods_are_explicit_classification_iid_ablation(method, tmp_path):
    case = payload()
    with pytest.raises(ValueError, match="classification only"):
        tabular.generate_features(case["X"], case["X"], case["y"], method, "regression", config(), tmp_path)
    with pytest.raises(ValueError, match="IID only"):
        tabular.generate_features(
            case["X"],
            case["X"],
            case["y"],
            method,
            "classification",
            {**config(), "groups": np.arange(len(case["X"]))},
            tmp_path,
        )
    assert method not in tabular.DEFAULT_METHODS


def test_target_feature_method_preserves_aligned_frequency_robust_identity(tmp_path):
    case = payload()
    method = "featcopilot-frequency-aligned-target-robust"
    with patch("featcopilot.AutoFeatureEngineer") as constructor:
        constructor.return_value.fit_transform.return_value = case["X"]
        constructor.return_value.transform.return_value = case["X"]
        constructor.return_value.selection_report_ = None
        tabular.generate_features(
            case["X"],
            case["X"],
            pd.Series(np.arange(len(case["X"])) % 2),
            method,
            "classification",
            {**config(), "selection_learner": "linear", "selection_model_parameters": {"regularization": 0.1}},
            tmp_path,
        )
    options = constructor.call_args.kwargs
    assert options["selection_estimator"].C == 0.1
    assert options["screening"] == "importance"
    assert options["selection_rule"] == "paired_gain"
    assert options["candidate_set"] == "extended"
    assert options["include_frequency"] is True
    assert options["numeric_clip_quantiles"] == (0.0, 1.0)
    assert options["categorical_encoding"] == "target"
    assert method in tabular.ALIGNED_METHODS & tabular.ROBUST_METHODS & tabular.PREDICTIVE_METHODS


def test_target_summary_does_not_credit_encoding_only_gains_as_feature_gains(tmp_path):
    rows = [
        fold_row("a", name, fold, loss)
        for fold in range(2)
        for name, loss in [
            ("baseline-robust", 10.0),
            ("baseline-target-robust", 5.0),
            ("featcopilot-frequency-aligned-target-robust", 4.0),
        ]
    ]
    for row in rows:
        row.update(inner_fe_seconds=0.1, outer_fe_seconds=0.1, n_output_features=3)
    manifest = {"config": {"folds": 2, "seeds": [42], "methods": [row["method"] for row in rows[:3]]}}
    tabular.write_report(tmp_path, manifest, rows)
    result = json.loads((tmp_path / "results.json").read_text())
    method = "featcopilot-frequency-aligned-target-robust"
    original = next(item for item in result["summary"] if item["method"] == method)
    matched = next(item for item in result["matched_target_summary"] if item["method"] == method)
    assert original["mean_loss_reduction"] == pytest.approx(0.6)
    assert matched["reference_method"] == "baseline-target-robust"
    assert matched["mean_loss_reduction"] == pytest.approx(0.2)


def test_timeout_has_no_score_and_stops_worker(tmp_path):
    rows = tabular.run_isolated(payload(), tmp_path / "case", timeout=0.01)
    assert rows[0]["status"] == "timeout"
    assert "loss" not in rows[0]
    assert rows[0]["worker_python_hash_seed"] == 42
    assert not (tmp_path / "case" / "input.pkl").exists()


@pytest.fixture
def fake_supervisor(monkeypatch):
    import psutil

    process = Mock(pid=424242, returncode=None)
    process.poll.side_effect = lambda: process.returncode
    process.kill.side_effect = lambda: setattr(process, "returncode", -9)
    process.wait.side_effect = lambda timeout: process.returncode
    parent = Mock(pid=process.pid)
    parent.children.return_value = []
    parent.memory_info.return_value = SimpleNamespace(rss=0)
    monkeypatch.setattr(tabular.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(psutil, "Process", Mock(return_value=parent))
    monkeypatch.setattr(psutil, "wait_procs", Mock(return_value=([], [])))
    return process, parent


@pytest.mark.parametrize("site", ["process", "children", "memory", "wait"])
@pytest.mark.parametrize("denied", [False, True])
def test_supervisor_monitor_errors_always_stop_worker_and_remove_input(
    tmp_path, monkeypatch, fake_supervisor, caplog, site, denied
):
    import psutil

    process, parent = fake_supervisor
    error = psutil.AccessDenied(process.pid) if denied else RuntimeError("Synthetic monitor failure")
    if site == "process":
        psutil.Process.side_effect = error
    elif site == "children":
        parent.children.side_effect = error
    elif site == "memory":
        parent.memory_info.side_effect = error
    else:
        process.wait.side_effect = [error, -9]
    rows = tabular.run_isolated(payload(), tmp_path / "case", 30)
    assert rows[0]["status"] == "error" and "loss" not in rows[0]
    assert type(error).__name__ in rows[0]["error"]
    assert type(error).__name__ in caplog.text
    process.kill.assert_called_once()
    assert process.returncode == -9
    assert not (tmp_path / "case" / "input.pkl").exists()
    assert all(call.kwargs["timeout"] <= 5 for call in process.wait.call_args_list)


def test_supervisor_interrupt_cleans_up_without_converting_cancellation(tmp_path, fake_supervisor):
    process, parent = fake_supervisor
    process.wait.side_effect = [KeyboardInterrupt, -9]
    with pytest.raises(KeyboardInterrupt):
        tabular.run_isolated(payload(), tmp_path / "case", 30)
    process.kill.assert_called_once()
    assert not (tmp_path / "case" / "input.pkl").exists()


def test_tree_cleanup_survives_inaccessible_children_and_has_bounded_waits(fake_supervisor, monkeypatch):
    import psutil

    process, parent = fake_supervisor
    inaccessible, accessible = Mock(pid=424243), Mock(pid=424244)
    inaccessible.kill.side_effect = psutil.AccessDenied(inaccessible.pid)
    parent.children.return_value = [inaccessible, accessible]
    process.wait.side_effect = tabular.subprocess.TimeoutExpired("mock worker", 5)
    psutil.wait_procs.return_value = ([accessible], [inaccessible])
    tabular.stop_process_tree(process)
    process.kill.assert_called()
    accessible.kill.assert_called_once()
    inaccessible.kill.assert_called()
    assert 0 <= process.wait.call_args.kwargs["timeout"] <= 5
    assert 0 <= psutil.wait_procs.call_args.kwargs["timeout"] <= 5


def test_cleanup_retains_observed_descendants_when_parent_has_disappeared(fake_supervisor):
    import psutil

    process, parent = fake_supervisor
    process.returncode = 0
    child = Mock(pid=424243)
    psutil.Process.side_effect = psutil.NoSuchProcess(process.pid)
    tabular.stop_process_tree(process, descendants=[child])
    child.kill.assert_called_once()
    process.kill.assert_not_called()
    assert psutil.wait_procs.call_args.args[0] == [child]


@pytest.mark.parametrize("failure", ["decode", "monitor"])
def test_cleanup_failure_does_not_replace_original_worker_failure(tmp_path, monkeypatch, fake_supervisor, failure):
    import psutil

    from benchmarks.execution import BenchmarkProtocolError

    process, parent = fake_supervisor

    def launch(command, **kwargs):
        if failure == "decode":
            process.returncode = 0
            Path(command[-1]).write_text("{")
        else:
            parent.memory_info.side_effect = psutil.AccessDenied(process.pid)
        return process

    monkeypatch.setattr(tabular.subprocess, "Popen", launch)
    monkeypatch.setattr(tabular, "stop_process_tree", Mock(side_effect=RuntimeError("Synthetic cleanup failure")))
    if failure == "decode":
        with pytest.raises(BenchmarkProtocolError, match="not valid JSON"):
            tabular.run_isolated(payload(), tmp_path / "case", 30)
    else:
        rows = tabular.run_isolated(payload(), tmp_path / "case", 30)
        assert "AccessDenied" in rows[0]["error"]
        assert "cleanup failure" not in rows[0]["error"]
    process.kill.assert_called_once()
    assert process.wait.call_args.kwargs["timeout"] == 5
    assert not (tmp_path / "case" / "input.pkl").exists()


@pytest.mark.parametrize("document", [b"{", b"null", b"[42]", b"[]", b"[null]", b'{"status":"success"}', b"\xff"])
def test_actual_worker_file_contract_is_validated_before_decoration(tmp_path, monkeypatch, document):
    import psutil

    from benchmarks.execution import BenchmarkProtocolError

    process = Mock(pid=424242, returncode=0)
    process.poll.return_value = 0

    def launch(command, **kwargs):
        Path(command[-1]).write_bytes(document)
        return process

    monkeypatch.setattr(tabular.subprocess, "Popen", launch)
    monkeypatch.setattr(psutil, "Process", Mock(side_effect=psutil.NoSuchProcess(process.pid)))
    with pytest.raises(BenchmarkProtocolError):
        tabular.run_isolated(payload(), tmp_path / "case", 30)
    assert not (tmp_path / "case" / "input.pkl").exists()


def test_worker_launch_failure_is_retryable_and_removes_serialized_input(tmp_path, monkeypatch):
    monkeypatch.setattr(tabular.subprocess, "Popen", Mock(side_effect=OSError("Synthetic launch failure")))
    rows = tabular.run_isolated(payload(), tmp_path / "case", 30)
    assert rows[0]["status"] == "error" and "OSError" in rows[0]["error"]
    assert not (tmp_path / "case" / "input.pkl").exists()


@pytest.mark.parametrize(
    "worker_module",
    [
        "benchmarks.tabular",
        "benchmarks.calibration_evaluation",
        "benchmarks.blending_evaluation",
        "benchmarks.regression_training_audit",
    ],
)
@pytest.mark.parametrize("seed", [0, 2031, 2**32 - 1])
def test_worker_hash_seed_is_effective_at_startup_and_ignores_parent_environment(
    tmp_path, monkeypatch, worker_module, seed
):
    real_popen = tabular.subprocess.Popen
    probe = (
        "import json, os, sys; from pathlib import Path; "
        "Path(sys.argv[1]).write_text(json.dumps([{'status': 'success', "
        "'observed_seed': os.environ['PYTHONHASHSEED'], "
        "'observed_hash': hash('autofeat-feature-order'), "
        "'observed_order': list({'feature-' + str(i) for i in range(30)})}]))"
    )

    def launch_probe(command, **kwargs):
        assert command[2] == worker_module
        return real_popen([command[0], "-c", probe, command[-1]], **kwargs)

    monkeypatch.setattr(tabular.subprocess, "Popen", launch_probe)
    case = payload()
    case["seed"] = seed
    observations = []
    for index, inherited in enumerate(["random", "123"]):
        monkeypatch.setenv("PYTHONHASHSEED", inherited)
        rows = tabular.run_isolated(case, tmp_path / f"probe-{index}", 30, worker_module=worker_module)
        assert rows[0]["status"] == "success"
        assert rows[0]["observed_seed"] == str(seed)
        assert rows[0]["worker_python_hash_seed"] == seed
        assert os.environ["PYTHONHASHSEED"] == inherited
        observations.append((rows[0]["observed_hash"], rows[0]["observed_order"]))
    assert observations[0] == observations[1]


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5, "42", None])
def test_invalid_worker_hash_seed_is_rejected_before_creating_files(tmp_path, seed):
    case = payload()
    case["seed"] = seed
    with pytest.raises(ValueError, match="Worker seed must be an integer between 0 and 4294967295"):
        tabular.run_isolated(case, tmp_path / "case", timeout=1)
    assert not (tmp_path / "case").exists()


def test_worker_crash_records_effective_hash_seed_without_inventing_a_score(tmp_path, fake_supervisor):
    process, parent = fake_supervisor
    process.returncode = 1
    rows = tabular.run_isolated(payload(), tmp_path / "case", timeout=1)
    assert rows[0]["status"] == "error" and "loss" not in rows[0]
    assert rows[0]["worker_python_hash_seed"] == 42


def test_unknown_worker_module_is_rejected_before_creating_files(tmp_path):
    with pytest.raises(ValueError, match="Unsupported benchmark worker module"):
        tabular.run_isolated(payload(), tmp_path / "case", timeout=1, worker_module="unregistered.worker")
    assert not (tmp_path / "case").exists()


def test_manifest_has_unique_real_dataset_origins_and_explicit_targets():
    suite = json.loads(tabular.SUITE_PATH.read_text())
    assert len(suite["revision"]) == 40
    assert len({spec["name"] for spec in suite["datasets"]}) == len(suite["datasets"])
    assert len({spec["openml_id"] for spec in suite["datasets"]}) == len(suite["datasets"])
    assert all(spec["target"] for spec in suite["datasets"])


def test_dataset_cache_provenance_and_sampling(tmp_path):
    spec = {"name": "test", "path": "reg_num/test.csv", "target": "target", "task": "regression", "categorical": []}
    path = tmp_path / "revision" / "reg_num" / "test.csv"
    path.parent.mkdir(parents=True)
    data = pd.DataFrame({"x": range(10), "target": range(10)})
    data.to_csv(path, index=False)
    X, y, metadata = tabular.load_real_dataset(spec, "revision", tmp_path, 5)
    assert len(X) == len(y) == 5
    assert metadata["sha256"] == tabular.digest(path.read_bytes())
    assert metadata["original_rows"] == 10
    assert metadata["source"] == "real_world"
    pd.testing.assert_series_equal(X.x0, y, check_names=False)


def test_bad_dataset_target_does_not_become_synthetic(tmp_path):
    spec = {"name": "test", "path": "test.csv", "target": "missing", "task": "regression", "categorical": []}
    path = tmp_path / "revision" / "test.csv"
    path.parent.mkdir(parents=True)
    path.write_text("x,target\n1,2\n")
    with pytest.raises(ValueError, match="Missing declared target"):
        tabular.load_real_dataset(spec, "revision", tmp_path, 0)


def test_json_rejects_nonfinite_scores(tmp_path):
    with pytest.raises(ValueError):
        tabular.write_json(Path(tmp_path / "results.json"), {"loss": float("nan")})


def test_live_llm_methods_are_opt_in_not_default():
    assert "llm-plan" in tabular.METHODS
    assert "llm-plan" not in tabular.DEFAULT_METHODS
    assert "llm-hybrid" not in tabular.DEFAULT_METHODS


def test_llm_generation_requires_explicit_approval_and_shared_ledger(tmp_path):
    case = payload()
    with patch("benchmarks.llm_evaluation.CopilotPlanProvider") as provider:
        with pytest.raises(ValueError, match="explicit approval"):
            tabular.generate_features(case["X"], case["X"], case["y"], "llm-plan", "regression", config(), tmp_path)
    provider.assert_not_called()


def test_llm_adapter_forwards_semantic_context_and_exports_definitions_not_learned_state(tmp_path):
    from featcopilot.engines import FittedRuleEngine

    @dataclass
    class Diagnostics:
        calls: int = 1
        token_cap_enforced: bool = False

    case = payload()
    train, test = case["X"].iloc[:40], case["X"].iloc[40:]

    def evaluate(train_X, y, test_X, **kwargs):
        assert kwargs["live_approved"] is True
        assert kwargs["column_descriptions"] == {"x0": "Synthetic measurement"}
        assert set(train_X.index).isdisjoint(test_X.index)
        engine = FittedRuleEngine([{"name": "scaled", "operator": "standardize", "columns": ["x0"]}])
        result = engine.fit_transform(train_X, y)
        return SimpleNamespace(
            train_X=result, test_X=engine.transform(test_X), diagnostics=Diagnostics(), plan_engine=engine
        )

    settings = {
        **config(),
        "live_llm": True,
        "llm_ledger": str(tmp_path / "budget"),
        "allow_soft_llm_tokens": True,
        "column_descriptions": {"x0": "Synthetic measurement"},
    }
    with (
        patch("benchmarks.llm_evaluation.evaluate_plan_fit", side_effect=evaluate),
        patch("benchmarks.llm_evaluation.CopilotPlanProvider"),
    ):
        transformed, _ = tabular.generate_features(
            train, test, case["y"].iloc[:40], "llm-plan", "regression", settings, tmp_path
        )
    assert transformed.attrs["llm_diagnostics"]["calls"] == 1
    assert set(transformed.attrs["llm_plan"][0]) == {"name", "operator", "columns", "parameters", "explanation"}
