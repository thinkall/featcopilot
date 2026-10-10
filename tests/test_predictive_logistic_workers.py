"""Output-preserving worker limits for single-task LogisticRegression clones."""

import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.multiclass import OneVsRestClassifier
from threadpoolctl import threadpool_limits

from featcopilot.engines.predictive import PredictiveTabularEngine

HAS_MULTI_CLASS = "multi_class" in LogisticRegression().get_params()


def engine(estimator, **kwargs):
    options = {"estimator": estimator, "task": "classification", "n_jobs": 8, "max_candidates": 0}
    options.update(kwargs)
    return PredictiveTabularEngine(**options)


@pytest.mark.parametrize("solver", ["lbfgs", "newton-cg"])
@pytest.mark.parametrize("requested_jobs", [None, 1, 3, -1])
def test_single_task_clone_changes_only_the_task_worker_parameter(solver, requested_jobs):
    supplied = LogisticRegression(
        solver=solver,
        C=0.37,
        tol=2e-5,
        max_iter=321,
        fit_intercept=False,
        class_weight="balanced",
        random_state=73,
        warm_start=True,
        n_jobs=requested_jobs,
    )
    original = supplied.get_params().copy()
    before = pickle.dumps(supplied)
    selected = engine(supplied)._learner("classification")
    assert selected is not supplied
    assert selected.get_params() == {**original, "n_jobs": 1}
    assert pickle.dumps(supplied) == before


def test_unspecified_sequential_workers_stay_unspecified():
    supplied = LogisticRegression(n_jobs=None, random_state=42)
    selected = engine(supplied, n_jobs=1)._learner("classification")
    assert selected.get_params() == supplied.get_params()
    assert selected.n_jobs is None
    assert supplied.n_jobs is None


@pytest.mark.skipif(not HAS_MULTI_CLASS, reason="Explicit multiclass modes were removed by sklearn")
@pytest.mark.parametrize("mode", ["auto", "deprecated", "multinomial"])
def test_supported_configured_joint_modes_are_single_task(mode):
    supplied = LogisticRegression(solver="lbfgs", multi_class=mode, n_jobs=6, random_state=42)
    selected = engine(supplied)._learner("classification")
    assert selected.get_params() == {**supplied.get_params(), "n_jobs": 1}


@pytest.mark.skipif(not HAS_MULTI_CLASS, reason="Explicit OVR is not configured on current LogisticRegression")
@pytest.mark.parametrize("solver", ["lbfgs", "newton-cg", "newton-cholesky"])
def test_explicit_one_vs_rest_keeps_existing_parallel_worker_policy(solver):
    supplied = LogisticRegression(solver=solver, multi_class="ovr", n_jobs=3, random_state=42)
    selected = engine(supplied)._learner("classification")
    assert selected.get_params() == {**supplied.get_params(), "n_jobs": 8}
    assert supplied.n_jobs == 3


@pytest.mark.parametrize("solver", ["liblinear", "sag", "saga", "newton-cholesky"])
def test_other_solvers_keep_existing_worker_policy(solver):
    supplied = LogisticRegression(solver=solver, n_jobs=3, random_state=42)
    selected = engine(supplied)._learner("classification")
    assert selected.get_params() == {**supplied.get_params(), "n_jobs": 8}
    assert supplied.n_jobs == 3


def test_missing_deprecated_multi_class_parameter_is_supported_without_version_checks():
    original_get_params = LogisticRegression.get_params

    def without_multi_class(estimator, deep=True):
        return {key: value for key, value in original_get_params(estimator, deep=deep).items() if key != "multi_class"}

    supplied = LogisticRegression(n_jobs=3, random_state=42)
    with patch.object(LogisticRegression, "get_params", without_multi_class):
        selected = engine(supplied)._learner("classification")
        assert "multi_class" not in selected.get_params()
        assert selected.get_params() == {**supplied.get_params(), "n_jobs": 1}


@pytest.mark.parametrize("mode", ["ovr", "unrecognized-future-mode", None])
def test_unknown_or_nonjoint_modes_are_not_assumed_single_task(mode):
    original_get_params = LogisticRegression.get_params

    def configured_mode(estimator, deep=True):
        return {**original_get_params(estimator, deep=deep), "multi_class": mode}

    supplied = LogisticRegression(n_jobs=3, random_state=42)
    # Inject only at _learner's parameter inspection, not through sklearn's
    # constructor validation, to exercise forward-compatible dispatch guards.
    configured = engine(supplied)
    selected_clone = clone(supplied)
    with (
        patch("featcopilot.engines.predictive._clone_estimator", return_value=selected_clone),
        patch.object(selected_clone, "get_params", side_effect=lambda deep=True: configured_mode(selected_clone, deep)),
    ):
        selected = configured._learner("classification")
    assert selected.n_jobs == 8
    assert supplied.n_jobs == 3


def test_cv_subclasses_and_tree_estimators_keep_existing_worker_policy():
    class CustomLogisticRegression(LogisticRegression):
        pass

    for supplied in [
        LogisticRegressionCV(cv=2, n_jobs=3, random_state=42),
        CustomLogisticRegression(n_jobs=3, random_state=42),
        RandomForestClassifier(n_estimators=2, n_jobs=3, random_state=42),
    ]:
        selected = engine(supplied)._learner("classification")
        assert selected.get_params() == {**supplied.get_params(), "n_jobs": 8}
        assert supplied.n_jobs == 3


def test_explicit_ovr_wrapper_and_its_nested_estimator_keep_existing_policy():
    supplied = OneVsRestClassifier(LogisticRegression(n_jobs=3, random_state=42), n_jobs=3)
    before = pickle.dumps(supplied)
    selected = engine(supplied)._learner("classification")
    assert selected.n_jobs == 8
    assert selected.estimator.n_jobs == 8
    assert selected.estimator.get_params() == {**supplied.estimator.get_params(), "n_jobs": 8}
    assert pickle.dumps(supplied) == before


@pytest.mark.parametrize("classes", [2, 3])
def test_binary_multiclass_fit_preserves_output_and_native_thread_cap(classes):
    rng = np.random.default_rng(991)
    X = pd.DataFrame(rng.normal(size=(120, 2)), columns=["a", "b"])
    latent = X["a"].to_numpy() + X["b"].to_numpy()
    y = (latent > 0).astype(int) if classes == 2 else np.digitize(latent, [-0.5, 0.5])
    supplied = LogisticRegression(C=0.7, max_iter=250, tol=1e-4, random_state=19, n_jobs=None)
    before = pickle.dumps(supplied)
    # Avoid launching process pools in unit tests; exact pinned-env old/new
    # process behavior is measured separately in the synthetic probe.
    with (
        threadpool_limits(limits=1),
        patch("featcopilot.engines.predictive.threadpool_limits", wraps=threadpool_limits) as native_limit,
    ):
        fitted = engine(supplied, stability_repeats=1)
        result = fitted.fit_transform(X, y)
        native_limit.assert_called_once_with(limits=8)
        selection_clone = fitted._learner("classification")
        assert selection_clone.get_params() == {**supplied.get_params(), "n_jobs": 1}
        actual = fitted._fit_learner(fitted._model_frame(result), y, selection_clone)
        reference = fitted._fit_learner(
            fitted._model_frame(X), y, LogisticRegression(C=0.7, max_iter=250, tol=1e-4, random_state=19, n_jobs=1)
        )
        np.testing.assert_array_equal(
            actual.predict_proba(fitted._model_frame(X)), reference.predict_proba(fitted._model_frame(X))
        )
    assert_frame_equal(result, X)
    assert fitted.selection_report_["selected_count"] == 0
    assert fitted.selection_report_["learner_fit_count"] == 1
    assert fitted.config.n_jobs == 8
    assert pickle.dumps(supplied) == before


def test_previously_fitted_caller_is_not_mutated():
    X = np.array([[-2], [-1], [1], [2]], dtype=float)
    supplied = LogisticRegression(random_state=42).fit(X, [0, 0, 1, 1])
    before = pickle.dumps(supplied)
    selected = engine(supplied)._learner("classification")
    assert not hasattr(selected, "coef_")
    assert selected.n_jobs == 1
    assert pickle.dumps(supplied) == before
