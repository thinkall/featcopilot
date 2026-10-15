"""Model-input target-encoding boundaries across selection and final learners."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from benchmarks import tabular
from featcopilot import AutoFeatureEngineer
from featcopilot.core.feature import Feature
from featcopilot.engines.predictive import PredictiveTabularEngine


class MatrixClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.training_matrix_ = np.asarray(X).copy()
        self.classes_, counts = np.unique(y, return_counts=True)
        self.priors_ = counts / counts.sum()
        self.feature_importances_ = np.arange(1, X.shape[1] + 1, dtype=float)
        return self

    def predict_proba(self, X):
        return np.tile(self.priors_, (len(X), 1))

    def predict(self, X):
        return np.full(len(X), self.classes_[self.priors_.argmax()])


def classification_data(classes=2):
    rows = 120
    X = pd.DataFrame({"numeric": np.arange(rows, dtype=float), "category": [f"id-{i}" for i in range(rows)]})
    y = np.arange(rows) % classes
    return X, y


@pytest.mark.parametrize("classes", [2, 3])
def test_internal_learner_trains_on_cross_fitted_not_full_mapping_matrix(classes):
    X, y = classification_data(classes)
    engine = PredictiveTabularEngine(
        task="classification",
        estimator=MatrixClassifier(),
        screening="importance",
        categorical_encoding="target",
    )
    prepared = engine._model_frame(X)
    pd.testing.assert_series_equal(prepared.category, X.category)
    model = engine._fit_selection_learner(prepared, y, MatrixClassifier())
    learner = model.named_steps["learner"]
    preprocessing = model.named_steps["preprocess"]
    span = preprocessing.output_indices_["categorical"]
    expected_width = 1 if classes == 2 else classes
    assert span.stop - span.start == expected_width
    np.testing.assert_allclose(learner.training_matrix_[:, span], np.full((len(X), expected_width), 1.0 / classes))
    full_mapping = preprocessing.transform(prepared)
    assert not np.allclose(full_mapping[:, span], learner.training_matrix_[:, span])
    assert not hasattr(engine.config.estimator, "training_matrix_")


@pytest.mark.parametrize("classes", [2, 3])
def test_final_learner_uses_cross_fitting_and_reports_actual_encoded_width(classes):
    X, y = classification_data(classes)
    model = tabular.make_model(
        X,
        "classification",
        "linear",
        {"regularization": 1.0},
        {"seed": 42, "threads": 1, "categorical_encoding": "target", "numeric_clip_quantiles": (0.0, 1.0)},
    )
    model.steps[-1] = ("recording", MatrixClassifier())
    model.fit(X, y)
    model_input = model.named_steps["recording"].training_matrix_
    expected_width = 1 if classes == 2 else classes
    assert model_input.shape == (len(X), 1 + expected_width)
    np.testing.assert_allclose(model_input[:, 1:], 1.0 / classes)
    report = tabular.target_preprocessing_diagnostics(model)
    assert report["categorical_encoded_width"] == expected_width
    assert report["total_encoded_width"] == 1 + expected_width
    assert "id-" not in str(report)


@pytest.mark.parametrize("classes", [2, 3])
def test_target_encoded_categorical_width_does_not_shift_numeric_candidate_attribution(classes):
    X, y = classification_data(classes)
    X["derived"] = np.sin(X.numeric)
    model = PredictiveTabularEngine._fit_learner(
        X, y, MatrixClassifier(), categorical_encoding="target", random_state=42
    )
    weights, _ = PredictiveTabularEngine._importance_values(model)
    positions = PredictiveTabularEngine._candidate_output_indices(model, [Feature(name="derived")], len(weights))
    assert positions == {"derived": 1}
    assert len(weights) == 2 + (1 if classes == 2 else classes)


@pytest.mark.parametrize("metadata", ["groups", "timestamps", "label_end_times"])
def test_target_encoding_rejects_non_iid_metadata_before_any_fit(metadata):
    X, y = classification_data()
    engine = PredictiveTabularEngine(
        task="classification", estimator=MatrixClassifier(), screening="importance", categorical_encoding="target"
    )
    with patch.object(engine, "_fit_learner") as fit:
        with pytest.raises(ValueError, match="IID only"):
            engine.fit(X, y, **{metadata: np.arange(len(X))})
    fit.assert_not_called()


def test_target_encoding_requires_classification_and_importance_without_changing_default():
    assert PredictiveTabularEngine().config.categorical_encoding == "one_hot"
    assert AutoFeatureEngineer().categorical_encoding == "one_hot"
    with pytest.raises(ValueError, match="importance"):
        PredictiveTabularEngine(categorical_encoding="target")
    with pytest.raises(ValueError, match="classification only"):
        PredictiveTabularEngine(categorical_encoding="target", screening="importance", task="regression")
    X, _ = classification_data()
    engine = PredictiveTabularEngine(
        categorical_encoding="target", screening="importance", task="auto", estimator=MatrixClassifier()
    )
    with pytest.raises(ValueError, match="classification only"):
        engine.fit(X, np.linspace(0, 1, len(X)))
    with pytest.raises(ValueError, match="predictive importance"):
        AutoFeatureEngineer(categorical_encoding="target")


def test_wrapper_clones_forwards_and_preserves_raw_schema_and_candidate_definitions():
    X, y = classification_data()
    X.category = np.where(X.numeric % 3 == 0, "rare", "common")
    wrapper = clone(
        AutoFeatureEngineer(
            selection_strategy="predictive",
            task="classification",
            selection_estimator=MatrixClassifier(),
            screening="importance",
            categorical_encoding="target",
            max_candidates=20,
            max_features=5,
            stability_repeats=1,
        )
    )
    result = wrapper.fit_transform(X, y)
    assert wrapper.get_params()["categorical_encoding"] == "target"
    assert wrapper._engine_instances["tabular"].config.categorical_encoding == "target"
    pd.testing.assert_frame_equal(result[X.columns], X)
    pd.testing.assert_frame_equal(result, wrapper.transform(X))
    assert wrapper.selection_report_["categorical_preprocessing"]["active"]
    with pytest.raises(ValueError, match="predictive importance"):
        wrapper.set_params(screening="auto")
    assert wrapper.screening == "importance"
    options = {
        "task": "classification",
        "estimator": MatrixClassifier(),
        "screening": "importance",
        "max_candidates": 20,
    }
    normal = PredictiveTabularEngine(**options)
    target = PredictiveTabularEngine(**options, categorical_encoding="target")
    first, _ = normal._plan_candidates(X, 3)
    second, _ = target._plan_candidates(X, 3)
    assert [feature.to_dict() for feature in first] == [feature.to_dict() for feature in second]


def test_numeric_only_target_mode_never_constructs_an_encoder():
    X, y = classification_data()
    X = X[["numeric"]]
    with patch(
        "featcopilot.utils.model_preprocessing.CrossFittedTargetEncoder",
        side_effect=AssertionError("Numeric-only inputs need no target encoder"),
    ):
        model = tabular.make_model(
            X, "classification", "linear", {"regularization": 1.0}, {"seed": 42, "categorical_encoding": "target"}
        )
        model.fit(X, y)
        assert tabular.target_preprocessing_diagnostics(model)["active"] is False
        engine = PredictiveTabularEngine(
            task="classification",
            estimator=MatrixClassifier(),
            screening="importance",
            categorical_encoding="target",
            max_candidates=3,
            max_features=1,
            stability_repeats=1,
        ).fit(X, y)
        assert engine.selection_report_["categorical_preprocessing"]["active"] is False


def test_target_encoder_support_errors_are_not_silently_downgraded():
    X = pd.DataFrame({"category": ["a", "b"] * 4})
    with pytest.raises(ValueError):
        PredictiveTabularEngine._fit_learner(X, np.array([0, 1] * 4), MatrixClassifier(), categorical_encoding="target")
    with pytest.raises(ValueError, match="native CatBoost"):
        tabular.make_model(
            X, "classification", "catboost", {"depth": 3}, {"seed": 42, "categorical_encoding": "target"}
        )


@pytest.mark.parametrize("value,unseen", [(1, "1"), (True, "True")])
def test_benchmark_target_preparation_preserves_category_types(value, unseen, tmp_path):
    train = pd.DataFrame({"category": pd.Series([value] * 20 + [2] * 40, dtype=object)})
    test = pd.DataFrame({"category": pd.Series([value, unseen], dtype=object)}, index=[0, 1])
    y = pd.Series([1] * 20 + [0] * 40)
    prepared, heldout = tabular.generate_features(
        train,
        test,
        y,
        "baseline-target-robust",
        "classification",
        {"max_added_features": 3},
        tmp_path,
    )
    pd.testing.assert_frame_equal(prepared, train)
    pd.testing.assert_frame_equal(heldout, test)
    assert prepared.category.dtype == object
    model = tabular.make_model(
        prepared,
        "classification",
        "linear",
        {"regularization": 1.0},
        {"seed": 42, "categorical_encoding": "target"},
    )
    model.fit(prepared, y)
    encoder = model.named_steps["columntransformer"].named_transformers_["categorical"]
    values = encoder.transform(heldout)
    assert values[0, 0] > 0.9
    assert values[1, 0] == pytest.approx(y.mean())


def test_explicit_boolean_categories_remain_target_encoded_in_final_model():
    X = pd.DataFrame({"category": pd.Categorical([True, False] * 30)})
    y = pd.Series(np.arange(len(X)) % 2)
    model = tabular.make_model(
        X,
        "classification",
        "linear",
        {"regularization": 1.0},
        {"seed": 42, "categorical_encoding": "target", "numeric_clip_quantiles": (0.0, 1.0)},
    )
    model.fit(X, y)
    report = tabular.target_preprocessing_diagnostics(model)
    assert report["active"]
    assert report["categorical_encoded_width"] == report["total_encoded_width"] == 1
