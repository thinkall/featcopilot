"""Exact matched-preprocessing behavior on synthetic numeric/mixed frames."""

import json
import pickle
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import sklearn
import sklearn.preprocessing
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from benchmarks import tabular
from featcopilot.utils import model_preprocessing
from featcopilot.utils.model_preprocessing import (
    CrossFittedTargetEncoder,
    make_classification_pipeline,
    normalize_categorical_frame,
    normalize_model_frame,
    validate_classification_frame,
)


@pytest.mark.parametrize("mixed", [False, True])
def test_normalization_and_model_pipeline_match_existing_benchmark(mixed: bool) -> None:
    rng = np.random.default_rng(918)
    train = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})
    test = pd.DataFrame({"a": [1e300, -1e300, np.nan, np.inf], "b": [None, 0.0, 1.0, -np.inf]})
    train.loc[0, "a"] = np.nan
    if mixed:
        train["category"] = pd.Categorical(["a", "b", None] * 20)
        train["flag"] = [True, False] * 30
        test["category"] = ["unseen", None, "missing:", "a"]
        test["flag"] = [False, True, True, False]
    original = train.copy()
    expected_train, expected_test = tabular.prepare_numeric(train, test, impute=False, preserve_boolean=True)
    schema = train.iloc[:0].copy()
    actual_train = normalize_model_frame(schema, train)
    actual_test = normalize_model_frame(schema, test)
    pd.testing.assert_frame_equal(actual_train, expected_train, check_exact=True)
    pd.testing.assert_frame_equal(actual_test, expected_test, check_exact=True)
    pd.testing.assert_frame_equal(original, train, check_exact=True)
    estimator = LogisticRegression(C=0.1, max_iter=2000, random_state=42)
    expected = tabular.make_model(
        expected_train,
        "classification",
        "linear",
        {"regularization": 0.1},
        {"seed": 42, "threads": 1, "numeric_clip_quantiles": (0.0, 1.0)},
    ).fit(expected_train, np.arange(60) % 2)
    actual = make_classification_pipeline(actual_train, clone(estimator)).fit(actual_train, np.arange(60) % 2)
    np.testing.assert_array_equal(actual.predict_proba(actual_test), expected.predict_proba(expected_test))
    assert actual.steps[0][1].bounds_ == expected.steps[0][1].bounds_
    np.testing.assert_array_equal(actual[:-1].transform(actual_train), expected[:-1].transform(expected_train))
    assert not hasattr(estimator, "classes_")


def test_normalization_is_stateless_batch_invariant_and_never_imputes() -> None:
    train = pd.DataFrame({"number": [0.0, np.nan, 10.0], "category": ["a", None, "missing:"]})
    schema = train.iloc[:0]
    normalized = normalize_model_frame(schema, train)
    assert pd.isna(normalized.number.iloc[1])
    assert normalized.category.tolist() == ["value:a", "missing:", "value:missing:"]
    batched = pd.concat([normalize_model_frame(schema, train.iloc[:1]), normalize_model_frame(schema, train.iloc[1:])])
    pd.testing.assert_frame_equal(batched, normalized)
    assert schema.empty


@pytest.mark.parametrize("values", [[1, 2, None], [True, False, None]])
@pytest.mark.parametrize("dtype", ["object", "category"])
def test_target_normalization_preserves_raw_category_scalars_and_dtype(values: list, dtype: str) -> None:
    train = pd.DataFrame({"category": pd.Series(values, dtype=dtype), "number": [1.0, 2.0, np.nan]})
    original = train.copy(deep=True)
    normalized = normalize_model_frame(train.iloc[:0], train, preserve_categorical_types=True)
    pd.testing.assert_frame_equal(normalized, train, check_exact=True)
    pd.testing.assert_frame_equal(train, original, check_exact=True)
    assert str(normalized.category.dtype) == dtype
    test = pd.DataFrame({"category": [values[0], values[1]], "number": [4.0, 5.0]})
    normalized_test = normalize_model_frame(train.iloc[:0], test, preserve_categorical_types=True)
    assert normalized_test.category.dtype == object
    assert normalized_test.category.tolist() == values[:2]
    assert not pd.api.types.is_numeric_dtype(normalized_test.category)
    pd.testing.assert_series_equal(normalized_test.number, test.number)


def test_target_normalization_retains_unseen_category_levels_instead_of_fitted_vocab() -> None:
    train = pd.DataFrame({"category": pd.Categorical([1, 2, 1])})
    test = pd.DataFrame({"category": pd.Categorical(["1", 1, "new"])})
    actual = normalize_model_frame(train.iloc[:0], test, preserve_categorical_types=True)
    pd.testing.assert_frame_equal(actual, test)
    assert actual.category.iloc[0] == "1"
    assert actual.category.iloc[1] == 1
    assert actual.category.iloc[2] == "new"


def test_preserving_types_is_target_only_and_default_normalization_is_unchanged() -> None:
    train = pd.DataFrame({"category": pd.Series([1, "1", True, "True", None, "missing:"], dtype=object)})
    schema = train.iloc[:0]
    default = normalize_model_frame(schema, train)
    explicit = normalize_model_frame(schema, train, preserve_categorical_types=False)
    pd.testing.assert_frame_equal(default, explicit)
    assert default.category.tolist() == ["value:1", "value:1", "value:True", "value:True", "missing:", "value:missing:"]
    target = normalize_model_frame(schema, train, preserve_categorical_types=True)
    pd.testing.assert_frame_equal(target, train)
    canonical = normalize_categorical_frame(target)
    assert canonical.category.iloc[0] != canonical.category.iloc[1]
    assert canonical.category.iloc[2] != canonical.category.iloc[3]
    assert pd.isna(canonical.category.iloc[4]) and isinstance(canonical.category.iloc[5], str)


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame(),
        pd.DataFrame([[1, 2]], columns=["a", "a"]),
        pd.DataFrame({0: [1]}),
        pd.DataFrame({"a": [1 + 2j]}),
    ],
)
def test_frame_guards_reject_invalid_schema(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        validate_classification_frame(frame)


def test_numeric_and_boolean_roles_do_not_follow_inference_batch_dtype() -> None:
    train = pd.DataFrame({"number": [1.0], "flag": [True], "category": ["x"]})
    with pytest.raises(TypeError, match="numeric"):
        normalize_model_frame(train.iloc[:0], train.assign(number="bad"))
    with pytest.raises(TypeError, match="boolean"):
        normalize_model_frame(train.iloc[:0], train.assign(flag=1))
    with pytest.raises(ValueError, match="schema"):
        normalize_model_frame(train.iloc[:0], train[["category", "flag", "number"]])


class _MatrixRecorder(ClassifierMixin, BaseEstimator):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "_MatrixRecorder":
        self.training_matrix_ = np.asarray(X).copy()
        self.classes_ = np.unique(y)
        return self


@pytest.mark.parametrize("n_classes", [2, 3])
def test_unique_ids_use_complement_priors_and_inference_uses_full_prior(n_classes: int) -> None:
    counts = [43, 17] if n_classes == 2 else [31, 19, 10]
    y = np.repeat(np.arange(n_classes), counts)
    X = pd.DataFrame({"private_id": [f"private-{index}" for index in range(len(y))]})
    adapter = CrossFittedTargetEncoder(random_state=42)
    encoded = adapter.fit_transform(X, y)
    expected = np.empty((len(y), 1 if n_classes == 2 else n_classes))
    for train, heldout in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        if n_classes == 2:
            expected[heldout, 0] = np.mean(y[train])
        else:
            expected[heldout] = np.mean(np.eye(n_classes)[y[train]], axis=0)
    np.testing.assert_array_equal(encoded, expected)
    assert not np.array_equal(adapter.transform(X), encoded)
    unseen = pd.DataFrame({"private_id": ["unknown-private-id"]})
    prior = [[np.mean(y)]] if n_classes == 2 else [np.mean(np.eye(n_classes)[y], axis=0)]
    np.testing.assert_array_equal(adapter.transform(unseen), prior)
    report = adapter.get_encoding_report()
    assert report["target_type"] == ("binary" if n_classes == 2 else "multiclass")
    assert report["class_counts"] == counts
    assert report["training_output_bytes"] == encoded.nbytes
    assert report["output_width"] == encoded.shape[1]
    assert report["output_dtype"] == str(encoded.dtype)
    assert report["training_representation"] == "cross_fitted"
    serialized = json.dumps(report, allow_nan=False)
    assert "private-" not in serialized
    for key in ("categories_", "encodings_", "target_mean_", "training_matrix"):
        assert key not in serialized
    assert not any(isinstance(value, np.ndarray) and value.shape == encoded.shape for value in vars(adapter).values())


@pytest.mark.parametrize("n_classes", [2, 3])
def test_pipeline_passes_cross_fitted_matrix_directly_to_learner(
    n_classes: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    y = np.arange(90) % n_classes
    X = pd.DataFrame({"numeric": np.arange(90, dtype=float), "id": [f"id-{index}" for index in range(90)]})
    matrices = []
    original = CrossFittedTargetEncoder.fit_transform

    def recorded(self: CrossFittedTargetEncoder, X: pd.DataFrame, y: np.ndarray, **kwargs: object) -> np.ndarray:
        output = original(self, X, y, **kwargs)
        matrices.append(output.copy())
        return output

    monkeypatch.setattr(CrossFittedTargetEncoder, "fit_transform", recorded)
    monkeypatch.setattr(
        CrossFittedTargetEncoder, "fit", Mock(side_effect=AssertionError("Pipeline must use fit_transform"))
    )
    pipeline = make_classification_pipeline(X, _MatrixRecorder(), categorical_encoding="target", random_state=42)
    pipeline.fit(X, y)
    assert len(matrices) == 1
    preprocessor = pipeline.steps[1][1]
    span = preprocessor.output_indices_["categorical"]
    np.testing.assert_array_equal(pipeline.steps[-1][1].training_matrix_[:, span], matrices[0])
    self_inclusive = pipeline[:-1].transform(X)
    assert not np.array_equal(self_inclusive[:, span], matrices[0])
    assert preprocessor.output_indices_["numeric"] == slice(0, 1)
    assert span == slice(1, 1 + (1 if n_classes == 2 else n_classes))


@pytest.mark.parametrize("n_classes", [2, 3])
def test_encoder_fold_labels_cannot_influence_own_training_encoding(
    n_classes: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    y = np.arange(90) % n_classes
    X = pd.DataFrame({"category": [f"level-{index % 7}" for index in range(90)]})
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    monkeypatch.setattr(StratifiedKFold, "split", lambda *args, **kwargs: iter(splits))
    original = CrossFittedTargetEncoder().fit_transform(X, y)
    changed_y = y.copy()
    heldout = splits[0][1]
    changed_y[heldout] = 1
    changed = CrossFittedTargetEncoder().fit_transform(X, changed_y)
    np.testing.assert_array_equal(changed[heldout], original[heldout])
    assert not np.array_equal(changed[splits[1][1]], original[splits[1][1]])


def test_typed_missing_categories_multiclass_shape_batch_and_pickle() -> None:
    X = pd.DataFrame(
        {
            "category": [1, "1", None, np.nan, "missing:", "builtins.int:1"] * 10,
            "other": [True, False, "True", None, "x", "y"] * 10,
        }
    )
    y = np.arange(60) % 3
    normalized = normalize_categorical_frame(X)
    assert normalized.category.iloc[0] != normalized.category.iloc[1]
    assert pd.isna(normalized.category.iloc[2]) and pd.isna(normalized.category.iloc[3])
    assert normalized.category.iloc[4] != normalized.category.iloc[5]
    assert normalized.other.iloc[0] != normalized.other.iloc[2]
    adapter = CrossFittedTargetEncoder(42)
    output = adapter.fit_transform(X, y)
    assert output.shape == (60, 6)
    assert adapter.get_encoding_report()["output_width"] == 6
    test = pd.DataFrame({"category": ["new", None, 1], "other": ["new", None, True]})
    prediction = adapter.transform(test)
    np.testing.assert_array_equal(
        np.vstack([adapter.transform(test.iloc[:1]), adapter.transform(test.iloc[1:])]), prediction
    )
    restored = pickle.loads(pickle.dumps(adapter))
    np.testing.assert_array_equal(restored.transform(test), prediction)
    unfitted = clone(adapter)
    assert unfitted.get_params() == {"random_state": 42}
    with pytest.raises(NotFittedError):
        unfitted.transform(test)
    with pytest.raises(ValueError, match="schema"):
        adapter.transform(test[["other", "category"]])
    report = adapter.get_encoding_report()
    report["class_counts"] = []
    assert adapter.get_encoding_report()["class_counts"] == [20, 20, 20]
    assert len(adapter.get_feature_names_out()) == 6


@pytest.mark.parametrize("counts", [[5, 5], [5, 5, 5]])
def test_exact_minimum_class_support_is_accepted(counts: list[int]) -> None:
    y = np.repeat(np.arange(len(counts)), counts)
    X = pd.DataFrame({"category": ["same"] * len(y)})
    assert np.isfinite(CrossFittedTargetEncoder().fit_transform(X, y)).all()


@pytest.mark.parametrize(
    "y", [np.array([0] * 4 + [1] * 6), np.zeros(10), np.linspace(0, 1, 10), np.array([0, 1] * 4 + [None, None])]
)
def test_invalid_encoder_targets_fail_before_construction(y: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> None:
    factory = Mock(side_effect=AssertionError("Invalid labels must fail before constructing sklearn encoder"))
    monkeypatch.setattr(model_preprocessing, "_make_target_encoder", factory)
    with pytest.raises(ValueError):
        CrossFittedTargetEncoder().fit_transform(pd.DataFrame({"category": ["same"] * len(y)}), y)
    factory.assert_not_called()


@pytest.mark.parametrize("key", ["groups", "timestamps", "label_end_times", "sample_weight", "eval_set"])
def test_encoder_fit_metadata_is_explicitly_rejected(key: str) -> None:
    with pytest.raises(TypeError, match="IID"):
        CrossFittedTargetEncoder().fit_transform(
            pd.DataFrame({"category": ["x"] * 10}), np.array([0, 1] * 5), **{key: None}
        )


@pytest.mark.parametrize("version", ["1.0.2", "1.3.2", "unknown"])
def test_target_encoder_version_requirement_is_feature_local(version: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sklearn, "__version__", version)
    X = pd.DataFrame({"category": ["a", "b"] * 6})
    y = np.array([0, 1] * 6)
    with pytest.raises(ImportError, match="scikit-learn >=1.4"):
        CrossFittedTargetEncoder().fit_transform(X, y)
    numeric = pd.DataFrame({"number": np.arange(8, dtype=float)})
    numeric_y = np.array([0, 1] * 4)
    default = make_classification_pipeline(numeric, LogisticRegression()).fit(numeric, numeric_y)
    explicit = make_classification_pipeline(numeric, LogisticRegression(), categorical_encoding="one_hot").fit(
        numeric, numeric_y
    )
    inactive = make_classification_pipeline(numeric, LogisticRegression(), categorical_encoding="target").fit(
        numeric, numeric_y
    )
    np.testing.assert_array_equal(default.predict_proba(numeric), explicit.predict_proba(numeric))
    np.testing.assert_array_equal(default.predict_proba(numeric), inactive.predict_proba(numeric))
    make_classification_pipeline(X, LogisticRegression()).fit(X, y)


@pytest.mark.parametrize("version", ["1.4.2", "1.8.0", "1.9.0", "2.0.0"])
def test_version_specific_constructor_uses_only_approved_fixed_arguments(
    version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    constructor = Mock(return_value=BaseEstimator())
    monkeypatch.setattr(sklearn, "__version__", version)
    monkeypatch.setattr(sklearn.preprocessing, "TargetEncoder", constructor)
    model_preprocessing._make_target_encoder("multiclass", 137)
    constructor.assert_called_once()
    kwargs = constructor.call_args.kwargs
    assert kwargs["target_type"] == "multiclass"
    assert kwargs["smooth"] == "auto"
    if version in ("1.4.2", "1.8.0"):
        assert kwargs["cv"] == 5
        assert kwargs["shuffle"] is True
        assert kwargs["random_state"] == 137
    else:
        assert "shuffle" not in kwargs and "random_state" not in kwargs
        assert isinstance(kwargs["cv"], StratifiedKFold)
        assert kwargs["cv"].n_splits == 5
        assert kwargs["cv"].shuffle is True
        assert kwargs["cv"].random_state == 137


@pytest.mark.parametrize("encoding", ["bad", "target_mean", None, True, ["target"]])
def test_invalid_encoding_mode_is_rejected_even_without_categorical_columns(encoding: object) -> None:
    with pytest.raises(ValueError, match="categorical_encoding"):
        model_preprocessing.validate_categorical_encoding(encoding)


def test_failed_encoder_refit_clears_inference_state() -> None:
    X = pd.DataFrame({"category": ["a", "b"] * 10})
    adapter = CrossFittedTargetEncoder().fit(X, np.array([0, 1] * 10))
    with pytest.raises(ValueError, match="five"):
        adapter.fit(X.iloc[:4], np.array([0, 1, 0, 1]))
    with pytest.raises(NotFittedError):
        adapter.transform(X)
