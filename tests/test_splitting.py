"""Shared split protocols for engine selection and benchmark evaluation."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from featcopilot.engines.predictive import PredictiveTabularEngine
from featcopilot.utils.splitting import (
    make_cv_splits,
    make_holdout_splits,
    validate_split_boundaries,
    validate_split_metadata,
)


def assert_same_splits(first, second):
    assert len(first) == len(second)
    for (first_train, first_test), (second_train, second_test) in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_train, second_train)
        np.testing.assert_array_equal(first_test, second_test)


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_iid_cv_matches_sklearn_and_returns_positions(task):
    target = np.arange(120) if task == "regression" else np.array(["common"] * 90 + ["rare"] * 30)
    y = pd.Series(target, index=np.arange(len(target)) * 5 + 1000)
    actual = make_cv_splits(y, task, 3, 42)
    splitter = (
        KFold(3, shuffle=True, random_state=42)
        if task == "regression"
        else StratifiedKFold(3, shuffle=True, random_state=42)
    )
    expected = list(splitter.split(target, target))
    assert_same_splits(actual, expected)
    np.testing.assert_array_equal(np.sort(np.concatenate([test for _, test in actual])), np.arange(len(y)))


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_repeated_iid_holdouts_match_independently_seeded_sklearn(task):
    y = np.arange(120) if task == "regression" else np.array(["common"] * 90 + ["rare"] * 30)
    actual = make_holdout_splits(y, task, 3, 2**32 - 1, test_size=0.2)
    expected = [
        train_test_split(
            np.arange(len(y)),
            test_size=0.2,
            random_state=seed,
            stratify=y if task == "classification" else None,
        )
        for seed in [2**32 - 1, 0, 1]
    ]
    assert_same_splits(actual, expected)
    assert len({tuple(test) for _, test in actual}) == 3


@pytest.mark.parametrize("function", [make_cv_splits, make_holdout_splits])
def test_group_splits_are_disjoint_and_metadata_is_positional(function):
    groups = np.repeat(np.arange(12), np.arange(1, 13))
    y = np.arange(len(groups), dtype=float)
    rng = np.random.default_rng(531)
    permutation = rng.permutation(len(groups))
    y, groups = y[permutation], groups[permutation]
    metadata = pd.Series(groups, index=np.arange(len(groups))[::-1] + 1000)
    actual = function(y, "regression", 3, 42, groups=metadata)
    assert_same_splits(actual, function(y, "regression", 3, 42, groups=groups))
    for train, test in actual:
        assert set(train).isdisjoint(test)
        assert set(groups[train]).isdisjoint(groups[test])
        assert set(train) | set(test) == set(range(len(y)))
        if function is make_holdout_splits:
            assert len(np.unique(groups[test])) == 3
    if function is make_cv_splits:
        np.testing.assert_array_equal(np.sort(np.concatenate([test for _, test in actual])), np.arange(len(y)))


@pytest.mark.parametrize("function", [make_cv_splits, make_holdout_splits])
def test_temporal_splits_use_atomic_blocks_and_disjoint_forward_windows(function):
    timestamps = np.repeat(pd.date_range("2024-01-01", periods=24, tz="UTC").to_numpy(), 3)
    y = np.arange(len(timestamps), dtype=float)
    permutation = np.random.default_rng(194).permutation(len(y))
    y, timestamps = y[permutation], timestamps[permutation]
    actual = function(y, "regression", 3, 42, timestamps=timestamps)
    assert_same_splits(actual, function(y, "regression", 3, 24, timestamps=timestamps))
    scored = set()
    for train, test in actual:
        assert timestamps[train].max() < timestamps[test].min()
        assert scored.isdisjoint(test)
        scored.update(test)
        for timestamp in timestamps[test]:
            assert set(np.flatnonzero(timestamps == timestamp)) <= set(test)
    assert 0 < len(scored) < len(y)


def test_temporal_holdout_fraction_is_applied_to_first_evaluation_prefix():
    timestamps = np.repeat(np.arange(36), 2)
    y = np.arange(len(timestamps), dtype=float)
    quarter = make_holdout_splits(y, "regression", 3, 42, timestamps=timestamps, test_size=0.25)
    half = make_holdout_splits(y, "regression", 3, 42, timestamps=timestamps, test_size=0.5)
    assert [len(test) for _, test in quarter] == [12, 12, 12]
    assert [len(test) for _, test in half] == [18, 18, 18]
    assert len(quarter[0][1]) / (len(quarter[0][0]) + len(quarter[0][1])) == 0.25
    assert len(half[0][1]) / (len(half[0][0]) + len(half[0][1])) == 0.5


def test_temporal_single_holdout_preserves_three_quarter_prefix():
    timestamps = np.repeat(np.arange(20), 2)
    y = np.arange(len(timestamps), dtype=float)
    train, test = make_holdout_splits(y, "regression", 1, 42, timestamps=timestamps)[0]
    np.testing.assert_array_equal(train, np.arange(30))
    np.testing.assert_array_equal(test, np.arange(30, 40))


def test_cv_feasibility_caps_are_explicit_in_returned_fold_count():
    assert len(make_cv_splits(["a", "b"] * 2, "classification", 5, 42)) == 2
    assert len(make_cv_splits(np.arange(12), "regression", 5, 42, groups=np.repeat(np.arange(3), 4))) == 3
    assert len(make_cv_splits(np.arange(9), "regression", 2, 42, timestamps=np.repeat(np.arange(3), 3))) == 1


def test_equal_instants_with_different_timezones_share_one_block():
    timestamps = [
        pd.Timestamp("2024-01-01T00:00:00Z"),
        pd.Timestamp("2023-12-31T19:00:00-05:00"),
        pd.Timestamp("2024-01-02T00:00:00Z"),
    ]
    metadata = validate_split_metadata(None, timestamps, len(timestamps))
    np.testing.assert_array_equal(metadata.values, [0, 0, 1])
    assert metadata.groups is None
    assert metadata.kind == "temporal"
    np.testing.assert_array_equal(metadata.subset(np.array([1, 2])).timestamps, [0, 1])


@pytest.mark.parametrize("function", [make_cv_splits, make_holdout_splits])
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"groups": np.arange(12), "timestamps": np.arange(12)}, "together"),
        ({"groups": np.arange(11)}, "same number"),
        ({"groups": np.ones((12, 1))}, "one-dimensional"),
        ({"groups": np.full(12, np.nan)}, "missing"),
        ({"timestamps": np.full(12, np.nan)}, "missing"),
        ({"timestamps": ["2024-01-01"] * 11 + ["NaT"]}, "missing"),
        ({"timestamps": ["2024-01-01"] * 11 + [""]}, "missing"),
        ({"timestamps": np.full(12, np.inf)}, "finite"),
        ({"timestamps": np.arange(12) + 1j}, "finite"),
        ({"timestamps": ["invalid"] * 12}, "valid datetimes"),
    ],
)
def test_invalid_metadata_is_never_a_random_fallback(function, kwargs, message):
    with pytest.raises(ValueError, match=message):
        function(np.arange(12), "regression", 2, 42, **kwargs)


@pytest.mark.parametrize("function", [make_cv_splits, make_holdout_splits])
@pytest.mark.parametrize(
    ("n_splits", "random_state", "message"),
    [
        (0, 42, "n_splits"),
        (2.5, 42, "n_splits"),
        (True, 42, "n_splits"),
        (2, -1, "random_state"),
        (2, 2**32, "random_state"),
    ],
)
def test_invalid_options_are_visible(function, n_splits, random_state, message):
    with pytest.raises(ValueError, match=message):
        function(np.arange(12), "regression", n_splits, random_state)


@pytest.mark.parametrize("test_size", [0, 1, -0.1, np.inf, np.nan, ".25", True])
def test_invalid_holdout_fraction(test_size):
    with pytest.raises(ValueError, match="test_size"):
        make_holdout_splits(np.arange(12), "regression", 2, 42, test_size=test_size)


@pytest.mark.parametrize("function", [make_cv_splits, make_holdout_splits])
@pytest.mark.parametrize(
    ("y", "message"),
    [
        (np.ones((12, 2)), "one-dimensional"),
        (np.array([]), "non-empty"),
        (np.ones(12), "constant"),
        (np.full(12, np.nan), "missing"),
    ],
)
def test_invalid_supervised_targets(function, y, message):
    with pytest.raises(ValueError, match=message):
        function(y, "regression", 2, 42)


def test_too_few_groups_or_time_blocks_raise_helpfully():
    y = np.arange(12)
    for function in [make_cv_splits, make_holdout_splits]:
        with pytest.raises(ValueError, match="groups"):
            function(y, "regression", 2, 42, groups=np.zeros(12))
        with pytest.raises(ValueError, match="time blocks"):
            function(y, "regression", 3, 42, timestamps=np.zeros(12))


def test_missing_historical_class_is_not_randomly_repaired():
    y = ["early"] * 18 + ["late"] * 6
    timestamps = np.repeat(np.arange(12), 2)
    for function in [make_cv_splits, make_holdout_splits]:
        with pytest.raises(ValueError, match="every class"):
            function(y, "classification", 3, 42, timestamps=timestamps)


@pytest.mark.parametrize("kind", ["iid", "grouped", "temporal"])
def test_engine_routes_holdout_and_nested_cv_through_shared_helpers(kind):
    rng = np.random.default_rng(331)
    X = pd.DataFrame(rng.normal(size=(160, 2)), columns=["a", "b"])
    y = X["a"] * X["b"] + rng.normal(scale=0.1, size=len(X))
    metadata = np.repeat(np.arange(20), 8)
    kwargs = {} if kind == "iid" else {"groups" if kind == "grouped" else "timestamps": metadata}
    with (
        threadpool_limits(limits=1),
        patch("featcopilot.engines.predictive.make_cv_splits", wraps=make_cv_splits) as cv,
        patch("featcopilot.engines.predictive.make_holdout_splits", wraps=make_holdout_splits) as holdout,
    ):
        engine = PredictiveTabularEngine(
            task="regression", estimator=Ridge(alpha=0.01), max_candidates=5, max_features=3, stability_repeats=1
        ).fit(X, y, **kwargs)
    holdout.assert_called_once()
    assert cv.call_count == 4
    assert engine.selection_report_["split_kind"] == kind


@pytest.mark.parametrize("function", [make_cv_splits, make_holdout_splits])
def test_label_horizons_purge_only_unavailable_training_rows(function):
    timestamps = np.repeat(np.arange(24, dtype=float), 2)
    ends = timestamps + 3
    ends[0] = 100
    y = np.arange(len(timestamps), dtype=float)
    unpurged = function(y, "regression", 3, 42, timestamps=timestamps)
    purged = function(y, "regression", 3, 42, timestamps=timestamps, label_end_times=ends)
    for (train, validation), (original_train, original_validation) in zip(purged, unpurged, strict=True):
        cutoff = timestamps[validation].min()
        expected_train = original_train[ends[original_train] < cutoff]
        np.testing.assert_array_equal(train, expected_train)
        np.testing.assert_array_equal(validation, original_validation)
        assert 0 not in train
        assert ends[train].max() < cutoff
        assert not set(np.flatnonzero(ends == cutoff)) & set(train)
        assert len(train) < len(original_train)
        for timestamp in np.unique(timestamps[validation]):
            assert set(np.flatnonzero(timestamps == timestamp)) <= set(validation)


@pytest.mark.parametrize("function", [make_cv_splits, make_holdout_splits])
def test_instantaneous_labels_and_explicit_none_preserve_existing_splits(function):
    y = np.arange(48, dtype=float)
    timestamps = np.repeat(np.arange(24), 2)
    original = function(y, "regression", 3, 42, timestamps=timestamps)
    assert_same_splits(original, function(y, "regression", 3, 42, timestamps=timestamps, label_end_times=timestamps))
    assert_same_splits(original, function(y, "regression", 3, 42, timestamps=timestamps, label_end_times=None))
    assert_same_splits(function(y, "regression", 3, 42), function(y, "regression", 3, 42, label_end_times=None))


@pytest.mark.parametrize("function", [make_cv_splits, make_holdout_splits])
def test_timezone_and_datetime_resolution_horizons_use_matching_instants(function):
    prediction = pd.date_range("2024-01-01", periods=24, tz="UTC")
    available = (prediction + pd.Timedelta(days=2)).tz_convert("US/Eastern")
    y = np.arange(24, dtype=float)
    actual = function(y, "regression", 3, 42, timestamps=prediction, label_end_times=available)
    expected = function(y, "regression", 3, 42, timestamps=np.arange(24), label_end_times=np.arange(24) + 2)
    assert_same_splits(actual, expected)
    seconds = prediction.tz_localize(None).to_numpy(dtype="datetime64[s]")
    nanoseconds = available.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")
    assert_same_splits(actual, function(y, "regression", 3, 42, timestamps=seconds, label_end_times=nanoseconds))


def test_label_horizon_codes_preserve_large_integer_precision_and_subset_alignment():
    timestamps = np.arange(12, dtype=np.uint64) + np.uint64(2**63)
    ends = timestamps + np.uint64(1)
    metadata = validate_split_metadata(None, timestamps, len(timestamps), label_end_times=ends)
    assert metadata.label_end_times is not None
    assert np.all(metadata.label_end_times > metadata.timestamps)
    indices = np.array([2, 4, 7, 9])
    subset = metadata.subset(indices)
    np.testing.assert_array_equal(subset.label_end_times, metadata.label_end_times[indices])
    renormalized = validate_split_metadata(
        None, subset.timestamps, len(indices), label_end_times=subset.label_end_times
    )
    for cutoff in range(len(indices)):
        np.testing.assert_array_equal(
            subset.label_end_times < subset.timestamps[cutoff],
            renormalized.label_end_times < renormalized.timestamps[cutoff],
        )


@pytest.mark.parametrize("function", [make_cv_splits, make_holdout_splits])
def test_mixed_availability_within_a_past_timestamp_never_splits_a_validation_block(function):
    timestamps = np.repeat(np.arange(24), 2)
    ends = timestamps.copy()
    ends[0] = 100
    y = np.arange(48)
    for train, validation in function(y, "regression", 3, 42, timestamps=timestamps, label_end_times=ends):
        assert 0 not in train
        assert 1 in train
        assert not set(timestamps[train]) & set(timestamps[validation])
        for timestamp in np.unique(timestamps[validation]):
            assert set(np.flatnonzero(timestamps == timestamp)) <= set(validation)


@pytest.mark.parametrize("function", [make_cv_splits, make_holdout_splits])
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"label_end_times": np.arange(24)}, "requires timestamps"),
        ({"groups": np.arange(24), "label_end_times": np.arange(24)}, "requires timestamps"),
        ({"timestamps": np.arange(24), "label_end_times": np.arange(23)}, "same number"),
        ({"timestamps": np.arange(24), "label_end_times": np.ones((24, 1))}, "one-dimensional"),
        ({"timestamps": np.arange(24), "label_end_times": np.full(24, np.nan)}, "missing"),
        ({"timestamps": np.arange(24), "label_end_times": np.full(24, np.inf)}, "finite"),
        ({"timestamps": np.arange(24), "label_end_times": np.arange(24) + 1j}, "finite"),
        ({"timestamps": np.arange(24), "label_end_times": np.arange(24) - 1}, "greater than or equal"),
        (
            {"timestamps": np.arange(24), "label_end_times": pd.date_range("2024-01-01", periods=24)},
            "same time domain",
        ),
        (
            {"timestamps": pd.date_range("2024-01-01", periods=24), "label_end_times": np.arange(24)},
            "same time domain",
        ),
        (
            {"timestamps": pd.date_range("2024-01-01", periods=24), "label_end_times": ["NaT"] * 24},
            "missing",
        ),
        (
            {"groups": np.arange(24), "timestamps": np.arange(24), "label_end_times": np.arange(24)},
            "together",
        ),
    ],
)
def test_invalid_label_horizons_raise_without_fallback(function, kwargs, message):
    with pytest.raises(ValueError, match=message):
        function(np.arange(24), "regression", 3, 42, **kwargs)


@pytest.mark.parametrize("function", [make_cv_splits, make_holdout_splits])
@pytest.mark.parametrize("allow_constant", [False, True])
def test_purging_all_available_history_is_an_explicit_error(function, allow_constant):
    with pytest.raises(ValueError, match="purging.*no.*training"):
        function(
            np.arange(24),
            "regression",
            3,
            42,
            timestamps=np.arange(24),
            label_end_times=np.full(24, 100),
            allow_constant_target=allow_constant,
        )


@pytest.mark.parametrize("function", [make_cv_splits, make_holdout_splits])
@pytest.mark.parametrize("kind", ["iid", "grouped", "temporal"])
def test_constant_regression_opt_in_is_label_independent(function, kind):
    values = np.repeat(np.arange(12), 4)
    metadata = {} if kind == "iid" else {"groups" if kind == "grouped" else "timestamps": values}
    if kind == "temporal":
        metadata["label_end_times"] = values + 1
    target = np.zeros(len(values))
    with pytest.raises(ValueError, match="constant"):
        function(target, "regression", 3, 42, **metadata)
    constant_splits = function(target, "regression", 3, 42, allow_constant_target=True, **metadata)
    perturbed = target.copy()
    perturbed[-1] = 1000
    assert_same_splits(
        constant_splits, function(perturbed, "regression", 3, 42, allow_constant_target=True, **metadata)
    )


@pytest.mark.parametrize("function", [make_cv_splits, make_holdout_splits])
def test_constant_regression_opt_in_allows_one_available_historical_label(function):
    timestamps = np.arange(12)
    ends = np.full(12, 100)
    ends[0] = 0
    partitions = function(
        np.ones(12),
        "regression",
        2,
        42,
        timestamps=timestamps,
        label_end_times=ends,
        allow_constant_target=True,
    )
    for train, _ in partitions:
        np.testing.assert_array_equal(train, [0])


@pytest.mark.parametrize("function", [make_cv_splits, make_holdout_splits])
def test_constant_target_flag_never_relaxes_classification_coverage(function):
    with pytest.raises(ValueError, match="constant"):
        function(["one"] * 24, "classification", 3, 42, allow_constant_target=True)
    with pytest.raises(ValueError, match="every class"):
        function(
            ["first"] * 18 + ["second"] * 6,
            "classification",
            3,
            42,
            timestamps=np.arange(24),
            allow_constant_target=True,
        )


@pytest.mark.parametrize("function", [make_cv_splits, make_holdout_splits])
@pytest.mark.parametrize("value", [None, 1, "true"])
def test_constant_target_flag_requires_an_explicit_boolean(function, value):
    with pytest.raises(ValueError, match="allow_constant_target"):
        function(np.arange(24), "regression", 3, 42, allow_constant_target=value)


def test_official_indices_are_validated_without_mutation_or_reordering():
    train = np.array([5, 1, 3, 0], dtype=np.int32)
    test = np.array([9, 7], dtype=np.int32)
    before_train, before_test = train.copy(), test.copy()
    timestamps = np.arange(12)
    ends = timestamps + 1
    assert validate_split_boundaries(train, test, timestamps=timestamps, label_end_times=ends) is None
    np.testing.assert_array_equal(train, before_train)
    np.testing.assert_array_equal(test, before_test)
    assert train.dtype == test.dtype == np.dtype(np.int32)


def test_official_unavailable_training_label_is_rejected_not_purged():
    train, test = np.array([0, 1, 2]), np.array([4, 5])
    ends = np.arange(6)
    ends[2] = 4
    with pytest.raises(ValueError, match="labels must be available strictly"):
        validate_split_boundaries(train, test, timestamps=np.arange(6), label_end_times=ends)
    np.testing.assert_array_equal(train, [0, 1, 2])
    np.testing.assert_array_equal(test, [4, 5])


def test_official_group_and_equal_time_boundary_checks():
    validate_split_boundaries([0, 1], [2, 3], groups=["a", "a", "b", "b"])
    with pytest.raises(ValueError, match="same group"):
        validate_split_boundaries([0, 1], [2, 3], groups=["a", "b", "b", "c"])
    with pytest.raises(ValueError, match="shared time blocks"):
        validate_split_boundaries([0, 1], [2, 3], timestamps=[0, 1, 1, 2])
    with pytest.raises(ValueError, match="strictly earlier"):
        validate_split_boundaries([0, 3], [1, 2], timestamps=np.arange(4))
    with pytest.raises(ValueError, match="together"):
        validate_split_boundaries([0, 1], [2, 3], groups=["a", "a", "b", "b"], timestamps=np.arange(4))


@pytest.mark.parametrize(
    ("train", "test", "kwargs", "message"),
    [
        ([], [1], {}, "nonempty"),
        ([0], [], {}, "nonempty"),
        ([[0, 1]], [2], {}, "one-dimensional"),
        ([0.0, 1.0], [2], {}, "integer"),
        ([True, False], [2], {}, "integer"),
        ([-1, 0], [2], {}, "nonnegative"),
        ([0, 0], [2], {}, "unique"),
        ([0], [2, 2], {}, "unique"),
        ([0, 1], [1, 2], {}, "disjoint"),
        ([0], [3], {"n_rows": 3}, "row count"),
        ([0], [3], {"groups": ["a", "b", "c"]}, "row count"),
        ([0], [2], {"n_rows": 4, "timestamps": [0, 1, 2]}, "same number"),
        ([0], [2], {"label_end_times": [0, 1, 2]}, "requires timestamps"),
    ],
)
def test_official_invalid_indices_and_metadata_are_explicit(train, test, kwargs, message):
    with pytest.raises(ValueError, match=message):
        validate_split_boundaries(train, test, **kwargs)


def test_generated_splits_also_pass_official_boundary_validation():
    timestamps = np.repeat(np.arange(24), 2)
    ends = timestamps + 2
    for function in [make_cv_splits, make_holdout_splits]:
        for train, test in function(np.arange(48), "regression", 3, 42, timestamps=timestamps, label_end_times=ends):
            validate_split_boundaries(train, test, timestamps=timestamps, label_end_times=ends, n_rows=48)
