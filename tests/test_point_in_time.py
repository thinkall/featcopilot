"""Cutoff, immutability and schema contracts for relational history features."""

import pickle

import numpy as np
import pandas as pd
import pytest

from featcopilot.engines.point_in_time import PointInTimeAggregation, PointInTimeRelationalEngine


def queries():
    return pd.DataFrame(
        {
            "customer": ["a", "a", "b", "new"],
            "when": pd.to_datetime(["2025-01-03", "2025-01-05", "2025-01-05", "2025-01-05"], utc=True),
        },
        index=[50, 10, 80, 20],
    )


def history():
    return pd.DataFrame(
        {
            "entity": ["a", "a", "a", "a", "b"],
            "available": pd.to_datetime(
                ["2025-01-01", "2025-01-03", "2025-01-04", "2025-01-06", "2025-01-02"], utc=True
            ),
            "value": [1.0, 3.0, 4.0, 1000.0, np.nan],
        }
    )


def engine(method="mean", window=None):
    return PointInTimeRelationalEngine(
        [
            {
                "name": "historical",
                "table": "events",
                "entity_column": "entity",
                "available_at": "available",
                "value_column": "value",
                "aggregation": method,
                "window": window,
            }
        ],
        entity_column="customer",
        prediction_time_column="when",
    )


def test_strict_per_row_cutoff_preserves_originals_and_order():
    X = queries()
    result = engine().fit_transform(X, related_tables={"events": history()})
    pd.testing.assert_frame_equal(result[X.columns], X)
    assert result.historical.iloc[0] == 1.0
    assert result.historical.iloc[1] == pytest.approx(8 / 3)
    assert result.historical.iloc[2:].isna().all()


def test_future_records_and_other_entities_cannot_change_past_features():
    X = queries()
    original = engine().fit_transform(X, related_tables={"events": history()})
    changed = history()
    changed.loc[3, "value"] = -1e10
    changed.loc[4, "value"] = 99.0
    revised = engine().fit_transform(X, related_tables={"events": changed})
    pd.testing.assert_series_equal(original.historical.iloc[:2], revised.historical.iloc[:2])


def test_trailing_window_includes_lower_bound_but_excludes_query_time():
    result = engine(window="2D").fit_transform(queries(), related_tables={"events": history()})
    assert result.historical.iloc[0] == 1.0
    assert result.historical.iloc[1] == 3.5


@pytest.mark.parametrize(
    "method,expected", [("count", 3), ("sum", 8), ("min", 1), ("max", 4), ("last", 4), ("std", np.std([1, 3, 4]))]
)
def test_aggregate_definitions(method, expected):
    result = engine(method).fit_transform(queries(), related_tables={"events": history()})
    assert result.historical.iloc[1] == pytest.approx(expected)


def test_row_count_can_omit_value_column():
    specification = {
        "name": "events_count",
        "table": "events",
        "entity_column": "entity",
        "available_at": "available",
        "aggregation": "count",
    }
    fitted = PointInTimeRelationalEngine([specification], "customer", "when").fit(
        queries(), related_tables={"events": history()}
    )
    assert fitted.transform(queries()).events_count.tolist() == [1, 3, 1, 0]


def test_frozen_history_batch_invariance_and_pickle():
    source = history()
    fitted = engine().fit(queries(), related_tables={"events": source})
    expected = fitted.transform(queries())
    source["value"] = -99.0
    restored = pickle.loads(pickle.dumps(fitted))
    parts = pd.concat([restored.transform(queries().iloc[:2]), restored.transform(queries().iloc[2:])])
    pd.testing.assert_frame_equal(expected, parts)
    assert fitted.get_feature_set()["historical"].metadata["point_in_time"]["available_at"] == "available"


def test_refit_resets_state_and_invalid_refit_is_unfitted():
    fitted = engine().fit(queries(), related_tables={"events": history()})
    with pytest.raises(ValueError, match="Missing related table"):
        fitted.fit(queries(), related_tables={})
    with pytest.raises(RuntimeError, match="fitted"):
        fitted.transform(queries())


def test_transform_cannot_replace_history_or_change_schema():
    fitted = engine().fit(queries(), related_tables={"events": history()})
    with pytest.raises(TypeError, match="frozen history"):
        fitted.transform(queries(), related_tables={"events": history()})
    with pytest.raises(ValueError, match="schema"):
        fitted.transform(queries().assign(unexpected=1))


@pytest.mark.parametrize(
    "changes", [{"window": "0D"}, {"aggregation": "unknown"}, {"value_column": None}, {"name": ""}]
)
def test_invalid_aggregation_specs_are_rejected(changes):
    base = {
        "name": "result",
        "table": "events",
        "entity_column": "entity",
        "available_at": "available",
        "value_column": "value",
    }
    with pytest.raises(ValueError):
        PointInTimeAggregation(**{**base, **changes})


def test_numeric_times_require_explicit_unit_conversion():
    X = queries()
    X["when"] = [1, 2, 3, 4]
    with pytest.raises(ValueError, match="explicit unit"):
        engine().fit(X, related_tables={"events": history()})


def test_collision_and_missing_history_metadata_fail_explicitly():
    with pytest.raises(ValueError, match="collides"):
        engine().fit(queries().assign(historical=0), related_tables={"events": history()})
    data = history()
    data.loc[0, "available"] = pd.NaT
    with pytest.raises(ValueError, match="missing timestamps"):
        engine().fit(queries(), related_tables={"events": data})


def test_complex_values_cannot_silently_lose_their_imaginary_part():
    data = history()
    data["value"] = np.array([1 + 2j] * len(data))
    with pytest.raises(ValueError, match="real numeric"):
        engine().fit(queries(), related_tables={"events": data})


def test_timezone_equivalent_cutoffs_and_duplicate_query_indices():
    X = pd.DataFrame(
        {"customer": ["a", "a"], "when": pd.to_datetime(["2025-01-01T04:00:00Z", "2025-01-01T05:00:00Z"])},
        index=[7, 7],
    )
    events = pd.DataFrame(
        {
            "entity": ["a"],
            "available": pd.to_datetime(["2025-01-01T06:00:00+02:00"]),
            "value": [10.0],
        }
    )
    result = engine("count").fit_transform(X, related_tables={"events": events})
    assert result.historical.tolist() == [0, 1]
    assert result.index.tolist() == [7, 7]


def test_failed_refit_clears_generated_metadata():
    fitted = engine().fit(queries(), related_tables={"events": history()})
    assert fitted.get_feature_metadata()
    with pytest.raises(ValueError):
        fitted.fit(queries(), related_tables={})
    assert fitted.get_feature_metadata() == {}
    assert fitted.get_feature_names() == []
