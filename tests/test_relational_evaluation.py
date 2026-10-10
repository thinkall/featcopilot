"""Tiny fixtures only: no downloads, LLM calls, or predictive model training."""

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from benchmarks import relational_evaluation as pilot
from benchmarks import tabular


def transaction(customer, date, quantity=1, price=2.0, invoice="12345"):
    return {
        "InvoiceNo": invoice,
        "StockCode": "A",
        "Description": "fixture",
        "Quantity": quantity,
        "InvoiceDate": date,
        "UnitPrice": price,
        "CustomerID": customer,
        "Country": "United Kingdom",
    }


@pytest.fixture
def raw():
    rows = [transaction("1", "2010-12-01 09:00"), transaction("2", "2010-12-03 10:00")]
    rows += [
        transaction("1", date.strftime("%Y-%m-%d %H:%M"), quantity=week + 1)
        for week, date in enumerate(pd.date_range("2011-01-03 09:00", "2011-12-05", freq="7D"))
    ]
    rows += [transaction("3", "2011-06-01 10:00", quantity=999)]
    return pd.DataFrame(rows)


@pytest.fixture
def prepared(raw):
    return pilot.prepare_data(raw, pilot.PilotConfig(customers=2))


def test_cleaning_preserves_signed_refunds_and_duplicates_with_explicit_counts():
    original = transaction("1", "12/1/2010 08:26", 3, 2)
    frame = pd.DataFrame(
        [
            original,
            original,
            transaction("1", "12/2/2010 08:26", -2, 2, "C123"),
            transaction(None, "12/3/2010 08:26"),
            transaction("", "12/3/2010 08:26"),
            transaction("1", "12/3/2010 08:26", price=0),
            transaction("1", "12/3/2010 08:26", price=-1),
            transaction("1", "12/3/2010 08:26", price=np.inf),
            transaction("1", "12/3/2010 08:26", price=np.nan),
        ]
    )
    cleaned, audit = pilot.clean_transactions(frame)
    assert cleaned.net_spend.tolist() == [6, 6, -4]
    assert cleaned.InvoiceDate.iloc[0] == pd.Timestamp("2010-12-01 08:26")
    assert audit["removed_missing_customer"] == 2
    assert audit["removed_nonpositive_or_nonfinite_price_after_missing_customer"] == 4
    assert audit["retained_rows"] == 3
    assert audit["retained_exact_duplicates"] == 1
    assert audit["retained_negative_quantity"] == audit["retained_cancellation_rows"] == 1


@pytest.mark.parametrize("column,value", [("Quantity", np.inf), ("InvoiceDate", "not-a-date"), ("CustomerID", "bad")])
def test_invalid_values_fail_explicitly_instead_of_unregistered_cleaning(column, value):
    row = transaction("1", "2010-12-01 09:00")
    row[column] = value
    with pytest.raises(ValueError):
        pilot.clean_transactions(pd.DataFrame([row]))


def test_cohort_selection_depends_only_on_december_history_and_seed():
    past = pd.DataFrame([transaction(str(customer), "2010-12-04 09:00") for customer in range(250)])
    later = pd.DataFrame([transaction("999", "2011-01-02 09:00", 1000000)])
    first, _ = pilot.clean_transactions(past)
    second, _ = pilot.clean_transactions(pd.concat([past, later], ignore_index=True))
    assert pilot.select_cohort(first) == pilot.select_cohort(second)
    assert len(pilot.select_cohort(first)) == 200
    assert "999" not in pilot.select_cohort(second)
    # Source row ordering is not an implicit sampling criterion.
    assert pilot.select_cohort(first.iloc[::-1]) == pilot.select_cohort(first)


def test_all_customers_remain_at_all_cutoffs_even_with_zero_future_spend(prepared):
    q = prepared.queries
    assert len(q) == 2 * 48
    assert q.groupby("cutoff").CustomerID.nunique().eq(2).all()
    assert q.groupby("CustomerID").size().eq(48).all()
    assert prepared.target.loc[q.CustomerID == "2"].eq(0).all()
    assert set(q.CustomerID) == {"1", "2"}
    assert q.label_end.max() <= pilot.COVERAGE_END
    assert q.cutoff.max() == pd.Timestamp("2011-11-28")
    assert prepared.provenance["requested_cutoffs"] == 49
    assert prepared.provenance["end_censored_queries_removed"] == 2


def test_signed_label_interval_is_left_closed_right_open():
    data = pd.DataFrame(
        [
            transaction("1", "2010-12-01 09:00"),
            transaction("1", "2011-01-02 23:59", 100),
            transaction("1", "2011-01-03 00:00", 5),
            transaction("1", "2011-01-09 23:59", -2, invoice="C1"),
            transaction("1", "2011-01-10 00:00", 1000),
        ]
    )
    prepared = pilot.prepare_data(data, pilot.PilotConfig())
    assert prepared.target.iloc[0] == 6
    assert prepared.features["simple_history"].spend_7d_sum.iloc[0] == 200
    assert prepared.features["expanded_pit"].spend_7d_count.iloc[0] == 1


@pytest.mark.parametrize("days", [7, 30, 90])
def test_each_history_window_includes_lower_boundary_but_excludes_cutoff(days):
    cutoff = pd.Timestamp("2011-06-06")
    lower = cutoff - pd.Timedelta(days=days)
    rows = [
        transaction("1", "2010-12-01 09:00"),
        transaction("1", (lower - pd.Timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M"), 1000),
        transaction("1", lower.strftime("%Y-%m-%d %H:%M"), 3),
        transaction("1", (cutoff - pd.Timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M"), -1),
        transaction("1", cutoff.strftime("%Y-%m-%d %H:%M"), 1000),
    ]
    data = pilot.prepare_data(pd.DataFrame(rows), pilot.PilotConfig())
    features = data.features["expanded_pit"].loc[data.queries.cutoff == cutoff].iloc[0]
    assert features[f"spend_{days}d_count"] == 2
    assert features[f"spend_{days}d_sum"] == 4
    assert features[f"spend_{days}d_last"] == -2


def test_future_perturbations_change_targets_but_not_features_or_cohort(raw):
    before = pilot.prepare_data(raw, pilot.PilotConfig(customers=2))
    cutoff = pd.Timestamp("2011-06-06")
    modified = raw.copy()
    future = pd.to_datetime(modified.InvoiceDate) >= cutoff
    modified.loc[future, "Quantity"] *= -100
    modified = pd.concat([modified, pd.DataFrame([transaction("999", "2011-06-06 00:00", 10**6)])], ignore_index=True)
    after = pilot.prepare_data(modified, pilot.PilotConfig(customers=2))
    assert before.provenance["cohort_sha256"] == after.provenance["cohort_sha256"]
    eligible = before.queries.cutoff <= cutoff
    for method in pilot.METHODS:
        pd.testing.assert_frame_equal(before.features[method].loc[eligible], after.features[method].loc[eligible])
    assert not before.target.loc[before.queries.cutoff == cutoff].equals(
        after.target.loc[after.queries.cutoff == cutoff]
    )


def test_every_aggregation_matches_strict_past_only_windows_and_schema(raw, prepared):
    cleaned, _ = pilot.clean_transactions(raw)
    features = prepared.features["expanded_pit"]
    forbidden = {"CustomerID", "InvoiceDate", "cutoff", "label_end", "next_7d_net_spend"}
    for frame in prepared.features.values():
        assert not forbidden.intersection(frame.columns)
        assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes)
    assert prepared.features["simple_history"].shape[1] == 6
    assert features.shape[1] == 22
    for position, query in prepared.queries.iterrows():
        history = cleaned[(cleaned.CustomerID == query.CustomerID) & (cleaned.InvoiceDate < query.cutoff)]
        assert (
            features.recency_days.iloc[position] == (query.cutoff - history.InvoiceDate.max()).total_seconds() / 86400
        )
        assert features.tenure_days.iloc[position] == (query.cutoff - history.InvoiceDate.min()).total_seconds() / 86400
        for days in (7, 30, 90):
            values = history.loc[history.InvoiceDate >= query.cutoff - pd.Timedelta(days=days), "net_spend"].to_numpy()
            expected = {
                "count": len(values),
                "sum": values.sum(),
                "mean": values.mean() if len(values) else np.nan,
                "std": values.std(ddof=0) if len(values) else np.nan,
                "max": values.max() if len(values) else np.nan,
                "last": values[-1] if len(values) else np.nan,
            }
            for operation, value in expected.items():
                assert features[f"spend_{days}d_{operation}"].iloc[position] == pytest.approx(value, nan_ok=True)


def test_outer_and_inner_folds_repeat_entities_but_purge_time_and_label_overlap(prepared):
    q = prepared.queries
    folds = pilot.make_folds(prepared)
    assert len(folds) == 3
    for fold in folds:
        assert set(fold.inner_train).issubset(fold.train)
        assert set(fold.inner_valid).issubset(fold.train)
        for train, valid in ((fold.train, fold.test), (fold.inner_train, fold.inner_valid)):
            assert set(q.CustomerID.iloc[train]) == set(q.CustomerID.iloc[valid])
            assert q.label_end.iloc[train].max() < q.cutoff.iloc[valid].min()
            assert not set(q.cutoff.iloc[train]).intersection(q.cutoff.iloc[valid])
            assert q.iloc[train].groupby("cutoff").size().eq(2).all()
            assert q.iloc[valid].groupby("cutoff").size().eq(2).all()
            equality = q.index[q.label_end == q.cutoff.iloc[valid].min()]
            assert not set(equality).intersection(train)
    assert pilot.fold_provenance(prepared, folds) == pilot.fold_provenance(prepared, pilot.make_folds(prepared))


@pytest.mark.parametrize("method", pilot.METHODS)
@pytest.mark.parametrize("learner", pilot.LEARNERS)
def test_adapter_hpo_and_model_boundaries_using_only_fake_models(monkeypatch, prepared, tmp_path, method, learner):
    fold = pilot.make_folds(prepared)[0]
    created = []
    fits = []
    scored = []
    predicted = []
    regression_metrics = tabular.regression_metrics

    class FakeModel:
        def fit(self, X, y):
            fits.append((X.copy(), y.copy()))
            return self

        def predict(self, X):
            predicted.append(X.copy())
            return np.zeros(len(X))

    def make_model(X, task, actual_learner, parameters, config):
        assert task == "regression" and actual_learner == learner
        created.append(X.index.to_numpy())
        return FakeModel()

    def score_predictions(prediction, y):
        scored.append((predicted[-1], y.copy()))
        return regression_metrics(prediction, y)

    monkeypatch.setattr(tabular, "make_model", make_model)
    monkeypatch.setattr(tabular, "regression_metrics", score_predictions)
    payload = pilot.cell_payload(prepared, fold, 0, method, learner, pilot.PilotConfig(customers=2))
    rows = tabular.run_case(payload, tmp_path)
    assert len(rows) == 1 and rows[0]["status"] == "success"
    assert len(fits) == len(created) == len(predicted) == len(scored) == 4
    for number, ((fit_X, fit_y), (score_X, score_y)) in enumerate(zip(fits, scored, strict=True)):
        train = fold.inner_train if number < 3 else fold.train
        valid = fold.inner_valid if number < 3 else fold.test
        assert fit_X.index.tolist() == fit_y.index.tolist() == train.tolist()
        assert score_X.index.tolist() == score_y.index.tolist() == valid.tolist()
        assert fit_X.columns.tolist() == score_X.columns.tolist() == prepared.features[method].columns.tolist()
        assert prepared.queries.label_end.iloc[train].max() < prepared.queries.cutoff.iloc[valid].min()


def test_adapter_imputation_is_fitted_on_each_training_boundary(monkeypatch, prepared, tmp_path):
    folds = pilot.make_folds(prepared)
    fold = folds[0]
    X = prepared.features["simple_history"].copy()
    X["fixture_missing"] = np.nan
    X.loc[fold.inner_train, "fixture_missing"] = 2.0
    X.loc[fold.test, "fixture_missing"] = 1000000.0
    changed = replace(prepared, features={**prepared.features, "simple_history": X})
    calls = []

    class FakeModel:
        def fit(self, X, y):
            return self

        def predict(self, X):
            calls.append(X.copy())
            return np.zeros(len(X))

    monkeypatch.setattr(tabular, "make_model", lambda *args: FakeModel())
    tabular.run_case(
        pilot.cell_payload(changed, fold, 0, "simple_history", "linear", pilot.PilotConfig(customers=2)), tmp_path
    )
    assert len(calls) == 4
    for validation in calls[:3]:
        assert validation.fixture_missing.eq(2).all()


@pytest.mark.parametrize(
    "values",
    [
        {"threads": 9},
        {"customers": 201},
        {"total_seconds": 7201},
        {"case_timeout": 481},
        {"total_seconds": float("nan")},
    ],
)
def test_hard_resource_ceiling_validation(values):
    with pytest.raises(ValidationError):
        pilot.PilotConfig(**values)


def test_cached_original_bytes_and_rights_hashes_are_enforced(tmp_path):
    csv = pd.DataFrame([transaction("1", "2010-12-01 09:00")]).to_csv(index=False).encode()
    evidence = {
        "raw_sha256": pilot._sha(csv),
        "rights_page_sha256": pilot._sha(b"rights"),
        "api_sha256": pilot._sha(b"metadata"),
        "provider_rows": 1,
    }
    (tmp_path / "data.csv").write_bytes(csv)
    (tmp_path / "rights.html").write_bytes(b"rights")
    (tmp_path / "metadata.json").write_bytes(b"metadata")
    (tmp_path / "source.json").write_text(json.dumps(evidence))
    frame, source = pilot.acquire_source(tmp_path)
    assert len(frame) == 1 and source == evidence
    (tmp_path / "data.csv").write_bytes(csv + b"\n")
    with pytest.raises(ValueError, match="changed"):
        pilot.acquire_source(tmp_path)


def test_protocol_is_row_free_fixed_and_discloses_scope():
    document = pilot.protocol(pilot.PilotConfig())
    assert document["expected_cells"] == 12
    assert document["max_model_fits"] == 48
    assert document["role"] == "development"
    assert "not unseen-entity" in document["entity_scope"]
    assert "no learned" in document["method_description"]
    assert document["candidate_parameters"]["linear"] == tabular.candidate_parameters("linear")
    assert len(pilot._planned_cells("pending")) == 12


def test_prepare_never_invokes_models_or_isolated_runner(monkeypatch, raw, tmp_path):
    import psutil

    monkeypatch.setattr(psutil, "Process", lambda: SimpleNamespace(cpu_affinity=lambda values=None: [0, 1]))
    monkeypatch.setattr(pilot, "acquire_source", lambda: (raw, {"raw_sha256": "fixture"}))
    monkeypatch.setattr(pilot, "_registration", lambda *args: {"frozen": "fixture"})
    monkeypatch.setattr(tabular, "run_isolated", lambda *args: pytest.fail("Preparation invoked models"))
    pilot.atomic_json(
        tmp_path / "report.json", {"status": "preparing", "cells": pilot._planned_cells("not_run_preparation")}
    )
    assert pilot._execute(tmp_path, pilot.PilotConfig(customers=2), False, None) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "prepared"
    assert all(cell["status"] == "not_run_preparation" for cell in report["cells"])


def test_changed_registration_blocks_all_models_and_persists_failed_cells(monkeypatch, raw, tmp_path):
    import psutil

    monkeypatch.setattr(psutil, "Process", lambda: SimpleNamespace(cpu_affinity=lambda values=None: [0]))
    monkeypatch.setattr(pilot, "acquire_source", lambda: (raw, {"raw_sha256": "fixture"}))
    monkeypatch.setattr(pilot, "_registration", lambda *args: {"version": 2})
    monkeypatch.setattr(tabular, "run_isolated", lambda *args: pytest.fail("Changed source invoked models"))
    pilot.atomic_json(tmp_path / "registration.json", {"version": 1})
    pilot.atomic_json(tmp_path / "report.json", {"status": "starting", "cells": pilot._planned_cells("pending")})
    sha = pilot._sha((tmp_path / "registration.json").read_bytes())
    assert pilot._execute(tmp_path, pilot.PilotConfig(customers=2), True, sha) == 1
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "error"
    assert all(cell["status"] == "error" for cell in report["cells"])


def test_failed_cells_are_not_replaced_by_baseline_scores(monkeypatch, raw, tmp_path):
    import psutil

    monkeypatch.setattr(psutil, "Process", lambda: SimpleNamespace(cpu_affinity=lambda values=None: [0]))
    monkeypatch.setattr(pilot, "acquire_source", lambda: (raw, {"raw_sha256": "fixture"}))
    monkeypatch.setattr(pilot, "_registration", lambda *args: {"version": 1})
    timeouts = []

    def isolated(payload, directory, timeout):
        timeouts.append(timeout)
        return [{"status": "timeout", "learner": payload["config"]["learners"][0], "fold": payload["fold"]}]

    monkeypatch.setattr(tabular, "run_isolated", isolated)
    pilot.atomic_json(tmp_path / "registration.json", {"version": 1})
    pilot.atomic_json(tmp_path / "report.json", {"status": "starting", "cells": pilot._planned_cells("pending")})
    sha = pilot._sha((tmp_path / "registration.json").read_bytes())
    assert pilot._execute(tmp_path, pilot.PilotConfig(customers=2), True, sha) == 1
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "incomplete"
    assert len(timeouts) == 12 and max(timeouts) <= 480
    assert all(cell["status"] == "timeout" and "loss" not in cell for cell in report["cells"])
    assert {cell["method"] for cell in report["cells"]} == set(pilot.METHODS)


def test_cli_supervisor_terminates_worker_tree_at_overall_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    waits = []
    stopped = []

    class FakeProcess:
        def wait(self, timeout):
            waits.append(timeout)
            raise subprocess.TimeoutExpired("fixture", timeout)

    process = FakeProcess()
    monkeypatch.setattr(pilot.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(tabular, "stop_process_tree", lambda value: stopped.append(value))
    output = tmp_path / "pilot"
    assert pilot.main(["--prepare-only", "--output", str(output), "--total-seconds", "15"]) == 1
    assert waits == [10]
    assert stopped == [process]
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "total_budget_exhausted"
    assert (output / "protocol.json").exists()
    assert not (output / "registration.json").exists()


def test_cli_requires_explicit_frozen_authorization_before_evaluation(tmp_path):
    with pytest.raises(SystemExit):
        pilot.main(["--evaluate", "--output", str(Path(pilot.ROOT) / "benchmarks" / ".cache" / "unused-fixture")])


def test_cli_charges_preparation_time_against_evaluation_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    config = pilot.PilotConfig(total_seconds=15)
    pilot.atomic_json(tmp_path / "registration.json", {"protocol": pilot.protocol(config)})
    sha = pilot._sha((tmp_path / "registration.json").read_bytes())
    pilot.atomic_json(
        tmp_path / "report.json",
        {"status": "prepared", "supervised_wall_seconds": 7, "cells": pilot._planned_cells("not_run_preparation")},
    )
    waits = []

    class FakeProcess:
        def wait(self, timeout):
            waits.append(timeout)
            raise subprocess.TimeoutExpired("fixture", timeout)

    monkeypatch.setattr(pilot.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(tabular, "stop_process_tree", lambda value: None)
    assert (
        pilot.main(["--evaluate", "--output", str(tmp_path), "--total-seconds", "15", "--registration-sha256", sha])
        == 1
    )
    assert waits == [3]
    report = json.loads((tmp_path / "report.json").read_bytes())
    assert report["preparation_wall_seconds"] == 7
    assert all(cell["status"] == "total_budget_exhausted" for cell in report["cells"])
