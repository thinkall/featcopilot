"""Synthetic-only checks for audit isolation and aggregate-only reporting."""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import training_selection_audit as audit
from sklearn.model_selection import KFold


def test_outer_training_isolation_precedes_any_value_processing():
    X = pd.DataFrame({"x": np.arange(60, dtype=float)})
    y = pd.Series(np.arange(60, dtype=float))
    partitions = list(KFold(3, shuffle=True, random_state=2027).split(np.empty((60, 0))))
    hashes = [audit.split_hash(train, test) for train, test in partitions]
    train, excluded = partitions[1]
    X.loc[excluded, "x"] = np.inf
    y.loc[excluded] = np.nan
    isolated_X, isolated_y, identity = audit.isolate_outer_train({"X": X, "y": y}, 1, hashes)
    assert isolated_X.index.tolist() == train.tolist()
    assert isolated_y.index.tolist() == train.tolist()
    assert np.isfinite(isolated_X.to_numpy()).all()
    assert isolated_y.notna().all()
    assert identity["outer_split_sha256"] == hashes[1]
    with pytest.raises(ValueError, match="fingerprint"):
        audit.isolate_outer_train({"X": X, "y": y}, 1, ["wrong"] * 3)


def test_nested_index_split_has_fixed_disjoint_boundaries():
    train, holdout, inner_train, inner_valid = audit.nested_partitions(139)
    assert set(train).isdisjoint(holdout)
    assert set(inner_train).isdisjoint(inner_valid)
    assert set(inner_train) | set(inner_valid) == set(train)
    assert set(train) | set(holdout) == set(range(139))
    assert len(holdout) == 35
    for first, second in zip(audit.nested_partitions(139), audit.nested_partitions(139), strict=True):
        np.testing.assert_array_equal(first, second)


def test_fit_guard_rejects_audit_rows_and_does_not_publish_indices():
    guard = audit.FitAudit()
    frame = pd.DataFrame({"x": [1, 2]}, index=[7, 9])
    guard.check(frame, {7, 9}, "valid")
    assert set(guard.records[0]) == {"stage", "rows", "row_positions_sha256"}
    with pytest.raises(AssertionError, match="boundary"):
        guard.check(frame, {7}, "audit-row-leak")


def test_full_partition_flow_never_fits_audit_rows_or_tunes_on_audit_labels(tmp_path):
    from benchmarks import tabular

    X = pd.DataFrame({"x": np.linspace(-1, 1, 80)})
    X.index = np.arange(80) * 5 + 2
    train, held, inner_train, inner_valid = audit.nested_partitions(len(X))
    target = pd.Series(2 * X["x"], index=X.index)
    target.iloc[held] = 987654.0
    fits = []
    generations = []

    class MeanModel:
        def __init__(self, alpha):
            self.alpha = alpha

        def fit(self, values, labels):
            assert not set(values.index) & set(X.index[held])
            assert not (np.asarray(labels) == 987654).any()
            fits.append((set(values.index), self.alpha))
            self.mean = float(np.mean(labels))
            return self

        def predict(self, values):
            return np.full(len(values), self.mean + self.alpha * 0.001)

    def make_model(frame, task, learner, params, config):
        return MeanModel(params["regularization"])

    def generate(train_frame, validation_frame, labels, config, guard, stage, directory):
        assert not set(train_frame.index) & set(X.index[held])
        assert not (np.asarray(labels) == 987654).any()
        expected = set(X.index[inner_train]) if stage == "inner" else set(X.index[train])
        assert set(train_frame.index) == expected
        generations.append(stage)
        guard.check(train_frame, expected, stage)
        names = ["synthetic_feature"]
        return (
            train_frame.assign(synthetic_feature=train_frame["x"] ** 2),
            validation_frame.assign(synthetic_feature=validation_frame["x"] ** 2),
            {"selected_names": names, "selected_count": 1},
        )

    def diagnostics(model, train_frame, audit_frame):
        return {"preprocessing_state_sha256": audit.schema_hash(train_frame)}

    with (
        patch.object(tabular, "make_model", side_effect=make_model),
        patch.object(audit, "generate_checked", side_effect=generate),
        patch.object(audit, "model_diagnostics", side_effect=diagnostics),
    ):
        result = audit.run_partition(X, target, {"seed": 2027, "threads": 8}, tmp_path)
    assert generations == ["inner", "final_audit_train"]
    assert len(fits) in (8, 9)
    assert all(rows <= set(X.index[train]) for rows, _ in fits)
    assert all(rows == set(X.index[inner_train]) for rows, _ in fits[:6])
    assert set(result["models"]) == {"selection_alpha_1", "inner_chosen_alpha", "raw_baseline"}
    assert result["audit_holdout_rows"] == len(held)
    assert not {"targets", "labels", "predictions", "rows"} & set(result)
    assert all("predictions" not in model for model in result["models"].values())


def test_model_diagnostics_omit_category_vocabulary_and_raw_rows():
    class Preprocessing:
        output_indices_ = {"numeric": slice(0, 1), "categorical": slice(1, 3)}
        transformers_ = [("numeric", None, ["numeric"]), ("categorical", None, ["category"])]

        def transform(self, frame):
            return np.column_stack([frame["numeric"].to_numpy(), np.ones(len(frame)), np.zeros(len(frame))])

    class Learner:
        coef_ = np.array([2.0, 3.0, 1.0])
        intercept_ = 5.0

    class Model:
        steps = [("preprocess", Preprocessing()), ("learner", Learner())]

    train = pd.DataFrame({"numeric": [-1.0, 1.0], "category": ["private-a", "private-b"]})
    holdout = pd.DataFrame({"numeric": [2.0, 3.0], "category": ["private-c", "private-d"]})
    with patch.object(audit.pickle, "dumps", return_value=b"synthetic-preprocessing-state"):
        result = audit.model_diagnostics(Model(), train, holdout)
    text = json.dumps(result)
    assert "private-" not in text
    assert "predictions" not in result
    assert [feature["feature"] for feature in result["features"]] == [
        "numeric",
        "categorical_output_1",
        "categorical_output_2",
    ]
    assert result["features"][0]["audit"]["max_abs"] == 3.0
    assert result["features"][0]["audit_contribution"]["max_abs"] == 6.0
