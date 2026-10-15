"""Synthetic-only tests for the TRAIN-only canonical instrumentation worker."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from benchmarks import regression_training_audit as audit
from benchmarks import tabular
from benchmarks.execution import split_fingerprint
from featcopilot.utils.numeric_clipping import NumericQuantileClipper


def _data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(781)
    X = pd.DataFrame({"a": rng.normal(size=80), "b": rng.normal(size=80)}, index=np.arange(80) * 7 + 101)
    y = pd.Series(2 * X.a - X.b + 0.2 * X.a**2 + rng.normal(scale=0.1, size=len(X)), index=X.index)
    return X, y


def _payload() -> dict:
    X, y = _data()
    return {
        "X": X.iloc[:60].copy(),
        "y": y.iloc[:60].copy(),
        "dataset": "synthetic-audit",
        "source": "synthetic",
        "task": "regression",
        "method": audit.METHOD,
        "seed": 42,
        "fold": 0,
        "split_sha256": split_fingerprint(np.arange(60), np.arange(60, 80)),
        "config": {
            "seed": 42,
            "threads": 1,
            "learners": ["linear"],
            "max_candidates": 12,
            "max_added_features": 4,
            "stability_repeats": 1,
            "numeric_clip_quantiles": [0.0, 1.0],
            "categorical_encoding": "one_hot",
            "target_transform": "identity",
        },
        "provenance": {"split_policy": "official", "outer_train_rows": 60, "outer_test_rows": 20},
        "package_versions": {"scikit-learn": importlib.metadata.version("scikit-learn")},
    }


def _direct(case: dict, method: str, workspace: Path) -> dict:
    X, y = _data()
    current = {
        **{key: value for key, value in case.items() if key not in ("provenance", "package_versions")},
        "X": X,
        "y": y,
        "method": method,
        "train_idx": np.arange(60),
        "test_idx": np.arange(60, 80),
    }
    return tabular.run_case(current, workspace)[0]


@pytest.mark.parametrize("seed", [42, 137, 2029])
def test_audit_replays_canonical_inner_losses_parameters_and_final_names(seed: int, tmp_path: Path) -> None:
    case = _payload()
    case["seed"] = case["config"]["seed"] = seed
    references = {}
    originals = {}
    for role, method in audit.COMPONENTS:
        original = _direct(case, method, tmp_path)
        originals[role] = original
        references[role] = {
            key: original[key]
            for key in ("parameters", "inner_losses", "outer_feature_names_sha256", "n_output_features")
        }
    case["expected_components"] = references
    rows = audit.run_case(case, tmp_path)
    assert len(rows) == 1 and rows[0]["status"] == "success"
    assert "loss" not in rows[0] and "r2" not in rows[0] and "mae" not in rows[0]
    report = rows[0]["audit"]
    assert report["reference_fields_verified"] == {role: sorted(audit._REFERENCE_KEYS) for role in references}
    for role, _ in audit.COMPONENTS:
        component = report["components"][role]
        for key, expected in references[role].items():
            assert component[key] == expected
        assert component["full_train"]["feature_names_sha256"] == originals[role]["outer_feature_names_sha256"]
        assert component["full_train"]["n_output_features"] == originals[role]["n_output_features"]
        assert component["winning_trial"] == np.argmin(originals[role]["inner_losses"])
        assert component["full_train"]["fit_rows"] == 60
        assert component["winning_inner"]["fit_rows"] == 45
        assert component["inner_validation_rows"] == 15
        assert component["prediction_pipeline_fit_count"] == 4
        names = component["full_train"]["feature_names"]
        assert component["full_train"]["feature_names_sha256"] == tabular.digest(json.dumps(names).encode())


def test_eight_pipeline_fits_and_no_outer_rows_or_outer_scoring(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = _payload()
    allowed = set(case["X"].index)
    fitting = []
    generating = []
    scoring = []
    models = []
    original_fit = Pipeline.fit
    original_generate = tabular.generate_features
    original_metrics = tabular.metrics

    def fit(self, X, y, **kwargs):
        if self.steps[-1][0] == "ridge":
            assert set(X.index) <= allowed and X.index.equals(y.index)
            fitting.append(len(X))
            models.append(self)
        return original_fit(self, X, y, **kwargs)

    def generate(train, other, y, method, task, config, workspace):
        assert set(train.index) <= allowed and set(other.index) <= allowed
        assert train.index.equals(y.index)
        assert config["numeric_clip_quantiles"] == (0.0, 1.0)
        assert config["selection_learner"] == "linear"
        generating.append((len(train), len(other), method, copy.deepcopy(config)))
        return original_generate(train, other, y, method, task, config, workspace)

    def metrics(model, X, y, task):
        assert set(X.index) <= allowed
        scoring.append(len(X))
        return original_metrics(model, X, y, task)

    # FE uses its own internal pipelines. Count canonical matched models via
    # the factory rather than conflating internal selection learners.
    original_make = tabular.make_model
    canonical_models = []

    def make(*args, **kwargs):
        result = original_make(*args, **kwargs)
        canonical_models.append(result)
        return result

    monkeypatch.setattr(Pipeline, "fit", fit)
    monkeypatch.setattr(tabular, "generate_features", generate)
    monkeypatch.setattr(tabular, "metrics", metrics)
    monkeypatch.setattr(tabular, "make_model", make)
    row = audit.run_case(case, tmp_path)[0]
    fitted_canonical = [model for model in canonical_models if any(model is fitted for fitted in models)]
    assert len(fitted_canonical) == 8
    assert row["audit"]["prediction_pipeline_fit_count"] == 8
    assert scoring == [15] * 6
    assert [(a, b) for a, b, _, _ in generating] == [(45, 15), (60, 1), (45, 15), (45, 15), (45, 15), (60, 1)]
    assert row["audit"]["components"]["raw"]["selected_name_turnover"] == {
        "common_count": 0,
        "removed": [],
        "added": [],
        "ordered_names_equal": True,
    }
    assert case["X"].index.tolist() == sorted(allowed)


@pytest.mark.parametrize(
    "matrix,rank,status",
    [
        (np.eye(3), 3, "full_column_rank"),
        (np.array([[1, 2], [2, 4], [3, 6]], dtype=float), 1, "rank_deficient"),
        (np.ones((4, 2)), 1, "rank_deficient"),
        (np.zeros((3, 2)), 0, "rank_deficient"),
        (np.array([[1.0, 0, 0], [0, 1, 0]]), 2, "rank_deficient"),
    ],
)
def test_design_rank_and_finite_json_condition_status(matrix: np.ndarray, rank: int, status: str) -> None:
    result = audit._matrix_diagnostics(matrix)
    assert result["rank"] == rank
    assert result["condition_status"] == status
    assert result["rows"] == matrix.shape[0] and result["columns"] == matrix.shape[1]
    assert result["condition_number"] == (1.0 if status == "full_column_rank" else None)
    assert "singular_values" not in result
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("matrix", [np.array([[np.nan]]), np.empty((0, 2)), np.array([1, 2])])
def test_design_invalid_values_raise(matrix: np.ndarray) -> None:
    with pytest.raises(ValueError, match="design"):
        audit._matrix_diagnostics(matrix)


def test_exact_training_clipping_counts_exclude_constant_undefined_boolean() -> None:
    X = pd.DataFrame(
        {
            "varying": [0.0, 0.0, 1.0, 2.0, np.nan],
            "constant": [7.0, 7.0, 7.0, np.nan, 7.0],
            "missing": [np.nan] * 5,
            "guarded": [-1e40, -2.0, 1.0, 2.0, 1e40],
            "flag": [True, False, True, False, True],
            "category": ["SECRET-A", "SECRET-B", "SECRET-A", "SECRET-B", "SECRET-A"],
        }
    )
    clipper = NumericQuantileClipper().fit(X)
    report = audit._clipping_diagnostics(clipper, X)
    assert report["eligible_finite_count"] == 9
    assert report["lower_count"] == 3
    assert report["upper_count"] == 2
    assert report["saturated_count"] == 5
    assert report["saturated_fraction"] == 5 / 9
    assert report["constant_columns"] == ["constant"]
    assert report["undefined_columns"] == ["missing"]
    assert report["excluded_boolean_columns"] == ["flag"]
    records = report["columns"]
    assert isinstance(records, list)
    assert records[1]["status"] == "constant" and records[1]["saturated_count"] is None
    assert records[2]["status"] == "undefined" and records[2]["saturated_fraction"] is None
    assert "SECRET" not in json.dumps(report)
    assert "bounds" not in json.dumps(report)


def test_all_excluded_clipping_is_explicitly_undefined() -> None:
    X = pd.DataFrame({"constant": [2.0, 2.0], "missing": [np.nan, np.nan]})
    report = audit._clipping_diagnostics(NumericQuantileClipper().fit(X), X)
    assert report["eligible_finite_count"] == 0
    assert report["saturated_fraction"] is None


def test_pipeline_diagnostics_match_training_design_coefficient_norms_and_residuals() -> None:
    X, y = _data()
    X["duplicate"] = 2 * X["a"]
    model = tabular.make_model(
        X,
        "regression",
        "linear",
        {"regularization": 1.0},
        {"seed": 42, "threads": 1, "numeric_clip_quantiles": (0.0, 1.0)},
    ).fit(X, y)
    result = audit._fit_diagnostics(model, X, y, ["a", "b"])
    expected = model[:-1].transform(X)
    singular = np.linalg.svd(expected, compute_uv=False)
    design = result["scaled_training_design"]
    assert isinstance(design, dict)
    assert design["rank"] == 2
    assert design["condition_number"] is None
    assert design["singular_value_min"] == float(singular[-1])
    assert design["singular_value_max"] == float(singular[0])
    coef = model.steps[-1][1].coef_
    assert result["coefficient_norms"] == {"l2": np.linalg.norm(coef), "linf": np.abs(coef).max()}
    assert result["training_residual_rmse"] == tabular.metrics(model, X, y, "regression")["loss"]
    assert result["selected_features"] == ["duplicate"]


def test_no_vocabulary_rows_coefficients_design_or_residual_vectors_in_json(tmp_path: Path) -> None:
    case = _payload()
    case["X"]["category"] = ["PRIVATE-ONE", "PRIVATE-TWO", None] * 20
    original = case["X"].copy(deep=True)
    result = audit.run_case(case, tmp_path)
    serialized = json.dumps(result, allow_nan=False)
    assert "PRIVATE-ONE" not in serialized and "PRIVATE-TWO" not in serialized
    assert '"loss":' in serialized  # Inner HPO scalar trials are permitted.
    assert not any(key in result[0] for key in ("loss", "r2", "mae", "test_X", "test_y"))
    assert "coefficients" not in serialized and "residuals" not in serialized
    assert "categories_" not in serialized
    pd.testing.assert_frame_equal(case["X"], original)


@pytest.mark.parametrize(
    "scope,key,value",
    [
        ("payload", "test_X", pd.DataFrame({"x": [1]})),
        ("payload", "test_y", pd.Series([1])),
        ("payload", "test_idx", np.arange(3)),
        ("payload", "data_path", "mixed-original-dataset.pkl"),
        ("payload", "groups", [1]),
        ("payload", "task", "classification"),
        ("payload", "seed", 137),
        ("payload", "fold", -1),
        ("payload", "split_sha256", "invalid"),
        ("config", "learners", ["lightgbm"]),
        ("config", "groups", [1]),
        ("config", "timestamps", [1]),
        ("config", "label_end_times", [1]),
        ("config", "target_transform", "log1p"),
        ("config", "categorical_encoding", "target"),
        ("config", "numeric_clip_quantiles", [0.01, 0.99]),
        ("config", "threads", 9),
        ("config", "threads", True),
        ("config", "max_added_features", 51),
        ("config", "max_candidates", 1001),
        ("config", "stability_repeats", 4),
        ("config", "test_rows", [1]),
        ("payload", "provenance", {"test_predictions": [1]}),
        ("payload", "provenance", {"test_idx": [1]}),
        ("payload", "package_versions", {"scikit-learn": "0.0.0"}),
        ("payload", "expected_components", {"other": {}}),
        ("payload", "expected_components", {"raw": {"test_loss": 3.0}}),
        ("payload", "expected_components", {"raw": {"inner_losses": [1.0]}}),
        ("payload", "expected_components", {"raw": {"inner_losses": [1.0, np.nan, 3.0]}}),
        ("payload", "expected_components", {"raw": {"parameters": {"regularization": np.array([1])}}}),
        ("payload", "expected_components", {"raw": {"outer_feature_names_sha256": "bad"}}),
        ("payload", "expected_components", {"raw": {"n_output_features": 0}}),
        ("payload", "training_rows", 59),
        ("payload", "original_train_indices_sha256", "bad"),
        ("config", "screening", "importance"),
        ("config", "memory_limit_mb", 4096.0),
    ],
)
def test_input_firewall_rejects_before_fitting(
    scope: str, key: str, value: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = _payload()
    (case if scope == "payload" else case["config"])[key] = value
    generate = Mock(side_effect=AssertionError("Must reject before fitting"))
    monkeypatch.setattr(tabular, "generate_features", generate)
    with pytest.raises((ValueError, TypeError)):
        audit.run_case(case, tmp_path)
    generate.assert_not_called()


@pytest.mark.parametrize("kind", ["index", "nan", "constant", "target2d", "features", "attrs"])
def test_invalid_training_inputs_rejected(kind: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    case = _payload()
    if kind == "index":
        case["y"] = case["y"].reset_index(drop=True)
    elif kind == "nan":
        case["y"].iloc[0] = np.nan
    elif kind == "constant":
        case["y"][:] = 1
    elif kind == "target2d":
        case["y"] = case["y"].to_frame()
    elif kind == "features":
        case["X"] = case["X"].to_numpy()
    else:
        case["X"].attrs["test_rows"] = np.array([1, 2])
    generate = Mock()
    monkeypatch.setattr(tabular, "generate_features", generate)
    with pytest.raises(ValueError):
        audit.run_case(case, tmp_path)
    generate.assert_not_called()


def test_train_only_artifact_sha_and_payload_schema(tmp_path: Path) -> None:
    case = _payload()
    content = pickle.dumps({"X": case.pop("X"), "y": case.pop("y")})
    path = tmp_path / "train-only.pkl"
    path.write_bytes(content)
    case["data_path"] = str(path)
    case["prepared_sha256"] = tabular.digest(content)
    row = audit.run_case(case, tmp_path)[0]
    assert row["audit"]["train_data_sha256"] == tabular.digest(content)
    path.write_bytes(content + b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        audit.run_case(case, tmp_path)
    bad = pickle.dumps({"X": pd.DataFrame({"x": [1]}), "y": pd.Series([1]), "test_y": [2]})
    path.write_bytes(bad)
    case["prepared_sha256"] = tabular.digest(bad)
    with pytest.raises(ValueError, match="exactly X and y"):
        audit.run_case(case, tmp_path)


@pytest.mark.parametrize(
    "field,expected",
    [
        ("parameters", {"regularization": 999}),
        ("inner_losses", [1.0, 2.0, 3.0]),
        ("outer_feature_names_sha256", "0" * 64),
        ("n_output_features", 999),
    ],
)
def test_reference_mismatch_is_not_success(field: str, expected: object, tmp_path: Path) -> None:
    case = _payload()
    case["expected_components"] = {"raw": {field: expected}}
    with pytest.raises(ValueError, match=f"raw.{field}"):
        audit.run_case(case, tmp_path)


def test_worker_failure_has_logging_and_standard_failure_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = _payload()
    source, output = tmp_path / "input.pkl", tmp_path / "output.json"
    source.write_bytes(pickle.dumps(case))
    failed = Mock(side_effect=RuntimeError("SVD audit failure"))
    logged = Mock()
    monkeypatch.setattr(audit, "run_case", failed)
    monkeypatch.setattr(audit.logger, "exception", logged)
    assert audit.main(["--worker", str(source), "--worker-output", str(output)]) == 0
    row = json.loads(output.read_text())[0]
    assert row["status"] == "error" and "SVD audit failure" in row["error"]
    assert "audit" not in row and "loss" not in row
    assert row["method"] == audit.METHOD and row["learner"] == "linear"
    logged.assert_called_once()


def test_subprocess_worker_cli_synthetic_only(tmp_path: Path) -> None:
    case = _payload()
    train_content = pickle.dumps({"X": case.pop("X"), "y": case.pop("y")})
    train_path = tmp_path / "train-only.pkl"
    train_path.write_bytes(train_content)
    case.update(
        data_path=str(train_path),
        prepared_sha256=tabular.digest(train_content),
        original_train_indices_sha256=tabular.digest(np.arange(60, dtype="<i8").tobytes()),
        training_rows=60,
        reference_run="synthetic-reference",
        role="development",
        expected_cells=30,
    )
    case["config"].update(memory_limit_mb=4096, screening="auto")
    source, output = tmp_path / "input.pkl", tmp_path / "output.json"
    source.write_bytes(pickle.dumps(case))
    environment = os.environ.copy()
    root = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = root + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.regression_training_audit",
            "--worker",
            str(source),
            "--worker-output",
            str(output),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    row = json.loads(output.read_text())[0]
    assert row["status"] == "success", row
    assert row["audit"]["prediction_pipeline_fit_count"] == 8
    assert row["method"] == "regression-training-audit"
    assert row["audit"]["provenance"]["training_rows"] == 60
    assert row["audit"]["provenance"]["reference_run"] == "synthetic-reference"
    assert row["audit"]["train_data_sha256"] == case["prepared_sha256"]
    assert "loss" not in row
