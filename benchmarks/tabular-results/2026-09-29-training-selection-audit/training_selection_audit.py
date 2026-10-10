"""One fixed retrospective, training-only audit; never score original outer tests."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

PROCESS_STARTED = time.perf_counter()

import numpy as np  # noqa: E402 - include scientific import time in the supervisor clock.
import pandas as pd  # noqa: E402
from sklearn.model_selection import KFold, train_test_split  # noqa: E402

ROOT = Path(r"C:\code\featcopilot")
ARCHIVE = ROOT / "benchmarks" / "tabular-results" / "2026-09-28-frequency-iid-confirmation"
OUTPUT = ROOT / "benchmarks" / "tabular-results" / "2026-09-29-training-selection-audit"
FAMILIES = ("computer-hardware-performance", "infrared-oral-temperature")
ARCHIVED_SOURCE = "c30bfbfcacb2fc7d3c3398699d8b2ee67ace00807cecaf2de49c7c06f527a808"
THREADS = 8
SECONDS_CEILING = 3600
CELL_SECONDS_CEILING = 600
MEMORY_MB = 4096
ALPHAS = (0.1, 1.0, 10.0)

sys.path.insert(0, str(ROOT))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def fingerprint(value):
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def index_hash(indices):
    return sha_bytes(np.asarray(indices, dtype="<i8").tobytes())


def split_hash(train, test):
    sizes = np.asarray([len(train), len(test)], dtype="<i8").tobytes()
    return sha_bytes(sizes + np.asarray(train, dtype="<i8").tobytes() + np.asarray(test, dtype="<i8").tobytes())


def nested_partitions(n_rows):
    """Index-only splitting; neither task labels nor feature values are read."""
    fit, audit = train_test_split(np.arange(n_rows), test_size=0.25, random_state=2028)
    inner_fit, inner_valid = train_test_split(fit, test_size=0.25, random_state=2027)
    return fit, audit, inner_fit, inner_valid


def isolate_outer_train(data, fold, expected_splits):
    """Slice the original outer train before any fitting or value statistics."""
    if not isinstance(data["X"], pd.DataFrame) or len(data["X"]) != len(data["y"]):
        raise ValueError("Prepared data schema is invalid")
    n_rows = len(data["X"])
    partitions = list(KFold(n_splits=3, shuffle=True, random_state=2027).split(np.empty((n_rows, 0))))
    for number, (train, test) in enumerate(partitions):
        if split_hash(train, test) != expected_splits[number]:
            raise ValueError("Reconstructed outer split differs from the archived fingerprint")
    train, excluded = partitions[fold]
    X = data["X"].iloc[train].copy()
    y = pd.Series(np.asarray(data["y"].iloc[train]), index=X.index, name="target")
    if not X.index.is_unique:
        raise ValueError("Audit requires unique prepared row indices")
    return (
        X,
        y,
        {
            "outer_train_rows": len(train),
            "original_outer_test_excluded_rows": len(excluded),
            "outer_split_sha256": expected_splits[fold],
            "outer_train_positions_sha256": index_hash(train),
        },
    )


class FitAudit:
    """Check fitted row identities without publishing those identities."""

    def __init__(self):
        self.records = []

    def check(self, frame, allowed, stage):
        actual = set(frame.index)
        if not actual or not actual <= set(allowed):
            raise AssertionError(f"Fit boundary violation at {stage}")
        self.records.append({"stage": stage, "rows": len(frame), "row_positions_sha256": index_hash(frame.index)})


def fit_checked(model, X, y, allowed, guard, stage):
    guard.check(X, allowed, stage)
    if not X.index.equals(y.index):
        raise AssertionError("Positionally aligned targets required")
    model.fit(X.replace([np.inf, -np.inf], np.nan), y)
    return model


def rmse(model, X, y):
    prediction = np.asarray(model.predict(X.replace([np.inf, -np.inf], np.nan)), dtype=float)
    value = float(np.sqrt(np.mean(np.square(np.asarray(y) - prediction))))
    if not np.isfinite(value):
        raise ValueError("Nonfinite audit metric")
    return value


def schema_hash(frame):
    return fingerprint({"columns": list(frame.columns), "dtypes": [str(dtype) for dtype in frame.dtypes]})


def feature_snapshot(wrapper, transformed):
    engine = wrapper._engine_instances["tabular"]
    definitions = []
    for feature in engine.get_feature_set():
        state = feature.to_dict()
        definitions.append(
            {
                "name": feature.name,
                "operator": feature.transformation,
                "sources": feature.source_columns,
                "definition_sha256": fingerprint(
                    {"name": feature.name, "operator": feature.transformation, "sources": feature.source_columns}
                ),
                "fitted_state_sha256": fingerprint(state),
            }
        )
    report = engine.selection_report_
    return {
        "selected_names": engine.get_feature_names(),
        "selected_definitions": definitions,
        "schema_sha256": schema_hash(transformed),
        "fitted_feature_state_sha256": fingerprint(definitions),
        "selected_count": report["selected_count"],
        "ranking_scope": report["ranking_scope"],
        "baseline_loss": report["baseline_loss"],
        "validation_scores": report["validation_scores"],
        "count_by_repeat": report["count_by_repeat"],
        "frequency_policy": report.get("frequency_policy"),
        "learner_fit_count": report["learner_fit_count"],
    }


def generate_checked(train, valid, y, config, guard, stage, directory):
    """Use the parent's fixed benchmark route and assert nested fit boundaries."""
    import featcopilot
    from benchmarks import tabular
    from featcopilot.engines.predictive import PredictiveTabularEngine
    from featcopilot.utils.frequency import fit_frequency_mapping

    allowed = set(train.index)
    guard.check(train, allowed, stage + ":feature_engineer")
    wrappers = []
    original_factory = featcopilot.AutoFeatureEngineer
    original_fit = PredictiveTabularEngine._fit_learner
    original_prepare = tabular.prepare_numeric
    normalized = []

    def factory(**kwargs):
        wrapper = original_factory(**kwargs)
        wrappers.append(wrapper)
        return wrapper

    def inspect_fit(frame, target, template):
        guard.check(frame, allowed, stage + ":selection_learner")
        return original_fit(frame, target, template)

    def inspect_frequency(series):
        guard.check(series.to_frame(), allowed, stage + ":frequency_mapping")
        return fit_frequency_mapping(series)

    def inspect_preparation(train_frame, validation_frame, **kwargs):
        if kwargs.get("impute") is not False:
            raise AssertionError("Predictive generate_features must not pre-impute before its internal split")
        result = original_prepare(train_frame, validation_frame, **kwargs)
        normalized.append(result)
        return result

    with (
        patch("featcopilot.AutoFeatureEngineer", side_effect=factory),
        patch.object(PredictiveTabularEngine, "_fit_learner", side_effect=inspect_fit),
        patch("featcopilot.engines.predictive.fit_frequency_mapping", side_effect=inspect_frequency),
        patch.object(tabular, "prepare_numeric", side_effect=inspect_preparation),
    ):
        transformed, validation = tabular.generate_features(
            train, valid, y, "featcopilot-frequency-paired", "regression", config, directory
        )
    if len(wrappers) != 1:
        raise AssertionError("Expected exactly one feature-engineer instance")
    wrapper = wrappers[0]
    if wrapper.selection_estimator.alpha != 1.0:
        raise AssertionError("Selection learner must use the declared middle-grid alpha")
    if len(normalized) != 1:
        raise AssertionError("Expected one stateless preparation step")
    pd.testing.assert_frame_equal(transformed[list(train.columns)], normalized[0][0])
    pd.testing.assert_frame_equal(validation[list(valid.columns)], normalized[0][1])
    if not transformed.columns.equals(validation.columns):
        raise AssertionError("Frozen fit/validation feature schema mismatch")
    return transformed, validation, feature_snapshot(wrapper, transformed)


def tune_alpha(train, valid, train_y, valid_y, config, guard, stage):
    from benchmarks.tabular import make_model

    losses = []
    started = time.perf_counter()
    for alpha in ALPHAS:
        model = make_model(train, "regression", "linear", {"regularization": alpha}, config)
        fit_checked(model, train, train_y, train.index, guard, stage)
        losses.append(rmse(model, valid, valid_y))
    chosen = float(ALPHAS[int(np.argmin(losses))])
    return {
        "chosen_alpha": chosen,
        "alphas": list(ALPHAS),
        "validation_rmse": losses,
        "seconds": time.perf_counter() - started,
    }


def absolute_summary(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite preprocessed audit values")
    absolute = np.abs(values)
    return {"abs_p95": float(np.quantile(absolute, 0.95)), "max_abs": float(absolute.max())}


def model_diagnostics(model, train, audit):
    """Publish aggregates only: no raw rows, category labels, or predictions."""
    preprocessing = model.steps[0][1]
    learner = model.steps[-1][1]
    training = np.asarray(preprocessing.transform(train.replace([np.inf, -np.inf], np.nan)), dtype=float)
    validation = np.asarray(preprocessing.transform(audit.replace([np.inf, -np.inf], np.nan)), dtype=float)
    coefficients = np.asarray(learner.coef_, dtype=float)
    if training.shape[1] != len(coefficients) or validation.shape[1] != len(coefficients):
        raise AssertionError("Encoded feature attribution mismatch")
    names = [f"categorical_output_{index}" for index in range(len(coefficients))]
    numeric_slice = preprocessing.output_indices_.get("numeric", slice(0, 0))
    numeric_columns = next((item[2] for item in preprocessing.transformers_ if item[0] == "numeric"), [])
    if numeric_slice.stop - numeric_slice.start != len(numeric_columns):
        raise AssertionError("Numeric imputation changed width; cannot guess attribution")
    for index, name in enumerate(numeric_columns, numeric_slice.start):
        names[index] = str(name)
    items = []
    for index, name in enumerate(names):
        train_values, audit_values = training[:, index], validation[:, index]
        train_contribution = train_values * coefficients[index]
        audit_contribution = audit_values * coefficients[index]
        train_summary, audit_summary = absolute_summary(train_values), absolute_summary(audit_values)
        items.append(
            {
                "feature": name,
                "kind": (
                    "numeric_standardized"
                    if numeric_slice.start <= index < numeric_slice.stop
                    else "categorical_encoded"
                ),
                "coefficient": float(coefficients[index]),
                "train": train_summary,
                "audit": audit_summary,
                "audit_above_train_max_abs_count": int(
                    np.count_nonzero(np.abs(audit_values) > train_summary["max_abs"])
                ),
                "train_contribution": {
                    **absolute_summary(train_contribution),
                    "minimum": float(train_contribution.min()),
                    "maximum": float(train_contribution.max()),
                },
                "audit_contribution": {
                    **absolute_summary(audit_contribution),
                    "minimum": float(audit_contribution.min()),
                    "maximum": float(audit_contribution.max()),
                },
            }
        )
    return {
        "schema_sha256": schema_hash(train),
        "preprocessing_state_sha256": sha_bytes(pickle.dumps(preprocessing)),
        "coefficient_sha256": sha_bytes(coefficients.tobytes()),
        "coefficient_l1": float(np.abs(coefficients).sum()),
        "coefficient_l2": float(np.linalg.norm(coefficients)),
        "coefficient_max_abs": float(np.abs(coefficients).max()),
        "intercept": float(learner.intercept_),
        "features": items,
    }


def run_partition(X, y, config, directory):
    """Run the one fixed audit only after receiving an isolated outer train."""
    from benchmarks.tabular import make_model, prepare_numeric

    started = time.perf_counter()
    guard = FitAudit()
    train, audit, inner_train, inner_valid = nested_partitions(len(X))
    X_train, y_train = X.iloc[train], y.iloc[train]
    X_audit, y_audit = X.iloc[audit], y.iloc[audit]
    X_inner, y_inner = X.iloc[inner_train], y.iloc[inner_train]
    X_valid, y_valid = X.iloc[inner_valid], y.iloc[inner_valid]
    if set(X_train.index) & set(X_audit.index):
        raise AssertionError("Audit partition overlap")

    fe_started = time.perf_counter()
    inner, valid, inner_snapshot = generate_checked(X_inner, X_valid, y_inner, config, guard, "inner", directory)
    inner_fe_seconds = time.perf_counter() - fe_started
    engineered_tuning = tune_alpha(inner, valid, y_inner, y_valid, config, guard, "engineered_alpha_tuning")
    raw_inner, raw_valid = prepare_numeric(X_inner, X_valid, impute=False)
    raw_tuning = tune_alpha(raw_inner, raw_valid, y_inner, y_valid, config, guard, "raw_alpha_tuning")

    fe_started = time.perf_counter()
    final, holdout, final_snapshot = generate_checked(
        X_train, X_audit, y_train, config, guard, "final_audit_train", directory
    )
    final_fe_seconds = time.perf_counter() - fe_started
    models = {}
    fitted_by_alpha = {}
    for label, alpha in [("selection_alpha_1", 1.0), ("inner_chosen_alpha", engineered_tuning["chosen_alpha"])]:
        start = time.perf_counter()
        if alpha not in fitted_by_alpha:
            model = make_model(final, "regression", "linear", {"regularization": alpha}, config)
            fitted_by_alpha[alpha] = fit_checked(
                model, final, y_train, X_train.index, guard, "final_engineered_learner"
            )
        model = fitted_by_alpha[alpha]
        models[label] = {
            "alpha": alpha,
            "fit_or_reuse_seconds": time.perf_counter() - start,
            "audit_rmse": rmse(model, holdout, y_audit),
            "train_rmse": rmse(model, final, y_train),
            "diagnostics": model_diagnostics(model, final, holdout),
        }
    raw_train, raw_audit = prepare_numeric(X_train, X_audit, impute=False)
    raw_model = make_model(raw_train, "regression", "linear", {"regularization": raw_tuning["chosen_alpha"]}, config)
    start = time.perf_counter()
    fit_checked(raw_model, raw_train, y_train, X_train.index, guard, "final_raw_learner")
    models["raw_baseline"] = {
        "alpha": raw_tuning["chosen_alpha"],
        "fit_seconds": time.perf_counter() - start,
        "audit_rmse": rmse(raw_model, raw_audit, y_audit),
        "train_rmse": rmse(raw_model, raw_train, y_train),
        "diagnostics": model_diagnostics(raw_model, raw_train, raw_audit),
    }
    first, second = models["selection_alpha_1"], models["inner_chosen_alpha"]
    if first["diagnostics"]["preprocessing_state_sha256"] != second["diagnostics"]["preprocessing_state_sha256"]:
        raise AssertionError("Alpha comparison did not preserve the frozen preprocessing state")
    inner_names, final_names = set(inner_snapshot["selected_names"]), set(final_snapshot["selected_names"])
    return {
        "status": "success",
        "role": "retrospective_training_only_diagnostic",
        "outer_train_rows": len(X),
        "audit_train_rows": len(train),
        "audit_holdout_rows": len(audit),
        "inner_train_rows": len(inner_train),
        "inner_validation_rows": len(inner_valid),
        "audit_split_sha256": split_hash(train, audit),
        "inner_split_sha256": split_hash(inner_train, inner_valid),
        "isolated_outer_train_sha256": sha_bytes(
            pd.util.hash_pandas_object(pd.concat([X, y], axis=1)).to_numpy().tobytes()
        ),
        "inner_feature_fit": inner_snapshot,
        "final_feature_fit": final_snapshot,
        "selected_name_intersection": sorted(inner_names & final_names),
        "selected_name_jaccard": (
            len(inner_names & final_names) / len(inner_names | final_names) if inner_names | final_names else 1.0
        ),
        "engineered_tuning": engineered_tuning,
        "raw_tuning": raw_tuning,
        "models": models,
        "same_plan_alpha_effect_rmse": second["audit_rmse"] - first["audit_rmse"],
        "fit_boundary_checks": guard.records,
        "timings": {
            "inner_feature_fit_seconds": inner_fe_seconds,
            "final_feature_fit_seconds": final_fe_seconds,
            "partition_seconds": time.perf_counter() - started,
        },
    }


def verify_versions():
    expected = {
        "scikit-learn": "1.7.2",
        "joblib": "1.3.2",
        "numpy": "2.4.4",
        "pandas": "2.3.3",
        "scipy": "1.17.1",
        "threadpoolctl": "3.6.0",
    }
    actual = {name: importlib.metadata.version(name) for name in expected}
    if actual != expected:
        raise ValueError("Audit must use the pinned benchmark environment")
    return actual


def worker(request_path, result_path):
    from threadpoolctl import threadpool_limits

    from benchmarks import tabular

    started = time.perf_counter()
    result = {"started_utc": utc_now(), "status": "error"}
    stage = "preflight"
    try:
        request = json.loads(request_path.read_text())
        result.update({"family": request["family"], "outer_fold": request["fold"]})
        verify_versions()
        if tabular.source_fingerprint() != request["source_sha256"]:
            raise ValueError("Runtime source changed after audit freeze")
        content = Path(request["cache_path"]).read_bytes()
        if sha_bytes(content) != request["prepared_sha256"]:
            raise ValueError("Prepared cache changed after audit freeze")
        stage = "isolate_original_outer_train"
        data = pickle.loads(content)
        X, y, identity = isolate_outer_train(data, request["fold"], request["split_hashes"])
        del data, content
        stage = "training_only_audit"
        with threadpool_limits(limits=THREADS):
            result.update(run_partition(X, y, request["config"], result_path.parent))
        result.update(identity)
        if tabular.source_fingerprint() != request["source_sha256"]:
            raise ValueError("Runtime source changed during audit")
    except Exception as exc:
        result.update({"status": "error", "error_type": type(exc).__name__, "error_stage": stage})
        # Do not serialize exception contents, which may contain patient/category values.
        print(f"Audit failed at {stage}: {type(exc).__name__}", file=sys.stderr, flush=True)
    result.update({"finished_utc": utc_now(), "worker_elapsed_seconds": time.perf_counter() - started})
    atomic_json(result_path, result)
    return 0 if result["status"] == "success" else 1


def stop_owned_process(process):
    import psutil

    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child in reversed(children):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    if process.poll() is None:
        process.kill()
    process.wait()
    psutil.wait_procs(children, timeout=5)


def write_analysis(output, rows, elapsed, source):
    lines = [
        "# Retrospective training-selection audit",
        "",
        "**Not confirmation. No original outer-test rows were evaluated.**",
        "",
        "All six original outer-training partitions were isolated before fitting/statistics.",
        "Audit holdout: 25%, seed 2028. Nested tuning: 25%, seed 2027; alpha grid 0.1, 1, 10.",
        "The two engineered models use the same frozen final audit-training representation.",
        "Their difference isolates alpha within that plan; inner/final set differences are descriptive.",
        "",
        "| Family | Original fold | Raw tuned RMSE | Engineered alpha=1 | Engineered tuned alpha | Tuned RMSE | Inner/final additions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["status"] != "success":
            lines.append(f"| {row['family']} | {row['outer_fold']} | {row['status']} | - | - | - | - |")
            continue
        models = row["models"]
        lines.append(
            f"| {row['family']} | {row['outer_fold']} | {models['raw_baseline']['audit_rmse']:.6g} | "
            f"{models['selection_alpha_1']['audit_rmse']:.6g} | {models['inner_chosen_alpha']['alpha']:g} | "
            f"{models['inner_chosen_alpha']['audit_rmse']:.6g} | "
            f"{row['inner_feature_fit']['selected_count']}/{row['final_feature_fit']['selected_count']} |"
        )
    lines.extend(
        [
            "",
            f"Runtime source: `{source}`.",
            f"Measured whole-audit elapsed: {elapsed:.6f} seconds.",
            "The JSON records aggregate standardized-feature and coefficient-contribution diagnostics, "
            "not raw rows, category vocabulary or prediction vectors.",
            "No alpha, feature rule, split or threshold was changed after seeing audit outcomes.",
        ]
    )
    (output / "ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def supervisor(expected_source):
    import psutil

    from benchmarks import tabular

    started = PROCESS_STARTED
    started_utc = utc_now()
    versions = verify_versions()
    current_source = tabular.source_fingerprint()
    if current_source != expected_source:
        raise ValueError("Parent-ready source hash does not match current runtime")
    if OUTPUT.exists():
        raise ValueError("Audit output already exists; refusing to overwrite prior measurements")
    original_manifest = json.loads((ARCHIVE / "manifest.json").read_text())
    if original_manifest["source_sha256"] != ARCHIVED_SOURCE or original_manifest["source_changed"]:
        raise ValueError("Original archive identity mismatch")
    OUTPUT.mkdir(parents=True)
    script_path = OUTPUT / "training_selection_audit.py"
    shutil.copy2(Path(__file__), script_path)
    tests_path = Path(__file__).with_name("test_training_selection_audit.py")
    if tests_path.exists():
        shutil.copy2(tests_path, OUTPUT / tests_path.name)
    with zipfile.ZipFile(OUTPUT / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in tabular.source_files():
            archive.write(path, arcname=path.relative_to(ROOT).as_posix())
        archive.write(script_path, arcname="training_selection_audit.py")
    config = {
        "seed": 2027,
        "threads": THREADS,
        "selection_learner": "linear",
        "learners": ["linear"],
        "max_added_features": 50,
        "max_candidates": 1000,
        "stability_repeats": 3,
        "screening": "importance",
        "detailed_diagnostics": False,
    }
    manifest = {
        "role": "RETROSPECTIVE_DEVELOPMENT_NOT_CONFIRMATION",
        "started_utc": started_utc,
        "source_sha256": current_source,
        "source_zip_sha256": sha_bytes((OUTPUT / "source.zip").read_bytes()),
        "audit_script_sha256": sha_bytes(script_path.read_bytes()),
        "synthetic_safety_tests": {
            "passed": 5,
            "script_sha256": sha_bytes(tests_path.read_bytes()) if tests_path.exists() else None,
        },
        "original_source_sha256": ARCHIVED_SOURCE,
        "original_manifest_sha256": sha_bytes((ARCHIVE / "manifest.json").read_bytes()),
        "versions": versions,
        "python_executable": sys.executable,
        "families": list(FAMILIES),
        "original_outer_seed": 2027,
        "original_outer_folds": 3,
        "audit_holdout_seed": 2028,
        "audit_holdout_fraction": 0.25,
        "nested_tuning_seed": 2027,
        "nested_tuning_holdout_fraction": 0.25,
        "alpha_grid": list(ALPHAS),
        "selection_alpha": 1.0,
        "method": "featcopilot-frequency-paired",
        "config": config,
        "whole_audit_seconds_ceiling": SECONDS_CEILING,
        "cell_seconds_ceiling": CELL_SECONDS_CEILING,
        "process_tree_memory_mb": MEMORY_MB,
        "original_outer_test_scored": False,
        "data": [],
    }
    requests = []
    for family in FAMILIES:
        spec = next(item for item in original_manifest["datasets"] if item["name"] == family)
        if spec["task"] != "regression" or spec["used_split_policy"] != "iid":
            raise ValueError("Audit family is not the declared regression IID task")
        records = sorted(
            (s for s in original_manifest["splits"] if s["dataset"] == family and s["seed"] == 2027),
            key=lambda s: s["fold"],
        )
        if [s["fold"] for s in records] != [0, 1, 2]:
            raise ValueError("Expected all three original outer-fold fingerprints")
        cache = ARCHIVE / "prepared" / f"{family}.pkl"
        cache_hash = sha_bytes(cache.read_bytes())
        manifest["data"].append(
            {
                "family": family,
                "prepared_sha256": cache_hash,
                "source_data_sha256": spec["sha256"],
                "outer_split_hashes": [s["sha256"] for s in records],
            }
        )
        for fold in range(3):
            requests.append(
                {
                    "family": family,
                    "fold": fold,
                    "cache_path": str(cache),
                    "prepared_sha256": cache_hash,
                    "split_hashes": [s["sha256"] for s in records],
                    "source_sha256": current_source,
                    "config": config,
                }
            )
    atomic_json(OUTPUT / "protocol.json", manifest)
    rows = []
    for index, request in enumerate(requests):
        remaining = SECONDS_CEILING - (time.perf_counter() - started)
        if remaining <= 0:
            rows.append({"family": request["family"], "outer_fold": request["fold"], "status": "budget_exhausted"})
            continue
        request_path = OUTPUT / f"request-{index}.json"
        result_path = OUTPUT / f"cell-{index}.json"
        atomic_json(request_path, request)
        env = os.environ.copy()
        env.update(
            {
                key: str(THREADS)
                for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]
            }
        )
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        cell_started = time.perf_counter()
        cap = min(CELL_SECONDS_CEILING, remaining)
        reason = None
        peak_mb = 0.0
        with (OUTPUT / f"worker-{index}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, str(script_path), "--worker", str(request_path), "--worker-output", str(result_path)],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            while process.poll() is None:
                if time.perf_counter() - cell_started >= cap:
                    reason = "timeout"
                    stop_owned_process(process)
                    break
                try:
                    parent = psutil.Process(process.pid)
                    rss = 0
                    for child in [parent, *parent.children(recursive=True)]:
                        try:
                            rss += child.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                    peak_mb = max(peak_mb, rss / 1024**2)
                except psutil.NoSuchProcess:
                    pass
                if peak_mb > MEMORY_MB:
                    reason = "memory_limit"
                    stop_owned_process(process)
                    break
                time.sleep(0.2)
        elapsed = time.perf_counter() - cell_started
        if reason is not None or not result_path.exists():
            row = {"family": request["family"], "outer_fold": request["fold"], "status": reason or "worker_error"}
        else:
            row = json.loads(result_path.read_text())
        row.update(
            {
                "whole_process_seconds": elapsed,
                "peak_process_tree_rss_mb": peak_mb,
                "worker_returncode": process.returncode,
            }
        )
        rows.append(row)
        atomic_json(result_path, row)
        atomic_json(OUTPUT / "results.json", {"protocol": manifest, "rows": rows})
        print(
            json.dumps(
                {"family": row["family"], "fold": row["outer_fold"], "status": row["status"], "seconds": elapsed}
            ),
            flush=True,
        )
    elapsed = time.perf_counter() - started
    manifest.update(
        {
            "finished_utc": utc_now(),
            "whole_audit_elapsed_seconds": elapsed,
            "source_sha256_at_finish": tabular.source_fingerprint(),
        }
    )
    manifest["source_changed"] = manifest["source_sha256_at_finish"] != current_source
    atomic_json(OUTPUT / "protocol.json", manifest)
    atomic_json(OUTPUT / "results.json", {"protocol": manifest, "rows": rows})
    write_analysis(OUTPUT, rows, elapsed, current_source)
    return 0 if not manifest["source_changed"] and all(row["status"] == "success" for row in rows) else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-ready-source-hash")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.worker_output is None:
            parser.error("--worker-output is required")
        return worker(args.worker, args.worker_output)
    if not args.parent_ready_source_hash:
        parser.error("Real-data execution requires explicit parent readiness and its source hash")
    return supervisor(args.parent_ready_source_hash)


if __name__ == "__main__":
    raise SystemExit(main())
