"""Fixed 24-cell robust replay on the exact prior training-only audit partitions."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import shutil
import sys
import threading
import time
import types
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROCESS_STARTED = time.perf_counter()

import numpy as np  # noqa: E402 - charge scientific imports to the supervisor clock.
import pandas as pd  # noqa: E402

ROOT = Path(r"C:\code\featcopilot")
PREVIOUS = ROOT / "benchmarks" / "tabular-results" / "2026-09-29-training-selection-audit"
ORIGINAL = ROOT / "benchmarks" / "tabular-results" / "2026-09-28-frequency-iid-confirmation"
ALIGNED_PREVIOUS = ROOT / "benchmarks" / "tabular-results" / "2026-09-29-aligned-training-audit"
OUTPUT = ROOT / "benchmarks" / "tabular-results" / "2026-09-29-robust-training-audit"
EXPECTED_SOURCE = "f99759f0c84f668fddab71de44944d68f079604f719d4589946d4e190cce2f44"
FAMILIES = ("computer-hardware-performance", "infrared-oral-temperature")
METHODS = ("baseline", "baseline-robust", "featcopilot-frequency-aligned", "featcopilot-frequency-aligned-robust")
ALPHAS = (0.1, 1.0, 10.0)
SECONDS_CEILING = 1500
REPORT_RESERVE_SECONDS = 60
CELL_SECONDS = 600
sys.path.insert(0, str(ROOT))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(content):
    return hashlib.sha256(content).hexdigest()


def file_hashes(directory):
    return {
        str(path.relative_to(directory)): sha(path.read_bytes())
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def load_previous_helpers(protocol):
    path = PREVIOUS / "training_selection_audit.py"
    source = path.read_bytes()
    if sha(source) != protocol["audit_script_sha256"]:
        raise ValueError("Prior audit helper checksum mismatch")
    # Execute source without import-loader bytecode writes into the preserved study.
    module = types.ModuleType("preserved_training_audit_helpers")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def isolate_verified_partition(data, family, fold, previous, helper):
    protocol = previous["protocol"]
    identity = next(item for item in protocol["data"] if item["family"] == family)
    X, y, outer = helper.isolate_outer_train(data, fold, identity["outer_split_hashes"])
    old = next(row for row in previous["rows"] if row["family"] == family and row["outer_fold"] == fold)
    train, audit, inner_train, inner_valid = helper.nested_partitions(len(X))
    data_hash = sha(pd.util.hash_pandas_object(pd.concat([X, y], axis=1)).to_numpy().tobytes())
    checks = {
        "outer_split_sha256": outer["outer_split_sha256"],
        "outer_train_positions_sha256": outer["outer_train_positions_sha256"],
        "audit_split_sha256": helper.split_hash(train, audit),
        "inner_split_sha256": helper.split_hash(inner_train, inner_valid),
        "isolated_outer_train_sha256": data_hash,
    }
    for key, value in checks.items():
        if value != old[key]:
            raise ValueError(f"Prior audit partition mismatch: {key}")
    if set(train) & set(audit) or not (set(inner_train) | set(inner_valid)) <= set(train):
        raise AssertionError("Nested audit split boundary violation")
    checks.update(
        {
            "family": family,
            "original_outer_fold": fold,
            "outer_train_rows": len(X),
            "original_outer_test_excluded_rows": outer["original_outer_test_excluded_rows"],
            "audit_train_rows": len(train),
            "audit_holdout_rows": len(audit),
            "inner_train_rows": len(inner_train),
            "inner_validation_rows": len(inner_valid),
            "verified_against_previous": True,
        }
    )
    return X, y, train, audit, checks


def make_payload(X, y, train, audit, identity, method):
    return {
        "X": X,
        "y": y,
        "groups": None,
        "timestamps": None,
        "label_end_times": None,
        "column_descriptions": {},
        "dataset": identity["family"],
        "family_id": identity["family"],
        "source": "real_world",
        "task": "regression",
        "role": "retrospective_development",
        "method": method,
        "seed": 2027,
        "fold": identity["original_outer_fold"],
        "split_sha256": identity["audit_split_sha256"],
        "train_idx": train,
        "test_idx": audit,
        "expected_cells": 3,
        "config": {
            "seed": 2027,
            "threads": 8,
            "learners": ["linear"],
            "max_added_features": 50,
            "max_candidates": 1000,
            "stability_repeats": 3,
            "screening": "importance",
            "memory_limit_mb": 4096,
            "detailed_diagnostics": False,
            "live_llm": False,
        },
    }


def validate_record(row, payload, previous):
    if row["status"] != "success":
        return
    if row["split_sha256"] != payload["split_sha256"]:
        raise ValueError("Worker scored a different audit partition")
    if row["n_train"] != len(payload["train_idx"]) or row["n_test"] != len(payload["test_idx"]):
        raise ValueError("Worker row counts differ from the locked audit")
    chosen_index = int(np.argmin(row["inner_losses"]))
    if row["parameters"] != {"regularization": ALPHAS[chosen_index]}:
        raise ValueError("Worker selected parameters inconsistent with inner losses")
    if row["method"] in ("featcopilot-frequency-aligned", "featcopilot-frequency-aligned-robust"):
        trials = row["inner_pipeline_trials"]
        if len(trials) != len(ALPHAS) or row["selection_model_parameters"] != row["parameters"]:
            raise ValueError("Aligned fit parameters do not match the chosen inner pipeline")
        for index, trial in enumerate(trials):
            if trial["parameters"] != {"regularization": ALPHAS[index]}:
                raise ValueError("Aligned trial grid differs from protocol")
            if trial["loss"] != row["inner_losses"][index]:
                raise ValueError("Aligned trial loss bookkeeping mismatch")
        if row["inner_selection_report"] != trials[chosen_index]["selection_report"]:
            raise ValueError("Chosen inner representation report does not match the winning trial")
        row["aligned_trial_parameter_validation_passed"] = True
    if row["method"] in ("baseline", "featcopilot-frequency-aligned"):
        old = next(
            item
            for item in previous["rows"]
            if item["dataset"] == row["dataset"] and item["fold"] == row["fold"] and item["method"] == row["method"]
        )
        row["previous_audit_rmse"] = old["loss"]
        row["previous_audit_rmse_abs_difference"] = abs(row["loss"] - old["loss"])
        row["previous_audit_parameters_match"] = row["parameters"] == old["parameters"]
        row["previous_audit_loss_matches"] = math.isclose(row["loss"], old["loss"], rel_tol=1e-10, abs_tol=1e-12)
        if row["method"] != "baseline":
            row["previous_audit_selected_names_match"] = (
                row["outer_selection_report"]["selected_features"] == old["outer_selection_report"]["selected_features"]
            )
    if row["method"].endswith("-robust"):
        if row.get("numeric_clip_quantiles") != [0.0, 1.0] or row.get("matched_baseline_method") != "baseline-robust":
            raise ValueError("Robust worker policy differs from the fixed min/max declaration")


def numeric_support_diagnostics(X, train, audit):
    """Aggregate original-numeric support only; no learner fit or row outputs."""
    from featcopilot.utils.numeric_clipping import NumericQuantileClipper, guard_numeric_values

    fitting, testing = X.iloc[train], X.iloc[audit]
    clipper = NumericQuantileClipper((0.0, 1.0)).fit(fitting)
    clipped = clipper.transform(testing)
    diagnostics = []
    for column in clipper.numeric_columns_:
        values = guard_numeric_values(fitting[column].to_numpy(dtype=float, na_value=np.nan))
        future = guard_numeric_values(testing[column].to_numpy(dtype=float, na_value=np.nan))
        finite = values[np.isfinite(values)]
        if not len(finite):
            diagnostics.append({"feature": column, "status": "no_finite_training_values"})
            continue
        median = float(np.median(finite))
        imputed = np.where(np.isfinite(values), values, median)
        center = float(imputed.mean())
        scale = float(imputed.std()) or 1.0
        before = (np.where(np.isfinite(future), future, median) - center) / scale
        after_values = clipped[column].to_numpy(dtype=float, na_value=np.nan)
        after = (np.where(np.isfinite(after_values), after_values, median) - center) / scale
        lower, upper = clipper.bounds_[column]
        diagnostics.append(
            {
                "feature": column,
                "status": "ok",
                "train_standardized_max_abs": float(np.max(np.abs((imputed - center) / scale))),
                "audit_standardized_max_abs_before": float(np.max(np.abs(before))),
                "audit_standardized_max_abs_after": float(np.max(np.abs(after))),
                "audit_below_training_bound": int(np.count_nonzero(future < lower)),
                "audit_above_training_bound": int(np.count_nonzero(future > upper)),
                "bounds_sha256": sha(np.asarray([lower, upper], dtype=float).tobytes()),
            }
        )
    return {
        "scope": "original_numeric_inputs_on_this_audit_train_and_holdout_only",
        "kind": "shared_clipper_and_training_median_population_scale_diagnostic_not_fitted_model_contributions",
        "features": diagnostics,
    }


def hard_stop():
    """Terminate only this supervisor and its descendants at the whole-wall limit."""
    import psutil

    try:
        parent = psutil.Process(os.getpid())
        children = parent.children(recursive=True)
        for child in reversed(children):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        if OUTPUT.exists():
            (OUTPUT / "hard_timeout.json").write_text(
                json.dumps({"status": "whole_wall_timeout", "elapsed_seconds": time.perf_counter() - PROCESS_STARTED}),
                encoding="utf-8",
            )
    finally:
        os._exit(124)


def write_analysis(output, rows, elapsed):
    lines = [
        "# Robust training-only retrospective audit",
        "",
        "**Retrospective development, not confirmation.** Each cell receives only its corresponding original",
        "outer-training partition. Its audit holdout and nested tuning split exactly match the prior study.",
        "The six audit partitions overlap; they are not six independent datasets.",
        "",
        "Methods use current benchmark run_isolated/run_case without runtime modifications.",
        "Aligned methods select among the fixed 0.1/1/10 alpha-specific pipelines using nested validation.",
        "Robust methods use fixed TRAIN-min/max caps (0,1) under the shared finite guard.",
        "No choice is made from audit-holdout outcomes.",
        "",
        "| Family | Original fold | Raw RMSE | Robust raw RMSE | Aligned RMSE | Robust aligned RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        for fold in range(3):
            selected = {row["method"]: row for row in rows if row["dataset"] == family and row["fold"] == fold}
            if set(selected) != set(METHODS) or any(row["status"] != "success" for row in selected.values()):
                lines.append(f"| {family} | {fold} | incomplete | incomplete | incomplete | incomplete |")
                continue
            raw, robust_raw, aligned, robust_aligned = (selected[method] for method in METHODS)
            lines.append(
                f"| {family} | {fold} | {raw['loss']:.8g} | {robust_raw['loss']:.8g} | {aligned['loss']:.8g} | "
                f"{robust_aligned['loss']:.8g} |"
            )
    lines.extend(
        [
            "",
            f"Supervisor elapsed through cell execution: {elapsed:.6f} seconds; external launcher timing",
            "records the complete process including startup and summary/artifact writing.",
            "All source/data/split hashes and prior-audit preservation checks are stored in protocol.json.",
            "FE attribution uses robust aligned versus the matched robust raw baseline, not unclipped raw alone.",
            "It cannot establish a universal accuracy gain or justify default promotion.",
        ]
    )
    (output / "ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def publish_archives(protocol, rows):
    """Publish bounded exact parts; large originals remain local and ignored."""
    from benchmarks.run_state import archive_results, atomic_json

    archive_results(OUTPUT / "results.json")
    auxiliary = []
    large = [path for path in OUTPUT.rglob("*") if path.is_file() and path.stat().st_size > 480 * 1024]
    ignored = {"results.json", "source.zip", "cells/", "__pycache__/"}
    for path in large:
        relative = path.relative_to(OUTPUT).as_posix()
        ignored.add(relative)
        if relative == "results.json":
            continue
        content = path.read_bytes()
        compression = "none" if path.suffix == ".zip" else "gzip"
        packed = content if compression == "none" else gzip.compress(content, mtime=0)
        parts = []
        for number, offset in enumerate(range(0, len(packed), 480 * 1024), 1):
            part = path.with_name(path.name + (".part" if compression == "none" else ".gz.part") + str(number))
            part.write_bytes(packed[offset : offset + 480 * 1024])
            parts.append(part.relative_to(OUTPUT).as_posix())
        reconstructed = b"".join((OUTPUT / part).read_bytes() for part in parts)
        if compression == "gzip":
            reconstructed = gzip.decompress(reconstructed)
        if reconstructed != content:
            raise ValueError("Lossless archive reconstruction mismatch")
        auxiliary.append({"name": relative, "compression": compression, "parts": parts, "sha256": sha(content)})
    atomic_json(OUTPUT / "auxiliary_archive.json", {"format": "bounded-lossless-artifacts-v1", "files": auxiliary})
    (OUTPUT / ".gitignore").write_text("\n".join(sorted(ignored)) + "\n", encoding="utf-8")
    files = {
        path.relative_to(OUTPUT).as_posix(): {"sha256": sha(path.read_bytes()), "bytes": path.stat().st_size}
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    atomic_json(OUTPUT / "artifact_hashes.json", {"format": "sha256-artifact-index-v1", "files": files})


def main(execute=False):
    if not execute:
        raise ValueError("Real replay requires explicit --execute")
    from benchmarks import tabular
    from benchmarks.run_state import atomic_json

    started_utc = utc_now()
    timer = threading.Timer(max(0.01, SECONDS_CEILING - (time.perf_counter() - PROCESS_STARTED) - 5), hard_stop)
    timer.daemon = True
    timer.start()
    versions = {
        name: importlib.metadata.version(name)
        for name in ["scikit-learn", "joblib", "numpy", "pandas", "scipy", "threadpoolctl"]
    }
    expected = {
        "scikit-learn": "1.7.2",
        "joblib": "1.3.2",
        "numpy": "2.4.4",
        "pandas": "2.3.3",
        "scipy": "1.17.1",
        "threadpoolctl": "3.6.0",
    }
    if versions != expected:
        raise ValueError("Pinned benchmark environment required")
    if tabular.source_fingerprint() != EXPECTED_SOURCE:
        raise ValueError("Runtime source does not match parent freeze")
    if tabular.candidate_parameters("linear") != [{"regularization": value} for value in ALPHAS]:
        raise ValueError("Unexpected benchmark learner grid")
    if tabular.ROBUST_CLIP_QUANTILES != (0.0, 1.0):
        raise ValueError("Robust bounds differ from preregistered min/max pair")
    previous_hashes = file_hashes(PREVIOUS)
    previous = json.loads((PREVIOUS / "results.json").read_text())
    aligned_hashes = file_hashes(ALIGNED_PREVIOUS)
    aligned_previous = json.loads((ALIGNED_PREVIOUS / "results.json").read_text())
    helper = load_previous_helpers(previous["protocol"])
    if sha((ORIGINAL / "manifest.json").read_bytes()) != previous["protocol"]["original_manifest_sha256"]:
        raise ValueError("Original confirmation manifest changed")
    if OUTPUT.exists():
        raise ValueError("Refusing to overwrite an existing aligned audit")
    OUTPUT.mkdir(parents=True)
    copied_script = OUTPUT / "robust_training_audit.py"
    shutil.copy2(Path(__file__), copied_script)
    tests = Path(__file__).with_name("test_robust_training_audit.py")
    if tests.exists():
        shutil.copy2(tests, OUTPUT / tests.name)
    with zipfile.ZipFile(OUTPUT / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as snapshot:
        for path in tabular.source_files():
            snapshot.write(path, arcname=path.relative_to(ROOT).as_posix())
        snapshot.write(copied_script, arcname=copied_script.name)
    protocol = {
        "role": "RETROSPECTIVE_DEVELOPMENT_NOT_CONFIRMATION",
        "started_utc": started_utc,
        "source_sha256": EXPECTED_SOURCE,
        "source_zip_sha256": sha((OUTPUT / "source.zip").read_bytes()),
        "script_sha256": sha(copied_script.read_bytes()),
        "previous_audit_artifact_sha256": previous_hashes,
        "previous_aligned_artifact_sha256": aligned_hashes,
        "versions": versions,
        "python_executable": sys.executable,
        "methods": list(METHODS),
        "learner": "linear",
        "alpha_grid": list(ALPHAS),
        "config_seed": 2027,
        "audit_seed": 2028,
        "audit_holdout_fraction": 0.25,
        "internal_holdout_fraction": 0.25,
        "stability_repeats": 3,
        "max_candidates": 1000,
        "max_added_features": 50,
        "native_threads": 8,
        "cell_seconds_ceiling": CELL_SECONDS,
        "global_seconds_ceiling": SECONDS_CEILING,
        "report_reserve_seconds": REPORT_RESERVE_SECONDS,
        "requested_cells": 24,
        "numeric_clip_quantiles": [0.0, 1.0],
        "memory_limit_mb": 4096,
        "partitions": [],
        "prepared_cache_hashes": {},
        "original_outer_test_scored": False,
        "synthetic_safety_tests": {"passed": 4, "whole_test_process_seconds": 29.5925588},
    }
    rows = [
        {
            "dataset": family,
            "family_id": family,
            "fold": fold,
            "method": method,
            "seed": 2027,
            "learner": "linear",
            "task": "regression",
            "role": "retrospective_development",
            "status": "not_started",
        }
        for family in FAMILIES
        for fold in range(3)
        for method in METHODS
    ]
    atomic_json(OUTPUT / "protocol.json", protocol)
    atomic_json(OUTPUT / "results.json", {"protocol": protocol, "rows": rows})
    cell_index = 0
    for family in FAMILIES:
        cache = ORIGINAL / "prepared" / f"{family}.pkl"
        content = cache.read_bytes()
        identity = next(item for item in previous["protocol"]["data"] if item["family"] == family)
        if sha(content) != identity["prepared_sha256"]:
            raise ValueError("Prepared cache differs from the previous audit")
        protocol["prepared_cache_hashes"][family] = identity["prepared_sha256"]
        # No fitting/statistics occurs while the original full cache is in scope.
        data = pickle.loads(content)
        isolated = [isolate_verified_partition(data, family, fold, previous, helper) for fold in range(3)]
        del data, content
        for X, y, train, audit, checks in isolated:
            if time.perf_counter() - PROCESS_STARTED < SECONDS_CEILING - REPORT_RESERVE_SECONDS:
                checks["numeric_support_diagnostics"] = numeric_support_diagnostics(X, train, audit)
            protocol["partitions"].append(checks)
            for method in METHODS:
                payload = make_payload(X, y, train, audit, checks, method)
                remaining = SECONDS_CEILING - REPORT_RESERVE_SECONDS - (time.perf_counter() - PROCESS_STARTED)
                directory = OUTPUT / "cells" / f"{family}-fold-{checks['original_outer_fold']}-{method}"
                atomic_json(OUTPUT / "protocol.json", protocol)
                call_started = time.perf_counter()
                if remaining <= 0:
                    result = tabular.failure_rows(
                        payload, "budget_exhausted", "Audit/report-reserve ceiling reached", 0.0
                    )
                else:
                    if tabular.source_fingerprint() != EXPECTED_SOURCE:
                        raise ValueError("Runtime changed during the audit")
                    result = tabular.run_isolated(payload, directory, timeout=min(CELL_SECONDS, remaining))
                if len(result) != 1:
                    raise ValueError("Expected exactly one benchmark row per method/partition")
                row = result[0]
                row["audit_supervisor_cell_seconds"] = time.perf_counter() - call_started
                row["original_outer_split_sha256"] = checks["outer_split_sha256"]
                row["audit_inner_split_sha256"] = checks["inner_split_sha256"]
                row["isolated_outer_train_sha256"] = checks["isolated_outer_train_sha256"]
                row["corresponding_original_outer_test_excluded"] = True
                row["role"] = "retrospective_development"
                validate_record(row, payload, aligned_previous)
                rows[cell_index] = row
                atomic_json(OUTPUT / f"record-{cell_index:02d}.json", row)
                cell_index += 1
                atomic_json(OUTPUT / "results.json", {"protocol": protocol, "rows": rows})
                print(
                    json.dumps(
                        {
                            "family": family,
                            "fold": row["fold"],
                            "method": method,
                            "status": row["status"],
                            "seconds": row["audit_supervisor_cell_seconds"],
                        }
                    ),
                    flush=True,
                )
                if directory.exists() and list(directory.glob("*.pkl")):
                    raise ValueError("Benchmark worker retained raw payload copies")
        del isolated
    elapsed = time.perf_counter() - PROCESS_STARTED
    preserved = file_hashes(PREVIOUS) == previous_hashes and file_hashes(ALIGNED_PREVIOUS) == aligned_hashes
    protocol.update(
        {
            "finished_utc": utc_now(),
            "supervisor_elapsed_seconds": elapsed,
            "source_sha256_at_finish": tabular.source_fingerprint(),
            "source_changed": tabular.source_fingerprint() != EXPECTED_SOURCE,
            "previous_audit_bytes_preserved": preserved,
            "attempted_cells": sum(row["status"] not in ("not_started", "budget_exhausted") for row in rows),
            "successful_cells": sum(row["status"] == "success" for row in rows),
            "child_worker_wall_seconds": sum(row.get("worker_wall_seconds", 0.0) for row in rows),
            "pipeline_seconds_sum": sum(row.get("pipeline_seconds", 0.0) for row in rows),
        }
    )
    raw_summary = tabular.summarize(rows, 3, 1)
    robust_summary = tabular.summarize(
        [
            dict(row, method="baseline") if row["method"] == "baseline-robust" else row
            for row in rows
            if row["method"] in ("baseline-robust", "featcopilot-frequency-aligned-robust")
        ],
        3,
        1,
    )
    atomic_json(OUTPUT / "protocol.json", protocol)
    atomic_json(
        OUTPUT / "results.json",
        {"protocol": protocol, "rows": rows, "summary": raw_summary, "matched_robust_summary": robust_summary},
    )
    write_analysis(OUTPUT, rows, elapsed)
    if not preserved:
        raise ValueError("Previous audit artifacts changed")
    publish_archives(protocol, rows)
    timer.cancel()
    return (
        0
        if len(rows) == 24 and not protocol["source_changed"] and all(row["status"] == "success" for row in rows)
        else 1
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(args.execute))
