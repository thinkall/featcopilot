"""Replay the exact prior six training-only audit partitions via benchmark workers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import pickle
import shutil
import sys
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
OUTPUT = ROOT / "benchmarks" / "tabular-results" / "2026-09-29-aligned-training-audit"
EXPECTED_SOURCE = "b7f0b38ae2f73780da84bb1452dcb28e1cec3bf1ec4c44487008fd2d0be2d4c1"
FAMILIES = ("computer-hardware-performance", "infrared-oral-temperature")
METHODS = ("baseline", "featcopilot-frequency-paired", "featcopilot-frequency-aligned")
ALPHAS = (0.1, 1.0, 10.0)
SECONDS_CEILING = 3600
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
    if row["method"] == "featcopilot-frequency-aligned":
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
    else:
        old = next(
            item for item in previous["rows"] if item["family"] == row["dataset"] and item["outer_fold"] == row["fold"]
        )
        prior_model = old["models"]["raw_baseline" if row["method"] == "baseline" else "inner_chosen_alpha"]
        row["previous_audit_rmse"] = prior_model["audit_rmse"]
        row["previous_audit_rmse_abs_difference"] = abs(row["loss"] - prior_model["audit_rmse"])
        row["previous_audit_parameters_match"] = row["parameters"] == {"regularization": prior_model["alpha"]}
        row["previous_audit_loss_matches"] = math.isclose(
            row["loss"], prior_model["audit_rmse"], rel_tol=1e-10, abs_tol=1e-12
        )
        if row["method"] != "baseline":
            row["previous_audit_selected_names_match"] = (
                row["outer_selection_report"]["selected_features"] == old["final_feature_fit"]["selected_names"]
            )


def write_analysis(output, rows, elapsed):
    lines = [
        "# Aligned training-only retrospective audit",
        "",
        "**Retrospective development, not confirmation.** Each cell receives only its corresponding original",
        "outer-training partition. Its audit holdout and nested tuning split exactly match the prior study.",
        "The six audit partitions overlap; they are not six independent datasets.",
        "",
        "Methods use current benchmark run_isolated/run_case without runtime modifications.",
        "Aligned selects among the fixed 0.1/1/10 alpha-specific feature pipelines using nested validation,",
        "then refits the chosen feature-selection/learner parameters. Paired keeps selection alpha=1.",
        "No choice is made from audit-holdout outcomes.",
        "",
        "| Family | Original fold | Raw RMSE | Paired RMSE | Aligned RMSE | Paired/Aligned alpha | Paired/Aligned added |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for family in FAMILIES:
        for fold in range(3):
            selected = {row["method"]: row for row in rows if row["dataset"] == family and row["fold"] == fold}
            if set(selected) != set(METHODS) or any(row["status"] != "success" for row in selected.values()):
                lines.append(f"| {family} | {fold} | incomplete | incomplete | incomplete | - | - |")
                continue
            raw, paired, aligned = (selected[method] for method in METHODS)
            lines.append(
                f"| {family} | {fold} | {raw['loss']:.8g} | {paired['loss']:.8g} | {aligned['loss']:.8g} | "
                f"{paired['parameters']['regularization']:g}/{aligned['parameters']['regularization']:g} | "
                f"{paired['n_output_features']-paired['n_input_features']}/"
                f"{aligned['n_output_features']-aligned['n_input_features']} |"
            )
    lines.extend(
        [
            "",
            f"Supervisor elapsed through cell execution: {elapsed:.6f} seconds; external launcher timing",
            "records the complete process including startup and summary/artifact writing.",
            "All source/data/split hashes and prior-audit preservation checks are stored in protocol.json.",
            "This alignment comparison changes the selected pipeline, not merely alpha on a fixed feature plan.",
            "It cannot establish a universal accuracy gain or justify default promotion.",
        ]
    )
    (output / "ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(execute=False):
    if not execute:
        raise ValueError("Real replay requires explicit --execute")
    from benchmarks import tabular
    from benchmarks.run_state import atomic_json

    started_utc = utc_now()
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
    previous_hashes = file_hashes(PREVIOUS)
    previous = json.loads((PREVIOUS / "results.json").read_text())
    helper = load_previous_helpers(previous["protocol"])
    if sha((ORIGINAL / "manifest.json").read_bytes()) != previous["protocol"]["original_manifest_sha256"]:
        raise ValueError("Original confirmation manifest changed")
    if OUTPUT.exists():
        raise ValueError("Refusing to overwrite an existing aligned audit")
    OUTPUT.mkdir(parents=True)
    copied_script = OUTPUT / "aligned_training_audit.py"
    shutil.copy2(Path(__file__), copied_script)
    tests = Path(__file__).with_name("test_aligned_training_audit.py")
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
        "memory_limit_mb": 4096,
        "partitions": [],
        "prepared_cache_hashes": {},
        "original_outer_test_scored": False,
    }
    rows = []
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
            protocol["partitions"].append(checks)
            for method in METHODS:
                payload = make_payload(X, y, train, audit, checks, method)
                remaining = SECONDS_CEILING - (time.perf_counter() - PROCESS_STARTED)
                directory = OUTPUT / "cells" / f"{family}-fold-{checks['original_outer_fold']}-{method}"
                atomic_json(OUTPUT / "protocol.json", protocol)
                call_started = time.perf_counter()
                if remaining <= 0:
                    result = tabular.failure_rows(payload, "budget_exhausted", "One-hour audit ceiling exhausted", 0.0)
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
                validate_record(row, payload, previous)
                rows.append(row)
                atomic_json(OUTPUT / f"record-{len(rows)-1:02d}.json", row)
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
    preserved = file_hashes(PREVIOUS) == previous_hashes
    protocol.update(
        {
            "finished_utc": utc_now(),
            "supervisor_elapsed_seconds": elapsed,
            "source_sha256_at_finish": tabular.source_fingerprint(),
            "source_changed": tabular.source_fingerprint() != EXPECTED_SOURCE,
            "previous_audit_bytes_preserved": preserved,
            "attempted_cells": len(rows),
            "successful_cells": sum(row["status"] == "success" for row in rows),
            "child_worker_wall_seconds": sum(row.get("worker_wall_seconds", 0.0) for row in rows),
            "pipeline_seconds_sum": sum(row.get("pipeline_seconds", 0.0) for row in rows),
        }
    )
    atomic_json(OUTPUT / "protocol.json", protocol)
    atomic_json(OUTPUT / "results.json", {"protocol": protocol, "rows": rows})
    write_analysis(OUTPUT, rows, elapsed)
    if not preserved:
        raise ValueError("Previous audit artifacts changed")
    return (
        0
        if len(rows) == 18 and not protocol["source_changed"] and all(row["status"] == "success" for row in rows)
        else 1
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(args.execute))
