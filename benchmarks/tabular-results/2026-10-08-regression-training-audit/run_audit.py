"""Supervise the preregistered hardware TRAIN-only numerical diagnostic."""

# ruff: noqa: E402
import time

STARTED = time.perf_counter()

import hashlib
import json
import logging
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import arff
import numpy as np
import pandas as pd

ROOT = Path(r"C:\code\featcopilot")
sys.path.insert(0, str(ROOT))

from benchmarks import catalog, execution, tabular
from benchmarks.run_state import ExposureLedger, RunState, archive_results, archive_source, atomic_json, fingerprint

OUTPUT = ROOT / "benchmarks" / "tabular-results" / "2026-10-08-regression-training-audit"
PROTOCOL = ROOT / "benchmarks" / "suites" / "regression_training_audit_20261008.json"
REFERENCES = {
    "2026-10-08-regression-average-development": {
        "results.json": "aaf0788d1d8b14ed366e53e273c0b7294d0e05cd7af92f2000257e4432fb8009",
        "controller.json": "bf618631121d33e87bb9259ce94c94b7a0742c3c64065b948802a7cff66477c7",
        "protocol.json": "929009d90b5019b25f4fb8d261e669fda29dd99cecd9b392fd72e3d5f5155c1d",
    },
    "2026-10-08-regression-policy-replication": {
        "results.json": "7f315b3f190fa6c542f86a70b72f4a0d0f37f0e45096254d5ee130123d5fd582",
        "controller.json": "ecd1d6ab9e1e1638d6686160bd0f55a7f947994004d2bd365a35dc05345c7646",
        "protocol.json": "66374f399b619e086e75e93e959e8b295abc1927a3008d3396c0a64638295c2d",
    },
}
METHOD = "regression-training-audit"
logger = logging.getLogger("regression_training_audit")


def load(path: Path) -> dict:
    """Read one JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    """Hash exact file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def references_unchanged() -> None:
    """Reject any modification to the scored historical result inputs."""
    for run, files in REFERENCES.items():
        for name, expected in files.items():
            assert sha(OUTPUT.parent / run / name) == expected, (run, name)


def prepare(protocol: dict) -> tuple[list[dict], dict]:
    """Extract only official outer-TRAIN frames and allowed train-only reference fields."""
    references_unchanged()
    name = protocol["dataset"]
    source = OUTPUT.parent / protocol["reference_runs"]["42"] / "hardware-official"
    manifest = load(source / "manifest.json")
    versions = execution.environment_versions()
    assert versions == manifest["versions"] and sys.version == manifest["python"]
    metadata = manifest["datasets"][0]
    assert metadata["name"] == name and metadata["task"] == "regression"
    prepared = source / "prepared" / f"{name}.pkl"
    assert sha(prepared) == "9b41a54261472f08dcc0d9996c7c8080d0bfb3f50923144c68ca71f2fc967f0c"
    data = pickle.loads(prepared.read_bytes())
    assert not any(data.get(key) is not None for key in ("groups", "timestamps", "label_end_times"))
    cache = (
        ROOT
        / "benchmarks"
        / ".cache"
        / name
        / "sources"
        / "539abb055e867a889bf7ffb4b6428a1c86098f1472cc3441651fdf67ca2dd079"
        / "splits.arff"
    )
    assert sha(cache) == metadata["official_splits_sha256"]
    parsed = arff.loads(cache.read_text(encoding="utf-8"))
    frame = pd.DataFrame(parsed["data"], columns=[entry[0] for entry in parsed["attributes"]])
    splits, ids = catalog._official_splits(frame, len(data["X"]))
    assert ids == metadata["official_split_ids"] and len(splits) == len(protocol["official_folds"]) == 10
    original_rows = {}
    for run in REFERENCES:
        original_rows[run] = [
            {
                **{key: row[key] for key in ("seed", "fold", "method", "split_sha256")},
                **{key: row[key] for key in protocol["reference_parity_fields"]},
            }
            for row in load(OUTPUT.parent / run / "results.json")["rows"]
            if row["dataset"] == name and row["learner"] == "linear" and row["method"] in protocol["component_methods"]
        ]
    jobs, preparation = [], {}
    for fold, (train, test) in enumerate(splits):
        split_hash = execution.split_fingerprint(train, test)
        # Only ordered outer-TRAIN rows cross the worker boundary.
        training = {"X": data["X"].iloc[train].copy(), "y": data["y"].iloc[train].copy()}
        train_bytes = pickle.dumps(training)
        train_hash = hashlib.sha256(train_bytes).hexdigest()
        train_indices_hash = hashlib.sha256(np.asarray(train, dtype="<i8").tobytes()).hexdigest()
        preparation[str(fold)] = {
            "training_rows": len(train),
            "original_train_indices_sha256": train_indices_hash,
            "original_split_sha256": split_hash,
            "prepared_sha256": train_hash,
            "bytes": train_bytes,
        }
        for seed in protocol["seeds"]:
            run = protocol["reference_runs"][str(seed)]
            expected = {}
            for role, method in zip(("raw", "fe"), protocol["component_methods"], strict=True):
                matching = [
                    row
                    for row in original_rows[run]
                    if row["seed"] == seed and row["fold"] == fold and row["method"] == method
                ]
                assert len(matching) == 1 and matching[0]["split_sha256"] == split_hash
                expected[role] = {key: matching[0][key] for key in protocol["reference_parity_fields"]}
            jobs.append(
                {
                    "dataset": name,
                    "family_id": name,
                    "source": "real_world",
                    "task": "regression",
                    "method": METHOD,
                    "seed": seed,
                    "fold": fold,
                    "split_sha256": split_hash,
                    "role": "development",
                    "expected_cells": protocol["expected_jobs"],
                    "training_rows": len(train),
                    "original_train_indices_sha256": train_indices_hash,
                    "expected_components": expected,
                    "package_versions": {name: version for name, version in versions.items() if version is not None},
                    "reference_run": run,
                    "prepared_sha256": train_hash,
                    "config": {
                        "learners": ["linear"],
                        "seed": seed,
                        "threads": protocol["threads"],
                        "memory_limit_mb": protocol["memory_limit_mb"],
                        "max_candidates": protocol["max_candidates"],
                        "max_added_features": protocol["max_added_features"],
                        "stability_repeats": protocol["stability_repeats"],
                        "numeric_clip_quantiles": protocol["numeric_clip_quantiles"],
                        "categorical_encoding": protocol["categorical_encoding"],
                        "target_transform": "identity",
                        "screening": "auto",
                    },
                }
            )
    return jobs, preparation


def publish_source_archive() -> None:
    """Publish the exact source ZIP in bounded ordered parts."""
    path = OUTPUT / "source.zip"
    content = path.read_bytes()
    parts = []
    for index, offset in enumerate(range(0, len(content), 480 * 1024), start=1):
        name = f"source.zip.part{index}"
        (OUTPUT / name).write_bytes(content[offset : offset + 480 * 1024])
        parts.append(name)
    assert b"".join((OUTPUT / name).read_bytes() for name in parts) == content
    atomic_json(
        OUTPUT / "auxiliary_archive.json",
        {
            "format": "bounded-lossless-artifacts-v1",
            "files": [{"name": "source.zip", "compression": "none", "parts": parts, "sha256": sha(path)}],
        },
    )


def main() -> int:
    """Run all 30 diagnostic jobs once under the existing resource supervisor."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    protocol = load(PROTOCOL)
    jobs, preparation = prepare(protocol)
    assert len(jobs) == protocol["expected_jobs"] == 30
    source = tabular.source_fingerprint()
    public_preparation = {
        fold: {key: value for key, value in item.items() if key != "bytes"} for fold, item in preparation.items()
    }
    identity = {
        "protocol": protocol,
        "source_sha256": source,
        "versions": execution.environment_versions(),
        "reference_artifact_sha256": REFERENCES,
        "training_preparation": public_preparation,
        "protocol_sha256": sha(PROTOCOL),
        "runner_sha256": sha(Path(__file__)),
    }
    state = RunState(OUTPUT, identity)
    shutil.copyfile(PROTOCOL, OUTPUT / "protocol.json")
    shutil.copyfile(__file__, OUTPUT / "run_audit.py")
    prepared_dir = OUTPUT / "prepared"
    prepared_dir.mkdir()
    for fold, record in preparation.items():
        path = prepared_dir / f"fold-{fold}-train.pkl"
        path.write_bytes(record["bytes"])
        assert sha(path) == record["prepared_sha256"]
    controller = {
        **identity,
        "run_id": state.run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_archive_sha256": archive_source(OUTPUT, ROOT, tabular.source_files(), protocol),
    }
    atomic_json(OUTPUT / "controller.json", controller)
    ExposureLedger(ROOT / "benchmarks" / ".cache" / "exposure.sqlite").admit(
        protocol["family_id"], state.run_id, "development"
    )
    interrupted = False
    try:
        for payload in jobs:
            payload["data_path"] = str(prepared_dir / f"fold-{payload['fold']}-train.pkl")
            assert sha(Path(payload["data_path"])) == payload["prepared_sha256"]
            cell_id = state.cell_id(payload["dataset"], payload["seed"], payload["fold"], METHOD, "linear")
            cell_hash = fingerprint({"identity": identity, "cell": cell_id, "expected": payload["expected_components"]})
            remaining = (
                protocol["combined_ceiling_seconds"]
                - (time.perf_counter() - STARTED)
                - protocol["publication_reserve_seconds"]
            )
            if interrupted:
                rows = tabular.failure_rows(payload, "interrupted", "Training diagnostic interrupted", 0.0)
            elif tabular.source_fingerprint() != source:
                rows = tabular.failure_rows(payload, "source_changed", "Audit source changed", 0.0)
            elif remaining <= 0:
                rows = tabular.failure_rows(payload, "budget_exhausted", "Audit time budget exhausted", 0.0)
            else:
                logger.info("TRAIN-only audit original fold %s seed %s", payload["fold"], payload["seed"])
                try:
                    rows = tabular.run_isolated(
                        payload,
                        state.next_attempt(cell_id),
                        min(protocol["case_timeout_seconds"], remaining),
                        worker_module="benchmarks.regression_training_audit",
                    )
                except KeyboardInterrupt:
                    interrupted = True
                    rows = tabular.failure_rows(payload, "interrupted", "Training diagnostic interrupted", 0.0)
            assert len(rows) == 1 and "loss" not in rows[0], "Diagnostic must never publish an outer evaluation score"
            rows[0].update(
                not_scored_evaluation=True,
                training_payload_sha256=payload["prepared_sha256"],
                original_train_indices_sha256=payload["original_train_indices_sha256"],
                reference_run=payload["reference_run"],
            )
            state.elapsed_seconds = time.perf_counter() - STARTED
            state.record(cell_id, cell_hash, rows[0])
            atomic_json(OUTPUT / "results.json", {"protocol": protocol, "rows": list(state.rows.values())})
        references_unchanged()
        archive_results(OUTPUT / "results.json")
        publish_source_archive()
    finally:
        controller["source_sha256_at_finish"] = tabular.source_fingerprint()
        controller["source_changed"] = source != controller["source_sha256_at_finish"]
        controller["active_seconds"] = time.perf_counter() - STARTED
        controller["budget_overshoot_seconds"] = max(
            0.0, controller["active_seconds"] - protocol["combined_ceiling_seconds"]
        )
        controller["finished_utc"] = datetime.now(timezone.utc).isoformat()
        controller["completed_jobs"] = len(state.rows)
        controller["successful_jobs"] = sum(row["status"] == "success" for row in state.rows.values())
        atomic_json(OUTPUT / "controller.json", controller)
        state.elapsed_seconds = controller["active_seconds"]
        state.save()
    return int(
        controller["source_changed"]
        or controller["successful_jobs"] != 30
        or controller["budget_overshoot_seconds"] > 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
