"""Replay three already-exposed Thoracic folds with an explicitly repaired reference."""

# Include dependency import time in the measured diagnostic budget.
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

ROOT = Path(r"C:\code\featcopilot")
sys.path.insert(0, str(ROOT))

from benchmarks import execution, tabular
from benchmarks.run_state import ExposureLedger, RunState, archive_results, archive_source, atomic_json, fingerprint

logger = logging.getLogger("safe_scale_diagnostic")
ORIGINAL = ROOT / "benchmarks" / "tabular-results" / "2026-10-04-thoracic-fixed-average-confirmation"
OUTPUT = ROOT / "benchmarks" / "tabular-results" / "2026-10-04-autofeat-safe-scale-diagnostic"
METHOD = "autofeat-native-safe-scale-v1-robust"
CEILING = 3600.0


def digest(path: Path) -> str:
    """Return the exact artifact checksum."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    """Run the fixed retrospective reference comparison."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    expected = {
        "results.json": "4e5ae41a54e77f7c4fd1926fe4621e6fd6737ad3d1cb364c8f143bf2e2f7e92a",
        "manifest.json": "99475fab6026e5872d033312a321bd9cd90eb753aa18d1e2175af9ecbb5d94b1",
    }
    for name, checksum in expected.items():
        if digest(ORIGINAL / name) != checksum:
            raise ValueError(f"Original evidence changed: {name}")
    original = json.loads((ORIGINAL / "manifest.json").read_text())
    metadata = original["datasets"][0]
    prepared = ORIGINAL / "prepared" / "thoracic-surgery.pkl"
    if digest(prepared) != metadata["prepared_sha256"]:
        raise ValueError("Original prepared dataset checksum differs")
    if digest(ORIGINAL / "source.zip") != original["source_archive_sha256"]:
        raise ValueError("Original source archive checksum differs")
    data = pickle.loads(prepared.read_bytes())
    config = {
        **original["config"],
        "allow_confirmation": False,
        "methods": [METHOD],
        "seed": 2031,
        "seeds": [2031],
        "learners": ["lightgbm"],
    }
    splits = execution._splits(data, "classification", config, None)
    recorded = {(item["seed"], item["fold"]): item["sha256"] for item in original["splits"]}
    if len(splits) != 3 or any(
        execution.split_fingerprint(train, test) != recorded[(seed, fold)] for seed, fold, train, test in splits
    ):
        raise ValueError("Retrospective split identities differ")
    source = tabular.source_fingerprint()
    protocol = {
        "protocol": "autofeat_safe_scale_retrospective_v1",
        "method": METHOD,
        "not_confirmation": True,
        "original_confirmation_decision": "blocked",
        "original_run_id": original["run_id"],
        "original_artifact_sha256": expected,
        "prepared_sha256": metadata["prepared_sha256"],
        "evaluation_tier": metadata["evaluation_tier"],
        "claim_boundary": metadata["claim_boundary"],
        "config": config,
        "splits": original["splits"],
        "expected_cells": 3,
        "total_seconds": CEILING,
        "source_sha256": source,
        "versions": execution.environment_versions(),
        "runner_sha256": digest(Path(__file__)),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    state = RunState(OUTPUT, protocol)
    shutil.copyfile(__file__, OUTPUT / "run_diagnostic.py")
    protocol["source_archive_sha256"] = archive_source(OUTPUT, ROOT, tabular.source_files(), original["suite"])
    atomic_json(OUTPUT / "protocol.json", protocol)
    ledger = ExposureLedger(ROOT / "benchmarks" / ".cache" / "exposure.sqlite")
    ledger.admit("thoracic-surgery", state.run_id, "development")
    interrupted = False
    for seed, fold, train, test in splits:
        payload = {
            "data_path": str(prepared),
            "prepared_sha256": metadata["prepared_sha256"],
            "dataset": "thoracic-surgery",
            "family_id": "thoracic-surgery",
            "role": "development",
            "source": "real_world",
            "task": "classification",
            "method": METHOD,
            "seed": seed,
            "fold": fold,
            "train_idx": train,
            "test_idx": test,
            "split_sha256": execution.split_fingerprint(train, test),
            "expected_cells": 3,
            "config": config,
        }
        cell_id = state.cell_id("thoracic-surgery", seed, fold, METHOD, "lightgbm")
        cell_hash = fingerprint(
            {
                "source": source,
                "prepared": metadata["prepared_sha256"],
                "split": payload["split_sha256"],
                "cell": cell_id,
            }
        )
        remaining = CEILING - (time.perf_counter() - STARTED) - 20.0
        if interrupted:
            result = tabular.failure_rows(payload, "interrupted", "Diagnostic interrupted", 0.0)[0]
        elif tabular.source_fingerprint() != source:
            result = tabular.failure_rows(payload, "source_changed", "Diagnostic source changed", 0.0)[0]
        elif remaining <= 0:
            result = tabular.failure_rows(payload, "budget_exhausted", "Diagnostic budget exhausted", 0.0)[0]
        else:
            logger.info("Measuring safe-scale compatibility on original fold %s", fold)
            try:
                results = tabular.run_isolated(payload, state.next_attempt(cell_id), min(900.0, remaining))
            except KeyboardInterrupt:
                interrupted = True
                results = tabular.failure_rows(payload, "interrupted", "Diagnostic interrupted", 0.0)
            if len(results) != 1:
                raise ValueError("Expected one reference result per fold")
            result = results[0]
        result.update(
            evaluation_tier=metadata["evaluation_tier"],
            claim_boundary=metadata["claim_boundary"],
            retrospective_diagnostic=True,
            original_confirmation_run=original["run_id"],
        )
        state.elapsed_seconds = time.perf_counter() - STARTED
        state.record(cell_id, cell_hash, result)
        atomic_json(
            OUTPUT / "results.json", {"protocol": protocol, "run_id": state.run_id, "rows": list(state.rows.values())}
        )
    for name, checksum in expected.items():
        if digest(ORIGINAL / name) != checksum:
            raise ValueError(f"Original evidence changed during the diagnostic: {name}")
    protocol["source_sha256_at_finish"] = tabular.source_fingerprint()
    protocol["source_changed"] = source != protocol["source_sha256_at_finish"]
    protocol["original_artifacts_unchanged"] = True
    protocol["finished_utc"] = datetime.now(timezone.utc).isoformat()
    archive_results(OUTPUT / "results.json")
    state.elapsed_seconds = time.perf_counter() - STARTED
    state.save()
    protocol["active_seconds"] = state.elapsed_seconds
    atomic_json(OUTPUT / "protocol.json", protocol)
    logger.info("Saved all %s retrospective result cells", len(state.rows))
    return int(protocol["source_changed"] or any(row["status"] != "success" for row in state.rows.values()))


if __name__ == "__main__":
    raise SystemExit(main())
