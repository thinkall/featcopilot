"""Replicate the frozen learner-specific regression policy without changing its source."""

# ruff: noqa: E402
import time

STARTED = time.perf_counter()

import gzip
import hashlib
import json
import logging
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\code\featcopilot")
sys.path.insert(0, str(ROOT))

from benchmarks import execution, tabular
from benchmarks.run_state import archive_results, atomic_json

OUTPUT = ROOT / "benchmarks" / "tabular-results" / "2026-10-08-regression-policy-replication"
PROTOCOL = ROOT / "benchmarks" / "suites" / "regression_policy_replication_20261008.json"
SUITES = [
    ROOT / "benchmarks" / "suites" / "frequency_development_replay_20260924.json",
    ROOT / "benchmarks" / "tabular_suite.json",
    ROOT / "benchmarks" / "suites" / "regression_exposed_stress_replay_20261008.json",
]
logger = logging.getLogger("regression_policy_replication")


def load(path: Path) -> dict:
    """Read one JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    """Hash exact file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_reference(protocol: dict) -> dict:
    """Require unchanged prior source members and dependency versions before creating a run."""
    reference = OUTPUT.parent / protocol["reference_run"]
    controller = load(reference / "controller.json")
    manifest = load(reference / "iid" / "manifest.json")
    assert controller["source_sha256"] == controller["source_sha256_at_finish"] == protocol["reference_source_sha256"]
    assert not controller["source_changed"]
    assert sha(reference / "iid" / "source.zip") == protocol["reference_source_archive_sha256"]
    with zipfile.ZipFile(reference / "iid" / "source.zip") as archive:
        previous = set(archive.namelist()) - {"resolved-suite.json"}
        for name in previous:
            assert archive.read(name) == (ROOT / name).read_bytes(), f"Frozen source changed: {name}"
    current = {path.relative_to(ROOT).as_posix() for path in tabular.source_files()}
    assert current - previous == set(protocol["allowed_added_source_files"])
    assert not previous - current
    assert execution.environment_versions() == manifest["versions"]
    assert sys.version == manifest["python"]
    return controller


def arguments(stage: dict, protocol: dict) -> list[str]:
    """Build the fixed canonical stage options."""
    result = [
        "--output",
        str(OUTPUT / stage["name"]),
        "--suite",
        *map(str, SUITES),
        "--datasets",
        *stage["datasets"],
        "--methods",
        *stage["methods"],
        "--learners",
        *stage["learners"],
        "--seeds",
        *map(str, protocol["seeds"]),
    ]
    for flag, value in {
        "--folds": stage["folds"],
        "--max-rows": stage["max_rows"],
        "--total-seconds": stage["budget_seconds"],
        "--case-timeout": protocol["case_timeout_seconds"],
        "--threads": protocol["threads"],
        "--memory-limit-mb": protocol["memory_limit_mb"],
        "--max-candidates": protocol["max_candidates"],
        "--max-added-features": protocol["max_added_features"],
        "--stability-repeats": protocol["stability_repeats"],
    }.items():
        result.extend([flag, str(value)])
    return result


def verify_preparation(stage: dict, protocol: dict, source: str, reference: dict) -> dict:
    """Check complete stages and exact original prepared bytes before any model fitting."""
    directory = OUTPUT / stage["name"]
    manifest = load(directory / "manifest.json")
    assert not manifest["source_changed"]
    assert manifest["source_sha256"] == manifest["source_sha256_at_finish"] == source
    assert manifest["planned_cells"] == stage["expected_cells"]
    assert not load(directory / "state.json")["records"]
    assert {item["name"] for item in manifest["datasets"]} == set(stage["datasets"])
    for key in ("methods", "learners"):
        assert manifest["config"][key] == stage[key]
    assert manifest["config"]["seeds"] == protocol["seeds"]
    prepared = {}
    for metadata in manifest["datasets"]:
        name = metadata["name"]
        assert "error" not in metadata, (name, metadata.get("error"))
        assert metadata["task"] == "regression" and metadata["role"] == "development"
        assert metadata.get("target_transform", "identity") == "identity"
        assert metadata["outer_split_count"] == stage["folds"] * len(protocol["seeds"])
        prepared[name] = sha(directory / "prepared" / f"{name}.pkl")
        assert prepared[name] == reference["stages"][stage["source_stage"]]["prepared_sha256"][name]
        splits = [item for item in manifest["splits"] if item["dataset"] == name]
        assert len(splits) == stage["folds"] * len(protocol["seeds"])
        assert {item["seed"] for item in splits} == set(protocol["seeds"])
        if stage["split_policy"] == "official":
            assert len(metadata["official_split_ids"]) == 10
            assert metadata["sampled_rows"] == metadata["original_rows"]
        else:
            assert metadata["sampled_rows"] <= stage["max_rows"]
    assert sha(directory / "source.zip") == manifest["source_archive_sha256"]
    return {"prepared_sha256": prepared, "source_archive_sha256": manifest["source_archive_sha256"]}


def publish(protocol: dict, controller: dict) -> None:
    """Retain all outcomes and exact bounded per-stage source/manifest archives."""
    rows, files = [], []
    for stage in protocol["stages"]:
        directory = OUTPUT / stage["name"]
        result = load(directory / "results.json")
        manifest = load(directory / "manifest.json")
        assert len(result["rows"]) == stage["expected_cells"]
        assert manifest["source_sha256"] == controller["source_sha256"] and not manifest["source_changed"]
        for name, expected in controller["stages"][stage["name"]]["prepared_sha256"].items():
            assert sha(directory / "prepared" / f"{name}.pkl") == expected
        for row in result["rows"]:
            cohort = "primary" if row["family_id"] in protocol["primary_families"] else f"stress:{row['family_id']}"
            rows.append({**row, "study_stage": stage["name"], "analysis_cohort": cohort})
        for name in ("manifest.json", "source.zip"):
            path = directory / name
            content = path.read_bytes()
            compression = "gzip" if name.endswith(".json") else "none"
            encoded = gzip.compress(content, mtime=0) if compression == "gzip" else content
            parts = []
            for index, offset in enumerate(range(0, len(encoded), 480 * 1024), start=1):
                part = f"{stage['name']}.{name}{'.gz' if compression == 'gzip' else ''}.part{index}"
                (OUTPUT / part).write_bytes(encoded[offset : offset + 480 * 1024])
                parts.append(part)
            restored = b"".join((OUTPUT / part).read_bytes() for part in parts)
            assert (gzip.decompress(restored) if compression == "gzip" else restored) == content
            files.append(
                {"name": f"{stage['name']}/{name}", "compression": compression, "parts": parts, "sha256": sha(path)}
            )
    assert len(rows) == protocol["expected_cells"]
    assert len({row["cell_id"] for row in rows}) == len(rows)
    atomic_json(OUTPUT / "results.json", {"protocol": protocol, "rows": rows})
    archive_results(OUTPUT / "results.json")
    atomic_json(OUTPUT / "auxiliary_archive.json", {"format": "bounded-lossless-artifacts-v1", "files": files})
    controller["scored_cells"] = sum(row["status"] == "success" for row in rows)
    controller["expected_cells"] = len(rows)


def main() -> int:
    """Prepare every stage, then replicate inside the shared seven-hour ceiling."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    protocol = load(PROTOCOL)
    reference = verify_reference(protocol)
    assert sum(stage["expected_cells"] for stage in protocol["stages"]) == protocol["expected_cells"] == 340
    assert sum(stage["budget_seconds"] for stage in protocol["stages"]) == protocol["combined_ceiling_seconds"] == 25200
    assert (
        protocol["prior_cycle_used_hours"] + protocol["combined_ceiling_seconds"] / 3600 < protocol["cycle_total_hours"]
    )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(PROTOCOL, OUTPUT / "protocol.json")
    shutil.copyfile(__file__, OUTPUT / "run_study.py")
    source = tabular.source_fingerprint()
    controller = {
        "source_sha256": source,
        "protocol_sha256": sha(PROTOCOL),
        "runner_sha256": sha(Path(__file__)),
        "reference_source_verified": True,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "stages": {},
    }
    try:
        for stage in protocol["stages"]:
            logger.info("Preparing frozen stage %s before any model fitting", stage["name"])
            code = tabular.main([*arguments(stage, protocol), "--prepare-only"])
            if code:
                raise RuntimeError(f"Stage {stage['name']} preparation failed with exit code {code}")
            controller["stages"][stage["name"]] = verify_preparation(stage, protocol, source, reference)
            atomic_json(OUTPUT / "controller.json", controller)
        for stage in protocol["stages"]:
            assert sha(PROTOCOL) == controller["protocol_sha256"] and tabular.source_fingerprint() == source
            manifest = load(OUTPUT / stage["name"] / "manifest.json")
            remaining = protocol["combined_ceiling_seconds"] - (time.perf_counter() - STARTED) - 120.0
            if remaining <= 0:
                raise TimeoutError("Shared replication budget exhausted before stage dispatch")
            budget = min(stage["budget_seconds"], remaining + manifest["active_seconds"])
            controller["stages"][stage["name"]]["effective_budget_seconds"] = budget
            atomic_json(OUTPUT / "controller.json", controller)
            logger.info("Replicating stage %s with cumulative ceiling %.3fs", stage["name"], budget)
            controller["stages"][stage["name"]]["exit_code"] = tabular.main(
                ["--output", str(OUTPUT / stage["name"]), "--resume", "--total-seconds", str(budget)]
            )
            atomic_json(OUTPUT / "controller.json", controller)
        publish(protocol, controller)
    except (AssertionError, OSError, ValueError, RuntimeError, KeyboardInterrupt) as exc:
        controller["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("Replication stopped; no successful outcome was substituted")
        raise
    finally:
        controller["source_sha256_at_finish"] = tabular.source_fingerprint()
        controller["source_changed"] = controller["source_sha256_at_finish"] != source
        controller["active_seconds"] = time.perf_counter() - STARTED
        controller["budget_overshoot_seconds"] = max(
            0.0, controller["active_seconds"] - protocol["combined_ceiling_seconds"]
        )
        controller["finished_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(OUTPUT / "controller.json", controller)
    return int(
        controller["source_changed"]
        or controller["scored_cells"] != protocol["expected_cells"]
        or controller["budget_overshoot_seconds"] > 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
