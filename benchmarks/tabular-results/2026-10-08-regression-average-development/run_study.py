"""Run the two predeclared regression stages under one measurement ceiling."""

# ruff: noqa: E402
import time

STARTED = time.perf_counter()

import gzip
import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\code\featcopilot")
sys.path.insert(0, str(ROOT))

from benchmarks import tabular
from benchmarks.run_state import archive_results, atomic_json

OUTPUT = ROOT / "benchmarks" / "tabular-results" / "2026-10-08-regression-average-development"
PROTOCOL = ROOT / "benchmarks" / "suites" / "regression_average_protocol_20261008.json"
SUITES = [
    ROOT / "benchmarks" / "suites" / "frequency_development_replay_20260924.json",
    ROOT / "benchmarks" / "tabular_suite.json",
    ROOT / "benchmarks" / "suites" / "regression_exposed_stress_replay_20261008.json",
]
logger = logging.getLogger("regression_average_study")


def load(path: Path) -> dict:
    """Load one JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    """Hash exact artifact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_arguments(stage: dict, protocol: dict) -> list[str]:
    """Construct only the fixed canonical CLI options."""
    args = [
        "--output",
        str(OUTPUT / stage["name"]),
        "--suite",
        *map(str, SUITES),
        "--datasets",
        *stage["datasets"],
        "--methods",
        *protocol["methods"],
        "--learners",
        *protocol["learners"],
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
        args.extend([flag, str(value)])
    return args


def verify_preparation(stage: dict, source: str) -> dict:
    """Reject changed identities, missing sources or incorrect split counts before fitting."""
    directory = OUTPUT / stage["name"]
    manifest = load(directory / "manifest.json")
    assert not manifest["source_changed"]
    assert manifest["source_sha256"] == manifest["source_sha256_at_finish"] == source
    assert manifest["planned_cells"] == stage["expected_cells"]
    assert {item["name"] for item in manifest["datasets"]} == set(stage["datasets"])
    assert not load(directory / "state.json")["records"]
    prepared = {}
    for item in manifest["datasets"]:
        assert "error" not in item, (item["name"], item.get("error"))
        assert item["task"] == "regression" and item["role"] == "development"
        assert item.get("target_transform", "identity") == "identity"
        assert item["outer_split_count"] == stage["folds"]
        assert item["sampled_rows"] <= stage["max_rows"] if stage["max_rows"] else True
        path = directory / "prepared" / f"{item['name']}.pkl"
        prepared[item["name"]] = sha(path)
        splits = [split for split in manifest["splits"] if split["dataset"] == item["name"]]
        assert len(splits) == stage["folds"] and {split["seed"] for split in splits} == {42}
        if stage["split_policy"] == "official":
            assert len(item["official_split_ids"]) == 10
            assert item["sampled_rows"] == item["original_rows"]
    assert sha(directory / "source.zip") == manifest["source_archive_sha256"]
    return {"prepared_sha256": prepared, "source_archive_sha256": manifest["source_archive_sha256"]}


def publish(protocol: dict, controller: dict) -> None:
    """Retain every stage outcome and exact bounded stage artifacts."""
    rows, artifacts = [], []
    for stage in protocol["stages"]:
        directory = OUTPUT / stage["name"]
        result = load(directory / "results.json")
        manifest = load(directory / "manifest.json")
        assert len(result["rows"]) == stage["expected_cells"]
        assert manifest["source_sha256"] == controller["source_sha256"]
        assert not manifest["source_changed"]
        for name, expected in controller["stages"][stage["name"]]["prepared_sha256"].items():
            assert sha(directory / "prepared" / f"{name}.pkl") == expected
        for row in result["rows"]:
            cohort = "primary" if row["family_id"] in protocol["primary_families"] else f"stress:{row['family_id']}"
            rows.append({**row, "study_stage": stage["name"], "analysis_cohort": cohort})
        for name in ("manifest.json", "source.zip"):
            path = directory / name
            content = path.read_bytes()
            compressed = name.endswith(".json")
            encoded = gzip.compress(content, mtime=0) if compressed else content
            parts = []
            for index, offset in enumerate(range(0, len(encoded), 480 * 1024), start=1):
                part = f"{stage['name']}.{name}{'.gz' if compressed else ''}.part{index}"
                (OUTPUT / part).write_bytes(encoded[offset : offset + 480 * 1024])
                parts.append(part)
            restored = b"".join((OUTPUT / part).read_bytes() for part in parts)
            assert (gzip.decompress(restored) if compressed else restored) == content
            artifacts.append(
                {
                    "name": f"{stage['name']}/{name}",
                    "compression": "gzip" if compressed else "none",
                    "parts": parts,
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
    assert len(rows) == protocol["expected_cells"]
    assert len({row["cell_id"] for row in rows}) == len(rows)
    atomic_json(OUTPUT / "results.json", {"protocol": protocol, "rows": rows})
    archive_results(OUTPUT / "results.json")
    atomic_json(OUTPUT / "auxiliary_archive.json", {"format": "bounded-lossless-artifacts-v1", "files": artifacts})
    controller["scored_cells"] = sum(row["status"] == "success" for row in rows)
    controller["expected_cells"] = len(rows)


def main() -> int:
    """Prepare both complete cohorts, then run under the fixed global/stage ceilings."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    protocol = load(PROTOCOL)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(PROTOCOL, OUTPUT / "protocol.json")
    shutil.copyfile(__file__, OUTPUT / "run_study.py")
    source = tabular.source_fingerprint()
    controller = {
        "source_sha256": source,
        "protocol_sha256": sha(PROTOCOL),
        "runner_sha256": sha(Path(__file__)),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "stages": {},
    }
    try:
        for stage in protocol["stages"]:
            logger.info("Preparing complete stage %s before any model fitting", stage["name"])
            code = tabular.main([*stage_arguments(stage, protocol), "--prepare-only"])
            if code:
                raise RuntimeError(f"Stage {stage['name']} preparation failed with exit code {code}")
            controller["stages"][stage["name"]] = verify_preparation(stage, source)
            atomic_json(OUTPUT / "controller.json", controller)
        for stage in protocol["stages"]:
            assert sha(PROTOCOL) == controller["protocol_sha256"]
            assert tabular.source_fingerprint() == source
            manifest = load(OUTPUT / stage["name"] / "manifest.json")
            remaining = protocol["combined_ceiling_seconds"] - (time.perf_counter() - STARTED) - 120.0
            if remaining <= 0:
                raise TimeoutError("Combined study budget exhausted before stage dispatch")
            budget = min(stage["budget_seconds"], remaining + manifest["active_seconds"])
            controller["stages"][stage["name"]]["effective_budget_seconds"] = budget
            atomic_json(OUTPUT / "controller.json", controller)
            logger.info("Running fixed stage %s with cumulative ceiling %.3fs", stage["name"], budget)
            code = tabular.main(["--output", str(OUTPUT / stage["name"]), "--resume", "--total-seconds", str(budget)])
            controller["stages"][stage["name"]]["exit_code"] = code
            atomic_json(OUTPUT / "controller.json", controller)
        publish(protocol, controller)
    except (AssertionError, OSError, ValueError, RuntimeError, KeyboardInterrupt) as exc:
        controller["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("Regression study stopped; no successful outcome was substituted")
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
