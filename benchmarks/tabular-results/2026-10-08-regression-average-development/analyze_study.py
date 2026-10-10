"""Verify the completed fixed regression study and publish scoped comparisons."""

import gzip
import hashlib
import json
import math
import pickle
import shutil
import sys
import zipfile
from pathlib import Path
from statistics import mean

import arff
import pandas as pd

ROOT = Path(r"C:\code\featcopilot")
sys.path.insert(0, str(ROOT))

from benchmarks import catalog, execution, tabular  # noqa: E402
from benchmarks.run_state import atomic_json  # noqa: E402

OUTPUT = ROOT / "benchmarks" / "tabular-results" / "2026-10-08-regression-average-development"
RAW = "baseline-robust"
FE = "featcopilot-frequency-aligned-robust"
AVERAGE = "featcopilot-regression-average-robust"


def load(path: Path) -> dict:
    """Read an artifact, including the PowerShell launcher's optional BOM."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    """Hash exact artifact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(row: dict) -> tuple:
    """Return the complete public cell identity."""
    return tuple(row[key] for key in ("study_stage", "dataset", "seed", "fold", "method", "learner", "split_sha256"))


def verify_results_archive(directory: Path) -> None:
    """Require lossless reconstruction using only index-listed ordered parts."""
    index = load(directory / "archive.json")
    content = gzip.decompress(b"".join((directory / part).read_bytes() for part in index["parts"]))
    assert content == (directory / "results.json").read_bytes()
    assert hashlib.sha256(content).hexdigest() == index["uncompressed_sha256"]


def main() -> None:
    """Check all declared cells, exact component parity and scope-specific gates."""
    protocol = load(OUTPUT / "protocol.json")
    controller = load(OUTPUT / "controller.json")
    timing = load(OUTPUT / "launcher_timing.json")
    result = load(OUTPUT / "results.json")
    rows = result["rows"]
    assert result["protocol"] == protocol
    assert sha(OUTPUT / "protocol.json") == controller["protocol_sha256"]
    assert sha(OUTPUT / "run_study.py") == controller["runner_sha256"]
    assert controller["source_sha256"] == controller["source_sha256_at_finish"] == tabular.source_fingerprint()
    assert not controller["source_changed"] and controller["budget_overshoot_seconds"] == 0
    assert timing["exit_code"] == 0
    assert 0 < controller["active_seconds"] <= timing["whole_process_seconds"] <= protocol["combined_ceiling_seconds"]
    assert len(rows) == protocol["expected_cells"] == controller["scored_cells"] == 204
    assert len({identity(row) for row in rows}) == len(rows)
    index = {identity(row): row for row in rows}
    expected_identities = set()
    manifests = {}
    for stage in protocol["stages"]:
        directory = OUTPUT / stage["name"]
        manifest = manifests[stage["name"]] = load(directory / "manifest.json")
        record = controller["stages"][stage["name"]]
        assert not manifest["source_changed"] and record["exit_code"] == 0
        assert manifest["source_sha256"] == manifest["source_sha256_at_finish"] == controller["source_sha256"]
        assert sha(directory / "source.zip") == manifest["source_archive_sha256"] == record["source_archive_sha256"]
        assert manifest["active_seconds"] <= record["effective_budget_seconds"] <= stage["budget_seconds"]
        assert manifest["planned_cells"] == stage["expected_cells"]
        for key in (
            "methods",
            "learners",
            "seeds",
            "threads",
            "max_candidates",
            "max_added_features",
            "stability_repeats",
        ):
            assert manifest["config"][key] == protocol[key], key
        assert manifest["config"]["folds"] == stage["folds"]
        assert manifest["config"]["max_rows"] == stage["max_rows"]
        assert manifest["config"]["case_timeout"] == protocol["case_timeout_seconds"]
        assert manifest["config"]["memory_limit_mb"] == protocol["memory_limit_mb"]
        assert not manifest["config"]["live_llm"] and not manifest["config"]["allow_confirmation"]
        with zipfile.ZipFile(directory / "source.zip") as archive:
            for path in tabular.source_files():
                assert archive.read(path.relative_to(ROOT).as_posix()) == path.read_bytes(), path
            assert json.loads(archive.read("resolved-suite.json")) == manifest["suite"]
        stage_rows = load(directory / "results.json")["rows"]
        assert len(stage_rows) == stage["expected_cells"]
        assert {item["name"] for item in manifest["datasets"]} == set(stage["datasets"])
        for metadata in manifest["datasets"]:
            name = metadata["name"]
            assert metadata["task"] == "regression" and metadata["role"] == "development"
            path = directory / "prepared" / f"{name}.pkl"
            assert sha(path) == record["prepared_sha256"][name]
            data = pickle.loads(path.read_bytes())
            official = None
            if stage["split_policy"] == "official":
                cache = (
                    ROOT
                    / "benchmarks"
                    / ".cache"
                    / "computer-hardware-performance"
                    / "sources"
                    / "539abb055e867a889bf7ffb4b6428a1c86098f1472cc3441651fdf67ca2dd079"
                    / "splits.arff"
                )
                assert sha(cache) == metadata["official_splits_sha256"]
                parsed = arff.loads(cache.read_text(encoding="utf-8"))
                frame = pd.DataFrame(parsed["data"], columns=[entry[0] for entry in parsed["attributes"]])
                official, identifiers = catalog._official_splits(frame, len(data["X"]))
                assert identifiers == metadata["official_split_ids"]
                assert len(official) == 10
                assert metadata["sample_indices"] == list(range(metadata["original_rows"]))
            splits = execution._splits(data, "regression", manifest["config"], official)
            assert len(splits) == stage["folds"]
            for seed, fold, train, test in splits:
                split_hash = execution.split_fingerprint(train, test)
                recorded = next(
                    item
                    for item in manifest["splits"]
                    if item["dataset"] == name and item["seed"] == seed and item["fold"] == fold
                )
                assert split_hash == recorded["sha256"]
                for method in protocol["methods"]:
                    for learner in protocol["learners"]:
                        key = (stage["name"], name, seed, fold, method, learner, split_hash)
                        expected_identities.add(key)
                        row = index[key]
                        assert row["n_train"] == len(train) and row["n_test"] == len(test)
                        assert row["expected_cells"] == len(splits)
                        assert row["family_id"] == metadata["family_id"]
            expected_cohort = (
                "primary"
                if metadata["family_id"] in protocol["primary_families"]
                else f"stress:{metadata['family_id']}"
            )
            assert all(row["analysis_cohort"] == expected_cohort for row in rows if row["dataset"] == name)
        for row in stage_rows:
            combined = index[identity({**row, "study_stage": stage["name"]})]
            assert row == {
                key: value for key, value in combined.items() if key not in ("study_stage", "analysis_cohort")
            }
        verify_results_archive(directory)
    assert expected_identities == set(index)
    for row in rows:
        assert row["status"] == "success" and row["attempt"] == 1 and row["role"] == "development"
        assert row["worker_python_hash_seed"] == row["seed"] == 42
        assert row["numeric_clip_quantiles"] == [0.0, 1.0]
        assert all(math.isfinite(row[key]) for key in ("loss", "r2", "mae", "pipeline_seconds"))
        assert row["loss"] >= 0
    verify_results_archive(OUTPUT)
    for artifact in load(OUTPUT / "auxiliary_archive.json")["files"]:
        content = b"".join((OUTPUT / part).read_bytes() for part in artifact["parts"])
        if artifact["compression"] == "gzip":
            content = gzip.decompress(content)
        assert content == (OUTPUT / artifact["name"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == artifact["sha256"]
        assert all((OUTPUT / part).stat().st_size <= 480 * 1024 for part in artifact["parts"])
    mismatches = []
    component_pairs = 0
    for row in rows:
        if row["method"] != AVERAGE:
            continue
        details = row["regression_average"]
        assert details["weights"] == protocol["weights"] and not details["weight_optimized"]
        assert details["complete_procedure_count"] == 2
        for role, method in (("raw", RAW), ("fe", FE)):
            component = details["components"][role]
            control = index[identity({**row, "method": method})]
            component_pairs += 1
            for key in protocol["required_parity_fields"]:
                if component[key] != control[key]:
                    mismatches.append(
                        {
                            "dataset": row["dataset"],
                            "learner": row["learner"],
                            "fold": row["fold"],
                            "role": role,
                            "field": key,
                        }
                    )
    assert component_pairs == 136
    families = []
    for learner in protocol["learners"]:
        for family in protocol["primary_families"] + protocol["stress_families"]:
            selected = [row for row in rows if row["learner"] == learner and row["family_id"] == family]
            losses = {
                method: mean(row["loss"] for row in selected if row["method"] == method)
                for method in protocol["methods"]
            }
            assert losses[RAW] > 0 and losses[FE] > 0
            families.append(
                {
                    "learner": learner,
                    "family_id": family,
                    "cohort": selected[0]["analysis_cohort"],
                    "folds_per_method": len(selected) // 3,
                    "mean_rmse": losses,
                    "fe_gain_vs_raw": 1 - losses[FE] / losses[RAW],
                    "average_gain_vs_raw": 1 - losses[AVERAGE] / losses[RAW],
                    "average_gain_vs_fe": 1 - losses[AVERAGE] / losses[FE],
                    "mean_pipeline_seconds": {
                        method: mean(row["pipeline_seconds"] for row in selected if row["method"] == method)
                        for method in protocol["methods"]
                    },
                }
            )
    summaries = []
    for learner in protocol["learners"]:
        selected = [family for family in families if family["learner"] == learner]
        primary = [family for family in selected if family["cohort"] == "primary"]
        primary_gain = mean(family["average_gain_vs_raw"] for family in primary)
        worst_gain = min(family["average_gain_vs_raw"] for family in selected)
        parity_ok = not any(item["learner"] == learner for item in mismatches)
        summaries.append(
            {
                "learner": learner,
                "primary_families": len(primary),
                "primary_average_gain_vs_raw": primary_gain,
                "primary_fe_gain_vs_raw": mean(family["fe_gain_vs_raw"] for family in primary),
                "primary_average_gain_vs_fe": mean(family["average_gain_vs_fe"] for family in primary),
                "primary_average_worst_relative_harm_vs_raw": max(0.0, -min(f["average_gain_vs_raw"] for f in primary)),
                "primary_fe_worst_relative_harm_vs_raw": max(0.0, -min(f["fe_gain_vs_raw"] for f in primary)),
                "worst_relative_harm_including_separate_stress": max(0.0, -worst_gain),
                "component_control_parity": parity_ok,
                "eligible_for_followup": parity_ok
                and primary_gain > 0
                and worst_gain >= -protocol["followup_criteria"]["maximum_family_relative_rmse_increase_vs_raw"],
            }
        )
    tails = []
    for learner in protocol["learners"]:
        for method in (FE, AVERAGE):
            selected = [row for row in rows if row["learner"] == learner and row["method"] == method]
            worst = max(selected, key=lambda row: row["loss"] / index[identity({**row, "method": RAW})]["loss"])
            tails.append(
                {
                    "learner": learner,
                    "method": method,
                    "dataset": worst["dataset"],
                    "cohort": worst["analysis_cohort"],
                    "seed": worst["seed"],
                    "fold": worst["fold"],
                    "relative_rmse_change_vs_raw": worst["loss"] / index[identity({**worst, "method": RAW})]["loss"]
                    - 1,
                }
            )
    used = protocol["prior_cycle_used_hours"] + timing["whole_process_seconds"] / 3600
    analysis = {
        "not_confirmation": True,
        "identity_and_archive_checks_passed": True,
        "source_sha256": controller["source_sha256"],
        "results_sha256": sha(OUTPUT / "results.json"),
        "protocol_sha256": sha(OUTPUT / "protocol.json"),
        "successful_cells": len(rows),
        "component_control_pairs": component_pairs,
        "component_control_mismatches": mismatches,
        "families": families,
        "summaries": summaries,
        "worst_individual_folds": tails,
        "physical_worker_seconds_sum": sum(row["worker_wall_seconds"] for row in rows),
        "measurement_hours": timing["whole_process_seconds"] / 3600,
        "cycle_used_hours": used,
        "cycle_remaining_hours": protocol["cycle_total_hours"] - used,
        "default_promotion": False,
        "analysis_runner_sha256": sha(Path(__file__)),
    }
    shutil.copyfile(__file__, OUTPUT / "analyze_study.py")
    atomic_json(OUTPUT / "analysis.json", analysis)
    print(
        json.dumps(
            {"summaries": summaries, "families": families, "mismatches": mismatches, "used_hours": used}, indent=2
        )
    )


if __name__ == "__main__":
    main()
