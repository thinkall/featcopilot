"""Verify frozen-source regression replication and its preregistered gates."""

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

OUTPUT = ROOT / "benchmarks" / "tabular-results" / "2026-10-08-regression-policy-replication"
RAW = "baseline-robust"
FE = "featcopilot-frequency-aligned-robust"
AVERAGE = "featcopilot-regression-average-robust"


def load(path: Path) -> dict:
    """Read one JSON artifact, accepting a PowerShell UTF-8 BOM."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    """Hash the exact file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(row: dict) -> tuple:
    """Return the full public cell identity."""
    return tuple(row[key] for key in ("study_stage", "dataset", "seed", "fold", "method", "learner", "split_sha256"))


def verify_archive(directory: Path) -> None:
    """Reconstruct raw results from only index-listed ordered parts."""
    index = load(directory / "archive.json")
    content = gzip.decompress(b"".join((directory / part).read_bytes() for part in index["parts"]))
    assert content == (directory / "results.json").read_bytes()
    assert hashlib.sha256(content).hexdigest() == index["uncompressed_sha256"]


def main() -> None:
    """Validate complete evidence, paired controls, seed-wise and pooled outcomes."""
    protocol = load(OUTPUT / "protocol.json")
    controller = load(OUTPUT / "controller.json")
    timing = load(OUTPUT / "launcher_timing.json")
    result = load(OUTPUT / "results.json")
    rows = result["rows"]
    reference = OUTPUT.parent / protocol["reference_run"]
    original_controller = load(reference / "controller.json")
    original_manifest = load(reference / "iid" / "manifest.json")
    assert result["protocol"] == protocol
    assert sha(OUTPUT / "protocol.json") == controller["protocol_sha256"]
    assert sha(OUTPUT / "run_study.py") == controller["runner_sha256"]
    assert controller["source_sha256"] == controller["source_sha256_at_finish"] == tabular.source_fingerprint()
    assert not controller["source_changed"] and controller["reference_source_verified"]
    assert controller["budget_overshoot_seconds"] == 0 and timing["exit_code"] == 0
    assert 0 < controller["active_seconds"] <= timing["whole_process_seconds"] < protocol["combined_ceiling_seconds"]
    assert len(rows) == controller["scored_cells"] == protocol["expected_cells"] == 340
    index = {identity(row): row for row in rows}
    assert len(index) == len(rows)
    assert sha(reference / "iid" / "source.zip") == protocol["reference_source_archive_sha256"]
    assert original_manifest["source_sha256"] == protocol["reference_source_sha256"]
    with zipfile.ZipFile(reference / "iid" / "source.zip") as archive:
        old_names = set(archive.namelist()) - {"resolved-suite.json"}
        for name in old_names:
            assert archive.read(name) == (ROOT / name).read_bytes(), name
    current_names = {path.relative_to(ROOT).as_posix() for path in tabular.source_files()}
    assert current_names - old_names == set(protocol["allowed_added_source_files"])
    assert not old_names - current_names
    assert execution.environment_versions() == original_manifest["versions"]
    expected_identities = set()
    stage_times = {}
    for stage in protocol["stages"]:
        directory = OUTPUT / stage["name"]
        manifest = load(directory / "manifest.json")
        record = controller["stages"][stage["name"]]
        assert record["exit_code"] == 0 and not manifest["source_changed"]
        assert manifest["source_sha256"] == manifest["source_sha256_at_finish"] == controller["source_sha256"]
        assert sha(directory / "source.zip") == manifest["source_archive_sha256"] == record["source_archive_sha256"]
        assert manifest["active_seconds"] <= record["effective_budget_seconds"] <= stage["budget_seconds"]
        stage_times[stage["name"]] = manifest["active_seconds"]
        assert manifest["planned_cells"] == stage["expected_cells"]
        assert manifest["versions"] == original_manifest["versions"]
        assert manifest["python"] == original_manifest["python"]
        for key in ("seeds", "threads", "max_candidates", "max_added_features", "stability_repeats"):
            assert manifest["config"][key] == protocol[key], key
        for key in ("methods", "learners", "folds", "max_rows"):
            assert manifest["config"][key] == stage[key], key
        assert manifest["config"]["case_timeout"] == protocol["case_timeout_seconds"]
        assert manifest["config"]["memory_limit_mb"] == protocol["memory_limit_mb"]
        assert not manifest["config"]["live_llm"] and not manifest["config"]["allow_confirmation"]
        with zipfile.ZipFile(directory / "source.zip") as archive:
            for path in tabular.source_files():
                assert archive.read(path.relative_to(ROOT).as_posix()) == path.read_bytes(), path
            assert json.loads(archive.read("resolved-suite.json")) == manifest["suite"]
        assert {item["name"] for item in manifest["datasets"]} == set(stage["datasets"])
        for metadata in manifest["datasets"]:
            name = metadata["name"]
            assert "error" not in metadata
            assert metadata["task"] == "regression" and metadata["role"] == "development"
            path = directory / "prepared" / f"{name}.pkl"
            assert sha(path) == record["prepared_sha256"][name]
            assert sha(path) == original_controller["stages"][stage["source_stage"]]["prepared_sha256"][name]
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
                assert identifiers == metadata["official_split_ids"] and len(official) == 10
                assert metadata["sample_indices"] == list(range(metadata["original_rows"]))
            splits = execution._splits(data, "regression", manifest["config"], official)
            assert len(splits) == metadata["outer_split_count"] == stage["folds"] * len(protocol["seeds"])
            for seed, fold, train, test in splits:
                split_hash = execution.split_fingerprint(train, test)
                recorded = next(
                    item
                    for item in manifest["splits"]
                    if item["dataset"] == name and item["seed"] == seed and item["fold"] == fold
                )
                assert recorded["sha256"] == split_hash
                for method in stage["methods"]:
                    for learner in stage["learners"]:
                        key = (stage["name"], name, seed, fold, method, learner, split_hash)
                        expected_identities.add(key)
                        row = index[key]
                        assert row["n_train"] == len(train) and row["n_test"] == len(test)
                        assert row["expected_cells"] == len(splits)
                        assert row["family_id"] == metadata["family_id"]
                        cohort = (
                            "primary"
                            if row["family_id"] in protocol["primary_families"]
                            else f"stress:{row['family_id']}"
                        )
                        assert row["analysis_cohort"] == cohort
        stage_rows = load(directory / "results.json")["rows"]
        assert len(stage_rows) == stage["expected_cells"]
        for row in stage_rows:
            combined = index[identity({**row, "study_stage": stage["name"]})]
            assert row == {
                key: value for key, value in combined.items() if key not in ("study_stage", "analysis_cohort")
            }
        verify_archive(directory)
    assert expected_identities == set(index)
    for row in rows:
        assert row["status"] == "success" and row["attempt"] == 1 and row["role"] == "development"
        assert row["worker_python_hash_seed"] == row["seed"] and row["seed"] in protocol["seeds"]
        assert row["numeric_clip_quantiles"] == [0.0, 1.0] and row["loss"] > 0
        assert all(math.isfinite(row[key]) for key in ("loss", "r2", "mae", "pipeline_seconds"))
    verify_archive(OUTPUT)
    for artifact in load(OUTPUT / "auxiliary_archive.json")["files"]:
        content = b"".join((OUTPUT / part).read_bytes() for part in artifact["parts"])
        if artifact["compression"] == "gzip":
            content = gzip.decompress(content)
        assert content == (OUTPUT / artifact["name"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == artifact["sha256"]
        assert all((OUTPUT / part).stat().st_size <= 480 * 1024 for part in artifact["parts"])
    mismatches, pairs = [], 0
    for row in rows:
        if row["method"] != AVERAGE:
            continue
        details = row["regression_average"]
        assert details["weights"] == protocol["weights"] and not details["weight_optimized"]
        assert details["complete_procedure_count"] == 2
        for role, method in (("raw", RAW), ("fe", FE)):
            component = details["components"][role]
            control = index[identity({**row, "method": method})]
            pairs += 1
            for key in protocol["required_parity_fields"]:
                if component[key] != control[key]:
                    mismatches.append(
                        {
                            **{key: row[key] for key in ("dataset", "learner", "seed", "fold")},
                            "role": role,
                            "field": key,
                        }
                    )
    assert pairs == protocol["expected_component_control_pairs"] == 136
    families, summaries, tails = [], [], []
    for label, seeds in [("seed137", [137]), ("seed2029", [2029]), ("pooled137_2029", protocol["seeds"])]:
        for learner, method in protocol["primary_methods"].items():
            learner_rows = [row for row in rows if row["learner"] == learner and row["seed"] in seeds]
            section = []
            for family in protocol["primary_families"] + protocol["stress_families"]:
                selected = [row for row in learner_rows if row["family_id"] == family]
                methods = (RAW, FE) if learner == "linear" else (RAW, FE, AVERAGE)
                losses = {name: mean(row["loss"] for row in selected if row["method"] == name) for name in methods}
                expected_folds = (10 if family == "computer-hardware-performance" else 3) * len(seeds)
                assert all(sum(row["method"] == name for row in selected) == expected_folds for name in methods)
                item = {
                    "scope": label,
                    "learner": learner,
                    "primary_method": method,
                    "family_id": family,
                    "cohort": selected[0]["analysis_cohort"],
                    "folds_per_method": expected_folds,
                    "mean_rmse": losses,
                    "policy_gain_vs_raw": 1 - losses[method] / losses[RAW],
                    "single_fe_gain_vs_raw": 1 - losses[FE] / losses[RAW],
                    "policy_gain_vs_single_fe": 1 - losses[method] / losses[FE],
                    "mean_pipeline_seconds": {
                        name: mean(row["pipeline_seconds"] for row in selected if row["method"] == name)
                        for name in methods
                    },
                }
                families.append(item)
                section.append(item)
            primary = [item for item in section if item["cohort"] == "primary"]
            gain = mean(item["policy_gain_vs_raw"] for item in primary)
            worst = min(item["policy_gain_vs_raw"] for item in section)
            parity_ok = not any(item["learner"] == learner and item["seed"] in seeds for item in mismatches)
            summaries.append(
                {
                    "scope": label,
                    "learner": learner,
                    "primary_method": method,
                    "primary_mean_gain_vs_raw": gain,
                    "primary_single_fe_mean_gain_vs_raw": mean(item["single_fe_gain_vs_raw"] for item in primary),
                    "primary_mean_gain_vs_single_fe": mean(item["policy_gain_vs_single_fe"] for item in primary),
                    "primary_worst_relative_harm_vs_raw": max(
                        0.0, -min(item["policy_gain_vs_raw"] for item in primary)
                    ),
                    "worst_relative_harm_including_separate_stress": max(0.0, -worst),
                    "component_control_parity": parity_ok,
                    "eligible_for_followup": parity_ok and gain > 0 and worst >= -0.05,
                }
            )
            for compared in (FE,) if learner == "linear" else (FE, AVERAGE):
                candidates = [row for row in learner_rows if row["method"] == compared]
                worst_row = max(
                    candidates, key=lambda row: row["loss"] / index[identity({**row, "method": RAW})]["loss"]
                )
                tails.append(
                    {
                        "scope": label,
                        "learner": learner,
                        "method": compared,
                        **{key: worst_row[key] for key in ("dataset", "seed", "fold", "analysis_cohort")},
                        "relative_rmse_change_vs_raw": worst_row["loss"]
                        / index[identity({**worst_row, "method": RAW})]["loss"]
                        - 1,
                    }
                )
    used = protocol["prior_cycle_used_hours"] + timing["whole_process_seconds"] / 3600
    analysis = {
        "not_confirmation": True,
        "identity_and_archive_checks_passed": True,
        "source_sha256": controller["source_sha256"],
        "prior_source_unchanged": True,
        "results_sha256": sha(OUTPUT / "results.json"),
        "protocol_sha256": sha(OUTPUT / "protocol.json"),
        "successful_cells": len(rows),
        "component_control_pairs": pairs,
        "component_control_mismatches": mismatches,
        "families": families,
        "summaries": summaries,
        "worst_individual_folds": tails,
        "replication_gate_passed": all(item["eligible_for_followup"] for item in summaries),
        "stage_active_seconds": stage_times,
        "physical_worker_seconds_sum": sum(row["worker_wall_seconds"] for row in rows),
        "measurement_hours": timing["whole_process_seconds"] / 3600,
        "cycle_used_hours": used,
        "cycle_remaining_hours": protocol["cycle_total_hours"] - used,
        "default_promotion": False,
        "analysis_runner_sha256": sha(Path(__file__)),
    }
    shutil.copyfile(__file__, OUTPUT / "analyze_study.py")
    atomic_json(OUTPUT / "analysis.json", analysis)
    stress = [item for item in families if item["cohort"] != "primary"]
    print(
        json.dumps(
            {
                "summaries": summaries,
                "stress": stress,
                "tails": tails,
                "mismatches": mismatches,
                "used_hours": used,
                "remaining_hours": protocol["cycle_total_hours"] - used,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
