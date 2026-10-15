"""Verify all training-only numerical audit artifacts without evaluating test data."""

import gzip
import hashlib
import json
import math
import pickle
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

import arff
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(r"C:\code\featcopilot")
sys.path.insert(0, str(ROOT))

from benchmarks import catalog, execution, tabular  # noqa: E402
from benchmarks.run_state import atomic_json  # noqa: E402

OUTPUT = ROOT / "benchmarks" / "tabular-results" / "2026-10-08-regression-training-audit"


def load(path: Path) -> dict:
    """Read JSON, accepting a PowerShell UTF-8 BOM."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    """Return the checksum of exact artifact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positions_sha(positions: np.ndarray) -> str:
    """Match the recorded ordered int64 positional checksum."""
    return hashlib.sha256(np.asarray(positions, dtype="<i8").tobytes()).hexdigest()


def verify_fitted_diagnostics(item: dict, expected_rows: int, input_width: int) -> None:
    """Validate compact numerical diagnostics without reconstructing predictions."""
    assert item["fit_rows"] == expected_rows
    names = item["feature_names"]
    assert len(names) == item["n_output_features"]
    assert item["selected_features"] == names[input_width:]
    assert hashlib.sha256(json.dumps(names).encode()).hexdigest() == item["feature_names_sha256"]
    design = item["scaled_training_design"]
    assert design["rows"] == expected_rows and 0 <= design["rank"] <= min(expected_rows, design["columns"])
    assert design["columns"] == len(names)  # This fixed hardware source has no categorical block.
    assert math.isfinite(design["singular_value_max"]) and math.isfinite(design["singular_value_min"])
    assert design["singular_value_max"] >= design["singular_value_min"] >= 0
    if design["rank"] < design["columns"]:
        assert design["condition_status"] == "rank_deficient" and design["condition_number"] is None
    else:
        assert design["condition_status"] == "full_column_rank"
        assert math.isfinite(design["condition_number"]) and design["condition_number"] >= 1
    assert math.isfinite(item["training_residual_rmse"]) and item["training_residual_rmse"] >= 0
    assert all(math.isfinite(value) and value >= 0 for value in item["coefficient_norms"].values())
    clipping = item["clipping_saturation"]
    eligible = [column for column in clipping["columns"] if column["status"] == "eligible"]
    assert clipping["eligible_finite_count"] == sum(column["finite_count"] for column in eligible)
    assert clipping["lower_count"] == sum(column["lower_count"] for column in eligible)
    assert clipping["upper_count"] == sum(column["upper_count"] for column in eligible)
    assert clipping["saturated_count"] == clipping["lower_count"] + clipping["upper_count"]
    for column in clipping["columns"]:
        assert column["finite_count"] + column["missing_count"] == expected_rows
        if column["status"] == "eligible":
            assert column["saturated_count"] == column["lower_count"] + column["upper_count"]
            assert 0 <= column["saturated_fraction"] <= 1
        else:
            assert column["saturated_count"] is None and column["saturated_fraction"] is None


def main() -> None:
    """Verify all 30 jobs, 60 component replays and ten original TRAIN boundaries."""
    protocol = load(OUTPUT / "protocol.json")
    controller = load(OUTPUT / "controller.json")
    timing = load(OUTPUT / "launcher_timing.json")
    result = load(OUTPUT / "results.json")
    rows = result["rows"]
    assert result["protocol"] == controller["protocol"] == protocol
    assert sha(OUTPUT / "protocol.json") == controller["protocol_sha256"]
    assert sha(OUTPUT / "run_audit.py") == controller["runner_sha256"]
    assert controller["source_sha256"] == controller["source_sha256_at_finish"] == tabular.source_fingerprint()
    assert not controller["source_changed"] and controller["budget_overshoot_seconds"] == 0
    assert timing["exit_code"] == 0
    assert 0 < controller["active_seconds"] <= timing["whole_process_seconds"] < protocol["combined_ceiling_seconds"]
    assert controller["versions"] == execution.environment_versions()
    assert len(rows) == controller["completed_jobs"] == controller["successful_jobs"] == protocol["expected_jobs"] == 30
    assert {(row["seed"], row["fold"]) for row in rows} == {
        (seed, fold) for seed in protocol["seeds"] for fold in protocol["official_folds"]
    }
    assert len({row["cell_id"] for row in rows}) == 30
    assert sha(OUTPUT / "source.zip") == controller["source_archive_sha256"]
    with zipfile.ZipFile(OUTPUT / "source.zip") as archive:
        for path in tabular.source_files():
            assert archive.read(path.relative_to(ROOT).as_posix()) == path.read_bytes()
        assert json.loads(archive.read("resolved-suite.json")) == protocol
    reference_fields = protocol["reference_parity_fields"]
    references = {}
    for run, artifacts in controller["reference_artifact_sha256"].items():
        for name, checksum in artifacts.items():
            assert sha(OUTPUT.parent / run / name) == checksum
        references[run] = {
            (row["seed"], row["fold"], row["method"]): {
                **{key: row[key] for key in reference_fields},
                "split_sha256": row["split_sha256"],
            }
            for row in load(OUTPUT.parent / run / "results.json")["rows"]
            if row["dataset"] == protocol["dataset"]
            and row["learner"] == "linear"
            and row["method"] in protocol["component_methods"]
        }
    original = OUTPUT.parent / protocol["reference_runs"]["42"] / "hardware-official"
    original_controller = load(original.parent / "controller.json")
    original_prepared = original / "prepared" / f"{protocol['dataset']}.pkl"
    assert (
        sha(original_prepared)
        == original_controller["stages"]["hardware-official"]["prepared_sha256"][protocol["dataset"]]
    )
    data = pickle.loads(original_prepared.read_bytes())
    metadata = load(original / "manifest.json")["datasets"][0]
    cache = (
        ROOT
        / "benchmarks"
        / ".cache"
        / protocol["dataset"]
        / "sources"
        / "539abb055e867a889bf7ffb4b6428a1c86098f1472cc3441651fdf67ca2dd079"
        / "splits.arff"
    )
    assert sha(cache) == metadata["official_splits_sha256"]
    parsed = arff.loads(cache.read_text(encoding="utf-8"))
    frame = pd.DataFrame(parsed["data"], columns=[entry[0] for entry in parsed["attributes"]])
    splits, ids = catalog._official_splits(frame, len(data["X"]))
    assert ids == metadata["official_split_ids"]
    for fold, (train, test) in enumerate(splits):
        path = OUTPUT / "prepared" / f"fold-{fold}-train.pkl"
        record = controller["training_preparation"][str(fold)]
        assert sha(path) == record["prepared_sha256"]
        training = pickle.loads(path.read_bytes())
        assert set(training) == {"X", "y"} and len(training["X"]) == record["training_rows"] == len(train)
        assert record["original_train_indices_sha256"] == positions_sha(train)
        assert record["original_split_sha256"] == execution.split_fingerprint(train, test)
        pd.testing.assert_frame_equal(training["X"], data["X"].iloc[train])
        pd.testing.assert_series_equal(training["y"], data["y"].iloc[train])
        assert np.intersect1d(train, test).size == 0
        # Verify only the original training rows; outer-test values are never selected or scored.
    compact = []
    for row in rows:
        assert row["status"] == "success" and row["attempt"] == 1 and row["not_scored_evaluation"]
        assert not ({"loss", "r2", "mae", "n_test", "prediction", "predictions", "test_idx"} & row.keys())
        assert row["method"] == "regression-training-audit" and row["learner"] == "linear"
        assert row["worker_python_hash_seed"] == row["seed"]
        audit = row["audit"]
        record = controller["training_preparation"][str(row["fold"])]
        assert audit["scope"] == "original_outer_train_only_no_outer_test_access_or_scoring"
        assert audit["train_data_sha256"] == row["training_payload_sha256"] == record["prepared_sha256"]
        assert row["original_train_indices_sha256"] == record["original_train_indices_sha256"]
        assert row["split_sha256"] == record["original_split_sha256"]
        assert audit["package_versions"] == controller["versions"]
        assert audit["prediction_pipeline_fit_count"] == 8
        assert set(audit["components"]) == {"raw", "fe"}
        for role, method in zip(("raw", "fe"), protocol["component_methods"], strict=True):
            component = audit["components"][role]
            expected = references[row["reference_run"]][row["seed"], row["fold"], method]
            assert component["method"] == method and row["split_sha256"] == expected["split_sha256"]
            assert audit["reference_fields_verified"][role] == sorted(reference_fields)
            for key in reference_fields:
                assert component[key] == expected[key], (row["seed"], row["fold"], role, key)
            train, valid = train_test_split(np.arange(row["n_train"]), test_size=0.25, random_state=row["seed"])
            assert component["inner_train_positions_sha256"] == positions_sha(train)
            assert component["inner_validation_positions_sha256"] == positions_sha(valid)
            verify_fitted_diagnostics(component["winning_inner"], len(train), row["n_input_features"])
            verify_fitted_diagnostics(component["full_train"], row["n_train"], row["n_input_features"])
            final, inner = component["full_train"], component["winning_inner"]
            a, b = inner["selected_features"], final["selected_features"]
            turnover = component["selected_name_turnover"]
            assert turnover["common_count"] == len(set(a) & set(b))
            assert turnover["removed"] == [name for name in a if name not in b]
            assert turnover["added"] == [name for name in b if name not in a]
            compact.append(
                {
                    "seed": row["seed"],
                    "fold": row["fold"],
                    "role": role,
                    "regularization": component["parameters"]["regularization"],
                    "widths": [inner["n_output_features"], final["n_output_features"]],
                    "ordered_names_equal": turnover["ordered_names_equal"],
                    "inner_design": inner["scaled_training_design"],
                    "final_design": final["scaled_training_design"],
                    "inner_coefficient_norms": inner["coefficient_norms"],
                    "final_coefficient_norms": final["coefficient_norms"],
                    "inner_training_residual_rmse": inner["training_residual_rmse"],
                    "final_training_residual_rmse": final["training_residual_rmse"],
                    "inner_training_boundary_fraction": inner["clipping_saturation"]["saturated_fraction"],
                    "final_training_boundary_fraction": final["clipping_saturation"]["saturated_fraction"],
                }
            )
    index = load(OUTPUT / "archive.json")
    reconstructed = gzip.decompress(b"".join((OUTPUT / name).read_bytes() for name in index["parts"]))
    assert reconstructed == (OUTPUT / "results.json").read_bytes()
    assert hashlib.sha256(reconstructed).hexdigest() == index["uncompressed_sha256"]
    for artifact in load(OUTPUT / "auxiliary_archive.json")["files"]:
        reconstructed = b"".join((OUTPUT / name).read_bytes() for name in artifact["parts"])
        assert reconstructed == (OUTPUT / artifact["name"]).read_bytes()
        assert hashlib.sha256(reconstructed).hexdigest() == artifact["sha256"]
        assert all((OUTPUT / name).stat().st_size <= 480 * 1024 for name in artifact["parts"])
    fe = [item for item in compact if item["role"] == "fe"]
    used = protocol["prior_cycle_used_hours"] + timing["whole_process_seconds"] / 3600
    summary = {
        "final_design_status_counts": dict(Counter(item["final_design"]["condition_status"] for item in fe)),
        "inner_design_status_counts": dict(Counter(item["inner_design"]["condition_status"] for item in fe)),
        "changed_ordered_plans": sum(not item["ordered_names_equal"] for item in fe),
        "changed_widths": sum(item["widths"][0] != item["widths"][1] for item in fe),
        "final_coefficient_l2_range": [
            min(item["final_coefficient_norms"]["l2"] for item in fe),
            max(item["final_coefficient_norms"]["l2"] for item in fe),
        ],
    }
    analysis = {
        "not_scored_evaluation": True,
        "not_confirmation": True,
        "identity_source_boundary_and_archive_checks_passed": True,
        "successful_jobs": 30,
        "exact_component_replays": 60,
        "fitted_designs_instrumented": 120,
        "prediction_pipeline_fits": 240,
        "source_sha256": controller["source_sha256"],
        "results_sha256": sha(OUTPUT / "results.json"),
        "protocol_sha256": sha(OUTPUT / "protocol.json"),
        "summary": summary,
        "components": compact,
        "measurement_hours": timing["whole_process_seconds"] / 3600,
        "cycle_used_hours": used,
        "cycle_remaining_hours": protocol["cycle_total_hours"] - used,
        "limitations": [
            "Diagnostic design conditioning is unpenalized and omits the intercept column; not the centered regularized Ridge solve condition.",
            "Coefficient norms are scale/representation dependent and do not identify a safe threshold.",
            "Training boundary counts are neither changed-value counts nor future clipping/extrapolation rates.",
            "No outer-test data were sent to workers or selected/scored by this verification.",
        ],
        "analysis_runner_sha256": sha(Path(__file__)),
    }
    shutil.copyfile(__file__, OUTPUT / "analyze_audit.py")
    atomic_json(OUTPUT / "analysis.json", analysis)
    print(
        json.dumps(
            {
                "summary": summary,
                "used_hours": used,
                "remaining_hours": protocol["cycle_total_hours"] - used,
                "fold6": [item for item in fe if item["fold"] == 6],
                "counterexamples": [item for item in fe if (item["seed"], item["fold"]) in ((2029, 0), (137, 3))],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
