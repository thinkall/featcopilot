"""Durability, exact-resume and budget identity tests for benchmark execution."""

import gzip
import json
import sqlite3
import zipfile
from unittest.mock import patch

import pytest

from benchmarks.run_state import ExposureLedger, RunState, archive_results, archive_source, fingerprint


def test_reuse_requires_matching_code_configuration_and_data(tmp_path):
    output = tmp_path / "run"
    state = RunState(output, {"code": "abc", "seed": 42})
    cell = state.cell_id("example", 42, 0, "baseline", "linear")
    state.next_attempt(cell)
    state.record(cell, "data-fingerprint", {"status": "success", "loss": 0.5})
    restored = RunState(output, {"seed": 42, "code": "abc"}, resume=True)
    assert restored.reusable(cell, "data-fingerprint")
    assert restored.rows[cell]["loss"] == 0.5
    with pytest.raises(ValueError, match="inputs or split"):
        restored.reusable(cell, "new-data")
    with pytest.raises(ValueError, match="Cannot resume"):
        RunState(output, {"code": "changed", "seed": 42}, resume=True)


def test_failed_retry_never_overwrites_successful_other_learner(tmp_path):
    state = RunState(tmp_path / "run", {"code": "abc"})
    first = state.cell_id("data", 42, 0, "method", "linear")
    second = state.cell_id("data", 42, 0, "method", "lightgbm")
    state.record(first, "first-data", {"status": "success", "loss": 0.1})
    state.record(second, "second-data", {"status": "timeout"})
    assert state.reusable(first, "first-data", retry_failed=True)
    assert not state.reusable(second, "second-data", retry_failed=True)
    assert state.reusable(second, "second-data")
    state.next_attempt(second)
    state.record(second, "second-data", {"status": "success", "loss": 0.2})
    assert state.rows[first]["loss"] == 0.1
    assert len(list((state.directory / "records").glob("*.json"))) == 3


@pytest.mark.parametrize("status", ["budget_exhausted", "interrupted"])
def test_unfinished_cells_resume_without_retry_flag(tmp_path, status):
    state = RunState(tmp_path / "run", {"config": "fixed"})
    state.record("cell", "data", {"status": status})
    assert not state.reusable("cell", "data")


def test_checkpoint_corruption_fails_explicitly(tmp_path):
    state = RunState(tmp_path / "run", {})
    state.record("cell", "data", {"status": "success", "loss": 1.0})
    record = next((state.directory / "records").glob("*.json"))
    record.write_text("{}")
    with pytest.raises(ValueError, match="Corrupt checkpoint"):
        RunState(state.directory, {}, resume=True)


def test_prepared_dataset_drift_fails_on_resume(tmp_path):
    state = RunState(tmp_path / "run", {})
    state.validate_dataset("d", {"sha256": "original", "sample": "fixed"})
    state.elapsed_seconds = 12.0
    state.save()
    restored = RunState(state.directory, {}, resume=True)
    assert restored.elapsed_seconds == 12.0
    with pytest.raises(ValueError, match="dataset identity changed"):
        restored.validate_dataset("d", {"sha256": "changed", "sample": "fixed"})


def test_confirmation_exposure_blocks_other_runs(tmp_path):
    ledger = ExposureLedger(tmp_path / "exposure.sqlite")
    ledger.admit("family", "development-run", "development")
    with pytest.raises(ValueError, match="already exposed"):
        ledger.admit("family", "confirmation-run", "confirmation")
    ledger.admit("new-family", "confirmation-run", "confirmation")
    ledger.admit("new-family", "confirmation-run", "confirmation")


def test_exposure_ledger_closes_database_connections(tmp_path):
    path = tmp_path / "exposure.sqlite"
    ledger = ExposureLedger(path)
    connection = sqlite3.connect(path)
    with patch("benchmarks.run_state.sqlite3.connect", return_value=connection):
        ledger.admit("family", "run", "development")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


def test_archive_retains_exact_content_and_authoritative_parts(tmp_path):
    path = tmp_path / "results.json"
    original = json.dumps({"rows": [{"loss": i / 100} for i in range(500)]}).encode()
    path.write_bytes(original)
    names = archive_results(path, part_bytes=100)
    index = json.loads((tmp_path / "archive.json").read_text())
    assert names == index["parts"]
    assert gzip.decompress(b"".join((tmp_path / name).read_bytes() for name in names)) == original
    assert all((tmp_path / name).stat().st_size <= 100 for name in names)


def test_canonical_hash_handles_key_order_but_not_invalid_numbers():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    with pytest.raises(ValueError):
        fingerprint({"loss": float("nan")})


def test_source_archive_preserves_dirty_code_for_reproduction(tmp_path):
    root, output = tmp_path / "repo", tmp_path / "run"
    root.mkdir()
    output.mkdir()
    source = root / "source.py"
    source.write_text("VALUE = 1\n")
    original = source.read_bytes()
    digest = archive_source(output, root, [source], {"name": "declared-suite"})
    source.write_text("VALUE = 2\n")
    with zipfile.ZipFile(output / "source.zip") as archive:
        assert archive.read("source.py") == original
        assert json.loads(archive.read("resolved-suite.json"))["name"] == "declared-suite"
    assert len(digest) == 64
