"""Durability, exact-resume and budget identity tests for benchmark execution."""

import gzip
import hashlib
import json
import sqlite3
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PureWindowsPath
from threading import Barrier
from unittest.mock import patch

import pytest

from benchmarks import run_state
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


@pytest.mark.parametrize("status", ["budget_exhausted", "interrupted"])
def test_unprepared_unfinished_cells_can_obtain_real_input_identity(tmp_path, status):
    state = RunState(tmp_path / "run", {"config": "fixed"})
    state.record(
        "cell", "unavailable", {"status": status, "preparation_state": "unprepared", "split_sha256": "unavailable"}
    )
    restored = RunState(state.directory, state.identity, resume=True)
    assert not restored.reusable("cell", "prepared-data")
    restored.record("cell", "prepared-data", {"status": "success", "loss": 0.5})
    with pytest.raises(ValueError, match="inputs or split"):
        restored.reusable("cell", "changed-data")


@pytest.mark.parametrize("status", ["budget_exhausted", "interrupted", "success", "timeout"])
def test_prepared_cells_still_reject_changed_inputs(tmp_path, status):
    state = RunState(tmp_path / "run", {})
    state.record("cell", "original-data", {"status": status})
    with pytest.raises(ValueError, match="inputs or split"):
        state.reusable("cell", "changed-data", retry_failed=True)


@pytest.mark.parametrize("status", ["success", "timeout", "dataset_error"])
def test_preparation_marker_does_not_relax_terminal_cell_identity(tmp_path, status):
    state = RunState(tmp_path / "run", {})
    state.record("cell", "original-data", {"status": status, "preparation_state": "unprepared"})
    with pytest.raises(ValueError, match="inputs or split"):
        state.reusable("cell", "changed-data")


@pytest.mark.parametrize("status", ["interrupted", "budget_exhausted"])
def test_unprepared_marker_cannot_bypass_an_already_known_split(tmp_path, status):
    state = RunState(tmp_path / "run", {})
    state.record(
        "cell", "original-data", {"status": status, "preparation_state": "unprepared", "split_sha256": "known-split"}
    )
    with pytest.raises(ValueError, match="inputs or split"):
        state.reusable("cell", "changed-data")


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


@pytest.mark.parametrize("failure", ["part", "index"])
def test_archive_publication_failure_keeps_previous_index_and_parts_valid(tmp_path, monkeypatch, failure):
    path = tmp_path / "results.json"
    original = bytes(range(256)) * 4
    path.write_bytes(original)
    previous_names = archive_results(path, part_bytes=40)
    previous_index = (tmp_path / "archive.json").read_bytes()
    previous_parts = {name: (tmp_path / name).read_bytes() for name in previous_names}
    replacement = bytes(reversed(range(256))) * 5
    path.write_bytes(replacement)
    replace = Path.replace
    atomic = run_state.atomic_json
    published = []

    def fail_part(source, target):
        if ".part" in target.name:
            published.append(target.name)
            if len(published) == 2:
                raise OSError("Synthetic part publication failure")
        return replace(source, target)

    def fail_index(target, value):
        if target.name == "archive.json":
            raise OSError("Synthetic index publication failure")
        return atomic(target, value)

    with monkeypatch.context() as patcher:
        if failure == "part":
            patcher.setattr(Path, "replace", fail_part)
        else:
            patcher.setattr(run_state, "atomic_json", fail_index)
        with pytest.raises(OSError, match="Synthetic"):
            archive_results(path, part_bytes=40)
    assert (tmp_path / "archive.json").read_bytes() == previous_index
    assert all((tmp_path / name).read_bytes() == content for name, content in previous_parts.items())
    assert gzip.decompress(b"".join((tmp_path / name).read_bytes() for name in previous_names)) == original
    current = archive_results(path, part_bytes=40)
    assert set(current).isdisjoint(previous_names)
    index = json.loads((tmp_path / "archive.json").read_text())
    assert index["parts"] == current
    assert gzip.decompress(b"".join((tmp_path / name).read_bytes() for name in index["parts"])) == replacement
    assert index["uncompressed_sha256"] == hashlib.sha256(replacement).hexdigest()


def test_archive_generations_are_immutable_idempotent_and_index_selected(tmp_path):
    path = tmp_path / "results.json"
    path.write_bytes(bytes(range(256)) * 4)
    old = archive_results(path, part_bytes=40)
    before = {name: ((tmp_path / name).read_bytes(), (tmp_path / name).stat().st_mtime_ns) for name in old}
    assert archive_results(path, part_bytes=40) == old
    assert all(
        ((tmp_path / name).read_bytes(), (tmp_path / name).stat().st_mtime_ns) == value
        for name, value in before.items()
    )
    path.write_bytes(b"small replacement")
    current = archive_results(path, part_bytes=40)
    assert set(current).isdisjoint(old)
    assert all((tmp_path / name).read_bytes() == value[0] for name, value in before.items())
    index = json.loads((tmp_path / "archive.json").read_text())
    assert gzip.decompress(b"".join((tmp_path / name).read_bytes() for name in index["parts"])) == b"small replacement"


def test_archive_hash_and_compression_use_one_input_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "results.json"
    path.write_bytes(b"first")
    read_bytes = Path.read_bytes
    reads = []

    def changing_file(candidate):
        if candidate == path:
            reads.append(None)
            return b"first" if len(reads) == 1 else b"changed"
        return read_bytes(candidate)

    monkeypatch.setattr(Path, "read_bytes", changing_file)
    names = archive_results(path)
    index = json.loads((tmp_path / "archive.json").read_text())
    assert len(reads) == 1
    assert gzip.decompress(b"".join((tmp_path / name).read_bytes() for name in names)) == b"first"
    assert index["uncompressed_sha256"] == hashlib.sha256(b"first").hexdigest()


def test_concurrent_archive_publishers_use_independent_index_staging(tmp_path, monkeypatch):
    path = tmp_path / "results.json"
    original = bytes(range(256)) * 4
    path.write_bytes(original)
    replace = Path.replace
    gate = Barrier(2)
    staged = set()

    def publish(source, target):
        if target.name == "archive.json" and source not in staged:
            staged.add(source)
            gate.wait(timeout=10)
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", publish)
    with ThreadPoolExecutor(max_workers=2) as pool:
        attempts = [pool.submit(archive_results, path, size) for size in (40, 80)]
        generations = [attempt.result(timeout=30) for attempt in attempts]
    index = json.loads((tmp_path / "archive.json").read_text())
    assert index["parts"] in generations
    assert gzip.decompress(b"".join((tmp_path / name).read_bytes() for name in index["parts"])) == original
    assert all((tmp_path / name).exists() for generation in generations for name in generation)


@pytest.mark.parametrize("persistent", [False, True])
def test_atomic_index_rename_retries_are_bounded_and_preserve_prior_content(tmp_path, monkeypatch, persistent):
    path = tmp_path / "archive.json"
    run_state.atomic_json(path, {"parts": ["previous"]})
    original = path.read_bytes()
    replace = Path.replace
    attempts = []
    delays = []

    def sharing_violation(source, target):
        attempts.append(None)
        if persistent or len(attempts) < 3:
            raise PermissionError("Synthetic sharing violation")
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", sharing_violation)
    monkeypatch.setattr(run_state.time, "sleep", delays.append)
    if persistent:
        with pytest.raises(PermissionError, match="sharing violation"):
            run_state.atomic_json(path, {"parts": ["replacement"]})
        assert path.read_bytes() == original
    else:
        run_state.atomic_json(path, {"parts": ["replacement"]})
        assert json.loads(path.read_text()) == {"parts": ["replacement"]}
    assert len(attempts) == 3
    assert delays == [0.01, 0.02]


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


def test_initialization_source_allowlist_matches_zip_names_with_windows_separators(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    source = root / "pkg" / "mod.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n")
    relative_to = Path.relative_to

    def windows_relative(path, *args, **kwargs):
        relative = relative_to(path, *args, **kwargs)
        return PureWindowsPath(*relative.parts) if path == source else relative

    monkeypatch.setattr(Path, "relative_to", windows_relative)
    assert str(source.relative_to(root)) == "pkg\\mod.py"
    state = RunState(tmp_path / "run", {"code": "fixed"}, initialization={"budget_seconds": 100})
    manifest = {"budget_seconds": 100}
    assert state.initialize_source(manifest, root, [source], {}, started=run_state.time.perf_counter())
    with zipfile.ZipFile(state.directory / "source.zip") as archive:
        members = set(archive.namelist()) - {"resolved-suite.json"}
    assert manifest["source_allowlist"] == ["pkg/mod.py"]
    assert set(manifest["source_allowlist"]) == members


@pytest.mark.parametrize("interrupted", [False, True])
def test_source_archive_failure_never_truncates_previous_snapshot(tmp_path, monkeypatch, interrupted):
    root, output = tmp_path / "repo", tmp_path / "run"
    root.mkdir()
    output.mkdir()
    source = root / "source.py"
    source.write_text("VALUE = 1\n")
    original_source = source.read_bytes()
    archive_source(output, root, [source], {"name": "original"})
    previous = (output / "source.zip").read_bytes()
    source.write_text("VALUE = 2\n")

    def fail(*args, **kwargs):
        raise KeyboardInterrupt if interrupted else OSError("Synthetic source write failure")

    monkeypatch.setattr(zipfile.ZipFile, "writestr", fail)
    with pytest.raises(KeyboardInterrupt if interrupted else OSError):
        archive_source(output, root, [source], {"name": "replacement"})
    assert (output / "source.zip").read_bytes() == previous
    with zipfile.ZipFile(output / "source.zip") as archive:
        assert archive.read("source.py") == original_source


def test_run_state_bootstrap_can_restore_missing_manifest_without_accepting_changed_identity(tmp_path):
    directory = tmp_path / "run"
    identity = {"config": {"methods": ["baseline"]}, "code": "fixed"}
    state = RunState(directory, identity, initialization={"config": identity["config"], "budget_seconds": 100})
    assert not (directory / "manifest.json").exists()
    manifest = run_state.load_run_manifest(directory)
    assert manifest["budget_seconds"] == 100
    assert manifest["config"] == identity["config"]
    assert manifest["run_id"] == state.run_id
    before = (directory / "state.json").read_bytes()
    with pytest.raises(ValueError, match="Cannot resume"):
        RunState(directory, {**identity, "code": "changed"}, resume=True)
    assert (directory / "state.json").read_bytes() == before
