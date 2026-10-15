"""Durable, fingerprinted benchmark cells and family-exposure bookkeeping."""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import sqlite3
import time
import zipfile
from collections.abc import Callable
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class BenchmarkProtocolError(ValueError):
    """The controller received an inconsistent plan or isolated-cell response."""


class ResumeMismatchError(ValueError):
    """A completed experiment identity no longer matches its checkpoint."""


def fingerprint(value: Any) -> str:
    """Hash a canonical JSON value without accepting non-finite numbers."""
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(content).hexdigest()


def _replace_file(source: Path, target: Path) -> None:
    for attempt in range(3):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == 2:
                raise
            # Windows sharing violations can briefly outlast a competing rename.
            time.sleep(0.01 * (attempt + 1))


def atomic_json(path: Path, value: Any) -> None:
    """Write complete JSON with atomic replacement and a final newline."""
    content = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def archive_results(path: Path, part_bytes: int = 480 * 1024) -> list[str]:
    """Publish immutable gzip parts, atomically switching their authoritative index.

    Notes
    -----
    Only filenames listed in archive.json belong to the current archive.
    Older or interrupted generations are retained and must never be globbed
    into a reconstruction. Existing historical archives are not modified.
    """
    if part_bytes < 1:
        raise ValueError("part_bytes must be positive")
    original = path.read_bytes()
    content = gzip.compress(original, mtime=0)
    generation = hashlib.sha256(content).hexdigest()
    names = []
    for index, offset in enumerate(range(0, len(content), part_bytes), start=1):
        part = path.with_name(f"{path.name}.gz.{generation}.{part_bytes}.part{index}")
        chunk = content[offset : offset + part_bytes]
        if part.exists():
            if part.read_bytes() != chunk:
                raise ValueError(f"Existing immutable archive part is corrupt: {part.name}")
        else:
            temporary = part.with_name(f"{part.name}.{uuid4().hex}.tmp")
            try:
                with temporary.open("xb") as stream:
                    stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
                _replace_file(temporary, part)
            finally:
                temporary.unlink(missing_ok=True)
        names.append(part.name)
    atomic_json(
        path.with_name("archive.json"),
        {
            "format": "concatenated-gzip-v1",
            "parts": names,
            "uncompressed_sha256": hashlib.sha256(original).hexdigest(),
        },
    )
    return names


def archive_source(directory: Path, root: Path, paths: list[Path], suite: dict) -> str:
    """Atomically publish the exact source ZIP without truncating an existing snapshot."""
    target = directory / "source.zip"
    temporary = directory / f"source.zip.{uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in paths:
                    archive.write(path, arcname=path.relative_to(root).as_posix())
                archive.writestr("resolved-suite.json", json.dumps(suite, indent=2, allow_nan=False))
            stream.flush()
            os.fsync(stream.fileno())
        checksum = hashlib.sha256(temporary.read_bytes()).hexdigest()
        _replace_file(temporary, target)
        return checksum
    finally:
        temporary.unlink(missing_ok=True)


def _has_run_evidence(directory: Path, checkpoint: dict) -> bool:
    return any(checkpoint.get(key) for key in ("records", "attempts", "datasets")) or any(
        (directory / name).exists()
        for name in ("results.json", "archive.json", "REPORT.md", "records", "cells", "prepared")
    )


def load_run_manifest(directory: Path) -> dict:
    """Read a manifest or the durable bootstrap of an otherwise empty initialization."""
    path = directory / "manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    checkpoint = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    initialization = checkpoint.get("initialization") or {}
    bootstrap = initialization.get("manifest")
    if not isinstance(bootstrap, dict) or _has_run_evidence(directory, checkpoint):
        raise ResumeMismatchError("Cannot resume: missing manifest is not a recoverable, empty initialization")
    return bootstrap.copy()


class ExposureLedger:
    """Record dataset-family exposure atomically without modifying source manifests."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with closing(sqlite3.connect(path, timeout=30)) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS exposure "
                "(family_id TEXT NOT NULL, run_id TEXT NOT NULL, role TEXT NOT NULL, created_at TEXT NOT NULL, "
                "PRIMARY KEY (family_id, run_id))"
            )

    def admit(self, family_id: str, run_id: str, role: str) -> None:
        """Atomically admit a family; reserve confirmation before any payload read.

        Confirmation callers retain the reservation even if preparation fails,
        because failure cannot establish that protected data was never opened.
        Retries by the same run remain permitted.
        """
        with closing(sqlite3.connect(self.path, timeout=30)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                "SELECT run_id FROM exposure WHERE family_id = ? AND run_id != ?", (family_id, run_id)
            ).fetchall()
            if role == "confirmation" and previous:
                raise ValueError(f"Confirmation family {family_id!r} was already exposed in another run")
            connection.execute(
                "INSERT OR IGNORE INTO exposure VALUES (?, ?, ?, ?)",
                (family_id, run_id, role, datetime.now(timezone.utc).isoformat()),
            )

    def check(self, family_id: str, run_id: str, role: str) -> None:
        """Check protected-family exposure before opening its data payload."""
        if role != "confirmation":
            return
        with closing(sqlite3.connect(self.path, timeout=30)) as connection:
            previous = connection.execute(
                "SELECT 1 FROM exposure WHERE family_id = ? AND run_id != ? LIMIT 1", (family_id, run_id)
            ).fetchone()
        if previous is not None:
            raise ValueError(f"Confirmation family {family_id!r} was already exposed in another run")


class RunState:
    """Persist independent result cells and validate exact resumability."""

    def __init__(self, directory: Path, identity: dict, *, resume: bool = False, initialization: dict | None = None):
        self.directory = directory
        self.path = directory / "state.json"
        self.identity_hash = fingerprint(identity)
        self._resuming = resume
        if resume:
            if not self.path.exists():
                raise ValueError("Resume requires a protocol-v3 state.json; legacy result files cannot be resumed")
            state = json.loads(self.path.read_text(encoding="utf-8"))
            if state["identity_sha256"] != self.identity_hash:
                raise ResumeMismatchError(
                    "Cannot resume: code, suite, dependencies or semantic configuration changed. Use a new output directory."
                )
            self.run_id: str = state["run_id"]
            self.elapsed_seconds: float = state["elapsed_seconds"]
            self.records: dict[str, dict] = state["records"]
            self.rows: dict[str, dict] = {}
            for cell_id, record in self.records.items():
                path = directory / "records" / record["file"]
                if path.parent.resolve() != (directory / "records").resolve():
                    raise ValueError("Invalid checkpoint record path")
                content = path.read_bytes()
                if hashlib.sha256(content).hexdigest() != record["sha256"]:
                    raise ValueError(f"Corrupt checkpoint record for cell {cell_id}")
                self.rows[cell_id] = json.loads(content)
            self.attempts: dict[str, int] = state["attempts"]
            self.datasets: dict[str, dict] = state.get("datasets", {})
            self.initialization: dict | None = state.get("initialization")
        else:
            directory.mkdir(parents=True, exist_ok=False)
            self.run_id = fingerprint(
                {"identity": self.identity_hash, "created": datetime.now(timezone.utc).isoformat()}
            )
            self.elapsed_seconds = 0.0
            self.rows = {}
            self.records = {}
            self.attempts = {}
            self.datasets = {}
            self.initialization = (
                {
                    "status": "pending",
                    "manifest": {
                        **initialization,
                        "run_id": self.run_id,
                        "started_utc": datetime.now(timezone.utc).isoformat(),
                        "source_archive_sha256": None,
                        "initialization_status": "pending",
                    },
                }
                if initialization is not None
                else None
            )
        self.identity = identity
        if not resume:
            self.save()

    def initialize_source(
        self,
        manifest: dict,
        root: Path,
        paths: list[Path],
        suite: dict,
        *,
        started: float,
        archive_writer: Callable = archive_source,
    ) -> bool:
        """Publish source once, or recover an exact-identity initialization without evidence.

        Returns
        -------
        bool
            Whether initialization completed. Failure/interrupt time and the
            recovery configuration are persisted before returning false.
        """
        phase = self.initialization or {}
        expected = manifest.get("source_archive_sha256")
        evidence = _has_run_evidence(
            self.directory, {"records": self.records, "attempts": self.attempts, "datasets": self.datasets}
        )
        if phase.get("status") == "ready" or (self._resuming and not phase):
            recorded = phase.get("source_archive_sha256", expected)
            if expected is None and not evidence and phase.get("status") == "ready":
                expected = recorded
            if expected is None or expected != recorded:
                raise ResumeMismatchError("Cannot resume: source archive identity is unavailable or changed")
            target = self.directory / "source.zip"
            if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ResumeMismatchError("Cannot resume: source archive changed")
            manifest.update(source_archive_sha256=expected, initialization_status="ready", initialization_error=None)
            if manifest.get("run_status", "").startswith("initialization_"):
                manifest.pop("run_status")
            return True
        if evidence:
            raise ResumeMismatchError(
                "Cannot recover source initialization after preparation, attempts or result evidence"
            )
        prior_elapsed = self.elapsed_seconds
        status, error, checksum = "pending", None, None
        manifest.update(
            source_archive_sha256=None,
            initialization_status=status,
            initialization_error=None,
            source_allowlist=[path.relative_to(root).as_posix() for path in paths],
            run_status="initialization_pending",
        )
        self.initialization = {"status": status, "manifest": manifest.copy()}
        self.save()
        try:
            atomic_json(self.directory / "manifest.json", manifest)
            if prior_elapsed + time.perf_counter() - started >= manifest["budget_seconds"]:
                status, error = "budget_exhausted", "Cumulative budget exhausted before source initialization"
            else:
                checksum = archive_writer(self.directory, root, paths, suite)
                status = "ready"
        except KeyboardInterrupt:
            status, error = "interrupted", "Source initialization interrupted"
            logger.warning(error)
        except Exception as exc:
            status, error = "error", f"{type(exc).__name__}: {exc}"
            logger.exception("Source initialization failed")
        finally:
            self.elapsed_seconds = prior_elapsed + time.perf_counter() - started
            manifest.update(
                source_archive_sha256=checksum,
                initialization_status=status,
                initialization_error=error,
                active_seconds=self.elapsed_seconds,
                budget_overrun_seconds=max(0.0, self.elapsed_seconds - manifest["budget_seconds"]),
            )
            if status == "ready":
                manifest.pop("run_status", None)
            else:
                manifest["run_status"] = f"initialization_{status}"
            self.initialization = (
                {"status": "ready", "source_archive_sha256": checksum}
                if status == "ready"
                else {"status": status, "manifest": manifest.copy()}
            )
            self.save()
            atomic_json(self.directory / "manifest.json", manifest)
        return status == "ready"

    @staticmethod
    def cell_id(dataset: str, seed: int, fold: int, method: str, learner: str) -> str:
        """Return a stable identifier for one requested evaluation cell."""
        return fingerprint([dataset, seed, fold, method, learner])[:24]

    def validate_dataset(self, name: str, metadata: dict) -> None:
        """Reject changed dataset bytes, sampling or split policy on resume."""
        previous = self.datasets.get(name)
        if previous is not None and fingerprint(previous) != fingerprint(metadata):
            raise ResumeMismatchError(f"Cannot resume: prepared dataset identity changed for {name!r}")
        self.datasets[name] = metadata

    def reusable(self, cell_id: str, cell_fingerprint: str, *, retry_failed: bool = False) -> bool:
        """Require identical prepared inputs; unprepared interruptions can acquire them."""
        previous = self.rows.get(cell_id)
        if previous is None:
            return False
        if (
            previous["status"] in {"budget_exhausted", "interrupted"}
            and previous.get("preparation_state") == "unprepared"
            and previous.get("split_sha256") == "unavailable"
        ):
            return False
        if previous["status"] == "dataset_error" and retry_failed:
            return False
        if previous.get("cell_fingerprint") != cell_fingerprint:
            raise ResumeMismatchError(f"Cannot resume: inputs or split changed for cell {cell_id}")
        if previous["status"] in {"budget_exhausted", "interrupted"}:
            return False
        return previous["status"] == "success" or not retry_failed

    def next_attempt(self, cell_id: str) -> Path:
        """Reserve a new scratch directory; never overwrite a previous attempt."""
        number = self.attempts.get(cell_id, 0) + 1
        self.attempts[cell_id] = number
        self.save()
        return self.directory / "cells" / cell_id / f"attempt-{number}"

    def record(self, cell_id: str, cell_fingerprint: str, row: dict) -> None:
        """Checkpoint a completed or explicitly unfinished cell."""
        self.rows[cell_id] = {
            **row,
            "cell_id": cell_id,
            "cell_fingerprint": cell_fingerprint,
            "attempt": self.attempts.get(cell_id, 0),
        }
        content_hash = fingerprint(self.rows[cell_id])
        record_dir = self.directory / "records"
        record_dir.mkdir(exist_ok=True)
        path = record_dir / f"{cell_id}-{content_hash[:16]}.json"
        atomic_json(path, self.rows[cell_id])
        self.records[cell_id] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.save()

    def save(self) -> None:
        """Persist resumable operational state independently of report rendering."""
        atomic_json(
            self.path,
            {
                "state_version": 1,
                "identity_sha256": self.identity_hash,
                "identity": self.identity,
                "run_id": self.run_id,
                "elapsed_seconds": self.elapsed_seconds,
                "records": self.records,
                "attempts": self.attempts,
                "datasets": self.datasets,
                "initialization": self.initialization,
            },
        )
