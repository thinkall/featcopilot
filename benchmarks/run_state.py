"""Durable, fingerprinted benchmark cells and family-exposure bookkeeping."""

from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ResumeMismatchError(ValueError):
    """A completed experiment identity no longer matches its checkpoint."""


def fingerprint(value: Any) -> str:
    """Hash a canonical JSON value without accepting non-finite numbers."""
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(content).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Write complete JSON with atomic replacement and a final newline."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def archive_results(path: Path, part_bytes: int = 480 * 1024) -> list[str]:
    """Archive exact result bytes in bounded gzip parts for version control."""
    if part_bytes < 1:
        raise ValueError("part_bytes must be positive")
    content = gzip.compress(path.read_bytes(), mtime=0)
    names = []
    for index, offset in enumerate(range(0, len(content), part_bytes), start=1):
        part = path.with_name(f"{path.name}.gz.part{index}")
        part.write_bytes(content[offset : offset + part_bytes])
        names.append(part.name)
    atomic_json(
        path.with_name("archive.json"),
        {
            "format": "concatenated-gzip-v1",
            "parts": names,
            "uncompressed_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    )
    return names


def archive_source(directory: Path, root: Path, paths: list[Path], suite: dict) -> str:
    """Retain the exact allowlisted source, including dirty files, for reproduction."""
    target = directory / "source.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, arcname=path.relative_to(root).as_posix())
        archive.writestr("resolved-suite.json", json.dumps(suite, indent=2, allow_nan=False))
    return hashlib.sha256(target.read_bytes()).hexdigest()


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
        """Prevent reused families from silently becoming fresh confirmation."""
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

    def __init__(self, directory: Path, identity: dict, *, resume: bool = False):
        self.directory = directory
        self.path = directory / "state.json"
        self.identity_hash = fingerprint(identity)
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
        self.identity = identity
        self.save()

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
        """Only reuse cells under identical inputs; unfinished cells always run."""
        previous = self.rows.get(cell_id)
        if previous is None:
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
            },
        )
