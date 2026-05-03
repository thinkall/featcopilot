"""Tests for the featcopilot CLI."""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from featcopilot import __version__
from featcopilot import cli as fc_cli


def _run(argv: list[str]) -> tuple[int, str, str]:
    """Invoke ``cli.main(argv)`` and capture exit code, stdout, stderr."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = fc_cli.main(argv)
    return rc, out.getvalue(), err.getvalue()


@pytest.fixture
def tabular_csv(tmp_path: Path) -> Path:
    """A small classification dataset written to CSV."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.integers(0, 5, size=n),
            "y": rng.integers(0, 2, size=n),
        }
    )
    path = tmp_path / "in.csv"
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------- info


def test_info_json_emits_supported_options():
    rc, out, err = _run(["info", "--json"])
    assert rc == 0, err
    payload = json.loads(out)
    assert payload["version"] == __version__
    assert "tabular" in payload["supported_engines"]
    assert "mutual_info" in payload["supported_selection_methods"]
    assert "warn" in payload["supported_leakage_guards"]
    # CSV/JSON are always supported; parquet is gated on engine availability.
    assert {"csv", "json"} <= set(payload["supported_input_formats"])
    assert {"csv", "json"} <= set(payload["supported_output_formats"])
    assert isinstance(payload["parquet_available"], bool)
    if payload["parquet_available"]:
        assert "parquet" in payload["supported_input_formats"]
        assert "parquet" in payload["supported_output_formats"]
    else:
        assert "parquet" not in payload["supported_input_formats"]
        assert "parquet" not in payload["supported_output_formats"]


def test_info_excludes_parquet_when_engine_missing(monkeypatch):
    """When no parquet engine can be imported, ``info`` must not advertise it."""
    monkeypatch.setattr(fc_cli, "_parquet_engine_available", lambda: False)
    rc, out, _ = _run(["info", "--json"])
    assert rc == 0
    payload = json.loads(out)
    assert payload["parquet_available"] is False
    assert "parquet" not in payload["supported_input_formats"]
    assert "parquet" not in payload["supported_output_formats"]


def test_info_includes_parquet_when_engine_present(monkeypatch):
    monkeypatch.setattr(fc_cli, "_parquet_engine_available", lambda: True)
    rc, out, _ = _run(["info", "--json"])
    assert rc == 0
    payload = json.loads(out)
    assert payload["parquet_available"] is True
    assert "parquet" in payload["supported_input_formats"]
    assert "parquet" in payload["supported_output_formats"]


def test_info_text_mode_is_human_readable():
    rc, out, _ = _run(["info"])
    assert rc == 0
    # Not JSON: parsing should fail.
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)
    assert "version" in out
    assert __version__ in out


def test_top_level_version_flag(capsys):
    # ``--version`` (argparse action) prints to stdout; main() now traps the
    # SystemExit and returns the code so the API contract is consistent.
    rc = fc_cli.main(["--version"])
    assert rc == 0
    assert __version__ in capsys.readouterr().out


# ----------------------------------------------------------------- transform


def test_transform_csv_to_csv(tmp_path: Path, tabular_csv: Path):
    out_path = tmp_path / "out.csv"
    rc, out, err = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(out_path),
            "--target",
            "y",
            "--max-features",
            "10",
            "--json",
        ]
    )
    assert rc == 0, err
    payload = json.loads(out)
    assert payload["status"] == "ok"
    assert payload["target"] == "y"
    assert payload["engines"] == ["tabular"]
    assert payload["selection_applied"] is True
    assert payload["n_input_columns"] == 3  # x1, x2, x3 (y is the target)

    # The output file exists and is readable as CSV.
    assert out_path.exists()
    written = pd.read_csv(out_path)
    assert written.shape[0] == 200
    assert "y" not in written.columns  # target excluded by default


def test_transform_include_target_round_trip(tmp_path: Path, tabular_csv: Path):
    out_path = tmp_path / "out.csv"
    rc, _, err = _run(
        [
            "transform",
            "-i",
            str(tabular_csv),
            "-o",
            str(out_path),
            "-t",
            "y",
            "--max-features",
            "10",
            "--include-target",
        ]
    )
    assert rc == 0, err
    written = pd.read_csv(out_path)
    assert "y" in written.columns


def test_transform_parquet_round_trip(tmp_path: Path):
    pytest.importorskip("pyarrow")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=120), "b": rng.normal(size=120), "y": rng.integers(0, 2, size=120)})
    in_path = tmp_path / "in.parquet"
    out_path = tmp_path / "out.parquet"
    df.to_parquet(in_path, index=False)

    rc, out, err = _run(
        [
            "transform",
            "--input",
            str(in_path),
            "--output",
            str(out_path),
            "--target",
            "y",
            "--max-features",
            "8",
            "--json",
        ]
    )
    assert rc == 0, err
    payload = json.loads(out)
    assert payload["input_format"] == "parquet"
    assert payload["output_format"] == "parquet"
    pd.read_parquet(out_path)  # readable


def test_transform_json_round_trip(tmp_path: Path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=80), "b": rng.normal(size=80), "y": rng.integers(0, 2, size=80)})
    in_path = tmp_path / "in.json"
    out_path = tmp_path / "out.json"
    df.to_json(in_path, orient="records")

    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(in_path),
            "--output",
            str(out_path),
            "--target",
            "y",
        ]
    )
    assert rc == 0, err
    written = pd.read_json(out_path, orient="records")
    assert written.shape[0] == 80


def test_transform_no_selection_skips_selector(tmp_path: Path, tabular_csv: Path):
    out_path = tmp_path / "out.csv"
    rc, out, err = _run(
        [
            "transform",
            "-i",
            str(tabular_csv),
            "-o",
            str(out_path),
            "-t",
            "y",
            "--no-selection",
            "--max-features",
            "5",
            "--json",
        ]
    )
    assert rc == 0, err
    payload = json.loads(out)
    assert payload["selection_applied"] is False


def test_transform_config_file_supplies_engineer_kwargs(tmp_path: Path, tabular_csv: Path):
    config_path = tmp_path / "cfg.json"
    config_path.write_text(
        json.dumps(
            {
                "engines": ["tabular"],
                "selection_methods": ["mutual_info"],
                "max_features": 7,
                "correlation_threshold": 0.9,
                "leakage_guard": "off",
            }
        )
    )
    out_path = tmp_path / "out.csv"
    rc, out, err = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(out_path),
            "--target",
            "y",
            "--config",
            str(config_path),
            "--json",
        ]
    )
    assert rc == 0, err
    payload = json.loads(out)
    assert payload["selection_methods"] == ["mutual_info"]
    assert payload["max_features"] == 7


def test_transform_cli_flags_override_config(tmp_path: Path, tabular_csv: Path):
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps({"max_features": 5, "engines": ["tabular"]}))
    out_path = tmp_path / "out.csv"
    rc, out, _ = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(out_path),
            "--target",
            "y",
            "--config",
            str(config_path),
            "--max-features",
            "12",
            "--json",
        ]
    )
    assert rc == 0
    assert json.loads(out)["max_features"] == 12


# ----------------------- _build_engineer config validation


def test_string_engines_in_config_returns_clean_exit_2(tmp_path: Path, tabular_csv: Path):
    """A misconfigured ``"engines": "tabular"`` (string instead of list) must
    surface ``AutoFeatureEngineer``'s precise type-validation error via the
    standard exit-2 path — *not* be silently coerced into a per-character list.
    """
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps({"engines": "tabular"}))
    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(tmp_path / "out.csv"),
            "--target",
            "y",
            "--config",
            str(config_path),
        ]
    )
    assert rc == 2
    assert "engines must be a list or tuple" in err


def test_empty_engines_list_in_config_returns_clean_exit_2(tmp_path: Path, tabular_csv: Path):
    """An explicit empty ``engines`` list in the config must propagate to the
    transformer's validation so the user sees the documented error, instead
    of being silently rewritten into the defaults.
    """
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps({"engines": []}))
    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(tmp_path / "out.csv"),
            "--target",
            "y",
            "--config",
            str(config_path),
        ]
    )
    assert rc == 2
    assert "at least one engine" in err.lower() or "empty sequence" in err.lower()


def test_empty_selection_methods_list_in_config_returns_clean_exit_2(tmp_path: Path, tabular_csv: Path):
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps({"selection_methods": []}))
    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(tmp_path / "out.csv"),
            "--target",
            "y",
            "--config",
            str(config_path),
        ]
    )
    assert rc == 2
    assert "at least one method" in err.lower() or "empty sequence" in err.lower()


# -------------------------------------------------------------- error paths


def test_transform_missing_input_returns_exit_2(tmp_path: Path):
    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(tmp_path / "nope.csv"),
            "--output",
            str(tmp_path / "out.csv"),
            "--target",
            "y",
        ]
    )
    assert rc == 2
    assert "Input file not found" in err


def test_transform_unknown_target_returns_exit_2(tmp_path: Path, tabular_csv: Path):
    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(tmp_path / "out.csv"),
            "--target",
            "does_not_exist",
        ]
    )
    assert rc == 2
    assert "does_not_exist" in err


def test_transform_unknown_extension_without_override(tmp_path: Path, tabular_csv: Path):
    out_path = tmp_path / "out.weird"
    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(out_path),
            "--target",
            "y",
        ]
    )
    assert rc == 2
    assert "infer format" in err.lower()


def test_transform_format_override_accepted(tmp_path: Path, tabular_csv: Path):
    out_path = tmp_path / "out.weird"
    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(out_path),
            "--target",
            "y",
            "--output-format",
            "csv",
        ]
    )
    assert rc == 0, err
    assert out_path.exists()


def test_invalid_config_file_returns_exit_2(tmp_path: Path, tabular_csv: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2, 3]")  # JSON, but not an object
    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(tmp_path / "o.csv"),
            "--target",
            "y",
            "--config",
            str(bad),
        ]
    )
    assert rc == 2
    assert "JSON object" in err


def test_directory_as_config_returns_exit_2(tmp_path: Path, tabular_csv: Path):
    """Pointing ``--config`` at a directory must surface as exit 2, not the
    generic ``exit 1`` backstop (``IsADirectoryError``).
    """
    cfg_dir = tmp_path / "not_a_file"
    cfg_dir.mkdir()
    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(tmp_path / "o.csv"),
            "--target",
            "y",
            "--config",
            str(cfg_dir),
        ]
    )
    assert rc == 2
    assert "directory" in err.lower()


def test_malformed_json_config_returns_exit_2(tmp_path: Path, tabular_csv: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json,}")
    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(tmp_path / "o.csv"),
            "--target",
            "y",
            "--config",
            str(bad),
        ]
    )
    assert rc == 2
    assert "valid json" in err.lower()


def test_non_dict_llm_config_returns_exit_2(tmp_path: Path, tabular_csv: Path):
    """A non-mapping ``llm_config`` (e.g. a string) must be rejected at
    config-load time with a clean exit 2, not bubble up as an
    ``AttributeError`` from ``.get(...)`` deep inside engine construction.
    """
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"engines": ["tabular"], "llm_config": "gpt-5"}))
    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(tmp_path / "o.csv"),
            "--target",
            "y",
            "--config",
            str(cfg),
        ]
    )
    assert rc == 2
    assert "llm_config" in err
    assert "JSON object" in err or "mapping" in err.lower()


def test_no_subcommand_exits_nonzero(capsys):
    # main() now returns the argparse-reported exit code (2 for usage error)
    # rather than letting SystemExit propagate, so programmatic callers get
    # an integer back even on parse-time failures.
    rc = fc_cli.main([])
    assert rc == 2


def test_unknown_flag_returns_exit_2(capsys):
    rc = fc_cli.main(["transform", "--no-such-flag"])
    assert rc == 2


def test_help_flag_returns_zero(capsys):
    rc = fc_cli.main(["--help"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "featcopilot" in captured.out


# ------------------------------------------------------------------ explain


def test_explain_emits_json_payload(tmp_path: Path, tabular_csv: Path):
    rc, out, err = _run(
        [
            "explain",
            "--input",
            str(tabular_csv),
            "--target",
            "y",
        ]
    )
    assert rc == 0, err
    payload = json.loads(out)
    assert payload["status"] == "ok"
    assert payload["engines"] == ["tabular"]
    assert isinstance(payload["features"], list)
    # The tabular engine actually generates derived features, and the explain
    # subcommand must materialize them by running the full fit_transform
    # pipeline (engines populate _feature_names during transform()).
    assert payload["n_features"] > 0
    assert len(payload["features"]) == payload["n_features"]
    # Each feature entry is a dict with the expected keys.
    entry = payload["features"][0]
    assert {"name", "explanation", "code"} <= set(entry.keys())
    assert entry["name"]


# --------------------------------------------------------------- parquet path


def test_transform_parquet_missing_engine_returns_exit_2(tmp_path, tabular_csv, monkeypatch):
    """When pyarrow/fastparquet is missing, the CLI should surface a clean
    user-facing dependency error (exit 2) rather than the generic exit 1
    backstop.
    """
    import pandas as pd

    def _raise_import_error(self, *args, **kwargs):  # noqa: ANN001
        raise ImportError("Missing optional dependency 'pyarrow' (simulated)")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _raise_import_error, raising=True)

    out_path = tmp_path / "out.parquet"
    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(out_path),
            "--target",
            "y",
            "--max-features",
            "5",
        ]
    )
    assert rc == 2
    assert "parquet engine" in err.lower()


def test_transform_read_parquet_missing_engine_returns_exit_2(tmp_path, tabular_csv, monkeypatch):
    """Symmetric coverage for reading a .parquet input when no engine is installed.

    The CLI must convert the ``ImportError`` from ``pd.read_parquet`` into
    the deterministic exit-2 path (with a user-facing install hint),
    just like the write path.
    """
    import pandas as pd

    # Make sure the input path has a .parquet suffix so format detection picks parquet.
    fake_pq = tmp_path / "fake.parquet"
    fake_pq.write_bytes(b"")  # contents don't matter; we'll intercept read_parquet

    def _raise_import_error(*args, **kwargs):
        raise ImportError("Missing optional dependency 'pyarrow' (simulated)")

    monkeypatch.setattr(pd, "read_parquet", _raise_import_error, raising=True)

    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(fake_pq),
            "--output",
            str(tmp_path / "out.csv"),
            "--target",
            "y",
        ]
    )
    assert rc == 2
    assert "parquet engine" in err.lower()


def test_parquet_engine_available_returns_false_when_neither_installed(monkeypatch):
    """Both probes return ``None`` from ``find_spec`` -> function returns False."""
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name in ("pyarrow", "fastparquet"):
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    assert fc_cli._parquet_engine_available() is False


def test_parquet_engine_available_returns_true_for_fastparquet_only(monkeypatch):
    """Even without pyarrow, finding fastparquet must report parquet as available."""
    import importlib.util

    class _FakeSpec:
        pass

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "pyarrow":
            return None
        if name == "fastparquet":
            return _FakeSpec()
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    assert fc_cli._parquet_engine_available() is True


def test_unreadable_config_returns_exit_2(tmp_path, tabular_csv, monkeypatch):
    """An ``OSError`` while opening the config (permission denied, broken
    symlink, etc.) is converted into the deterministic exit-2 path.
    """
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}")

    real_open = Path.open

    def _raise_oserror(self, *args, **kwargs):
        if self == cfg:
            raise PermissionError("simulated read failure")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _raise_oserror, raising=True)

    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(tmp_path / "out.csv"),
            "--target",
            "y",
            "--config",
            str(cfg),
        ]
    )
    assert rc == 2
    assert "could not be read" in err.lower()


# --------------------------------------------------------------- python -m


def test_dunder_main_module_runs(monkeypatch, capsys):
    """``cli.main`` is invoked via the same code path as ``python -m featcopilot``."""
    monkeypatch.setattr(sys, "argv", ["featcopilot", "info", "--json"])
    rc = fc_cli.main(["info", "--json"])
    assert rc == 0


def test_dunder_main_subprocess_invocation():
    """``python -m featcopilot info --json`` must succeed in a real subprocess.

    Exercises ``featcopilot/__main__.py`` end-to-end so a regression in
    module-form invocation (e.g. a broken import path) actually breaks the
    test, not just the unit-level call to ``cli.main``.
    """
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "featcopilot", "info", "--json"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["version"] == __version__
    assert "tabular" in payload["supported_engines"]


def test_dunder_main_subprocess_version_flag():
    """``python -m featcopilot --version`` must print and exit 0."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "featcopilot", "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert __version__ in result.stdout


# ------------------------------------------------------- console script


def test_console_script_subprocess_invocation():
    """The installed ``featcopilot`` console script must be on PATH and runnable.

    Exercises the ``[project.scripts] featcopilot = "featcopilot.cli:main"``
    entry point end-to-end so a typo or packaging regression in
    ``pyproject.toml`` would actually break the suite. Skipped when the
    script isn't on ``PATH`` (e.g. running tests without ``pip install``).
    """
    import shutil
    import subprocess

    script = shutil.which("featcopilot")
    if script is None:
        pytest.skip(
            "featcopilot console script not on PATH (install the package "
            "with `pip install -e .` to exercise the entry point)"
        )

    result = subprocess.run(
        [script, "info", "--json"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["version"] == __version__
    assert "tabular" in payload["supported_engines"]


def test_console_script_version_flag():
    import shutil
    import subprocess

    script = shutil.which("featcopilot")
    if script is None:
        pytest.skip("featcopilot console script not on PATH")

    result = subprocess.run(
        [script, "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert __version__ in result.stdout
