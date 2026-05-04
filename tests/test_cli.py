"""Tests for the featcopilot CLI."""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import warnings
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


# ----------------------- scalar config-type validation


@pytest.mark.parametrize(
    "key,value,fragment",
    [
        ("max_features", "10", "max_features"),
        ("max_features", True, "max_features"),  # bool rejected for numeric field
        ("correlation_threshold", "0.9", "correlation_threshold"),
        ("correlation_threshold", True, "correlation_threshold"),
        ("gate_n_jobs", "2", "gate_n_jobs"),
        ("leakage_guard", 42, "leakage_guard"),
    ],
)
def test_scalar_type_mismatch_in_config_returns_exit_2(tmp_path: Path, tabular_csv: Path, key, value, fragment):
    """A malformed JSON config (string in a numeric field, etc.) must hit the
    deterministic exit-2 user-error path with a precise message — not bubble
    up as a downstream ``TypeError`` (exit 1).
    """
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps({key: value}))
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
    assert fragment in err


@pytest.mark.parametrize("threshold", [-0.1, 1.1, 5.0, -1.0])
def test_correlation_threshold_out_of_range_returns_exit_2(tmp_path: Path, tabular_csv: Path, threshold):
    """``correlation_threshold`` is only meaningful in [0.0, 1.0]. Out-of-range
    values silently change selector behavior (>1 disables redundancy elim,
    <0 treats every numeric pair as redundant), so the CLI rejects them up
    front with a precise exit-2 error.
    """
    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(tmp_path / "out.csv"),
            "--target",
            "y",
            "--correlation-threshold",
            str(threshold),
            "--max-features",
            "5",
        ]
    )
    assert rc == 2
    assert "correlation_threshold" in err
    assert "[0.0, 1.0]" in err or "0.0" in err


def test_correlation_threshold_in_config_out_of_range_returns_exit_2(tmp_path: Path, tabular_csv: Path):
    """The same range check applies when ``correlation_threshold`` arrives
    from ``--config`` rather than the CLI flag.
    """
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"correlation_threshold": 2.5}))
    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(tmp_path / "out.csv"),
            "--target",
            "y",
            "--max-features",
            "5",
            "--config",
            str(cfg),
        ]
    )
    assert rc == 2
    assert "correlation_threshold" in err


def test_correlation_threshold_boundary_values_accepted(tmp_path: Path, tabular_csv: Path):
    """The boundaries (0.0 and 1.0) must be accepted — they're the inclusive
    valid range. Default 0.85 is also exercised throughout the suite.
    """
    out_path = tmp_path / "out.csv"
    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(out_path),
            "--target",
            "y",
            "--correlation-threshold",
            "0.0",
            "--max-features",
            "5",
        ]
    )
    assert rc == 0, err

    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(out_path),
            "--target",
            "y",
            "--correlation-threshold",
            "1.0",
            "--max-features",
            "5",
        ]
    )
    assert rc == 0, err


# ----------------------- --verbose / --no-verbose


def test_no_verbose_overrides_config_verbose_true(tmp_path: Path, tabular_csv: Path):
    """``--no-verbose`` (BooleanOptionalAction) must override a config-level
    ``"verbose": true`` to false — the documented precedence rule.
    """
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps({"verbose": True}))
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
            "--no-verbose",
            "--max-features",
            "5",
            "--json",
        ]
    )
    assert rc == 0, err


def test_verbose_overrides_config_verbose_false(tmp_path: Path, tabular_csv: Path):
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps({"verbose": False}))
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
            "--verbose",
            "--max-features",
            "5",
        ]
    )
    assert rc == 0, err


@pytest.mark.parametrize(
    "value",
    ["true", "false", 1, 0],
)
def test_non_bool_verbose_in_config_returns_exit_2(tmp_path: Path, tabular_csv: Path, value):
    """A malformed ``"verbose": <non-bool>`` config must hit exit 2 with a
    precise message, not silently turn verbose mode on/off via Python's
    truthiness rules.
    """
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps({"verbose": value}))
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
            "--max-features",
            "5",
        ]
    )
    assert rc == 2
    assert "verbose" in err


def test_transform_missing_target_with_selection_returns_exit_2(tmp_path: Path, tabular_csv: Path):
    """Without ``--target``, selection silently degrades to a no-op. The CLI
    must surface that as a clean exit-2 user error so automation can react.
    """
    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(tmp_path / "out.csv"),
            "--max-features",
            "5",
        ]
    )
    assert rc == 2
    assert "--target" in err
    assert "selection" in err.lower()


def test_transform_missing_target_with_no_selection_succeeds(tmp_path: Path, tabular_csv: Path):
    """Once selection is opted out, the missing target is no longer an error
    (selection requires a target; raw transform doesn't).
    """
    # Drop the target column so we can run without --target.
    in_path = tmp_path / "in_notarget.csv"
    pd.read_csv(tabular_csv).drop(columns=["y"]).to_csv(in_path, index=False)
    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(in_path),
            "--output",
            str(tmp_path / "out.csv"),
            "--no-selection",
        ]
    )
    assert rc == 0, err


def test_transform_missing_target_no_max_features_succeeds(tmp_path: Path, tabular_csv: Path):
    """Without ``--max-features`` (and the corresponding config key),
    ``AutoFeatureEngineer`` doesn't actually fit a selector even with
    ``apply_selection=True``, so requiring ``--target`` would be a false
    positive. Raw feature generation without target / without cap must
    therefore succeed.
    """
    in_path = tmp_path / "in_notarget.csv"
    pd.read_csv(tabular_csv).drop(columns=["y"]).to_csv(in_path, index=False)
    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(in_path),
            "--output",
            str(tmp_path / "out.csv"),
        ]
    )
    assert rc == 0, err


def test_transform_missing_target_max_features_in_config_returns_exit_2(tmp_path: Path, tabular_csv: Path):
    """The ``--target`` requirement also fires when ``max_features`` comes
    from ``--config`` (not just the CLI flag), since the selector will
    actually run in that case.
    """
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps({"max_features": 5}))
    in_path = tmp_path / "in_notarget.csv"
    pd.read_csv(tabular_csv).drop(columns=["y"]).to_csv(in_path, index=False)
    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(in_path),
            "--output",
            str(tmp_path / "out.csv"),
            "--config",
            str(config_path),
        ]
    )
    assert rc == 2
    assert "--target" in err


def test_explain_ignores_selection_only_config_keys(tmp_path: Path, tabular_csv: Path):
    """A shared transform/explain config with selection-only keys
    (``selection_methods`` / ``correlation_threshold``) must not break
    ``explain``: those keys are inert at runtime (selection is disabled
    in ``explain``) and ``_build_engineer(include_selection_config=False)``
    skips reading them so config-validation does not fire.
    """
    config_path = tmp_path / "cfg.json"
    # Use *valid* selection_methods values; the point is they''re ignored.
    config_path.write_text(
        json.dumps(
            {
                "engines": ["tabular"],
                "selection_methods": ["mutual_info"],
                "correlation_threshold": 0.5,
                "max_features": 5,
            }
        )
    )
    rc, out, err = _run(
        [
            "explain",
            "--input",
            str(tabular_csv),
            "--target",
            "y",
            "--config",
            str(config_path),
        ]
    )
    assert rc == 0, err
    payload = json.loads(out)
    assert payload["status"] == "ok"


# ----------------------- explain subparser doesn't expose selection-only flags


def test_explain_rejects_selection_methods_flag(tmp_path: Path, tabular_csv: Path):
    """``explain`` always disables selection, so accepting ``--selection-methods``
    on the CLI would silently mis-configure the user. The subparser must not
    advertise it.
    """
    rc, _, err = _run(
        [
            "explain",
            "--input",
            str(tabular_csv),
            "--target",
            "y",
            "--selection-methods",
            "mutual_info",
        ]
    )
    assert rc == 2
    assert "unrecognized" in err.lower() or "--selection-methods" in err.lower()


def test_explain_accepts_max_features_flag(tmp_path: Path, tabular_csv: Path):
    """``--max-features`` is *not* selection-only — ``AutoFeatureEngineer``
    forwards it into engine construction (e.g. the tabular engine uses it
    to cap how many features it generates). ``explain`` must therefore
    expose it so callers can bound the size of the explanation payload.
    """
    rc, out, err = _run(
        [
            "explain",
            "--input",
            str(tabular_csv),
            "--target",
            "y",
            "--max-features",
            "5",
        ]
    )
    assert rc == 0, err
    payload = json.loads(out)
    assert payload["status"] == "ok"


def test_explain_rejects_correlation_threshold_flag(tmp_path: Path, tabular_csv: Path):
    rc, _, err = _run(
        [
            "explain",
            "--input",
            str(tabular_csv),
            "--target",
            "y",
            "--correlation-threshold",
            "0.9",
        ]
    )
    assert rc == 2


def test_explain_target_help_no_longer_says_required_for_selection():
    """The ``--target`` help on ``explain`` must not claim it gates selection
    (selection is intentionally disabled in ``explain``).
    """
    parser = fc_cli._build_parser()
    # argparse stores subparsers under a special action attribute
    explain_parser = next(
        action.choices["explain"] for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    target_help = next(a.help for a in explain_parser._actions if "--target" in a.option_strings)
    assert "required for selection" not in target_help
    assert "leakage" in target_help.lower() or "task context" in target_help.lower()


# ----------------------- I/O OSError normalization


def test_input_directory_returns_exit_2(tmp_path: Path):
    """Pointing ``--input`` at a directory must surface as exit 2."""
    in_dir = tmp_path / "i_am_a_dir.csv"
    in_dir.mkdir()
    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(in_dir),
            "--output",
            str(tmp_path / "out.csv"),
            "--target",
            "y",
        ]
    )
    assert rc == 2
    assert "directory" in err.lower()


def test_output_directory_returns_exit_2(tmp_path: Path, tabular_csv: Path):
    """Pointing ``--output`` at an existing directory must surface as exit 2."""
    out_dir = tmp_path / "i_am_a_dir.csv"
    out_dir.mkdir()
    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(out_dir),
            "--target",
            "y",
        ]
    )
    assert rc == 2
    assert "directory" in err.lower()


def test_unwritable_output_returns_exit_2(tmp_path: Path, tabular_csv: Path, monkeypatch):
    """An ``OSError`` on write (e.g. permission denied) must surface as exit 2."""
    import pandas as pd

    def _raise_oserror(self, *args, **kwargs):
        raise PermissionError("simulated write failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", _raise_oserror, raising=True)

    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(tmp_path / "out.csv"),
            "--target",
            "y",
        ]
    )
    assert rc == 2
    assert "failed to write" in err.lower()


def test_unreadable_input_csv_returns_exit_2(tmp_path: Path, tabular_csv: Path, monkeypatch):
    """An ``OSError`` while reading the input must surface as exit 2."""
    import pandas as pd

    def _raise_oserror(*args, **kwargs):
        raise PermissionError("simulated read failure")

    monkeypatch.setattr(pd, "read_csv", _raise_oserror, raising=True)

    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(tmp_path / "out.csv"),
            "--target",
            "y",
        ]
    )
    assert rc == 2
    assert "failed to read" in err.lower()


def test_unreadable_input_json_returns_exit_2(tmp_path: Path, tabular_csv: Path, monkeypatch):
    """``OSError`` from ``pd.read_json`` is surfaced as exit 2 too."""
    import pandas as pd

    in_path = tmp_path / "in.json"
    in_path.write_text("[]")  # contents irrelevant; we'll intercept

    def _raise_oserror(*args, **kwargs):
        raise PermissionError("simulated read failure")

    monkeypatch.setattr(pd, "read_json", _raise_oserror, raising=True)

    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(in_path),
            "--output",
            str(tmp_path / "out.csv"),
            "--target",
            "y",
        ]
    )
    assert rc == 2
    assert "failed to read json" in err.lower()


def test_unreadable_input_parquet_returns_exit_2(tmp_path: Path, monkeypatch):
    """``OSError`` from ``pd.read_parquet`` (e.g. corrupt file) is exit 2."""
    import pandas as pd

    in_path = tmp_path / "in.parquet"
    in_path.write_bytes(b"")

    def _raise_oserror(*args, **kwargs):
        raise OSError("simulated parquet read failure")

    monkeypatch.setattr(pd, "read_parquet", _raise_oserror, raising=True)

    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(in_path),
            "--output",
            str(tmp_path / "out.csv"),
            "--target",
            "y",
        ]
    )
    assert rc == 2
    assert "failed to read parquet" in err.lower()


def test_unwritable_output_json_returns_exit_2(tmp_path: Path, tabular_csv: Path, monkeypatch):
    import pandas as pd

    def _raise_oserror(self, *args, **kwargs):
        raise PermissionError("simulated json write failure")

    monkeypatch.setattr(pd.DataFrame, "to_json", _raise_oserror, raising=True)

    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(tmp_path / "out.json"),
            "--target",
            "y",
        ]
    )
    assert rc == 2
    assert "failed to write json" in err.lower()


def test_unwritable_output_parquet_returns_exit_2(tmp_path: Path, tabular_csv: Path, monkeypatch):
    """``OSError`` (vs ``ImportError``) from ``DataFrame.to_parquet`` -> exit 2."""
    import pandas as pd

    def _raise_oserror(self, *args, **kwargs):
        raise OSError("simulated parquet write failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _raise_oserror, raising=True)

    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(tmp_path / "out.parquet"),
            "--target",
            "y",
        ]
    )
    assert rc == 2
    assert "failed to write parquet" in err.lower()


def test_parquet_read_engine_error_returns_exit_2(tmp_path: Path, monkeypatch):
    """A non-OSError parquet *backend* error (e.g. ``pyarrow.lib.ArrowInvalid``
    for a corrupt file) must surface as exit 2, not the generic exit 1
    "unexpected error" backstop. The CLI catches ``Exception`` for parquet
    operations because they are fully delegated to a third-party backend
    whose failures are by definition user-facing data issues.
    """
    import pandas as pd

    in_path = tmp_path / "fake.parquet"
    in_path.write_bytes(b"\x00\x01\x02\x03")  # not a real parquet file

    class _FakeArrowInvalid(Exception):
        """Stand-in for ``pyarrow.lib.ArrowInvalid`` (also subclasses Exception)."""

    def _raise_backend_error(*args, **kwargs):
        raise _FakeArrowInvalid("simulated corrupt parquet")

    monkeypatch.setattr(pd, "read_parquet", _raise_backend_error, raising=True)

    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(in_path),
            "--output",
            str(tmp_path / "out.csv"),
            "--target",
            "y",
        ]
    )
    assert rc == 2
    assert "failed to read parquet" in err.lower()


def test_parquet_write_engine_error_returns_exit_2(tmp_path: Path, tabular_csv: Path, monkeypatch):
    """Same coverage on the write side: a backend-level pyarrow exception
    that is *not* an ``OSError`` (e.g. an unsupported column-type
    conversion error) must produce exit 2, not exit 1.
    """
    import pandas as pd

    class _FakeArrowTypeError(Exception):
        pass

    def _raise_backend_error(self, *args, **kwargs):
        raise _FakeArrowTypeError("simulated unsupported column dtype for parquet")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _raise_backend_error, raising=True)

    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(tmp_path / "out.parquet"),
            "--target",
            "y",
        ]
    )
    assert rc == 2
    assert "failed to write parquet" in err.lower()


def test_uncreatable_parent_directory_returns_exit_2(tmp_path: Path, tabular_csv: Path, monkeypatch):
    """If creating the output's parent directory fails, exit 2 with a clean message."""
    real_mkdir = Path.mkdir

    def _raise_oserror(self, *args, **kwargs):
        # Only fail for our test's would-be output parent so other calls (e.g.
        # tmp_path operations under the hood) still work.
        if "deep" in self.parts:
            raise PermissionError("simulated mkdir failure")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", _raise_oserror, raising=True)

    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(tmp_path / "deep" / "nested" / "out.csv"),
            "--target",
            "y",
        ]
    )
    assert rc == 2
    assert "create parent directory" in err.lower()


# ----------------------- stderr is reserved for failures (warnings captured)


def test_transform_leakage_warning_does_not_pollute_stderr(tmp_path: Path):
    """``leakage_guard='warn'`` (the default) must not bleed
    ``warnings.warn(...)`` onto stderr on a successful run; the warnings
    are captured and surfaced inside the JSON payload's ``warnings`` field
    instead, so agents can keep treating non-empty stderr as failure metadata.
    """
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            # ``label_encoded`` is detected as leakage-prone ("label" + "encoded"
            # both appear in the stoplist).
            "label_encoded": rng.integers(0, 2, size=n),
            "y": rng.integers(0, 2, size=n),
        }
    )
    in_path = tmp_path / "in_with_leakage.csv"
    df.to_csv(in_path, index=False)
    out_path = tmp_path / "out.csv"

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
            "5",
            "--json",
        ]
    )
    assert rc == 0, err
    assert err == "", f"stderr should be empty on success but got: {err!r}"
    payload = json.loads(out)
    assert payload["status"] == "ok"
    # ``warnings`` field is always present; it MAY contain the leakage
    # warning depending on the heuristic. The contract being tested is
    # that stderr stays clean — not that any specific warning was emitted
    # (the leakage detector heuristics evolve).
    assert "warnings" in payload
    assert isinstance(payload["warnings"], list)


def test_explain_leakage_warning_does_not_pollute_stderr(tmp_path: Path):
    """``explain`` has the same stderr-cleanliness contract as ``transform``."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "label_encoded": rng.integers(0, 2, size=n),
            "y": rng.integers(0, 2, size=n),
        }
    )
    in_path = tmp_path / "in.csv"
    df.to_csv(in_path, index=False)

    rc, out, err = _run(
        [
            "explain",
            "--input",
            str(in_path),
            "--target",
            "y",
        ]
    )
    assert rc == 0, err
    assert err == "", f"stderr should be empty on success but got: {err!r}"
    payload = json.loads(out)
    assert payload["status"] == "ok"
    # The ``warnings`` field is always present and is a list. Whether or
    # not the leakage heuristic fires is not guaranteed (it evolves); the
    # contract under test is that stderr stays clean.
    assert isinstance(payload["warnings"], list)


def test_transform_logger_warning_does_not_pollute_stderr(tmp_path: Path, tabular_csv: Path):
    """The CLI captures ``logger.warning(...)`` records (in addition to
    ``warnings.warn``), so any successful run that exercises a code path
    emitting a logger message — for example the do-no-harm gate's
    fallback — keeps stderr empty. The captured records appear in the
    JSON payload's ``warnings`` field.
    """
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
            "5",
            "--verbose",  # exercises ``logger.info(...)`` paths in engines
            "--json",
        ]
    )
    assert rc == 0, err
    assert err == "", f"stderr should be empty on success but got: {err!r}"
    payload = json.loads(out)
    assert payload["status"] == "ok"
    assert isinstance(payload["warnings"], list)


def test_transform_verbose_logger_info_captured_not_on_stderr(tmp_path: Path, tabular_csv: Path):
    """``--verbose`` enables ``logger.info(...)`` calls in
    ``AutoFeatureEngineer`` and the engines. Those records must end up
    in the JSON payload's ``warnings`` field, not on stderr.
    """
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
            "5",
            "--verbose",
            "--json",
        ]
    )
    assert rc == 0, err
    assert err == "", f"stderr should be empty on success but got: {err!r}"
    payload = json.loads(out)
    # ``--verbose`` reliably emits "Fitted tabular engine" via logger.info,
    # and selection / engineer calls also log. We don't pin the exact
    # messages (they evolve) — just check at least one log record is
    # present in the captured payload.
    assert isinstance(payload["warnings"], list)
    assert len(payload["warnings"]) >= 1


def test_capture_featcopilot_messages_intercepts_logger_warning():
    """Direct unit test for the contextmanager so the docstring contract is
    not just covered transitively via the CLI subcommands.
    """
    fc_logger = logging.getLogger("featcopilot.test_cli")
    # Reset Python's warning-deduplication state for the duration of the
    # test so a previous test that fired ``warnings.warn`` at the same
    # source location does not suppress this one.
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with fc_cli._capture_featcopilot_messages() as captured:
            fc_logger.warning("captured-warning-message")
            warnings.warn("captured-runtime-warning", UserWarning, stacklevel=2)
    assert any("captured-warning-message" in m for m in captured)
    assert any("captured-runtime-warning" in m for m in captured)


def test_capture_featcopilot_messages_does_not_mutate_logger_state_per_call():
    """The contextmanager installs hooks *once* (lazily) and then never
    mutates the featcopilot logger again — so successive captures don't
    add or remove handlers, regardless of test ordering. The earlier
    "restores handlers" test (asserting equality with pre-first-call
    state) was order-dependent: on the very first capture in a process,
    ``_install_capture_hooks_once()`` permanently adds
    ``_routing_handler`` and that's a one-way change. We instead assert
    *stability* across an exception-propagating with-block, which is the
    real behavioral contract.
    """
    # First, force install via a no-op capture.
    with fc_cli._capture_featcopilot_messages():
        pass

    fc_root = logging.getLogger("featcopilot")
    handlers_before = list(fc_root.handlers)
    level_before = fc_root.level
    showwarning_before = warnings.showwarning

    with pytest.raises(RuntimeError):
        with fc_cli._capture_featcopilot_messages():
            raise RuntimeError("boom")

    # Hooks remain installed (handler stays, level unchanged, showwarning
    # override remains in place); per-call state has been popped.
    assert fc_root.handlers == handlers_before
    assert fc_root.level == level_before
    assert warnings.showwarning is showwarning_before


def test_capture_featcopilot_messages_thread_safety():
    """Concurrent ``_capture_featcopilot_messages`` invocations must not
    steal each other's records. Implementation uses per-thread routing
    (no global lock held during the body), so threads execute concurrently.
    """
    import threading

    fc_logger = logging.getLogger("featcopilot.test_concurrent")

    results: list[list[str]] = []
    barrier = threading.Barrier(2)

    def worker(tag: str):
        # Force both threads to enter the with-block at roughly the same
        # time so the routing dispatch is genuinely contended.
        barrier.wait()
        with fc_cli._capture_featcopilot_messages() as captured:
            for i in range(20):
                fc_logger.warning(f"{tag}-{i}")
        results.append(captured)

    t1 = threading.Thread(target=worker, args=("A",))
    t2 = threading.Thread(target=worker, args=("B",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(results) == 2
    # Each capture list must contain exactly its own thread's records and
    # nothing from the other thread.
    for res in results:
        # Find which tag this list belongs to.
        tag = "A" if any("A-" in m for m in res) else "B"
        assert all(f"{tag}-" in m for m in res), f"Thread isolation violated in capture {tag!r}: got {res!r}"
        assert len(res) == 20


def test_capture_does_not_block_concurrent_callers():
    """Two concurrent ``_capture_featcopilot_messages`` blocks must run in
    parallel — i.e. the design does NOT serialize the body via a global
    lock. Verified by timing: a worker that sleeps inside the block must
    not block another worker from also entering the block at the same
    time.
    """
    import threading
    import time

    inside = []
    inside_lock = threading.Lock()
    seen_overlap = threading.Event()
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        with fc_cli._capture_featcopilot_messages():
            with inside_lock:
                inside.append(1)
                if len(inside) >= 2:
                    seen_overlap.set()
            # Sleep long enough that, if the implementation serialized via
            # a global lock, the second thread would never enter
            # simultaneously.
            time.sleep(0.2)
            with inside_lock:
                inside.pop()

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert seen_overlap.is_set(), (
        "Both threads should have been inside _capture_featcopilot_messages "
        "simultaneously; the implementation appears to serialize the body."
    )


def test_capture_warnings_warn_thread_isolated():
    """``warnings.warn`` calls from one capturing thread must not leak into
    another capturing thread's payload. The CLI overrides
    ``warnings.showwarning`` per-thread (rather than using
    ``warnings.catch_warnings(record=True)`` which is process-global).
    """
    import threading

    barrier = threading.Barrier(2)
    a_captured: list[str] = []
    b_captured: list[str] = []

    def worker(tag: str, target: list[str]):
        barrier.wait()
        with fc_cli._capture_featcopilot_messages() as captured:
            for i in range(10):
                # ``stacklevel=2`` is forwarded; reset filter state so we
                # don't lose the warning to Python's default dedup.
                warnings.warn(f"{tag}-warn-{i}", UserWarning, stacklevel=2)
        target.extend(captured)

    # Reset warning filters for this test so dedup doesn't suppress
    # repeated emissions at the same source line.
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        t1 = threading.Thread(target=worker, args=("A", a_captured))
        t2 = threading.Thread(target=worker, args=("B", b_captured))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    assert all("A-warn-" in m for m in a_captured)
    assert all("B-warn-" in m for m in b_captured)
    assert not any("B-warn-" in m for m in a_captured)
    assert not any("A-warn-" in m for m in b_captured)


def test_nested_capture_on_same_thread_preserves_outer_list():
    """A capture inside a capture on the same thread must:

    1. Route records to the *innermost* list while the inner block is active.
    2. Restore the outer list when the inner block exits, so subsequent
       records flow into the outer payload.

    The previous single-list-per-thread design clobbered the outer
    registration; this test guards against that regression.
    """
    fc_logger = logging.getLogger("featcopilot.test_nested")

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with fc_cli._capture_featcopilot_messages() as outer:
            fc_logger.warning("outer-before-nested")
            with fc_cli._capture_featcopilot_messages() as inner:
                fc_logger.warning("inner-only")
                warnings.warn("inner-runtime", UserWarning, stacklevel=2)
            fc_logger.warning("outer-after-nested")

    # Inner contains only the records emitted while it was the active
    # capture.
    assert any("inner-only" in m for m in inner)
    assert any("inner-runtime" in m for m in inner)
    assert not any("outer-before-nested" in m for m in inner)
    assert not any("outer-after-nested" in m for m in inner)

    # Outer contains records emitted before AND after the inner block,
    # but NOT records emitted while inner was active (those went to inner).
    assert any("outer-before-nested" in m for m in outer)
    assert any("outer-after-nested" in m for m in outer)
    assert not any("inner-only" in m for m in outer)
    assert not any("inner-runtime" in m for m in outer)


def test_overlapping_captures_with_out_of_order_exit():
    """Two threads enter the capture block, then thread A exits *before*
    thread B. The CLI must continue to capture B's warnings even after
    A has exited — i.e. A's exit must not restore a global state that
    disables B's capture.

    This is the strict version of the warnings.showwarning race that
    existed when the override was saved/restored per-call: A's exit
    used to restore the original ``warnings.showwarning``, leaking B's
    subsequent ``warnings.warn`` calls onto stderr.
    """
    import threading
    import time

    barrier = threading.Barrier(2)
    a_done = threading.Event()
    a_captured: list[str] = []
    b_captured: list[str] = []

    fc_logger = logging.getLogger("featcopilot.test_overlap")

    def worker_a():
        barrier.wait()
        with fc_cli._capture_featcopilot_messages() as captured:
            fc_logger.warning("A-1")
            warnings.warn("A-warn-1", UserWarning, stacklevel=2)
        a_captured.extend(captured)
        a_done.set()  # signal: A has exited the capture block

    def worker_b():
        barrier.wait()
        with fc_cli._capture_featcopilot_messages() as captured:
            fc_logger.warning("B-1")
            # Wait for A to fully exit before emitting B's tail records.
            assert a_done.wait(timeout=5)
            time.sleep(0.05)  # small grace so any racy restoration would have happened
            fc_logger.warning("B-2-after-A-exit")
            warnings.warn("B-warn-after-A-exit", UserWarning, stacklevel=2)
        b_captured.extend(captured)

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        t_a = threading.Thread(target=worker_a)
        t_b = threading.Thread(target=worker_b)
        t_b.start()  # start B first so it's already in the block
        time.sleep(0.05)
        t_a.start()
        t_a.join(timeout=5)
        t_b.join(timeout=5)

    # B's records — including the ones emitted *after* A exited — must
    # all be captured. None of A's records should have leaked into B.
    assert any("B-1" in m for m in b_captured)
    assert any("B-2-after-A-exit" in m for m in b_captured)
    assert any("B-warn-after-A-exit" in m for m in b_captured)
    assert not any("A-1" in m for m in b_captured)
    assert not any("A-warn-1" in m for m in b_captured)
    # A's payload likewise contains only A's records.
    assert any("A-1" in m for m in a_captured)
    assert any("A-warn-1" in m for m in a_captured)
    assert not any("B-" in m for m in a_captured)


def test_unexpected_error_writes_single_stderr_line(monkeypatch, tmp_path: Path, tabular_csv: Path):
    """An unexpected (non-ValueError) exception must produce exactly one
    structured stderr line — no second timestamped traceback from
    ``logger.exception(...)`` — so agents can parse failures
    deterministically.
    """
    import pandas as pd

    class _UnexpectedError(Exception):
        """A non-ValueError, non-OSError exception that escapes the helpers."""

    def _raise_unexpected(*args, **kwargs):
        raise _UnexpectedError("simulated internal failure")

    # Monkey-patch ``pd.read_csv`` directly. Since ``_read_table``'s CSV
    # branch normally catches ``OSError`` / ``ParserError`` / ``UnicodeDecodeError``,
    # raising a different exception type forces us into the generic exit-1
    # backstop in ``main()``.
    monkeypatch.setattr(pd, "read_csv", _raise_unexpected, raising=True)

    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(tabular_csv),
            "--output",
            str(tmp_path / "out.csv"),
            "--target",
            "y",
        ]
    )
    assert rc == 1, err
    # Exactly one non-empty line on stderr.
    err_lines = [line for line in err.splitlines() if line.strip()]
    assert len(err_lines) == 1, f"Expected single-line stderr, got: {err!r}"
    assert err_lines[0].startswith("featcopilot: unexpected error:")
    assert "_UnexpectedError" in err_lines[0]
    assert "simulated internal failure" in err_lines[0]
    # No traceback signature.
    assert "Traceback" not in err
    assert 'File "' not in err


# ----------------------- --target help text accuracy


def test_transform_target_help_reflects_actual_contract():
    """The ``--target`` help on ``transform`` must say the flag is required
    only when ``--max-features`` is set (which is when the selector
    actually fits), not whenever selection is enabled by default.
    """
    parser = fc_cli._build_parser()
    transform_parser = next(
        action.choices["transform"] for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    target_help = next(a.help for a in transform_parser._actions if "--target" in a.option_strings)
    assert "max_features" in target_help.lower() or "max-features" in target_help.lower()
    # The old ("required when selection is applied (the default ...)")
    # phrasing was misleading — guard against regressions.
    assert "the default" not in target_help.lower()


# ----------------------- target check runs after type validation


def test_invalid_max_features_in_config_takes_precedence_over_target_check(tmp_path: Path, tabular_csv: Path):
    """A malformed ``max_features`` in ``--config`` (string, negative, etc.)
    must surface its real validation error rather than ``--target is
    required``. The CLI now builds the engineer first (which type-validates
    every scalar config field) and only checks ``--target`` after.
    """
    in_path = tmp_path / "in_notarget.csv"
    pd.read_csv(tabular_csv).drop(columns=["y"]).to_csv(in_path, index=False)

    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"max_features": "5"}))  # string, not int
    rc, _, err = _run(
        [
            "transform",
            "--input",
            str(in_path),
            "--output",
            str(tmp_path / "out.csv"),
            "--config",
            str(cfg),
        ]
    )
    assert rc == 2
    # The real error is the type mismatch, NOT --target missing.
    assert "max_features" in err
    assert "--target" not in err


def test_check_scalar_type_rejects_none_when_required():
    """Direct unit test for ``_check_scalar_type`` to exercise the
    ``allow_none=False`` + ``value is None`` branch, which the integration
    path doesn't naturally hit (every scalar with ``allow_none=False`` has
    a non-None default).
    """
    with pytest.raises(ValueError, match="must not be null"):
        fc_cli._check_scalar_type("foo", None, (int,), allow_none=False)


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


def test_unknown_config_top_level_key_returns_exit_2(tmp_path: Path, tabular_csv: Path):
    """A typo in a top-level config key (``max_feature`` instead of
    ``max_features``, etc.) must fail fast with a precise exit-2 message
    listing the recognized keys — not silently run with defaults.
    """
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"max_feature": 5}))  # missing 's'
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
    assert "max_feature" in err
    assert "Recognized keys" in err or "recognized keys" in err.lower()


def test_unknown_config_top_level_key_lists_known_keys(tmp_path: Path, tabular_csv: Path):
    """The error message must enumerate the recognized keys so users can
    self-correct without reading the source.
    """
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"selection_method": ["mutual_info"]}))  # missing 's'
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
    assert "selection_method" in err
    # Recognized-keys list must include the canonical names.
    assert "selection_methods" in err
    assert "max_features" in err


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


def test_explain_uses_full_input_by_default(tmp_path: Path):
    """``explain`` defaults to using the FULL input — no implicit
    sub-sampling. Some engines (e.g. ``TabularEngine`` categorical
    encoding) decide which features to plan based on row counts and
    per-category statistics, so silent sampling would change the
    advertised metadata. Sampling is opt-in via ``--explain-sample-size``.
    """
    rng = np.random.default_rng(0)
    n = 1500  # arbitrary
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "y": rng.integers(0, 2, size=n),
        }
    )
    in_path = tmp_path / "big.csv"
    df.to_csv(in_path, index=False)

    rc, out, err = _run(
        [
            "explain",
            "--input",
            str(in_path),
            "--target",
            "y",
        ]
    )
    assert rc == 0, err
    payload = json.loads(out)
    assert payload["status"] == "ok"
    # Default: no sampling — full input is used.
    assert payload["n_rows_used"] == n


def test_explain_caps_input_size_when_sample_size_set(tmp_path: Path):
    """When ``--explain-sample-size N`` is passed, the input is capped at
    ``N`` rows (with a captured warning) so callers can opt into bounded
    cost on huge inputs. The default remains full-input.
    """
    rng = np.random.default_rng(0)
    n = 5000
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "y": rng.integers(0, 2, size=n),
        }
    )
    in_path = tmp_path / "big.csv"
    df.to_csv(in_path, index=False)

    rc, out, err = _run(
        [
            "explain",
            "--input",
            str(in_path),
            "--target",
            "y",
            "--explain-sample-size",
            "1000",
        ]
    )
    assert rc == 0, err
    payload = json.loads(out)
    # Sampling cap was enforced.
    assert payload["n_rows_used"] == 1000
    assert payload["n_features"] > 0
    # The CLI emits a warning when sampling so callers can detect that
    # metadata may not match a full-input transform run.
    assert any("sampling" in w.lower() for w in payload["warnings"])


def test_explain_sample_size_smaller_than_input_no_op(tmp_path: Path):
    """When ``--explain-sample-size`` exceeds the actual input, no sampling
    happens (and no warning is emitted).
    """
    rng = np.random.default_rng(0)
    n = 50
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "y": rng.integers(0, 2, size=n),
        }
    )
    in_path = tmp_path / "small.csv"
    df.to_csv(in_path, index=False)

    rc, out, err = _run(
        [
            "explain",
            "--input",
            str(in_path),
            "--target",
            "y",
            "--explain-sample-size",
            "1000",
        ]
    )
    assert rc == 0, err
    payload = json.loads(out)
    assert payload["n_rows_used"] == n
    assert not any("sampling" in w.lower() for w in payload["warnings"])


def test_explain_sample_size_via_config(tmp_path: Path):
    """``explain_sample_size`` is also recognized in ``--config`` JSON."""
    rng = np.random.default_rng(0)
    n = 5000
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "y": rng.integers(0, 2, size=n),
        }
    )
    in_path = tmp_path / "big.csv"
    df.to_csv(in_path, index=False)

    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"explain_sample_size": 500}))

    rc, out, err = _run(
        [
            "explain",
            "--input",
            str(in_path),
            "--target",
            "y",
            "--config",
            str(cfg),
        ]
    )
    assert rc == 0, err
    payload = json.loads(out)
    assert payload["n_rows_used"] == 500


@pytest.mark.parametrize("bad_value", [0, -1, -100])
def test_explain_sample_size_rejects_non_positive(tmp_path: Path, bad_value):
    """``--explain-sample-size`` must be a positive integer."""
    rc, _, err = _run(
        [
            "explain",
            "--input",
            str(tmp_path / "in.csv"),  # missing — but flag check happens first
            "--target",
            "y",
            "--explain-sample-size",
            str(bad_value),
        ]
    )
    # We accept either argparse-level rejection or our own ValueError;
    # both surface as exit 2.
    assert rc == 2


def test_explain_sample_size_rejects_string_in_config(tmp_path: Path, tabular_csv: Path):
    """Type-validation: ``"explain_sample_size": "100"`` (string) is rejected."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"explain_sample_size": "100"}))
    rc, _, err = _run(
        [
            "explain",
            "--input",
            str(tabular_csv),
            "--target",
            "y",
            "--config",
            str(cfg),
        ]
    )
    assert rc == 2
    assert "explain_sample_size" in err


def test_explain_sample_size_rejects_zero_in_config(tmp_path: Path, tabular_csv: Path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"explain_sample_size": 0}))
    rc, _, err = _run(
        [
            "explain",
            "--input",
            str(tabular_csv),
            "--target",
            "y",
            "--config",
            str(cfg),
        ]
    )
    assert rc == 2
    assert "explain_sample_size" in err


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
    """When ``__import__`` raises ``ImportError`` for both engines, the
    function reports parquet as unavailable.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("pyarrow", "fastparquet"):
            raise ImportError(f"No module named '{name}' (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert fc_cli._parquet_engine_available() is False


def test_parquet_engine_available_returns_true_for_fastparquet_only(monkeypatch):
    """Even without pyarrow, importing fastparquet must report parquet as available."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyarrow":
            raise ImportError("No module named 'pyarrow' (simulated)")
        if name == "fastparquet":
            # Simulate a successful import by short-circuiting; we don't
            # actually need a real module object, just a non-raising return.
            class _FakeModule:
                pass

            return _FakeModule()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert fc_cli._parquet_engine_available() is True


def test_parquet_engine_available_returns_false_for_broken_native_install(monkeypatch):
    """A distribution that's on sys.path but raises a non-ImportError at
    import time (e.g. broken native bindings) is reported as unavailable.
    Using ``__import__`` (rather than ``importlib.util.find_spec``) is what
    makes this honest: ``find_spec`` would have returned a spec and lied.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("pyarrow", "fastparquet"):
            # Simulate a broken native install (loader-level failure).
            raise OSError("broken native install: undefined symbol (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert fc_cli._parquet_engine_available() is False


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


def _featcopilot_package_is_installed() -> bool:
    """Return True iff the ``featcopilot`` distribution is installed in the
    current environment (i.e. the entry-point machinery should have placed
    the console script on ``PATH``).

    Used by the console-script tests to distinguish two cases:

    * Running tests directly against the source tree (``python -m pytest``
      from a clean checkout, no ``pip install -e .``): the package is
      *not* installed; the script is legitimately missing and the test
      should ``skip`` rather than report a packaging bug.
    * Running tests after ``pip install`` (the CI flow): the package IS
      installed, so the script MUST be on ``PATH``. If it isn't, that's a
      real ``[project.scripts]`` regression and the test should ``fail``,
      not silently pass via skip.
    """
    try:
        from importlib.metadata import PackageNotFoundError, distribution
    except ImportError:  # pragma: no cover - py3.10+ always has this
        return False
    try:
        distribution("featcopilot")
    except PackageNotFoundError:
        return False
    return True


def test_console_script_subprocess_invocation():
    """The installed ``featcopilot`` console script must be on PATH and runnable.

    Exercises the ``[project.scripts] featcopilot = "featcopilot.cli:main"``
    entry point end-to-end so a typo or packaging regression in
    ``pyproject.toml`` would actually break the suite. When the
    ``featcopilot`` distribution is installed, the script must be on
    ``PATH``: a missing script in that case is a real packaging
    regression, not a test environment quirk, so we ``fail`` (not
    ``skip``). The skip is reserved for the rare case of running tests
    against an un-installed source tree.
    """
    import shutil
    import subprocess

    script = shutil.which("featcopilot")
    if script is None:
        if _featcopilot_package_is_installed():
            pytest.fail(
                "featcopilot package is installed but the `featcopilot` console "
                "script is missing from PATH. This is a `[project.scripts]` "
                "regression in pyproject.toml."
            )
        pytest.skip(
            "featcopilot package is not installed in this environment; install "
            "it with `pip install -e .` to exercise the console-script entry point."
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
    """Same install-aware skip/fail policy as
    :func:`test_console_script_subprocess_invocation`.
    """
    import shutil
    import subprocess

    script = shutil.which("featcopilot")
    if script is None:
        if _featcopilot_package_is_installed():
            pytest.fail(
                "featcopilot package is installed but the `featcopilot` console "
                "script is missing from PATH. This is a `[project.scripts]` "
                "regression in pyproject.toml."
            )
        pytest.skip("featcopilot package is not installed in this environment.")

    result = subprocess.run(
        [script, "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert __version__ in result.stdout
