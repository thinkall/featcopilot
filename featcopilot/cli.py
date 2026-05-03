"""
FeatCopilot command-line interface.

Provides a stable, agent-friendly CLI for invoking FeatCopilot from shells,
notebooks, agentic workflows (e.g. Copilot/LLM tool-use), and CI pipelines
without writing Python glue code.

Subcommands
-----------
info
    Print version and supported engines/methods. Always machine-readable
    when ``--json`` is passed.
transform
    Run :class:`featcopilot.AutoFeatureEngineer` on a tabular input file
    (CSV / Parquet / JSON) and write engineered features to an output file.
    Emits a JSON status line on stdout when ``--json`` is passed so that
    agents can parse the result deterministically.
explain
    Fit the engineer and print a JSON document describing each generated
    feature (name, explanation, code) for downstream LLM consumption.

Examples
--------
Agentic usage (machine-readable result on stdout, errors on stderr)::

    featcopilot info --json
    featcopilot transform \\
        --input data.csv --target label --output features.parquet \\
        --engines tabular --max-features 50 --json
    featcopilot explain --input data.csv --target label --json

Equivalent module invocation::

    python -m featcopilot info --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from featcopilot import __version__
from featcopilot.transformers.sklearn_compat import AutoFeatureEngineer
from featcopilot.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_INPUT_FORMATS = ("csv", "parquet", "json")
SUPPORTED_OUTPUT_FORMATS = ("csv", "parquet", "json")


def _detect_format(path: Path, override: str | None) -> str:
    """Return one of ``SUPPORTED_INPUT_FORMATS`` for ``path``.

    Parameters
    ----------
    path : pathlib.Path
        File path whose suffix is inspected when ``override`` is ``None``.
    override : str or None
        Explicit format override (``csv`` / ``parquet`` / ``json``).

    Raises
    ------
    ValueError
        If the format cannot be determined or is not supported.
    """
    if override is not None:
        fmt = override.lower()
        if fmt not in SUPPORTED_INPUT_FORMATS:
            raise ValueError(
                f"Unsupported format {override!r}; expected one of {SUPPORTED_INPUT_FORMATS}"
            )
        return fmt

    suffix = path.suffix.lower().lstrip(".")
    aliases = {"pq": "parquet", "parq": "parquet"}
    fmt = aliases.get(suffix, suffix)
    if fmt not in SUPPORTED_INPUT_FORMATS:
        raise ValueError(
            f"Cannot infer format from extension {path.suffix!r}; "
            f"pass --input-format / --output-format (one of {SUPPORTED_INPUT_FORMATS})."
        )
    return fmt


def _read_table(path: Path, fmt: str):
    """Read a tabular file into a pandas DataFrame."""
    import pandas as pd

    if fmt == "csv":
        return pd.read_csv(path)
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "json":
        # ``orient='records'`` is the agent-friendly default; fall back to
        # pandas' auto-detection when the file isn't a records list.
        try:
            return pd.read_json(path, orient="records")
        except ValueError:
            return pd.read_json(path)
    raise ValueError(f"Unsupported input format: {fmt}")


def _write_table(df, path: Path, fmt: str) -> None:
    """Write a pandas DataFrame to ``path`` in ``fmt``."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "json":
        df.to_json(path, orient="records", indent=2)
    else:
        raise ValueError(f"Unsupported output format: {fmt}")


def _load_config(config_path: str | None) -> dict[str, Any]:
    """Load a JSON config file (or return an empty dict)."""
    if config_path is None:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"Config file {config_path!r} must contain a JSON object at the top level"
        )
    return data


def _emit(payload: dict[str, Any], *, as_json: bool, stream=None) -> None:
    """Emit a payload to stdout, JSON-encoded when ``as_json`` is true."""
    stream = stream if stream is not None else sys.stdout
    if as_json:
        stream.write(json.dumps(payload, default=str, sort_keys=True))
        stream.write("\n")
    else:
        for key, value in payload.items():
            stream.write(f"{key}: {value}\n")
    stream.flush()


def _build_engineer(args: argparse.Namespace) -> AutoFeatureEngineer:
    """Construct an :class:`AutoFeatureEngineer` from parsed CLI args.

    Precedence: explicit CLI flags override values from ``--config``.
    """
    config = _load_config(args.config)

    def pick(flag_value, config_key, default):
        if flag_value is not None:
            return flag_value
        return config.get(config_key, default)

    engines = pick(args.engines, "engines", None) or ["tabular"]
    selection_methods = pick(args.selection_methods, "selection_methods", None) or [
        "mutual_info",
        "importance",
    ]
    max_features = pick(args.max_features, "max_features", None)
    correlation_threshold = pick(args.correlation_threshold, "correlation_threshold", 0.85)
    leakage_guard = pick(args.leakage_guard, "leakage_guard", "warn")
    gate_n_jobs = pick(args.gate_n_jobs, "gate_n_jobs", 1)
    llm_config = config.get("llm_config", {}) or {}
    verbose = bool(pick(args.verbose, "verbose", False))

    return AutoFeatureEngineer(
        engines=list(engines),
        max_features=max_features,
        selection_methods=list(selection_methods),
        correlation_threshold=correlation_threshold,
        llm_config=llm_config,
        verbose=verbose,
        leakage_guard=leakage_guard,
        gate_n_jobs=gate_n_jobs,
    )


def _split_xy(df, target: str | None):
    """Split a DataFrame into ``(X, y)``; ``y`` is ``None`` when no target."""
    if target is None:
        return df, None
    if target not in df.columns:
        raise ValueError(
            f"Target column {target!r} not found in input. "
            f"Available columns: {list(df.columns)[:20]}{'...' if len(df.columns) > 20 else ''}"
        )
    y = df[target]
    X = df.drop(columns=[target])
    return X, y


def _cmd_info(args: argparse.Namespace) -> int:
    """Print version + supported engines/methods."""
    payload = {
        "version": __version__,
        "supported_engines": sorted(AutoFeatureEngineer.SUPPORTED_ENGINES),
        "supported_selection_methods": sorted(AutoFeatureEngineer.SUPPORTED_SELECTION_METHODS),
        "supported_leakage_guards": sorted(AutoFeatureEngineer.SUPPORTED_LEAKAGE_GUARDS),
        "supported_input_formats": list(SUPPORTED_INPUT_FORMATS),
        "supported_output_formats": list(SUPPORTED_OUTPUT_FORMATS),
    }
    _emit(payload, as_json=args.json)
    return 0


def _cmd_transform(args: argparse.Namespace) -> int:
    """Read input, fit/transform, write output."""
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    output_path = Path(args.output)

    in_fmt = _detect_format(input_path, args.input_format)
    out_fmt = _detect_format(output_path, args.output_format)

    df = _read_table(input_path, in_fmt)
    X, y = _split_xy(df, args.target)

    engineer = _build_engineer(args)
    transformed = engineer.fit_transform(
        X,
        y,
        task_description=args.task_description or "prediction task",
        target_name=args.target,
        apply_selection=not args.no_selection,
    )

    if args.include_target and y is not None:
        # Re-attach the target column so downstream training scripts can
        # consume the engineered file as a single artifact.
        target_name = args.target if args.target in df.columns else "target"
        transformed = transformed.copy()
        transformed[target_name] = y.values

    _write_table(transformed, output_path, out_fmt)

    payload = {
        "status": "ok",
        "input": str(input_path),
        "output": str(output_path),
        "input_format": in_fmt,
        "output_format": out_fmt,
        "n_rows": int(transformed.shape[0]),
        "n_features": int(transformed.shape[1]),
        "n_input_columns": int(X.shape[1]),
        "n_generated_features": len(engineer.get_feature_names()),
        "engines": list(engineer.engines),
        "selection_methods": list(engineer.selection_methods),
        "max_features": engineer.max_features,
        "target": args.target,
        "selection_applied": engineer._selector is not None,
    }
    _emit(payload, as_json=args.json)
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    """Fit engines and print feature explanations + code as JSON."""
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    in_fmt = _detect_format(input_path, args.input_format)
    df = _read_table(input_path, in_fmt)
    X, y = _split_xy(df, args.target)

    engineer = _build_engineer(args)
    engineer.fit(
        X,
        y,
        task_description=args.task_description or "prediction task",
        target_name=args.target,
    )

    explanations = engineer.explain_features()
    code = engineer.get_feature_code()
    feature_names = engineer.get_feature_names()

    payload = {
        "status": "ok",
        "input": str(input_path),
        "n_features": len(feature_names),
        "engines": list(engineer.engines),
        "features": [
            {
                "name": name,
                "explanation": explanations.get(name, ""),
                "code": code.get(name, ""),
            }
            for name in feature_names
        ],
    }

    # explain always emits JSON to stdout (it's the only sensible format),
    # but we still respect ``--json`` for symmetry with other subcommands.
    _emit(payload, as_json=True)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="featcopilot",
        description=(
            "FeatCopilot CLI — automated feature engineering from the command line. "
            "Designed for scripting and agentic usage; pass --json to any subcommand "
            "for machine-readable stdout."
        ),
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"featcopilot {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # ----- info ---------------------------------------------------------
    p_info = subparsers.add_parser(
        "info",
        help="Print version and supported engines/methods.",
        description="Print the installed FeatCopilot version and the supported engines, "
        "selection methods, leakage guards, and I/O formats.",
    )
    p_info.add_argument("--json", action="store_true", help="Emit JSON to stdout.")
    p_info.set_defaults(func=_cmd_info)

    # ----- transform ----------------------------------------------------
    p_transform = subparsers.add_parser(
        "transform",
        help="Run feature engineering on a tabular file.",
        description="Read INPUT, run AutoFeatureEngineer, and write engineered features to OUTPUT.",
    )
    _add_io_args(p_transform)
    _add_engineer_args(p_transform)
    p_transform.add_argument(
        "--no-selection",
        action="store_true",
        help="Disable feature selection (skip do-no-harm gate).",
    )
    p_transform.add_argument(
        "--include-target",
        action="store_true",
        help="Include the target column in the output file.",
    )
    p_transform.add_argument("--json", action="store_true", help="Emit a JSON status line on stdout.")
    p_transform.set_defaults(func=_cmd_transform)

    # ----- explain ------------------------------------------------------
    p_explain = subparsers.add_parser(
        "explain",
        help="Print JSON feature explanations and code for agent consumption.",
        description="Fit AutoFeatureEngineer on INPUT and emit a JSON document "
        "describing each generated feature (name, explanation, code).",
    )
    p_explain.add_argument("--input", "-i", required=True, help="Path to input file (CSV / Parquet / JSON).")
    p_explain.add_argument("--input-format", choices=SUPPORTED_INPUT_FORMATS, help="Override input format detection.")
    p_explain.add_argument("--target", "-t", help="Target column name (required for selection).")
    p_explain.add_argument(
        "--task-description",
        help="Natural-language ML task description (used by the LLM engine).",
    )
    _add_engineer_args(p_explain)
    p_explain.add_argument("--json", action="store_true", help="(Always JSON — flag accepted for symmetry.)")
    p_explain.set_defaults(func=_cmd_explain)

    return parser


def _add_io_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", "-i", required=True, help="Path to input file (CSV / Parquet / JSON).")
    p.add_argument("--output", "-o", required=True, help="Path to output file (CSV / Parquet / JSON).")
    p.add_argument("--input-format", choices=SUPPORTED_INPUT_FORMATS, help="Override input format detection.")
    p.add_argument("--output-format", choices=SUPPORTED_OUTPUT_FORMATS, help="Override output format detection.")
    p.add_argument("--target", "-t", help="Target column name (required for selection).")
    p.add_argument(
        "--task-description",
        help="Natural-language ML task description (used by the LLM engine).",
    )


def _add_engineer_args(p: argparse.ArgumentParser) -> None:
    """Add ``AutoFeatureEngineer``-related flags to a subparser."""
    p.add_argument(
        "--engines",
        nargs="+",
        choices=sorted(AutoFeatureEngineer.SUPPORTED_ENGINES),
        help="Engines to use (default: tabular).",
    )
    p.add_argument(
        "--selection-methods",
        nargs="+",
        choices=sorted(AutoFeatureEngineer.SUPPORTED_SELECTION_METHODS),
        help="Selection methods (default: mutual_info importance).",
    )
    p.add_argument("--max-features", type=int, help="Maximum number of features to keep.")
    p.add_argument(
        "--correlation-threshold",
        type=float,
        help="Maximum pairwise correlation in redundancy elimination (default: 0.85).",
    )
    p.add_argument(
        "--leakage-guard",
        choices=sorted(AutoFeatureEngineer.SUPPORTED_LEAKAGE_GUARDS),
        help="How to handle suspicious column names (default: warn).",
    )
    p.add_argument(
        "--gate-n-jobs",
        type=int,
        help="Parallelism for the do-no-harm gate's RF (default: 1; -1 = all cores).",
    )
    p.add_argument(
        "--config",
        help="Path to a JSON config file. CLI flags take precedence over config keys. "
        "Use this to pass nested keys such as ``llm_config``.",
    )
    p.add_argument("--verbose", action="store_true", default=None, help="Enable verbose logging.")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns the process exit code; suitable for both the ``console_scripts``
    entry point (``featcopilot``) and ``python -m featcopilot``.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        # User-facing input/config errors: print a clean message to stderr
        # without a traceback so agents can parse the failure.
        sys.stderr.write(f"featcopilot: error: {exc}\n")
        return 2
    except KeyboardInterrupt:
        sys.stderr.write("featcopilot: interrupted\n")
        return 130
    except Exception as exc:  # pragma: no cover - defensive backstop
        sys.stderr.write(f"featcopilot: unexpected error: {type(exc).__name__}: {exc}\n")
        logger.exception("Unhandled CLI exception")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
