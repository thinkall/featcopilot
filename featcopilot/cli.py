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
        --input data.csv --target label --output features.csv \\
        --engines tabular --max-features 50 --json
    featcopilot explain --input data.csv --target label --json

Equivalent module invocation::

    python -m featcopilot info --json

Parquet I/O is supported only when ``pyarrow`` or ``fastparquet`` is
installed (FeatCopilot's base distribution does not pin either); ``info``
reports the runtime availability via ``parquet_available``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import threading
import warnings
from pathlib import Path
from typing import Any

from featcopilot import __version__
from featcopilot.transformers.sklearn_compat import AutoFeatureEngineer
from featcopilot.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_INPUT_FORMATS = ("csv", "parquet", "json")
SUPPORTED_OUTPUT_FORMATS = ("csv", "parquet", "json")


def _parquet_engine_available() -> bool:
    """Return ``True`` if a parquet engine (pyarrow or fastparquet) can be imported.

    FeatCopilot's base install pins neither ``pyarrow`` nor ``fastparquet``;
    parquet I/O is therefore opportunistic. ``info`` uses this probe so the
    machine-readable capability output reflects what will actually work in
    the current environment, rather than always advertising parquet.

    Uses ``__import__`` (not ``importlib.util.find_spec``) so the probe is
    *correct* even on environments with a broken native install:
    ``find_spec`` only confirms a distribution is on ``sys.path``; it does
    not prove the C extensions can actually load. A real import is the
    only way to verify the engine is usable.
    """
    for name in ("pyarrow", "fastparquet"):
        try:
            __import__(name)
            return True
        except Exception:  # noqa: BLE001  - any import-time failure means unusable
            continue
    return False


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
            raise ValueError(f"Unsupported format {override!r}; expected one of {SUPPORTED_INPUT_FORMATS}")
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


def _read_table(path: Path, fmt: str, *, nrows: int | None = None, suppress_truncation_warning: bool = False):
    """Read a tabular file into a pandas DataFrame.

    All user-facing failure modes (missing parquet engine, ``--input``
    pointing at a directory, permission denied, malformed JSON/CSV,
    decoding errors) are normalized into :class:`ValueError` so the CLI's
    top-level handler routes them to the deterministic ``exit 2``
    user-error path. The generic ``exit 1`` backstop is reserved for
    truly unexpected (i.e. CLI-internal) errors.

    Parameters
    ----------
    path : pathlib.Path
        File to read.
    fmt : str
        One of ``csv`` / ``parquet`` / ``json``.
    nrows : int or None, optional
        Cap the number of rows returned. For ``csv``, this is propagated
        directly to :func:`pandas.read_csv` so the underlying read is
        memory-bounded. For ``parquet`` and ``json``, pandas does not
        expose a native row limit, so the file is fully read and then
        truncated; a :class:`UserWarning` is issued in that case (unless
        ``suppress_truncation_warning`` is true) so the caller knows the
        bound is post-read (not memory-bounded). The ``nrows`` cap is
        applied with a deterministic head slice so re-runs on the same
        input produce the same metadata.
    suppress_truncation_warning : bool, optional
        When True, the post-read truncation notice (parquet / JSON only)
        is *not* emitted from this helper. Used by callers that emit
        their own consolidated, user-facing warning so users don't see
        a confusing pair of messages — see ``_cmd_explain``'s
        ``--explain-sample-size`` handling.
    """
    import pandas as pd

    if path.is_dir():
        raise ValueError(f"--input expects a file, but {str(path)!r} is a directory.")

    if fmt == "csv":
        try:
            # ``nrows`` is the only memory-bound knob native to read_csv;
            # passing it here is what lets ``--explain-sample-size`` actually
            # cap memory on huge CSV inputs (rather than loading the entire
            # file and then trimming).
            df = pd.read_csv(path, nrows=nrows)
        except (
            OSError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
            UnicodeDecodeError,
        ) as exc:
            # ``EmptyDataError`` fires for headerless / zero-byte CSVs;
            # without it, those inputs would fall into the generic exit-1
            # "unexpected error" path instead of the documented exit-2
            # user-input error.
            raise ValueError(f"Failed to read CSV from {str(path)!r}: {exc}") from exc
    elif fmt == "parquet":
        try:
            df = pd.read_parquet(path)
        except ImportError as exc:
            raise ValueError(
                f"Reading parquet requires a parquet engine (pyarrow or fastparquet); "
                f"install one of them, or convert the input to CSV/JSON. Original error: {exc}"
            ) from exc
        except Exception as exc:
            # Catch *any* backend failure (``OSError`` for I/O,
            # ``pyarrow.lib.ArrowInvalid`` for corrupt files,
            # ``ValueError`` from ``fastparquet`` for malformed metadata,
            # etc.) and surface it via the deterministic exit-2 path.
            # Catching ``Exception`` is appropriate here because the entire
            # operation is delegated to a third-party backend; any error
            # raised is by definition an I/O or data issue, not a CLI bug.
            raise ValueError(f"Failed to read parquet from {str(path)!r}: {exc}") from exc
        if nrows is not None and len(df) > nrows:
            if not suppress_truncation_warning:
                warnings.warn(
                    f"--explain-sample-size cap is applied post-read for parquet "
                    f"(loaded {len(df)} rows, truncating to {nrows}). pandas "
                    "does not expose a native parquet row-limit, so the full "
                    "file is materialized in memory before the cap. For hard "
                    "memory bounds on huge inputs, convert to CSV first.",
                    UserWarning,
                    stacklevel=2,
                )
            df = df.iloc[:nrows]
    elif fmt == "json":
        # ``orient='records'`` is the agent-friendly default; fall back to
        # pandas' auto-detection when the file isn't a records list.
        try:
            df = pd.read_json(path, orient="records")
        except ValueError:
            try:
                df = pd.read_json(path)
            except ValueError as exc:
                raise ValueError(f"Failed to read JSON from {str(path)!r}: {exc}") from exc
        except OSError as exc:
            raise ValueError(f"Failed to read JSON from {str(path)!r}: {exc}") from exc
        if nrows is not None and len(df) > nrows:
            if not suppress_truncation_warning:
                warnings.warn(
                    f"--explain-sample-size cap is applied post-read for JSON "
                    f"(loaded {len(df)} rows, truncating to {nrows}). pandas "
                    "does not expose a native JSON row-limit, so the full "
                    "file is materialized in memory before the cap. For hard "
                    "memory bounds on huge inputs, convert to CSV first.",
                    UserWarning,
                    stacklevel=2,
                )
            df = df.iloc[:nrows]
    else:
        raise ValueError(f"Unsupported input format: {fmt}")

    # Reject "header-only" / empty inputs across every supported format.
    # ``pd.read_csv`` returns an empty DataFrame (no exception) when the
    # CSV has headers but zero data rows; the same goes for an empty
    # parquet file or ``[]`` JSON body. Without this check, the CLI
    # would pass an empty frame into ``TabularEngine``, which divides by
    # ``len(X)`` while fitting categorical encoding and exits via the
    # generic ``unexpected error`` path. Surface the issue as a clean
    # exit-2 user-input error.
    if df.empty:
        raise ValueError(
            f"Input file {str(path)!r} is empty (zero data rows). "
            "Feature engineering requires at least one row of data."
        )
    return df


def _write_table(df, path: Path, fmt: str) -> None:
    """Write a pandas DataFrame to ``path`` in ``fmt``.

    All user-facing failure modes (missing parquet engine, ``--output``
    pointing at a directory, permission denied, parent-directory creation
    failures) are normalized into :class:`ValueError` so the CLI surfaces a
    clean stderr message via the standard ``exit 2`` path instead of the
    generic ``exit 1`` "unexpected error" backstop.
    """
    if path.exists() and path.is_dir():
        raise ValueError(f"--output expects a file, but {str(path)!r} is an existing directory.")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"Cannot create parent directory for {str(path)!r}: {exc}") from exc

    if fmt == "csv":
        try:
            df.to_csv(path, index=False)
        except OSError as exc:
            raise ValueError(f"Failed to write CSV to {str(path)!r}: {exc}") from exc
    elif fmt == "parquet":
        try:
            df.to_parquet(path, index=False)
        except ImportError as exc:
            raise ValueError(
                f"Writing parquet requires a parquet engine (pyarrow or fastparquet); "
                f"install one of them, or pick CSV/JSON via --output-format. Original error: {exc}"
            ) from exc
        except Exception as exc:
            # Same broad-catch rationale as ``_read_table``: parquet write
            # is fully delegated to a backend (``pyarrow``/``fastparquet``)
            # whose errors include ``OSError`` (I/O), engine-specific type
            # / conversion exceptions for unsupported column values, etc.
            # All of these are user-facing data issues, not CLI bugs, so
            # they should produce a clean exit-2 failure.
            raise ValueError(f"Failed to write parquet to {str(path)!r}: {exc}") from exc
    elif fmt == "json":
        try:
            df.to_json(path, orient="records", indent=2)
        except OSError as exc:
            raise ValueError(f"Failed to write JSON to {str(path)!r}: {exc}") from exc
    else:
        raise ValueError(f"Unsupported output format: {fmt}")


# Top-level keys recognized in a ``--config`` JSON file. The CLI rejects
# any other top-level key with a precise exit-2 error so typos like
# ``max_feature`` (no s) fail fast in automation rather than silently
# running with defaults.
_KNOWN_CONFIG_KEYS = frozenset(
    {
        "engines",
        "selection_methods",
        "max_features",
        "correlation_threshold",
        "leakage_guard",
        "gate_n_jobs",
        "llm_config",
        "verbose",
        "explain_sample_size",
    }
)


def _load_config(config_path: str | None) -> dict[str, Any]:
    """Load a JSON config file (or return an empty dict).

    Normalizes user-input mistakes (missing path, directory passed instead
    of a file, invalid JSON, non-object root, unknown top-level keys) into
    :class:`ValueError` / :class:`FileNotFoundError` so the CLI's top-level
    error handler can route them all to the deterministic ``exit 2``
    user-error path (rather than e.g. ``IsADirectoryError`` falling into
    the generic ``exit 1`` "unexpected error" backstop, or a typo silently
    being ignored).
    """
    if config_path is None:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if path.is_dir():
        raise ValueError(f"--config expects a JSON file, but {config_path!r} is a directory.")
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Config file {config_path!r} is not valid JSON: {exc}") from exc
    except OSError as exc:
        # Catch-all for unreadable files (permission denied, broken symlink,
        # etc.). Surface as a user-facing error rather than the generic
        # exit-1 backstop.
        raise ValueError(f"Config file {config_path!r} could not be read: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Config file {config_path!r} must contain a JSON object at the top level")
    unknown = sorted(set(data.keys()) - _KNOWN_CONFIG_KEYS)
    if unknown:
        raise ValueError(
            f"Config file {config_path!r} has unknown top-level key(s): {unknown}. "
            f"Recognized keys: {sorted(_KNOWN_CONFIG_KEYS)}."
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


def _check_scalar_type(
    name: str,
    value: Any,
    expected: tuple[type, ...],
    *,
    allow_none: bool = False,
    allow_bool: bool = True,
) -> None:
    """Validate a scalar value's type for ``--config``-supplied keys.

    Raises :class:`ValueError` (caught by ``main()`` -> exit 2) when the
    value's type does not match. ``bool`` is a subclass of ``int`` in
    Python; pass ``allow_bool=False`` to reject ``True``/``False`` for
    numeric-only fields like ``max_features`` / ``correlation_threshold``.
    """
    if value is None:
        if allow_none:
            return
        raise ValueError(f"`{name}` must not be null in --config")
    if not allow_bool and isinstance(value, bool):
        raise ValueError(
            f"`{name}` in --config must be a {' or '.join(t.__name__ for t in expected)}; " f"got bool={value!r}."
        )
    if not isinstance(value, expected):
        raise ValueError(
            f"`{name}` in --config must be a {' or '.join(t.__name__ for t in expected)}; "
            f"got {type(value).__name__}={value!r}."
        )


def _build_engineer(args: argparse.Namespace, *, include_selection_config: bool = True) -> AutoFeatureEngineer:
    """Construct an :class:`AutoFeatureEngineer` from parsed CLI args.

    Precedence: explicit CLI flags override values from ``--config``;
    explicit config values (including empty lists) override the defaults.
    Empty / non-list values are propagated unchanged so that
    :meth:`AutoFeatureEngineer._validate_configuration` produces its
    canonical (and deterministic) error path — the CLI's wrapper must not
    silently rewrite a misconfigured config into something that looks
    different from what the user wrote.

    ``include_selection_config=False`` (used by the ``explain`` subcommand)
    skips reading selection-only config keys (``selection_methods``,
    ``correlation_threshold``) so a shared config file with selection
    settings does not cause ``explain`` to fail config-validation for keys
    that are inert at runtime (selection is disabled in ``explain``).
    """
    config = _load_config(args.config)

    def pick(flag_value, config_key, default):
        # Explicit CLI flag wins. Otherwise honor an explicit config entry
        # — even a falsy one such as ``[]`` — so AutoFeatureEngineer can
        # raise its own clear "must contain at least one" error rather than
        # the CLI silently swapping in defaults. Only fall back to the
        # default when the key is *absent* from the config.
        if flag_value is not None:
            return flag_value
        if config_key in config:
            return config[config_key]
        return default

    engines = pick(args.engines, "engines", ["tabular"])
    # ``explain`` exposes ``--engines`` and ``--max-features`` (engine-level
    # caps) but not the selection-only flags ``--selection-methods`` and
    # ``--correlation-threshold``. When ``include_selection_config`` is
    # False (i.e. we're called from ``explain``) we also skip reading the
    # selection-only keys from the config file, so a shared transform/explain
    # config with selection settings won't trip ``explain`` over keys that
    # have no effect on its runtime behavior.
    if include_selection_config:
        selection_methods = pick(
            getattr(args, "selection_methods", None),
            "selection_methods",
            ["mutual_info", "importance"],
        )
        correlation_threshold = pick(getattr(args, "correlation_threshold", None), "correlation_threshold", 0.85)
    else:
        selection_methods = ["mutual_info", "importance"]
        correlation_threshold = 0.85
    max_features = pick(args.max_features, "max_features", None)
    leakage_guard = pick(args.leakage_guard, "leakage_guard", "warn")
    gate_n_jobs = pick(args.gate_n_jobs, "gate_n_jobs", 1)

    # Type-check scalar config fields here so the CLI surfaces a clean
    # exit-2 error instead of a downstream ``TypeError`` (e.g. from
    # ``self.max_features <= 0`` when the JSON config supplied a string).
    # ``argparse`` already enforces types for the flag side; this only
    # guards against malformed ``--config`` JSON.
    _check_scalar_type("max_features", max_features, (int,), allow_none=True, allow_bool=False)
    _check_scalar_type("correlation_threshold", correlation_threshold, (int, float), allow_bool=False)
    _check_scalar_type("gate_n_jobs", gate_n_jobs, (int,), allow_bool=False)
    _check_scalar_type("leakage_guard", leakage_guard, (str,))

    # Range-check ``correlation_threshold``: it's only meaningful in
    # ``[0.0, 1.0]``. Values above 1 silently disable redundancy
    # elimination (``FeatureSelector.fit`` only runs it when threshold
    # < 1.0); values below 0 effectively treat every numeric pair as
    # redundant. Reject out-of-range up front so the CLI doesn't quietly
    # change selector behavior.
    if not (0.0 <= float(correlation_threshold) <= 1.0):
        raise ValueError(f"`correlation_threshold` must be in the range [0.0, 1.0]; got {correlation_threshold!r}.")
    # ``max_features`` must be positive when set (matches
    # AutoFeatureEngineer's own validation). Surface that here too so
    # the message says ``max_features`` rather than the more cryptic
    # transformer error.
    if max_features is not None and max_features <= 0:
        raise ValueError(f"`max_features` must be a positive integer when set; got {max_features!r}.")

    # Validate ``llm_config`` is a JSON object (i.e. a Python dict) before
    # forwarding it. Without this check, a misconfigured non-dict value
    # would only fail at engine-construction time inside
    # ``AutoFeatureEngineer._create_engine`` via ``self.llm_config.get(...)``,
    # raising an ``AttributeError`` that bypasses the structured exit-2
    # user-error path (the CLI would surface it as exit 1 "unexpected
    # error", which is a poor agent contract for a documented config key).
    llm_config_raw = config.get("llm_config")
    if llm_config_raw is None:
        llm_config: dict[str, Any] = {}
    elif isinstance(llm_config_raw, dict):
        llm_config = llm_config_raw
    else:
        raise ValueError(
            "`llm_config` in the --config file must be a JSON object (mapping); "
            f"got {type(llm_config_raw).__name__}={llm_config_raw!r}."
        )

    # ``verbose`` is type-checked before being forwarded so a malformed
    # config like ``{"verbose": "false"}`` (truthy string) does NOT silently
    # turn verbose mode on — instead it raises a clean exit-2 error
    # consistent with the other scalar fields. ``args.verbose`` is already
    # a bool / None thanks to ``BooleanOptionalAction``; only the config
    # path can introduce a non-bool.
    verbose_raw = pick(args.verbose, "verbose", False)
    _check_scalar_type("verbose", verbose_raw, (bool,))
    verbose = bool(verbose_raw)

    # Pass ``engines`` / ``selection_methods`` through *unchanged* (no
    # ``list(...)`` wrapping). Coercion would convert a misconfigured
    # JSON string like ``"tabular"`` into ``['t','a','b','u','l','a','r']``,
    # turning a clear type error into a confusing "Unknown engines" path.
    # AutoFeatureEngineer.__init__ rejects non-list/tuple inputs with a
    # precise message — let it.
    return AutoFeatureEngineer(
        engines=engines,
        max_features=max_features,
        selection_methods=selection_methods,
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
    """Print version + supported engines/methods.

    Parquet appears in ``supported_input_formats`` / ``supported_output_formats``
    only when an actual parquet engine (``pyarrow`` or ``fastparquet``) can
    be imported in the current environment — otherwise the ``info`` output
    would advertise a format that immediately fails on use, which is
    misleading for the agentic capability-discovery the CLI is designed to
    support.
    """
    parquet_ok = _parquet_engine_available()
    input_formats = [f for f in SUPPORTED_INPUT_FORMATS if f != "parquet" or parquet_ok]
    output_formats = [f for f in SUPPORTED_OUTPUT_FORMATS if f != "parquet" or parquet_ok]
    payload = {
        "version": __version__,
        "supported_engines": sorted(AutoFeatureEngineer.SUPPORTED_ENGINES),
        "supported_selection_methods": sorted(AutoFeatureEngineer.SUPPORTED_SELECTION_METHODS),
        "supported_leakage_guards": sorted(AutoFeatureEngineer.SUPPORTED_LEAKAGE_GUARDS),
        "supported_input_formats": input_formats,
        "supported_output_formats": output_formats,
        "parquet_available": parquet_ok,
    }
    _emit(payload, as_json=args.json)
    return 0


def _fit_transform_capturing_warnings(engineer, X, y, **kwargs):
    """Run ``engineer.fit_transform(X, y, **kwargs)`` while capturing both
    Python ``warnings.warn(...)`` and FeatCopilot logger records.

    The CLI contract is that stdout carries the JSON payload and stderr is
    reserved for failures. Two sources can otherwise bleed onto stderr on
    a successful run:

    * ``warnings.warn(...)`` — emitted by ``AutoFeatureEngineer.fit`` for
      leakage-prone column names under the default ``leakage_guard='warn'``.
    * ``logger.warning(...)`` / ``logger.info(...)`` — emitted by e.g.
      ``_do_no_harm_gate`` on validation-failure fallback, and by every
      engine when ``--verbose`` is set.

    The single ``featcopilot`` root logger (``propagate=False``) receives
    every child logger's records by ordinary Python logging propagation;
    we swap in a capture handler for the duration of the call so the JSON
    payload can surface those messages instead of stderr.

    Returns
    -------
    (messages, result)
        ``messages`` is a list of ``str`` (warnings then logs, in
        emission order). ``result`` is whatever ``fit_transform`` returned.
    """
    with _capture_featcopilot_messages() as captured:
        result = engineer.fit_transform(X, y, **kwargs)
    return captured, result


class _ThreadCaptureState:
    """Holds per-thread capture *stacks* with strict per-thread isolation
    and a *narrow* cross-thread fallback for LLM-client log records.

    Each thread maps to a stack of capture lists. Nested
    :func:`_capture_featcopilot_messages` calls on the same thread push
    onto the stack; the innermost active capture is always at the top
    and receives records / warnings until its block exits, at which
    point the outer capture (if any) becomes active again.

    **Strict thread isolation by default.** :meth:`get` returns a target
    list ONLY for the calling thread itself. This avoids the
    misattribution that an unconditional single-active-capture fallback
    would cause: any featcopilot log on any thread would otherwise be
    silently swallowed into the active CLI run's payload, including
    output from unrelated background work happening in the same process.

    **Narrow LLM-client fallback.** :meth:`get_for_llm_record` is the
    one exception: when a record originates from one of the *exact*
    sync LLM client modules whitelisted in
    :attr:`_LLM_FALLBACK_LOGGER_NAMES` (the modules whose synchronous
    entry points fall back to ``ThreadPoolExecutor`` in event-loop
    environments), and exactly one capture is active in the process,
    the record is routed to that capture even when emitted on a worker
    thread. This addresses the common case where an LLM client's
    mock-mode startup warning fires on a worker that ``submit()``
    spawned and would otherwise bleed onto stderr. The whitelist is an
    explicit set of module names — *not* a prefix — so unrelated
    ``featcopilot.llm.*`` loggers (e.g. ``semantic_engine``,
    ``code_generator``, ``transform_rule_generator``, ``explainer``)
    that never run on worker threads do not trigger the fallback.

    **Atomic per-record decision.** :meth:`resolve_for_record` caches
    the lookup on the record itself the first time it is called, so
    :class:`_ThreadRoutingHandler` (which routes records) and
    :class:`_SuppressCapturingFilter` (which suppresses them from
    stderr) cannot disagree under concurrent push/pop activity from
    other threads. Without that caching, ``filter`` could see one
    state and the later ``emit`` could see another, breaking the
    "captured XOR on stderr" invariant.

    Shared by :class:`_ThreadRoutingHandler` (writes records),
    :class:`_SuppressCapturingFilter` (suppresses stderr), and the
    routing ``warnings.showwarning`` override.
    """

    # Logger names whose records are eligible for the narrow cross-thread
    # fallback. Only the synchronous LLM client modules whose ``run`` /
    # batch entry points fall back to ``ThreadPoolExecutor`` workers in
    # event-loop environments are listed; their startup / mock-mode
    # warnings legitimately fire from those worker threads. Other
    # ``featcopilot.llm.*`` loggers are intentionally NOT included so
    # cross-thread records from unrelated background work cannot be
    # silently swallowed into an active CLI capture.
    _LLM_FALLBACK_LOGGER_NAMES = frozenset(
        {
            "featcopilot.llm.copilot_client",
            "featcopilot.llm.litellm_client",
            "featcopilot.llm.openai_client",
        }
    )

    # Sentinel for "no cached decision yet" on a log record. Distinct
    # from ``None``, which is itself a valid resolved decision meaning
    # "no capture target — let the record flow to stderr".
    _UNCACHED = object()

    def __init__(self):
        self._per_thread: dict[int, list[list[str]]] = {}
        self._lock = threading.Lock()

    def push(self, tid: int, target: list[str]) -> None:
        with self._lock:
            self._per_thread.setdefault(tid, []).append(target)

    def pop(self, tid: int) -> None:
        with self._lock:
            stack = self._per_thread.get(tid)
            if stack:
                stack.pop()
                if not stack:
                    del self._per_thread[tid]

    def get(self, tid: int) -> list[str] | None:
        # Strict per-thread lookup. No cross-thread fallback: an
        # unconditional fallback was too broad and could silently
        # swallow unrelated background log output into a CLI run's
        # payload. The narrow LLM-client fallback lives in
        # :meth:`get_for_llm_record` instead, opted into by name.
        with self._lock:
            stack = self._per_thread.get(tid)
            if stack:
                return stack[-1]
            return None

    def get_for_llm_record(self, tid: int, logger_name: str) -> list[str] | None:
        """Per-thread lookup with a narrow cross-thread fallback for LLM
        client records.

        When the calling thread has its own active capture, that's used.
        Otherwise — and only when the record originates from a
        whitelisted sync LLM client module
        (:attr:`_LLM_FALLBACK_LOGGER_NAMES`) AND exactly one capture is
        active in the process — the record is routed to that single
        capture so it doesn't bleed onto stderr. The whitelist is an
        explicit set of module names so unrelated ``featcopilot.llm.*``
        loggers (e.g. ``semantic_engine``, ``code_generator``) that
        never hop onto worker threads do not trigger the fallback.

        This method takes the lock once and returns a snapshot
        decision; callers should generally use :meth:`resolve_for_record`
        to additionally cache the result on the record so that paired
        filter / emit calls can never disagree under concurrent
        push/pop on other threads.
        """
        with self._lock:
            stack = self._per_thread.get(tid)
            if stack:
                return stack[-1]
            if logger_name not in self._LLM_FALLBACK_LOGGER_NAMES:
                return None
            if len(self._per_thread) != 1:
                # Either no captures are active (nothing to route to) or
                # multiple are active (ambiguous — keep strict isolation
                # so concurrent CLI calls don't cross-contaminate).
                return None
            only_stack = next(iter(self._per_thread.values()))
            if only_stack:
                return only_stack[-1]
            return None

    def resolve_for_record(self, record: logging.LogRecord) -> list[str] | None:
        """Resolve the capture target for ``record`` exactly once.

        The first call computes the decision via
        :meth:`get_for_llm_record` and caches it on the record itself
        as ``record._featcopilot_capture_target``; subsequent calls
        return the cached value. This is what makes the routing
        handler's ``emit`` and the suppression filter's ``filter``
        atomic with respect to each other: they always see the same
        decision for a given record even if another thread pops or
        pushes a capture between the two calls.

        Logging records are produced and dispatched to handlers in the
        same thread, so caching directly on the record is safe — there
        is no concurrent reader of the same record's attributes.
        """
        cached = getattr(record, "_featcopilot_capture_target", self._UNCACHED)
        if cached is not self._UNCACHED:
            return cached
        target = self.get_for_llm_record(threading.get_ident(), record.name)
        record._featcopilot_capture_target = target
        return target


class _ThreadRoutingHandler(logging.Handler):
    """Logging handler that routes records to the calling thread's capture list.

    Attached once to the ``featcopilot`` root logger. Records propagated
    from any ``featcopilot.*`` child logger reach this handler in the same
    way they reach the existing stderr handler. If the calling thread has
    a registered capture list, the record is appended to it.
    Otherwise, for records originating from a ``featcopilot.llm.*``
    logger AND when exactly one capture is active in the process, the
    record is routed to that capture (the narrow LLM-client cross-thread
    fallback — see :class:`_ThreadCaptureState`). Records from any
    other thread / logger combination flow through to the existing
    stderr handler.
    """

    def __init__(self, state: _ThreadCaptureState):
        super().__init__(logging.DEBUG)
        self._state = state
        self.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        # Use the cached resolver so this handler and the paired
        # ``_SuppressCapturingFilter`` always see the same decision
        # for a given record, even if another thread pushes or pops
        # a capture between the filter and emit phases.
        target = self._state.resolve_for_record(record)
        if target is None:
            return
        try:
            target.append(self.format(record))
        except Exception:  # pragma: no cover - never let logging crash the CLI
            target.append(record.getMessage())


class _SuppressCapturingFilter(logging.Filter):
    """Filter for the *existing* handlers: drops records being captured.

    Without this filter, every record routed by
    :class:`_ThreadRoutingHandler` to a capture list would still hit the
    featcopilot root logger's stderr ``StreamHandler`` and bleed onto
    stderr — breaking the CLI's "stderr reserved for failures" contract.
    The filter mirrors the routing handler's policy so the two stay in
    lockstep: anything captured (current-thread record OR cross-thread
    LLM-client record under the narrow fallback) is also suppressed
    from the original handlers; anything else (records from
    non-capturing threads / unrelated background work) flows through to
    stderr unchanged.
    """

    def __init__(self, state: _ThreadCaptureState):
        super().__init__()
        self._state = state

    def filter(self, record: logging.LogRecord) -> bool:
        # Use the cached resolver so this filter and the paired
        # ``_ThreadRoutingHandler`` always see the same decision for a
        # given record, even if another thread pushes or pops a capture
        # between this call and the routing handler's emit. Without the
        # cache, the two could disagree and the same record could end
        # up both captured and on stderr (or suppressed without being
        # captured) — breaking the "stderr reserved for failures"
        # contract under concurrent CLI calls.
        return self._state.resolve_for_record(record) is None


# Module-level singletons. Installed exactly once on the featcopilot root
# logger / its existing handlers; subsequent ``_capture_featcopilot_messages``
# calls just push/pop thread state. No global lock is held during the slow
# ``fit_transform`` body — concurrent threads each capture their own records
# independently.
_capture_state = _ThreadCaptureState()
_routing_handler = _ThreadRoutingHandler(_capture_state)
_suppress_filter = _SuppressCapturingFilter(_capture_state)
_install_lock = threading.Lock()
_install_done = False
# Captures the original ``warnings.showwarning`` at first install so the
# routing override can chain to it for non-capturing threads (and so we
# never mutate it again on subsequent capture calls — the previous
# per-call save/restore raced under concurrent overlapping captures).
_original_showwarning = None


def _routing_showwarning(message, category, filename, lineno, file=None, line=None):
    """Permanent ``warnings.showwarning`` override (installed once).

    Routes warnings to the *innermost* capturing list for the current
    thread (via :class:`_ThreadCaptureState` stack lookup). If the
    current thread is not capturing, chains to the original
    ``warnings.showwarning`` so non-capturing threads keep their normal
    behavior.

    Installed once globally — *not* swapped per-call — so concurrent
    overlapping captures on different threads cannot race on the
    process-global ``warnings.showwarning`` slot.
    """
    target = _capture_state.get(threading.get_ident())
    if target is not None:
        target.append(str(message))
        return
    if _original_showwarning is not None:
        _original_showwarning(message, category, filename, lineno, file, line)


def _install_capture_hooks_once() -> None:
    """Install the routing handler + suppress filter + showwarning override.

    The logger handler and filter are installed exactly once (idempotent).
    The ``warnings.showwarning`` override is re-installed every call if
    something else has replaced it — this is necessary because external
    code (most commonly ``warnings.catch_warnings()`` blocks) can reset
    the global ``warnings.showwarning`` and undo a previous install. The
    fresh re-install captures the current (caller's) ``showwarning`` as
    the new "original" to chain to, so non-capturing threads still see
    whatever warning behavior the caller had set up.

    All hooks themselves dispatch on :class:`_ThreadCaptureState` which
    uses a per-thread stack, so they are no-ops for threads that aren't
    currently capturing.
    """
    global _install_done, _original_showwarning
    with _install_lock:
        # Logger handler/filter install (truly once — these can't be
        # silently undone by external code in the way ``warnings.showwarning``
        # can).
        if not _install_done:
            fc_root = logging.getLogger("featcopilot")
            if _routing_handler not in fc_root.handlers:
                fc_root.addHandler(_routing_handler)
            for handler in list(fc_root.handlers):
                if handler is _routing_handler:
                    continue
                if _suppress_filter not in handler.filters:
                    handler.addFilter(_suppress_filter)
            _install_done = True

        # ``warnings.showwarning`` install — re-check every entry. A
        # caller's ``warnings.catch_warnings()`` block restores the
        # previous ``showwarning`` on exit, undoing our install. Re-
        # installing on next entry is what makes overlapping captures
        # robust against caller-side warning context manipulation.
        if warnings.showwarning is not _routing_showwarning:
            _original_showwarning = warnings.showwarning
            warnings.showwarning = _routing_showwarning


@contextlib.contextmanager
def _capture_featcopilot_messages():
    """Capture FeatCopilot log records and ``warnings.warn`` calls emitted
    on the *current thread*.

    Yields a list that the caller can read after the with-block exits.
    The list contains formatted log records (in emission order) and any
    Python warning messages emitted during the with-block on this thread.

    Concurrency model
    -----------------
    * **Logger records** are routed *per-thread* via
      :class:`_ThreadRoutingHandler` (added once to the ``featcopilot``
      root logger) and a :class:`_SuppressCapturingFilter` on the existing
      handlers. Two threads can capture concurrently without blocking
      each other; each sees only its own records, and other threads'
      records still flow normally to stderr.
    * **``warnings.warn`` records** are intercepted via a permanent
      :func:`_routing_showwarning` override installed once. The override
      routes by ``threading.get_ident()`` and chains to the original
      ``warnings.showwarning`` for non-capturing threads. The override is
      *not* swapped per-call, so concurrent overlapping captures on
      different threads cannot race on the process-global
      ``warnings.showwarning`` slot.
    * **Nested captures** on the same thread are supported via a
      per-thread stack in :class:`_ThreadCaptureState`. Records and
      warnings always go to the innermost active capture; when the inner
      block exits, the outer capture is automatically reactivated.

    The contextmanager does NOT hold any lock for the duration of the
    with-block — only briefly during install/push/pop — so long-running
    ``fit_transform`` calls in one thread do not block other threads
    from running concurrently.
    """
    _install_capture_hooks_once()

    captured: list[str] = []
    tid = threading.get_ident()
    _capture_state.push(tid, captured)
    try:
        yield captured
    finally:
        _capture_state.pop(tid)


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

    # Build the engineer first: ``_build_engineer`` runs all scalar / list /
    # dict type validation on the merged CLI-flag + config view, so any
    # malformed value (e.g. ``"max_features": "5"``, ``"verbose": "false"``)
    # surfaces a precise exit-2 error here rather than down the wrong
    # ``--target is required`` rabbit hole.
    engineer = _build_engineer(args)

    # Selection requires a target column to fit against. ``AutoFeatureEngineer``
    # only actually fits a selector when ``y is not None`` AND ``max_features``
    # is set; without ``max_features`` the call is a raw feature-generation
    # run and does not need a target. The CLI mirrors that contract: only
    # require ``--target`` when both selection is enabled (the default) AND
    # ``max_features`` is configured (CLI flag or config), so commands like
    # ``featcopilot transform --input in.csv --output out.csv`` (no target,
    # no cap) still work. Using ``engineer.max_features`` here means the
    # value has already been type-validated, so we never report
    # ``--target is required`` when the real problem is a malformed
    # ``max_features`` config value.
    if not args.no_selection and args.target is None and engineer.max_features is not None:
        raise ValueError(
            "--target is required when feature selection is applied "
            "(i.e. when --max-features / config max_features is set). "
            "Pass --target <column>, or pass --no-selection / drop --max-features to skip selection."
        )

    captured_warnings, transformed = _fit_transform_capturing_warnings(
        engineer,
        X,
        y,
        task_description=args.task_description or "prediction task",
        target_name=args.target,
        apply_selection=not args.no_selection,
    )

    if args.include_target and y is not None:
        # Re-attach the target column so downstream training scripts can
        # consume the engineered file as a single artifact. Detect column
        # collisions: if an engineered feature happens to share the
        # target's column name (e.g. a target named ``foo_pow2`` matching
        # a tabular-engine derived feature), blindly assigning ``transformed[
        # target_name] = y.values`` would silently overwrite the engineered
        # column. Surface that as a clean exit-2 error instead. Callers
        # who knowingly want to overwrite can rename their target before
        # invoking ``transform`` (or skip ``--include-target``).
        target_name = args.target if args.target in df.columns else "target"
        if target_name in transformed.columns:
            raise ValueError(
                f"--include-target would overwrite engineered feature {target_name!r} "
                "with the target values. Rename the target column in the input file, "
                "or drop --include-target."
            )
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
        "warnings": captured_warnings,
    }
    _emit(payload, as_json=args.json)
    return 0


# Default ``explain`` behavior is to use the full input so the metadata
# is a faithful description of what a corresponding ``transform`` run
# would do — engines like ``TabularEngine._fit_categorical_encoding``
# use ``n_rows`` and per-category counts to decide e.g. one-hot vs.
# target-encoding, so subsampling can silently change which features
# appear. Callers who knowingly accept that trade-off can opt in via
# ``--explain-sample-size`` (set to ``None``/absent to disable, any
# positive integer to cap).
def _cmd_explain(args: argparse.Namespace) -> int:
    """Fit + transform engines and print feature explanations + code as JSON.

    The built-in engines populate their internal feature-name registry during
    :meth:`transform`, not :meth:`fit` (planning happens in ``fit`` but feature
    objects are materialized in ``transform``). We therefore call
    :meth:`AutoFeatureEngineer.fit_transform` so ``get_feature_names()``,
    :meth:`explain_features` and :meth:`get_feature_code` all return the
    actual generated features. Selection is intentionally skipped here so the
    payload describes every candidate feature the engines produced, not just
    the post-selection survivors.

    Performance vs. faithfulness
    ---------------------------
    By default ``explain`` runs on the *full* input so the reported
    metadata is a faithful description of what a corresponding
    ``transform`` would generate. Some engines (notably
    :class:`TabularEngine`) consult row counts and per-category
    statistics when deciding which features to plan, so blind
    subsampling can silently change the result.

    For very large inputs where the metadata-only nature of ``explain``
    really should not pay full memory / compute cost, callers can pass
    ``--explain-sample-size N`` (or set ``"explain_sample_size": N`` in
    ``--config``) to cap the rows fed to the engineer. The cap is a
    deterministic *head slice* (the first N rows): for CSV the cap is
    threaded through ``pd.read_csv(nrows=N)`` so memory is bounded
    natively; for parquet/JSON pandas has no native row-limit so the
    file is fully read and then truncated, with a UserWarning explaining
    the limitation. The cap is NOT a random sample — callers who need
    randomness should sample externally before invoking ``explain``.
    A "metadata may differ" UserWarning is emitted (captured into the
    JSON payload's ``warnings`` field) only when the cap actually
    truncated the input. The ``n_rows_used`` field reports the effective
    sample size.
    """
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    in_fmt = _detect_format(input_path, args.input_format)

    # Apply opt-in sample cap from CLI flag or config (CLI flag wins).
    # Resolve and validate it BEFORE reading the input so the cap can be
    # threaded into ``_read_table(... nrows=sample_size)`` to bound memory
    # on huge inputs (CSV uses ``pd.read_csv(nrows=...)`` natively;
    # parquet/JSON fall back to post-read truncation with a UserWarning
    # since pandas doesn't expose a native row-limit for those formats).
    sample_size = getattr(args, "explain_sample_size", None)
    if sample_size is None and args.config is not None:
        sample_size = _load_config(args.config).get("explain_sample_size")
    if sample_size is not None:
        _check_scalar_type("explain_sample_size", sample_size, (int,), allow_bool=False)
        if sample_size <= 0:
            raise ValueError(f"`explain_sample_size` must be a positive integer when set; got {sample_size!r}.")

    engineer = _build_engineer(args, include_selection_config=False)

    # Run the sample-warning AND ``fit_transform`` inside a single
    # capture context so the sampling notice ends up in the JSON
    # payload's ``warnings`` field instead of bleeding onto stderr.
    with _capture_featcopilot_messages() as captured_warnings:
        # Choose a read strategy that gives us:
        #   1. a memory bound where pandas allows it (CSV ``nrows``), and
        #   2. enough information to detect truncation without ``_read_table``
        #      emitting its own (slightly off-by-one) warning that would
        #      then double up with ours.
        #
        # For CSV we ask for ``sample_size + 1`` rows: ``pd.read_csv``
        # reads at most that many, AND ``len(df) > sample_size`` becomes
        # a strict proof of truncation. We pass ``suppress_truncation_warning``
        # so ``_read_table`` doesn't emit its own message — ``_cmd_explain``
        # is the single source of truth for the sampling notice and uses
        # the user-facing ``sample_size`` value.
        #
        # For parquet/JSON pandas exposes no native row-limit, so we
        # always read fully (``nrows=None``) and let ``_cmd_explain``
        # both detect truncation and emit a single notice that includes
        # the "memory bound is post-read" caveat. Asking ``_read_table``
        # for a limit there would only have caused it to truncate at
        # ``sample_size + 1`` and emit a confusing duplicate warning.
        if sample_size is not None and in_fmt == "csv":
            read_nrows: int | None = sample_size + 1
        else:
            read_nrows = None
        df = _read_table(input_path, in_fmt, nrows=read_nrows, suppress_truncation_warning=True)
        X, y = _split_xy(df, args.target)
        n_sampled = len(X)
        if sample_size is not None and n_sampled > sample_size:
            # Strict proof of truncation: file had at least one more row
            # than the requested cap. Trim to the exact cap and emit the
            # "metadata may differ" notice.
            X = X.iloc[:sample_size]
            if y is not None:
                y = y.iloc[:sample_size]
            original_len = n_sampled
            n_sampled = sample_size
            msg = f"explain: capping input to {sample_size} rows (head slice). "
            if in_fmt != "csv":
                # For parquet/JSON we read the whole file before truncation
                # (no native row-limit). Surface that fact so callers know
                # memory wasn't bounded.
                msg += (
                    f"For {in_fmt}, pandas does not expose a native row-limit, "
                    f"so the full file ({original_len}+ rows) was loaded into "
                    "memory before truncation. For hard memory bounds on huge "
                    "inputs, convert to CSV first. "
                )
            msg += (
                "Some engines (e.g. TabularEngine categorical encoding) decide which "
                "features to plan based on row counts and per-category statistics, "
                "so the reported metadata may differ from a full-input transform run."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)

        engineer.fit_transform(
            X,
            y,
            task_description=args.task_description or "prediction task",
            target_name=args.target,
            apply_selection=False,
        )

    explanations = engineer.explain_features()
    code = engineer.get_feature_code()
    feature_names = engineer.get_feature_names()

    payload = {
        "status": "ok",
        "input": str(input_path),
        "n_features": len(feature_names),
        "n_rows_used": n_sampled,
        "engines": list(engineer.engines),
        "features": [
            {
                "name": name,
                "explanation": explanations.get(name, ""),
                "code": code.get(name, ""),
            }
            for name in feature_names
        ],
        "warnings": captured_warnings,
    }

    # explain always emits JSON to stdout (it's the only sensible format),
    # but we still respect ``--json`` for symmetry with other subcommands.
    _emit(payload, as_json=True)
    return 0


class _StructuredArgumentParser(argparse.ArgumentParser):
    """``argparse.ArgumentParser`` that emits the CLI's structured single-line
    error format on usage failures.

    The default ``argparse`` ``error()`` method writes the multi-line
    usage banner ("usage: featcopilot ...\\n featcopilot: error: ...") to
    stderr before raising :class:`SystemExit`. That breaks the CLI
    contract that stderr carries exactly one ``featcopilot: error: ...``
    line per failure — agents parsing stderr deterministically would see
    the banner and the actual error mixed together, with no easy way to
    tell which is which.

    This subclass overrides :meth:`error` to write a single line and
    skip the banner, so usage failures (missing required argument,
    unknown flag, missing subcommand, etc.) follow the same single-line
    contract as the rest of the CLI's exit-2 paths.
    """

    def error(self, message: str) -> None:  # type: ignore[override]
        sys.stderr.write(f"featcopilot: error: {message}\n")
        # ``ArgumentParser.error`` is documented to terminate; ``SystemExit(2)``
        # is what the parent class would do after writing the banner.
        # ``main()``'s ``except SystemExit`` handler converts this to an int
        # return value so callers still see the documented exit-2 contract.
        raise SystemExit(2)


def _build_parser() -> argparse.ArgumentParser:
    parser = _StructuredArgumentParser(
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
    # Use the structured parser class for subparsers too so any
    # subcommand-specific usage error (unknown flag, missing required
    # arg) follows the same single-line stderr contract.
    subparsers = parser.add_subparsers(
        dest="command", required=True, metavar="COMMAND", parser_class=_StructuredArgumentParser
    )

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
        "describing each generated feature (name, explanation, code). Selection is "
        "intentionally disabled, so all candidate features are reported.",
    )
    p_explain.add_argument("--input", "-i", required=True, help="Path to input file (CSV / Parquet / JSON).")
    p_explain.add_argument("--input-format", choices=SUPPORTED_INPUT_FORMATS, help="Override input format detection.")
    p_explain.add_argument(
        "--target",
        "-t",
        help="Target column name. Used by leakage-guard checks and as task context "
        "for the LLM engine. (Selection is disabled in `explain`, so this flag "
        "does not gate selector behavior.)",
    )
    p_explain.add_argument(
        "--task-description",
        help="Natural-language ML task description (used by the LLM engine).",
    )
    p_explain.add_argument(
        "--explain-sample-size",
        type=int,
        default=None,
        help="Cap the input fed to the engineer at this many rows. The cap is "
        "applied as a deterministic head slice (the first N rows of the input — "
        "for CSV via `pd.read_csv(nrows=N)` so memory is bounded; for parquet/JSON "
        "the file is fully read and then truncated, with a warning). OFF by default: "
        "the full input is used so the metadata is a faithful description of what a "
        "corresponding `transform` would generate. Pass a positive integer ONLY when "
        "you knowingly accept that (a) the analyzed rows are the first N (not a "
        "random sample), and (b) some engines (e.g. TabularEngine categorical "
        "encoding) decide which features to plan based on row counts and per-category "
        "statistics, so the reported metadata may differ from a full-input run.",
    )
    _add_engineer_args(p_explain, include_selection_args=False)
    p_explain.add_argument("--json", action="store_true", help="(Always JSON — flag accepted for symmetry.)")
    p_explain.set_defaults(func=_cmd_explain)

    return parser


def _add_io_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", "-i", required=True, help="Path to input file (CSV / Parquet / JSON).")
    p.add_argument("--output", "-o", required=True, help="Path to output file (CSV / Parquet / JSON).")
    p.add_argument("--input-format", choices=SUPPORTED_INPUT_FORMATS, help="Override input format detection.")
    p.add_argument("--output-format", choices=SUPPORTED_OUTPUT_FORMATS, help="Override output format detection.")
    p.add_argument(
        "--target",
        "-t",
        help="Target column name. Required when feature selection is applied "
        "(i.e. when --max-features / config max_features is set so the "
        "selector actually fits). With no max_features, raw feature "
        "generation runs without a target.",
    )
    p.add_argument(
        "--task-description",
        help="Natural-language ML task description (used by the LLM engine).",
    )


def _add_engineer_args(p: argparse.ArgumentParser, *, include_selection_args: bool = True) -> None:
    """Add ``AutoFeatureEngineer``-related flags to a subparser.

    ``include_selection_args=False`` omits selection-only flags
    (``--selection-methods`` and ``--correlation-threshold``) — these are
    silently ignored by the ``explain`` subcommand, which always runs with
    selection disabled. ``--max-features`` is *not* selection-only:
    ``AutoFeatureEngineer`` forwards it into engine construction (e.g. the
    tabular engine uses it to cap the number of generated features), so it
    is exposed even when ``include_selection_args=False`` to give callers
    a CLI-level handle on the engine output size.
    """
    p.add_argument(
        "--engines",
        nargs="+",
        choices=sorted(AutoFeatureEngineer.SUPPORTED_ENGINES),
        help="Engines to use (default: tabular).",
    )
    # ``--max-features`` is exposed on every engineer-using subcommand
    # because it caps engine output, not just selection — see the
    # ``AutoFeatureEngineer`` constructor and ``TabularEngine``.
    p.add_argument(
        "--max-features",
        type=int,
        help="Maximum number of features to generate / keep (forwarded to engines and selector).",
    )
    if include_selection_args:
        p.add_argument(
            "--selection-methods",
            nargs="+",
            choices=sorted(AutoFeatureEngineer.SUPPORTED_SELECTION_METHODS),
            help="Selection methods (default: mutual_info importance).",
        )
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
    # ``BooleanOptionalAction`` (Python 3.9+) provides both ``--verbose``
    # and ``--no-verbose`` so a config-supplied ``"verbose": true`` can be
    # explicitly turned off from the command line. ``default=None`` so the
    # absence of either flag means "fall through to config / default".
    p.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable verbose logging (or --no-verbose to override config).",
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns the process exit code; suitable for both the ``console_scripts``
    entry point (``featcopilot``) and ``python -m featcopilot``. Argparse
    usage errors (missing subcommand, unknown flag) and the cooperative
    ``--help`` / ``--version`` actions all normally raise :class:`SystemExit`;
    we trap those here and return their exit code so that programmatic
    callers (and agent harnesses) get a consistent integer-returning API.
    """
    parser = _build_parser()

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse uses SystemExit(0) for ``--help`` / ``--version`` and
        # SystemExit(2) for usage errors (also writing to stderr). We let the
        # output through but convert the exit into a return value so
        # ``main(argv) -> int`` is honored even on parse-time failures.
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        # Non-int code (e.g. error string): print to stderr, return 2.
        sys.stderr.write(f"{code}\n")
        return 2

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
        # Single deterministic stderr line so agents can parse the failure.
        # We deliberately do NOT call ``logger.exception(...)`` here:
        # FeatCopilot loggers write to stderr, which would append a second
        # timestamped traceback after our structured line and break the
        # CLI's "stderr is exactly one error message" contract. Internal
        # failure introspection is the caller's job (e.g. set
        # ``PYTHONFAULTHANDLER=1`` or attach a debugger).
        sys.stderr.write(f"featcopilot: unexpected error: {type(exc).__name__}: {exc}\n")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
