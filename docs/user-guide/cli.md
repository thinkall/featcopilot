# Command-Line Interface

FeatCopilot ships a stable, agent-friendly `featcopilot` CLI for using the
library from shells, CI pipelines, and **agentic / LLM tool-use** workflows
without writing Python glue. All subcommands accept `--json` for
machine-readable stdout; user-facing errors are written to **stderr** with
a non-zero exit code so that automation can parse failures
deterministically.

The CLI is installed automatically with the package via the
`[project.scripts]` entry point (`featcopilot = "featcopilot.cli:main"`),
so after `pip install featcopilot` the `featcopilot` command is available
on `$PATH`. The equivalent module form `python -m featcopilot ...` always
works regardless of how the package was installed.

## Subcommands

| Command | Purpose |
| --- | --- |
| `featcopilot info` | Print version, supported engines, selection methods, leakage guards, I/O formats, and a runtime `parquet_available` flag. |
| `featcopilot transform` | Read a CSV / Parquet / JSON file, run [`AutoFeatureEngineer`](../user-guide/overview.md), and write engineered features to an output file. |
| `featcopilot explain` | Fit and print a JSON document with `{name, explanation, code}` per feature for downstream LLM consumption (no output file is written). |

Run any subcommand with `--help` to see the full flag list:

```bash
featcopilot --help
featcopilot transform --help
featcopilot explain --help
```

## Output contract

All three subcommands honor the same agent-friendly contract:

* **`stdout`** carries the result. With `--json` (always implicit for
  `explain`), exactly one JSON document is written.
* **`stderr`** is reserved for failures. A successful run keeps `stderr`
  empty even when `AutoFeatureEngineer` emits leakage warnings or
  `verbose` logger output ─ those are surfaced via the JSON payload's
  `warnings` field instead. This same contract covers warnings emitted
  during pandas / pyarrow read or write phases (e.g. `DtypeWarning` on
  mixed-type CSVs, `FutureWarning` from a successful Parquet write):
  they are routed to the JSON `warnings` field, never to `stderr`.
* **Exit codes**: `0` on success; `2` for user-input errors (missing
  files, malformed config, unknown target, etc.); `1` for unexpected
  internal errors.

## `featcopilot info`

Discover capabilities without running an engineer:

```bash
featcopilot info --json
```

Sample (truncated) output:

```json
{
  "version": "0.3.7",
  "supported_engines": ["relational", "tabular", "text", "timeseries"],
  "supported_selection_methods": ["correlation", "importance", "mutual_info"],
  "supported_leakage_guards": ["block", "ignore", "warn"],
  "supported_input_formats": ["csv", "json"],
  "supported_output_formats": ["csv", "json"],
  "parquet_available": false
}
```

`parquet_available` reflects whether `pyarrow` or `fastparquet` is
importable in the current environment. The base FeatCopilot install does
not pin a parquet engine; install one with
`pip install pyarrow` (or `fastparquet`) to enable Parquet I/O.

## `featcopilot transform`

Run feature engineering on a tabular input and write the engineered
features to disk:

```bash
featcopilot transform \
    --input data.csv --target label --output features.csv \
    --engines tabular --max-features 50 \
    --json
```

Common flags:

| Flag | Purpose |
| --- | --- |
| `--input / -i` | Path to input file (CSV / Parquet / JSON). Required. |
| `--output / -o` | Path to output file. Required. |
| `--target / -t` | Target column. Required when feature selection is applied (i.e. when `--max-features` / config `max_features` is set). |
| `--input-format` / `--output-format` | Override format detection (`csv` / `parquet` / `json`). |
| `--engines` | One or more engines to enable (default: `tabular`). |
| `--max-features N` | Cap on engine output / selection. Forwarded both to engine constructors and to the selector. |
| `--no-selection` | Skip feature selection entirely (raw feature generation). |
| `--selection-methods` | Override the default `mutual_info importance` selection set. |
| `--leakage-guard` | How to handle suspicious column names: `warn` (default), `block`, or `ignore`. |
| `--include-target` | Re-attach the target column to the output file (collision-safe). |
| `--task-description` | Free-form ML task description forwarded to LLM-aware engines. |
| `--config FILE` | JSON config with nested keys (e.g. `llm_config`, `selection_methods`). CLI flags override config values. |
| `--verbose / --no-verbose` | Toggle verbose logging. With `--json`, log records are routed to the JSON `warnings` field rather than `stderr`. |
| `--gate-n-jobs` | Parallelism for the do-no-harm gate's RF (default 1; `-1` = all cores). |
| `--json` | Emit a one-line JSON status object on stdout instead of human-readable text. |

A successful `--json` run prints something like:

```json
{
  "status": "ok",
  "input": "data.csv",
  "output": "features.csv",
  "input_format": "csv",
  "output_format": "csv",
  "n_rows": 1000,
  "n_features": 47,
  "n_input_columns": 12,
  "n_generated_features": 47,
  "engines": ["tabular"],
  "selection_methods": ["mutual_info", "importance"],
  "max_features": 50,
  "target": "label",
  "selection_applied": true,
  "warnings": []
}
```

## `featcopilot explain`

Fit the engineer (without writing any output file) and print a JSON
catalog of generated features for downstream LLM consumption:

```bash
featcopilot explain --input data.csv --target label
```

Each entry in the `features` array contains the feature `name`, an
LLM-style natural-language `explanation`, and the executable Python
`code` used to produce it.

`explain` defaults to running on the **full** input so the metadata is
a faithful description of what a corresponding `transform` would
generate. Some engines (notably the tabular engine's categorical
encoding) consult per-row / per-category statistics when planning
features, so blind subsampling can silently change results. For very
large inputs where metadata-only `explain` should not pay full memory
or compute cost, opt in with:

```bash
featcopilot explain --input big.csv --target label --explain-sample-size 5000
```

The cap is a deterministic *head slice* (the first N rows), threaded
through `pd.read_csv(nrows=N)` for CSV so memory is bounded natively.
For Parquet / JSON pandas has no native row-limit, so the file is
fully read and then truncated; a `UserWarning` explaining the
limitation is emitted (and surfaced in the JSON `warnings` field) only
when the cap actually truncates the input.

## Configuration files

Pass `--config config.json` to provide nested keys that don't have
matching CLI flags, such as the `llm_config` engine kwargs:

```json
{
  "engines": ["tabular", "llm"],
  "max_features": 80,
  "selection_methods": ["mutual_info", "importance"],
  "llm_config": {
    "backend": "litellm",
    "model": "gpt-4o",
    "max_suggestions": 20
  }
}
```

Explicit CLI flags override values from the config file. Any malformed
scalar (e.g. `"max_features": "5"`, `"verbose": "false"`) is rejected
with a clean exit-2 error rather than failing later inside the
engineer.

## Parquet I/O

The base FeatCopilot install does not pin a parquet engine. To use
`--input file.parquet` / `--output file.parquet` (or the `parquet`
value of `--input-format` / `--output-format`), install one of:

```bash
pip install pyarrow      # recommended
# or
pip install fastparquet
```

Confirm with `featcopilot info --json`:

```json
{ "parquet_available": true, ... }
```

If neither engine is installed, attempting Parquet I/O fails with a
clean exit-2 error pointing at the missing dependency.

## Agentic-usage tips

* Always pass `--json`. Treat anything on `stderr` as a hard failure;
  treat anything on `stdout` as the JSON result.
* Treat the JSON `warnings` field as a list of human-readable
  diagnostic strings ─ it is non-empty for `transform` runs that
  generated leakage / mock-mode / sampling notices, and empty for
  fully clean runs.
* For long-running batch jobs, prefer `featcopilot transform` to
  `python -m featcopilot transform` only because the former is shorter;
  both invoke the exact same entry point.

## See also

* [Overview](overview.md) ─ the underlying `AutoFeatureEngineer` API.
* [Engines](engines.md) ─ what each engine generates.
* [LLM Features](llm-features.md) ─ configuring the LLM backend (used
  by `--config llm_config`).
