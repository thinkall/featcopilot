"""Bounded UCI Online Retail development pilot, not an official RelBench task.

First run ``python -m benchmarks.relational_evaluation --prepare-only --output PATH``.
After freezing source, use ``--evaluate --registration-sha256 SHA --output PATH``;
SHA is the SHA256 of PATH/registration.json. Preparation fits no predictive model.
Evaluation refuses changed source, data, configuration, features, or fold boundaries.

The fixed comparison is calendar/simple history versus a fixed relational
aggregation library, NOT automatic multi-hop discovery. Both use identical
train-only preprocessing/HPO from benchmarks.tabular. Twelve sequential cells
(two representations, two learners, three forward folds) require at most 48
model fits. The supervised CLI caps cumulative preparation plus evaluation at
two hours, and each cell at eight minutes; eight CPU cores/threads
and 4 GiB per-cell worker-tree RSS are upper bounds. No LLM is used.

Raw bytes and short-lived worker payloads stay under benchmarks/.cache. Reports
contain counts, hashes, temporal boundaries and metrics, never customer IDs,
individual targets, predictions, or transaction rows. Do not commit the cache.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from benchmarks import tabular
from benchmarks.run_state import atomic_json, fingerprint
from featcopilot.engines.point_in_time import PointInTimeAggregation, PointInTimeRelationalEngine
from featcopilot.utils.logger import get_logger
from featcopilot.utils.splitting import make_cv_splits, make_holdout_splits

logger = get_logger(__name__)
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "benchmarks" / ".cache" / "online-retail-352"
SOURCE_URL = "https://archive.ics.uci.edu/dataset/352/online+retail"
API_URL = "https://archive.ics.uci.edu/api/dataset?id=352"
DATA_URL = "https://archive.ics.uci.edu/static/public/352/data.csv"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/legalcode"
COLLECTION_START = pd.Timestamp("2010-12-01")
COHORT_END = pd.Timestamp("2011-01-01")
# The provider gives a final date, not full-day completeness: exclude that day.
COVERAGE_END = pd.Timestamp("2011-12-09")
METHODS = ("simple_history", "expanded_pit")
LEARNERS = ("linear", "lightgbm")
SEED = 42


class PilotConfig(BaseModel):
    """Immutable resource limits; statistical choices are not CLI tunables."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    customers: int = Field(default=200, ge=1, le=200, strict=True)
    threads: int = Field(default=8, ge=1, le=8, strict=True)
    case_timeout: float = Field(default=480, ge=1, le=480, allow_inf_nan=False)
    total_seconds: float = Field(default=7200, ge=10, le=7200, allow_inf_nan=False)
    memory_limit_mb: int = Field(default=4096, ge=128, le=4096, strict=True)


@dataclass
class PilotData:
    """Local-only matrices and safe, row-free preparation provenance."""

    queries: pd.DataFrame
    target: pd.Series
    features: dict[str, pd.DataFrame]
    provenance: dict[str, Any]


@dataclass
class PilotFold:
    """Global positional indices for an outer fold and its purged HPO holdout."""

    train: np.ndarray
    test: np.ndarray
    inner_train: np.ndarray
    inner_valid: np.ndarray


def protocol(config: PilotConfig) -> dict[str, Any]:
    """Describe the complete fixed development protocol before data/model work."""
    return {
        "version": 1,
        "role": "development",
        "dataset": "uci-online-retail-352",
        "task": "next_7_day_net_line_item_spend_returning_customers",
        "source": SOURCE_URL,
        "data_url": DATA_URL,
        "attribution": "Daqing Chen, Online Retail, UCI, DOI:10.24432/C5BW33",
        "license": "CC BY 4.0",
        "license_url": LICENSE_URL,
        "config": config.model_dump(),
        "seed": SEED,
        "cohort": "Uniform seeded sample of sorted unique cleaned December 2010 CustomerIDs, no future filtering",
        "cohort_start": COLLECTION_START.isoformat(),
        "cohort_end_exclusive": COHORT_END.isoformat(),
        "cutoffs": "Mondays 2011-01-03 through 2011-12-05, dropping label ends after coverage_end",
        "coverage_end_exclusive": COVERAGE_END.isoformat(),
        "label": "Sum Quantity*UnitPrice in [cutoff, cutoff+7 days), zero if no recorded line items",
        "availability": "Invoice-generated time assumed immediately available; no posting-delay metadata",
        "clock": "Unshifted source wall clock; UTC normalization is arithmetic only, not a timezone claim",
        "duplicates": "Retain all exact duplicate records; no source line-item key resolves legitimate repeats",
        "refunds": "Keep signed Quantity including cancellations; do not flip signs using InvoiceNo",
        "missing_history": "Empty count/sum = 0; other aggregates missing; train-only median/zero imputation",
        "outer": "3 expanding forward folds via make_cv_splits; entire equal-time blocks retained",
        "inner": "1 forward 25% holdout per outer train via make_holdout_splits; 3 HPO candidates",
        "purge": "Training label_end_times must be strictly before validation cutoff; equality is purged",
        "entity_scope": "Same-customer future-period prediction, not unseen-entity transfer",
        "methods": list(METHODS),
        "method_description": "Fixed relational aggregation library; no learned feature or multi-hop discovery",
        "precomputation": "Label-free PIT features are computed once; each query independently excludes all later rows",
        "timing": "Shared PIT preparation counts toward the total budget; per-cell FE time measures only train-only imputation",
        "windows_days": [7, 30, 90],
        "aggregations": ["count", "sum", "mean", "std", "max", "last"],
        "std": "Population standard deviation (ddof=0)",
        "last_ties": "Stable original CSV row order for equal invoice timestamps",
        "learners": list(LEARNERS),
        "candidate_parameters": {learner: tabular.candidate_parameters(learner) for learner in LEARNERS},
        "primary_metric": "RMSE in recorded sterling net line-item spend (lower is better)",
        "secondary_metrics": ["MAE", "R2"],
        "expected_cells": 12,
        "max_model_fits": 48,
        "interpretation": [
            "Recorded net spend is a proxy, not profit, invoice totals, or complete customer purchasing activity.",
            "Zero means no qualifying recorded spend, not proof of no purchases.",
            "Source coverage boundaries are provider dates, not inferred from sampled customer activity.",
            "Historical prefix includes December 2010; early 90-day windows are necessarily left-truncated.",
            "Known line items may update between query cutoffs within a held-out block; models do not refit.",
            "One small preselected cohort, one seed, one source; no population or multi-dataset claims.",
            "Repeated-customer/time errors are dependent; fold summaries are descriptive, not IID intervals.",
        ],
    }


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _fetch(url: str, limit: int) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as response:
        content: bytes = response.read(limit + 1)
    if len(content) > limit:
        raise ValueError("Approved source exceeds its fixed byte limit")
    return content


def acquire_source(cache: Path = CACHE) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Cache approved original bytes with primary-source rights evidence and hashes."""
    cache.mkdir(parents=True, exist_ok=True)
    csv_path, provenance_path = cache / "data.csv", cache / "source.json"
    if provenance_path.exists():
        evidence = json.loads(provenance_path.read_text(encoding="utf-8"))
        content = csv_path.read_bytes()
        if evidence["raw_sha256"] != _sha(content):
            raise ValueError("Cached original CSV changed; do not silently replace a frozen source")
    else:
        page = _fetch(SOURCE_URL, 4 * 1024**2)
        if LICENSE_URL.encode() not in page or b"Creative Commons Attribution 4.0 International" not in page:
            raise ValueError("Primary-source explicit CC BY 4.0 evidence is absent")
        api = _fetch(API_URL, 1024**2)
        metadata = json.loads(api)["data"]
        if metadata["data_url"] != DATA_URL or metadata["num_instances"] != 541909:
            raise ValueError("UCI source metadata changed from the approved dataset")
        content = _fetch(DATA_URL, 128 * 1024**2)
        evidence = {
            "source_url": SOURCE_URL,
            "api_url": API_URL,
            "data_url": DATA_URL,
            "attribution": "Daqing Chen, Online Retail, DOI:10.24432/C5BW33",
            "license": "CC BY 4.0",
            "license_url": LICENSE_URL,
            "license_statement": "This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.",
            "rights_page_sha256": _sha(page),
            "api_sha256": _sha(api),
            "raw_sha256": _sha(content),
            "raw_bytes": len(content),
            "provider_rows": 541909,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        }
        (cache / "rights.html").write_bytes(page)
        (cache / "metadata.json").write_bytes(api)
        csv_path.write_bytes(content)
        atomic_json(provenance_path, evidence)
    for name, key in (("rights.html", "rights_page_sha256"), ("metadata.json", "api_sha256")):
        if _sha((cache / name).read_bytes()) != evidence[key]:
            raise ValueError("Cached source/rights evidence changed")
    frame = pd.read_csv(io.BytesIO(content), dtype={"CustomerID": "string", "InvoiceNo": "string"})
    if len(frame) != evidence["provider_rows"]:
        raise ValueError("Original CSV row count differs from approved source metadata")
    return frame, evidence


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply explicit sequential exclusions without deleting signed refunds or duplicates."""
    required = {
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
    }
    if not required.issubset(raw.columns):
        raise ValueError("Online Retail source lacks required columns")
    frame = raw.copy()
    audit: dict[str, Any] = {
        "input_rows": len(frame),
        "exact_duplicate_rows_retained_raw": int(frame.duplicated().sum()),
    }
    missing = frame.CustomerID.isna() | frame.CustomerID.astype("string").str.strip().eq("").fillna(False)
    audit["removed_missing_customer"] = int(missing.sum())
    frame = frame.loc[~missing].copy()
    price = pd.to_numeric(frame.UnitPrice, errors="coerce")
    invalid_price = price.isna() | ~np.isfinite(price) | (price <= 0)
    audit["removed_nonpositive_or_nonfinite_price_after_missing_customer"] = int(invalid_price.sum())
    frame = frame.loc[~invalid_price].copy()
    frame["UnitPrice"] = price.loc[~invalid_price]
    quantity = pd.to_numeric(frame.Quantity, errors="coerce")
    if quantity.isna().any() or not np.isfinite(quantity).all():
        raise ValueError("Nonfinite/malformed quantities require an explicit new source policy")
    frame["Quantity"] = quantity
    customer = pd.to_numeric(frame.CustomerID, errors="coerce")
    if customer.isna().any() or not np.isfinite(customer).all() or (customer % 1 != 0).any():
        raise ValueError("CustomerID must identify an integral customer")
    frame["CustomerID"] = customer.astype("int64").astype(str)
    if pd.api.types.is_datetime64_any_dtype(frame.InvoiceDate):
        dates = pd.to_datetime(frame.InvoiceDate)
    else:
        # Explicit US-style source dates plus unambiguous ISO, never inferred day-first.
        dates = pd.to_datetime(frame.InvoiceDate, format="%m/%d/%Y %H:%M", errors="coerce")
        for fmt in ("%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            dates = dates.fillna(pd.to_datetime(frame.InvoiceDate, format=fmt, errors="coerce"))
    if dates.isna().any() or dates.dt.tz is not None:
        raise ValueError("InvoiceDate requires valid naive source-clock timestamps")
    if (dates < COLLECTION_START).any() or (dates >= COVERAGE_END + pd.Timedelta(days=1)).any():
        raise ValueError("InvoiceDate is outside provider-declared collection dates")
    frame["InvoiceDate"] = dates
    frame["net_spend"] = frame.Quantity * frame.UnitPrice
    if not np.isfinite(frame.net_spend).all():
        raise ValueError("Nonfinite signed net spend is unsupported")
    audit.update(
        retained_rows=len(frame),
        retained_exact_duplicates=int(frame[list(raw.columns)].duplicated().sum()),
        retained_negative_quantity=int((frame.Quantity < 0).sum()),
        retained_zero_quantity=int((frame.Quantity == 0).sum()),
        retained_cancellation_rows=int(frame.InvoiceNo.astype(str).str.upper().str.startswith("C").sum()),
        exclusion_order=["missing_customer", "nonpositive_or_nonfinite_price"],
    )
    return frame.reset_index(drop=True), audit


def select_cohort(history: pd.DataFrame, limit: int = 200) -> list[str]:
    """Sample only customers observed in cleaned December 2010 history, seed 42."""
    if not 1 <= limit <= 200:
        raise ValueError("Cohort size must be between 1 and 200")
    december = history.loc[(history.InvoiceDate >= COLLECTION_START) & (history.InvoiceDate < COHORT_END)]
    eligible = np.asarray(sorted(december.CustomerID.unique()))
    if not len(eligible):
        raise ValueError("No attributable December 2010 customer history")
    chosen = np.random.default_rng(SEED).choice(eligible, min(limit, len(eligible)), replace=False)
    return sorted(chosen.tolist())


def aggregation_specs() -> list[PointInTimeAggregation]:
    """Declare the fixed label-independent history library, including recency inputs."""
    operations: tuple[Literal["count", "sum", "mean", "std", "max", "last"], ...] = (
        "count",
        "sum",
        "mean",
        "std",
        "max",
        "last",
    )
    timing_specs: tuple[tuple[str, Literal["min", "last"]], ...] = (
        ("first_invoice_days", "min"),
        ("last_invoice_days", "last"),
    )
    specs = [
        PointInTimeAggregation(
            name=f"spend_{days}d_{operation}",
            table="line_items",
            entity_column="CustomerID",
            available_at="InvoiceDate",
            aggregation=operation,
            value_column=None if operation == "count" else "net_spend",
            window=f"{days}D",
        )
        for days in (7, 30, 90)
        for operation in operations
    ]
    specs += [
        PointInTimeAggregation(
            name=name,
            table="line_items",
            entity_column="CustomerID",
            available_at="InvoiceDate",
            aggregation=operation,
            value_column="invoice_days",
        )
        for name, operation in timing_specs
    ]
    return specs


def prepare_data(raw: pd.DataFrame, config: PilotConfig) -> PilotData:
    """Build all fixed-cohort weekly queries, past-only features and uncensored labels."""
    history, audit = clean_transactions(raw)
    cohort = select_cohort(history, config.customers)
    history = history.loc[history.CustomerID.isin(cohort)].copy()
    history["invoice_days"] = (history.InvoiceDate - COLLECTION_START).dt.total_seconds() / 86400
    requested = pd.date_range("2011-01-03", "2011-12-05", freq="7D")
    cutoffs = requested[requested + pd.Timedelta(days=7) <= COVERAGE_END]
    queries = pd.MultiIndex.from_product([cutoffs, cohort], names=["cutoff", "CustomerID"]).to_frame(index=False)
    queries["label_end"] = queries.cutoff + pd.Timedelta(days=7)
    keys = queries[["CustomerID", "cutoff"]]
    specs = aggregation_specs()
    engine = PointInTimeRelationalEngine([*specs], "CustomerID", "cutoff")
    engine.fit(keys, related_tables={"line_items": history})
    aggregate = engine.transform(keys)
    calendar = pd.DataFrame(
        {
            "annual_sin": np.sin(2 * np.pi * queries.cutoff.dt.dayofyear / 365.25),
            "annual_cos": np.cos(2 * np.pi * queries.cutoff.dt.dayofyear / 365.25),
            "spend_7d_count": aggregate.spend_7d_count,
            "spend_7d_sum": aggregate.spend_7d_sum,
            "recency_days": (queries.cutoff - COLLECTION_START).dt.total_seconds() / 86400
            - aggregate.last_invoice_days,
            "tenure_days": (queries.cutoff - COLLECTION_START).dt.total_seconds() / 86400
            - aggregate.first_invoice_days,
        }
    )
    expanded = calendar.copy()
    for spec in specs:
        if spec.window is not None:
            expanded[spec.name] = aggregate[spec.name]
    label_spec = next(spec for spec in specs if spec.name == "spend_7d_sum")
    labels = PointInTimeRelationalEngine([label_spec], "CustomerID", "cutoff")
    labels.fit(keys, related_tables={"line_items": history})
    label_queries = keys.assign(cutoff=queries.label_end)
    target = labels.transform(label_queries)["spend_7d_sum"].rename("next_7d_net_spend")
    features = dict(zip(METHODS, (calendar, expanded), strict=True))
    provenance = {
        "cleaning": audit,
        "cohort_customers": len(cohort),
        "cohort_sha256": fingerprint(cohort),
        "cohort_history_rows": len(history),
        "requested_cutoffs": len(requested),
        "retained_cutoffs": len(cutoffs),
        "end_censored_queries_removed": (len(requested) - len(cutoffs)) * len(cohort),
        "query_rows": len(queries),
        "first_cutoff": cutoffs.min().isoformat(),
        "last_cutoff": cutoffs.max().isoformat(),
        "last_label_end": queries.label_end.max().isoformat(),
        "aggregations": [spec.model_dump() for spec in specs],
        "feature_columns": {method: frame.columns.tolist() for method, frame in features.items()},
        "feature_sha256": {method: _frame_sha(frame) for method, frame in features.items()},
        "queries_sha256": _frame_sha(queries),
        "target_sha256": _frame_sha(target.to_frame()),
    }
    return PilotData(queries, target, features, provenance)


def _frame_sha(frame: pd.DataFrame) -> str:
    return _sha(pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype="<u8").tobytes())


def make_folds(data: PilotData) -> list[PilotFold]:
    """Use the same purged inner holdout as tabular.run_case, with three outer folds."""
    q, y = data.queries, data.target
    outer = make_cv_splits(y, "regression", 3, SEED, timestamps=q.cutoff, label_end_times=q.label_end)
    if len(outer) != 3:
        raise ValueError("The preregistered task requires exactly three outer folds")
    result = []
    for train, test in outer:
        inner_train, inner_valid = make_holdout_splits(
            y.iloc[train],
            "regression",
            1,
            SEED,
            test_size=0.25,
            timestamps=q.cutoff.iloc[train],
            label_end_times=q.label_end.iloc[train],
        )[0]
        result.append(PilotFold(train, test, train[inner_train], train[inner_valid]))
    return result


def fold_provenance(data: PilotData, folds: list[PilotFold]) -> list[dict[str, Any]]:
    """Describe every temporal boundary and hash positions without exporting rows."""
    result = []
    for number, fold in enumerate(folds):
        entry: dict[str, Any] = {"fold": number}
        for name in ("train", "test", "inner_train", "inner_valid"):
            positions = getattr(fold, name)
            subset = data.queries.iloc[positions]
            entry[name] = {
                "rows": len(positions),
                "position_sha256": _sha(positions.astype("<i8").tobytes()),
                "first_cutoff": subset.cutoff.min().isoformat(),
                "last_cutoff": subset.cutoff.max().isoformat(),
                "last_label_end": subset.label_end.max().isoformat(),
                "customers": subset.CustomerID.nunique(),
            }
        result.append(entry)
    return result


def cell_payload(
    data: PilotData, fold: PilotFold, number: int, method: str, learner: str, config: PilotConfig
) -> dict[str, Any]:
    """Adapt precomputed, label-free PIT features to the existing bounded runner."""
    if method not in METHODS or learner not in LEARNERS:
        raise ValueError("Unknown preregistered representation or learner")
    return {
        "X": data.features[method],
        "y": data.target,
        "timestamps": data.queries.cutoff,
        "label_end_times": data.queries.label_end,
        "train_idx": fold.train,
        "test_idx": fold.test,
        "dataset": "uci-online-retail-352",
        "family_id": "uci-online-retail",
        "source": "UCI",
        "task": "regression",
        # Baseline here means pass-through of the declared, precomputed representation,
        # NOT that an unsuccessful expansion can fall back to simple-history scores.
        "method": "baseline",
        "seed": SEED,
        "fold": number,
        "split_sha256": fingerprint([fold.train.tolist(), fold.test.tolist()]),
        "role": "development",
        "expected_cells": 12,
        "config": {**config.model_dump(), "seed": SEED, "learners": [learner]},
    }


def _planned_cells(status: str) -> list[dict[str, Any]]:
    return [
        {"fold": fold, "method": method, "learner": learner, "status": status}
        for fold in range(3)
        for learner in LEARNERS
        for method in METHODS
    ]


def _versions() -> dict[str, str]:
    packages = ("numpy", "pandas", "scikit-learn", "lightgbm", "pydantic", "threadpoolctl", "psutil")
    result = {"python": sys.version}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not installed"
    return result


def _registration(
    data: PilotData, folds: list[PilotFold], source: dict[str, Any], config: PilotConfig
) -> dict[str, Any]:
    return {
        "protocol": protocol(config),
        "source_fingerprint": tabular.source_fingerprint(),
        "source_files": {str(path.relative_to(ROOT)): _sha(path.read_bytes()) for path in tabular.source_files()},
        "versions": _versions(),
        "dataset": source,
        "preparation": data.provenance,
        "folds": fold_provenance(data, folds),
    }


def _finish_pending(report: dict[str, Any], status: str) -> None:
    for cell in report["cells"]:
        if cell["status"] == "pending":
            cell["status"] = status


def _execute(output: Path, config: PilotConfig, evaluate: bool, registration_sha256: str | None) -> int:
    import psutil
    from threadpoolctl import threadpool_limits

    process = psutil.Process()
    affinity = process.cpu_affinity()[: config.threads]
    process.cpu_affinity(affinity)  # Inherited by every model child: a hard CPU-core cap.
    report_path = output / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    started = time.monotonic()
    try:
        with threadpool_limits(limits=config.threads):
            raw, source = acquire_source()
            data = prepare_data(raw, config)
            folds = make_folds(data)
        registration = _registration(data, folds, source, config)
        registration_path = output / "registration.json"
        if not evaluate:
            atomic_json(registration_path, registration)
            report.update(status="prepared", registration_sha256=_sha(registration_path.read_bytes()))
        else:
            frozen = registration_path.read_bytes()
            if _sha(frozen) != registration_sha256 or json.loads(frozen) != registration:
                raise ValueError("Frozen registration differs from current source, data, environment, or protocol")
            report.update(status="running", registration_sha256=registration_sha256)
            atomic_json(report_path, report)
            workspace = CACHE / "cells" / uuid.uuid4().hex
            for index, cell in enumerate(report["cells"]):
                remaining = config.total_seconds - (time.monotonic() - started) - 10
                if remaining <= 0:
                    _finish_pending(report, "total_budget_exhausted")
                    break
                payload = cell_payload(data, folds[cell["fold"]], cell["fold"], cell["method"], cell["learner"], config)
                rows = tabular.run_isolated(payload, workspace / str(index), min(config.case_timeout, remaining))
                if len(rows) != 1:
                    raise ValueError("The bounded adapter requires exactly one learner result per cell")
                report["cells"][index] = {
                    **rows[0],
                    "method": cell["method"],
                    "worker_adapter": "tabular baseline pass-through of fixed PIT features",
                    "log": str((workspace / str(index) / "worker.log").relative_to(ROOT)),
                }
                atomic_json(report_path, report)
            report["status"] = "complete" if all(c["status"] == "success" for c in report["cells"]) else "incomplete"
        report["cpu_affinity"] = affinity
        report["worker_seconds"] = time.monotonic() - started
        atomic_json(report_path, report)
        return 0 if report["status"] in {"prepared", "complete"} else 1
    except Exception as exc:
        logger.exception("Relational pilot failed; no scores will be substituted")
        report.update(status="error", error_type=type(exc).__name__)
        _finish_pending(report, "error")
        atomic_json(report_path, report)
        return 1


def main(argv: list[str] | None = None) -> int:
    """Prepare or explicitly authorize one frozen, wall-clock-supervised pilot."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    parser.add_argument("--output", type=Path, default=CACHE / "pilot")
    parser.add_argument("--registration-sha256")
    parser.add_argument("--customers", type=int, default=200)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--case-timeout", type=float, default=480)
    parser.add_argument("--total-seconds", type=float, default=7200)
    parser.add_argument("--memory-limit-mb", type=int, default=4096)
    parser.add_argument("--managed-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    config = PilotConfig(**{name: getattr(args, name) for name in PilotConfig.model_fields})
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("Keep pilot artifacts inside the repository (raw cache remains ignored)")
    if args.evaluate and not args.registration_sha256:
        parser.error("--evaluate requires the explicitly frozen --registration-sha256")
    if args.managed_worker:
        return _execute(output, config, args.evaluate, args.registration_sha256)
    report_path = output / "report.json"
    prior_seconds = 0.0
    if args.prepare_only:
        output.mkdir(parents=True, exist_ok=True)
        if report_path.exists() or (output / "registration.json").exists():
            parser.error("Use a new output directory; previous evidence is never overwritten")
        atomic_json(output / "protocol.json", protocol(config))
        report: dict[str, Any] = {"status": "preparing", "cells": _planned_cells("not_run_preparation")}
    else:
        frozen = (output / "registration.json").read_bytes()
        if _sha(frozen) != args.registration_sha256:
            parser.error("Registration SHA256 does not match")
        registration = json.loads(frozen)
        if registration["protocol"] != protocol(config):
            parser.error("Configuration differs from frozen registration")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["status"] != "prepared":
            parser.error("Only an unmeasured prepared registration can be evaluated; no silent retries")
        prior_seconds = report["supervised_wall_seconds"]
        report["preparation_wall_seconds"] = prior_seconds
        report.update(status="starting", cells=_planned_cells("pending"))
    atomic_json(report_path, report)
    if prior_seconds >= config.total_seconds - 5:
        report["status"] = "total_budget_exhausted"
        _finish_pending(report, "total_budget_exhausted")
        atomic_json(report_path, report)
        return 1
    command = [
        sys.executable,
        "-m",
        "benchmarks.relational_evaluation",
        "--evaluate" if args.evaluate else "--prepare-only",
        "--managed-worker",
        "--output",
        str(output),
    ]
    for name, value in config.model_dump().items():
        command.extend(["--" + name.replace("_", "-"), str(value)])
    if args.registration_sha256:
        command.extend(["--registration-sha256", args.registration_sha256])
    environment = os.environ.copy()
    environment.update({name: str(config.threads) for name in tabular.THREAD_VARIABLES})
    started = time.monotonic()
    with (output / ("evaluate.log" if args.evaluate else "prepare.log")).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            # Reserve cleanup time inside the overall cap; this also bounds download
            # and feature preparation, unlike only limiting model subprocesses.
            code = process.wait(timeout=max(1, config.total_seconds - prior_seconds - 5))
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            tabular.stop_process_tree(process)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            status = "total_budget_exhausted" if isinstance(exc, subprocess.TimeoutExpired) else "interrupted"
            report["status"] = status
            _finish_pending(report, status)
            atomic_json(report_path, report)
            code = 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if code and report["status"] in {"preparing", "starting", "running"}:
        report["status"] = "worker_error"
        _finish_pending(report, "worker_error")
    report["supervised_wall_seconds"] = time.monotonic() - started
    report["cumulative_wall_seconds"] = prior_seconds + report["supervised_wall_seconds"]
    report["total_seconds_cap"] = config.total_seconds
    atomic_json(report_path, report)
    logger.info("Pilot status: %s; report: %s", report["status"], report_path)
    if report.get("registration_sha256"):
        logger.info("Frozen registration SHA256: %s", report["registration_sha256"])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
