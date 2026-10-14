## Review Summary
- **Round**: 6
- **Theme**: Polish & hardening
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\code\featcopilot\reviews\pr9\task-9-attempt-1-stage-attempt-2-review-6-grok-4.7.md
- **Issues Found**: 5
- **Verdict**: ISSUES_FOUND

## Evidence Checklist
- [x] Read `AGENTS.md` logging, docstring, and naming rules; compared them with current worktree producers rather than HEAD-only diffs.
- [x] Inspected `benchmarks/tabular.py` `source_files()` (lines 1530-1544) and `source_fingerprint()` (lines 1546-1550). Provider-free probe on this worktree: listing took 16.187s for 121 returned paths / 5,147,316 bytes; a second `source_fingerprint()` call took 15.130s. Control: `benchmarks` `rglob("*.py")` took 15.623s (40 py files); the same tree with `tabular-results`, `.cache`, and `__pycache__` pruned during the walk took 0.001s (27 py files). `featcopilot` `rglob` took 0.005s. No frozen archive bodies were opened.
- [x] Traced live fingerprint calls in the budgeted loops: `benchmarks/execution.py` lines 696, 977, 1012, and 1060; `benchmarks/blending_evaluation.py` lines 1658, 1809, 1867, 1921, and 1976; `benchmarks/calibration_evaluation.py` lines 1267, 1525, 1585, and 1614. Controller tests stub the helper (`tests/test_benchmark_execution.py:33`, `tests/test_blending_evaluation.py:50`, `tests/test_calibration_evaluation.py:54`). `tests/test_benchmark_tabular.py:60-70` only checks a tiny temp tree's returned set.
- [x] Inspected the worktree cell loop in `benchmarks/execution.py` lines 971-1024 and the final publication block at lines 1055-1084. Compared with `git diff` against HEAD: per-cell `write_report` is new in the worktree; base `bf119812162af87c29bc8d09bb3c705576b94f27` has no `source_fingerprint`.
- [x] Compared `docs/user-guide/benchmarks.md` lines 47-69 with `benchmarks/README.md` lines 29-36 and 617-618, `benchmarks/TABULAR_PROTOCOL.md` cleanup/resume claims, `benchmarks/tabular.py` `stop_process_tree` lines 1070-1071, and `pyproject.toml` `benchmark-tabular` pins. Those user-guide sentences are additions versus base.
- [x] Inspected untracked `featcopilot/llm/_plan_transport.py` lines 175-208 against `docs/user-guide/llm-features.md` isolation/cleanup claims and `featcopilot/llm/copilot_client.py` synchronous bridge. No live provider call.
- [x] Confirmed `benchmarks/datasets.py` warning `print()` calls at lines 678, 716, and 756 are identical at base. Worktree diff adds no `print(` lines.

## Limitations
- Theme-limited review: performance, observability, documentation accuracy, and naming. Algorithmic leakage, training boundaries, and security were not re-litigated.
- No full pytest/coverage run, package install, provider call, real-data benchmark, or read of frozen result/archive bodies. The parent owns the full suite.
- Fingerprint timing is one Windows worktree measurement. A checkout without local `benchmarks/tabular-results` would not pay the 15s walk until local evidence accumulates; the pruned-walk control isolates the cause from hashing the 5MB allowlist.
- Did not time `write_report` on a large row set. The call pattern is sufficient to show repeated full publication; the absolute per-call cost is unmeasured.
- Other reviewers' current artifacts and driver metadata were not read.

## Issues

### Issue 1: Source identity checks walk generated benchmark trees on every cell
- **Severity**: High
- **Classification**: Introduced. `source_fingerprint` is absent at base `bf119812162af87c29bc8d09bb3c705576b94f27`. The helper and its per-cell callers are in the current worktree.
- **File**: benchmarks/tabular.py
- **Line(s)**: 1530-1550; callers at benchmarks/execution.py:977 and 1012, benchmarks/blending_evaluation.py:1867 and 1921, benchmarks/calibration_evaluation.py:1525 and 1585
- **Description**: `source_files()` documents that generated results are excluded, but it first `rglob`s all of `benchmarks` and only then drops paths whose parts include `tabular-results`. On this worktree that descent took about 15.6 seconds and returned 40 Python files; pruning those directories during the walk took 0.001 seconds. `source_fingerprint()` repeats that walk and then hashes about 5MB. The canonical, blending, and calibration controllers call it again inside job loops, including after each worker and, until a change is detected, before dispatch. `execution.py` then stores `perf_counter()` delta as `active_seconds`, so the walk is charged to the cumulative budget. There is no log that this recomputation is happening. Unit tests either stub the helper or point `ROOT` at a tiny tree, so CI does not see the cost.
- **Risk**: A normal local evidence tree can add roughly 15 seconds before each unfinished cell and another 15 seconds after each worker, before any model fit. Short cells and smoke runs become controller-bound. A multi-hundred-cell study can exhaust `--total-seconds` on directory walks, record that time as active experiment time, and still miss source edits that occur between the slow checks. Operators see unexplained gaps rather than a fingerprint log.
- **Suggested Fix**: Prune `tabular-results`, `.cache`, and `__pycache__` before descending. Cache the digest with a cheap signature of the included files (relative POSIX path, size, mtime) and rehash only when that signature changes. Log duration when a rehash runs. Add a temp-tree regression with a large excluded directory and assert the walk does not stat it. Do not regenerate frozen archives.
- **Reproduction**: From the repo root, time `benchmarks.tabular.source_files()` and `source_fingerprint()`, then time a `benchmarks` walk that removes `tabular-results`, `.cache`, and `__pycache__` from `dirnames` before descending. Observed on this worktree: 16.187s list, 15.130s fingerprint, 15.623s unpruned `rglob`, 0.001s pruned walk.

### Issue 2: Canonical controller republishes the full report after every recorded cell
- **Severity**: Medium
- **Classification**: Introduced in the current worktree. Base has no controller. HEAD published the report at the end; the worktree calls `write_report` inside the cell loop.
- **File**: benchmarks/execution.py
- **Line(s)**: 1014-1024, 1084
- **Description**: After every non-reusable cell, the loop sets `publication_pending` if records changed and never clears it until final publication. Each such cell rewrites `results.json` and `REPORT.md` through `tabular.write_report`, which rebuilds paired summaries and a 5,000-draw dataset bootstrap. The reserve then becomes at least twice that latest full-write duration and is subtracted from remaining cell budget. The later comment says publication is charged once; wall-clock accounting does not add it twice, but the loop still performs a full publication per recorded cell. Durable per-cell records already exist via `RunState.record`.
- **Risk**: Publication cost grows with the retained row set and is paid on every subsequent cell. That time is part of `active_seconds` and inflates `publication_reserve_seconds`, so later cells can be marked `budget_exhausted` even though the final archive is a similar write already performed. Large runs amplify both disk traffic and descriptive bootstrap work without making the immutable score archive more correct.
- **Suggested Fix**: Keep per-cell checkpoints as the crash record. Recalculate the publication reserve from a sampled or final write, not a full bootstrap report after every cell. Clear or narrow `publication_pending` so unchanged aggregate output is not rewritten. Align the "charged once" comment with the actual schedule. Preserve the end-of-run atomic archive switch.

### Issue 3: User guide contradicts the current runner contract
- **Severity**: Medium
- **Classification**: Introduced. These sentences are additions versus base and were not corrected by the worktree doc updates.
- **File**: docs/user-guide/benchmarks.md
- **Line(s)**: 47, 52, 68-69
- **Description**: The guide says output directories must be new and that cached scores are never reused. `benchmarks/README.md` lines 617-618 and `benchmarks/tabular.py` `--resume` say the opposite for an explicit resume: fingerprint-identical cells are reused and previous attempts are not overwritten. The same guide says external worker deadlines actually stop work, including child processes. `stop_process_tree` is a best-effort kill with a shared five-second wait, and the protocol says OS-denied cleanup is reported rather than a guaranteed hard kill. The extra pin sentence says only that sklearn is below 1.8. `pyproject.toml` and `benchmarks/README.md` lines 29-36 now pin scikit-learn `>=1.7.2,<1.8` and AutoFeat `==2.1.1`, and they distinguish sklearn 1.8/1.9 support as outside that extra.
- **Risk**: Operators following the user guide will discard resumable runs or believe a timeout always reaps children. Installers can also miss the new lower sklearn bound and the AutoFeat exact pin that the benchmark README already states.
- **Suggested Fix**: State that a new directory is required unless `--resume`, and that resume reuses only fingerprint-identical cells. Describe worker stop as bounded best-effort termination with logged cleanup failures. Copy the current extra pins and the separately managed 1.8/1.9 note from `benchmarks/README.md`.

### Issue 4: Failed plan-workspace cleanup does not identify the leftover directory
- **Severity**: Low
- **Classification**: Introduced. `featcopilot/llm/_plan_transport.py` is an untracked worktree file; base has no shared plan transport.
- **File**: featcopilot/llm/_plan_transport.py
- **Line(s)**: 191, 203-207
- **Description**: Session disconnect/stop failures are logged with only the exception type. If `TemporaryDirectory.cleanup()` raises `OSError`, the warning says an isolated workspace may remain, but it omits `directory.name`, `errno`, and `strerror`. The user guide correctly says a warning is emitted and the workspace may remain; it does not say the path is recoverable from that warning. The prefix `featcopilot-plan-` is not a substitute on a busy temp directory.
- **Risk**: An operator who must delete an OS-busy plan workspace cannot find it from the log. Retrying or inspecting the original request error will not reveal the leftover path. Logging only the type also hides permission-versus-busy distinctions.
- **Suggested Fix**: Include the workspace path and the OS error's errno/strerror in the cleanup warning. For session shutdown, log the exception message when it does not contain prompt text. Keep the original request error authoritative.

### Issue 5: Dataset loaders still warn with print and fall back to synthetic data
- **Severity**: Low
- **Classification**: Pre-existing, not introduced. The same `print(f"Warning: Could not load real dataset: {e}")` calls exist at base lines 678, 716, 756, and others. The worktree diff does not add or remove `print(` lines.
- **File**: benchmarks/datasets.py
- **Line(s)**: 678
- **Description**: Real-data load failures are reported with `print()` and then replaced by synthetic frames. That bypasses `featcopilot.utils.logger` and can mix warnings into stdout. `AGENTS.md` forbids `print()` for logging. This is legacy loader behavior, not a new benchmark-controller defect.
- **Risk**: Scripted callers can miss the substitution or treat the warning as payload output. The synthetic fallback itself is existing behavior and should not be changed under this polish finding.
- **Suggested Fix**: Route these warnings through `get_logger(__name__).warning` without changing the fallback policy. Defer if this pass must stay limited to introduced controller and documentation defects.

## Resolution Log
_Updated by the driving agent as findings are addressed._

### Issue 1
- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

### Issue 2
- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

### Issue 3
- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

### Issue 4
- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

### Issue 5
- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

## Driver resolution update

- **Issues 1-2 - Repair in progress**: Preserve exact byte-based source identity
  while pruning excluded trees before descent, and remove redundant per-cell
  full-report work without weakening durable checkpoints, atomic final
  publication or cumulative active-time accounting. An mtime/size-only digest
  cache is intentionally not planned: same-size edits with restored mtimes must
  still invalidate source identity.
- **Issue 3 - Fixed**: The user guide now distinguishes new runs from explicit,
  fingerprint-validated resume; describes bounded best-effort process-tree
  cleanup; and records the current reference-compatible dependency pins and
  separately managed sklearn 1.8/1.9 environments.
- **Issue 4 - Fixed**: Leftover-workspace warnings include the generated
  directory, error class and numeric errno. Raw SDK/error messages, arbitrary
  strerror and exception filenames remain excluded to avoid disclosing
  provider data. Original primary exceptions and successful outcomes remain
  authoritative. Plan-mode tests passed 133 cases on each SDK 0.3.0 and 1.0.16,
  including exact workspace/errno, privacy and outcome-preservation assertions.
- **Issue 5 - Deferred, pre-existing**: The independently verified legacy
  dataset-loader warning/fallback behavior is outside these introduced repairs.
- **Stage outcome**: This second-attempt verdict remains `ISSUES_FOUND`. The
  High stage advances with unresolved findings, not a clean verdict; subsequent
  repair notes do not constitute a third independent review.

### Driver follow-up: Issues 1-2 repaired

- **Issue 1 - Fixed**: Top-down discovery prunes excluded result/cache trees
  before descent while retaining the existing allowlist order and POSIX hash
  names. Every identity check still hashes included bytes. Deterministic
  regressions cover pruning and same-size, same-mtime source edits; metadata-only
  caching was rejected because it weakens the identity guarantee.
- **Issue 2 - Fixed**: Full reporting and descriptive bootstrap now occur once
  at finalization. Per-cell durable checkpoints remain the crash record.
  Resume recovery, immutable atomic archive publication, charged publication
  time, estimated reserves and explicit overruns are preserved.
- **How verified**: The controller/source-discovery repair suite passed 512
  tests, with Black, Ruff, targeted mypy and whitespace checks. The protocol
  documents finalization-only full reporting and byte-based identity checks.
- **Limits**: Included-byte hashing still costs time, and publication reserves
  remain estimates. No frozen result or descriptor was regenerated.
