## Review Summary
- **Round**: 2
- **Theme**: Architecture & patterns
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\code\featcopilot\reviews\pr9\task-9-attempt-1-stage-attempt-2-review-2-grok-4.7.md
- **Issues Found**: 1
- **Verdict**: ISSUES_FOUND

## Evidence Checklist
- [x] Scope map `review-pr9/scope.json`: 721 paths, 127 non-generated. Architecture inspection covered the typed plan transport, public preprocessor/class-label contracts, Tabular positional API, and controller/state direction. Generated archives were not reread as raw evidence.
- [x] Base `bf119812162af87c29bc8d09bb3c705576b94f27` versus worktree, including uncommitted fixes and untracked `featcopilot/llm/_plan_transport.py`. AST comparison of changed public `__init__`/`fit`/`transform` signatures found no mid-list positional insertion. Live check: `TabularEngine(2, False, ["log"], None, False, True, 0.05, 0.5, 1)` keeps `min_samples_per_category=1` and `target_encode_cv=5`; a tenth positional argument raises `TypeError`.
- [x] `featcopilot/llm/_plan_transport.py` is the single isolation session used by `featcopilot/llm/copilot_client.py` (`isolated_plan`) and `benchmarks/llm_evaluation.py` (`CopilotPlanProvider`). `plan_generator.py` stays on `FeatureClient.send_prompt`; `SemanticEngine` opts Copilot plan mode into that transport. AST scan: no `featcopilot` module imports `benchmarks`. Legacy approve-all sessions remain the documented non-plan path.
- [x] Original-label weights go through shared `encode_class_weights` (`featcopilot/utils/model_preprocessing.py`) from both `classification.py` and `engines/predictive.py`. Catalog `class_labels` are restored in `benchmarks/execution.py` before `evaluation_classes` stringification; diagnostic providers are explicitly excluded. That matches `tests/test_benchmark_execution.py` preparation contracts.
- [x] Documented experimental policies, not defects: `docs/user-guide/classification.md` marks the ensemble experimental and says numeric/OHE is not native CatBoost; `selection.md` keeps predictive output raw and maps `class_weight` from original labels; target encoding is the shared `CrossFittedTargetEncoder` for selection and the final classifier; soft Copilot token requests remain explicitly unenforced.
- [x] Controller dependency direction is `benchmarks/execution.py` → `benchmarks/run_state.py`, with tabular imported lazily inside `execute`. Resume no longer rewrites `state.json` on open (`run_state.py` save only when `not resume`). Identity rejection skips that save.
- [ ] Full 789-test/mypy/wheel suite and 154 transport tests were not rerun, per the stage instructions. Assessment is independent source and small in-memory checks only.

## Scope and limitations
Reviewed the current worktree against the stated base, not HEAD `62cbb1c`. Did not read other review artifacts or driver progress. Did not inspect raw generated evidence bodies. Findings below are architectural invariant gaps, not style preferences.

## Issues

### Issue 1: Protocol failures consume resume budget without a terminal run status
- **Severity**: Medium
- **File**: benchmarks/execution.py
- **Line(s)**: 803, 909, 928-948, 949-1026
- **Description**: `execute` documents that final operational `run_status` lives in `manifest.json` (`benchmarks/TABULAR_PROTOCOL.md` lines 105-112). That write happens only after the dispatch `try` returns. `plan_jobs` raises `ValueError` when a locked confirmation cell count disagrees with preregistration (line 803), and dispatch raises `ValueError` when an isolated cell does not return exactly one result (line 909). Neither is `ResumeMismatchError` or `KeyboardInterrupt`, so the `finally` still saves elapsed time and any in-memory dataset identity, then the exception propagates and never reaches the status writer at lines 967-1026. Identity rejection correctly does not save; this path is the opposite. A later `--resume` still matches identity, re-prepares, hits the same deterministic raise, and charges more active time without recording `failed`.
- **Risk**: A non-retryable protocol failure looks resumable. Repeated resumes burn the preregistered active-time ceiling while `manifest.json` never becomes the operational record the protocol requires. Completed cells are not rewritten, but the run has no terminal status that stops the retry.
- **Suggested Fix**: Catch these controller failures separately from `ResumeMismatchError`. Write a terminal `run_status` such as `failed` to `manifest.json` before returning or re-raising, and refuse a later resume of that terminal protocol failure the same way a changed source is refused. Keep the current identity-rejection path unsaved. Do not rely on an uncaught `ValueError` to both persist budget consumption and skip the status writer.

## Resolution Log
_Updated by the driving agent as findings are addressed._

### Issue 1
- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

### Driver resolution update

- **Status**: Fixed after the architecture stage's two-attempt cap.
- **What changed**: Typed controller protocol failures now persist consumed time, terminal `protocol_error` status and a logged reason. A later resume is rejected before state access or clock changes. Malformed worker result contracts are rejected before checkpointing.
- **Why**: Terminal controller failures must not look like retryable evaluation failures or repeatedly consume an otherwise unchanged run's budget. Normal failed-cell retries, identity rejection and prior successful evidence remain separate.
- **How verified**: Controlled-clock protocol-count and malformed-worker regressions verify the terminal result and byte-identical rejected-resume outputs. The implementation execution/state suite passed 307 tests. The driver's combined execution/state/fitted-rule/plan suite then passed 465 tests, package mypy passed 52 files and the wheel built.
