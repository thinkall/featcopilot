## Review Summary
- **Round**: 3
- **Theme**: Edge cases & robustness
- **Mode**: resolved-plan
- **Model**: gpt-6.1-sol, reasoning=max; supplied High/two-model human override
- **Artifact**: C:\code\featcopilot\reviews\pr9\task-9-attempt-1-stage-attempt-2-review-3-gpt-6.1-sol.md
- **Scope**: Base `bf119812162af87c29bc8d09bb3c705576b94f27` to the current worktree, including pending changes and `featcopilot\llm\_plan_transport.py`.
- **Issues Found**: 1 introduced issue: Low
- **Verdict**: ISSUES_FOUND

## Evidence Checklist
- [x] Read the generated `stage3-attempt2\review-round-3.md` lines 1–100, the parent `scope.json` metadata, and repository `AGENTS.md`. Inspected actual implementations rather than relying on the input map. Did not read other current reviews or driver progress.
- [x] Reviewed fitted-state reset and publication paths in `classification.py`, `engines\predictive.py`, `engines\fitted_rules.py`, `engines\tabular.py`, `llm\semantic_engine.py`, and `transformers\sklearn_compat.py`. In-memory synthetic checks confirmed failed-refit invalidation for TabularEngine, FittedRuleEngine, predeclared-plan SemanticEngine, and NestedBaseClassifier.
- [x] Reviewed `utils\splitting.py`, fitted-rule target cross-fitting, numeric clipping, and categorical preprocessing. A 24-row synthetic temporal CV check produced three folds, each satisfying strict training-label availability before validation; the existing-split validator accepted those boundaries.
- [x] Reviewed all three provider wrappers and the new shared plan transport. Twelve concurrent fake requests per OpenAI/LiteLLM synchronous wrapper used exactly one worker/event loop; subsequent stop closed that loop/executor. Stop did not overlap requests.
- [x] In-memory fake plan-transport checks verified cooperative send timeout and subsequent disconnect/stop, including a hanging disconnect bounded by its cleanup grace. Replaced workspace creation with an in-memory context; no provider, subprocess transport, or temporary workspace was started. Locally inspected installed SDK 0.3.0 signatures/auth-response handling without starting a client.
- [x] Reviewed `benchmarks\tabular.py:1054–1241` and `benchmarks\execution.py:555–1083`. Synthetic protocol checks accepted a valid row and rejected five malformed shapes, values, or identities. Mocked process cleanup verified that a denied descendant kill does not prevent sibling/owned-parent cleanup and that the parent reap receives a bounded timeout.
- [x] Reviewed `benchmarks\run_state.py`, canonical confirmation reservation before acquisition, and the calibration/blending controllers' normalization/resume paths. In-memory RunState checks confirmed strict completed-cell/known-split identities and the narrowly scoped unprepared-interruption exception.
- [x] Inspected immutable archive generation/index publication and the failure/concurrent-publication tests in `tests\test_benchmark_run_state.py:134–277`; inspected canonical concurrent-admission tests in `tests\test_benchmark_execution.py:1584–1651`. These filesystem/concurrency tests were read, not executed.
- [x] Reproduced Issue 1 through all three current public asynchronous validation APIs without starting any provider. Executed the corresponding validators extracted from the base revision in memory: all three rejected the same input, establishing introduction.

## Limitations
- This is a targeted robustness review across the substantive PR, not an assertion that every line or generated evidence archive was exhaustively inspected. Generated evidence was approached through producers and contracts, not raw archive bodies.
- No downloads, installs, provider requests, real-data benchmarks, nested agents, source edits, or git writes were performed. Checks were small synthetic/in-memory probes; the parent's reported 1,033 tests, mypy, and wheel results were not independently rerun.
- Real OS process-tree races, crash durability, SQLite interprocess admission, and filesystem replacement were not exercised. Mock checks establish control flow, not OS-level guarantees.
- Cancellation-resistant/blocking SDK work and OS-denied process termination remain documented cooperative/best-effort limitations, not findings asserting nonexistent hard guarantees. No unsupported concurrent-stop contract was assumed. The modern SDK transport branch received static inspection, not validation against an installed modern SDK.

## Issues

### Issue 1: Validation-only calls now accept code that Python cannot compile
- **Severity**: Low
- **File**: `featcopilot\llm\_execution.py`
- **Line(s)**: 88–95; delegates at `llm\copilot_client.py:538`, `llm\openai_client.py:525`, and `llm\litellm_client.py:487`
- **Description**: The shared validator replaces the providers' previous `compile(code, ..., "exec")` syntax check with `ast.parse(code)`. Parsing alone does not perform compiler context/scope validation. With the public default `sample_data=None`, code containing an assignment followed by top-level `break`, `continue`, or `return` consequently returns `valid=True` despite being non-compilable.
- **Reproduction**:
  ```python
  from featcopilot.llm._execution import validate_feature_code

  code = "result = 1\nbreak"
  validate_feature_code(code)
  # {'valid': True, 'error': None, 'warnings': []}
  compile(code, "<feature>", "exec")
  # SyntaxError: 'break' outside loop
  ```
  The three public clients' `await client.validate_feature_code(code)` returned the same false-positive result, with `_is_started` remaining false. The analogous `return` and `continue` inputs also reproduced it.
- **Base comparison**: The OpenAI, LiteLLM, and Copilot validators at the stated base all compile before optional sample execution. Executing those extracted base methods with the same code returned `valid=False` and a syntax error in all three cases.
- **Risk**: A caller using the advertised validation-only API can accept/persist an invalid feature and discover the failure only during later computation. This does **not** bypass runtime compilation: the sample-execution path still catches the error, so impact is limited to validation without sample data.
- **Suggested Fix**: Compile the parsed AST in `"exec"` mode before returning success, while retaining the assignment/output checks and existing `SyntaxError` handling. Add no-sample regression cases for compiler-only errors through the shared helper and provider delegates.

## Pre-existing Concerns — Not Counted as Introduced Issues
- **Singleton legacy target encoding**: `engines\tabular.py:301–307` falls back to full-training target maps when cross-fitting has fewer than two rows. With one categorical row, `min_samples_per_category=1`, `onehot_ratio_threshold=0.0`, and `target_encode_ratio_threshold=1.0`, a target of `17.0` produces training encoding `[17.0]`; changing the target changes that feature directly. Independently executing the base TabularEngine source in memory produced the same `[17.0]`. This is therefore an existing own-target boundary concern, not a newly introduced regression. A separate follow-up could reject singleton target encoding or supply an explicit label-independent no-history policy.

## Resolution Log
_Updated by the driving agent as findings are addressed._

### Issue 1
- **Status**: Open
- **What changed**: No implementation changes; this review writes only its assigned artifact.
- **Why**: Restore compiler-level validation without executing generated code or requiring sample data.
- **How verified**: Current helper/public-client reproductions and independently executed base validators establish the regression. Post-fix verification remains pending.

### Driver resolution update

- **Status**: Fixed after the robustness stage's second-attempt cap.
- **What changed**: The shared validator compiles the parsed AST in `exec`
  mode before checking output assignment and optionally executing sample data.
- **Why**: Compiler scope validation is necessary even when callers intentionally
  omit samples; compilation itself must not execute provider-generated code.
- **How verified**: Sixteen helper/provider-delegate cases for top-level
  `break`, `continue`, `return` and `nonlocal` failed before the fix and passed
  afterward. A valid-but-raising program verifies that validation without
  samples does not execute it. All 84 engine-quality tests passed, with
  Black and Ruff passing.
