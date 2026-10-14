## Review Summary
- **Round**: 2
- **Theme**: Architecture & patterns
- **Mode**: resolved-plan
- **Model**: gpt-6.1-sol, reasoning=max
- **Artifact**: C:\code\featcopilot\reviews\pr9\task-9-attempt-1-stage-attempt-2-review-2-gpt-6.1-sol.md
- **Issues Found**: 1
- **Verdict**: ISSUES_FOUND

One low-severity inconsistency remains in the shared fitted-rule numeric boundary. No Medium, High, or Critical architectural findings were established in the inspected paths.

## Evidence Checklist
- [x] Read the supplied generated header, lines 1–100, and `scope.json`; used `bf119812162af87c29bc8d09bb3c705576b94f27`, not the artifact-only HEAD, for source comparisons. Final SHA-256 checks matched all 103 non-generated Python paths in the intake map; this is a stability check, not a claim of line-by-line coverage.
- [x] Read `AGENTS.md` and `pyproject.toml`. Statically inspected imports in all 52 package Python modules: no production-to-`benchmarks` import statements. Public package/classifier imports also passed with Copilot, OpenAI, LiteLLM, benchmark modules, and optional learner imports deliberately blocked.
- [x] Traced `featcopilot\llm\_plan_transport.py` into both `copilot_client.py` and `benchmarks\llm_evaluation.py`. Twelve independent fake-SDK checks covered modern/legacy construction, restrictive session settings, explicit model identity, existing-login rejection, and cleanup after startup/session/body/disconnect failures. Valid text and four rejected response cases also passed. Temporary-directory creation was mocked; no SDK processes or providers ran.
- [x] Compared the actual `TabularEngine.__init__` AST against the exact base: all nine legacy positional parameters remain in order; `target_encode_cv` is keyword-only. A corrected nine-argument invocation passed. The initial probe mistakenly supplied ten positional arguments; that probe error is not a product finding.
- [x] Exercised public target preprocessing with mixed typed categories, multiclass cross-fitting, batch-invariant inference, uncached training matrices, metadata rejection, and failed-refit invalidation. Numeric clipping preserved boolean roles and cleared fitted state after a rejected complex-valued refit.
- [x] Exercised the public classifier with original labels `[-7, 11, 40]`, original-label class weights, owned estimator copies, cloning, target encoding, JSON-safe diagnostics, and failed-refit invalidation. All checks passed.
- [x] Independently tested `benchmarks\execution.py::_prepare_dataset` using an in-memory catalog stub: observed codes `[2, 0, 2, 0]` with an unused middle class retained the correct original evaluation labels after contiguous re-encoding.
- [x] Traced `BaseEngine._transform_fit_data`, `FittedRuleEngine`, `SemanticEngine`, and both sklearn wrappers. Provider-free grouped target-mean probes verified group-excluded training encodings, positional duplicate-index preservation, and separate frozen inference maps.
- [x] Inspected `run_state.py`, general execution, calibration/blending controllers, and outcome normalizers for identity validation, whole-bundle checkpointing, exposure admission, source invalidation, and cumulative-time accounting. Pure in-memory `RunState` probes rejected changed completed-cell/dataset identities and allowed genuinely unprepared interrupted cells to acquire inputs. The source archive allowlist includes the untracked shared transport and excludes reviews.
- [x] Independently reproduced Issue 1 on both fitting and inference paths, compared rejection behavior at adjacent real-valued boundaries, and verified that `featcopilot\engines\fitted_rules.py` does not exist in the exact base.

## Scope and Limitations

This is an independent architecture review of the current worktree, including untracked transport code and uncommitted fixes. Examination concentrated on the runtime interfaces, dependency direction, preprocessing contracts, and controller/state boundaries above; it was not a line-by-line review of all 721 code/evidence paths. Other reviews, driver progress, and other agents' artifacts were not read.

Synthetic probes used the existing `benchmark-env\Scripts\python.exe` with NumPy 2.4.4, pandas 2.3.3, and sklearn 1.7.2, with bytecode writing disabled. No full suite, real benchmark, acquisition, installation, authentication, or live provider call was performed. The supplied 789-test, mypy-52-module, wheel, and 154-transport-follow-up results were not independently rerun or treated as substitutes for source inspection.

The documented experimental restrictions are intentional, not findings: exactly three parameter candidates; fixed clipping/feature budgets; explicit prediction modes without a fabricated union feature space; five-fold categorical target encoding with strict class support; distinct legacy one-hot normalization versus opt-in type-preserving target normalization; and soft-token Copilot evaluation that does not claim an enforced token ceiling. No generic controller refactor or abstraction replacement is requested.

## Issues

### Issue 1: Shared statistical-rule numeric conversion does not enforce the real-valued boundary
- **Severity**: Low
- **File**: `featcopilot\engines\fitted_rules.py`
- **Line(s)**: 60–64; affected callers at 276–277, 319, and 383–384
- **Description**: `_numeric` accepts complex dtypes because pandas classifies them as numeric, then converts them to `float`. Consequently, `standardize`, `clip`, and fitting of `group_mean` discard imaginary components instead of rejecting unsupported inputs. The same engine explicitly rejects complex operands for its rowwise operators at lines 67–70; `NumericQuantileClipper` and classification-frame validation likewise reject them. The shared conversion boundary therefore gives the statistical operators a different, lossy dtype contract.
- **Evidence**: In the established benchmark environment, fitting these three rules to numeric values `[1+10j, 2+20j, 3+30j]` succeeded and produced the real-only results, emitting `ComplexWarning` rather than a validation error. Separately, after fitting on real values `[1, 2, 3]`, transforming `[1+100j, 2-200j, 3+300j]` produced exactly the same `standardize` and `clip` outputs as the original real batch. The adjacent public preprocessors rejected that complex dtype. Exact-base tree inspection confirms this engine/helper is newly introduced; this is not attributed to the artifact-only HEAD or presented as a regression in legacy `TabularEngine`.
- **Risk**: Unsupported input can become a successfully fitted/exported statistical plan over only part of the supplied numerical data. A cast warning is weaker than the explicit fail-closed validation used by the neighboring numeric APIs. The impact is confined to complex-valued rule sources, hence Low severity.
- **Suggested Fix**: Reject complex source dtypes in `_numeric` before conversion, giving all numerical fitted operators the same real-valued boundary. Add focused tests for complex `standardize`/`clip` inputs during both fit and transform, and for complex `group_mean` value columns during fit. Preserve existing real, nullable, and all-missing behavior.

## Resolution Log

### Issue 1
- **Status**: Open
- **What changed**: Pending parent disposition. This reviewer changed no source or tests and performed no git writes.
- **Why**: Central validation prevents the lossy conversion at the existing shared boundary without redesigning the operator architecture or altering documented experimental policies.
- **How verified**: Two independent in-memory probes confirmed complex-source acceptance and imaginary-component loss, including inference after a real-valued fit; adjacent public-boundary rejection and exact-base introduction were checked independently. No fix has been applied or verified.

### Driver resolution update

- **Status**: Fixed after the architecture stage's second-attempt cap.
- **What changed**: The existing shared `_numeric` helper rejects complex dtypes before any lossy float conversion.
- **Why**: Statistical rules now honor the same real-valued input boundary as rowwise rules without changing real, nullable or all-missing behavior.
- **How verified**: Five new fit/inference regressions failed with `ComplexWarning` and no exception before the fix, then passed. The complete fitted-rule/typed-plan/wrapper suite passed 168 tests; Black and Ruff passed.
