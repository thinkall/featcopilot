## Review Summary
- **Round**: 5
- **Theme**: Testing & coverage
- **Mode**: resolved-plan; High, stage attempt 2, top-level attempt 1
- **Model**: gpt-6.1-sol, reasoning=max; caller-assigned explicit two-model override
- **Artifact**: C:\code\featcopilot\reviews\pr9\task-9-attempt-1-stage-attempt-2-review-5-gpt-6.1-sol.md
- **Issues Found**: 1
- **Verdict**: ISSUES_FOUND

## Evidence Checklist
- [x] Read the first 100 lines of the supplied stage prompt and its parent-directory `scope.json`. Used `bf119812162af87c29bc8d09bb3c705576b94f27` to the current working tree, not the artifact-only HEAD change. Included the untracked `featcopilot\llm\_plan_transport.py`. Independently compared all 103 substantive Python source/test entries in the supplied map with their current SHA-256 values: no drift.
- [x] Examined meaningful training-boundary assertions, not just feature counts: `tests\test_importance_screening.py:239-388` exercises real tree/logistic improvements, coefficient attribution and held-out perturbations; `tests\test_importance_resampling.py:388-495` checks bag-local mappings/preprocessing and final full-training frequency state; `tests\test_plan_validation.py:290-413` instruments actual fit/discovery boundaries and independently perturbs gate and outer-test data. Also inspected corresponding predictive, preprocessing and plan implementations.
- [x] Traced failure and exhausted-resume handling through `benchmarks\execution.py:789-1027` and `benchmarks\run_state.py`, against `tests\test_benchmark_execution.py:861-1026,1869-1891`. Assertions check retained metadata/split identity before writes, successful-cell preservation, no exhausted acquisition, and unchanged evidence. Inspected source-initialization interruption/recovery and immutable archive publication tests rather than treating preserved historical evidence as a coverage defect.
- [x] Checked cross-platform archive naming against actual producers: `benchmarks\run_state.py` uses POSIX ZIP member/allowlist names, as does the calibration manifest. `tests\test_benchmark_run_state.py:300-319` and `tests\test_calibration_evaluation.py:189-214` deliberately introduce Windows separators and compare allowlists with ZIP members, including resume preservation.
- [x] Inspected the SDK boundary in `_plan_transport.py`, both Copilot adapters, and `tests\test_llm_plan_mode.py`. Fakes autospec the installed SDK and use its actual `AssistantMessageData` type. Deadline tests cover startup, authentication, session creation and send; separate shutdown grace, cancellation and original-error identity are asserted. Inspected real worker-file decoding/supervisor cleanup assertions in `tests\test_benchmark_tabular.py:843-1053` and controller protocol-error tests in `tests\test_benchmark_execution.py:453-640`.
- [x] Examined public classification/ensemble, regression-averaging, relational, reference-adapter, target-encoding and task-routing tests alongside selected source paths. Distinct mocks isolate orchestration; real sklearn fits, independent arithmetic, fold-local encodings, inference/pickle parity, duplicate indexes, nullable values and train-frozen preprocessing provide behavioral checks beyond those mocks.
- [x] Independently ran **36 focused existing tests: passed** using the established `files\benchmark-env\Scripts\python.exe` (NumPy 2.4.4, pandas 2.3.3, sklearn 1.7.2, Copilot SDK 0.3.0). Selected exact tests from numeric clipping, model preprocessing, fitted rules, predictive/extended candidates, importance screening, classification, splitting, point-in-time and LLM execution. Included signed-zero bit preservation, nullable/category/boolean dtypes, duplicate indexes, large-integer time codes, independent synthetic gain and own-target exclusion.
- [x] Independently ran **29 additional focused existing tests: passed** for all provider deadline phases, shared startup deadline, bounded shutdown, preservation of original failures, workspace cleanup errors, concurrent pooled-client initialization and descendant cleanup. SDK/process objects remained mocked; an outer in-memory workspace stand-in prevented filesystem creation. This validates async/error logic, not real filesystem isolation.
- [x] Ran **two additional in-memory desired-regression assertions: both failed**, each because `BenchmarkProtocolError` was not raised for an impossible success payload. Separately passed accepted negative-loss rows through the actual `tabular.summarize`: a 0.5 baseline and -1.0 candidate produced `mean_loss_reduction=3.0`. See Issue 1.

## Scope / Limits

This is an independent full-PR-scope testing review, not an exhaustive line-by-line reading of every changed file or generated archive. The 721-entry input map includes substantial generated/frozen evidence; its raw bodies were not exhaustively inspected. Source/test hash checking establishes snapshot identity, not test correctness.

The independent executions used `-B`, disabled pytest caching/plugin autoload, explicitly loaded pytest-asyncio, and made no provider/network calls. No source/test files, git state, dependencies or historical evidence were changed. No full suite, real benchmark, package install, download, commit or nested agent was run. The parent's reported affected-test, mypy and wheel results were not treated as independent proof.

Durability/archive tests requiring filesystem fixtures were inspected, not rerun here. The SDK tests use the installed 0.3.0 contract; the modern `RuntimeConnection` branch, actual CLI/authentication, OS-busy workspace deletion, and forced termination of cancellation-resistant SDK code were not independently exercised. Native OpenFE/AutoFeat searches, optional learner combinations, historical archive regeneration, Linux and the complete supported Python/dependency matrix were not rerun. Optional/version skips and deliberate historical-archive preservation are not, by themselves, findings.

## Issues

### Issue 1: Worker-protocol tests miss semantically impossible successful results
- **Severity**: Medium
- **File**: `benchmarks\execution.py`
- **Line(s)**: 581-591; related missing cases in `tests\test_benchmark_execution.py:516-577`
- **Description**: `_validated_worker_row` checks finiteness for both metrics and feature counts, but explicitly exempts `loss` from the nonnegative check and accepts fractional `n_output_features`. Thus an otherwise valid success with `loss=-1.0`, or with `n_output_features=1.5`, passes the new protocol validator. The actual producer defines loss as RMSE or log loss (`benchmarks\tabular.py:743-787`), neither of which can legitimately be negative; a feature count cannot be fractional. The malformed-worker test matrix covers nonfinite loss, missing counts and negative runtime, but neither of these domains. This is introduced: both the validator and its test module are additions relative to the requested base.
- **Risk**: An invalid worker result is accepted and checkpointed as a reusable success instead of becoming a terminal protocol failure. This is not merely a missing assertion: actual accepted -1.0 candidate rows against 0.5 baseline rows produce a claimed **300% loss reduction** in the real paired summarizer (`benchmarks\tabular.py:1273-1277`). No existing real-data result is alleged to contain such a value.
- **Reproduction**: Run the following provider-free test against the current source; both parametrizations should raise, but currently do not. Equivalent in-memory `unittest.assertRaises` probes were executed independently and produced two failures, without filesystem fixtures.

  ```python
  import pytest
  from benchmarks.execution import BenchmarkProtocolError, _validated_worker_row

  @pytest.mark.parametrize("changed", [
      {"loss": -1.0},
      {"n_output_features": 1.5},
  ])
  def test_impossible_success_is_a_protocol_error(changed):
      payload = {
          "dataset": "synthetic", "family_id": "synthetic-family",
          "role": "development", "source": "synthetic",
          "task": "classification", "method": "baseline",
          "seed": 42, "fold": 0, "split_sha256": "fixed",
          "expected_cells": 1, "config": {"learners": ["linear"]},
      }
      row = {key: value for key, value in payload.items() if key != "config"}
      row.update(
          learner="linear", status="success", loss=0.5,
          pipeline_seconds=1.0, inner_fe_seconds=0.0,
          outer_fe_seconds=0.0, n_input_features=1, n_output_features=1,
      )
      row.update(changed)
      with pytest.raises(BenchmarkProtocolError):
          _validated_worker_row([row], payload)
  ```

- **Suggested Fix**: Validate nonnegative loss and a non-boolean integral output-feature count separately from runtime fields. Add these cases to the existing controller malformed-output matrix, asserting terminal `protocol_error`, no record for the malformed cell, preservation of previous successes/evidence, and resume rejection. Retain acceptance of legitimate zero loss and integer counts.

## Resolution Log

### Issue 1
- **Status**: Open
- **What changed**: Review artifact only; source and tests intentionally unchanged.
- **Why**: This invocation is an independent read-only reviewer. The driving agent owns any implementation/test changes.
- **How verified**: Two in-memory expected-rejection tests fail on the reviewed tree; a separate actual-summary probe returns a 3.0 candidate loss reduction. The 65 selected existing tests pass but do not cover these invalid domains.

### Driver resolution update

- **Status**: Fixed after the testing stage's second-attempt cap.
- **What changed**: Successful worker losses must be nonnegative; output
  feature counts must be non-boolean nonnegative integers, separate from
  floating timing/metric validation.
- **Why**: Impossible worker evidence must never become a reusable success
  or enter a paired improvement summary.
- **How verified**: Added negative-loss, fractional-count and boolean-count
  cases to the terminal controller matrix, with and without preceding successes.
  The invalid cases assert terminal status, unchanged prior evidence and
  rejected resume. A zero-loss/integer-count success case preserves legitimate
  perfect predictions. The negative/fractional cases reproduced failures
  before the fix; the full affected modules are being validated.

The full canonical/controller modules completed with 473 tests passing,
including the new domain and source-fingerprint regressions.
