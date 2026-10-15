## Review Summary
- **Round**: 6
- **Theme**: Polish & hardening
- **Mode**: resolved-plan
- **Model**: gpt-6.1-sol, reasoning=max
- **Artifact**: C:\code\featcopilot\reviews\pr9\task-9-attempt-1-stage-attempt-1-review-6-gpt-6.1-sol.md
- **Issues Found**: 3
- **Verdict**: ISSUES_FOUND

## Evidence Checklist
- [x] Read the first 100 lines of the supplied `stage6-attempt1\review-round-6.md` and the parent `scope.json` inventory. Reviewed against base `bf119812162af87c29bc8d09bb3c705576b94f27` and worktree HEAD `62cbb1cf4f434a6bbfd956b34c2615e761441e46`, including relevant pending changes and `featcopilot\llm\_plan_transport.py`. Did not read current review artifacts or driver progress.
- [x] Examined predictive candidate limits, sequential evaluation, thread policy and fit accounting in `featcopilot\engines\predictive.py`. A provider-free 48-row, three-column synthetic Ridge check with `max_candidates=100` observed exactly **22**, **38**, and **16** learner fits for `(importance_resamples, stability_repeats)` of `(1,3)`, `(5,3)`, and `(5,1)` respectively; all matched `selection_report_["learner_fit_count"]` and the documented bounds.
- [x] Traced classifier search/refit, per-member inference, diagnostics and serialization through `featcopilot\classification.py` and shared preprocessing. Reproduced Issue 1 using four small raw-classifier fits, not a benchmark.
- [x] Read SDK startup/send/error/cleanup paths in the Copilot, OpenAI and LiteLLM clients, typed-plan generation and the shared isolated transport. Local signature inspection accepted the installed SDK 0.3.0 isolation contract without constructing or starting a real SDK client.
- [x] Compared `pyproject.toml`, installation/authentication documentation and reference loaders. Local environment: Python 3.13.13, sklearn 1.8.0, SciPy 1.17.1, Pydantic 2.13.3, Copilot SDK 0.3.0. Checked the Python/Linux/x86-64-specific SciPy exclusion and parsed core Python sources with Python 3.10 grammar. The mocked version-gate check reproduced Issue 2 without changing dependencies.
- [x] Inspected worker supervision, sampled RSS/deadline accounting, publication and archive producers in `benchmarks\tabular.py`, `benchmarks\execution.py` and `benchmarks\run_state.py`. New result parts have generation-specific identities and an authoritative ordered index; source publication is atomic. Also inspected the calibration/blending orchestration and reference method identities. Did not reconstruct or inspect historical raw result archives.
- [x] Cross-checked public predictive, classifier, fitted-plan and sklearn examples against their implementations and restrictions. Read only narrative `ANALYSIS.md` evidence for the fixed-average development and regression-policy replication links. Their development-only qualifications, negative outcomes and unchanged-default boundaries agree with the main README; no SOTA/no-harm promotion is warranted or asserted here.
- [x] Independently checked introduction against the base: `classification.py`, `reference_methods.py` and the classifier guide are additions; the benchmark extra is new. The optional text-provider fallback already existed at base `semantic_engine.py:279-300`, whereas the unconditional error guarantee in Issue 3 is newly added documentation.

## Limitations
- Focused final-lens review, not an exhaustive reread of every source/test line or a replacement for the parent's final gates.
- No source edits, git writes, subagents, installs, downloads, live providers, real-data measurements or full test-suite runs. Synthetic checks used in-memory data and disabled Python bytecode writes.
- Actual runtime checks used the installed Python 3.13/sklearn 1.8/SDK 0.3 environment. Python 3.10 was grammar-checked, not executed; other SDK versions and minimum dependency combinations were not installed or exercised.
- Historical performance numbers were checked through the allowed narrative and current producers, not independently recomputed from frozen raw results.

## Issues

### Issue 1: Optional fitted-state fingerprinting still aborts fitting for valid serialization-hook failures
- **Severity**: Medium
- **File**: `featcopilot\classification.py`
- **Line(s)**: 590-592; fit-time caller at 813-815
- **Description**: `diagnostics()` treats only `PicklingError`, `TypeError` and `AttributeError` as unavailable serialization. `pickle.dumps()` also propagates exceptions from a supplied estimator's `__getstate__`/`__reduce__` hooks, including `ValueError` and `RuntimeError`. Because the public classifier calls member diagnostics before publishing its fitted state, such an exception turns otherwise successful in-memory model training into a failed `FeatureEnsembleClassifier.fit()`. This contradicts the newly documented behavior at `docs\user-guide\classification.md:159-162`, where unavailable pickling should yield a missing hash and warning, not prevent in-memory use.
- **Risk**: A cloneable, probabilistic classifier that deliberately cannot export its native state is unusable through the public wrapper, even with `engineered=False`; completed training is discarded solely because optional observability cannot serialize it.
- **Reproduction**: Executed this provider-free synthetic case:

  ```python
  import numpy as np
  import pandas as pd
  from sklearn.linear_model import LogisticRegression
  from featcopilot.classification import FeatureEnsembleClassifier, NestedBaseClassifier

  class InMemoryClassifier(LogisticRegression):
      def __getstate__(self):
          raise ValueError("serialization deliberately unavailable")

  X = pd.DataFrame({
      "x": np.linspace(-2, 2, 40),
      "z": np.tile([0.0, 1.0, 2.0, 3.0], 10),
  })
  y = np.tile([0, 1], 20)
  parameters = [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}]
  base = NestedBaseClassifier(
      InMemoryClassifier(max_iter=100), parameters, engineered=False, n_jobs=1,
  ).fit(X, y)
  assert base.predict_proba(X.iloc[:2]).shape == (2, 2)
  FeatureEnsembleClassifier(
      InMemoryClassifier(max_iter=100), parameters,
      mode="single", engineered=False, n_jobs=1,
  ).fit(X, y)
  ```

  The standalone base fitted and predicted successfully. The public wrapper raised `ValueError: serialization deliberately unavailable` and had no `estimators_`. Calling the fitted base's diagnostics failed identically.
- **Suggested Fix**: Separate serialization from hashing so ordinary exceptions raised specifically by `pickle.dumps()` can be translated into an explicit unavailable fingerprint on the public non-strict path. Keep unrelated diagnostics/hash failures visible and preserve the benchmark's strict fingerprint behavior. Add a real serialization-hook regression test, rather than only monkeypatching `_sha256` for the three currently handled exception classes.
- **Introduction**: New public classifier and new serialization-availability contract; this file/API is absent from the base.

### Issue 2: The benchmark installation extra admits AutoFeat versions that every native adapter rejects
- **Severity**: Medium
- **File**: `pyproject.toml`
- **Line(s)**: 81
- **Description**: `benchmark-tabular` declares `autofeat>=2.1.0`, but `benchmarks\reference_methods.py:409-414` requires exactly `2.1.1` for `autofeat-default`, `autofeat-native` and their native downstream variants. Installing the documented extra can therefore leave an already installed 2.1.0 untouched even though it cannot run those documented references. The open-ended declaration also accepts subsequent versions that the loader intentionally rejects. This is a dependency contract mismatch, not a request to relax the reference's reproducibility pin.
- **Risk**: Users can complete the documented installation successfully and then spend preparation/dispatch resources on native-reference cells that all fail the adapter's version gate, leaving incomplete comparator coverage.
- **Reproduction**: Parsed the actual extra with `tomllib` and `packaging.requirements.Requirement`. With version lookup mocked to AutoFeat `2.1.0`, pandas `2.3.3` and sklearn `1.7.2`, the requirement accepted AutoFeat while `_require("autofeat", "2.1.1")` raised `ReferenceDependencyError: autofeat reference requires 2.1.1; found 2.1.0`. The identical check accepted 2.1.1. No package installation or upstream fitting was performed.
- **Suggested Fix**: Align the benchmark-only dependency with the existing loader contract, e.g. `autofeat==2.1.1`, and test that the declared extra resolves within the adapter's accepted release set. Keep the separate safe-scale diagnostic's additional exact runtime pins explicit; do not weaken frozen-reference validation.
- **Introduction**: Both `benchmark-tabular` and the exact native adapter are new relative to the base. This does not allege that the legacy restricted `autofeat` arm uses the same exact-version gate.

### Issue 3: The new unconditional provider-error guarantee omits the preserved text-only degradation path
- **Severity**: Low
- **File**: `docs\user-guide\llm-features.md`
- **Line(s)**: 34-36
- **Description**: The guide says provider, timeout and empty-response failures “raise errors,” but the code-mode `SemanticEngine` still catches failures from optional text suggestions at `featcopilot\llm\semantic_engine.py:483-506`, logs a warning and continues with built-in text features. For text-only inputs there is no subsequent general-feature request to propagate the error. The provider clients raise correctly; the documented engine-level guarantee needs this distinction.
- **Risk**: A caller relying on the new guide to detect provider/model outages through failed `fit()` calls can instead receive a successfully fitted, built-in-only text representation. This is observable by warning, not by the promised exception.
- **Reproduction**: Injected a trusted fake client whose `suggest_features` raises `TimeoutError("synthetic provider deadline")` and whose validation delegates to the local validator. On 16 distinct strings longer than 20 characters in a single text column, `SemanticEngine().fit_transform(X)` succeeded after one failed provider attempt, returned shape `(16, 12)`, and retained 11 features with origin `custom`. It emitted the explicit “using built-in text features” warning.
- **Suggested Fix**: Qualify the documentation: startup and required general/plan requests fail, while legacy optional code-mode text suggestions can degrade to explicitly logged built-in features. Document how callers identify that degradation. Preserve the legacy behavior rather than changing defaults just to match the new prose.
- **Introduction**: Documentation-only finding. The text fallback is independently verified as pre-existing in the base; the unconditional failure guarantee is newly introduced.

## Resolution Log
_Updated by the driving agent as findings are addressed._

### Issue 1
- **Status**: Open
- **What changed**: Pending; reviewer made no source changes.
- **Why**: Optional serialization diagnostics currently reject an otherwise usable in-memory classifier.
- **How verified**: Synthetic classifier with a real failing `__getstate__` hook; base prediction succeeded while public wrapper fitting failed.

### Issue 2
- **Status**: Open
- **What changed**: Pending; reviewer made no dependency changes.
- **Why**: Installation metadata and the intentionally exact native-reference runtime gate disagree.
- **How verified**: Actual declared requirement plus mocked dependency-version lookup; 2.1.0 satisfies the extra but fails the loader, while 2.1.1 passes.

### Issue 3
- **Status**: Open
- **What changed**: Pending; reviewer made no documentation or legacy-behavior changes.
- **Why**: Newly added prose promises stronger failure propagation than the retained text-engine path provides.
- **How verified**: Text-only fake-provider timeout produced a warning and successful built-in-only output; base fallback and new documentation introduction were checked separately.

### Driver resolution update

- **Issue 1 - Fixed, pending retry**: Optional fitted-state serialization is
  isolated from hashing. Ordinary exceptions raised specifically by estimator
  pickle hooks produce an explicit unavailable hash and warning on the public
  non-strict path. Strict benchmark diagnostics still re-raise; unrelated hash
  failures and actual model export failures remain visible. Real `__getstate__`
  `ValueError`/`RuntimeError` regressions reproduced the failure before repair
  and now permit in-memory fit/prediction without exposing private error text.
- **Issue 2 - Fixed, pending retry**: The benchmark extra pins AutoFeat 2.1.1
  and uses sklearn `>=1.7.2,<1.8`, matching the native and calibration contracts.
  Metadata tests reject the previously permitted incompatible versions.
- **Issue 3 - Documentation fixed**: The guide now distinguishes required
  provider/plan failures from logged legacy optional text degradation. It
  explains that built-in features have `custom` origin and do not establish
  provider success. Legacy behavior is unchanged.
- **How verified**: Sixteen focused serialization/dependency tests passed;
  package mypy passed 52 files, with Black and Ruff passing. Broader final
  validation follows after all hardening fixes are complete.

### Driver final validation follow-up

The full-suite run exposed one stale benchmark test injection: it patched the
former combined `_sha256` helper instead of the now-separated serialization
boundary. Production strict mode was unchanged and still re-raises serializer
errors. The test now injects at `pickle.dumps` and covers `PicklingError`,
`ValueError` and `RuntimeError`, asserting original exception identity.
All 18 related strict/public diagnostic tests passed after this test-only
correction. Exact source hashes confirm no production change since the
completed full-suite run; its 93.42% coverage remains applicable.
