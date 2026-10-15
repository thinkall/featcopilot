## Review Summary
- **Round**: 4
- **Theme**: Detailed correctness
- **Mode**: resolved-plan
- **Model**: gpt-6.1-sol, reasoning=max
- **Artifact**: C:\code\featcopilot\reviews\pr9\task-9-attempt-1-stage-attempt-2-review-4-gpt-6.1-sol.md
- **Issues Found**: 3
- **Verdict**: ISSUES_FOUND

Independent High review, stage attempt 2, under the human two-model override. Two Medium correctness regressions and one Low optional-metadata integration defect were reproduced. No source changes were made.

## Evidence Checklist
- [x] Read the first 100 lines of the assigned generated `stage4-attempt2\review-round-4.md` and the parent `scope.json`; assessed actual repository source rather than treating the trimmed map as a complete diff. Did not consult other reviews or driver progress.
- [x] Compared changes against `bf119812162af87c29bc8d09bb3c705576b94f27`. Reviewed HEAD remained `62cbb1cf4f434a6bbfd956b34c2615e761441e46`; uncommitted fixes and the new `featcopilot\llm\_plan_transport.py` were included.
- [x] Traced fitted-target, numeric, index and selection data flow through `featcopilot\engines\{tabular,fitted_rules,predictive,point_in_time}.py`, `classification.py`, the sklearn wrappers, and the shared frequency/splitting/clipping/model-preprocessing helpers. Read the corresponding relevant test contracts.
- [x] Inspected the schema-only plan, generated-code and provider paths in `semantic_engine.py`, `plan_generator.py`, `_plan_transport.py`, `_execution.py`, and the Copilot/OpenAI/LiteLLM clients. Provider-free checks confirmed lexical scope, Series index realignment and serialized ownership of concurrent synchronous wrapper calls. Local SDK inspection/imports did not start a client or probe authentication.
- [x] Traced catalog admission/cache/split policy, canonical nested tuning and result normalization in `benchmarks\catalog.py`, `execution.py`, `tabular.py`, `calibration_evaluation.py`, `prediction_ensemble.py`, `oof_blending.py`, `plan_validation.py`, `reference_methods.py` and `run_state.py`; inspected the blending normalization and regression-averaging integration as well.
- [x] Used benchmark Python `C:\Users\lijiang1\.copilot\session-state\fc6747ca-1d37-46b8-a1c4-fb650ac71116\files\benchmark-env\Scripts\python.exe -B` for the synthetic checks: NumPy 2.4.4, pandas 2.3.3, sklearn 1.7.2, Pydantic 2.13.3. Executed base `TabularEngine` source obtained with read-only `git show` in memory for the two compatibility comparisons.
- [x] Synthetic fitted-rule checks passed own-target perturbation exclusion for IID/grouped/purged-temporal splits, duplicate-index positional behavior, JSON inference-state round trips, and training-safe output through both public wrapper paths with provider access forbidden.
- [x] Synthetic checks passed q0/q1 signed-zero/subnormal bit preservation, categorical/boolean passthrough, clipping pickle/refit behavior, complete pair enumeration for widths 1–17, and stored-code versus engine execution parity for 26 numeric/categorical/frequency candidates.
- [x] Tiny classifier checks passed all three modes with `engineered=False`, original-label NumPy-scalar class weights, active typed target encoding, probability/class alignment, clone isolation, pickle parity and `set_params` fitted-state invalidation.
- [x] Offline validation admitted the 14-entry `frequency_development_replay_20260924.json` without acquisition. An entirely memory-backed archive check passed indexed reconstruction/checksum, idempotence, immutable old-generation retention and interruption before authoritative-index replacement.
- [x] Independently reproduced all three findings below. The stage-tagged regression example failed while its otherwise identical untagged control succeeded.

## Limits

- This is a focused detailed-correctness review, not a claim that every substantive PR file or every historical audit script was exhaustively reviewed.
- No agents, source edits, staging, commits, pushes, downloads, installs, provider requests or real-data benchmark runs were performed. Synthetic models used only small in-memory data. No full test suite was rerun; the parent's test/mypy/wheel results were not treated as independent correctness proof.
- Frozen result/archive bodies were not inspected. Archive publication was exercised with memory-backed paths, not a Windows crash/locking test. Live SDK/CLI lifecycle, actual authentication and the modern SDK branch were not exercised.
- Reviewed source SHA-256: `featcopilot\engines\tabular.py` = `7fbac7b2dbcc072e3a9134afc78375f8054b9320df7fbdbbe964f105246b4090`; `benchmarks\regression_averaging.py` = `89ee9191729c97035aaabe7874bf5f63a4a6bc81dcb67a3d3414f778aa2895c9`; `featcopilot\llm\_plan_transport.py` = `3e24122c8895fb0563bfc81feb1c39bb01720f3132118580cddc997c18de4895`.

## Issues

### Issue 1: New cross-fitting path fails on valid pandas categorical predictors
- **Severity**: Medium
- **File**: `featcopilot\engines\tabular.py`
- **Line(s)**: 318–322, specifically 321
- **Description**: Mapping a categorical Series to distinct fold means can retain `CategoricalDtype`. Calling `fillna(fold_global_mean)` then rejects the fold prior when it is not one of the mapped categories, even if that particular validation slice has no missing values. The new training cross-fit path therefore raises for supported categorical inputs instead of producing numeric encodings.
- **Risk**: Valid `TabularEngine.fit_transform` and default `AutoFeatureEngineer.fit_transform` calls fail. This is not the intended difference between OOF training values and full-training inference maps.
- **Reproduction**:

  ```python
  import numpy as np
  import pandas as pd
  from featcopilot.engines.tabular import TabularEngine

  X = pd.DataFrame({"category": pd.Categorical(["a"] * 10 + ["b"] * 10)})
  y = np.tile(np.arange(10, dtype=float), 2)
  TabularEngine(polynomial_degree=1, include_transforms=[]).fit_transform(X, y)
  ```

  Actual worktree result: `TypeError: Cannot setitem on a Categorical with a new category (4.8125), set the categories first`. The same example executed with the base class succeeds and returns twenty `4.5` encodings. The equal full-training category means deliberately keep that base case successful; the regression is triggered by the newly introduced fold-local mapping. Converting this same predictor to object dtype succeeds in the worktree. The default `AutoFeatureEngineer` wrapper reproduces the same exception.
- **Suggested Fix**: Convert the mapped fold means to a real numeric Series before filling the fold prior, rather than retaining a categorical container. Add a categorical-dtype regression using this equal-full-means/unequal-fold-means case, including the default wrapper.

### Issue 2: Positional target conversion discards numeric categorical-label semantics
- **Severity**: Medium
- **File**: `featcopilot\engines\tabular.py`
- **Line(s)**: 257–264
- **Description**: `_encode_target` first applies `np.asarray(y)` and then constructs a new Series from that ndarray. A categorical target Series with numeric categories loses `CategoricalDtype` before the subsequent categorical check. Consequently the check no longer recognizes these as categorical labels, no `LabelEncoder` is fitted, and arbitrary numeric label values are averaged as regression measurements.
- **Risk**: Existing supported categorical classification targets silently change the learned feature values and label interpretation. In multiclass cases, arbitrary numeric class spacing can change the meaning of the scalar encoding, not merely its scale. Fixing positional alignment does not require changing target semantics.
- **Reproduction**:

  ```python
  X = pd.DataFrame({
      "category": pd.Series(["a"] * 10 + ["b"] * 10, dtype=object)
  })
  y = pd.Series(pd.Categorical(
      [10] * 6 + [20] * 4 + [10] * 4 + [20] * 6
  ))
  engine = TabularEngine(polynomial_degree=1, include_transforms=[]).fit(X, y)
  # Inspect engine._target_encode_maps and engine._target_encode_global_mean,
  # or transform(X)["category_target_encoded"] at positions 0 and 10.
  ```

  Independently executed base result: maps `{"a": 0.4, "b": 0.6}`, prior `0.5`, label vocabulary `["10", "20"]`. Worktree result: maps `{"a": 14.0, "b": 16.0}`, prior `15.0`, label encoder `None`. This compares fitted inference state, not the deliberately changed OOF training output.
- **Suggested Fix**: Preserve or inspect the original target's pandas categorical dtype before converting/aliging its values positionally. For a Series, its extension array can preserve dtype without reindexing by labels. Test numeric categorical targets, inference-state compatibility and mismatched target indexes together.

### Issue 3: Optional stage-tagged regression averaging always fails its component identity check
- **Severity**: Low
- **File**: `benchmarks\regression_averaging.py`
- **Line(s)**: 124–125 and 262–272; interacting result assembly at `benchmarks\tabular.py:859`
- **Description**: The averaging wrapper explicitly preserves an optional payload `stage` in its expected identity and forwards it to each canonical component. However, `tabular.run_case` omits that field from its returned row. `_component` therefore rejects every otherwise valid stage-tagged run as an identity mismatch. `tests\test_regression_averaging.py:401–406` asserts this optional contract only through a mocked canonical runner that supplies the field, hiding the real integration failure.
- **Risk**: An optional experiment/provenance discriminator turns an otherwise successful synthetic or tagged study cell into a failure. Untagged execution is unaffected.
- **Reproduction**: With the real canonical functions, construct:

  ```python
  from pathlib import Path
  from benchmarks import tabular
  from benchmarks.execution import split_fingerprint

  X = pd.DataFrame({
      "a": np.linspace(-2.0, 2.0, 80),
      "b": np.sin(np.arange(80)),
  })
  y = pd.Series(3.0 * X.a + X.b, name="target")
  train, test = np.arange(60), np.arange(60, 80)
  payload = {
      "dataset": "synthetic-review", "family_id": "synthetic-review",
      "source": "synthetic", "role": "diagnostic", "task": "regression",
      "method": "featcopilot-regression-average-robust",
      "seed": 42, "fold": 0, "stage": "locked-replication",
      "split_sha256": split_fingerprint(train, test),
      "train_idx": train, "test_idx": test, "X": X, "y": y,
      "config": {
          "learners": ["linear"], "seed": 42, "threads": 1,
          "max_added_features": 1, "max_candidates": 8,
          "stability_repeats": 1, "numeric_clip_quantiles": (0.0, 1.0),
      },
  }
  tabular.run_case(payload, Path.cwd())
  ```

  Actual: `ValueError: raw component stage identity differs from the requested cell`. Removing only `stage` succeeds with loss `1.826174278752301`. The reproduction uses 80 synthetic rows and writes no model/workspace artifacts. This averaging module and its inconsistent stage contract are new relative to the base.
- **Suggested Fix**: Propagate the optional stage discriminator through canonical result assembly, retaining the identity check. Add a small real-component integration test rather than relying solely on the mock that already echoes `stage`.

## Resolution Log
_Updated by the driving agent as findings are addressed._

### Issue 1
- **Status**: Open
- **What changed**: No source changes; pending driver resolution.
- **Why**: Review-only assignment.
- **How verified**: Benchmark-Python synthetic reproduction, base-class control, object-dtype control and default-wrapper reproduction.

### Issue 2
- **Status**: Open
- **What changed**: No source changes; pending driver resolution.
- **Why**: Review-only assignment.
- **How verified**: Same categorical target compared against base and worktree fitted maps, priors and label-encoder state.

### Issue 3
- **Status**: Open
- **What changed**: No source changes; pending driver resolution.
- **Why**: Review-only assignment.
- **How verified**: Real in-memory canonical regression components; adding/removing only the optional stage field changes failure to success.

### Driver resolution update

- **Issues 1-2 - Fixed after the stage cap**: Mapped target means are explicitly
  converted to numeric values before prior filling in both training and inference.
  Positional target alignment now preserves a Series' extension dtype, including
  numeric categorical labels, rather than discarding that contract via NumPy.
- **Issue 3 - Fixed after the stage cap**: Canonical success and failure rows
  preserve the optional stage discriminator. Real raw/FE integration tests now
  exercise both tagged and untagged runs for linear and LightGBM learners.
- **How verified**: New categorical predictor/wrapper, numeric categorical target
  and real-component stage tests reproduced the failures before repair.
  The combined engine/wrapper/task/regression-average/canonical suite passed
  555 tests with two optional skips; package mypy passed 52 files and the wheel built.
