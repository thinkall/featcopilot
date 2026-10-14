## Review Summary
- **Round**: 4
- **Theme**: Detailed correctness
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\code\featcopilot\reviews\pr9\task-9-attempt-1-stage-attempt-2-review-4-grok-4.7.md
- **Issues Found**: 1
- **Verdict**: ISSUES_FOUND

## Evidence Checklist
- [x] Inspected current worktree against base `bf119812162af87c29bc8d09bb3c705576b94f27` (HEAD `62cbb1cf4f434a6bbfd956b34c2615e761441e46` plus uncommitted source), using `git diff` and full reads of `featcopilot/engines/tabular.py`, `featcopilot/engines/fitted_rules.py`, `featcopilot/engines/predictive.py`, `featcopilot/classification.py`, `featcopilot/utils/model_preprocessing.py`, `featcopilot/utils/numeric_clipping.py`, `featcopilot/utils/splitting.py`, `featcopilot/llm/_plan_transport.py`, `featcopilot/llm/copilot_client.py`, `featcopilot/llm/openai_client.py`, `featcopilot/llm/litellm_client.py`, `featcopilot/transformers/sklearn_compat.py`, `benchmarks/catalog.py`, `benchmarks/execution.py`, `benchmarks/run_state.py`, `benchmarks/calibration_evaluation.py`, `benchmarks/blending_evaluation.py`, `benchmarks/prediction_ensemble.py`, `benchmarks/llm_evaluation.py`, and `benchmarks/tabular.py`.
- [x] Compared the new tabular target-encoding helper with base via `git diff bf119812162af87c29bc8d09bb3c705576b94f27 -- featcopilot/engines/tabular.py`. The cross-fit path and `n_splits < 2` full-map fallback are introduced in this change.
- [x] Synthetic check: `TabularEngine` with `min_samples_per_category=1` and `target_encode_cv=5` on one row `cat=a`, `y=1` returns `cat_target_encoded=1.0`, equal to that row's own target. Two- and three-row cases excluded the encoded row (`[1.0, 0.0]` for `y=[0, 1]`; `[1.0, 0.5, 0.5]` for `y=[0, 1, 1]`). Duplicate index `[5, 5, 7, 7]` stayed positionally aligned.
- [x] Synthetic check: sklearn 1.8 `train_test_split(..., test_size=0.15, stratify=y)` sizes matched `ceil(0.15 * n)` for n=8..79, consistent with `make_subsample85_splits`. `PredictiveTabularEngine._balanced_pairs` covered every unordered pair exactly once for 0..8 columns. `encode_class_weights` mapped original string and integer keys onto `LabelEncoder` positions.
- [x] Pandas 3.0.2 probe: `is_string_dtype(series)` is False for object-integer columns while `is_string_dtype(dtype)` is True. The pending `TabularEngine._encode_target` change follows the series, so object-integer targets stay numeric. Not a regression.
- [ ] Did not run the parent 566-test suite, mypy, wheel build, real dataset benchmarks, provider calls, or package installs. Passing those would not by itself prove the one-row path.

## Issues

### Issue 1: One-row target encoding writes the row's own target into the training matrix
- **Severity**: Low
- **File**: featcopilot/engines/tabular.py
- **Line(s)**: 67, 294-306, 328-336
- **Description**: `_cross_fit_target_encodings` is documented as encoding each training row from folds that exclude that row, and `target_encode_cv` is documented as producing leakage-safe training encodings. `target_encode_cv` is at least 2, so `n_splits = min(self.config.target_encode_cv, len(X))` is below 2 only when the frame has fewer than two rows. That branch returns the full-data category map. With `min_samples_per_category=1`, a one-row fit therefore stores the row's own target (`1.0` for `y=1`) in `cat_target_encoded`. `fit_transform` uses this helper, so the training matrix is not leakage-safe on that input. For two or more rows the KFold path does exclude the encoded row; duplicate indexes stayed positional in a synthetic check. This fallback is new relative to base, which had no cross-fit helper.
- **Risk**: A one-row, or otherwise under-sized, supervised fit can train on a feature that is a copy of `y`. Downstream models can look perfect on that row and the public "leakage-safe" contract is false for a reachable parameter combination (`min_samples_per_category` has no lower bound above 1).
- **Suggested Fix**: Do not apply the full-data map to training rows when a row-excluding fold cannot be built. Reject target-encoded `fit`/`fit_transform` when `len(X) < 2`, or fill those training rows with an explicit prior that does not include the row's target (and test that prior). Keep the full-data map for later `transform` only.

## Resolution Log
_Updated by the driving agent as findings are addressed._

### Issue 1
- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

### Driver resolution update

- **Status**: Fixed after the stage cap as a tightly coupled cross-fit contract correction.
- **Classification**: The own-target singleton output itself also exists at the
  exact PR base, as independently executed by the GPT reviewers. However, the new
  explicit cross-fit training contract makes retaining that fallback inconsistent.
  This revises the earlier deferral without falsely claiming a wholly new behavior.
- **What changed**: When target encoding is activated, fitting requires at least
  two rows; the cross-fit helper cannot fall back to a full-data own-target map.
  Non-target-encoded singleton behavior and ordinary inference maps are unchanged.
- **How verified**: A focused singleton regression failed before the change and
  now raises the documented `ValueError` with the engine unfitted. The combined
  555-test validation, package mypy and wheel build passed.
