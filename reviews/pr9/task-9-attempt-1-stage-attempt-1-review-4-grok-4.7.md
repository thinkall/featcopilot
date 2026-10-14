## Review Summary
- **Round**: 4
- **Theme**: Detailed correctness
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\code\featcopilot\reviews\pr9\task-9-attempt-1-stage-attempt-1-review-4-grok-4.7.md
- **Issues Found**: 0
- **Verdict**: CLEAN

## Evidence Checklist
- [x] Scope index `review-pr9/scope.json`: 721 paths, 127 substantive, 594 frozen/generated excluded from body review. Base `bf119812162af87c29bc8d09bb3c705576b94f27`, HEAD `62cbb1cf4f434a6bbfd956b34c2615e761441e46`, plus uncommitted worktree including `featcopilot/llm/_plan_transport.py`.
- [x] Positional/group/time splits in `featcopilot/utils/splitting.py`: temporal holdout with `label_end == cutoff` purged row 5 and kept a strictly earlier prefix (`train=[0,1,2,3,4]`, `valid=[6,7]` on 4 blocks). Group codes are positional via `np.asarray`, not pandas index labels. `make_holdout_splits` window end is `first + n_splits * width == len(blocks)`.
- [x] Target cross-fit in `featcopilot/engines/fitted_rules.py` `_cross_fit` / `_target_splits`: non-monotonic index preserved; no OOF row equaled its own target on a one-category regression probe. IID splits covered every row (`prior_row_positions=[]`).
- [x] Frequency/category types in `featcopilot/utils/frequency.py`: `True`/`1`/`"1"` keys differ; `1`, `1.0`, and `np.int64(1)` share a number key; missing/nonfinite map to 0 and keep the input index.
- [x] Task encoding in `featcopilot/utils/task.py`: integer-valued float auto-routing is classification, matching the docstring; explicit classification `prepare_target` returns contiguous codes.
- [x] Class-weight and row alignment in `featcopilot/utils/model_preprocessing.py`, `featcopilot/classification.py`, and `featcopilot/engines/predictive.py`: `encode_class_weights` maps original labels, including `np.str_`/`int64` classes, onto `LabelEncoder` indices. A second encode inside the predictive learner is an identity on already-encoded `0..k-1` labels. `make_classification_pipeline` target encoding fitted and predicted with a numeric-only block and a mixed frame (`(40, 2)`).
- [x] Catalog/OpenML index contract in `benchmarks/catalog.py`: official row ids are finite integral positions in `[0, n)`; `OpenMLSplit.get` returns the same stored pairs. Nonzero `max_rows` is rejected unless policy is IID, so official indices are not sampled out from under the frames.
- [x] Worker/archive normalization: synthetic `archive_results` (`part_bytes=30`, 1000-byte payload) restored by concatenating `archive.json` parts in listed order. `_validated_worker_row` rejects boolean `loss` and accepts a finite success row. Official fold precedence over `--folds` is documented in `benchmarks/README.md` / `TABULAR_PROTOCOL.md`, not a silent override.
- [x] Provider text contracts: OpenAI/LiteLLM empty or non-string `content` raises and no longer falls back to mocks (`featcopilot/llm/openai_client.py`, `litellm_client.py`). `plan_message` rejects empty text and tool requests. Isolated transport cleanup cannot replace a primary error; workspace `OSError` is logged without replacing the outcome.

## Limitations
- Generated/frozen evidence bodies were not reviewed; producers and a synthetic archive round-trip were.
- No full test suite, dataset download, package install, or live provider call.
- Not every one of the 127 substantive files was read line-by-line. Review concentrated on split, target, catalog, classifier, worker, and provider dataflow, with spot checks of OOF/ensemble/regression-average alignment.
- `PointInTimeRelationalEngine._aggregate` `last` skips nonfinite values and therefore matches `DataFrame.groupby().last()` (`skipna=True`). That is undocumented but not contradicted by `tests/test_point_in_time.py`, so it is not filed as a defect.
- Group factorize follows Python equality (`True == 1`). That matches the hashable-label contract and sklearn grouping; it is not treated as a new splitter defect.
