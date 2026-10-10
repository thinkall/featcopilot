# AutoFeat safe-scale: retrospective Thoracic diagnostic

**The separately named compatibility reference completed all three folds.**
This resolves the observed NaN-input failure for this run, not the original
confirmation decision. The original native run remains **14/15, BLOCKED**,
with its failed fold and all evidence unchanged.

The compatibility reference's mean log loss is **0.503197**, versus the
archived FeatCopilot fixed average's **0.508215**. Lower is better: the
candidate does not outperform this reference on the retrospective
comparison. This is neither fresh confirmation nor evidence of SOTA.

## Fixed scope

This run used the exact prepared Thoracic Surgery payload and all three
registered IID splits, seed 2031, LightGBM, eight threads, matched final
train-min/max clipping, 900 seconds per job and 4,096 MiB RSS limits.
There were three physical jobs, three scored outcomes and no retries.
No raw, FeatCopilot or OpenFE models were refitted; their values below
come from the immutable original experiment.

The method is **`autofeat-native-safe-scale-v1-robust`**, not
`autofeat-native-robust`. The pinned AutoFeat 2.1.1 compatibility path
uses denominator one only for wholly finite, actually constant feature
or shuffled-noise columns whose upstream standard deviation is exactly
zero. It does not repair nonconstant underflow or nonfinite inputs,
change target scaling, drop columns, change the declared search policy,
or impose FeatCopilot's selected-addition cap. Upstream MIT notices,
dependency pins and source hashes are retained.

The qualified **`standard_iid_unverified_entity`** tier is unchanged.
These row-level results imply neither patient/entity independence nor
clinical/prospective generalization. They must not be pooled with
verified-entity evidence. Three folds of one exposed source are not
three independent datasets.

## All-fold losses

| Method | Scored folds | Mean log loss | Evidence |
|---|---:|---:|---|
| Raw robust LightGBM | 3/3 | 0.510180 | Original confirmation |
| FeatCopilot single FE | 3/3 | 0.520143 | Original confirmation |
| FeatCopilot fixed 50:50 average | 3/3 | 0.508215 | Original confirmation |
| OpenFE-inductive-native-robust | 3/3 | 0.510180 | Original confirmation |
| AutoFeat-native-robust | 2/3 | Undefined | Original confirmation |
| AutoFeat-native-safe-scale-v1-robust | 3/3 | 0.503197 | Retrospective diagnostic |

No native successful-fold average is used, and no repaired score is
spliced into the original native result.

## Fold-level compatibility outcomes

| Original fold | Native log loss | Safe-scale log loss | Native output width | Safe-scale output width |
|---|---:|---:|---:|---:|
| 0 | 0.485965 | 0.516047 | 66 | 59 |
| 1 | 0.466254 | 0.507780 | 63 | 49 |
| 2 | Error: NaN input | 0.485765 | Unavailable | 47 |

Widths include the reference's encoded originals and generated features;
they are not the number of additions. All three safe-scale fits select
the registered `num_leaves=7` setting.

The previously successful native folds do **not** reproduce exactly:
both their scores and output widths differ. The archived outcomes alone
do not identify how much of that difference comes from the guard versus
other execution variability. No output-parity or causal accuracy-gain
claim follows from the narrow implementation change.

### Read-only follow-up audit

Full per-cell diagnostics match on seed 2031, dependency versions,
thread count, input mappings, medians, categorical preprocessing,
native width policy, transformations and selection settings. No
unintended seed, default-parameter or dispatch mismatch was found.
The feature-selection outputs themselves differ before learner fitting:
fold 0's inner/outer additions change from 45/50 to 41/43, and fold 1's
from 40/47 to 33/33. The latter compatibility fit retains the encoded
originals but no extra symbolic features; categorical columns were not
discarded.

Pinned upstream `autofeat/featsel.py:135` applies a seeded permutation
to `list(set(df.columns).difference(initial_cols))`; line 170 also turns
a set into a list. These operations depend on Python string hash order,
which NumPy seeding does not control. The private compatibility copy
preserves them. Neither historical launcher explicitly pinned or
recorded `PYTHONHASHSEED`. Its inherited historical value is unknown.
This identifies an existing source of run variability, **not proof
that it caused these particular score differences**. Guard-activation
counts and intermediate selection matrices were not recorded, so a
guard-only causal comparison remains unavailable.

After this diagnostic, the shared benchmark launcher was changed to
pin Python's hash seed before worker startup and record it on outcomes
for future runs. That policy does not apply retroactively to this
source snapshot or the original confirmation; neither was rerun.

## Integrity, archives and resources

Verification checked all three unique identities, exact reconstructed
splits, prepared-data checksum, tier and claim boundaries, all six
inner/outer compatibility records, installed dependency/source pins,
source snapshot contents, original artifacts and ordered archive
reconstruction. The run's source fingerprint remained:
`489c114e8b36609a602bd77ecd54e86b99ed53d23f75700cfa81b5efcb82a36b`.

The whole-process stopwatch recorded **448.7185198 seconds
(0.124644 hours)**, including interpreter startup, imports, workers and
initial archives. Publication verification is separate. Peak recorded
worker-tree RSS was 2,505.86 MiB. Current-cycle measurement use is
**61.05288 of 72 hours**, leaving **10.94712 hours**. No LLM calls were
made; the generation ledger remains 96/100.

- [protocol.json](protocol.json): final protocol, source pins and elapsed time.
- [verification.json](verification.json): exact comparisons and integrity checks.
- [archive.json](archive.json): ordered, losslessly compressed raw results.
- [auxiliary_archive.json](auxiliary_archive.json): bounded source ZIP parts.
- [launcher_timing.json](launcher_timing.json): whole-process timing.
- [Original report](../2026-10-04-thoracic-fixed-average-confirmation/REPORT.md):
  unchanged authoritative blocked decision.

Defaults remain unchanged. Further independent evidence must compare
against an explicitly registered reference identity rather than
retroactively replacing failed native outcomes.
