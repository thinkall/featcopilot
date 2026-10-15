# Two-source qualified standard-IID policy confirmation

**The predeclared overall decision is FAIL**, despite complete coverage
and no greater-than-5% raw-baseline family harms. The fixed LightGBM
FE85 policy passes its scoped comparisons, but linear single FE falls
behind native AutoFeat by 1.28% on the two-family mean.
Do not change that criterion or choose a different linear method afterward.

All **96 prediction outcomes succeeded** from **72 physical jobs**, using
**1.74079 cumulative active hours**. There were no retries, timeouts,
substitutions or incomplete/undefined decision comparisons.

## Scope must stay qualified

This is the explicitly user-approved
**`standard_iid_unverified_entity`** tier: row-level public IID benchmark
evidence with unresolved patient/entity mapping. It does **not** establish
patient/entity-independent, clinical, prospective or general-SOTA
performance and must not be pooled with verified-entity cohorts.
Known group/time constraints remain binding.

Mammographic Mass retains Age, Shape, Margin and Density with BI-RADS
excluded; Haberman retains its original three predictors and five-year
endpoint. Sources, primary CC-BY-4.0 rights, exclusions, full rows,
three IID folds and seed 2030 were locked before acquisition.

The fixed policy is single robust FE for linear and plain FE85 averaging
for LightGBM. Sigmoid results are diagnostic only. Native OpenFE/AutoFeat
references are the declared native-width, single-model adaptations with
matched final numeric caps—not ensemble or paper-best reproductions.

## Locked policy comparisons

Positive mean gain means lower log loss, first averaged over each family's
three folds, then over both families.

| Policy | Versus raw single | Versus raw85 | Versus native OpenFE | Versus native AutoFeat |
|---|---|---|---|---|
| Linear single FE | +0.41% | +0.11% | +0.04% | **-1.28%** |
| LightGBM plain FE85 | +2.25% | +0.22% | +7.33% | +7.51% |

LightGBM's architecture-level gain over raw single is mostly shared with
raw85 itself. Its incremental FE gain over matched raw85 is only 0.22%.
This distinction prevents crediting prediction averaging alone as a
feature-engineering breakthrough.

The one failed locked comparison is **linear versus AutoFeat**.
All raw-comparator mean/harm criteria and all LightGBM comparisons pass
within this narrow tier. The authoritative `confirmation_decision` in
[results](archive.json) retains both families, all four comparators,
absolute losses, complete-fold coverage and the unchanged decision rules.

## Family-level absolute losses

| Learner / family | Fixed-policy loss | Raw single | Raw85 | OpenFE | AutoFeat |
|---|---|---|---|---|---|
| Linear / Haberman | 0.539784 | 0.544439 | 0.545396 | 0.544439 | 0.529202 |
| Linear / Mammographic | 0.458232 | 0.458107 | 0.454518 | 0.454721 | 0.455694 |
| LightGBM / Haberman | 0.605216 | 0.647250 | 0.605112 | 0.653349 | 0.707254 |
| LightGBM / Mammographic | 0.485819 | 0.476306 | 0.488045 | 0.524055 | 0.488751 |

LightGBM FE85 improves Haberman by 6.49% versus raw single but is nearly
identical to matched raw85. Mammographic is 2.00% worse than raw single
and 0.46% better than raw85. These are small, task-dependent results;
two datasets do not supply broad generalization evidence.

Sigmoid happens to improve the LightGBM diagnostics on these two tasks,
unlike its prior development average. It was not the registered policy
and is not substituted into the decision after seeing the outcomes.

## Integrity and resources

All 96 identities are unique, all 72 physical records are present, and
every outcome retains the restricted evaluation tier. Start/end/current
source hashes matched:
`d8952d7715d9d5aaf786708822f32b4f554e9dca21c7b2d7f90dfe0bce02d826`.
Exact result/source checksums and bounded archive reconstruction passed.
Physical shared training was counted once.

- [REPORT.md](REPORT.md): authoritative locked decision and all diagnostics.
- [archive.json](archive.json): ordered lossless result parts.
- [auxiliary_archive.json](auxiliary_archive.json): bounded source/manifest.
- [Locked suite](../../suites/standard_iid_policy_confirmation_20261003.json)
  and [release audit](../../suites/standard_iid_policy_confirmation_release_20261003.json).

Both origins are now exposed within this tier. No library default is
promoted and no SOTA claim is established. The current 72-hour cycle has
used **48.36633 hours**, leaving **23.63367**. No additional LLM calls occurred.
