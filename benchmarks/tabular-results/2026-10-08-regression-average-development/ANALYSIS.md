# Fixed regression averaging: complete development result

**All 204 cells scored successfully, and all 136 component/control pairs
matched exactly on the registered prediction hashes, feature-name hashes,
parameters, inner losses, widths and metrics.** The source remained fixed.
This is development evidence on exposed datasets, not confirmation.

The useful outcome is learner-specific: fixed averaging improves the
LightGBM comparison, whereas single FE remains the stronger linear
regression method on every declared family mean.

## Seven-family primary panel

Values below are equal-family means of relative RMSE reductions, after
averaging fold losses within each family. Positive values are better.
The matched comparator is the identically capped raw learner.

| Learner | Single FE vs raw | Fixed average vs raw | Fixed average vs single FE | Worst average family harm vs raw |
|---|---:|---:|---:|---:|
| Linear / Ridge | +24.46% | +16.35% | **-31.32%** | 0.00% |
| LightGBM | +2.09% | **+3.36%** | **+1.20%** | 0.00% |

The comparison with single FE is computed directly from paired family
losses, not by subtracting the two raw-relative percentages. Energy's
large relative tradeoff contributes substantially to the linear result:
RMSE is 2.844552 for raw, 0.529265 for single FE and 1.507610 for the
fixed average. Averaging gives up substantial FE signal there.

LightGBM averaging improves all seven primary family means versus raw.
Single FE harms Concrete by 4.12% and Auto MPG by 2.91%; averaging changes
those comparisons to improvements of 1.68% and 0.64%, respectively.
This does not prove universal protection against harm.

## Separate stress cases

These selected stress families are **not included in the primary mean**.
Infrared uses three IID folds; hardware uses all ten official folds.

| Stress family | Learner | Raw RMSE | Single-FE RMSE | Average RMSE | Average reduction vs raw |
|---|---|---:|---:|---:|---:|
| Infrared oral temperature | Linear | 0.262123 | 0.253557 | 0.255405 | +2.56% |
| Hardware performance | Linear | 68.547871 | 42.087304 | 50.650235 | +26.11% |
| Infrared oral temperature | LightGBM | 0.263895 | 0.265001 | 0.263936 | **-0.016%** |
| Hardware performance | LightGBM | 68.794830 | 69.831744 | 66.227722 | +3.73% |

**Fold-level risk remains.** Hardware's LightGBM fold 0 increases RMSE
by 73.94% with single FE and by 8.97% with averaging versus its matched
raw fold. The registered 5% gate concerns family-mean harm, not every
individual fold. Do not turn its passing result into a per-fold or
per-example no-harm claim.

The linear method's worst individual single-FE fold is hardware fold 7
at +1.56% harm. Linear averaging improves every individual fold in this
run, but is worse than single FE on every family mean across all nine
families. Both the accuracy and tail tradeoffs must remain visible.

## Interpretation and next candidate

Both averaging learners satisfy the preregistered necessary follow-up
conditions: complete coverage, exact component/control parity, positive
primary mean gain versus raw, and no family-mean raw harm exceeding 5%
in either declared cohort. Those conditions do not authorize promotion.

The development-derived policy to replicate is **single FE for linear,
fixed averaging for LightGBM**, retaining matched raw controls and the
LightGBM single-FE comparator. It keeps the stronger simpler linear
method rather than applying averaging everywhere. Freeze the policy
before additional seeds; do not optimize mixture weights from these
test results. Further origin-independent and native-reference evidence
is still required before a broad superiority claim.

## Fixed protocol and claim limits

The primary panel contains Concrete Strength, Energy Efficiency,
Combined Cycle Power, Real Estate Sindian, Auto MPG, Wine Quality and
California Housing. All use seed 42, three folds and at most 5,000 rows.
Energy retains its simulation designation. Houses/wine remain
cached-only legacy inputs with their original `legacy-not-reviewed`
rights labels and target representations.

Hardware retains its complete official ten-fold design and all 209 rows.
It is **not** a replay of the earlier three-fold IID hardware failure.
Infrared remains contemporaneous measurement evidence, not clinical,
prospective or entity-disjoint generalization. No origin is fresh.

Both components use their unchanged canonical complete procedure,
including the same three-setting inner search, train-only caps,
learner-aligned FE selection and final refit. The fixed rule averages
predictions, not RMSE scores. Both fits are charged; reported feature
width is the maximum separate plan width, never a feature union.
Within-worker warmup can make component timings differ from separately
started controls, so smaller displayed stage times are not a causal
speedup claim.

## Integrity and resources

Verification reconstructed all three-fold IID splits and all ten
hardware folds from the checksummed cached official split file. Every
planned identity, prepared payload, source ZIP, source-file content,
registered seed, component comparison and ordered archive was checked.
No retries, dropped cells or fallback scores were used.

Source fingerprint:
`f94e23533f6b4e24a64542c6a307abaa09a0439ca084ad3632e07c4024c5a3f7`.

The complete run used **12,026.224867 seconds (3.34062 hours)**, including
startup, both preparations, workers and initial publication archives.
It stayed below the shared four-hour ceiling. Current-cycle use is
**64.39350 of 72 hours**, leaving **7.60650 hours**. No LLM calls were
made; the logical-attempt count remains 96/100.

- [analysis.json](analysis.json): absolute family losses, paired gains,
  fold tails, parity checks and accounting.
- [analyze_study.py](analyze_study.py): reproducible verification/analysis.
- [protocol.json](protocol.json): unchanged pre-result declaration.
- [controller.json](controller.json) and [launcher_timing.json](launcher_timing.json):
  source continuity, stage ceilings and measured time.
- [archive.json](archive.json) and [auxiliary_archive.json](auxiliary_archive.json):
  exact results and bounded source/manifest reconstruction.

Production defaults and every prior confirmation decision remain unchanged.
