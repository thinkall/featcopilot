# Discovery-independent fixed-plan gate

**The gate executes correctly, but does not improve this development
comparison.** It underperforms the existing single-fit robust aligned
pipeline in every observational task/learner mean and worsens classification
tail harm. Do not promote it, tighten its threshold after seeing these
results, or treat accepted gates as external performance guarantees.

All **384 cells succeeded in 5.63226 active hours** on the original sixteen
exposed development families. All **192 existing control losses reproduce
the preceding run exactly**. Split identities, dependency versions,
three-setting learner grids, candidate limits and resource policies
remained frozen.

## Protocol, execution and identity checks

Each fit reserves 25% of its allowed training data as an independent gate.
All raw and engineered hyperparameter search, preprocessing and discovery
use the other 75%. The winning discovery plan is refitted on discovery
rows before the gate and then evaluated once against its separately
discovery-tuned raw control.

The fixed criterion is strictly more than 1% additive-loss reduction:
classification log loss or regression MSE, with positive raw loss.
Paired standard error is diagnostic only. Accepted formulas, operands,
constants, order and schema remain frozen; only unsupervised frequency
state and downstream preprocessing/model coefficients are refitted on
all allowed training rows. Valid rejection chooses the discovery-matched
raw branch. Errors are not converted to raw successes.

Across 96 feature-method cells:

- **50** accept a plan.
- **35** produce no added-feature proposal.
- **11** reject for insufficient gate improvement.

All gate decisions, discovery split hashes and selected raw parameters
were checked. Every rejected/no-feature cell exactly matches its
`baseline-discovery-robust` loss, learner parameters and output width.
Every accepted cell meets the fixed gate criterion and preserves plan
identity and ordered names across state refitting. There is no evidence
here of a gate implementation or identity-freezing failure.

## Both raw comparisons matter

Positive percentages mean lower primary loss. Classification covers nine
observational families; regression six.

| Task / learner | Incumbent vs ordinary robust raw | Plan gate vs ordinary robust raw | Plan gate vs discovery-matched raw |
|---|---|---|---|
| Classification / linear | +16.12% | +11.38% | +10.57% |
| Classification / LightGBM | +8.12% | +3.20% | +3.64% |
| Regression / linear | +14.97% | +13.09% | +13.18% |
| Regression / LightGBM | +1.39% | +1.27% | +2.18% |

The last column isolates the proposal's value relative to its raw rejection
branch. It must not be substituted for comparison against the ordinary
raw pipeline, whose model tuning uses more of the outer-training data.
For example, German Credit/LightGBM is unchanged versus its discovery
control but 6.01% worse than the ordinary robust raw learner.

Matched-gate medians are +1.07%, 0.00%, +9.06% and +1.29% in the same
row order. The classification means have wide descriptive family-bootstrap
intervals; the gate's internal gain is not fresh confirmatory evidence.

Energy Efficiency remains separate simulation evidence. Linear gain
versus discovery raw is +81.23%, while LightGBM worsens **7.12%**;
neither is included in the observational means.

## Classification failures remain despite accepted gates

Wisconsin Diagnostic worsens **42.35% for linear** and **46.66% for
LightGBM** versus the discovery-matched raw controls. Against ordinary
robust raw, the increases are 34.23% and 44.73%.

On Wisconsin fold 1, the LightGBM gate saw primary loss fall from
**0.106101 to 0.022999**. The accepted, identity-frozen procedure then
scored **0.343719** on the untouched outer test, versus discovery raw
**0.208548**. The linear gate saw **0.035706 to 0.032947**, followed by
outer-test **0.236868** versus discovery raw **0.131588**.

Freezing formulas removes one procedural mismatch, but the final model
and preprocessing still change when refitted, and small validation samples
can give optimistic comparisons. These records do not establish which
mechanism caused each error, nor justify changing the gate margin based
on these folds. A gate is a heuristic—not a no-harm certificate.

## Cost

Mean pipeline seconds per family/cell:

| Task / learner | Incumbent | Independent gate |
|---|---|---|
| Classification / linear | 28.22 | 20.92 |
| Classification / LightGBM | 61.29 | 49.37 |
| Regression / linear | 10.13 | 6.97 |
| Regression / LightGBM | 26.12 | 21.78 |

Discovery uses fewer rows, explaining why added gate work need not increase
runtime. This is not a free accuracy improvement: the smaller discovery
sample also changes the information available to feature/model selection.

## Evidence and next boundary

All 384 identities are unique and scored. Start/end/current source hashes
matched before further work:
`46d7e04ecb077c2481045dd7d4b6816a4f8ab97b79c6c44b673ccac1ef2130c5`.
Exact source/result checksums, bounded archive reconstruction, all gate
invariants and unchanged controls were verified.

- [REPORT.md](REPORT.md): ordinary and discovery-matched summaries.
- [archive.json](archive.json): exact ordered result parts.
- [auxiliary_archive.json](auxiliary_archive.json): bounded manifest/source.
- [decision.json](decision.json): unsuccessful promotion and gate counts.

Keep both five-bag and fixed-plan variants experimental. The existing
single-fit, robust aligned configuration remains the declared incumbent
for a separately frozen comparison on the four reserved families; it is
not promoted to the library default. Those families include only one
regression origin, so even a favorable result cannot establish broad SOTA.

The new cycle has used **11.99674 of 72 measurement hours**, leaving
**60.00326 hours**. No live LLM calls were made.
