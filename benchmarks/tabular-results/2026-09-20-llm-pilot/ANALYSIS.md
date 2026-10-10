# Budgeted typed-LLM development pilot

**Typed LLM proposals helped the linear Concrete model substantially, but this
two-dataset development pilot does not establish a general LLM advantage or SOTA.**
The requested model was `gpt-5.6-sol`. No model substitution was used.

## Protocol and accounting

Two reviewed development families, three folds and seeds 42/137 were evaluated
with linear and LightGBM learners. Raw inputs, deterministic predictive FE,
typed LLM plans and a hybrid were compared using training-only proposals,
training-only learned statistics and inner model tuning. Original column
meanings were supplied through descriptions; test rows/labels were not sent
to the provider.

All 96 requested learner cells were accounted for in **1.27 active hours**.
Raw, deterministic and typed-plan methods completed 24/24 cells each. Hybrid
completed 22/24; one plan-validation failure and one generated-name collision
remain unscored. No fallback scores were invented.

The shared allowance records **96 logical attempts total**: 94 in the pilot,
plus two readiness attempts (one authentication-configuration failure and one
successful synthetic request). Four remain unused. The counter includes
failed attempts and persists across processes and resumes.

The user explicitly relaxed the hard 4,096-token ceiling. It was a best-effort
request, with `token_cap_enforced=False`, not an enforced limit. The 92 completed
response diagnostics in result rows reported 517,358 input tokens and 45,289
output tokens through SDK usage events, with a maximum observed completion of
744 tokens. Those observations exclude unreported failed-response usage and
are not a complete provider billing guarantee.

## Complete-pair results

Mean primary-loss reductions versus the matched raw learner are:

| Dataset / learner | Deterministic predictive | Typed LLM | Hybrid |
|---|---|---|---|
| Online Shoppers / linear | +2.21% | +2.77% | Incomplete: 5/6 cells |
| Online Shoppers / LightGBM | +0.00% | -0.03% | Incomplete: 5/6 cells |
| Concrete / linear | +10.75% | +34.55% | +33.67% |
| Concrete / LightGBM | -0.46% | -0.68% | -1.31% |

Classification uses log loss; regression uses RMSE. Do not pool these two
families into a universal score or compare complete-pair averages against
partial hybrid folds. The gain depends strongly on the downstream learner.

These methods use the nine-operator typed plan vocabulary, not unrestricted
LLM-generated Python. Hybrid representation construction can fail when its
two generators choose colliding names; that limitation is recorded rather
than silently overwriting columns. Invalid model output remains a measured
failure. No failed cells were selectively replaced.

## Reproducibility and next steps

The original source and result archives were verified. See [REPORT.md](REPORT.md),
[manifest.json](manifest.json), [source.zip](source.zip), and
[archive.json](archive.json) for the authoritative compressed result parts.
Result records include proposal definitions, diagnostics and contextual hashes,
not raw prompts or learned category maps.

Before broader claims, expand the semantic panel, measure proposal variability,
compare against same-model/call-budget FE references, and make hybrid naming
composition robust without hiding proposal failures. Any retry or follow-up
must retain the existing attempt history and remain within the shared allowance.
