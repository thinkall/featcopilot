# Locked confirmation: first-attempt admission status

**Completion update:** the [separate two-family repair](../2026-09-21-confirmation-completion/ANALYSIS.md)
subsequently completed all 480 affected cells. The combined original six-family
outcome is available there, with explicit lineage. The failures and first-attempt
completion figures below remain unchanged.

The first attempt accounted for all **1,440 preregistered cells** in approximately
10.65 active hours. There were no source changes during execution.

Four families prepared successfully and produced 960 scored cells. Two families
failed schema admission before any of their models ran:

| Family / task dataset | Failure | Unscored cells |
|---|---|---:|
| COIL insurance / OpenML 46916 | `avgAge`: nonnumeric predictor not explicitly declared categorical | 240 |
| In-vehicle coupon / OpenML 46937 | `coupon`: nonnumeric predictor not explicitly declared categorical | 240 |

These failures remain in the original report and denominator. **This is not a
complete six-family confirmation result**, and it does not meet the proposed
95% completion gate.

A metadata-only repair audit is underway for exactly these two families.
No confirmation score values were inspected to choose the repair. Dataset
versions, targets, feature membership, official splits, methods, learner search
and seeds must remain unchanged. Any correction must have primary schema
evidence, create a separate repair record/manifest, and retain this failed
attempt rather than overwrite it or replace the families.

The source archive and all requested-cell identities were verified, and the
lossless result parts reconstruct the local results exactly. See
[manifest.json](manifest.json), [source.zip](source.zip) and
[archive.json](archive.json). Overall comparative conclusions are deferred
until the admission discrepancy is resolved or explicitly left incomplete.
