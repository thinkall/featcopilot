# Public classifier extraction: real-data parity

**All 162 prediction outcomes reproduce the pre-extraction results exactly.**
Moving the complete learning procedure into the production package did
not change the measured classifier behavior on this replay. This is
migration evidence, not a new accuracy improvement or SOTA result.

The replay covered nine canonical development classification families,
LightGBM, seed 42, three outer folds and the six existing single/plain/
sigmoid raw/FE arms. All **108 physical jobs** and **162 outcomes**
succeeded in **3.01883 cumulative active hours**.

## Exact comparisons

Against seed 42 in the archived `2026-10-03-subsample85-development` run:

- All log-loss, Brier-sum, accuracy, balanced-accuracy, macro-F1, ROC-AUC
  and average-precision values match exactly.
- Outer split identities, input widths and output widths match.
- Single-model parameters and inner losses match.
- Every ensemble member's selected parameters, feature order, width,
  inner split, base/calibration row identities and trial losses match.

There are **324 member comparisons including both shared prediction
views**, not 324 independently fitted models. Shared plain/calibrated
base identities remain internally consistent. Timing, generated identity
tokens and pickle class locations are not mistaken for semantic
prediction equality. See [parity.json](parity.json).

## Usable production artifact

`featcopilot.classification.FeatureEnsembleClassifier` now exposes the
plain procedure without importing benchmark modules or constructing
calibrators. It accepts a supplied bare classifier, three native
hyperparameter dictionaries and an explicit `single`/`subsample85` mode.
Each retained member owns its feature plan and preprocessing; there is
no fictitious common transformed feature matrix.

The benchmark uses the shared production fitting procedure while retaining
its own parameter aliases, calibration adapter and native CatBoost
preprocessing. Existing `AutoFeatureEngineer` defaults are unchanged.

Separate frozen synthetic comparisons covered binary/multiclass and mixed
inputs. A built wheel also fitted and predicted with weighted string labels
and three member plans while benchmark/optional learner/provider imports
and calibrator construction were blocked. Public class-weight copies and
non-pickleable in-memory estimator diagnostics received explicit regression
coverage. These checks do not expand the verified sklearn-version matrix
beyond the tested 1.7.2/1.8 environments.

## Integrity, artifacts and resources

All outcome identities are unique and scored. Dependencies and data/splits
match the pre-extraction replay. Runtime source remained:
`dc8931dd6286083af1048a3d728e0dcb2cdad0b7b3debcb797cea71f75436055`.
Exact source/result checksums and bounded reconstruction passed.

- [REPORT.md](REPORT.md): unchanged measured outcomes and timing.
- [parity.json](parity.json): checked fields and zero differences.
- [archive.json](archive.json): exact result parts.
- [auxiliary_archive.json](auxiliary_archive.json): bounded manifest/source.

Current-cycle measured execution totals **51.38516 of 72 hours**, leaving
**20.61484 hours**. No new data sources or live LLM calls were used.
Earlier failed or qualified confirmation decisions are not reclassified
by this successful engineering extraction.
