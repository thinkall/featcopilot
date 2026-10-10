# Robust classification calibration development study

Locked qualified confirmation: standard public row-level IID evidence with unresolved patient/entity mapping.
Evaluation tier: `standard_iid_unverified_entity`. No patient/entity-independent, clinical-generalization, prospective clinical-use or verified-entity pooled claim is permitted. Known group/time constraints remain binding.
Sampling mode: `subsample85`; protocol: `classification-qualified-confirmation-subsample85-v1`.
Three repeated stratified training bags request 85% training / 15% calibration per member. Returned row order is preserved; overlapping bags are not a three-fold partition or pooled OOF sample. Calibrated outcomes and every contrast involving them are diagnostic-only, not a variant-selection grid.
Member feature counts / maximum width describe distinct fold-local plans, not a common feature union.
Brier-sum is mean(sum over classes of squared probability error), including both binary classes.

Bundle plain/calibrated outcomes reuse exactly the same three fitted base procedures. Row pipeline_seconds are standalone attributions with shared training, not additive measured cost. Physical-job dispatch wall time is recorded once per physical_job_id, including retries; active_seconds in manifest.json is cumulative whole-controller wall time, including preparation, reporting and archiving. The ten-second report reserve is conservative, not a hard bound; actual budget overshoot is recorded.

Planned: 72 physical jobs / 96 prediction rows.
Scored: 96 / 96 prediction rows.

## Locked confirmation decision (authoritative; descriptive summaries below are not acceptance gates)

**FAIL**: 96/96 valid prediction rows; 72/72 complete physical jobs; all 2 registered families retained.
Only the locked two-family standard IID tier with unverified entity mapping; no robust, patient-independent, clinical, general-SOTA or default-promotion claim.
Undefined gains remain undefined; no epsilon, successful-family-only mean, or calibrated-policy substitution. Even PASS never authorizes automatic SOTA claims or default promotion.

| Learner | Fixed policy | Comparator | Defined / requested families | Mean gain | Median gain | Mean gate | Raw harm gate | Decision |
|---|---|---|---:|---:|---:|---|---|---|
| linear | featcopilot-frequency-aligned-robust | baseline-robust | 2 / 2 | 0.004139472722212045 | 0.004139472722212045 | True | True | pass |
| linear | featcopilot-frequency-aligned-robust | baseline-ensemble-robust | 2 / 2 | 0.001059819653583094 | 0.001059819653583094 | True | True | pass |
| linear | featcopilot-frequency-aligned-robust | openfe-inductive-native-robust | 2 / 2 | 0.0004149922342991941 | 0.0004149922342991941 | True | None | pass |
| linear | featcopilot-frequency-aligned-robust | autofeat-native-robust | 2 / 2 | -0.012782148436294993 | -0.012782148436294993 | False | None | fail |
| lightgbm | featcopilot-ensemble-robust | baseline-robust | 2 / 2 | 0.022484353331669427 | 0.022484353331669427 | True | True | pass |
| lightgbm | featcopilot-ensemble-robust | baseline-ensemble-robust | 2 / 2 | 0.0021943517510573107 | 0.0021943517510573107 | True | True | pass |
| lightgbm | featcopilot-ensemble-robust | openfe-inductive-native-robust | 2 / 2 | 0.07331563935758645 | 0.07331563935758645 | True | None | pass |
| lightgbm | featcopilot-ensemble-robust | autofeat-native-robust | 2 / 2 | 0.07513645589154422 | 0.07513645589154422 | True | None | pass |

### Both-family absolute losses and policy gains

| Learner | Comparator | Family | Policy mean loss | Reference mean loss | Gain | Undefined reason | Raw harm passed |
|---|---|---|---:|---:|---:|---|---|
| linear | baseline-robust | haberman-survival | 0.5397839920949035 | 0.5444392682907083 | 0.00855058859075375 | None | True |
| linear | baseline-robust | mammographic-mass | 0.45823182483998637 | 0.4581073831090818 | -0.0002716431463296587 | None | True |
| linear | baseline-ensemble-robust | haberman-survival | 0.5397839920949035 | 0.5453963425793514 | 0.010290407262185552 | None | True |
| linear | baseline-ensemble-robust | mammographic-mass | 0.45823182483998637 | 0.45451806321409916 | -0.008170767955019365 | None | True |
| linear | openfe-inductive-native-robust | haberman-survival | 0.5397839920949035 | 0.5444392682907083 | 0.00855058859075375 | None | None |
| linear | openfe-inductive-native-robust | mammographic-mass | 0.45823182483998637 | 0.45472110321606535 | -0.007720604122155361 | None | None |
| linear | autofeat-native-robust | haberman-survival | 0.5397839920949035 | 0.5292021346011666 | -0.01999587076819309 | None | None |
| linear | autofeat-native-robust | mammographic-mass | 0.45823182483998637 | 0.45569432466688575 | -0.005568426104396895 | None | None |
| lightgbm | baseline-robust | haberman-survival | 0.6052159694514235 | 0.647250443946363 | 0.06494313737144819 | None | True |
| lightgbm | baseline-robust | mammographic-mass | 0.4858194340332341 | 0.47630550277221917 | -0.019974430708109336 | None | True |
| lightgbm | baseline-ensemble-robust | haberman-survival | 0.6052159694514235 | 0.605112040471607 | -0.00017175163088056777 | None | True |
| lightgbm | baseline-ensemble-robust | mammographic-mass | 0.4858194340332341 | 0.4880451420062297 | 0.004560455132995189 | None | True |
| lightgbm | openfe-inductive-native-robust | haberman-survival | 0.6052159694514235 | 0.6533487466207555 | 0.07367088009012623 | None | None |
| lightgbm | openfe-inductive-native-robust | mammographic-mass | 0.4858194340332341 | 0.5240546717882206 | 0.07296039862504666 | None | None |
| lightgbm | autofeat-native-robust | haberman-survival | 0.6052159694514235 | 0.7072542815726564 | 0.14427386978038448 | None | None |
| lightgbm | autofeat-native-robust | mammographic-mass | 0.4858194340332341 | 0.4887514746585945 | 0.005999042002703958 | None | None |

Failed locked criteria:
- linear/autofeat-native-robust: locked mean-gain gate failed

## Relative to raw single-model baseline

| Learner | Method | Complete families / requested | Mean loss reduction | Diagnostic only |
|---|---|---:|---:|---|
| lightgbm | autofeat-native-robust | 2 / 2 | -0.059418 | False |
| lightgbm | baseline-calibrated-robust | 2 / 2 | 0.054062 | True |
| lightgbm | baseline-ensemble-robust | 2 / 2 | 0.020228 | False |
| lightgbm | baseline-robust | 2 / 2 | 0.000000 | False |
| lightgbm | featcopilot-calibrated-robust | 2 / 2 | 0.056093 | True |
| lightgbm | featcopilot-ensemble-robust | 2 / 2 | 0.022484 | False |
| lightgbm | featcopilot-frequency-aligned-robust | 2 / 2 | 0.000000 | False |
| lightgbm | openfe-inductive-native-robust | 2 / 2 | -0.054835 | False |
| linear | autofeat-native-robust | 2 / 2 | 0.016627 | False |
| linear | baseline-calibrated-robust | 2 / 2 | -0.011269 | True |
| linear | baseline-ensemble-robust | 2 / 2 | 0.003039 | False |
| linear | baseline-robust | 2 / 2 | 0.000000 | False |
| linear | featcopilot-calibrated-robust | 2 / 2 | -0.015013 | True |
| linear | featcopilot-ensemble-robust | 2 / 2 | -0.002029 | False |
| linear | featcopilot-frequency-aligned-robust | 2 / 2 | 0.004139 | False |
| linear | openfe-inductive-native-robust | 2 / 2 | 0.003696 | False |

Predeclared policy: linear uses single FE; LightGBM uses plain subsample85 FE. No post-score learner, dataset or method choice. Calibration remains diagnostic-only.
OpenFE/AutoFeat references use the existing native-width single-model adaptations with matched final clipping, not an ensemble architecture or a paper-best performance claim.
No robust guarantee follows from this two-family tier.

Full claim boundary: `{"claim_scope": "Row-level standard public IID benchmark evidence only, under explicitly unresolved patient/entity mapping.", "clinical_generalization_claims_allowed": false, "evaluation_tier": "standard_iid_unverified_entity", "future_confirmation_requires_explicit_tier_binding": true, "known_group_time_constraints_must_be_preserved": true, "limits": "This is not a waiver for known repeat-entity or temporal leakage. If later evidence identifies available group IDs, repeated-entity links or mandatory temporal boundaries, stop and review; do not drop or ignore them to preserve IID eligibility.", "patient_entity_disjoint_claims_allowed": false, "patient_entity_independence_verified": false, "pool_with_verified_entity_cohorts": false, "prospective_clinical_decision_use_authorized": false}`.

## All arms and native references versus matched raw subsample85

| Learner | Method | Complete families / requested | Mean loss reduction | Diagnostic only |
|---|---|---:|---:|---:|
| lightgbm | autofeat-native-robust | 2 / 2 | -0.085123 | False |
| lightgbm | baseline-calibrated-robust | 2 / 2 | 0.036138 | True |
| lightgbm | baseline-ensemble-robust | 2 / 2 | 0.000000 | False |
| lightgbm | baseline-robust | 2 / 2 | -0.022791 | False |
| lightgbm | featcopilot-calibrated-robust | 2 / 2 | 0.038114 | True |
| lightgbm | featcopilot-ensemble-robust | 2 / 2 | 0.002194 | False |
| lightgbm | featcopilot-frequency-aligned-robust | 2 / 2 | -0.022791 | False |
| lightgbm | openfe-inductive-native-robust | 2 / 2 | -0.076749 | False |
| linear | autofeat-native-robust | 2 / 2 | 0.013552 | False |
| linear | baseline-calibrated-robust | 2 / 2 | -0.014331 | True |
| linear | baseline-ensemble-robust | 2 / 2 | 0.000000 | False |
| linear | baseline-robust | 2 / 2 | -0.003071 | False |
| linear | featcopilot-calibrated-robust | 2 / 2 | -0.018088 | True |
| linear | featcopilot-ensemble-robust | 2 / 2 | -0.005075 | False |
| linear | featcopilot-frequency-aligned-robust | 2 / 2 | 0.001060 | False |
| linear | openfe-inductive-native-robust | 2 / 2 | 0.000654 | False |

## FE versus matched raw representation

| Learner | Architecture | Complete families / requested | Mean loss reduction | Diagnostic only |
|---|---|---:|---:|---|
| lightgbm | single | 2 / 2 | 0.000000 | False |
| linear | single | 2 / 2 | 0.004139 | False |
| lightgbm | plain-ensemble | 2 / 2 | 0.002194 | False |
| linear | plain-ensemble | 2 / 2 | -0.005075 | False |
| lightgbm | sigmoid-ensemble | 2 / 2 | 0.001972 | True |
| linear | sigmoid-ensemble | 2 / 2 | -0.003706 | True |

## Calibration operation: calibrated minus plain (diagnostic-only)

Equal-weight complete-fold families; negative loss/Brier differences favor calibration.

| Learner | Representation | Complete / requested families | Loss difference | Brier-sum difference |
|---|---|---:|---:|---:|
| linear | raw | 2 / 2 | 0.007354 | 0.003780 |
| linear | fe | 2 / 2 | 0.006601 | 0.003536 |
| lightgbm | raw | 2 / 2 | -0.021804 | -0.013078 |
| lightgbm | fe | 2 / 2 | -0.021699 | -0.012751 |

## Repeated stratified training bags: observed membership

One row per scored physical bundle (not per prediction arm). Fractions use actual member sizes. Coverage counts outer-training rows held out 0, 1, 2, or 3 times. Overlap lists pairwise train/calibration intersections for pairs 0-1, 0-2, 1-2; no pooled OOF claim.

| Physical job | Actual train fractions | Actual calibration fractions | Coverage 0..3 | Overlap | Distinct bags |
|---|---|---|---|---|---:|
| 9ccbc3128cdfd946f110ae04 | [0.8480392156862745, 0.8480392156862745, 0.8480392156862745] | [0.15196078431372548, 0.15196078431372548, 0.15196078431372548] | [121, 73, 10, 0] | [{'members': [0, 1], 'base_train': 144, 'calibration_validation': 2}, {'members': [0, 2], 'base_train': 148, 'calibration_validation': 6}, {'members': [1, 2], 'base_train': 144, 'calibration_validation': 2}] | 3 |
| 9c448c943cdc23f621d3acb9 | [0.8480392156862745, 0.8480392156862745, 0.8480392156862745] | [0.15196078431372548, 0.15196078431372548, 0.15196078431372548] | [121, 73, 10, 0] | [{'members': [0, 1], 'base_train': 144, 'calibration_validation': 2}, {'members': [0, 2], 'base_train': 148, 'calibration_validation': 6}, {'members': [1, 2], 'base_train': 144, 'calibration_validation': 2}] | 3 |
| f51963bc93337542867550a5 | [0.8480392156862745, 0.8480392156862745, 0.8480392156862745] | [0.15196078431372548, 0.15196078431372548, 0.15196078431372548] | [121, 73, 10, 0] | [{'members': [0, 1], 'base_train': 144, 'calibration_validation': 2}, {'members': [0, 2], 'base_train': 148, 'calibration_validation': 6}, {'members': [1, 2], 'base_train': 144, 'calibration_validation': 2}] | 3 |
| 32b9c0f5a87164aac9a8eb91 | [0.8480392156862745, 0.8480392156862745, 0.8480392156862745] | [0.15196078431372548, 0.15196078431372548, 0.15196078431372548] | [121, 73, 10, 0] | [{'members': [0, 1], 'base_train': 144, 'calibration_validation': 2}, {'members': [0, 2], 'base_train': 148, 'calibration_validation': 6}, {'members': [1, 2], 'base_train': 144, 'calibration_validation': 2}] | 3 |
| ff8960a03e0dc937ad28d65a | [0.85, 0.85, 0.85] | [0.15, 0.15, 0.15] | [399, 197, 41, 3] | [{'members': [0, 1], 'base_train': 467, 'calibration_validation': 19}, {'members': [0, 2], 'base_train': 460, 'calibration_validation': 12}, {'members': [1, 2], 'base_train': 467, 'calibration_validation': 19}] | 3 |
| a27000f1c12ff7ad2ee3350a | [0.85, 0.85, 0.85] | [0.15, 0.15, 0.15] | [399, 197, 41, 3] | [{'members': [0, 1], 'base_train': 467, 'calibration_validation': 19}, {'members': [0, 2], 'base_train': 460, 'calibration_validation': 12}, {'members': [1, 2], 'base_train': 467, 'calibration_validation': 19}] | 3 |
| 09581ee6814f19e8ec0b435b | [0.85, 0.85, 0.85] | [0.15, 0.15, 0.15] | [399, 197, 41, 3] | [{'members': [0, 1], 'base_train': 467, 'calibration_validation': 19}, {'members': [0, 2], 'base_train': 460, 'calibration_validation': 12}, {'members': [1, 2], 'base_train': 467, 'calibration_validation': 19}] | 3 |
| ea04f8a7b669a606db06f79f | [0.85, 0.85, 0.85] | [0.15, 0.15, 0.15] | [399, 197, 41, 3] | [{'members': [0, 1], 'base_train': 467, 'calibration_validation': 19}, {'members': [0, 2], 'base_train': 460, 'calibration_validation': 12}, {'members': [1, 2], 'base_train': 467, 'calibration_validation': 19}] | 3 |
| ab919e48c19a94bf7b536749 | [0.8480392156862745, 0.8480392156862745, 0.8480392156862745] | [0.15196078431372548, 0.15196078431372548, 0.15196078431372548] | [121, 73, 10, 0] | [{'members': [0, 1], 'base_train': 144, 'calibration_validation': 2}, {'members': [0, 2], 'base_train': 148, 'calibration_validation': 6}, {'members': [1, 2], 'base_train': 144, 'calibration_validation': 2}] | 3 |
| 7b2e4447db35547f98634dc0 | [0.8480392156862745, 0.8480392156862745, 0.8480392156862745] | [0.15196078431372548, 0.15196078431372548, 0.15196078431372548] | [121, 73, 10, 0] | [{'members': [0, 1], 'base_train': 144, 'calibration_validation': 2}, {'members': [0, 2], 'base_train': 148, 'calibration_validation': 6}, {'members': [1, 2], 'base_train': 144, 'calibration_validation': 2}] | 3 |
| a3b076a53f0962aa6a448e5b | [0.8480392156862745, 0.8480392156862745, 0.8480392156862745] | [0.15196078431372548, 0.15196078431372548, 0.15196078431372548] | [121, 73, 10, 0] | [{'members': [0, 1], 'base_train': 144, 'calibration_validation': 2}, {'members': [0, 2], 'base_train': 148, 'calibration_validation': 6}, {'members': [1, 2], 'base_train': 144, 'calibration_validation': 2}] | 3 |
| fe885ed56361f4894e250568 | [0.8480392156862745, 0.8480392156862745, 0.8480392156862745] | [0.15196078431372548, 0.15196078431372548, 0.15196078431372548] | [121, 73, 10, 0] | [{'members': [0, 1], 'base_train': 144, 'calibration_validation': 2}, {'members': [0, 2], 'base_train': 148, 'calibration_validation': 6}, {'members': [1, 2], 'base_train': 144, 'calibration_validation': 2}] | 3 |
| 26d2b2f5b01ca571b097b8e5 | [0.8486739469578783, 0.8486739469578783, 0.8486739469578783] | [0.15132605304212168, 0.15132605304212168, 0.15132605304212168] | [395, 204, 39, 3] | [{'members': [0, 1], 'base_train': 465, 'calibration_validation': 18}, {'members': [0, 2], 'base_train': 459, 'calibration_validation': 12}, {'members': [1, 2], 'base_train': 465, 'calibration_validation': 18}] | 3 |
| dd5a6d88ec803e4cc984ee09 | [0.8486739469578783, 0.8486739469578783, 0.8486739469578783] | [0.15132605304212168, 0.15132605304212168, 0.15132605304212168] | [395, 204, 39, 3] | [{'members': [0, 1], 'base_train': 465, 'calibration_validation': 18}, {'members': [0, 2], 'base_train': 459, 'calibration_validation': 12}, {'members': [1, 2], 'base_train': 465, 'calibration_validation': 18}] | 3 |
| 3bb55dd6e5b12b54c741fb77 | [0.8486739469578783, 0.8486739469578783, 0.8486739469578783] | [0.15132605304212168, 0.15132605304212168, 0.15132605304212168] | [395, 204, 39, 3] | [{'members': [0, 1], 'base_train': 465, 'calibration_validation': 18}, {'members': [0, 2], 'base_train': 459, 'calibration_validation': 12}, {'members': [1, 2], 'base_train': 465, 'calibration_validation': 18}] | 3 |
| c6aa8fde2bc4e3c556363756 | [0.8486739469578783, 0.8486739469578783, 0.8486739469578783] | [0.15132605304212168, 0.15132605304212168, 0.15132605304212168] | [395, 204, 39, 3] | [{'members': [0, 1], 'base_train': 465, 'calibration_validation': 18}, {'members': [0, 2], 'base_train': 459, 'calibration_validation': 12}, {'members': [1, 2], 'base_train': 465, 'calibration_validation': 18}] | 3 |
| a8af9d8d06cc8ee906724528 | [0.8480392156862745, 0.8480392156862745, 0.8480392156862745] | [0.15196078431372548, 0.15196078431372548, 0.15196078431372548] | [121, 73, 10, 0] | [{'members': [0, 1], 'base_train': 144, 'calibration_validation': 2}, {'members': [0, 2], 'base_train': 148, 'calibration_validation': 6}, {'members': [1, 2], 'base_train': 144, 'calibration_validation': 2}] | 3 |
| 4028fce5ace81c3f0d700013 | [0.8480392156862745, 0.8480392156862745, 0.8480392156862745] | [0.15196078431372548, 0.15196078431372548, 0.15196078431372548] | [121, 73, 10, 0] | [{'members': [0, 1], 'base_train': 144, 'calibration_validation': 2}, {'members': [0, 2], 'base_train': 148, 'calibration_validation': 6}, {'members': [1, 2], 'base_train': 144, 'calibration_validation': 2}] | 3 |
| 92a550fc1d5500292c34216f | [0.8480392156862745, 0.8480392156862745, 0.8480392156862745] | [0.15196078431372548, 0.15196078431372548, 0.15196078431372548] | [121, 73, 10, 0] | [{'members': [0, 1], 'base_train': 144, 'calibration_validation': 2}, {'members': [0, 2], 'base_train': 148, 'calibration_validation': 6}, {'members': [1, 2], 'base_train': 144, 'calibration_validation': 2}] | 3 |
| 7cf5464527c2a9e47aa2cf7c | [0.8480392156862745, 0.8480392156862745, 0.8480392156862745] | [0.15196078431372548, 0.15196078431372548, 0.15196078431372548] | [121, 73, 10, 0] | [{'members': [0, 1], 'base_train': 144, 'calibration_validation': 2}, {'members': [0, 2], 'base_train': 148, 'calibration_validation': 6}, {'members': [1, 2], 'base_train': 144, 'calibration_validation': 2}] | 3 |
| f908ac45beca96b5fb8ffdde | [0.8486739469578783, 0.8486739469578783, 0.8486739469578783] | [0.15132605304212168, 0.15132605304212168, 0.15132605304212168] | [395, 204, 39, 3] | [{'members': [0, 1], 'base_train': 465, 'calibration_validation': 18}, {'members': [0, 2], 'base_train': 459, 'calibration_validation': 12}, {'members': [1, 2], 'base_train': 465, 'calibration_validation': 18}] | 3 |
| 1f76a97a55bd0c767d10d914 | [0.8486739469578783, 0.8486739469578783, 0.8486739469578783] | [0.15132605304212168, 0.15132605304212168, 0.15132605304212168] | [395, 204, 39, 3] | [{'members': [0, 1], 'base_train': 465, 'calibration_validation': 18}, {'members': [0, 2], 'base_train': 459, 'calibration_validation': 12}, {'members': [1, 2], 'base_train': 465, 'calibration_validation': 18}] | 3 |
| 2cc7107e373572c40104eb67 | [0.8486739469578783, 0.8486739469578783, 0.8486739469578783] | [0.15132605304212168, 0.15132605304212168, 0.15132605304212168] | [395, 204, 39, 3] | [{'members': [0, 1], 'base_train': 465, 'calibration_validation': 18}, {'members': [0, 2], 'base_train': 459, 'calibration_validation': 12}, {'members': [1, 2], 'base_train': 465, 'calibration_validation': 18}] | 3 |
| 47c45b546c1f077b72432fa0 | [0.8486739469578783, 0.8486739469578783, 0.8486739469578783] | [0.15132605304212168, 0.15132605304212168, 0.15132605304212168] | [395, 204, 39, 3] | [{'members': [0, 1], 'base_train': 465, 'calibration_validation': 18}, {'members': [0, 2], 'base_train': 459, 'calibration_validation': 12}, {'members': [1, 2], 'base_train': 465, 'calibration_validation': 18}] | 3 |

Matched raw/FE architecture summaries, all errors and original dataset metadata: `results.json`.
Exact lossless gzip parts (each <=480 KiB): `archive.json`. Source allowlist: `manifest.json`.
Final cumulative controller time and any budget overshoot: `manifest.json`.
