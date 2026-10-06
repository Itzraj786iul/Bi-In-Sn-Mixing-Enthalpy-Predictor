# Step 11B — Scientific and Documentation Corrections

Documentation and wording corrections only. The scientific model is frozen. No retraining, coefficient,
dataset, RKM, validation-code, API-behaviour or frontend-design change was made. Nothing was committed,
pushed or deployed.

## 1. Files changed in Step 11B

| File | Correction category |
| --- | --- |
| `README.md` | RKM protocol table + caveat; "lower error … not strictly equivalent" wording; boundary limitation |
| `web/README.md` | Intro label ("the final model"); RKM protocol table + caveat replacing the "LOCSO" RKM table; boundary limitation |
| `web/frontend/src/ScientificBlocks.jsx` | Validation table headers/caption (RKM = author-fitted reference, not LOCSO); provenance caveat; boundary limitation in Scope / Limitations (text only, no layout/design change) |
| `web/backend/model.py` | Docstring only ("Frozen final … trained on the 104 experimental observations only; primary validation is LOCSO"). Coefficients and code untouched |
| `reports/controlled_model_comparison.md` | RKM-not-LOCSO caveat after pooled table; L174 "RKM had no fitting" corrected to the author-fitted wording; "not refitted here" wording |
| `reports/project_work_report.md` | Historical banner; "at the time of writing" wording; §24 renamed "Status at the end of Stage 1 (historical)" |
| `docs/ml_validation_strategy.md` | Historical banner; "Not started at the time of writing" |
| `docs/understanding_synthetic_data.md` | Historical banner; synthetic augmentation wording; pipeline diagram ends at "not used to train the final predictor"; §14 marked historical; past-tense status |
| `docs/understanding_rkm.md` | Historical banner; L419 "No ML model has been trained" → past tense + final LOCSO metrics + like-for-like caveat; synthetic-label sentence updated |
| `docs/understanding_ml_baseline.md` | Historical (superseded baseline) banner with the required synthetic wording; "synthetic training arm of this baseline experiment"; §11 synthetic wording |
| `docs/rkm_model.md` | "Before any ML work" → "Note on synthetic data" + ablation outcome |
| `reports/ml_composition_representation_results.md` | Label note: local Model B = adopted final Direct Poly D2; unrelated to residual Model A / path-aware Model B; RKM row caveat |
| `reports/final_model_validation.md` | "Final candidate model" → note that it was adopted as the frozen final model; KFold 30.34 vs StratifiedKFold 31.22 (both secondary); RKM caveat |
| `reports/final_model_equation.md` | "captures experimental scatter" → "captures systematic deviations from RKM and temperature trends present in the measurements"; like-for-like caveat; boundary limitation |
| `reports/final_surrogate_analysis.md` | "Dataset B was used only as an evaluation grid; it was not used for training" |
| `reports/ml_polynomial_temperature_results.md`, `reports/ml_polynomial_synthetic_temperature_results.md`, `reports/ml_model_diagnostics.md` | Superseded notes (not the final model; final is xBi, xIn, T, LOCSO 41.10) |
| `reports/final_cleanup_inventory.md`, `reports/final_cleanup_summary.md` | Stage-1 (pre-ML) cleanup record notes |
| `reports/final_model_website_audit.md` (untracked) | Superseded by `step_8_consistency_fix.md`; pre-fix values (−1014.6924) no longer current |

`git diff --stat` for tracked files also includes the earlier Step 8/9 edits (T² precision, README rewrite), which remain uncommitted.

## 2. Categories of corrections

1. **Historical "ML not started" documents.** These documents keep their original content and context. A blockquote "Historical note" was added under each title, and only statements that would falsely imply ML is still unimplemented were changed to the past tense ("at the time of writing", "completed later"). No document was rewritten as though authored after the final model existed.
2. **Synthetic terminology.** The required wording is used: RKM-derived synthetic datasets were investigated as an augmentation strategy. They did not improve LOCSO prediction of the real observations (54.32 vs 41.10 J/mol). They are not used to train the final predictor and are retained for exploratory analysis. No document claims synthetic data are inherently invalid.
3. **RKM fitting statement.** `controlled_model_comparison.md` now says the RKM ternary parameters were fitted by the original authors using the paper's experimental measurements, and that RKM was not independently trained under LOCSO.
4. **RKM labelled as LOCSO.** In the website, both READMEs and the reports, RKM is now presented as a physics-model reference with author-fitted parameters, evaluated on the same 104 observations. It is no longer presented as a LOCSO score. No number was changed.
5. **Pure-component boundary limitation.** This limitation is now disclosed (see §4).
6. **Confusing labels.** Context notes were added for "Model A/B", "Final candidate model" and the backend docstring.
7. **Consistent final metrics and wording.** ML 41.10 / 56.34 / 0.9712; RKM 57.13 / 73.49 / 0.9510. R² is never called accuracy (all remaining hits are explicit negations). "Lower error under the reported evaluation", not universal superiority.
8. **Historical/superseded reports.** Short notes were added. Nothing was deleted or rewritten.

## 3. RKM caveat (as used)

| Model | Evaluation protocol | MAE (J/mol) | RMSE (J/mol) | R² |
| --- | --- | ---: | ---: | ---: |
| Direct Poly D2 (final) | LOCSO, held-out cross-section | 41.10 | 56.34 | 0.9712 |
| RKM | Published parameters evaluated on the 104 experimental observations (ternary parameters fitted to these measurements by the original authors) | 57.13 | 73.49 | 0.9510 |

"These are not strictly like-for-like out-of-sample scores: the ML result uses cross-section holdout, whereas
the RKM parameters were fitted to the experimental measurements by the original authors. The RKM value is
therefore a physics-model reference rather than an independent LOCSO validation score."

"The Direct Poly D2 model achieves lower error than the RKM reference under the reported evaluation, but the
two evaluation protocols are not strictly equivalent because the RKM parameters were fitted to the
experimental measurements."

## 4. Pure-component disclosure

This disclosure was added to `README.md`, `web/README.md`, `reports/final_model_equation.md` and the website's Scope / Limitations list. The polynomial is an empirical predictive surrogate and was not constrained to satisfy ΔmixH = 0 at the pure components. It therefore gives non-zero endpoint values, for example about −392 J/mol for pure Bi at 767 K. These boundary extrapolations are not physically valid pure-component thermodynamic predictions. The model was not modified.

## 5. Historical-document treatment

`project_work_report.md`, `ml_validation_strategy.md`, `understanding_synthetic_data.md`,
`understanding_rkm.md` and `understanding_ml_baseline.md` keep their structure, explanations and original
figures. Each has a banner that states what the document covers and points to the final results. The
Stage-1 status table in `project_work_report.md` still shows "Not started"; it is now titled
"Status at the end of Stage 1 (historical)" and has an explanatory line. `project_progress_report_initial_modeling.md`
was deliberately left untouched because it is a dated checkpoint whose PDF has already been delivered.

## 6. Validation results

| Check | Result |
| --- | --- |
| `python scripts/validate_dataset.py` | DATASET VALIDATION: PASS — 65 checks |
| Backend `COEF_T2` | 0.0009741759040480247 (exact) |
| `/predict` (xBi 0.2509, xIn 0.4982, 813 K) | −1014.5761 J/mol |
| RKM at same point | −1028.5461 J/mol |
| `/surface` at 767 / 813 / 855 K | HTTP 200 |
| LOCSO recomputation (3 folds, in memory) | MAE 41.10, RMSE 56.34, R² 0.9712 (unchanged) |
| RKM on 104 points | MAE 57.13, RMSE 73.49, R² 0.9510 (unchanged) |
| Lints on `ScientificBlocks.jsx` | none |

### Pattern search classification

| Pattern | Remaining hits | Classification |
| --- | --- | --- |
| "ML not started" / "not started" | `project_work_report.md` banner and §24 historical table; `ml_validation_strategy.md` "Not started at the time of writing"; `final_cleanup_inventory.md` L25 (now under a Stage-1 note); Step 8/9/10/11A reports quoting the old text | Valid historical context |
| "synthetic training data" | `understanding_ml_baseline.md` L140 ("a good score on synthetic training data does not prove the model learned physics") | Valid discussion |
| "RKM had no fitting" | Only in `step_11a_repository_curation.md` (describing the issue) | Valid discussion; source fixed |
| "RKM.*LOCSO" | New caveats ("not an out-of-sample LOCSO score"); residual reports ("RKM + residual … under LOCSO", a hybrid model that is LOCSO-validated); audit reports describing the issue; `project_progress_report_initial_modeling.md` L37/L440/L539 ("comparable to RKM under LOCSO") | Valid discussion, except the progress-report lines, which are stale but intentionally untouched (warning W-c) |
| "R².*accuracy" | All are explicit negations ("not 95.1% accurate", "not an accuracy measure") or audit questions | Valid discussion |

## 7. Build result

`npm run build` in `web/frontend`: exit code 0, built in 32.6 s. The only output was Vite's standard
"chunks larger than 500 kB" advisory (pre-existing, harmless).

## 8. Confirmation: model, data and API unchanged

- No coefficient changed (T² still the Step 8 full-precision value; the other nine unchanged).
- No dataset file was touched; validator passes 65 checks.
- `src/rkm_model.py`, validation scripts and all `scripts/*.py` are untouched.
- API responses reproduce the frozen benchmark values. The only backend edit is a docstring.
- The frontend edit is limited to text, captions and one added list item; no layout or styling changes.
- No commit, push or deployment.

## 9. Remaining warnings

- **W-a. Regeneration would revert wording.** Several reports are produced by scripts that still contain the old text. Rerunning the scripts would overwrite the Step 11B notes:
  - `step_8_controlled_model_comparison.py` would overwrite `controlled_model_comparison.md` (L174 text and the RKM caveat).
  - `analyze_composition_representation`, `validate_final_model`, `build_final_surrogate` and `extract_final_model` would overwrite their reports.
  - The scripts were not edited because validation code is frozen.
- **W-b.** `reports/project_work_report.pdf` no longer matches the edited `.md` and has no generator.
- **W-c.** `reports/project_progress_report_initial_modeling.md`/`.pdf` is a dated checkpoint and was left unchanged. It contains "comparable to RKM … under the same LOCSO pooling" and refers to the composition-only Poly D2 (54.64).
- **W-d.** The deployed website and the remote repository still show pre-Step-8/11B content until the user commits and redeploys (blocker B4 from Step 10; this is the user's decision).
- **W-e.** Five figures have no generator script: `experimental_cross_sections`, `rkm_vs_experiment`, `rkm_vs_experiment_cross_sections`, `ternary_enthalpy_map_813K`, `experimental_vs_synthetic`. `rkm_validation.md` also has no generator.
- **W-f.** `reports/final_surrogate_predictions.csv` (8.6 MB) is unreferenced and regenerable. Untracking it is recommended (Step 11A).
