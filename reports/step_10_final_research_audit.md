# Step 10 — Final Research-Grade Audit

**Scope:** read-only audit of the whole repository and the website, carried out as if the project were about to be submitted to the supervising professor.

**Question:** can every important scientific claim be traced to the experimental data, the professor's paper, the implemented RKM, or a documented ML experiment?

**Method:**
- Independent in-memory recomputation of data checks, RKM metrics, the LOCSO metrics and the final coefficients from `data/original_experimental_data.csv`. No files were written.
- Code inspection of `src/`, `web/backend/` and `web/frontend/src/`.
- Full read of `README.md`, `web/README.md`, all 5 files in `docs/` and all 26 `.md` files in `reports/`.
- Repository hygiene scan.

**Constraint:** nothing was modified. This report is the only new file.

**Verdict:** **23 PASS, 16 WARNING, 4 BLOCKER. Submission readiness: NOT READY.**

All four blockers are documentation or process issues. None affects the model, the data or the validated results, which reproduce exactly.

---

## 1. Experimental data provenance

| Check | Evidence (recomputed) | Status |
|---|---|---|
| 104 observations | 104 rows; IDs `EXP_0001`–`EXP_0104` unique; `source` = `paper_experiment` only; `source_table` = `Table III` only | PASS |
| Provenance from the paper | `reports/project_work_report.md` §4–5 (Table III, PDF pages 8100–8103); `professor_paper.pdf` tracked; `data/extraction_audit.csv` has 208 rows (one sign note and one composition-reconstruction note per observation), all `confidence = high` | PASS |
| Composition reconstruction | Max closure error \|xBi + xIn + xSn − 1\| = 2.2 × 10⁻¹⁶. Bi/(Bi+Sn) is exactly constant within each section: 0.667519 (Bi-rich), 0.500000 (Equiatomic), 0.333423 (Sn-rich) | PASS |
| Temperature / cross-section assignment | 9 series, one per (section, T): Bi-rich 11/11/12, Equiatomic 12/11/11, Sn-rich 12/12/12 points at 767/813/855 K. Target range −1413.0 to −64.32 J/mol, matching the README | PASS |
| No synthetic rows in the final predictor | Validator check "original file has no synthetic rows" passes. A refit of Poly D2 on the 104 rows reproduces `reports/final_model_coefficients.csv` with max difference 0.0. Six positive In-partial-enthalpy rows are annotated as "sign confirmed on rendered Table III" | PASS |

## 2. RKM audit

| Check | Evidence | Status |
|---|---|---|
| Described as a thermodynamic benchmark, not ML | README, website ("RKM thermodynamic model"), `docs/understanding_rkm.md` L475 ("not a neural network") | PASS |
| Inputs/outputs and binary + ternary terms match the code | `src/rkm_model.py`: three binary RK terms (Bi–In ν=0–2, Bi–Sn ν=0–3, In–Sn ν=0–1) plus a ternary term. The Table IV values in the code equal `docs/rkm_model.md` L38–62 and `docs/understanding_rkm.md` L45–50 and L145–185 | PASS |
| (In, Bi, Sn) index-ordering issue documented honestly | Code docstring L16–18, `docs/rkm_model.md` L64–88 (literal Bi–Sn–In order fails Fig. 10), `docs/understanding_rkm.md` §9, `reports/rkm_validation.md` | PASS |
| Temperature independence | Code ignores `T` (verified: identical output at 767 and 855 K). Documented in `docs/rkm_model.md` L109 and in the website and README | PASS |

## 3. Final ML model audit

| Check | Evidence | Status |
|---|---|---|
| Direct Poly D2 on xBi, xIn, T; xSn = 1 − xBi − xIn | `web/backend/model.py`, `scripts/extract_final_model.py` (`FEATURE_COLS = ["xBi","xIn","temperature_K"]`, `DEGREE = 2`) | PASS |
| Trained on the 104 real observations only; synthetic data not a dependency | `extract_final_model.py` reads only the experimental CSV. The backend reads only the experimental CSV, for map overlays. `build_final_surrogate.py` uses Dataset B as an evaluation grid only | PASS |
| Backend coefficients match the coefficient record; T² exact | T² = `0.0009741759040480247`, bit-identical to the CSV. The other nine are at 6 decimals (\|Δ\| ≤ 4.2 × 10⁻⁷; effect ≤ 9.2 × 10⁻⁵ J/mol) | PASS |

## 4. Validation audit

All values were recomputed independently.

| Fold held out | n_test / n_train | ML MAE | RKM MAE |
|---|---|---:|---:|
| Bi-rich (Sn0.33Bi0.67) | 34 / 70 | 49.67 | 88.38 |
| Equiatomic (Sn0.50Bi0.50) | 34 / 70 | 37.35 | 40.98 |
| Sn-rich (Sn0.67Bi0.33) | 36 / 68 | 36.56 | 42.88 |

| Model | MAE | RMSE | R² | Status |
|---|---:|---:|---:|---|
| ML Poly D2, pooled LOCSO (104 out-of-fold) | 41.10 | 56.34 | 0.9712 | PASS (reproduced) |
| RKM on the same 104 rows | 57.13 | 73.49 | 0.9510 | PASS (reproduced) |

| Check | Finding | Status |
|---|---|---|
| Same observations for ML and RKM | Yes, all 104 rows | PASS |
| No leakage | Each fold fits only on the other two cross-sections; the held-out path is never seen | PASS |
| Random 5-fold is secondary | `final_model_validation.md` L20 and L52–54, `controlled_model_comparison.md`: labelled secondary and interpolation-oriented | PASS |
| R² never called accuracy | Every occurrence of "accuracy" near R² is an explicit negation (e.g. `docs/understanding_rkm.md` L425, `project_work_report.md` L342) | PASS |
| Like-for-like ML vs RKM comparison | RKM's ternary L parameters were fitted by the paper's authors to these same Table III data, and the (In, Bi, Sn) ordering was chosen by fit to Table III. RKM's score is therefore effectively in-sample, while ML's is out-of-fold. This favours RKM, so ML's LOCSO advantage is conservative, but no document says so. See W2 | WARNING |

## 5. Synthetic-data audit

| Check | Evidence | Status |
|---|---|---|
| Current-state documents say the right thing: RKM-derived, tested as augmentation, no LOCSO benefit (41.10 vs 54.32), not used in the final predictor, exploratory use retained | `README.md` "Synthetic data", website Methodology note and the separate S1–S4 workflow, `reports/synthetic_data_ablation.md` L82–87 and L164 | PASS |
| No contradicting document | **Contradicted** in `docs/understanding_synthetic_data.md` L425–431 ("Future ML model learns from those labels") and `docs/understanding_ml_baseline.md` L134 ("Synthetic rows are useful for training density") | **BLOCKER (B2)** |

## 6. Website scientific claim audit

| Claim | Source check | Status |
|---|---|---|
| 104 observations; 3 cross-sections; 767/813/855 K; cross-section formulas | Matches the data | PASS |
| LOCSO ML 41.10 / 56.34 / 0.9712; RKM 57.13 / 73.49 / 0.9510 | Matches the recomputation (`ScientificBlocks.jsx` L41–53, L186–198, L473–491) | PASS |
| Benchmark from the API | `/predict` −1014.5761; `/comparison` EXP_0051 RKM −1028.5461, difference +13.9700 (Step 9 run; no source changes since) | PASS |
| Equation display | 10 coefficients at 6 decimals, equal to the backend, with a precision note on T² | PASS |
| No full-simplex validation claim | "Predictions at unsampled ternary compositions are model estimates and are not experimentally validated" (L111–112); "the full ternary simplex was not experimentally mapped" (L224–225); "Full-simplex model comparison — not experimental validation" (`ComparisonMap.jsx` L252); difference note (L194–195) | PASS |
| "experimentally validated surrogate" (`App.jsx` L248, `ScientificBlocks.jsx` L436) | Acceptable given the caveats, but slightly broad. See W13 | WARNING |
| Validation table rows "LOCSO MAE/RMSE/R²" include the RKM column | RKM is not LOCSO-validated; it is fixed, with authors' ternary fit on the same data. See W2 | WARNING |

## 7. Repository documentation audit

Line numbers refer to the current files.

### Blockers (current-state claims that are false today)

| ID | File / location | Problem | Recommended correction |
|---|---|---|---|
| B1 | `reports/project_work_report.md` L5, L25, L493, L596, L611, L618–633, L639–658 ("## 24. Current status": "ML model development \| **Not started**" … "Experimental ML validation \| **Not started**"), L683, L703 | Tracked "main" report states that no ML model exists. It is the README's "Details" link | Add a banner under the title: "Historical Stage-1 record (data + RKM stage). For the final model see `final_model_equation.md`, `final_model_validation.md`, `synthetic_data_ablation.md`. Final predictor: Direct Poly D2 (xBi, xIn, T), 104 experimental rows only, LOCSO MAE 41.10 J/mol." Rename §24 to "Status at end of Stage 1". Also note the stale "87 checks" (L652, L700; now 65) and the deleted files referenced at L386–391 |
| B1 | `docs/ml_validation_strategy.md` L3 ("before any ML code is written"), L5 ("**Not started yet:** model training…") | False status | Add a banner: "Pre-ML design document (historical). Outcome: LOCSO primary; Direct Poly D2 on xBi, xIn, T; random 5-fold secondary; synthetic augmentation rejected." |
| B1 | `docs/understanding_synthetic_data.md` L463 ("NEXT STAGE: ML surrogate ← NOT STARTED"), L475 ("Machine learning has **not** been implemented yet.") | False status | Replace with the final status and add a historical banner |
| B1 | `docs/understanding_rkm.md` L419 ("No ML model has been trained.") | False status | "The final ML model (Direct Poly D2) is compared against these numbers under LOCSO: MAE 41.10, RMSE 56.34, R² 0.9712." |
| B2 | `docs/understanding_synthetic_data.md` L425–431 (diagram "→ Future ML model learns from those labels") | Shows synthetic data as ML training input | "→ tested as augmentation, rejected → final ML trained on the 104 experimental rows only" |
| B2 | `docs/understanding_ml_baseline.md` L134 ("Synthetic rows are useful for training density…") | General claim contradicted by the ablation | "Synthetic rows were tested as augmentation but worsened LOCSO MAE (54.32 vs 41.10 J/mol) and are not used in the final predictor." |
| B3 | `reports/controlled_model_comparison.md` L174 ("RKM is a fixed, temperature-independent model from binary parameters with no fitting to these data") | Factually wrong. Table IV ternary L parameters were fitted by the authors to these calorimetric data (`docs/rkm_model.md` L93; `project_work_report.md` L111) | "RKM is a fixed, temperature-independent model (binary parameters from the authors' earlier work; ternary parameters fitted by the authors to these calorimetric data) that is never refitted in this project; …" |

### Warnings (stale or imprecise wording)

| ID | File / location | Problem | Recommended correction |
|---|---|---|---|
| W3 | `docs/understanding_rkm.md` L11, L433–437, L477; `docs/understanding_ml_baseline.md` L3, L5, L41, L52–54, L153, L158–164; `docs/ml_validation_strategy.md` L11–14 ("three kinds" lists two), L32, L44–46 (5-fold presented first, with xSn), L206; `docs/understanding_synthetic_data.md` L11, L17 (overstates trust in RKM interior), L209, L402; `docs/rkm_model.md` L127 ("Before any ML work") | Pre-ML future tense, superseded baselines with no pointer to the final model | A one-paragraph "historical / superseded — see final model" banner at the top of each `docs/` note. The B1/B2 line fixes cover the rest |
| W4 | `reports/final_cleanup_inventory.md` (L23) and `final_cleanup_summary.md` (L5–35 tree has no ML scripts or `web/`); `project_progress_report_initial_modeling.md` L29–35 and L467 ("Polynomial Degree 2" 54.64 is composition-only), L442 typo "≈ 57–57"; `ml_composition_representation_results.md` L13–21 ("Model A/B" collides with residual Model A / path-aware Model B, and does not say "Model B" became final); `ml_polynomial_temperature_results.md` L20 and `ml_polynomial_synthetic_temperature_results.md` L15 (43.81 model includes xSn, not final); `ml_model_diagnostics.md` L14 (features not stated); `final_model_validation.md` L3 ("Final candidate model") and random 5-fold 30.34 (KFold) vs 31.22 (StratifiedKFold in `controlled_model_comparison.md`) not cross-referenced | Historical reports that could be misread as describing the final model | Add stage or "superseded" one-liners; relabel "Poly D2 (composition-only)", "3-var + T / 2-var + T (adopted as final)"; note the two 5-fold protocols differ (both secondary) |
| W5 | `reports/final_surrogate_analysis.md` L1–17 | Dataset B role only implied | Add: "Dataset B was used only as an evaluation grid; it was not used for training." |
| W6 | `reports/final_model_equation.md` L105–106 ("ML captures experimental scatter…") | Reads as fitting noise | "captures systematic deviations from RKM and temperature trends present in the measurements" |
| W7 | `reports/controlled_model_comparison.md` L166 | Hard-to-parse sentence | "only Direct Poly D2 is below RKM on every held-out path; its margin on the Equiatomic and Sn-rich paths is small" |
| W12 | `web/README.md` L3 and `web/backend/model.py` docstring: "(Step 4A)" | Internal label collides with `scripts/step_4_residual_learning.py` (a residual model) | Replace with "(final model)" |
| W14 | `scripts/step_4…step_9_*.py` vs `reports/step_8_consistency_fix.md` / `step_9_documentation_sync.md` / `step_10_final_research_audit.md` | Two unrelated "step" numbering schemes (e.g. `scripts/step_8_controlled_model_comparison.py` ≠ `reports/step_8_consistency_fix.md`) | Add a short index in the README mapping scripts to reports, or rename one series |
| W15 | `reports/final_model_website_audit.md` L96–115, L143, L174 | Pre-fix values (−1014.69, +13.854) correctly reflect the audit time, but there is no forward pointer | Add "Superseded by `step_8_consistency_fix.md`" |

### Checked and consistent

`README.md`, `web/README.md` (apart from W12), `docs/rkm_model.md` (apart from L127), `reports/rkm_validation.md`, `synthetic_data_ablation.md`, `final_surface_analysis.md`, `residual_learning_results.md`, `residual_diagnostic_analysis.md`, `path_aware_residual_analysis.md`, `cross_section_residual_structure.md`, `ml_baseline_results.md`, `ml_polynomial_results.md`, `ml_random_forest_results.md`, `ml_gradient_boosting_results.md`, `step_8_consistency_fix.md`, `step_9_documentation_sync.md`. No document claims ML is universally superior to RKM; `README.md` L55 and `final_surface_analysis.md` L85 explicitly deny it.

## 8. Reproducibility audit

| Item | Where documented | Status |
|---|---|---|
| Experimental data | `data/original_experimental_data.csv`, README file table | PASS |
| How RKM is run | `src/rkm_model.rkm_delta_mix_h(x_bi, x_in, x_sn, T=None)`, `docs/rkm_model.md` | PASS |
| How ML is validated | LOCSO in `scripts/analyze_composition_representation.py` → `reports/ml_composition_representation_results.md`; secondary 5-fold in `scripts/validate_final_model.py`; coefficients in `scripts/extract_final_model.py` | PASS |
| Backend ↔ frontend | `web/README.md` (FastAPI endpoints, `VITE_API_BASE_URL`, Vite proxy, CORS `FRONTEND_ORIGIN`); `web/frontend/src/api.js` | PASS |
| How figures are generated | Most figures map to scripts (`analyze_ml_models.py`, `build_final_surrogate.py`, `analyze_final_surface.py`, `step_5…step_8`). **No generating script in the repo** for 5 tracked figures (`experimental_cross_sections.png`, `experimental_vs_synthetic.png`, `rkm_vs_experiment.png`, `rkm_vs_experiment_cross_sections.png`, `ternary_enthalpy_map_813K.png`) or for `reports/rkm_validation.md`. Its numbers do reproduce from `src/rkm_model.py` (verified: 57.13 / 73.49 / 0.9510) | WARNING (W9) |
| Dependencies | `requirements.txt` (numpy, pandas, scikit-learn, matplotlib) and `web/backend/requirements.txt` (fastapi, uvicorn[standard], pydantic, numpy): unpinned. `scripts/md_to_pdf_progress_report.py` needs `markdown` and `xhtml2pdf`, which are not listed. The frontend is pinned via `package-lock.json` | WARNING (W10) |

## 9. Repository hygiene (candidates only; nothing deleted)

| Item | Finding | Status |
|---|---|---|
| Secrets / API keys | None found. `.env`, `.env.local` ignored; only `.env.example` files tracked; `web/README.md` states no secrets are required | PASS |
| Caches / build output | `scripts/__pycache__/`, `src/__pycache__/`, `web/backend/__pycache__/`, `web/frontend/dist/`, `node_modules/` exist locally, all git-ignored (not tracked) | PASS |
| Large generated artifact | `reports/final_surrogate_predictions.csv` (8.2 MB, tracked), regenerable by `scripts/build_final_surrogate.py`. Candidate for removal from git or for documenting as generated | WARNING (W11) |
| Superseded experiment scripts | `train_baseline.py`, `train_random_forest.py`, `train_gradient_boosting.py`, `train_polynomial_baseline.py`, `train_polynomial_synthetic_temperature.py` and the `step_4…step_9` research scripts are a legitimate experiment record, not junk. Keep them, ideally indexed (W14) | — |
| Untracked deliverables | 11 figures, 15 reports and 7 scripts from Steps 1–9 are untracked, and 4 tracked files are modified. See B4 | **BLOCKER (B4)** |
| Bundle size | Vite chunk > 500 kB (Plotly); performance only | WARNING (W16) |

## 10. Professor-defense audit

### Q1. Why use ML if RKM already exists?

- **Evidence:** LOCSO results above; `README.md` L55.
- **Answer:** On the defined LOCSO protocol, Poly D2 has lower error than RKM: MAE 41.10 vs 57.13 J/mol, RMSE 56.34 vs 73.49 J/mol. ML is a data-driven surrogate of the measurements, not a replacement thermodynamic theory. RKM remains the physics-based benchmark.
- **Limitation:** The advantage is concentrated on the Bi-rich path (49.67 vs 88.38). On the Equiatomic path (37.35 vs 40.98) and the Sn-rich path (36.56 vs 42.88) the margin is small. RKM's score is effectively in-sample (W2), which makes the ML advantage conservative, but it is still one dataset with three paths.

### Q2. Why is LOCSO the primary validation?

- **Evidence:** `final_model_validation.md` L14–15 and L52–54; `controlled_model_comparison.md`.
- **Answer:** The experiment measured three composition paths. Random splits let neighbouring points on the same path fall in both train and test, which inflates the score (random 5-fold MAE about 30–31 vs LOCSO 41.10). Holding out a whole path tests generalisation to an unseen composition path.
- **Limitation:** There are only three folds, each trained on two paths, so the estimate has high variance. LOCSO still only tests along measured paths, not the triangle interior.

### Q3. Why did synthetic data hurt performance?

- **Evidence:** `synthetic_data_ablation.md`: real-only 41.10, real + Dataset A 54.32, Dataset A only 54.99 J/mol.
- **Answer:** Synthetic labels are RKM evaluations, so they carry RKM's systematic error (RKM MAE 57.13), and they outnumber the real training rows by about 100×. The model is pulled toward RKM: the real + A result (54.32) sits close to RKM (57.13) and to A-only (54.99).
- **Limitation:** The Dataset B variant was not run because no leakage-safe exclusion threshold was justified. Other schemes, such as sample weighting, were not tested, so the conclusion applies to the augmentation as tested.

### Q4. Why is xSn excluded from the final regression?

- **Evidence:** `ml_composition_representation_results.md` L20–21.
- **Answer:** xBi + xIn + xSn = 1 makes xSn an exact linear combination of the intercept, xBi and xIn, so including it is redundant and gives a rank-deficient design. Empirically, 2 composition variables + T (10 terms) gives LOCSO 41.10 / 56.34 / 0.9712, versus 43.81 / 61.17 / 0.9661 for 3 variables + T (15 terms). xSn is still reported, derived as 1 − xBi − xIn.
- **Limitation:** The choice of which two fractions to keep is a modelling convention. Degree-2 terms make the fitted coefficients basis-dependent, although predictions in the 10-term space are not.

### Q5. Does R² = 0.9712 mean 97.12% accuracy?

- **Evidence:** `docs/understanding_rkm.md` L425; `controlled_model_comparison.md` L175.
- **Answer:** No. R² is the fraction of variance explained relative to predicting the mean. Because ΔmixH spans about −64 to −1413 J/mol, R² stays high even with 50–100 J/mol errors. MAE and RMSE in J/mol are the primary error measures.
- **Limitation:** None; this is a matter of interpretation.

### Q6. Can the model be trusted outside the measured composition region?

- **Evidence:** Website caveats (§6); `final_surface_analysis.md` L84. New recomputation in this audit:

| Composition | ML at 767 K | ML at 855 K | RKM |
|---|---:|---:|---:|
| Pure Bi | −391.5 | −130.0 | 0 |
| Pure Sn | +33.2 | −59.1 | 0 |
| Pure In | −76.6 | +9.3 | 0 |
| In₅₀Sn₅₀ | −155.5 | −158.7 | −495.0 |
| Bi₅₀In₅₀ | −2194.7 | −2021.0 | −1654.2 |

- **Answer:** No. Predictions are supported only along the three measured paths (xIn ≈ 0.09–0.91) within 767–855 K. The polynomial does not enforce ΔmixH = 0 at the pure elements, and it departs strongly from RKM on the binary edges.
- **Limitation:** This specific behaviour (non-zero pure-element values) is **not disclosed** anywhere (W1). The general "unsampled = not validated" caveat is present.

### Q7. Why does the ML surface disagree with RKM?

- **Evidence:** `/comparison` at 813 K: mean \|ML − RKM\| 111.99, max 541.89 J/mol; `final_surface_analysis.md` L70–77 (largest gaps near xBi ≈ 0.75–0.81, xSn ≈ 0, in "unsampled_extrapolation" rows).
- **Answer:** The disagreement is largest where there are no measurements (the Bi–In edge region). There the unconstrained quadratic extrapolates, while RKM follows its binary parameters. The website labels this "model disagreement, not experimental confirmation of either model".
- **Limitation:** Neither model is validated there, so the size of the disagreement cannot be resolved without new measurements.

### Q8. Is the model physically interpretable?

- **Evidence:** Website equation disclaimer; `final_model_equation.md`.
- **Answer:** Only loosely. The coefficients are empirical regression parameters, not Redlich–Kister interaction parameters, and the website and reports state this.
- **Limitation:** There are no thermodynamic constraints: no zero at pure elements (Q6) and no consistency with binary data. Interpretation belongs to RKM.

### Q9. Why does temperature appear in the ML model when the paper calls ΔmixH "almost temperature-independent"?

- **Evidence:** Composition-only Poly D2 LOCSO MAE 54.64 (`ml_model_diagnostics.md` / progress report) vs 41.10 with T. At the benchmark composition, dΔmixH/dT ≈ +0.97 J/mol/K, and ML goes from −1057.2 J/mol (767 K) to −972.0 J/mol (855 K).
- **Answer:** Adding T reduced LOCSO error, and the implied change (about 85 J/mol over 88 K) is of the same order as the scatter between isotherms. It is reported as an empirical trend.
- **Limitation:** Each (cross-section, T) pair is a single calorimetric series (series_id 1–9 map one-to-one), so the T terms may partly absorb run-specific offsets rather than a true thermal effect. There are only three temperatures, and RKM has no T dependence to compare against.

### Q10. What is genuinely novel in this project?

- **Evidence:** README; `extraction_audit.csv`; `docs/rkm_model.md`; ablation, residual and controlled-comparison reports; `web/`.
- **Answer:**
  - An audited, reproducible digitisation of Table III.
  - A documented resolution of the RKM ternary index ordering.
  - A strict path-level (LOCSO) comparison of ML against RKM.
  - A documented negative result: RKM-derived synthetic augmentation does not help on real data.
  - Residual and path-aware experiments.
  - An interactive tool that explicitly scopes what is and is not validated.
- **Limitation:** There are no new measurements and no new thermodynamic model. The claims are methodological and computational, on one 104-point dataset.

## 11. Final verdict

### A. PASS (23)

1. 104 experimental observations, Table III only, unique IDs.
2. Provenance from the paper documented (`project_work_report.md` §4–5, `extraction_audit.csv`).
3. Composition reconstruction exact and documented.
4. Temperature / cross-section / series assignments consistent.
5. No synthetic row in the experimental file or in final training.
6. RKM implementation matches the documented equation and Table IV.
7. (In, Bi, Sn) ordering issue documented honestly.
8. RKM temperature independence correct; never described as ML.
9. Final model is Direct Poly D2 (xBi, xIn, T); refit equals the coefficient record exactly.
10. Backend coefficients match; T² exact (`0.0009741759040480247`).
11. ML LOCSO 41.10 / 56.34 / 0.9712 reproduced.
12. RKM 57.13 / 73.49 / 0.9510 reproduced on the same 104 rows.
13. LOCSO has no leakage.
14. Random 5-fold presented as secondary.
15. R² never called accuracy.
16. Current-state synthetic-data statements correct (README, website, ablation report).
17. Production backend has no synthetic, residual or path-aware dependency.
18. Website metrics, benchmark and equation display match the sources.
19. Website does not claim full-simplex experimental validation.
20. `README.md` and `web/README.md` consistent with the final model.
21. No secrets; caches and build output git-ignored.
22. Backend ↔ frontend communication documented.
23. `python scripts/validate_dataset.py`: PASS, 65 checks (re-run in this step). Build: PASS (Step 9; no source changes since).

### B. WARNINGS (16)

| ID | Item |
|---|---|
| W1 | Poly D2 gives non-zero ΔmixH at pure elements (e.g. pure Bi −391.5 J/mol at 767 K) and departs strongly from RKM on binary edges; not disclosed. Recommend one limitation sentence in `README.md` and the website "Scope / Limitations" list |
| W2 | ML vs RKM is not strictly like-for-like: RKM's ternary parameters and index order were fitted/chosen on the same Table III data (RKM effectively in-sample). The website table labels RKM rows "LOCSO". Recommend stating "RKM: fixed model, evaluated on all 104 points" |
| W3 | Pre-ML future-tense wording throughout `docs/` (beyond the blocker lines) |
| W4 | Historical reports without superseded markers or with confusable labels (§7) |
| W5 | `final_surrogate_analysis.md` does not state explicitly that Dataset B is an evaluation grid only |
| W6 | `final_model_equation.md` L105–106 "captures experimental scatter" |
| W7 | `controlled_model_comparison.md` L166 unclear sentence |
| W8 | `project_work_report.md` secondary staleness: 87 vs 65 checks; deleted files referenced (fixed together with B1) |
| W9 | 5 tracked figures and `rkm_validation.md` have no generating script in the repo |
| W10 | Unpinned Python requirements; `markdown` and `xhtml2pdf` missing for the PDF script |
| W11 | 8.2 MB regenerable `reports/final_surrogate_predictions.csv` tracked |
| W12 | "(Step 4A)" label in `web/README.md` and `model.py` docstring |
| W13 | "experimentally validated surrogate" wording on the website is broad (caveats exist) |
| W14 | Two unrelated "step_N" numbering schemes (scripts vs reports) |
| W15 | `final_model_website_audit.md` lacks a forward pointer to the Step 8 fix |
| W16 | Frontend bundle chunk > 500 kB (performance only) |

### C. BLOCKERS (4)

| ID | Item | Files requiring action |
|---|---|---|
| B1 | Tracked documents state that ML has not started / has not been implemented | `reports/project_work_report.md`, `docs/ml_validation_strategy.md`, `docs/understanding_synthetic_data.md`, `docs/understanding_rkm.md` |
| B2 | Documents describe synthetic data as ML training input, contradicting the final predictor and the ablation | `docs/understanding_synthetic_data.md` (L425–431), `docs/understanding_ml_baseline.md` (L134) |
| B3 | Factual error: RKM described as having "no fitting to these data" | `reports/controlled_model_comparison.md` (L174) |
| B4 | The final state exists only in the working tree. `origin/main` (last commit `861bbb8`, 2026-09-02) still has the README saying "Machine learning has not been implemented yet", the rounded T², and the website text "RKM-generated training data". `README.md` links to untracked reports (`synthetic_data_ablation.md` and others). The live Vercel/Render deployment, which builds from the repository, cannot contain the Step 8–9 corrections | Commit the 4 modified files and the intended untracked deliverables, push, then redeploy frontend and backend (user decision; not done in this audit) |

B1–B3 can each be fixed with a short banner or a one-sentence edit. No change to the model, data or results is needed.

**Submission readiness: NOT READY.** It becomes READY once B1–B4 are resolved; no scientific rework is required.

## 12. Git status at end of audit

- Branch `main`, up to date with `origin/main`.
- Modified (Steps 8–9): `README.md`, `web/README.md`, `web/backend/model.py`, `web/frontend/src/ScientificBlocks.jsx`.
- Untracked:
  - 11 figures (residual, path-aware, model comparison).
  - Reports: `controlled_model_comparison.{md,csv}`, `cross_section_residual_structure.md`, `final_model_website_audit.md`, `path_aware_residual_analysis.md`, `path_aware_residual_results.csv`, `project_progress_report_initial_modeling.{md,pdf}`, `residual_diagnostic_analysis.md`, `residual_learning_results.{md,csv}`, `step_8_consistency_fix.md`, `step_9_documentation_sync.md`, `synthetic_data_ablation.{md,csv}`, and this report.
  - Scripts: `md_to_pdf_progress_report.py`, `step_4…step_9_*.py`.

Nothing was committed or pushed.
