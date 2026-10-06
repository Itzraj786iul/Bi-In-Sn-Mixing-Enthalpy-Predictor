# Step 9 — Documentation Synchronization

This step only updated documentation, so that it matches the finalized experimental-only Direct Polynomial Degree-2 predictor. Nothing was retrained. Coefficients, metrics, datasets, RKM, API behaviour, validation methodology and frontend layout are unchanged.

## 1. Files changed in this step

| File | Change |
|------|--------|
| `web/README.md` | Benchmark updated; RKM metrics added next to the ML metrics; coefficient-precision note; experimental-only provenance stated |
| `README.md` (root) | Stale status and pipeline replaced with the finalized methodology, results and synthetic-data status |
| `web/frontend/src/ScientificBlocks.jsx` | One extra `equation-disclaimer` paragraph inside the existing "View fitted equation" block. Text only; existing class; no layout change |
| `reports/step_9_documentation_sync.md` | New file (this report) |

`web/backend/model.py` was not touched in this step. It shows as modified in git only because of the Step 8 T² precision fix.

## 2. Documentation inconsistencies corrected

| # | Location | Before | After |
|---|----------|--------|-------|
| 1 | `web/README.md` example response | `delta_mix_H_J_mol: -1014.69` (rounded-coefficient value from before Step 8) | `-1014.5761` |
| 2 | `web/README.md` benchmark note | "RKM … approximately −1028.55 J/mol" | Benchmark table at xBi 0.2509, xIn 0.4982, xSn 0.2509, T 813 K: ML −1014.5761, RKM −1028.5461, ML − RKM +13.9700 J/mol |
| 3 | `web/README.md` model summary | ML LOCSO metrics only | ML (41.10 / 56.34 / 0.9712) and RKM (57.13 / 73.49 / 0.9510) side by side; LOCSO named as primary validation |
| 4 | `web/README.md` model line | "Hard-coded coefficients (not refit)" | Adds the coefficient source CSV, training on 104 experimental observations only, and that no synthetic data are loaded |
| 5 | `web/README.md` | No note on precision | States that the backend uses the full-precision T² coefficient `0.0009741759040480247` and the website displays coefficients to 6 decimals |
| 6 | Root `README.md` | "Machine learning has not been implemented yet." | "Project status" section: final Direct Poly D2, features xBi, xIn, T; xSn = 1 − xBi − xIn; coefficient source; RKM benchmark |
| 7 | Root `README.md` pipeline | Validated RKM → Synthetic Datasets A/B → "Future ML model ← not started" | Two branches: the final predictor (104 experiments → reconstruction → fixed RKM benchmark → Direct Poly D2 → LOCSO → web app) and a separate exploratory synthetic branch ending in "exploratory analysis only" |
| 8 | Root `README.md` | No ML results | LOCSO table for ML and RKM, with the interpretation that the fitted model performs better under this LOCSO protocol but is not universally superior, and that RKM remains the physics-based benchmark |
| 9 | Root `README.md` synthetic data | "Before ML" section implied synthetic data were for future ML training | "Synthetic data" section: generated from RKM; augmentation evaluated; no improvement under strict LOCSO (54.32 vs 41.10 J/mol pooled MAE); not used in the final predictor; retained for exploratory analysis and visualization |
| 10 | Root `README.md` file table | No final-model files | Adds `extract_final_model.py`, `final_model_coefficients.csv`, the LOCSO results report, `final_model_validation.md`, `synthetic_data_ablation.md`, `web/`; synthetic CSVs marked exploratory only |
| 11 | Website fitted equation | No precision note | Added: "Coefficients are displayed to 6 decimal places; calculations use the full-precision fitted T² coefficient, the term most sensitive to rounding." |

On item 11: the website note names the T² term, not "full-precision fitted coefficients" in general. Only T² is stored at full precision in `model.py`. The other nine coefficients use the established 6-decimal values, which shifts predictions by at most 9.2 × 10⁻⁵ J/mol (Step 8). The generic wording would have overstated the precision.

R² is reported as "R² = 0.9712" everywhere and is never called "accuracy". A search of both READMEs and the frontend source finds no remaining "accuracy", "not started", "not been implemented", "−1014.69", "−1028.55" or "training data" wording.

## 3. Confirmation of no scientific or behavioural change

- Model: frozen experimental-only Direct Poly D2 (xBi, xIn, T). Unchanged.
- Coefficients: unchanged from Step 8; `COEF_T2 == 0.0009741759040480247` confirmed True.
- Datasets, synthetic datasets, RKM implementation and validation methodology: unchanged. LOCSO is still primary.
- API endpoints and responses: identical to the Step 8 results (below).
- Frontend: one text paragraph added in an existing block; no layout or style change.

## 4. Validation results

| Check | Result |
|-------|--------|
| `python scripts/validate_dataset.py` | DATASET VALIDATION: PASS, 65 checks passed |
| `POST /predict` benchmark | 200, −1014.5760950762894 J/mol |
| `POST /predict` xBi + xIn > 1 | 400 |
| `POST /predict` xBi < 0 | 422 |
| `POST /predict` T = 900 K or 766.9 K | 400 |
| `GET /surface` 767 K | 200, 5151 points, range −2197.91 … 313.72 J/mol, 35 experimental points |
| `GET /surface` 813 K | 200, 5151 points, range −2107.28 … 339.66 J/mol, 34 experimental points |
| `GET /surface` 855 K | 200, 5151 points, range −2021.65 … 374.94 J/mol, 35 experimental points |
| `GET /surface` 800 K | 400 |
| `GET /comparison` modes ml / rkm / difference (813 K) | 200 each; mean \|ML − RKM\| 111.99, max 541.89 J/mol; EXP_0051 ML −1014.5761, RKM −1028.5461, difference +13.9700 |
| `GET /comparison` T = 800 K or invalid mode | 400 |

All status codes were as expected.

## 5. Build result

`npm run build` in `web/frontend` exits 0 (43 modules, built in 40.5 s). The only warning is the existing chunk-size warning for the Plotly bundle. Output goes to the git-ignored `dist/`.

## 6. Final benchmark

xBi = 0.2509, xIn = 0.4982, xSn = 0.2509, T = 813 K (EXP_0051, measured −1017.0 J/mol):

| Quantity | Value |
|----------|------:|
| ML (Direct Poly D2) | −1014.5761 J/mol |
| RKM | −1028.5461 J/mol |
| ML − RKM | +13.9700 J/mol |

## 7. Final LOCSO metrics (unchanged)

| Model | MAE | RMSE | R² |
|-------|----:|-----:|---:|
| Direct Poly D2 (final) | 41.10 J/mol | 56.34 J/mol | 0.9712 |
| RKM benchmark | 57.13 J/mol | 73.49 J/mol | 0.9510 |

## 8. Git status

Modified, not committed:

- `README.md`: Step 9
- `web/README.md`: Step 9
- `web/backend/model.py`: Step 8, T² precision only
- `web/frontend/src/ScientificBlocks.jsx`: Steps 8 and 9

New in this step: `reports/step_9_documentation_sync.md`.

Earlier untracked files are unchanged by this step: the research scripts, reports and figures from the residual, ablation and comparison steps, `reports/final_model_website_audit.md`, `reports/step_8_consistency_fix.md`, and the progress report.

Nothing was committed or pushed.
