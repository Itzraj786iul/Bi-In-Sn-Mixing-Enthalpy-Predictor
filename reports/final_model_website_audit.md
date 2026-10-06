# Final Model ↔ Website Audit

> **Historical note (superseded):** This audit records the state before the Step 8 fix. The issues it reports (T² coefficient rounding and the website's synthetic-data wording) were corrected in `step_8_consistency_fix.md`. Pre-fix values quoted here (e.g. benchmark /predict −1014.6924) are no longer current. The current value is −1014.5761.

Audit-only review of whether the deployed website (FastAPI backend + React frontend) uses the
scientifically selected final model. No source, model, dataset, report, website, or package file
was modified. Numerical checks were run in memory (temporary script outside the repository,
bytecode writing disabled); the frontend build wrote only to the git-ignored `web/frontend/dist/`.
`git status` was identical before and after the audit (apart from this report).

Classification: **PASS** = consistent; **WARNING** = consistent in prediction behaviour but with a
minor numerical or documentation issue; **MISMATCH** = the website does not implement the final model.

---

## 1. Scientific final-model decision

| Item | Value | Status |
|------|-------|--------|
| Final model | Direct Polynomial Degree-2 regression, experimental data only | PASS |
| Features | `xBi`, `xIn`, `temperature_K` (10 terms incl. intercept) | PASS |
| `xSn` | Not a feature; derived as `xSn = 1 − xBi − xIn` (avoids the exact collinearity xBi + xIn + xSn = 1) | PASS |
| Training data | 104 rows of `data/original_experimental_data.csv`, all `source = paper_experiment` | PASS |
| Fit method | Minimum-norm least squares (`np.linalg.lstsq`) on `PolynomialFeatures(degree=2, include_bias=True)` | PASS |
| Coefficient source | `scripts/extract_final_model.py` → `reports/final_model_coefficients.csv` (reads experimental CSV only) | PASS |
| Primary validation (LOCSO) | MAE 41.10, RMSE 56.34, R² 0.9712 J/mol (RKM: 57.13 / 73.49 / 0.9510) | PASS |
| Excluded from production | Synthetic Datasets A/B (exploratory only, Step 9 ablation); residual Model A and path-aware Model B (research only, Steps 4–8) | PASS |

In-memory refit of the 10-term Poly D2 on the 104 experimental rows reproduces
`reports/final_model_coefficients.csv` to a maximum absolute difference of **2.3 × 10⁻¹³**.
The CSV is therefore exactly the experimental-only final model.

## 2. Current backend model

| Item | Finding | Status |
|------|---------|--------|
| Implementation | `web/backend/model.py`, hard-coded constants, docstring "Frozen Polynomial Degree-2 mixing enthalpy surrogate" | PASS |
| Frozen vs dynamic | Frozen. No training, no CSV coefficient loading, no pickle/joblib, no sklearn at runtime | PASS |
| Equation | `INTERCEPT + b1·xBi + b2·xIn + b3·T + b4·xBi² + b5·xBi·xIn + b6·xBi·T + b7·xIn² + b8·xIn·T + b9·T²` (degree 2, inputs xBi/xIn/T only) | PASS |
| xSn in equation | Absent (intentional) | PASS |
| Endpoints using it | `POST /predict` (`predict` → `predict_delta_mix_h`), `GET /surface` and `GET /comparison` (`predict_delta_mix_h_batch`) | PASS |
| Other model code in backend | None. Only additional import is `src.rkm_model.rkm_delta_mix_h` in `comparison.py` (benchmark only) | PASS |
| External ML API | None | PASS |

## 3. Current coefficients

| Term | Expected (established) | Backend `model.py` | Frontend `FROZEN_EQUATION` | Full-precision CSV | Status |
|------|-----------------------:|-------------------:|---------------------------:|-------------------:|--------|
| intercept | 1477.018848 | 1477.018848 | 1477.018848 | 1477.0188480073768 | PASS |
| xBi | −1633.602767 | −1633.602767 | −1633.602767 | −1633.602766580109 | PASS |
| xIn | −2198.688284 | −2198.688284 | −2198.688284 | −2198.688284216979 | PASS |
| T | −2.629607 | −2.629607 | −2.629607 | −2.629607106032224 | PASS |
| xBi² | −1875.297539 | −1875.297539 | −1875.297539 | −1875.2975391483956 | PASS |
| xBi·xIn | −9182.903143 | −9182.903143 | −9182.903143 | −9182.90314265283 | PASS |
| xBi·T | +4.021166 | +4.021166 | +4.021166 | 4.02116633233977 | PASS |
| xIn² | +535.330326 | +535.330326 | +535.330326 | 535.3303257923645 | PASS |
| xIn·T | +2.025539 | +2.025539 | +2.025539 | 2.025539140923673 | PASS |
| T² | +0.000974 | +0.000974 | +0.000974 | 0.000974175904048 | **WARNING** |

All ten backend coefficients equal the established set exactly; the frontend's displayed equation
equals the backend. **Coefficient match: PASS.**

**WARNING (numerical, minor):** the backend stores coefficients rounded to 6 decimals. For the nine
large-magnitude terms this is negligible (|Δ| ≤ 4.2 × 10⁻⁷). For T², the 6-decimal rounding drops
1.76 × 10⁻⁷, which is multiplied by T² ≈ 6–7 × 10⁵. Combined effect of all rounding vs the
full-precision CSV:

| T (K) | Backend − full-precision prediction |
|------:|------------------------------------:|
| 767 | −0.104 J/mol |
| 813 | −0.116 J/mol |
| 855 | −0.129 J/mol |

Over all 104 experimental rows the maximum shift is 0.129 J/mol (training MAE 27.10 vs 27.09 J/mol).
This is ~0.3 % of the LOCSO MAE and far below experimental uncertainty, so it does not change any
scientific conclusion. It is recorded only because the backend is not bit-identical to the CSV.

## 4. Current website prediction path

| Step | Finding | Status |
|------|---------|--------|
| Frontend input | `App.jsx` sends `{xBi, xIn, temperature_K}` to `POST /predict` via `apiUrl()` (`VITE_API_BASE_URL`) | PASS |
| xSn derivation | Frontend preview `1 − xBi − xIn`; backend returns `xSn = 1 − xBi − xIn` | PASS |
| Client composition checks | NaN check; 0 ≤ xBi ≤ 1; 0 ≤ xIn ≤ 1; xBi + xIn ≤ 1 | PASS |
| Client temperature check | 767 ≤ T ≤ 855 K | PASS |
| Server schema (pydantic) | xBi, xIn ∈ [0, 1] → HTTP 422 otherwise | PASS |
| Server model validation | `validate_inputs`: xBi + xIn ≤ 1 (+1e-12 tolerance), 767 ≤ T ≤ 855 → HTTP 400 otherwise | PASS |
| Model evaluated | Frozen Poly D2 (`_evaluate_polynomial`) | PASS |
| Synthetic / residual / path-aware / external API | None in path | PASS |

The temperature range is enforced in `model.py`, not the pydantic schema. The API still rejects
out-of-range values (verified below), so this is a design note, not a defect.

Live `TestClient` checks (in memory, no server started):

| Request | Response |
|---------|----------|
| xBi 0.2509, xIn 0.4982, T 813 | 200, ΔmixH −1014.692 J/mol, xSn 0.2509, "Exothermic" |
| xBi 0.6, xIn 0.5, T 813 | 400 "Bi and In mole fractions must sum to at most 1" |
| T 700 / T 900 | 400 "Temperature must be between 767 K and 855 K." |
| xBi −0.1 | 422 (pydantic ge=0) |
| T 790 (in range, not a measured isotherm) | 200, −224.49 J/mol (continuous T accepted for `/predict`) |

## 5. Benchmark verification

Point: xBi = 0.2509, xIn = 0.4982, xSn = 0.2509, T = 813 K (this is experimental row `EXP_0051`,
equiatomic Bi/Sn cross-section, measured ΔmixH = −1017.0 J/mol).

| Quantity | Expected | Computed | Status |
|----------|---------:|---------:|--------|
| ML (backend, via `/predict`) | ≈ −1014.69 | −1014.6924 J/mol | PASS |
| ML (full-precision CSV) | — | −1014.5761 J/mol | (see §3 WARNING) |
| RKM (`src/rkm_model`) | ≈ −1028.55 | −1028.5461 J/mol | PASS |
| ML − RKM | ≈ +13.86 | +13.854 J/mol | PASS |
| Experiment | — | −1017.0 J/mol | ML error +2.31, RKM error −11.55 |

The benchmark values quoted in `web/README.md` (−1014.69 ML; ≈ −1028.55 RKM) and the default
inputs in `App.jsx` (0.2509 / 0.4982 / 813) agree with these results.

## 6. Surface verification

| Check | Finding | Status |
|-------|---------|--------|
| Model | `build_surface` → `predict_delta_mix_h_batch` (frozen Poly D2) | PASS |
| Grid | 0.01 step, xBi + xIn ≤ 1, 5151 points, xSn = 1 − xBi − xIn (max closure error 1.1 × 10⁻¹⁶) | PASS |
| Allowed temperatures | 767, 813, 855 K only; T = 800 → HTTP 400 | PASS |
| API values vs independent recomputation | max abs difference 0 at all three temperatures | PASS |
| Ranges | 767 K: −2198.01 … +313.62; 813 K: −2107.39 … +339.54; 855 K: −2021.78 … +374.81 J/mol | PASS (informational) |
| Overlay points | Experimental CSV only: 35 / 34 / 35 points at 767 / 813 / 855 K (104 total) | PASS |
| Synthetic data | Not loaded | PASS |

The minimum grid xSn is −1.1 × 10⁻¹⁶, which is floating-point round-off at the xSn = 0 edge, not
an invalid composition.

## 7. Comparison verification

| Check | Finding | Status |
|-------|---------|--------|
| ML side | Frozen Poly D2 (`predict_delta_mix_h_batch`) | PASS |
| RKM side | Fixed `src.rkm_model.rkm_delta_mix_h` (unchanged, temperature-independent benchmark) | PASS |
| Difference | `ml − rkm` elementwise; frontend label "ML − RKM" with correct sign interpretation | PASS |
| Modes | `ml`, `rkm`, `difference` all 200; invalid mode → 400; T = 800 → 400 | PASS |
| API vs recomputation (813 K) | max abs difference 0 for `delta_mix_H`, `delta_mix_H_ml`, `delta_mix_H_rkm`, `difference` | PASS |
| Statistics (813 K) | ML range −2107.39 … +339.54; RKM range −1654.25 … +134.73; mean ML−RKM 112.02; max ML−RKM 542.01 J/mol | PASS (recomputed identically) |
| Benchmark point in overlay | EXP_0051: experiment −1017.0, ML −1014.69, RKM −1028.55 → ML − RKM = +13.85 | PASS |
| Frontend caveat | "Full-simplex model comparison — not experimental validation." | PASS |

## 8. Synthetic-data dependency audit

| Location | Finding | Status |
|----------|---------|--------|
| `web/backend/*.py` | No reference to `synthetic_cross_sections.csv`, `synthetic_full_ternary.csv`, Dataset A/B, `generate_synthetic_data`, residual, path-aware, Model A/B | PASS |
| Backend data reads | Only `data/original_experimental_data.csv` (experimental overlay) | PASS |
| Coefficient provenance | `scripts/extract_final_model.py` reads only the experimental CSV; refit reproduces CSV to 2.3 × 10⁻¹³ | PASS |
| `scripts/build_final_surrogate.py` | Reads Dataset B only as an **evaluation grid** for plots/predictions; trains on experimental data only; not imported by the web app | PASS |
| `web/frontend/src` (prediction logic) | No synthetic data loaded or used | PASS |
| `web/frontend/src/ScientificBlocks.jsx` text | Methodology pipeline step reads **"Synthetic data — RKM-generated training data"** (line 297) | **WARNING** |
| `web/frontend/src/ScientificBlocks.jsx` text | Provenance step 03: "Synthetic data — Generated from the implemented RKM model for ML development." (lines 373–376) | **WARNING** (minor) |
| `README.md` | Describes Datasets A/B as RKM evaluations, not calorimetry, and warns against scoring on synthetic rows | PASS |

**Synthetic data in production prediction path: NONE.** The two WARNINGs are text only. Line 297
in particular says synthetic data are *training* data, which contradicts the final decision and
the actual deployed model (trained on 104 experimental rows only). Nearby website text ("trained
on 104 experimental calorimetry observations") is correct, so the site is internally inconsistent
at this one point.

## 9. Stale-model audit

| Search target | Finding | Status |
|---------------|---------|--------|
| Old/alternative coefficients in `web/` | None; only the established set appears (backend + frontend display) | PASS |
| Old metrics (e.g. 43.81, 54.64, random-forest / gradient-boosting values) in `web/` or `README.md` | None | PASS |
| Website metrics | LOCSO MAE 41.10, RMSE 56.34, R² 0.9712; RKM 57.13 / 73.49 / 0.9510 — match final reports | PASS |
| Residual Model A / path-aware Model B in `web/` | None | PASS |
| Model naming | "Frozen Polynomial Degree-2 surrogate (Step 4A)" in `web/README.md` and `model.py`; FastAPI description "104 experimental observations" | PASS |
| Benchmark values in docs | `web/README.md` −1014.69 / ≈ −1028.55 — correct | PASS |
| Pipeline description | "RKM-generated training data" (see §8) | **WARNING** |
| Other docs/reports | Benchmark values also appear in historical prediction CSVs and docs (`docs/understanding_rkm.md`, `docs/understanding_synthetic_data.md`); these are research records, not served by the website | PASS (not production) |

## 10. Test / build status

| Test | Result | Status |
|------|--------|--------|
| Backend import (`main`, `model`, `surface`, `comparison`, `src.rkm_model`) + FastAPI `TestClient` on all endpoints | Imported and responded as described above | PASS |
| `npm run build` (`web/frontend`) | Exit 0; Vite build succeeded. Only warning: bundle chunk > 500 kB (Plotly), performance only | PASS |
| `python scripts/validate_dataset.py` | "DATASET VALIDATION: PASS — Passed 65 checks." | PASS |
| Package files | Not modified (`git status` unchanged) | PASS |
| Tests | None created or modified | PASS |

## 11. Discrepancies

No **MISMATCH** found. The website, backend, displayed equation, benchmark, surface and comparison
all implement the experimental-only Direct Poly D2 final model. No fixes were made, per audit
instructions.

| # | Item | Classification | Impact |
|---|------|----------------|--------|
| D1 | Backend coefficients rounded to 6 d.p.; T² rounding gives a −0.10 to −0.13 J/mol shift vs full-precision CSV | WARNING | Negligible (≤ 0.13 J/mol; ≈ 0.3 % of LOCSO MAE). Not scientifically material |
| D2 | `ScientificBlocks.jsx` line 297: methodology pipeline says synthetic data are "RKM-generated training data" | WARNING | Documentation only; contradicts the final decision (synthetic data are exploratory only, not used for training) |
| D3 | `ScientificBlocks.jsx` lines 373–376: provenance step "Synthetic data — Generated from the implemented RKM model for ML development" placed in the pipeline before "ML surrogate" | WARNING (minor) | Documentation only; implies synthetic data feed the deployed model |
| D4 | `/predict` temperature range enforced in `model.py` rather than in the pydantic schema | PASS (design note) | None; out-of-range requests correctly return HTTP 400 |
| D5 | Vite chunk-size warning | PASS (informational) | Performance only |

**Overall scientific consistency: CONSISTENT.** The deployed prediction path matches the final
scientific model. The only discrepancies are a sub-0.13 J/mol coefficient-rounding effect and
two website text passages that overstate the role of synthetic data.
