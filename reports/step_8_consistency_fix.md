# Step 8 — Final-Model Consistency Fix

Surgical correction of the three issues found by the Step 7 audit
(`reports/final_model_website_audit.md`). No retraining, refitting, architecture change, RKM
change, dataset change, validation-method change, API change, or layout redesign.

## 1. Coefficient precision correction

| Item | Value |
|------|-------|
| File | `web/backend/model.py` |
| Constant | `COEF_T2` (temperature_K² term) |
| Before | `0.000974` (rounded to 6 decimals) |
| After | `0.0009741759040480247` |
| Source | `reports/final_model_coefficients.csv`, row `temperature_K^2`, column `coefficient`. The raw CSV string was copied exactly; it was not reconstructed or rounded. |
| Exactness check | `model.COEF_T2 == float("0.0009741759040480247")` → True; also equals the CSV read with `float_precision="round_trip"` |

The other nine coefficients were left exactly as they were. Their 6-decimal rounding errors are at most 4.2 × 10⁻⁷ each. Across all 104 experimental rows, the backend now agrees with a full-precision evaluation of the CSV to within 9.2 × 10⁻⁵ J/mol; before the fix the gap was up to 0.129 J/mol.

The model is still the frozen, experimental-only Direct Polynomial Degree-2 in xBi, xIn and T, with 10 terms and xSn = 1 − xBi − xIn as a derived quantity.

## 2. Benchmark before / after

Point: xBi = 0.2509, xIn = 0.4982, xSn = 0.2509, T = 813 K (experimental row EXP_0051, measured ΔmixH = −1017.0 J/mol).

| Quantity | Before (rounded T²) | After (full-precision T²) | Change |
|----------|-------------------:|-------------------------:|-------:|
| ML backend (`POST /predict`) | −1014.6924 J/mol | −1014.5761 J/mol | +0.1163 |
| Independent full-precision calculation from CSV | −1014.5761 J/mol | −1014.5761 J/mol | — |
| Backend − independent | −0.1163 J/mol | −3.9 × 10⁻⁵ J/mol | — |
| RKM (`src/rkm_model`, unchanged) | −1028.5461 J/mol | −1028.5461 J/mol | 0 |
| ML − RKM | +13.854 J/mol | +13.970 J/mol | +0.116 |

The prediction moved by about 0.1 J/mol, as expected. The `/comparison` statistics at 813 K shifted by the same order: mean |ML − RKM| went from 112.02 to 111.99 J/mol, and max |ML − RKM| from 542.01 to 541.89 J/mol. Training-set MAE went from 27.10 to 27.09 J/mol.

The reported scientific metrics were not changed. They come from the LOCSO evaluation in the research scripts, not from the web backend, and the website still shows:

- ML (LOCSO): MAE 41.10 J/mol, RMSE 56.34 J/mol, R² 0.9712
- RKM: MAE 57.13 J/mol, RMSE 73.49 J/mol, R² 0.9510

## 3. Synthetic-data wording corrections

File: `web/frontend/src/ScientificBlocks.jsx`. Only text and data arrays changed. The existing components and CSS classes were reused, with no layout or style changes.

### 3a. Methodology pipeline (`MethodologySection`)

Before:

```text
Experiment → Thermodynamic model → Synthetic data ("RKM-generated training data") → Machine learning → Validation
```

After:

```text
Experiment (104 calorimetry observations) → Data reconstruction (sign and composition checks)
→ Physics benchmark (Redlich-Kister-Muggianu, fixed) → Machine learning (Direct Polynomial Degree-2,
experimental data only) → Validation (Leave-One-Cross-Section-Out)
```

A second `methodology-note` paragraph was added:

> Synthetic datasets generated from the RKM model were evaluated during the research as an
> augmentation strategy. Under LOCSO, RKM-derived augmentation did not improve prediction of the
> real experimental observations, so synthetic data are not used to train the final predictor.
> They remain useful for exploratory visualization, sensitivity, and hypothesis analysis.

### 3b. Research / provenance workflow (`PROVENANCE_STEPS`, formerly lines 373–376)

The final workflow no longer contains a synthetic-data step:

```text
01 Experimental calorimetry (104 real observations)
02 Data cleaning / reconstruction
03 RKM physics benchmark (fixed)
04 Direct Polynomial Degree-2 (experimental data only)
05 Experimental validation (LOCSO)
```

Synthetic data now appear in a separate block, headed "Synthetic-data experiment (not used by the final predictor)", which uses the same workflow component:

```text
S1 RKM-derived synthetic datasets
S2 Augmentation / sensitivity experiment (evaluated under LOCSO)
S3 Not beneficial for final training
S4 Retained for exploratory analysis
```

These statements match `reports/synthetic_data_ablation.md`. Real + Dataset A had a pooled LOCSO MAE of 54.32 J/mol, compared with 41.10 J/mol for real data only. The wording does not claim that synthetic data are inherently bad.

## 4. Validation status

| Test | Result |
|------|--------|
| `python scripts/validate_dataset.py` | DATASET VALIDATION: PASS, 65 checks passed |
| `POST /predict`, valid inputs (benchmark; 0/0 at 767 K; 0.5/0.5 at 855 K; T = 790 K) | 200 |
| `POST /predict`, xBi + xIn > 1 | 400 |
| `POST /predict`, xBi < 0 or xIn > 1 | 422 |
| `POST /predict`, T = 766.9 K or 900 K | 400 |
| `GET /surface` at 767, 813, 855 K | 200; 5151 grid points; API equals model exactly; within 1.9 × 10⁻⁴ J/mol of the full-precision CSV |
| `GET /surface` at T = 800 K | 400 |
| `GET /comparison`, modes ml / rkm / difference | 200; difference = ML − RKM exactly |
| `GET /comparison` with T = 800 K or an invalid mode | 400 |

## 5. Build status

`npm run build` in `web/frontend` exits 0 (43 modules, built in 40 s). The only warning is the existing chunk-size warning for the Plotly bundle. Output goes to the git-ignored `dist/` folder.

## 6. Scientific consistency check

| Check | Status |
|-------|--------|
| A. Backend uses the full-precision T² from the final coefficients CSV; the other nine are unchanged as instructed | PASS |
| B. Prediction still uses only the experimental-trained Direct Poly D2 (xBi, xIn, T) | PASS |
| C. Production backend loads no synthetic data | PASS |
| D. Website describes synthetic data as exploratory and ablation data, not final training data | PASS |
| E. RKM is unchanged and remains the fixed physics benchmark | PASS |
| F. LOCSO remains the primary validation; website metrics unchanged | PASS |

## 7. Confirmation

No model architecture, feature set, training data, coefficient value other than the T² precision, RKM implementation, dataset, validation method, API endpoint, page layout, or scientific conclusion was changed.

Changed files:

- `web/backend/model.py`: T² precision only
- `web/frontend/src/ScientificBlocks.jsx`: two wording and workflow corrections

New file:

- `reports/step_8_consistency_fix.md`: this report

## 8. Remaining items not fixed (out of scope)

- `web/README.md` still quotes the old rounded benchmark of −1014.69 J/mol; the backend now returns −1014.58 J/mol.
- The fitted equation displayed on the website shows every coefficient to 6 decimals, including T² as 0.000974. This affects display only.
- The root `README.md` pipeline is from before the ML work: it says ML is "not started" and shows synthetic data feeding a future model.
