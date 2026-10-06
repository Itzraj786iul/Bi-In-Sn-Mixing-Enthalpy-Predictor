# Final Model Validation

## Final candidate model

> **Note:** This candidate was subsequently adopted as the final frozen Direct Polynomial Degree-2 model (features xBi, xIn, T; 104 experimental observations; primary validation LOCSO MAE 41.10 / RMSE 56.34 / R² 0.9712). The random 5-fold result below (plain `KFold`, MAE 30.34) is secondary. It differs slightly from the stratified 5-fold value (31.22) in `controlled_model_comparison.md` because the fold assignment differs. The RKM values are a physics-model reference on the same 104 points, not an out-of-sample score.

PolynomialFeatures(degree=2) + LinearRegression (minimum-norm OLS)

Features: xBi, xIn, temperature_K

Target: `integral_mixing_enthalpy_J_per_mol`

Data: 104 original experimental observations only.

## Validation design

### Primary validation: LOCSO (leave-one-cross-section-out)
Held-out entire composition cross-sections. Tests extrapolation along unseen paths.
This remains the **primary defensible metric** for this project.

Reference result (Step 3H, not rerun): MAE = 41.10, RMSE = 56.34, R² = 0.9712

### Secondary validation: random 5-fold CV
`KFold(n_splits=5, shuffle=True, random_state=42)`

Points on the same cross-section (and often neighbouring xIn values) can appear in both training and test folds. This is an **easier, interpolation-oriented** check.

## Secondary validation — pooled metrics (104 OOF)

- **Poly D2 (random 5-fold):** MAE = 30.34 J/mol, RMSE = 40.02 J/mol, R² = 0.9855 (N = 104)
- **RKM (same 104 points):** MAE = 57.13 J/mol, RMSE = 73.49 J/mol, R² = 0.9510 (N = 104)

## Fold-level metrics (random 5-fold)

| Fold | Train N | Test N | Poly D2 MAE | Poly D2 RMSE | Poly D2 R² | RKM MAE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 83 | 21 | 22.93 | 29.83 | 0.9908 | 57.60 |
| 2 | 83 | 21 | 31.75 | 36.77 | 0.9903 | 53.83 |
| 3 | 83 | 21 | 28.19 | 41.94 | 0.9875 | 71.24 |
| 4 | 83 | 21 | 40.90 | 52.33 | 0.9467 | 46.33 |
| 5 | 84 | 20 | 27.82 | 35.39 | 0.9865 | 56.65 |

**Pooled (104 OOF):** MAE = 30.34 J/mol, RMSE = 40.02 J/mol, R² = 0.9855 (N = 104)

## Comparison summary

| Validation | Model | MAE | RMSE | R² |
| --- | --- | ---: | ---: | ---: |
| Primary LOCSO | Poly D2 | 41.10 | 56.34 | 0.9712 |
| Secondary 5-fold | Poly D2 | 30.34 | 40.02 | 0.9855 |
| All 104 points | RKM | 57.13 | 73.49 | 0.9510 |

## Interpretation

Random 5-fold MAE (30.34) is **substantially better** than LOCSO (41.10; ΔMAE = -10.76).

A better 5-fold score **should be expected** when it occurs: neighbouring compositions on the same measured cross-section often leak between folds, so the model interpolates rather than extrapolates to a full unseen path. LOCSO is stricter and more aligned with the paper's experimental design.

RKM reference (frozen): MAE = 57.13, RMSE = 73.49, R² = 0.9510
