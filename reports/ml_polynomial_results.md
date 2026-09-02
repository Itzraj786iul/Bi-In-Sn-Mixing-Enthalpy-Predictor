# ML Polynomial Baseline Results

## 1. Model
PolynomialFeatures (include_bias=True) + LinearRegression (fit_intercept=False)

No duplicate intercept column. OLS coefficients from minimum-norm least squares
(numpy lstsq) when the expanded design matrix is rank-deficient on compositional data.

## 2. Features
xBi, xIn, xSn (expanded to polynomial terms per degree)

## 3. Target
ΔmixH — experimental: `integral_mixing_enthalpy_J_per_mol`; synthetic train: `delta_mix_H_J_per_mol`

## 4. Validation
Leave-one-cross-section-out (LOCSO), same folds as Step 3B.

## Degree 2 results

### Experimental Only

#### Fold 1 — held out: `(Sn0.33Bi0.67)1-xInx`
- Train N = 70, Test N = 34, poly features = 10
- ML: MAE = 58.04 J/mol, RMSE = 69.98 J/mol, R² = 0.9571 (N = 34)
- RKM: MAE = 88.38 J/mol, RMSE = 105.90 J/mol, R² = 0.9018 (N = 34)

#### Fold 2 — held out: `(Sn0.50Bi0.50)1-xInx`
- Train N = 70, Test N = 34, poly features = 10
- ML: MAE = 48.87 J/mol, RMSE = 62.47 J/mol, R² = 0.9476 (N = 34)
- RKM: MAE = 40.98 J/mol, RMSE = 50.63 J/mol, R² = 0.9656 (N = 34)

#### Fold 3 — held out: `(Sn0.67Bi0.33)1-xInx`
- Train N = 68, Test N = 36, poly features = 10
- ML: MAE = 56.87 J/mol, RMSE = 79.66 J/mol, R² = 0.8453 (N = 36)
- RKM: MAE = 42.88 J/mol, RMSE = 50.88 J/mol, R² = 0.9369 (N = 36)

**Overall (104 OOF):** MAE = 54.64 J/mol, RMSE = 71.23 J/mol, R² = 0.9540 (N = 104)

### Synthetic Only

#### Fold 1 — held out: `(Sn0.33Bi0.67)1-xInx`
- Train N = 4926, Test N = 34, poly features = 10
- ML: MAE = 71.34 J/mol, RMSE = 87.88 J/mol, R² = 0.9324 (N = 34)
- RKM: MAE = 88.38 J/mol, RMSE = 105.90 J/mol, R² = 0.9018 (N = 34)

#### Fold 2 — held out: `(Sn0.50Bi0.50)1-xInx`
- Train N = 4926, Test N = 34, poly features = 10
- ML: MAE = 49.63 J/mol, RMSE = 61.29 J/mol, R² = 0.9495 (N = 34)
- RKM: MAE = 40.98 J/mol, RMSE = 50.63 J/mol, R² = 0.9656 (N = 34)

#### Fold 3 — held out: `(Sn0.67Bi0.33)1-xInx`
- Train N = 4926, Test N = 36, poly features = 10
- ML: MAE = 51.17 J/mol, RMSE = 94.21 J/mol, R² = 0.7837 (N = 36)
- RKM: MAE = 42.88 J/mol, RMSE = 50.88 J/mol, R² = 0.9369 (N = 36)

**Overall (104 OOF):** MAE = 57.26 J/mol, RMSE = 82.61 J/mol, R² = 0.9381 (N = 104)

### Combined

#### Fold 1 — held out: `(Sn0.33Bi0.67)1-xInx`
- Train N = 4996, Test N = 34, poly features = 10
- ML: MAE = 71.02 J/mol, RMSE = 87.49 J/mol, R² = 0.9330 (N = 34)
- RKM: MAE = 88.38 J/mol, RMSE = 105.90 J/mol, R² = 0.9018 (N = 34)

#### Fold 2 — held out: `(Sn0.50Bi0.50)1-xInx`
- Train N = 4996, Test N = 34, poly features = 10
- ML: MAE = 49.40 J/mol, RMSE = 61.06 J/mol, R² = 0.9499 (N = 34)
- RKM: MAE = 40.98 J/mol, RMSE = 50.63 J/mol, R² = 0.9656 (N = 34)

#### Fold 3 — held out: `(Sn0.67Bi0.33)1-xInx`
- Train N = 4994, Test N = 36, poly features = 10
- ML: MAE = 50.63 J/mol, RMSE = 93.46 J/mol, R² = 0.7871 (N = 36)
- RKM: MAE = 42.88 J/mol, RMSE = 50.88 J/mol, R² = 0.9369 (N = 36)

**Overall (104 OOF):** MAE = 56.90 J/mol, RMSE = 82.13 J/mol, R² = 0.9388 (N = 104)

## Degree 3 results

### Experimental Only

#### Fold 1 — held out: `(Sn0.33Bi0.67)1-xInx`
- Train N = 70, Test N = 34, poly features = 20
- ML: MAE = 61.10 J/mol, RMSE = 75.30 J/mol, R² = 0.9503 (N = 34)
- RKM: MAE = 88.38 J/mol, RMSE = 105.90 J/mol, R² = 0.9018 (N = 34)

#### Fold 2 — held out: `(Sn0.50Bi0.50)1-xInx`
- Train N = 70, Test N = 34, poly features = 20
- ML: MAE = 42.23 J/mol, RMSE = 52.63 J/mol, R² = 0.9628 (N = 34)
- RKM: MAE = 40.98 J/mol, RMSE = 50.63 J/mol, R² = 0.9656 (N = 34)

#### Fold 3 — held out: `(Sn0.67Bi0.33)1-xInx`
- Train N = 68, Test N = 36, poly features = 20
- ML: MAE = 68.35 J/mol, RMSE = 81.86 J/mol, R² = 0.8367 (N = 36)
- RKM: MAE = 42.88 J/mol, RMSE = 50.88 J/mol, R² = 0.9369 (N = 36)

**Overall (104 OOF):** MAE = 57.44 J/mol, RMSE = 71.26 J/mol, R² = 0.9539 (N = 104)

### Synthetic Only

#### Fold 1 — held out: `(Sn0.33Bi0.67)1-xInx`
- Train N = 4926, Test N = 34, poly features = 20
- ML: MAE = 99.61 J/mol, RMSE = 115.15 J/mol, R² = 0.8839 (N = 34)
- RKM: MAE = 88.38 J/mol, RMSE = 105.90 J/mol, R² = 0.9018 (N = 34)

#### Fold 2 — held out: `(Sn0.50Bi0.50)1-xInx`
- Train N = 4926, Test N = 34, poly features = 20
- ML: MAE = 48.82 J/mol, RMSE = 59.73 J/mol, R² = 0.9521 (N = 34)
- RKM: MAE = 40.98 J/mol, RMSE = 50.63 J/mol, R² = 0.9656 (N = 34)

#### Fold 3 — held out: `(Sn0.67Bi0.33)1-xInx`
- Train N = 4926, Test N = 36, poly features = 20
- ML: MAE = 67.15 J/mol, RMSE = 90.25 J/mol, R² = 0.8015 (N = 36)
- RKM: MAE = 42.88 J/mol, RMSE = 50.88 J/mol, R² = 0.9369 (N = 36)

**Overall (104 OOF):** MAE = 71.77 J/mol, RMSE = 91.22 J/mol, R² = 0.9245 (N = 104)

### Combined

#### Fold 1 — held out: `(Sn0.33Bi0.67)1-xInx`
- Train N = 4996, Test N = 34, poly features = 20
- ML: MAE = 98.15 J/mol, RMSE = 113.58 J/mol, R² = 0.8870 (N = 34)
- RKM: MAE = 88.38 J/mol, RMSE = 105.90 J/mol, R² = 0.9018 (N = 34)

#### Fold 2 — held out: `(Sn0.50Bi0.50)1-xInx`
- Train N = 4996, Test N = 34, poly features = 20
- ML: MAE = 48.36 J/mol, RMSE = 59.27 J/mol, R² = 0.9528 (N = 34)
- RKM: MAE = 40.98 J/mol, RMSE = 50.63 J/mol, R² = 0.9656 (N = 34)

#### Fold 3 — held out: `(Sn0.67Bi0.33)1-xInx`
- Train N = 4994, Test N = 36, poly features = 20
- ML: MAE = 67.39 J/mol, RMSE = 90.38 J/mol, R² = 0.8009 (N = 36)
- RKM: MAE = 42.88 J/mol, RMSE = 50.88 J/mol, R² = 0.9369 (N = 36)

**Overall (104 OOF):** MAE = 71.22 J/mol, RMSE = 90.52 J/mol, R² = 0.9257 (N = 104)

## Comparison summary (overall 104 OOF)

| Training type | Linear (Step 3B) | Poly deg 2 | Poly deg 3 | RKM |
| --- | --- | --- | --- | --- |
| experimental_only | MAE 250.2, R² 0.197 | MAE 54.6, R² 0.954 | MAE 57.4, R² 0.954 | MAE 57.13, R² 0.9510 |
| synthetic_only | MAE 268.7, R² 0.070 | MAE 57.3, R² 0.938 | MAE 71.8, R² 0.925 | — |
| combined | MAE 270.7, R² 0.054 | MAE 56.9, R² 0.939 | MAE 71.2, R² 0.926 | — |

RKM overall (all experiments): MAE = 57.13 J/mol, RMSE = 73.49 J/mol, R² = 0.9510

Reference: MAE = 57.13 J/mol, RMSE = 73.49 J/mol, R² = 0.9510
