# ML Random Forest Baseline Results

## 1. Model
`sklearn.ensemble.RandomForestRegressor(n_estimators=200, random_state=42, max_features=1.0)`

No hyperparameter tuning.

## 2. Features
xBi, xIn, xSn

## 3. Target
ΔmixH — experimental: `integral_mixing_enthalpy_J_per_mol`; synthetic train: `delta_mix_H_J_per_mol`

## 4. Validation
Leave-one-cross-section-out (LOCSO), same folds as Steps 3B–3C.

## Experimental-only

### Fold 1 — held out: `(Sn0.33Bi0.67)1-xInx`
- Train N = 70, Test N = 34
- Random Forest: MAE = 181.83 J/mol, RMSE = 223.59 J/mol, R² = 0.5621 (N = 34)
- RKM: MAE = 88.38 J/mol, RMSE = 105.90 J/mol, R² = 0.9018 (N = 34)

### Fold 2 — held out: `(Sn0.50Bi0.50)1-xInx`
- Train N = 70, Test N = 34
- Random Forest: MAE = 69.40 J/mol, RMSE = 92.65 J/mol, R² = 0.8847 (N = 34)
- RKM: MAE = 40.98 J/mol, RMSE = 50.63 J/mol, R² = 0.9656 (N = 34)

### Fold 3 — held out: `(Sn0.67Bi0.33)1-xInx`
- Train N = 68, Test N = 36
- Random Forest: MAE = 141.85 J/mol, RMSE = 169.76 J/mol, R² = 0.2977 (N = 36)
- RKM: MAE = 42.88 J/mol, RMSE = 50.88 J/mol, R² = 0.9369 (N = 36)

**Overall (104 OOF):** MAE = 131.24 J/mol, RMSE = 170.67 J/mol, R² = 0.7358 (N = 104)

## Synthetic-only

### Fold 1 — held out: `(Sn0.33Bi0.67)1-xInx`
- Train N = 4926, Test N = 34
- Random Forest: MAE = 227.45 J/mol, RMSE = 264.82 J/mol, R² = 0.3857 (N = 34)
- RKM: MAE = 88.38 J/mol, RMSE = 105.90 J/mol, R² = 0.9018 (N = 34)

### Fold 2 — held out: `(Sn0.50Bi0.50)1-xInx`
- Train N = 4926, Test N = 34
- Random Forest: MAE = 98.08 J/mol, RMSE = 136.01 J/mol, R² = 0.7515 (N = 34)
- RKM: MAE = 40.98 J/mol, RMSE = 50.63 J/mol, R² = 0.9656 (N = 34)

### Fold 3 — held out: `(Sn0.67Bi0.33)1-xInx`
- Train N = 4926, Test N = 36
- Random Forest: MAE = 117.03 J/mol, RMSE = 140.77 J/mol, R² = 0.5171 (N = 36)
- RKM: MAE = 42.88 J/mol, RMSE = 50.88 J/mol, R² = 0.9369 (N = 36)

**Overall (104 OOF):** MAE = 146.93 J/mol, RMSE = 189.30 J/mol, R² = 0.6750 (N = 104)

## Combined

### Fold 1 — held out: `(Sn0.33Bi0.67)1-xInx`
- Train N = 4996, Test N = 34
- Random Forest: MAE = 226.17 J/mol, RMSE = 263.65 J/mol, R² = 0.3911 (N = 34)
- RKM: MAE = 88.38 J/mol, RMSE = 105.90 J/mol, R² = 0.9018 (N = 34)

### Fold 2 — held out: `(Sn0.50Bi0.50)1-xInx`
- Train N = 4996, Test N = 34
- Random Forest: MAE = 103.35 J/mol, RMSE = 140.22 J/mol, R² = 0.7358 (N = 34)
- RKM: MAE = 40.98 J/mol, RMSE = 50.63 J/mol, R² = 0.9656 (N = 34)

### Fold 3 — held out: `(Sn0.67Bi0.33)1-xInx`
- Train N = 4994, Test N = 36
- Random Forest: MAE = 116.19 J/mol, RMSE = 138.85 J/mol, R² = 0.5302 (N = 36)
- RKM: MAE = 42.88 J/mol, RMSE = 50.88 J/mol, R² = 0.9369 (N = 36)

**Overall (104 OOF):** MAE = 147.95 J/mol, RMSE = 189.28 J/mol, R² = 0.6751 (N = 104)

## Comparison summary (overall 104 OOF)

| Training type | Linear | Poly deg 2 | Poly deg 3 | Random Forest | RKM |
| --- | --- | --- | --- | --- | --- |
| experimental_only | MAE 250.2, R² 0.197 | MAE 54.6, R² 0.954 | MAE 57.4, R² 0.954 | MAE 131.2, R² 0.736 | MAE 57.13, R² 0.9510 |
| synthetic_only | MAE 268.7, R² 0.070 | MAE 57.3, R² 0.938 | MAE 71.8, R² 0.924 | MAE 146.9, R² 0.675 | — |
| combined | MAE 270.7, R² 0.054 | MAE 56.9, R² 0.939 | MAE 71.2, R² 0.926 | MAE 147.9, R² 0.675 | — |

RKM overall: MAE = 57.13 J/mol, RMSE = 73.49 J/mol, R² = 0.9510

Reference: MAE = 57.13 J/mol, RMSE = 73.49 J/mol, R² = 0.9510
