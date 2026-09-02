# ML Baseline Results

## 1. Model
Linear Regression (`sklearn.linear_model.LinearRegression`)

## 2. Features
xBi, xIn, xSn

## 3. Target
ΔmixH — experimental: `integral_mixing_enthalpy_J_per_mol`; synthetic train: `delta_mix_H_J_per_mol`

## 4. Validation
Leave-one-cross-section-out (LOCSO) on the three experimental cross-sections.

## 5. Experimental-only results

### Fold 1 — held out: `(Sn0.33Bi0.67)1-xInx`
- Train N = 70, Test N = 34
- ML: MAE = 309.46 J/mol, RMSE = 374.22 J/mol, R² = -0.2267 (N = 34)
- RKM (same test points): MAE = 88.38 J/mol, RMSE = 105.90 J/mol, R² = 0.9018 (N = 34)

### Fold 2 — held out: `(Sn0.50Bi0.50)1-xInx`
- Train N = 70, Test N = 34
- ML: MAE = 240.25 J/mol, RMSE = 273.04 J/mol, R² = -0.0016 (N = 34)
- RKM (same test points): MAE = 40.98 J/mol, RMSE = 50.63 J/mol, R² = 0.9656 (N = 34)

### Fold 3 — held out: `(Sn0.67Bi0.33)1-xInx`
- Train N = 68, Test N = 36
- ML: MAE = 203.57 J/mol, RMSE = 230.65 J/mol, R² = -0.2965 (N = 36)
- RKM (same test points): MAE = 42.88 J/mol, RMSE = 50.88 J/mol, R² = 0.9369 (N = 36)

### Overall (all 104 out-of-fold predictions)
- ML: MAE = 250.18 J/mol, RMSE = 297.61 J/mol, R² = 0.1967 (N = 104)

## 6. Synthetic-only → experimental results

### Fold 1 — held out: `(Sn0.33Bi0.67)1-xInx`
- Train N = 4926 (Dataset A, excluding held-out section)
- Test N = 34 (experimental)
- ML: MAE = 292.55 J/mol, RMSE = 338.25 J/mol, R² = -0.0022 (N = 34)
- RKM (same test points): MAE = 88.38 J/mol, RMSE = 105.90 J/mol, R² = 0.9018 (N = 34)

### Fold 2 — held out: `(Sn0.50Bi0.50)1-xInx`
- Train N = 4926 (Dataset A, excluding held-out section)
- Test N = 34 (experimental)
- ML: MAE = 261.21 J/mol, RMSE = 297.71 J/mol, R² = -0.1907 (N = 34)
- RKM (same test points): MAE = 40.98 J/mol, RMSE = 50.63 J/mol, R² = 0.9656 (N = 34)

### Fold 3 — held out: `(Sn0.67Bi0.33)1-xInx`
- Train N = 4926 (Dataset A, excluding held-out section)
- Test N = 36 (experimental)
- ML: MAE = 253.26 J/mol, RMSE = 323.18 J/mol, R² = -1.5453 (N = 36)
- RKM (same test points): MAE = 42.88 J/mol, RMSE = 50.88 J/mol, R² = 0.9369 (N = 36)

### Overall (all 104 out-of-fold predictions)
- ML: MAE = 268.70 J/mol, RMSE = 320.21 J/mol, R² = 0.0701 (N = 104)

## 7. Combined experimental + synthetic results

### Fold 1 — held out: `(Sn0.33Bi0.67)1-xInx`
- Train N = 4996 (experiment + Dataset A, excluding held-out section)
- Test N = 34 (experimental)
- ML: MAE = 292.90 J/mol, RMSE = 338.85 J/mol, R² = -0.0058 (N = 34)
- RKM (same test points): MAE = 88.38 J/mol, RMSE = 105.90 J/mol, R² = 0.9018 (N = 34)

### Fold 2 — held out: `(Sn0.50Bi0.50)1-xInx`
- Train N = 4996 (experiment + Dataset A, excluding held-out section)
- Test N = 34 (experimental)
- ML: MAE = 262.85 J/mol, RMSE = 299.60 J/mol, R² = -0.2059 (N = 34)
- RKM (same test points): MAE = 40.98 J/mol, RMSE = 50.63 J/mol, R² = 0.9656 (N = 34)

### Fold 3 — held out: `(Sn0.67Bi0.33)1-xInx`
- Train N = 4994 (experiment + Dataset A, excluding held-out section)
- Test N = 36 (experimental)
- ML: MAE = 257.03 J/mol, RMSE = 328.91 J/mol, R² = -1.6363 (N = 36)
- RKM (same test points): MAE = 42.88 J/mol, RMSE = 50.88 J/mol, R² = 0.9369 (N = 36)

### Overall (all 104 out-of-fold predictions)
- ML: MAE = 270.66 J/mol, RMSE = 323.00 J/mol, R² = 0.0538 (N = 104)

## 8. RKM benchmark

### Fold 1 — held out: `(Sn0.33Bi0.67)1-xInx`
- RKM: MAE = 88.38 J/mol, RMSE = 105.90 J/mol, R² = 0.9018 (N = 34)

### Fold 2 — held out: `(Sn0.50Bi0.50)1-xInx`
- RKM: MAE = 40.98 J/mol, RMSE = 50.63 J/mol, R² = 0.9656 (N = 34)

### Fold 3 — held out: `(Sn0.67Bi0.33)1-xInx`
- RKM: MAE = 42.88 J/mol, RMSE = 50.88 J/mol, R² = 0.9369 (N = 36)

### Overall (all 104 experimental points, out-of-fold)
- RKM: MAE = 57.13 J/mol, RMSE = 73.49 J/mol, R² = 0.9510 (N = 104)

Reference validation on all 104 points: MAE = 57.13 J/mol, RMSE = 73.49 J/mol, R² = 0.9510
