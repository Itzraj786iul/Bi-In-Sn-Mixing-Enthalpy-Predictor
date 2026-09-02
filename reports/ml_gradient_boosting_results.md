# Gradient Boosting Baseline Results

## Model
`GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=2, random_state=42)`

## Features
xBi, xIn, temperature_K (experimental-only LOCSO)

## Pooled metrics (104 OOF)

- Gradient Boosting: MAE = 166.31 J/mol, RMSE = 219.06 J/mol, R² = 0.5648 (N = 104)
- Residual mean = -10.86, std = 218.79, max |residual| = 521.24
- RKM: MAE = 57.13 J/mol, RMSE = 73.49 J/mol, R² = 0.9510 (N = 104)

## Comparison with prior baselines (104 OOF, experimental-only)

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Linear Regression | 250.18 | 297.61 | 0.1967 |
| Poly D2 composition-only | 54.64 | 71.23 | 0.9540 |
| Poly D2 comp+T (3 vars) | 43.81 | 61.17 | 0.9661 |
| Poly D2 comp+T (2 vars) | 41.10 | 56.34 | 0.9712 |
| Random Forest | 131.24 | 170.67 | 0.7358 |
| RKM | 57.13 | 73.49 | 0.9510 |
| **Gradient Boosting (this step)** | **166.31** | **219.06** | **0.5648** |

**Versus best Poly D2 (MAE 41.10):** Gradient Boosting is **worse** than the current best Poly D2 baseline (ΔMAE = +125.21).

## Fold-level metrics

| Fold | Held-out section | Train N | Test N | GB MAE | GB RMSE | GB R² | RKM MAE |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `(Sn0.33Bi0.67)1-xInx` | 70 | 34 | 217.06 | 267.42 | 0.3736 | 88.38 |
| 2 | `(Sn0.50Bi0.50)1-xInx` | 70 | 34 | 110.03 | 165.84 | 0.6305 | 40.98 |
| 3 | `(Sn0.67Bi0.33)1-xInx` | 68 | 36 | 171.53 | 212.40 | -0.0994 | 42.88 |

**Pooled:** MAE = 166.31 J/mol, RMSE = 219.06 J/mol, R² = 0.5648 (N = 104)

## Metrics by temperature

| T (K) | N | GB MAE | GB RMSE | GB R² | RKM MAE | Poly D2 2-var MAE* |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 767 | 35 | 166.83 | 224.65 | 0.5628 | 72.77 | 46.36 |
| 813 | 34 | 176.35 | 231.94 | 0.5111 | 51.14 | 41.26 |
| 855 | 35 | 156.03 | 199.66 | 0.6074 | 47.32 | 35.69 |

*Poly D2 2-var MAE from Step 3H `ml_composition_representation_results.md`.

## Residual diagnostics

- Residual mean: -10.86 J/mol
- Residual std: 218.79 J/mol
- Max |residual|: 521.24 J/mol

### Largest absolute residuals

| experiment_id | cross_section | T (K) | xIn | experimental | prediction | residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| EXP_0004 | `(Sn0.33Bi0.67)1-xInx` | 767 | 0.4103 | -1346.0 | -824.8 | 521.2 |
| EXP_0094 | `(Sn0.67Bi0.33)1-xInx` | 855 | 0.2125 | -251.5 | -731.1 | -479.6 |
| EXP_0005 | `(Sn0.33Bi0.67)1-xInx` | 767 | 0.4942 | -1413.0 | -940.1 | 472.9 |
| EXP_0003 | `(Sn0.33Bi0.67)1-xInx` | 767 | 0.3081 | -1142.0 | -678.0 | 464.0 |
| EXP_0082 | `(Sn0.67Bi0.33)1-xInx` | 813 | 0.2163 | -283.3 | -745.6 | -462.3 |

## Regional diagnostics (Step 3E focus)

### Bi-rich fold (fold 1) (N = 34)
- MAE = 217.06, RMSE = 267.42, R² = 0.3736
- Residual mean = 209.15, std = 166.65

### High xIn (xIn ≥ 0.70) (N = 49)
- MAE = 66.26, RMSE = 91.65, R² = 0.8496
- Residual mean = -5.39, std = 91.49

### 767 K (N = 35)
- MAE = 166.83, RMSE = 224.65, R² = 0.5628
- Residual mean = 20.22, std = 223.73

### 855 K (N = 35)
- MAE = 156.03, RMSE = 199.66, R² = 0.6074
- Residual mean = -37.46, std = 196.11

### Bi-rich + high xIn (N = 16)
- MAE = 90.25, RMSE = 118.26, R² = 0.7475
- Residual mean = 73.44, std = 92.69
