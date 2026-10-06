# Polynomial Degree 2 — Temperature Feature Experiment

> **Historical note (superseded):** This is an intermediate experiment. Neither model here is the final model: the temperature model uses xBi, xIn, xSn, T (MAE 43.81). The final frozen Direct Poly D2 uses xBi, xIn, T and achieves LOCSO MAE 41.10 / RMSE 56.34 / R² 0.9712 (see `final_model_validation.md`). RKM rows are a physics-model reference with author-fitted parameters, not an out-of-sample LOCSO score.

## Scientific question
Does adding experimental temperature improve Poly D2 predictions of integral mixing enthalpy?

## Setup
- Model: PolynomialFeatures (degree 2) + minimum-norm OLS (same as Step 3C)
- Training: **experimental-only**, LOCSO (3 folds)
- No synthetic data, no hyperparameter tuning

## Feature sets compared
- `composition_only`: xBi, xIn, xSn
- `composition_temperature`: xBi, xIn, xSn, temperature_K

## Pooled metrics (104 OOF)

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Poly D2 composition only | 54.64 | 71.23 | 0.9540 |
| Poly D2 composition temperature | 43.81 | 61.17 | 0.9661 |
| RKM (benchmark) | 57.13 | 73.49 | 0.9510 |

Step 3C reference (composition-only): MAE = 54.64, RMSE = 71.23, R² = 0.9540

## Fold-level metrics — composition only

| Fold | Held-out section | Train N | Test N | Poly features | MAE | RMSE | R² |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `(Sn0.33Bi0.67)1-xInx` | 70 | 34 | 10 | 58.04 | 69.98 | 0.9571 |
| 2 | `(Sn0.50Bi0.50)1-xInx` | 70 | 34 | 10 | 48.87 | 62.47 | 0.9476 |
| 3 | `(Sn0.67Bi0.33)1-xInx` | 68 | 36 | 10 | 56.87 | 79.66 | 0.8453 |

**Pooled (104 OOF):** MAE = 54.64 J/mol, RMSE = 71.23 J/mol, R² = 0.9540 (N = 104)

## Fold-level metrics — composition temperature

| Fold | Held-out section | Train N | Test N | Poly features | MAE | RMSE | R² |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `(Sn0.33Bi0.67)1-xInx` | 70 | 34 | 15 | 39.50 | 52.79 | 0.9756 |
| 2 | `(Sn0.50Bi0.50)1-xInx` | 70 | 34 | 15 | 39.13 | 53.32 | 0.9618 |
| 3 | `(Sn0.67Bi0.33)1-xInx` | 68 | 36 | 15 | 52.30 | 74.10 | 0.8662 |

**Pooled (104 OOF):** MAE = 43.81 J/mol, RMSE = 61.17 J/mol, R² = 0.9661 (N = 104)

## Metrics by temperature

| T (K) | N | Comp-only MAE | Comp-only RMSE | Comp+T MAE | Comp+T RMSE | RKM MAE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 767 | 35 | 68.15 | 84.51 | 50.20 | 66.24 | 72.77 |
| 813 | 34 | 42.83 | 60.45 | 43.58 | 59.66 | 51.14 |
| 855 | 35 | 52.59 | 66.22 | 37.65 | 57.19 | 47.32 |

## Diagnostic comparison (Step 3E focus regions)

Residual = prediction − experimental ΔmixH.
Positive `abs_improvement` means composition+temperature has smaller absolute error.

### Bi-rich cross-section (N = 34)
- Composition-only MAE: 58.04 J/mol
- Composition+temperature MAE: 39.50 J/mol
- Temperature model better on 21/34 points
- Mean residual (comp): 30.74 → (temp): 35.54

### 767 K (N = 35)
- Composition-only MAE: 68.15 J/mol
- Composition+temperature MAE: 50.20 J/mol
- Temperature model better on 26/35 points
- Mean residual (comp): 60.06 → (temp): 21.27

### High xIn (xIn ≥ 0.70) (N = 49)
- Composition-only MAE: 35.05 J/mol
- Composition+temperature MAE: 20.12 J/mol
- Temperature model better on 33/49 points
- Mean residual (comp): 0.12 → (temp): 2.49

### Bi-rich + 767 K + high xIn (N = 5)
- Composition-only MAE: 69.69 J/mol
- Composition+temperature MAE: 32.82 J/mol
- Temperature model better on 5/5 points
- Mean residual (comp): 69.69 → (temp): 32.82
