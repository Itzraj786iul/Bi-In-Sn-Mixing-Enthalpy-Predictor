# Compositional Feature Representation Check

## Purpose
Test whether temperature-aware Poly D2 is sensitive to including redundant xSn
given xBi + xIn + xSn = 1.

## Setup
- Model: PolynomialFeatures (degree 2) + minimum-norm OLS
- Training: experimental-only LOCSO (3 folds)
- No synthetic data

## Representations
- **Model A (`composition_temperature_3vars`):** xBi, xIn, xSn, temperature_K
- **Model B (`composition_temperature_2vars`):** xBi, xIn, temperature_K (xSn omitted)

## Pooled metrics (104 OOF)

| Model | MAE | RMSE | R² | Poly input features |
| --- | ---: | ---: | ---: | ---: |
| Model A — 3 composition vars + T | 43.81 | 61.17 | 0.9661 | 15 |
| Model B — 2 composition vars + T | 41.10 | 56.34 | 0.9712 | 10 |
| RKM (benchmark) | 57.13 | 73.49 | 0.9510 | — |

Step 3F reference (Model A): MAE = 43.81, RMSE = 61.17, R² = 0.9661

## Fold-level — Model A (3 vars + T)

| Fold | Held-out section | Train N | Test N | Poly features | MAE | RMSE | R² |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `(Sn0.33Bi0.67)1-xInx` | 70 | 34 | 15 | 39.50 | 52.79 | 0.9756 |
| 2 | `(Sn0.50Bi0.50)1-xInx` | 70 | 34 | 15 | 39.13 | 53.32 | 0.9618 |
| 3 | `(Sn0.67Bi0.33)1-xInx` | 68 | 36 | 15 | 52.30 | 74.10 | 0.8662 |

**Pooled (104 OOF):** MAE = 43.81 J/mol, RMSE = 61.17 J/mol, R² = 0.9661 (N = 104)

## Fold-level — Model B (2 vars + T)

| Fold | Held-out section | Train N | Test N | Poly features | MAE | RMSE | R² |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `(Sn0.33Bi0.67)1-xInx` | 70 | 34 | 10 | 49.67 | 67.32 | 0.9603 |
| 2 | `(Sn0.50Bi0.50)1-xInx` | 70 | 34 | 10 | 37.35 | 49.00 | 0.9677 |
| 3 | `(Sn0.67Bi0.33)1-xInx` | 68 | 36 | 10 | 36.56 | 51.21 | 0.9361 |

**Pooled (104 OOF):** MAE = 41.10 J/mol, RMSE = 56.34 J/mol, R² = 0.9712 (N = 104)

## Metrics by temperature

| T (K) | N | Model A MAE | Model A RMSE | Model B MAE | Model B RMSE | RKM MAE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 767 | 35 | 50.20 | 66.24 | 46.36 | 59.84 | 72.77 |
| 813 | 34 | 43.58 | 59.66 | 41.26 | 55.33 | 51.14 |
| 855 | 35 | 37.65 | 57.19 | 35.69 | 53.65 | 47.32 |

## Conclusion

Removing xSn changes performance (ΔMAE = -2.71, ΔR² = +0.0051).

- Model A vs Step 3F: MAE Δ = +0.00, R² Δ = -0.0000
- Model B vs Model A: MAE Δ = -2.71, RMSE Δ = -4.82, R² Δ = +0.0051
