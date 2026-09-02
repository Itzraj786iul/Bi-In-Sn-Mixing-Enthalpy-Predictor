# Temperature-Aware Poly D2 with Synthetic Training

## Scientific question
Does Dataset A synthetic RKM data help a temperature-aware Poly D2 model predict experimental ΔmixH?

## Setup
- Model: PolynomialFeatures (degree 2) + minimum-norm OLS
- Features: xBi, xIn, xSn, temperature_K (`composition_temperature`)
- LOCSO on three cross-sections; Dataset A only (held-out section excluded from synthetic train)

## Pooled metrics (104 OOF)

| Model / training | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Temp-aware Poly D2 — experimental-only | 43.81 | 61.17 | 0.9661 |
| Temp-aware Poly D2 — synthetic-only | 57.35 | 82.79 | 0.9378 |
| Temp-aware Poly D2 — combined | 56.73 | 82.09 | 0.9389 |
| RKM (benchmark) | 57.13 | 73.49 | 0.9510 |

### Historical comparison (prior steps)

| Reference | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Step 3C comp-only experimental-only | 54.64 | 71.23 | 0.9540 |
| Step 3C comp-only synthetic-only | 57.26 | 82.61 | 0.9381 |
| Step 3C comp-only combined | 56.90 | 82.13 | 0.9388 |
| Step 3F temp-aware experimental-only | 43.81 | 61.17 | 0.9661 |

## Fold-level — experimental only

| Fold | Held-out section | Train N | Test N | Poly features | MAE | RMSE | R² | RKM MAE |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `(Sn0.33Bi0.67)1-xInx` | 70 | 34 | 15 | 39.50 | 52.79 | 0.9756 | 88.38 |
| 2 | `(Sn0.50Bi0.50)1-xInx` | 70 | 34 | 15 | 39.13 | 53.32 | 0.9618 | 40.98 |
| 3 | `(Sn0.67Bi0.33)1-xInx` | 68 | 36 | 15 | 52.30 | 74.10 | 0.8662 | 42.88 |

**Pooled (104 OOF):** MAE = 43.81 J/mol, RMSE = 61.17 J/mol, R² = 0.9661 (N = 104)

## Fold-level — synthetic only

| Fold | Held-out section | Train N | Test N | Poly features | MAE | RMSE | R² | RKM MAE |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `(Sn0.33Bi0.67)1-xInx` | 4926 | 34 | 15 | 71.43 | 88.25 | 0.9318 | 88.38 |
| 2 | `(Sn0.50Bi0.50)1-xInx` | 4926 | 34 | 15 | 49.60 | 61.29 | 0.9495 | 40.98 |
| 3 | `(Sn0.67Bi0.33)1-xInx` | 4926 | 36 | 15 | 51.36 | 94.34 | 0.7831 | 42.88 |

**Pooled (104 OOF):** MAE = 57.35 J/mol, RMSE = 82.79 J/mol, R² = 0.9378 (N = 104)

## Fold-level — combined

| Fold | Held-out section | Train N | Test N | Poly features | MAE | RMSE | R² | RKM MAE |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `(Sn0.33Bi0.67)1-xInx` | 4996 | 34 | 15 | 71.00 | 87.57 | 0.9328 | 88.38 |
| 2 | `(Sn0.50Bi0.50)1-xInx` | 4996 | 34 | 15 | 49.11 | 60.76 | 0.9504 | 40.98 |
| 3 | `(Sn0.67Bi0.33)1-xInx` | 4994 | 36 | 15 | 50.46 | 93.48 | 0.7871 | 42.88 |

**Pooled (104 OOF):** MAE = 56.73 J/mol, RMSE = 82.09 J/mol, R² = 0.9389 (N = 104)

## Metrics by temperature

| T (K) | N | Exp-only MAE | Syn-only MAE | Combined MAE | RKM MAE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 767 | 35 | 50.20 | 82.65 | 81.54 | 72.77 |
| 813 | 34 | 43.58 | 52.96 | 52.37 | 51.14 |
| 855 | 35 | 37.65 | 36.30 | 36.16 | 47.32 |

## Effect of Dataset A synthetic data

Synthetic targets are RKM-generated and **temperature-independent**; temperature is included only as an input feature.

| Training type | MAE | RMSE | R² | ΔMAE vs experimental-only |
| --- | ---: | ---: | ---: | ---: |
| Experimental-only | 43.81 | 61.17 | 0.9661 | +0.00 |
| Synthetic-only | 57.35 | 82.79 | 0.9378 | +13.53 |
| Combined | 56.73 | 82.09 | 0.9389 | +12.92 |

**Synthetic data makes the temperature-aware model worse** than experimental-only training.

## Step 3F reproduction check

Experimental-only: MAE = 43.81 (ref 43.81), RMSE = 61.17 (ref 61.17), R² = 0.9661 (ref 0.9661)
