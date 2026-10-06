# ML Model Diagnostics

> **Historical note (superseded):** These diagnostics cover the earlier composition-only models (e.g. Polynomial Degree 2 without temperature, MAE 54.64), as do the `ml_diag_*` figures. They do not describe the final frozen Direct Poly D2 (xBi, xIn, T; LOCSO MAE 41.10 / RMSE 56.34 / R² 0.9712). RKM rows are a physics-model reference with author-fitted parameters, not an out-of-sample LOCSO score.

Analysis of **104 out-of-fold experimental predictions** from LOCSO
(`experimental_only` training setup). No new models were trained.

Residual = prediction − experimental ΔmixH.

## Overall metrics

| Model | MAE | RMSE | R² | Residual mean | Residual std | Max |residual| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RKM | 57.13 | 73.49 | 0.9510 | 21.35 | 70.32 | 213.27 |
| Linear Regression | 250.18 | 297.61 | 0.1967 | 22.27 | 296.77 | 701.97 |
| Polynomial Degree 2 | 54.64 | 71.23 | 0.9540 | 17.57 | 69.03 | 251.42 |
| Polynomial Degree 3 | 57.44 | 71.26 | 0.9539 | 11.71 | 70.29 | 187.35 |
| Random Forest | 131.24 | 170.67 | 0.7358 | -18.56 | 169.65 | 434.86 |

## RKM

### MAE / RMSE by cross-section

| Cross-section | N | MAE | RMSE |
| --- | ---: | ---: | ---: |
| `(Sn0.33Bi0.67)1-xInx` | 34 | 88.38 | 105.90 |
| `(Sn0.50Bi0.50)1-xInx` | 34 | 40.98 | 50.63 |
| `(Sn0.67Bi0.33)1-xInx` | 36 | 42.88 | 50.88 |

**Worst cross-section (MAE):** `(Sn0.33Bi0.67)1-xInx`

### MAE / RMSE by temperature

| T (K) | N | MAE | RMSE |
| --- | ---: | ---: | ---: |
| 767 | 35 | 72.77 | 93.73 |
| 813 | 34 | 51.14 | 64.56 |
| 855 | 35 | 47.32 | 56.69 |

**Worst temperature (MAE):** 767 K

### Largest absolute residuals

| experiment_id | cross_section | T (K) | xIn | experimental | prediction | residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| EXP_0009 | `(Sn0.33Bi0.67)1-xInx` | 767 | 0.8552 | -792.5 | -579.2 | 213.3 |
| EXP_0008 | `(Sn0.33Bi0.67)1-xInx` | 767 | 0.8075 | -959.0 | -750.8 | 208.2 |
| EXP_0010 | `(Sn0.33Bi0.67)1-xInx` | 767 | 0.8844 | -656.3 | -467.6 | 188.7 |
| EXP_0007 | `(Sn0.33Bi0.67)1-xInx` | 767 | 0.7103 | -1227.0 | -1043.7 | 183.3 |
| EXP_0011 | `(Sn0.33Bi0.67)1-xInx` | 767 | 0.9037 | -565.5 | -391.6 | 173.9 |

## Linear Regression

### MAE / RMSE by cross-section

| Cross-section | N | MAE | RMSE |
| --- | ---: | ---: | ---: |
| `(Sn0.33Bi0.67)1-xInx` | 34 | 309.46 | 374.22 |
| `(Sn0.50Bi0.50)1-xInx` | 34 | 240.25 | 273.04 |
| `(Sn0.67Bi0.33)1-xInx` | 36 | 203.57 | 230.65 |

**Worst cross-section (MAE):** `(Sn0.33Bi0.67)1-xInx`

### MAE / RMSE by temperature

| T (K) | N | MAE | RMSE |
| --- | ---: | ---: | ---: |
| 767 | 35 | 250.58 | 301.68 |
| 813 | 34 | 249.76 | 297.44 |
| 855 | 35 | 250.20 | 293.64 |

**Worst temperature (MAE):** 767 K

### Largest absolute residuals

| experiment_id | cross_section | T (K) | xIn | experimental | prediction | residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| EXP_0023 | `(Sn0.33Bi0.67)1-xInx` | 855 | 0.0999 | -295.1 | -997.1 | -702.0 |
| EXP_0012 | `(Sn0.33Bi0.67)1-xInx` | 813 | 0.1035 | -343.5 | -995.5 | -652.0 |
| EXP_0006 | `(Sn0.33Bi0.67)1-xInx` | 767 | 0.6026 | -1389.0 | -771.2 | 617.8 |
| EXP_0001 | `(Sn0.33Bi0.67)1-xInx` | 767 | 0.0980 | -380.4 | -997.9 | -617.5 |
| EXP_0005 | `(Sn0.33Bi0.67)1-xInx` | 767 | 0.4942 | -1413.0 | -819.9 | 593.1 |

## Polynomial Degree 2

### MAE / RMSE by cross-section

| Cross-section | N | MAE | RMSE |
| --- | ---: | ---: | ---: |
| `(Sn0.33Bi0.67)1-xInx` | 34 | 58.04 | 69.98 |
| `(Sn0.50Bi0.50)1-xInx` | 34 | 48.87 | 62.47 |
| `(Sn0.67Bi0.33)1-xInx` | 36 | 56.87 | 79.66 |

**Worst cross-section (MAE):** `(Sn0.33Bi0.67)1-xInx`

### MAE / RMSE by temperature

| T (K) | N | MAE | RMSE |
| --- | ---: | ---: | ---: |
| 767 | 35 | 68.15 | 84.51 |
| 813 | 34 | 42.83 | 60.45 |
| 855 | 35 | 52.59 | 66.22 |

**Worst temperature (MAE):** 767 K

### Largest absolute residuals

| experiment_id | cross_section | T (K) | xIn | experimental | prediction | residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| EXP_0069 | `(Sn0.67Bi0.33)1-xInx` | 767 | 0.0964 | -105.2 | 146.2 | 251.4 |
| EXP_0081 | `(Sn0.67Bi0.33)1-xInx` | 813 | 0.1062 | -92.7 | 112.9 | 205.6 |
| EXP_0001 | `(Sn0.33Bi0.67)1-xInx` | 767 | 0.0980 | -380.4 | -196.9 | 183.5 |
| EXP_0093 | `(Sn0.67Bi0.33)1-xInx` | 855 | 0.1047 | -64.3 | 117.9 | 182.2 |
| EXP_0060 | `(Sn0.50Bi0.50)1-xInx` | 855 | 0.2988 | -632.3 | -798.0 | -165.7 |

## Polynomial Degree 3

### MAE / RMSE by cross-section

| Cross-section | N | MAE | RMSE |
| --- | ---: | ---: | ---: |
| `(Sn0.33Bi0.67)1-xInx` | 34 | 61.10 | 75.30 |
| `(Sn0.50Bi0.50)1-xInx` | 34 | 42.23 | 52.63 |
| `(Sn0.67Bi0.33)1-xInx` | 36 | 68.35 | 81.86 |

**Worst cross-section (MAE):** `(Sn0.67Bi0.33)1-xInx`

### MAE / RMSE by temperature

| T (K) | N | MAE | RMSE |
| --- | ---: | ---: | ---: |
| 767 | 35 | 62.94 | 81.12 |
| 813 | 34 | 48.27 | 61.41 |
| 855 | 35 | 60.85 | 69.62 |

**Worst temperature (MAE):** 767 K

### Largest absolute residuals

| experiment_id | cross_section | T (K) | xIn | experimental | prediction | residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| EXP_0069 | `(Sn0.67Bi0.33)1-xInx` | 767 | 0.0964 | -105.2 | 82.1 | 187.3 |
| EXP_0003 | `(Sn0.33Bi0.67)1-xInx` | 767 | 0.3081 | -1142.0 | -967.7 | 174.3 |
| EXP_0081 | `(Sn0.67Bi0.33)1-xInx` | 813 | 0.1062 | -92.7 | 61.6 | 154.4 |
| EXP_0002 | `(Sn0.33Bi0.67)1-xInx` | 767 | 0.2133 | -841.6 | -689.5 | 152.1 |
| EXP_0004 | `(Sn0.33Bi0.67)1-xInx` | 767 | 0.4103 | -1346.0 | -1195.8 | 150.2 |

## Random Forest

### MAE / RMSE by cross-section

| Cross-section | N | MAE | RMSE |
| --- | ---: | ---: | ---: |
| `(Sn0.33Bi0.67)1-xInx` | 34 | 181.83 | 223.59 |
| `(Sn0.50Bi0.50)1-xInx` | 34 | 69.40 | 92.65 |
| `(Sn0.67Bi0.33)1-xInx` | 36 | 141.85 | 169.76 |

**Worst cross-section (MAE):** `(Sn0.33Bi0.67)1-xInx`

### MAE / RMSE by temperature

| T (K) | N | MAE | RMSE |
| --- | ---: | ---: | ---: |
| 767 | 35 | 129.41 | 171.89 |
| 813 | 34 | 126.91 | 171.36 |
| 855 | 35 | 137.26 | 168.75 |

**Worst temperature (MAE):** 855 K

### Largest absolute residuals

| experiment_id | cross_section | T (K) | xIn | experimental | prediction | residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| EXP_0005 | `(Sn0.33Bi0.67)1-xInx` | 767 | 0.4942 | -1413.0 | -978.1 | 434.9 |
| EXP_0004 | `(Sn0.33Bi0.67)1-xInx` | 767 | 0.4103 | -1346.0 | -942.8 | 403.2 |
| EXP_0094 | `(Sn0.67Bi0.33)1-xInx` | 855 | 0.2125 | -251.5 | -646.5 | -395.0 |
| EXP_0016 | `(Sn0.33Bi0.67)1-xInx` | 813 | 0.5006 | -1366.0 | -978.1 | 387.9 |
| EXP_0006 | `(Sn0.33Bi0.67)1-xInx` | 767 | 0.6026 | -1389.0 | -1010.0 | 379.0 |

## Polynomial Degree 2 vs RKM

- Polynomial D2 has lower absolute residual on **56/104** points (53.8%).

### By cross-section

| Cross-section | N | Poly2 MAE | RKM MAE | Poly2 better (count) |
| --- | ---: | ---: | ---: | ---: |
| `(Sn0.33Bi0.67)1-xInx` | 34 | 58.04 | 88.38 | 21/34 |
| `(Sn0.50Bi0.50)1-xInx` | 34 | 48.87 | 40.98 | 16/34 |
| `(Sn0.67Bi0.33)1-xInx` | 36 | 56.87 | 42.88 | 19/36 |

### By temperature

| T (K) | N | Poly2 MAE | RKM MAE | Poly2 better (count) |
| --- | ---: | ---: | ---: | ---: |
| 767 | 35 | 68.15 | 72.77 | 20/35 |
| 813 | 34 | 42.83 | 51.14 | 18/34 |
| 855 | 35 | 52.59 | 47.32 | 18/35 |

### By LOCSO fold (held-out section)

| Fold | Held-out section | Poly2 MAE | RKM MAE | Poly2 better (count) |
| ---: | --- | ---: | ---: | ---: |
| 1 | `(Sn0.33Bi0.67)1-xInx` | 58.04 | 88.38 | 21/34 |
| 2 | `(Sn0.50Bi0.50)1-xInx` | 48.87 | 40.98 | 16/34 |
| 3 | `(Sn0.67Bi0.33)1-xInx` | 56.87 | 42.88 | 19/36 |

## Figures

- `figures/ml_diag_pred_vs_exp_rkm.png`
- `figures/ml_diag_pred_vs_exp_poly2.png`
- `figures/ml_diag_pred_vs_exp_rf.png`
- `figures/ml_diag_residual_vs_xin_poly2.png`
- `figures/ml_diag_residual_vs_xin_rkm.png`
