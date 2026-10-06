# Final ML Surrogate Analysis (Dataset B)

## Important disclaimer
Dataset B was used only as an evaluation grid for the frozen model; it was not used for training. The final model is trained on the 104 experimental observations only.

The per-point prediction file `reports/final_surrogate_predictions.csv` (≈8.6 MB) is a generated artifact and is not tracked in Git; regenerate it with `python scripts/build_final_surrogate.py`.

Dataset B is **RKM-generated** and covers many compositions **without experimental measurements**.
ML predictions on Dataset B are **surrogate / model-exploration outputs only**.
They are **not experimentally validated** in unsampled ternary regions.

## Final frozen model
PolynomialFeatures(degree=2) + minimum-norm OLS

Features: xBi, xIn, temperature_K

Trained on: 104 original experimental observations only.

## Experimentally validated performance (104 calorimetry points only)

These metrics come from **prior validation** on experimental data. Dataset B is not used here.

### Primary validation — LOCSO (Step 3H, Poly D2 2-var + T)
- ML: MAE = 41.10 J/mol, RMSE = 56.34 J/mol, R² = 0.9712
- RKM: MAE = 57.13 J/mol, RMSE = 73.49 J/mol, R² = 0.9510

### Secondary validation — random 5-fold (Step 3J)
- ML: MAE = 30.34 J/mol, RMSE = 40.02 J/mol, R² = 0.9855

### LOCSO ML vs RKM on same 104 points (from Step 3H predictions file)

- ML: MAE = 41.10, RMSE = 56.34, R² = 0.9712
- RKM: MAE = 57.13, RMSE = 73.49, R² = 0.9510

## Dataset B surrogate predictions (NOT experimental validation)

- Rows predicted: **15453**
- ML prediction range: -2197.91 to 374.94 J/mol
- RKM prediction range: -1654.25 to 134.73 J/mol
- ML − RKM difference range: -656.81 to 375.14 J/mol
- Mean |ML − RKM|: 118.81 J/mol
- Maximum |ML − RKM|: 656.81 J/mol

### By temperature

| T (K) | N | ML min | ML max | RKM min | RKM max | Mean |ML−RKM| | Max |ML−RKM| |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 767 | 5151 | -2197.91 | 313.72 | -1654.25 | 134.73 | 144.91 | 656.81 |
| 813 | 5151 | -2107.28 | 339.66 | -1654.25 | 134.73 | 111.99 | 541.89 |
| 855 | 5151 | -2021.65 | 374.94 | -1654.25 | 134.73 | 99.52 | 434.65 |

### Largest |ML − RKM| compositions

| id | T (K) | xBi | xIn | xSn | region_type | ML | RKM | ML−RKM |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| SYN_B_04920 | 767 | 0.790 | 0.210 | 0.000 | unsampled_ternary_region | -1626.5 | -969.7 | -656.8 |
| SYN_B_04898 | 767 | 0.780 | 0.220 | 0.000 | unsampled_ternary_region | -1668.0 | -1011.5 | -656.5 |
| SYN_B_04941 | 767 | 0.800 | 0.200 | 0.000 | unsampled_ternary_region | -1583.3 | -927.0 | -656.4 |
| SYN_B_04875 | 767 | 0.770 | 0.230 | 0.000 | unsampled_ternary_region | -1708.0 | -1052.5 | -655.6 |
| SYN_B_04961 | 767 | 0.810 | 0.190 | 0.000 | unsampled_ternary_region | -1538.7 | -883.4 | -655.2 |
| SYN_B_04851 | 767 | 0.760 | 0.240 | 0.000 | unsampled_ternary_region | -1746.4 | -1092.4 | -654.0 |
| SYN_B_04980 | 767 | 0.820 | 0.180 | 0.000 | unsampled_ternary_region | -1492.4 | -839.2 | -653.2 |
| SYN_B_04826 | 767 | 0.750 | 0.250 | 0.000 | unsampled_ternary_region | -1783.3 | -1131.4 | -651.9 |
| SYN_B_04998 | 767 | 0.830 | 0.170 | 0.000 | unsampled_ternary_region | -1444.6 | -794.3 | -650.3 |
| SYN_B_04800 | 767 | 0.740 | 0.260 | 0.000 | unsampled_ternary_region | -1818.6 | -1169.2 | -649.3 |

## Figures (Cartesian xBi–xIn maps; surrogate exploration only)

- `figures/ml_surrogate_dmixH_767K.png`
- `figures/ml_surrogate_dmixH_813K.png`
- `figures/ml_surrogate_dmixH_855K.png`
- `figures/ml_surrogate_diff_rkm_813K.png`
