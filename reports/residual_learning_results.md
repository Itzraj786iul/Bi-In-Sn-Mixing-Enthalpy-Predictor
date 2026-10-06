# Residual Learning Experiment

Script: `scripts/step_4_residual_learning.py`. Per-observation output: `reports/residual_learning_results.csv`.

## Objective

Test whether a machine-learning model can learn the **systematic deviations of the experimental measurements from the RKM thermodynamic baseline**, using only real calorimetry. RKM-generated synthetic data are not independent experimental information (their targets are RKM evaluations), so this experiment learns the correction to RKM directly from the 104 measured points.

## Data

- `data/original_experimental_data.csv`: **104 real experimental observations** (Table III, Kumar, Mohan and Behera 2019).
- Three cross-sections × three temperatures (767, 813, 855 K).
- **No synthetic data were used in this experiment.** Dataset A and Dataset B were not loaded.
- RKM predictions come from the existing `src/rkm_model.rkm_delta_mix_h` (unchanged, no refit).

## Residual definition

```text
residual = ΔmixH_exp − ΔmixH_RKM
```

RKM is a fixed physics-based model (Table IV parameters), so computing its residual on every observation does not use any fold-specific fitting.

## Hybrid prediction

```text
residual_ML  = PolyD2(xBi, xIn, T)      # PolynomialFeatures(degree=2), minimum-norm least squares
ΔmixH_hybrid = ΔmixH_RKM + residual_ML
```

The residual model uses the same model family, features and fitting procedure as the current best direct baseline (Poly D2 on `xBi, xIn, temperature_K`; 10 polynomial terms).

## Validation protocol

Leave-one-cross-section-out (LOCSO), identical folds to the existing ML experiments. In each fold the entire held-out cross-section is excluded from fitting; the residual model is trained only on the other two sections and predicts residuals for the held-out section. The held-out experimental ΔmixH is never used for fitting that fold. Pooled metrics use all 104 out-of-fold predictions.

| Fold | Held-out section | Train sections | Train N | Test N |
| ---: | --- | --- | ---: | ---: |
| 1 | Bi-rich `(Sn0.33Bi0.67)1-xInx` | Equiatomic, Sn-rich | 70 | 34 |
| 2 | Equiatomic `(Sn0.50Bi0.50)1-xInx` | Bi-rich, Sn-rich | 70 | 34 |
| 3 | Sn-rich `(Sn0.67Bi0.33)1-xInx` | Bi-rich, Equiatomic | 68 | 36 |

## Results

### Pooled LOCSO (104 out-of-fold predictions)

| Model | MAE (J/mol) | RMSE (J/mol) | R² |
| --- | ---: | ---: | ---: |
| RKM | 57.13 | 73.49 | 0.9510 |
| RKM + Residual ML (Poly D2) | 44.66 | 55.75 | 0.9718 |

Reference (not recomputed here): existing direct Poly D2 on `xBi, xIn, T`, LOCSO — MAE 41.10, RMSE 56.34, R² 0.9712 (`reports/ml_composition_representation_results.md`).

### By held-out cross-section

| Cross-section | N | RKM MAE | Hybrid MAE | RKM RMSE | Hybrid RMSE | RKM R² | Hybrid R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bi-rich `(Sn0.33Bi0.67)1-xInx` | 34 | 88.38 | 54.74 | 105.90 | 68.56 | 0.9018 | 0.9588 |
| Equiatomic `(Sn0.50Bi0.50)1-xInx` | 34 | 40.98 | 23.95 | 50.63 | 29.93 | 0.9656 | 0.9880 |
| Sn-rich `(Sn0.67Bi0.33)1-xInx` | 36 | 42.88 | 54.70 | 50.88 | 60.78 | 0.9369 | 0.9100 |

### By temperature

| Temperature (K) | N | RKM MAE | Hybrid MAE | RKM RMSE | Hybrid RMSE | RKM R² | Hybrid R² |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 767 | 35 | 72.77 | 48.61 | 93.73 | 59.55 | 0.9239 | 0.9693 |
| 813 | 34 | 51.14 | 44.24 | 64.56 | 54.34 | 0.9621 | 0.9732 |
| 855 | 35 | 47.32 | 41.13 | 56.69 | 53.11 | 0.9683 | 0.9722 |

### Residual-model diagnostics

| Quantity | Value (J/mol) |
| --- | ---: |
| Mean experimental residual (exp − RKM) | -21.35 |
| Std of experimental residual | 70.66 |
| Mean predicted residual (OOF) | -11.53 |
| MAE of residual prediction | 44.66 |
| RMSE of residual prediction | 55.75 |
| Max absolute residual-prediction error | 175.65 |

Because ΔmixH_exp − ΔmixH_hybrid = residual − residual_ML, the residual-prediction MAE/RMSE equal the hybrid MAE/RMSE on ΔmixH.

## Interpretation

Pooled over all 104 out-of-fold predictions, RKM + residual ML **reduces the error relative to RKM**: MAE 57.13 → 44.66 J/mol (-12.47 J/mol (-21.8%)), RMSE 73.49 → 55.75 J/mol, R² 0.9510 → 0.9718.

- Bi-rich held out: MAE 88.38 → 54.74 J/mol (-33.64 J/mol (-38.1%)).
- Equiatomic held out: MAE 40.98 → 23.95 J/mol (-17.03 J/mol (-41.6%)).
- Sn-rich held out: MAE 42.88 → 54.70 J/mol (+11.82 J/mol (+27.6%)).
- 767 K: MAE 72.77 → 48.61 J/mol (-24.16 J/mol (-33.2%)).
- 813 K: MAE 51.14 → 44.24 J/mol (-6.90 J/mol (-13.5%)).
- 855 K: MAE 47.32 → 41.13 J/mol (-6.20 J/mol (-13.1%)).

Sections where the hybrid MAE is lower than RKM: Bi-rich, Equiatomic. Sections where it is not: Sn-rich.
Temperatures where the hybrid MAE is lower than RKM: 767 K, 813 K, 855 K. Temperatures where it is not: none.

Against the existing direct Poly D2 reference (MAE 41.10, RMSE 56.34, R² 0.9712), the hybrid has MAE +3.56 J/mol, RMSE -0.59 J/mol and R² +0.0006; these differences are small relative to the fold-to-fold variation and do not by themselves justify replacing the current model.

Caveats: each LOCSO fold trains the residual model on only two composition paths (68–70 points), and each test fold has 34–36 points, so fold-level differences carry considerable uncertainty. RKM itself is temperature-independent, so any temperature dependence in the hybrid comes entirely from the residual model. Improvements on the three measured cross-sections do not establish behaviour in unsampled regions of the ternary.
