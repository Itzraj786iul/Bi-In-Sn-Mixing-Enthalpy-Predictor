# Final ML Surface Analysis (Step 4B)

## Scope
Response-surface analysis of the **frozen** Poly D2 model (Step 4A coefficients).
No refitting. Coefficients are **empirical ML weights**, not RKM thermodynamic parameters.

## Primary validated performance (unchanged)
- LOCSO OOF: MAE = 41.10, RMSE = 56.34, R² = 0.9712
- RKM benchmark: MAE = 57.13, RMSE = 73.49, R² = 0.9510

## A. Composition response

Ternary grid: xBi ≥ 0, xIn ≥ 0, xBi+xIn ≤ 1, step = 0.01.

The **xBi·xIn** term (−9182.903 J/mol per unit product) modulates curvature between Bi and In.
Its **contribution at a composition** is coefficient × xBi × xIn (not the raw coefficient alone).
Raw coefficient magnitude must **not** be read as physical importance (different variable scales).

| T (K) | ML min | ML max | xBi·xIn term min | xBi·xIn term max |
| ---: | ---: | ---: | ---: | ---: |
| 767 | -2197.91 | 313.72 | -2295.73 | -0.00 |
| 813 | -2107.28 | 339.66 | -2295.73 | -0.00 |
| 855 | -2021.65 | 374.94 | -2295.73 | -0.00 |

## B. Temperature response

- **767 K:** ML range -2197.91 to 313.72 J/mol; mean temperature-term contribution 102.13 J/mol
- **813 K:** ML range -2107.28 to 339.66 J/mol; mean temperature-term contribution 144.69 J/mol
- **855 K:** ML range -2021.65 to 374.94 J/mol; mean temperature-term contribution 187.14 J/mol
- Mean ML change 767→855 K (same compositions): 85.01 J/mol; max |change| = 261.51 J/mol

The model captures **temperature-dependent variation present in the 104 calorimetry points**. This is **not** a claim of fundamental thermodynamic T-dependence discovered by ML.

## C. Experimental-region analysis

104 experiments span: xBi 0.031–0.602, xIn 0.095–0.907, T 767–855 K.
Experimental ΔmixH: -1413.0 to -64.3 J/mol (all exothermic).

Near-experimental grid points: within 0.03 in (xBi,xIn) of a measured point at the same T.

### 767 K — near-measured region (N = 759)
- ML range: -1563.35 to 67.23 J/mol
- ML mean: -758.42 J/mol
- Fraction ML > 0: 1.3%

### 813 K — near-measured region (N = 759)
- ML range: -1499.40 to 45.89 J/mol
- ML mean: -720.65 J/mol
- Fraction ML > 0: 1.4%

### 855 K — near-measured region (N = 750)
- ML range: -1426.60 to 59.48 J/mol
- ML mean: -676.50 J/mol
- Fraction ML > 0: 1.9%

- ML on the **104 experimental rows** (in-sample check): range -1411.09 to -17.61 J/mol

## D. RKM comparison at 813 K

- ML range: -2107.28 to 339.66 J/mol
- RKM range: -1654.25 to 134.73 J/mol
- ML−RKM range: -541.89 to 363.62 J/mol
- Mean |ML−RKM|: 111.99 J/mol
- Max |ML−RKM|: 541.89 J/mol

### Largest |ML−RKM| at 813 K

| xBi | xIn | xSn | region | ML | RKM | ML−RKM |
| ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 0.780 | 0.220 | -0.000 | unsampled_extrapolation | -1553.4 | -1011.5 | -541.9 |
| 0.770 | 0.230 | -0.000 | unsampled_extrapolation | -1594.3 | -1052.5 | -541.9 |
| 0.790 | 0.210 | -0.000 | unsampled_extrapolation | -1510.9 | -969.7 | -541.3 |
| 0.760 | 0.240 | 0.000 | unsampled_extrapolation | -1633.7 | -1092.4 | -541.2 |
| 0.750 | 0.250 | 0.000 | unsampled_extrapolation | -1671.4 | -1131.4 | -540.0 |
| 0.800 | 0.200 | -0.000 | unsampled_extrapolation | -1466.9 | -927.0 | -539.9 |
| 0.740 | 0.260 | 0.000 | unsampled_extrapolation | -1707.6 | -1169.2 | -538.4 |
| 0.810 | 0.190 | -0.000 | unsampled_extrapolation | -1421.3 | -883.4 | -537.8 |

## E. Physical sanity checks

- At 813 K, **572** simplex grid points (11.1%) have ML ΔmixH > 0.
- Of those, **11** overlap the near-measured region — some positive predictions occur near experiments.
- Extreme ML values occur primarily at **unsampled** compositions (especially Bi–In rich, low xSn).
- **Extrapolation warning:** predictions far from the three measured cross-sections are **not experimentally validated**.
- ML is **not** claimed physically superior to RKM; RKM remains the physics-based benchmark.

## Figures

- `figures/final_surface_767K.png`
- `figures/final_surface_813K.png`
- `figures/final_surface_855K.png`
- `figures/final_surface_temperature_effect.png`
- `figures/final_surface_vs_rkm_813K.png`
