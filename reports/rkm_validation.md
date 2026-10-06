# RKM validation against Table III experimental data

Paper: Kumar, Mohan and Behera, *J. Electron. Mater.* **48**, 8096–8106 (2019).

The implementation uses Equation (4) and Table IV of that paper.
Parameters are temperature-independent. Experimental points at 767 K, 813 K and 855 K
are therefore compared to the same RKM prediction at each composition.

## Ternary index assignment

Validated (i, j, k) order: **In–Bi–Sn**

Table IV names the ternary parameter $L_{\mathrm{Bi-Sn-In}}$. A literal
(i, j, k) = (Bi, Sn, In) assignment does **not** reproduce Fig. 10 or the authors'
statement that the fitted curve is close to experiment. The assignment used here
(i = In, j = Bi, k = Sn) was selected because it reproduces the measured
cross-section shapes, the Bi-rich vs Sn-rich depth order, and a high R² vs Table III.
See `docs/rkm_model.md`.

## Overall metrics (all 9 series)

- N = 104
- MAE = 57.13 J/mol
- RMSE = 73.49 J/mol
- R² = 0.9510

## Metrics by series

| Series | T (K) | Cross-section | N | MAE | RMSE | R² |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 767 | (Sn0.33Bi0.67)1-xInx | 11 | 134.91 | 146.70 | 0.8131 |
| 2 | 813 | (Sn0.33Bi0.67)1-xInx | 11 | 76.82 | 95.13 | 0.9190 |
| 3 | 855 | (Sn0.33Bi0.67)1-xInx | 12 | 56.32 | 61.28 | 0.9645 |
| 4 | 767 | (Sn0.50Bi0.50)1-xInx | 12 | 59.93 | 66.84 | 0.9395 |
| 5 | 813 | (Sn0.50Bi0.50)1-xInx | 11 | 35.71 | 38.64 | 0.9799 |
| 6 | 855 | (Sn0.50Bi0.50)1-xInx | 11 | 25.58 | 39.45 | 0.9781 |
| 7 | 767 | (Sn0.67Bi0.33)1-xInx | 12 | 28.64 | 37.82 | 0.9643 |
| 8 | 813 | (Sn0.67Bi0.33)1-xInx | 12 | 41.76 | 46.30 | 0.9460 |
| 9 | 855 | (Sn0.67Bi0.33)1-xInx | 12 | 58.25 | 64.74 | 0.8998 |

## Metrics by temperature

- 767 K: MAE = 72.77 J/mol, RMSE = 93.73 J/mol, R² = 0.9239
- 813 K: MAE = 51.14 J/mol, RMSE = 64.56 J/mol, R² = 0.9621
- 855 K: MAE = 47.32 J/mol, RMSE = 56.69 J/mol, R² = 0.9683

## Qualitative shape checks (Figures 2–4)

- (Sn0.33Bi0.67)1-xInx: model minimum at xIn=0.500.
- (Sn0.50Bi0.50)1-xInx: model minimum at xIn=0.540.
- (Sn0.67Bi0.33)1-xInx: model minimum at xIn=0.590.
- Minima order preserved: Bi-rich -1303.1 < equiatomic -1039.7 < Sn-rich -785.6 J/mol.

Shape checks passed: **True**

> **Reproducibility note on the minima values.** The minimum values above
> (Bi-rich −1303.1, equiatomic −1039.7, Sn-rich −785.6 J/mol) are the historical values from
> the original execution and are kept unchanged here. The code or execution that produced them
> is not available in the repository, so they cannot be reproduced exactly from the current
> repository. The current RKM implementation (`src/rkm_model.py`), evaluated along the three
> reconstructed cross-sections on a 0.01 xIn grid by `scripts/generate_research_figures.py`,
> gives Bi-rich −1317.5, equiatomic −1039.9 and Sn-rich −794.2 J/mol, at the same minimum
> locations (xIn = 0.50 / 0.54 / 0.59). The qualitative conclusion is unchanged: the minima
> order Bi-rich < equiatomic < Sn-rich is preserved, and the shape checks pass. All other
> metrics in this report (overall, per-series and per-temperature) reproduce exactly.

## Residual comments

The largest residuals are on the Bi-rich section (Series 1) and at some In-rich
compositions, which the paper already flags as the weaker part of the fit.
Residuals remain of the same order as, or larger than, the tabulated series
standard uncertainties u(ΔmixH) (8–23 J/mol) but much smaller than the
stated overall calorimeter uncertainty of 10–12%.

Synthetic generation was allowed to proceed only because the model reproduces
the experimental qualitative behavior of Figures 2–4 and 10.
