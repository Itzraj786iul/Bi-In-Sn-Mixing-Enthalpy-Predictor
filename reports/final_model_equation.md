# Final Polynomial Degree-2 Model Equation

## Frozen model specification
- `PolynomialFeatures(degree=2, include_bias=True)` + minimum-norm OLS
- Features: **xBi, xIn, temperature_K** (xSn omitted; see below)
- Target: `integral_mixing_enthalpy_J_per_mol` (ΔmixH, J/mol)
- Training data: all **104** original experimental observations

## Primary validated performance (LOCSO — do not replace with training metrics)

These are the **primary defensible** out-of-fold metrics from Step 3H:

- **ML (LOCSO OOF):** MAE = 41.10 J/mol, RMSE = 56.34 J/mol, R² = 0.9712
- **RKM (same 104 test points):** MAE = 57.13 J/mol, RMSE = 73.49 J/mol, R² = 0.9510

## Training-set fit (104 observations — not primary validation)

The model below is refit on **all 104 points** for coefficient extraction.
These in-sample metrics are reported for completeness only:

- ML (training): MAE = 27.09 J/mol, RMSE = 34.64 J/mol, R² = 0.9891
- RKM (same 104 points): MAE = 57.13 J/mol, RMSE = 73.49 J/mol, R² = 0.9510

## Fitted equation

ΔmixH (J/mol) =

  +1477.0188
  − 1633.6028·xBi
  − 2198.6883·xIn
  − 2.6296·temperature_K
  − 1875.2975·xBi²
  − 9182.9031·xBi·xIn
  + 4.0212·xBi·temperature_K
  + 535.3303·xIn²
  + 2.0255·xIn·temperature_K
  + 0.0010·temperature_K²

## Coefficient table

| Term | Coefficient (J/mol per term unit) | Sign | |coefficient| |
| --- | ---: | --- | ---: |
| intercept | 1477.018848 | positive | 1477.018848 |
| xBi | -1633.602767 | negative | 1633.602767 |
| xIn | -2198.688284 | negative | 2198.688284 |
| temperature_K | -2.629607 | negative | 2.629607 |
| xBi² | -1875.297539 | negative | 1875.297539 |
| xBi·xIn | -9182.903143 | negative | 9182.903143 |
| xBi·temperature_K | 4.021166 | positive | 4.021166 |
| xIn² | 535.330326 | positive | 535.330326 |
| xIn·temperature_K | 2.025539 | positive | 2.025539 |
| temperature_K² | 0.000974 | positive | 0.000974 |

Note: |coefficient| alone does **not** rank physical importance because xBi, xIn
are mole fractions (0–1) while temperature_K is hundreds of kelvin.

## Why degree 2 gives ten terms

With three input variables (xBi, xIn, T) and degree 2, PolynomialFeatures builds:

1. intercept (constant 1)
2. three linear terms: xBi, xIn, T
3. six quadratic terms: xBi², xIn², T², xBi·xIn, xBi·T, xIn·T

Total = 1 + 3 + 6 = **10** terms.

There are no xSn terms because xSn was deliberately omitted from the feature vector.

## Why xSn is absent and how it is recovered

Mole fractions satisfy **xBi + xIn + xSn = 1**, so xSn is not an independent variable.
Given any (xBi, xIn) within the ternary simplex:

**xSn = 1 − xBi − xIn**

Including both xSn and (xBi, xIn) would create redundant columns and an ill-conditioned
polynomial design matrix (as seen in Step 3H).

## Meaning of interaction terms (mathematical, not thermodynamic)

- **xBi·xIn**: allows the effect of In to depend on Bi level (and vice versa) beyond linear addition.
- **xBi·T** and **xIn·T**: allow composition effects to change with temperature.
- **xBi², xIn², T²**: allow curvature along each axis.

These are **empirical regression shapes**, not Redlich–Kister or RKM interaction parameters.

## Coefficients are not thermodynamic interaction parameters

The fitted numbers are ordinary least-squares weights chosen to minimize error on 104
calorimetry points. They should **not** be read as Lᵢⱼ or ternary indices from Table IV.

Different variable scaling (especially T in kelvin vs mole fractions) means coefficient
magnitude comparisons across term types are misleading without standardization.

## Comparison with RKM

| Aspect | RKM (Eq. 4, Table IV) | Final ML polynomial |
| --- | --- | --- |
| Origin | Paper thermodynamic model | Fit to 104 experiments |
| Parameters | Fixed Lᵢⱼ from literature | 10 learned coefficients |
| Temperature | Not used in current implementation | Explicit feature T |
| Composition | xBi, xIn, xSn in formula | xBi, xIn (+ xSn = 1−xBi−xIn) |
| Validation | MAE 57.13 on 104 exp. points | LOCSO MAE 41.10 on 104 OOF points |

RKM remains the physics-based benchmark; ML captures experimental scatter and
temperature trends that the temperature-independent RKM surface cannot represent.
