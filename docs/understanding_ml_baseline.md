# Understanding the ML Baseline

This note explains the **first machine-learning experiment** in the project. It uses ordinary linear regression on composition only, evaluated with leave-one-cross-section-out (LOCSO). Results are in `reports/ml_baseline_results.md`.

No Random Forest, no boosting, no temperature feature yet — just a transparent pipeline check.

---

## 1. What regression means

We want to predict a number (\(\Delta_{\mathrm{mix}}H\)) from inputs (mole fractions). **Regression** means fitting a formula that maps inputs → a continuous output, not a category.

We already have a regression-like model: RKM is a fixed polynomial. ML regression **learns** coefficients from data instead of taking them from Table IV.

---

## 2. What Linear Regression does

**Linear Regression** fits a straight-line combination of features:

\[
\Delta_{\mathrm{mix}}H \approx w_0 + w_1 x_{\mathrm{Bi}} + w_2 x_{\mathrm{In}} + w_3 x_{\mathrm{Sn}}
\]

`sklearn.linear_model.LinearRegression` finds the weights \(w\) that best match the training targets (least squares). It is the simplest reasonable ML baseline: easy to interpret, no hyperparameters to tune.

It cannot capture curved iso-enthalpy lines by itself unless the physics is nearly linear in composition — which it is not for this alloy system.

---

## 3. Input features

First baseline uses **three features only**:

| Feature | Column |
| --- | --- |
| Bismuth mole fraction | `xBi` |
| Indium mole fraction | `xIn` |
| Tin mole fraction | `xSn` |

Temperature is **not** included yet. A separate run with `[xBi, xIn, xSn, temperature_K]` will come later using the same folds.

---

## 4. Target

**Experimental evaluation always uses real calorimetry:**

- column: `integral_mixing_enthalpy_J_per_mol`
- meaning: integral molar mixing enthalpy \(\Delta_{\mathrm{mix}}H\) (J/mol)

**Synthetic training** uses:

- column: `delta_mix_H_J_per_mol` from Dataset A (RKM + noise)

The target is never used as an input feature.

---

## 5. What LOCSO means

**Leave-one-cross-section-out** (LOCSO) splits by the three experimental composition paths:

1. `(Sn0.33Bi0.67)1-xInx` — Bi-rich bath  
2. `(Sn0.50Bi0.50)1-xInx` — equiatomic bath  
3. `(Sn0.67Bi0.33)1-xInx` — Sn-rich bath  

Each fold **holds out one entire cross-section** for testing and trains on the other two. Every experimental point is tested exactly once across the three folds.

This tests: *Can the model predict mixing enthalpy on a composition path it never saw during training?*

---

## 6. What one fold looks like

Example — Fold 2 holds out the equiatomic section:

```text
TRAIN:  all experimental points on Bi-rich + Sn-rich sections (70 points)
TEST:   all experimental points on equiatomic section (34 points)
```

For synthetic experiments, **all Dataset A rows on the held-out section are removed from training** (4926 synthetic rows remain — the two other sections × 3 temperatures × 821 grid points).

Metrics (MAE, RMSE, R²) are computed on the test section. After three folds, we pool all 104 out-of-fold predictions for an **overall** metric.

---

## 7. Experimental-only

**Train:** experimental points from two cross-sections.  
**Test:** experimental points from the held-out cross-section.

Question: *With only 104 real measurements and a linear model, can we interpolate to a new line?*

This is the honest small-data baseline. No synthetic help.

---

## 8. Synthetic-only

**Train:** Dataset A synthetic rows from **two** cross-sections (held-out section excluded).  
**Test:** experimental points on the held-out section.

Question: *If ML learns from RKM labels on two lines, can it predict **real** calorimetry on the third line?*

Important: training labels are still RKM-based, not new experiments. Excluding the test section from synthetic training avoids the worst overlap (grid points 0.001 away on the same line).

---

## 9. Combined

**Train:** experimental points from two sections **plus** Dataset A synthetic from those same two sections.  
**Test:** experimental points on the held-out section.

Question: *Does adding dense synthetic data alongside sparse experiment improve prediction of real held-out experiment?*

Same exclusion rule: no synthetic rows from the test section.

---

## 10. Why RKM is used as a benchmark

Every test fold compares ML against **RKM on identical experimental points** using `src/rkm_model.rkm_delta_mix_h`.

RKM is the validated thermodynamic model (overall MAE 57.13 J/mol on all 104 points). If linear ML beats RKM on held-out experiment, that would be interesting. If ML only beats RKM on synthetic data, that means nothing about calorimetry.

Fair comparison requires the same test points and the same target column.

---

## 11. Why the 104 experimental observations remain important

Synthetic rows are useful for training density, but they inherit RKM assumptions. Only Table III rows are direct measurements.

**Ultimate reference:** `data/original_experimental_data.csv` evaluated through LOCSO.

A good score on synthetic training data does not prove the model learned physics. A good score on **held-out experimental cross-sections** is the meaningful check — and even then, with only ~34–36 test points per fold, uncertainty is large.

---

## 12. What this first experiment can and cannot tell us

**Can tell us:**

- The ML pipeline (load → split → train → predict → metrics) works.
- LOCSO and synthetic section exclusion behave as designed (sanity checks pass).
- A **linear** composition model is much worse than RKM on this task (expected).
- Synthetic training without the held-out section still does not beat RKM on real experiment in this baseline.

**Cannot tell us:**

- Whether a more flexible model (polynomial features, trees, etc.) would beat RKM.
- Whether temperature helps.
- Whether Dataset B interior points can be predicted experimentally (no labels there).
- That high accuracy on synthetic data implies calorimetric accuracy.

**First-run overall results (104 pooled OOF predictions):**

| Training type | ML MAE | ML R² | RKM MAE | RKM R² |
| --- | --- | --- | --- | --- |
| Experimental-only | 250.18 J/mol | 0.20 | 57.13 J/mol | 0.951 |
| Synthetic-only | 268.70 J/mol | 0.07 | 57.13 J/mol | 0.951 |
| Combined | 270.66 J/mol | 0.05 | 57.13 J/mol | 0.951 |

Negative fold-level R² values mean the linear model does worse than predicting the mean on that fold — a sign the baseline is too simple for cross-section generalization.

---

*Script: `scripts/train_baseline.py`. Strategy: `docs/ml_validation_strategy.md`. Predictions: `reports/ml_baseline_predictions.csv`.*
