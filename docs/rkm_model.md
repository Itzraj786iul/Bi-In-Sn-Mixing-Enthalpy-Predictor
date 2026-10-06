# Redlich–Kister–Muggianu model (Equation 4, Table IV)

Implementation: `src/rkm_model.py`  
Function: `rkm_delta_mix_h(x_bi, x_in, x_sn, temperature=None)`  
Units: J mol\(^{-1}\).

## Equation

From Kumar, Mohan and Behera (2019), Equation (4):

\[
\Delta_{\mathrm{mix}}H
=
\sum_i \sum_{j>i}
x_i x_j
\left[
\sum_{\nu}
L^{(\nu)}_{i:j}
(x_i - x_j)^{\nu}
\right]
+
x_i x_j x_k
\left(
L^{(0)}_{i:j:k}\, x_i
+
L^{(1)}_{i:j:k}\, x_j
+
L^{(2)}_{i:j:k}\, x_k
\right)
\]

The first (double) sum is the three **binary** Redlich–Kister contributions. The second term is a single **ternary** Muggianu-type contribution. There is no explicit temperature in Table IV or Equation (4).

## Binary interaction terms

Named pairs and coefficients from Table IV (rendered page 8106; minus signs verified):

| Pair \((i,j)\) | \(\nu\) | \(L^{(\nu)}_{i:j}\) (J/mol) | \((x_i-x_j)\) |
| --- | --- | --- | --- |
| Bi–In | 0, 1, 2 | −6617, 133, 2066 | \(x_{\mathrm{Bi}}-x_{\mathrm{In}}\) |
| Bi–Sn | 0, 1, 2, 3 | 520, 235, −157, −119 | \(x_{\mathrm{Bi}}-x_{\mathrm{Sn}}\) |
| In–Sn | 0, 1 | −1980, −893 | \(x_{\mathrm{In}}-x_{\mathrm{Sn}}\) |

Each binary contribution is \(x_i x_j \sum_\nu L^{(\nu)}(x_i-x_j)^\nu\).

Physical reading consistent with the paper:

- Bi–In: large negative \(L^{(0)}\) → strongly exothermic binary (iso-enthalpies bend toward Bi–In).
- Bi–Sn: positive \(L^{(0)}\) → endothermic binary (Table II).
- In–Sn: negative \(L^{(0)}\) → moderately exothermic binary.

The paper states that these binary parameters come from the authors’ **unpublished** work, not from a re-fit of Table II in this article. The RKM Bi–Sn edge therefore need not match every Table II point.

## Ternary interaction terms

Table IV lists \(L_{\mathrm{Bi-Sn-In}}^{(t)}\):

| \(t\) | Value (J/mol) |
| --- | --- |
| 0 | −3874 |
| 1 | −12777 |
| 2 | 24187 |

Equation (4) multiplies \(L^{(0)}\) by \(x_i\), \(L^{(1)}\) by \(x_j\), \(L^{(2)}\) by \(x_k\). The printed name \(L_{\mathrm{Bi-Sn-In}}\) would suggest \((i,j,k)=(\mathrm{Bi},\mathrm{Sn},\mathrm{In})\). That literal assignment **fails** to reproduce Fig. 10 and the authors’ claim that the fitted curve is close to experiment (large systematic error on the Bi-rich section).

The implementation therefore uses the unique permutation that recovers the measured cross-section shapes and a high \(R^2\) versus Table III:

\[
(i,j,k) = (\mathrm{In},\ \mathrm{Bi},\ \mathrm{Sn})
\]

\[
\Delta H_{\mathrm{ternary}}
=
x_{\mathrm{In}}\, x_{\mathrm{Bi}}\, x_{\mathrm{Sn}}
\left(
L^{(0)} x_{\mathrm{In}}
+ L^{(1)} x_{\mathrm{Bi}}
+ L^{(2)} x_{\mathrm{Sn}}
\right)
\]

This assignment also matches two qualitative facts in the PDF:

- \(L^{(2)}=+24187\) multiplies \(x_{\mathrm{Sn}}\): minima become less exothermic as \(x_{\mathrm{Sn}}/x_{\mathrm{Bi}}\) increases.
- \(L^{(1)}=-12777\) multiplies \(x_{\mathrm{Bi}}\): mixing is more negative toward Bi–In (Fig. 9).

The ambiguity is recorded in `reports/rkm_validation.md`. Synthetic data were not generated from the literal Bi–Sn–In index order because that order does not reproduce the paper’s own fit.

## Meaning of the \(L\) parameters

- Binary \(L^{(\nu)}_{i:j}\): coefficients of the Redlich–Kister expansion of the \(i\)–\(j\) contribution to \(\Delta_{\mathrm{mix}}H\).
- Ternary \(L^{(t)}_{i:j:k}\): coefficients of the composition-weighted ternary correction. They were obtained by least squares using the binary \(L\) values plus the calorimetric ternary data of this paper.
- All \(L\) are in J/mol and are treated as **enthalpy** parameters (no \(T\) factor in Table IV).

## Composition constraint

Mole fractions must satisfy

\[
x_{\mathrm{Bi}} \ge 0,\quad
x_{\mathrm{In}} \ge 0,\quad
x_{\mathrm{Sn}} \ge 0,\quad
x_{\mathrm{Bi}}+x_{\mathrm{In}}+x_{\mathrm{Sn}}=1.
\]

The code raises if the sum is not 1 within \(10^{-8}\). On a cross-section \((Sn_a Bi_b)_{1-x}In_x\), two mole fractions are determined by \(x_{\mathrm{In}}\) and the bath ratio; the third is the remainder.

The optional argument `T` is accepted so that `rkm_delta_mix_h(xBi, xIn, xSn, T)` is a valid call. **`T` is not used.** The same composition yields the same \(\Delta_{\mathrm{mix}}H\) at 767 K, 813 K and 855 K, in line with the paper’s “almost temperature-independent” conclusion.

## Relation to the experimental data

- Table III \(\Delta_{\mathrm{mix}}H\) is the integral quantity from Eq. (3), including the Bi–Sn bath enthalpy.
- RKM \(\Delta_{\mathrm{mix}}H\) is the model of that same integral quantity from pure liquid elements.
- The model was checked on every Table III composition before any synthetic row was written (`reports/rkm_validation.md`).
- It reproduces: exothermic mixing; a minimum near \(x_{\mathrm{In}}\approx 0.55\); Bi-rich deeper than Sn-rich; overlapping temperatures.

## Limitations of using the fitted model as a synthetic generator

1. The experimental design samples **three lines**, not the interior of the Gibbs triangle. Interior Dataset B points are model evaluations, not interpolations of measurements.
2. RKM synthetic labels **inherit every assumption** of Equation (4) and Table IV (polynomial form, unpublished binaries, ternary index assignment).
3. The authors already report larger deviations at some **In-rich** compositions.
4. There is **no fitted temperature dependence**; repeating the grid at three temperatures does not create a thermal effect.
5. Training an ML model on RKM labels and testing it on other RKM labels tests reproduction of the polynomial, not calorimetric accuracy.
6. Noise added to synthetic rows is a statistical model of tabulated \(u(\Delta_{\mathrm{mix}}H)\), not a new measurement.

**Note on synthetic data:** synthetic \(\Delta_{\mathrm{mix}}H\) is not independent experimental evidence. RKM-derived synthetic augmentation was later tested and did not improve LOCSO prediction of the real observations, so it is not used by the final predictor. Train/test scores on RKM-generated labels only show that a model copied the polynomial. Held-out Table III rows (`source = paper_experiment`) remain the external check.

Experimental Table III rows remain the only calorimetric anchors.
