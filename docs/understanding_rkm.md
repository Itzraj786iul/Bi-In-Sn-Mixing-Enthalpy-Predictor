# Understanding the RKM Model

> **Historical note:** This document describes an earlier stage of the project, written before any ML model existed. The RKM description itself remains accurate and RKM is the fixed thermodynamic benchmark. The final implementation subsequently developed an experimental-only Direct Polynomial Degree-2 predictor (xBi, xIn, T) and evaluated it using strict leave-one-cross-section-out (LOCSO) validation. RKM-derived synthetic data were tested as augmentation and are not used to train the final predictor. Refer to `README.md` and the final research reports for the completed methodology and results.

This note explains the Redlich–Kister–Muggianu (RKM) calculation that is actually sitting in this repository. It is based on `src/rkm_model.py`, Table III and Table IV in `professor_paper.pdf`, `docs/rkm_model.md`, and `reports/rkm_validation.md`. No new parameters were invented here.

---

## 1. Why do we need RKM?

The professor’s paper measured mixing enthalpy by dropping indium into a liquid Bi–Sn bath. That is real calorimetry. It is also slow. They only measured **104** compositions, and those 104 points lie on **three lines** in the Bi–In–Sn triangle, not all over the triangle.

Our project still needs a number for \(\Delta_{\mathrm{mix}}H\) at many other compositions. Before any machine-learning model exists, we need a thermodynamic formula that can take a composition and return a mixing enthalpy.

That formula is the RKM polynomial from the paper (Equation 4, Table IV).

In one sentence: we have a mixture of Bi, In and Sn, and we want an estimate of how much heat is involved when those three liquids mix. RKM is the paper’s mathematical model for that heat.

---

## 2. What does RKM do?

Think of RKM as a calculator with a very specific job.

You tell it:

- how much bismuth,
- how much indium,
- how much tin.

It returns:

- the integral molar mixing enthalpy, \(\Delta_{\mathrm{mix}}H\), in J/mol.

It does **not** look at the calorimeter files. It does **not** interpolate neighbouring experimental points by distance. It evaluates a polynomial whose coefficients were taken from Table IV of the paper.

If the three mole fractions are a valid liquid composition (non-negative and summing to 1), RKM always returns a number. That is why it can later fill the rest of the triangle.

---

## 3. Inputs and output

The function is `rkm_delta_mix_h` in `src/rkm_model.py`.

**Inputs**

| Input | Meaning | Actually used? |
| --- | --- | --- |
| `x_bi` | mole fraction of Bi | yes |
| `x_in` | mole fraction of In | yes |
| `x_sn` | mole fraction of Sn | yes |
| `temperature` | bath temperature in K | **no** |

The code accepts temperature so that a call like `rkm_delta_mix_h(xBi, xIn, xSn, T)` is valid. The temperature value is then ignored. Table IV has no \(T\) dependence, and the paper says mixing enthalpy is almost temperature-independent between 767 K and 855 K. The same composition therefore gives the same RKM value at all three experimental temperatures.

The mole fractions must satisfy

\[
x_{\mathrm{Bi}} \ge 0,\quad
x_{\mathrm{In}} \ge 0,\quad
x_{\mathrm{Sn}} \ge 0,\quad
x_{\mathrm{Bi}} + x_{\mathrm{In}} + x_{\mathrm{Sn}} = 1.
\]

The code raises an error if the sum is not 1 within \(10^{-8}\).

**Output**

- quantity: integral molar mixing enthalpy \(\Delta_{\mathrm{mix}}H\)
- units: J/mol
- sign: negative means exothermic (heat is released on mixing); positive means endothermic

This is **not** the partial enthalpy of indium. Table III reports both. In our experimental CSV they are separate columns:

- `partial_enthalpy_In_J_per_mol` — partial molar enthalpy of In
- `integral_mixing_enthalpy_J_per_mol` — \(\Delta_{\mathrm{mix}}H\), the target

RKM models the second one.

---

## 4. Physical meaning of mixing enthalpy

Pure liquid Bi, pure liquid In and pure liquid Sn each have their own enthalpy. When they mix, the enthalpy of the liquid is not just the weighted average of the three pure liquids. The difference is the mixing enthalpy.

If unlike atoms attract each other more strongly than like atoms, mixing releases heat and \(\Delta_{\mathrm{mix}}H\) is negative (exothermic). If they prefer their own kind, mixing absorbs heat and \(\Delta_{\mathrm{mix}}H\) is positive (endothermic).

In this paper the **ternary** measurements on the three sections are exothermic. The Bi–Sn **binary**, by contrast, is endothermic. That is already visible in the signs of the Table IV parameters, and it shows up later on the Bi–Sn edge of Dataset B.

RKM is a way of writing that mixing enthalpy as a function of composition. It starts from the three binary edges and then adds a ternary correction because three metals together are not just three independent pairs.

---

## 5. RKM equation

The paper’s Equation (4) is the equation implemented in `src/rkm_model.py`:

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

There are no extra terms in the code. The implementation is exactly:

```text
ΔmixH
  = Bi–In binary contribution
  + Bi–Sn binary contribution
  + In–Sn binary contribution
  + ternary contribution
```

That is the whole model.

---

## 6. Binary interaction terms

A binary term answers: “if only this pair of metals existed, how much would they contribute to mixing enthalpy at these mole fractions?”

For a pair \((i,j)\) the code computes

\[
x_i x_j \sum_n L_n (x_i - x_j)^n
\]

The product \(x_i x_j\) is zero if either metal is missing, so a binary contribution automatically vanishes on the opposite edge. The polynomial in \((x_i - x_j)\) lets the pair’s mixing enthalpy change as the two metals become unequal.

Our three pairs, from Table IV and `_binary_term` in the code:

| Pair \((i,j)\) | What the code calls | \(L\) values (J/mol) | Difference used |
| --- | --- | --- | --- |
| Bi–In | `_binary_term(x_bi, x_in, L_BI_IN)` | −6617, 133, 2066 | \(x_{\mathrm{Bi}}-x_{\mathrm{In}}\) |
| Bi–Sn | `_binary_term(x_bi, x_sn, L_BI_SN)` | 520, 235, −157, −119 | \(x_{\mathrm{Bi}}-x_{\mathrm{Sn}}\) |
| In–Sn | `_binary_term(x_in, x_sn, L_IN_SN)` | −1980, −893 | \(x_{\mathrm{In}}-x_{\mathrm{Sn}}\) |

In plain language:

- **Bi–In** is the strong exothermic pair. \(L^{(0)} = -6617\) J/mol is large and negative. This is why the paper’s iso-enthalpy curves (Fig. 9) bend toward Bi–In.
- **Bi–Sn** is the endothermic pair. \(L^{(0)} = +520\) J/mol is positive. Mixing Bi with Sn costs heat.
- **In–Sn** is moderately exothermic. \(L^{(0)} = -1980\) J/mol.

The number of \(L\) coefficients is not the same for every pair. That is how Table IV is printed: Bi–In has three, Bi–Sn has four, In–Sn has two. The code just loops over whatever array is stored.

---

## 7. Ternary interaction term

Three metals can do something that no pair can describe on its own. The second part of Equation (4) is that extra correction.

In the code, with \((i,j,k) = (\mathrm{In},\mathrm{Bi},\mathrm{Sn})\):

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

with

\[
L^{(0)} = -3874,\quad
L^{(1)} = -12777,\quad
L^{(2)} = 24187
\quad \text{J/mol}.
\]

The product \(x_{\mathrm{In}} x_{\mathrm{Bi}} x_{\mathrm{Sn}}\) is zero on every binary edge and at every pure metal. So the ternary term only exists inside the triangle. That is physically reasonable: you should not need a three-metal correction when one metal is absent.

The expression in parentheses weights the three ternary \(L\) values by composition. Which metal multiplies which \(L\) is the \((i,j,k)\) convention discussed below.

---

## 8. Interaction parameters

The \(L\) numbers are the interaction parameters. They are not mole fractions and they are not the mixing enthalpy itself. They are the fitted constants inside the polynomial.

**Where they came from**

All of them are copied from Table IV of Kumar, Mohan and Behera (2019), page 8106. Signs were checked on the printed table.

The paper is explicit about two different origins:

- **Binary \(L\) values** were taken from the authors’ **unpublished** work. They were not re-fitted from Table II in this article.
- **Ternary \(L\) values** were obtained by a least-squares fit of Equation (4) to **this paper’s** calorimetric ternary data, using those binary parameters as already known.

That is why we treat Table IV as given. We did not re-optimise anything.

**What they are doing in the calculation**

Each binary \(L^{(\nu)}\) scales one power of \((x_i - x_j)\). \(L^{(0)}\) is the leading, roughly “how strongly this pair mixes.” Higher \(\nu\) terms tilt and bend that pair’s contribution as the two mole fractions become unequal.

Each ternary \(L^{(t)}\) scales one metal’s mole fraction inside the three-metal correction. \(L^{(1)} = -12777\) J/mol multiplies \(x_{\mathrm{Bi}}\), which pulls the surface more negative toward Bi. \(L^{(2)} = +24187\) J/mol multiplies \(x_{\mathrm{Sn}}\), which makes mixing less exothermic as tin replaces bismuth. That matches Fig. 9 and the three experimental sections.

All \(L\) are enthalpy parameters in J/mol. There is no \(RT\) factor and no temperature multiplier in Table IV or in our code.

Without these numbers, Equation (4) is only a shape. The parameters are what make it *this* alloy’s model rather than a blank polynomial.

---

## 9. \((i,j,k) = (\mathrm{In},\mathrm{Bi},\mathrm{Sn})\) convention

Equation (4) writes the ternary piece as \(L^{(0)} x_i + L^{(1)} x_j + L^{(2)} x_k\). Table IV names the ternary parameter \(L_{\mathrm{Bi-Sn-In}}\).

If we read that name literally, we would set

\[
(i,j,k) = (\mathrm{Bi},\mathrm{Sn},\mathrm{In}).
\]

That assignment is **not** what the code uses. Our documentation (`docs/rkm_model.md` and `reports/rkm_validation.md`) treats this as an indexing / order convention, not as a claim that the paper is wrong.

The implementation uses

\[
(i,j,k) = (\mathrm{In},\mathrm{Bi},\mathrm{Sn}).
\]

That is the comment next to the ternary term in `src/rkm_model.py`:

```text
ternary = x_in * x_bi * x_sn * (
    L_TERNARY[0] * x_in + L_TERNARY[1] * x_bi + L_TERNARY[2] * x_sn
)
```

**Why the order matters**

The three ternary \(L\) values are very different (−3874, −12777, +24187). Swapping which metal multiplies which \(L\) completely changes the ternary surface. The binary terms are unaffected, because each binary pair is named explicitly (Bi–In, Bi–Sn, In–Sn).

**How this convention was chosen**

The literal Bi–Sn–In assignment does not reproduce Fig. 10 or the paper’s statement that the fitted curve is close to experiment. In particular it gives a large systematic error on the Bi-rich section. The In–Bi–Sn order recovers:

- the measured cross-section shapes,
- Bi-rich minima deeper than Sn-rich minima,
- \(R^2 = 0.9510\) versus all 104 Table III points.

That is the validated convention recorded in `reports/rkm_validation.md`. Synthetic data were generated from this order, not from the literal name order.

If the order were changed, the RKM numbers in this repository would no longer match the validation report, and they would no longer be the surface that Dataset A and Dataset B were built from.

---

## 10. One real experimental example

The walkthrough uses **EXP_0051** from `data/original_experimental_data.csv`.

Why this row: it is Series 5, 813 K, on the equiatomic section \((Sn_{0.50}Bi_{0.50})_{1-x}In_x\). That is the same section and temperature as Fig. 10 in the paper. \(x_{\mathrm{In}} = 0.4982\) sits near the reported minimum around 0.55, and the three mole fractions are easy to read.

| Field | Value |
| --- | --- |
| id | EXP_0051 |
| source | paper_experiment (Table III) |
| series | 5 |
| cross-section | `(Sn0.50Bi0.50)1-xInx` |
| temperature | 813 K (recorded; **not used by RKM**) |
| \(x_{\mathrm{Bi}}\) | 0.2509 |
| \(x_{\mathrm{In}}\) | 0.4982 |
| \(x_{\mathrm{Sn}}\) | 0.2509 |
| sum of mole fractions | 1.0000 |
| partial enthalpy of In | −1603 J/mol |
| **experimental \(\Delta_{\mathrm{mix}}H\)** | **−1017.0 J/mol** |

Notice that the partial enthalpy (−1603) is not the same as the integral mixing enthalpy (−1017). RKM is compared with −1017, not −1603.

Because this section is labelled 50:50 Bi:Sn, \(x_{\mathrm{Bi}} = x_{\mathrm{Sn}}\). That will make the Bi–Sn polynomial collapse to only \(L^{(0)}\), which is a useful check that the formula is doing what we think.

---

## 11. RKM vs experiment

Pass this composition into `rkm_delta_mix_h(0.2509, 0.4982, 0.2509)`.

### Bi–In contribution

\[
x_{\mathrm{Bi}}-x_{\mathrm{In}} = 0.2509 - 0.4982 = -0.2473
\]

\[
\begin{aligned}
\sum L_n (x_{\mathrm{Bi}}-x_{\mathrm{In}})^n
&= -6617 + 133(-0.2473) + 2066(-0.2473)^2 \\
&= -6617 - 32.89 + 126.35 \\
&= -6523.54
\end{aligned}
\]

\[
x_{\mathrm{Bi}} x_{\mathrm{In}} \times (-6523.54) = 0.12499838 \times (-6523.54) = -815.43\ \mathrm{J/mol}
\]

This is the dominant term. Almost all of the exothermic enthalpy at this point comes from Bi–In.

### Bi–Sn contribution

\[
x_{\mathrm{Bi}}-x_{\mathrm{Sn}} = 0
\]

Every power of the difference higher than 0 is therefore 0, and only \(L^{(0)} = 520\) survives.

\[
x_{\mathrm{Bi}} x_{\mathrm{Sn}} \times 520 = 0.06295081 \times 520 = +32.73\ \mathrm{J/mol}
\]

Small and **positive**: Bi and Sn still contribute an endothermic binary piece even inside the ternary alloy.

### In–Sn contribution

\[
x_{\mathrm{In}}-x_{\mathrm{Sn}} = 0.2473
\]

\[
-1980 + (-893)(0.2473) = -2200.84
\]

\[
x_{\mathrm{In}} x_{\mathrm{Sn}} \times (-2200.84) = -275.10\ \mathrm{J/mol}
\]

### Ternary contribution

\[
x_{\mathrm{In}} x_{\mathrm{Bi}} x_{\mathrm{Sn}} = 0.03136209
\]

\[
\begin{aligned}
L^{(0)} x_{\mathrm{In}} + L^{(1)} x_{\mathrm{Bi}} + L^{(2)} x_{\mathrm{Sn}}
&= (-3874)(0.4982) + (-12777)(0.2509) + (24187)(0.2509) \\
&= -1930.03 - 3205.75 + 6068.52 \\
&= +932.74
\end{aligned}
\]

\[
0.03136209 \times 932.74 = +29.25\ \mathrm{J/mol}
\]

At this particular composition the ternary correction is small and slightly positive. It does not dominate the answer. The binaries already put us near −1000 J/mol.

### Total

```text
Bi–In binary     =  −815.43 J/mol
Bi–Sn binary     =    +32.73 J/mol
In–Sn binary     =   −275.10 J/mol
Ternary          =    +29.25 J/mol
--------------------------------
RKM ΔmixH        =  −1028.55 J/mol
```

Passing the same numbers through `rkm_delta_mix_h` returns **−1028.55 J/mol**. Adding temperature = 813 K does not change it.

```text
Experimental ΔmixH  =  −1017.0 J/mol
RKM ΔmixH           =  −1028.55 J/mol
RKM − experiment    =    −11.55 J/mol
```

So for EXP_0051 the model is 11.55 J/mol more exothermic than the calorimeter. That is a close point. Series 1 (Bi-rich, 767 K) is much worse on average; this equiatomic 813 K series is one of the better ones (MAE 35.71 J/mol, \(R^2 = 0.9799\)).

Flow for this row:

```text
xBi=0.2509, xIn=0.4982, xSn=0.2509
        ↓
binary terms (−815.43 + 32.73 − 275.10)
        ↓
ternary term (+29.25)
        ↓
RKM ΔmixH = −1028.55 J/mol
        ↓
compare with experimental −1017.0 J/mol
        ↓
difference = −11.55 J/mol
```

---

## 12. MAE, RMSE and R²

The same comparison was done for **all 104** experimental rows:

1. Read \(x_{\mathrm{Bi}}\), \(x_{\mathrm{In}}\), \(x_{\mathrm{Sn}}\) from Table III.
2. Compute RKM \(\Delta_{\mathrm{mix}}H\).
3. Subtract from the experimental integral value (not the partial enthalpy).
4. Repeat 104 times.
5. Summarise the 104 residuals.

Overall result, from `reports/rkm_validation.md` and `scripts/validate_dataset.py`:

- MAE = 57.13 J/mol
- RMSE = 73.49 J/mol
- \(R^2\) = 0.9510

These are **RKM-versus-experiment** numbers. They are not machine-learning scores. When this note was written, no ML model had been trained. The final ML model (Direct Poly D2) was later evaluated under LOCSO (MAE 41.10, RMSE 56.34, R² 0.9712). Those scores are not strictly like-for-like with the RKM numbers above, because the RKM ternary parameters were fitted to these measurements by the paper's authors.

**MAE.** Mean absolute error. On average, RKM is 57 J/mol away from the calorimeter. Experimental \(\Delta_{\mathrm{mix}}H\) in this file runs from −1413.0 to −64.32 J/mol, so 57 J/mol is a modest fraction of that range, but it is larger than the series uncertainties \(u(\Delta_{\mathrm{mix}}H)\) (about 8–23 J/mol).

**RMSE.** Root-mean-square error. Same idea, but large mistakes count more. RMSE (73.49) is higher than MAE (57.13) because some points, especially Series 1, have bigger residuals.

**\(R^2\).** This is **not** “95.1% accurate.” It is the fraction of *variation* in the 104 experimental integral enthalpies that the RKM predictions follow. \(R^2 = 0.9510\) means the RKM values track the up-and-down shape of the measurements quite closely. A model that always predicted the mean experimental value would have \(R^2 = 0\). A perfect point-by-point match would have \(R^2 = 1\).

The fit is not uniform. Series 1 is the weakest (\(R^2 = 0.8131\), MAE = 134.91 J/mol). Series 5, which contains EXP_0051, is much closer (\(R^2 = 0.9799\)). The paper already notes larger deviations at some indium-rich compositions.

---

## 13. Why RKM validation matters

We were about to use this polynomial as a generator of extra compositions. If the polynomial did not even follow the 104 measured points, those extra numbers would be a different surface from the paper.

Validation asked a simple question: when we evaluate Equation (4) at the **same compositions the calorimeter used**, do we get mixing enthalpies that resemble Table III?

The answer in this repository is: yes, well enough to keep going. Qualitative checks also passed (exothermic sections, minimum near \(x_{\mathrm{In}} \approx 0.55\), Bi-rich deeper than Sn-rich). That is why synthetic generation was allowed to proceed.

Validation does **not** prove that RKM is correct in the unmeasured interior of the triangle. It only shows agreement on the three measured lines.

---

## 14. How this leads to synthetic data

Once the 104-point check looked reasonable, RKM could be evaluated at compositions the calorimeter never visited.

```text
104 experimental points
        ↓
validate RKM against those points
        ↓
RKM follows Table III closely enough
        ↓
evaluate RKM at additional compositions
        ↓
synthetic Dataset A and Dataset B
```

**Dataset A** (`data/synthetic_cross_sections.csv`): 7389 points still on the **same three experimental lines**, with a dense \(x_{\mathrm{In}}\) grid and a little noise. The `base_model_prediction` column is pure RKM. The target column is RKM plus noise.

**Dataset B** (`data/synthetic_full_ternary.csv`): 15453 points on a composition grid covering the **whole triangle**. The target column is pure RKM (no noise). Most of those rows are `unsampled_ternary_region`: the calorimeter was never there.

This note stops there. The important idea for now is: synthetic \(\Delta_{\mathrm{mix}}H\) is RKM output, not a new drop-calorimetry experiment.

---

## 15. What RKM does NOT mean

| Quantity | Meaning | Source |
| --- | --- | --- |
| Experimental \(\Delta_{\mathrm{mix}}H\) | Heat of mixing measured in the drop calorimeter, converted to the integral molar value | Professor’s Table III (`source = paper_experiment`) |
| RKM \(\Delta_{\mathrm{mix}}H\) | Value of Equation (4) at a given composition | Thermodynamic polynomial in `src/rkm_model.py` |
| Synthetic \(\Delta_{\mathrm{mix}}H\) | RKM evaluated on a grid (Dataset B), or RKM plus noise on the three lines (Dataset A) | RKM-generated (`source = synthetic`) |

RKM is not a measurement. It is not a neural network. It does not “know” temperature. It does not predict partial enthalpy of indium. \(R^2 = 0.9510\) is not a statement that the model is 95.1% accurate everywhere in the triangle.

A good score of an ML model on synthetic labels would only mean that it copied this polynomial. The final ML model was trained on the 104 experimental rows only, and the 104 Table III rows remain the only calorimetric check.

---

*Code path: `src/rkm_model.py` → `rkm_delta_mix_h`. Paper: Kumar, Mohan and Behera, J. Electron. Mater. 48 (2019) 8096–8106, Equation (4) and Table IV.*
