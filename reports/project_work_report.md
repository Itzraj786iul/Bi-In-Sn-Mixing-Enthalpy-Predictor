# Project Work Report — Stage 1

**Project title:** AI-Based Surrogate Model for Predicting Mixing Enthalpy of Bi-In-Sn Lead-Free Solder Alloy

**This document:** technical record of the work completed so far (experimental-data reconstruction and thermodynamics-based synthetic-data generation). Machine-learning model training has not been started.

**Primary scientific source:** M.R. Kumar, S. Mohan and C.K. Behera, “Measurements of Mixing Enthalpy for a Lead-Free Solder Bi-In-Sn System,” *Journal of Electronic Materials*, Vol. 48, No. 12, 2019, pp. 8096–8106 (PDF supplied with the project).

No existing datasets, RKM code, or generation scripts were changed while writing this report.

---

## 1. Project overview

This semester project is about predicting the integral molar enthalpy of mixing, \(\Delta_{\mathrm{mix}}H\), of liquid Bi-In-Sn alloys from composition (and, later, possibly temperature). Bi-In-Sn is a lead-free solder system. The measurements we are using come from a drop-calorimetry study carried out by Kumar, Mohan and Behera (2019).

The paper measured mixing enthalpy at three bath temperatures:

- 767 K
- 813 K
- 855 K

along three composition lines (cross-sections) running from the Bi–Sn binary edge toward the indium corner. Those measurements are laboratory work. They took time, required a calorimeter, and cover only a limited set of compositions.

The long-term aim of *our* project is to train a surrogate model that can estimate \(\Delta_{\mathrm{mix}}H\) for other compositions without repeating a drop-calorimetry experiment every time. That surrogate has **not** been trained yet. Stage 1 only prepared and checked the data that such a model would use.

It is important to keep two pieces of work separate:

| Whose work | What it is |
| --- | --- |
| **Professor’s original research (the paper)** | Drop calorimetry of liquid Bi-In-Sn; Table III experimental enthalpies; Redlich–Kister–Muggianu (RKM) fit in Equation (4) and Table IV |
| **Our semester project (this repository)** | Reading the PDF, extracting Table III, reconstructing ternary compositions, implementing Equation (4), checking the implementation against Table III, generating clearly labelled synthetic data, and writing validation scripts |

We have not claimed new calorimetric measurements. We have not claimed that an AI model already predicts mixing enthalpy.

---

## 2. Scientific background

### 2.1 What is a Bi-In-Sn alloy?

It is a ternary mixture of bismuth (Bi), indium (In) and tin (Sn). Composition is written as mole fractions:

\[
x_{\mathrm{Bi}} + x_{\mathrm{In}} + x_{\mathrm{Sn}} = 1
\]

with each \(x_i \ge 0\). The paper studies the **liquid** alloy.

### 2.2 Why lead-free solders?

Conventional electronics solders are based on lead–tin. Lead is toxic, and the paper discusses health and environmental reasons for replacing it. No single metal replaces lead well, so candidate solders are multi-component alloys. Bi-In-Sn is one such system (and a subsystem of Bi-In-Sn-Zn). Thermodynamic data, including mixing enthalpy, are used when phase diagrams are assessed.

### 2.3 Mixing and enthalpy

**Mixing** here means forming a liquid solution from the pure liquid metals.

**Enthalpy** \(H\) is the heat content of the system at constant pressure. The **enthalpy of mixing** (mixing enthalpy) is the enthalpy change when the pure liquids are mixed at the same temperature and pressure:

\[
\Delta_{\mathrm{mix}}H = H_{\mathrm{mixture}} - \sum_i x_i H_i^{\mathrm{pure}}
\]

If \(\Delta_{\mathrm{mix}}H < 0\), mixing is **exothermic**: heat is released and unlike atoms attract each other more than in an ideal mixture. If \(\Delta_{\mathrm{mix}}H > 0\), mixing is **endothermic**. The paper reports that the **ternary** Bi-In-Sn mixing enthalpies on the three measured sections are exothermic (negative). The Bi–Sn **binary**, by contrast, is endothermic in Table II of the paper. That distinction matters later when Dataset B includes the Bi–Sn edge.

### 2.4 Partial vs integral mixing enthalpy

When a small amount of indium is dropped into a Bi–Sn bath, the heat associated with that drop is used to estimate the **partial** molar enthalpy of indium, \(\Delta \bar{H}_{\mathrm{In}}\) (Equation (2) in the paper). That quantity describes the effect of adding indium at one composition.

The **integral** molar enthalpy of mixing, \(\Delta_{\mathrm{mix}}H\), describes the whole alloy relative to the pure liquids (Equation (3) in the paper). Table III lists both.

The target for the future ML model is the **integral** value:

\[
\Delta_{\mathrm{mix}}H \quad \text{(J mol}^{-1}\text{)}
\]

not \(\Delta \bar{H}_{\mathrm{In}}\). Partial enthalpy is kept in the experimental file because it is in Table III, but it is not the prediction target.

---

## 3. Understanding the professor’s paper

The parts of the PDF that this project actually uses are summarised below. This is not a rewrite of the whole article.

| Item | From the PDF |
| --- | --- |
| Title | Measurements of Mixing Enthalpy for a Lead-Free Solder Bi-In-Sn System |
| Authors | M.R. Kumar, S. Mohan, C.K. Behera |
| Year | 2019 (*J. Electron. Mater.* 48, 8096–8106) |
| Experimental objective | Measure integral mixing enthalpy of liquid Bi-In-Sn by drop calorimetry and fit a ternary RKM polynomial |
| Materials (Table I) | In ingot, Sn shots, Bi lumps; all 99.999 wt.%, Johnson Matthey. Argon from Indian Oxygen Limited |
| Atmosphere | Argon, \(p = 0.1\) MPa (Table III headers) |
| Calorimeter | MHTC 96 Line Evo (Setaram), drop technique |
| Calibration | \(\alpha\)-Al\(_2\)O\(_3\) needles (NIST) dropped at the end of each run; calibration constant \(K\) |
| Drop temperature \(T_D\) | 298 K |
| Bath temperatures \(T_M\) | 767 K, 813 K, 855 K |
| Software | CALISTO (integration of heat-flow peaks) |
| Overall calorimeter uncertainty | 10–12% (Results and Discussion) |
| Calibration error | less than \(\pm 1.5\%\) |
| \(u(T_M)\) | 1 K (Table III footnote a) |

Indium was dropped into a liquid Bi–Sn bath. The three cross-sections are lines in the Gibbs triangle, not scattered ternary points:

1. \((Sn_{0.33}Bi_{0.67})_{1-x}In_x\) — Bi-rich Bi–Sn bath; indium mole fraction \(x\) increases toward pure In.
2. \((Sn_{0.50}Bi_{0.50})_{1-x}In_x\) — equiatomic Bi–Sn bath.
3. \((Sn_{0.67}Bi_{0.33})_{1-x}In_x\) — Sn-rich Bi–Sn bath.

There are **nine** experimental series (three sections × three temperatures). The paper reports that \(\Delta_{\mathrm{mix}}H\) is almost temperature-independent between 767 K and 855 K, that the ternary mixing is exothermic, and that the minimum along the sections is near \(x_{\mathrm{In}} \approx 0.55\) (the Bi-rich section is described as almost symmetric). Minima become less exothermic as the Sn/Bi ratio increases. Iso-enthalpy curves at 813 K (Fig. 9) bend toward the Bi–In binary.

Equation (4) is the RKM polynomial used to obtain ternary interaction parameters by least squares. Binary \(L\) parameters in Table IV are taken from the authors’ unpublished work; ternary \(L\) parameters are fitted to this calorimetric study.

---

## 4. Experimental data extraction

The experimental file is built from **Table III** only (pages 8100–8103 of the PDF). Table II (binary Bi–Sn) was read for context and is used inside Equation (3) in the paper, but it is not copied into our ternary experimental CSV.

**Number of original experimental observations = 104.**

| Series | Cross-section | \(T_M\) (K) | \(N\) |
| --- | --- | --- | --- |
| S1 | \((Sn_{0.33}Bi_{0.67})_{1-x}In_x\) | 767 | 11 |
| S2 | same | 813 | 11 |
| S3 | same | 855 | 12 |
| S4 | \((Sn_{0.50}Bi_{0.50})_{1-x}In_x\) | 767 | 12 |
| S5 | same | 813 | 11 |
| S6 | same | 855 | 11 |
| S7 | \((Sn_{0.67}Bi_{0.33})_{1-x}In_x\) | 767 | 12 |
| S8 | same | 813 | 12 |
| S9 | same | 855 | 12 |
| **Total** | | | **104** |

Table III columns that were extracted (where present) include: moles of indium in the drop, \(x_{\mathrm{In}}\), \(u(x_{\mathrm{In}})\), heat effect, heat-effect uncertainty, heat of reaction, partial enthalpy of In, and integral \(\Delta_{\mathrm{mix}}H\).

**ML target:** integral mixing enthalpy \(\Delta_{\mathrm{mix}}H\), stored as `integral_mixing_enthalpy_J_per_mol` (and as `delta_mix_H_J_per_mol` in the combined files).

**Not the ML target:** `partial_enthalpy_In_J_per_mol`.

The numeric table is stored in `data/original_experimental_data.csv`.

---

## 5. PDF extraction issues

Text extracted from the PDF does not always keep minus signs. In Table III the minus often appears as a replacement character. If those signs were ignored, every ternary mixing enthalpy would look positive, which contradicts Figures 2–4 and the paper’s statement that the mixing is exothermic.

What was done:

1. Numbers were read from Table III.
2. Signs were checked on the **rendered PDF pages**, not taken on trust from OCR.
3. Integral \(\Delta_{\mathrm{mix}}H\) values in the experimental file are negative, matching the printed table.
4. A few In-rich **partial** enthalpies are positive in the PDF (and in our file). Those signs were also left as printed. Integral values on the same rows remain negative.
5. Each experimental row is recorded in `data/extraction_audit.csv` (208 audit rows: enthalpy-sign note plus composition-reconstruction note per observation). The audit does not invent new numbers. For enthalpy it records that the printed minus was restored and that the numeric magnitude was not changed.

This was done so that a later reader can see what was checked, instead of finding silently edited science data.

A second extraction issue is composition. Table III reports \(x_{\mathrm{In}}\) but not \(x_{\mathrm{Bi}}\) and \(x_{\mathrm{Sn}}\). The section labels use rounded bath ratios 0.33/0.67, 0.50/0.50 and 0.67/0.33. The headers also give starting amounts \(n_{\mathrm{Bi}}\) and \(n_{\mathrm{Sn}}\). Those moles were used to rebuild the actual bath ratio (Section 6). The rounded labels and the mole-based ratios are close but not identical:

| Label in the paper | Starting moles (Table III header) | Mole-based Bi : Sn |
| --- | --- | --- |
| 0.67 Bi / 0.33 Sn | \(n_{\mathrm{Bi}}=0.002088\), \(n_{\mathrm{Sn}}=0.00104\) | about **0.6675 : 0.3325** |
| 0.50 / 0.50 | \(n_{\mathrm{Bi}}=n_{\mathrm{Sn}}=0.001699\) | **0.50 : 0.50** |
| 0.33 Bi / 0.67 Sn | \(n_{\mathrm{Bi}}=0.001238\), \(n_{\mathrm{Sn}}=0.002475\) | about **0.3334 : 0.6666** |

This matters because the experimental points lie on the real bath lines, not on a slightly shifted 0.33/0.67 line. Synthetic Dataset A was generated on the same mole-based lines so that it sits on top of the measurements.

---

## 6. Composition reconstruction

For a section \((Sn_a Bi_b)_{1-x}In_x\) with \(a+b=1\):

\[
x_{\mathrm{In}} = x, \qquad
x_{\mathrm{Sn}} = a(1-x), \qquad
x_{\mathrm{Bi}} = b(1-x)
\]

and therefore

\[
x_{\mathrm{Bi}} + x_{\mathrm{In}} + x_{\mathrm{Sn}} = 1.
\]

In the code, \(a\) and \(b\) are not the rounded labels. They are

\[
b = \frac{n_{\mathrm{Bi}}}{n_{\mathrm{Bi}}+n_{\mathrm{Sn}}}, \qquad
a = \frac{n_{\mathrm{Sn}}}{n_{\mathrm{Bi}}+n_{\mathrm{Sn}}}
\]

taken from the series header, and \(x\) is the tabulated \(x_{\mathrm{In}}\). Function: `reconstruct_composition` in `scripts/generate_synthetic_data.py`.

Every experimental row was required to satisfy the sum within \(10^{-10}\). All 104 rows passed. The later validation script repeats the sum check with tolerance \(10^{-8}\) on the written CSV. That check also passed.

Mole fractions were also required to lie in \([0,1]\). All experimental rows passed.

---

## 7. Original dataset

**File:** `data/original_experimental_data.csv`

This file contains **only** the 104 Table III observations. No RKM values and no synthetic rows are written into it. That is deliberate. If experimental and synthetic numbers share one “raw data” file, it becomes easy to treat model output as calorimetry.

Provenance:

- `source = paper_experiment`
- `source_table = Table III`
- `id = EXP_0001` … `EXP_0104`

Each row also has `series_id` (1–9), `cross_section`, `temperature_K`, drop size `nIn_mol`, reconstructed `xBi`, `xIn`, `xSn`, tabulated uncertainties and heat quantities, `partial_enthalpy_In_J_per_mol`, and the target `integral_mixing_enthalpy_J_per_mol`.

The `notes` field is used when a row needs a comment (for example a positive partial enthalpy that was confirmed on the PDF). It is not used to change the target.

---

## 8. RKM model

The paper already represents \(\Delta_{\mathrm{mix}}H\) with a Redlich–Kister–Muggianu polynomial (Equation (4)). We implemented that polynomial so that:

1. we could check that we had read Table IV correctly, by comparing the model with Table III, and
2. we could evaluate the same function on a denser composition grid to build **labelled synthetic** data.

The model has two kinds of term.

**Binary terms.** For each pair Bi–In, Bi–Sn and In–Sn there is a Redlich–Kister expansion

\[
x_i x_j \sum_{\nu} L^{(\nu)}_{i:j} (x_i - x_j)^{\nu}.
\]

These \(L\) coefficients describe how far that binary liquid is from ideal mixing. Bi–In is strongly exothermic (\(L^{(0)}\) negative). Bi–Sn is endothermic (\(L^{(0)}\) positive), which matches Table II. In–Sn is moderately exothermic.

**Ternary term.** A single extra term

\[
x_i x_j x_k \left( L^{(0)} x_i + L^{(1)} x_j + L^{(2)} x_k \right)
\]

corrects for three-component interactions. The \(L^{(t)}\) values were obtained in the paper by least squares.

**Why this is useful for synthetic data.** Once the polynomial is fixed, any valid \((x_{\mathrm{Bi}}, x_{\mathrm{In}}, x_{\mathrm{Sn}})\) can be given a \(\Delta_{\mathrm{mix}}H\) without a new drop. That is convenient, but those numbers are **not** new experiments. They follow the polynomial, including in parts of the triangle where the calorimeter was never used.

**Code:** `src/rkm_model.py`

The public function is `rkm_delta_mix_h(xBi, xIn, xSn, T=None)`. It:

- checks that mole fractions are in range and sum to 1,
- adds the three binary Redlich–Kister contributions,
- adds the ternary term,
- returns \(\Delta_{\mathrm{mix}}H\) in J mol\(^{-1}\).

`T` is accepted so that a call can include temperature, but **it is not used**. Table IV has no temperature-dependent coefficients, and the paper treats mixing enthalpy as almost independent of \(T\) in this range. Evaluating the same composition at 767 K, 813 K and 855 K therefore gives the same RKM number.

---

## 9. RKM parameters

Parameters are taken from Table IV of the PDF (page 8106). Signs were checked on the rendered table.

**Binary parameters (J/mol)**

| Interaction | \(\nu\) | \(L^{(\nu)}\) (J/mol) |
| --- | --- | --- |
| \(L_{\mathrm{Bi-In}}\) | 0 | −6617 |
| | 1 | 133 |
| | 2 | 2066 |
| \(L_{\mathrm{Bi-Sn}}\) | 0 | 520 |
| | 1 | 235 |
| | 2 | −157 |
| | 3 | −119 |
| \(L_{\mathrm{In-Sn}}\) | 0 | −1980 |
| | 1 | −893 |

**Ternary parameters listed in Table IV as \(L_{\mathrm{Bi-Sn-In}}^{(t)}\) (J/mol)**

| \(t\) | Value (J/mol) |
| --- | --- |
| 0 | −3874 |
| 1 | −12777 |
| 2 | 24187 |

How \(t = 0,1,2\) is attached to \(x_i, x_j, x_k\) is the index-order question in the next section. The numerical values themselves were not altered.

---

## 10. RKM index-order issue

Equation (4) writes the ternary contribution as \(L^{(0)} x_i + L^{(1)} x_j + L^{(2)} x_k\). Table IV names the parameter \(L_{\mathrm{Bi-Sn-In}}\). A literal reading would be

\[
(i,j,k) = (\mathrm{Bi},\ \mathrm{Sn},\ \mathrm{In}).
\]

That assignment was tried first. It did **not** reproduce the paper’s statement that the fitted curve is close to experiment, and it did not match the shape of Figure 10 (equiatomic section at 813 K). The Bi-rich section in particular showed a large systematic offset.

Other assignments of \((i,j,k)\) were then compared with all 104 Table III points. The assignment that recovered the measured curves, the Bi-rich vs Sn-rich depth order, and an \(R^2\) of 0.951 was

\[
(i,j,k) = (\mathrm{In},\ \mathrm{Bi},\ \mathrm{Sn}),
\]

so that

\[
\Delta H_{\mathrm{ternary}}
= x_{\mathrm{In}}\, x_{\mathrm{Bi}}\, x_{\mathrm{Sn}}
\left( L^{(0)} x_{\mathrm{In}} + L^{(1)} x_{\mathrm{Bi}} + L^{(2)} x_{\mathrm{Sn}} \right).
\]

This is the assignment stored as `TERNARY_COMPONENT_ORDER = ("In", "Bi", "Sn")` in `rkm_model.py`.

This was **not** a silent edit of Table IV. The three numbers −3874, −12777 and 24187 are unchanged. Only the mapping from those numbers onto mole fractions was chosen so that Equation (4) behaves like the paper’s own fitted curves. The paper does not print a worked numerical example that would remove the ambiguity, so we do not call it an author error. We treat it as an indexing/order ambiguity found during implementation.

It is written up in:

- `docs/rkm_model.md`
- `reports/rkm_validation.md`

Synthetic data were generated only after this assignment was in place and the validation metrics below were obtained.

---

## 11. RKM validation

The implemented polynomial was evaluated at every experimental composition **before** any synthetic file was written. `scripts/generate_synthetic_data.py` stops if the qualitative shape checks fail or if \(R^2 < 0.85\).

**Overall (N = 104):**

| Metric | Value |
| --- | --- |
| MAE | 57.13 J/mol |
| RMSE | 73.49 J/mol |
| \(R^2\) | 0.9510 |

Meaning, in ordinary language:

- **MAE** is the average absolute difference between RKM and Table III. About 57 J/mol on values that range from roughly −64 to −1413 J/mol.
- **RMSE** weights larger errors more than MAE. About 73 J/mol here.
- **\(R^2 = 0.951\)** means the model explains about 95.1% of the **variation** in the experimental integral enthalpies in this comparison. It is **not** “95.1% accurate” and it is not a classification score.

The fit is not equally good on every series. Series 1 (Bi-rich, 767 K) is the weakest (\(R^2 = 0.8131\), MAE = 134.91 J/mol). The equiatomic section at 813 K, which is the section shown with the theoretical curve in Fig. 10, is much closer (Series 5: \(R^2 = 0.9799\)). The paper already notes larger deviations at some indium-rich compositions.

Qualitative checks that passed:

- mixing remains exothermic on the three sections in the measured \(x_{\mathrm{In}}\) window,
- model minima at \(x_{\mathrm{In}} \approx 0.50\), 0.54 and 0.59 (paper: near 0.55),
- Bi-rich minimum deeper than equiatomic, which is deeper than Sn-rich.

Figures used for this check include `figures/rkm_vs_experiment.png` and `figures/rkm_vs_experiment_cross_sections.png`.

Full tables: `reports/rkm_validation.md`.

---

## 12. Synthetic data generation

Two synthetic datasets were produced after the RKM check. Both are tagged `source = synthetic` and `synthetic = True`.

### Dataset A — `data/synthetic_cross_sections.csv`

**7,389 rows.**

This set stays on the **same three experimental cross-sections** (using the mole-based Bi:Sn bath ratios). \(x_{\mathrm{In}}\) is filled on a grid from 0.09 to 0.91 in steps of 0.001, at 767 K, 813 K and 855 K. That is denser than Table III, but it does not leave the measured lines.

The enthalpy is the RKM value at that composition, plus a small Gaussian perturbation whose standard deviation is the series-specific \(u(\Delta_{\mathrm{mix}}H)\) from Table III footnote a. The generation method written in the file is `RKM_interpolation_plus_noise`.

These rows are **synthetic**. They are not extra calorimeter drops.

### Dataset B — `data/synthetic_full_ternary.csv`

**15,453 rows.**

This set covers the whole composition triangle with a barycentric (simplex) grid of step 0.01: every combination of non-negative mole fractions that sum to 1. The same grid is written at 767 K, 813 K and 855 K. Because the RKM function does not depend on \(T\), the three temperatures have the same thermodynamic value at a given composition. Temperature is stored as a label only.

The generation method in this file is `RKM_full_ternary_grid` (no noise). Interior compositions are not calorimetry. Most rows are tagged `unsampled_ternary_region`. Rows near the three measured lines are tagged `near_experimental_cross_section` or `experimental_anchor` depending on distance to Table III points.

**Dataset B is not experimental data.**

---

## 13. Deterministic vs noisy data

Two extra files collect A and B together under a single noise policy:

| File | Contents |
| --- | --- |
| `data/synthetic_rkm_deterministic.csv` | Dataset A + Dataset B with **no** random noise (`noise_added_J_per_mol = 0`) |
| `data/synthetic_rkm_noisy.csv` | Dataset A + Dataset B with Gaussian noise |

Deterministic output is exactly the polynomial. Noisy output is

\[
y_{\mathrm{synthetic}} = y_{\mathrm{RKM}} + \varepsilon, \qquad \varepsilon \sim \mathrm{Normal}(0,\sigma)
\]

with \(\sigma\) taken from the paper’s tabulated uncertainties (Section 14), not from a round number such as 0.1 or 0.5.

**Random seed: 42.**

The seed is fixed so that `python scripts/generate_synthetic_data.py` writes the same noisy files each time. That matters if we later compare ML runs: changing noise from run to run would mix data uncertainty with training luck.

A noise-free mode exists specifically so we can train or plot against a smooth thermodynamic surface when we want that, without pretending the noise is a new measurement.

Each file contains 22,842 synthetic rows (7,389 + 15,453).

---

## 14. Experimental uncertainty

The paper gives two kinds of uncertainty statement.

1. **Overall calorimeter uncertainty: 10–12%** (construction, calibration, integration, dissolution, impurities).
2. **Series-specific** \(u(\Delta_{\mathrm{mix}}H)\) in Table III footnote a (kJ mol\(^{-1}\)):

| Series | \(u(\Delta_{\mathrm{mix}}H)\) (kJ/mol) | used as \(\sigma\) (J/mol) |
| --- | --- | --- |
| 1 | 0.0233 | 23.3 |
| 2 | 0.0197 | 19.7 |
| 3 | 0.01608 | 16.08 |
| 4 | 0.01817 | 18.17 |
| 5 | 0.0144 | 14.4 |
| 6 | 0.0127 | 12.7 |
| 7 | 0.0132 | 13.2 |
| 8 | 0.0105 | 10.5 |
| 9 | 0.0079 | 7.9 |

Dataset A uses the matching series \(\sigma\). The noisy copy of Dataset B uses the mean of these nine values, **15.106 J/mol**.

The 10–12% overall figure was **not** used as \(\sigma\). On a mixing enthalpy of 1000 J/mol that would be 100–120 J/mol of scatter, which is larger than the wiggles in Figures 2–4 and would wash out the composition dependence. The series \(u(\Delta_{\mathrm{mix}}H)\) values are the ones that belong to the integral enthalpy column we are copying.

Noise is only there to give synthetic rows a similar **size** of scatter to the tabulated standard uncertainties. It does not turn those rows into measurements.

---

## 15. Dataset boundaries

| Set | Composition coverage |
| --- | --- |
| Experiment (Table III) | \(x_{\mathrm{In}}\) from **0.0953 to 0.9073**, only on three lines |
| Dataset A | \(x_{\mathrm{In}}\) from **0.09 to 0.91** on those same three lines (interpolation range; not extended to pure In or to the Bi–Sn edge) |
| Dataset B | **full ternary simplex**, including binary edges and the three pure metals |

Language used in the files:

- `experimental_anchor` — at or very near a Table III composition
- `near_experimental_cross_section` — close to one of the three measured lines
- `unsampled_ternary_region` — remainder of the triangle (and most of Dataset B: 11,856 of 15,453 rows)

Unsampled regions are **model-generated**. They are not experimentally validated. Fig. 1 of the paper shows why: the calorimeter was only run along three paths.

---

## 16. Negative and positive values

| Set | \(\Delta_{\mathrm{mix}}H\) range (J/mol) |
| --- | --- |
| Experimental Table III | **−1413.0 to −64.32** (all 104 values negative) |
| Dataset A (noisy cross-sections) | about **−1372 to +9.8** |
| Dataset B (full ternary, deterministic) | about **−1654 to +135** |

Experimental ternary observations on the three sections are exothermic, as the paper states.

Dataset A contains a **small number of positive values** (4 rows; largest about +9.8 J/mol). They sit near the In-rich end of the grid, where the RKM curve is already close to zero, so a draw from \(\mathrm{Normal}(0,\sigma)\) can cross the axis. These rows were **not** clipped. They should be looked at again before final ML training (for example, whether to use the deterministic A file, to reject \(y>0\) on those sections, or to leave them as a test of noise). That decision has **not** been made yet.

Dataset B can be positive on the **Bi–Sn binary edge**. That is expected from the paper: Table II and \(L_{\mathrm{Bi-Sn}}^{(0)}=+520\) J/mol both describe endothermic Bi–Sn mixing. The largest Dataset B value, about +135 J/mol, is in that binary-like region, not a sign error in the ternary well. Pure-element corners are essentially 0, which the validation script checks.

Positive values were left in the files on purpose. Hiding them would make the synthetic data look cleaner than the model (and the binaries) actually are.

---

## 17. Validation checks

**Script:** `scripts/validate_dataset.py`

**Last run recorded for this report:** `DATASET VALIDATION: PASS`.

The script does **not** retrain models and does not regenerate data. It loads the CSVs and the RKM function and tests them.

Checks that the script actually performs (grouped):

- **Files present:** the four data CSVs listed in Section 18.
- **Experimental provenance:** 104 rows; `source = paper_experiment`; `source_table = Table III`; unique ids; temperatures only in {767, 813, 855}; all integral \(\Delta_{\mathrm{mix}}H\) negative; range −1413.0 to −64.32 J/mol; target has no NaN; audit file non-empty; partial |H_In| larger on average than |ΔmixH| (columns not swapped); nine series; original file contains no synthetic rows.
- **Composition:** \(x_{\mathrm{Bi}}+x_{\mathrm{In}}+x_{\mathrm{Sn}}=1\); each mole fraction in [0, 1]; no NaN or inf — for experiment, Dataset A, and Dataset B.
- **Cross-section ratios:** Bi/(Bi+Sn) constant on each experimental section and within 0.015 of the labelled 0.67 / 0.50 / 0.33 (actual values 0.6675, 0.5000, 0.3334). Dataset A is checked the same way.
- **RKM vs experiment:** MAE = 57.13 J/mol, RMSE = 73.49 J/mol, R² = 0.9510.
- **Dataset A:** `source = synthetic`; `synthetic = True`; 7389 rows; temperatures allowed; `base_model_prediction` matches `rkm_delta_mix_h` on a sample of 400 rows; |noise residual| < 200 J/mol on that sample.
- **Dataset B:** `source = synthetic`; 15453 rows; method name contains `RKM_full_ternary`; `region_type` in the three allowed strings; both unsampled and near-experimental tags present; targets match RKM on a 400-row sample (deterministic grid); near-pure vertices present with \(\Delta_{\mathrm{mix}}H \approx 0\).
- **Reproducibility:** RKM returns the same number for the same composition; sample point (0.25, 0.50, 0.25) is unchanged; `RANDOM_SEED == 42` in the validator.

The script does not check ML accuracy, because no ML model has been trained.

---

## 18. File structure

Current layout (see also `README.md`):

```text
project/
├── professor_paper.pdf
├── data/
│   ├── original_experimental_data.csv
│   ├── extraction_audit.csv
│   ├── synthetic_cross_sections.csv
│   └── synthetic_full_ternary.csv
├── src/
│   └── rkm_model.py              # Equation (4)
├── scripts/
│   ├── generate_synthetic_data.py
│   └── validate_dataset.py
├── figures/
├── docs/
│   └── rkm_model.md
├── reports/
│   ├── rkm_validation.md
│   └── project_work_report.md    # this file
└── requirements.txt
```

**Data files, briefly**

- `original_experimental_data.csv` — Table III only (104 rows).
- `extraction_audit.csv` — sign and composition reconstruction notes.
- `synthetic_cross_sections.csv` — Dataset A (7389 noisy cross-section points).
- `synthetic_full_ternary.csv` — Dataset B (15453 deterministic grid points).

**Figures** kept for explanation: experimental cross-sections, RKM vs experiment (parity and sections), experimental vs synthetic overlay, and the 813 K ternary enthalpy map.

`docs/rkm_model.md` states the equation, Table IV parameters, and the ternary index convention. The ML leakage warning is in that file and in Section 21 below. `requirements.txt` lists numpy and pandas.

---

## 19. Reproducibility

From the project root:

```text
python scripts/validate_dataset.py
```

`validate_dataset.py` only reads what is already on disk and prints `DATASET VALIDATION: PASS` or `FAIL`.

`generate_synthetic_data.py` can rebuild Dataset A and Dataset B from the experimental CSV and the RKM model (seed 42). It does not rewrite the experimental file. Do not run it unless you intend to overwrite the synthetic CSVs.

**Random seed: 42** (noisy synthetic rows only). The RKM polynomial itself is deterministic.

This report was written **without** re-running generation, as requested, so that the datasets on disk stay as they are.

---

## 20. Data provenance

Two chains exist. They must not be merged in the reader’s mind.

```text
PDF Table III
    → extraction + sign check + composition reconstruction
    → data/original_experimental_data.csv
    → source = paper_experiment
```

```text
PDF Equation (4) + Table IV
    → rkm_delta_mix_h (after index-order check vs Table III)
    → dense grid / cross-section grid  [+ optional noise]
    → synthetic CSV files
    → source = synthetic
```

The two chains stay in separate files (`original_experimental_data.csv` vs the two synthetic CSVs). Each synthetic row keeps `source`, `synthetic`, `generation_method` and `region_type`.

RKM-generated values cannot be treated as independent experimental evidence. They are the same polynomial that was fitted (in the paper) using those experiments, plus our implementation choices. Using them as if they were extra calorimetry would count the model twice.

---

## 21. Data leakage / ML warning

The same warning is recorded in `docs/rkm_model.md`. The main point is simple.

If synthetic labels come from RKM, and an ML model is trained and tested only on those labels, a good test score mainly shows that the ML model copied the polynomial. It does **not** show that the ML model predicts drop-calorimetry.

The warning also lists practical mistakes to avoid later:

- training and testing only inside Dataset B,
- letting synthetic rows dominate a mixed test set,
- using `base_model_prediction_J_per_mol` as an input feature when the target is also RKM-based,
- ignoring `region_type` and reporting interior-triangle error as “validated”,
- treating temperature as a useful feature on synthetic rows (the generator writes three temperatures with the same RKM value),
- treating noise as a substitute for a held-out experiment.

What should remain the external reference is the 104 rows with `source = paper_experiment`.

That protocol has **not** been run yet. The warning is documentation for the next stage, not a result.

---

## 22. Current limitations

The following limitations are supported by the files and the paper:

1. Only **104** original experimental observations are available.
2. Those observations lie on **three cross-sections**, not across the full triangle.
3. Full-ternary synthetic data (Dataset B) come from the **RKM model**, not from new measurements.
4. Synthetic targets therefore inherit Table IV, the unpublished binary \(L\) parameters, the ternary index assignment we selected, and the assumption of no temperature dependence in 767–855 K.
5. The only experimental temperatures in this project are **767 K, 813 K and 855 K**.
6. The paper reports nearly temperature-independent mixing enthalpy in that window; our RKM code does not contain a \(T\) term.
7. Dataset A has a few **positive** noisy values near the In-rich end (4 rows, up to about +9.8 J/mol). They should be reviewed before final training. They were not clipped.
8. Random train/test splits on a concatenated experimental+synthetic table would be misleading: most rows would be synthetic, neighbouring grid points are almost the same polynomial, and experimental points on a section are sequential drops. Grouped or experimental hold-out splits are needed later. **Not implemented yet.**
9. Synthetic data must not be treated as independent experimental evidence (Section 20–21).
10. Series 1 is the weakest RKM comparison (MAE 134.91 J/mol). Using RKM as a generator does not remove that residual.
11. No ML feature table has been trained. Combined / ML-ready CSVs were removed because they can be rebuilt later from the three primary files.

---

## 23. What has not been done yet

The following stages are **not implemented yet**:

- ML model training (linear models, trees, neural nets, or any other regressor)
- model comparison
- hyperparameter tuning
- experimental-only benchmark
- synthetic-only benchmark
- combined experimental + synthetic benchmark
- cross-section holdout validation
- composition-grouped or temperature-grouped splits
- Gaussian Process (or other) uncertainty modelling
- final AI surrogate selection
- candidate composition screening
- any claim that the surrogate replaces calorimetry

Feature tables for a later comparison of composition-only vs composition+temperature inputs can be built from the three primary CSVs when ML work starts. They are not stored in this folder.

---

## 24. Current status

| Stage | Status |
| --- | --- |
| Paper study | Completed |
| Experimental data extraction (Table III) | Completed |
| Data cleaning / sign check / audit | Completed |
| Composition reconstruction | Completed |
| RKM implementation (Equation 4, Table IV) | Completed |
| RKM validation vs Table III | Completed |
| Synthetic Dataset A | Completed |
| Synthetic Dataset B | Completed |
| Deterministic and noisy combined synthetic files | Completed |
| Dataset validation script (87 checks, PASS) | Completed |
| Documentation (dictionary, leakage warning, RKM notes) | Completed |
| ML-ready CSV export | Completed (files only) |
| ML model development | **Not started** |
| Experimental-only / synthetic-only / combined ML benchmarks | **Not started** |
| Final surrogate selection | **Not started** |
| Experimental ML validation | **Not started** |

“Completed” above means the corresponding files exist and the validation script passed. It does not mean the scientific questions are finished.

---

## 25. Next recommended stage

The next stage should be a **careful ML benchmark**, not more data generation.

That work has **not** been performed. A reasonable plan, to be implemented later, is:

1. **Experimental-only training** — train on a subset of the 104 Table III rows, test on held-out experimental rows.
2. **Synthetic-only training** — train on RKM synthetic rows, still **test on held-out experimental** rows (this shows how well the polynomial, as seen by an ML model, matches calorimetry).
3. **Experimental + synthetic training** — add Dataset A (and, if used at all, Dataset B with `region_type` in mind), still evaluate on experimental hold-outs.

Hold-outs should respect the experimental design (for example leaving out one cross-section or one temperature), rather than shuffling neighbouring points at random.

Two feature sets are already prepared for a later comparison:

- composition only: \(x_{\mathrm{Bi}}, x_{\mathrm{In}}, x_{\mathrm{Sn}}\)
- composition + `temperature_K`

On experimental rows, adding temperature may or may not help; the paper says mixing enthalpy is almost independent of \(T\) here. On synthetic rows, temperature cannot help for physical reasons, because it was never put into the labels.

Until that benchmark exists, there is no trained surrogate to report.

---

## 26. Final summary

| Quantity | Value |
| --- | --- |
| Original experimental observations | **104** |
| Synthetic Dataset A | **7,389** |
| Synthetic Dataset B | **15,453** |
| Temperatures | **767 K, 813 K, 855 K** |
| Experimental \(x_{\mathrm{In}}\) | **0.0953–0.9073** |
| Experimental \(\Delta_{\mathrm{mix}}H\) | **−1413.0 to −64.32 J/mol** |
| RKM MAE | **57.13 J/mol** |
| RKM RMSE | **73.49 J/mol** |
| RKM \(R^2\) | **0.9510** |
| Dataset validation | **PASS, 87 checks** |
| Random seed | **42** |

At this stage, the project has completed the experimental-data reconstruction and thermodynamics-based synthetic-data generation stage. The machine-learning surrogate-model stage has not yet been completed.
