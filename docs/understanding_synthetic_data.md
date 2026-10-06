# Understanding the Synthetic Data

> **Historical note:** This document describes an earlier stage of the project, written when the synthetic datasets were created and before any ML model existed. RKM-derived synthetic datasets were investigated as a possible augmentation strategy. Under strict LOCSO validation, augmentation did not improve prediction of the real experimental observations (pooled MAE 54.32 J/mol for real + Dataset A vs 41.10 J/mol for real data only; `reports/synthetic_data_ablation.md`). Therefore synthetic data are not used to train the final predictor and are retained only for exploratory analysis. The final predictor is an experimental-only Direct Polynomial Degree-2 model (xBi, xIn, T) trained on the 104 experimental observations. The description of the datasets below remains accurate.

This note explains the synthetic datasets that already exist in this repository. It is based on the actual CSV files and `scripts/generate_synthetic_data.py`. Nothing here was regenerated.

---

## 1. Why synthetic data are needed

The professor’s paper gives us **104 real calorimetry measurements**. That sounds like a lot, but in composition space they only cover **three thin lines** inside the Bi–In–Sn triangle — not the whole alloy system.

Our long-term goal is a surrogate model that can estimate mixing enthalpy at many compositions. At the time, the idea was that denser `(composition → ΔmixH)` pairs than Table III provides might help such a model; this was later tested and not adopted (see the historical note above).

**Synthetic data** in this project means:

> A composition where **ΔmixH was calculated by the validated RKM polynomial**, not measured in the calorimeter.

They are useful because RKM can be evaluated anywhere on the triangle, although it is only checked against experiment on the three measured lines. Synthetic rows give us dense coverage along the measured lines (Dataset A) and across the full triangle (Dataset B).

They are **not** a substitute for experiments. They are model-generated labels.

---

## 2. The original 104 experimental points

Source: `data/original_experimental_data.csv`, extracted from Table III of `professor_paper.pdf`.

Each row is a **drop-calorimetry measurement**: indium was dropped into a liquid Bi–Sn bath at fixed temperature. The target column is `integral_mixing_enthalpy_J_per_mol` — the integral molar mixing enthalpy \(\Delta_{\mathrm{mix}}H\) in J/mol. This is **not** the partial enthalpy of indium (`partial_enthalpy_In_J_per_mol` is a different column).

**Where they sit in composition space**

The paper did not scatter points randomly across the triangle. It measured along **three composition cross-sections** (three lines from the Bi–Sn edge toward the In corner), at **three temperatures**:

| Temperature | 767 K | 813 K | 855 K |
| --- | --- | --- | --- |
| Points per section (approx.) | 11–12 | 11–12 | 11–12 |

Breakdown in our file:

| Cross-section | Total rows |
| --- | --- |
| `(Sn0.33Bi0.67)1-xInx` (Bi-rich bath) | 34 |
| `(Sn0.50Bi0.50)1-xInx` (equiatomic bath) | 34 |
| `(Sn0.67Bi0.33)1-xInx` (Sn-rich bath) | 36 |
| **Total** | **104** |

Experimental \(\Delta_{\mathrm{mix}}H\) range: **−1413.0 to −64.32 J/mol** (all exothermic on these sections).

The actual Bi:Sn bath ratios from starting moles are not exactly the rounded labels:

- Bi-rich: \(x_{\mathrm{Bi}}/(x_{\mathrm{Bi}}+x_{\mathrm{Sn}}) \approx 0.6675\) (labelled 0.67)
- Equiatomic: **0.5000**
- Sn-rich: **0.3334** (labelled 0.33)

---

## 3. What a composition cross-section means

A cross-section is a **line** in the ternary diagram, not the whole triangle.

Take the equiatomic section from the paper:

\[
(Sn_{0.50}Bi_{0.50})_{1-x}In_x
\]

Read this as: along the Bi–Sn edge the bath is 50% Bi and 50% Sn (by mole). Then indium is added. Only **\(x_{\mathrm{In}}\)** changes; Bi and Sn stay in that fixed ratio among themselves.

Using the label ratio 0.50 : 0.50:

\[
x_{\mathrm{In}} = x
\]
\[
x_{\mathrm{Bi}} = 0.50\,(1 - x_{\mathrm{In}})
\]
\[
x_{\mathrm{Sn}} = 0.50\,(1 - x_{\mathrm{In}})
\]

Check the sum:

\[
x_{\mathrm{Bi}} + x_{\mathrm{In}} + x_{\mathrm{Sn}}
= 0.50(1-x) + x + 0.50(1-x)
= 1
\]

**What stays fixed:** the Bi : Sn ratio in the bath (here 1 : 1).

**What changes:** how much indium was added (\(x_{\mathrm{In}}\)).

**Physical picture:** imagine a line starting on the Bi–Sn edge and moving toward pure In. The calorimeter walked along that line at discrete \(x_{\mathrm{In}}\) values. Table III has roughly 11–12 drops per section per temperature — sparse compared to the synthetic grid we built later.

Example from a real experimental row (`EXP_0051`, equiatomic, 813 K):

- \(x_{\mathrm{Bi}} = 0.2509\)
- \(x_{\mathrm{In}} = 0.4982\)
- \(x_{\mathrm{Sn}} = 0.2509\)
- experimental \(\Delta_{\mathrm{mix}}H = -1017.0\) J/mol

Here \(x_{\mathrm{Bi}} = x_{\mathrm{Sn}}\) because the bath is 50 : 50.

---

## 4. Dataset A

**File:** `data/synthetic_cross_sections.csv`  
**Rows:** **7389**  
**Method:** `RKM_interpolation_plus_noise`

### How it was generated

From `scripts/generate_synthetic_data.py`:

1. Read the experimental CSV (for bath ratios and anchor tagging).
2. For **each of 3 cross-sections**:
3. For **each of 3 temperatures** (767, 813, 855 K):
4. For **each** \(x_{\mathrm{In}}\) on a grid from **0.09 to 0.91** in steps of **0.001**:
5. Reconstruct \(x_{\mathrm{Bi}}\) and \(x_{\mathrm{Sn}}\) from the bath ratio.
6. Compute RKM → `base_model_prediction_J_per_mol`.
7. Add Gaussian noise → `delta_mix_H_J_per_mol`.

### The arithmetic behind 7389

Grid length:

\[
\text{points per } x_{\mathrm{In}} = \frac{0.91 - 0.09}{0.001} + 1 = 821
\]

Total:

\[
3\ \text{sections} \times 3\ \text{temperatures} \times 821\ x_{\mathrm{In}}\ \text{values}
= 7389
\]

Per section: \(3 \times 821 = 2463\) rows.  
Per section per temperature: **821** rows.

### Deterministic or noisy?

**Noisy.** Every row has non-zero noise (`noise_added_J_per_mol ≠ 0`). The underlying RKM surface is stored separately in `base_model_prediction_J_per_mol`; the target column `delta_mix_H_J_per_mol` is RKM + noise.

There is **one version** on disk: the noisy cross-section file. (An older pipeline also built deterministic/noisy combined files; those were removed in cleanup. Dataset A as stored is always noisy.)

### Important columns

| Column | Meaning |
| --- | --- |
| `source` | Always `synthetic` |
| `synthetic` | Always `True` |
| `generation_method` | `RKM_interpolation_plus_noise` |
| `cross_section` | Which of the three measured lines |
| `temperature_K` | 767, 813, or 855 (label only for RKM — same prediction at all T) |
| `xBi`, `xIn`, `xSn` | Composition |
| `base_model_prediction_J_per_mol` | Pure RKM \(\Delta_{\mathrm{mix}}H\) |
| `noise_added_J_per_mol` | \(\varepsilon \sim N(0,\sigma)\) draw |
| `delta_mix_H_J_per_mol` | **Target** = base + noise |
| `uncertainty_J_per_mol` | \(\sigma\) used for that row (series-specific, J/mol) |
| `region_type` | `experimental_anchor` or `near_experimental_cross_section` (never `unsampled_ternary_region` — all points are on the three lines) |
| `nearest_experimental_id` | Closest Table III row by composition |
| `experimental_anchor_value` | That row’s experimental \(\Delta_{\mathrm{mix}}H\) |
| `synthetic_minus_experimental` | Synthetic target minus that anchor |

Region counts in the actual file:

- `near_experimental_cross_section`: 4377
- `experimental_anchor`: 3012

Dataset A \(\Delta_{\mathrm{mix}}H\) range: about **−1371.9 to +9.8 J/mol** (only 4 rows slightly positive — see Section 9).

---

## 5. One Dataset A example

**Row:** `SYN_A_03693`

| Field | Value |
| --- | --- |
| `cross_section` | `(Sn0.50Bi0.50)1-xInx` |
| `temperature_K` | 813 |
| `xBi` | 0.251 |
| `xIn` | 0.498 |
| `xSn` | 0.251 |
| `base_model_prediction_J_per_mol` (RKM) | −1028.44 J/mol |
| `noise_added_J_per_mol` | −27.92 J/mol |
| `delta_mix_H_J_per_mol` (target) | −1056.36 J/mol |
| `source` | synthetic |
| `synthetic` | True |
| `region_type` | experimental_anchor |
| `nearest_experimental_id` | EXP_0051 |
| `experimental_anchor_value` | −1017.0 J/mol |

**Why this composition was selected**

The generator did not “pick” it individually. It is one grid point: equiatomic section, 813 K, \(x_{\mathrm{In}} = 0.498\) (step 0.001 grid from 0.09 to 0.91). That places it almost exactly on the experimental point EXP_0051 (\(x_{\mathrm{In}} = 0.4982\)).

**How RKM calculated ΔmixH**

Same composition is passed to `rkm_delta_mix_h(0.251, 0.498, 0.251)` → **−1028.44 J/mol**. Then noise from Series 5 (\(\sigma = 14.4\) J/mol) was added: −1028.44 + (−27.92) = −1056.36 J/mol.

**Does it correspond to an experimentally sampled region?**

Yes — it lies on a measured cross-section and is within 0.0002 (composition distance) of EXP_0051. That is why `region_type = experimental_anchor`.

**Why it is still synthetic**

Even on top of a real experimental composition, the **target** is RKM + random noise, not the calorimeter reading (−1017.0 J/mol). The row is tagged `source = synthetic` and the notes say it is not an experimental measurement. If we trained ML on this row’s target, we would be learning the model surface (plus artificial scatter), not copying Table III exactly.

---

## 6. Dataset B

**File:** `data/synthetic_full_ternary.csv`  
**Rows:** **15453**  
**Method:** `RKM_full_ternary_grid`

### What “full ternary composition space” means

Any liquid alloy composition satisfies:

\[
x_{\mathrm{Bi}} + x_{\mathrm{In}} + x_{\mathrm{Sn}} = 1,\quad
x_{\mathrm{Bi}}, x_{\mathrm{In}}, x_{\mathrm{Sn}} \ge 0
\]

That is the **Gibbs triangle**: all possible Bi–In–Sn mixtures. Dataset B places a grid on **every** valid combination at step 0.01, including binary edges and pure-element corners — not just the three cross-sections.

### How the grid was created

Function `simplex_grid(0.01)` in the generation script:

- Set \(n = 1 / 0.01 = 100\).
- For all non-negative integers \(i, j, k\) with \(i + j + k = n\), store
  \[
  x_{\mathrm{Bi}} = i/n,\quad x_{\mathrm{In}} = j/n,\quad x_{\mathrm{Sn}} = k/n.
  \]

That yields **5151** unique compositions on the triangle.

Then repeat for **3 temperatures** (767, 813, 855 K):

\[
5151 \times 3 = 15453
\]

### Deterministic or noisy?

**Deterministic.** `noise_added_J_per_mol = 0` on every row. `delta_mix_H_J_per_mol` equals `base_model_prediction_J_per_mol` equals pure RKM.

RKM does not use temperature, so the three temperature copies of the same composition have **identical** \(\Delta_{\mathrm{mix}}H\). Temperature is stored for bookkeeping only.

Dataset B \(\Delta_{\mathrm{mix}}H\) range: about **−1654.25 to +134.73 J/mol**.

Region counts:

| `region_type` | Rows |
| --- | --- |
| `unsampled_ternary_region` | 11856 |
| `near_experimental_cross_section` | 3147 |
| `experimental_anchor` | 450 |

---

## 7. The ternary composition space

In a ternary diagram, the three corners are:

- **Bi-rich** — pure bismuth (\(x_{\mathrm{Bi}} = 1\))
- **In-rich** — pure indium (\(x_{\mathrm{In}} = 1\))
- **Sn-rich** — pure tin (\(x_{\mathrm{Sn}} = 1\))

**Edges** are binary alloys (one component zero):

- Bi–Sn edge: \(x_{\mathrm{In}} = 0\)
- Bi–In edge: \(x_{\mathrm{Sn}} = 0\)
- In–Sn edge: \(x_{\mathrm{Bi}} = 0\)

**Interior** points have all three components positive.

**Where the 104 experiments sit**

Three lines from the Bi–Sn edge toward In — one for each bath ratio. They do **not** fill the interior. Most of the triangle was never calorimetrically measured.

**Where Dataset B adds points**

Everywhere on the triangle at 0.01 resolution: edges, interior, and corners. That is **11856** rows in `unsampled_ternary_region` — compositions the calorimeter never visited.

```text
104 experimental points     =  three thin lines (sparse drops)
Dataset B (15453 points)    =  dense RKM grid over the whole triangle
```

---

## 8. Experimental vs unsampled regions

The column `region_type` tags how close a synthetic row is to real measurements:

| Label | Meaning |
| --- | --- |
| `experimental_anchor` | Very close to a Table III composition (distance ≤ 0.012 in mole-fraction space) |
| `near_experimental_cross_section` | Close to one of the three measured lines (distance ≤ 0.03) but not necessarily on an exact drop |
| `unsampled_ternary_region` | Everything else — **interior and most edges away from the three lines** |

Classification uses only composition distance, not temperature.

**Critical point:** A row tagged `unsampled_ternary_region` is an **RKM model prediction**. It was never measured. Calling it “experimental” would be wrong even though it sits in a physically plausible composition range.

Dataset A has **no** `unsampled_ternary_region` rows — by design it only lives on the three lines.

---

## 9. Why positive values can occur

Two different mechanisms appear in the actual files.

### A. Dataset B — positive values from RKM itself (1566 rows)

All 1566 positive rows have **positive `base_model_prediction_J_per_mol`**. Noise is zero, so this is not a noise artefact.

The largest value is **+134.73 J/mol** at compositions like \(x_{\mathrm{Bi}} = 0.58\), \(x_{\mathrm{In}} = 0\), \(x_{\mathrm{Sn}} = 0.42\) — essentially the **Bi–Sn binary edge** with little or no indium.

That matches the paper and Table IV: the Bi–Sn binary has \(L^{(0)}_{\mathrm{Bi-Sn}} = +520\) J/mol (endothermic). Ternary measurements on the three sections are exothermic, but the **model** correctly allows endothermic mixing on the Bi–Sn edge. Pure-element corners are ~0 J/mol.

So positive Dataset B values are **physically consistent with the RKM parameters**, not sign errors.

### B. Dataset A — 4 slightly positive rows from noise

Only **4** rows (all on the Sn-rich section at 767 K, \(x_{\mathrm{In}} \approx 0.09\)–0.096) have positive targets. In each case **RKM is still negative** (e.g. base ≈ −6.7 J/mol) but a large positive noise draw (σ = 13.2 J/mol for Series 7) pushes the target slightly above zero (max **+9.84 J/mol**).

These were **not clipped**. They are rare edge cases where the true surface is near zero and artificial noise crosses the axis.

**Summary**

| Cause | Where | Mechanism |
| --- | --- | --- |
| RKM physics (Bi–Sn endothermic edge) | Dataset B, 1566 rows | Model prediction genuinely positive |
| Artificial noise | Dataset A, 4 rows | RKM negative, noise pushes target positive |

---

## 10. Deterministic vs noisy data

| | Dataset A (on disk) | Dataset B (on disk) |
| --- | --- | --- |
| RKM part | `base_model_prediction_J_per_mol` | same as target |
| Noise | Yes | No |
| Target | base + ε | base only |

**Deterministic synthetic data** = \(\Delta_{\mathrm{mix}}H\) exactly equals the RKM polynomial. Dataset B is fully deterministic. Dataset A’s `base_model_prediction` column is the deterministic part; the main target column is noisy.

**Noisy synthetic data** = RKM + Gaussian perturbation:

\[
y_{\mathrm{synthetic}} = y_{\mathrm{RKM}} + \varepsilon,\qquad \varepsilon \sim N(0,\sigma)
\]

**Why noise was added (Dataset A only)**

Real Table III values have reported uncertainty \(u(\Delta_{\mathrm{mix}}H)\) from footnote a (about 7.9–23.3 J/mol per series). Noise imitates that scatter so synthetic rows are not unnaturally smooth. It does **not** create new experiments.

**Where σ comes from**

Hard-coded in `generate_synthetic_data.py` as `SERIES_U_J`, converted from Table III footnote a (kJ/mol → J/mol):

| Series | T (K) | σ (J/mol) |
| --- | --- | --- |
| 1 | 767 | 23.3 |
| 2 | 813 | 19.7 |
| 3 | 855 | 16.08 |
| 4 | 767 | 18.17 |
| 5 | 813 | 14.4 |
| 6 | 855 | 12.7 |
| 7 | 767 | 13.2 |
| 8 | 813 | 10.5 |
| 9 | 855 | 7.9 |

Each Dataset A row uses the σ for its `(cross_section, temperature)` series.

**Random seed = 42**

`RANDOM_SEED = 42` fixes the noise draws. Re-running the generator with the same seed reproduces the same 7389 noisy targets. Changing the seed would change the noise but not the RKM surface.

Noise changes the **numeric target** slightly; it does **not** change the scientific meaning — these rows remain RKM-based synthetic labels, not calorimetry.

---

## 11. Experimental uncertainty and artificial noise

Table III reports uncertainty on the **experimental** integral enthalpy. Our synthetic noise uses those σ values as the standard deviation of a normal draw.

That is a **statistical imitation**, not a new drop-calorimetry experiment.

```text
Real experiment:     calorimeter → measured ΔmixH ± u(ΔmixH)
Synthetic (Dataset A): RKM → base ΔmixH, then + artificial ε ~ N(0, σ)
Synthetic (Dataset B): RKM → ΔmixH only (no ε)
```

Confusing noisy synthetic targets with independent experimental evidence is the main mistake to avoid in the ML stage.

---

## 12. Dataset comparison

| Dataset | Rows | File | Source | Real experiment? | Noise? | Main purpose |
| --- | --- | --- | --- | --- | --- | --- |
| Original experimental | 104 | `original_experimental_data.csv` | Table III / calorimeter | **YES** | N/A (measured scatter in u columns) | Ground truth / reference |
| Dataset A | 7389 | `synthetic_cross_sections.csv` | RKM + noise | **NO** | Yes | Dense coverage on the three measured lines |
| Dataset B | 15453 | `synthetic_full_ternary.csv` | RKM only | **NO** | No | Dense coverage of full composition triangle |

Additional tags:

- Dataset A: only on three cross-sections; `generation_method = RKM_interpolation_plus_noise`
- Dataset B: full simplex grid step 0.01; `generation_method = RKM_full_ternary_grid`; most rows `unsampled_ternary_region`

---

## 13. Important ML leakage warning

Synthetic \(\Delta_{\mathrm{mix}}H\) labels come **from the same RKM equation** we validated on the 104 experiments. The chain is:

```text
RKM (Equation 4, Table IV)
        ↓
Synthetic ΔmixH (Dataset A and B)
        ↓
Tested as ML augmentation (Dataset A) under strict LOCSO
        ↓
No improvement on the real observations → not used to train the final predictor
```

The final predictor was trained on the 104 experimental rows only. The warning below explains why scores on synthetic labels alone would not have been evidence of calorimetric accuracy.

If we train and test **only** on synthetic rows, a good score often means the ML model **reproduced RKM**, not that it predicts calorimetry.

RKM already uses the experimental points (indirectly — the ternary parameters were fitted to them in the paper). Testing ML on more RKM labels does not add independent physical evidence.

**Experimental rows must stay the ultimate reference.** Hold out `source = paper_experiment` for evaluation. Be careful with `region_type`: interior Dataset B points were never measured. Reporting low error on `unsampled_ternary_region` is not “validation against experiment.”

This warning is also in `docs/rkm_model.md` and the project README.

---

## 14. Project pipeline at the time of writing (historical)

```text
Professor's paper (Table III + Equation 4 + Table IV)
        ↓
104 experimental measurements          ← REAL (calorimeter)
        ↓
RKM implementation (src/rkm_model.py)  ← MODEL (thermodynamic equation)
        ↓
Validate RKM vs all 104 points         ← CHECK (MAE 57.13, R² 0.9510)
        ↓
Validated RKM surface
        ↓
Generate extra compositions            ← CHOSEN GRIDS (not measured)
        ↓
RKM calculates ΔmixH at each           ← MODEL OUTPUT
        ↓
Dataset A (7389, noisy, cross-sections)  ← SYNTHETIC
Dataset B (15453, deterministic, full triangle)  ← SYNTHETIC
        ↓
NEXT STAGE (at the time): ML surrogate
```

The ML stage was completed later. The final ML surrogate is the Direct Poly D2 (xBi, xIn, T), trained on the 104 experimental rows only, with LOCSO MAE 41.10 J/mol. Synthetic augmentation was tested and rejected (54.32 vs 41.10 J/mol), and Datasets A and B are exploratory only.

At each stage:

| Stage | Real? | Model-generated? |
| --- | --- | --- |
| 104 Table III rows | Yes | No |
| RKM equation / parameters | From paper | Not new experiments |
| RKM predictions at test compositions | No | Yes |
| Dataset A & B targets | No | Yes (RKM ± noise for A) |

When this note was written, machine learning had not been implemented. The question it posed was whether the surrogate could match **experiments** on held-out Table III rows, not just match RKM on synthetic grids. The final model was evaluated in exactly that way, by holding out whole experimental cross-sections (LOCSO).

---

*Based on: `data/synthetic_cross_sections.csv`, `data/synthetic_full_ternary.csv`, `data/original_experimental_data.csv`, and `scripts/generate_synthetic_data.py`. RKM validation: `reports/rkm_validation.md`.*
