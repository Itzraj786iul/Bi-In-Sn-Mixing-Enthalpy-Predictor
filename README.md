# Bi-In-Sn mixing enthalpy — semester project

Surrogate modelling of the integral molar mixing enthalpy \(\Delta_{\mathrm{mix}}H\) for liquid Bi–In–Sn. Experimental values come from the professor’s paper:

Kumar, Mohan and Behera, *J. Electron. Mater.* **48**, 8096–8106 (2019) (`professor_paper.pdf`).

## Project status

The final predictor is a **Direct Polynomial Degree-2 regression trained only on the 104 real experimental observations**. It is deployed in the web application (`web/`, see `web/README.md`).

- **Features:** xBi, xIn, T (`temperature_K`)
- **Derived:** xSn = 1 − xBi − xIn. xSn is not a feature, because it is exactly determined by xBi and xIn.
- **Output:** integral \(\Delta_{\mathrm{mix}}H\) in J/mol
- **Coefficients:** `reports/final_model_coefficients.csv`, produced by `scripts/extract_final_model.py`
- **Benchmark:** the fixed Redlich–Kister–Muggianu (RKM) thermodynamic model, `src/rkm_model.py`

## Methodology

1. Professor-paper experimental data: 104 calorimetry observations (Table III), three composition cross-sections, 767 / 813 / 855 K.
2. Experimental data reconstruction and validation: minus signs and compositions checked against the source tables (`data/extraction_audit.csv`, `scripts/validate_dataset.py`).
3. Fixed RKM thermodynamic benchmark: Equation 4 and Table IV of the paper, not refitted.
4. Direct Polynomial Degree-2 ML model using xBi, xIn and T.
5. Strict leave-one-cross-section-out (LOCSO) validation: each composition cross-section is held out in turn.
6. Ternary prediction and RKM comparison surfaces in the web application.

### Pipeline

The final predictor and the synthetic-data experiment are separate branches. Synthetic data do not feed the final predictor.

```text
FINAL PREDICTOR                              SEPARATE EXPLORATORY BRANCH

104 real experiments (Table III)             RKM-derived synthetic datasets (A, B)
        ↓                                            ↓
Reconstruction / cleaning                    Augmentation experiment
        ↓                                            ↓
Fixed RKM benchmark                          Strict LOCSO evaluation
        ↓                                            ↓
Direct Poly D2 (xBi, xIn, T)                 No improvement on real observations
        ↓                                            ↓
LOCSO validation                             Exploratory analysis only
        ↓
Web app: prediction and ML vs RKM surfaces
```

## Results

Primary validation of the ML model is LOCSO on the 104 experimental observations. RKM is reported as a physics-model reference on the same observations.

| Model | Evaluation protocol | MAE (J/mol) | RMSE (J/mol) | R² |
| --- | --- | ---: | ---: | ---: |
| Direct Poly D2 (final) | LOCSO, held-out cross-section | 41.10 | 56.34 | 0.9712 |
| RKM | Evaluated on the 104 experimental observations using the published parameters (ternary parameters fitted to these measurements by the original authors) | 57.13 | 73.49 | 0.9510 |

These are not strictly like-for-like out-of-sample scores: the ML result uses cross-section holdout, whereas the RKM parameters were fitted to the experimental measurements by the original authors. The RKM value is therefore a physics-model reference rather than an independent LOCSO validation score.

The Direct Poly D2 model achieves lower error than the RKM reference under the reported evaluation, but the two protocols are not strictly equivalent, and this does not make ML universally superior. RKM remains the physics-based thermodynamic benchmark, and predictions away from the measured cross-sections are not experimentally validated.

**Thermodynamic boundary limitation.** The fitted polynomial is an empirical predictive surrogate and was not constrained to satisfy ΔmixH = 0 at the pure components. It therefore produces non-zero endpoint values. For example, the backend gives about −392 J/mol for pure Bi at 767 K (−130 J/mol at 855 K), about +33 / −59 J/mol for pure Sn, and about −77 / +9 J/mol for pure In, at 767 / 855 K respectively. RKM gives exactly 0 at every pure component. These boundary extrapolations should not be interpreted as physically valid pure-component thermodynamic predictions.

Benchmark composition: xBi = 0.2509, xIn = 0.4982, xSn = 0.2509, T = 813 K. ML gives −1014.5761 J/mol, RKM −1028.5461 J/mol, so ML − RKM = +13.9700 J/mol. The measured value is −1017.0 J/mol.

## Synthetic data

- Synthetic Datasets A and B were generated from the RKM model. Their labels are RKM evaluations, not new calorimetry.
- Synthetic-data augmentation was evaluated experimentally (`reports/synthetic_data_ablation.md`).
- Under strict LOCSO, synthetic augmentation did not improve prediction of the real experimental observations. Real data plus Dataset A gave a pooled MAE of 54.32 J/mol, against 41.10 J/mol for real data only.
- Therefore synthetic data are **not** used to train the final predictor.
- The synthetic datasets are retained for exploratory analysis and visualization only.

A high score on synthetic rows only shows that a model reproduced the RKM polynomial. Always evaluate on `source = paper_experiment` rows. Interior Dataset B points are labelled `unsampled_ternary_region`.

## What is in this folder

| File | Role |
| --- | --- |
| `professor_paper.pdf` | Source paper (Table III, Equation 4, Table IV) |
| `data/original_experimental_data.csv` | 104 calorimetry rows (training and validation data) |
| `data/extraction_audit.csv` | Minus-sign and composition reconstruction notes |
| `data/synthetic_cross_sections.csv` | Dataset A, 7389 points, `source = synthetic` (exploratory only) |
| `data/synthetic_full_ternary.csv` | Dataset B, 15453 RKM grid points (exploratory only) |
| `src/rkm_model.py` | RKM calculation (fixed benchmark) |
| `scripts/validate_dataset.py` | Checks the CSV files already on disk |
| `scripts/extract_final_model.py` | Fits the final Poly D2 on the 104 experimental rows and writes the coefficients |
| `reports/final_model_coefficients.csv` | Final model coefficients (full precision) |
| `reports/ml_composition_representation_results.md` | LOCSO results for the final feature set |
| `reports/final_model_validation.md` | Final model validation (LOCSO primary, random 5-fold secondary) |
| `reports/synthetic_data_ablation.md` | Synthetic-data augmentation experiment |
| `reports/rkm_validation.md` | RKM MAE / RMSE / R² |
| `scripts/generate_research_figures.py` | Regenerates the data/RKM figures and `reports/rkm_validation.md` |
| `docs/rkm_model.md` | Equation, parameters, ternary index convention |
| `reports/project_work_report.md` | Stage-1 report (data, RKM, synthetic data); historical notes added later |
| `reports/project_work_report.pdf` | Historical Stage-1 PDF; **not** synchronized with the current Markdown |
| `web/` | Web application (FastAPI backend and React frontend) |

Target: integral \(\Delta_{\mathrm{mix}}H\) in J/mol, **not** the partial enthalpy of indium. Mole fractions \(x_{\mathrm{Bi}}+x_{\mathrm{In}}+x_{\mathrm{Sn}}=1\).

## How to run

```text
pip install -r requirements.txt
python scripts/validate_dataset.py
```

That should print `DATASET VALIDATION: PASS`.

`python scripts/generate_synthetic_data.py` rebuilds Dataset A and B (seed 42). Do not run it unless you intend to overwrite those two CSV files. It does not change the experimental data.

## Frozen final predictor vs model-selection scripts

The **frozen final predictor** is defined by `reports/final_model_coefficients.csv` (written by `scripts/extract_final_model.py`) and served by `web/backend/model.py`. Its supporting analyses are `scripts/validate_final_model.py` (secondary random 5-fold), `scripts/analyze_final_surface.py` (response surface) and `scripts/build_final_surrogate.py` (Dataset B evaluation grid).

### Model-selection and exploratory analysis scripts

These scripts record how the final model was chosen and which alternatives were investigated. They are research history, not part of the final predictor, and none of them changes it.

| Script | Purpose |
| --- | --- |
| `train_baseline.py` | First linear-regression baseline (composition only) |
| `train_polynomial_baseline.py` | Polynomial degree 2/3 baselines (composition only) |
| `train_random_forest.py`, `train_gradient_boosting.py` | Tree-based baselines |
| `train_polynomial_temperature.py` | Adds temperature to Poly D2 (xBi, xIn, xSn, T) |
| `train_polynomial_synthetic_temperature.py` | Early synthetic-training test for the temperature model |
| `analyze_composition_representation.py` | xSn included vs omitted; selected the final feature set (xBi, xIn, T) |
| `analyze_ml_models.py`, `analyze_ml_overlap.py` | Diagnostics of the earlier composition-only models; experiment/synthetic overlap |
| `step_4_residual_learning.py` … `step_7_cross_section_residual_structure.py` | RKM + residual-learning hybrids and residual diagnostics (not adopted) |
| `step_8_controlled_model_comparison.py` | Controlled comparison of RKM, direct Poly D2 and the hybrids |
| `step_9_synthetic_data_ablation.py` | Definitive synthetic-augmentation ablation for the final model |

## Reproducing the research outputs

Run from the repository root after `pip install -r requirements.txt`. Each script overwrites the report/figures named in its header.

| Output | Command |
| --- | --- |
| Dataset checks (65) | `python scripts/validate_dataset.py` |
| Final coefficients and equation | `python scripts/extract_final_model.py` |
| LOCSO metrics of the final feature set | `python scripts/analyze_composition_representation.py` |
| Secondary random 5-fold validation | `python scripts/validate_final_model.py` |
| Final response surfaces | `python scripts/analyze_final_surface.py` |
| Dataset B prediction grid (untracked CSV, ≈8.6 MB) | `python scripts/build_final_surrogate.py` |
| Controlled model comparison | `python scripts/step_8_controlled_model_comparison.py` |
| Synthetic-data ablation | `python scripts/step_9_synthetic_data_ablation.py` |
| Data/RKM figures and `rkm_validation.md` | `python scripts/generate_research_figures.py` (use `--out-dir` to write elsewhere) |

Caveats: several generated reports carry explanatory notes added by hand after generation (see `reports/step_11b_scientific_documentation_corrections.md`), and rerunning their scripts would remove those notes. `generate_research_figures.py` reproduces the data and RKM content of its figures, but its images are not pixel-identical to the committed PNGs (the original plotting code was not preserved). See `reports/step_11c_final_repository_assembly.md`.

## Results that must stay the same

- Experiments: 104; \(\Delta_{\mathrm{mix}}H\) from −1413.0 to −64.32 J/mol
- RKM vs experiment (all 104 points, published parameters): MAE 57.13 J/mol, RMSE 73.49 J/mol, R² 0.9510
- Final Poly D2 (LOCSO): MAE 41.10 J/mol, RMSE 56.34 J/mol, R² 0.9712
- Dataset A: 7389 points; Dataset B: 15453 points; seed 42

Details: `reports/final_model_equation.md`, `reports/final_model_validation.md`, `reports/controlled_model_comparison.md`, `reports/synthetic_data_ablation.md` and `docs/rkm_model.md`. The Stage-1 history is in `reports/project_work_report.md`.
