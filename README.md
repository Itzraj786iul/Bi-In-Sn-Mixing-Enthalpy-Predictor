# Bi-In-Sn mixing enthalpy — semester project

Surrogate modelling of the integral molar mixing enthalpy \(\Delta_{\mathrm{mix}}H\) for liquid Bi–In–Sn. Experimental values come from the professor’s paper:

Kumar, Mohan and Behera, *J. Electron. Mater.* **48**, 8096–8106 (2019) (`professor_paper.pdf`).

**Machine learning has not been implemented yet.**

## Pipeline

```text
Professor's paper
        ↓
104 experimental observations (Table III)
        ↓
RKM thermodynamic model (Equation 4, Table IV)
        ↓
Validation against experiments
        ↓
Validated RKM
        ↓
Synthetic Dataset A (cross-sections) and Dataset B (full ternary)
        ↓
Future ML model  ← not started
```

## What is in this folder

| File | Role |
| --- | --- |
| `professor_paper.pdf` | Source paper (Table III, Equation 4, Table IV) |
| `data/original_experimental_data.csv` | 104 calorimetry rows |
| `data/synthetic_cross_sections.csv` | Dataset A, 7389 points, `source = synthetic` |
| `data/synthetic_full_ternary.csv` | Dataset B, 15453 RKM grid points |
| `data/extraction_audit.csv` | Minus-sign and composition reconstruction notes |
| `src/rkm_model.py` | RKM calculation |
| `scripts/validate_dataset.py` | Checks the CSV files already on disk |
| `docs/rkm_model.md` | Equation, parameters, ternary index convention |
| `reports/project_work_report.md` | Stage-1 write-up |
| `reports/rkm_validation.md` | MAE / RMSE / R² |

Target: integral \(\Delta_{\mathrm{mix}}H\) in J/mol, **not** the partial enthalpy of indium. Mole fractions \(x_{\mathrm{Bi}}+x_{\mathrm{In}}+x_{\mathrm{Sn}}=1\).

## How to run

```text
pip install -r requirements.txt
python scripts/validate_dataset.py
```

That should print `DATASET VALIDATION: PASS`.

`python scripts/generate_synthetic_data.py` rebuilds Dataset A and B (seed 42). Do not run it unless you intend to overwrite those two CSV files. It does not change the experimental data.

## Results that must stay the same

- Experiments: 104; \(\Delta_{\mathrm{mix}}H\) from −1413.0 to −64.32 J/mol
- RKM vs experiment: MAE 57.13 J/mol, RMSE 73.49 J/mol, R² 0.9510
- Dataset A: 7389 points; Dataset B: 15453 points; seed 42

## Before ML

Synthetic labels are RKM evaluations, not new calorimetry. A high score on synthetic rows only means a model copied the polynomial. Hold out `source = paper_experiment` rows. Interior Dataset B points are `unsampled_ternary_region`.

Details: `reports/project_work_report.md` and `docs/rkm_model.md`.
