# Final cleanup summary

## 1. Final directory structure

```text
project/
├── README.md
├── requirements.txt
├── professor_paper.pdf
├── .gitignore
├── data/
│   ├── original_experimental_data.csv
│   ├── extraction_audit.csv
│   ├── synthetic_cross_sections.csv
│   └── synthetic_full_ternary.csv
├── src/
│   └── rkm_model.py
├── scripts/
│   ├── generate_synthetic_data.py
│   └── validate_dataset.py
├── reports/
│   ├── project_work_report.md
│   ├── project_work_report.pdf
│   ├── rkm_validation.md
│   ├── final_cleanup_inventory.md
│   └── final_cleanup_summary.md
├── docs/
│   └── rkm_model.md
└── figures/
    ├── experimental_cross_sections.png
    ├── rkm_vs_experiment.png
    ├── rkm_vs_experiment_cross_sections.png
    ├── experimental_vs_synthetic.png
    └── ternary_enthalpy_map_813K.png
```

## 2. Files removed

- Python cache (`__pycache__/`, `*.pyc`)
- Personal image (`WhatsApp Image 2026-08-31 at 7.44.47 PM.jpeg`)
- `data/derived/` (master, ML tables, deterministic/noisy concatenations)
- `src/data/table_iii.py` and nested packages
- `figures/archive/` and five extra diagnostic plots
- `reports/project_cleanup_plan.md`, `reports/project_cleanup_summary.md`

Datasets were **not** regenerated. Experimental and synthetic CSV contents are unchanged.

## 3. Files merged / moved

- `src/thermodynamics/rkm_model.py` → `src/rkm_model.py` (same equation and parameters)
- Paper PDF → `professor_paper.pdf`
- Generation logic no longer depends on `table_iii.py`; it reads the experimental CSV
- Leakage warning kept in `docs/rkm_model.md` and the work report (no separate leakage file)

## 4. Files retained

Listed in the tree above.

## 5. Why each major retained file exists

| File | Why |
| --- | --- |
| `professor_paper.pdf` | Source of Table III, Equation 4, Table IV |
| `original_experimental_data.csv` | 104 experimental observations |
| `extraction_audit.csv` | Minus-sign / composition reconstruction record |
| `synthetic_cross_sections.csv` | Dataset A (7389) |
| `synthetic_full_ternary.csv` | Dataset B (15453) |
| `src/rkm_model.py` | RKM implementation |
| `scripts/validate_dataset.py` | Confirms the files on disk |
| `scripts/generate_synthetic_data.py` | How A and B were built (do not re-run casually) |
| `docs/rkm_model.md` | Equation, Table IV, (i,j,k) convention, ML warning |
| `reports/project_work_report.md` | Stage-1 write-up for the professor |
| `reports/project_work_report.pdf` | Printable copy of that write-up |
| `reports/rkm_validation.md` | MAE / RMSE / R² |
| `figures/*.png` | Five plots that explain experiment, RKM fit, and the triangle |
| `README.md` | Pipeline and how to run validation |

## 6. Validation result

```text
DATASET VALIDATION: PASS
Passed 65 checks.
```

Scientific numbers (unchanged):

- Experimental observations: **104**
- Experimental ΔmixH: **−1413.0 to −64.32 J/mol**
- RKM: **MAE = 57.13 J/mol**, **RMSE = 73.49 J/mol**, **R² = 0.9510**
- Dataset A: **7389**
- Dataset B: **15453**
- Random seed: **42**
- Flattened RKM sample point matches the previous implementation: **max difference = 0**

Check **count** is 65, not 87. The missing checks were only for the deleted derived CSVs (master, ML tables, A-det/B-noisy concatenations). No scientific result changed.

## 7. Remaining extra material

- `reports/final_cleanup_inventory.md` and this summary exist only because this cleanup asked for them.
- `.gitignore` only ignores cache files.
