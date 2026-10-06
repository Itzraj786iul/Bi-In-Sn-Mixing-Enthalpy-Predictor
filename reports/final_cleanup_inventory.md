# Final cleanup inventory

> **Historical note:** This inventory records the Stage-1 repository cleanup, done before the ML work. "Final" refers to that cleanup, not to the completed project. For the current repository state see `step_11a_repository_curation.md`.

Classifications before any deletion. Reasons are one line each.

## Root

| File | Action | Reason |
| --- | --- | --- |
| README.md | KEEP | Entry point for the project |
| requirements.txt | KEEP | Needed to install libraries |
| s11664-019-07646-0 (MRK Sir Project).pdf | MOVE | Rename to `professor_paper.pdf`; this is the source paper |
| WhatsApp Image 2026-08-31 at 7.44.47 PM.jpeg | DELETE | Personal/handwritten notes image; not needed to run or reproduce the science |

## data/

| File | Action | Reason |
| --- | --- | --- |
| original_experimental_data.csv | KEEP | 104 Table III measurements |
| synthetic_cross_sections.csv | KEEP | Dataset A (7389) |
| synthetic_full_ternary.csv | KEEP | Dataset B (15453) |
| extraction_audit.csv | KEEP | Documents minus-sign / composition reconstruction |
| derived/bisn_biin_synthetic_master.csv | DELETE | Concatenation of files we already keep |
| derived/ml_dataset.csv | DELETE | Column subset for ML that has not started; can be rebuilt later |
| derived/ml_dataset_no_temperature.csv | DELETE | Same rows without temperature |
| derived/synthetic_rkm_deterministic.csv | DELETE | A+B without noise; B is already deterministic; A-det can be rebuilt |
| derived/synthetic_rkm_noisy.csv | DELETE | A is already noisy; B-noisy is not needed now |
| derived/ | DELETE | Empty after the CSVs go |

## src/

| File | Action | Reason |
| --- | --- | --- |
| thermodynamics/rkm_model.py | MOVE | Flatten to `src/rkm_model.py` |
| data/table_iii.py | DELETE | Extraction is finished; numbers live in `original_experimental_data.csv` |
| **/__init__.py | DELETE | Not needed once `src/` is a single module |
| **/__pycache__/ | DELETE | Python cache |

## scripts/

| File | Action | Reason |
| --- | --- | --- |
| generate_synthetic_data.py | KEEP (simplify) | Reproducibility of A and B; stop writing derived files |
| validate_dataset.py | KEEP (simplify) | Must still PASS on the retained CSVs |

## figures/

| File | Action | Reason |
| --- | --- | --- |
| experimental_cross_sections.png | KEEP | Shows the three experimental sections |
| rkm_vs_experiment.png | KEEP | Parity plot for RKM vs Table III |
| rkm_vs_experiment_cross_sections.png | KEEP | Shape of the fit along each section |
| ternary_enthalpy_map_813K.png | KEEP | Full-triangle RKM map |
| experimental_vs_synthetic.png | KEEP | Distinguishes experiment vs Dataset A |
| experimental_all_temperatures.png | DELETE | Covered by cross-section plot |
| difference_map.png | DELETE | Extra diagnostic; not needed to explain the project |
| distribution_experimental_vs_synthetic.png | DELETE | Histogram not used in the main argument |
| ternary_composition_scatter.png | DELETE | The enthalpy map already shows the three lines |
| ternary_isoenthalpy_813K.png | DELETE | Same story as the enthalpy map |
| figures/archive/ (all 11 pngs) | DELETE | Duplicates / per-temperature extras |

## docs/

| File | Action | Reason |
| --- | --- | --- |
| rkm_model.md | KEEP | Equation, Table IV, (i,j,k) convention |

## reports/

| File | Action | Reason |
| --- | --- | --- |
| project_work_report.md | KEEP | Stage-1 scientific write-up |
| project_work_report.pdf | KEEP | Useful to hand to the professor |
| rkm_validation.md | KEEP | MAE / RMSE / R² |
| project_cleanup_plan.md | DELETE | Housekeeping note, not science |
| project_cleanup_summary.md | DELETE | Housekeeping note, not science |
| final_cleanup_inventory.md | KEEP | This inventory (this task asked for it) |
| final_cleanup_summary.md | KEEP | This task asked for it |

## Uncertain → decision

None. Every file above has a KEEP / MOVE / DELETE.
