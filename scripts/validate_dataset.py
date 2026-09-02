"""
Check the experimental and synthetic CSV files.

    python scripts/validate_dataset.py

Prints DATASET VALIDATION: PASS or FAIL.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rkm_model import rkm_delta_mix_h

DATA = ROOT / "data"
RANDOM_SEED = 42
COMPOSITION_TOL = 1e-8
ALLOWED_T = {767, 813, 855}
CROSS_SECTION_RATIO = {
    "(Sn0.33Bi0.67)1-xInx": 0.67,
    "(Sn0.50Bi0.50)1-xInx": 0.50,
    "(Sn0.67Bi0.33)1-xInx": 0.33,
}
RATIO_TOL = 0.015  # labelled 0.67 vs actual ~0.6675 from starting moles

passed = []
failed = []


def check(condition, message):
    if condition:
        passed.append(message)
    else:
        failed.append(message)


def read_csv(name):
    path = DATA / name
    check(path.exists(), f"file exists: {name}")
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def check_compositions(df, label):
    total = df["xBi"] + df["xIn"] + df["xSn"]
    check(bool(np.allclose(total, 1.0, atol=COMPOSITION_TOL)), f"{label}: xBi+xIn+xSn = 1")
    for col in ("xBi", "xIn", "xSn"):
        in_range = ((df[col] >= -COMPOSITION_TOL) & (df[col] <= 1 + COMPOSITION_TOL)).all()
        check(bool(in_range), f"{label}: 0 <= {col} <= 1")
    check(not df[["xBi", "xIn", "xSn"]].isna().any().any(), f"{label}: no NaN mole fractions")
    check(not np.isinf(df[["xBi", "xIn", "xSn"]].to_numpy()).any(), f"{label}: no inf mole fractions")


def rkm_metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return mae, rmse, r2


def main():
    exp = read_csv("original_experimental_data.csv")
    audit = read_csv("extraction_audit.csv")
    syn_a = read_csv("synthetic_cross_sections.csv")
    syn_b = read_csv("synthetic_full_ternary.csv")

    check(len(exp) == 104, f"experimental N={len(exp)} (expected 104)")
    check((exp["source"] == "paper_experiment").all(), "experimental source tag")
    check((exp["source_table"] == "Table III").all(), "experimental source_table")
    check(not exp["id"].duplicated().any(), "experimental IDs unique")
    check(set(exp["temperature_K"].astype(int)) <= ALLOWED_T, "experimental temperatures")
    check_compositions(exp, "experimental")
    y_exp = exp["integral_mixing_enthalpy_J_per_mol"]
    check((y_exp < 0).all(), "all experimental dmixH exothermic")
    check(not y_exp.isna().any(), "experimental target has no NaN")
    check(float(y_exp.min()) == -1413.0, f"experimental dmixH min = {float(y_exp.min())} (expected -1413.0)")
    check(float(y_exp.max()) == -64.32, f"experimental dmixH max = {float(y_exp.max())} (expected -64.32)")
    check(len(audit) > 0, "extraction audit is non-empty")
    check(
        exp["partial_enthalpy_In_J_per_mol"].abs().mean()
        > y_exp.abs().mean(),
        "partial |H_In| larger on average than |dmixH| (not swapped columns)",
    )

    for name, labelled_ratio in CROSS_SECTION_RATIO.items():
        subset = exp[exp["cross_section"] == name]
        ratio = subset["xBi"] / (subset["xBi"] + subset["xSn"])
        check(bool(np.allclose(ratio, ratio.iloc[0], atol=1e-10)), f"{name}: constant Bi/(Bi+Sn)")
        check(
            bool(abs(ratio.iloc[0] - labelled_ratio) < RATIO_TOL),
            f"{name}: Bi/(Bi+Sn) ~ {labelled_ratio} (got {ratio.iloc[0]:.4f})",
        )

    check(exp.groupby("series_id").ngroups == 9, "nine experimental series")

    pred = np.array([rkm_delta_mix_h(r.xBi, r.xIn, r.xSn) for r in exp.itertuples()])
    mae, rmse, r2 = rkm_metrics(y_exp.to_numpy(), pred)
    check(round(mae, 2) == 57.13, f"RKM MAE = {mae:.2f} J/mol (expected 57.13)")
    check(round(rmse, 2) == 73.49, f"RKM RMSE = {rmse:.2f} J/mol (expected 73.49)")
    check(round(r2, 4) == 0.9510, f"RKM R2 = {r2:.4f} (expected 0.9510)")

    check((syn_a["source"] == "synthetic").all(), "Dataset A source=synthetic")
    check(syn_a["synthetic"].astype(str).str.lower().isin(["true", "1"]).all(), "Dataset A synthetic=True")
    check(len(syn_a) == 7389, f"Dataset A size = {len(syn_a)} (expected 7389)")
    check_compositions(syn_a, "Dataset A")
    check(set(syn_a["temperature_K"].astype(int)) <= ALLOWED_T, "Dataset A temperatures")
    for name, labelled_ratio in CROSS_SECTION_RATIO.items():
        subset = syn_a[syn_a["cross_section"] == name]
        ratio = subset["xBi"] / (subset["xBi"] + subset["xSn"])
        check(bool((np.abs(ratio - labelled_ratio) < RATIO_TOL).all()), f"Dataset A {name} ratio")

    sample_a = syn_a.sample(n=min(400, len(syn_a)), random_state=RANDOM_SEED)
    pred_a = np.array([rkm_delta_mix_h(r.xBi, r.xIn, r.xSn) for r in sample_a.itertuples()])
    check(
        bool(np.allclose(pred_a, sample_a["base_model_prediction_J_per_mol"], atol=1e-6)),
        "Dataset A base_model_prediction matches RKM",
    )
    noise_size = (sample_a["delta_mix_H_J_per_mol"] - sample_a["base_model_prediction_J_per_mol"]).abs()
    check(bool((noise_size < 200).all()), "Dataset A noise residuals are bounded")

    check((syn_b["source"] == "synthetic").all(), "Dataset B source=synthetic")
    check(len(syn_b) == 15453, f"Dataset B size = {len(syn_b)} (expected 15453)")
    check_compositions(syn_b, "Dataset B")
    check(syn_b["generation_method"].astype(str).str.contains("RKM_full_ternary").all(), "Dataset B method")
    check(
        syn_b["region_type"].isin(
            ["experimental_anchor", "near_experimental_cross_section", "unsampled_ternary_region"]
        ).all(),
        "Dataset B region_type values",
    )
    check((syn_b["region_type"] == "unsampled_ternary_region").any(), "Dataset B contains unsampled region")
    check(
        (syn_b["region_type"] == "near_experimental_cross_section").any()
        or (syn_b["region_type"] == "experimental_anchor").any(),
        "Dataset B contains near-experimental region",
    )

    sample_b = syn_b.sample(n=min(400, len(syn_b)), random_state=RANDOM_SEED)
    pred_b = np.array([rkm_delta_mix_h(r.xBi, r.xIn, r.xSn) for r in sample_b.itertuples()])
    check(
        bool(np.allclose(pred_b, sample_b["delta_mix_H_J_per_mol"], atol=1e-5)),
        "Dataset B targets equal RKM (deterministic grid)",
    )

    pure = syn_b[(syn_b["xBi"] >= 0.999) | (syn_b["xIn"] >= 0.999) | (syn_b["xSn"] >= 0.999)]
    check(len(pure) > 0, "Dataset B includes near-pure vertices")
    check(bool((pure["delta_mix_H_J_per_mol"].abs() < 1e-6).all()), "pure-element dmixH ~ 0")

    check(rkm_delta_mix_h(0.25, 0.50, 0.25) == rkm_delta_mix_h(0.25, 0.50, 0.25), "RKM deterministic for identical composition")
    check(abs(float(rkm_delta_mix_h(0.25, 0.50, 0.25)) - (-1029.4375)) < 1e-9, "RKM sample point unchanged")
    check(RANDOM_SEED == 42, "RANDOM_SEED is 42")
    check(~exp["source"].astype(str).str.contains("synthetic").any(), "original file has no synthetic rows")

    print("")
    if failed:
        print("DATASET VALIDATION: FAIL")
        for message in failed:
            print(" -", message)
        print(f"\nPassed {len(passed)} checks.")
        return 1

    print("DATASET VALIDATION: PASS")
    print(f"Passed {len(passed)} checks.")
    for message in passed:
        print(" -", message)
    print("")
    print("WARNING: Synthetic data are model-generated and are NOT experimental observations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
