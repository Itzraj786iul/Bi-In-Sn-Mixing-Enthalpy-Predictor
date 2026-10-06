"""
Residual-learning experiment (experimental-only LOCSO).

    python scripts/step_4_residual_learning.py

Why this experiment exists:
  - RKM (Equation 4, Table IV) is the fixed physics-based baseline.
  - RKM-generated synthetic data are not independent experimental observations:
    their targets are RKM evaluations, so training on them mostly teaches the
    model RKM itself.
  - This experiment therefore asks whether ML can learn the systematic
    experimental correction to RKM directly from the 104 real observations:

        residual          = dmixH_exp - dmixH_RKM
        residual_ML       = PolyD2(xBi, xIn, temperature_K)
        dmixH_hybrid      = dmixH_RKM + residual_ML

Only data/original_experimental_data.csv is loaded. No synthetic data.
Model family, fitting (minimum-norm least squares) and LOCSO folds are reused
from scripts/analyze_composition_representation.py. RKM is reused from
src/rkm_model.py without modification.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_composition_representation import (  # noqa: E402
    EXP_TARGET,
    calculate_metrics,
    get_locsos_folds,
    predict_polynomial,
    rkm_predict,
    train_polynomial,
)

EXP_PATH = ROOT / "data" / "original_experimental_data.csv"
REPORT_PATH = ROOT / "reports" / "residual_learning_results.md"
PRED_PATH = ROOT / "reports" / "residual_learning_results.csv"

FEATURES = ["xBi", "xIn", "temperature_K"]
EXPECTED_N = 104
EXPECTED_FOLDS = 3

SECTION_LABELS = {
    "(Sn0.33Bi0.67)1-xInx": "Bi-rich",
    "(Sn0.50Bi0.50)1-xInx": "Equiatomic",
    "(Sn0.67Bi0.33)1-xInx": "Sn-rich",
}

# Existing direct Poly D2 (xBi, xIn, T) LOCSO result, reports/ml_composition_representation_results.md
DIRECT_POLY_D2_REF = {"MAE": 41.10, "RMSE": 56.34, "R2": 0.9712}


def load_experimental() -> pd.DataFrame:
    exp = pd.read_csv(EXP_PATH)
    if len(exp) != EXPECTED_N:
        raise RuntimeError(f"Expected {EXPECTED_N} experimental rows, got {len(exp)}")
    if "source" in exp.columns and set(exp["source"].unique()) != {"paper_experiment"}:
        raise RuntimeError(f"Unexpected source values: {sorted(exp['source'].unique())}")
    exp = exp.copy()
    exp["rkm_prediction"] = rkm_predict(exp)
    exp["residual"] = exp[EXP_TARGET] - exp["rkm_prediction"]
    return exp


def run_locso(exp: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    folds = get_locsos_folds(exp)
    if len(folds) != EXPECTED_FOLDS:
        raise RuntimeError(f"Expected {EXPECTED_FOLDS} LOCSO folds, got {len(folds)}")

    rows: list[pd.DataFrame] = []
    fold_info: list[dict] = []
    for fold_idx, (held_out, train, test) in enumerate(folds, start=1):
        if set(train["id"]) & set(test["id"]):
            raise RuntimeError(f"Fold {fold_idx}: test id present in training set")
        if held_out in set(train["cross_section"]):
            raise RuntimeError(f"Fold {fold_idx}: held-out section present in training set")

        poly, model = train_polynomial(
            train[FEATURES].to_numpy(dtype=float),
            train["residual"].to_numpy(dtype=float),
        )
        predicted_residual = predict_polynomial(poly, model, test[FEATURES].to_numpy(dtype=float))

        out = test[["id", "cross_section", "temperature_K", "xBi", "xIn", "xSn"]].copy()
        out["experimental_delta_mixH"] = test[EXP_TARGET].to_numpy(dtype=float)
        out["rkm_prediction"] = test["rkm_prediction"].to_numpy(dtype=float)
        out["residual"] = test["residual"].to_numpy(dtype=float)
        out["predicted_residual"] = predicted_residual
        out["hybrid_prediction"] = out["rkm_prediction"] + out["predicted_residual"]
        out["fold"] = fold_idx
        out["held_out_section"] = held_out
        rows.append(out)

        fold_info.append(
            {
                "fold": fold_idx,
                "held_out_section": held_out,
                "n_train": len(train),
                "n_test": len(test),
                "train_sections": sorted(train["cross_section"].unique()),
            }
        )

    pred = pd.concat(rows, ignore_index=True).rename(columns={"id": "experiment_id"})
    if len(pred) != EXPECTED_N or pred["experiment_id"].duplicated().any():
        raise RuntimeError("Each experimental observation must be predicted exactly once")
    return pred, fold_info


def both_metrics(df: pd.DataFrame) -> tuple[dict, dict]:
    y = df["experimental_delta_mixH"]
    return calculate_metrics(y, df["rkm_prediction"]), calculate_metrics(y, df["hybrid_prediction"])


def grouped(df: pd.DataFrame, col: str) -> list[tuple[object, dict, dict]]:
    return [(key, *both_metrics(grp)) for key, grp in df.groupby(col, sort=True)]


def residual_diagnostics(df: pd.DataFrame) -> dict[str, float]:
    err = df["residual"] - df["predicted_residual"]
    return {
        "mean_residual": float(df["residual"].mean()),
        "std_residual": float(df["residual"].std(ddof=1)),
        "mean_predicted_residual": float(df["predicted_residual"].mean()),
        "residual_prediction_MAE": float(np.mean(np.abs(err))),
        "residual_prediction_RMSE": float(np.sqrt(np.mean(err**2))),
        "max_abs_residual_error": float(np.max(np.abs(err))),
    }


def delta_text(rkm: dict, hyb: dict) -> str:
    d = hyb["MAE"] - rkm["MAE"]
    pct = 100.0 * d / rkm["MAE"]
    return f"{d:+.2f} J/mol ({pct:+.1f}%)"


def write_report(pred: pd.DataFrame, fold_info: list[dict]) -> None:
    rkm_all, hyb_all = both_metrics(pred)
    by_section = grouped(pred, "held_out_section")
    by_temp = grouped(pred, "temperature_K")
    diag = residual_diagnostics(pred)

    improved_sections = [SECTION_LABELS[s] for s, r, h in by_section if h["MAE"] < r["MAE"]]
    worse_sections = [SECTION_LABELS[s] for s, r, h in by_section if h["MAE"] >= r["MAE"]]
    improved_temps = [f"{int(t)} K" for t, r, h in by_temp if h["MAE"] < r["MAE"]]
    worse_temps = [f"{int(t)} K" for t, r, h in by_temp if h["MAE"] >= r["MAE"]]
    pooled_better = hyb_all["MAE"] < rkm_all["MAE"] and hyb_all["RMSE"] < rkm_all["RMSE"]

    L: list[str] = []
    L += [
        "# Residual Learning Experiment",
        "",
        "Script: `scripts/step_4_residual_learning.py`. Per-observation output: `reports/residual_learning_results.csv`.",
        "",
        "## Objective",
        "",
        "Test whether a machine-learning model can learn the **systematic deviations of the experimental "
        "measurements from the RKM thermodynamic baseline**, using only real calorimetry. RKM-generated "
        "synthetic data are not independent experimental information (their targets are RKM evaluations), "
        "so this experiment learns the correction to RKM directly from the 104 measured points.",
        "",
        "## Data",
        "",
        f"- `data/original_experimental_data.csv`: **{len(pred)} real experimental observations** "
        "(Table III, Kumar, Mohan and Behera 2019).",
        "- Three cross-sections × three temperatures (767, 813, 855 K).",
        "- **No synthetic data were used in this experiment.** Dataset A and Dataset B were not loaded.",
        "- RKM predictions come from the existing `src/rkm_model.rkm_delta_mix_h` (unchanged, no refit).",
        "",
        "## Residual definition",
        "",
        "```text",
        "residual = ΔmixH_exp − ΔmixH_RKM",
        "```",
        "",
        "RKM is a fixed physics-based model (Table IV parameters), so computing its residual on every "
        "observation does not use any fold-specific fitting.",
        "",
        "## Hybrid prediction",
        "",
        "```text",
        "residual_ML  = PolyD2(xBi, xIn, T)      # PolynomialFeatures(degree=2), minimum-norm least squares",
        "ΔmixH_hybrid = ΔmixH_RKM + residual_ML",
        "```",
        "",
        "The residual model uses the same model family, features and fitting procedure as the current best "
        "direct baseline (Poly D2 on `xBi, xIn, temperature_K`; 10 polynomial terms).",
        "",
        "## Validation protocol",
        "",
        "Leave-one-cross-section-out (LOCSO), identical folds to the existing ML experiments. In each fold the "
        "entire held-out cross-section is excluded from fitting; the residual model is trained only on the "
        "other two sections and predicts residuals for the held-out section. The held-out experimental "
        "ΔmixH is never used for fitting that fold. Pooled metrics use all 104 out-of-fold predictions.",
        "",
        "| Fold | Held-out section | Train sections | Train N | Test N |",
        "| ---: | --- | --- | ---: | ---: |",
    ]
    for f in fold_info:
        train_lbl = ", ".join(SECTION_LABELS[s] for s in f["train_sections"])
        L.append(
            f"| {f['fold']} | {SECTION_LABELS[f['held_out_section']]} `{f['held_out_section']}` | "
            f"{train_lbl} | {f['n_train']} | {f['n_test']} |"
        )

    L += [
        "",
        "## Results",
        "",
        "### Pooled LOCSO (104 out-of-fold predictions)",
        "",
        "| Model | MAE (J/mol) | RMSE (J/mol) | R² |",
        "| --- | ---: | ---: | ---: |",
        f"| RKM | {rkm_all['MAE']:.2f} | {rkm_all['RMSE']:.2f} | {rkm_all['R2']:.4f} |",
        f"| RKM + Residual ML (Poly D2) | {hyb_all['MAE']:.2f} | {hyb_all['RMSE']:.2f} | {hyb_all['R2']:.4f} |",
        "",
        f"Reference (not recomputed here): existing direct Poly D2 on `xBi, xIn, T`, LOCSO — "
        f"MAE {DIRECT_POLY_D2_REF['MAE']:.2f}, RMSE {DIRECT_POLY_D2_REF['RMSE']:.2f}, "
        f"R² {DIRECT_POLY_D2_REF['R2']:.4f} (`reports/ml_composition_representation_results.md`).",
        "",
        "### By held-out cross-section",
        "",
        "| Cross-section | N | RKM MAE | Hybrid MAE | RKM RMSE | Hybrid RMSE | RKM R² | Hybrid R² |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s, r, h in by_section:
        L.append(
            f"| {SECTION_LABELS[s]} `{s}` | {r['N']} | {r['MAE']:.2f} | {h['MAE']:.2f} | "
            f"{r['RMSE']:.2f} | {h['RMSE']:.2f} | {r['R2']:.4f} | {h['R2']:.4f} |"
        )

    L += [
        "",
        "### By temperature",
        "",
        "| Temperature (K) | N | RKM MAE | Hybrid MAE | RKM RMSE | Hybrid RMSE | RKM R² | Hybrid R² |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for t, r, h in by_temp:
        L.append(
            f"| {int(t)} | {r['N']} | {r['MAE']:.2f} | {h['MAE']:.2f} | "
            f"{r['RMSE']:.2f} | {h['RMSE']:.2f} | {r['R2']:.4f} | {h['R2']:.4f} |"
        )

    L += [
        "",
        "### Residual-model diagnostics",
        "",
        "| Quantity | Value (J/mol) |",
        "| --- | ---: |",
        f"| Mean experimental residual (exp − RKM) | {diag['mean_residual']:.2f} |",
        f"| Std of experimental residual | {diag['std_residual']:.2f} |",
        f"| Mean predicted residual (OOF) | {diag['mean_predicted_residual']:.2f} |",
        f"| MAE of residual prediction | {diag['residual_prediction_MAE']:.2f} |",
        f"| RMSE of residual prediction | {diag['residual_prediction_RMSE']:.2f} |",
        f"| Max absolute residual-prediction error | {diag['max_abs_residual_error']:.2f} |",
        "",
        "Because ΔmixH_exp − ΔmixH_hybrid = residual − residual_ML, the residual-prediction MAE/RMSE equal the "
        "hybrid MAE/RMSE on ΔmixH.",
        "",
        "## Interpretation",
        "",
    ]

    if pooled_better:
        L.append(
            f"Pooled over all 104 out-of-fold predictions, RKM + residual ML **reduces the error relative to RKM**: "
            f"MAE {rkm_all['MAE']:.2f} → {hyb_all['MAE']:.2f} J/mol ({delta_text(rkm_all, hyb_all)}), "
            f"RMSE {rkm_all['RMSE']:.2f} → {hyb_all['RMSE']:.2f} J/mol, "
            f"R² {rkm_all['R2']:.4f} → {hyb_all['R2']:.4f}."
        )
    else:
        L.append(
            f"Pooled over all 104 out-of-fold predictions, RKM + residual ML **does not improve on RKM**: "
            f"MAE {rkm_all['MAE']:.2f} → {hyb_all['MAE']:.2f} J/mol ({delta_text(rkm_all, hyb_all)}), "
            f"RMSE {rkm_all['RMSE']:.2f} → {hyb_all['RMSE']:.2f} J/mol."
        )
    L.append("")
    for s, r, h in by_section:
        L.append(f"- {SECTION_LABELS[s]} held out: MAE {r['MAE']:.2f} → {h['MAE']:.2f} J/mol ({delta_text(r, h)}).")
    for t, r, h in by_temp:
        L.append(f"- {int(t)} K: MAE {r['MAE']:.2f} → {h['MAE']:.2f} J/mol ({delta_text(r, h)}).")
    L += [
        "",
        f"Sections where the hybrid MAE is lower than RKM: {', '.join(improved_sections) or 'none'}. "
        f"Sections where it is not: {', '.join(worse_sections) or 'none'}.",
        f"Temperatures where the hybrid MAE is lower than RKM: {', '.join(improved_temps) or 'none'}. "
        f"Temperatures where it is not: {', '.join(worse_temps) or 'none'}.",
        "",
        f"Against the existing direct Poly D2 reference (MAE {DIRECT_POLY_D2_REF['MAE']:.2f}, "
        f"RMSE {DIRECT_POLY_D2_REF['RMSE']:.2f}, R² {DIRECT_POLY_D2_REF['R2']:.4f}), the hybrid has "
        f"MAE {hyb_all['MAE'] - DIRECT_POLY_D2_REF['MAE']:+.2f} J/mol, "
        f"RMSE {hyb_all['RMSE'] - DIRECT_POLY_D2_REF['RMSE']:+.2f} J/mol and "
        f"R² {hyb_all['R2'] - DIRECT_POLY_D2_REF['R2']:+.4f}; these differences are small relative to the "
        "fold-to-fold variation and do not by themselves justify replacing the current model.",
        "",
        "Caveats: each LOCSO fold trains the residual model on only two composition paths (68–70 points), and "
        "each test fold has 34–36 points, so fold-level differences carry considerable uncertainty. RKM itself "
        "is temperature-independent, so any temperature dependence in the hybrid comes entirely from the "
        "residual model. Improvements on the three measured cross-sections do not establish behaviour in "
        "unsampled regions of the ternary.",
        "",
    ]

    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    exp = load_experimental()
    print(f"Experimental N = {len(exp)} (synthetic datasets not loaded)")

    pred, fold_info = run_locso(exp)
    for f in fold_info:
        print(
            f"Fold {f['fold']}: held out {f['held_out_section']} "
            f"(train N={f['n_train']}, test N={f['n_test']}, train sections={f['train_sections']})"
        )

    rkm_all, hyb_all = both_metrics(pred)
    print(f"RKM     pooled: MAE={rkm_all['MAE']:.2f} RMSE={rkm_all['RMSE']:.2f} R2={rkm_all['R2']:.4f}")
    print(f"Hybrid  pooled: MAE={hyb_all['MAE']:.2f} RMSE={hyb_all['RMSE']:.2f} R2={hyb_all['R2']:.4f}")
    for label, rows in (("section", grouped(pred, "held_out_section")), ("temperature", grouped(pred, "temperature_K"))):
        for key, r, h in rows:
            print(
                f"  {label} {key}: RKM MAE={r['MAE']:.2f} RMSE={r['RMSE']:.2f} R2={r['R2']:.4f} | "
                f"Hybrid MAE={h['MAE']:.2f} RMSE={h['RMSE']:.2f} R2={h['R2']:.4f}"
            )
    for k, v in residual_diagnostics(pred).items():
        print(f"  {k} = {v:.2f}")

    pred.to_csv(PRED_PATH, index=False)
    write_report(pred, fold_info)
    print(f"Wrote {PRED_PATH}")
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
