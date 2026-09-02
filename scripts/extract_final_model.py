"""
Extract and document the final frozen Poly D2 model coefficients.

    python scripts/extract_final_model.py

Trains on all 104 experimental observations (same config as final surrogate).
Reports coefficients and training-set fit — NOT primary validation metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rkm_model import rkm_delta_mix_h

FEATURE_COLS = ["xBi", "xIn", "temperature_K"]
FEATURE_LABELS = ["xBi", "xIn", "temperature_K"]
EXP_TARGET = "integral_mixing_enthalpy_J_per_mol"
DEGREE = 2
REPORT_PATH = ROOT / "reports" / "final_model_equation.md"
COEF_PATH = ROOT / "reports" / "final_model_coefficients.csv"

LOCSO_REF = {"MAE": 41.10, "RMSE": 56.34, "R2": 0.9712}
RKM_REF = {"MAE": 57.13, "RMSE": 73.49, "R2": 0.9510}


def load_experimental() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "original_experimental_data.csv")


def features(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLS].to_numpy(dtype=float)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "N": int(len(y_true))}


def train_final_model(exp: pd.DataFrame) -> tuple[PolynomialFeatures, LinearRegression, np.ndarray]:
    X = features(exp)
    y = exp[EXP_TARGET].to_numpy(dtype=float)
    poly = PolynomialFeatures(degree=DEGREE, include_bias=True)
    X_poly = poly.fit_transform(X)
    coef, _, _, _ = np.linalg.lstsq(X_poly, y, rcond=None)
    model = LinearRegression(fit_intercept=False)
    model.coef_ = coef
    model.intercept_ = 0.0
    return poly, model, coef


def pretty_term(sklearn_name: str) -> str:
    mapping = {
        "1": "intercept",
        "xBi": "xBi",
        "xIn": "xIn",
        "temperature_K": "temperature_K",
        "xBi^2": "xBi²",
        "xBi xIn": "xBi·xIn",
        "xBi temperature_K": "xBi·temperature_K",
        "xIn^2": "xIn²",
        "xIn temperature_K": "xIn·temperature_K",
        "temperature_K^2": "temperature_K²",
    }
    return mapping.get(sklearn_name, sklearn_name)


def format_equation_term(coef: float, term: str) -> str:
    if term == "intercept":
        return f"{coef:+.4f}"
    sign = "+" if coef >= 0 else "−"
    return f"{sign} {abs(coef):.4f}·{term}"


def build_coefficient_table(poly: PolynomialFeatures, coef: np.ndarray) -> pd.DataFrame:
    names = poly.get_feature_names_out(FEATURE_LABELS)
    rows = []
    for name, c in zip(names, coef, strict=True):
        display = pretty_term(str(name))
        rows.append(
            {
                "term_sklearn": str(name),
                "term_display": display,
                "coefficient": float(c),
                "sign": "positive" if c >= 0 else "negative",
                "magnitude": float(abs(c)),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    coef_df: pd.DataFrame,
    train_ml: dict[str, float],
    train_rkm: dict[str, float],
    equation_lines: list[str],
) -> None:
    lines = [
        "# Final Polynomial Degree-2 Model Equation",
        "",
        "## Frozen model specification",
        "- `PolynomialFeatures(degree=2, include_bias=True)` + minimum-norm OLS",
        "- Features: **xBi, xIn, temperature_K** (xSn omitted; see below)",
        "- Target: `integral_mixing_enthalpy_J_per_mol` (ΔmixH, J/mol)",
        "- Training data: all **104** original experimental observations",
        "",
        "## Primary validated performance (LOCSO — do not replace with training metrics)",
        "",
        "These are the **primary defensible** out-of-fold metrics from Step 3H:",
        "",
        f"- **ML (LOCSO OOF):** MAE = {LOCSO_REF['MAE']:.2f} J/mol, RMSE = {LOCSO_REF['RMSE']:.2f} J/mol, R² = {LOCSO_REF['R2']:.4f}",
        f"- **RKM (same 104 test points):** MAE = {RKM_REF['MAE']:.2f} J/mol, RMSE = {RKM_REF['RMSE']:.2f} J/mol, R² = {RKM_REF['R2']:.4f}",
        "",
        "## Training-set fit (104 observations — not primary validation)",
        "",
        "The model below is refit on **all 104 points** for coefficient extraction.",
        "These in-sample metrics are reported for completeness only:",
        "",
        f"- ML (training): MAE = {train_ml['MAE']:.2f} J/mol, RMSE = {train_ml['RMSE']:.2f} J/mol, R² = {train_ml['R2']:.4f}",
        f"- RKM (same 104 points): MAE = {train_rkm['MAE']:.2f} J/mol, RMSE = {train_rkm['RMSE']:.2f} J/mol, R² = {train_rkm['R2']:.4f}",
        "",
        "## Fitted equation",
        "",
        "ΔmixH (J/mol) =",
        "",
    ]
    lines.extend(equation_lines)
    lines.append("")

    lines.extend(
        [
            "## Coefficient table",
            "",
            "| Term | Coefficient (J/mol per term unit) | Sign | |coefficient| |",
            "| --- | ---: | --- | ---: |",
        ]
    )
    for _, row in coef_df.iterrows():
        lines.append(
            f"| {row['term_display']} | {row['coefficient']:.6f} | {row['sign']} | {row['magnitude']:.6f} |"
        )

    lines.extend(
        [
            "",
            "Note: |coefficient| alone does **not** rank physical importance because xBi, xIn",
            "are mole fractions (0–1) while temperature_K is hundreds of kelvin.",
            "",
            "## Why degree 2 gives ten terms",
            "",
            "With three input variables (xBi, xIn, T) and degree 2, PolynomialFeatures builds:",
            "",
            "1. intercept (constant 1)",
            "2. three linear terms: xBi, xIn, T",
            "3. six quadratic terms: xBi², xIn², T², xBi·xIn, xBi·T, xIn·T",
            "",
            "Total = 1 + 3 + 6 = **10** terms.",
            "",
            "There are no xSn terms because xSn was deliberately omitted from the feature vector.",
            "",
            "## Why xSn is absent and how it is recovered",
            "",
            "Mole fractions satisfy **xBi + xIn + xSn = 1**, so xSn is not an independent variable.",
            "Given any (xBi, xIn) within the ternary simplex:",
            "",
            "**xSn = 1 − xBi − xIn**",
            "",
            "Including both xSn and (xBi, xIn) would create redundant columns and an ill-conditioned",
            "polynomial design matrix (as seen in Step 3H).",
            "",
            "## Meaning of interaction terms (mathematical, not thermodynamic)",
            "",
            "- **xBi·xIn**: allows the effect of In to depend on Bi level (and vice versa) beyond linear addition.",
            "- **xBi·T** and **xIn·T**: allow composition effects to change with temperature.",
            "- **xBi², xIn², T²**: allow curvature along each axis.",
            "",
            "These are **empirical regression shapes**, not Redlich–Kister or RKM interaction parameters.",
            "",
            "## Coefficients are not thermodynamic interaction parameters",
            "",
            "The fitted numbers are ordinary least-squares weights chosen to minimize error on 104",
            "calorimetry points. They should **not** be read as Lᵢⱼ or ternary indices from Table IV.",
            "",
            "Different variable scaling (especially T in kelvin vs mole fractions) means coefficient",
            "magnitude comparisons across term types are misleading without standardization.",
            "",
            "## Comparison with RKM",
            "",
            "| Aspect | RKM (Eq. 4, Table IV) | Final ML polynomial |",
            "| --- | --- | --- |",
            "| Origin | Paper thermodynamic model | Fit to 104 experiments |",
            "| Parameters | Fixed Lᵢⱼ from literature | 10 learned coefficients |",
            "| Temperature | Not used in current implementation | Explicit feature T |",
            "| Composition | xBi, xIn, xSn in formula | xBi, xIn (+ xSn = 1−xBi−xIn) |",
            "| Validation | MAE 57.13 on 104 exp. points | LOCSO MAE 41.10 on 104 OOF points |",
            "",
            "RKM remains the physics-based benchmark; ML captures experimental scatter and",
            "temperature trends that the temperature-independent RKM surface cannot represent.",
            "",
        ]
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


def main() -> int:
    exp = load_experimental()
    print(f"Experimental N = {len(exp)}")

    poly, model, coef = train_final_model(exp)
    coef_df = build_coefficient_table(poly, coef)
    COEF_PATH.parent.mkdir(parents=True, exist_ok=True)
    coef_df.to_csv(COEF_PATH, index=False)
    print(f"Wrote {COEF_PATH}")

    y = exp[EXP_TARGET].to_numpy(dtype=float)
    ml_pred = model.predict(poly.transform(features(exp)))
    rkm_pred = np.array([rkm_delta_mix_h(r.xBi, r.xIn, r.xSn) for r in exp.itertuples()])

    train_ml = calculate_metrics(y, ml_pred)
    train_rkm = calculate_metrics(y, rkm_pred)

    equation_lines = []
    for i, row in coef_df.iterrows():
        equation_lines.append(f"  {format_equation_term(row['coefficient'], row['term_display'])}")

    write_report(coef_df, train_ml, train_rkm, equation_lines)

    print("\nCoefficients:")
    for _, row in coef_df.iterrows():
        print(f"  {row['term_display']:20s} {row['coefficient']:+.6f}")

    print(
        f"\nTraining fit: MAE={train_ml['MAE']:.2f}, RMSE={train_ml['RMSE']:.2f}, R²={train_ml['R2']:.4f}"
    )
    print(
        f"LOCSO reference (primary): MAE={LOCSO_REF['MAE']:.2f}, "
        f"RMSE={LOCSO_REF['RMSE']:.2f}, R²={LOCSO_REF['R2']:.4f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
