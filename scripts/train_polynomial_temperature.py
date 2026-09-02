"""
Polynomial degree-2 temperature-feature experiment (experimental-only LOCSO).

    python scripts/train_polynomial_temperature.py

Compares:
  - composition only: xBi, xIn, xSn
  - composition + temperature: xBi, xIn, xSn, temperature_K

No synthetic data. No hyperparameter tuning.
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

EXP_TARGET = "integral_mixing_enthalpy_J_per_mol"
DEGREE = 2
REPORT_PATH = ROOT / "reports" / "ml_polynomial_temperature_results.md"
PRED_PATH = ROOT / "reports" / "ml_polynomial_temperature_predictions.csv"

FEATURE_SETS = {
    "composition_only": ["xBi", "xIn", "xSn"],
    "composition_temperature": ["xBi", "xIn", "xSn", "temperature_K"],
}

# Step 3C/3E reference (experimental-only, poly deg 2, composition only)
COMP_ONLY_REFERENCE = {"MAE": 54.64, "RMSE": 71.23, "R2": 0.9540}
RKM_REFERENCE = {"MAE": 57.13, "RMSE": 73.49, "R2": 0.9510}

BI_RICH = "(Sn0.33Bi0.67)1-xInx"
HIGH_XIN_THRESHOLD = 0.70


def load_data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "original_experimental_data.csv")


def get_locsos_folds(exp: pd.DataFrame) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    sections = sorted(exp["cross_section"].unique())
    return [
        (held_out, exp[exp["cross_section"] != held_out].copy(), exp[exp["cross_section"] == held_out].copy())
        for held_out in sections
    ]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "N": int(len(y_true))}


def features(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return df[feature_cols].to_numpy(dtype=float)


def rkm_predict(df: pd.DataFrame) -> np.ndarray:
    return np.array([rkm_delta_mix_h(r.xBi, r.xIn, r.xSn) for r in df.itertuples()])


def train_polynomial(X_train: np.ndarray, y_train: np.ndarray, degree: int) -> tuple[PolynomialFeatures, LinearRegression]:
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X_train)
    coef, _, _, _ = np.linalg.lstsq(X_poly, y_train, rcond=None)
    model = LinearRegression(fit_intercept=False)
    model.coef_ = coef
    model.intercept_ = 0.0
    return poly, model


def predict_polynomial(poly: PolynomialFeatures, model: LinearRegression, X: np.ndarray) -> np.ndarray:
    return model.predict(poly.transform(X))


def run_locsos(feature_set: str, exp: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    feature_cols = FEATURE_SETS[feature_set]
    fold_metrics = []
    oof_rows = []

    for fold_idx, (held_out, exp_train, exp_test) in enumerate(get_locsos_folds(exp), start=1):
        X_train = features(exp_train, feature_cols)
        y_train = exp_train[EXP_TARGET].to_numpy(dtype=float)
        X_test = features(exp_test, feature_cols)
        y_test = exp_test[EXP_TARGET].to_numpy(dtype=float)

        poly, model = train_polynomial(X_train, y_train, DEGREE)
        ml_pred = predict_polynomial(poly, model, X_test)
        rkm_pred = rkm_predict(exp_test)

        ml_m = calculate_metrics(y_test, ml_pred)
        rkm_m = calculate_metrics(y_test, rkm_pred)

        fold_metrics.append(
            {
                "feature_set": feature_set,
                "fold": fold_idx,
                "held_out_section": held_out,
                "n_train": len(exp_train),
                "n_test": len(exp_test),
                "n_poly_features": int(poly.n_output_features_),
                "ml": ml_m,
                "rkm": rkm_m,
            }
        )

        for j, (_, row) in enumerate(exp_test.iterrows()):
            oof_rows.append(
                {
                    "experiment_id": row["id"],
                    "cross_section": row["cross_section"],
                    "temperature_K": row["temperature_K"],
                    "xBi": row["xBi"],
                    "xIn": row["xIn"],
                    "xSn": row["xSn"],
                    "experimental_delta_mixH": row[EXP_TARGET],
                    "rkm_prediction": float(rkm_pred[j]),
                    "ml_prediction": float(ml_pred[j]),
                    "fold": fold_idx,
                    "held_out_section": held_out,
                    "feature_set": feature_set,
                }
            )

    return fold_metrics, pd.DataFrame(oof_rows)


def grouped_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for key, grp in df.groupby(group_col, sort=True):
        m = calculate_metrics(grp["experimental_delta_mixH"], grp["ml_prediction"])
        rows.append({"group": key, "MAE": m["MAE"], "RMSE": m["RMSE"], "R2": m["R2"], "N": m["N"]})
    return pd.DataFrame(rows)


def sanity_checks(exp: pd.DataFrame, pred_df: pd.DataFrame) -> None:
    for fs in FEATURE_SETS:
        subset = pred_df[pred_df["feature_set"] == fs]
        if len(subset) != len(exp):
            raise RuntimeError(f"{fs}: expected {len(exp)} predictions, got {len(subset)}")
        if subset["experiment_id"].duplicated().any():
            raise RuntimeError(f"{fs}: duplicate experiment_id")

    for held_out, exp_train, exp_test in get_locsos_folds(exp):
        if set(exp_test["id"]) & set(exp_train["id"]):
            raise RuntimeError("Experimental leakage: test id in train")

    for col in FEATURE_SETS["composition_only"] + FEATURE_SETS["composition_temperature"]:
        if col == EXP_TARGET:
            raise RuntimeError("Target must not be used as a feature")

    print("Sanity checks passed.")


def format_metrics(m: dict[str, float]) -> str:
    return f"MAE = {m['MAE']:.2f} J/mol, RMSE = {m['RMSE']:.2f} J/mol, R² = {m['R2']:.4f} (N = {m['N']})"


def diagnostic_comparison(comp_df: pd.DataFrame, temp_df: pd.DataFrame) -> list[str]:
    merged = comp_df.merge(
        temp_df[["experiment_id", "ml_prediction"]].rename(columns={"ml_prediction": "ml_prediction_temp"}),
        on="experiment_id",
    )
    merged["residual_comp"] = merged["ml_prediction"] - merged["experimental_delta_mixH"]
    merged["residual_temp"] = merged["ml_prediction_temp"] - merged["experimental_delta_mixH"]
    merged["abs_improvement"] = merged["residual_comp"].abs() - merged["residual_temp"].abs()

    lines = ["## Diagnostic comparison (Step 3E focus regions)", ""]
    lines.append("Residual = prediction − experimental ΔmixH.")
    lines.append("Positive `abs_improvement` means composition+temperature has smaller absolute error.")
    lines.append("")

    def region_block(label: str, mask: pd.Series) -> None:
        grp = merged[mask]
        if len(grp) == 0:
            return
        comp_mae = calculate_metrics(grp["experimental_delta_mixH"], grp["ml_prediction"])["MAE"]
        temp_mae = calculate_metrics(grp["experimental_delta_mixH"], grp["ml_prediction_temp"])["MAE"]
        better = int((grp["abs_improvement"] > 0).sum())
        lines.append(f"### {label} (N = {len(grp)})")
        lines.append(f"- Composition-only MAE: {comp_mae:.2f} J/mol")
        lines.append(f"- Composition+temperature MAE: {temp_mae:.2f} J/mol")
        lines.append(f"- Temperature model better on {better}/{len(grp)} points")
        lines.append(f"- Mean residual (comp): {grp['residual_comp'].mean():.2f} → (temp): {grp['residual_temp'].mean():.2f}")
        lines.append("")

    region_block("Bi-rich cross-section", merged["cross_section"] == BI_RICH)
    region_block("767 K", merged["temperature_K"] == 767)
    region_block("High xIn (xIn ≥ 0.70)", merged["xIn"] >= HIGH_XIN_THRESHOLD)
    region_block(
        "Bi-rich + 767 K + high xIn",
        (merged["cross_section"] == BI_RICH) & (merged["temperature_K"] == 767) & (merged["xIn"] >= HIGH_XIN_THRESHOLD),
    )

    return lines


def write_report(
    all_metrics: list[dict],
    comp_oof: pd.DataFrame,
    temp_oof: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> None:
    lines = [
        "# Polynomial Degree 2 — Temperature Feature Experiment",
        "",
        "## Scientific question",
        "Does adding experimental temperature improve Poly D2 predictions of integral mixing enthalpy?",
        "",
        "## Setup",
        "- Model: PolynomialFeatures (degree 2) + minimum-norm OLS (same as Step 3C)",
        "- Training: **experimental-only**, LOCSO (3 folds)",
        "- No synthetic data, no hyperparameter tuning",
        "",
        "## Feature sets compared",
        "- `composition_only`: xBi, xIn, xSn",
        "- `composition_temperature`: xBi, xIn, xSn, temperature_K",
        "",
        "## Pooled metrics (104 OOF)",
        "",
        "| Model | MAE | RMSE | R² |",
        "| --- | ---: | ---: | ---: |",
    ]

    for fs, ref in [
        ("composition_only", COMP_ONLY_REFERENCE),
        ("composition_temperature", None),
    ]:
        oof = pred_df[pred_df["feature_set"] == fs]
        m = calculate_metrics(oof["experimental_delta_mixH"], oof["ml_prediction"])
        label = fs.replace("_", " ")
        lines.append(f"| Poly D2 {label} | {m['MAE']:.2f} | {m['RMSE']:.2f} | {m['R2']:.4f} |")

    rkm_m = calculate_metrics(comp_oof["experimental_delta_mixH"], comp_oof["rkm_prediction"])
    lines.append(f"| RKM (benchmark) | {rkm_m['MAE']:.2f} | {rkm_m['RMSE']:.2f} | {rkm_m['R2']:.4f} |")
    lines.append("")
    lines.append(
        f"Step 3C reference (composition-only): MAE = {COMP_ONLY_REFERENCE['MAE']:.2f}, "
        f"RMSE = {COMP_ONLY_REFERENCE['RMSE']:.2f}, R² = {COMP_ONLY_REFERENCE['R2']:.4f}"
    )
    lines.append("")

    for fs in FEATURE_SETS:
        lines.append(f"## Fold-level metrics — {fs.replace('_', ' ')}")
        lines.append("")
        fold_rows = [m for m in all_metrics if m["feature_set"] == fs]
        lines.append("| Fold | Held-out section | Train N | Test N | Poly features | MAE | RMSE | R² |")
        lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for fm in fold_rows:
            m = fm["ml"]
            lines.append(
                f"| {fm['fold']} | `{fm['held_out_section']}` | {fm['n_train']} | {fm['n_test']} | "
                f"{fm['n_poly_features']} | {m['MAE']:.2f} | {m['RMSE']:.2f} | {m['R2']:.4f} |"
            )
        oof = pred_df[pred_df["feature_set"] == fs]
        overall = calculate_metrics(oof["experimental_delta_mixH"], oof["ml_prediction"])
        lines.append("")
        lines.append(f"**Pooled (104 OOF):** {format_metrics(overall)}")
        lines.append("")

    lines.append("## Metrics by temperature")
    lines.append("")
    lines.append("| T (K) | N | Comp-only MAE | Comp-only RMSE | Comp+T MAE | Comp+T RMSE | RKM MAE |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    rkm_by_t = grouped_metrics(comp_oof.assign(ml_prediction=comp_oof["rkm_prediction"]), "temperature_K")
    comp_by_t = grouped_metrics(comp_oof, "temperature_K")
    temp_by_t = grouped_metrics(temp_oof, "temperature_K")

    for temp in sorted(comp_oof["temperature_K"].unique()):
        c = comp_by_t[comp_by_t["group"] == temp].iloc[0]
        t = temp_by_t[temp_by_t["group"] == temp].iloc[0]
        r = rkm_by_t[rkm_by_t["group"] == temp].iloc[0]
        lines.append(
            f"| {int(temp)} | {int(c['N'])} | {c['MAE']:.2f} | {c['RMSE']:.2f} | "
            f"{t['MAE']:.2f} | {t['RMSE']:.2f} | {r['MAE']:.2f} |"
        )
    lines.append("")

    lines.extend(diagnostic_comparison(comp_oof, temp_oof))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


def main() -> int:
    exp = load_data()
    print(f"Experimental N = {len(exp)}")

    all_metrics: list[dict] = []
    all_oof: list[pd.DataFrame] = []

    for feature_set in FEATURE_SETS:
        metrics, oof = run_locsos(feature_set, exp)
        all_metrics.extend(metrics)
        all_oof.append(oof)
        overall = calculate_metrics(oof["experimental_delta_mixH"], oof["ml_prediction"])
        print(
            f"{feature_set}: MAE={overall['MAE']:.2f}, "
            f"RMSE={overall['RMSE']:.2f}, R²={overall['R2']:.4f}"
        )

    pred_df = pd.concat(all_oof, ignore_index=True)
    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(PRED_PATH, index=False)
    print(f"Wrote {PRED_PATH}")

    sanity_checks(exp, pred_df)

    comp_oof = pred_df[pred_df["feature_set"] == "composition_only"].copy()
    temp_oof = pred_df[pred_df["feature_set"] == "composition_temperature"].copy()
    write_report(all_metrics, comp_oof, temp_oof, pred_df)

    rkm_m = calculate_metrics(comp_oof["experimental_delta_mixH"], comp_oof["rkm_prediction"])
    print(f"RKM overall: MAE={rkm_m['MAE']:.2f}, RMSE={rkm_m['RMSE']:.2f}, R²={rkm_m['R2']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
