"""
Compositional feature-representation check (experimental-only LOCSO).

    python scripts/analyze_composition_representation.py

Compares temperature-aware Poly D2 with:
  A) xBi, xIn, xSn, temperature_K  (composition_temperature_3vars)
  B) xBi, xIn, temperature_K       (composition_temperature_2vars; xSn redundant)

No synthetic data. No new model family.
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
REPORT_PATH = ROOT / "reports" / "ml_composition_representation_results.md"
PRED_PATH = ROOT / "reports" / "ml_composition_representation_predictions.csv"

FEATURE_SETS = {
    "composition_temperature_3vars": ["xBi", "xIn", "xSn", "temperature_K"],
    "composition_temperature_2vars": ["xBi", "xIn", "temperature_K"],
}

STEP3F_REF = {"MAE": 43.81, "RMSE": 61.17, "R2": 0.9661}
RKM_REF = {"MAE": 57.13, "RMSE": 73.49, "R2": 0.9510}


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


def train_polynomial(X_train: np.ndarray, y_train: np.ndarray) -> tuple[PolynomialFeatures, LinearRegression]:
    poly = PolynomialFeatures(degree=DEGREE, include_bias=True)
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

        poly, model = train_polynomial(X_train, y_train)
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

    if EXP_TARGET in FEATURE_SETS["composition_temperature_3vars"]:
        raise RuntimeError("Target must not be used as a feature")

    print("Sanity checks passed.")


def format_metrics(m: dict[str, float]) -> str:
    return f"MAE = {m['MAE']:.2f} J/mol, RMSE = {m['RMSE']:.2f} J/mol, R² = {m['R2']:.4f} (N = {m['N']})"


def material_change(m3: dict[str, float], m2: dict[str, float], threshold_mae: float = 0.5) -> str:
    delta_mae = abs(m3["MAE"] - m2["MAE"])
    delta_r2 = abs(m3["R2"] - m2["R2"])
    if delta_mae < threshold_mae and delta_r2 < 0.005:
        return "Removing xSn does **not** materially change performance."
    return (
        f"Removing xSn changes performance (ΔMAE = {m2['MAE'] - m3['MAE']:+.2f}, "
        f"ΔR² = {m2['R2'] - m3['R2']:+.4f})."
    )


def write_report(all_metrics: list[dict], pred_df: pd.DataFrame) -> None:
    oof3 = pred_df[pred_df["feature_set"] == "composition_temperature_3vars"]
    oof2 = pred_df[pred_df["feature_set"] == "composition_temperature_2vars"]
    m3 = calculate_metrics(oof3["experimental_delta_mixH"], oof3["ml_prediction"])
    m2 = calculate_metrics(oof2["experimental_delta_mixH"], oof2["ml_prediction"])
    rkm_m = calculate_metrics(oof3["experimental_delta_mixH"], oof3["rkm_prediction"])

    lines = [
        "# Compositional Feature Representation Check",
        "",
        "## Purpose",
        "Test whether temperature-aware Poly D2 is sensitive to including redundant xSn",
        "given xBi + xIn + xSn = 1.",
        "",
        "## Setup",
        "- Model: PolynomialFeatures (degree 2) + minimum-norm OLS",
        "- Training: experimental-only LOCSO (3 folds)",
        "- No synthetic data",
        "",
        "## Representations",
        "- **Model A (`composition_temperature_3vars`):** xBi, xIn, xSn, temperature_K",
        "- **Model B (`composition_temperature_2vars`):** xBi, xIn, temperature_K (xSn omitted)",
        "",
        "## Pooled metrics (104 OOF)",
        "",
        "| Model | MAE | RMSE | R² | Poly input features |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for fs, label in [
        ("composition_temperature_3vars", "Model A — 3 composition vars + T"),
        ("composition_temperature_2vars", "Model B — 2 composition vars + T"),
    ]:
        oof = pred_df[pred_df["feature_set"] == fs]
        m = calculate_metrics(oof["experimental_delta_mixH"], oof["ml_prediction"])
        n_poly = next(x["n_poly_features"] for x in all_metrics if x["feature_set"] == fs)
        lines.append(f"| {label} | {m['MAE']:.2f} | {m['RMSE']:.2f} | {m['R2']:.4f} | {n_poly} |")

    lines.append(f"| RKM (benchmark) | {rkm_m['MAE']:.2f} | {rkm_m['RMSE']:.2f} | {rkm_m['R2']:.4f} | — |")
    lines.append("")
    lines.append(
        f"Step 3F reference (Model A): MAE = {STEP3F_REF['MAE']:.2f}, "
        f"RMSE = {STEP3F_REF['RMSE']:.2f}, R² = {STEP3F_REF['R2']:.4f}"
    )
    lines.append("")

    for fs, label in [
        ("composition_temperature_3vars", "Model A (3 vars + T)"),
        ("composition_temperature_2vars", "Model B (2 vars + T)"),
    ]:
        lines.append(f"## Fold-level — {label}")
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
    lines.append("| T (K) | N | Model A MAE | Model A RMSE | Model B MAE | Model B RMSE | RKM MAE |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    rkm_by_t = grouped_metrics(oof3.assign(ml_prediction=oof3["rkm_prediction"]), "temperature_K")
    a_by_t = grouped_metrics(oof3, "temperature_K")
    b_by_t = grouped_metrics(oof2, "temperature_K")

    for temp in sorted(oof3["temperature_K"].unique()):
        a = a_by_t[a_by_t["group"] == temp].iloc[0]
        b = b_by_t[b_by_t["group"] == temp].iloc[0]
        r = rkm_by_t[rkm_by_t["group"] == temp].iloc[0]
        lines.append(
            f"| {int(temp)} | {int(a['N'])} | {a['MAE']:.2f} | {a['RMSE']:.2f} | "
            f"{b['MAE']:.2f} | {b['RMSE']:.2f} | {r['MAE']:.2f} |"
        )
    lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    lines.append(material_change(m3, m2))
    lines.append("")
    lines.append(
        f"- Model A vs Step 3F: MAE Δ = {m3['MAE'] - STEP3F_REF['MAE']:+.2f}, "
        f"R² Δ = {m3['R2'] - STEP3F_REF['R2']:+.4f}"
    )
    lines.append(
        f"- Model B vs Model A: MAE Δ = {m2['MAE'] - m3['MAE']:+.2f}, "
        f"RMSE Δ = {m2['RMSE'] - m3['RMSE']:+.2f}, R² Δ = {m2['R2'] - m3['R2']:+.4f}"
    )
    lines.append("")

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
    write_report(all_metrics, pred_df)

    rkm_m = calculate_metrics(
        pred_df[pred_df["feature_set"] == "composition_temperature_3vars"]["experimental_delta_mixH"],
        pred_df[pred_df["feature_set"] == "composition_temperature_3vars"]["rkm_prediction"],
    )
    print(f"RKM overall: MAE={rkm_m['MAE']:.2f}, RMSE={rkm_m['RMSE']:.2f}, R²={rkm_m['R2']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
