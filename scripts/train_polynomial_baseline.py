"""
Polynomial regression baseline with LOCSO evaluation.

    python scripts/train_polynomial_baseline.py

Uses PolynomialFeatures (degree 2 and 3) + LinearRegression.
Same LOCSO splits and training experiments as scripts/train_baseline.py.
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

FEATURE_COLS = ["xBi", "xIn", "xSn"]
EXP_TARGET = "integral_mixing_enthalpy_J_per_mol"
SYN_TARGET = "delta_mix_H_J_per_mol"
DEGREES = (2, 3)
REPORT_PATH = ROOT / "reports" / "ml_polynomial_results.md"
PRED_PATH = ROOT / "reports" / "ml_polynomial_predictions.csv"

# Step 3B linear baseline (for comparison table only)
LINEAR_REFERENCE = {
    "experimental_only": {"MAE": 250.18, "RMSE": 297.61, "R2": 0.1967},
    "synthetic_only": {"MAE": 268.70, "RMSE": 320.21, "R2": 0.0701},
    "combined": {"MAE": 270.66, "RMSE": 323.00, "R2": 0.0538},
}
RKM_REFERENCE = {"MAE": 57.13, "RMSE": 73.49, "R2": 0.9510}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    exp = pd.read_csv(ROOT / "data" / "original_experimental_data.csv")
    syn_a = pd.read_csv(ROOT / "data" / "synthetic_cross_sections.csv")
    return exp, syn_a


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


def features(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLS].to_numpy(dtype=float)


def rkm_predict(df: pd.DataFrame) -> np.ndarray:
    return np.array([rkm_delta_mix_h(r.xBi, r.xIn, r.xSn) for r in df.itertuples()])


def train_polynomial(X_train: np.ndarray, y_train: np.ndarray, degree: int) -> tuple[PolynomialFeatures, LinearRegression]:
    """PolynomialFeatures + OLS via LinearRegression.

    include_bias=True with fit_intercept=False avoids a duplicate intercept column.
    Rank-deficient design matrices (compositional xBi+xIn+xSn=1 on 1D cross-sections)
    can make sklearn's default solver unstable; numpy lstsq gives the minimum-norm OLS
    solution, which we assign to the LinearRegression object for prediction.
    """
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X_train)
    coef, _, _, _ = np.linalg.lstsq(X_poly, y_train, rcond=None)
    model = LinearRegression(fit_intercept=False)
    model.coef_ = coef
    model.intercept_ = 0.0
    return poly, model


def predict_polynomial(poly: PolynomialFeatures, model: LinearRegression, X: np.ndarray) -> np.ndarray:
    return model.predict(poly.transform(X))


def build_exp_only(held_out: str, exp_train: pd.DataFrame, syn_a: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    _ = held_out, syn_a
    return exp_train.copy(), EXP_TARGET


def build_syn_only(held_out: str, exp_train: pd.DataFrame, syn_a: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    _ = exp_train
    return syn_a[syn_a["cross_section"] != held_out].copy(), SYN_TARGET


def build_combined(held_out: str, exp_train: pd.DataFrame, syn_a: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    syn_train = syn_a[syn_a["cross_section"] != held_out].copy()
    exp_part = exp_train[FEATURE_COLS].copy()
    exp_part["target"] = exp_train[EXP_TARGET].to_numpy()
    syn_part = syn_train[FEATURE_COLS].copy()
    syn_part["target"] = syn_train[SYN_TARGET].to_numpy()
    return pd.concat([exp_part, syn_part], ignore_index=True), "target"


def run_locsos_experiment(
    name: str,
    degree: int,
    exp: pd.DataFrame,
    syn_a: pd.DataFrame,
    build_train_fn,
) -> tuple[list[dict], pd.DataFrame]:
    fold_metrics = []
    oof_rows = []

    for fold_idx, (held_out, exp_train, exp_test) in enumerate(get_locsos_folds(exp), start=1):
        train_df, target_col = build_train_fn(held_out, exp_train, syn_a)

        X_train = features(train_df)
        y_train = train_df[target_col].to_numpy(dtype=float)
        X_test = features(exp_test)
        y_test = exp_test[EXP_TARGET].to_numpy(dtype=float)

        poly, model = train_polynomial(X_train, y_train, degree)
        ml_pred = predict_polynomial(poly, model, X_test)
        rkm_pred = rkm_predict(exp_test)

        ml_m = calculate_metrics(y_test, ml_pred)
        rkm_m = calculate_metrics(y_test, rkm_pred)

        fold_metrics.append(
            {
                "training_type": name,
                "model_degree": degree,
                "fold": fold_idx,
                "held_out_section": held_out,
                "n_train": len(train_df),
                "n_test": len(exp_test),
                "n_features": int(poly.n_output_features_),
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
                    "training_type": name,
                    "model_degree": degree,
                }
            )

    return fold_metrics, pd.DataFrame(oof_rows)


def sanity_checks(exp: pd.DataFrame, syn_a: pd.DataFrame, pred_df: pd.DataFrame) -> None:
    for degree in DEGREES:
        for tt in ("experimental_only", "synthetic_only", "combined"):
            subset = pred_df[(pred_df["training_type"] == tt) & (pred_df["model_degree"] == degree)]
            if len(subset) != len(exp):
                raise RuntimeError(f"{tt} degree {degree}: expected {len(exp)} predictions, got {len(subset)}")
            if subset["experiment_id"].duplicated().any():
                raise RuntimeError(f"{tt} degree {degree}: duplicate experiment_id")

    for held_out, exp_train, exp_test in get_locsos_folds(exp):
        if set(exp_test["id"]) & set(exp_train["id"]):
            raise RuntimeError("Experimental leakage: test id in train")
        syn_train = syn_a[syn_a["cross_section"] != held_out]
        if (syn_train["cross_section"] == held_out).any():
            raise RuntimeError(f"Synthetic leakage: held-out section {held_out} in synthetic train")

    print("Sanity checks passed.")


def format_metrics(m: dict[str, float]) -> str:
    return f"MAE = {m['MAE']:.2f} J/mol, RMSE = {m['RMSE']:.2f} J/mol, R² = {m['R2']:.4f} (N = {m['N']})"


def write_report(all_metrics: list[dict], pred_df: pd.DataFrame) -> None:
    lines = [
        "# ML Polynomial Baseline Results",
        "",
        "## 1. Model",
        "PolynomialFeatures (include_bias=True) + LinearRegression (fit_intercept=False)",
        "",
        "No duplicate intercept column. OLS coefficients from minimum-norm least squares",
        "(numpy lstsq) when the expanded design matrix is rank-deficient on compositional data.",
        "",
        "## 2. Features",
        "xBi, xIn, xSn (expanded to polynomial terms per degree)",
        "",
        "## 3. Target",
        "ΔmixH — experimental: `integral_mixing_enthalpy_J_per_mol`; synthetic train: `delta_mix_H_J_per_mol`",
        "",
        "## 4. Validation",
        "Leave-one-cross-section-out (LOCSO), same folds as Step 3B.",
        "",
    ]

    for degree in DEGREES:
        lines.append(f"## Degree {degree} results")
        lines.append("")
        for training_type in ("experimental_only", "synthetic_only", "combined"):
            lines.append(f"### {training_type.replace('_', ' ').title()}")
            lines.append("")
            fold_rows = [m for m in all_metrics if m["model_degree"] == degree and m["training_type"] == training_type]
            for fm in fold_rows:
                lines.append(f"#### Fold {fm['fold']} — held out: `{fm['held_out_section']}`")
                lines.append(f"- Train N = {fm['n_train']}, Test N = {fm['n_test']}, poly features = {fm['n_features']}")
                lines.append(f"- ML: {format_metrics(fm['ml'])}")
                lines.append(f"- RKM: {format_metrics(fm['rkm'])}")
                lines.append("")
            oof = pred_df[(pred_df["model_degree"] == degree) & (pred_df["training_type"] == training_type)]
            overall = calculate_metrics(oof["experimental_delta_mixH"], oof["ml_prediction"])
            lines.append(f"**Overall (104 OOF):** {format_metrics(overall)}")
            lines.append("")

    lines.extend(["## Comparison summary (overall 104 OOF)", ""])
    lines.append("| Training type | Linear (Step 3B) | Poly deg 2 | Poly deg 3 | RKM |")
    lines.append("| --- | --- | --- | --- | --- |")

    rkm_oof = pred_df[pred_df["model_degree"] == 2].drop_duplicates("experiment_id")
    rkm_m = calculate_metrics(rkm_oof["experimental_delta_mixH"], rkm_oof["rkm_prediction"])

    for tt in ("experimental_only", "synthetic_only", "combined"):
        lin = LINEAR_REFERENCE[tt]
        row = [tt]
        row.append(f"MAE {lin['MAE']:.1f}, R² {lin['R2']:.3f}")
        for degree in DEGREES:
            oof = pred_df[(pred_df["model_degree"] == degree) & (pred_df["training_type"] == tt)]
            m = calculate_metrics(oof["experimental_delta_mixH"], oof["ml_prediction"])
            row.append(f"MAE {m['MAE']:.1f}, R² {m['R2']:.3f}")
        if tt == "experimental_only":
            row.append(f"MAE {rkm_m['MAE']:.2f}, R² {rkm_m['R2']:.4f}")
        else:
            row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append(f"RKM overall (all experiments): MAE = {rkm_m['MAE']:.2f} J/mol, RMSE = {rkm_m['RMSE']:.2f} J/mol, R² = {rkm_m['R2']:.4f}")
    lines.append("")
    lines.append("Reference: MAE = 57.13 J/mol, RMSE = 73.49 J/mol, R² = 0.9510")
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


def main() -> int:
    exp, syn_a = load_data()
    print(f"Experimental N = {len(exp)}, Dataset A N = {len(syn_a)}")

    all_metrics: list[dict] = []
    all_oof: list[pd.DataFrame] = []

    builders = [
        ("experimental_only", build_exp_only),
        ("synthetic_only", build_syn_only),
        ("combined", build_combined),
    ]

    for degree in DEGREES:
        for name, build_fn in builders:
            metrics, oof = run_locsos_experiment(name, degree, exp, syn_a, build_fn)
            all_metrics.extend(metrics)
            all_oof.append(oof)
            overall = calculate_metrics(oof["experimental_delta_mixH"], oof["ml_prediction"])
            print(
                f"degree={degree} {name}: MAE={overall['MAE']:.2f}, "
                f"RMSE={overall['RMSE']:.2f}, R²={overall['R2']:.4f}"
            )

    pred_df = pd.concat(all_oof, ignore_index=True)
    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(PRED_PATH, index=False)
    print(f"Wrote {PRED_PATH}")

    sanity_checks(exp, syn_a, pred_df)
    write_report(all_metrics, pred_df)

    rkm_m = calculate_metrics(
        pred_df[pred_df["model_degree"] == 2].drop_duplicates("experiment_id")["experimental_delta_mixH"],
        pred_df[pred_df["model_degree"] == 2].drop_duplicates("experiment_id")["rkm_prediction"],
    )
    print(f"RKM overall: MAE={rkm_m['MAE']:.2f}, RMSE={rkm_m['RMSE']:.2f}, R²={rkm_m['R2']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
