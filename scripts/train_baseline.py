"""
First ML baseline: Linear Regression with LOCSO evaluation.

    python scripts/train_baseline.py

Experiments:
  A) experimental-only
  B) synthetic-only (Dataset A, held-out section excluded from train)
  C) combined (experiment + synthetic from non-held-out sections)

No hyperparameter tuning. Composition features only: xBi, xIn, xSn.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rkm_model import rkm_delta_mix_h

FEATURE_COLS = ["xBi", "xIn", "xSn"]
EXP_TARGET = "integral_mixing_enthalpy_J_per_mol"
SYN_TARGET = "delta_mix_H_J_per_mol"
REPORT_PATH = ROOT / "reports" / "ml_baseline_results.md"
PRED_PATH = ROOT / "reports" / "ml_baseline_predictions.csv"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    exp = pd.read_csv(ROOT / "data" / "original_experimental_data.csv")
    syn_a = pd.read_csv(ROOT / "data" / "synthetic_cross_sections.csv")
    return exp, syn_a


def get_locsos_folds(exp: pd.DataFrame) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Return (held_out_section, train_df, test_df) for each cross-section."""
    sections = sorted(exp["cross_section"].unique())
    folds = []
    for held_out in sections:
        test = exp[exp["cross_section"] == held_out].copy()
        train = exp[exp["cross_section"] != held_out].copy()
        folds.append((held_out, train, test))
    return folds


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "N": int(len(y_true))}


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def predict(model: LinearRegression, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


def rkm_predict(df: pd.DataFrame) -> np.ndarray:
    return np.array([rkm_delta_mix_h(r.xBi, r.xIn, r.xSn) for r in df.itertuples()])


def features(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLS].to_numpy(dtype=float)


def run_locsos_experiment(
    name: str,
    exp: pd.DataFrame,
    syn_a: pd.DataFrame,
    build_train_fn,
) -> tuple[list[dict], pd.DataFrame]:
    """
    build_train_fn(held_out_section, exp_train, syn_a) -> train DataFrame with FEATURE_COLS + target column name in 'y'
    Returns fold metrics and out-of-fold prediction rows.
    """
    fold_metrics = []
    oof_rows = []

    for fold_idx, (held_out, exp_train, exp_test) in enumerate(get_locsos_folds(exp), start=1):
        train_df, target_col = build_train_fn(held_out, exp_train, syn_a)

        X_train = features(train_df)
        y_train = train_df[target_col].to_numpy(dtype=float)
        X_test = features(exp_test)
        y_test = exp_test[EXP_TARGET].to_numpy(dtype=float)

        model = train_model(X_train, y_train)
        ml_pred = predict(model, X_test)
        rkm_pred = rkm_predict(exp_test)

        ml_m = calculate_metrics(y_test, ml_pred)
        rkm_m = calculate_metrics(y_test, rkm_pred)

        fold_metrics.append(
            {
                "experiment": name,
                "fold": fold_idx,
                "held_out_section": held_out,
                "n_train": len(train_df),
                "n_test": len(exp_test),
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
                }
            )

    return fold_metrics, pd.DataFrame(oof_rows)


def build_exp_only(held_out: str, exp_train: pd.DataFrame, syn_a: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    _ = held_out, syn_a
    out = exp_train.copy()
    return out, EXP_TARGET


def build_syn_only(held_out: str, exp_train: pd.DataFrame, syn_a: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    _ = exp_train
    syn_train = syn_a[syn_a["cross_section"] != held_out].copy()
    return syn_train, SYN_TARGET


def build_combined(held_out: str, exp_train: pd.DataFrame, syn_a: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    syn_train = syn_a[syn_a["cross_section"] != held_out].copy()
    exp_part = exp_train[FEATURE_COLS].copy()
    exp_part["target"] = exp_train[EXP_TARGET].to_numpy()
    syn_part = syn_train[FEATURE_COLS].copy()
    syn_part["target"] = syn_train[SYN_TARGET].to_numpy()
    return pd.concat([exp_part, syn_part], ignore_index=True), "target"


def sanity_checks(
    exp: pd.DataFrame,
    syn_a: pd.DataFrame,
    exp_folds: list,
    syn_folds: list,
    comb_folds: list,
    pred_df: pd.DataFrame,
) -> None:
    sections = sorted(exp["cross_section"].unique())

    for name, fold_list, build_fn in [
        ("experimental_only", exp_folds, build_exp_only),
        ("synthetic_only", syn_folds, build_syn_only),
        ("combined", comb_folds, build_combined),
    ]:
        subset = pred_df[pred_df["training_type"] == name]
        if len(subset) != len(exp):
            raise RuntimeError(f"{name}: expected {len(exp)} predictions, got {len(subset)}")
        if subset["experiment_id"].duplicated().any():
            raise RuntimeError(f"{name}: duplicate experiment_id in predictions")

    for held_out, exp_train, exp_test in get_locsos_folds(exp):
        test_ids = set(exp_test["id"])
        train_ids = set(exp_train["id"])
        if test_ids & train_ids:
            raise RuntimeError("Experimental leakage: test id in train")

        syn_train = syn_a[syn_a["cross_section"] != held_out]
        if (syn_train["cross_section"] == held_out).any():
            raise RuntimeError(f"Synthetic leakage: held-out section {held_out} in synthetic train")

    for tt in pred_df["training_type"].unique():
        ids = set(pred_df[pred_df["training_type"] == tt]["experiment_id"])
        if ids != set(exp["id"]):
            raise RuntimeError(f"Missing experiment ids for {tt}")

    _ = sections
    print("Sanity checks passed.")


def format_metrics(m: dict[str, float]) -> str:
    return f"MAE = {m['MAE']:.2f} J/mol, RMSE = {m['RMSE']:.2f} J/mol, R² = {m['R2']:.4f} (N = {m['N']})"


def write_report(
    exp_metrics: list,
    syn_metrics: list,
    comb_metrics: list,
    exp_oof: pd.DataFrame,
    syn_oof: pd.DataFrame,
    comb_oof: pd.DataFrame,
) -> None:
    def overall(oof: pd.DataFrame) -> dict:
        return calculate_metrics(
            oof["experimental_delta_mixH"].to_numpy(),
            oof["ml_prediction"].to_numpy(),
        )

    def overall_rkm(oof: pd.DataFrame) -> dict:
        return calculate_metrics(
            oof["experimental_delta_mixH"].to_numpy(),
            oof["rkm_prediction"].to_numpy(),
        )

    exp_all = overall(exp_oof)
    syn_all = overall(syn_oof)
    comb_all = overall(comb_oof)
    rkm_all = overall_rkm(exp_oof)

    lines = [
        "# ML Baseline Results",
        "",
        "## 1. Model",
        "Linear Regression (`sklearn.linear_model.LinearRegression`)",
        "",
        "## 2. Features",
        "xBi, xIn, xSn",
        "",
        "## 3. Target",
        "ΔmixH — experimental: `integral_mixing_enthalpy_J_per_mol`; synthetic train: `delta_mix_H_J_per_mol`",
        "",
        "## 4. Validation",
        "Leave-one-cross-section-out (LOCSO) on the three experimental cross-sections.",
        "",
        "## 5. Experimental-only results",
        "",
    ]

    for fm in exp_metrics:
        lines.append(f"### Fold {fm['fold']} — held out: `{fm['held_out_section']}`")
        lines.append(f"- Train N = {fm['n_train']}, Test N = {fm['n_test']}")
        lines.append(f"- ML: {format_metrics(fm['ml'])}")
        lines.append(f"- RKM (same test points): {format_metrics(fm['rkm'])}")
        lines.append("")

    lines.append("### Overall (all 104 out-of-fold predictions)")
    lines.append(f"- ML: {format_metrics(exp_all)}")
    lines.append("")

    lines.extend(["## 6. Synthetic-only → experimental results", ""])
    for fm in syn_metrics:
        lines.append(f"### Fold {fm['fold']} — held out: `{fm['held_out_section']}`")
        lines.append(f"- Train N = {fm['n_train']} (Dataset A, excluding held-out section)")
        lines.append(f"- Test N = {fm['n_test']} (experimental)")
        lines.append(f"- ML: {format_metrics(fm['ml'])}")
        lines.append(f"- RKM (same test points): {format_metrics(fm['rkm'])}")
        lines.append("")

    lines.append("### Overall (all 104 out-of-fold predictions)")
    lines.append(f"- ML: {format_metrics(syn_all)}")
    lines.append("")

    lines.extend(["## 7. Combined experimental + synthetic results", ""])
    for fm in comb_metrics:
        lines.append(f"### Fold {fm['fold']} — held out: `{fm['held_out_section']}`")
        lines.append(f"- Train N = {fm['n_train']} (experiment + Dataset A, excluding held-out section)")
        lines.append(f"- Test N = {fm['n_test']} (experimental)")
        lines.append(f"- ML: {format_metrics(fm['ml'])}")
        lines.append(f"- RKM (same test points): {format_metrics(fm['rkm'])}")
        lines.append("")

    lines.append("### Overall (all 104 out-of-fold predictions)")
    lines.append(f"- ML: {format_metrics(comb_all)}")
    lines.append("")

    lines.extend(["## 8. RKM benchmark", ""])
    for fm in exp_metrics:
        lines.append(f"### Fold {fm['fold']} — held out: `{fm['held_out_section']}`")
        lines.append(f"- RKM: {format_metrics(fm['rkm'])}")
        lines.append("")

    lines.append("### Overall (all 104 experimental points, out-of-fold)")
    lines.append(f"- RKM: {format_metrics(rkm_all)}")
    lines.append("")
    lines.append("Reference validation on all 104 points: MAE = 57.13 J/mol, RMSE = 73.49 J/mol, R² = 0.9510")
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


def main() -> int:
    exp, syn_a = load_data()
    sections = sorted(exp["cross_section"].unique())
    print("Cross-sections:", sections)
    print(f"Experimental N = {len(exp)}, Dataset A N = {len(syn_a)}")

    exp_metrics, exp_oof = run_locsos_experiment("experimental_only", exp, syn_a, build_exp_only)
    syn_metrics, syn_oof = run_locsos_experiment("synthetic_only", exp, syn_a, build_syn_only)
    comb_metrics, comb_oof = run_locsos_experiment("combined", exp, syn_a, build_combined)

    pred_df = pd.concat([exp_oof, syn_oof, comb_oof], ignore_index=True)
    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(PRED_PATH, index=False)
    print(f"Wrote {PRED_PATH}")

    sanity_checks(exp, syn_a, exp_metrics, syn_metrics, comb_metrics, pred_df)
    write_report(exp_metrics, syn_metrics, comb_metrics, exp_oof, syn_oof, comb_oof)

    # Print summary
    for label, oof in [("Experimental-only", exp_oof), ("Synthetic-only", syn_oof), ("Combined", comb_oof)]:
        m = calculate_metrics(oof["experimental_delta_mixH"], oof["ml_prediction"])
        print(f"{label} overall ML: MAE={m['MAE']:.2f}, RMSE={m['RMSE']:.2f}, R²={m['R2']:.4f}")

    rkm_m = calculate_metrics(exp_oof["experimental_delta_mixH"], exp_oof["rkm_prediction"])
    print(f"RKM overall: MAE={rkm_m['MAE']:.2f}, RMSE={rkm_m['RMSE']:.2f}, R²={rkm_m['R2']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
