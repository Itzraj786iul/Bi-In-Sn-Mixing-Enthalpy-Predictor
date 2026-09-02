"""
Gradient Boosting experimental-only LOCSO baseline.

    python scripts/train_gradient_boosting.py

Features: xBi, xIn, temperature_K (no xSn).
Fixed GradientBoostingRegressor config — no hyperparameter tuning.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rkm_model import rkm_delta_mix_h

FEATURE_COLS = ["xBi", "xIn", "temperature_K"]
EXP_TARGET = "integral_mixing_enthalpy_J_per_mol"
MODEL_NAME = "gradient_boosting"
REPORT_PATH = ROOT / "reports" / "ml_gradient_boosting_results.md"
PRED_PATH = ROOT / "reports" / "ml_gradient_boosting_predictions.csv"

BI_RICH = "(Sn0.33Bi0.67)1-xInx"
HIGH_XIN_THRESHOLD = 0.70

# Prior baselines (comparison table only, experimental-only LOCSO)
REFERENCES = {
    "Linear Regression": {"MAE": 250.18, "RMSE": 297.61, "R2": 0.1967},
    "Poly D2 composition-only": {"MAE": 54.64, "RMSE": 71.23, "R2": 0.9540},
    "Poly D2 comp+T (3 vars)": {"MAE": 43.81, "RMSE": 61.17, "R2": 0.9661},
    "Poly D2 comp+T (2 vars)": {"MAE": 41.10, "RMSE": 56.34, "R2": 0.9712},
    "Random Forest": {"MAE": 131.24, "RMSE": 170.67, "R2": 0.7358},
    "RKM": {"MAE": 57.13, "RMSE": 73.49, "R2": 0.9510},
}
BEST_BASELINE = REFERENCES["Poly D2 comp+T (2 vars)"]


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
    residual = y_pred - y_true
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual, ddof=0)),
        "max_abs_residual": float(np.max(np.abs(residual))),
        "N": int(len(y_true)),
    }


def features(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLS].to_numpy(dtype=float)


def rkm_predict(df: pd.DataFrame) -> np.ndarray:
    return np.array([rkm_delta_mix_h(r.xBi, r.xIn, r.xSn) for r in df.itertuples()])


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def run_locsos(exp: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    fold_metrics = []
    oof_rows = []

    for fold_idx, (held_out, exp_train, exp_test) in enumerate(get_locsos_folds(exp), start=1):
        X_train = features(exp_train)
        y_train = exp_train[EXP_TARGET].to_numpy(dtype=float)
        X_test = features(exp_test)
        y_test = exp_test[EXP_TARGET].to_numpy(dtype=float)

        model = train_model(X_train, y_train)
        ml_pred = model.predict(X_test)
        rkm_pred = rkm_predict(exp_test)

        ml_m = calculate_metrics(y_test, ml_pred)
        rkm_m = calculate_metrics(y_test, rkm_pred)

        fold_metrics.append(
            {
                "fold": fold_idx,
                "held_out_section": held_out,
                "n_train": len(exp_train),
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
                    "model": MODEL_NAME,
                }
            )

    return fold_metrics, pd.DataFrame(oof_rows)


def grouped_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for key, grp in df.groupby(group_col, sort=True):
        m = calculate_metrics(grp["experimental_delta_mixH"], grp["ml_prediction"])
        rows.append({"group": key, **m})
    return pd.DataFrame(rows)


def sanity_checks(exp: pd.DataFrame, pred_df: pd.DataFrame) -> None:
    if len(pred_df) != len(exp):
        raise RuntimeError(f"expected {len(exp)} predictions, got {len(pred_df)}")
    if pred_df["experiment_id"].duplicated().any():
        raise RuntimeError("duplicate experiment_id")
    if set(pred_df["experiment_id"]) != set(exp["id"]):
        raise RuntimeError("missing or extra experiment ids")

    for held_out, exp_train, exp_test in get_locsos_folds(exp):
        if set(exp_test["id"]) & set(exp_train["id"]):
            raise RuntimeError("Experimental leakage: test id in train")

    if EXP_TARGET in FEATURE_COLS:
        raise RuntimeError("Target must not be used as a feature")

    print("Sanity checks passed.")


def format_metrics(m: dict[str, float]) -> str:
    return f"MAE = {m['MAE']:.2f} J/mol, RMSE = {m['RMSE']:.2f} J/mol, R² = {m['R2']:.4f} (N = {m['N']})"


def worst_observations(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    out = df.copy()
    out["residual"] = out["ml_prediction"] - out["experimental_delta_mixH"]
    out["abs_residual"] = out["residual"].abs()
    return out.sort_values("abs_residual", ascending=False).head(n)


def region_diagnostics(df: pd.DataFrame) -> list[str]:
    df = df.copy()
    df["residual"] = df["ml_prediction"] - df["experimental_delta_mixH"]

    lines = ["## Regional diagnostics (Step 3E focus)", ""]
    regions = [
        ("Bi-rich fold (fold 1)", df["fold"] == 1),
        ("High xIn (xIn ≥ 0.70)", df["xIn"] >= HIGH_XIN_THRESHOLD),
        ("767 K", df["temperature_K"] == 767),
        ("855 K", df["temperature_K"] == 855),
        (
            "Bi-rich + high xIn",
            (df["cross_section"] == BI_RICH) & (df["xIn"] >= HIGH_XIN_THRESHOLD),
        ),
    ]
    for label, mask in regions:
        grp = df[mask]
        if len(grp) == 0:
            continue
        m = calculate_metrics(grp["experimental_delta_mixH"], grp["ml_prediction"])
        lines.append(f"### {label} (N = {len(grp)})")
        lines.append(f"- MAE = {m['MAE']:.2f}, RMSE = {m['RMSE']:.2f}, R² = {m['R2']:.4f}")
        lines.append(f"- Residual mean = {m['residual_mean']:.2f}, std = {m['residual_std']:.2f}")
        lines.append("")
    return lines


def write_report(fold_metrics: list[dict], oof: pd.DataFrame) -> None:
    overall = calculate_metrics(oof["experimental_delta_mixH"], oof["ml_prediction"])
    rkm_overall = calculate_metrics(oof["experimental_delta_mixH"], oof["rkm_prediction"])

    lines = [
        "# Gradient Boosting Baseline Results",
        "",
        "## Model",
        "`GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=2, random_state=42)`",
        "",
        "## Features",
        "xBi, xIn, temperature_K (experimental-only LOCSO)",
        "",
        "## Pooled metrics (104 OOF)",
        "",
        f"- Gradient Boosting: {format_metrics(overall)}",
        f"- Residual mean = {overall['residual_mean']:.2f}, std = {overall['residual_std']:.2f}, "
        f"max |residual| = {overall['max_abs_residual']:.2f}",
        f"- RKM: {format_metrics(rkm_overall)}",
        "",
        "## Comparison with prior baselines (104 OOF, experimental-only)",
        "",
        "| Model | MAE | RMSE | R² |",
        "| --- | ---: | ---: | ---: |",
    ]

    for name, ref in REFERENCES.items():
        lines.append(f"| {name} | {ref['MAE']:.2f} | {ref['RMSE']:.2f} | {ref['R2']:.4f} |")

    gb_row = f"| **Gradient Boosting (this step)** | **{overall['MAE']:.2f}** | **{overall['RMSE']:.2f}** | **{overall['R2']:.4f}** |"
    lines.append(gb_row)
    lines.append("")

    delta = overall["MAE"] - BEST_BASELINE["MAE"]
    if overall["MAE"] < BEST_BASELINE["MAE"]:
        verdict = f"Gradient Boosting **improves** on the current best Poly D2 baseline (ΔMAE = {delta:.2f})."
    elif abs(delta) < 0.5:
        verdict = f"Gradient Boosting is **approximately unchanged** vs Poly D2 2-var baseline (ΔMAE = {delta:+.2f})."
    else:
        verdict = f"Gradient Boosting is **worse** than the current best Poly D2 baseline (ΔMAE = {delta:+.2f})."
    lines.append(f"**Versus best Poly D2 (MAE 41.10):** {verdict}")
    lines.append("")

    lines.append("## Fold-level metrics")
    lines.append("")
    lines.append("| Fold | Held-out section | Train N | Test N | GB MAE | GB RMSE | GB R² | RKM MAE |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for fm in fold_metrics:
        m, r = fm["ml"], fm["rkm"]
        lines.append(
            f"| {fm['fold']} | `{fm['held_out_section']}` | {fm['n_train']} | {fm['n_test']} | "
            f"{m['MAE']:.2f} | {m['RMSE']:.2f} | {m['R2']:.4f} | {r['MAE']:.2f} |"
        )
    lines.append("")
    lines.append(f"**Pooled:** {format_metrics(overall)}")
    lines.append("")

    lines.append("## Metrics by temperature")
    lines.append("")
    lines.append("| T (K) | N | GB MAE | GB RMSE | GB R² | RKM MAE | Poly D2 2-var MAE* |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    poly2_by_t = {767: 46.36, 813: 41.26, 855: 35.69}  # Step 3H reference
    gb_by_t = grouped_metrics(oof, "temperature_K")
    rkm_by_t = grouped_metrics(oof.assign(ml_prediction=oof["rkm_prediction"]), "temperature_K")
    for temp in sorted(oof["temperature_K"].unique()):
        g = gb_by_t[gb_by_t["group"] == temp].iloc[0]
        r = rkm_by_t[rkm_by_t["group"] == temp].iloc[0]
        p2 = poly2_by_t[int(temp)]
        lines.append(
            f"| {int(temp)} | {int(g['N'])} | {g['MAE']:.2f} | {g['RMSE']:.2f} | {g['R2']:.4f} | "
            f"{r['MAE']:.2f} | {p2:.2f} |"
        )
    lines.append("")
    lines.append("*Poly D2 2-var MAE from Step 3H `ml_composition_representation_results.md`.")
    lines.append("")

    lines.append("## Residual diagnostics")
    lines.append("")
    lines.append(f"- Residual mean: {overall['residual_mean']:.2f} J/mol")
    lines.append(f"- Residual std: {overall['residual_std']:.2f} J/mol")
    lines.append(f"- Max |residual|: {overall['max_abs_residual']:.2f} J/mol")
    lines.append("")
    lines.append("### Largest absolute residuals")
    lines.append("")
    lines.append("| experiment_id | cross_section | T (K) | xIn | experimental | prediction | residual |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for _, row in worst_observations(oof).iterrows():
        lines.append(
            f"| {row['experiment_id']} | `{row['cross_section']}` | {int(row['temperature_K'])} | "
            f"{row['xIn']:.4f} | {row['experimental_delta_mixH']:.1f} | "
            f"{row['ml_prediction']:.1f} | {row['residual']:.1f} |"
        )
    lines.append("")

    lines.extend(region_diagnostics(oof))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


def main() -> int:
    exp = load_data()
    print(f"Experimental N = {len(exp)}")

    fold_metrics, oof = run_locsos(exp)
    overall = calculate_metrics(oof["experimental_delta_mixH"], oof["ml_prediction"])
    print(f"Gradient Boosting: MAE={overall['MAE']:.2f}, RMSE={overall['RMSE']:.2f}, R²={overall['R2']:.4f}")

    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(PRED_PATH, index=False)
    print(f"Wrote {PRED_PATH}")

    sanity_checks(exp, oof)
    write_report(fold_metrics, oof)

    rkm_m = calculate_metrics(oof["experimental_delta_mixH"], oof["rkm_prediction"])
    print(f"RKM overall: MAE={rkm_m['MAE']:.2f}, RMSE={rkm_m['RMSE']:.2f}, R²={rkm_m['R2']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
