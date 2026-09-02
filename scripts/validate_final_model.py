"""
Secondary validation: random 5-fold CV of the final Poly D2 baseline.

    python scripts/validate_final_model.py

Final model: PolynomialFeatures(degree=2) + OLS on xBi, xIn, temperature_K.
Primary validation remains LOCSO (reported for context only; not rerun here).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rkm_model import rkm_delta_mix_h

FEATURE_COLS = ["xBi", "xIn", "temperature_K"]
EXP_TARGET = "integral_mixing_enthalpy_J_per_mol"
DEGREE = 2
VALIDATION_TYPE = "random_5fold"
REPORT_PATH = ROOT / "reports" / "final_model_validation.md"
PRED_PATH = ROOT / "reports" / "final_model_5fold_predictions.csv"

# Primary LOCSO reference (Step 3H, not rerun)
LOCSO_REF = {"MAE": 41.10, "RMSE": 56.34, "R2": 0.9712}
RKM_REF = {"MAE": 57.13, "RMSE": 73.49, "R2": 0.9510}


def load_data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "original_experimental_data.csv")


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


def run_random_5fold(exp: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    oof_rows = []

    X_all = features(exp)
    y_all = exp[EXP_TARGET].to_numpy(dtype=float)
    rkm_all = rkm_predict(exp)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_all), start=1):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        poly, model = train_polynomial(X_train, y_train)
        ml_pred = predict_polynomial(poly, model, X_test)
        rkm_pred = rkm_all[test_idx]

        ml_m = calculate_metrics(y_test, ml_pred)
        rkm_m = calculate_metrics(y_test, rkm_pred)

        fold_metrics.append(
            {
                "fold": fold_idx,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "ml": ml_m,
                "rkm": rkm_m,
            }
        )

        test_df = exp.iloc[test_idx]
        for j, (_, row) in enumerate(test_df.iterrows()):
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
                    "validation_type": VALIDATION_TYPE,
                }
            )

    return fold_metrics, pd.DataFrame(oof_rows)


def sanity_checks(exp: pd.DataFrame, pred_df: pd.DataFrame) -> None:
    if len(pred_df) != len(exp):
        raise RuntimeError(f"expected {len(exp)} predictions, got {len(pred_df)}")
    if pred_df["experiment_id"].duplicated().any():
        raise RuntimeError("duplicate experiment_id in OOF predictions")
    if set(pred_df["experiment_id"]) != set(exp["id"]):
        raise RuntimeError("missing or extra experiment ids")
    if (pred_df["validation_type"] != VALIDATION_TYPE).any():
        raise RuntimeError("unexpected validation_type")
    if EXP_TARGET in FEATURE_COLS:
        raise RuntimeError("target must not be used as a feature")
    print("Sanity checks passed.")


def format_metrics(m: dict[str, float]) -> str:
    return f"MAE = {m['MAE']:.2f} J/mol, RMSE = {m['RMSE']:.2f} J/mol, R² = {m['R2']:.4f} (N = {m['N']})"


def write_report(fold_metrics: list[dict], oof: pd.DataFrame) -> None:
    ml_overall = calculate_metrics(oof["experimental_delta_mixH"], oof["ml_prediction"])
    rkm_overall = calculate_metrics(oof["experimental_delta_mixH"], oof["rkm_prediction"])

    delta_mae = ml_overall["MAE"] - LOCSO_REF["MAE"]
    substantially_better = ml_overall["MAE"] < LOCSO_REF["MAE"] - 5.0

    lines = [
        "# Final Model Validation",
        "",
        "## Final candidate model",
        "PolynomialFeatures(degree=2) + LinearRegression (minimum-norm OLS)",
        "",
        "Features: xBi, xIn, temperature_K",
        "",
        "Target: `integral_mixing_enthalpy_J_per_mol`",
        "",
        "Data: 104 original experimental observations only.",
        "",
        "## Validation design",
        "",
        "### Primary validation: LOCSO (leave-one-cross-section-out)",
        "Held-out entire composition cross-sections. Tests extrapolation along unseen paths.",
        "This remains the **primary defensible metric** for this project.",
        "",
        f"Reference result (Step 3H, not rerun): MAE = {LOCSO_REF['MAE']:.2f}, "
        f"RMSE = {LOCSO_REF['RMSE']:.2f}, R² = {LOCSO_REF['R2']:.4f}",
        "",
        "### Secondary validation: random 5-fold CV",
        "`KFold(n_splits=5, shuffle=True, random_state=42)`",
        "",
        "Points on the same cross-section (and often neighbouring xIn values) can appear in "
        "both training and test folds. This is an **easier, interpolation-oriented** check.",
        "",
        "## Secondary validation — pooled metrics (104 OOF)",
        "",
        f"- **Poly D2 (random 5-fold):** {format_metrics(ml_overall)}",
        f"- **RKM (same 104 points):** {format_metrics(rkm_overall)}",
        "",
        "## Fold-level metrics (random 5-fold)",
        "",
        "| Fold | Train N | Test N | Poly D2 MAE | Poly D2 RMSE | Poly D2 R² | RKM MAE |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for fm in fold_metrics:
        m, r = fm["ml"], fm["rkm"]
        lines.append(
            f"| {fm['fold']} | {fm['n_train']} | {fm['n_test']} | "
            f"{m['MAE']:.2f} | {m['RMSE']:.2f} | {m['R2']:.4f} | {r['MAE']:.2f} |"
        )

    lines.extend(
        [
            "",
            f"**Pooled (104 OOF):** {format_metrics(ml_overall)}",
            "",
            "## Comparison summary",
            "",
            "| Validation | Model | MAE | RMSE | R² |",
            "| --- | --- | ---: | ---: | ---: |",
            f"| Primary LOCSO | Poly D2 | {LOCSO_REF['MAE']:.2f} | {LOCSO_REF['RMSE']:.2f} | {LOCSO_REF['R2']:.4f} |",
            f"| Secondary 5-fold | Poly D2 | {ml_overall['MAE']:.2f} | {ml_overall['RMSE']:.2f} | {ml_overall['R2']:.4f} |",
            f"| All 104 points | RKM | {rkm_overall['MAE']:.2f} | {rkm_overall['RMSE']:.2f} | {rkm_overall['R2']:.4f} |",
            "",
            "## Interpretation",
            "",
        ]
    )

    if substantially_better:
        lines.append(
            f"Random 5-fold MAE ({ml_overall['MAE']:.2f}) is **substantially better** than LOCSO "
            f"({LOCSO_REF['MAE']:.2f}; ΔMAE = {delta_mae:+.2f})."
        )
    elif ml_overall["MAE"] < LOCSO_REF["MAE"]:
        lines.append(
            f"Random 5-fold MAE ({ml_overall['MAE']:.2f}) is **modestly better** than LOCSO "
            f"({LOCSO_REF['MAE']:.2f}; ΔMAE = {delta_mae:+.2f})."
        )
    else:
        lines.append(
            f"Random 5-fold MAE ({ml_overall['MAE']:.2f}) is **not better** than LOCSO "
            f"({LOCSO_REF['MAE']:.2f}; ΔMAE = {delta_mae:+.2f})."
        )

    lines.extend(
        [
            "",
            "A better 5-fold score **should be expected** when it occurs: neighbouring compositions "
            "on the same measured cross-section often leak between folds, so the model interpolates "
            "rather than extrapolates to a full unseen path. LOCSO is stricter and more aligned with "
            "the paper's experimental design.",
            "",
            f"RKM reference (frozen): MAE = {RKM_REF['MAE']:.2f}, RMSE = {RKM_REF['RMSE']:.2f}, R² = {RKM_REF['R2']:.4f}",
            "",
        ]
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


def main() -> int:
    exp = load_data()
    print(f"Experimental N = {len(exp)}")

    fold_metrics, oof = run_random_5fold(exp)
    ml_m = calculate_metrics(oof["experimental_delta_mixH"], oof["ml_prediction"])
    print(f"Random 5-fold Poly D2: MAE={ml_m['MAE']:.2f}, RMSE={ml_m['RMSE']:.2f}, R²={ml_m['R2']:.4f}")

    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(PRED_PATH, index=False)
    print(f"Wrote {PRED_PATH}")

    sanity_checks(exp, oof)
    write_report(fold_metrics, oof)

    rkm_m = calculate_metrics(oof["experimental_delta_mixH"], oof["rkm_prediction"])
    print(f"RKM (same points): MAE={rkm_m['MAE']:.2f}, RMSE={rkm_m['RMSE']:.2f}, R²={rkm_m['R2']:.4f}")
    print(f"LOCSO reference: MAE={LOCSO_REF['MAE']:.2f}, RMSE={LOCSO_REF['RMSE']:.2f}, R²={LOCSO_REF['R2']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
