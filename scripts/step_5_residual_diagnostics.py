"""
Residual-learning diagnostic analysis (diagnosis only; no model changes).

    python scripts/step_5_residual_diagnostics.py

Reads the existing residual-learning out-of-fold predictions
(reports/residual_learning_results.csv, produced by
scripts/step_4_residual_learning.py) and the 104 experimental observations,
and characterises why RKM + residual Poly D2 improves RKM overall but not on
the Sn-rich held-out cross-section.

No model is refit except to reproduce the existing residual model's
in-sample fit per fold (same features, same fitting code) for the
train-versus-test comparison. No synthetic data are loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_composition_representation import (  # noqa: E402
    EXP_TARGET,
    calculate_metrics,
    predict_polynomial,
    train_polynomial,
)
from step_4_residual_learning import FEATURES  # noqa: E402

EXP_PATH = ROOT / "data" / "original_experimental_data.csv"
OOF_PATH = ROOT / "reports" / "residual_learning_results.csv"
REPORT_PATH = ROOT / "reports" / "residual_diagnostic_analysis.md"
FIG_DIR = ROOT / "figures"

SECTION_ORDER = ["(Sn0.33Bi0.67)1-xInx", "(Sn0.50Bi0.50)1-xInx", "(Sn0.67Bi0.33)1-xInx"]
SECTION_LABELS = {
    "(Sn0.33Bi0.67)1-xInx": "Bi-rich",
    "(Sn0.50Bi0.50)1-xInx": "Equiatomic",
    "(Sn0.67Bi0.33)1-xInx": "Sn-rich",
}
SECTION_COLORS = {"Bi-rich": "tab:blue", "Equiatomic": "tab:green", "Sn-rich": "tab:red"}
TEMP_MARKERS = {767: "o", 813: "s", 855: "^"}
SN_RICH = "(Sn0.67Bi0.33)1-xInx"

XIN_BINS = [0.09, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.91]
REGION_BINS = [0.0, 0.30, 0.60, 1.0]
REGION_LABELS = ["In-lean (xIn < 0.30)", "Mid (0.30 ≤ xIn < 0.60)", "In-rich (xIn ≥ 0.60)"]


def load() -> pd.DataFrame:
    exp = pd.read_csv(EXP_PATH)
    oof = pd.read_csv(OOF_PATH)
    if len(exp) != 104 or len(oof) != 104:
        raise RuntimeError("Expected 104 experimental and 104 out-of-fold rows")
    if set(exp["id"]) != set(oof["experiment_id"]):
        raise RuntimeError("Out-of-fold file does not match experimental IDs")
    merged = oof.merge(exp[["id", EXP_TARGET]], left_on="experiment_id", right_on="id")
    if not np.allclose(merged[EXP_TARGET], merged["experimental_delta_mixH"]):
        raise RuntimeError("Out-of-fold experimental targets differ from data file")

    df = oof.copy()
    df["section"] = df["cross_section"].map(SECTION_LABELS)
    df["held_out"] = df["held_out_section"].map(SECTION_LABELS)
    df["bi_ratio"] = df["xBi"] / (df["xBi"] + df["xSn"])
    df["rkm_error"] = df["experimental_delta_mixH"] - df["rkm_prediction"]
    df["hybrid_error"] = df["experimental_delta_mixH"] - df["hybrid_prediction"]
    df["rkm_abs_error"] = df["rkm_error"].abs()
    df["hybrid_abs_error"] = df["hybrid_error"].abs()
    df["residual_pred_error"] = df["residual"] - df["predicted_residual"]
    df["xin_bin"] = pd.cut(df["xIn"], XIN_BINS, include_lowest=True)
    df["region"] = pd.cut(df["xIn"], REGION_BINS, labels=REGION_LABELS, right=False)
    return df


# ---------------------------------------------------------------- statistics


def residual_stats(g: pd.DataFrame) -> dict:
    r = g["residual"]
    return {
        "n": len(g),
        "mean": r.mean(),
        "median": r.median(),
        "std": r.std(ddof=1) if len(g) > 1 else float("nan"),
        "rkm_mae": g["rkm_abs_error"].mean(),
        "rkm_rmse": float(np.sqrt(np.mean(g["rkm_error"] ** 2))),
        "mean_abs": r.abs().mean(),
        "min": r.min(),
        "max": r.max(),
        "hybrid_mae": g["hybrid_abs_error"].mean(),
        "hybrid_rmse": float(np.sqrt(np.mean(g["hybrid_error"] ** 2))),
        "mean_pred": g["predicted_residual"].mean(),
    }


def table(df: pd.DataFrame, col: str, order=None) -> pd.DataFrame:
    rows = []
    keys = order if order is not None else sorted(df[col].dropna().unique())
    for k in keys:
        g = df[df[col] == k]
        if len(g):
            rows.append({"group": k, **residual_stats(g)})
    return pd.DataFrame(rows)


def bias_variance(g: pd.DataFrame) -> dict:
    e = g["residual_pred_error"].to_numpy()
    mse = float(np.mean(e**2))
    bias = float(np.mean(e))
    return {
        "mse": mse,
        "bias": bias,
        "bias_sq_share": bias**2 / mse if mse > 0 else float("nan"),
        "var_share": float(np.var(e)) / mse if mse > 0 else float("nan"),
        "corr": float(np.corrcoef(g["residual"], g["predicted_residual"])[0, 1]),
    }


def fold_train_fit(df: pd.DataFrame, held_out_label: str) -> dict:
    """Reproduce the existing fold's residual model to report its in-sample fit."""
    train = df[df["section"] != held_out_label]
    poly, model = train_polynomial(train[FEATURES].to_numpy(float), train["residual"].to_numpy(float))
    fitted = predict_polynomial(poly, model, train[FEATURES].to_numpy(float))
    test = df[df["section"] == held_out_label]
    test_pred = predict_polynomial(poly, model, test[FEATURES].to_numpy(float))
    if not np.allclose(test_pred, test["predicted_residual"].to_numpy(float), atol=1e-6):
        raise RuntimeError("Reproduced fold predictions differ from residual_learning_results.csv")
    m = calculate_metrics(train["residual"].to_numpy(float), fitted)
    return {"train_mae": m["MAE"], "train_rmse": m["RMSE"], "train_mean_resid": train["residual"].mean()}


def coverage(df: pd.DataFrame, held_out_label: str) -> dict:
    train = df[df["section"] != held_out_label]
    test = df[df["section"] == held_out_label]
    hull = Delaunay(train[["xBi", "xIn"]].to_numpy(float))
    inside = hull.find_simplex(test[["xBi", "xIn"]].to_numpy(float)) >= 0
    tr = train[["xBi", "xIn", "xSn"]].to_numpy(float)
    te = test[["xBi", "xIn", "xSn"]].to_numpy(float)
    nn = np.min(np.linalg.norm(te[:, None, :] - tr[None, :, :], axis=2), axis=1)
    return {
        "train_ratio_range": (train["bi_ratio"].min(), train["bi_ratio"].max()),
        "test_ratio": test["bi_ratio"].mean(),
        "frac_inside_hull": float(inside.mean()),
        "n_inside": int(inside.sum()),
        "outside_xin": sorted(round(float(v), 4) for v in test.loc[~inside, "xIn"]),
        "n_test": len(test),
        "nn_median": float(np.median(nn)),
        "nn_max": float(np.max(nn)),
        "train_xbi": (train["xBi"].min(), train["xBi"].max()),
        "test_xbi": (test["xBi"].min(), test["xBi"].max()),
        "train_xsn": (train["xSn"].min(), train["xSn"].max()),
        "test_xsn": (test["xSn"].min(), test["xSn"].max()),
        "train_xin": (train["xIn"].min(), train["xIn"].max()),
        "test_xin": (test["xIn"].min(), test["xIn"].max()),
    }


def matched_profile_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Mean residual per section on common xIn bins, per temperature."""
    g = df.groupby(["temperature_K", "xin_bin", "section"], observed=True)["residual"].mean().unstack("section")
    return g[[SECTION_LABELS[s] for s in SECTION_ORDER]]


# ---------------------------------------------------------------- figures


def scatter_by_section(ax, df, ycol):
    ax.figure.set_size_inches(8, 6)
    for sec in [SECTION_LABELS[s] for s in SECTION_ORDER]:
        for t, mk in TEMP_MARKERS.items():
            g = df[(df["section"] == sec) & (df["temperature_K"] == t)]
            ax.scatter(g["xIn"], g[ycol], c=SECTION_COLORS[sec], marker=mk, s=28, alpha=0.8,
                       edgecolors="none", label=f"{sec}, {t} K")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("xIn")
    ax.grid(alpha=0.3)


def make_figures(df: pd.DataFrame) -> list[str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    names = []

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter_by_section(ax, df, "residual")
    ax.set_ylabel("RKM residual = ΔmixH_exp − ΔmixH_RKM (J/mol)")
    ax.set_title("RKM residual vs xIn (104 experimental observations)")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residual_vs_xIn.png", dpi=150)
    plt.close(fig)
    names.append("residual_vs_xIn.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter_by_section(ax, df, "residual_pred_error")
    ax.set_ylabel("Residual − predicted residual (J/mol)\n(= hybrid error, held-out LOCSO)")
    ax.set_title("Residual-prediction error vs xIn (each point predicted with its section held out)")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residual_prediction_error_vs_xIn.png", dpi=150)
    plt.close(fig)
    names.append("residual_prediction_error_vs_xIn.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    labels = [SECTION_LABELS[s] for s in SECTION_ORDER]
    for ax, col, title in zip(axes, ["rkm_abs_error", "hybrid_abs_error"], ["RKM", "RKM + residual Poly D2 (LOCSO)"]):
        data = [df.loc[df["section"] == s, col] for s in labels]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55)
        ax.set_xticks(range(1, len(labels) + 1), labels)
        for patch, s in zip(bp["boxes"], labels):
            patch.set_facecolor(SECTION_COLORS[s])
            patch.set_alpha(0.35)
        for i, s in enumerate(labels, start=1):
            y = df.loc[df["section"] == s, col]
            ax.scatter(np.full(len(y), i) + np.random.default_rng(0).uniform(-0.12, 0.12, len(y)), y,
                       s=10, c=SECTION_COLORS[s], alpha=0.7)
            ax.text(i, y.max() + 8, f"MAE {y.mean():.1f}", ha="center", fontsize=8)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("|ΔmixH_exp − prediction| (J/mol)")
    fig.suptitle("Absolute error by cross-section")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rkm_vs_hybrid_error_by_section.png", dpi=150)
    plt.close(fig)
    names.append("rkm_vs_hybrid_error_by_section.png")

    sn = df[df["section"] == "Sn-rich"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, t in zip(axes, sorted(TEMP_MARKERS)):
        g = sn[sn["temperature_K"] == t].sort_values("xIn")
        ax.plot(g["xIn"], g["residual"], "o-", c="tab:red", label="actual residual")
        ax.plot(g["xIn"], g["predicted_residual"], "s--", c="tab:gray", label="predicted residual (held out)")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"Sn-rich, {t} K")
        ax.set_xlabel("xIn")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Residual (J/mol)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residual_sn_rich_actual_vs_predicted.png", dpi=150)
    plt.close(fig)
    names.append("residual_sn_rich_actual_vs_predicted.png")

    fig, ax = plt.subplots(figsize=(6, 5.5))
    for sec in [SECTION_LABELS[s] for s in SECTION_ORDER]:
        g = df[df["section"] == sec]
        ax.scatter(g["xBi"], g["xIn"], c=SECTION_COLORS[sec], s=18, label=sec)
    for r in (0.33, 0.50, 0.67):
        ax.plot([0, r], [1, 0], c="lightgray", lw=0.8, zorder=0)
    ax.plot([0, 1, 0, 0], [0, 0, 1, 0], c="black", lw=0.8)
    ax.set_xlabel("xBi")
    ax.set_ylabel("xIn")
    ax.set_title("Experimental composition paths (xSn = 1 − xBi − xIn)")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residual_composition_coverage.png", dpi=150)
    plt.close(fig)
    names.append("residual_composition_coverage.png")
    return names


# ---------------------------------------------------------------- main


def fmt_stats_table(t: pd.DataFrame, label: str) -> list[str]:
    L = [
        f"| {label} | n | Mean resid. | Median | Std | Min | Max | Mean abs. resid. | RKM MAE | RKM RMSE | Hybrid MAE | Hybrid RMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in t.iterrows():
        L.append(
            f"| {r['group']} | {r['n']} | {r['mean']:.2f} | {r['median']:.2f} | {r['std']:.2f} | {r['min']:.2f} | "
            f"{r['max']:.2f} | {r['mean_abs']:.2f} | {r['rkm_mae']:.2f} | {r['rkm_rmse']:.2f} | "
            f"{r['hybrid_mae']:.2f} | {r['hybrid_rmse']:.2f} |"
        )
    return L


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    df = load()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    overall = residual_stats(df)
    print("OVERALL", {k: round(v, 2) for k, v in overall.items()})

    sec = table(df, "section", [SECTION_LABELS[s] for s in SECTION_ORDER])
    print("\nBY SECTION\n", sec.round(2))
    temp = table(df, "temperature_K")
    print("\nBY TEMPERATURE\n", temp.round(2))
    xin = table(df, "xin_bin", list(df["xin_bin"].cat.categories))
    print("\nBY xIn BIN\n", xin.round(2))
    region = table(df, "region", REGION_LABELS)
    print("\nBY REGION\n", region.round(2))
    df["xbi_bin"] = pd.cut(df["xBi"], [0.0, 0.10, 0.20, 0.30, 0.40, 0.61], include_lowest=True)
    xbi = table(df, "xbi_bin", list(df["xbi_bin"].cat.categories))
    print("\nBY xBi BIN\n", xbi.round(2))

    st = df.groupby(["section", "temperature_K"])
    sec_temp = st.apply(lambda g: pd.Series(residual_stats(g)), include_groups=False)
    print("\nBY SECTION x TEMPERATURE\n", sec_temp[["n", "mean", "std", "mean_pred", "rkm_mae", "hybrid_mae"]].round(2))

    print("\nFOLDS")
    folds = {}
    for s in SECTION_ORDER:
        lab = SECTION_LABELS[s]
        g = df[df["section"] == lab]
        bv = bias_variance(g)
        tf = fold_train_fit(df, lab)
        cov = coverage(df, lab)
        folds[lab] = {**bv, **tf, **cov, "test_mean_resid": g["residual"].mean(), "test_mean_pred": g["predicted_residual"].mean()}
        print(lab, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in folds[lab].items()})

    print("\nMATCHED PROFILE (mean residual per xIn bin, temperature)\n", matched_profile_gap(df).round(1))

    sn = df[df["section"] == "Sn-rich"].copy()
    sn = sn.sort_values(["temperature_K", "xIn"])
    cols = ["experiment_id", "temperature_K", "xBi", "xIn", "xSn", "experimental_delta_mixH", "rkm_prediction",
            "residual", "predicted_residual", "hybrid_prediction", "hybrid_error"]
    print("\nSN-RICH\n", sn[cols].round(2).to_string(index=False))

    corr = df.groupby("section")[["residual", "xIn", "temperature_K"]].corr().round(3)
    print("\nCORR\n", corr)

    names = make_figures(df)
    print("\nFigures:", names)

    write_report(df, overall, sec, temp, xin, region, xbi, sec_temp, folds, sn, names)
    print(f"Wrote {REPORT_PATH}")
    return 0


def write_report(df, overall, sec, temp, xin, region, xbi, sec_temp, folds, sn, names) -> None:
    S = sec.set_index("group")
    F = folds
    snf, eqf, bif = F["Sn-rich"], F["Equiatomic"], F["Bi-rich"]
    corr = {s: float(np.corrcoef(g["residual"], g["xIn"])[0, 1]) for s, g in df.groupby("section")}
    prof = matched_profile_gap(df)
    hi = df[df["xIn"] > 0.80].groupby(["temperature_K", "section"])["residual"].mean().unstack("section")
    st = sec_temp

    L: list[str] = []
    L += [
        "# Residual Learning Diagnostic Analysis",
        "",
        "Script: `scripts/step_5_residual_diagnostics.py`. Inputs: `data/original_experimental_data.csv` (104 rows), "
        "`reports/residual_learning_results.csv` (existing LOCSO out-of-fold predictions from "
        "`scripts/step_4_residual_learning.py`), RKM from `src/rkm_model.py`. No model, dataset or existing report "
        "was modified, and no synthetic data were loaded.",
        "",
        "Labels used below: **Observed** = directly computed from the data; **Interpretation** = what the "
        "observation most plausibly means; **Hypothesis** = a possible explanation that this analysis does not test.",
        "",
        "## Objective",
        "",
        "RKM + residual Poly D2 (residual = ΔmixH_exp − ΔmixH_RKM, learned as PolyD2(xBi, xIn, T)) improves "
        "RKM under LOCSO overall (MAE 57.13 → 44.66 J/mol) and on the Bi-rich and Equiatomic held-out sections, "
        "but is worse than RKM on the Sn-rich section (MAE 42.88 → 54.70 J/mol). This analysis investigates why, "
        "without changing any model.",
        "",
        "## Overall residual behavior",
        "",
        f"Over all 104 observations the RKM residual has mean {overall['mean']:.2f} J/mol, median "
        f"{overall['median']:.2f}, standard deviation {overall['std']:.2f}, range {overall['min']:.2f} to "
        f"{overall['max']:.2f} J/mol. RKM MAE = {overall['rkm_mae']:.2f}, RMSE = {overall['rkm_rmse']:.2f} "
        "(identical to the frozen RKM benchmark).",
        "",
        "![RKM residual vs xIn](../figures/residual_vs_xIn.png)",
        "",
        "Composition-region summary (In content):",
        "",
        *fmt_stats_table(region, "Region"),
        "",
        "Residual by xBi bin (note: xBi is strongly confounded with cross-section and xIn on these three paths, "
        "so this table is descriptive only):",
        "",
        *fmt_stats_table(xbi.assign(group=xbi["group"].astype(str)), "xBi bin"),
        "",
        "## Cross-section comparison",
        "",
        *fmt_stats_table(sec, "Cross-section"),
        "",
        "Hybrid MAE/RMSE are the held-out LOCSO values (each section predicted by a residual model trained on "
        "the other two).",
        "",
        "Correlation of the residual with xIn and temperature, within each section:",
        "",
        "| Cross-section | corr(residual, xIn) | corr(residual, T) |",
        "| --- | ---: | ---: |",
    ]
    for s in [SECTION_LABELS[x] for x in SECTION_ORDER]:
        g = df[df["section"] == s]
        L.append(f"| {s} | {corr[s]:+.3f} | {np.corrcoef(g['residual'], g['temperature_K'])[0, 1]:+.3f} |")

    L += [
        "",
        f"**Observed.** The three sections have clearly different residual distributions. Bi-rich residuals are "
        f"strongly negative (mean {S.loc['Bi-rich','mean']:.1f}, std {S.loc['Bi-rich','std']:.1f}); Equiatomic "
        f"residuals are mildly negative (mean {S.loc['Equiatomic','mean']:.1f}, std {S.loc['Equiatomic','std']:.1f}); "
        f"Sn-rich residuals are positive on average (mean {S.loc['Sn-rich','mean']:.1f}, std "
        f"{S.loc['Sn-rich','std']:.1f}). The sign of the residual–xIn relationship flips: it is negative in the "
        f"Bi-rich ({corr['Bi-rich']:+.2f}) and Equiatomic ({corr['Equiatomic']:+.2f}) sections and positive in the "
        f"Sn-rich section ({corr['Sn-rich']:+.2f}).",
        "",
        "Mean residual for In-rich compositions (xIn > 0.80), by temperature:",
        "",
        "| T (K) | Bi-rich | Equiatomic | Sn-rich |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for t, r in hi.iterrows():
        L.append(f"| {int(t)} | {r['Bi-rich']:.1f} | {r['Equiatomic']:.1f} | {r['Sn-rich']:.1f} |")
    L += [
        "",
        "**Observed.** At In-rich compositions the residual is ordered Bi-rich < Equiatomic < Sn-rich at every "
        "temperature, with roughly equal steps between neighbouring sections (≈ 75–120 J/mol). At In-lean "
        "compositions (xIn < 0.20) there is no such consistent ordering (see the matched-bin table under "
        "*Composition-space coverage*).",
        "",
        "**Interpretation.** RKM's systematic error depends on the Bi/(Bi+Sn) ratio, mainly at In-rich "
        "compositions. A residual pattern learned on one or two ratios is therefore not directly transferable to "
        "another ratio; it changes monotonically with ratio rather than being shared.",
        "",
        "![RKM vs hybrid error by section](../figures/rkm_vs_hybrid_error_by_section.png)",
        "",
        "## xIn dependence",
        "",
        "| xIn bin | n | Mean residual | MAE of residual (= RKM MAE) | RKM MAE | Hybrid MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in xin.iterrows():
        L.append(
            f"| {r['group']} | {r['n']} | {r['mean']:.2f} | {r['mean_abs']:.2f} | {r['rkm_mae']:.2f} | {r['hybrid_mae']:.2f} |"
        )
    L += [
        "",
        "The bins hold unequal numbers of points because the experimental titrations are dense near the "
        "In-rich end (36 of 104 points have xIn > 0.80).",
        "",
        "**Observed.** RKM error is largest at In-rich compositions (xIn > 0.6, RKM MAE ≈ 56–75 J/mol). The "
        "hybrid helps most for 0.6 < xIn ≤ 0.8 (RKM ≈ 56–75 → hybrid ≈ 27–28 J/mol) and also improves "
        "xIn > 0.8 (74.4 → 52.7 J/mol). In every bin with xIn ≤ 0.6, where RKM is already comparatively "
        "accurate (RKM MAE ≈ 29–45 J/mol), the pooled hybrid MAE is higher than RKM.",
        "",
        "![Residual prediction error vs xIn](../figures/residual_prediction_error_vs_xIn.png)",
        "",
        "## Sn-rich fold diagnosis",
        "",
        "Per-fold comparison of actual vs predicted residual on the held-out section. Error = actual − predicted "
        "residual (equal to the hybrid error on ΔmixH). MSE share from bias = bias² / MSE; the remainder is the "
        "spread of the error around its mean.",
        "",
        "| Held-out fold | Test mean actual resid. | Test mean predicted resid. | Bias of error | Bias² share of MSE | Spread share of MSE | corr(actual, predicted) | Train-fit residual MAE | Held-out residual MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in [SECTION_LABELS[x] for x in SECTION_ORDER]:
        f = F[s]
        L.append(
            f"| {s} | {f['test_mean_resid']:.2f} | {f['test_mean_pred']:.2f} | {f['bias']:+.2f} | "
            f"{100*f['bias_sq_share']:.1f}% | {100*f['var_share']:.1f}% | {f['corr']:.3f} | {f['train_mae']:.2f} | "
            f"{S.loc[s,'hybrid_mae']:.2f} |"
        )

    L += [
        "",
        "![Sn-rich actual vs predicted residual](../figures/residual_sn_rich_actual_vs_predicted.png)",
        "",
        "Every Sn-rich test observation (held out in fold 3):",
        "",
        "| ID | T (K) | xBi | xIn | xSn | ΔmixH_exp | RKM | Actual resid. | Predicted resid. | Hybrid | Hybrid error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in sn.iterrows():
        L.append(
            f"| {r['experiment_id']} | {int(r['temperature_K'])} | {r['xBi']:.4f} | {r['xIn']:.4f} | {r['xSn']:.4f} | "
            f"{r['experimental_delta_mixH']:.2f} | {r['rkm_prediction']:.2f} | {r['residual']:.2f} | "
            f"{r['predicted_residual']:.2f} | {r['hybrid_prediction']:.2f} | {r['hybrid_error']:+.2f} |"
        )
    top = sn.reindex(sn["hybrid_error"].abs().sort_values(ascending=False).index).head(8)
    L += ["", "Largest Sn-rich hybrid errors (by |error|):", "",
          "| ID | T (K) | xIn | Actual resid. | Predicted resid. | Hybrid error | RKM abs. error |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for _, r in top.iterrows():
        L.append(
            f"| {r['experiment_id']} | {int(r['temperature_K'])} | {r['xIn']:.4f} | {r['residual']:.2f} | "
            f"{r['predicted_residual']:.2f} | {r['hybrid_error']:+.2f} | {abs(r['residual']):.2f} |"
        )

    L += [
        "",
        "**Observed.**",
        "",
        f"- The Sn-rich fold has almost no systematic bias: mean actual residual {snf['test_mean_resid']:.2f} vs mean "
        f"predicted {snf['test_mean_pred']:.2f} J/mol (bias {snf['bias']:+.2f}; {100*snf['bias_sq_share']:.1f}% of MSE).",
        f"- Nearly all of the error ({100*snf['var_share']:.1f}% of MSE) is in the *shape*: correlation between actual and "
        f"predicted residual is only {snf['corr']:.2f}, compared with {bif['corr']:.2f} (Bi-rich) and {eqf['corr']:.2f} (Equiatomic).",
        "- The predicted residual is a hump over xIn: positive and up to ≈ +125 J/mol around xIn ≈ 0.4–0.6, then "
        "falling steeply toward the In corner and turning negative at the highest xIn (from xIn ≈ 0.79 at 767 K, "
        "≈ 0.87 at 813 K, 0.91 at 855 K). The actual Sn-rich residual instead rises with xIn and stays positive "
        "(≈ +6 to +100 J/mol) at the In-rich end.",
        "- The errors therefore have opposite signs in two regions. For 0.3 ≤ xIn ≤ 0.6 the hybrid is less "
        "exothermic than measured by ≈ 40–96 J/mol; for xIn ≥ 0.79 it is more exothermic than measured by "
        "≈ 35–100 J/mol. The largest errors (|error| > 84 J/mol) are at xIn ≈ 0.89–0.91 and xIn ≈ 0.50.",
        f"- The residual data in the Sn-rich section are not unusually noisy: std {S.loc['Sn-rich','std']:.1f} J/mol "
        f"vs {S.loc['Equiatomic','std']:.1f} (Equiatomic) and {S.loc['Bi-rich','std']:.1f} (Bi-rich), and each "
        "temperature series is smooth in xIn (figure above).",
        f"- The residual model fits its training data well (training residual MAE {snf['train_mae']:.2f} J/mol), so "
        "the failure is not under-fitting of the training sections.",
        f"- RKM is already relatively accurate on the Sn-rich section (RKM MAE {S.loc['Sn-rich','rkm_mae']:.2f}; at 767 K "
        f"only {st.loc[('Sn-rich', 767), 'rkm_mae']:.2f}), so there is little error for a correction to remove and an "
        "incorrectly shaped correction adds error.",
        "",
        "**Interpretation (assessment of candidate causes).**",
        "",
        "| Candidate cause | Supported? | Evidence |",
        "| --- | --- | --- |",
        f"| Systematic bias | No | Bias {snf['bias']:+.2f} J/mol, {100*snf['bias_sq_share']:.1f}% of MSE. |",
        "| High variance / noisy data | No | Sn-rich residual std is comparable to Equiatomic and smaller than Bi-rich; series are smooth. |",
        "| Temperature dependence | No (not the main cause) | Mean predicted residual per T matches mean actual per T closely (see next section); hybrid MAE is similar at all three T. |",
        f"| Composition dependence | Yes | Residual–xIn correlation flips sign ({corr['Sn-rich']:+.2f} vs {corr['Bi-rich']:+.2f} / {corr['Equiatomic']:+.2f}); predicted shape is wrong (corr {snf['corr']:.2f}). |",
        "| Extrapolation | Yes | Sn-rich lies outside the composition range spanned by the training paths (0/36 test points inside the training convex hull; see coverage section). |",
        "| Insufficient training coverage | Yes, as the mechanism behind extrapolation | Only two training Bi/(Bi+Sn) ratios (0.50, 0.67), both on the same side of the Sn-rich ratio (0.33). |",
        "",
        "The data support the conclusion that the Sn-rich failure is a **composition-dependent extrapolation "
        "error**: the residual shape learned from the Bi-rich and Equiatomic paths does not carry over to the "
        "Sn-rich path. They do not support bias, noise or temperature as the primary cause.",
        "",
        "## Temperature dependence",
        "",
        *fmt_stats_table(temp.assign(group=temp["group"].astype(int).astype(str) + " K"), "Temperature"),
        "",
        "Mean actual vs mean predicted residual per section and temperature (held-out predictions):",
        "",
        "| Section | T (K) | n | Mean actual resid. | Mean predicted resid. | Std actual | RKM MAE | Hybrid MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (s, t), r in st.iterrows():
        L.append(
            f"| {s} | {int(t)} | {int(r['n'])} | {r['mean']:.2f} | {r['mean_pred']:.2f} | {r['std']:.2f} | "
            f"{r['rkm_mae']:.2f} | {r['hybrid_mae']:.2f} |"
        )
    L += [
        "",
        "**Observed.** RKM has no temperature term, yet the residual rises with temperature in every section "
        "(767 → 855 K: Bi-rich −134.9 → −17.0, Equiatomic −58.7 → +20.8, Sn-rich −0.3 → +51.9 J/mol). Pooled "
        "residual mean goes from −62.6 (767 K) to +18.5 J/mol (855 K). The temperature trend has the same sign in "
        "all three sections, though its size varies (≈ 118, 80 and 52 J/mol over 88 K).",
        "",
        "**Observed (Sn-rich).** The held-out model reproduces the Sn-rich *per-temperature mean* residual "
        "closely (−5.2 vs −0.3, 22.5 vs 27.2, 51.1 vs 51.9 J/mol), and the Sn-rich hybrid MAE is similar at all "
        "three temperatures (60.8, 52.6, 50.7 J/mol). The hybrid is worse than RKM at 767 K and 813 K and better "
        "at 855 K, mainly because RKM itself is most accurate at 767 K for this section.",
        "",
        "**Interpretation.** Temperature is an important part of the overall residual (it is the main thing RKM "
        "cannot represent) and its *direction* transfers across sections. It is not the primary cause of the "
        "Sn-rich failure.",
        "",
        "## Composition-space coverage",
        "",
        "All three paths run from the In-lean side to the In corner along a fixed Bi/(Bi+Sn) ratio "
        "(0.67, 0.50, 0.33). Holding one out leaves the model with two lines in the (xBi, xIn, xSn) triangle.",
        "",
        "![Composition coverage](../figures/residual_composition_coverage.png)",
        "",
        "| Held-out fold | Test Bi/(Bi+Sn) | Train Bi/(Bi+Sn) range | Test points inside training convex hull (xBi, xIn) | Median nearest-train distance | Max nearest-train distance | Test xBi range | Train xBi range | Test xSn range | Train xSn range |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for s in [SECTION_LABELS[x] for x in SECTION_ORDER]:
        f = F[s]
        L.append(
            f"| {s} | {f['test_ratio']:.3f} | {f['train_ratio_range'][0]:.3f}–{f['train_ratio_range'][1]:.3f} | "
            f"{f['n_inside']}/{f['n_test']} | {f['nn_median']:.3f} | {f['nn_max']:.3f} | "
            f"{f['test_xbi'][0]:.3f}–{f['test_xbi'][1]:.3f} | {f['train_xbi'][0]:.3f}–{f['train_xbi'][1]:.3f} | "
            f"{f['test_xsn'][0]:.3f}–{f['test_xsn'][1]:.3f} | {f['train_xsn'][0]:.3f}–{f['train_xsn'][1]:.3f} |"
        )
    L += [
        "",
        "Distances are Euclidean in (xBi, xIn, xSn). The convex-hull test uses (xBi, xIn), which fully determines "
        "composition since xSn = 1 − xBi − xIn.",
        "",
        "**Observed.**",
        "",
        "- The xIn range is the same in every fold (≈ 0.095–0.907), so comparing xIn alone would suggest no "
        "extrapolation.",
        "- In full composition space, the Equiatomic fold is an interpolation (31/34 test points inside the "
        "training hull; the 3 outside are at the extreme ends of the path). The Bi-rich and Sn-rich folds are "
        "extrapolations (0/34 and 0/36 inside): their paths lie entirely outside the region between the two "
        "training paths. The Sn-rich test path also has higher xSn (up to 0.602) than any training point (max 0.452).",
        "- Nearest-neighbour distances are similar across folds (median ≈ 0.08–0.09), so the extrapolation is not "
        "about being far from the data; it is about lying on the far side of both training paths in the "
        "Bi/(Bi+Sn) direction.",
        "",
        "Mean residual per xIn bin, section and temperature (blank = no observation in that bin):",
        "",
        "| T (K) | xIn bin | Bi-rich | Equiatomic | Sn-rich |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for (t, b), r in prof.iterrows():
        cells = " | ".join("" if pd.isna(r[c]) else f"{r[c]:.1f}" for c in prof.columns)
        L.append(f"| {int(t)} | {b} | {cells} |")
    L += [
        "",
        "**Interpretation.** LOCSO creates a genuine cross-section extrapolation problem for the two outer "
        "sections (Bi-rich and Sn-rich), and an interpolation problem for the Equiatomic section. This matches the "
        "performance ranking: Equiatomic (interpolated) has the best hybrid result (MAE 23.95). Bi-rich is also "
        f"extrapolated and shows a large bias (mean error {bif['bias']:+.1f} J/mol: the model under-predicts how "
        "negative the Bi-rich residual is), but it still improves on RKM because RKM's Bi-rich error is large and "
        f"the predicted shape is roughly right (corr {bif['corr']:.2f}). For Sn-rich the RKM error is smaller and "
        "the predicted shape is wrong, so the hybrid loses.",
        "",
        "**Hypothesis (not tested here).** At In-rich compositions the three paths are very close together in "
        "xBi (e.g. at xIn ≈ 0.85, xBi ≈ 0.10, 0.075 and 0.05), but their residuals differ by ≈ 75–120 J/mol "
        "between neighbouring paths. A single global degree-2 polynomial in (xBi, xIn, T), trained on two paths, "
        "may extend the In-rich trend of the training paths (residual becoming more negative toward the In corner) "
        "rather than the ratio-dependent trend seen across all three paths. The observed predicted residual "
        "(turning negative for xIn > 0.85 in the Sn-rich fold) is consistent with this, but this analysis does not "
        "isolate the cause.",
        "",
        "## Main findings",
        "",
        "**Observed results**",
        "",
        f"1. RKM residuals differ systematically between sections: mean {S.loc['Bi-rich','mean']:.1f} (Bi-rich), "
        f"{S.loc['Equiatomic','mean']:.1f} (Equiatomic), {S.loc['Sn-rich','mean']:+.1f} (Sn-rich) J/mol. The "
        "residual–xIn trend reverses sign in the Sn-rich section.",
        "2. Residual rises with temperature in all three sections (RKM is temperature-independent); this "
        "direction is shared across sections.",
        f"3. Sn-rich held-out error is almost entirely shape error, not bias (bias {snf['bias']:+.2f} J/mol, "
        f"{100*snf['bias_sq_share']:.1f}% of MSE; corr(actual, predicted) = {snf['corr']:.2f}).",
        "4. The largest Sn-rich errors occur at the In-rich end (xIn ≥ 0.84, predicted residual far below the "
        "positive actual residual and often negative) and around xIn ≈ 0.5 (predicted residual ≈ +77 to +124 vs "
        "actual ≈ −10 to +28 J/mol).",
        "5. In full composition space, the Sn-rich and Bi-rich held-out paths lie outside the training paths' "
        "convex hull (0% inside); the Equiatomic path lies inside (91%). xIn ranges are identical in all folds.",
        "",
        "**Interpretation**",
        "",
        "- The Sn-rich failure is best described as composition-dependent extrapolation: the Sn-rich path has a "
        "Bi/(Bi+Sn) ratio outside the two training ratios, and the composition dependence of the residual "
        "changes with that ratio. Systematic bias, data noise and temperature are not supported as primary causes.",
        "- Temperature dependence of the residual appears transferable across sections (same direction, and the "
        "Sn-rich per-temperature mean is predicted well). The composition dependence of the residual is not "
        "transferable from two sections to a third lying outside them.",
        "- The residual behaviour is consistent with a roughly monotone dependence on Bi/(Bi+Sn) at In-rich "
        "compositions, which an interpolated section can benefit from but an extrapolated section cannot.",
        "",
        "**Hypotheses (require a separate test)**",
        "",
        "- The global quadratic form in (xBi, xIn) may not represent the steep across-path residual change near "
        "the In corner when trained on only two paths.",
        "- With three composition paths in total, any LOCSO assessment of an outer path is a one-sided "
        "extrapolation; performance on outer paths may be inherently less reliable than on the middle path "
        "regardless of the residual model.",
        "",
        "No new model is recommended in this step.",
        "",
        "## Figures",
        "",
        *[f"- `figures/{n}`" for n in names],
        "",
    ]
    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
