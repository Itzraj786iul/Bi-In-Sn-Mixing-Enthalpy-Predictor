"""
Cross-section structure of the experimental RKM residual (diagnostic only).

    python scripts/step_7_cross_section_residual_structure.py

Asks whether residual = dmixH_exp - dmixH_RKM contains a systematic
cross-section (Bi/Sn composition-path) component, how strong it is near the
In-rich corner, whether it persists at fixed temperature, and whether two
paths can constrain it for the third. Descriptive statistics and one
transparent linear variance decomposition only; nothing is predicted and no
model is trained. Existing Step 1 / Step 3 out-of-fold residual predictions are
read from their CSV files, not recomputed. Only the 104 experimental
observations are loaded.
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_composition_representation import EXP_TARGET  # noqa: E402
from step_4_residual_learning import SECTION_LABELS, load_experimental  # noqa: E402

STEP1_PATH = ROOT / "reports" / "residual_learning_results.csv"
STEP3_PATH = ROOT / "reports" / "path_aware_residual_results.csv"
REPORT_PATH = ROOT / "reports" / "cross_section_residual_structure.md"
FIG_DIR = ROOT / "figures"
FIGS = {
    "by_section": "residual_by_section_vs_xIn.png",
    "high_in": "residual_high_in_section_comparison.png",
    "sec_temp": "residual_section_temperature_comparison.png",
    "level_shape": "residual_sn_rich_level_vs_shape.png",
}

SECTIONS = ["Bi-rich", "Equiatomic", "Sn-rich"]
COLORS = {"Bi-rich": "tab:blue", "Equiatomic": "tab:green", "Sn-rich": "tab:red"}
TEMPS = [767, 813, 855]
MATCH_TOL = 0.01
REGIONS = [
    ("xIn < 0.50", 0.0, 0.50),
    ("0.50 ≤ xIn < 0.70", 0.50, 0.70),
    ("0.70 ≤ xIn < 0.80", 0.70, 0.80),
    ("xIn ≥ 0.80", 0.80, 1.01),
]
HIGH_IN = [("xIn ≥ 0.70", 0.70), ("xIn ≥ 0.80", 0.80)]


def load() -> pd.DataFrame:
    exp = load_experimental()
    if len(exp) != 104:
        raise RuntimeError("Expected 104 experimental observations")
    df = exp.rename(columns={"id": "experiment_id", EXP_TARGET: "experimental_delta_mixH"})
    df["section"] = df["cross_section"].map(SECTION_LABELS)
    df["bi_sn_fraction"] = df["xBi"] / (df["xBi"] + df["xSn"])
    df["region"] = pd.cut(df["xIn"], [r[1] for r in REGIONS] + [REGIONS[-1][2]], labels=[r[0] for r in REGIONS], right=False)

    s1 = pd.read_csv(STEP1_PATH)[["experiment_id", "residual", "predicted_residual"]]
    s3 = pd.read_csv(STEP3_PATH)[["experiment_id", "residual", "predicted_residual_A", "predicted_residual_B"]]
    df = df.merge(s1.rename(columns={"residual": "residual_s1", "predicted_residual": "pred_A_s1"}), on="experiment_id")
    df = df.merge(s3.rename(columns={"residual": "residual_s3", "predicted_residual_A": "pred_A", "predicted_residual_B": "pred_B"}), on="experiment_id")
    if len(df) != 104:
        raise RuntimeError("Step 1 / Step 3 outputs do not cover the 104 observations")
    for col in ("residual_s1", "residual_s3"):
        if not np.allclose(df[col], df["residual"], atol=1e-9):
            raise RuntimeError(f"RKM residual differs from {col}: RKM or data changed")
    if not np.allclose(df["pred_A"], df["pred_A_s1"], atol=1e-9):
        raise RuntimeError("Model A predictions differ between Step 1 and Step 3 outputs")
    return df.drop(columns=["residual_s1", "residual_s3", "pred_A_s1"])


def stats_row(r: pd.Series) -> dict:
    return {
        "n": len(r),
        "mean": r.mean(),
        "median": r.median(),
        "std": r.std(ddof=1) if len(r) > 1 else float("nan"),
        "MAE": r.abs().mean(),
        "RMSE": float(np.sqrt(np.mean(r**2))),
    }


# ------------------------------------------------------------ xIn overlap / matching


def xin_overlap(df: pd.DataFrame, tol: float = MATCH_TOL) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-to-one matching at the same temperature (no point is reused).
    Pairs: |ΔxIn| ≤ tol. Triplets: all three pairwise |ΔxIn| ≤ tol.
    Greedy by smallest xIn spread.
    """
    pairs, triplets = [], []
    for t in TEMPS:
        sub = df[df["temperature_K"] == t]
        by = {s: sub[sub["section"] == s] for s in SECTIONS}
        for a, b in [("Bi-rich", "Equiatomic"), ("Equiatomic", "Sn-rich"), ("Bi-rich", "Sn-rich")]:
            cands = sorted(
                (abs(ra.xIn - rb.xIn), ia, ib)
                for ia, ra in by[a].iterrows() for ib, rb in by[b].iterrows() if abs(ra.xIn - rb.xIn) <= tol
            )
            used_a, used_b = set(), set()
            for d, ia, ib in cands:
                if ia in used_a or ib in used_b:
                    continue
                used_a.add(ia)
                used_b.add(ib)
                ra, rb = df.loc[ia], df.loc[ib]
                pairs.append({"T": t, "pair": f"{a} − {b}", "xIn_a": ra["xIn"], "xIn_b": rb["xIn"], "dxIn": d,
                              "diff": ra["residual"] - rb["residual"]})
        cands = []
        for ib, rb in by["Bi-rich"].iterrows():
            for ie, re in by["Equiatomic"].iterrows():
                for i_s, rs in by["Sn-rich"].iterrows():
                    x = (rb.xIn, re.xIn, rs.xIn)
                    if max(x) - min(x) <= tol:
                        cands.append((max(x) - min(x), ib, ie, i_s))
        used = set()
        for spread, *idx in sorted(cands):
            if used & set(idx):
                continue
            used |= set(idx)
            rows = dict(zip(SECTIONS, (df.loc[i] for i in idx)))
            triplets.append({
                "T": t,
                "xIn_mean": float(np.mean([r["xIn"] for r in rows.values()])),
                "spread": spread,
                **{f"xIn_{s}": rows[s]["xIn"] for s in SECTIONS},
                **{f"res_{s}": rows[s]["residual"] for s in SECTIONS},
                **{f"A_{s}": rows[s]["pred_A"] for s in SECTIONS},
                **{f"B_{s}": rows[s]["pred_B"] for s in SECTIONS},
            })
    tri = pd.DataFrame(triplets).sort_values(["T", "xIn_mean"]).reset_index(drop=True)
    if len(tri):
        tri["gap_Sn_minus_Bi"] = tri["res_Sn-rich"] - tri["res_Bi-rich"]
        tri["ordered"] = (tri["res_Bi-rich"] < tri["res_Equiatomic"]) & (tri["res_Equiatomic"] < tri["res_Sn-rich"])
        tri["mid_dev"] = tri["res_Equiatomic"] - 0.5 * (tri["res_Bi-rich"] + tri["res_Sn-rich"])
    return pd.DataFrame(pairs), tri


# ------------------------------------------------------------ variance decomposition


def anova_cells(df: pd.DataFrame, mask: pd.Series) -> dict:
    """One-way ANOVA on section *within* (temperature, xIn-region) cells, pooled over cells."""
    sub = df[mask]
    ss_between = ss_within = 0.0
    df_between = df_within = 0
    for (_, _), cell in sub.groupby(["temperature_K", "region"], observed=True):
        groups = [g["residual"].to_numpy() for _, g in cell.groupby("section") if len(g)]
        if len(groups) < 2:
            continue
        m = cell["residual"].mean()
        ss_between += sum(len(g) * (g.mean() - m) ** 2 for g in groups)
        ss_within += sum(((g - g.mean()) ** 2).sum() for g in groups)
        df_between += len(groups) - 1
        df_within += sum(len(g) - 1 for g in groups)
    F = (ss_between / df_between) / (ss_within / df_within) if df_within > 0 and ss_within > 0 else float("nan")
    p = float(stats.f.sf(F, df_between, df_within)) if np.isfinite(F) else float("nan")
    return {
        "n": len(sub),
        "ss_between": ss_between,
        "ss_within": ss_within,
        "eta2": ss_between / (ss_between + ss_within),
        "F": F,
        "df": (df_between, df_within),
        "p": p,
        "sd_within": float(np.sqrt(ss_within / df_within)) if df_within else float("nan"),
    }


def ols(X: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    rss = float(np.sum((y - X @ coef) ** 2))
    return rss, int(np.linalg.matrix_rank(X))


def linear_decomposition(df: pd.DataFrame, mask: pd.Series) -> dict:
    """
    common(xIn, T): temperature indicators + per-temperature linear and quadratic xIn terms.
    + section: adds two section indicators (constant section offset).
    + section × xIn: lets the section offset change linearly with xIn.
    Nested F-tests; descriptive only.
    """
    sub = df[mask]
    y = sub["residual"].to_numpy(float)
    xin = sub["xIn"].to_numpy(float) - 0.5
    T = [(sub["temperature_K"] == t).to_numpy(float) for t in TEMPS]
    common = np.column_stack([*T, *[t * xin for t in T], *[t * xin**2 for t in T]])
    S = [(sub["section"] == s).to_numpy(float) for s in ("Bi-rich", "Sn-rich")]
    with_sec = np.column_stack([common, *S])
    with_int = np.column_stack([with_sec, *[s * xin for s in S]])
    tss = float(np.sum((y - y.mean()) ** 2))
    out = {"n": len(y), "tss": tss}
    prev = None
    for name, X in (("common", common), ("common + section", with_sec), ("common + section + section×xIn", with_int)):
        rss, k = ols(X, y)
        row = {"rss": rss, "k": k, "R2": 1 - rss / tss, "sd_resid": float(np.sqrt(rss / (len(y) - k)))}
        if prev is not None:
            dk = k - prev["k"]
            F = ((prev["rss"] - rss) / dk) / (rss / (len(y) - k))
            row.update({"F": F, "p": float(stats.f.sf(F, dk, len(y) - k)), "dR2": row["R2"] - prev["R2"], "dk": dk})
        out[name] = row
        prev = row
    return out


# ------------------------------------------------------------ figures


def fig_by_section(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for ax, t in zip(axes, TEMPS):
        for s in SECTIONS:
            g = df[(df["section"] == s) & (df["temperature_K"] == t)].sort_values("xIn")
            ax.plot(g["xIn"], g["residual"], "o-", c=COLORS[s], ms=4, label=s)
        ax.axhline(0, c="black", lw=0.7)
        ax.axvspan(0.80, 0.92, color="0.9", zorder=0)
        ax.set_title(f"{t} K")
        ax.set_xlabel("xIn")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Residual = ΔmixH_exp − ΔmixH_RKM (J/mol)")
    axes[0].legend(title="Composition path")
    fig.suptitle("Experimental RKM residual by composition path (shaded: xIn ≥ 0.80)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / FIGS["by_section"], dpi=150)
    plt.close(fig)


def fig_high_in(df: pd.DataFrame) -> None:
    hi = df[df["xIn"] >= 0.80]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    for s in SECTIONS:
        for t, mk in zip(TEMPS, "os^"):
            g = hi[(hi["section"] == s) & (hi["temperature_K"] == t)].sort_values("xIn")
            ax.plot(g["xIn"], g["residual"], marker=mk, c=COLORS[s], ls="-", lw=0.8, ms=5,
                    label=f"{s}, {t} K")
    ax.axhline(0, c="black", lw=0.7)
    ax.set_xlabel("xIn")
    ax.set_ylabel("Residual (J/mol)")
    ax.set_title("xIn ≥ 0.80: residual vs xIn")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))

    ax = axes[1]
    width = 0.25
    for i, s in enumerate(SECTIONS):
        means = [hi[(hi["section"] == s) & (hi["temperature_K"] == t)]["residual"].mean() for t in TEMPS]
        sds = [hi[(hi["section"] == s) & (hi["temperature_K"] == t)]["residual"].std(ddof=1) for t in TEMPS]
        ax.bar(np.arange(3) + (i - 1) * width, means, width, yerr=sds, capsize=3, color=COLORS[s], alpha=0.8, label=s)
    ax.set_xticks(range(3), [f"{t} K" for t in TEMPS])
    ax.axhline(0, c="black", lw=0.7)
    ax.set_ylabel("Mean residual ± 1 SD (J/mol)")
    ax.set_title("xIn ≥ 0.80: mean residual by path and temperature")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / FIGS["high_in"], dpi=150)
    plt.close(fig)


def fig_sec_temp(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    width = 0.25
    for ax, (label, lo, hi) in zip(axes, REGIONS):
        for i, s in enumerate(SECTIONS):
            means = []
            for t in TEMPS:
                g = df[(df["section"] == s) & (df["temperature_K"] == t) & (df["xIn"] >= lo) & (df["xIn"] < hi)]
                means.append(g["residual"].mean() if len(g) else np.nan)
            ax.bar(np.arange(3) + (i - 1) * width, means, width, color=COLORS[s], alpha=0.85, label=s)
        ax.set_xticks(range(3), [f"{t}" for t in TEMPS])
        ax.set_xlabel("Temperature (K)")
        ax.set_title(label)
        ax.axhline(0, c="black", lw=0.7)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("Mean residual (J/mol)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Mean RKM residual by composition path, temperature and xIn region")
    fig.tight_layout()
    fig.savefig(FIG_DIR / FIGS["sec_temp"], dpi=150)
    plt.close(fig)


def fig_level_shape(df: pd.DataFrame) -> None:
    sn = df[df["section"] == "Sn-rich"]
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for t, mk in zip(TEMPS, "os^"):
        g = sn[sn["temperature_K"] == t]
        ax.scatter(g["residual"], g["pred_A"], marker=mk, facecolors="none", edgecolors="tab:gray", label=f"Model A, {t} K")
        ax.scatter(g["residual"], g["pred_B"], marker=mk, c="black", s=22, label=f"Model B, {t} K")
    lim = [min(sn[["residual", "pred_A", "pred_B"]].min()) - 10, max(sn[["residual", "pred_A", "pred_B"]].max()) + 10]
    ax.plot(lim, lim, c="tab:red", lw=1, label="perfect prediction")
    off = (sn["residual"] - sn["pred_B"]).mean()
    ax.plot(lim, [v - off for v in lim], c="black", lw=0.8, ls="--", label=f"y = x − {off:.0f} (Model B mean offset)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Actual Sn-rich residual (J/mol)")
    ax.set_ylabel("Held-out predicted residual (J/mol)")
    ax.set_title("Sn-rich fold: shape vs level")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / FIGS["level_shape"], dpi=150)
    plt.close(fig)


# ------------------------------------------------------------ model comparison


def error_decomposition(actual: np.ndarray, pred: np.ndarray) -> dict:
    """MSE = bias² + (sd_a − sd_p)² + 2·sd_a·sd_p·(1 − r) (population SDs)."""
    e = actual - pred
    r = float(np.corrcoef(actual, pred)[0, 1])
    sa, sp = float(np.std(actual)), float(np.std(pred))
    bias = float(np.mean(e))
    mse = float(np.mean(e**2))
    return {
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(mse)),
        "bias": bias,
        "corr": r,
        "sd_actual": sa,
        "sd_pred": sp,
        "bias_sq": bias**2,
        "scale_sq": (sa - sp) ** 2,
        "shape": 2 * sa * sp * (1 - r),
        "mse": mse,
        "MAE_offset_removed": float(np.mean(np.abs(e - bias))),
    }


# ------------------------------------------------------------ main / report


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f"N = {len(df)}; only data/original_experimental_data.csv loaded")
    xin_table = df.pivot_table(index=["temperature_K", "section"], values="xIn", aggfunc=lambda v: ", ".join(f"{x:.3f}" for x in sorted(v)))
    print(xin_table.to_string())

    pairs, tri = xin_overlap(df)
    sensitivity = {tol: (len(xin_overlap(df, tol)[0]), len(xin_overlap(df, tol)[1])) for tol in (0.005, 0.01, 0.02)}
    print(f"\nMatched pairs (±{MATCH_TOL}): {len(pairs)}; triplets: {len(tri)}; sensitivity {sensitivity}")
    print(pairs.groupby("pair")["diff"].agg(["count", "mean", "std", "min", "max"]).round(1))
    print(tri.drop(columns=[c for c in tri.columns if c.startswith(("A_", "B_"))]).round(3).to_string())

    cells = []
    for t in TEMPS:
        for label, lo, hi_ in REGIONS:
            row = {"T": t, "region": label}
            for s in SECTIONS:
                g = df[(df["temperature_K"] == t) & (df["section"] == s) & (df["xIn"] >= lo) & (df["xIn"] < hi_)]
                row[s] = g["residual"].mean() if len(g) else np.nan
                row[f"n_{s}"] = len(g)
            row["ordered"] = bool(row["Bi-rich"] < row["Equiatomic"] < row["Sn-rich"])
            row["range"] = max(row[s] for s in SECTIONS) - min(row[s] for s in SECTIONS)
            cells.append(row)
    cells = pd.DataFrame(cells)
    print(cells.round(1).to_string())

    anova = {lab: anova_cells(df, df["xIn"] >= lo) for lab, lo in [("all", 0.0), *HIGH_IN]}
    lin = {lab: linear_decomposition(df, df["xIn"] >= lo) for lab, lo in [("all", 0.0), *HIGH_IN]}
    for k in anova:
        print(k, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in anova[k].items()})
        for kk, v in lin[k].items():
            print("   ", kk, v if not isinstance(v, dict) else {a: round(b, 4) if isinstance(b, float) else b for a, b in v.items()})

    hi = df[df["xIn"] >= 0.80]
    slopes = []
    for s, t in product(SECTIONS, TEMPS):
        g = hi[(hi["section"] == s) & (hi["temperature_K"] == t)]
        res = stats.linregress(g["xIn"], g["residual"])
        slopes.append({"section": s, "T": t, "n": len(g), "slope": res.slope, "slope_se": res.stderr, "mean": g["residual"].mean()})
    slopes = pd.DataFrame(slopes)
    print(slopes.round(2).to_string())

    sn = df[df["section"] == "Sn-rich"]
    decomp = {}
    for lab, m in [("Sn-rich, all", sn["xIn"] >= 0), ("Sn-rich, xIn ≥ 0.70", sn["xIn"] >= 0.70), ("Sn-rich, xIn ≥ 0.80", sn["xIn"] >= 0.80)]:
        g = sn[m]
        decomp[lab] = {k: error_decomposition(g["residual"].to_numpy(), g[f"pred_{k}"].to_numpy()) for k in ("A", "B")}
        for k in ("A", "B"):
            print(lab, k, {a: round(b, 2) for a, b in decomp[lab][k].items()})

    fig_by_section(df)
    fig_high_in(df)
    fig_sec_temp(df)
    fig_level_shape(df)
    write_report(df, xin_table, pairs, tri, sensitivity, cells, anova, lin, slopes, decomp)
    print(f"Wrote {REPORT_PATH}")
    return 0


def fmt_stats(title: str, groups: list[tuple[str, pd.Series]]) -> list[str]:
    L = [f"| {title} | n | Mean | Median | SD | MAE | RMSE |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, r in groups:
        s = stats_row(r)
        L.append(f"| {name} | {s['n']} | {s['mean']:.1f} | {s['median']:.1f} | {s['std']:.1f} | {s['MAE']:.1f} | {s['RMSE']:.1f} |")
    return L


def write_report(df, xin_table, pairs, tri, sensitivity, cells, anova, lin, slopes, decomp) -> None:
    hi7, hi8 = df[df["xIn"] >= 0.70], df[df["xIn"] >= 0.80]
    low = cells[cells["region"] == "xIn < 0.50"]
    n_ord = int(cells["ordered"].sum())
    hi_cells = cells[cells["region"].isin(["0.70 ≤ xIn < 0.80", "xIn ≥ 0.80"])]
    tri_hi = tri[tri["xIn_mean"] >= 0.70]
    tri_lo = tri[tri["xIn_mean"] < 0.70]
    pg = pairs.groupby("pair")["diff"].agg(["count", "mean", "std", "min", "max"])
    sn_all_B = decomp["Sn-rich, all"]["B"]
    sn8 = decomp["Sn-rich, xIn ≥ 0.80"]

    L: list[str] = [
        "# Cross-Section Residual Structure",
        "",
        "Script: `scripts/step_7_cross_section_residual_structure.py`. Data: `data/original_experimental_data.csv` only "
        "(104 observations); RKM from `src/rkm_model.py`. Model A / Model B held-out predicted residuals are read from "
        "`reports/residual_learning_results.csv` and `reports/path_aware_residual_results.csv` (nothing retrained). "
        "No synthetic data, Dataset A or Dataset B were loaded. Diagnostic only: no prediction model is proposed.",
        "",
        "## Objective",
        "",
        "Under LOCSO, residual learning on two composition paths fails on the Sn-rich path (MAE: RKM 42.88, Model A "
        "54.70, path-aware Model B 55.15 J/mol). Model B reproduces the Sn-rich residual *shape* (correlation ≈ 0.91) "
        "but misses its *level* (mean error ≈ +55 J/mol). This diagnostic asks whether the experimental RKM residual "
        "contains a systematic cross-section (Bi/Sn composition-path) component that two paths cannot pin down for "
        "the third.",
        "",
        "## Experimental residual definition",
        "",
        "```text",
        "residual = ΔmixH_exp − ΔmixH_RKM",
        "```",
        "",
        "RKM is temperature-independent, so any temperature dependence in the residual comes from the measurements. "
        "Residuals were recomputed here and agree with the Step 1 and Step 3 files to < 1e-9 J/mol.",
        "",
        "Actual xIn values (no interpolation is used anywhere in this analysis):",
        "",
        "| T (K) | Path | xIn values |",
        "| ---: | --- | --- |",
    ]
    for (t, s), r in xin_table.iterrows():
        L.append(f"| {t} | {s} | {r['xIn']} |")
    L += [
        "",
        "All three paths span nearly the same xIn range (≈ 0.095–0.907) with a similar titration grid, but the "
        "individual xIn points do not coincide exactly; the closest matches are quantified under *Matched-composition "
        "comparison*.",
        "",
        "## Cross-section comparison",
        "",
        *fmt_stats("Path (all xIn)", [(s, df.loc[df["section"] == s, "residual"]) for s in SECTIONS]),
        "",
        *fmt_stats("Temperature (all paths)", [(f"{t} K", df.loc[df["temperature_K"] == t, "residual"]) for t in TEMPS]),
        "",
        *fmt_stats("Path, xIn ≥ 0.70", [(s, hi7.loc[hi7["section"] == s, "residual"]) for s in SECTIONS]),
        "",
        *fmt_stats("Path, xIn ≥ 0.80", [(s, hi8.loc[hi8["section"] == s, "residual"]) for s in SECTIONS]),
        "",
        "MAE here is the mean absolute residual (= RKM MAE on that subset).",
        "",
        "**OBSERVATION.** Pooled over temperature, the mean residual is ordered Bi-rich < Equiatomic < Sn-rich, and "
        "the separation is much larger at high In content than overall.",
        "",
        "## Temperature-controlled comparison",
        "",
        "Mean residual (J/mol) by path at fixed temperature and xIn region (n per path in brackets). "
        "\"Ordered\" = Bi-rich < Equiatomic < Sn-rich.",
        "",
        "| T (K) | xIn region | Bi-rich | Equiatomic | Sn-rich | Range across paths | Ordered |",
        "| ---: | --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for _, r in cells.iterrows():
        vals = " | ".join(f"{r[s]:.1f} ({r[f'n_{s}']})" for s in SECTIONS)
        L.append(f"| {r['T']} | {r['region']} | {vals} | {r['range']:.1f} | {'yes' if r['ordered'] else 'no'} |")
    L += [
        "",
        f"**OBSERVATION.** The ordering holds in {n_ord} of {len(cells)} temperature × region cells. It holds in all "
        f"{int(hi_cells['ordered'].sum())} of {len(hi_cells)} cells with xIn ≥ 0.70 "
        f"(range across paths {hi_cells['range'].min():.0f}–{hi_cells['range'].max():.0f} J/mol), and in "
        f"{int(low['ordered'].sum())} of {len(low)} cells with xIn < 0.50 (range {low['range'].min():.0f}–"
        f"{low['range'].max():.0f} J/mol). The 0.50–0.70 cells are ordered but contain only 1–3 points per path. "
        "The path effect is therefore present at fixed temperature, and it is "
        "concentrated at high In content. Its size decreases with temperature in the In-rich cells.",
        "",
        "![Residual by section vs xIn](../figures/residual_by_section_vs_xIn.png)",
        "",
        "![Residual by section and temperature](../figures/residual_section_temperature_comparison.png)",
        "",
        "## High-In residual behavior",
        "",
        "Within xIn ≥ 0.80 (4 points per path per temperature): mean residual and least-squares slope of residual "
        "vs xIn. Slopes from 4 points spanning only ≈ 0.1 in xIn are imprecise (standard errors shown).",
        "",
        "| Path | T (K) | n | Mean residual | Slope d(residual)/d(xIn) | Slope SE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in slopes.iterrows():
        L.append(f"| {r['section']} | {r['T']} | {r['n']} | {r['mean']:.1f} | {r['slope']:.0f} | {r['slope_se']:.0f} |")
    m8 = slopes.pivot(index="T", columns="section", values="mean")
    L += [
        "",
        "Differences between path means at xIn ≥ 0.80:",
        "",
        "| T (K) | Equiatomic − Bi-rich | Sn-rich − Equiatomic | Sn-rich − Bi-rich |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for t, r in m8.iterrows():
        L.append(f"| {t} | {r['Equiatomic'] - r['Bi-rich']:.1f} | {r['Sn-rich'] - r['Equiatomic']:.1f} | {r['Sn-rich'] - r['Bi-rich']:.1f} |")
    L += [
        "",
        "![High-In section comparison](../figures/residual_high_in_section_comparison.png)",
        "",
        "**OBSERVATION.** Near the In-rich corner the three paths are cleanly separated at every temperature: the "
        f"Sn-rich − Bi-rich gap is {(m8['Sn-rich'] - m8['Bi-rich']).min():.0f}–{(m8['Sn-rich'] - m8['Bi-rich']).max():.0f} "
        "J/mol, compared with within-path scatter of ≈ 14 J/mol (next sections). The gap shrinks as temperature rises "
        "(767 → 855 K), while all three paths shift upward with temperature. Within xIn ≥ 0.80 the residual trend "
        "with xIn differs in sign between paths (positive slopes for Bi-rich, mostly negative for Sn-rich), but "
        "these slopes are imprecise.",
        "",
        "## Matched-composition comparison",
        "",
        f"To rule out that path differences merely reflect different xIn values, observations were matched one-to-one "
        f"at the **same temperature** with |ΔxIn| ≤ {MATCH_TOL} (no point reused; no interpolation). For triplets, "
        f"all three pairwise |ΔxIn| ≤ {MATCH_TOL}.",
        "",
        f"Sensitivity of match counts to the tolerance (pairs / triplets): "
        + "; ".join(f"±{k}: {v[0]} / {v[1]}" for k, v in sensitivity.items()) + ".",
        "",
        "Matched pairs (residual difference, first path minus second, J/mol):",
        "",
        "| Pair | n | Mean diff | SD | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for p, r in pg.iterrows():
        L.append(f"| {p} | {int(r['count'])} | {r['mean']:.1f} | {r['std']:.1f} | {r['min']:.1f} | {r['max']:.1f} |")
    L += [
        "",
        f"Matched triplets ({len(tri)}). `Mid. dev.` = Equiatomic − ½(Bi-rich + Sn-rich): the deviation of the middle "
        "path from the midpoint of the outer two. Since bi_sn_fraction is 0.6675 / 0.5000 / 0.3334, the Equiatomic "
        "path lies almost exactly halfway, so a small `Mid. dev.` relative to the gap means the three residuals are "
        "close to evenly spaced in this coordinate (a description of three points, not evidence of a linear law).",
        "",
        "| T (K) | xIn (B / E / S) | Bi-rich | Equiatomic | Sn-rich | Sn − Bi gap | Ordered | Mid. dev. |",
        "| ---: | --- | ---: | ---: | ---: | ---: | :---: | ---: |",
    ]
    for _, r in tri.iterrows():
        L.append(
            f"| {r['T']} | {r['xIn_Bi-rich']:.3f} / {r['xIn_Equiatomic']:.3f} / {r['xIn_Sn-rich']:.3f} | "
            f"{r['res_Bi-rich']:.1f} | {r['res_Equiatomic']:.1f} | {r['res_Sn-rich']:.1f} | {r['gap_Sn_minus_Bi']:.1f} | "
            f"{'yes' if r['ordered'] else 'no'} | {r['mid_dev']:+.1f} |"
        )
    L += [
        "",
        f"**OBSERVATION.** {len(tri)} matched triplets exist ({len(tri_hi)} with mean xIn ≥ 0.70). The ordering "
        f"holds in {int(tri_hi['ordered'].sum())}/{len(tri_hi)} high-In triplets and {int(tri_lo['ordered'].sum())}/"
        f"{len(tri_lo)} triplets below xIn 0.70. For high-In triplets the Sn − Bi gap is "
        f"{tri_hi['gap_Sn_minus_Bi'].min():.0f}–{tri_hi['gap_Sn_minus_Bi'].max():.0f} J/mol and |Mid. dev.| is "
        f"{tri_hi['mid_dev'].abs().min():.0f}–{tri_hi['mid_dev'].abs().max():.0f} J/mol (median "
        f"{tri_hi['mid_dev'].abs().median():.0f}). The path differences therefore persist at matched composition "
        "and temperature; they are not an artefact of comparing different xIn values.",
        "",
        "## Between-section vs within-section variation",
        "",
        "**(a) ANOVA-style decomposition within cells.** Observations are grouped into temperature × xIn-region "
        "cells (regions as above). Within each cell, the residual sum of squares is split into between-path and "
        "within-path parts, then pooled over cells. η² = SS_between / (SS_between + SS_within). Temperature and "
        "coarse xIn effects are removed by construction because comparisons are only made inside cells.",
        "",
        "| Subset | n | SS between paths | SS within paths | η² (path) | Within-path SD (J/mol) | F (df) | p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for k, a in anova.items():
        lab = "All xIn" if k == "all" else k
        L.append(
            f"| {lab} | {a['n']} | {a['ss_between']:.0f} | {a['ss_within']:.0f} | {a['eta2']:.3f} | {a['sd_within']:.1f} | "
            f"{a['F']:.1f} ({a['df'][0]}, {a['df'][1]}) | {a['p']:.1e} |"
        )
    L += [
        "",
        "**(b) Transparent linear diagnostic (not a prediction model; in-sample, no validation).** "
        "`common(xIn, T)` = separate intercept, linear and quadratic xIn terms for each temperature (9 parameters). "
        "`+ section` adds a constant offset for two of the paths. `+ section×xIn` lets that offset change linearly "
        "with xIn. Nested F-tests compare successive rows.",
        "",
        "| Subset | Terms | Parameters | R² | Residual SD (J/mol) | ΔR² | F | p |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for k, d in lin.items():
        lab = "All xIn" if k == "all" else k
        for name in ("common", "common + section", "common + section + section×xIn"):
            r = d[name]
            extra = f"{r['dR2']:+.3f} | {r['F']:.1f} | {r['p']:.1e}" if "F" in r else "— | — | —"
            L.append(f"| {lab} (n={d['n']}) | {name} | {r['k']} | {r['R2']:.3f} | {r['sd_resid']:.1f} | {extra} |")

    a7, a8 = anova["xIn ≥ 0.70"], anova["xIn ≥ 0.80"]
    l7, l8, la = lin["xIn ≥ 0.70"], lin["xIn ≥ 0.80"], lin["all"]
    L += [
        "",
        f"**OBSERVATION.** At fixed temperature and xIn region, the path accounts for η² = {anova['all']['eta2']:.2f} of "
        f"residual variation over all xIn, {a7['eta2']:.2f} for xIn ≥ 0.70 and {a8['eta2']:.2f} for xIn ≥ 0.80; the "
        f"within-path SD at high In is only ≈ {a8['sd_within']:.0f} J/mol. In the linear diagnostic for xIn ≥ 0.80, a "
        f"common xIn–T trend alone explains R² = {l8['common']['R2']:.2f}; adding one constant offset per path raises "
        f"this to {l8['common + section']['R2']:.2f} (residual SD {l8['common']['sd_resid']:.0f} → "
        f"{l8['common + section']['sd_resid']:.0f} J/mol). Over all xIn, a constant path offset is not enough "
        f"(R² {la['common + section']['R2']:.2f}); allowing it to vary with xIn raises R² to "
        f"{la['common + section + section×xIn']['R2']:.2f}, consistent with a path effect that is small at low xIn "
        "and large near the In corner.",
        "",
        "Caveat: these are in-sample descriptive fits on 104 (or 49 / 36) correlated points from only three paths; "
        "p-values assume independent errors and should be read as indicative only.",
        "",
        "## Implications for residual learning",
        "",
        "Model A and Model B held-out predictions on the Sn-rich path (read from existing outputs). Error "
        "decomposition: MSE = bias² + (SD_actual − SD_pred)² + 2·SD_actual·SD_pred·(1 − r). `MAE after removing mean "
        "offset` is the MAE that would remain if the mean error were subtracted (diagnostic only).",
        "",
        "| Subset | Model | MAE | Mean error (bias) | corr | SD actual | SD pred | bias² | scale term | shape term | MAE after removing mean offset |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lab, dd in decomp.items():
        for k in ("A", "B"):
            d = dd[k]
            L.append(
                f"| {lab} | {k} | {d['MAE']:.1f} | {d['bias']:+.1f} | {d['corr']:.2f} | {d['sd_actual']:.1f} | {d['sd_pred']:.1f} | "
                f"{d['bias_sq']:.0f} | {d['scale_sq']:.0f} | {d['shape']:.0f} | {d['MAE_offset_removed']:.1f} |"
            )
    L += [
        "",
        "Matched high-In triplets: actual vs held-out predicted residual for the **Sn-rich** point (each model "
        "trained on Bi-rich + Equiatomic only):",
        "",
        "| T (K) | xIn (Sn-rich) | Actual | Model A | Model B | Bi-rich actual | Equiatomic actual |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in tri_hi.iterrows():
        L.append(
            f"| {r['T']} | {r['xIn_Sn-rich']:.3f} | {r['res_Sn-rich']:.1f} | {r['A_Sn-rich']:.1f} | {r['B_Sn-rich']:.1f} | "
            f"{r['res_Bi-rich']:.1f} | {r['res_Equiatomic']:.1f} |"
        )
    L += [
        "",
        "![Sn-rich level vs shape](../figures/residual_sn_rich_level_vs_shape.png)",
        "",
        "**OBSERVATION.**",
        "",
        f"- Model B's Sn-rich error is dominated by bias: bias² = {sn_all_B['bias_sq']:.0f} of MSE {sn_all_B['mse']:.0f} "
        f"({100*sn_all_B['bias_sq']/sn_all_B['mse']:.0f}%); the shape term is only {sn_all_B['shape']:.0f}. With the mean "
        f"offset removed its MAE would be {sn_all_B['MAE_offset_removed']:.1f} J/mol instead of {sn_all_B['MAE']:.1f}. "
        "Correlation is insensitive to a constant offset, which is how r ≈ 0.91 and MAE ≈ 55 J/mol coexist.",
        f"- The bias is largest exactly where the path separation is largest: for Sn-rich xIn ≥ 0.80, Model A bias "
        f"{sn8['A']['bias']:+.1f} and Model B bias {sn8['B']['bias']:+.1f} J/mol, with correlations "
        f"{sn8['A']['corr']:.2f} / {sn8['B']['corr']:.2f}. Both models predict Sn-rich In-rich residuals that lie "
        "close to, or below, the Equiatomic level, whereas the measured Sn-rich residuals lie well above it.",
        f"- Model B also compresses the Sn-rich residual range (SD predicted {sn_all_B['sd_pred']:.1f} vs actual "
        f"{sn_all_B['sd_actual']:.1f} J/mol).",
        "",
        "**INTERPRETATION.**",
        "",
        "- The experimental residual contains a systematic path-dependent component that is large relative to "
        "within-path scatter, persists at fixed temperature and at matched composition, and is concentrated near "
        "the In-rich corner. In LOCSO, the level of this component for the held-out path is exactly the quantity a "
        "two-path model must extrapolate.",
        "- Model B learned the within-path shape (common xIn–T structure) but did not place the Sn-rich level beyond "
        "the Equiatomic level by the measured amount. Its failure is therefore a failure to extrapolate the path "
        "offset, not a failure to learn the shape.",
        "- With three paths, the across-path behaviour at a given (xIn, T) is described by three numbers. A two-path "
        "training fold sees two of them, which fixes at most a straight line in the Bi/Sn coordinate and gives no "
        "way to check it. The full data set gives one degree of freedom (`Mid. dev.`) to check the spacing, and only "
        "at the matched high-In compositions.",
        "",
        "**HYPOTHESIS (not tested here).**",
        "",
        "- At high In the three paths' residuals are close to evenly spaced in bi_sn_fraction (small `Mid. dev.` "
        "relative to the gap). If so, a model whose path term were allowed to vary jointly with xIn and T might "
        "extrapolate the Sn-rich level better than the degree-2 models tested, which can represent only limited "
        "path × xIn × T interaction and are rank-deficient with two training paths (Step 3). This is a hypothesis "
        "about representation, not a recommendation, and three paths may be too few to validate it.",
        "- The decrease of the path gap with temperature could reflect a temperature-dependent interaction that RKM "
        "(temperature-independent) cannot represent; no mechanism is claimed.",
        "",
        "## Conclusion",
        "",
        f"1. **Is there evidence for a cross-section-dependent residual component?** Yes. At fixed temperature and xIn "
        f"region the path explains η² = {anova['all']['eta2']:.2f} of residual variation overall, and the "
        "difference persists for matched compositions at the same temperature.",
        f"2. **Is it especially strong near the In-rich corner?** Yes. η² = {a7['eta2']:.2f} (xIn ≥ 0.70) and "
        f"{a8['eta2']:.2f} (xIn ≥ 0.80), with path gaps of ≈ {(m8['Sn-rich'] - m8['Bi-rich']).min():.0f}–"
        f"{(m8['Sn-rich'] - m8['Bi-rich']).max():.0f} J/mol versus ≈ {a8['sd_within']:.0f} J/mol within-path SD. "
        "At xIn < 0.50 the ordering is not consistent and the differences are small.",
        f"3. **Is the effect present at fixed temperature?** Yes. The Bi-rich < Equiatomic < Sn-rich ordering holds "
        f"in all {len(hi_cells)} temperature × region cells with xIn ≥ 0.70. Its size decreases from 767 to 855 K.",
        "4. **Can the available three paths adequately constrain that effect?** Only weakly. The path effect is "
        "sampled at three Bi/Sn values; under LOCSO a model sees two, which cannot determine curvature in the "
        "Bi/Sn coordinate or check extrapolation. The full data set offers a single check (the middle path), which "
        "at high In is roughly consistent with even spacing, but that cannot be confirmed with three paths.",
        "5. **Does this explain why a model trained on two paths struggles on the third?** It is consistent with it "
        "and accounts for the Sn-rich error pattern: the error is concentrated where the path effect is largest, "
        "and Model B's error there is almost entirely a level (bias) error while the shape is captured. This "
        "establishes an empirical pattern; it does not identify a thermodynamic mechanism.",
        "",
        "## Figures",
        "",
        *[f"- `figures/{v}`" for v in FIGS.values()],
        "",
    ]
    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
