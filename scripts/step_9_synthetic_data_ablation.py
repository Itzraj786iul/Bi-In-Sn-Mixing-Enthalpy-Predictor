"""
Ablation study of RKM-derived synthetic data for the direct Poly D2 model.

    python scripts/step_9_synthetic_data_ablation.py

Question: do RKM-derived synthetic data (Dataset A, Dataset B) add predictive
information for the 104 real experimental measurements under strict LOCSO?

Model: existing direct Poly D2 on (xBi, xIn, temperature_K), minimum-norm lstsq,
no weighting, no tuning. Test set in every experiment: the held-out real
experimental cross-section only.

  A  Real only           104 real (LOCSO)
  B  Real + Dataset A    real + Dataset A from the two training paths only
  C  Real + Dataset B    NOT TRAINED - no scientifically justified exclusion
                         threshold exists in the project (see report); a
                         geometry-only diagnostic is reported instead
  D  Dataset A only      Dataset A from the two training paths only
                         ("RKM-derived synthetic-to-experimental transfer diagnostic")

Synthetic files are read, never written or regenerated. RKM is the fixed
src/rkm_model.py prediction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_composition_representation import (  # noqa: E402
    EXP_TARGET,
    calculate_metrics,
    get_locsos_folds,
    predict_polynomial,
    train_polynomial,
)
from generate_synthetic_data import (  # noqa: E402
    ANCHOR_DISTANCE,
    NEAR_CS_DISTANCE,
    SERIES_U_J,
    bath_ratio,
    distance_to_cross_section_line,
)
from step_4_residual_learning import DIRECT_POLY_D2_REF, SECTION_LABELS, load_experimental  # noqa: E402

SYN_A_PATH = ROOT / "data" / "synthetic_cross_sections.csv"
SYN_B_PATH = ROOT / "data" / "synthetic_full_ternary.csv"
REPORT_PATH = ROOT / "reports" / "synthetic_data_ablation.md"
CSV_PATH = ROOT / "reports" / "synthetic_data_ablation.csv"

FEATURES = ["xBi", "xIn", "temperature_K"]
SYN_TARGET = "delta_mix_H_J_per_mol"
SECTIONS = ["Bi-rich", "Equiatomic", "Sn-rich"]
TEMPS = [767, 813, 855]
REGIMES = {
    "RKM": "RKM (fixed)",
    "A": "Real only",
    "B": "Real + Dataset A",
    "D": "Dataset A only",
}
RKM_REF = {"MAE": 57.13, "RMSE": 73.49, "R2": 0.9510}


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exp = load_experimental()
    exp["section"] = exp["cross_section"].map(SECTION_LABELS)
    syn_a = pd.read_csv(SYN_A_PATH)
    syn_b = pd.read_csv(SYN_B_PATH)
    if len(exp) != 104 or len(syn_a) != 7389 or len(syn_b) != 15453:
        raise RuntimeError("Unexpected dataset sizes")
    if set(syn_a["cross_section"]) != set(exp["cross_section"]):
        raise RuntimeError("Dataset A cross-section labels do not match the experimental labels")
    if not (syn_a["synthetic"].astype(bool).all() and syn_b["synthetic"].astype(bool).all()):
        raise RuntimeError("Synthetic files must be flagged synthetic")
    return exp, syn_a, syn_b


def dataset_a_provenance(exp: pd.DataFrame, syn_a: pd.DataFrame) -> dict:
    """Empirical check of how Dataset A targets relate to RKM and to the experiment."""
    recon = syn_a["base_model_prediction_J_per_mol"] + syn_a["noise_added_J_per_mol"]
    noise = syn_a["noise_added_J_per_mol"]
    per_series = syn_a.groupby("series_id").agg(sd_noise=("noise_added_J_per_mol", "std"),
                                               mean_noise=("noise_added_J_per_mol", "mean"),
                                               sigma=("uncertainty_J_per_mol", "first"))
    # For each experimental point: the nearest Dataset A point on the same path and temperature.
    rows = []
    for _, e in exp.iterrows():
        cand = syn_a[(syn_a["cross_section"] == e["cross_section"]) & (syn_a["temperature_K"] == e["temperature_K"])]
        d = np.sqrt((cand["xBi"] - e["xBi"]) ** 2 + (cand["xIn"] - e["xIn"]) ** 2 + (cand["xSn"] - e["xSn"]) ** 2)
        s = cand.loc[d.idxmin()]
        rows.append({"d": float(d.min()), "exp_resid": e["residual"], "syn_minus_rkm": s[SYN_TARGET] - s["base_model_prediction_J_per_mol"],
                     "syn_minus_exp": s[SYN_TARGET] - e[EXP_TARGET]})
    nn = pd.DataFrame(rows)
    r, p = stats.pearsonr(nn["syn_minus_rkm"], nn["exp_resid"])
    return {
        "corr_p": float(p),
        "target_equals_rkm_plus_noise": float(np.max(np.abs(recon - syn_a[SYN_TARGET]))),
        "noise_mean": float(noise.mean()),
        "noise_sd": float(noise.std(ddof=1)),
        "per_series": per_series,
        "nn_max_d": float(nn["d"].max()),
        "corr_noise_vs_exp_residual": float(np.corrcoef(nn["syn_minus_rkm"], nn["exp_resid"])[0, 1]),
        "mean_abs_syn_minus_exp": float(nn["syn_minus_exp"].abs().mean()),
        "mean_abs_exp_resid": float(nn["exp_resid"].abs().mean()),
    }


def fit_predict(X_train, y_train, X_test) -> tuple[np.ndarray, int, int]:
    poly, model = train_polynomial(X_train, y_train)
    X_poly = poly.transform(X_train)
    return predict_polynomial(poly, model, X_test), int(np.linalg.matrix_rank(X_poly)), int(X_poly.shape[1])


def run(exp: pd.DataFrame, syn_a: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    oof, info = [], []
    for fold, (held_out, train, test) in enumerate(get_locsos_folds(exp), start=1):
        if set(train["id"]) & set(test["id"]) or held_out in set(train["cross_section"]):
            raise RuntimeError(f"Fold {fold}: real held-out data in training")
        syn_train = syn_a[syn_a["cross_section"] != held_out]
        if (syn_train["cross_section"] == held_out).any():
            raise RuntimeError(f"Fold {fold}: Dataset A from held-out path in training")

        X_test = test[FEATURES].to_numpy(float)
        X_real, y_real = train[FEATURES].to_numpy(float), train[EXP_TARGET].to_numpy(float)
        X_syn, y_syn = syn_train[FEATURES].to_numpy(float), syn_train[SYN_TARGET].to_numpy(float)

        part = test[["id", "section", "temperature_K", "xBi", "xIn", "xSn", EXP_TARGET, "rkm_prediction"]].copy()
        part["pred_RKM"] = test["rkm_prediction"].to_numpy(float)
        part["pred_A"], rA, nA = fit_predict(X_real, y_real, X_test)
        part["pred_B"], rB, nB = fit_predict(np.vstack([X_real, X_syn]), np.concatenate([y_real, y_syn]), X_test)
        part["pred_D"], rD, nD = fit_predict(X_syn, y_syn, X_test)
        part["fold"] = fold
        oof.append(part)
        info.append({
            "fold": fold, "held_out": SECTION_LABELS[held_out], "test": len(test),
            "A": {"real": len(train), "syn": 0, "rank": f"{rA}/{nA}"},
            "B": {"real": len(train), "syn": len(syn_train), "rank": f"{rB}/{nB}"},
            "D": {"real": 0, "syn": len(syn_train), "rank": f"{rD}/{nD}"},
        })
    oof = pd.concat(oof, ignore_index=True)
    if len(oof) != 104 or oof["id"].duplicated().any():
        raise RuntimeError("Each experimental observation must be predicted exactly once")
    return oof, info


def dataset_b_geometry(exp: pd.DataFrame, syn_b: pd.DataFrame) -> dict:
    """Why Dataset B cannot be separated from a held-out path without an arbitrary threshold."""
    comps = syn_b.drop_duplicates(["xBi", "xIn", "xSn"])[["xBi", "xIn", "xSn"]].reset_index(drop=True)
    ratios = {SECTION_LABELS[cs]: bath_ratio(exp, cs) for cs in SECTION_LABELS}
    d = {s: distance_to_cross_section_line(comps["xBi"], comps["xIn"], comps["xSn"], r) for s, r in ratios.items()}
    out = {"n_comps": len(comps), "n_rows": len(syn_b), "ratios": ratios, "per_path": {}}
    for s in SECTIONS:
        others = [o for o in SECTIONS if o != s]
        d_other = np.minimum(d[others[0]], d[others[1]])
        test = exp[exp["section"] == s]
        # Nearest Dataset B composition to each real point on the held-out path
        p = test[["xBi", "xIn", "xSn"]].to_numpy(float)
        c = comps.to_numpy(float)
        nn = np.sqrt(((p[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        near = d[s] <= NEAR_CS_DISTANCE
        out["per_path"][s] = {
            "conv_rows_removed": int(near.sum()) * len(TEMPS),
            "conv_rows_retained": int((~near).sum()) * len(TEMPS),
            "conv_min_d_removed": float(d[s][near].min()),
            "conv_min_d_retained": float(d[s][~near].min()),
            "nn_to_real_max": float(nn.max()),
            "nn_to_real_median": float(np.median(nn)),
            "n_within_conv": int(near.sum()),
            "n_within_conv_also_near_training": int((near & (d_other <= NEAR_CS_DISTANCE)).sum()),
            "n_closer_to_heldout_than_training": int((d[s] < d_other).sum()),
        }
    # Separation between adjacent experimental paths at the highest measured xIn
    xin_max = float(exp["xIn"].max())
    sep = {}
    for a, b in [("Bi-rich", "Equiatomic"), ("Equiatomic", "Sn-rich")]:
        sep[f"{a}–{b}"] = float(np.sqrt(2) * abs(ratios[a] - ratios[b]) * (1 - xin_max))
    out["xin_max"] = xin_max
    out["path_separation_at_xin_max"] = sep
    out["path_separation_at_xin_0"] = {k: v / (1 - xin_max) for k, v in sep.items()}
    return out


def metrics_table(oof: pd.DataFrame) -> dict:
    res = {}
    for k in REGIMES:
        res[k] = {"all": calculate_metrics(oof[EXP_TARGET], oof[f"pred_{k}"])}
        for s in SECTIONS:
            g = oof[oof["section"] == s]
            res[k][s] = calculate_metrics(g[EXP_TARGET], g[f"pred_{k}"])
        for t in TEMPS:
            g = oof[oof["temperature_K"] == t]
            res[k][t] = calculate_metrics(g[EXP_TARGET], g[f"pred_{k}"])
    return res


def build_csv(res: dict, info: list[dict]) -> pd.DataFrame:
    rows = []
    fold_of = {i["held_out"]: i for i in info}
    for k in REGIMES:
        for scope, m in res[k].items():
            if scope in SECTIONS:
                f = fold_of[scope]
                tr = f.get(k, {"real": 0, "syn": 0})
                rows.append({"experiment": k if k != "RKM" else "benchmark", "fold": f["fold"], "model_data_regime": REGIMES[k],
                             "section": scope, "temperature_K": "all",
                             "train_real_count": tr["real"], "train_synthetic_count": tr["syn"], "test_count": m["N"],
                             "MAE": m["MAE"], "RMSE": m["RMSE"], "R2": m["R2"]})
            else:
                rows.append({"experiment": k if k != "RKM" else "benchmark", "fold": "pooled", "model_data_regime": REGIMES[k],
                             "section": "all", "temperature_K": scope,
                             "train_real_count": "", "train_synthetic_count": "", "test_count": m["N"],
                             "MAE": m["MAE"], "RMSE": m["RMSE"], "R2": m["R2"]})
    rows.append({"experiment": "C", "fold": "not run", "model_data_regime": "Real + Dataset B",
                 "section": "all", "temperature_K": "all", "train_real_count": "", "train_synthetic_count": "",
                 "test_count": "", "MAE": np.nan, "RMSE": np.nan, "R2": np.nan})
    return pd.DataFrame(rows)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    exp, syn_a, syn_b = load()
    print(f"Real N = {len(exp)}; Dataset A = {len(syn_a)}; Dataset B = {len(syn_b)} (read-only)")
    prov = dataset_a_provenance(exp, syn_a)
    print({k: v for k, v in prov.items() if k != "per_series"})
    print(prov["per_series"].round(2))

    oof, info = run(exp, syn_a)
    res = metrics_table(oof)
    for ref, k in ((RKM_REF, "RKM"), (DIRECT_POLY_D2_REF, "A")):
        m = res[k]["all"]
        if abs(m["MAE"] - ref["MAE"]) > 0.01 or abs(m["RMSE"] - ref["RMSE"]) > 0.01:
            raise RuntimeError(f"{k} benchmark not reproduced: {m}")
    for f in info:
        print(f)
    for k in REGIMES:
        r = res[k]
        print(f"{REGIMES[k]:18s} " + " | ".join(f"{s}: {r[s]['MAE']:.2f}" for s in ["all", *SECTIONS, *TEMPS])
              + f" | RMSE {r['all']['RMSE']:.2f} R2 {r['all']['R2']:.4f}")
    geo = dataset_b_geometry(exp, syn_b)
    print(geo)

    build_csv(res, info).to_csv(CSV_PATH, index=False)
    write_report(exp, syn_a, syn_b, prov, oof, info, res, geo)
    print(f"Wrote {CSV_PATH}\nWrote {REPORT_PATH}")
    return 0


def write_report(exp, syn_a, syn_b, prov, oof, info, res, geo) -> None:
    R = res
    mae = "MAE"
    d = {k: R[k]["all"]["MAE"] - R["A"]["all"]["MAE"] for k in ("B", "D")}
    ratio = {f["held_out"]: f["B"]["syn"] / f["B"]["real"] for f in info}
    L: list[str] = [
        "# Synthetic Data Ablation Study",
        "",
        "Script: `scripts/step_9_synthetic_data_ablation.py`. Machine-readable results: `reports/synthetic_data_ablation.csv`. "
        "Synthetic files were read only; nothing was regenerated, modified or added to any model or the website.",
        "",
        "## Motivation",
        "",
        "Methodological question: **\"Why use synthetic data if RKM already provides the values?\"** This study tests, "
        "with the existing direct Poly D2 model and strict LOCSO, whether training on RKM-derived synthetic data "
        "improves prediction of the 104 real calorimetric measurements compared with training on the real "
        "measurements alone. It is not a search for a better model.",
        "",
        "## Dataset provenance",
        "",
        "### Experimental data",
        "",
        f"`data/original_experimental_data.csv`: {len(exp)} calorimetric measurements (Table III) on three composition "
        "paths (Bi/(Bi+Sn) = 0.6675, 0.5000, 0.3334) at 767, 813 and 855 K. These are the only independent "
        "measurements and the only test data in this study.",
        "",
        "### Dataset A",
        "",
        f"`data/synthetic_cross_sections.csv`: {len(syn_a)} rows generated by `scripts/generate_synthetic_data.py` on the "
        "same three paths (xIn grid 0.09–0.91, step 0.001, three temperatures). Construction, checked against the file:",
        "",
        "```text",
        "target = RKM(xBi, xIn, xSn) + ε,   ε ~ N(0, σ_series),   σ_series = Table III footnote u(ΔmixH)",
        "```",
        "",
        f"- `delta_mix_H_J_per_mol` equals `base_model_prediction_J_per_mol + noise_added_J_per_mol` to "
        f"{prov['target_equals_rkm_plus_noise']:.0e} J/mol on every row; `scripts/validate_dataset.py` separately confirms "
        "`base_model_prediction_J_per_mol` equals the RKM function.",
        f"- The added noise has mean {prov['noise_mean']:.2f} J/mol and SD {prov['noise_sd']:.2f} J/mol overall; per series "
        "its SD matches the assigned σ:",
        "",
        "| Series | σ assigned (J/mol) | Noise SD in file | Noise mean |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for sid, r in prov["per_series"].iterrows():
        L.append(f"| {int(sid)} | {r['sigma']:.2f} | {r['sd_noise']:.2f} | {r['mean_noise']:+.2f} |")
    L += [
        "",
        f"- Every experimental point has a Dataset A point on the same path and temperature within "
        f"{prov['nn_max_d']:.4f} in composition. At those nearest points, the synthetic target differs from the "
        f"measured value by {prov['mean_abs_syn_minus_exp']:.2f} J/mol on average, essentially the RKM error "
        f"({prov['mean_abs_exp_resid']:.2f} J/mol). The synthetic deviation from RKM (the noise) has correlation "
        f"{prov['corr_noise_vs_exp_residual']:.2f} (p = {prov['corr_p']:.2f}, n = 104) with the experimental deviation "
        "from RKM; the generator draws the noise from a seeded random-number generator without reference to the "
        "measured values, so this weak correlation is consistent with chance.",
        "",
        "### Dataset B",
        "",
        f"`data/synthetic_full_ternary.csv`: {len(syn_b)} rows = {geo['n_comps']} compositions on a 0.01 simplex grid × 3 "
        "temperatures. Target = RKM exactly (noise = 0), identical at all three temperatures. Most rows lie in "
        "regions that were never measured.",
        "",
        "## Validation protocol",
        "",
        "Strict leave-one-cross-section-out (LOCSO), the same three folds as all previous steps. In each fold the "
        "entire held-out experimental path is the test set (34 or 36 real observations); all real observations on "
        "that path are excluded from training, and **all Dataset A rows labelled with that path are excluded** "
        "(checked in code). Every experiment is evaluated on the same 104 real observations; no synthetic row is "
        "ever in a test set. RKM predictions come from the unchanged `src/rkm_model.py` and are never fitted.",
        "",
        "Model in every experiment: existing direct Poly D2, `PolynomialFeatures(degree=2)` on (xBi, xIn, T), "
        "minimum-norm `np.linalg.lstsq`, unweighted, no tuning. Real and synthetic rows enter the least-squares fit "
        "with equal weight (no re-weighting was introduced).",
        "",
        "## Experiments",
        "",
        "- **A. Real only** — train on real observations of the two training paths.",
        "- **B. Real + Dataset A** — A plus Dataset A rows on the two training paths.",
        "- **C. Real + Dataset B** — **not run** (see below).",
        "- **D. Dataset A only** — Dataset A rows on the two training paths; no real data. Reported as an "
        "*RKM-derived synthetic-to-experimental transfer diagnostic*, not as experimental validation of a "
        "synthetic model.",
        "",
        "### Why Experiment C was not run",
        "",
        "A leakage-safe version requires removing Dataset B points that are \"sufficiently close\" to the held-out "
        "path. The project contains no scientifically justified threshold for this:",
        "",
        f"- The only path distance in the project is `NEAR_CS_DISTANCE = {NEAR_CS_DISTANCE}` (and `ANCHOR_DISTANCE = "
        f"{ANCHOR_DISTANCE}`) in `scripts/generate_synthetic_data.py`. They are labels for the `region_type` column "
        "with no documented derivation from experimental uncertainty or physics.",
        "- `docs/ml_validation_strategy.md` explicitly states that a distance exclusion zone is \"possible but "
        "arbitrary threshold; **not recommended**\" and that cross-section exclusion should be preferred \"over "
        "arbitrary distance cutoffs\". Cross-section exclusion is not available for Dataset B because its rows have "
        "no path label.",
        f"- Geometry makes any fixed threshold problematic: all three paths meet at the pure-In corner. The "
        f"distance between adjacent experimental paths is ≈ {list(geo['path_separation_at_xin_0'].values())[0]:.3f} at "
        f"xIn = 0 but only ≈ {list(geo['path_separation_at_xin_max'].values())[0]:.3f} at the highest measured xIn "
        f"({geo['xin_max']:.3f}), smaller than {NEAR_CS_DISTANCE}. A threshold large enough to protect the held-out "
        "path at low xIn also removes Dataset B points next to the training paths at high xIn; a smaller one leaves "
        "RKM grid points within a fraction of a grid step of the held-out experimental compositions.",
        "",
        "Following the experiment's instructions, no threshold was chosen and Experiment C was not trained. For "
        "transparency only, this is what the existing `region_type` labelling distance would imply "
        "(**illustrative; not used for any training**):",
        "",
        "| Held-out path | Nearest Dataset B composition to a real test point (median / max) | Rows removed at d ≤ 0.03 | Rows retained | Min distance of removed rows | Min distance of retained rows | Removed compositions also ≤ 0.03 from a training path |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in SECTIONS:
        g = geo["per_path"][s]
        L.append(
            f"| {s} | {g['nn_to_real_median']:.4f} / {g['nn_to_real_max']:.4f} | {g['conv_rows_removed']} | {g['conv_rows_retained']} | "
            f"{g['conv_min_d_removed']:.4f} | {g['conv_min_d_retained']:.4f} | {g['n_within_conv_also_near_training']} of {g['n_within_conv']} |"
        )
    L += [
        "",
        "Every real test point has a Dataset B grid neighbour within ≈ 0.008 in composition, and a non-trivial "
        "share of compositions near each held-out path are equally near a training path. Question 3 is therefore "
        "left unanswered rather than answered with an arbitrary cutoff.",
        "",
        "## Results",
        "",
        "Pooled over the 104 real observations (LOCSO):",
        "",
        "| Model/Data regime | MAE (J/mol) | RMSE (J/mol) | R² |",
        "| --- | ---: | ---: | ---: |",
    ]
    for k, lab in (("RKM", "RKM (fixed benchmark)"), ("A", "Real only"), ("B", "Real + Dataset A"), (None, "Real + Dataset B"), ("D", "Dataset A only")):
        if k is None:
            L.append(f"| {lab} | not run | not run | not run |")
        else:
            m = R[k]["all"]
            L.append(f"| {lab} | {m['MAE']:.2f} | {m['RMSE']:.2f} | {m['R2']:.4f} |")
    L += [
        "",
        "Real-only and RKM reproduce the existing benchmarks (41.10 / 56.34 / 0.9712 and 57.13 / 73.49 / 0.9510).",
        "",
        "Section-wise MAE (J/mol; each section is its own held-out LOCSO fold):",
        "",
        "| Section | RKM | Real only | Real + Dataset A | Dataset A only |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for s in SECTIONS:
        L.append(f"| {s} | " + " | ".join(f"{R[k][s]['MAE']:.2f}" for k in ("RKM", "A", "B", "D")) + " |")
    L += ["", "Temperature-wise MAE (J/mol; pooled held-out predictions):", "",
          "| Temperature | RKM | Real only | Real + Dataset A | Dataset A only |",
          "| ---: | ---: | ---: | ---: | ---: |"]
    for t in TEMPS:
        L.append(f"| {t} K | " + " | ".join(f"{R[k][t]['MAE']:.2f}" for k in ("RKM", "A", "B", "D")) + " |")

    D, A_, B_ = R["D"], R["A"], R["B"]
    L += [
        "",
        "## Synthetic-to-real transfer",
        "",
        f"*RKM-derived synthetic-to-experimental transfer diagnostic* (Experiment D): pooled MAE {D['all']['MAE']:.2f}, "
        f"RMSE {D['all']['RMSE']:.2f} J/mol, compared with RKM itself at {R['RKM']['all']['MAE']:.2f} / "
        f"{R['RKM']['all']['RMSE']:.2f}. A Poly D2 trained only on RKM + noise along two paths reproduces the real "
        "measurements on the third path with errors of the same order as RKM: somewhat lower pooled MAE and lower "
        f"on Bi-rich ({D['Bi-rich']['MAE']:.2f} vs {R['RKM']['Bi-rich']['MAE']:.2f}), higher on Equiatomic "
        f"({D['Equiatomic']['MAE']:.2f} vs {R['RKM']['Equiatomic']['MAE']:.2f}) and Sn-rich "
        f"({D['Sn-rich']['MAE']:.2f} vs {R['RKM']['Sn-rich']['MAE']:.2f}), and a slightly higher pooled RMSE. This is "
        "expected: the model is "
        "approximating RKM, and its error relative to experiment is inherited from RKM plus the approximation "
        "error of a quadratic surface and of extrapolating to an unseen path.",
        "",
        "## Effect of synthetic augmentation",
        "",
        "ΔMAE = MAE(regime) − MAE(Real only); negative = improvement, positive = degradation.",
        "",
        "| Regime | Pooled ΔMAE | Bi-rich ΔMAE | Equiatomic ΔMAE | Sn-rich ΔMAE | 767 K | 813 K | 855 K |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for k in ("B", "D"):
        L.append(f"| {REGIMES[k]} | {R[k]['all']['MAE'] - A_['all']['MAE']:+.2f} | "
                 + " | ".join(f"{R[k][s]['MAE'] - A_[s]['MAE']:+.2f}" for s in [*SECTIONS, *TEMPS]) + " |")
    L.append("| Real + Dataset B | not run | | | | | | |")
    L += [
        "",
        f"**Magnitude.** Adding Dataset A raises pooled MAE by {d['B']:+.2f} J/mol ({100 * d['B'] / A_['all']['MAE']:+.0f}%) "
        f"and RMSE from {A_['all']['RMSE']:.2f} to {B_['all']['RMSE']:.2f} J/mol. The degradation is present on all "
        f"three held-out paths ({', '.join(f'{s} {B_[s][mae] - A_[s][mae]:+.1f}' for s in SECTIONS)} J/mol) "
        "and is largest at 767 K; at 855 K the change is negligible "
        f"({B_[855]['MAE'] - A_[855]['MAE']:+.2f} J/mol). These differences are large compared with the "
        "tabulated measurement uncertainties (7.9–23.3 J/mol) and consistent across folds, so they are not a "
        "rounding-level effect; with only three folds they are nevertheless descriptive, not a formal test.",
        "",
        "## Data dominance",
        "",
        "| Fold | Held-out path | Regime | Real training | Synthetic training | Total | Synthetic / real | Test (real) |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for f in info:
        for k in ("A", "B", "D"):
            tr = f[k]
            rr = f"{tr['syn'] / tr['real']:.1f}" if tr["real"] else "— (no real)"
            L.append(f"| {f['fold']} | {f['held_out']} | {REGIMES[k]} | {tr['real']} | {tr['syn']} | {tr['real'] + tr['syn']} | {rr} | {f['test']} |")
    L += [
        "",
        f"With unweighted least squares, the ≈ {min(ratio.values()):.0f}–{max(ratio.values()):.0f} : 1 synthetic-to-real "
        f"ratio means the real observations carry about 1.4% of the training weight. Real + Dataset A "
        f"(MAE {B_['all']['MAE']:.2f}) is therefore almost identical to Dataset A only (MAE {D['all']['MAE']:.2f}): the "
        "fitted surface is essentially the quadratic approximation of RKM along the training paths, and the "
        "experimental deviations from RKM that the real-only model learns are swamped.",
        "",
        "## Scientific interpretation",
        "",
        "**Observation**",
        "",
        f"- Real only: MAE {A_['all']['MAE']:.2f}, RMSE {A_['all']['RMSE']:.2f}, R² {A_['all']['R2']:.4f}.",
        f"- Real + Dataset A: MAE {B_['all']['MAE']:.2f}, RMSE {B_['all']['RMSE']:.2f}, R² {B_['all']['R2']:.4f} — worse on "
        "every held-out path.",
        f"- Dataset A only: MAE {D['all']['MAE']:.2f}, RMSE {D['all']['RMSE']:.2f}, R² {D['all']['R2']:.4f} — close to RKM.",
        "- Real + Dataset B: not evaluated (no leakage-safe protocol).",
        "- The same pattern was found earlier with a four-feature variant (xBi, xIn, xSn, T): experimental-only "
        "43.81, synthetic-only 57.35, combined 56.73 J/mol (`reports/ml_polynomial_synthetic_temperature_results.md`).",
        "",
        "**Interpretation**",
        "",
        "- *Question 1 — Does Dataset A provide independent experimental information?* No. By construction its "
        "target is the RKM prediction plus Gaussian noise whose size is set from the reported measurement "
        "uncertainty; the file confirms this exactly. The synthetic targets encode the existing RKM model and "
        "therefore do not constitute independent experimental information. The noise adds variability, not "
        "information about how real alloys deviate from RKM.",
        f"- *Question 2 — Does Dataset A improve prediction of real experimental measurements?* Under the tested "
        f"LOCSO protocol, RKM-derived synthetic augmentation did not provide evidence of improved prediction of the "
        f"real experimental observations; pooled MAE increased by {d['B']:.2f} J/mol and every held-out path got worse.",
        "- *Question 3 — Does Dataset B improve prediction of real experimental measurements?* Not determined. The "
        "experiment cannot be made leakage-safe without defining a path-exclusion threshold, and the project has "
        "no justified one.",
        f"- *Question 4 — Does synthetic data outperform simply using the 104 real observations?* No, for the "
        f"configurations that could be tested: Real only ({A_['all']['MAE']:.2f}) < Real + Dataset A "
        f"({B_['all']['MAE']:.2f}) ≈ Dataset A only ({D['all']['MAE']:.2f}), both close to RKM "
        f"({R['RKM']['all']['MAE']:.2f}; RMSE {R['RKM']['all']['RMSE']:.2f} vs {B_['all']['RMSE']:.2f} / {D['all']['RMSE']:.2f}).",
        "- *Question 5 — Should synthetic data be used in the final predictor?* See Recommendation.",
        "",
        "**Limitation**",
        "",
        "- One model family (direct Poly D2) and unweighted fitting only; no re-weighting of real vs synthetic rows "
        "was tested, by design (no tuning). A different weighting would be a new methodological choice that "
        "would need its own justification and validation.",
        "- Three folds; differences are descriptive.",
        "- Dataset B could not be evaluated as a training source.",
        "- These results concern supervised training for predicting the measured paths. They say nothing against "
        "the use of Dataset B for exploratory surface analysis, sensitivity analysis, visualisation or hypothesis "
        "generation, for which it remains suitable; its use for supervised training requires careful separation "
        "from experimental test regions, which this project has not defined.",
        "",
        "## Recommendation",
        "",
        "Based on experimental validation, RKM-derived synthetic data should **not** be part of the training set "
        "of the final supervised predictor. Under strict LOCSO, adding Dataset A made predictions of the real "
        "measurements worse on every held-out path, and the Dataset A-only model performed at roughly the level of "
        "RKM itself (better on one path, worse on two), which is available directly without any training. RKM "
        "should remain the fixed physics-based "
        "benchmark, reported alongside the ML model; Dataset A and Dataset B remain useful as documented "
        "RKM-derived reference data for visualisation and exploratory analysis, clearly labelled as not "
        "experimental. No change to the current final model or the website was made in this step.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
