"""
ML vs RKM ternary surface comparison for the web application.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from model import predict_delta_mix_h_batch
from src.rkm_model import rkm_delta_mix_h
from surface import GRID_STEP, SURFACE_TEMPERATURES, build_ternary_grid, load_experiments

COMPARISON_MODES = ("ml", "rkm", "difference")


def _validate_temperature(temperature_k: float) -> None:
    if temperature_k not in SURFACE_TEMPERATURES:
        allowed = ", ".join(str(t) for t in SURFACE_TEMPERATURES)
        raise ValueError(f"Temperature must be one of: {allowed} K.")


def _validate_mode(mode: str) -> None:
    if mode not in COMPARISON_MODES:
        allowed = ", ".join(COMPARISON_MODES)
        raise ValueError(f"Mode must be one of: {allowed}.")


def _rkm_batch(x_bi: list[float], x_in: list[float], x_sn: list[float]) -> list[float]:
    values = rkm_delta_mix_h(
        np.asarray(x_bi, dtype=float),
        np.asarray(x_in, dtype=float),
        np.asarray(x_sn, dtype=float),
    )
    return np.asarray(values, dtype=float).tolist()


def _experiments_at_temperature(temperature_k: float) -> list[dict]:
    rows: list[dict] = []
    for row in load_experiments():
        if row["temperature_K"] != temperature_k:
            continue
        enriched = dict(row)
        enriched["rkm_prediction_J_mol"] = rkm_delta_mix_h(
            row["xBi"], row["xIn"], row["xSn"]
        )
        rows.append(enriched)
    return rows


def _surface_statistics(ml_values: list[float], rkm_values: list[float]) -> dict:
    differences = [m - r for m, r in zip(ml_values, rkm_values, strict=True)]
    abs_differences = [abs(d) for d in differences]
    return {
        "ml_range_J_mol": [min(ml_values), max(ml_values)],
        "rkm_range_J_mol": [min(rkm_values), max(rkm_values)],
        "mean_abs_difference_J_mol": sum(abs_differences) / len(abs_differences),
        "max_abs_difference_J_mol": max(abs_differences),
    }


def build_comparison(temperature_k: float, mode: str) -> dict:
    _validate_temperature(temperature_k)
    _validate_mode(mode)

    x_bi, x_in, x_sn = build_ternary_grid()
    ml_values = predict_delta_mix_h_batch(x_bi, x_in, temperature_k)
    rkm_values = _rkm_batch(x_bi, x_in, x_sn)
    difference = [m - r for m, r in zip(ml_values, rkm_values, strict=True)]

    experiments = _experiments_at_temperature(temperature_k)
    all_experiments = load_experiments()
    statistics = _surface_statistics(ml_values, rkm_values)

    response: dict = {
        "temperature_K": temperature_k,
        "mode": mode,
        "grid_step": GRID_STEP,
        "n_grid_points": len(x_bi),
        "xBi": x_bi,
        "xIn": x_in,
        "xSn": x_sn,
        "statistics": statistics,
        "experimental_points": experiments,
        "experimental_count_at_temperature": len(experiments),
        "experimental_total": len(all_experiments),
    }

    if mode == "ml":
        response["delta_mix_H"] = ml_values
    elif mode == "rkm":
        response["delta_mix_H"] = rkm_values
    else:
        response["delta_mix_H_ml"] = ml_values
        response["delta_mix_H_rkm"] = rkm_values
        response["difference"] = difference

    return response
