"""
Frozen Polynomial Degree-2 mixing enthalpy surrogate (Step 4A).

Features: xBi, xIn, temperature_K
Output: integral molar mixing enthalpy ΔmixH in J/mol
"""

from __future__ import annotations

# Coefficients from reports/final_model_coefficients.csv (do not change)
INTERCEPT = 1477.018848
COEF_XBI = -1633.602767
COEF_XIN = -2198.688284
COEF_T = -2.629607
COEF_XBI2 = -1875.297539
COEF_XBI_XIN = -9182.903143
COEF_XBI_T = 4.021166
COEF_XIN2 = 535.330326
COEF_XIN_T = 2.025539
COEF_T2 = 0.000974

T_MIN_K = 767.0
T_MAX_K = 855.0
ZERO_THRESHOLD_J_PER_MOL = 10.0


def validate_inputs(x_bi: float, x_in: float, temperature_k: float) -> None:
    if not (0.0 <= x_bi <= 1.0):
        raise ValueError("Bi mole fraction must be between 0 and 1.")
    if not (0.0 <= x_in <= 1.0):
        raise ValueError("In mole fraction must be between 0 and 1.")
    if x_bi + x_in > 1.0 + 1e-12:
        raise ValueError("Bi and In mole fractions must sum to at most 1 (Sn = 1 − xBi − xIn).")
    if not (T_MIN_K <= temperature_k <= T_MAX_K):
        raise ValueError(f"Temperature must be between {int(T_MIN_K)} K and {int(T_MAX_K)} K.")


def predict_delta_mix_h(x_bi: float, x_in: float, temperature_k: float) -> float:
    """Return ΔmixH in J/mol using the frozen polynomial equation."""
    validate_inputs(x_bi, x_in, temperature_k)
    return _evaluate_polynomial(x_bi, x_in, temperature_k)


def _evaluate_polynomial(x_bi: float, x_in: float, temperature_k: float) -> float:
    return (
        INTERCEPT
        + COEF_XBI * x_bi
        + COEF_XIN * x_in
        + COEF_T * temperature_k
        + COEF_XBI2 * x_bi**2
        + COEF_XBI_XIN * x_bi * x_in
        + COEF_XBI_T * x_bi * temperature_k
        + COEF_XIN2 * x_in**2
        + COEF_XIN_T * x_in * temperature_k
        + COEF_T2 * temperature_k**2
    )


def predict_delta_mix_h_batch(
    x_bi: list[float], x_in: list[float], temperature_k: float
) -> list[float]:
    """Vectorized batch prediction for ternary grid points at fixed temperature."""
    return [_evaluate_polynomial(b, i, temperature_k) for b, i in zip(x_bi, x_in, strict=True)]


def mixing_type(delta_mix_h: float) -> str:
    if abs(delta_mix_h) <= ZERO_THRESHOLD_J_PER_MOL:
        return "Approximately zero"
    if delta_mix_h < 0:
        return "Exothermic"
    return "Endothermic"


def predict(x_bi: float, x_in: float, temperature_k: float) -> dict[str, float | str]:
    delta_mix_h = predict_delta_mix_h(x_bi, x_in, temperature_k)
    x_sn = 1.0 - x_bi - x_in
    return {
        "xBi": x_bi,
        "xIn": x_in,
        "xSn": x_sn,
        "temperature_K": temperature_k,
        "delta_mix_H_J_mol": delta_mix_h,
        "mixing_type": mixing_type(delta_mix_h),
    }
