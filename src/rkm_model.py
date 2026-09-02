"""
Redlich-Kister-Muggianu mixing enthalpy for liquid Bi-In-Sn.

Paper: Kumar, Mohan and Behera, J. Electron. Mater. 48 (2019) 8096-8106.
Equation (4) and Table IV. Units: J/mol.

    ΔmixH = sum of three binary Redlich-Kister terms
          + ternary term

Binary term for pair (i, j):
    x_i * x_j * sum_n L_n * (x_i - x_j)**n

Ternary term, with (i, j, k) = (In, Bi, Sn):
    x_In * x_Bi * x_Sn * (L0 * x_In + L1 * x_Bi + L2 * x_Sn)

Table IV names the ternary parameter L_Bi-Sn-In. Using
(i, j, k) = (Bi, Sn, In) does not match Figure 10. The order
(In, Bi, Sn) does. See docs/rkm_model.md.

Table IV has no temperature dependence. The temperature argument
is accepted but not used.
"""

import numpy as np

# Table IV, J/mol. Signs checked on the printed table.
L_BI_IN = np.array([-6617.0, 133.0, 2066.0])
L_BI_SN = np.array([520.0, 235.0, -157.0, -119.0])
L_IN_SN = np.array([-1980.0, -893.0])
L_TERNARY = np.array([-3874.0, -12777.0, 24187.0])  # L0, L1, L2


def _binary_term(x_i, x_j, L):
    """x_i x_j * polynomial in (x_i - x_j)."""
    delta = x_i - x_j
    poly = 0.0
    for n, L_n in enumerate(L):
        poly = poly + L_n * delta**n
    return x_i * x_j * poly


def rkm_delta_mix_h(x_bi, x_in, x_sn, temperature=None, composition_tol=1e-8):
    """
    Integral molar enthalpy of mixing (J/mol).

    x_bi, x_in, x_sn must be >= 0 and sum to 1.
    Arrays are allowed. temperature is ignored.
    """
    x_bi = np.asarray(x_bi, dtype=float)
    x_in = np.asarray(x_in, dtype=float)
    x_sn = np.asarray(x_sn, dtype=float)
    x_bi, x_in, x_sn = np.broadcast_arrays(x_bi, x_in, x_sn)

    if np.any(x_bi < -composition_tol) or np.any(x_in < -composition_tol) or np.any(x_sn < -composition_tol):
        raise ValueError("Mole fractions must be >= 0.")
    if np.any(x_bi > 1.0 + composition_tol) or np.any(x_in > 1.0 + composition_tol) or np.any(x_sn > 1.0 + composition_tol):
        raise ValueError("Mole fractions must be <= 1.")
    if np.any(np.abs(x_bi + x_in + x_sn - 1.0) > composition_tol):
        raise ValueError("Mole fractions must satisfy x_bi + x_in + x_sn = 1.")

    binary = (
        _binary_term(x_bi, x_in, L_BI_IN)
        + _binary_term(x_bi, x_sn, L_BI_SN)
        + _binary_term(x_in, x_sn, L_IN_SN)
    )

    # (i, j, k) = (In, Bi, Sn)
    ternary = x_in * x_bi * x_sn * (
        L_TERNARY[0] * x_in + L_TERNARY[1] * x_bi + L_TERNARY[2] * x_sn
    )

    mixing_enthalpy = binary + ternary
    _ = temperature

    if np.ndim(mixing_enthalpy) == 0:
        return float(mixing_enthalpy)
    return mixing_enthalpy
