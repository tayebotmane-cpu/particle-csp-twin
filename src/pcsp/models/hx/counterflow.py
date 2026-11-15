"""Steady 1-D counterflow heat exchanger solved as enthalpy balances.

The hot stream enters at face 0 (``z = 0``) and flows towards face ``N``;
the cold stream enters at face ``N`` and flows towards face 0. On each of the
``N`` cells ``[z_i, z_{i+1}]`` the duty is

.. math::  Q_i = (UA)_i \\, \\frac{\\Delta T_i + \\Delta T_{i+1}}{2},
           \\qquad \\Delta T = T_h - T_c ,

(trapezoidal rule, second-order accurate in the cell size) and each stream's
enthalpy changes by exactly that duty:

.. math::
    \\dot m_h \\left[h_h(T_{h,i}) - h_h(T_{h,i+1})\\right] = Q_i, \\qquad
    \\dot m_c \\left[h_c(T_{c,i}) - h_c(T_{c,i+1})\\right] = Q_i .

Summing over cells shows that the two streams exchange the same total duty
to the solver tolerance, whatever the property functions or ``UA`` profile.
The ``2N`` equations for the ``2N`` unknown face temperatures are solved by
Newton's method with a sparse Jacobian. ``UA`` is re-evaluated from the
current temperatures at every iteration but its temperature derivative is
left out of the Jacobian, so the iteration converges linearly in the (weak)
dependence of ``UA`` on temperature and quadratically otherwise.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

Property = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class CounterflowSolution:
    """Converged face temperatures [K], cell duties [W] and cell conductances [W K^-1]."""

    T_hot: np.ndarray
    T_cold: np.ndarray
    Q_cells: np.ndarray
    UA_cells: np.ndarray
    iterations: int


def solve_counterflow(
    n_cells: int,
    m_hot: float,
    m_cold: float,
    h_hot: Property,
    cp_hot: Property,
    h_cold: Property,
    cp_cold: Property,
    T_hot_in: float,
    T_cold_in: float,
    ua_cells: Callable[[np.ndarray, np.ndarray], np.ndarray],
    rtol: float = 1e-12,
    max_iter: int = 100,
) -> CounterflowSolution:
    """Solve the discrete counterflow balances.

    Parameters
    ----------
    n_cells : number of cells ``N``
    m_hot, m_cold : mass flow rates [kg s^-1]
    h_hot, h_cold : specific enthalpy functions of temperature [J kg^-1]
    cp_hot, cp_cold : their derivatives [J kg^-1 K^-1]
    T_hot_in, T_cold_in : inlet temperatures [K] (hot at face 0, cold at face N)
    ua_cells : ``(T_hot_faces, T_cold_faces) -> UA`` per cell [W K^-1], shape ``(N,)``
    rtol : convergence tolerance on the residual relative to the maximum duty
    max_iter : iteration limit (``RuntimeError`` if exceeded)
    """
    if T_hot_in <= T_cold_in:
        raise ValueError("hot inlet must be hotter than cold inlet")
    N = n_cells
    dT_in = T_hot_in - T_cold_in
    q_scale = min(
        m_hot * (h_hot(np.array([T_hot_in])) - h_hot(np.array([T_cold_in])))[0],
        m_cold * (h_cold(np.array([T_hot_in])) - h_cold(np.array([T_cold_in])))[0],
    )
    s = np.linspace(0.0, 1.0, N + 1)
    T_h = T_hot_in - 0.5 * dT_in * s
    T_c = T_cold_in + 0.5 * dT_in * (1.0 - s)

    cells = np.arange(N)
    for iteration in range(1, max_iter + 1):
        UA = ua_cells(T_h, T_c)
        dT = T_h - T_c
        Q = 0.5 * UA * (dT[:-1] + dT[1:])
        hh = h_hot(T_h)
        hc = h_cold(T_c)
        r = np.concatenate([m_hot * (hh[:-1] - hh[1:]) - Q, m_cold * (hc[:-1] - hc[1:]) - Q])
        J = _jacobian(N, m_hot * cp_hot(T_h), m_cold * cp_cold(T_c), UA, cells)
        step = spsolve(J, -r)
        T_h[1:] += step[:N]
        T_c[:-1] += step[N:]
        np.clip(T_h, T_cold_in, T_hot_in, out=T_h)
        np.clip(T_c, T_cold_in, T_hot_in, out=T_c)
        if np.max(np.abs(r)) < rtol * q_scale and np.max(np.abs(step)) < 1e-9 * dT_in:
            UA = ua_cells(T_h, T_c)
            dT = T_h - T_c
            Q = 0.5 * UA * (dT[:-1] + dT[1:])
            return CounterflowSolution(T_h, T_c, Q, UA, iteration)
    raise RuntimeError(f"counterflow solver did not converge in {max_iter} iterations")


def _jacobian(
    N: int, mcp_h: np.ndarray, mcp_c: np.ndarray, UA: np.ndarray, cells: np.ndarray
) -> coo_matrix:
    """Sparse Jacobian of the residuals with UA frozen.

    Unknown ordering: ``T_hot[1..N]`` -> columns ``0..N-1``,
    ``T_cold[0..N-1]`` -> columns ``N..2N-1``. Rows: hot balances ``0..N-1``,
    cold balances ``N..2N-1``.
    """
    half = 0.5 * UA
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    def add(r: np.ndarray, c: np.ndarray, v: np.ndarray, mask: np.ndarray) -> None:
        rows.append(r[mask])
        cols.append(c[mask])
        vals.append(v[mask])

    i = cells
    has_hot_i = i >= 1  # T_hot[i] is an unknown only for i >= 1
    has_cold_next = i + 1 <= N - 1  # T_cold[i+1] is an unknown only for i+1 <= N-1
    all_cells = np.ones(N, dtype=bool)
    # Hot balance: m_h (h(Th_i) - h(Th_i+1)) - UA/2 (Th_i - Tc_i + Th_i+1 - Tc_i+1)
    add(i, i - 1, mcp_h[i] - half, has_hot_i)
    add(i, i, -mcp_h[i + 1] - half, all_cells)
    add(i, N + i, half, all_cells)
    add(i, N + i + 1, half, has_cold_next)
    # Cold balance: m_c (h(Tc_i) - h(Tc_i+1)) - UA/2 (...)
    add(N + i, i - 1, -half, has_hot_i)
    add(N + i, i, -half, all_cells)
    add(N + i, N + i, mcp_c[i] + half, all_cells)
    add(N + i, N + i + 1, -mcp_c[i + 1] + half, has_cold_next)
    return coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * N, 2 * N)
    ).tocsc()


def effectiveness_ntu_counterflow(ntu: float, c_ratio: float) -> float:
    """Analytic counterflow effectiveness for constant properties and U [-].

    .. math::
        \\varepsilon = \\frac{1 - e^{-NTU(1-C_r)}}{1 - C_r e^{-NTU(1-C_r)}}
        \\quad (C_r < 1), \\qquad \\varepsilon = \\frac{NTU}{1 + NTU} \\quad (C_r = 1),

    with :math:`NTU = UA / C_{min}` and :math:`C_r = C_{min}/C_{max}`
    (Kays & London, Compact Heat Exchangers; Incropera et al., Table 11.3).
    """
    if not 0.0 <= c_ratio <= 1.0:
        raise ValueError("c_ratio must be in [0, 1]")
    if np.isclose(c_ratio, 1.0, rtol=0.0, atol=1e-12):
        return ntu / (1.0 + ntu)
    e = np.exp(-ntu * (1.0 - c_ratio))
    return float((1.0 - e) / (1.0 - c_ratio * e))
