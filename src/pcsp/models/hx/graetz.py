"""Thermal entry of a plug (slug) flow between parallel plates: the Graetz problem.

Problem
-------
A dense moving bed is a continuum with uniform velocity ``u`` [m s^-1] and
thermal diffusivity ``alpha`` [m^2 s^-1] flowing between walls at
:math:`y = \\pm b` (gap ``2b``). Axial conduction is neglected (Peclet number
:math:`u b / \\alpha \\gg 1`). Steady energy conservation is

.. math::  u \\frac{\\partial T}{\\partial z} = \\alpha \\frac{\\partial^2 T}{\\partial y^2},
           \\qquad T(y, 0) = T_0, \\qquad \\partial_y T(0, z) = 0 .

With :math:`\\eta = y/b` and the dimensionless entry length

.. math::  \\xi = \\frac{\\alpha z}{u b^2} = \\frac{16}{Gz}, \\qquad
           Gz = \\frac{D_h \\, Pe_{D_h}}{z}, \\quad Pe_{D_h} = \\frac{u D_h}{\\alpha} ,

this is the heat equation :math:`\\partial_\\xi \\theta = \\partial_\\eta^2 \\theta`
on :math:`0 \\le \\eta \\le 1`.

Nusselt-number definition (stated once, used everywhere in this package):

.. math::  Nu = \\frac{h \\, D_h}{k_{eff}}, \\qquad D_h = 4 b = 2 \\times \\text{gap},
           \\qquad h = \\frac{q_w}{T_w - T_m} ,

with :math:`T_m` the cross-section mean (= bulk, since the velocity is uniform)
temperature and :math:`q_w` the wall heat flux [W m^-2]. This is the
convention of Albrecht & Ho (2019) and Adapa et al. (2024).

Constant wall temperature (boundary condition ``"T"``)
------------------------------------------------------
With :math:`\\theta = (T - T_w)/(T_0 - T_w)` and :math:`\\theta(1, \\xi) = 0`,
separation of variables gives eigenvalues :math:`\\lambda_n = (2n-1)\\pi/2` and

.. math::  \\theta = \\sum_n \\frac{2 (-1)^{n+1}}{\\lambda_n}
                    \\cos(\\lambda_n \\eta) e^{-\\lambda_n^2 \\xi},
           \\qquad \\theta_m = \\sum_n \\frac{2}{\\lambda_n^2} e^{-\\lambda_n^2 \\xi} .

The wall flux is :math:`q_w = (k/b)(T_0 - T_w) \\sum_n 2 e^{-\\lambda_n^2 \\xi}`, so

.. math::  Nu_T(\\xi) = 4 \\frac{\\sum_n e^{-\\lambda_n^2 \\xi}}
                               {\\sum_n \\lambda_n^{-2} e^{-\\lambda_n^2 \\xi}}
           \\;\\xrightarrow{\\xi \\to \\infty}\\; 4 \\lambda_1^2 = \\pi^2 \\approx 9.8696 .

The mean coefficient over :math:`[0, z]` is the arithmetic mean of the local
one. From the bulk energy balance :math:`\\rho c u b \\, dT_m/dz = -h (T_m - T_w)`,

.. math::  \\overline{Nu}_T(\\xi) = \\frac{1}{\\xi}\\int_0^\\xi Nu_T \\, d\\xi'
                                = -\\frac{4}{\\xi} \\ln \\theta_m(\\xi) .

Constant wall heat flux (boundary condition ``"H"``)
----------------------------------------------------
With :math:`\\theta = k (T - T_0) / (q_w b)` and :math:`\\partial_\\eta \\theta(1) = 1`,

.. math::  \\theta = \\xi + \\frac{\\eta^2}{2} - \\frac16
                    - \\sum_{n \\ge 1} \\frac{2(-1)^n}{n^2\\pi^2}
                      \\cos(n\\pi\\eta) e^{-n^2\\pi^2\\xi} ,

(the series coefficients are the cosine-Fourier coefficients of
:math:`\\eta^2/2 - 1/6`, so :math:`\\theta(\\eta, 0) = 0`). Then
:math:`\\theta_m = \\xi` and the wall-to-mean difference is

.. math::  \\Delta(\\xi) = \\theta_w - \\theta_m
           = \\frac13 - \\sum_n \\frac{2}{n^2\\pi^2} e^{-n^2\\pi^2\\xi},
           \\qquad Nu_H = \\frac{4}{\\Delta(\\xi)} \\;\\xrightarrow{\\xi\\to\\infty}\\; 12 .

The mean Nusselt number uses the length-averaged temperature difference (the
natural mean at fixed flux, as in Adapa et al. 2024, eq. 8-9):

.. math::  \\overline{Nu}_H(\\xi) = \\frac{4}{\\bar\\Delta}, \\qquad
           \\bar\\Delta = \\frac13 - \\sum_n \\frac{2}{n^4\\pi^4 \\xi}
                         \\left(1 - e^{-n^2\\pi^2\\xi}\\right) .

Short entry lengths
-------------------
For :math:`\\xi < 0.01` the series needs many terms. There the walls do not
yet see each other and the semi-infinite-solid (penetration) solution is used:
:math:`\\theta_m = 1 - 2\\sqrt{\\xi/\\pi}` (``"T"``) and
:math:`\\Delta = 2\\sqrt{\\xi/\\pi} - \\xi` (``"H"``). Their error relative to the
exact slab solution comes from the image source at distance ``2b`` and is of
order :math:`e^{-1/\\xi} < 10^{-43}` at the switch point. For :math:`\\xi \\ge 0.01`
the series is truncated where :math:`\\lambda_n^2 \\xi > 60`.

Near-wall resistance
--------------------
A contact or gas-film resistance ``R_w`` [m^2 K W^-1] between wall and bed acts
in series: :math:`1/h_{tot} = 1/h + R_w`. For the ``"H"`` condition this is
exact (Adapa et al. 2024, eq. 2-4); for ``"T"`` it is the usual approximation
(the exact condition would be a Robin condition with its own eigenvalues).

References
----------
* Albrecht, K.J., Ho, C.K., 2019. Design and operating considerations for a
  shell-and-plate, moving packed-bed, particle-to-sCO2 heat exchanger.
  Solar Energy 178, 331-340 (Fig. 2 quotes the asymptotes 9.87 and 12).
* Adapa, S.R. et al., 2024. Heat transfer coefficients of moving particle beds
  from flow-dependent particle bed thermal conductivity and near-wall
  resistance. Solar Energy 282, 112960; arXiv:2403.19892, eq. 1-11.
* Muzychka, Y.S., Walsh, E., Walsh, P., 2010. Simple models for laminar
  thermally developing slug flow in noncircular ducts and channels.
  J. Heat Transfer 132, 111702.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

WallCondition = Literal["T", "H"]

NU_FULLY_DEVELOPED = {"T": np.pi**2, "H": 12.0}
"""Fully developed slug-flow Nusselt numbers (D_h = 4b), derived above."""

XI_SHORT = 0.01
"""Below this dimensionless length the penetration solution is used."""

_EXPONENT_CUTOFF = 60.0


def _check_bc(bc: str) -> None:
    if bc not in NU_FULLY_DEVELOPED:
        raise ValueError(f"wall condition must be 'T' or 'H', got {bc!r}")


def _eigenvalues(bc: WallCondition, xi_min: float) -> np.ndarray:
    """Eigenvalues lambda_n with lambda_n^2 xi_min <= cutoff (at least 3 terms)."""
    n_max = max(3, int(np.ceil(np.sqrt(_EXPONENT_CUTOFF / xi_min) / np.pi)) + 2)
    n = np.arange(1, n_max + 1, dtype=float)
    return (2.0 * n - 1.0) * np.pi / 2.0 if bc == "T" else n * np.pi


def _series(xi: np.ndarray, bc: WallCondition, kind: str) -> np.ndarray:
    """Series sums for xi >= XI_SHORT, vectorised over xi."""
    lam = _eigenvalues(bc, float(np.min(xi)))[:, None]
    decay = np.exp(-(lam**2) * xi[None, :])
    if bc == "T":
        if kind == "theta_m":
            return np.sum(2.0 / lam**2 * decay, axis=0)
    else:
        if kind == "delta":
            return 1.0 / 3.0 - np.sum(2.0 / lam**2 * decay, axis=0)
        if kind == "delta_mean":
            # sum_n 2 / (n^4 pi^4 xi) = 1 / (45 xi) exactly (sum 1/n^4 = pi^4/90); only the
            # decaying part needs the truncated series.
            return (
                1.0 / 3.0 - 1.0 / (45.0 * xi) + np.sum(2.0 / (lam**4 * xi[None, :]) * decay, axis=0)
            )
    raise ValueError(kind)


def _log_theta_m_series(xi: np.ndarray) -> np.ndarray:
    """ln(theta_m) for constant wall temperature, free of underflow at large xi."""
    lam = _eigenvalues("T", float(np.min(xi)))[:, None]
    lam1 = lam[0]
    ratio = np.sum((lam1 / lam) ** 2 * np.exp(-(lam**2 - lam1**2) * xi[None, :]), axis=0)
    return np.log(2.0 / lam1**2) - lam1**2 * xi + np.log(ratio)


def _log_mean_temperature(xi: np.ndarray) -> np.ndarray:
    """ln(theta_m) for xi >= 0 (``"T"`` condition)."""
    out = np.zeros_like(xi)
    short = (xi > 0.0) & (xi < XI_SHORT)
    out[short] = np.log(1.0 - 2.0 * np.sqrt(xi[short] / np.pi))
    long = xi >= XI_SHORT
    if np.any(long):
        out[long] = _log_theta_m_series(xi[long])
    return out


def _as_positive_array(xi: np.ndarray | float) -> np.ndarray:
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    if np.any(xi <= 0.0):
        raise ValueError("dimensionless length xi must be positive")
    return xi


def mean_temperature(xi: np.ndarray | float) -> np.ndarray:
    """Constant wall temperature: :math:`\\theta_m = (T_m - T_w)/(T_0 - T_w)` [-].

    ``xi = 0`` is allowed and returns 1.
    """
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    out = np.ones_like(xi)
    short = (xi > 0.0) & (xi < XI_SHORT)
    out[short] = 1.0 - 2.0 * np.sqrt(xi[short] / np.pi)
    long = xi >= XI_SHORT
    if np.any(long):
        out[long] = _series(xi[long], "T", "theta_m")
    return out


def wall_excess(xi: np.ndarray | float) -> np.ndarray:
    """Constant wall flux: :math:`\\Delta = k (T_w - T_m) / (q_w b)` [-]. ``xi = 0`` gives 0."""
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    out = np.zeros_like(xi)
    short = (xi > 0.0) & (xi < XI_SHORT)
    out[short] = 2.0 * np.sqrt(xi[short] / np.pi) - xi[short]
    long = xi >= XI_SHORT
    if np.any(long):
        out[long] = _series(xi[long], "H", "delta")
    return out


def nusselt_local(xi: np.ndarray | float, bc: WallCondition = "T") -> np.ndarray:
    """Local Nusselt number :math:`Nu = h D_h / k_{eff}` at dimensionless length ``xi`` [-]."""
    _check_bc(bc)
    xi = _as_positive_array(xi)
    if bc == "H":
        return 4.0 / wall_excess(xi)
    out = np.empty_like(xi)
    short = xi < XI_SHORT
    # Short: -d(theta_m)/dxi = 1/sqrt(pi xi); Nu = 4 (-d theta_m/d xi) / theta_m.
    out[short] = 4.0 / (np.sqrt(np.pi * xi[short]) * mean_temperature(xi[short]))
    long = ~short
    if np.any(long):
        # Both sums scaled by exp(lambda_1^2 xi) to avoid underflow at large xi.
        x = xi[long]
        lam = _eigenvalues("T", float(np.min(x)))[:, None]
        scaled = np.exp(-(lam**2 - lam[0] ** 2) * x[None, :])
        out[long] = 4.0 * np.sum(scaled, axis=0) / np.sum(scaled / lam**2, axis=0)
    return out


def nusselt_mean(xi: np.ndarray | float, bc: WallCondition = "T") -> np.ndarray:
    """Mean Nusselt number over the entry length ``[0, xi]`` [-] (definitions in module doc)."""
    _check_bc(bc)
    xi = _as_positive_array(xi)
    if bc == "T":
        return -4.0 / xi * _log_mean_temperature(xi)
    delta_mean = np.empty_like(xi)
    short = xi < XI_SHORT
    delta_mean[short] = 4.0 / 3.0 * np.sqrt(xi[short] / np.pi) - xi[short] / 2.0
    long = ~short
    if np.any(long):
        delta_mean[long] = _series(xi[long], "H", "delta_mean")
    return 4.0 / delta_mean


def nusselt_cell(
    xi_lo: np.ndarray | float, xi_hi: np.ndarray | float, bc: WallCondition = "T"
) -> np.ndarray:
    """Nusselt number averaged over the interval ``[xi_lo, xi_hi]`` [-].

    Consistent with :func:`nusselt_mean`: for ``"T"`` the local coefficient is
    averaged arithmetically,
    :math:`(\\xi_2 \\overline{Nu}_2 - \\xi_1 \\overline{Nu}_1)/(\\xi_2 - \\xi_1)`;
    for ``"H"`` the local *resistance* is averaged,
    :math:`(\\xi_2/\\overline{Nu}_2 - \\xi_1/\\overline{Nu}_1)/(\\xi_2 - \\xi_1)`. The
    integrable :math:`\\xi^{-1/2}` singularity at the inlet is therefore handled
    exactly; ``xi_lo = 0`` is allowed.
    """
    _check_bc(bc)
    xi_lo = np.atleast_1d(np.asarray(xi_lo, dtype=float))
    xi_hi = np.atleast_1d(np.asarray(xi_hi, dtype=float))
    if np.any(xi_lo < 0.0) or np.any(xi_hi <= xi_lo):
        raise ValueError("need 0 <= xi_lo < xi_hi")

    def integral(xi: np.ndarray) -> np.ndarray:
        # T: int_0^xi Nu dxi' = -4 ln theta_m.  H: int_0^xi (1/Nu) dxi' = xi * Delta_mean / 4.
        if bc == "T":
            return -4.0 * _log_mean_temperature(xi)
        out = np.zeros_like(xi)
        pos = xi > 0.0
        if np.any(pos):
            out[pos] = xi[pos] / nusselt_mean(xi[pos], "H")
        return out

    span = xi_hi - xi_lo
    avg = (integral(xi_hi) - integral(xi_lo)) / span
    return avg if bc == "T" else 1.0 / avg
