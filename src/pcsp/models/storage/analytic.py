"""Analytic Schumann / Anzelius solution for a step change in inlet temperature.

Model (constant properties, no axial conduction, no losses)
-----------------------------------------------------------
With :math:`G` the mass flux [kg m^-2 s^-1], :math:`c_f` the fluid specific heat
[J kg^-1 K^-1], :math:`C_f = \\varepsilon \\rho_f c_f` and
:math:`C_s = (1-\\varepsilon) \\rho_s c_s` the fluid and solid volumetric heat
capacities [J m^-3 K^-1] and :math:`h_v` the volumetric heat-transfer
coefficient [W m^-3 K^-1]:

.. math::
    C_f \\partial_t T_f + G c_f \\partial_z T_f = h_v (T_s - T_f), \\qquad
    C_s \\partial_t T_s = h_v (T_f - T_s).

The bed starts at :math:`T_0` and the inlet steps to :math:`T_{in}` at
:math:`t = 0`. In the dimensionless variables

.. math::
    \\xi = \\frac{h_v z}{G c_f}, \\qquad
    \\eta = \\frac{h_v}{C_s}\\left(t - \\frac{C_f z}{G c_f}\\right), \\qquad
    \\theta = \\frac{T - T_0}{T_{in} - T_0},

the fluid storage term becomes a pure time delay (the residence time
:math:`C_f z / (G c_f) = \\varepsilon \\rho_f z / G`) and the system reduces to
:math:`\\partial_\\xi \\theta_f = \\theta_s - \\theta_f`,
:math:`\\partial_\\eta \\theta_s = \\theta_f - \\theta_s` (Anzelius 1926,
Z. Angew. Math. Mech. 6, 291-294; Schumann 1929, J. Franklin Inst. 208,
405-416). Its solution is written with the J-function

.. math::
    J(x, y) = 1 - e^{-y} \\int_0^x e^{-s} I_0\\big(2\\sqrt{y s}\\big)\\, ds,

as :math:`\\theta_f = J(\\xi, \\eta)` and :math:`\\theta_s = 1 - J(\\eta, \\xi)`
for :math:`\\eta > 0`, and :math:`\\theta_f = \\theta_s = 0` for :math:`\\eta < 0`
(the fluid front has not arrived yet). These forms are verified directly
against the PDE and boundary conditions in ``tests/models/storage/test_analytic.py``.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import i0e


def _j_scalar(x: float, y: float) -> float:
    if x <= 0.0:
        return 1.0
    if y <= 0.0:
        return float(np.exp(-x))

    # exp(-y - s) I0(2 sqrt(y s)) = exp(-(sqrt(y) - sqrt(s))^2) i0e(2 sqrt(y s)):
    # the scaled form never overflows, even for x, y of several thousand.
    def integrand(s: float) -> float:
        return float(np.exp(-((np.sqrt(y) - np.sqrt(s)) ** 2)) * i0e(2.0 * np.sqrt(y * s)))

    # The integrand is bounded by exp(-(sqrt(s) - sqrt(y))^2) (i0e <= 1): it
    # lives in a window around s = y and is below exp(-81) outside
    # sqrt(s) = sqrt(y) -/+ 9, so integrate over that window only.
    lo = max(np.sqrt(y) - 9.0, 0.0) ** 2
    hi = min((np.sqrt(y) + 9.0) ** 2, x)
    if hi <= lo:
        return 1.0
    breaks = [y] if lo < y < hi else None
    integral, _ = quad(integrand, lo, hi, points=breaks, limit=500, epsabs=1e-15, epsrel=1e-13)
    return 1.0 - integral


def j_function(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
    """Anzelius-Schumann J-function (see module docstring), element-wise [-]."""
    x_arr, y_arr = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    out = np.empty(x_arr.shape)
    for idx in np.ndindex(x_arr.shape):
        out[idx] = _j_scalar(float(x_arr[idx]), float(y_arr[idx]))
    return out


def schumann_step_response(
    z: np.ndarray | float,
    t: float,
    *,
    h_v: float,
    mass_flux: float,
    cp_f: float,
    fluid_capacity: float,
    solid_capacity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Dimensionless fluid and solid temperatures after an inlet step.

    Parameters
    ----------
    z : distance from the inlet [m]
    t : time since the step [s]
    h_v : volumetric fluid-solid heat-transfer coefficient [W m^-3 K^-1]
    mass_flux : superficial mass flux G [kg m^-2 s^-1]
    cp_f : fluid specific heat [J kg^-1 K^-1]
    fluid_capacity : eps * rho_f * cp_f [J m^-3 K^-1]
    solid_capacity : (1 - eps) * rho_s * cp_s [J m^-3 K^-1]

    Returns
    -------
    theta_f, theta_s : ``(T - T_0) / (T_in - T_0)`` for fluid and solid [-]
    """
    z = np.asarray(z, dtype=float)
    xi = h_v * z / (mass_flux * cp_f)
    eta = h_v * (t - fluid_capacity * z / (mass_flux * cp_f)) / solid_capacity
    arrived = eta > 0.0
    theta_f = np.where(arrived, j_function(xi, np.maximum(eta, 0.0)), 0.0)
    theta_s = np.where(arrived, 1.0 - j_function(np.maximum(eta, 0.0), xi), 0.0)
    return theta_f, theta_s
