"""Steady two-dimensional reference solution of the moving-bed heat exchanger.

The reduced model (:mod:`pcsp.models.hx.model`) replaces conduction across
the particle channel by a Nusselt number from the Graetz problem with a
*uniform* wall condition. This module solves the conjugate problem without
that approximation, as formulated by Albrecht & Ho (2019, eq. 9-10, 16-17),
and is used to check the reduced model.

Equations
---------
``x`` runs down from the particle inlet, ``y`` from the wall (``y = 0``) to
the channel centre (``y = b``, half the gap). The bed moves as a plug with
velocity ``u``:

.. math::
    \\rho_b c_p(T) \\, u \\, \\frac{\\partial T}{\\partial x}
      = \\frac{\\partial}{\\partial y}\\left(k_{eff}(T) \\frac{\\partial T}{\\partial y}\\right),
    \\qquad \\left.\\frac{\\partial T}{\\partial y}\\right|_{y=b} = 0 ,

with the wall condition

.. math::
    q''(x) = k_{eff} \\left.\\frac{\\partial T}{\\partial y}\\right|_{y=0}
           = \\frac{T(x, 0) - T_{CO_2}(x)}{R_w + t/k_m + r_{CO_2}} ,

where ``R_w`` is the near-wall resistance of the closure (at the mean bed
temperature of the cross-section), ``t / k_m`` the plate and ``r_CO2`` the
sCO2 resistance of :func:`pcsp.models.hx.model.co2_side`. The sCO2 flows up
in counterflow:

.. math::
    \\dot m_{CO_2} \\, c_{p,CO_2} \\, \\frac{dT_{CO_2}}{dx} = -2 W q'' .

Numerical method
----------------
Finite volumes in ``y`` (harmonic-mean face conductivity, wall flux through
half a cell in series with the wall resistances), stiff integration in ``x``
(BDF), and shooting on the sCO2 outlet temperature at ``x = 0`` so that the
sCO2 inlet temperature at ``x = H`` is met.

Like the reduced model, this neglects conduction along ``x``, heat loss and
maldistribution, and assumes plug flow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from pcsp.materials import sco2
from pcsp.models.hx.model import (
    SCO2_TABLE_T_MAX,
    SCO2_TABLE_T_MIN,
    MovingBedHXParams,
    co2_side,
)


@dataclass(frozen=True)
class ConjugateSolution:
    """Profiles along ``x`` and scalar results of :func:`solve_conjugate`.

    Attributes
    ----------
    x : depth below the particle inlet [m], shape ``(n,)``
    T_particle : cross-section mean bed temperature [K]
    T_co2 : sCO2 temperature [K]
    T_plate : plate surface temperature on the particle side [K]
    heat_flux : wall heat flux [W m^-2]
    h_particle_wall : ``q'' / (T_particle - T_plate)`` [W m^-2 K^-1]; includes
        the near-wall resistance, as in Albrecht & Ho's eq. 5
    nu_bulk : Nusselt number of the bed continuum alone,
        ``q'' D_h / (k_eff (T_particle - T(x, 0)))`` with ``D_h`` twice the gap [-]
    duty : heat transferred per unit cell [W]
    U_mean : ``duty / (A * LMTD)`` [W m^-2 K^-1]
    h_particle_wall_mean : mean of ``h_particle_wall`` over the height [W m^-2 K^-1]
    """

    x: np.ndarray
    T_particle: np.ndarray
    T_co2: np.ndarray
    T_plate: np.ndarray
    heat_flux: np.ndarray
    h_particle_wall: np.ndarray
    nu_bulk: np.ndarray
    duty: float
    U_mean: float
    h_particle_wall_mean: float

    @property
    def T_particle_out(self) -> float:
        return float(self.T_particle[-1])

    @property
    def T_co2_out(self) -> float:
        return float(self.T_co2[0])


def solve_conjugate(
    params: MovingBedHXParams, n_y: int = 40, n_out: int = 201, rtol: float = 1e-8
) -> ConjugateSolution:
    """Solve the 2-D conjugate problem for the parameters of the reduced model.

    Parameters
    ----------
    params : the same inputs as :func:`pcsp.models.hx.simulate` (``wall_bc`` and
        ``n_cells`` are not used)
    n_y : finite volumes across the half gap
    n_out : output points along ``x``
    rtol : relative tolerance of the integration along ``x``
    """
    if n_y < 2 or n_out < 2:
        raise ValueError("need n_y >= 2 and n_out >= 2")
    p = params
    part, closure = p.particle, p.closure
    co2 = sco2.isobar(p.p_co2, SCO2_TABLE_T_MIN, SCO2_TABLE_T_MAX)
    u = p.particle_velocity
    dy = 0.5 * p.channel_gap / n_y
    r_metal = p.wall_thickness / p.wall_conductivity
    T_hot_in = p.T_particle_in

    def wall(T: np.ndarray, T_c: float) -> tuple[float, float, float]:
        """Wall heat flux and the resistances behind the bed and behind the plate surface."""
        r_fluid = r_metal + float(co2_side(p, T_c)[2])
        r_near = float(closure.r_wall(np.mean(T), u, part))
        r_half_cell = 0.5 * dy / float(closure.k_eff(T[0], u, part))
        return (T[0] - T_c) / (r_near + r_fluid + r_half_cell), r_near, r_fluid

    def rhs(_x: float, state: np.ndarray) -> np.ndarray:
        T, T_c = state[:-1], float(state[-1])
        k = closure.k_eff(T, u, part)
        k_face = 2.0 * k[:-1] * k[1:] / (k[:-1] + k[1:])
        towards_wall = k_face * (T[1:] - T[:-1]) / dy
        q = wall(T, T_c)[0]
        net = np.empty_like(T)
        net[0] = towards_wall[0] - q
        net[1:-1] = towards_wall[1:] - towards_wall[:-1]
        net[-1] = -towards_wall[-1]
        dT = net / (dy * part.rho_bulk * part.cp(T) * u)
        dT_c = -2.0 * p.width * q / (p.m_dot_co2 * float(co2.cp(T_c)))
        return np.concatenate([dT, [dT_c]])

    def march(T_c_out: float):
        start = np.concatenate([np.full(n_y, T_hot_in), [T_c_out]])
        return solve_ivp(
            rhs, (0.0, p.height), start, method="BDF", rtol=rtol, atol=1e-8, dense_output=True
        )

    def miss(T_c_out: float) -> float:
        return float(march(T_c_out).y[-1, -1]) - p.T_co2_in

    span = T_hot_in - p.T_co2_in
    T_c_out = brentq(miss, p.T_co2_in + 1e-6 * span, T_hot_in - 1e-6 * span, xtol=1e-8)
    solution = march(T_c_out)

    x = np.linspace(0.0, p.height, n_out)
    states = solution.sol(x)
    rows = []
    for i in range(n_out):
        T, T_c = states[:-1, i], float(states[-1, i])
        q, r_near, r_fluid = wall(T, T_c)
        T_mean = float(np.mean(T))
        T_plate = T_c + q * r_fluid
        T_bed_wall = T_plate + q * r_near
        k_mean = float(closure.k_eff(T_mean, u, part))
        nu = q * 2.0 * p.channel_gap / (k_mean * (T_mean - T_bed_wall))
        rows.append((T_mean, T_c, T_plate, q, q / (T_mean - T_plate), nu))
    T_mean, T_c, T_plate, q, h_pw, nu = np.array(rows).T

    duty = p.m_dot_particle * (part.enthalpy(T_hot_in) - part.enthalpy(T_mean[-1]))
    dT_top, dT_bottom = T_hot_in - T_c[0], T_mean[-1] - T_c[-1]
    lmtd = (dT_top - dT_bottom) / np.log(dT_top / dT_bottom)
    return ConjugateSolution(
        x=x,
        T_particle=T_mean,
        T_co2=T_c,
        T_plate=T_plate,
        heat_flux=q,
        h_particle_wall=h_pw,
        nu_bulk=nu,
        duty=float(duty),
        U_mean=float(duty / (p.heat_transfer_area * lmtd)),
        h_particle_wall_mean=float(np.sum(0.5 * (h_pw[1:] + h_pw[:-1]) * np.diff(x)) / p.height),
    )
