"""Transient (control-oriented) model of the moving packed-bed particle-to-sCO2 heat exchanger.

The steady models of this package (:mod:`.model`, :mod:`.conjugate`) give
outlet temperatures for fixed inputs. This module adds time: the bed, the
plate and their heat capacities, so that the response of the outlet
temperatures to steps in the inlet conditions, and to the two manipulated
inputs of a controller (particle mass flow, sCO2 bypass), can be simulated,
linearised and optimised. It implements
:class:`~pcsp.core.interfaces.ControlModel` and is written once for NumPy and
CasADi (:mod:`pcsp.core.backend`).

Geometry
--------
The unit cell is that of :class:`~pcsp.models.hx.model.MovingBedHXParams`:
one particle channel of gap ``2b`` and width ``W`` between two plate faces,
and the sCO2 flow of one plate. By symmetry only one half channel
(``0 <= y <= b``, wall at ``y = 0``) and one plate face are resolved; every
extensive result (mass flows, duty, stored energy) is reported per unit cell,
i.e. twice the half-cell value. ``z`` runs down from the particle inlet
(``z = 0``) to the outlet (``z = H``); the sCO2 enters at ``z = H`` and leaves
at ``z = 0`` (counterflow).

States and equations
--------------------
``n_z`` cells along ``z`` (``dz = H / n_z``) and ``n_y`` cells across the
half gap (``dy = b / n_y``). The states are the bed temperatures
:math:`T_{ij}` (``i`` along ``z``, ``j = 0`` at the wall) and the plate
temperatures :math:`T_{w,i}`, ``n_z (n_y + 1)`` in total.

*Bed* (plug flow at uniform velocity, conduction across the channel only):

.. math::
    \\rho_b c_p(T_{ij})\\,\\Delta V \\,\\dot T_{ij}
      = \\frac{\\dot m_p}{2 n_y}\\,\\big[h_p(T_{i-1,j}) - h_p(T_{ij})\\big]
        + Q_{y,j-1/2} - Q_{y,j+1/2} ,
    \\qquad \\Delta V = W\\,dz\\,dy ,

with first-order upwind enthalpy transport (``T_{-1,j}`` is the particle
inlet temperature), :math:`Q_{y,j+1/2} = k_{j+1/2} W dz (T_{ij} -
T_{i,j+1})/dy` towards the centre, :math:`k_{j+1/2}` the harmonic mean of
:math:`k_{eff}` of the two cells, zero flux at the centre ``y = b``, and at
the wall

.. math::
    q''_i = \\frac{T_{i0} - T_{w,i}}
               {dy / (2 k_{eff}(T_{i0})) + R_w(\\bar T_i) + t / (2 k_m)} ,

where :math:`R_w` is the near-wall resistance of the closure at the
cross-section mean temperature :math:`\\bar T_i` (as in
:mod:`.conjugate`), and ``t / (2 k_m)`` is the conduction resistance of half
the plate.

*Plate* (one node across its thickness, at mid-plane; no conduction along ``z``):

.. math::
    C_w W dz\\,\\dot T_{w,i} = q''_i W dz - \\dot Q_{CO_2,i} ,

with :math:`C_w = \\rho_w c_w t_w` [J m^-2 K^-1] per unit wall area.

*sCO2* (quasi-steady; see below). In cell ``i`` the plate temperature is
uniform, so with properties at the cell's sCO2 inlet temperature the exact
solution of :math:`(\\dot m_{c}/2) c_{p,c} dT_c = (T_w - T_c) W dz' / R_c` is

.. math::
    T_{c,i}^{out} = T_{w,i} - (T_{w,i} - T_{c,i}^{in})
       \\exp\\!\\left(-\\frac{W dz}{R_c \\, (\\dot m_{c}/2)\\, c_{p,c}(T_{c,i}^{in})}\\right),
    \\qquad R_c = \\frac{t}{2 k_m} + r_{CO_2}(T_{c,i}^{in}) ,

marched from the sCO2 inlet (``i = n_z - 1``) to the outlet (``i = 0``),
and the heat taken from the plate is defined as the enthalpy rise
:math:`\\dot Q_{CO_2,i} = (\\dot m_c/2)\\,[h_c(T^{out}_{c,i}) - h_c(T^{in}_{c,i})]`.
:math:`\\dot m_c = (1 - \\beta) \\dot m_{CO_2}` is the flow through the
exchanger, ``beta`` the bypass fraction.

Because every heat flow leaves one control volume exactly as it enters
another, and the advection is written in enthalpy, the semi-discrete model
conserves energy to round-off:

.. math::
    \\frac{d}{dt}\\Big[2 \\sum \\rho_b \\Delta V h_p(T_{ij})
        + 2 \\sum C_w W dz\\, T_{w,i}\\Big]
    = \\dot m_p \\big[h_p(T_{p,in}) - \\overline{h_p}(T_{n_z-1,\\cdot})\\big]
      - \\dot m_c \\big[h_c(T_{c,0}^{out}) - h_c(T_{CO_2,in})\\big] ,

(:meth:`MovingBedHXDynamic.stored_energy`,
:meth:`MovingBedHXDynamic.energy_rates`). In steady state the model is a
discretisation of the 2-D conjugate problem of :mod:`.conjugate` and
converges to it as the grid is refined (first order in ``dz``).

Outputs
-------
* particle outlet temperature: the mixing-cup temperature of the bottom row
  (equal flow in every column), :math:`h_p(T_{p,out}) = \\overline{h_p}(T_{n_z-1,j})`;
* mixed sCO2 temperature after the bypass rejoins:
  :math:`\\dot m_{CO_2} h_c(T_{mix}) = \\dot m_c h_c(T_{c,0}^{out}) +
  \\beta \\dot m_{CO_2} h_c(T_{CO_2,in})` (an enthalpy balance; Fang et al.
  write it with constant :math:`c_p`, their eq. 10b).

Both inversions of ``h`` use Newton steps (three for sCO2, two for the
particles) from a linear first guess; the residual is below 1e-9 K over the
fitted range (checked in the tests).

Assumptions
-----------
* **Quasi-steady sCO2.** Its residence time is ``H / u_CO2``, about 2 s at
  0.5 m/s, and its heat capacity per unit wall area,
  :math:`\\rho_c c_{p,c} s / 2 \\approx 90` J m^-2 K^-1 for a 1 mm slot,
  is about 2 % of the plate's (4400) and under 1 % of plate plus bed. The
  slowest processes it couples to have time constants of 100-1000 s, so the
  neglected storage shifts them by about 1 %.
* **Plate.** Lumped across its thickness (its half-thickness resistance
  :math:`t / (2 k_m) \\approx 2\\times10^{-5}` m^2 K/W is under 1 % of the
  series resistance, about 9e-3), no conduction along ``z`` (in the Fang
  et al. base case the plate carries :math:`k_m t\\, dT/dz \\approx 23 \\times
  10^{-3} \\times 200` = 5 W per metre of width along ``z``, against a duty of
  about 13 kW per metre of width). Default :math:`C_w = 8000 \\times 550
  \\times t` J m^-2 K^-1, i.e. :data:`PLATE_DENSITY` and
  :data:`PLATE_HEAT_CAPACITY` - **assumptions** for a stainless or nickel
  alloy near 900 K, not taken from a source.
* **Bed.** Plug flow at a uniform velocity, no axial conduction (the axial
  Peclet number :math:`u H / \\alpha` is about 2e4), no heat loss. The
  near-wall layer is a resistance without heat capacity, and it does not
  take area from the flow (as in :mod:`.conjugate`); Fang et al. make it a
  stagnant layer ``d_p/2`` thick, which for ``d_p = 250`` um is 2.5 % of
  the half gap.
* **Closures as polynomials.** :math:`k_{eff}(T)` and :math:`R_w(T)` of the
  chosen :class:`~pcsp.models.hx.closures.BedClosure` are evaluated (with
  NumPy) at construction, at the nominal bed velocity, and fitted by
  polynomials in temperature over ``T_range``, so that ``rhs`` stays
  CasADi-compatible. All closures in :mod:`.closures` are independent of the
  velocity; a velocity-dependent closure would be frozen at the nominal
  velocity. The fit errors are in :attr:`MovingBedHXDynamic.fit_errors`.
* **sCO2 properties as polynomials.** :math:`c_p`, :math:`\\mu` and ``k``
  on the isobar at ``params.p_co2`` (from :mod:`pcsp.materials.sco2`, i.e.
  CoolProp) are fitted over ``T_range``; ``h`` is the exact integral of the
  fitted :math:`c_p`, anchored at the table enthalpy at the lower end, so
  ``h`` and :math:`c_p` are consistent. Pressure drop is neglected.
* **sCO2 coefficient** (``co2_htc``):

  - ``"gnielinski"`` (default): the correlation of :mod:`.channel` with
    bulk properties, as in :func:`pcsp.models.hx.model.co2_side`. That
    function raises below ``Re = 2300``; here, to keep the model defined when
    the bypass lowers the flow, :math:`Nu = \\max(Nu_{Gn}(\\max(Re, 2300)),
    Nu_{lam})` - an **assumption** (no transition model).
  - ``"laminar"``: fully developed laminar flow in a slot heated on both
    sides with uniform flux, :math:`Nu = 8.235` on the hydraulic diameter
    twice the slot width (Shah & London, 1978, as tabulated in standard
    texts). This assumes the slot encoding of
    :mod:`pcsp.models.hx.albrecht_ho` (``co2_channel_diameter`` = twice the
    slot width, which is then the hydraulic diameter). Thermal entry effects
    are neglected, so this is a lower bound for a laminar solver.
  - a float: a fixed coefficient [W m^-2 K^-1] on the channel wetted area.

The reference case of Fang, Chen & Shi (2022) and its published response
times are at the end of this module (:func:`fang2022_params`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve

from pcsp.core.backend import ops_for
from pcsp.core.interfaces import ControlModelBase
from pcsp.materials import sco2
from pcsp.materials.particles import CARBO_HSP_40_70
from pcsp.models.hx.channel import RE_MIN
from pcsp.models.hx.closures import BedClosure, YagiKuniiBed
from pcsp.models.hx.model import MovingBedHXParams

PLATE_DENSITY = 8000.0
"""Plate density [kg m^-3]. **Assumption** (stainless steel / nickel alloy)."""

PLATE_HEAT_CAPACITY = 550.0
"""Plate specific heat capacity [J kg^-1 K^-1]. **Assumption** (stainless or
nickel alloy near 900 K; not checked against a primary table)."""

NU_LAMINAR_SLOT_H = 8.235
"""Fully developed laminar Nusselt number of a parallel-plate channel heated on
both sides with uniform flux, on the hydraulic diameter (twice the gap) [-]."""

CO2_HTC_OPTIONS = ("gnielinski", "laminar")


def _horner(coeffs: Sequence[float], tau: Any) -> Any:
    """Polynomial with coefficients highest power first, backend-agnostic."""
    result = coeffs[0] + 0.0 * tau
    for c in coeffs[1:]:
        result = result * tau + c
    return result


@dataclass(frozen=True)
class PolyFit:
    """Polynomial ``p(tau)`` in the scaled variable ``tau = (T - T_mid) / T_half``.

    ``coeffs`` highest power first. :meth:`__call__` works for floats, NumPy
    arrays and CasADi expressions.
    """

    coeffs: tuple[float, ...]
    T_mid: float
    T_half: float

    def tau(self, T: Any) -> Any:
        return (T - self.T_mid) / self.T_half

    def __call__(self, T: Any) -> Any:
        return _horner(self.coeffs, self.tau(T))

    def integral(self, T: Any, T_ref: float) -> Any:
        """:math:`\\int_{T_{ref}}^{T} p \\, dT` [units of p times K]."""
        n = len(self.coeffs)
        anti = [c / (n - k) for k, c in enumerate(self.coeffs)] + [0.0]
        return self.T_half * (_horner(anti, self.tau(T)) - _horner(anti, self.tau(T_ref)))


def fit_polynomial(
    f: Callable[[np.ndarray], np.ndarray], T_range: tuple[float, float], degree: int
) -> tuple[PolyFit, float]:
    """Least-squares fit of ``f(T)`` on Chebyshev points; return the fit and its max relative error.

    The error is measured on 2001 uniform points over ``T_range``.
    """
    lo, hi = T_range
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    n_pts = max(4 * (degree + 1), 64)
    tau = np.cos(np.pi * (np.arange(n_pts) + 0.5) / n_pts)
    coeffs = np.polyfit(tau, np.asarray(f(mid + half * tau), dtype=float), degree)
    fit = PolyFit(tuple(float(c) for c in coeffs), mid, half)
    T_check = np.linspace(lo, hi, 2001)
    exact = np.asarray(f(T_check), dtype=float)
    return fit, float(np.max(np.abs(fit(T_check) / exact - 1.0)))


@dataclass(frozen=True)
class ScaledClosure:
    """A :class:`BedClosure` with its conductivity and near-wall resistance scaled.

    ``k_eff = k_factor * base.k_eff`` and ``r_wall = r_factor * base.r_wall``.
    For sensitivity studies only.
    """

    base: BedClosure
    k_factor: float = 1.0
    r_factor: float = 1.0
    name: str = "scaled"

    def __post_init__(self) -> None:
        if self.k_factor <= 0.0 or self.r_factor < 0.0:
            raise ValueError("need k_factor > 0 and r_factor >= 0")

    def k_eff(self, T: np.ndarray, u: float, particle: Any) -> np.ndarray:
        return self.k_factor * self.base.k_eff(T, u, particle)

    def r_wall(self, T: np.ndarray, u: float, particle: Any) -> np.ndarray:
        return self.r_factor * self.base.r_wall(T, u, particle)


class SteadyState(NamedTuple):
    """An equilibrium ``rhs(x, u, d) = 0`` of :class:`MovingBedHXDynamic`."""

    x: np.ndarray
    u: np.ndarray
    d: np.ndarray
    y: np.ndarray
    residual: float
    """max |dx/dt| at ``x`` [K s^-1]"""


class MovingBedHXDynamic(ControlModelBase):
    """Transient 2-D moving-bed heat exchanger (see the module docstring).

    =========  ======================  ==========================================  =======
    vector     name                    meaning                                     units
    =========  ======================  ==========================================  =======
    x          ``T_bed[i,j]``          bed temperature, cell ``i`` along z, ``j``  K
                                       across (j = 0 at the wall); column-major:
                                       ``x[j * n_z + i]``
    x          ``T_plate[i]``          plate temperature, ``x[n_y * n_z + i]``     K
    u[0]       ``m_dot_particle``      particle mass flow per unit cell            kg s-1
    u[1]       ``bypass_fraction``     fraction of the sCO2 flow that bypasses     -
                                       the exchanger (0 = none)
    d[0]       ``T_particle_in``       particle inlet temperature                  K
    d[1]       ``T_co2_in``            sCO2 inlet temperature                      K
    d[2]       ``m_dot_co2``           total sCO2 mass flow per unit cell          kg s-1
    y[0]       ``T_co2_mixed``         sCO2 temperature after the bypass rejoins   K
    y[1]       ``T_particle_out``      particle outlet (mixing-cup) temperature    K
    =========  ======================  ==========================================  =======

    Parameters
    ----------
    params : geometry, materials, closure and sCO2 pressure. Its operating
        point (``m_dot_particle``, ``m_dot_co2``, ``T_particle_in``,
        ``T_co2_in``) is only the nominal point (:meth:`nominal_inputs`),
        and the bed velocity at which the closure is fitted. ``n_cells`` and
        ``wall_bc`` are not used.
    n_z, n_y : cells along the flow and across the half gap
    plate_heat_capacity : :math:`C_w` per unit wall area [J m^-2 K^-1];
        default ``PLATE_DENSITY * PLATE_HEAT_CAPACITY * params.wall_thickness``
        (assumption)
    co2_htc : ``"gnielinski"``, ``"laminar"`` or a fixed coefficient
        [W m^-2 K^-1] (see the module docstring)
    T_range : temperature range [K] of the polynomial fits; states must stay
        inside it (:meth:`check_range`)
    fit_degree : degree of all polynomial fits
    bypass_max : the bypass fraction is clipped to ``[0, bypass_max]``
    """

    input_names = ("m_dot_particle", "bypass_fraction")
    disturbance_names = ("T_particle_in", "T_co2_in", "m_dot_co2")
    output_names = ("T_co2_mixed", "T_particle_out")

    def __init__(
        self,
        params: MovingBedHXParams,
        n_z: int = 50,
        n_y: int = 8,
        plate_heat_capacity: float | None = None,
        co2_htc: str | float = "gnielinski",
        T_range: tuple[float, float] = (650.0, 1200.0),
        fit_degree: int = 5,
        bypass_max: float = 0.95,
    ) -> None:
        if n_z < 2 or n_y < 1:
            raise ValueError(f"need n_z >= 2 and n_y >= 1, got {n_z}, {n_y}")
        if isinstance(co2_htc, str):
            if co2_htc not in CO2_HTC_OPTIONS:
                raise ValueError(f"co2_htc must be one of {CO2_HTC_OPTIONS} or a number")
        elif float(co2_htc) <= 0.0:
            raise ValueError("a fixed co2_htc must be positive")
        if not 0.0 <= bypass_max < 1.0:
            raise ValueError("bypass_max must be in [0, 1)")
        lo, hi = T_range
        if not lo < hi:
            raise ValueError("T_range must be increasing")
        if not (lo <= params.T_co2_in and params.T_particle_in <= hi):
            raise ValueError("the nominal inlet temperatures must lie inside T_range")

        self.params = params
        self.n_z = int(n_z)
        self.n_y = int(n_y)
        self.co2_htc = co2_htc
        self.T_range = (float(lo), float(hi))
        self.bypass_max = float(bypass_max)
        self.plate_heat_capacity = (
            PLATE_DENSITY * PLATE_HEAT_CAPACITY * params.wall_thickness
            if plate_heat_capacity is None
            else float(plate_heat_capacity)
        )
        if self.plate_heat_capacity <= 0.0:
            raise ValueError("plate_heat_capacity must be positive")

        p = params
        self.dz = p.height / n_z
        self.dy = 0.5 * p.channel_gap / n_y
        self.r_metal_half = 0.5 * p.wall_thickness / p.wall_conductivity

        self.state_names = tuple(f"T_bed[{i},{j}]" for j in range(n_y) for i in range(n_z)) + tuple(
            f"T_plate[{i}]" for i in range(n_z)
        )

        # --- polynomial fits (NumPy at construction, backend-agnostic afterwards) -----------
        u_nom = p.particle_velocity
        part = p.particle
        co2 = sco2.isobar(p.p_co2, min(300.0, lo), max(1100.0, hi))
        self._co2_table = co2
        fits: dict[str, PolyFit] = {}
        errors: dict[str, float] = {}
        sources = {
            "k_eff": lambda T: p.closure.k_eff(T, u_nom, part),
            "r_wall": lambda T: p.closure.r_wall(T, u_nom, part),
            "cp_co2": co2.cp,
            "mu_co2": co2.mu,
            "k_co2": co2.k,
        }
        for name, f in sources.items():
            fits[name], errors[name] = fit_polynomial(f, self.T_range, fit_degree)
        self._fits = fits
        self._h_co2_ref = float(co2.h(lo))
        T_check = np.linspace(lo, hi, 2001)
        h_err = np.abs(self.h_co2(T_check) - co2.h(T_check))
        errors["h_co2_abs_over_cp"] = float(np.max(h_err / co2.cp(T_check)))
        self.fit_errors = errors
        """Max relative error of each fit over ``T_range`` [-]; ``h_co2_abs_over_cp``
        is the enthalpy error expressed in kelvin."""

    # --- properties --------------------------------------------------------------------------

    def k_eff(self, T: Any) -> Any:
        """Fitted bed conductivity [W m^-1 K^-1]."""
        return self._fits["k_eff"](T)

    def r_wall(self, T: Any) -> Any:
        """Fitted near-wall resistance [m^2 K W^-1]."""
        return self._fits["r_wall"](T)

    def cp_co2(self, T: Any) -> Any:
        """Fitted sCO2 isobaric heat capacity [J kg^-1 K^-1]."""
        return self._fits["cp_co2"](T)

    def h_co2(self, T: Any) -> Any:
        """sCO2 enthalpy [J kg^-1], exact integral of :meth:`cp_co2` (CoolProp reference)."""
        return self._h_co2_ref + self._fits["cp_co2"].integral(T, self.T_range[0])

    def T_from_h_co2(self, h: Any, T_guess: Any, iterations: int = 3) -> Any:
        """Invert :meth:`h_co2` by Newton steps from ``T_guess`` [K]."""
        T = T_guess
        for _ in range(iterations):
            T = T - (self.h_co2(T) - h) / self.cp_co2(T)
        return T

    def T_from_h_particle(self, h: Any, T_guess: Any, iterations: int = 2) -> Any:
        """Invert the particle enthalpy by Newton steps from ``T_guess`` [K]."""
        part = self.params.particle
        T = T_guess
        for _ in range(iterations):
            T = T - (part.enthalpy(T) - h) / part.cp(T)
        return T

    def r_co2(self, T: Any, m_dot_hx: Any) -> Any:
        """sCO2 resistance referred to the particle-side wall area [m^2 K W^-1].

        ``m_dot_hx`` is the sCO2 flow through the exchanger per unit cell [kg s^-1].
        """
        xp = ops_for(T, m_dot_hx)
        p = self.params
        d = p.co2_channel_diameter
        k = self._fits["k_co2"](T)
        if isinstance(self.co2_htc, str) and self.co2_htc == "gnielinski":
            mu = self._fits["mu_co2"](T)
            pr = self.cp_co2(T) * mu / k
            re = xp.maximum(4.0 * m_dot_hx / p.n_co2_channels / (np.pi * d * mu), RE_MIN)
            f8 = (0.790 * xp.log(re) - 1.64) ** -2 / 8.0
            nu_gn = f8 * (re - 1000.0) * pr / (1.0 + 12.7 * xp.sqrt(f8) * (pr ** (2.0 / 3.0) - 1.0))
            h = xp.maximum(nu_gn, NU_LAMINAR_SLOT_H) * k / d
        elif isinstance(self.co2_htc, str):  # "laminar"
            h = NU_LAMINAR_SLOT_H * k / d + 0.0 * m_dot_hx
        else:
            h = float(self.co2_htc) + 0.0 * T + 0.0 * m_dot_hx
        return 2.0 * p.width / (h * p.n_co2_channels * np.pi * d)

    # --- state layout ---------------------------------------------------------------------

    def bed_column(self, x: Any, j: int) -> Any:
        """Bed temperatures of column ``j`` (along z) [K]."""
        return x[j * self.n_z : (j + 1) * self.n_z]

    def plate(self, x: Any) -> Any:
        """Plate temperatures along z [K]."""
        return x[self.n_y * self.n_z : (self.n_y + 1) * self.n_z]

    def _bypass(self, u: Any) -> Any:
        xp = ops_for(u)
        return xp.minimum(xp.maximum(u[1], 0.0), self.bypass_max)

    # --- the model --------------------------------------------------------------------------

    def _evaluate(self, x: Any, u: Any, d: Any) -> dict[str, Any]:
        """All fluxes; backend-agnostic. Per half cell (one wall face)."""
        xp = ops_for(x, u, d)
        p = self.params
        part = p.particle
        nz, ny, dz, dy, W = self.n_z, self.n_y, self.dz, self.dy, p.width
        m_p, T_p_in = u[0], d[0]
        beta = self._bypass(u)
        m_hx = (1.0 - beta) * d[2]

        cols = [self.bed_column(x, j) for j in range(ny)]
        Tw = self.plate(x)
        k = [self.k_eff(c) for c in cols]

        T_mean = cols[0]
        for c in cols[1:]:
            T_mean = T_mean + c
        T_mean = T_mean / ny

        # wall heat flux [W m^-2], bed -> plate
        r_bed_side = 0.5 * dy / k[0] + self.r_wall(T_mean) + self.r_metal_half
        q_wall = (cols[0] - Tw) / r_bed_side

        # conduction between columns [W per cell face], towards the centre
        q_y = []
        for j in range(ny - 1):
            k_face = 2.0 * k[j] * k[j + 1] / (k[j] + k[j + 1])
            q_y.append(k_face * W * dz * (cols[j] - cols[j + 1]) / dy)

        # advection in enthalpy, first-order upwind (flow in +z)
        m_col = m_p / (2.0 * ny)
        h_in = part.enthalpy(T_p_in)
        net = []
        for j in range(ny):
            h = part.enthalpy(cols[j])
            h_up = xp.concat([h_in, h[: nz - 1]])
            # the wall column loses heat to the plate, the others gain it from the wall side
            from_wall_side = -q_wall * W * dz if j == 0 else q_y[j - 1]
            flow = m_col * (h_up - h) + from_wall_side
            if j < ny - 1:
                flow = flow - q_y[j]
            net.append(flow)

        # quasi-steady sCO2, marched from the inlet (i = nz - 1) to the outlet (i = 0)
        m_face = 0.5 * m_hx
        T_c = d[1]
        h_c = self.h_co2(T_c)
        Q_co2 = [None] * nz
        T_c_faces = [None] * (nz + 1)
        T_c_faces[nz] = T_c
        for i in range(nz - 1, -1, -1):
            R_c = self.r_metal_half + self.r_co2(T_c, m_hx)
            ntu = W * dz / (R_c * m_face * self.cp_co2(T_c))
            T_w_i = Tw[i]
            T_out = T_w_i - (T_w_i - T_c) * xp.exp(-ntu)
            h_out = self.h_co2(T_out)
            Q_co2[i] = m_face * (h_out - h_c)
            T_c, h_c = T_out, h_out
            T_c_faces[i] = T_out
        Q_co2_vec = xp.concat(Q_co2)
        plate_net = q_wall * W * dz - Q_co2_vec

        return {
            "net_bed": net,
            "plate_net": plate_net,
            "cols": cols,
            "q_wall": q_wall,
            "Q_co2": Q_co2_vec,
            "T_co2_faces": T_c_faces,
            "T_co2_hx_out": T_c,
            "h_co2_hx_out": h_c,
            "m_dot_hx": m_hx,
            "bypass": beta,
            "h_particle_in": h_in,
        }

    def rhs(self, x: Any, u: Any, d: Any) -> Any:
        """``dx/dt`` [K s^-1]."""
        xp = ops_for(x, u, d)
        e = self._evaluate(x, u, d)
        rho_b = self.params.particle.rho_bulk
        volume = self.params.width * self.dz * self.dy
        parts = [
            e["net_bed"][j] / (rho_b * self.params.particle.cp(c) * volume)
            for j, c in enumerate(e["cols"])
        ]
        parts.append(e["plate_net"] / (self.plate_heat_capacity * self.params.width * self.dz))
        return xp.concat(parts)

    def _particle_out(self, x: Any) -> tuple[Any, Any]:
        """Mixing-cup outlet enthalpy and temperature of the bottom row."""
        part = self.params.particle
        last = self.n_z - 1
        h_sum = 0.0
        T_sum = 0.0
        for j in range(self.n_y):
            T_j = x[j * self.n_z + last]
            h_sum = h_sum + part.enthalpy(T_j)
            T_sum = T_sum + T_j
        h_out = h_sum / self.n_y
        return h_out, self.T_from_h_particle(h_out, T_sum / self.n_y)

    def output(self, x: Any, u: Any, d: Any) -> Any:
        """``[T_co2_mixed, T_particle_out]`` [K]."""
        xp = ops_for(x, u, d)
        e = self._evaluate(x, u, d)
        beta = e["bypass"]
        h_mix = (1.0 - beta) * e["h_co2_hx_out"] + beta * self.h_co2(d[1])
        T_guess = (1.0 - beta) * e["T_co2_hx_out"] + beta * d[1]
        T_mix = self.T_from_h_co2(h_mix, T_guess)
        return xp.concat([T_mix, self._particle_out(x)[1]])

    # --- diagnostics (NumPy) ----------------------------------------------------------------

    def co2_hx_outlet(self, x: np.ndarray, u: np.ndarray, d: np.ndarray) -> float:
        """sCO2 temperature at the exchanger outlet, before the bypass rejoins [K]."""
        return float(self._evaluate(np.asarray(x), np.asarray(u), np.asarray(d))["T_co2_hx_out"])

    def stored_energy(self, x: np.ndarray) -> float:
        """Enthalpy held by bed and plates per unit cell [J] (particle reference of
        :meth:`ParticleMaterial.enthalpy`, plate referenced to 0 K)."""
        x = np.asarray(x, dtype=float)
        p = self.params
        bed = x[: self.n_y * self.n_z]
        e_bed = p.particle.rho_bulk * p.width * self.dz * self.dy * np.sum(p.particle.enthalpy(bed))
        e_plate = self.plate_heat_capacity * p.width * self.dz * np.sum(self.plate(x))
        return float(2.0 * (e_bed + e_plate))

    def energy_rates(self, x: Any, u: Any, d: Any) -> tuple[Any, Any]:
        """``(P_particle, P_co2)`` per unit cell [W], backend-agnostic.

        ``P_particle`` = :math:`\\dot m_p [h_p(T_{in}) - h_p(T_{out})]`, heat given up by the
        particle stream; ``P_co2`` = :math:`\\dot m_c [h_c(T_{c,out}) - h_c(T_{c,in})]`, heat
        taken up by the sCO2. ``d(stored_energy)/dt = P_particle - P_co2``.
        """
        e = self._evaluate(x, u, d)
        h_out = self._particle_out(x)[0]
        P_particle = u[0] * (e["h_particle_in"] - h_out)
        P_co2 = e["m_dot_hx"] * (e["h_co2_hx_out"] - self.h_co2(d[1]))
        return P_particle, P_co2

    def profiles(self, x: np.ndarray, u: np.ndarray, d: np.ndarray) -> dict[str, np.ndarray]:
        """Fields for plotting: ``z_cell``, ``z_face``, ``T_bed`` (n_z, n_y), ``T_bed_mean``,
        ``T_plate``, ``T_co2`` (faces), ``heat_flux`` [W m^-2]."""
        x = np.asarray(x, dtype=float)
        e = self._evaluate(x, np.asarray(u, dtype=float), np.asarray(d, dtype=float))
        T_bed = np.stack([np.asarray(c) for c in e["cols"]], axis=1)
        z_face = np.linspace(0.0, self.params.height, self.n_z + 1)
        return {
            "z_cell": 0.5 * (z_face[:-1] + z_face[1:]),
            "z_face": z_face,
            "T_bed": T_bed,
            "T_bed_mean": T_bed.mean(axis=1),
            "T_plate": np.asarray(self.plate(x)),
            "T_co2": np.array([float(t) for t in e["T_co2_faces"]]),
            "heat_flux": np.asarray(e["q_wall"]),
        }

    def check_range(self, x: np.ndarray) -> None:
        """Raise ``ValueError`` if a state lies outside the fitted ``T_range``."""
        x = np.asarray(x)
        lo, hi = self.T_range
        if np.min(x) < lo or np.max(x) > hi:
            raise ValueError(
                f"state outside the fitted range [{lo}, {hi}] K: {np.min(x):.1f}-{np.max(x):.1f}"
            )

    def nominal_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        """``(u, d)`` of ``params``' operating point, with no bypass."""
        p = self.params
        return (
            np.array([p.m_dot_particle, 0.0]),
            np.array([p.T_particle_in, p.T_co2_in, p.m_dot_co2]),
        )

    def initial_guess(self, u: np.ndarray, d: np.ndarray) -> np.ndarray:
        """A rough state (linear profiles between the inlet temperatures) to start a solve."""
        T_p, T_c = float(d[0]), float(d[1])
        s = (np.arange(self.n_z) + 0.5) / self.n_z
        bed = T_p - 0.8 * (T_p - T_c) * s
        plate = bed - 0.1 * (T_p - T_c)
        return np.concatenate([np.tile(bed, self.n_y), plate])

    # --- compiled functions ------------------------------------------------------------------

    def casadi(self) -> dict[str, ca.Function]:
        """Cached CasADi functions ``f``, ``jac`` (d f / d x, sparse), ``h``, ``energy``."""
        if getattr(self, "_casadi", None) is None:
            x = ca.SX.sym("x", self.nx)
            u = ca.SX.sym("u", self.nu)
            d = ca.SX.sym("d", self.nd)
            f = self.rhs(x, u, d)
            P_p, P_c = self.energy_rates(x, u, d)
            self._casadi = {
                "f": ca.Function("f", [x, u, d], [f]),
                "jac": ca.Function("dfdx", [x, u, d], [ca.jacobian(f, x)]),
                "h": ca.Function("h", [x, u, d], [self.output(x, u, d)]),
                "energy": ca.Function("energy", [x, u, d], [ca.vertcat(P_p, P_c)]),
            }
        return self._casadi


def steady_state(
    model: MovingBedHXDynamic,
    u: Sequence[float],
    d: Sequence[float],
    x_guess: np.ndarray | None = None,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> SteadyState:
    """Equilibrium of ``model`` for inputs ``u`` and disturbances ``d``.

    Pseudo-transient continuation: implicit-Euler steps
    :math:`(I/\\Delta t - J)\\,\\delta x = f(x)` with the exact sparse Jacobian
    (CasADi), the pseudo time step doubling after each step that reduces
    :math:`\\max|f|`, which turns into Newton's method once the step is large.
    Stops when :math:`\\max|\\delta x| <` ``tol`` K and returns the state, the
    outputs and the final residual :math:`\\max|f|` [K s^-1].
    """
    u = np.asarray(u, dtype=float)
    d = np.asarray(d, dtype=float)
    fun = model.casadi()
    x = model.initial_guess(u, d) if x_guess is None else np.array(x_guess, dtype=float)
    eye = identity(model.nx, format="csc")
    dt = 50.0
    f = np.asarray(fun["f"](x, u, d)).ravel()
    res = float(np.max(np.abs(f)))
    for _ in range(max_iter):
        J = fun["jac"](x, u, d).tocsc()
        dx = spsolve(eye / dt - J, f)
        x_new = x + dx
        f_new = np.asarray(fun["f"](x_new, u, d)).ravel()
        res_new = float(np.max(np.abs(f_new)))
        if np.all(np.isfinite(f_new)) and res_new <= 1.5 * res:
            x, f, res = x_new, f_new, res_new
            dt = min(dt * 4.0, 1e12)
            if float(np.max(np.abs(dx))) < tol:
                break
        else:
            dt *= 0.25
    else:
        raise RuntimeError(f"steady state not converged: max |dx/dt| = {res:.3g} K/s")
    y = np.asarray(fun["h"](x, u, d)).ravel()
    return SteadyState(x, u, d, y, res)


class Trajectory(NamedTuple):
    """Result of :func:`simulate_response`."""

    t: np.ndarray
    x: np.ndarray
    """states, shape ``(len(t), nx)``"""
    y: np.ndarray
    """outputs, shape ``(len(t), ny)``"""
    energy_in: np.ndarray
    """:math:`\\int_0^t (P_{particle} - P_{CO_2})\\,dt'` per unit cell [J]"""


def simulate_response(
    model: MovingBedHXDynamic,
    x0: np.ndarray,
    u: Sequence[float],
    d: Sequence[float],
    t_eval: np.ndarray,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> Trajectory:
    """Integrate from ``x0`` with ``u`` and ``d`` held constant (a step if ``x0`` is the
    steady state of other inputs).

    Implicit BDF (:func:`scipy.integrate.solve_ivp`) with the exact sparse
    Jacobian; the net heat input :math:`P_{particle} - P_{CO_2}` is integrated
    alongside as one extra state, for energy checks.
    """
    u = np.asarray(u, dtype=float)
    d = np.asarray(d, dtype=float)
    fun = model.casadi()
    nx = model.nx

    def rhs(_t: float, z: np.ndarray) -> np.ndarray:
        x = z[:nx]
        P = np.asarray(fun["energy"](x, u, d)).ravel()
        return np.concatenate([np.asarray(fun["f"](x, u, d)).ravel(), [P[0] - P[1]]])

    def jac(_t: float, z: np.ndarray) -> Any:
        from scipy.sparse import bmat

        J = fun["jac"](z[:nx], u, d).tocsc()
        return bmat([[J, None], [None, np.zeros((1, 1))]], format="csc")

    t_eval = np.asarray(t_eval, dtype=float)
    z0 = np.concatenate([np.asarray(x0, dtype=float), [0.0]])
    sol = solve_ivp(
        rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        z0,
        method="BDF",
        t_eval=t_eval,
        jac=jac,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    X = sol.y[:nx].T
    Y = np.array([np.asarray(fun["h"](xk, u, d)).ravel() for xk in X])
    return Trajectory(sol.t, X, Y, sol.y[nx])


# --- Reference case: Fang, Chen & Shi (2022) -----------------------------------------------

FANG2022_RESPONSE_TIMES = {
    "T_particle_in": {"T_co2_out": 338.0, "T_particle_out": 969.0},
    "T_co2_in": {"T_co2_out": 560.0, "T_particle_out": 461.0},
    "u_co2_in": {"T_co2_out": 367.0, "T_particle_out": 979.0},
}
"""95 % response times [s] after a step at one inlet (Fang et al. 2022, sections 3.1.1-3.1.2).
They equal ``A + 3 B`` of the fits ``1 - exp(-(t - A)/B)`` printed in their Figs. 3, 5
and 7: (A, B) = (0, 112.7), (342.1, 209.3); (0, 186.9), (0, 154.0); (0, 122.4),
(43.5, 312.3)."""

FANG2022_FITS = {
    "T_particle_in": {"T_co2_out": (0.0, 112.7), "T_particle_out": (342.1, 209.3)},
    "T_co2_in": {"T_co2_out": (0.0, 186.9), "T_particle_out": (0.0, 154.0)},
    "u_co2_in": {"T_co2_out": (0.0, 122.4), "T_particle_out": (43.5, 312.3)},
}
"""``(A, B)`` [s] of the exponential fits in Fang et al. (2022) Figs. 3(c,d), 5(c,d), 7(c,d)."""

FANG2022_STEPS = {
    "T_particle_in": (988.0, 1008.0, 1028.0, 1068.0, 1088.0, 1108.0),
    "T_co2_in": (763.0, 783.0, 803.0, 843.0, 863.0, 883.0),
    "u_co2_in": (0.35, 0.40, 0.45, 0.55, 0.60, 0.65),
}
"""New values of the stepped input [K or m s^-1] in Fang et al. (2022), sections 3.1.1-3.1.2."""

FANG2022_TABLE2 = {
    "case 1": (998.0, 773.0, 0.25),
    "case 2": (1023.0, 823.0, 0.25),
    "case 3": (1048.0, 773.0, 0.25),
    "case 4": (1048.0, 823.0, 0.25),
    "case 5": (1023.0, 773.0, 0.5),
    "case 6": (1048.0, 773.0, 0.5),
}
"""Combined disturbances ``(T_S_in [K], T_CO2_in [K], u_CO2_in [m s^-1])``, their Table 2."""

FANG2022_BASE = {
    "height": 1.0,
    "channel_gap": 0.01,
    "co2_slot": 0.001,
    "plate_thickness": 0.001,
    "T_particle_in": 1048.0,
    "u_particle": 0.003,
    "rho_bulk": 2000.0,
    "T_co2_in": 823.0,
    "u_co2_in": 0.5,
    "p_co2": 25e6,
}
"""Base case, Fang et al. (2022) Table 1 (SI units)."""


def fang2022_co2_mass_flow(u_co2_in: float, T_co2_in: float = 823.0, width: float = 1.0) -> float:
    """sCO2 mass flow per unit cell [kg s^-1] for an inlet velocity [m s^-1].

    ``rho(T_co2_in, 25 MPa) * u * s * W`` with ``s`` the 1 mm slot: a full
    slot per full particle channel, Fang et al.'s half slot per half channel.
    Using the inlet density is an **assumption**; they solve the flow with
    properties at 800 K (their section 2.2), which would give 3 % more flow.
    """
    rho = float(sco2.isobar(FANG2022_BASE["p_co2"]).rho(T_co2_in))
    return rho * u_co2_in * FANG2022_BASE["co2_slot"] * width


def fang2022_params(
    d_p: float = 250e-6,
    closure: BedClosure | None = None,
    width: float = 1.0,
    wall_conductivity: float = 23.0,
    **overrides: Any,
) -> MovingBedHXParams:
    """Base case of Fang, Chen & Shi (2022), Energy 256, 124597, Table 1, per unit cell.

    From the paper: height 1 m, plate spacing 10 mm, sCO2 slot 1 mm, plate
    1 mm, particle inlet 1048 K at 0.003 m/s, bulk density 2000 kg/m3,
    sCO2 inlet 823 K at 0.5 m/s and 25 MPa, near-wall layer ``d_p / 2`` with
    the thin-film conductivity of Albrecht & Ho (2019).

    Not in the paper, chosen here (**assumptions**):

    * particles: :data:`~pcsp.materials.particles.CARBO_HSP_40_70` (sintered
      bauxite; bulk density 2000 kg/m3 as in the paper; heat capacity
      :math:`365 (T - 273.15)^{0.18}`, 1150-1210 J/kg/K over 850-1050 K), with
      diameter ``d_p`` (default 250 um, that of Albrecht & Ho's case against
      which Fang et al. validated). The paper fits ``c_p(T)`` and ``k(T)`` to
      Baumann & Zunft (2015) data for sintered bauxite but does not print the fits.
    * bed closure: :class:`~pcsp.models.hx.closures.YagiKuniiBed` (Albrecht &
      Ho's bulk conductivity, about 0.27 W/m/K at 700 degC, and their
      near-wall layer), unless ``closure`` is given;
    * plate conductivity ``wall_conductivity`` (23 W/m/K);
    * width ``W`` (1 m; all results scale with it);
    * the slot is entered as circular channels of diameter ``2 s`` at pitch
      ``pi s``, the convention of :mod:`pcsp.models.hx.albrecht_ho`.
    """
    b = FANG2022_BASE
    particle = CARBO_HSP_40_70.with_(d_p=d_p)
    m_p = b["rho_bulk"] * b["u_particle"] * b["channel_gap"] * width
    values: dict[str, Any] = {
        "m_dot_particle": m_p,
        "m_dot_co2": fang2022_co2_mass_flow(b["u_co2_in"], b["T_co2_in"], width),
        "T_particle_in": b["T_particle_in"],
        "T_co2_in": b["T_co2_in"],
        "p_co2": b["p_co2"],
        "particle": particle,
        "closure": closure if closure is not None else YagiKuniiBed(),
        "height": b["height"],
        "width": width,
        "channel_gap": b["channel_gap"],
        "wall_thickness": b["plate_thickness"],
        "wall_conductivity": wall_conductivity,
        "co2_channel_diameter": 2.0 * b["co2_slot"],
        "co2_channel_pitch": np.pi * b["co2_slot"],
    }
    values.update(overrides)
    return MovingBedHXParams(**values)


def with_params(model: MovingBedHXDynamic, **changes: Any) -> MovingBedHXDynamic:
    """Copy of ``model`` with ``params`` fields replaced (same grid and options)."""
    return MovingBedHXDynamic(
        dataclasses.replace(model.params, **changes),
        n_z=model.n_z,
        n_y=model.n_y,
        plate_heat_capacity=model.plate_heat_capacity,
        co2_htc=model.co2_htc,
        T_range=model.T_range,
        bypass_max=model.bypass_max,
        fit_degree=len(model._fits["k_eff"].coeffs) - 1,
    )
