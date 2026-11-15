"""Steady one-dimensional model of a free-falling particle curtain receiver.

Model structure
---------------
The structure follows the reduced-order receiver model of González-Portillo,
Albrecht & Ho (2021), *Techno-Economic Optimization of CSP Plants with
Free-Falling Particle Receivers*, Entropy 23(1):76, doi:10.3390/e23010076,
Section 2.5 (their Eqs. 4-17), which they in turn base on Albrecht, Bauer &
Ho (ASME ES2019). Like theirs, it resolves only the fall direction ``y``
(downwards from the release slot, metres) and uses two lumped loss
parameters that González-Portillo et al. fitted to Sandia CFD (Mills & Ho,
AIP Conf. Proc. 2126, 2019): an aperture escape factor ``F`` and an
advective heat-transfer coefficient ``h_adv``. Differences from the paper
are listed under *Departures* below.

Governing equations (per unit curtain width, steady)
----------------------------------------------------
Mass and momentum (G-P Eqs. 4-5, free fall, drag on the curtain neglected)::

    phi * t_c * rho_p * v * w = m_dot                                   [kg s^-1]
    v(y) = sqrt(v0^2 + 2 g y)                                           [m s^-1]

Energy (G-P Eq. 6), marched from the slot (``y = 0``) to the bottom::

    m_dot * dh_p/dy = w * q_net(T_p, y)                                 [W m^-1]
    q_net = g_s - F * J_front - q_wall - q_adv                           [W m^-2]

with ``h_p`` the particle specific enthalpy (``ParticleMaterial.enthalpy``,
temperature-dependent cp), ``g_s`` the concentrated solar irradiance on the
curtain front, and, per unit curtain area:

* front radiosity (G-P Eq. 8)
  ``J_front = rho_s g_s + eps_c sigma T_p^4 + tau_t g_back``;
  a fraction ``F`` of it leaves through the aperture, the rest is returned to
  the curtain by the cavity (G-P treat it as re-absorbed);
* back radiosity (G-P Eq. 10) ``J_back = eps_c sigma T_p^4 + rho_t g_back +
  tau_s g_s``, received by the back wall, which returns
  ``g_back = J_back - q_wall`` (G-P Eqs. 9, 13, 14 with a re-radiating wall);
* back-wall conduction loss (G-P Eq. 15 with a series resistance)
  ``q_wall = U_wall (T_wall - T_amb)``, ``U_wall = 1/(t_w/k_w + 1/h_out)``;
* advective loss (G-P Eq. 11) ``q_adv = h_adv (T_p - T_amb)``.

Eliminating ``g_back`` gives the explicit closed form used in the code::

    J_back = (eps_c sigma T_p^4 + tau_s g_s - rho_t q_wall) / (1 - rho_t)

The loss breakdown follows directly: reflective ``F rho_s g_s``, radiative
``F (eps_c sigma T_p^4 + tau_t g_back)``, back wall ``q_wall``, advective
``q_adv``. Their sum plus ``q_net`` equals ``g_s`` identically, so the
discrete model conserves energy to round-off (see :func:`simulate`).

Curtain optics
--------------
Particles are large compared with the wavelength (``d_p = 350 um``), so each
intercepts radiation over its projected area (extinction efficiency
``Q_ext = 1``, diffraction counted as transmitted) and the extinction optical
thickness of the curtain is (independent scattering, geometric optics)::

    tau_e = 1.5 Q_ext phi t_c / d_p = 1.5 Q_ext m_dot / (rho_p w v d_p)   [-]

which for a non-scattering curtain gives Beer's law ``exp(-1.5 phi t_c /
d_p)`` used by Sandia for curtain opacity (Kim et al., Solar Energy 83, 2009).
Scattering is handled with the Kubelka-Munk two-flux solution for a slab with
non-reflecting boundaries (see :func:`two_flux_slab`), using the particle
absorptivity as single-particle absorption (single-scattering albedo
``omega = 1 - alpha_p``) and a backscatter fraction ``b``. The solar band
uses the particle solar absorptivity; the thermal band the particle
emissivity; Kirchhoff's law gives the curtain emissivity ``eps_c = 1 - rho_t
- tau_t``. González-Portillo et al. use their own analytical optics model
(Solar Energy 213, 2021), which could not be retrieved; the two-flux model
here is a documented substitute.

Departures from González-Portillo et al. (2021)
-----------------------------------------------
* Optics: two-flux slab (above) instead of their analytical model.
* Back wall: no axial conduction; each wall element re-radiates locally and
  its conduction loss uses the *adiabatic* wall temperature
  ``sigma T_wall^4 = J_back(q_wall = 0)``, which keeps the scheme explicit.
  With the default wall (``U ~ 2.9 W m^-2 K^-1``) this loss is < 1 % of
  the incident power, so the approximation affects only a small term.
* The back cavity is grey: solar radiation transmitted through the curtain
  and returned by the wall is treated in the thermal band.
* The incident flux is uniform over the curtain band that faces the
  aperture (``aperture_top`` to ``aperture_top + aperture_height``) and
  averaged over the curtain width.
* As in G-P, ambient radiation entering the aperture is neglected
  (``sigma T_amb^4 ~ 0.46 kW m^-2`` at 300 K, about 1 % of the curtain's
  emission at 900 K). The model therefore does not hold particles at exactly
  ``T_amb`` without sun; it is meant for operating temperatures.
* Curtain drag is neglected (G-P Eq. 5). The terminal velocity of an
  isolated 350 um bauxite sphere in 900 K air is of order 3 m/s (our
  Schiller-Naumann estimate), below the ~6.6 m/s reached at 2.25 m, so free fall
  relies on the dense curtain dragging the surrounding air along with it.
  This is an assumption. It makes the residence time a lower bound, and
  an unknown slow-down would raise the opacity.

Numerics
--------
Heun's method (explicit trapezoidal rule, second order) in ``y`` on the
enthalpy, with the solar irradiance integrated exactly over each segment.
There are no implicit solves and no branching on the state, so the same
code runs on floats or CasADi symbols (:mod:`pcsp.core.backend`). The local
loss breakdown is averaged with the same Heun weights as the enthalpy
update, which is what makes the energy balance close to round-off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import xarray as xr

from pcsp.constants import SIGMA_SB, T_ZERO_C, G
from pcsp.core.backend import ops_for
from pcsp.data.schema import Coord, Var, new_dataset
from pcsp.materials.particles import CARBO_HSP_40_70, ParticleMaterial

MODEL_NAME = "fpr_curtain_1d"
MODEL_VERSION = "0.1.0"

M_DOT_MIN = 1e-6
"""Floor on the mass flow [kg s^-1] that keeps the model finite at zero flow."""


@dataclass(frozen=True)
class FPRParams:
    """Inputs of the steady curtain model.

    Default geometry and loss parameters are the "Small Scale" free-falling
    receiver of González-Portillo et al. (2021), Table 6, whose ``F`` and
    ``h_adv`` were fitted there to Sandia CFD of the 1 MWt NSTTF receiver
    (Mills & Ho 2019). Whether every Table 6 dimension equals the NSTTF
    hardware could not be checked (the NSTTF papers are on OSTI, which was
    not reachable); Sandia reports a ~1 m^2 square aperture for the NSTTF
    receiver (Ho et al., SolarPACES 2020; Mills et al., Energies 15:1657,
    2022), which is what ``aperture_height`` assumes.

    Attributes
    ----------
    mass_flow : particle mass flow rate [kg s^-1]
    T_in : particle temperature at the release slot [K]
    Q_in : concentrated solar power entering the aperture [W]
    T_amb : ambient air temperature [K]
    curtain_width : curtain width ``w`` [m] (G-P Table 6: 1.52 m)
    fall_height : irradiated-cavity fall height ``H`` [m] (G-P Table 6: 2.25 m)
    aperture_height : height of the curtain band facing the aperture [m]
        (assumption: 1 m, square ~1 m^2 NSTTF aperture)
    aperture_top : depth of that band's top edge below the release slot [m]
        (assumption: band centred on the fall, ``(H - aperture_height) / 2``)
    h_adv : advective heat-transfer coefficient [W m^-2 K^-1], per unit
        curtain area. **The dominant uncertain parameter.** 38 W m^-2 K^-1 is
        González-Portillo et al.'s fit to CFD of the NSTTF receiver (Table 6);
        CFD put advective losses of that receiver at ~10-25 % of the incident
        power (Mills et al. 2022, Sec. 3.4), and a reduced-volume cavity cut
        them by ~40 % in ground tests (Ho et al. 2020).
    aperture_escape_factor : fraction ``F`` of the curtain-front radiosity
        lost through the aperture [-] (G-P Table 6 fit: 0.65 small scale)
    wall_conductivity, wall_thickness, wall_outer_htc : back-wall insulation
        [W m^-1 K^-1], [m], [W m^-2 K^-1] (G-P Table 7: 0.2, 0.05, 10, all
        labelled "assumed" by them)
    extinction_efficiency : particle extinction efficiency ``Q_ext`` [-]
        (1: geometric-optics large-particle limit, diffraction excluded)
    backscatter_fraction : fraction ``b`` of scattered radiation sent
        backwards in the two-flux model [-] (0.5: isotropic; nominal,
        unverified for bauxite spheres)
    solar_path_factor, thermal_path_factor : two-flux path-length factors
        [-] (1 for the near-collimated solar beam, consistent with the Beer
        law of Kim et al. 2009; 2 for diffuse thermal radiation, the standard
        two-flux hemispherical value)
    thickness_growth : curtain thickness growth ``dt_c/dy`` [m m^-1]
        (0.0087, Kim et al. 2009 as quoted in G-P Eq. 17)
    material : particle material (cp, density, radiative properties)
    n_segments : number of segments along the fall
    """

    mass_flow: float = 6.0
    T_in: float = 823.15
    Q_in: float = 600e3
    T_amb: float = 300.0
    curtain_width: float = 1.52
    fall_height: float = 2.25
    aperture_height: float = 1.0
    aperture_top: float = 0.625
    h_adv: float = 38.0
    aperture_escape_factor: float = 0.65
    wall_conductivity: float = 0.2
    wall_thickness: float = 0.05
    wall_outer_htc: float = 10.0
    extinction_efficiency: float = 1.0
    backscatter_fraction: float = 0.5
    solar_path_factor: float = 1.0
    thermal_path_factor: float = 2.0
    thickness_growth: float = 0.0087
    material: ParticleMaterial = CARBO_HSP_40_70
    n_segments: int = 50

    def __post_init__(self) -> None:
        positive = ("curtain_width", "fall_height", "aperture_height", "T_amb", "T_in")
        positive += ("wall_conductivity", "wall_thickness", "wall_outer_htc")
        positive += ("extinction_efficiency", "solar_path_factor", "thermal_path_factor")
        for name in positive:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        for name in ("mass_flow", "Q_in", "h_adv", "thickness_growth", "aperture_top"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        for name in ("aperture_escape_factor", "backscatter_fraction"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {getattr(self, name)}")
        if self.aperture_top + self.aperture_height > self.fall_height:
            raise ValueError("the aperture band must lie within the fall height")
        if self.n_segments < 1:
            raise ValueError(f"n_segments must be >= 1, got {self.n_segments}")

    @property
    def wall_U(self) -> float:
        """Overall back-wall loss coefficient ``1/(t_w/k_w + 1/h_out)`` [W m^-2 K^-1]."""
        return 1.0 / (self.wall_thickness / self.wall_conductivity + 1.0 / self.wall_outer_htc)


def temperature_from_enthalpy(material: ParticleMaterial, h: Any, T_ref: float = 298.15) -> Any:
    """Exact inverse of :meth:`ParticleMaterial.enthalpy` [K].

    For ``cp = a (T - 273.15)^b`` the enthalpy is
    ``h = a/n ((T - 273.15)^n - (T_ref - 273.15)^n)`` with ``n = 1 + b``, so

    .. math::  T = 273.15 + \\left(\\frac{n h}{a} + (T_{ref} - 273.15)^n\\right)^{1/n}

    Valid above 0 degC, like the cp correlation itself.
    """
    n = 1.0 + material.cp_b
    return T_ZERO_C + (n * h / material.cp_a + (T_ref - T_ZERO_C) ** n) ** (1.0 / n)


def two_flux_slab(tau_e: Any, albedo: float, backscatter: float, path_factor: float) -> Any:
    """Reflectance and transmittance of a scattering slab (Kubelka-Munk) [-].

    Two-flux equations for forward/backward fluxes ``I+``, ``I-`` in a slab of
    extinction optical thickness ``tau_e`` (Kubelka & Munk 1931; e.g. Modest,
    *Radiative Heat Transfer*, two-flux approximation)::

        dI+/dx = -(K + S) I+ + S I-,      -dI-/dx = -(K + S) I- + S I+

    with absorption ``K = D (1 - omega) tau_e`` and backscatter
    ``S = D b omega tau_e`` (optical thicknesses), path factor ``D``. For
    non-reflecting boundaries the exact solution is, with
    ``kappa = sqrt(K (K + 2 S))``,

    .. math::
        R = \\frac{S \\tanh\\kappa}{(K + S) \\tanh\\kappa + \\kappa}, \\qquad
        T = \\frac{\\kappa}{\\cosh\\kappa \\, [(K + S) \\tanh\\kappa + \\kappa]}

    Limits: ``omega = 0`` gives Beer's law ``T = exp(-D tau_e)``, ``R = 0``;
    ``tau_e -> inf`` gives ``T -> 0`` and
    ``R -> 1 + K/S - sqrt((K/S)^2 + 2 K/S)``.

    Requires ``albedo < 1`` and ``tau_e > 0``. Returns ``(R, T)``.
    """
    xp = ops_for(tau_e)
    K = path_factor * (1.0 - albedo) * tau_e
    S = path_factor * backscatter * albedo * tau_e
    kappa = xp.sqrt(K * (K + 2.0 * S))
    th = xp.tanh(kappa)
    denom = (K + S) * th + kappa
    cosh = 0.5 * (xp.exp(kappa) + xp.exp(-kappa))
    return S * th / denom, kappa / (cosh * denom)


def release_thickness(p: FPRParams, m_dot: Any) -> Any:
    """Curtain thickness at the slot ``t_c0`` [m] (G-P 2021 Eq. 16).

    .. math::  t_{c,0} = \\left(\\frac{60\\,\\dot m}{62\\, w\\, \\phi_0\\, \\rho_p \\sqrt{g}}
               \\right)^{1/1.5} + 1.4\\, d_p

    a Beverloo-type slot-discharge law attributed by González-Portillo et al.
    to Ho et al. (J. Sol. Energy Eng. 139, 2017); transcribed from G-P, the
    original was not checked. ``phi_0`` is the packed-bed solid fraction.
    """
    mat = p.material
    base = 60.0 * m_dot / (62.0 * p.curtain_width * mat.packing_fraction * mat.rho_p * G**0.5)
    return base ** (1.0 / 1.5) + 1.4 * mat.d_p


def release_velocity(p: FPRParams, m_dot: Any) -> Any:
    """Particle velocity at the slot, from mass conservation at ``t_c0`` [m s^-1]."""
    bulk_density = p.material.packing_fraction * p.material.rho_p
    return m_dot / (bulk_density * p.curtain_width * release_thickness(p, m_dot))


def aperture_fractions(p: FPRParams, n: int) -> np.ndarray:
    """Fraction of ``Q_in`` falling on each of ``n`` equal segments [-] (sums to 1)."""
    edges = np.linspace(0.0, p.fall_height, n + 1)
    top, bottom = p.aperture_top, p.aperture_top + p.aperture_height
    overlap = np.clip(np.minimum(edges[1:], bottom) - np.maximum(edges[:-1], top), 0.0, None)
    return overlap / p.aperture_height


class LocalState(NamedTuple):
    """Curtain state and fluxes at one point of the fall (fluxes in W m^-2)."""

    T_p: Any
    velocity: Any
    solid_fraction: Any
    thickness: Any
    tau_solar: Any
    alpha_solar: Any
    eps_curtain: Any
    T_wall: Any
    q_net: Any
    q_reflective: Any
    q_radiative: Any
    q_wall: Any
    q_advective: Any


class CurtainSolution(NamedTuple):
    """Result of :func:`march_curtain`.

    ``h`` holds the ``n + 1`` enthalpies at the segment edges [J kg^-1];
    ``segments`` the ``n`` Heun-averaged :class:`LocalState` that the
    enthalpy update actually used; ``solar_flux`` the segment-averaged
    incident irradiance [W m^-2].
    """

    h: list[Any]
    segments: list[LocalState]
    solar_flux: list[Any]


def local_state(p: FPRParams, m_dot: Any, v0: Any, y: float, h: Any, g_s: Any) -> LocalState:
    """Evaluate the curtain at depth ``y`` [m] for enthalpy ``h`` and irradiance ``g_s``."""
    xp = ops_for(m_dot, h, g_s)
    mat = p.material
    T = temperature_from_enthalpy(mat, h)
    v = xp.sqrt(v0**2 + 2.0 * G * y)
    t_c = release_thickness(p, m_dot) + p.thickness_growth * y
    phi = m_dot / (mat.rho_p * p.curtain_width * t_c * v)
    tau_e = 1.5 * p.extinction_efficiency * m_dot / (mat.rho_p * p.curtain_width * v * mat.d_p)
    b = p.backscatter_fraction
    rho_s, tau_s = two_flux_slab(tau_e, 1.0 - mat.solar_absorptivity, b, p.solar_path_factor)
    rho_t, tau_t = two_flux_slab(tau_e, 1.0 - mat.emissivity, b, p.thermal_path_factor)
    eps_c = 1.0 - rho_t - tau_t
    emission = eps_c * SIGMA_SB * T**4
    # Back wall: adiabatic temperature first, then its small conduction loss.
    T_wall = ((emission + tau_s * g_s) / ((1.0 - rho_t) * SIGMA_SB)) ** 0.25
    q_wall = p.wall_U * (T_wall - p.T_amb)
    J_back = (emission + tau_s * g_s - rho_t * q_wall) / (1.0 - rho_t)
    g_back = J_back - q_wall
    F = p.aperture_escape_factor
    q_reflective = F * rho_s * g_s
    q_radiative = F * (emission + tau_t * g_back)
    q_adv = p.h_adv * (T - p.T_amb)
    q_net = g_s - q_reflective - q_radiative - q_wall - q_adv
    return LocalState(
        T, v, phi, t_c, tau_s, 1.0 - rho_s - tau_s, eps_c, T_wall,
        q_net, q_reflective, q_radiative, q_wall, q_adv,
    )  # fmt: skip


def march_curtain(p: FPRParams, n: int, m_dot: Any, T_in: Any, Q_in: Any) -> CurtainSolution:
    """March the steady curtain energy equation from the slot to the bottom.

    Backend-agnostic: ``m_dot`` [kg s^-1], ``T_in`` [K] and ``Q_in`` [W] may
    be floats or CasADi symbols; ``p`` supplies everything else (its own
    ``mass_flow``, ``T_in``, ``Q_in`` and ``n_segments`` are ignored here).

    Stability: the explicit march behaves like Heun's method on
    ``dT/dy = -k T`` with ``k = w (dq_loss/dT) / (m_dot cp)``; it needs
    ``k dy`` well below 2. For the default receiver this holds for
    ``m_dot >= 0.5 kg/s`` with any ``n >= 1`` and fails (NaN) for one or two
    segments near 0.2 kg/s. Keep optimisers to physical flows (~1-11 kg/s in
    Sandia's NSTTF tests, Mills et al. 2022 Sec. 4.2) or use more segments.
    """
    xp = ops_for(m_dot, T_in, Q_in)
    m = xp.maximum(m_dot, M_DOT_MIN)
    dy = p.fall_height / n
    area = p.curtain_width * dy
    fractions = aperture_fractions(p, n)
    v0 = release_velocity(p, m)
    h = p.material.enthalpy(T_in)
    hs, segments, solar = [h], [], []
    for i in range(n):
        g_s = Q_in * float(fractions[i]) / area
        s1 = local_state(p, m, v0, i * dy, h, g_s)
        s2 = local_state(p, m, v0, (i + 1) * dy, h + area * s1.q_net / m, g_s)
        seg = LocalState(*(0.5 * (a + b) for a, b in zip(s1, s2, strict=True)))
        h = h + area * seg.q_net / m
        hs.append(h)
        segments.append(seg)
        solar.append(g_s)
    return CurtainSolution(hs, segments, solar)


def outlet_temperature(p: FPRParams, n: int, m_dot: Any, T_in: Any, Q_in: Any) -> Any:
    """Particle temperature leaving the curtain [K] (backend-agnostic)."""
    sol = march_curtain(p, n, m_dot, T_in, Q_in)
    return temperature_from_enthalpy(p.material, sol.h[-1])


def simulate(params: FPRParams) -> xr.Dataset:
    """Run the steady curtain model and return a schema-conforming dataset.

    Thermal efficiency is defined as in Sandia's on-sun tests (Ho et al.,
    SolarPACES 2020, Eq. 1; Mills et al., Energies 2022, Eq. 1)::

        eta = m_dot (h_out - h_in) / Q_in

    with ``Q_in`` the solar power entering the aperture. The loss breakdown
    satisfies ``Q_in = Q_absorbed + Q_reflective + Q_radiative + Q_advective
    + Q_wall``; ``energy_residual`` reports the round-off left in that sum.
    """
    p = params
    n = p.n_segments
    sol = march_curtain(p, n, p.mass_flow, p.T_in, p.Q_in)
    area = p.curtain_width * p.fall_height / n
    m = max(p.mass_flow, M_DOT_MIN)
    y = np.linspace(0.0, p.fall_height, n + 1)
    y_seg = 0.5 * (y[1:] + y[:-1])
    # Nodal kinematics and optics do not depend on the irradiance, hence g_s = 0.
    v0 = release_velocity(p, m)
    nodes = [local_state(p, m, v0, float(yi), hi, 0.0) for yi, hi in zip(y, sol.h, strict=True)]
    seg = {k: np.array([getattr(s, k) for s in sol.segments]) for k in LocalState._fields}
    node = {k: np.array([getattr(s, k) for s in nodes]) for k in LocalState._fields}
    Q_abs = m * (sol.h[-1] - sol.h[0])
    losses = {
        k: float(np.sum(seg[k]) * area)
        for k in ("q_reflective", "q_radiative", "q_advective", "q_wall")
    }
    residual = p.Q_in - Q_abs - sum(losses.values())
    residence = float((node["velocity"][-1] - node["velocity"][0]) / G)  # exact for free fall
    eta = Q_abs / p.Q_in if p.Q_in > 0.0 else np.nan

    def node_var(key: str, units: str, name: str) -> Var:
        return Var("y", node[key], units, name)

    def seg_var(key: str, units: str, name: str) -> Var:
        return Var("y_segment", seg[key], units, name)

    data_vars = {
        "T_p": node_var("T_p", "K", "particle temperature"),
        "velocity": node_var("velocity", "m s-1", "particle velocity"),
        "solid_fraction": node_var("solid_fraction", "1", "curtain solid volume fraction"),
        "thickness": node_var("thickness", "m", "curtain thickness"),
        "tau_solar": node_var("tau_solar", "1", "curtain solar transmittance"),
        "solar_flux": Var("y_segment", np.array(sol.solar_flux), "W m-2", "incident solar flux"),
        "q_absorbed": seg_var("q_net", "W m-2", "net flux absorbed by particles"),
        "q_reflective": seg_var("q_reflective", "W m-2", "solar reflected out of aperture"),
        "q_radiative": seg_var("q_radiative", "W m-2", "thermal radiation out of aperture"),
        "q_advective": seg_var("q_advective", "W m-2", "advective loss"),
        "q_wall": seg_var("q_wall", "W m-2", "back-wall conduction loss"),
        "alpha_solar": seg_var("alpha_solar", "1", "curtain solar absorptance"),
        "eps_curtain": seg_var("eps_curtain", "1", "curtain thermal emissivity"),
        "T_wall": seg_var("T_wall", "K", "back-wall temperature"),
        "T_out": Var((), node["T_p"][-1], "K", "particle outlet temperature"),
        "efficiency": Var((), eta, "1", "thermal efficiency m_dot*(h_out-h_in)/Q_in"),
        "Q_in": Var((), p.Q_in, "W", "solar power entering the aperture"),
        "Q_absorbed": Var((), Q_abs, "W", "particle enthalpy rise m_dot*(h_out-h_in)"),
        "Q_reflective": Var((), losses["q_reflective"], "W", "reflective loss"),
        "Q_radiative": Var((), losses["q_radiative"], "W", "thermal radiative loss"),
        "Q_advective": Var((), losses["q_advective"], "W", "advective loss"),
        "Q_wall": Var((), losses["q_wall"], "W", "back-wall conduction loss"),
        "energy_residual": Var((), residual, "W", "Q_in minus absorbed and all losses"),
        "residence_time": Var((), residence, "s", "particle residence time in the fall"),
    }
    coords = {
        "y": Coord(y, "m", "depth below release slot (segment edges)"),
        "y_segment": Coord(y_seg, "m", "depth below release slot (segment centres)"),
    }
    return new_dataset(
        component="receiver_fpr",
        fidelity="reduced",
        model=MODEL_NAME,
        model_version=MODEL_VERSION,
        params=params,
        coords=coords,
        data_vars=data_vars,
    )
