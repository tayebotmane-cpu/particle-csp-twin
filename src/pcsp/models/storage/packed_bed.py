"""1-D two-phase (Schumann-type) packed-bed thermocline storage model.

A vertical bed of particles or rocks of length :math:`L` and diameter :math:`D`
is charged by hot gas entering at the *hot end* (:math:`z = 0`) and
discharged by cold gas entering at the *cold end* (:math:`z = L`). The
coordinate ``z`` is fixed to the bed and always measured from the hot end, so
the end state of one phase is the start state of the next.

Governing equations
-------------------
Written in the flow coordinate :math:`x` (distance from the inlet of the
current phase; :math:`x = z` when charging, :math:`x = L - z` when
discharging):

.. math::
    \\varepsilon \\rho_f c_f \\frac{\\partial T_f}{\\partial t}
        + G \\frac{\\partial h_f}{\\partial x}
        = h_v (T_s - T_f) - \\frac{4 U_w}{D} (T_f - T_{amb})
    \\qquad [W\\,m^{-3}]

.. math::
    \\frac{\\partial H_s}{\\partial t}
        = h_v (T_f - T_s)
          + \\frac{\\partial}{\\partial x}\\left(k_{eff} \\frac{\\partial T_s}{\\partial x}\\right)
    \\qquad [W\\,m^{-3}]

* :math:`G = \\dot m / A` [kg m^-2 s^-1] is the superficial mass flux. Gas
  mass storage is neglected (quasi-steady continuity), so :math:`G` - not the
  velocity :math:`u_s = G / \\rho_f` - is uniform along the bed.
  :math:`G \\partial_x h_f = G c_f \\partial_x T_f`.
* :math:`h_f(T) = \\int_{T_{cold}}^{T} c_f\\,dT` [J kg^-1] is the gas enthalpy and
  :math:`H_s(T) = (1-\\varepsilon) \\rho_s \\int_{T_{cold}}^{T} c_s\\,dT`
  [J m^-3] the solid volumetric enthalpy.
* :math:`h_v = 6 (1-\\varepsilon) / d_p \\cdot h_{eff}` [W m^-3 K^-1] with the
  particle coefficient from Wakao & Kaguei (1982),
  :math:`Nu = h d_p / k_f = 2 + 1.1 Re_p^{0.6} Pr^{1/3}`,
  :math:`Re_p = G d_p / \\mu_f` (:func:`pcsp.models.correlations.nu_wakao_kaguei`).
* Intra-particle conduction (optional): the lumped solid temperature is the
  particle *mean*, so the internal resistance is added in series,
  :math:`1/h_{eff} = 1/h + d_p / (10 k_s)`, i.e. :math:`h_{eff} = h / (1 + Bi/5)`
  with :math:`Bi = h d_p / (2 k_s)`. The factor 1/5 is exact for a sphere in
  the quasi-steady (uniformly heating) regime: the parabolic internal profile
  gives :math:`T_{surface} - \\bar T = q'' R / (5 k_s)`. This is the correction
  usually attributed to Jeffreson (1972, AIChE J. 18, 409-416); the original
  paper was not consulted. It is an approximation for slow transients and
  moderate :math:`Bi`; for :math:`Bi \\gtrsim 1` a resolved particle model is
  needed.
* :math:`k_{eff}` (optional axial solid conduction) is the stagnant bed
  conductivity of Zehner & Schlunder (1970)
  (:func:`pcsp.models.correlations.k_eff_zehner_schlunder`), with the gas
  conductivity evaluated at the face solid temperature. Bed ends are adiabatic
  for conduction.
* :math:`U_w` [W m^-2 K^-1] (optional) is an overall wall loss coefficient
  based on the inner wall area; perimeter / area of a cylinder is :math:`4/D`.

Properties: dry air (:mod:`pcsp.materials.air`, ideal gas at the constant
operating pressure ``pressure``; the bed pressure drop is assumed small
against it) and the solid ``cp(T)`` of the
:class:`~pcsp.materials.particles.ParticleMaterial`. ``property_mode =
"constant"`` freezes every property at ``T_property`` (needed for the
analytic Schumann solution).

Numerics
--------
Finite volumes on ``n_cells`` uniform cells (method of lines), integrated by
SciPy's variable-order BDF with a sparse finite-difference Jacobian (column
colouring of the known stencil, 8 RHS evaluations per Jacobian). Tolerances:
``rtol`` and ``atol = 0.01 rtol (T_hot - T_cold)`` on states that are all in
kelvin. The system is very stiff: the gas responds in milliseconds
(:math:`\\varepsilon\\rho_f c_f/h_v`) while the thermocline moves through the bed
over hours.

* State: :math:`T_f` [K] and the solid enthalpy :math:`H_s` (scaled to kelvin
  by a constant capacity). Advancing :math:`H_s` rather than :math:`T_s` keeps
  the solid energy a *linear* function of the state, which BDF conserves to
  round-off; with the enthalpy flux form below the whole discrete energy
  balance telescopes. Cumulative inflow, outflow and per-cell wall losses are
  integrated as extra states for an exact budget.
* Advection: second-order MUSCL reconstruction of the face temperature,
  upwinded in the flow direction, with the face enthalpy
  :math:`h_f(T_{face})` as the flux. The slope limiter is van Albada's,
  :math:`s = ab(a+b)/(a^2+b^2+\\delta^2)` for :math:`ab > 0` and 0 otherwise
  (:math:`a, b` the upwind and downwind differences; van Albada, van Leer &
  Roberts 1982, Astron. Astrophys. 108, 76-84), with
  :math:`\\delta = 10^{-6}(T_{hot} - T_{cold})` only to avoid 0/0. It is TVD
  (:math:`\\phi(r) \\le \\min(2r, 1.21)`), so the sharp front does not
  oscillate. It was chosen over the van Leer limiter because its smoother
  :math:`\\phi(r)` keeps BDF's Newton iteration well behaved: with van Leer,
  some tolerances made the step count jump 25-fold.
* Numerical diffusion: first-order upwinding would add a gas-side
  conductivity :math:`G c_f \\Delta x / 2`, i.e. a front diffusivity
  :math:`u_{th}\\Delta x/2` (:math:`u_{th} = G c_f / C_s`, the thermal front
  speed). The physical spreading of a Schumann front is equivalent to a
  diffusivity :math:`u_{th}^2 C_s/h_v` (substitute the quasi-steady gas
  balance :math:`T_f - T_s \\approx -(G c_f/h_v)\\partial_x T` into the summed
  energy equation). Their ratio is :math:`NTU_{cell}/2` with
  :math:`NTU_{cell} = h_v \\Delta x / (G c_f)`, which is of order one or more
  for rock beds, so upwinding would roughly double the front width. The
  MUSCL scheme's leading truncation error is :math:`O(\\Delta x^2)` and the
  limiter reduces it to first order only at extrema of the profile.
  Measured against the Anzelius solution, the order is 2.0 in both the L1 and
  max norms (``tests/models/storage/test_verification.py``).
* Ghost cells are linear extrapolations: through the known inlet face value
  :math:`T_{in}` upstream, and from the last two cells downstream, so the
  outlet face (and hence ``T_out``) is second order as well. The downstream
  extrapolation is not TVD-bounded. The overshoot it produces as a front
  leaves the bed was measured at < 1e-6 of the temperature range.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp
import xarray as xr
from scipy.integrate import solve_ivp

from pcsp.constants import P_ATM, T_ZERO_C
from pcsp.data.schema import Coord, Var, new_dataset
from pcsp.materials import air
from pcsp.materials.particles import ParticleMaterial
from pcsp.models import correlations as corr

MODEL_NAME = "packed_bed_schumann_1d"
MODEL_VERSION = "0.1.0"

THERMOCLINE_LEVELS = (0.1, 0.9)
"""Dimensionless solid-temperature levels bounding the thermocline [-]."""

# Gauss-Legendre rules. Four nodes integrate polynomials up to degree 7
# exactly, so the gas enthalpy of the quartic cp(T) fit in pcsp.materials.air
# is exact to round-off. rho_f*cp_f ~ cp/T is not a polynomial; with 16 nodes
# it agrees with adaptive quadrature to 1e-12 relative over 300-1600 K (tested).
_GL4 = np.polynomial.legendre.leggauss(4)
_GL16 = np.polynomial.legendre.leggauss(16)


@dataclass(frozen=True)
class PackedBedParams:
    """Complete, JSON-serialisable input of one charge or discharge phase.

    Attributes
    ----------
    material : bed particles; supplies ``rho_p``, ``d_p``, ``k_s``, ``cp(T)``
        and the void fraction ``material.void_fraction``
    length : bed length in the flow direction [m]
    diameter : bed (inner wall) diameter [m]
    T_hot : charging inlet temperature [K]
    T_cold : discharging inlet temperature, also the energy reference [K]
    mass_flow : gas mass flow rate, >= 0 [kg s^-1]
    duration : simulated time [s]
    mode : ``"charge"`` (gas at ``T_hot`` enters at z = 0) or ``"discharge"``
        (gas at ``T_cold`` enters at z = L, flowing towards z = 0)
    initial_T_s : initial solid temperature [K]: a scalar, or one value per
        cell ordered by z from the hot end. ``None`` means a fully discharged
        bed (``T_cold``) for a charge and a fully charged bed (``T_hot``) for a
        discharge.
    initial_T_f : initial gas temperature, same format; ``None`` -> equal to
        the solid
    pressure : operating (absolute) pressure for the gas density [Pa]
    property_mode : ``"variable"`` (T-dependent air and solid cp) or
        ``"constant"`` (all properties at ``T_property``)
    T_property : property temperature in constant mode [K]; ``None`` ->
        ``(T_hot + T_cold) / 2``
    h_v : fixed volumetric heat-transfer coefficient [W m^-3 K^-1] replacing
        the Wakao-Kaguei closure (for verification); ``None`` -> correlation
    intraparticle_conduction : apply the ``1/h_eff = 1/h + d_p/(10 k_s)``
        correction (ignored when ``h_v`` is given)
    solid_axial_conduction : include Zehner-Schlunder axial conduction
    U_wall : wall loss coefficient on the inner wall area [W m^-2 K^-1]
    T_ambient : ambient temperature for wall losses [K]
    n_cells : number of finite volumes
    n_output : number of equally spaced output instants (including t = 0)
    rtol : relative tolerance of the time integrator
    """

    material: ParticleMaterial
    length: float
    diameter: float
    T_hot: float
    T_cold: float
    mass_flow: float
    duration: float
    mode: Literal["charge", "discharge"] = "charge"
    initial_T_s: float | tuple[float, ...] | None = None
    initial_T_f: float | tuple[float, ...] | None = None
    pressure: float = P_ATM
    property_mode: Literal["variable", "constant"] = "variable"
    T_property: float | None = None
    h_v: float | None = None
    intraparticle_conduction: bool = True
    solid_axial_conduction: bool = False
    U_wall: float = 0.0
    T_ambient: float = 298.15
    n_cells: int = 200
    n_output: int = 121
    rtol: float = 1e-6

    def __post_init__(self) -> None:
        for name in ("length", "diameter", "duration", "pressure", "rtol"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.mass_flow < 0.0:
            raise ValueError(f"mass_flow must be >= 0, got {self.mass_flow}")
        if self.U_wall < 0.0:
            raise ValueError(f"U_wall must be >= 0, got {self.U_wall}")
        if self.h_v is not None and self.h_v <= 0.0:
            raise ValueError(f"h_v must be positive, got {self.h_v}")
        if not self.T_hot > self.T_cold > 0.0:
            raise ValueError("need T_hot > T_cold > 0 K")
        if self.mode not in ("charge", "discharge"):
            raise ValueError(f"mode must be 'charge' or 'discharge', got {self.mode!r}")
        if self.property_mode not in ("variable", "constant"):
            raise ValueError(f"unknown property_mode {self.property_mode!r}")
        if self.n_cells < 3 or self.n_output < 2:
            raise ValueError("need n_cells >= 3 and n_output >= 2")
        for name in ("initial_T_s", "initial_T_f"):
            value = getattr(self, name)
            if isinstance(value, list):  # keep the dataclass hashable and frozen
                object.__setattr__(self, name, tuple(float(v) for v in value))
                value = getattr(self, name)
            if isinstance(value, tuple) and len(value) != self.n_cells:
                raise ValueError(f"{name} has {len(value)} values, expected n_cells")
        if self.property_mode == "variable":
            lo, hi = self.T_cold, self.T_hot
            if lo < air.T_MIN or hi > air.T_MAX:
                raise ValueError(f"air properties are valid for {air.T_MIN}-{air.T_MAX} K only")
            if self.material.cp_b != 0.0 and lo <= T_ZERO_C:
                raise ValueError("power-law solid cp needs temperatures above 273.15 K")

    @property
    def area(self) -> float:
        """Bed cross-section [m^2]."""
        return np.pi * self.diameter**2 / 4.0

    @property
    def mass_flux(self) -> float:
        """Superficial mass flux G [kg m^-2 s^-1]."""
        return self.mass_flow / self.area

    @property
    def T_inlet(self) -> float:
        """Gas inlet temperature of this phase [K]."""
        return self.T_hot if self.mode == "charge" else self.T_cold

    @property
    def T_prop(self) -> float:
        """Property temperature of the constant-property mode [K]."""
        if self.T_property is not None:
            return self.T_property
        return 0.5 * (self.T_hot + self.T_cold)

    def with_(self, **changes: object) -> PackedBedParams:
        """Copy with some fields replaced."""
        return dataclasses.replace(self, **changes)


@dataclass(frozen=True)
class Phase:
    """One step of a cycle schedule for :func:`run_cycles`."""

    mode: Literal["charge", "discharge"]
    duration: float
    mass_flow: float


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def gas_properties(params: PackedBedParams, T: np.ndarray) -> tuple[np.ndarray, ...]:
    """Gas ``rho`` [kg m^-3], ``cp`` [J kg^-1 K^-1], ``mu`` [Pa s], ``k`` [W m^-1 K^-1]."""
    T = np.asarray(T, dtype=float)
    if params.property_mode == "constant":
        Tp = params.T_prop
        ones = np.ones_like(T)
        return (
            air.rho(Tp, params.pressure) * ones,
            air.cp(Tp) * ones,
            air.mu(Tp) * ones,
            air.k(Tp) * ones,
        )
    return air.rho(T, params.pressure), air.cp(T), air.mu(T), air.k(T)


def _gauss(f, a: float, b: np.ndarray, rule: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Integral of ``f`` from ``a`` to each ``b`` by Gauss-Legendre."""
    nodes, weights = rule
    half = 0.5 * (np.asarray(b, dtype=float) - a)
    mid = 0.5 * (np.asarray(b, dtype=float) + a)
    return half * (f(mid[..., None] + half[..., None] * nodes) @ weights)


def gas_enthalpy(params: PackedBedParams, T: np.ndarray) -> np.ndarray:
    """Gas specific enthalpy relative to ``T_cold`` [J kg^-1]."""
    if params.property_mode == "constant":
        return air.cp(params.T_prop) * (np.asarray(T, dtype=float) - params.T_cold)
    return _gauss(air.cp, params.T_cold, T, _GL4)


def gas_energy_density(params: PackedBedParams, T: np.ndarray) -> np.ndarray:
    """Gas energy per bed volume relative to ``T_cold``, ``eps * int rho cp dT`` [J m^-3].

    With quasi-steady continuity the gas capacity term of the energy equation
    is ``eps rho_f(T) c_f(T) dT_f/dt``; this is the energy whose rate of change
    that term is, so the energy budget closes exactly.
    """
    eps = params.material.void_fraction
    if params.property_mode == "constant":
        rho, cp, _, _ = gas_properties(params, params.T_prop)
        return eps * rho * cp * (np.asarray(T, dtype=float) - params.T_cold)
    return eps * _gauss(lambda x: air.rho(x, params.pressure) * air.cp(x), params.T_cold, T, _GL16)


def _solid_cp_a_b(params: PackedBedParams) -> tuple[float, float]:
    """Power-law coefficients ``cp = a (T - 273.15)^b`` of the solid in the active mode."""
    if params.property_mode == "constant":
        return float(params.material.cp(params.T_prop)), 0.0
    return params.material.cp_a, params.material.cp_b


def solid_enthalpy_density(params: PackedBedParams, T: np.ndarray) -> np.ndarray:
    """Solid enthalpy per bed volume relative to ``T_cold``, :math:`H_s(T)` [J m^-3]."""
    a, b = _solid_cp_a_b(params)
    n = 1.0 + b
    rho_s = (1.0 - params.material.void_fraction) * params.material.rho_p
    Tc = np.asarray(T, dtype=float) - T_ZERO_C
    return rho_s * a / n * (Tc**n - (params.T_cold - T_ZERO_C) ** n)


def solid_temperature(params: PackedBedParams, H: np.ndarray) -> np.ndarray:
    """Inverse of :func:`solid_enthalpy_density` (closed form) [K]."""
    a, b = _solid_cp_a_b(params)
    n = 1.0 + b
    rho_s = (1.0 - params.material.void_fraction) * params.material.rho_p
    base = n * np.asarray(H, dtype=float) / (rho_s * a) + (params.T_cold - T_ZERO_C) ** n
    return T_ZERO_C + np.maximum(base, 0.0) ** (1.0 / n)


def volumetric_heat_transfer_coefficient(params: PackedBedParams, T_f: np.ndarray) -> np.ndarray:
    """Fluid-solid coefficient :math:`h_v` [W m^-3 K^-1] at gas temperature ``T_f``.

    ``h_v = 6 (1 - eps) / d_p * h_eff`` with Wakao-Kaguei ``h`` and, if
    enabled, the intra-particle series resistance (module docstring).
    """
    T_f = np.asarray(T_f, dtype=float)
    if params.h_v is not None:
        return params.h_v * np.ones_like(T_f)
    mat = params.material
    _, cp, mu, k = gas_properties(params, T_f)
    Re = corr.reynolds_particle(params.mass_flux, mat.d_p, 1.0, mu)  # rho_f u_s = G
    h = corr.nu_wakao_kaguei(Re, cp * mu / k) * k / mat.d_p
    if params.intraparticle_conduction:
        h = h / (1.0 + h * mat.d_p / (10.0 * mat.k_s))
    return 6.0 * (1.0 - mat.void_fraction) / mat.d_p * h


def pressure_gradient(params: PackedBedParams, T_f: np.ndarray) -> np.ndarray:
    """Ergun pressure gradient at local gas temperature, ``u_s = G / rho_f(T_f)`` [Pa m^-1]."""
    rho, _, mu, _ = gas_properties(params, T_f)
    mat = params.material
    return corr.ergun_pressure_gradient(params.mass_flux / rho, mat.d_p, mat.void_fraction, rho, mu)


def thermal_front_speed(params: PackedBedParams) -> float:
    """Speed of the thermocline for ``h_v -> inf``, constant properties [m s^-1].

    ``u_th = G c_f / (eps rho_f c_f + (1 - eps) rho_s c_s)`` at ``T_prop``.
    """
    rho, cp, _, _ = gas_properties(params, params.T_prop)
    a, _ = _solid_cp_a_b(params.with_(property_mode="constant"))
    eps = params.material.void_fraction
    capacity = eps * rho * cp + (1.0 - eps) * params.material.rho_p * a
    return float(params.mass_flux * cp / capacity)


def thermocline_thickness(z: np.ndarray, theta: np.ndarray) -> float:
    """Distance between the 0.9 and 0.1 levels of a dimensionless profile [m].

    ``theta = (T_s - T_cold) / (T_hot - T_cold)`` sampled at ``z`` (measured
    from the hot end). Each level's position is the first downward crossing
    from the hot end, by linear interpolation. Returns NaN unless both levels
    are crossed inside the sampled bed.
    """
    hi, lo = THERMOCLINE_LEVELS[1], THERMOCLINE_LEVELS[0]

    def first_crossing(level: float) -> float:
        below = np.nonzero(theta <= level)[0]
        if below.size == 0 or below[0] == 0:
            return np.nan
        j = below[0]
        frac = (theta[j - 1] - level) / (theta[j - 1] - theta[j])
        return float(z[j - 1] + frac * (z[j] - z[j - 1]))

    return first_crossing(lo) - first_crossing(hi)


# ---------------------------------------------------------------------------
# Discretisation
# ---------------------------------------------------------------------------


def _van_albada_slope(a: np.ndarray, b: np.ndarray, delta2: float) -> np.ndarray:
    """Limited slope ``ab(a+b)/(a^2+b^2+delta2)`` if ``ab > 0`` else 0 (van Albada 1982)."""
    return np.where(a * b > 0.0, a * b * (a + b) / (a * a + b * b + delta2), 0.0)


class _Semidiscrete:
    """Right-hand side of the method-of-lines system in the flow coordinate.

    State layout (``N = n_cells``, cell 0 at the inlet):
    ``[T_f (N) | H_s/C_ref (N) | loss/C_ref (N) | E_in/(C_ref L), E_out/(C_ref L)]``
    with ``C_ref = (1-eps) rho_s cp_s(T_prop)`` so every state is in kelvin.
    """

    def __init__(self, params: PackedBedParams) -> None:
        self.p = params
        self.N = params.n_cells
        self.dx = params.length / self.N
        self.G = params.mass_flux
        self.eps = params.material.void_fraction
        a, _ = _solid_cp_a_b(params.with_(property_mode="constant"))
        self.C_ref = (1.0 - self.eps) * params.material.rho_p * a
        self.h_in = float(gas_enthalpy(params, np.array(params.T_inlet)))
        self.loss_per_K = 4.0 * params.U_wall / params.diameter
        self.limiter_delta2 = (1e-6 * (params.T_hot - params.T_cold)) ** 2

    def unpack(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        N = self.N
        return y[:N], solid_temperature(self.p, y[N : 2 * N] * self.C_ref)

    def face_temperatures(self, T_f: np.ndarray) -> np.ndarray:
        """Gas temperature at the N+1 faces (MUSCL, van Albada limiter, upwind)."""
        T_in = self.p.T_inlet
        ext = np.concatenate(([2.0 * T_in - T_f[0]], T_f, [2.0 * T_f[-1] - T_f[-2]]))  # ghosts
        slope = _van_albada_slope(ext[1:-1] - ext[:-2], ext[2:] - ext[1:-1], self.limiter_delta2)
        return np.concatenate(([T_in], T_f + 0.5 * slope))

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        p, dx = self.p, self.dx
        T_f, T_s = self.unpack(y)
        rho, cp, _, _ = gas_properties(p, T_f)
        h_face = gas_enthalpy(p, self.face_temperatures(T_f))
        h_face[0] = self.h_in
        exchange = volumetric_heat_transfer_coefficient(p, T_f) * (T_s - T_f)
        loss = self.loss_per_K * (T_f - p.T_ambient)
        dT_f = (self.G * (h_face[:-1] - h_face[1:]) / dx + exchange - loss) / (self.eps * rho * cp)
        dH_s = -exchange
        if p.solid_axial_conduction:
            T_face = 0.5 * (T_s[1:] + T_s[:-1])
            k_f = gas_properties(p, T_face)[3]
            k_eff = corr.k_eff_zehner_schlunder(k_f, p.material.k_s, self.eps)
            flux = np.concatenate(([0.0], -k_eff * np.diff(T_s) / dx, [0.0]))
            dH_s = dH_s + (flux[:-1] - flux[1:]) / dx
        scale = self.C_ref * p.length
        return np.concatenate(
            (
                dT_f,
                dH_s / self.C_ref,
                loss / self.C_ref,
                [self.G * self.h_in / scale, self.G * h_face[-1] / scale],
            )
        )

    def jacobian(self, t: float, y: np.ndarray) -> sp.csc_matrix:
        """Forward-difference Jacobian using column groups (graph colouring).

        Gas column ``j`` influences gas rows ``j-1..j+2`` (MUSCL stencil),
        solid row ``j``, loss row ``j`` and the outlet-flux row; solid column
        ``j`` influences gas row ``j`` and solid rows ``j-1..j+1``. Columns
        five (gas) or three (solid) apart therefore never share a row and can
        be perturbed together: 8 RHS evaluations per Jacobian whatever ``N``.
        The accumulator states influence nothing, so their columns are zero.
        """
        N = self.N
        n = y.size
        f0 = self.rhs(t, y)
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        cells = np.arange(N)
        for offset, period, reach in ((0, 5, (-1, 2)), (N, 3, (-1, 1))):
            for colour in range(period):
                j = cells[colour::period]
                step = np.sqrt(np.finfo(float).eps) * np.maximum(np.abs(y[offset + j]), 1.0)
                y_pert = y.copy()
                y_pert[offset + j] += step
                df = (self.rhs(t, y_pert) - f0)[None, :]
                for block, lo, hi in ((0, *reach), (N, -1, 1), (2 * N, 0, 0)):
                    for d in range(lo, hi + 1):
                        r = j + d
                        ok = (r >= 0) & (r < N)
                        rows.append(block + r[ok])
                        cols.append(offset + j[ok])
                        vals.append(df[0, block + r[ok]] / step[ok])
                if offset == 0:  # outlet enthalpy flux depends on the last gas cells
                    last = j[j >= N - 3]
                    rows.append(np.full(last.size, n - 1))
                    cols.append(last)
                    vals.append(df[0, n - 1] / step[j >= N - 3])
        J = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
        )
        return J.tocsc()


def _initial_profile(value: float | tuple[float, ...] | None, default: float, n: int) -> np.ndarray:
    if value is None:
        return np.full(n, default)
    return np.broadcast_to(np.asarray(value, dtype=float), (n,)).copy()


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------


def simulate(params: PackedBedParams) -> xr.Dataset:
    """Simulate one charge or discharge phase.

    Returns a :func:`pcsp.data.schema.new_dataset` dataset (component
    ``storage_bin``, fidelity ``reduced``). Energies are referenced to the bed
    and gas at ``T_cold``. Time series: ``T_f``, ``T_s`` (time, z), ``T_out``,
    ``energy_stored``, ``thermocline_thickness``, ``pressure_drop``,
    ``pumping_power``. Scalars: see each variable's ``long_name``.
    """
    model = _Semidiscrete(params)
    N, L = params.n_cells, params.length
    z = (np.arange(N) + 0.5) * L / N  # from the hot end
    default_T = params.T_cold if params.mode == "charge" else params.T_hot
    T_s0 = _initial_profile(params.initial_T_s, default_T, N)
    T_f0 = T_s0 if params.initial_T_f is None else _initial_profile(params.initial_T_f, 0.0, N)
    flip = params.mode == "discharge"  # flow coordinate x = L - z
    if flip:
        T_s0, T_f0 = T_s0[::-1], T_f0[::-1]

    y0 = np.concatenate(
        (T_f0, solid_enthalpy_density(params, T_s0) / model.C_ref, np.zeros(N), [0.0, 0.0])
    )
    t_out = np.linspace(0.0, params.duration, params.n_output)
    atol = 1e-2 * params.rtol * (params.T_hot - params.T_cold)
    sol = solve_ivp(
        model.rhs,
        (0.0, params.duration),
        y0,
        method="BDF",
        t_eval=t_out,
        rtol=params.rtol,
        atol=atol,
        jac=model.jacobian,
    )
    if not sol.success:
        raise RuntimeError(f"packed-bed integration failed: {sol.message}")

    Y = sol.y.T  # (time, state)
    T_f = Y[:, :N]
    H_s = Y[:, N : 2 * N] * model.C_ref
    T_s = solid_temperature(params, H_s)
    T_out = np.array([model.face_temperatures(row)[-1] for row in T_f])  # outlet face
    if flip:
        T_f, T_s, H_s = T_f[:, ::-1], T_s[:, ::-1], H_s[:, ::-1]

    A, dz = params.area, L / N
    energy_stored = A * dz * np.sum(H_s + gas_energy_density(params, T_f), axis=1)
    scale = model.C_ref * L * A
    E_in, E_out = Y[-1, -2] * scale, Y[-1, -1] * scale
    E_loss = model.C_ref * A * dz * float(np.sum(Y[-1, 2 * N : 3 * N]))
    dE = energy_stored[-1] - energy_stored[0]
    residual = dE - (E_in - E_out - E_loss)

    capacity = (
        A
        * L
        * float(
            solid_enthalpy_density(params, np.array(params.T_hot))
            + gas_energy_density(params, np.array(params.T_hot))
        )
    )
    if params.mode == "charge":
        efficiency = dE / E_in if E_in > 0.0 else np.nan
        eff_name = "charge efficiency: stored-energy gain / gas enthalpy supplied (ref. T_cold)"
    else:
        efficiency = E_out / (-dE) if dE < 0.0 else np.nan
        eff_name = "discharge efficiency: gas enthalpy recovered (ref. T_cold) / stored-energy drop"

    theta_s = (T_s - params.T_cold) / (params.T_hot - params.T_cold)
    thickness = np.array([thermocline_thickness(z, row) for row in theta_s])
    dp_dz = pressure_gradient(params, T_f)
    rho_f = gas_properties(params, T_f)[0]
    pressure_drop = dz * np.sum(dp_dz, axis=1)
    pumping_power = params.mass_flow * dz * np.sum(dp_dz / rho_f, axis=1)
    pumping_energy = float(np.trapezoid(pumping_power, t_out))

    lo, hi = THERMOCLINE_LEVELS
    data_vars = {
        "T_f": Var(("time", "z"), T_f, "K", "gas temperature (cell average)"),
        "T_s": Var(("time", "z"), T_s, "K", "solid temperature (cell average, particle mean)"),
        "T_out": Var("time", T_out, "K", "gas temperature leaving the bed"),
        "energy_stored": Var(
            "time", energy_stored, "J", "solid + gas thermal energy of the bed relative to T_cold"
        ),
        "thermocline_thickness": Var(
            "time",
            thickness,
            "m",
            f"distance between solid theta={hi} and theta={lo} levels, "
            "theta=(T_s-T_cold)/(T_hot-T_cold); NaN if not both inside the bed",
        ),
        "pressure_drop": Var("time", pressure_drop, "Pa", "Ergun pressure drop across the bed"),
        "pumping_power": Var(
            "time",
            pumping_power,
            "W",
            "hydraulic pumping power, mass_flow * integral of (dp/dz)/rho_f dz (no fan efficiency)",
        ),
        "capacity": Var((), capacity, "J", "energy stored at uniform T_hot relative to T_cold"),
        "energy_in": Var((), E_in, "J", "gas enthalpy entering the bed (ref. T_cold)"),
        "energy_out": Var((), E_out, "J", "gas enthalpy leaving the bed (ref. T_cold)"),
        "energy_loss": Var((), E_loss, "J", "heat lost through the wall"),
        "energy_balance_residual": Var(
            (),
            residual,
            "J",
            "stored-energy change - (energy_in - energy_out - energy_loss); numerical error",
        ),
        "state_of_charge": Var(
            (), energy_stored[-1] / capacity, "1", "final energy_stored/capacity"
        ),
        "phase_efficiency": Var((), efficiency, "1", eff_name),
        "pumping_energy": Var(
            (), pumping_energy, "J", "time integral of pumping_power (trapezoid on output times)"
        ),
    }
    coords = {
        "time": Coord(t_out, "s", "time since start of phase"),
        "z": Coord(z, "m", "cell-centre distance from the hot end of the bed"),
    }
    return new_dataset(
        component="storage_bin",
        fidelity="reduced",
        model=MODEL_NAME,
        model_version=MODEL_VERSION,
        params=params,
        coords=coords,
        data_vars=data_vars,
        attrs={"mode": params.mode, "n_rhs_evaluations": int(sol.nfev)},
    )


def run_cycles(params: PackedBedParams, phases: list[Phase]) -> list[xr.Dataset]:
    """Chain charge/discharge phases, each starting from the previous end state.

    The first phase starts from ``params.initial_T_s`` / ``initial_T_f``
    (``None`` meaning the fully discharged or charged bed for that phase's
    mode). Every other field of ``params`` is shared; ``mode``, ``duration``
    and ``mass_flow`` come from each :class:`Phase`. Each returned dataset
    records its own complete input, including its initial profiles.
    """
    results: list[xr.Dataset] = []
    T_s0, T_f0 = params.initial_T_s, params.initial_T_f
    for phase in phases:
        ds = simulate(
            params.with_(
                mode=phase.mode,
                duration=phase.duration,
                mass_flow=phase.mass_flow,
                initial_T_s=T_s0,
                initial_T_f=T_f0,
            )
        )
        results.append(ds)
        T_s0 = tuple(float(v) for v in ds["T_s"].values[-1])
        T_f0 = tuple(float(v) for v in ds["T_f"].values[-1])
    return results
