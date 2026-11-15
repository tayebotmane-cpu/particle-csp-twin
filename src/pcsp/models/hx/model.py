"""Steady reduced-order model of a moving packed-bed particle-to-sCO2 heat exchanger.

Geometry (one *unit cell*: one particle channel and one plate)
----------------------------------------------------------------
Hot particles flow down, as a dense plug, through a channel of gap ``2b``
between two vertical plates of height ``H`` and width ``W``. Each plate
carries a row of circular sCO2 channels of diameter ``d`` at pitch ``p``
across the width, with metal thickness ``t`` between a channel and each plate
face. sCO2 flows up (global counterflow) along ``z``. A plate is shared by
the two particle channels on its faces, so per particle channel the heat
transfer area is :math:`A = 2 H W` (both faces) and the sCO2 flow is that of
one plate, :math:`n_{ch} = W / p` channels. All extensive outputs (duty,
mass flows) are per unit cell.

Local overall coefficient
-------------------------
Referred to the particle-side wall area, with resistances in series:

.. math::
    \\frac{1}{U(z)} = \\underbrace{\\frac{D_h}{Nu(z)\\,k_{eff}}}_{\\text{bed (Graetz)}}
      + \\underbrace{R_w}_{\\text{near wall}}
      + \\underbrace{\\frac{t}{k_m}}_{\\text{metal}}
      + \\underbrace{\\frac{2 W}{h_{CO_2} \\, n_{ch} \\pi d}}_{\\text{sCO}_2} ,

where :math:`D_h = 4b`, ``Nu`` is the slug-flow thermal-entry Nusselt
number of :mod:`pcsp.models.hx.graetz` averaged over each cell, ``k_eff``
and ``R_w`` come from the chosen :class:`~pcsp.models.hx.closures.BedClosure`,
and :math:`h_{CO_2}` from :mod:`pcsp.models.hx.channel` on the channel
wetted area :math:`n_{ch}\\pi d` per unit height. The metal is a plane wall of
thickness ``t`` (2-D conduction around the channels and fin effects are
neglected).

Thermal entry with temperature-dependent properties
--------------------------------------------------
If :math:`\\alpha = k_{eff}/(\\rho_b c_p)` varies along ``z`` only, the bed
equation :math:`u\\,\\partial_z T = \\alpha(z)\\,\\partial_y^2 T` becomes the
constant-property one in the stretched coordinate
:math:`\\xi(z) = \\int_0^z \\alpha \\, dz' / (u b^2)`; the model uses this
:math:`\\xi`, with :math:`\\alpha` at the local bed bulk temperature. The
entry starts at the particle inlet ``z = 0``. The Graetz solution assumes a
uniform wall condition (``"T"`` or ``"H"``); the real condition is neither.
``"H"`` is the default, following Adapa et al. (2024); in near-balanced
counterflow the wall heat flux is close to uniform.

Energy balance and outputs
--------------------------
Solved with :func:`pcsp.models.hx.counterflow.solve_counterflow`, as
enthalpy balances with :meth:`ParticleMaterial.enthalpy` and real-gas CO2
enthalpy from :mod:`pcsp.materials.sco2` (one isobar at the inlet pressure;
the sCO2 pressure drop is far below 1 % of the pressure).

* Effectiveness :math:`\\varepsilon = Q / Q_{max}` with
  :math:`Q_{max} = \\min[\\dot m_p (h_p(T_{p,in}) - h_p(T_{c,in})),\\;
  \\dot m_c (h_c(T_{p,in}) - h_c(T_{c,in}))]` (enthalpy form, valid for
  variable heat capacity).
* Mean overall coefficient :math:`\\bar U = Q / (A \\, \\Delta T_{lm})`, the
  definition used by Albrecht & Ho (2019, eq. 13-14); the area average of
  ``U(z)`` is also reported.
* sCO2 pressure drop: Darcy-Weisbach friction (Petukhov factor) summed over
  cells plus the acceleration term :math:`G^2 (1/\\rho_{out} - 1/\\rho_{in})`;
  inlet/outlet and header losses are not included.

Particle conduction along ``z``, heat loss to the surroundings and
maldistribution between channels are neglected.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xarray as xr

from pcsp.data.schema import Coord, Var, new_dataset
from pcsp.materials import sco2
from pcsp.materials.particles import CARBO_HSP_40_70, ParticleMaterial
from pcsp.models.hx import graetz
from pcsp.models.hx.channel import friction_factor_petukhov, nusselt_gnielinski
from pcsp.models.hx.closures import BedClosure, StagnantBed
from pcsp.models.hx.counterflow import CounterflowSolution, solve_counterflow

MODEL_NAME = "moving_bed_graetz_1d"
MODEL_VERSION = "0.1.0"

SCO2_TABLE_T_MIN = 300.0
SCO2_TABLE_T_MAX = 1100.0
"""Temperature range [K] of the sCO2 property table used by the model."""


@dataclass(frozen=True)
class MovingBedHXParams:
    """Inputs of :func:`simulate`. All values SI, per unit cell (see module doc).

    Attributes
    ----------
    m_dot_particle : particle mass flow per particle channel [kg s^-1]
    m_dot_co2 : sCO2 mass flow per plate [kg s^-1]
    T_particle_in : particle inlet temperature (top, z = 0) [K]
    T_co2_in : sCO2 inlet temperature (bottom, z = H) [K]
    p_co2 : sCO2 inlet pressure [Pa]
    particle : particle material (cp, density, packing, d_p, k_s, emissivity)
    closure : particle-side closure (a frozen dataclass from
        :mod:`pcsp.models.hx.closures`)
    height : plate height H, flow length [m]
    width : plate width W [m]
    channel_gap : particle channel gap 2b [m]
    wall_thickness : metal between plate face and sCO2 channel t [m]
    wall_conductivity : metal conductivity k_m [W m^-1 K^-1]. The default
        23 W/m/K is a *nominal, unverified* value for AISI 316 near 950 K
        (not checked against a primary table).
    co2_channel_diameter : sCO2 channel diameter d [m]
    co2_channel_pitch : centre-to-centre spacing of sCO2 channels p [m]
    wall_bc : Graetz wall condition, ``"H"`` (uniform flux) or ``"T"``
    n_cells : number of cells along z
    """

    m_dot_particle: float
    m_dot_co2: float
    T_particle_in: float
    T_co2_in: float
    p_co2: float = 20e6
    particle: ParticleMaterial = CARBO_HSP_40_70
    closure: BedClosure = field(default_factory=StagnantBed)
    height: float = 1.0
    width: float = 0.5
    channel_gap: float = 6e-3
    wall_thickness: float = 1e-3
    wall_conductivity: float = 23.0
    co2_channel_diameter: float = 1e-3
    co2_channel_pitch: float = 2e-3
    wall_bc: str = "H"
    n_cells: int = 200

    def __post_init__(self) -> None:
        positive = (
            "m_dot_particle",
            "m_dot_co2",
            "p_co2",
            "height",
            "width",
            "channel_gap",
            "wall_thickness",
            "wall_conductivity",
            "co2_channel_diameter",
            "co2_channel_pitch",
        )
        for name in positive:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.co2_channel_pitch <= self.co2_channel_diameter:
            raise ValueError("co2_channel_pitch must exceed co2_channel_diameter")
        if self.T_particle_in <= self.T_co2_in:
            raise ValueError("particles must enter hotter than the sCO2")
        if self.wall_bc not in graetz.NU_FULLY_DEVELOPED:
            raise ValueError(f"wall_bc must be 'H' or 'T', got {self.wall_bc!r}")
        if self.n_cells < 2:
            raise ValueError("n_cells must be >= 2")
        if not (isinstance(self.closure, BedClosure) and dataclasses.is_dataclass(self.closure)):
            raise TypeError("closure must be a dataclass implementing the BedClosure protocol")

    @property
    def n_co2_channels(self) -> float:
        """sCO2 channels per plate, ``W / p`` [-]."""
        return self.width / self.co2_channel_pitch

    @property
    def heat_transfer_area(self) -> float:
        """Particle-side wall area per unit cell, ``2 H W`` [m^2]."""
        return 2.0 * self.height * self.width

    @property
    def particle_velocity(self) -> float:
        """Bed plug velocity ``m_dot / (rho_bulk * gap * W)`` [m s^-1]."""
        return self.m_dot_particle / (self.particle.rho_bulk * self.channel_gap * self.width)


def _co2_table(params: MovingBedHXParams) -> sco2.SCO2Isobar:
    return sco2.isobar(params.p_co2, SCO2_TABLE_T_MIN, SCO2_TABLE_T_MAX)


def local_coefficients(
    params: MovingBedHXParams, T_particle: np.ndarray, T_co2: np.ndarray
) -> dict[str, np.ndarray]:
    """Cell quantities for given face temperature profiles (length ``n_cells + 1``).

    Returns cell-centred arrays: ``k_eff``, ``r_near_wall``, ``xi_faces``
    (length ``n_cells + 1``), ``nu_bed``, ``h_bed``, ``re_co2``, ``h_co2``,
    ``r_co2`` (referred to the particle-side area), ``r_metal`` (scalar) and
    ``U``.
    """
    p = params
    co2 = _co2_table(p)
    dz = p.height / p.n_cells
    b = 0.5 * p.channel_gap
    u = p.particle_velocity
    Tp = 0.5 * (T_particle[:-1] + T_particle[1:])
    Tc = 0.5 * (T_co2[:-1] + T_co2[1:])

    k_eff = p.closure.k_eff(Tp, u, p.particle)
    r_near_wall = p.closure.r_wall(Tp, u, p.particle)
    alpha = k_eff / (p.particle.rho_bulk * p.particle.cp(Tp))
    xi_faces = np.concatenate([[0.0], np.cumsum(alpha * dz / (u * b**2))])
    nu_bed = graetz.nusselt_cell(xi_faces[:-1], xi_faces[1:], p.wall_bc)
    h_bed = nu_bed * k_eff / (4.0 * b)

    d = p.co2_channel_diameter
    m_channel = p.m_dot_co2 / p.n_co2_channels
    re_co2 = 4.0 * m_channel / (np.pi * d * co2.mu(Tc))
    h_co2 = nusselt_gnielinski(re_co2, co2.prandtl(Tc)) * co2.k(Tc) / d
    r_co2 = 2.0 * p.width / (h_co2 * p.n_co2_channels * np.pi * d)
    r_metal = p.wall_thickness / p.wall_conductivity

    U = 1.0 / (1.0 / h_bed + r_near_wall + r_metal + r_co2)
    return {
        "k_eff": k_eff,
        "r_near_wall": r_near_wall,
        "xi_faces": xi_faces,
        "nu_bed": nu_bed,
        "h_bed": h_bed,
        "re_co2": re_co2,
        "h_co2": h_co2,
        "r_co2": r_co2,
        "r_metal": np.full_like(Tp, r_metal),
        "U": U,
    }


def solve(params: MovingBedHXParams) -> CounterflowSolution:
    """Converged temperature profiles (particles hot, sCO2 cold)."""
    p = params
    co2 = _co2_table(p)
    cell_area = p.heat_transfer_area / p.n_cells

    def ua_cells(T_hot: np.ndarray, T_cold: np.ndarray) -> np.ndarray:
        return local_coefficients(p, T_hot, T_cold)["U"] * cell_area

    return solve_counterflow(
        n_cells=p.n_cells,
        m_hot=p.m_dot_particle,
        m_cold=p.m_dot_co2,
        h_hot=p.particle.enthalpy,
        cp_hot=p.particle.cp,
        h_cold=co2.h,
        cp_cold=co2.cp,
        T_hot_in=p.T_particle_in,
        T_cold_in=p.T_co2_in,
        ua_cells=ua_cells,
    )


def _log_mean(dT_a: float, dT_b: float) -> float:
    if np.isclose(dT_a, dT_b, rtol=1e-10, atol=0.0):
        return 0.5 * (dT_a + dT_b)
    return (dT_a - dT_b) / np.log(dT_a / dT_b)


def summarize(params: MovingBedHXParams, sol: CounterflowSolution) -> dict[str, Any]:
    """Scalar quantities of interest and profiles from a converged solution."""
    p = params
    co2 = _co2_table(p)
    local = local_coefficients(p, sol.T_hot, sol.T_cold)
    Tp, Tc = sol.T_hot, sol.T_cold
    Tp_in, Tp_out = Tp[0], Tp[-1]
    Tc_in, Tc_out = Tc[-1], Tc[0]

    q_particle = p.m_dot_particle * (p.particle.enthalpy(Tp_in) - p.particle.enthalpy(Tp_out))
    q_co2 = p.m_dot_co2 * float(co2.h(Tc_out) - co2.h(Tc_in))
    duty = float(np.sum(sol.Q_cells))
    q_max = min(
        p.m_dot_particle * (p.particle.enthalpy(Tp_in) - p.particle.enthalpy(Tc_in)),
        p.m_dot_co2 * float(co2.h(Tp_in) - co2.h(Tc_in)),
    )
    lmtd = _log_mean(Tp_in - Tc_out, Tp_out - Tc_in)

    dz = p.height / p.n_cells
    d = p.co2_channel_diameter
    G = p.m_dot_co2 / p.n_co2_channels / (0.25 * np.pi * d**2)
    Tc_cell = 0.5 * (Tc[:-1] + Tc[1:])
    rho_cell = co2.rho(Tc_cell)
    dp_friction = np.sum(friction_factor_petukhov(local["re_co2"]) * dz / d * G**2 / (2 * rho_cell))
    dp_accel = G**2 * (1.0 / co2.rho(Tc_out) - 1.0 / co2.rho(Tc_in))

    cell_area = p.heat_transfer_area / p.n_cells
    heat_flux = sol.Q_cells / cell_area
    T_wall = Tc_cell + heat_flux * (local["r_metal"] + local["r_co2"])
    return {
        "duty": duty,
        "q_particle": float(q_particle),
        "q_co2": q_co2,
        "energy_residual": float((q_particle - q_co2) / duty),
        "effectiveness": duty / float(q_max),
        "lmtd": float(lmtd),
        "U_mean": duty / (p.heat_transfer_area * lmtd),
        "U_area_mean": float(np.mean(local["U"])),
        "T_particle_out": float(Tp_out),
        "T_co2_out": float(Tc_out),
        "dp_co2": float(dp_friction + dp_accel),
        "heat_flux": heat_flux,
        "T_wall": T_wall,
        **local,
    }


def simulate(params: MovingBedHXParams) -> xr.Dataset:
    """Run the steady model; return profiles along ``z`` and scalar QoIs.

    ``z`` (cell faces) and ``z_cell`` (cell centres) are measured downward
    from the particle inlet at the top of the plate.
    """
    p = params
    sol = solve(p)
    s = summarize(p, sol)
    z = np.linspace(0.0, p.height, p.n_cells + 1)
    z_cell = 0.5 * (z[:-1] + z[1:])
    per_cell = "per unit cell (one particle channel + one plate)"
    data_vars = {
        "T_particle": Var("z", sol.T_hot, "K", "particle bulk temperature"),
        "T_co2": Var("z", sol.T_cold, "K", "sCO2 bulk temperature"),
        "inverse_graetz": Var(
            "z", s["xi_faces"] / 16.0, "1", "inverse Graetz number z/(D_h Pe), D_h = 2 gap"
        ),
        "U_local": Var(
            "z_cell", s["U"], "W m-2 K-1", "local overall coefficient (particle-side area)"
        ),
        "h_bed": Var("z_cell", s["h_bed"], "W m-2 K-1", "bed (Graetz) heat transfer coefficient"),
        "nu_bed": Var("z_cell", s["nu_bed"], "1", "bed Nusselt number h D_h / k_eff, D_h = 2 gap"),
        "k_eff": Var("z_cell", s["k_eff"], "W m-1 K-1", "effective bed conductivity"),
        "r_near_wall": Var("z_cell", s["r_near_wall"], "m2 K W-1", "near-wall bed resistance"),
        "h_co2": Var("z_cell", s["h_co2"], "W m-2 K-1", "sCO2 coefficient (channel wetted area)"),
        "re_co2": Var("z_cell", s["re_co2"], "1", "sCO2 channel Reynolds number"),
        "heat_flux": Var("z_cell", s["heat_flux"], "W m-2", "heat flux at the particle-side wall"),
        "T_wall": Var("z_cell", s["T_wall"], "K", "particle-side wall surface temperature"),
        "duty": Var((), s["duty"], "W", f"heat duty {per_cell}"),
        "effectiveness": Var((), s["effectiveness"], "1", "heat exchanger effectiveness Q/Q_max"),
        "U_mean": Var((), s["U_mean"], "W m-2 K-1", "mean overall coefficient Q/(A LMTD)"),
        "U_area_mean": Var((), s["U_area_mean"], "W m-2 K-1", "area-averaged local U"),
        "lmtd": Var((), s["lmtd"], "K", "log-mean temperature difference"),
        "T_particle_out": Var((), s["T_particle_out"], "K", "particle outlet temperature"),
        "T_co2_out": Var((), s["T_co2_out"], "K", "sCO2 outlet temperature"),
        "dp_co2": Var((), s["dp_co2"], "Pa", "sCO2 pressure drop (friction + acceleration)"),
        "particle_velocity": Var((), p.particle_velocity, "m s-1", "bed plug velocity"),
        "heat_transfer_area": Var((), p.heat_transfer_area, "m2", f"wall area {per_cell}"),
        "energy_residual": Var(
            (), s["energy_residual"], "1", "(Q_particle - Q_sCO2) / Q, energy balance closure"
        ),
    }
    return new_dataset(
        component="hx_moving_bed",
        fidelity="reduced",
        model=MODEL_NAME,
        model_version=MODEL_VERSION,
        params=p,
        coords={
            "z": Coord(z, "m", "depth below particle inlet (cell faces)"),
            "z_cell": Coord(z_cell, "m", "depth below particle inlet (cell centres)"),
        },
        data_vars=data_vars,
        attrs={"closure": p.closure.name, "wall_bc": p.wall_bc},
    )
