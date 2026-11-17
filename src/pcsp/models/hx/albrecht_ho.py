"""Reference cases from Albrecht & Ho for the moving-bed heat exchanger.

Two parameter sets live here.

1. The published design study (:func:`paper_params`)
----------------------------------------------------
Albrecht, K.J., Ho, C.K., 2019. Heat transfer models of moving packed-bed
particle-to-sCO2 heat exchangers. J. Sol. Energy Eng. 141, 031006,
doi:10.1115/1.4041546 (accepted manuscript of 2018 consulted). Their 2-D
continuum model of one particle channel, 1 m high and 0.5 m wide (1.0 m2 of
wall), cooling particles from 775 to 570 degC while heating sCO2 from 550 to
700 degC at 25 MPa.

Inputs printed in the paper (their Table 1 and text) and used as they are:
particle diameter 250 um, bulk density 2000 kg/m3, particle heat capacity
1200 J/kg/K, solid conductivity 2 W/m/K, bulk voidage 0.40, plate
conductivity 23 W/m/K, plate spacing 6 mm, sCO2 channel 0.5 mm wide. Results
printed in the paper (their Tables 2 and 3) are in :data:`PAPER_CASES`.

Not printed in the paper, and chosen here:

* The four constants of their bed closure: see
  :class:`pcsp.models.hx.closures.YagiKuniiBed`, the default closure of
  :func:`paper_params`.
* Plate thickness: 1 mm. It is 3 % of the non-particle resistance.
* sCO2 coefficient: the Gnielinski correlation of
  :mod:`pcsp.models.hx.channel`. Their sCO2 channel is a slot heated on both
  sides. A slot of width ``s`` has the same Reynolds number, coefficient and
  wetted area as the model's circular channels with diameter ``2 s`` at pitch
  ``pi s``, which is how it is entered here.

  Both choices are confirmed by the same group's transient study of this
  geometry (Fernandez-Torrijos, Albrecht & Ho, Applied Energy 226 (2018)
  595-606): its Table 2 gives a 1 mm plate, and it states that the sCO2
  coefficient is the Gnielinski correlation on a hydraulic diameter of twice
  the slot width, 600 W/m2/K at its design point (20 MPa, 26.7 g/s). The
  model gives 607 there (``tests/models/hx/test_model.py``).
* Interstitial gas: air.

How the comparison comes out is recorded in ``tests/models/hx/test_model.py``.
In short: with these inputs a converged 2-D solution of the paper's equations
(:mod:`pcsp.models.hx.conjugate`) gives an overall coefficient near 160
W/m2/K where the paper reports 144; the reduced model gives 165 with a
uniform-flux wall condition and 145 with a uniform-temperature one. The
paper's Tables 2 and 3 follow the uniform-temperature Nusselt number.

2. A generic operating point (:func:`nominal_params`)
-----------------------------------------------------
Assembled before the paper above was available, from the inputs of the same
group's later study (Albrecht & Ho, Solar Energy 178 (2019) 331-340, Tables
1-3: solid density 3300 kg/m3, solid fraction 0.55, particle diameter 280 um,
emissivity 0.9, 1 mm metal, 1 mm sCO2 channels at 2 mm pitch) with the
project's CARBO HSP heat capacity, the default stagnant-bed closure (which
includes radiation) and 20 MPa. It is **not** the case of the 2018 paper: its
overall coefficient of 235 W/m2/K differs from the paper's 144 because the
inputs and the closure differ. It is kept as the default operating point of
the model's own tests. Its mass flows are back-calculated from the design
temperatures and a coefficient of 144 W/m2/K.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

from scipy.optimize import root

from pcsp.constants import T_ZERO_C
from pcsp.materials import sco2
from pcsp.materials.particles import CARBO_HSP_40_70, ParticleMaterial
from pcsp.models.hx.closures import BedClosure, StagnantBed, YagiKuniiBed
from pcsp.models.hx.model import (
    SCO2_TABLE_T_MAX,
    SCO2_TABLE_T_MIN,
    MovingBedHXParams,
    solve,
)

U_REPORTED = 144.0
"""Overall coefficient of the paper's nominal case [W m^-2 K^-1] (Table 2)."""

H_PARTICLE_WALL_REPORTED = 182.0
"""Average particle-wall coefficient of the paper's nominal case [W m^-2 K^-1] (Table 2)."""

T_PARTICLE_IN = 775.0 + T_ZERO_C
T_PARTICLE_OUT = 570.0 + T_ZERO_C
T_CO2_IN = 550.0 + T_ZERO_C
T_CO2_OUT = 700.0 + T_ZERO_C


def design_lmtd() -> float:
    """Log-mean temperature difference of the design temperatures [K] (41.61 K in the paper)."""
    dT_hot_end = T_PARTICLE_IN - T_CO2_OUT
    dT_cold_end = T_PARTICLE_OUT - T_CO2_IN
    return (dT_hot_end - dT_cold_end) / math.log(dT_hot_end / dT_cold_end)


# --- 1. The published design study ---------------------------------------------------------

PAPER_P_CO2 = 25e6
"""sCO2 pressure of the paper's cases [Pa]."""

PAPER_EFFECTIVENESS = 0.915
PAPER_NTU = 5.03
"""Effectiveness and number of transfer units, the same for all cases (Tables 2 and 3)."""


@dataclass(frozen=True)
class PaperCase:
    """One column of the paper's Table 2 or 3 (per channel, 1.0 m2 of wall).

    Attributes
    ----------
    d_p : particle diameter [m]
    channel_gap : plate spacing [m]
    U : overall coefficient [W m^-2 K^-1]
    h_particle_wall : average particle-wall coefficient [W m^-2 K^-1]
    duty : heat transferred [W]
    m_dot_particle, m_dot_co2 : mass flows that give the design temperatures [kg s^-1]
    """

    d_p: float
    channel_gap: float
    U: float
    h_particle_wall: float
    duty: float
    m_dot_particle: float
    m_dot_co2: float


PAPER_CASES: dict[str, PaperCase] = {
    "nominal": PaperCase(250e-6, 6e-3, 144.0, 182.0, 5860.0, 23.8e-3, 31.3e-3),
    "d_p_50um": PaperCase(50e-6, 6e-3, 163.0, 208.0, 6630.0, 27.0e-3, 35.4e-3),
    "d_p_750um": PaperCase(750e-6, 6e-3, 110.0, 139.0, 4500.0, 18.3e-3, 24.0e-3),
    "gap_3mm": PaperCase(250e-6, 3e-3, 246.0, 315.0, 10000.0, 40.8e-3, 53.6e-3),
    "gap_9mm": PaperCase(250e-6, 9e-3, 100.0, 128.0, 4090.0, 16.6e-3, 21.8e-3),
}
"""Tables 2 (particle diameter) and 3 (plate spacing) of the paper."""

_PAPER_CO2_SLOT = 0.5e-3


def paper_particle(d_p: float = 250e-6) -> ParticleMaterial:
    """Particles of the paper's Table 1 (emissivity is not used by its closure)."""
    solid_fraction = 0.60
    return ParticleMaterial(
        name="albrecht_ho_2019_table1",
        rho_p=2000.0 / solid_fraction,
        d_p=d_p,
        k_s=2.0,
        emissivity=0.9,
        solar_absorptivity=0.9,
        cp_a=1200.0,
        cp_b=0.0,
        packing_fraction=solid_fraction,
        source="Albrecht & Ho, J. Sol. Energy Eng. 141 (2019) 031006, Table 1",
    )


def paper_params(
    case: str = "nominal",
    *,
    closure: BedClosure | None = None,
    wall_bc: str = "H",
    **overrides: Any,
) -> MovingBedHXParams:
    """Inputs of one of :data:`PAPER_CASES`, at the mass flows the paper reports."""
    ref = PAPER_CASES[case]
    values: dict[str, Any] = {
        "m_dot_particle": ref.m_dot_particle,
        "m_dot_co2": ref.m_dot_co2,
        "T_particle_in": T_PARTICLE_IN,
        "T_co2_in": T_CO2_IN,
        "p_co2": PAPER_P_CO2,
        "particle": paper_particle(ref.d_p),
        "closure": closure if closure is not None else YagiKuniiBed(),
        "height": 1.0,
        "width": 0.5,
        "channel_gap": ref.channel_gap,
        "wall_thickness": 1e-3,
        "wall_conductivity": 23.0,
        "co2_channel_diameter": 2.0 * _PAPER_CO2_SLOT,
        "co2_channel_pitch": math.pi * _PAPER_CO2_SLOT,
        "wall_bc": wall_bc,
    }
    values.update(overrides)
    return MovingBedHXParams(**values)


def at_design_temperatures(params: MovingBedHXParams) -> MovingBedHXParams:
    """``params`` with both mass flows adjusted to give the design outlet temperatures.

    The paper fixes the four temperatures (particles 775 to 570 degC, sCO2 550
    to 700 degC) and reports the mass flows that achieve them; this does the
    same for the reduced model.
    """

    def with_flows(scale: Any) -> MovingBedHXParams:
        return dataclasses.replace(
            params,
            m_dot_particle=params.m_dot_particle * float(scale[0]),
            m_dot_co2=params.m_dot_co2 * float(scale[1]),
        )

    def miss(scale: Any) -> list[float]:
        sol = solve(with_flows(scale))
        return [float(sol.T_hot[-1]) - T_PARTICLE_OUT, float(sol.T_cold[0]) - T_CO2_OUT]

    result = root(miss, [1.0, 1.0], tol=1e-9)
    if not result.success:
        raise RuntimeError(f"no mass flows give the design temperatures: {result.message}")
    return with_flows(result.x)


# --- 2. A generic operating point ----------------------------------------------------------

P_CO2 = 20e6

PARTICLE: ParticleMaterial = CARBO_HSP_40_70.with_(
    d_p=280e-6, packing_fraction=0.55, emissivity=0.9, k_s=2.0, rho_p=3300.0
)


def nominal_params(closure: BedClosure | None = None, **overrides: Any) -> MovingBedHXParams:
    """The generic operating point (section 2 of the module docstring)."""
    height, width = 1.0, 0.5
    duty = U_REPORTED * 2.0 * height * width * design_lmtd()
    co2 = sco2.isobar(P_CO2, SCO2_TABLE_T_MIN, SCO2_TABLE_T_MAX)
    m_particle = duty / (PARTICLE.enthalpy(T_PARTICLE_IN) - PARTICLE.enthalpy(T_PARTICLE_OUT))
    m_co2 = duty / float(co2.h(T_CO2_OUT) - co2.h(T_CO2_IN))
    values: dict[str, Any] = {
        "m_dot_particle": float(m_particle),
        "m_dot_co2": m_co2,
        "T_particle_in": T_PARTICLE_IN,
        "T_co2_in": T_CO2_IN,
        "p_co2": P_CO2,
        "particle": PARTICLE,
        "closure": closure if closure is not None else StagnantBed(),
        "height": height,
        "width": width,
        "channel_gap": 6e-3,
        "wall_thickness": 1e-3,
        "co2_channel_diameter": 1e-3,
        "co2_channel_pitch": 2e-3,
    }
    values.update(overrides)
    return MovingBedHXParams(**values)
