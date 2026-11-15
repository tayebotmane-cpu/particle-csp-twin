"""Nominal operating point of the Sandia single-bank moving-bed HX (Albrecht & Ho).

Validation target: the 2-D continuum model of Albrecht & Ho (ASME
PowerEnergy2017-3377; journal version J. Sol. Energy Eng. 141 (2019) 031006)
for a single channel bank reported an overall coefficient of 144 W m^-2 K^-1
and an average particle-wall coefficient of 182 W m^-2 K^-1 (conference
slides SAND2017-6879C, slide 12, OSTI 1459786). The full paper could not be
retrieved (OSTI was unreachable), so the parameter set below is assembled
from the sources listed per value.

Sourced values
--------------
* Plate height 1 m, particle channel gap 6 mm, particle 775 -> 570 degC,
  sCO2 550 -> 700 degC: slides SAND2017-6879C, slide 11.
* Solid density 3300 kg/m^3, solid volume fraction 0.55, particle diameter
  280 um, k_s = 2.0 W/m/K, emissivity 0.9, metal thickness 1 mm, sCO2
  channel diameter 1 mm and spacing 1 mm (pitch 2 mm), SS316 plates:
  Albrecht & Ho, Solar Energy 178 (2019) 331-340, Tables 1-3. These are the
  *later* (4-bank) study's baseline values; that they equal the 2017 inputs
  is an assumption.

Assumed values (not found in an accessible source)
--------------------------------------------------
* Plate width 0.5 m: from the task brief; every output used for validation is
  per unit width, so it only sets the number of sCO2 channels.
* sCO2 pressure 20 MPa: Solar Energy (2019) only states "> 20 MPa".
* sCO2 channels circular and running vertically (counterflow), as in the
  1-D sCO2 equation of the 2017 2-D model (slide 10). The 2017 sCO2 channel
  geometry itself is unknown.
* Particle cp: the project's CARBO HSP correlation (365 T_C^0.18) rather than
  the 148.2 T^0.3093 of Solar Energy (2019), Table 2 (about 5 % apart at
  670 degC; whether T there is in K is not stated).
* SS316 conductivity: model default (nominal, unverified).
* Mass flows: not given. They are back-calculated so that the reported
  temperatures and U are mutually consistent: :math:`Q = U_{ref} A
  \\Delta T_{lm}` with :math:`U_{ref} = 144`, and each stream's flow from its
  enthalpy change. This fixes the operating point (and hence the capacity
  ratio and the particle velocity) at which our U and effectiveness are
  compared; it does not touch any closure.
"""

from __future__ import annotations

import math
from typing import Any

from pcsp.constants import T_ZERO_C
from pcsp.materials import sco2
from pcsp.materials.particles import CARBO_HSP_40_70, ParticleMaterial
from pcsp.models.hx.closures import BedClosure, StagnantBed
from pcsp.models.hx.model import SCO2_TABLE_T_MAX, SCO2_TABLE_T_MIN, MovingBedHXParams

U_REPORTED = 144.0
"""Overall coefficient of the single-bank 2-D model [W m^-2 K^-1] (slide 12)."""

H_PARTICLE_WALL_REPORTED = 182.0
"""Average particle-wall coefficient of the same model [W m^-2 K^-1] (slide 12)."""

T_PARTICLE_IN = 775.0 + T_ZERO_C
T_PARTICLE_OUT = 570.0 + T_ZERO_C
T_CO2_IN = 550.0 + T_ZERO_C
T_CO2_OUT = 700.0 + T_ZERO_C
P_CO2 = 20e6

PARTICLE: ParticleMaterial = CARBO_HSP_40_70.with_(
    d_p=280e-6, packing_fraction=0.55, emissivity=0.9, k_s=2.0, rho_p=3300.0
)


def design_lmtd() -> float:
    """Log-mean temperature difference of the design temperatures [K] (41.61 K in the paper)."""
    dT_hot_end = T_PARTICLE_IN - T_CO2_OUT
    dT_cold_end = T_PARTICLE_OUT - T_CO2_IN
    return (dT_hot_end - dT_cold_end) / math.log(dT_hot_end / dT_cold_end)


def nominal_params(closure: BedClosure | None = None, **overrides: Any) -> MovingBedHXParams:
    """Parameters of the nominal point; see the module docstring for every source."""
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
