"""Solid particle materials.

All property methods are backend-agnostic (plain arithmetic only), so they can
be called with floats, NumPy arrays or CasADi symbols.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from pcsp.constants import T_ZERO_C


@dataclass(frozen=True)
class ParticleMaterial:
    """A granular heat-transfer / storage medium.

    Specific heat follows a power law in Celsius temperature,

    .. math::  c_p(T) = a \\, (T - 273.15)^b ,

    which covers both the constant-:math:`c_p` case (``cp_b = 0``) and the
    Sandia correlation for sintered-bauxite proppants (``cp_b = 0.18``).

    Attributes
    ----------
    name : human-readable identifier
    rho_p : particle (envelope) density [kg m^-3]
    d_p : Sauter-mean particle diameter [m]
    k_s : thermal conductivity of the solid material [W m^-1 K^-1]
    emissivity : total hemispherical thermal emissivity [-]
    solar_absorptivity : solar-weighted absorptivity [-]
    cp_a, cp_b : specific-heat power-law coefficients (see above) [J kg^-1 K^-1]
    packing_fraction : solid volume fraction of the loosely packed bed [-]
    source : where the numbers come from, and their validity range
    """

    name: str
    rho_p: float
    d_p: float
    k_s: float
    emissivity: float
    solar_absorptivity: float
    cp_a: float
    cp_b: float = 0.0
    packing_fraction: float = 0.6
    source: str = ""

    def __post_init__(self) -> None:
        for field in ("rho_p", "d_p", "k_s", "cp_a"):
            if getattr(self, field) <= 0.0:
                raise ValueError(f"{field} must be positive, got {getattr(self, field)}")
        for field in ("emissivity", "solar_absorptivity", "packing_fraction"):
            if not 0.0 < getattr(self, field) <= 1.0:
                raise ValueError(f"{field} must be in (0, 1], got {getattr(self, field)}")

    @property
    def rho_bulk(self) -> float:
        """Bulk density of the packed bed [kg m^-3]."""
        return self.rho_p * self.packing_fraction

    @property
    def void_fraction(self) -> float:
        """Void (gas) volume fraction of the packed bed [-]."""
        return 1.0 - self.packing_fraction

    def cp(self, T: Any) -> Any:
        """Specific heat capacity [J kg^-1 K^-1] at temperature ``T`` [K]."""
        if self.cp_b == 0.0:
            return self.cp_a + 0.0 * T
        return self.cp_a * (T - T_ZERO_C) ** self.cp_b

    def enthalpy(self, T: Any, T_ref: float = 298.15) -> Any:
        """Specific enthalpy [J kg^-1] relative to ``T_ref`` [K].

        Exact integral of :meth:`cp`, so energy balances written in enthalpy
        close to round-off.
        """
        n = 1.0 + self.cp_b
        return self.cp_a / n * ((T - T_ZERO_C) ** n - (T_ref - T_ZERO_C) ** n)

    def with_(self, **changes: Any) -> ParticleMaterial:
        """Copy with some attributes replaced (for sensitivity studies)."""
        return dataclasses.replace(self, **changes)


CARBO_HSP_40_70 = ParticleMaterial(
    name="CARBO HSP 40/70",
    rho_p=3300.0,
    d_p=350e-6,
    k_s=2.0,
    emissivity=0.843,
    solar_absorptivity=0.934,
    cp_a=365.0,
    cp_b=0.18,
    packing_fraction=0.606,
    source=(
        "Sintered-bauxite proppant used in Sandia falling-particle receivers. "
        "cp = 365*T_C^0.18 J/kg/K for 50-1100 degC (Sandia correlation, Ho et al.). "
        "Radiative properties: as-received values reported by Siegel et al. (ASME ES2014). "
        "rho_p, d_p, k_s and packing fraction are nominal literature values; "
        "treat all of them as uncertain inputs in any quantitative study."
    ),
)

SILICA_SAND = ParticleMaterial(
    name="silica sand (generic)",
    rho_p=2650.0,
    d_p=300e-6,
    k_s=3.0,
    emissivity=0.8,
    solar_absorptivity=0.55,
    cp_a=1000.0,
    cp_b=0.0,
    packing_fraction=0.6,
    source=(
        "Generic order-of-magnitude values for quartz sand with a constant mean "
        "cp. Placeholder for scoping studies only; replace with measured data."
    ),
)
