"""Particle-side closures: effective bed conductivity and near-wall resistance.

The moving-bed heat-exchanger model needs exactly two closures from the
particle side, both functions of the local bed state:

* ``k_eff(T, u, particle)`` - effective (continuum) conductivity of the bulk
  bed [W m^-1 K^-1];
* ``r_wall(T, u, particle)`` - near-wall thermal resistance between the wall
  and the bulk continuum [m^2 K W^-1], acting in series with the Graetz
  resistance (see :mod:`pcsp.models.hx.graetz`).

``T`` is the bed bulk temperature [K], ``u`` the bed (plug) velocity [m s^-1].
Learning these two closures from CFD-DEM data is a planned thesis
contribution, so they sit behind the :class:`BedClosure` protocol. Every
closure is a frozen dataclass with a ``name`` field plus numeric settings, so
the parameter set stays JSON-serialisable (:func:`pcsp.data.schema.to_jsonable`)
and reproducible. To add a closure (e.g. one trained on CFD-DEM), write a new
frozen dataclass with a unique ``name`` and the two methods.

The near-wall resistance of every closure here is evaluated with the gas
conductivity at the *bed bulk* temperature, as in Adapa et al. (2024). The
gas film actually sits between bed and wall temperature; the bed-to-wall
difference in the nominal Sandia case is a few tens of kelvin, which changes
:math:`k_{air}` by roughly 2 %.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from pcsp.constants import SIGMA_SB, T_ZERO_C
from pcsp.materials import air
from pcsp.materials.particles import ParticleMaterial
from pcsp.models.correlations import k_eff_zehner_schlunder


@runtime_checkable
class BedClosure(Protocol):
    """Effective conductivity and near-wall resistance of a moving packed bed."""

    name: str

    def k_eff(self, T: np.ndarray, u: float, particle: ParticleMaterial) -> np.ndarray:
        """Bulk effective conductivity [W m^-1 K^-1] at bed temperature ``T`` [K]."""
        ...

    def r_wall(self, T: np.ndarray, u: float, particle: ParticleMaterial) -> np.ndarray:
        """Near-wall thermal resistance [m^2 K W^-1] at bed temperature ``T`` [K]."""
        ...


def _check_name(obj: Any, expected: str) -> None:
    if obj.name != expected:
        raise ValueError(f"{type(obj).__name__}.name must be {expected!r}, got {obj.name!r}")


@dataclass(frozen=True)
class ConstantBed:
    """Fixed ``k_eff`` [W m^-1 K^-1] and ``r_wall`` [m^2 K W^-1].

    For verification, and for plugging in single values extracted from
    CFD-DEM or experiments.
    """

    k_eff_value: float
    r_wall_value: float = 0.0
    name: str = "constant"

    def __post_init__(self) -> None:
        _check_name(self, "constant")
        if self.k_eff_value <= 0.0 or self.r_wall_value < 0.0:
            raise ValueError("need k_eff_value > 0 and r_wall_value >= 0")

    def k_eff(self, T: np.ndarray, u: float, particle: ParticleMaterial) -> np.ndarray:
        return np.full_like(np.asarray(T, dtype=float), self.k_eff_value)

    def r_wall(self, T: np.ndarray, u: float, particle: ParticleMaterial) -> np.ndarray:
        return np.full_like(np.asarray(T, dtype=float), self.r_wall_value)


def k_radiation_breitbach_barthels(
    T: np.ndarray, d_p: float, eps: float, emissivity: float, k_s: float
) -> np.ndarray:
    """Radiative conductivity of a packed bed of spheres [W m^-1 K^-1].

    Breitbach & Barthels (1980), as reproduced by van Antwerpen (eq. 3.150):

    .. math::
        k_{rad} = \\left\\{ \\left[1 - \\sqrt{1-\\varepsilon}\\right]\\varepsilon
          + \\frac{\\sqrt{1-\\varepsilon}}{2/e_r - 1} \\, \\frac{B+1}{B} \\,
            \\frac{1}{1 + \\left[(2/e_r - 1)\\Lambda_s\\right]^{-1}} \\right\\} 4 \\sigma T^3 d_p ,

    with :math:`\\Lambda_s = k_s / (4 \\sigma T^3 d_p)` and
    :math:`B = 1.25 \\, ((1-\\varepsilon)/\\varepsilon)^{10/9}`.

    Parameters
    ----------
    T : bed temperature [K]
    d_p : particle diameter [m]
    eps : void fraction [-]
    emissivity : particle surface emissivity :math:`e_r` [-]
    k_s : solid conductivity [W m^-1 K^-1]

    References: Breitbach, G., Barthels, H., 1980. The radiant heat transfer in
    the high temperature reactor core after failure of the afterheat removal
    systems. Nucl. Technol. 49, 392-399 (original not consulted). van Antwerpen,
    W. Modelling the effective thermal conductivity in the near-wall region of
    a packed pebble bed. Dissertation, North-West University, chapter 3,
    eq. 3.150-3.152 (the chapter consulted does not state the year). The same
    authors' review: van Antwerpen, du Toit, Rousseau, 2010, Nucl. Eng. Des.
    240, 1803-1818.
    """
    T = np.asarray(T, dtype=float)
    k_black = 4.0 * SIGMA_SB * T**3 * d_p
    lam_s = k_s / k_black
    B = 1.25 * ((1.0 - eps) / eps) ** (10.0 / 9.0)
    root = np.sqrt(1.0 - eps)
    shape = 2.0 / emissivity - 1.0
    factor = (1.0 - root) * eps + root / shape * (B + 1.0) / B / (1.0 + 1.0 / (shape * lam_s))
    return factor * k_black


@dataclass(frozen=True)
class StagnantBed:
    """Default closure built from stagnant-bed models.

    * Conduction: Zehner & Schlunder (1970) unit-cell model,
      :func:`pcsp.models.correlations.k_eff_zehner_schlunder`, with air at
      the bed temperature (:mod:`pcsp.materials.air`) as interstitial gas.
    * Radiation: Breitbach & Barthels (1980),
      :func:`k_radiation_breitbach_barthels`, added in parallel
      (:math:`k_{eff} = k_{ZS} + k_{rad}`). The additive superposition of a
      conduction and a radiation model is an assumption of this closure.
    * Near-wall resistance: a gas film of thickness :math:`\\chi d_p` between
      wall and first particle layer (Sullivan & Sabersky, 1975),
      :math:`R_w = \\chi d_p / k_{gas}(T)`. Their fitted value is
      :math:`\\chi = 0.085`; Denloye & Botterill (1977) report about 0.1 and
      Natarajan & Hunt (1997) 0.09-0.114 (all as summarised by Adapa et al.
      2024, introduction; the original papers were not consulted here).

    Neither part depends on the bed velocity: this is a *stagnant-bed*
    closure. Adapa et al. (2024) measured 11-27 % lower conductivity and a
    roughly twice thicker gas gap in flowing beds than in packed beds.

    Attributes
    ----------
    gas_film_factor : :math:`\\chi` [-]
    radiation : include the radiative contribution
    """

    gas_film_factor: float = 0.085
    radiation: bool = True
    name: str = "stagnant_zs_bb_ss"

    def __post_init__(self) -> None:
        _check_name(self, "stagnant_zs_bb_ss")
        if self.gas_film_factor < 0.0:
            raise ValueError("gas_film_factor must be non-negative")

    def k_eff(self, T: np.ndarray, u: float, particle: ParticleMaterial) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        eps = particle.void_fraction
        k = k_eff_zehner_schlunder(air.k(T), particle.k_s, eps)
        if self.radiation:
            k = k + k_radiation_breitbach_barthels(
                T, particle.d_p, eps, particle.emissivity, particle.k_s
            )
        return k

    def r_wall(self, T: np.ndarray, u: float, particle: ParticleMaterial) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        return self.gas_film_factor * particle.d_p / air.k(T)


ADAPA_2024_TABLE7 = {
    "CP 40/100": {"d_air": (1.3e-2, 25.0), "k_eff": (2.8e-4, 0.13)},
    "HSP 40/70": {"d_air": (2e-2, 22.0), "k_eff": (1.5e-4, 0.23)},
}
"""Adapa et al. (2024) Table 7: linear fits ``y = m * T_C + c`` (T in degC) of the
flowing-bed properties. ``d_air`` in micrometres, ``k_eff`` in W m^-1 K^-1."""


@dataclass(frozen=True)
class AdapaFlowingBed:
    """Flowing-bed closure of Adapa et al. (2024), Solar Energy 282, 112960.

    Bulk conductivity and near-wall resistance of a *flowing* bed, measured
    in situ by modulated photothermal radiometry in a 5 mm deep channel
    (velocities 5-15 mm/s, 300-650 degC) and fitted linearly in temperature
    (their Table 7, :data:`ADAPA_2024_TABLE7`):

    .. math::
        k_{eff} = m_k T_C + c_k, \\qquad D_{air} = (m_D T_C + c_D) \\times 10^{-6}\\,\\mathrm{m},
        \\qquad R_w = D_{air} / k_{air}(T) ,

    with :math:`T_C` the bed temperature in degC and :math:`k_{air}` the air
    conductivity at that temperature (their nomenclature: :math:`R_{nw} =
    D_{air}/k_{air}`). They found no clear velocity trend in 5-15 mm/s and
    used the fits as velocity-independent; so does this class.

    What the paper does *not* give, and therefore is not implemented:
    a velocity dependence; fits for HSP 16/30 (only three data points, their
    Table 6); a dependence on particle diameter other than through the two
    measured grades; any data above 650 degC. Using the fits above 650 degC or
    for a grade with a different size distribution is an extrapolation.

    Attributes
    ----------
    grade : ``"CP 40/100"`` (mean d_p 275 um) or ``"HSP 40/70"`` (404 um)
    """

    grade: str = "HSP 40/70"
    name: str = "adapa2024_flowing"

    def __post_init__(self) -> None:
        _check_name(self, "adapa2024_flowing")
        if self.grade not in ADAPA_2024_TABLE7:
            raise ValueError(
                f"grade must be one of {sorted(ADAPA_2024_TABLE7)}, got {self.grade!r}"
            )

    def k_eff(self, T: np.ndarray, u: float, particle: ParticleMaterial) -> np.ndarray:
        m, c = ADAPA_2024_TABLE7[self.grade]["k_eff"]
        return m * (np.asarray(T, dtype=float) - T_ZERO_C) + c

    def gap_thickness(self, T: np.ndarray) -> np.ndarray:
        """Effective near-wall air-gap thickness :math:`D_{air}` [m]."""
        m, c = ADAPA_2024_TABLE7[self.grade]["d_air"]
        return 1e-6 * (m * (np.asarray(T, dtype=float) - T_ZERO_C) + c)

    def r_wall(self, T: np.ndarray, u: float, particle: ParticleMaterial) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        return self.gap_thickness(T) / air.k(T)
