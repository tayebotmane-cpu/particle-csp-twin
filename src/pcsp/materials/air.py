"""Dry-air properties at near-atmospheric pressure, in closed form.

Closed-form expressions (rather than calls into a property library) keep the
functions fast, vectorised and usable with CasADi symbols.

``cp``, ``mu`` and ``k`` are fourth-order polynomials in ``T / 1000 K``,
least-squares fitted to CoolProp (Lemmon et al. reference equations) at
1 atm over 250-1600 K. Maximum deviation from CoolProp over that range:
0.15 % (cp), 0.44 % (mu), 0.37 % (k). ``tests/materials/test_air.py`` guards
those bounds. Do not use outside 250-1600 K.
"""

from __future__ import annotations

from typing import Any

from pcsp.constants import P_ATM, R_UNIVERSAL

MOLAR_MASS = 0.0289647
"""Molar mass of dry air [kg mol^-1]."""

T_MIN = 250.0
T_MAX = 1600.0

# Polynomial coefficients, highest power of (T/1000) first.
_CP = (1.784385502e02, -7.750714491e02, 1.126682911e03, -4.466853293e02, 1.057974270e03)
_MU = (-3.930801225e-06, 2.008093541e-05, -4.190699274e-05, 6.745949473e-05, 1.564211179e-06)
_K = (-4.756672122e-03, 2.431605531e-02, -5.040813952e-02, 9.743260234e-02, 1.077812113e-03)


def _poly(coeffs: tuple[float, ...], T: Any) -> Any:
    tau = T / 1000.0
    result = coeffs[0] + 0.0 * tau
    for c in coeffs[1:]:
        result = result * tau + c
    return result


def rho(T: Any, p: Any = P_ATM) -> Any:
    """Density [kg m^-3] from the ideal-gas law."""
    return p * MOLAR_MASS / (R_UNIVERSAL * T)


def cp(T: Any) -> Any:
    """Isobaric specific heat capacity [J kg^-1 K^-1]."""
    return _poly(_CP, T)


def mu(T: Any) -> Any:
    """Dynamic viscosity [Pa s]."""
    return _poly(_MU, T)


def k(T: Any) -> Any:
    """Thermal conductivity [W m^-1 K^-1]."""
    return _poly(_K, T)


def prandtl(T: Any) -> Any:
    """Prandtl number [-]."""
    return cp(T) * mu(T) / k(T)
