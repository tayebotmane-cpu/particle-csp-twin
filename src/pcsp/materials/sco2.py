"""Carbon-dioxide properties along an isobar, tabulated from CoolProp.

Calling CoolProp inside a solver loop costs tens of microseconds per state. For
heat-exchanger models the pressure changes by well under 1 % along the flow
path, so properties are tabulated once along a single isobar and interpolated:

* ``h(T)`` is a cubic Hermite spline that uses the exact derivative
  :math:`(\\partial h / \\partial T)_p = c_p` at every node, so it is
  fourth-order accurate and consistent with ``cp``. ``T(h)`` is the Hermite
  spline of the inverse function (derivative :math:`1/c_p`).
* ``cp``, ``rho``, ``mu`` and ``k`` use monotone piecewise-cubic (PCHIP)
  interpolation, which cannot overshoot near the pseudo-critical peak of
  :math:`c_p`.

The node set is the union of a uniform temperature grid and a grid that is
uniform in enthalpy. The second set concentrates nodes where :math:`c_p` is
large (spacing :math:`\\Delta T = \\Delta h / c_p`), i.e. around the
pseudo-critical temperature, where a uniform temperature grid is too coarse.

Sources (as reported by CoolProp 8 for the ``HEOS::CO2`` backend):

* equation of state: Span & Wagner, J. Phys. Chem. Ref. Data 25 (1996) 1509;
* viscosity: Laesecke & Muzny, J. Phys. Chem. Ref. Data 46 (2017) 013107;
* thermal conductivity: Huber et al., J. Phys. Chem. Ref. Data 45 (2016) 013102.

The validity ranges of those correlations are given in the papers; this module
does not check them. The default table spans 300-1100 K. Accuracy of the table
against direct CoolProp calls is guarded by ``tests/materials/test_sco2.py``.

Functions are NumPy only (not CasADi-compatible).
"""

from __future__ import annotations

from functools import lru_cache

import CoolProp.CoolProp as CP
import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

MOLAR_MASS = 0.0440098
"""Molar mass of CO2 [kg mol^-1] (Span & Wagner, 1996)."""

T_CRITICAL = 304.1282
"""Critical temperature of CO2 [K] (Span & Wagner, 1996)."""

P_CRITICAL = 7.3773e6
"""Critical pressure of CO2 [Pa] (Span & Wagner, 1996)."""


class SCO2Isobar:
    """Interpolated CO2 properties at fixed pressure ``p`` [Pa] over ``[T_min, T_max]`` [K].

    Build with :func:`isobar` to reuse cached tables. All methods accept floats
    or NumPy arrays and raise ``ValueError`` outside the tabulated range.

    Parameters
    ----------
    p : pressure [Pa]
    T_min, T_max : tabulated temperature range [K]
    n_nodes : number of nodes of each of the two grids (uniform in T and
        uniform in h); the table holds at most ``2 * n_nodes`` nodes.
    """

    def __init__(self, p: float, T_min: float = 300.0, T_max: float = 1100.0, n_nodes: int = 800):
        if not T_min < T_max:
            raise ValueError(f"need T_min < T_max, got {T_min}, {T_max}")
        if n_nodes < 10:
            raise ValueError(f"n_nodes must be >= 10, got {n_nodes}")
        self.p = float(p)
        self.T_min = float(T_min)
        self.T_max = float(T_max)

        state = CP.AbstractState("HEOS", "CO2")
        T_uniform = np.linspace(T_min, T_max, n_nodes)
        h_uniform, cp_uniform = _coolprop_h_cp(state, self.p, T_uniform)
        # Nodes equally spaced in enthalpy, located with the Hermite spline of T(h).
        coarse_T_of_h = CubicHermiteSpline(h_uniform, T_uniform, 1.0 / cp_uniform)
        T_equal_h = coarse_T_of_h(np.linspace(h_uniform[0], h_uniform[-1], n_nodes))
        T = np.unique(np.concatenate([T_uniform, T_equal_h]))
        T = T[np.concatenate([[True], np.diff(T) > 1e-9 * T_max])]

        n = T.size
        h, cp, rho, mu, k = (np.empty(n) for _ in range(5))
        for i, Ti in enumerate(T):
            state.update(CP.PT_INPUTS, self.p, Ti)
            h[i] = state.hmass()
            cp[i] = state.cpmass()
            rho[i] = state.rhomass()
            mu[i] = state.viscosity()
            k[i] = state.conductivity()
        self.T_nodes = T
        self._h = CubicHermiteSpline(T, h, cp)
        self._T_of_h = CubicHermiteSpline(h, T, 1.0 / cp)
        self._cp = PchipInterpolator(T, cp)
        self._rho = PchipInterpolator(T, rho)
        self._mu = PchipInterpolator(T, mu)
        self._k = PchipInterpolator(T, k)
        self.h_min = float(h[0])
        self.h_max = float(h[-1])

    def _check_T(self, T: np.ndarray) -> None:
        if T.min() < self.T_min or T.max() > self.T_max:
            raise ValueError(
                f"temperature outside the sCO2 table range [{self.T_min}, {self.T_max}] K: "
                f"min {np.min(T):.6g}, max {np.max(T):.6g}"
            )

    def h(self, T: np.ndarray | float) -> np.ndarray:
        """Specific enthalpy [J kg^-1] (CoolProp reference state)."""
        T = np.asarray(T, dtype=float)
        self._check_T(T)
        return self._h(T)

    def T_from_h(self, h: np.ndarray | float) -> np.ndarray:
        """Temperature [K] from specific enthalpy [J kg^-1]."""
        h = np.asarray(h, dtype=float)
        if np.any(h < self.h_min) or np.any(h > self.h_max):
            raise ValueError("enthalpy outside the sCO2 table range")
        return self._T_of_h(h)

    def cp(self, T: np.ndarray | float) -> np.ndarray:
        """Isobaric specific heat capacity [J kg^-1 K^-1]."""
        T = np.asarray(T, dtype=float)
        self._check_T(T)
        return self._cp(T)

    def rho(self, T: np.ndarray | float) -> np.ndarray:
        """Density [kg m^-3]."""
        T = np.asarray(T, dtype=float)
        self._check_T(T)
        return self._rho(T)

    def mu(self, T: np.ndarray | float) -> np.ndarray:
        """Dynamic viscosity [Pa s]."""
        T = np.asarray(T, dtype=float)
        self._check_T(T)
        return self._mu(T)

    def k(self, T: np.ndarray | float) -> np.ndarray:
        """Thermal conductivity [W m^-1 K^-1]."""
        T = np.asarray(T, dtype=float)
        self._check_T(T)
        return self._k(T)

    def prandtl(self, T: np.ndarray | float) -> np.ndarray:
        """Prandtl number [-]."""
        return self.cp(T) * self.mu(T) / self.k(T)

    def pseudocritical_temperature(self) -> float:
        """Temperature of the tabulated :math:`c_p` maximum [K] (supercritical isobars only)."""
        return float(self.T_nodes[np.argmax(self._cp(self.T_nodes))])


def _coolprop_h_cp(state: CP.AbstractState, p: float, T: np.ndarray) -> tuple[np.ndarray, ...]:
    h = np.empty(T.size)
    cp = np.empty(T.size)
    for i, Ti in enumerate(T):
        state.update(CP.PT_INPUTS, p, Ti)
        h[i] = state.hmass()
        cp[i] = state.cpmass()
    return h, cp


@lru_cache(maxsize=16)
def isobar(p: float, T_min: float = 300.0, T_max: float = 1100.0, n_nodes: int = 800) -> SCO2Isobar:
    """Cached :class:`SCO2Isobar` (building one takes a few tens of milliseconds)."""
    return SCO2Isobar(p, T_min, T_max, n_nodes)


def coolprop_state(p: float, T: float) -> dict[str, float]:
    """Direct (slow) CoolProp evaluation of one state, for reference and tests.

    Returns ``h`` [J kg^-1], ``cp`` [J kg^-1 K^-1], ``rho`` [kg m^-3],
    ``mu`` [Pa s] and ``k`` [W m^-1 K^-1].
    """
    state = CP.AbstractState("HEOS", "CO2")
    state.update(CP.PT_INPUTS, p, T)
    return {
        "h": state.hmass(),
        "cp": state.cpmass(),
        "rho": state.rhomass(),
        "mu": state.viscosity(),
        "k": state.conductivity(),
    }
