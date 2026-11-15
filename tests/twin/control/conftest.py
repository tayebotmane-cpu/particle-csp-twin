"""Toy plants for the control tests, written against the ControlModel protocol only.

Fixtures return the *classes*, so a test can build a plant and a deliberately
different controller model from the same family.
"""

import numpy as np
import pytest
from scipy.optimize import brentq, fsolve

from pcsp.core import ControlModelBase, ops_for

STEFAN_BOLTZMANN = 5.670374419e-8  # W / (m^2 K^4)


class FirstOrder(ControlModelBase):
    """``tau dx/dt = -x + K u + d``, ``y = x``: the textbook first-order lag."""

    state_names = ("x",)
    input_names = ("u",)
    disturbance_names = ("d",)
    output_names = ("y",)

    def __init__(self, gain: float = 2.0, tau: float = 10.0) -> None:
        self.gain, self.tau = gain, tau

    def rhs(self, x, u, d):
        xp = ops_for(x, u, d)
        return xp.concat([(-x[0] + self.gain * u[0] + d[0]) / self.tau])

    def output(self, x, u, d):
        return ops_for(x, u, d).concat([x[0]])


class LinearSystem(ControlModelBase):
    """``dx/dt = A x + B u``, ``y = C x`` without disturbances (portable, no ``@``)."""

    disturbance_names = ()

    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> None:
        self.A, self.B, self.C = (np.asarray(M, dtype=float) for M in (A, B, C))
        self.state_names = tuple(f"x{i}" for i in range(self.A.shape[0]))
        self.input_names = tuple(f"u{i}" for i in range(self.B.shape[1]))
        self.output_names = tuple(f"y{i}" for i in range(self.C.shape[0]))

    def _matvec(self, M, v, n):
        return [sum(M[i, j] * v[j] for j in range(M.shape[1])) for i in range(n)]

    def rhs(self, x, u, d):
        xp = ops_for(x, u, d)
        Ax = self._matvec(self.A, x, self.nx)
        Bu = self._matvec(self.B, u, self.nx)
        return xp.concat([a + b for a, b in zip(Ax, Bu, strict=True)])

    def output(self, x, u, d):
        return ops_for(x, u, d).concat(self._matvec(self.C, x, self.ny))


class ToyReceiver(ControlModelBase):
    """Falling-particle receiver caricature: three particle cells in series plus a thermocouple.

    Cell ``i`` (heat capacity ``M c_p``) receives particles at the temperature
    of the cell above (``T_in`` for the first), absorbs a third of the absorbed
    solar power and loses heat by radiation::

        M c_p dT_i/dt = m_dot c_p (T_{i-1} - T_i) + alpha Q / 3 - eps A sigma (T_i^4 - T_amb^4)
        tau_tc dT_tc/dt = T_3 - T_tc,       y = T_tc

    The gain dT_out/dm_dot is negative and strongly state dependent.
    """

    state_names = ("T1", "T2", "T3", "T_tc")
    input_names = ("m_dot",)
    disturbance_names = ("Q_solar",)
    output_names = ("T_tc",)

    def __init__(
        self,
        cell_mass: float = 60.0,  # kg
        cp: float = 1200.0,  # J/(kg K)
        absorptance: float = 0.9,
        emissivity_area: float = 0.85,  # m^2
        T_in: float = 873.15,  # K
        T_amb: float = 300.0,  # K
        tau_tc: float = 5.0,  # s
    ) -> None:
        self.M, self.cp, self.alpha = cell_mass, cp, absorptance
        self.eA, self.T_in, self.T_amb, self.tau_tc = emissivity_area, T_in, T_amb, tau_tc

    def rhs(self, x, u, d):
        xp = ops_for(x, u, d)
        cap = self.M * self.cp
        q = self.alpha * d[0] / 3.0
        flow = u[0] * self.cp
        upstream = [self.T_in, x[0], x[1]]
        dT = [
            (
                flow * (upstream[i] - x[i])
                + q
                - self.eA * STEFAN_BOLTZMANN * (x[i] ** 4 - self.T_amb**4)
            )
            / cap
            for i in range(3)
        ]
        return xp.concat([*dT, (x[2] - x[3]) / self.tau_tc])

    def output(self, x, u, d):
        return ops_for(x, u, d).concat([x[3]])

    def steady_state(self, m_dot: float, Q: float) -> np.ndarray:
        """Equilibrium state for constant flow [kg/s] and solar power [W]."""
        return fsolve(lambda x: self.rhs(x, np.array([m_dot]), np.array([Q])), np.full(4, 1000.0))

    def flow_for(self, T_out: float, Q: float) -> float:
        """Steady mass flow [kg/s] that gives outlet temperature ``T_out`` [K]."""
        return brentq(lambda m: self.steady_state(m, Q)[3] - T_out, 0.5, 20.0)


@pytest.fixture(scope="session")
def first_order():
    return FirstOrder


@pytest.fixture(scope="session")
def linear_system():
    return LinearSystem


@pytest.fixture(scope="session")
def toy_receiver():
    return ToyReceiver
