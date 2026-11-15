"""Toy models shared by the estimation tests."""

import numpy as np
import pytest

from pcsp.core import ControlModelBase, ops_for


class Lorenz63(ControlModelBase):
    """Lorenz (1963) system with the classical parameters; all states measured."""

    state_names = ("x", "y", "z")
    output_names = ("x", "y", "z")

    def rhs(self, x, u, d):
        xp = ops_for(x, u, d)
        return xp.concat(
            [
                10.0 * (x[1] - x[0]),
                x[0] * (28.0 - x[2]) - x[1],
                x[0] * x[1] - (8.0 / 3.0) * x[2],
            ]
        )

    def output(self, x, u, d):
        xp = ops_for(x, u, d)
        return xp.concat([x[0], x[1], x[2]])


class ToyReceiver(ControlModelBase):
    """Two-lump radiating receiver: a wall heated by the sun, particles heated by the wall.

    .. math::

        C_w \\dot T_w = Q_{sun} + Q_{aux} - hA (T_w - T_p)
                        - \\epsilon\\sigma A (T_w^4 - T_{amb}^4) \\\\
        C_p \\dot T_p = hA (T_w - T_p) - \\dot m c_p (T_p - T_{in})

    Only the particle outlet temperature is measured. ``Q_aux`` enters exactly
    like ``Q_sun``, so the two cannot be told apart from any sensor.
    """

    state_names = ("T_w", "T_p")
    input_names = ("m_dot",)
    disturbance_names = ("Q_sun", "T_amb", "Q_aux")
    output_names = ("T_p",)

    C_w = 2.0e5  # J/K
    C_p = 6.0e5  # J/K
    hA = 5.0e3  # W/K
    eps_sigma_A = 0.9 * 5.670374419e-8 * 2.0  # W/K^4
    cp = 1200.0  # J/(kg K)
    T_in = 600.0  # K

    def rhs(self, x, u, d):
        xp = ops_for(x, u, d)
        conv = self.hA * (x[0] - x[1])
        rad = self.eps_sigma_A * (x[0] ** 4 - d[1] ** 4)
        return xp.concat(
            [
                (d[0] + d[2] - conv - rad) / self.C_w,
                (conv - u[0] * self.cp * (x[1] - self.T_in)) / self.C_p,
            ]
        )

    def output(self, x, u, d):
        xp = ops_for(x, u, d)
        return xp.concat([x[1]])


class LinearCV(ControlModelBase):
    """Damped oscillator ``ẋ = M x + B u`` with position measured (linear ControlModel)."""

    state_names = ("pos", "vel")
    input_names = ("force",)
    output_names = ("pos",)
    M = np.array([[0.0, 1.0], [-2.0, -0.3]])

    def rhs(self, x, u, d):
        xp = ops_for(x, u, d)
        return xp.concat([x[1], -2.0 * x[0] - 0.3 * x[1] + u[0]])

    def output(self, x, u, d):
        xp = ops_for(x, u, d)
        return xp.concat([x[0]])


@pytest.fixture
def lorenz():
    return Lorenz63()


@pytest.fixture
def receiver():
    return ToyReceiver()


@pytest.fixture
def linear_cv():
    return LinearCV()
