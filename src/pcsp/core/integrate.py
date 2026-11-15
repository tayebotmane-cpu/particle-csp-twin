"""Time integration helpers for :class:`~pcsp.core.interfaces.ControlModel`.

All helpers use the classical fixed-step RK4 scheme with a zero-order hold on
inputs and disturbances. RK4 is explicit: for stiff models either raise
``substeps`` or integrate with an implicit scheme of your own.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import casadi as ca
import numpy as np

from pcsp.core.interfaces import ControlModel

Rhs = Callable[[Any, Any, Any], Any]


def rk4_step(rhs: Rhs, x: Any, u: Any, d: Any, dt: float, substeps: int = 1) -> Any:
    """Advance ``x`` by ``dt`` with ``substeps`` RK4 steps (``u``, ``d`` held).

    Backend-agnostic: works for NumPy arrays and CasADi symbols alike.
    """
    if substeps < 1:
        raise ValueError(f"substeps must be >= 1, got {substeps}")
    h = dt / substeps
    for _ in range(substeps):
        k1 = rhs(x, u, d)
        k2 = rhs(x + 0.5 * h * k1, u, d)
        k3 = rhs(x + 0.5 * h * k2, u, d)
        k4 = rhs(x + h * k3, u, d)
        x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


def discretize(
    model: ControlModel, dt: float, substeps: int = 1
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Return the numerical one-sample map ``x_next = F(x, u, d)``."""

    def step(x: np.ndarray, u: np.ndarray, d: np.ndarray) -> np.ndarray:
        return rk4_step(model.rhs, x, u, d, dt, substeps)

    return step


def simulate(
    model: ControlModel,
    x0: np.ndarray,
    u: np.ndarray,
    d: np.ndarray,
    dt: float,
    substeps: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Open-loop simulation under piecewise-constant inputs.

    Parameters
    ----------
    x0 : initial state, shape ``(nx,)``
    u : inputs held over each sample period, shape ``(N, nu)``
    d : disturbances held over each sample period, shape ``(N, nd)``
    dt : sample period [s]

    Returns
    -------
    X : states at the sample instants, shape ``(N + 1, nx)``; ``X[0] == x0``
    Y : outputs evaluated at the start of each period, shape ``(N, ny)``
    """
    u = np.asarray(u, dtype=float)
    d = np.asarray(d, dtype=float)
    n_steps = u.shape[0]
    if d.shape[0] != n_steps:
        raise ValueError(f"u has {n_steps} rows but d has {d.shape[0]}")
    X = np.empty((n_steps + 1, model.nx))
    Y = np.empty((n_steps, model.ny))
    X[0] = np.asarray(x0, dtype=float)
    for k in range(n_steps):
        Y[k] = model.output(X[k], u[k], d[k])
        X[k + 1] = rk4_step(model.rhs, X[k], u[k], d[k], dt, substeps)
    return X, Y


def casadi_functions(model: ControlModel) -> tuple[ca.Function, ca.Function]:
    """Symbolic ``f(x, u, d) -> xdot`` and ``h(x, u, d) -> y`` for ``model``."""
    x, u, d = _symbols(model)
    f = ca.Function("f", [x, u, d], [model.rhs(x, u, d)], ["x", "u", "d"], ["xdot"])
    h = ca.Function("h", [x, u, d], [model.output(x, u, d)], ["x", "u", "d"], ["y"])
    return f, h


def casadi_discretize(model: ControlModel, dt: float, substeps: int = 1) -> ca.Function:
    """Symbolic one-sample map ``F(x, u, d) -> x_next`` (RK4, zero-order hold)."""
    x, u, d = _symbols(model)
    x_next = rk4_step(model.rhs, x, u, d, dt, substeps)
    return ca.Function("F", [x, u, d], [x_next], ["x", "u", "d"], ["x_next"])


def _symbols(model: ControlModel) -> tuple[ca.SX, ca.SX, ca.SX]:
    return (
        ca.SX.sym("x", model.nx),
        ca.SX.sym("u", model.nu),
        ca.SX.sym("d", model.nd),
    )
