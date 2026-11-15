"""Exact Jacobians of the discrete-time model via CasADi automatic differentiation.

Shared by the extended Kalman filter and the observability analysis. The
Jacobian is that of the *discretised* map ``F`` (RK4 with zero-order hold), not
of the continuous right-hand side, so it is consistent with the prediction step.
"""

from __future__ import annotations

import casadi as ca

from pcsp.core.integrate import casadi_discretize, casadi_functions
from pcsp.core.interfaces import ControlModel


def discrete_jacobians(
    model: ControlModel, dt: float, substeps: int = 1
) -> tuple[ca.Function, ca.Function]:
    """Return two CasADi functions of ``(x, u, d)``.

    * ``transition``: ``(x_next, A)`` with ``x_next = F(x, u, d)`` and ``A = dF/dx``.
    * ``measurement``: ``(y, C)`` with ``y = h(x, u, d)`` and ``C = dh/dx``.
    """
    F = casadi_discretize(model, dt, substeps)
    _, h = casadi_functions(model)
    x = ca.SX.sym("x", model.nx)
    u = ca.SX.sym("u", model.nu)
    d = ca.SX.sym("d", model.nd)
    x_next = F(x, u, d)
    y = h(x, u, d)
    transition = ca.Function("transition", [x, u, d], [x_next, ca.jacobian(x_next, x)])
    measurement = ca.Function("measurement", [x, u, d], [y, ca.jacobian(y, x)])
    return transition, measurement
