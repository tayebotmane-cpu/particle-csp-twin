"""Write a model once, evaluate it with NumPy or CasADi.

Control-oriented models in this package have to serve two masters: fast
numerical simulation (NumPy) and gradient-based optimisation (CasADi symbols,
for MPC and moving-horizon estimation). Rather than maintain two copies of the
physics, model code asks for an *ops* namespace and uses only what it offers::

    from pcsp.core.backend import ops_for

    def rhs(self, x, u, d):
        xp = ops_for(x, u, d)
        loss = xp.maximum(x[0] - 300.0, 0.0) ** 4
        return xp.concat([-loss + u[0], x[0] - x[1]])

Rules that keep code portable across both backends:

* Vectors are 1-D NumPy arrays or CasADi ``SX`` column vectors. Index with
  integers or slices only (``x[0]``, ``x[1:]``).
* Use plain arithmetic operators (``+ - * / **``); they are element-wise in
  both backends. Do not use ``@`` on state vectors.
* Use ``xp.exp``, ``xp.sqrt``, ``xp.maximum`` ... instead of ``np.*`` / ``math.*``.
* Build vectors with ``xp.concat([...])``; never ``np.array([...])`` of states.
* Branch with ``xp.where(cond, a, b)``; Python ``if`` on a state is a bug.
* Wrap constant arrays in ``xp.const(values)`` before combining with states.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import casadi as ca
import numpy as np

_CASADI_TYPES = (ca.SX, ca.MX, ca.DM)


def is_symbolic(*values: Any) -> bool:
    """True if any argument is a CasADi object (``SX``, ``MX`` or ``DM``)."""
    return any(isinstance(v, _CASADI_TYPES) for v in values)


class NumpyOps:
    """Numerical backend: operands are floats or NumPy arrays."""

    name = "numpy"

    exp = staticmethod(np.exp)
    log = staticmethod(np.log)
    sqrt = staticmethod(np.sqrt)
    tanh = staticmethod(np.tanh)
    abs = staticmethod(np.abs)
    maximum = staticmethod(np.maximum)
    minimum = staticmethod(np.minimum)

    @staticmethod
    def concat(parts: Sequence[Any]) -> np.ndarray:
        """Stack scalars and 1-D vectors into one 1-D vector."""
        return np.concatenate([np.atleast_1d(np.asarray(p, dtype=float)) for p in parts])

    @staticmethod
    def where(cond: Any, if_true: Any, if_false: Any) -> Any:
        return np.where(cond, if_true, if_false)

    @staticmethod
    def const(values: Any) -> np.ndarray:
        return np.asarray(values, dtype=float)

    @staticmethod
    def sum(x: Any) -> Any:
        return np.sum(x)


class CasadiOps:
    """Symbolic backend: operands are CasADi ``SX`` (preferred) expressions."""

    name = "casadi"

    exp = staticmethod(ca.exp)
    log = staticmethod(ca.log)
    sqrt = staticmethod(ca.sqrt)
    tanh = staticmethod(ca.tanh)
    abs = staticmethod(ca.fabs)
    maximum = staticmethod(ca.fmax)
    minimum = staticmethod(ca.fmin)

    @staticmethod
    def concat(parts: Sequence[Any]) -> Any:
        """Stack scalars and column vectors into one column vector."""
        return ca.vertcat(*parts)

    @staticmethod
    def where(cond: Any, if_true: Any, if_false: Any) -> Any:
        return ca.if_else(cond, if_true, if_false)

    @staticmethod
    def const(values: Any) -> ca.DM:
        return ca.DM(np.asarray(values, dtype=float))

    @staticmethod
    def sum(x: Any) -> Any:
        return ca.sum1(x)


def ops_for(*values: Any) -> type[NumpyOps] | type[CasadiOps]:
    """Return the ops namespace matching the operands.

    CasADi wins if any operand is symbolic, so a model called with symbolic
    states and numeric parameters is evaluated symbolically.
    """
    return CasadiOps if is_symbolic(*values) else NumpyOps
