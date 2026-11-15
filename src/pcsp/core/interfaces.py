"""Contracts shared by physics models, estimators and controllers.

Anything in :mod:`pcsp.twin` is written against these protocols, never against
a concrete model, so a reduced physics model can later be swapped for a ROM or
a neural surrogate without touching the estimation or control code.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ControlModel(Protocol):
    """Continuous-time control-oriented model.

    .. math::  \\dot x = f(x, u, d), \\qquad y = h(x, u, d)

    ``x`` is the state, ``u`` the manipulated inputs, ``d`` the measured or
    estimated disturbances and ``y`` the measured outputs.

    ``rhs`` and ``output`` must be *backend-agnostic*: called with 1-D NumPy
    arrays they return 1-D NumPy arrays; called with CasADi ``SX`` column
    vectors they return ``SX`` column vectors. See :mod:`pcsp.core.backend`.

    The ``*_names`` tuples define the vector layouts and therefore the sizes
    ``nx``, ``nu``, ``nd``, ``ny``. Inherit from :class:`ControlModelBase` to
    get the sizes for free.
    """

    state_names: tuple[str, ...]
    input_names: tuple[str, ...]
    disturbance_names: tuple[str, ...]
    output_names: tuple[str, ...]

    @property
    def nx(self) -> int: ...
    @property
    def nu(self) -> int: ...
    @property
    def nd(self) -> int: ...
    @property
    def ny(self) -> int: ...

    def rhs(self, x: Any, u: Any, d: Any) -> Any:
        """State derivative ``dx/dt`` [state units per second]."""
        ...

    def output(self, x: Any, u: Any, d: Any) -> Any:
        """Measured outputs ``y``."""
        ...


class ControlModelBase:
    """Mixin deriving ``nx``, ``nu``, ``nd``, ``ny`` from the name tuples."""

    state_names: tuple[str, ...] = ()
    input_names: tuple[str, ...] = ()
    disturbance_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()

    @property
    def nx(self) -> int:
        return len(self.state_names)

    @property
    def nu(self) -> int:
        return len(self.input_names)

    @property
    def nd(self) -> int:
        return len(self.disturbance_names)

    @property
    def ny(self) -> int:
        return len(self.output_names)


@runtime_checkable
class Estimator(Protocol):
    """Recursive state estimator running at a fixed sample period.

    One cycle is ``predict(u, d)`` (advance the estimate by one sample period
    under the inputs that were applied) followed by ``update(y, u, d)``
    (assimilate the measurement taken at the end of that period).

    The measurement map is ``y = h(x, u, d)``, so ``update`` also takes the
    input and disturbance that were in effect when ``y`` was taken. They may be
    omitted after a ``predict``, in which case that call's ``u`` and ``d`` are
    reused. They must be given when ``update`` is the first call - a loop that
    assimilates the measurement at ``t = 0`` before any prediction - unless the
    estimator's measurement map does not depend on them.
    """

    def predict(self, u: np.ndarray, d: np.ndarray) -> None: ...

    def update(
        self, y: np.ndarray, u: np.ndarray | None = None, d: np.ndarray | None = None
    ) -> None: ...

    @property
    def x_hat(self) -> np.ndarray:
        """Current state estimate, shape ``(nx,)``."""
        ...

    @property
    def P(self) -> np.ndarray:
        """Current state-error covariance estimate, shape ``(nx, nx)``."""
        ...


@runtime_checkable
class Controller(Protocol):
    """Discrete-time feedback controller.

    ``step`` is called once per sample period and returns the input ``u`` to
    hold over the next period. Output-feedback controllers (PID) use ``y`` and
    ``r``; model-based controllers (MPC) use ``x_hat`` and ``d_hat`` as well.
    Controllers must ignore arguments they do not need.
    """

    def reset(self) -> None:
        """Clear internal state (integrators, warm starts)."""
        ...

    def step(
        self,
        t: float,
        y: np.ndarray,
        r: np.ndarray,
        x_hat: np.ndarray | None = None,
        d_hat: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the input vector ``u``, shape ``(nu,)``.

        Parameters
        ----------
        t : current time [s]
        y : latest measurement, shape ``(ny,)``
        r : setpoint for the controlled outputs
        x_hat : state estimate, if an estimator is in the loop
        d_hat : disturbance estimate or measurement, if available
        """
        ...
