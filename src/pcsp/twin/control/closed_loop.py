"""Fixed-sample-period closed-loop simulation: plant + controller (+ estimator).

Timeline of sample ``k`` (``t_k = k * dt``):

1. The plant output is measured, ``y_k = h_p(x_k, u_{k-1}, d_{k-1}) + v_k``
   with ``v_k ~ N(0, diag(noise_std^2))`` from a seeded generator. (``u_{-1}``
   is ``u0`` and ``d_{-1}`` is ``d(t_0)``.)
2. If an estimator is present: ``predict(u_{k-1}, dk_{k-1})`` (skipped at
   ``k = 0``, where the estimator's initial estimate is the prior) and then
   ``update(y_k, u_{k-1}, dk_{k-1})`` - the input and known disturbance that
   were in effect for that measurement. ``dk`` is the *known* disturbance (see
   below), given in the layout the estimator's model expects.
3. The controller returns ``u_cmd_k = step(t_k, y_k, r_k, x_hat, d_hat)``.
4. The optional actuator rate limit gives the applied input
   ``u_k = clip(u_cmd_k, u_{k-1} - rho dt, u_{k-1} + rho dt)``; it acts on the
   plant independently of any limits inside the controller.
5. The plant is integrated over ``[t_k, t_{k+1}]`` by RK4 with ``u_k`` and
   ``d_k = d(t_k)`` held (zero-order hold).

What the controller sees:

* ``x_hat``: with an estimator, ``estimate_to_state(estimator.x_hat)``
  (identity by default - the estimator may run on an augmented model, so the
  caller maps its estimate to the controller's state); without one, the true
  plant state if ``full_state_feedback`` is set (upper-bound studies), else
  ``None``.
* ``d_hat``: ``estimate_to_disturbance(estimator.x_hat)`` if that function is
  given, else the known disturbance ``dk_k``. The known disturbance is
  ``known_disturbance`` if given (for example a wrong constant, or only the
  measured components) and the true plant disturbance otherwise.

The plant is any :class:`~pcsp.core.interfaces.ControlModel`; it may differ
from the controller's internal model to study model mismatch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pcsp.core.integrate import rk4_step
from pcsp.core.interfaces import Controller, ControlModel, Estimator
from pcsp.twin.control.profiles import as_profile


@dataclass(frozen=True)
class ClosedLoopResult:
    """Time histories of a closed-loop run with ``n`` samples.

    Row ``k`` of every ``(n, ...)`` array belongs to the sample instant
    ``t[k]``; inputs are held over ``[t[k], t[k] + dt)``.

    Attributes
    ----------
    t : sample instants, shape ``(n,)`` [s]
    x : true plant state at ``t[k]``, shape ``(n, nx)``
    x_final : true plant state at ``t[-1] + dt``, shape ``(nx,)``
    y_true : noise-free plant output, shape ``(n, ny)``
    y_meas : measured output given to the estimator and controller, ``(n, ny)``
    u_cmd : controller output, shape ``(n, nu)``
    u : input applied to the plant after the actuator rate limit, ``(n, nu)``
    r : setpoint, shape ``(n, nr)``
    d : true plant disturbance, shape ``(n, nd)``
    d_hat : disturbance passed to the controller, shape ``(n, nd_c)``
    x_hat : state passed to the controller, or ``None`` if none was passed
    x_est : full estimator estimate (possibly augmented), or ``None``
    diagnostics : per-step controller diagnostics (e.g. NMPC solve time), each
        an array of shape ``(n,)``; missing entries are NaN
    """

    t: np.ndarray
    x: np.ndarray
    x_final: np.ndarray
    y_true: np.ndarray
    y_meas: np.ndarray
    u_cmd: np.ndarray
    u: np.ndarray
    r: np.ndarray
    d: np.ndarray
    d_hat: np.ndarray
    x_hat: np.ndarray | None
    x_est: np.ndarray | None
    diagnostics: dict[str, np.ndarray]


def run_closed_loop(
    plant: ControlModel,
    controller: Controller,
    x0: Sequence[float],
    u0: Sequence[float],
    *,
    dt: float,
    n_steps: int,
    setpoint: Any,
    disturbance: Any,
    plant_substeps: int = 1,
    noise_std: float | Sequence[float] = 0.0,
    seed: int = 0,
    known_disturbance: Any = None,
    estimator: Estimator | None = None,
    estimate_to_state: Callable[[np.ndarray], np.ndarray] | None = None,
    estimate_to_disturbance: Callable[[np.ndarray], np.ndarray] | None = None,
    full_state_feedback: bool = False,
    actuator_rate_max: float | Sequence[float] | None = None,
) -> ClosedLoopResult:
    """Simulate ``n_steps`` sample periods of the loop described in the module docstring.

    The controller is reset first; the estimator is used as passed (its current
    estimate is the prior at ``t = 0``).

    Parameters
    ----------
    plant : the simulated "true" system
    controller : any :class:`~pcsp.core.interfaces.Controller`
    x0 : initial plant state
    u0 : input applied before ``t = 0`` (actuator start position)
    dt : sample period [s]
    n_steps : number of sample periods
    setpoint, disturbance, known_disturbance : profiles - a callable of time,
        a constant, or an ``(n_steps, dim)`` array (see
        :func:`pcsp.twin.control.profiles.as_profile`)
    plant_substeps : RK4 substeps per sample for the plant
    noise_std : standard deviation of the additive Gaussian measurement noise,
        scalar or per output [output units]
    seed : seed of the noise generator; equal seeds give identical runs
    estimator, estimate_to_state, estimate_to_disturbance : see module docstring
    full_state_feedback : without an estimator, pass the true state as ``x_hat``
    actuator_rate_max : plant actuator rate limit [input units / s]
    """
    if n_steps < 1 or dt <= 0.0:
        raise ValueError("need n_steps >= 1 and dt > 0")
    if estimator is not None and full_state_feedback:
        raise ValueError("full_state_feedback is only meaningful without an estimator")
    if (estimate_to_state or estimate_to_disturbance) and estimator is None:
        raise ValueError("estimate_to_* functions need an estimator")
    controller.reset()
    nx, nu, ny = plant.nx, plant.nu, plant.ny
    x = np.asarray(x0, dtype=float).reshape(nx)
    u_prev = np.asarray(u0, dtype=float).reshape(nu)
    sigma = np.broadcast_to(np.asarray(noise_std, dtype=float), (ny,))
    rate = None if actuator_rate_max is None else np.broadcast_to(actuator_rate_max, (nu,))
    r_of = as_profile(setpoint, n_steps, dt)
    d_of = as_profile(disturbance, n_steps, dt)
    dk_of = d_of if known_disturbance is None else as_profile(known_disturbance, n_steps, dt)
    rng = np.random.default_rng(seed)

    rows: dict[str, list[np.ndarray]] = {
        k: [] for k in ("x", "y_true", "y_meas", "u_cmd", "u", "r", "d", "d_hat", "x_hat", "x_est")
    }
    diag_rows: list[dict[str, float]] = []
    d_prev = d_of(0.0)
    dk_prev = dk_of(0.0)
    for k in range(n_steps):
        t = k * dt
        d_k, dk_k, r_k = d_of(t), dk_of(t), r_of(t)
        y_true = np.asarray(plant.output(x, u_prev, d_prev), dtype=float).reshape(ny)
        y_meas = y_true + sigma * rng.standard_normal(ny)

        x_c: np.ndarray | None = None
        d_c = dk_k
        if estimator is not None:
            if k > 0:
                estimator.predict(u_prev, dk_prev)
            estimator.update(y_meas, u_prev, dk_prev)
            x_est = np.array(estimator.x_hat, dtype=float)
            rows["x_est"].append(x_est)
            x_c = x_est if estimate_to_state is None else np.asarray(estimate_to_state(x_est))
            if estimate_to_disturbance is not None:
                d_c = np.atleast_1d(np.asarray(estimate_to_disturbance(x_est), dtype=float))
        elif full_state_feedback:
            x_c = x.copy()

        u_cmd = np.asarray(controller.step(t, y_meas, r_k, x_hat=x_c, d_hat=d_c), dtype=float)
        u_cmd = u_cmd.reshape(nu)
        if not np.all(np.isfinite(u_cmd)):
            raise FloatingPointError(f"controller returned non-finite input {u_cmd} at t={t}")
        u_k = u_cmd if rate is None else np.clip(u_cmd, u_prev - rate * dt, u_prev + rate * dt)
        diagnostics = getattr(controller, "diagnostics", None)
        diag_rows.append(diagnostics() if callable(diagnostics) else {})

        for name, value in (
            ("x", x),
            ("y_true", y_true),
            ("y_meas", y_meas),
            ("u_cmd", u_cmd),
            ("u", u_k),
            ("r", r_k),
            ("d", d_k),
            ("d_hat", d_c),
        ):
            rows[name].append(np.array(value, dtype=float))
        if x_c is not None:
            rows["x_hat"].append(np.array(x_c, dtype=float))

        x = np.asarray(rk4_step(plant.rhs, x, u_k, d_k, dt, plant_substeps), dtype=float)
        u_prev, d_prev, dk_prev = u_k, d_k, dk_k

    keys = sorted({key for row in diag_rows for key in row})
    diag = {key: np.array([row.get(key, np.nan) for row in diag_rows]) for key in keys}
    stacked = {name: np.array(vals) if vals else None for name, vals in rows.items()}
    return ClosedLoopResult(
        t=dt * np.arange(n_steps),
        x_final=x,
        diagnostics=diag,
        **stacked,
    )
