"""Nonlinear model-predictive control by direct multiple shooting (CasADi + IPOPT).

Optimal control problem solved at every sample instant ``t_k`` (period ``dt``
[s], horizon ``N`` samples). Decision variables are the predicted states
``x_0 ... x_N``, the inputs ``u_0 ... u_{N-1}`` and, if soft output bounds are
configured, non-negative slacks ``eps_1 ... eps_N``:

.. math::

    \\min \\; & \\sum_{j=1}^{N} \\lVert y^c_j - r \\rVert^2_Q
        + \\sum_{j=0}^{N-1} \\Big( \\lVert u_j - u_{j-1} \\rVert^2_R
        + \\lVert u_j - u_\\text{ref} \\rVert^2_S \\Big)
        + \\sum_{j=1}^{N} \\big( w_2^\\top \\epsilon_j^2 + w_1^\\top \\epsilon_j \\big)

    \\text{s.t.} \\; & x_0 = \\hat x, \\qquad u_{-1} = u_\\text{prev}

    & x_{j+1} = F(x_j, u_j, d_j) \\quad \\text{(RK4, ``substeps`` per sample)}

    & y_{j+1} = h(x_{j+1}, u_j, d_j), \\qquad y^c = \\text{tracked outputs of } y

    & u_\\min \\le u_j \\le u_\\max, \\qquad |u_j - u_{j-1}| \\le \\rho\\,dt

    & y_\\min - \\epsilon_j \\le y_j \\le y_\\max + \\epsilon_j, \\qquad \\epsilon_j \\ge 0

``Q``, ``R``, ``S``, ``w_2``, ``w_1`` are diagonal weights given in physical
units (e.g. ``Q`` in 1/K^2, ``R`` in 1/(kg/s)^2 per sample). The output
``y_{j+1}`` is evaluated with the input held over the interval that ends at
``x_{j+1}``, which is exactly how the closed-loop simulator measures the plant.
The disturbances ``d_j`` are held at ``d_hat`` over the horizon unless a
preview (forecast) is supplied.

Offset-free tracking is *not* built in: it comes from the caller passing an
estimated disturbance as ``d_hat`` (e.g. from an estimator running on a model
augmented with integrating disturbances). With the move-penalty cost above and
``S = 0``, any closed-loop steady state then has zero offset.

Implementation notes
--------------------
* The NLP is built once as a parametric problem with parameters
  ``(x_hat, d_0..d_{N-1}, r, u_prev, u_ref)`` and re-solved every step.
* Warm start: the previous primal solution shifted by one sample (last element
  duplicated). After a failed solve the next solve starts cold.
* Scaling: decision variables are affine-scaled, ``x = x_offset + x_scale * z``
  and ``u = u_offset + u_scale * v``, and the shooting, rate and soft-bound
  constraints are divided by ``x_scale``, ``u_scale`` and ``y_scale``, so IPOPT
  sees O(1) numbers when the scales are set to typical ranges (say 100 K and
  1 kg/s for a receiver at 1000 K).
* If IPOPT does not report success, or returns a non-finite input, the
  controller holds the previous input and records the failure.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import casadi as ca
import numpy as np

from pcsp.core.integrate import casadi_discretize, casadi_functions
from pcsp.core.interfaces import ControlModel


@dataclass(frozen=True)
class NMPCConfig:
    """Tuning and constraints of :class:`NMPC`. Vectors are per input/output.

    Attributes
    ----------
    dt : sample period [s]
    horizon : prediction horizon ``N`` [samples]
    tracked_outputs : indices into the model outputs that track the setpoint
    output_weight : ``Q``, one entry per tracked output [1/output unit^2]
    move_weight : ``R`` on ``u_j - u_{j-1}`` [1/input unit^2]
    u_min, u_max : hard input bounds [input units]
    u_init : previous input assumed at the first step / after ``reset``
    rate_max : hard rate bound ``rho`` [input units / s]; ``None`` for none
    input_weight, u_ref : ``S`` and ``u_ref`` of the input-reference penalty
    y_soft_min, y_soft_max : soft output bounds, one entry per model output;
        use ``-inf`` / ``inf`` to leave an output unbounded
    soft_weight, soft_weight_linear : ``w_2`` and ``w_1`` per model output
    substeps : RK4 substeps per sample in the prediction model
    x_scale, x_offset, u_scale, u_offset, y_scale : variable scaling (see module
        docstring); default scale 1 and offset 0
    ipopt_options : extra IPOPT options, e.g. ``{"max_iter": 50}``
    """

    dt: float
    horizon: int
    tracked_outputs: Sequence[int]
    output_weight: Sequence[float]
    move_weight: Sequence[float]
    u_min: Sequence[float]
    u_max: Sequence[float]
    u_init: Sequence[float]
    rate_max: Sequence[float] | None = None
    input_weight: Sequence[float] | None = None
    u_ref: Sequence[float] | None = None
    y_soft_min: Sequence[float] | None = None
    y_soft_max: Sequence[float] | None = None
    soft_weight: Sequence[float] | None = None
    soft_weight_linear: Sequence[float] | None = None
    substeps: int = 1
    x_scale: Sequence[float] | None = None
    x_offset: Sequence[float] | None = None
    u_scale: Sequence[float] | None = None
    u_offset: Sequence[float] | None = None
    y_scale: Sequence[float] | None = None
    ipopt_options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SolveRecord:
    """Outcome of one NMPC solve.

    ``success`` is IPOPT's verdict; when it is ``False`` the previous input was
    held. ``solve_time`` is wall-clock time of the solver call [s]; ``cost`` is
    the optimal objective (NaN after an exception).
    """

    t: float
    success: bool
    status: str
    iterations: int
    solve_time: float
    cost: float


def _vec(values: Sequence[float] | None, n: int, default: float, name: str) -> np.ndarray:
    if values is None:
        return np.full(n, default)
    arr = np.atleast_1d(np.asarray(values, dtype=float))
    if arr.shape != (n,):
        raise ValueError(f"{name} must have {n} entries, got shape {arr.shape}")
    return arr


class NMPC:
    """Nonlinear MPC implementing :class:`pcsp.core.interfaces.Controller`.

    Build it from any :class:`~pcsp.core.interfaces.ControlModel` whose
    ``rhs``/``output`` accept CasADi ``SX`` symbols. See the module docstring
    for the optimal control problem.

    Attributes
    ----------
    solve_log : one :class:`SolveRecord` per call of :meth:`step`
    u_ref : input reference of the ``S`` penalty; may be changed between steps
        (it is an NLP parameter, so no rebuild is needed)
    """

    def __init__(self, model: ControlModel, config: NMPCConfig) -> None:
        self.model = model
        self.config = config
        c = config
        nx, nu, ny = model.nx, model.nu, model.ny
        if c.horizon < 1 or c.dt <= 0.0:
            raise ValueError("need horizon >= 1 and dt > 0")
        self.tracked = np.asarray(c.tracked_outputs, dtype=int)
        if np.any((self.tracked < 0) | (self.tracked >= ny)):
            raise ValueError(f"tracked_outputs must index the {ny} model outputs")
        nt = self.tracked.size
        self._q = _vec(c.output_weight, nt, 0.0, "output_weight")
        self._r = _vec(c.move_weight, nu, 0.0, "move_weight")
        self._s = _vec(c.input_weight, nu, 0.0, "input_weight")
        self.u_ref = _vec(c.u_ref, nu, 0.0, "u_ref")
        if c.input_weight is not None and c.u_ref is None:
            raise ValueError("input_weight requires u_ref")
        self._u_min = _vec(c.u_min, nu, 0.0, "u_min")
        self._u_max = _vec(c.u_max, nu, 0.0, "u_max")
        self._u_init = _vec(c.u_init, nu, 0.0, "u_init")
        if np.any(self._u_min >= self._u_max):
            raise ValueError("need u_min < u_max")
        if np.any((self._u_init < self._u_min) | (self._u_init > self._u_max)):
            raise ValueError("u_init must lie within [u_min, u_max]")
        self._rate = None if c.rate_max is None else _vec(c.rate_max, nu, 0.0, "rate_max")
        self._x_scale = _vec(c.x_scale, nx, 1.0, "x_scale")
        self._x_offset = _vec(c.x_offset, nx, 0.0, "x_offset")
        self._u_scale = _vec(c.u_scale, nu, 1.0, "u_scale")
        self._u_offset = _vec(c.u_offset, nu, 0.0, "u_offset")
        self._y_scale = _vec(c.y_scale, ny, 1.0, "y_scale")
        y_lo = _vec(c.y_soft_min, ny, -np.inf, "y_soft_min")
        y_hi = _vec(c.y_soft_max, ny, np.inf, "y_soft_max")
        self._soft = np.flatnonzero(np.isfinite(y_lo) | np.isfinite(y_hi))
        if self._soft.size and c.soft_weight is None:
            raise ValueError("soft output bounds require soft_weight")
        self._y_lo, self._y_hi = y_lo, y_hi
        self._w2 = _vec(c.soft_weight, ny, 0.0, "soft_weight")
        self._w1 = _vec(c.soft_weight_linear, ny, 0.0, "soft_weight_linear")

        self._build()
        self.reset()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        """Construct the parametric NLP and the IPOPT solver (called once)."""
        m, c = self.model, self.config
        nx, nu, nd = m.nx, m.nu, m.nd
        n_soft, N = self._soft.size, c.horizon
        F = casadi_discretize(m, c.dt, c.substeps)
        _, h = casadi_functions(m)

        Z = ca.SX.sym("z", nx, N + 1)
        V = ca.SX.sym("v", nu, N)
        E = ca.SX.sym("eps", n_soft, N)
        x0 = ca.SX.sym("x0", nx)
        D = ca.SX.sym("d", nd, N)
        r = ca.SX.sym("r", self.tracked.size)
        u_prev = ca.SX.sym("u_prev", nu)
        u_ref = ca.SX.sym("u_ref", nu)

        xs, xo = ca.DM(self._x_scale), ca.DM(self._x_offset)
        us, uo = ca.DM(self._u_scale), ca.DM(self._u_offset)
        X = [xo + xs * Z[:, k] for k in range(N + 1)]
        U = [uo + us * V[:, k] for k in range(N)]

        g = [(X[0] - x0) / xs]
        lbg = [np.zeros(nx)]
        ubg = [np.zeros(nx)]
        J = 0
        u_last = u_prev
        for k in range(N):
            g.append((F(X[k], U[k], D[:, k]) - X[k + 1]) / xs)
            lbg.append(np.zeros(nx))
            ubg.append(np.zeros(nx))
            du = U[k] - u_last
            J += ca.sum1(ca.DM(self._r) * du**2) + ca.sum1(ca.DM(self._s) * (U[k] - u_ref) ** 2)
            if self._rate is not None:
                g.append(du / us)
                lim = self._rate * c.dt / self._u_scale
                lbg.append(-lim)
                ubg.append(lim)
            y = h(X[k + 1], U[k], D[:, k])
            J += ca.sum1(ca.DM(self._q) * (y[self.tracked.tolist()] - r) ** 2)
            for j, i in enumerate(self._soft):
                ys = self._y_scale[i]
                eps = ys * E[j, k]
                if np.isfinite(self._y_lo[i]):  # y + eps >= y_min
                    g.append((y[i] + eps) / ys)
                    lbg.append(np.array([self._y_lo[i] / ys]))
                    ubg.append(np.array([np.inf]))
                if np.isfinite(self._y_hi[i]):  # y - eps <= y_max
                    g.append((y[i] - eps) / ys)
                    lbg.append(np.array([-np.inf]))
                    ubg.append(np.array([self._y_hi[i] / ys]))
                J += self._w2[i] * eps**2 + self._w1[i] * eps
            u_last = U[k]

        w = ca.vertcat(ca.vec(Z), ca.vec(V), ca.vec(E))
        p = ca.vertcat(x0, ca.vec(D), r, u_prev, u_ref)
        ipopt = {"print_level": 0, "sb": "yes", "tol": 1e-8, **dict(c.ipopt_options)}
        opts = {"ipopt": ipopt, "print_time": False, "error_on_fail": False}
        self._solver = ca.nlpsol(
            "nmpc", "ipopt", {"x": w, "p": p, "f": J, "g": ca.vertcat(*g)}, opts
        )
        self._lbg = np.concatenate(lbg)
        self._ubg = np.concatenate(ubg)
        nz, nv = nx * (N + 1), nu * N
        self._lbw = np.concatenate(
            [
                np.full(nz, -np.inf),
                np.tile((self._u_min - self._u_offset) / self._u_scale, N),
                np.zeros(n_soft * N),
            ]
        )
        self._ubw = np.concatenate(
            [
                np.full(nz, np.inf),
                np.tile((self._u_max - self._u_offset) / self._u_scale, N),
                np.full(n_soft * N, np.inf),
            ]
        )
        self._split = (nz, nz + nv)

    # ------------------------------------------------------------- interface
    def reset(self) -> None:
        """Forget the warm start and restart from ``u_init``; clears ``solve_log``."""
        self._u_prev = self._u_init.copy()
        self._w_warm: np.ndarray | None = None
        self.solve_log: list[SolveRecord] = []

    def step(
        self,
        t: float,
        y: np.ndarray,
        r: np.ndarray,
        x_hat: np.ndarray | None = None,
        d_hat: np.ndarray | None = None,
        *,
        d_preview: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve the OCP from ``x_hat`` and return ``u_0`` (shape ``(nu,)``).

        Parameters
        ----------
        t : current time [s] (only recorded)
        y : measurement (unused; feedback enters through ``x_hat`` and ``d_hat``)
        r : setpoint of the tracked outputs, shape ``(len(tracked_outputs),)``
        x_hat : full model state, shape ``(nx,)``; required
        d_hat : disturbance held over the horizon, shape ``(nd,)``; required
            when ``nd > 0`` unless ``d_preview`` is given
        d_preview : optional forecast, shape ``(N, nd)``; row ``j`` is ``d_j``
        """
        m, N = self.model, self.config.horizon
        if x_hat is None:
            raise ValueError("NMPC needs the full state: pass x_hat")
        x0 = np.asarray(x_hat, dtype=float).reshape(-1)
        if x0.shape != (m.nx,):
            raise ValueError(f"x_hat must have {m.nx} entries, got {x0.shape}")
        if d_preview is not None:
            D = np.asarray(d_preview, dtype=float).reshape(N, m.nd)
        elif m.nd == 0:
            D = np.zeros((N, 0))
        elif d_hat is None:
            raise ValueError("model has disturbances: pass d_hat or d_preview")
        else:
            D = np.tile(np.asarray(d_hat, dtype=float).reshape(1, m.nd), (N, 1))
        r_vec = np.asarray(r, dtype=float).reshape(-1)
        if r_vec.shape != (self.tracked.size,):
            raise ValueError(f"r must have {self.tracked.size} entries, got {r_vec.shape}")
        p = np.concatenate([x0, D.reshape(-1), r_vec, self._u_prev, self.u_ref])

        w0 = self._initial_guess(x0)
        tic = time.perf_counter()
        try:
            sol = self._solver(
                x0=w0, p=p, lbx=self._lbw, ubx=self._ubw, lbg=self._lbg, ubg=self._ubg
            )
            stats = self._solver.stats()
            w = np.asarray(sol["x"], dtype=float).reshape(-1)
            cost = float(sol["f"])
            success = bool(stats["success"])
            status = str(stats["return_status"])
            iterations = int(stats.get("iter_count", -1))
        except RuntimeError as err:  # e.g. NaN in a CasADi function evaluation
            w, cost, success, status, iterations = w0, np.nan, False, f"exception: {err}", -1
        solve_time = time.perf_counter() - tic

        i0 = self._split[0]
        u_opt = self._u_offset + self._u_scale * w[i0 : i0 + m.nu]
        if success and np.all(np.isfinite(w)):
            self._u_prev = u_opt
            self._w_warm = w
        else:
            success = False
            self._w_warm = None
        self.solve_log.append(SolveRecord(float(t), success, status, iterations, solve_time, cost))
        return self._u_prev.copy()

    def diagnostics(self) -> dict[str, float]:
        """Last solve as floats: ``success`` (1/0), ``iterations``, ``solve_time``, ``cost``."""
        if not self.solve_log:
            return {}
        rec = self.solve_log[-1]
        return {
            "success": float(rec.success),
            "iterations": float(rec.iterations),
            "solve_time": rec.solve_time,
            "cost": rec.cost,
        }

    # --------------------------------------------------------------- helpers
    def _initial_guess(self, x0: np.ndarray) -> np.ndarray:
        """Shifted previous solution, or a constant trajectory on a cold start."""
        nx, nu, N = self.model.nx, self.model.nu, self.config.horizon
        z0 = (x0 - self._x_offset) / self._x_scale
        i0, i1 = self._split
        if self._w_warm is None:
            Z = np.tile(z0[:, None], (1, N + 1))
            V = np.tile(((self._u_prev - self._u_offset) / self._u_scale)[:, None], (1, N))
            E = np.zeros((self._soft.size, N))
        else:
            w = self._w_warm
            Z = w[:i0].reshape(nx, N + 1, order="F")
            V = w[i0:i1].reshape(nu, N, order="F")
            E = w[i1:].reshape(self._soft.size, N, order="F")
            Z = np.concatenate([Z[:, 1:], Z[:, -1:]], axis=1)
            V = np.concatenate([V[:, 1:], V[:, -1:]], axis=1)
            E = np.concatenate([E[:, 1:], E[:, -1:]], axis=1)
            Z[:, 0] = z0
        return np.concatenate(
            [Z.reshape(-1, order="F"), V.reshape(-1, order="F"), E.reshape(-1, order="F")]
        )
