"""Discrete single-input single-output PID controller.

Control law (parallel form), evaluated once per sample period ``h`` [s]:

.. math::

    e_k &= s\\,(r_k - y_k), \\qquad s = +1 \\text{ (reverse action)},\\; -1 \\text{ (direct)}

    P_k &= K_p e_k

    D_k &= \\frac{T_f}{T_f + h} D_{k-1} - s\\,\\frac{K_d}{T_f + h}\\,(y_k - y_{k-1})

    F_k &= \\operatorname{clip}\\big(u_\\text{ff}(d_k),\\;
           F_{k-1} - \\rho h,\\; F_{k-1} + \\rho h\\big), \\qquad F_{-1} = 0

    v_k &= u_\\text{bias} + P_k + I_k + D_k + F_k

    u_k &= \\operatorname{clip}\\big(v_k,\\;
           \\max(u_\\min, u_{k-1} - \\rho h),\\; \\min(u_\\max, u_{k-1} + \\rho h)\\big)

    I_{k+1} &= I_k + K_i h e_k + \\frac{h}{T_t}(u_k - v_k)

* The derivative acts on the measurement, not the error, so setpoint steps
  cause no derivative kick. It is the backward-Euler discretisation of
  :math:`K_d s / (1 + T_f s)`, stable for every ``T_f >= 0``.
* ``u_k`` is limited by output saturation ``[u_min, u_max]`` and by the slew
  rate ``rho`` [input units / s] in one clip; ``u_{k-1}`` is the previous
  *applied* output.
* Anti-windup is back-calculation (Åström & Hägglund, *Advanced PID Control*,
  2006, sec. 3.5) driven by the difference between the applied and the
  unlimited output. Because ``u_k`` already includes both the saturation and
  the rate limit, the integrator stops winding up under either. With
  ``T_t = h`` the integrator is reset in one sample so that the next unlimited
  output equals the applied one.
* The feedforward term ``F_k`` follows ``u_ff(d_k)`` no faster than the slew
  rate. A feedforward step larger than one slew step would otherwise saturate
  the rate limit, and back-calculation would move the clipped part into the
  integral, cancelling the feedforward for good (with ``T_t = h``, entirely).
* Sign convention (ISA): a *reverse-acting* controller raises its output when
  the measurement falls, which suits a positive-gain process (heater power ->
  temperature). A *direct-acting* controller raises its output when the
  measurement rises, which suits a negative-gain process such as particle mass
  flow -> receiver outlet temperature. Gains are always non-negative; the sign
  comes from ``action``.
* The integrator uses forward Euler, so ``u_k`` depends on ``e_k`` only through
  ``P_k`` and ``D_k`` (no algebraic loop with the limiter).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np


class PIDController:
    """Discrete PID implementing :class:`pcsp.core.interfaces.Controller`.

    Parameters
    ----------
    kp : proportional gain [input units / output units]
    ki : integral gain [input units / (output units * s)]
    kd : derivative gain [input units * s / output units]
    dt : sample period ``h`` [s]
    action : ``"reverse"`` (``e = r - y``) or ``"direct"`` (``e = y - r``)
    bias : output when all terms are zero; also the initial output, so a
        controller started at ``y == r`` with ``bias`` equal to the current
        actuator position starts bumplessly [input units]
    u_min, u_max : output saturation limits [input units]
    rate_max : slew-rate limit [input units / s]; ``None`` for no limit
    derivative_filter_time : ``T_f`` [s]; default ``T_d / 10`` with
        ``T_d = kd / kp`` (``dt`` if ``kp == 0``)
    tracking_time : back-calculation time constant ``T_t`` [s]; default
        ``sqrt(T_i T_d)`` (Åström & Hägglund's rule of thumb, ``T_i = kp / ki``),
        never smaller than ``dt`` so the discrete reset is stable
    anti_windup : ``False`` gives a naive integrator (for comparison studies)
    feedforward : optional static feedforward ``u_ff(d)``: a callable of the
        measured disturbance vector returning the input *deviation* from
        ``bias`` that cancels it [input units]. It starts from zero and is
        applied no faster than ``rate_max``
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        dt: float,
        *,
        action: Literal["reverse", "direct"] = "reverse",
        bias: float = 0.0,
        u_min: float = -np.inf,
        u_max: float = np.inf,
        rate_max: float | None = None,
        derivative_filter_time: float | None = None,
        tracking_time: float | None = None,
        anti_windup: bool = True,
        feedforward: Callable[[np.ndarray], float] | None = None,
    ) -> None:
        if min(kp, ki, kd) < 0.0:
            raise ValueError("gains must be non-negative; choose the sign with `action`")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        if action not in ("reverse", "direct"):
            raise ValueError(f"action must be 'reverse' or 'direct', got {action!r}")
        if not u_min < u_max:
            raise ValueError(f"need u_min < u_max, got [{u_min}, {u_max}]")
        if rate_max is not None and rate_max <= 0.0:
            raise ValueError(f"rate_max must be positive, got {rate_max}")
        self.kp, self.ki, self.kd, self.dt = float(kp), float(ki), float(kd), float(dt)
        self.sign = 1.0 if action == "reverse" else -1.0
        self.bias = float(bias)
        self.u_min, self.u_max = float(u_min), float(u_max)
        self.rate_max = np.inf if rate_max is None else float(rate_max)
        self.anti_windup = anti_windup
        self.feedforward = feedforward

        t_d = kd / kp if kp > 0.0 else 0.0
        if derivative_filter_time is None:
            derivative_filter_time = t_d / 10.0 if kp > 0.0 else dt
        if derivative_filter_time < 0.0:
            raise ValueError("derivative_filter_time must be non-negative")
        self.t_f = float(derivative_filter_time)
        if tracking_time is None:
            t_i = kp / ki if ki > 0.0 else np.inf
            tracking_time = np.sqrt(t_i * t_d) if t_d > 0.0 else dt
        self.t_t = max(float(tracking_time), dt)
        self.reset()

    def reset(self) -> None:
        """Zero the integral and derivative states; output restarts at ``bias``."""
        self._i = 0.0
        self._d = 0.0
        self._ff = 0.0
        self._y_prev: float | None = None
        self._u_prev = float(np.clip(self.bias, self.u_min, self.u_max))
        self._last: dict[str, float] = {}

    def step(
        self,
        t: float,
        y: np.ndarray,
        r: np.ndarray,
        x_hat: np.ndarray | None = None,
        d_hat: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return ``u_k`` (shape ``(1,)``) from measurement ``y`` and setpoint ``r``.

        ``x_hat`` is ignored. ``d_hat`` is used only by the feedforward term.
        """
        y_k = float(np.asarray(y, dtype=float).reshape(-1)[0])
        r_k = float(np.asarray(r, dtype=float).reshape(-1)[0])
        h = self.dt
        e = self.sign * (r_k - y_k)

        p = self.kp * e
        dy = 0.0 if self._y_prev is None else y_k - self._y_prev
        self._d = (self.t_f * self._d - self.sign * self.kd * dy) / (self.t_f + h)
        ff = 0.0
        if self.feedforward is not None:
            if d_hat is None:
                raise ValueError("feedforward is configured but no disturbance d_hat was given")
            target = float(self.feedforward(np.asarray(d_hat, dtype=float)))
            ff = min(max(target, self._ff - self.rate_max * h), self._ff + self.rate_max * h)
            self._ff = ff

        v = self.bias + p + self._i + self._d + ff
        lo = max(self.u_min, self._u_prev - self.rate_max * h)
        hi = min(self.u_max, self._u_prev + self.rate_max * h)
        u = min(max(v, lo), hi)

        self._last = {"p": p, "i": self._i, "d": self._d, "ff": ff, "u_unlimited": v}
        self._i += self.ki * h * e
        if self.anti_windup:
            self._i += (h / self.t_t) * (u - v)
        self._y_prev = y_k
        self._u_prev = u
        return np.array([u])

    def diagnostics(self) -> dict[str, float]:
        """Terms of the last step: ``p``, ``i``, ``d``, ``ff`` and ``u_unlimited``."""
        return dict(self._last)
