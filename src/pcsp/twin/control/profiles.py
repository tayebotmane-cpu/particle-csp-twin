"""Setpoint and disturbance profiles: functions of time returning 1-D arrays.

A *profile* is any callable ``p(t) -> np.ndarray`` with ``t`` in seconds. The
closed-loop simulator samples profiles at the controller sample instants and
holds the value over the sample period (zero-order hold).

Example: a 20 % drop in incident solar power at ``t = 60 s``::

    q_solar = step(60.0, before=1.0e6, after=0.8e6)   # W
    q_solar(30.0)   # array([1000000.])
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

Profile = Callable[[float], np.ndarray]


def constant(value: float | Sequence[float]) -> Profile:
    """Profile that always returns ``value``."""
    v = np.atleast_1d(np.asarray(value, dtype=float))

    def profile(t: float) -> np.ndarray:
        return v.copy()

    return profile


def step(t_step: float, before: float | Sequence[float], after: float | Sequence[float]) -> Profile:
    """``before`` for ``t < t_step``, ``after`` for ``t >= t_step``."""
    return piecewise_constant([t_step], [before, after])


def ramp(
    t_start: float,
    t_end: float,
    start: float | Sequence[float],
    end: float | Sequence[float],
) -> Profile:
    """``start`` until ``t_start``, linear to ``end`` at ``t_end``, then ``end``."""
    if t_end <= t_start:
        raise ValueError(f"t_end ({t_end}) must exceed t_start ({t_start})")
    v0 = np.atleast_1d(np.asarray(start, dtype=float))
    v1 = np.atleast_1d(np.asarray(end, dtype=float))

    def profile(t: float) -> np.ndarray:
        frac = min(max((t - t_start) / (t_end - t_start), 0.0), 1.0)
        return v0 + frac * (v1 - v0)

    return profile


def piecewise_constant(
    breakpoints: Sequence[float], values: Sequence[float | Sequence[float]]
) -> Profile:
    """Piecewise-constant sequence.

    ``values[0]`` holds for ``t < breakpoints[0]``, ``values[i]`` for
    ``breakpoints[i-1] <= t < breakpoints[i]`` and ``values[-1]`` after the last
    breakpoint, so ``len(values) == len(breakpoints) + 1``.
    """
    times = np.asarray(breakpoints, dtype=float)
    vals = [np.atleast_1d(np.asarray(v, dtype=float)) for v in values]
    if len(vals) != times.size + 1:
        raise ValueError(f"need {times.size + 1} values for {times.size} breakpoints")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("breakpoints must be strictly increasing")

    def profile(t: float) -> np.ndarray:
        return vals[int(np.searchsorted(times, t, side="right"))].copy()

    return profile


def as_profile(spec: Any, n_samples: int, dt: float) -> Profile:
    """Turn a user specification into a profile.

    Accepted ``spec``:

    * a callable ``t -> array`` (returned unchanged, output made 1-D);
    * a scalar or 1-D sequence: a constant profile;
    * a 2-D array of shape ``(n_samples, dim)``: row ``k`` is the value at the
      sample instant ``t_k = k * dt`` and is held until ``t_{k+1}``.
    """
    if callable(spec):
        return lambda t: np.atleast_1d(np.asarray(spec(t), dtype=float))
    arr = np.asarray(spec, dtype=float)
    if arr.ndim <= 1:
        return constant(arr)
    if arr.ndim != 2 or arr.shape[0] != n_samples:
        raise ValueError(f"array profile must have shape ({n_samples}, dim), got {arr.shape}")

    def profile(t: float) -> np.ndarray:
        k = min(max(int(np.floor(t / dt + 1e-9)), 0), n_samples - 1)
        return arr[k].copy()

    return profile
