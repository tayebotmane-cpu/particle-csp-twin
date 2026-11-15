"""Step-response and regulation metrics for sampled closed-loop signals.

All functions take sample times ``t`` [s] and signals sampled at those
instants (for example the arrays of a
:class:`~pcsp.twin.control.closed_loop.ClosedLoopResult`). Integrals use the
trapezoidal rule. ``t_start`` restricts a metric to ``t >= t_start``, e.g. to
the period after a disturbance step.

The Sandia falling-particle-receiver criterion (outlet temperature within
+/- 2 sigma <= 10 degC over a 5-minute window) is evaluated with
:func:`two_sigma_band` using ``window=300.0`` and comparing to 10 K.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _after(t: np.ndarray, t_start: float | None, *signals: np.ndarray) -> list[np.ndarray]:
    t = np.asarray(t, dtype=float)
    mask = np.ones(t.shape, bool) if t_start is None else t >= t_start
    if not np.any(mask):
        raise ValueError(f"no samples at or after t_start={t_start}")
    return [t[mask], *(np.asarray(s, dtype=float)[mask] for s in signals)]


def integral_absolute_error(
    t: np.ndarray, y: np.ndarray, target: np.ndarray | float, t_start: float | None = None
) -> float:
    """IAE ``= int |y - target| dt`` [output unit * s]."""
    t, y, ref = _after(t, t_start, y, np.broadcast_to(target, np.shape(y)))
    return float(np.trapezoid(np.abs(y - ref), t))


def integral_squared_error(
    t: np.ndarray, y: np.ndarray, target: np.ndarray | float, t_start: float | None = None
) -> float:
    """ISE ``= int (y - target)^2 dt`` [output unit^2 * s]."""
    t, y, ref = _after(t, t_start, y, np.broadcast_to(target, np.shape(y)))
    return float(np.trapezoid((y - ref) ** 2, t))


def peak_deviation(
    t: np.ndarray, y: np.ndarray, target: np.ndarray | float, t_start: float | None = None
) -> float:
    """Signed deviation ``y - target`` of largest magnitude [output unit]."""
    _, y, ref = _after(t, t_start, y, np.broadcast_to(target, np.shape(y)))
    e = y - ref
    return float(e[np.argmax(np.abs(e))])


def overshoot(y: np.ndarray, y_initial: float, y_final: float) -> float:
    """Percent overshoot of a step response from ``y_initial`` to ``y_final``.

    ``100 * max excursion beyond y_final (in the direction of the step) /
    |y_final - y_initial|``; zero if ``y`` never passes ``y_final``.
    """
    span = y_final - y_initial
    if span == 0.0:
        raise ValueError("overshoot is undefined for a zero-size step")
    beyond = np.sign(span) * (np.asarray(y, dtype=float) - y_final)
    return float(100.0 * max(beyond.max(), 0.0) / abs(span))


def settling_time(
    t: np.ndarray,
    y: np.ndarray,
    target: np.ndarray | float,
    band: float,
    t_start: float = 0.0,
) -> float:
    """Time after ``t_start`` from which ``|y - target| <= band`` for all later samples [s].

    Returns ``inf`` if the last sample is outside the band. The answer is
    resolved to one sample period.
    """
    t, y, ref = _after(t, t_start, y, np.broadcast_to(target, np.shape(y)))
    outside = np.flatnonzero(np.abs(y - ref) > band)
    if outside.size == 0:
        return 0.0
    if outside[-1] == t.size - 1:
        return np.inf
    return float(t[outside[-1] + 1] - t_start)


def two_sigma_band(
    t: np.ndarray,
    y: np.ndarray,
    window: float,
    stride: float | None = None,
    t_start: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Twice the sample standard deviation of ``y`` over windows of length ``window``.

    Each window holds ``round(window / dt) + 1`` consecutive samples (so it
    spans ``window`` seconds) and the standard deviation is taken about the
    window mean with ``ddof=1``. Windows advance by ``stride`` seconds: one
    sample (default) gives a sliding window, ``stride = window`` gives fixed,
    back-to-back windows (sharing their boundary sample). Uniform sampling is
    assumed.

    The window sums are computed from cumulative sums of the mean-centred
    signal, so cost and memory are O(len(y)) whatever the window length.

    Returns
    -------
    t_end : time of the last sample in each window [s]
    two_sigma : ``2 * std`` in each window [output unit]
    """
    t, y = _after(t, t_start, y)
    dt = float(np.median(np.diff(t)))
    n_win = int(round(window / dt)) + 1
    if n_win < 2 or n_win > t.size:
        raise ValueError(f"window {window} s must span 2 to {t.size} samples of dt = {dt} s")
    n_stride = 1 if stride is None else max(int(round(stride / dt)), 1)
    yc = y - y.mean()
    s1 = np.concatenate([[0.0], np.cumsum(yc)])
    s2 = np.concatenate([[0.0], np.cumsum(yc**2)])
    start = np.arange(0, t.size - n_win + 1, n_stride)
    end = start + n_win
    win_sum, win_sq = s1[end] - s1[start], s2[end] - s2[start]
    var = np.maximum(win_sq - win_sum**2 / n_win, 0.0) / (n_win - 1)
    return t[end - 1], 2.0 * np.sqrt(var)


def total_variation(u: np.ndarray) -> np.ndarray:
    """Input total variation ``sum_k |u_{k+1} - u_k|`` per input (actuator travel)."""
    u = np.asarray(u, dtype=float)
    u = u.reshape(u.shape[0], -1)
    return np.abs(np.diff(u, axis=0)).sum(axis=0)


@dataclass(frozen=True)
class RegulationMetrics:
    """Summary of one controlled output after ``t_start`` (units as the output)."""

    iae: float
    ise: float
    peak_deviation: float
    settling_time: float
    two_sigma_max: float
    total_variation: np.ndarray


def regulation_metrics(
    t: np.ndarray,
    y: np.ndarray,
    r: np.ndarray | float,
    u: np.ndarray,
    *,
    t_start: float,
    band: float,
    window: float = 300.0,
) -> RegulationMetrics:
    """All metrics for output ``y`` against setpoint ``r`` from ``t_start`` on.

    ``two_sigma_max`` is the worst sliding-window ``2 sigma`` (Sandia
    criterion: ``<= 10 K`` with ``window = 300 s``); ``total_variation`` is
    computed for ``u`` after ``t_start``.
    """
    _, u_after = _after(t, t_start, u)
    return RegulationMetrics(
        iae=integral_absolute_error(t, y, r, t_start),
        ise=integral_squared_error(t, y, r, t_start),
        peak_deviation=peak_deviation(t, y, r, t_start),
        settling_time=settling_time(t, y, r, band, t_start),
        two_sigma_max=float(two_sigma_band(t, y, window, t_start=t_start)[1].max()),
        total_variation=total_variation(u_after),
    )
