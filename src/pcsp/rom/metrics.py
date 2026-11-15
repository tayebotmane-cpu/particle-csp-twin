"""Error measures and timing for reduced-order models.

Snapshot convention: matrices have shape ``(n_dof, n_snapshots)``.

Two errors must be reported separately for any projection-based ROM:

* the **projection error** ``||X - P X|| / ||X||`` where ``P`` is the (affine,
  weighted-orthogonal) projection onto the POD basis: the best any model
  living in that basis can do;
* the **ROM error** ``||X - X_rom|| / ||X||`` of what the ROM actually predicts.

The ROM error is never below the projection error; their gap is the error of the
reduced dynamics or of the coefficient interpolation. With quadrature weights
``w`` all norms are the weighted Frobenius norm ``||X||_W^2 = sum_ij w_i X_ij^2``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from pcsp.rom.pod import POD


def relative_error(
    X_true: ArrayLike,
    X_approx: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    axis: int | None = None,
) -> float | np.ndarray:
    """Relative (weighted) L2 / Frobenius error ``||X_true - X_approx||_W / ||X_true||_W``.

    Inputs are snapshot matrices of shape ``(n_dof, n_snapshots)`` or single fields
    of shape ``(n_dof,)``. ``weights`` has shape ``(n_dof,)``. With ``axis=None``
    the Frobenius norm of the whole matrix is used and a float is returned; with
    ``axis=0`` the error of each snapshot (column) is returned.
    """
    X_true = np.asarray(X_true, dtype=float)
    X_approx = np.asarray(X_approx, dtype=float)
    if X_true.shape != X_approx.shape:
        raise ValueError(f"shapes differ: {X_true.shape} vs {X_approx.shape}")
    w = np.ones(X_true.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    if X_true.ndim == 2:
        w = w[:, None]
    num = np.sum(w * (X_true - X_approx) ** 2, axis=axis)
    den = np.sum(w * X_true**2, axis=axis)
    result = np.sqrt(num / den)
    return float(result) if axis is None else result


def projection_error(pod: POD, X: ArrayLike, *, axis: int | None = None) -> float | np.ndarray:
    """Relative error of the best approximation of ``X`` in the POD basis.

    ``X`` has shape ``(n_dof, n_snapshots)``. The error is measured in the POD's own
    weighted inner product: ``||X - (xbar + Phi Phi^T W (X - xbar))||_W / ||X||_W``.
    """
    X = np.asarray(X, dtype=float)
    return relative_error(X, pod.reconstruct(pod.project(X)), weights=pod.weights, axis=axis)


@dataclass(frozen=True)
class Timing:
    """Wall-clock seconds of one full-model and one ROM evaluation, and their ratio."""

    full_seconds: float
    rom_seconds: float

    @property
    def speedup(self) -> float:
        """``full_seconds / rom_seconds``."""
        return self.full_seconds / self.rom_seconds


def time_full_vs_rom(
    full: Callable[[], object], rom: Callable[[], object], *, repeats: int = 5
) -> Timing:
    """Time the online cost of a full model and of a ROM producing the same output.

    ``full`` and ``rom`` are zero-argument callables (wrap arguments with a
    ``lambda``). Each is run ``repeats`` times and the minimum wall-clock time
    is kept, the usual estimator for a deterministic workload because noise only
    ever adds time (Python ``timeit`` documentation). Offline costs (sampling,
    SVD, fitting) are deliberately excluded and should be reported separately.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    return Timing(full_seconds=_best_time(full, repeats), rom_seconds=_best_time(rom, repeats))


def _best_time(fn: Callable[[], object], repeats: int) -> float:
    best = np.inf
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best
