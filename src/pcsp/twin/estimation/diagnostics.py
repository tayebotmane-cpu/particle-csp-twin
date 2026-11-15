"""Filter consistency statistics and local observability analysis.

Consistency (Bar-Shalom, Li & Kirubarajan, 2001, sec. 5.4)
------------------------------------------------------------
For a filter whose reported covariances are correct, with estimation error
``e = x - x̂`` (true minus estimated) and innovation ``ν = y - ŷ``:

.. math::

    \\epsilon = e^\\top P^{-1} e \\sim \\chi^2_{n_x} \\quad \\text{(NEES)}, \\qquad
    \\epsilon_\\nu = \\nu^\\top S^{-1} \\nu \\sim \\chi^2_{n_y} \\quad \\text{(NIS)}.

Averaged over ``M`` independent Monte Carlo runs, ``M ε̄ ~ χ²_{M n}``, which
gives the two-sided acceptance interval of :func:`chi2_bounds`. A mean NEES
above the interval means the filter is over-confident (``P`` too small); below
it, under-confident.

Observability (local, discrete time)
------------------------------------
Linearising the sampled model at ``(x, u, d)`` gives ``A = ∂F/∂x`` and
``C = ∂h/∂x``. The pair is (locally, weakly) observable iff the observability
matrix ``O = [C; C A; ...; C A^{L-1}]`` has rank ``n_x`` (``L = n_x`` suffices by
Cayley-Hamilton). Its condition number measures how strongly the noise is
amplified when the state is reconstructed; it depends on units, so pass
typical magnitudes in ``state_scale`` and ``output_scale``. This is a local
test at one operating point (Hermann & Krener, 1977 for the nonlinear notion).

References
----------
* Y. Bar-Shalom, X. R. Li, T. Kirubarajan (2001), *Estimation with Applications
  to Tracking and Navigation*, Wiley, sec. 5.4.
* R. Hermann, A. J. Krener (1977), "Nonlinear controllability and
  observability", *IEEE Trans. Automatic Control* 22(5), 728-740.
* C.-T. Chen (1999), *Linear System Theory and Design*, 3rd ed., OUP, ch. 6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from pcsp.core.interfaces import ControlModel
from pcsp.twin.estimation._linearise import discrete_jacobians


def nees(error: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Normalised estimation error squared ``eᵀ P⁻¹ e``.

    ``error`` has shape ``(..., n)`` and ``P`` shape ``(..., n, n)``; leading
    dimensions (time, Monte Carlo run) broadcast. Returns shape ``(...)``.
    """
    return _normalised_squares(error, P)


def nis(innovation: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Normalised innovation squared ``νᵀ S⁻¹ ν``; shapes as in :func:`nees`."""
    return _normalised_squares(innovation, S)


def chi2_bounds(dof: int, n_runs: int = 1, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided acceptance interval for the mean of ``n_runs`` i.i.d. ``χ²_dof`` values.

    Returns ``(χ²⁻¹_{Mn}(α/2) / M, χ²⁻¹_{Mn}(1-α/2) / M)`` with ``α = 1 - confidence``,
    ``M = n_runs`` and ``n = dof``.
    """
    alpha = 1.0 - confidence
    k = dof * n_runs
    lo, hi = stats.chi2.ppf([alpha / 2.0, 1.0 - alpha / 2.0], k)
    return float(lo / n_runs), float(hi / n_runs)


def rmse(estimate: np.ndarray, truth: np.ndarray, axis: int | None = 0) -> np.ndarray:
    """Root-mean-square error ``sqrt(mean((estimate - truth)²))`` along ``axis``.

    With the default ``axis=0`` and arrays of shape ``(n_steps, n)`` this is
    the per-component RMSE over time; ``axis=None`` pools everything.
    """
    diff = np.asarray(estimate, dtype=float) - np.asarray(truth, dtype=float)
    return np.sqrt(np.mean(diff**2, axis=axis))


@dataclass(frozen=True)
class ObservabilityResult:
    """Outcome of :func:`observability`.

    Attributes
    ----------
    rank : numerical rank of the (scaled) observability matrix
    n_states : ``nx``; the model is locally observable iff ``rank == n_states``
    singular_values : singular values of the scaled observability matrix, descending
    condition_number : largest over smallest singular value (``inf`` if rank-deficient)
    unobservable_directions : orthonormal basis (columns, in scaled state
        coordinates) of the numerically unobservable subspace; shape ``(nx, nx - rank)``
    """

    rank: int
    n_states: int
    singular_values: np.ndarray
    condition_number: float
    unobservable_directions: np.ndarray

    @property
    def observable(self) -> bool:
        return self.rank == self.n_states


def observability(
    model: ControlModel,
    x: np.ndarray,
    u: np.ndarray,
    d: np.ndarray,
    dt: float,
    substeps: int = 1,
    horizon: int | None = None,
    state_scale: np.ndarray | None = None,
    output_scale: np.ndarray | None = None,
    rtol: float = 1e-9,
) -> ObservabilityResult:
    """Local observability of the sampled model at the operating point ``(x, u, d)``.

    Builds ``O = [C; C A; ...; C A^{L-1}]`` with exact CasADi Jacobians of the
    RK4 map, in scaled coordinates ``x = D_x x̃``, ``y = D_y ỹ``
    (``Õ = blkdiag(D_y⁻¹) O D_x``). A singular value counts towards the rank
    if it exceeds ``rtol`` times the largest one.

    Parameters
    ----------
    model, x, u, d : model and operating point
    dt, substeps : sample period [s] and RK4 substeps of the filter
    horizon : number of block rows ``L`` (default ``nx``)
    state_scale, output_scale : typical magnitudes ``D_x``, ``D_y`` (default ones)
    rtol : relative singular-value threshold for the numerical rank
    """
    transition, measurement = discrete_jacobians(model, dt, substeps)
    x, u, d = (np.asarray(v, dtype=float).ravel() for v in (x, u, d))
    A = transition(x, u, d)[1].full()
    C = measurement(x, u, d)[1].full()
    nx = model.nx
    Dx = np.ones(nx) if state_scale is None else np.asarray(state_scale, dtype=float)
    Dy = np.ones(model.ny) if output_scale is None else np.asarray(output_scale, dtype=float)
    A_s = A * Dx[None, :] / Dx[:, None]  # D_x⁻¹ A D_x
    C_s = C * Dx[None, :] / Dy[:, None]  # D_y⁻¹ C D_x

    blocks, CA = [], C_s
    for _ in range(nx if horizon is None else horizon):
        blocks.append(CA)
        CA = CA @ A_s
    _, sv, Vt = np.linalg.svd(np.vstack(blocks))  # Vt is (nx, nx): full_matrices=True
    sv_full = np.zeros(nx)
    sv_full[: sv.size] = sv
    rank = int(np.sum(sv_full > rtol * sv_full[0])) if sv_full[0] > 0 else 0
    cond = float(sv_full[0] / sv_full[-1]) if rank == nx else float("inf")
    return ObservabilityResult(
        rank=rank,
        n_states=nx,
        singular_values=sv_full,
        condition_number=cond,
        unobservable_directions=Vt[rank:].T,
    )


def _normalised_squares(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    M = np.asarray(M, dtype=float)
    return np.einsum("...i,...i->...", v, np.linalg.solve(M, v[..., None])[..., 0])
