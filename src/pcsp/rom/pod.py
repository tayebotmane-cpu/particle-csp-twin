"""Proper Orthogonal Decomposition (POD) of a snapshot matrix.

Snapshot convention (used throughout :mod:`pcsp.rom`): a snapshot matrix ``X``
has shape ``(n_dof, n_snapshots)``; column ``j`` is one full field.

Formulation
-----------
Let ``w`` (length ``n_dof``, all positive) be quadrature weights, e.g. cell
volumes, defining the inner product ``<u, v>_W = u^T W v`` with
``W = diag(w)`` (``w = 1`` gives the Euclidean inner product). Optionally
subtract the snapshot mean ``xbar = X 1 / n_snapshots``. Take the thin SVD of
the weighted, centred data

    W^{1/2} (X - xbar 1^T) = U S V^T,   s_1 >= s_2 >= ... >= 0,

and define the POD modes ``Phi = W^{-1/2} U[:, :r]``. They are orthonormal in
the weighted inner product, ``Phi^T W Phi = I_r``. A field is projected and
reconstructed as

    a = Phi^T W (x - xbar),          x_r = xbar + Phi a.

By the Eckart-Young-Mirsky theorem (Eckart & Young 1936) this rank-``r``
reconstruction is optimal in the weighted Frobenius norm and its error is the
discarded energy,

    || W^{1/2} (X - X_r) ||_F^2 = sum_{i > r} s_i^2.

The cumulative energy is ``E_r = sum_{i <= r} s_i^2 / sum_i s_i^2``, so the
relative reconstruction error of the (centred) training data is
``sqrt(1 - E_r)``.

References
----------
Holmes, Lumley, Berkooz & Rowley (2012), *Turbulence, Coherent Structures,
Dynamical Systems and Symmetry*, 2nd ed., Cambridge University Press, ch. 3.
Volkwein (2013), *Proper orthogonal decomposition: theory and reduced-order
modelling*, lecture notes, Univ. Konstanz (weighted inner products, sec. 1.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class POD:
    """A fitted POD basis. Build it with :func:`fit_pod`.

    Snapshot matrices passed to the methods have shape ``(n_dof, n_snapshots)``;
    a single field of shape ``(n_dof,)`` is also accepted.

    Attributes
    ----------
    modes : ``(n_dof, rank)`` basis ``Phi``, orthonormal in the weighted inner product.
    singular_values : all singular values ``s_i`` of ``W^{1/2}(X - xbar)``,
        not only the retained ones, so the energy spectrum stays available.
    mean : ``(n_dof,)`` snapshot mean ``xbar`` (zeros if the data were not centred).
    weights : ``(n_dof,)`` quadrature weights ``w``, or ``None`` for ``w = 1``.
    """

    modes: FloatArray
    singular_values: FloatArray
    mean: FloatArray
    weights: FloatArray | None

    @property
    def rank(self) -> int:
        """Number of retained modes ``r``."""
        return self.modes.shape[1]

    @property
    def n_dof(self) -> int:
        """Length of one snapshot."""
        return self.modes.shape[0]

    def cumulative_energy(self) -> FloatArray:
        """``E_k = sum_{i<=k} s_i^2 / sum_i s_i^2`` for every available ``k = 1, 2, ...``."""
        return _cumulative_energy(self.singular_values)

    def project(self, X: ArrayLike) -> FloatArray:
        """POD coefficients ``a = Phi^T W (x - xbar)``.

        ``X`` has shape ``(n_dof, n_snapshots)`` (returns ``(rank, n_snapshots)``)
        or ``(n_dof,)`` (returns ``(rank,)``).
        """
        X = np.asarray(X, dtype=float)
        if X.shape[0] != self.n_dof:
            raise ValueError(f"snapshots have {X.shape[0]} rows, the basis has {self.n_dof}")
        fluct = X - self.mean[:, None] if X.ndim == 2 else X - self.mean
        if self.weights is not None:
            fluct = fluct * (self.weights[:, None] if X.ndim == 2 else self.weights)
        return self.modes.T @ fluct

    def reconstruct(self, A: ArrayLike) -> FloatArray:
        """Fields ``x = xbar + Phi a`` from coefficients.

        ``A`` has shape ``(rank, n_snapshots)`` (returns ``(n_dof, n_snapshots)``)
        or ``(rank,)`` (returns ``(n_dof,)``).
        """
        A = np.asarray(A, dtype=float)
        if A.shape[0] != self.rank:
            raise ValueError(f"coefficients have {A.shape[0]} rows, the basis has {self.rank}")
        out = self.modes @ A
        return out + self.mean[:, None] if A.ndim == 2 else out + self.mean

    def truncate(self, rank: int) -> POD:
        """The same basis keeping only the first ``rank`` modes (for rank sweeps)."""
        if not 1 <= rank <= self.rank:
            raise ValueError(f"rank must be in [1, {self.rank}], got {rank}")
        return POD(self.modes[:, :rank], self.singular_values, self.mean, self.weights)


def rank_for_energy(singular_values: ArrayLike, energy: float) -> int:
    """Smallest ``r`` with cumulative energy ``E_r >= energy`` (``0 < energy <= 1``)."""
    if not 0.0 < energy <= 1.0:
        raise ValueError(f"energy must be in (0, 1], got {energy}")
    cumulative = _cumulative_energy(np.asarray(singular_values, dtype=float))
    rank = int(np.searchsorted(cumulative, energy)) + 1
    return min(rank, cumulative.size)


def fit_pod(
    X: ArrayLike,
    *,
    rank: int | None = None,
    energy: float | None = None,
    center: bool = False,
    weights: ArrayLike | None = None,
) -> POD:
    """Compute a POD basis of the snapshot matrix ``X`` of shape ``(n_dof, n_snapshots)``.

    Parameters
    ----------
    X : snapshot matrix, shape ``(n_dof, n_snapshots)``.
    rank : number of modes to keep. Mutually exclusive with ``energy``.
    energy : keep the smallest number of modes whose cumulative energy reaches this
        fraction (e.g. ``0.9999``). If neither ``rank`` nor ``energy`` is given,
        all ``min(n_dof, n_snapshots)`` modes are kept.
    center : subtract the snapshot mean before the SVD (the basis then spans
        fluctuations about the mean, and the energy refers to fluctuations).
    weights : positive quadrature weights of shape ``(n_dof,)`` (e.g. cell volumes)
        defining the inner product ``u^T diag(w) v``. ``None`` means ``w = 1``.

    See the module docstring for the formulation.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be a (n_dof, n_snapshots) matrix, got shape {X.shape}")
    if rank is not None and energy is not None:
        raise ValueError("give either rank or energy, not both")
    n_dof = X.shape[0]

    mean = X.mean(axis=1) if center else np.zeros(n_dof)
    fluct = X - mean[:, None]

    w: FloatArray | None = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n_dof,) or not np.all(w > 0.0):
            raise ValueError(f"weights must be positive with shape ({n_dof},)")
        fluct = fluct * np.sqrt(w)[:, None]

    U, s, _ = np.linalg.svd(fluct, full_matrices=False)

    if energy is not None:
        rank = rank_for_energy(s, energy)
    elif rank is None:
        rank = s.size
    if not 1 <= rank <= s.size:
        raise ValueError(f"rank must be in [1, {s.size}], got {rank}")

    modes = U[:, :rank]
    if w is not None:
        modes = modes / np.sqrt(w)[:, None]
    return POD(modes=modes, singular_values=s, mean=mean, weights=w)


def _cumulative_energy(singular_values: FloatArray) -> FloatArray:
    energy = np.cumsum(singular_values**2)
    return energy / energy[-1]
