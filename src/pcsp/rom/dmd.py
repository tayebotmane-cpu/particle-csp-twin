"""Dynamic Mode Decomposition: exact DMD and DMD with control (DMDc).

Snapshot convention: a snapshot matrix has shape ``(n_dof, n_snapshots)``.
Data are snapshot pairs ``(x_k, y_k)`` with ``y_k`` the state one sampling interval
``dt`` after ``x_k``, stacked as columns of ``X`` and ``Y``. A single trajectory
``[x_0, ..., x_m]`` gives ``X = [x_0 .. x_{m-1}]``, ``Y = [x_1 .. x_m]``; several
trajectories (e.g. runs) are handled by concatenating their pairs.

Exact DMD (Tu et al. 2014, Algorithm 2)
---------------------------------------
Seek the best-fit linear map ``Y ~= A X``. With the rank-``r`` thin SVD
``X ~= U_r S_r V_r^T``:

    A_tilde = U_r^T Y V_r S_r^{-1}            (r x r, = U_r^T A U_r)
    A_tilde W = W Lambda                      (DMD eigenvalues lambda_i)
    Phi = Y V_r S_r^{-1} W                    (exact DMD modes, A Phi = Phi Lambda)

Prediction from ``x_0``: ``x_k = Phi Lambda^k b`` with amplitudes ``b = Phi^+ x_0``
(least squares). Continuous-time eigenvalues are ``omega_i = log(lambda_i) / dt``.

DMD with control (Proctor, Brunton & Kutz 2016, unknown B)
---------------------------------------------------------
Seek ``Y ~= A X + B Upsilon`` with inputs ``Upsilon`` (``n_u x m``). With the
rank-``p`` SVD of the stacked data ``Omega = [X; Upsilon] ~= U~ S~ V~^T`` and
``U~ = [U~_1; U~_2]`` (``n_dof`` and ``n_u`` rows), and the rank-``r`` SVD
``Y ~= U^ S^ V^^T`` of the outputs:

    A_tilde = U^^T Y V~ S~^{-1} U~_1^T U^     (r x r)
    B_tilde = U^^T Y V~ S~^{-1} U~_2^T        (r x n_u)

and the reduced model ``z_{k+1} = A_tilde z_k + B_tilde u_k``, ``x_k ~= U^ z_k``.
DMDc modes are ``Phi = Y V~ S~^{-1} U~_1^T U^ W`` with ``A_tilde W = W Lambda``.

Stability of the identified discrete-time operator: the free response decays iff
the spectral radius ``max_i |lambda_i| < 1`` (equivalently ``Re omega_i < 0``).

References
----------
Tu, Rowley, Luchtenburg, Brunton & Kutz (2014), "On dynamic mode decomposition:
theory and applications", J. Comput. Dyn. 1(2), 391-421.
Proctor, Brunton & Kutz (2016), "Dynamic mode decomposition with control",
SIAM J. Appl. Dyn. Syst. 15(1), 142-161.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class DMD:
    """Exact DMD model. Build it with :func:`fit_dmd`.

    Attributes
    ----------
    eigenvalues : ``(r,)`` complex discrete-time DMD eigenvalues ``lambda_i``.
    modes : ``(n_dof, r)`` complex exact DMD modes ``Phi``.
    basis : ``(n_dof, r)`` POD basis ``U_r`` of the input snapshots ``X``.
    reduced_operator : ``(r, r)`` real matrix ``A_tilde = U_r^T A U_r``.
    singular_values : all singular values of ``X`` (for choosing the rank).
    """

    eigenvalues: np.ndarray
    modes: np.ndarray
    basis: np.ndarray
    reduced_operator: np.ndarray
    singular_values: np.ndarray

    @property
    def rank(self) -> int:
        return self.eigenvalues.size

    def amplitudes(self, x0: ArrayLike) -> np.ndarray:
        """Mode amplitudes ``b = Phi^+ x0`` (least squares) for an ``(n_dof,)`` state."""
        return np.linalg.lstsq(self.modes, np.asarray(x0, dtype=float), rcond=None)[0]

    def predict(self, x0: ArrayLike, n_steps: int) -> np.ndarray:
        """Snapshot matrix ``[x_0, x_1, ..., x_{n_steps}]`` of shape ``(n_dof, n_steps + 1)``.

        ``x_k = Re(Phi Lambda^k b)``. For real data eigenvalues and modes come in
        complex-conjugate pairs, so the discarded imaginary part is round-off.
        """
        b = self.amplitudes(x0)
        powers = self.eigenvalues[:, None] ** np.arange(n_steps + 1)[None, :]
        return np.real(self.modes @ (b[:, None] * powers))

    def continuous_eigenvalues(self, dt: float) -> np.ndarray:
        """``omega_i = log(lambda_i) / dt`` (principal branch), in 1/s."""
        return np.log(self.eigenvalues.astype(complex)) / dt

    def spectral_radius(self) -> float:
        """``max_i |lambda_i|`` of the identified discrete-time operator."""
        return float(np.max(np.abs(self.eigenvalues)))

    def is_stable(self, tol: float = 1e-10) -> bool:
        """True if ``max_i |lambda_i| <= 1 + tol`` (no growing mode)."""
        return self.spectral_radius() <= 1.0 + tol


@dataclass(frozen=True)
class DMDc:
    """DMD with control. Build it with :func:`fit_dmdc`.

    Attributes
    ----------
    eigenvalues : ``(r,)`` complex eigenvalues of ``A_tilde``.
    modes : ``(n_dof, r)`` complex DMDc modes.
    basis : ``(n_dof, r)`` output basis ``U^`` (POD basis of ``Y``).
    A_reduced : ``(r, r)`` reduced state matrix ``A_tilde``.
    B_reduced : ``(r, n_u)`` reduced input matrix ``B_tilde``.
    """

    eigenvalues: np.ndarray
    modes: np.ndarray
    basis: np.ndarray
    A_reduced: np.ndarray
    B_reduced: np.ndarray

    @property
    def rank(self) -> int:
        return self.eigenvalues.size

    def full_operators(self) -> tuple[np.ndarray, np.ndarray]:
        """``(A, B) = (U^ A_tilde U^^T, U^ B_tilde)`` in full coordinates.

        ``A`` is dense ``(n_dof, n_dof)``: only form it for small ``n_dof``.
        """
        return self.basis @ self.A_reduced @ self.basis.T, self.basis @ self.B_reduced

    def predict(self, x0: ArrayLike, inputs: ArrayLike) -> np.ndarray:
        """Simulate ``z_{k+1} = A_tilde z_k + B_tilde u_k`` from ``z_0 = U^^T x0``.

        ``inputs`` has shape ``(n_u, n_steps)``; returns the snapshot matrix
        ``[x_0, ..., x_{n_steps}]`` of shape ``(n_dof, n_steps + 1)`` with
        ``x_k = U^ z_k`` (so ``x_0`` is the projection of the given state).
        """
        U = np.asarray(inputs, dtype=float)
        if U.ndim != 2 or U.shape[0] != self.B_reduced.shape[1]:
            raise ValueError(f"inputs must be ({self.B_reduced.shape[1]}, n_steps)")
        z = np.empty((self.rank, U.shape[1] + 1))
        z[:, 0] = self.basis.T @ np.asarray(x0, dtype=float)
        for k in range(U.shape[1]):
            z[:, k + 1] = self.A_reduced @ z[:, k] + self.B_reduced @ U[:, k]
        return self.basis @ z

    def spectral_radius(self) -> float:
        """``max_i |lambda_i|`` of ``A_tilde``."""
        return float(np.max(np.abs(self.eigenvalues)))

    def is_stable(self, tol: float = 1e-10) -> bool:
        """True if ``max_i |lambda_i| <= 1 + tol`` (unforced response does not grow)."""
        return self.spectral_radius() <= 1.0 + tol


def fit_dmd(X: ArrayLike, Y: ArrayLike | None = None, *, rank: int | None = None) -> DMD:
    """Exact DMD of snapshot data (snapshot matrices have shape ``(n_dof, n_snapshots)``).

    Parameters
    ----------
    X : ``(n_dof, m + 1)`` sequential snapshots if ``Y`` is ``None``, otherwise the
        ``(n_dof, m)`` "before" snapshots.
    Y : optional ``(n_dof, m)`` "after" snapshots, ``Y[:, k]`` one step after ``X[:, k]``.
    rank : SVD truncation rank ``r``; default is the numerical rank of ``X``.
    """
    X, Y = _pairs(X, Y)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    r = _check_rank(s, rank, "X")
    U_r, s_r, V_r = U[:, :r], s[:r], Vt[:r].T
    YVS = Y @ (V_r / s_r)  # Y V_r S_r^{-1}
    A_tilde = U_r.T @ YVS
    eigenvalues, W = np.linalg.eig(A_tilde)
    return DMD(
        eigenvalues=eigenvalues,
        modes=YVS @ W,
        basis=U_r,
        reduced_operator=A_tilde,
        singular_values=s,
    )


def fit_dmdc(
    X: ArrayLike,
    inputs: ArrayLike,
    Y: ArrayLike | None = None,
    *,
    rank_input: int | None = None,
    rank_output: int | None = None,
) -> DMDc:
    """DMD with control, unknown input matrix (snapshot matrices are ``(n_dof, n_snapshots)``).

    Parameters
    ----------
    X : ``(n_dof, m + 1)`` sequential snapshots if ``Y`` is ``None``, else ``(n_dof, m)``.
    inputs : ``(n_u, m)``; ``inputs[:, k]`` acts between ``X[:, k]`` and ``Y[:, k]``.
    Y : optional ``(n_dof, m)`` "after" snapshots.
    rank_input : truncation ``p`` of ``Omega = [X; inputs]``; default its numerical rank.
    rank_output : truncation ``r`` of ``Y``; default its numerical rank.
    """
    X, Y = _pairs(X, Y)
    Ups = np.asarray(inputs, dtype=float)
    if Ups.ndim != 2 or Ups.shape[1] != X.shape[1]:
        raise ValueError(f"inputs must have shape (n_u, {X.shape[1]}), got {Ups.shape}")
    n_dof = X.shape[0]

    Ut, st, Vtt = np.linalg.svd(np.vstack([X, Ups]), full_matrices=False)
    p = _check_rank(st, rank_input, "[X; inputs]")
    Ut, st, Vt_p = Ut[:, :p], st[:p], Vtt[:p].T
    U1, U2 = Ut[:n_dof], Ut[n_dof:]

    Uh, sh, _ = np.linalg.svd(Y, full_matrices=False)
    r = _check_rank(sh, rank_output, "Y")
    Uh = Uh[:, :r]

    YVS = Y @ (Vt_p / st)  # Y V~ S~^{-1}
    A_tilde = Uh.T @ YVS @ (U1.T @ Uh)
    B_tilde = Uh.T @ YVS @ U2.T
    eigenvalues, W = np.linalg.eig(A_tilde)
    return DMDc(
        eigenvalues=eigenvalues,
        modes=YVS @ (U1.T @ Uh) @ W,
        basis=Uh,
        A_reduced=A_tilde,
        B_reduced=B_tilde,
    )


def _pairs(X: ArrayLike, Y: ArrayLike | None) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be (n_dof, n_snapshots), got shape {X.shape}")
    if Y is None:
        return X[:, :-1], X[:, 1:]
    Y = np.asarray(Y, dtype=float)
    if Y.shape != X.shape:
        raise ValueError(f"X and Y must have the same shape, got {X.shape} and {Y.shape}")
    return X, Y


def _check_rank(s: np.ndarray, rank: int | None, what: str) -> int:
    """Validate a truncation rank against the numerical rank (singular values above tol)."""
    tol = s[0] * max(s.size, 1) * np.finfo(float).eps * 10 if s.size else 0.0
    numerical_rank = int(np.sum(s > tol))
    if rank is None:
        rank = numerical_rank
    if not 1 <= rank <= numerical_rank:
        raise ValueError(
            f"rank of {what} must be in [1, {numerical_rank}] (its numerical rank), got {rank}"
        )
    return rank
