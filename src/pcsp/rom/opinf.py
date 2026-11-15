"""Operator inference (OpInf): learn polynomial reduced models from data.

Snapshot convention: a snapshot matrix has shape ``(n_dof, n_snapshots)``.
Here the "snapshots" are reduced states: ``Q`` has shape ``(r, K)`` (e.g. POD
coefficients ``Q = pod.project(X)``), ``dQ`` its time derivative, ``U`` the
inputs ``(m, K)``.

Model
-----
    dq/dt = c + A q + H (q (x) q) + B u,

with ``c in R^r``, ``A in R^{r x r}``, ``B in R^{r x m}`` and the quadratic term in
compact form ``H (q (x)^ q)``, ``H in R^{r x r(r+1)/2}``. The full Kronecker
product ``q (x) q`` repeats every product ``q_i q_j`` (``i != j``) twice, which
makes the full ``H`` non-unique; the compact product keeps each product once,

    q (x)^ q = [q_1 q_1,  q_2 q_1, q_2 q_2,  q_3 q_1, q_3 q_2, q_3 q_3, ...]

(row ``i`` contributes ``q_i q_j`` for ``j <= i``). Any subset of the terms
``"c"``, ``"A"``, ``"H"``, ``"B"`` may be chosen.

Learning problem (Peherstorfer & Willcox 2016, sec. 3; Tikhonov
regularisation as in McQuarrie, Huang & Willcox 2021)
------------------------------------------------------------------
Stack the data matrix ``D = [1, Q^T, (Q (x)^ Q)^T, U^T]`` (``K x d``, only the
chosen blocks) and the operator matrix ``O = [c, A, H, B]`` (``r x d``). Solve

    min_O  || D O^T - dQ^T ||_F^2 + || Gamma O^T ||_F^2,

with ``Gamma = diag(lambda_1, ..., lambda_1, lambda_2, ..., lambda_2)``: ``lambda_2``
on the quadratic columns and ``lambda_1`` on all others. The minimiser is computed
as the ordinary least-squares solution of the augmented system
``[D; Gamma] O^T = [dQ^T; 0]`` with an SVD-based solver, which avoids squaring the
condition number of ``D`` in the normal equations. With ``lambda = 0`` and clean
data from a system that really has this form the operators are recovered exactly
whenever ``D`` has full column rank.

References
----------
Peherstorfer & Willcox (2016), "Data-driven operator inference for nonintrusive
projection-based model reduction", Comput. Methods Appl. Mech. Eng. 306, 196-215.
McQuarrie, Huang & Willcox (2021), "Data-driven reduced-order models via
regularised operator inference for a single-injector combustion process",
J. R. Soc. N. Z. 51(2), 194-211.
Fornberg (1988), "Generation of finite difference formulas on arbitrarily spaced
grids", Math. Comput. 51(184), 699-706.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

_TERMS = "cAHB"


def compact_kron(q: ArrayLike) -> np.ndarray:
    """Compact Kronecker product ``q (x)^ q`` (length ``r(r+1)/2``) of ``(r,)`` or ``(r, K)``."""
    q = np.asarray(q, dtype=float)
    return np.concatenate([q[i] * q[: i + 1] for i in range(q.shape[0])], axis=0)


def expand_quadratic(H_compact: ArrayLike) -> np.ndarray:
    """Full symmetric ``H`` of shape ``(r, r^2)`` with ``H (q (x) q) = H_c (q (x)^ q)``.

    Each off-diagonal coefficient is split equally between columns ``(i, j)`` and
    ``(j, i)`` of the full Kronecker ordering (``(q (x) q)[i r + j] = q_i q_j``).
    """
    Hc = np.asarray(H_compact, dtype=float)
    r = _quadratic_dim(Hc.shape[1])
    H = np.zeros((Hc.shape[0], r * r))
    col = 0
    for i in range(r):
        for j in range(i + 1):
            if i == j:
                H[:, i * r + i] = Hc[:, col]
            else:
                H[:, i * r + j] = H[:, j * r + i] = Hc[:, col] / 2.0
            col += 1
    return H


def compress_quadratic(H: ArrayLike) -> np.ndarray:
    """Compact ``H_c`` of shape ``(r, r(r+1)/2)`` from a full ``(r, r^2)`` quadratic operator."""
    H = np.asarray(H, dtype=float)
    r = int(round(np.sqrt(H.shape[1])))
    if r * r != H.shape[1]:
        raise ValueError(f"full quadratic operator must have r^2 columns, got {H.shape[1]}")
    columns = []
    for i in range(r):
        for j in range(i + 1):
            columns.append(H[:, i * r + i] if i == j else H[:, i * r + j] + H[:, j * r + i])
    return np.stack(columns, axis=1)


def time_derivative(Q: ArrayLike, t: ArrayLike, *, order: int = 4) -> np.ndarray:
    """Finite-difference ``dQ/dt`` of ``(r, K)`` states sampled at times ``t`` (``(K,)``).

    Each derivative uses ``order + 1`` consecutive samples: a centred stencil in the
    interior (for even ``order``) and one-sided stencils of the same width near the
    ends, so the truncation error is ``O(h^order)`` everywhere. The weights solve the
    Taylor (Vandermonde) conditions ``sum_j w_j (t_j - t_i)^k / k! = delta_{k1}``,
    ``k = 0..order`` (Fornberg 1988), so non-uniform grids are allowed. High orders
    amplify noise; for noisy data prefer ``order=2`` or smooth first.
    """
    Q = np.asarray(Q, dtype=float)
    t = np.asarray(t, dtype=float)
    K = t.size
    if Q.ndim != 2 or Q.shape[1] != K:
        raise ValueError(f"Q must have shape (r, {K}), got {Q.shape}")
    if order < 1 or order + 1 > K:
        raise ValueError(f"order must be >= 1 and needs at least order + 1 = {order + 1} samples")
    if not np.all(np.diff(t) > 0):
        raise ValueError("t must be strictly increasing")

    n_pts = order + 1
    start = np.clip(np.arange(K) - order // 2, 0, K - n_pts)
    idx = start[:, None] + np.arange(n_pts)[None, :]  # (K, n_pts) stencil indices
    h = np.median(np.diff(t))
    s = (t[idx] - t[:, None]) / h  # scaled offsets keep the Vandermonde system O(1)
    V = s[:, None, :] ** np.arange(n_pts)[None, :, None]  # V[i, k, j] = s_j^k
    rhs = np.zeros((K, n_pts))
    rhs[:, 1] = 1.0  # first derivative: k! delta_{k1} = delta_{k1}
    weights = np.linalg.solve(V, rhs[:, :, None])[:, :, 0] / h  # (K, n_pts)
    return np.einsum("ij,rij->ri", weights, Q[:, idx])


@dataclass(frozen=True)
class OpInfModel:
    """A learned model ``dq/dt = c + A q + H (q (x)^ q) + B u``. Build it with :func:`fit_opinf`.

    Absent terms are ``None``. ``H`` is in compact form, shape ``(r, r(r+1)/2)``;
    see :func:`expand_quadratic` for the full ``(r, r^2)`` operator.
    """

    c: np.ndarray | None
    A: np.ndarray | None
    H: np.ndarray | None
    B: np.ndarray | None
    rank: int

    def rhs(self, q: ArrayLike, u: ArrayLike | None = None) -> np.ndarray:
        """``dq/dt`` for a state ``(r,)`` or states ``(r, K)``; ``u`` is ``(m,)`` or ``(m, K)``."""
        q = np.asarray(q, dtype=float)
        out = np.zeros_like(q)
        if self.c is not None:
            out += self.c if q.ndim == 1 else self.c[:, None]
        if self.A is not None:
            out += self.A @ q
        if self.H is not None:
            out += self.H @ compact_kron(q)
        if self.B is not None:
            if u is None:
                raise ValueError("this model has an input term B; pass u")
            out += self.B @ np.asarray(u, dtype=float)
        return out

    def predict(
        self,
        q0: ArrayLike,
        t: ArrayLike,
        u: ArrayLike | Callable[[float], ArrayLike] | None = None,
        *,
        method: str = "RK45",
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ) -> np.ndarray:
        """Integrate the learned model from ``q0`` (``(r,)``); returns states ``(r, len(t))``.

        ``u`` is a callable ``u(t) -> (m,)`` or an ``(m, len(t))`` array sampled at
        ``t`` (linearly interpolated between samples, which adds an ``O(h^2)`` input
        error; pass a callable when the input is known analytically). Integration uses
        :func:`scipy.integrate.solve_ivp`; a ``RuntimeError`` is raised if it fails
        (learned models can be unstable, which is itself a result worth reporting).
        """
        t = np.asarray(t, dtype=float)
        u_of_t = _input_function(u, t, self.B)
        sol = solve_ivp(
            lambda time, q: self.rhs(q, None if u_of_t is None else u_of_t(time)),
            (t[0], t[-1]),
            np.asarray(q0, dtype=float),
            method=method,
            t_eval=t,
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"integration of the learned model failed: {sol.message}")
        return sol.y


def fit_opinf(
    Q: ArrayLike,
    dQ: ArrayLike,
    U: ArrayLike | None = None,
    *,
    terms: str = "AH",
    reg: float = 0.0,
    reg_quadratic: float | None = None,
) -> OpInfModel:
    """Learn reduced operators by regularised least squares (see the module docstring).

    Parameters
    ----------
    Q : ``(r, K)`` reduced states (snapshot matrix of reduced coordinates).
    dQ : ``(r, K)`` their time derivatives, e.g. from :func:`time_derivative`.
    U : ``(m, K)`` inputs, required if ``"B"`` is in ``terms``.
    terms : which operators to learn, any combination of ``"c"``, ``"A"``, ``"H"``, ``"B"``.
    reg : Tikhonov weight ``lambda_1`` on ``c``, ``A`` and ``B``.
    reg_quadratic : Tikhonov weight ``lambda_2`` on ``H`` (default ``reg``).
    """
    Q = np.asarray(Q, dtype=float)
    dQ = np.asarray(dQ, dtype=float)
    if Q.ndim != 2 or dQ.shape != Q.shape:
        raise ValueError(f"Q and dQ must both be (r, K), got {Q.shape} and {dQ.shape}")
    if not terms or any(term not in _TERMS for term in terms) or len(set(terms)) != len(terms):
        raise ValueError(f"terms must be a non-empty combination of {_TERMS!r}, got {terms!r}")
    if ("B" in terms) != (U is not None):
        raise ValueError("pass inputs U if and only if 'B' is in terms")
    if reg < 0 or (reg_quadratic is not None and reg_quadratic < 0):
        raise ValueError("regularisation weights must be non-negative")
    lam2 = reg if reg_quadratic is None else reg_quadratic
    r, K = Q.shape

    blocks: dict[str, np.ndarray] = {}
    if "c" in terms:
        blocks["c"] = np.ones((K, 1))
    if "A" in terms:
        blocks["A"] = Q.T
    if "H" in terms:
        blocks["H"] = compact_kron(Q).T
    if "B" in terms:
        U = np.asarray(U, dtype=float)
        if U.ndim != 2 or U.shape[1] != K:
            raise ValueError(f"U must have shape (m, {K}), got {U.shape}")
        blocks["B"] = U.T
    order = [term for term in _TERMS if term in blocks]  # canonical column order
    D = np.hstack([blocks[term] for term in order])
    gamma = np.concatenate(
        [np.full(blocks[term].shape[1], lam2 if term == "H" else reg) for term in order]
    )
    if D.shape[0] < D.shape[1] and np.all(gamma == 0):
        raise ValueError(f"{K} samples cannot determine {D.shape[1]} unknowns per row")

    lhs = np.vstack([D, np.diag(gamma)])
    rhs = np.vstack([dQ.T, np.zeros((D.shape[1], r))])
    O = np.linalg.lstsq(lhs, rhs, rcond=None)[0].T  # (r, d)

    operators: dict[str, np.ndarray | None] = dict.fromkeys(_TERMS)
    col = 0
    for term in order:
        width = blocks[term].shape[1]
        block = O[:, col : col + width]
        operators[term] = block[:, 0] if term == "c" else block
        col += width
    return OpInfModel(
        c=operators["c"], A=operators["A"], H=operators["H"], B=operators["B"], rank=r
    )


def _quadratic_dim(n_columns: int) -> int:
    """``r`` such that ``r (r + 1) / 2 = n_columns``."""
    r = int(round((np.sqrt(8 * n_columns + 1) - 1) / 2))
    if r * (r + 1) // 2 != n_columns:
        raise ValueError(f"{n_columns} is not a valid compact quadratic width r(r+1)/2")
    return r


def _input_function(
    u: ArrayLike | Callable[[float], ArrayLike] | None,
    t: np.ndarray,
    B: np.ndarray | None,
) -> Callable[[float], np.ndarray] | None:
    if B is None:
        if u is not None:
            raise ValueError("this model has no input term; do not pass u")
        return None
    if u is None:
        raise ValueError("this model has an input term B; pass u")
    if callable(u):
        return lambda time: np.atleast_1d(np.asarray(u(time), dtype=float))
    samples = np.asarray(u, dtype=float)
    if samples.shape != (B.shape[1], t.size):
        raise ValueError(f"u must have shape ({B.shape[1]}, {t.size}), got {samples.shape}")
    return lambda time: np.array([np.interp(time, t, row) for row in samples])
