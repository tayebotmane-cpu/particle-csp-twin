"""Ensemble Kalman filter (stochastic and deterministic square-root analyses).

The EnKF represents the state distribution by ``N`` members ``x_i`` and needs
only forward evaluations of the model, so the same code serves the
few-state physics models of today and the reduced-order models of tomorrow.

Notation (column convention; the code stores members as rows)
-------------------------------------------------------------
Ensemble mean ``x̄ = (1/N) Σ x_i`` and anomaly matrix
``X = [x_1 - x̄, ..., x_N - x̄] ∈ R^{n×N}``, so the sample covariance is
``P = X Xᵀ / (N - 1)``. Predicted outputs ``y_i = h(x_i, u, d)`` with mean ``ȳ``
and anomalies ``Y ∈ R^{m×N}``.

Forecast (both variants)
    ``x_i ← F(x_i, u, d) + w_i`` with ``w_i ~ N(0, Q)``, then multiplicative
    inflation of the anomalies ``x_i ← x̄ + λ (x_i - x̄)`` (covariance × λ²;
    Anderson & Anderson, 1999), then clipping to the state bounds.

Stochastic analysis, perturbed observations (Burgers et al., 1998; Evensen, 2003)
    ``P_xy = X Yᵀ/(N-1)``, ``S = Y Yᵀ/(N-1) + R``, ``K = P_xy S⁻¹`` and
    ``x_i ← x_i + K (y + v_i - y_i)`` with ``v_i ~ N(0, R)``. The perturbations
    are re-centred to zero mean so that, for linear ``h``, the analysis mean is
    exactly ``x̄ + K (y - ȳ)`` and only the spread carries sampling noise.

Deterministic square-root analysis, ETKF (Bishop et al., 2001; Hunt et al., 2007)
    Work in the N-dimensional ensemble space:

    .. math::

        \\tilde P^a = \\left[(N-1) I + Y^\\top R^{-1} Y\\right]^{-1}, \\quad
        \\bar w = \\tilde P^a Y^\\top R^{-1} (y - \\bar y), \\quad
        W = \\left[(N-1) \\tilde P^a\\right]^{1/2}

    with the symmetric square root, and ``x_i ← x̄ + X (w̄ + W_{:,i})``. For
    linear ``h`` this reproduces the Kalman mean ``x̄ + K (y - ȳ)`` and the
    covariance ``(I - K H) P`` of the forecast ensemble exactly, with no random
    observation perturbations; the symmetric root keeps the mean unchanged
    (Ott et al., 2004; Wang et al., 2004).

After either analysis the members are clipped to the state bounds. Clipping is
a projection, not part of Kalman theory: it keeps temperatures and mass flows
physical at the cost of a small bias when many members sit on a bound.

References
----------
* G. Evensen (1994), *J. Geophys. Res.* 99(C5), 10143-10162 (the EnKF).
* G. Burgers, P. J. van Leeuwen, G. Evensen (1998), "Analysis scheme in the
  ensemble Kalman filter", *Mon. Weather Rev.* 126, 1719-1724.
* G. Evensen (2003), "The ensemble Kalman filter: theoretical formulation and
  practical implementation", *Ocean Dynamics* 53, 343-367.
* C. H. Bishop, B. J. Etherton, S. J. Majumdar (2001), "Adaptive sampling with
  the ensemble transform Kalman filter", *Mon. Weather Rev.* 129, 420-436.
* B. R. Hunt, E. J. Kostelich, I. Szunyogh (2007), "Efficient data assimilation
  for spatiotemporal chaos: a local ensemble transform Kalman filter",
  *Physica D* 230, 112-126 (eqs. 20-25 give the ETKF used here).
* J. L. Anderson, S. L. Anderson (1999), "A Monte Carlo implementation of the
  nonlinear filtering problem to produce ensemble assimilations and
  forecasts", *Mon. Weather Rev.* 127, 2741-2758 (multiplicative inflation).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

from pcsp.twin.estimation._validate import check_cov, check_vec, resolve_ud

Transition = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
Measurement = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


class EnsembleKalmanFilter:
    """Ensemble Kalman filter implementing :class:`pcsp.core.interfaces.Estimator`.

    Parameters
    ----------
    transition : one-sample map ``F(x, u, d) -> x_next``, e.g.
        ``pcsp.core.integrate.discretize(model, dt)``
    measurement : output map ``h(x, u, d) -> y``, e.g. ``model.output``
    Q : process-noise covariance per sample period, shape ``(nx, nx)``; may be
        singular (zero rows for noise-free states)
    R : measurement-noise covariance, shape ``(ny, ny)``, positive definite
    x0, P0 : mean and covariance of the initial ensemble
    n_members : ensemble size ``N``
    method : ``"stochastic"`` (perturbed observations) or ``"sqrt"`` (ETKF)
    inflation : multiplicative anomaly inflation ``λ >= 1`` applied after each
        forecast; the covariance grows by ``λ²``
    lower, upper : optional state bounds, shape ``(nx,)``; use ``±np.inf`` for
        unbounded components
    seed : seed of the filter's own random generator (initial ensemble,
        process noise and observation perturbations)

    Notes
    -----
    The output map needs ``(u, d)``: :meth:`update` uses the pair passed to the
    last :meth:`predict`, or the ``u``, ``d`` keywords when given.
    """

    def __init__(
        self,
        transition: Transition,
        measurement: Measurement,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
        n_members: int,
        method: Literal["stochastic", "sqrt"] = "stochastic",
        inflation: float = 1.0,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        if method not in ("stochastic", "sqrt"):
            raise ValueError(f"method must be 'stochastic' or 'sqrt', got {method!r}")
        if n_members < 2:
            raise ValueError(f"n_members must be >= 2, got {n_members}")
        if inflation < 1.0:
            raise ValueError(f"inflation must be >= 1, got {inflation}")
        x0 = np.asarray(x0, dtype=float).ravel()
        nx = x0.size
        self.transition = transition
        self.measurement = measurement
        self.Q = check_cov(Q, nx, "Q")
        self.R = np.atleast_2d(np.asarray(R, dtype=float))
        self.R = check_cov(self.R, self.R.shape[0], "R")
        self.method = method
        self.inflation = float(inflation)
        self.lower = np.full(nx, -np.inf) if lower is None else check_vec(lower, nx, "lower")
        self.upper = np.full(nx, np.inf) if upper is None else check_vec(upper, nx, "upper")
        if np.any(self.lower > self.upper):
            raise ValueError("lower bounds must not exceed upper bounds")
        self._rng = np.random.default_rng(seed)
        self._sqrt_Q = _psd_sqrt(self.Q)
        self._sqrt_R = np.linalg.cholesky(self.R)
        P0 = check_cov(P0, nx, "P0")
        # Members are the rows of _E, shape (N, nx).
        z = self._rng.standard_normal((n_members, nx))
        self._E = self._clip(x0 + z @ _psd_sqrt(P0).T)
        self._last_ud: tuple[np.ndarray, np.ndarray] | None = None
        self.innovation: np.ndarray | None = None
        self.innovation_cov: np.ndarray | None = None

    # ------------------------------------------------------------------ cycle
    def predict(self, u: np.ndarray, d: np.ndarray) -> None:
        u, d = np.asarray(u, dtype=float), np.asarray(d, dtype=float)
        E = np.array([self.transition(x, u, d) for x in self._E], dtype=float)
        E += self._rng.standard_normal(E.shape) @ self._sqrt_Q.T
        if self.inflation != 1.0:
            mean = E.mean(axis=0)
            E = mean + self.inflation * (E - mean)
        self._E = self._clip(E)
        self._last_ud = (u, d)

    def update(
        self, y: np.ndarray, u: np.ndarray | None = None, d: np.ndarray | None = None
    ) -> None:
        u, d = resolve_ud(u, d, self._last_ud)
        y = np.asarray(y, dtype=float).ravel()
        N = self.n_members
        Yf = np.array([self.measurement(x, u, d) for x in self._E], dtype=float).reshape(N, -1)
        x_mean, y_mean = self._E.mean(axis=0), Yf.mean(axis=0)
        X = (self._E - x_mean).T  # (nx, N) anomalies
        Y = (Yf - y_mean).T  # (ny, N) anomalies
        self.innovation = y - y_mean
        self.innovation_cov = Y @ Y.T / (N - 1) + self.R
        if self.method == "stochastic":
            E = self._stochastic_analysis(y, Yf, X, Y)
        else:
            E = self._sqrt_analysis(x_mean, y_mean, y, X, Y)
        self._E = self._clip(E)

    def _stochastic_analysis(
        self, y: np.ndarray, Yf: np.ndarray, X: np.ndarray, Y: np.ndarray
    ) -> np.ndarray:
        N = self.n_members
        v = self._rng.standard_normal((N, y.size)) @ self._sqrt_R.T
        v -= v.mean(axis=0)
        P_xy = X @ Y.T / (N - 1)
        K = np.linalg.solve(self.innovation_cov, P_xy.T).T  # P_xy S⁻¹ (S symmetric)
        return self._E + (y + v - Yf) @ K.T

    def _sqrt_analysis(
        self, x_mean: np.ndarray, y_mean: np.ndarray, y: np.ndarray, X: np.ndarray, Y: np.ndarray
    ) -> np.ndarray:
        # With R = L Lᵀ and the thin SVD  S = L⁻¹ Y / sqrt(N-1) = U Σ Vᵀ:
        #   (N-1) P̃ᵃ = (I + SᵀS)⁻¹ = I - V diag(σ²/(1+σ²)) Vᵀ
        #   W = (I + SᵀS)^{-1/2}  = I + V diag((1+σ²)^{-1/2} - 1) Vᵀ   (symmetric root)
        #   w̄ = (I + SᵀS)⁻¹ Sᵀ L⁻¹ (y - ȳ) / sqrt(N-1) = V diag(σ/(1+σ²)) Uᵀ L⁻¹ (y - ȳ) / sqrt(N-1)
        # This equals the N×N formula in the module docstring but costs O(N ny²), not O(N³).
        N = self.n_members
        S = np.linalg.solve(self._sqrt_R, Y) / np.sqrt(N - 1)
        U, sig, Vt = np.linalg.svd(S, full_matrices=False)
        r = np.linalg.solve(self._sqrt_R, y - y_mean)
        w_mean = Vt.T @ (sig / (1.0 + sig**2) * (U.T @ r)) / np.sqrt(N - 1)
        XV = X @ Vt.T  # (nx, r)
        X_a = X + (XV * (1.0 / np.sqrt(1.0 + sig**2) - 1.0)) @ Vt  # X W
        return (x_mean + X @ w_mean)[None, :] + X_a.T

    # -------------------------------------------------------------- accessors
    @property
    def x_hat(self) -> np.ndarray:
        """Ensemble mean, shape ``(nx,)``."""
        return self._E.mean(axis=0)

    @property
    def P(self) -> np.ndarray:
        """Sample covariance of the ensemble (``N - 1`` normalisation)."""
        A = self._E - self._E.mean(axis=0)
        return A.T @ A / (self.n_members - 1)

    @property
    def ensemble(self) -> np.ndarray:
        """Copy of the members, shape ``(n_members, nx)``."""
        return self._E.copy()

    @property
    def n_members(self) -> int:
        return self._E.shape[0]

    def _clip(self, E: np.ndarray) -> np.ndarray:
        return np.clip(E, self.lower, self.upper)


def _psd_sqrt(M: np.ndarray) -> np.ndarray:
    """A factor ``L`` with ``L Lᵀ = M`` for symmetric positive semi-definite ``M``.

    Uses the eigendecomposition so that singular matrices (e.g. no process
    noise on some states) are accepted; tiny negative eigenvalues from rounding
    are set to zero.
    """
    lam, V = np.linalg.eigh(M)
    if lam.min(initial=0.0) < -1e-10 * np.abs(lam).max(initial=0.0):
        raise ValueError("covariance matrix is not positive semi-definite")
    return V * np.sqrt(np.clip(lam, 0.0, None))
