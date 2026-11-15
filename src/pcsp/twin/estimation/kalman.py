"""Exact linear Kalman filter and extended Kalman filter.

Both implement :class:`pcsp.core.interfaces.Estimator`. The linear filter is the
reference against which the ensemble filter is verified; the extended filter is
the classical nonlinear baseline.

Measurement update (shared by both filters)
-------------------------------------------
With prior mean ``x⁻``, prior covariance ``P⁻``, predicted output ``ŷ``, output
Jacobian ``C`` and measurement-noise covariance ``R``:

.. math::

    \\nu &= y - \\hat y                       && \\text{innovation} \\\\
    S &= C P^- C^\\top + R                    && \\text{innovation covariance} \\\\
    K &= P^- C^\\top S^{-1}                    && \\text{Kalman gain} \\\\
    x^+ &= x^- + K \\nu \\\\
    P^+ &= (I - K C) P^- (I - K C)^\\top + K R K^\\top  && \\text{Joseph form}

The Joseph form is algebraically equal to ``(I - K C) P⁻`` for the optimal gain
but stays symmetric positive semi-definite under rounding (Bucy & Joseph, 1968).

References
----------
* R. E. Kalman (1960), "A new approach to linear filtering and prediction
  problems", *J. Basic Engineering* 82(1), 35-45.
* A. H. Jazwinski (1970), *Stochastic Processes and Filtering Theory*,
  Academic Press, ch. 7 and 8 (Kalman and extended Kalman filters).
* Y. Bar-Shalom, X. R. Li, T. Kirubarajan (2001), *Estimation with Applications
  to Tracking and Navigation*, Wiley, ch. 5 and 10.
"""

from __future__ import annotations

import numpy as np

from pcsp.core.interfaces import ControlModel
from pcsp.twin.estimation._linearise import discrete_jacobians
from pcsp.twin.estimation._validate import check_cov, check_vec, resolve_ud


class KalmanFilter:
    """Exact Kalman filter for a linear Gaussian discrete-time system.

    .. math::

        x_{k+1} = A x_k + B u_k + G d_k + w_k, \\qquad w_k \\sim N(0, Q) \\\\
        y_k = C x_k + v_k, \\qquad v_k \\sim N(0, R)

    Prediction: ``x⁻ = A x + B u + G d`` and ``P⁻ = A P Aᵀ + Q``; the update is
    given in the module docstring.

    Parameters
    ----------
    A, C, Q, R : system, output, process-noise and measurement-noise matrices
    x0, P0 : initial mean, shape ``(nx,)``, and covariance, shape ``(nx, nx)``
    B, G : input and disturbance matrices; ``None`` means the term is absent
    """

    def __init__(
        self,
        A: np.ndarray,
        C: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
        B: np.ndarray | None = None,
        G: np.ndarray | None = None,
    ) -> None:
        self.A = np.atleast_2d(np.asarray(A, dtype=float))
        self.C = np.atleast_2d(np.asarray(C, dtype=float))
        self.B = None if B is None else np.atleast_2d(np.asarray(B, dtype=float))
        self.G = None if G is None else np.atleast_2d(np.asarray(G, dtype=float))
        nx, ny = self.A.shape[0], self.C.shape[0]
        self.Q = check_cov(Q, nx, "Q")
        self.R = check_cov(R, ny, "R")
        self._x = check_vec(x0, nx, "x0")
        self._P = check_cov(P0, nx, "P0")
        self.innovation: np.ndarray | None = None
        self.innovation_cov: np.ndarray | None = None

    def predict(self, u: np.ndarray, d: np.ndarray) -> None:
        x = self.A @ self._x
        if self.B is not None:
            x = x + self.B @ np.asarray(u, dtype=float)
        if self.G is not None:
            x = x + self.G @ np.asarray(d, dtype=float)
        self._x = x
        self._P = self.A @ self._P @ self.A.T + self.Q

    def update(
        self, y: np.ndarray, u: np.ndarray | None = None, d: np.ndarray | None = None
    ) -> None:
        """Assimilate ``y``. ``u`` and ``d`` are accepted for the
        :class:`~pcsp.core.interfaces.Estimator` protocol and ignored, because
        the linear measurement ``y = C x`` does not depend on them."""
        self._x, self._P, self.innovation, self.innovation_cov = kalman_update(
            self._x, self._P, np.asarray(y, dtype=float), self.C @ self._x, self.C, self.R
        )

    @property
    def x_hat(self) -> np.ndarray:
        return self._x.copy()

    @property
    def P(self) -> np.ndarray:
        return self._P.copy()


class ExtendedKalmanFilter:
    """Extended Kalman filter for a :class:`ControlModel` sampled every ``dt``.

    .. math::

        x_{k+1} = F(x_k, u_k, d_k) + w_k, \\qquad y_k = h(x_k, u_k, d_k) + v_k

    where ``F`` is the RK4 zero-order-hold map of :func:`pcsp.core.integrate.casadi_discretize`.
    Prediction: ``x⁻ = F(x, u, d)`` and ``P⁻ = A P Aᵀ + Q`` with ``A = ∂F/∂x``
    evaluated at the previous estimate; the update linearises ``h`` at ``x⁻``
    (``C = ∂h/∂x``). Both Jacobians are exact (CasADi automatic differentiation).

    The output map needs ``(u, d)``: :meth:`update` uses the pair passed to the
    last :meth:`predict`, or the ``u``, ``d`` keywords when given (for example to
    assimilate a measurement before the first prediction).

    Parameters
    ----------
    model : the continuous-time model
    dt : sample period [s]
    Q, R : process-noise covariance (per sample period) and measurement-noise covariance
    x0, P0 : initial mean and covariance
    substeps : RK4 substeps per sample period
    """

    def __init__(
        self,
        model: ControlModel,
        dt: float,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
        substeps: int = 1,
    ) -> None:
        self.model = model
        self.Q = check_cov(Q, model.nx, "Q")
        self.R = check_cov(R, model.ny, "R")
        self._x = check_vec(x0, model.nx, "x0")
        self._P = check_cov(P0, model.nx, "P0")
        self._transition, self._measurement = discrete_jacobians(model, dt, substeps)
        self._last_ud: tuple[np.ndarray, np.ndarray] | None = None
        self.innovation: np.ndarray | None = None
        self.innovation_cov: np.ndarray | None = None

    def predict(self, u: np.ndarray, d: np.ndarray) -> None:
        u, d = np.asarray(u, dtype=float), np.asarray(d, dtype=float)
        x_next, A = self._transition(self._x, u, d)
        A = A.full()
        self._x = x_next.full().ravel()
        self._P = A @ self._P @ A.T + self.Q
        self._last_ud = (u, d)

    def update(
        self, y: np.ndarray, u: np.ndarray | None = None, d: np.ndarray | None = None
    ) -> None:
        u, d = resolve_ud(u, d, self._last_ud)
        y_pred, C = self._measurement(self._x, u, d)
        self._x, self._P, self.innovation, self.innovation_cov = kalman_update(
            self._x, self._P, np.asarray(y, dtype=float), y_pred.full().ravel(), C.full(), self.R
        )

    @property
    def x_hat(self) -> np.ndarray:
        return self._x.copy()

    @property
    def P(self) -> np.ndarray:
        return self._P.copy()


def kalman_update(
    x: np.ndarray, P: np.ndarray, y: np.ndarray, y_pred: np.ndarray, C: np.ndarray, R: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Kalman measurement update (Joseph form); returns ``(x⁺, P⁺, ν, S)``."""
    nu = y - y_pred
    S = C @ P @ C.T + R
    K = np.linalg.solve(S, C @ P).T  # P Cᵀ S⁻¹, using the symmetry of P and S
    I_KC = np.eye(x.size) - K @ C
    P_new = I_KC @ P @ I_KC.T + K @ R @ K.T
    return x + K @ nu, 0.5 * (P_new + P_new.T), nu, S
