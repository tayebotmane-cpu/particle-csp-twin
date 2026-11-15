"""Exact Kalman filter and EKF: hand-checkable values, Riccati limit, consistency."""

import numpy as np
import pytest
from scipy.linalg import solve_discrete_are

from pcsp.core import Estimator
from pcsp.twin.estimation import (
    ExtendedKalmanFilter,
    KalmanFilter,
    chi2_bounds,
    nees,
    nis,
)

NO_INPUT = np.zeros(0)

# A lightly damped oscillator sampled at 0.1 s, position measured.
A2 = np.array([[1.0, 0.1], [-0.2, 0.95]])
C2 = np.array([[1.0, 0.0]])
Q2 = np.diag([0.01, 0.02])
R2 = np.array([[0.1]])
P02 = np.diag([0.5, 0.5])


def test_filters_satisfy_estimator_protocol(linear_cv):
    kf = KalmanFilter(A2, C2, Q2, R2, np.zeros(2), P02)
    ekf = ExtendedKalmanFilter(linear_cv, 0.1, Q2, R2, np.zeros(2), P02)
    assert isinstance(kf, Estimator) and isinstance(ekf, Estimator)


def test_scalar_random_walk_by_hand():
    # x+ = x + w, y = x + v with Q = R = P0 = 1, x0 = 0, first measurement y = 3.
    # Predict: P⁻ = 2. Gain K = 2/3. Update: x = 2, P = 2/3.
    kf = KalmanFilter([[1.0]], [[1.0]], [[1.0]], [[1.0]], [0.0], [[1.0]])
    kf.predict(NO_INPUT, NO_INPUT)
    assert kf.P[0, 0] == pytest.approx(2.0)
    kf.update(np.array([3.0]))
    assert kf.x_hat[0] == pytest.approx(2.0)
    assert kf.P[0, 0] == pytest.approx(2.0 / 3.0)
    assert kf.innovation[0] == pytest.approx(3.0)
    assert kf.innovation_cov[0, 0] == pytest.approx(3.0)
    # Steady state: P⁻² - P⁻ - 1 = 0, so P⁻ = golden ratio and P⁺ = 1/golden ratio.
    for _ in range(60):
        kf.predict(NO_INPUT, NO_INPUT)
        kf.update(np.array([0.0]))
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    assert kf.P[0, 0] == pytest.approx(1.0 / golden, rel=1e-12)


def test_covariance_converges_to_riccati_solution():
    kf = KalmanFilter(A2, C2, Q2, R2, np.zeros(2), P02)
    for _ in range(500):
        kf.predict(NO_INPUT, NO_INPUT)
        kf.update(np.zeros(1))
    # solve_discrete_are gives the prior (predicted) covariance of the filter.
    P_prior = solve_discrete_are(A2.T, C2.T, Q2, R2)
    kf.predict(NO_INPUT, NO_INPUT)
    np.testing.assert_allclose(kf.P, P_prior, rtol=1e-10)


def test_input_and_disturbance_terms_enter_prediction():
    B, G = np.array([[0.0], [0.1]]), np.array([[0.5], [0.0]])
    kf = KalmanFilter(A2, C2, Q2, R2, np.array([1.0, 2.0]), P02, B=B, G=G)
    kf.predict(np.array([3.0]), np.array([4.0]))
    np.testing.assert_allclose(kf.x_hat, A2 @ [1.0, 2.0] + B @ [3.0] + G @ [4.0])


def test_ekf_on_linear_model_equals_kalman_filter(linear_cv):
    # RK4 applied to ẋ = M x + B u is exactly x+ = Φ x + Γ u with the
    # 4th-order Taylor polynomials below; the EKF must reproduce that KF to rounding.
    dt, M, B = 0.1, linear_cv.M, np.array([[0.0], [1.0]])
    Mh = dt * M
    Phi = np.eye(2) + Mh + Mh @ Mh / 2 + Mh @ Mh @ Mh / 6 + Mh @ Mh @ Mh @ Mh / 24
    Gam = (np.eye(2) + Mh / 2 + Mh @ Mh / 6 + Mh @ Mh @ Mh / 24) @ B * dt
    x0 = np.array([0.3, -0.1])
    kf = KalmanFilter(Phi, C2, Q2, R2, x0, P02, B=Gam)
    ekf = ExtendedKalmanFilter(linear_cv, dt, Q2, R2, x0, P02)
    rng = np.random.default_rng(3)
    for _ in range(50):
        u, y = rng.normal(size=1), rng.normal(size=1)
        for f in (kf, ekf):
            f.predict(u, NO_INPUT)
            f.update(y)
    np.testing.assert_allclose(ekf.x_hat, kf.x_hat, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(ekf.P, kf.P, rtol=1e-11, atol=1e-13)


def test_ekf_update_before_predict_needs_inputs(linear_cv):
    ekf = ExtendedKalmanFilter(linear_cv, 0.1, Q2, R2, np.zeros(2), P02)
    with pytest.raises(RuntimeError, match="predict"):
        ekf.update(np.zeros(1))
    ekf.update(np.zeros(1), u=np.zeros(1), d=NO_INPUT)  # explicit inputs are accepted


def test_kalman_nees_and_nis_are_chi_square_consistent():
    """Monte Carlo consistency test of Bar-Shalom et al. (2001), sec. 5.4."""
    n_runs, n_steps, nx, ny = 100, 60, 2, 1
    rng = np.random.default_rng(42)
    L_Q, L_R, L_P0 = (np.linalg.cholesky(M) for M in (Q2, R2, P02))
    x0_mean = np.array([1.0, 0.0])
    eps = np.empty((n_runs, n_steps))
    eps_nu = np.empty((n_runs, n_steps))
    for r in range(n_runs):
        x = x0_mean + L_P0 @ rng.standard_normal(nx)
        kf = KalmanFilter(A2, C2, Q2, R2, x0_mean, P02)
        for k in range(n_steps):
            x = A2 @ x + L_Q @ rng.standard_normal(nx)
            y = C2 @ x + L_R @ rng.standard_normal(ny)
            kf.predict(NO_INPUT, NO_INPUT)
            kf.update(y)
            eps[r, k] = nees(x - kf.x_hat, kf.P)
            eps_nu[r, k] = nis(kf.innovation, kf.innovation_cov)

    lo, hi = chi2_bounds(nx, n_runs, 0.95)
    nees_mean = eps.mean(axis=0)
    frac_nees = np.mean((nees_mean > lo) & (nees_mean < hi))
    lo_nu, hi_nu = chi2_bounds(ny, n_runs, 0.95)
    nis_mean = eps_nu.mean(axis=0)
    frac_nis = np.mean((nis_mean > lo_nu) & (nis_mean < hi_nu))
    # Expected fraction inside a 95 % interval is 0.95; allow for sampling spread.
    assert frac_nees >= 0.88, f"NEES inside 95% bounds at only {frac_nees:.0%} of steps"
    assert frac_nis >= 0.88, f"NIS inside 95% bounds at only {frac_nis:.0%} of steps"
    # Innovations of the optimal filter are white, so pooling over time is exact for NIS.
    lo_all, hi_all = chi2_bounds(ny, n_runs * n_steps, 0.99)
    assert lo_all < eps_nu.mean() < hi_all
    # Time-averaged NEES must be close to nx (errors are correlated in time, so no exact bound).
    assert eps.mean() == pytest.approx(nx, rel=0.05)
