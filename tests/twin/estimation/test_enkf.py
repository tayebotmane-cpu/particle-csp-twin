"""Ensemble Kalman filter: agreement with the exact KF, nonlinear tracking, inflation, bounds."""

import time

import numpy as np
import pytest

from pcsp.core import Estimator, discretize
from pcsp.twin.estimation import (
    AugmentedModel,
    EnsembleKalmanFilter,
    ExtendedKalmanFilter,
    KalmanFilter,
    chi2_bounds,
    nees,
    rmse,
)

NO_INPUT = np.zeros(0)
METHODS = ("stochastic", "sqrt")

A2 = np.array([[1.0, 0.1], [-0.2, 0.95]])
C2 = np.array([[1.0, 0.0]])
Q2 = np.diag([0.01, 0.02])
R2 = np.array([[0.1]])
X02 = np.array([1.0, 0.0])
P02 = np.diag([0.5, 0.5])


def linear_F(x, u, d):
    return A2 @ x


def linear_h(x, u, d):
    return C2 @ x


def linear_measurements(n_steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = X02 + np.linalg.cholesky(P02) @ rng.standard_normal(2)
    ys = []
    for _ in range(n_steps):
        x = A2 @ x + np.linalg.cholesky(Q2) @ rng.standard_normal(2)
        ys.append(C2 @ x + np.sqrt(R2[0, 0]) * rng.standard_normal(1))
    return np.array(ys)


def test_enkf_satisfies_estimator_protocol():
    f = EnsembleKalmanFilter(linear_F, linear_h, Q2, R2, X02, P02, n_members=10, seed=0)
    assert isinstance(f, Estimator)
    assert f.ensemble.shape == (10, 2) and f.P.shape == (2, 2)


@pytest.mark.parametrize("method", METHODS)
def test_single_analysis_matches_kalman_update_of_the_ensemble(method):
    """For linear h the analysis mean is x̄ + K(y - Hx̄) with the ensemble gain K.

    The square-root (ETKF) analysis also reproduces the covariance (I - KH)P
    exactly; the stochastic one only in expectation.
    """
    f = EnsembleKalmanFilter(
        linear_F, linear_h, Q2, R2, X02, P02, n_members=40, method=method, seed=1
    )
    f.predict(NO_INPUT, NO_INPUT)
    x_f, P_f, y = f.x_hat, f.P, np.array([1.7])
    K = P_f @ C2.T @ np.linalg.inv(C2 @ P_f @ C2.T + R2)
    f.update(y)
    np.testing.assert_allclose(f.x_hat, x_f + K @ (y - C2 @ x_f), rtol=1e-12)
    if method == "sqrt":
        np.testing.assert_allclose(f.P, (np.eye(2) - K @ C2) @ P_f, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("method", METHODS)
def test_enkf_converges_to_kalman_filter_as_ensemble_grows(method):
    """Monte Carlo error of the EnKF moments decays like N^{-1/2}.

    Measured (4 seeds, 40 steps; mean error in KF standard deviations / relative
    Frobenius error of P):

    ====== =============== ===============
    N      stochastic      sqrt
    ====== =============== ===============
    25     0.186 / 0.289   0.192 / 0.208
    100    0.108 / 0.157   0.090 / 0.098
    400    0.052 / 0.065   0.039 / 0.051
    1600   0.030 / 0.033   0.024 / 0.031
    ====== =============== ===============

    A 64-fold ensemble gives 6-9 times smaller errors (theory: 8).
    """
    ys = linear_measurements(40, seed=1)
    kf = KalmanFilter(A2, C2, Q2, R2, X02, P02)
    kf_x, kf_P = [], []
    for y in ys:
        kf.predict(NO_INPUT, NO_INPUT)
        kf.update(y)
        kf_x.append(kf.x_hat)
        kf_P.append(kf.P)

    sizes = (25, 100, 400, 1600)
    mean_err, cov_err = [], []
    for N in sizes:
        e_x, e_P = [], []
        for seed in range(4):
            f = EnsembleKalmanFilter(
                linear_F, linear_h, Q2, R2, X02, P02, N, method=method, seed=seed
            )
            for k, y in enumerate(ys):
                f.predict(NO_INPUT, NO_INPUT)
                f.update(y)
                e_x.append(np.sqrt(np.mean((f.x_hat - kf_x[k]) ** 2 / np.diag(kf_P[k]))))
                e_P.append(np.linalg.norm(f.P - kf_P[k]) / np.linalg.norm(kf_P[k]))
        mean_err.append(np.mean(e_x))
        cov_err.append(np.mean(e_P))

    assert np.all(np.diff(mean_err) < 0), mean_err
    assert np.all(np.diff(cov_err) < 0), cov_err
    assert mean_err[0] / mean_err[-1] > 5.0, mean_err
    assert cov_err[0] / cov_err[-1] > 5.0, cov_err
    assert mean_err[-1] < 0.04 and cov_err[-1] < 0.04


def test_enkf_nees_tends_to_state_dimension():
    """NEES with the *sample* covariance is biased high for small N.

    Even for an unbiased Gaussian ensemble, E[P̂⁻¹] = (N-1)/(N-n-2) P⁻¹
    (inverse-Wishart mean), so E[NEES] ≈ n (N-1)/(N-n-2): 2.38 at N=20 and
    2.03 at N=200 for n=2. Measured (50 runs x 40 steps, ETKF): 2.85 and 2.16.
    The excess over the Wishart value at N=20 is genuine spread deficit, the
    reason inflation exists.
    """
    n_runs, n_steps = 50, 40
    mean_nees = {}
    for N in (20, 200):
        rng = np.random.default_rng(7)
        eps = np.empty((n_runs, n_steps))
        for r in range(n_runs):
            x = X02 + np.linalg.cholesky(P02) @ rng.standard_normal(2)
            f = EnsembleKalmanFilter(
                linear_F, linear_h, Q2, R2, X02, P02, N, method="sqrt", seed=100 + r
            )
            for k in range(n_steps):
                x = A2 @ x + np.linalg.cholesky(Q2) @ rng.standard_normal(2)
                f.predict(NO_INPUT, NO_INPUT)
                f.update(C2 @ x + np.sqrt(R2[0, 0]) * rng.standard_normal(1))
                eps[r, k] = nees(x - f.x_hat, f.P)
        mean_nees[N] = eps.mean()
    assert mean_nees[200] == pytest.approx(2.0 * 199 / 196, rel=0.08)
    lo, hi = chi2_bounds(2, n_runs * n_steps, 0.95)
    assert mean_nees[20] > hi  # small ensembles are over-confident, as theory predicts
    assert mean_nees[20] > mean_nees[200]


# ----------------------------------------------------------------- nonlinear
LORENZ_DT, LORENZ_SUBSTEPS = 0.05, 5
LORENZ_SIGMA = np.sqrt(2.0)  # measurement noise std of every component


def lorenz_truth(model, n_steps: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    F = discretize(model, LORENZ_DT, LORENZ_SUBSTEPS)
    rng = np.random.default_rng(seed)
    x = np.ones(3)
    for _ in range(400):  # spin up onto the attractor
        x = F(x, NO_INPUT, NO_INPUT)
    X, Y = [], []
    for _ in range(n_steps):
        x = F(x, NO_INPUT, NO_INPUT)
        X.append(x)
        Y.append(x + LORENZ_SIGMA * rng.standard_normal(3))
    return np.array(X), np.array(Y)


def run_filter(f, Y: np.ndarray) -> np.ndarray:
    est = []
    for y in Y:
        f.predict(NO_INPUT, NO_INPUT)
        f.update(y)
        est.append(f.x_hat)
    return np.array(est)


LORENZ_X0_ERROR = np.array([2.0, -2.0, 3.0])
LORENZ_P0 = 9.0 * np.eye(3)
LORENZ_R = LORENZ_SIGMA**2 * np.eye(3)


@pytest.mark.parametrize("method", METHODS)
def test_enkf_tracks_lorenz63_below_measurement_noise(lorenz, method):
    """Perfect-model Lorenz-63, all states observed every 0.05 time units with σ = √2.

    Measured RMSE (N = 20, λ = 1.02, seeds 0-2): stochastic 0.25-0.33,
    sqrt 0.26-0.38, versus 1.36-1.47 for the raw measurements.
    """
    n_steps = 400
    X, Y = lorenz_truth(lorenz, n_steps, seed=0)
    F = discretize(lorenz, LORENZ_DT, LORENZ_SUBSTEPS)
    f = EnsembleKalmanFilter(
        F,
        lorenz.output,
        np.zeros((3, 3)),
        LORENZ_R,
        X[0] + LORENZ_X0_ERROR,
        LORENZ_P0,
        n_members=20,
        method=method,
        inflation=1.02,
        seed=0,
    )
    err = rmse(run_filter(f, Y)[n_steps // 2 :], X[n_steps // 2 :], axis=None)
    assert err < 0.3 * LORENZ_SIGMA, f"RMSE {err:.3f} vs noise {LORENZ_SIGMA:.3f}"


def test_ekf_tracks_lorenz63_below_measurement_noise(lorenz):
    n_steps = 400
    X, Y = lorenz_truth(lorenz, n_steps, seed=0)
    ekf = ExtendedKalmanFilter(
        lorenz,
        LORENZ_DT,
        0.01 * np.eye(3),  # small model-error covariance keeps the EKF from over-confidence
        LORENZ_R,
        X[0] + LORENZ_X0_ERROR,
        LORENZ_P0,
        substeps=LORENZ_SUBSTEPS,
    )
    err = rmse(run_filter(ekf, Y)[n_steps // 2 :], X[n_steps // 2 :], axis=None)
    assert err < 0.3 * LORENZ_SIGMA, f"RMSE {err:.3f} vs noise {LORENZ_SIGMA:.3f}"


def test_small_ensemble_diverges_without_inflation_and_recovers_with_it(lorenz):
    """Stochastic EnKF, N = 5, perfect model (Q = 0).

    Measured RMSE over the second half of 400 steps (seeds 0, 1, 2):
    no inflation 12.0, 2.4, 4.1 (worse than the raw measurements, σ = 1.41);
    λ = 1.05: 0.25, 0.46, 0.45.
    The ETKF does not diverge in this setting (RMSE 0.19-0.35 without inflation):
    it adds no observation-perturbation sampling noise.
    """
    n_steps = 400
    F = discretize(lorenz, LORENZ_DT, LORENZ_SUBSTEPS)
    for seed in range(3):
        X, Y = lorenz_truth(lorenz, n_steps, seed=seed)
        errors = {}
        for inflation in (1.0, 1.05):
            f = EnsembleKalmanFilter(
                F,
                lorenz.output,
                np.zeros((3, 3)),
                LORENZ_R,
                X[0] + LORENZ_X0_ERROR,
                LORENZ_P0,
                n_members=5,
                inflation=inflation,
                seed=seed,
            )
            errors[inflation] = rmse(run_filter(f, Y)[n_steps // 2 :], X[n_steps // 2 :], None)
        assert errors[1.0] > LORENZ_SIGMA, (seed, errors)
        assert errors[1.05] < 0.5 * LORENZ_SIGMA, (seed, errors)


def test_inflation_scales_forecast_covariance():
    lam = 1.3
    plain = EnsembleKalmanFilter(linear_F, linear_h, Q2, R2, X02, P02, 50, seed=5)
    inflated = EnsembleKalmanFilter(linear_F, linear_h, Q2, R2, X02, P02, 50, inflation=lam, seed=5)
    plain.predict(NO_INPUT, NO_INPUT)
    inflated.predict(NO_INPUT, NO_INPUT)
    np.testing.assert_allclose(inflated.P, lam**2 * plain.P, rtol=1e-12)
    np.testing.assert_allclose(inflated.x_hat, plain.x_hat, rtol=1e-12)


# -------------------------------------------------------------------- bounds
@pytest.mark.parametrize("method", METHODS)
def test_members_never_leave_bounds(receiver, method):
    """Solar input actually zero (cloud); a wide prior pushes members below Q_sun = 0."""
    aug = AugmentedModel(receiver, disturbances=("Q_sun",))
    dt = 5.0
    F = discretize(aug, dt)
    lower = np.array([300.0, 300.0, 0.0])
    upper = np.array([2000.0, 2000.0, 2.0e6])
    z0 = np.array([900.0, 880.0, 5.0e4])
    P0 = np.diag([50.0**2, 50.0**2, (2.0e5) ** 2])
    Q = np.diag([0.1**2, 0.1**2, (5.0e3) ** 2])
    R = np.array([[1.0]])
    u = np.array([1.0])
    d_plant = np.array([0.0, 300.0, 0.0])  # Q_sun, T_amb, Q_aux
    d = aug.measured_disturbances(d_plant)
    bounded = EnsembleKalmanFilter(
        F, aug.output, Q, R, z0, P0, 40, method=method, lower=lower, upper=upper, seed=2
    )
    free = EnsembleKalmanFilter(F, aug.output, Q, R, z0, P0, 40, method=method, seed=2)

    rng = np.random.default_rng(2)
    x = np.array([900.0, 880.0])
    plant_F = discretize(receiver, dt)
    free_went_negative = False
    for _ in range(60):
        x = plant_F(x, u, d_plant)
        y = x[1:] + rng.standard_normal(1)
        for f in (bounded, free):
            f.predict(u, d)
            if f is bounded:
                assert np.all(f.ensemble >= lower) and np.all(f.ensemble <= upper)
            f.update(y)
        assert np.all(bounded.ensemble >= lower) and np.all(bounded.ensemble <= upper)
        free_went_negative |= bool(np.any(free.ensemble[:, 2] < 0.0))
    assert free_went_negative, "test is vacuous: unbounded members never crossed a bound"


# -------------------------------------------------------------------- timing
@pytest.mark.parametrize("n_members", [30, 100])
def test_cycle_cost_is_within_twin_budget(receiver, n_members):
    """One predict + update on the 3-state augmented receiver (RK4, 1 substep).

    Measured on the development laptop: ~2 ms (N = 30) and 6-8 ms (N = 100),
    dominated by the per-member Python model calls. Budget: well under 1 s.
    """
    aug = AugmentedModel(receiver, disturbances=("Q_sun",))
    F = discretize(aug, 5.0)
    P0 = np.diag([10.0, 10.0, 1.0e8])
    Q = np.diag([0.01, 0.01, 1.0e6])
    f = EnsembleKalmanFilter(
        F, aug.output, Q, np.eye(1), [1200.0, 1090.0, 8e5], P0, n_members, seed=0
    )
    u, d, y = np.array([1.0]), np.array([300.0, 0.0]), np.array([1090.0])
    n_cycles = 20
    t0 = time.perf_counter()
    for _ in range(n_cycles):
        f.predict(u, d)
        f.update(y)
    per_cycle = (time.perf_counter() - t0) / n_cycles
    print(f"EnKF N={n_members}: {1e3 * per_cycle:.2f} ms per predict+update")
    assert per_cycle < 0.1
