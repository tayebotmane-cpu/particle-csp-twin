"""Disturbance augmentation: layout, backend agreement, observability, step recovery."""

import casadi as ca
import numpy as np
import pytest

from pcsp.core import ControlModel, casadi_discretize, casadi_functions, discretize, simulate
from pcsp.twin.estimation import (
    AugmentedModel,
    EnsembleKalmanFilter,
    ExtendedKalmanFilter,
    observability,
    rmse,
)

DT = 5.0  # s


def test_layout_and_indices(receiver):
    aug = AugmentedModel(receiver, disturbances=("Q_sun",), output_biases=("T_p",))
    assert isinstance(aug, ControlModel)
    assert aug.state_names == ("T_w", "T_p", "d:Q_sun", "bias:T_p")
    assert aug.disturbance_names == ("T_amb", "Q_aux")
    assert aug.input_names == receiver.input_names
    assert aug.output_names == receiver.output_names
    assert aug.physical_state_indices == (0, 1)
    assert aug.disturbance_state_indices == (2,)
    assert aug.bias_state_indices == (3,)


def test_rejects_unknown_or_duplicate_names(receiver):
    with pytest.raises(ValueError, match="unknown disturbance"):
        AugmentedModel(receiver, disturbances=("Q_moon",))
    with pytest.raises(ValueError, match="duplicate"):
        AugmentedModel(receiver, disturbances=("Q_sun", "Q_sun"))
    with pytest.raises(ValueError, match="unknown output"):
        AugmentedModel(receiver, output_biases=("T_w",))


def test_join_and_split_round_trip(receiver):
    aug = AugmentedModel(receiver, disturbances=("Q_aux", "Q_sun"), output_biases=("T_p",))
    x, d, b = np.array([1200.0, 1090.0]), np.array([8e5, 290.0, 1e4]), np.array([-2.0])
    z = aug.join(x, d, b)
    np.testing.assert_array_equal(z, [1200.0, 1090.0, 1e4, 8e5, -2.0])
    d_meas = aug.measured_disturbances(d)
    np.testing.assert_array_equal(d_meas, [290.0])
    x_hat, d_hat = aug.split(z, d_meas)
    np.testing.assert_array_equal(x_hat, x)
    np.testing.assert_array_equal(d_hat, d)
    np.testing.assert_array_equal(aug.output_bias(z), b)
    with pytest.raises(ValueError, match="d_measured"):
        aug.split(z)


def test_augmented_model_reproduces_original_dynamics(receiver):
    aug = AugmentedModel(receiver, disturbances=("Q_sun",), output_biases=("T_p",))
    x, u, d = np.array([1200.0, 1090.0]), np.array([1.2]), np.array([8e5, 290.0, 3e4])
    z = aug.join(x, d, np.array([1.5]))
    z_dot = aug.rhs(z, u, aug.measured_disturbances(d))
    np.testing.assert_allclose(z_dot[:2], receiver.rhs(x, u, d), rtol=1e-15)
    np.testing.assert_array_equal(z_dot[2:], 0.0)
    y = aug.output(z, u, aug.measured_disturbances(d))
    np.testing.assert_allclose(y, receiver.output(x, u, d) + 1.5, rtol=1e-15)


@pytest.mark.parametrize(
    ("disturbances", "biases"),
    [(("Q_sun",), ()), (("Q_sun", "T_amb", "Q_aux"), ("T_p",)), ((), ("T_p",))],
)
def test_numpy_and_casadi_augmented_model_agree(receiver, disturbances, biases):
    aug = AugmentedModel(receiver, disturbances=disturbances, output_biases=biases)
    f, h = casadi_functions(aug)
    F_ca, F_np = casadi_discretize(aug, DT, 2), discretize(aug, DT, 2)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.uniform(600.0, 1400.0, 2)
        d = np.array([rng.uniform(0.0, 1e6), rng.uniform(270.0, 310.0), rng.uniform(0, 1e5)])
        z = aug.join(x, d, rng.normal(size=1))
        u, dm = rng.uniform(0.5, 2.0, 1), aug.measured_disturbances(d)
        np.testing.assert_allclose(np.array(f(z, u, dm)).ravel(), aug.rhs(z, u, dm), rtol=1e-12)
        np.testing.assert_allclose(np.array(h(z, u, dm)).ravel(), aug.output(z, u, dm), rtol=1e-12)
        np.testing.assert_allclose(np.array(F_ca(z, u, dm)).ravel(), F_np(z, u, dm), rtol=1e-12)
    assert isinstance(
        aug.rhs(ca.SX.sym("z", aug.nx), ca.SX.sym("u", 1), ca.SX.sym("d", aug.nd)), ca.SX
    )


# ------------------------------------------------------------- observability
OP_X, OP_U = np.array([1205.0, 1088.0]), np.array([1.0])
OP_D = np.array([8e5, 300.0, 0.0])  # Q_sun, T_amb, Q_aux


def test_observability_full_rank_with_one_augmented_heat_input(receiver):
    aug = AugmentedModel(receiver, disturbances=("Q_sun",))
    res = observability(
        aug,
        aug.join(OP_X, OP_D),
        OP_U,
        aug.measured_disturbances(OP_D),
        DT,
        state_scale=np.array([1000.0, 1000.0, 1e6]),
        output_scale=np.array([1000.0]),
    )
    assert res.observable and res.rank == 3
    assert np.isfinite(res.condition_number)
    assert res.unobservable_directions.shape == (3, 0)


def test_observability_detects_indistinguishable_disturbances(receiver):
    # Q_sun and Q_aux enter the wall balance identically: only their sum is observable.
    aug = AugmentedModel(receiver, disturbances=("Q_sun", "Q_aux"))
    res = observability(aug, aug.join(OP_X, OP_D), OP_U, aug.measured_disturbances(OP_D), DT)
    assert not res.observable and res.rank == 3
    assert res.condition_number == np.inf
    direction = res.unobservable_directions[:, 0]
    np.testing.assert_allclose(
        np.abs(direction), [0.0, 0.0, 1 / np.sqrt(2), 1 / np.sqrt(2)], atol=1e-6
    )
    assert direction[2] * direction[3] < 0  # the difference Q_sun - Q_aux is invisible


def test_observability_disturbance_plus_bias_exceeds_one_sensor(receiver):
    # Two integrating states (Q_sun and the T_p bias) but one sensor: at steady
    # state they cannot be separated (Pannocchia & Rawlings, 2003).
    aug = AugmentedModel(receiver, disturbances=("Q_sun",), output_biases=("T_p",))
    res = observability(aug, aug.join(OP_X, OP_D), OP_U, aug.measured_disturbances(OP_D), DT)
    assert not res.observable


# --------------------------------------------------------------- step recovery
def receiver_step_experiment(receiver, n_steps: int = 600, t_step: float = 500.0):
    """Plant at steady state with Q_sun = 800 kW, stepping to 600 kW at t_step."""
    F = discretize(receiver, DT)
    u = np.full((n_steps, 1), 1.0)
    q_sun = np.where(np.arange(n_steps) * DT < t_step, 8.0e5, 6.0e5)
    d = np.column_stack([q_sun, np.full(n_steps, 300.0), np.zeros(n_steps)])
    x = np.array([1000.0, 900.0])
    for _ in range(3000):
        x = F(x, u[0], d[0])
    X, _ = simulate(receiver, x, u, d, DT)
    rng = np.random.default_rng(0)
    Y = X[1:, 1:2] + 1.0 * rng.standard_normal((n_steps, 1))  # thermocouple, σ = 1 K
    return X, u, d, Y


def settling_time(error: np.ndarray, start: int, band: float) -> float:
    """Time after sample ``start`` until ``|error|`` enters ``band`` for good [s]."""
    outside = np.nonzero(np.abs(error[start:]) > band)[0]
    return 0.0 if outside.size == 0 else (outside[-1] + 1) * DT


def make_filter(kind: str, aug: AugmentedModel, z0, P0, Q, R):
    if kind == "ekf":
        return ExtendedKalmanFilter(aug, DT, Q, R, z0, P0)
    lower = np.array([300.0, 300.0, 0.0])
    return EnsembleKalmanFilter(
        discretize(aug, DT), aug.output, Q, R, z0, P0, 50, method=kind, lower=lower, seed=1
    )


@pytest.mark.parametrize("kind", ["stochastic", "sqrt", "ekf"])
def test_augmented_filter_recovers_unmeasured_solar_step(receiver, kind):
    """A 200 kW drop in solar input, seen only through one thermocouple on T_p.

    Measured (σ_T = 1 K, random-walk std 2 kW per 5 s sample, N = 50):
    settling into ±10 % of the step after 140-160 s (28-32 samples); bias over
    the last 400 s below 0.5 % of the step; T_p RMSE ≈ 0.75 K < σ.
    """
    X, u, d, Y = receiver_step_experiment(receiver)
    aug = AugmentedModel(receiver, disturbances=("Q_sun",))
    z0 = aug.join(X[0] + np.array([20.0, -10.0]), np.array([7.0e5, 300.0, 0.0]))
    P0 = np.diag([30.0**2, 30.0**2, (2.0e5) ** 2])
    Q = np.diag([0.1**2, 0.1**2, (2.0e3) ** 2])
    f = make_filter(kind, aug, z0, P0, Q, np.array([[1.0]]))

    q_hat, x_hat = [], []
    for k in range(len(Y)):
        f.predict(u[k], aug.measured_disturbances(d[k]))
        f.update(Y[k])
        xk, dk = aug.split(f.x_hat, aug.measured_disturbances(d[k]))
        x_hat.append(xk)
        q_hat.append(dk[0])
    error = np.array(q_hat) - d[:, 0]  # estimate at (k+1)Δt vs input held over step k

    step, k_step = 2.0e5, int(500.0 / DT)
    t_settle = settling_time(error, k_step, 0.1 * step)
    final_bias = error[-80:].mean()
    print(f"{kind}: settling {t_settle:.0f} s, final bias {final_bias:.0f} W")
    assert t_settle <= 250.0
    assert abs(final_bias) < 0.01 * step
    assert rmse(np.array(x_hat)[:, 1], X[1:, 1], axis=None) < 1.0  # below thermocouple σ
