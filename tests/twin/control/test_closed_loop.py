"""Closed-loop simulator: timeline, noise seeding, actuator limits, mismatch, estimator."""

import numpy as np
import pytest
from scipy.signal import place_poles

from pcsp.core import ControlModelBase, Estimator, discretize, ops_for
from pcsp.twin.control import (
    NMPC,
    NMPCConfig,
    PIDController,
    piecewise_constant,
    ramp,
    run_closed_loop,
    step,
)


class AugmentedLuenberger:
    """Stub estimator: first-order model augmented with an integrating input disturbance.

    ``z = (x, d_hat)``, ``z+ = (F(x, u, d_hat), d_hat)``, ``y = x``. Filter form:
    ``predict`` propagates the model, ``update`` adds ``L (y - x^-)``; ``L`` places
    the eigenvalues of the error dynamics ``(I - L C) A`` at 0.6 and 0.7.
    """

    def __init__(self, model, dt: float, z0) -> None:
        self._F = discretize(model, dt)
        a = self._F(np.ones(1), np.zeros(1), np.zeros(1))[0]
        e = self._F(np.zeros(1), np.zeros(1), np.ones(1))[0]
        A, C = np.array([[a, e], [0.0, 1.0]]), np.array([[1.0, 0.0]])
        self.L = place_poles(A.T, (C @ A).T, [0.6, 0.7]).gain_matrix.T[:, 0]
        self._z = np.asarray(z0, dtype=float)
        self.calls: list[tuple] = []

    def predict(self, u, d):
        self.calls.append(("predict", float(u[0]), float(d[0])))
        self._z = np.array([self._F(self._z[:1], u, self._z[1:])[0], self._z[1]])

    def update(self, y, u=None, d=None):
        self.calls.append(("update",))
        self._z = self._z + self.L * (y[0] - self._z[0])

    @property
    def x_hat(self):
        return self._z.copy()

    @property
    def P(self):
        return np.eye(2)


def nmpc_for(model, u_init=0.0):
    cfg = NMPCConfig(
        dt=1.0,
        horizon=15,
        tracked_outputs=[0],
        output_weight=[1.0],
        move_weight=[0.1],
        u_min=[-1e3],
        u_max=[1e3],
        u_init=[u_init],
    )
    return NMPC(model, cfg)


def test_mismatch_gives_offset_and_disturbance_estimator_removes_it(first_order):
    """Controller model K=2, tau=10; plant K=2.6, tau=13 with load d=0.3.

    With the true state and true d the mismatch leaves an offset. An estimator
    on the augmented model (2 states, not the plant's 1) feeds the controller
    ``x_hat = z[:1]`` and ``d_hat = z[1:]``; integrating-disturbance theory
    (Pannocchia & Rawlings 2003) gives zero offset.
    """
    model, plant = first_order(2.0, 10.0), first_order(2.6, 13.0)
    common = dict(dt=1.0, n_steps=200, setpoint=1.0, disturbance=step(20.0, 0.0, 0.3))
    no_est = run_closed_loop(
        plant, nmpc_for(model), [0.0], [0.0], full_state_feedback=True, **common
    )
    est = AugmentedLuenberger(model, 1.0, [0.0, 0.0])
    assert isinstance(est, Estimator)
    with_est = run_closed_loop(
        plant,
        nmpc_for(model),
        [0.0],
        [0.0],
        estimator=est,
        estimate_to_state=lambda z: z[:1],
        estimate_to_disturbance=lambda z: z[1:],
        **common,
    )
    offset_no_est = no_est.y_true[-1, 0] - 1.0
    offset_est = with_est.y_true[-1, 0] - 1.0
    print(
        f"mismatch offset: true state {offset_no_est:.4f}, "
        f"with disturbance estimator {offset_est:.1e}"
    )
    assert abs(offset_no_est) > 0.01
    assert abs(offset_est) < 1e-6
    assert with_est.x_est.shape == (200, 2) and with_est.x_hat.shape == (200, 1)
    np.testing.assert_allclose(with_est.d_hat[:, 0], with_est.x_est[:, 1])


def test_estimator_cycle_order_and_inputs(first_order):
    """``update(y_0)``, then ``predict(u_{k-1}, d_{k-1})`` + ``update(y_k)`` each sample."""
    model = first_order()
    est = AugmentedLuenberger(model, 1.0, [0.0, 0.0])
    res = run_closed_loop(
        model,
        PIDController(1.0, 0.1, 0.0, 1.0),
        [0.0],
        [0.0],
        dt=1.0,
        n_steps=4,
        setpoint=1.0,
        disturbance=step(1.5, 0.0, 0.5),
        estimator=est,
    )
    expected = [("update",)]
    for k in range(1, 4):
        expected += [("predict", res.u[k - 1, 0], res.d[k - 1, 0]), ("update",)]
    assert est.calls == expected


def test_same_seed_reproduces_run_and_noise_has_requested_std(first_order):
    def run(seed):
        return run_closed_loop(
            first_order(),
            PIDController(1.0, 0.1, 0.0, 0.5),
            [0.0],
            [0.0],
            dt=0.5,
            n_steps=2000,
            setpoint=1.0,
            disturbance=0.0,
            noise_std=0.05,
            seed=seed,
        )

    a, b, c = run(7), run(7), run(8)
    np.testing.assert_array_equal(a.y_meas, b.y_meas)
    np.testing.assert_array_equal(a.u, b.u)
    assert not np.array_equal(a.y_meas, c.y_meas)
    # Sample std of 2000 draws: relative standard error ~1.6 %, so 5 % is a 3-sigma bound.
    assert np.std(a.y_meas - a.y_true) == pytest.approx(0.05, rel=0.05)


def test_actuator_rate_limit_acts_on_plant_input_only(first_order):
    rate, dt = 0.1, 0.5
    res = run_closed_loop(
        first_order(),
        PIDController(5.0, 0.0, 0.0, dt),
        [0.0],
        [0.0],
        dt=dt,
        n_steps=100,
        setpoint=1.0,
        disturbance=0.0,
        actuator_rate_max=rate,
    )
    du = np.abs(np.diff(np.concatenate([[0.0], res.u[:, 0]])))
    assert du.max() <= rate * dt * (1 + 1e-12)
    assert res.u_cmd[0, 0] == pytest.approx(5.0)  # the controller itself is not limited
    assert res.u[0, 0] == pytest.approx(rate * dt)


def test_measurement_uses_input_held_over_previous_period():
    """``y_k = h(x_k, u_{k-1}, d_{k-1})`` for a model with direct feedthrough."""

    class Feedthrough(ControlModelBase):
        state_names, input_names, disturbance_names, output_names = ("x",), ("u",), ("d",), ("y",)

        def rhs(self, x, u, d):
            return ops_for(x, u, d).concat([-x[0] + u[0]])

        def output(self, x, u, d):
            return ops_for(x, u, d).concat([x[0] + 10.0 * u[0] + 100.0 * d[0]])

    res = run_closed_loop(
        Feedthrough(),
        PIDController(0.3, 0.0, 0.0, 1.0, bias=0.2),
        [1.0],
        [0.7],
        dt=1.0,
        n_steps=5,
        setpoint=0.0,
        disturbance=step(2.0, 0.0, 0.01),
        plant_substeps=4,
    )
    u_held = np.concatenate([[0.7], res.u[:-1, 0]])
    d_held = np.concatenate([res.d[:1, 0], res.d[:-1, 0]])
    np.testing.assert_allclose(res.y_true[:, 0], res.x[:, 0] + 10.0 * u_held + 100.0 * d_held)


def test_array_and_callable_profiles_are_equivalent(first_order):
    n, dt = 50, 1.0
    profile = piecewise_constant([10.0, 30.0], [0.0, 1.0, 0.5])
    as_array = np.array([profile(k * dt) for k in range(n)])
    runs = [
        run_closed_loop(
            first_order(),
            PIDController(1.0, 0.1, 0.0, dt),
            [0.0],
            [0.0],
            dt=dt,
            n_steps=n,
            setpoint=spec,
            disturbance=0.0,
        )
        for spec in (profile, as_array)
    ]
    np.testing.assert_array_equal(runs[0].y_meas, runs[1].y_meas)
    np.testing.assert_array_equal(runs[0].r[:, 0], as_array[:, 0])


def test_profile_helpers():
    s = step(5.0, 1.0, 2.0)
    assert s(4.999)[0] == 1.0 and s(5.0)[0] == 2.0
    rp = ramp(10.0, 20.0, [0.0, 100.0], [1.0, 0.0])
    np.testing.assert_allclose(rp(0.0), [0.0, 100.0])
    np.testing.assert_allclose(rp(15.0), [0.5, 50.0])
    np.testing.assert_allclose(rp(99.0), [1.0, 0.0])
    pc = piecewise_constant([1.0, 2.0], [3.0, 4.0, 5.0])
    assert [pc(t)[0] for t in (0.0, 1.0, 1.5, 2.0, 9.0)] == [3.0, 4.0, 4.0, 5.0, 5.0]
    with pytest.raises(ValueError):
        piecewise_constant([1.0], [1.0])
