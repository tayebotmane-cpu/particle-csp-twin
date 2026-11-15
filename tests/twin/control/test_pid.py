"""PID behaviour against analytic closed-loop responses on a first-order plant.

Plant: ``tau dy/dt = -y + K u + d``. With the input held over each sample
period ``h`` its exact discretisation is ``y+ = a y + (1 - a)(K u + d)``,
``a = exp(-h / tau)``; the simulator integrates it with fine RK4 substeps.
"""

import numpy as np
import pytest

from pcsp.core import Controller
from pcsp.twin.control import (
    PIDController,
    overshoot,
    run_closed_loop,
    settling_time,
    step,
)


def simulate(plant, pid, *, dt, n_steps, setpoint, disturbance=0.0, u0=0.0, x0=0.0):
    return run_closed_loop(
        plant,
        pid,
        [x0],
        [u0],
        dt=dt,
        n_steps=n_steps,
        setpoint=setpoint,
        disturbance=disturbance,
        plant_substeps=20,
    )


def test_pid_satisfies_controller_protocol():
    assert isinstance(PIDController(1.0, 0.1, 0.0, dt=1.0), Controller)


def test_p_control_matches_exact_discrete_closed_loop(first_order):
    """``u_k = kp (r - y_k)`` gives ``y_k = y_ss + (y_0 - y_ss) lambda^k`` exactly.

    ``lambda = a - (1 - a) K kp`` and ``y_ss = K kp r / (1 + K kp)``.
    """
    K, tau, kp, h, r = 2.0, 10.0, 1.5, 1.0, 1.0
    res = simulate(
        first_order(K, tau), PIDController(kp, 0.0, 0.0, h), dt=h, n_steps=60, setpoint=r
    )
    a = np.exp(-h / tau)
    lam = a - (1.0 - a) * K * kp
    y_ss = K * kp * r / (1.0 + K * kp)
    k = np.arange(60)
    np.testing.assert_allclose(res.y_meas[:, 0], y_ss * (1.0 - lam**k), rtol=0, atol=1e-10)


@pytest.mark.parametrize("h", [0.1, 0.05])
def test_pi_with_pole_zero_cancellation_converges_to_continuous_response(first_order, h):
    """With ``T_i = tau`` the continuous loop is ``y = r (1 - exp(-t K kp / tau))``.

    Sampling, the zero-order hold and the forward-Euler integrator are
    first-order accurate in ``h``: the error must be below ``0.5 h / tau_cl``
    (a half-sample delay on a first-order response) and halve with ``h``.
    """
    K, tau, kp, r = 2.0, 10.0, 1.0, 1.0
    tau_cl = tau / (K * kp)
    n = int(round(30.0 / h))
    res = simulate(
        first_order(K, tau), PIDController(kp, kp / tau, 0.0, h), dt=h, n_steps=n, setpoint=r
    )
    err = np.max(np.abs(res.y_meas[:, 0] - r * (1.0 - np.exp(-res.t / tau_cl))))
    assert err < 0.5 * h / tau_cl
    if h == 0.05:
        res_coarse = simulate(
            first_order(K, tau),
            PIDController(kp, kp / tau, 0.0, 0.1),
            dt=0.1,
            n_steps=300,
            setpoint=r,
        )
        err_coarse = np.max(
            np.abs(res_coarse.y_meas[:, 0] - r * (1 - np.exp(-res_coarse.t / tau_cl)))
        )
        assert err_coarse / err == pytest.approx(2.0, rel=0.05)


def test_integral_action_removes_steady_state_offset(first_order):
    """P-only leaves the analytic offset ``(d - r)/(1 + K kp)``; PI removes it."""
    K, tau, kp, r, d = 2.0, 10.0, 1.5, 1.0, 0.8
    dist = step(20.0, 0.0, d)
    p_only = simulate(
        first_order(K, tau),
        PIDController(kp, 0.0, 0.0, 0.5),
        dt=0.5,
        n_steps=400,
        setpoint=r,
        disturbance=dist,
    )
    assert p_only.y_meas[-1, 0] - r == pytest.approx((d - r) / (1 + K * kp), abs=1e-9)
    pi = simulate(
        first_order(K, tau),
        PIDController(kp, 0.3, 0.0, 0.5),
        dt=0.5,
        n_steps=400,
        setpoint=r,
        disturbance=dist,
    )
    assert abs(pi.y_meas[-1, 0] - r) < 1e-6


def _windup_case(first_order, anti_windup: bool, **limits):
    pid = PIDController(5.0, 1.0, 0.0, 0.1, anti_windup=anti_windup, **limits)
    return simulate(first_order(1.0, 10.0), pid, dt=0.1, n_steps=1500, setpoint=1.0)


@pytest.mark.parametrize(
    "limits", [{"u_min": 0.0, "u_max": 1.2}, {"rate_max": 0.02}], ids=["saturation", "rate_limit"]
)
def test_anti_windup_recovers_much_faster_than_naive_integrator(first_order, limits):
    """A 0 -> 1 setpoint step that the limited actuator cannot follow at once.

    The naive integrator accumulates error while the output is limited and
    overshoots; back-calculation on the *applied* output does not. Under the
    rate limit the naive loop even falls into a growing limit cycle, so its
    settling time is infinite.
    """
    aw = _windup_case(first_order, True, **limits)
    naive = _windup_case(first_order, False, **limits)
    y_aw, y_naive = aw.y_meas[:, 0], naive.y_meas[:, 0]
    ts_aw = settling_time(aw.t, y_aw, 1.0, band=0.02)
    ts_naive = settling_time(naive.t, y_naive, 1.0, band=0.02)
    print(f"{limits}: settling time (2 %) anti-windup {ts_aw:.1f} s, naive {ts_naive:.1f} s")
    assert np.isfinite(ts_aw) and ts_aw < 0.5 * ts_naive
    assert overshoot(y_aw, 0.0, 1.0) < 2.0 < overshoot(y_naive, 0.0, 1.0)
    for res in (aw, naive):  # limits are never exceeded
        du = np.abs(np.diff(res.u[:, 0]))
        assert du.max() <= limits.get("rate_max", np.inf) * 0.1 * (1 + 1e-12)
        assert (
            limits.get("u_min", -np.inf)
            <= res.u.min()
            <= res.u.max()
            <= limits.get("u_max", np.inf)
        )


def test_direct_action_controls_a_negative_gain_plant(first_order):
    """ISA convention: a negative-gain process needs a *direct-acting* controller.

    (Called "reverse action" in some texts: the error sign is reversed with
    respect to the default ``e = r - y``.) The wrong action is unstable.
    """
    plant = first_order(-2.0, 10.0)
    direct = simulate(
        plant,
        PIDController(0.5, 0.05, 0.0, 0.5, action="direct"),
        dt=0.5,
        n_steps=400,
        setpoint=1.0,
    )
    assert abs(direct.y_meas[-1, 0] - 1.0) < 1e-6
    wrong = simulate(
        plant,
        PIDController(0.5, 0.05, 0.0, 0.5, action="reverse"),
        dt=0.5,
        n_steps=100,
        setpoint=1.0,
    )
    assert abs(wrong.y_meas[-1, 0]) > 100.0


def test_derivative_acts_on_filtered_measurement_only():
    """Backward-Euler ``kd s / (1 + T_f s)`` on ``-y``; no kick on setpoint steps."""
    kd, t_f, h, bias = 2.0, 1.0, 0.5, 3.0
    pid = PIDController(0.0, 0.0, kd, h, bias=bias, derivative_filter_time=t_f)
    # Bumpless start and setpoint step: output stays exactly at the bias.
    assert pid.step(0.0, np.array([0.0]), np.array([0.0]))[0] == bias
    assert pid.step(0.5, np.array([0.0]), np.array([5.0]))[0] == bias
    # Unit measurement step: D_1 = -kd / (T_f + h), then decays by T_f / (T_f + h).
    u = [pid.step(1.0 + 0.5 * k, np.array([1.0]), np.array([5.0]))[0] for k in range(6)]
    alpha = t_f / (t_f + h)
    np.testing.assert_allclose(
        np.array(u) - bias, -kd / (t_f + h) * alpha ** np.arange(6), atol=1e-15
    )


def test_static_feedforward_cancels_a_measured_disturbance(first_order):
    """``u_ff = -d / K`` cancels ``d`` exactly, so P-only control is unaffected by it."""
    K, kp = 2.0, 1.5
    dist = step(10.0, 0.0, 0.8)
    pid_ff = PIDController(kp, 0.0, 0.0, 0.5, feedforward=lambda d: -d[0] / K)
    with_ff = simulate(
        first_order(K, 10.0), pid_ff, dt=0.5, n_steps=200, setpoint=1.0, disturbance=dist
    )
    no_dist = simulate(
        first_order(K, 10.0), PIDController(kp, 0.0, 0.0, 0.5), dt=0.5, n_steps=200, setpoint=1.0
    )
    np.testing.assert_allclose(with_ff.y_meas, no_dist.y_meas, atol=1e-12)


def test_feedforward_without_disturbance_is_an_error():
    pid = PIDController(1.0, 0.0, 0.0, 1.0, feedforward=lambda d: d[0])
    with pytest.raises(ValueError, match="d_hat"):
        pid.step(0.0, np.array([0.0]), np.array([1.0]))
