"""NMPC against analytic LQ solutions, constraint satisfaction and offset theory."""

import casadi as ca
import numpy as np
import pytest

from pcsp.core import Controller, ControlModelBase, casadi_discretize, ops_for
from pcsp.twin.control import (
    NMPC,
    NMPCConfig,
    PIDController,
    integral_absolute_error,
    piecewise_constant,
    run_closed_loop,
    step,
)

# --------------------------------------------------------------------------- helpers


def discrete_linear_maps(
    model, dt: float, substeps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(A_d, B_d, E_d)`` with ``x+ = A_d x + B_d u + E_d d`` for a linear model.

    Exact Jacobians of the RK4 map the NMPC itself uses, so the LQ solutions
    below refer to precisely the same discrete-time system.
    """
    x, u, d = ca.SX.sym("x", model.nx), ca.SX.sym("u", model.nu), ca.SX.sym("d", model.nd)
    x_next = casadi_discretize(model, dt, substeps)(x, u, d)
    jac = ca.Function("linear_maps", [x, u, d], [ca.jacobian(x_next, v) for v in (x, u, d)])
    return tuple(np.array(M, dtype=float) for M in jac(0, 0, 0))


def lq_first_move(A, B, E, C, *, x0, u_prev, r, d, Q, R, S, u_ref, N):
    """First input of the unconstrained finite-horizon LQ problem (batch least squares).

    Minimises ``sum_{j=1}^N |C x_j - r|_Q^2 + sum_{j=0}^{N-1} (|u_j - u_{j-1}|_R^2
    + |u_j - u_ref|_S^2)`` s.t. ``x_{j+1} = A x_j + B u_j + E d_j``: the cost is a
    sum of squares of affine functions of ``U = (u_0, ..., u_{N-1})``, so the
    minimiser is a linear least-squares solution. ``d`` is ``(nd,)`` or ``(N, nd)``.
    """
    nx, nu = B.shape
    D = np.broadcast_to(np.asarray(d, float).reshape(-1, E.shape[1]), (N, E.shape[1]))
    sq, sr, ss = np.sqrt(Q), np.sqrt(R), np.sqrt(S)
    rows, rhs = [], []
    x_free = np.asarray(x0, float)
    gamma = np.zeros((nx, N * nu))  # d x_j / d U
    for j in range(N):
        x_free = A @ x_free + E @ D[j]
        gamma = A @ gamma
        gamma[:, j * nu : (j + 1) * nu] += B
        rows.append(sq[:, None] * (C @ gamma))
        rhs.append(sq * (r - C @ x_free))
        move = np.zeros((nu, N * nu))
        move[:, j * nu : (j + 1) * nu] = np.eye(nu)
        if j > 0:
            move[:, (j - 1) * nu : j * nu] = -np.eye(nu)
        rows.append(sr[:, None] * move)
        rhs.append(sr * (np.asarray(u_prev, float) if j == 0 else np.zeros(nu)))
        pick = np.zeros((nu, N * nu))
        pick[:, j * nu : (j + 1) * nu] = np.eye(nu)
        rows.append(ss[:, None] * pick)
        rhs.append(ss * np.asarray(u_ref, float))
    U, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)
    return U[:nu]


def riccati_first_move(A, B, C, *, x0, u_prev, Q, R, S, N):
    """Regulator (``r = 0``, ``u_ref = 0``) first move by the backward Riccati recursion.

    Augmented state ``z = (x, u_{j-1})`` and input ``v = u_j - u_{j-1}``; stage
    cost ``v'Rv + (Ez+v)'S(Ez+v) + [j >= 1] |C x_j|_Q^2`` with ``E z = u_{j-1}``,
    i.e. a cross term ``2 z' E'S v``. Terminal cost ``|C x_N|_Q^2``.
    """
    nx, nu = B.shape
    Az = np.block([[A, B], [np.zeros((nu, nx)), np.eye(nu)]])
    Bz = np.vstack([B, np.eye(nu)])
    Cz = np.hstack([C, np.zeros((C.shape[0], nu))])
    Ez = np.hstack([np.zeros((nu, nx)), np.eye(nu)])
    Qy, Rm, Sm = np.diag(Q), np.diag(R), np.diag(S)
    P = Cz.T @ Qy @ Cz
    for j in range(N - 1, -1, -1):
        Qj = Ez.T @ Sm @ Ez + (Cz.T @ Qy @ Cz if j >= 1 else 0.0)
        H = Rm + Sm + Bz.T @ P @ Bz
        G = Bz.T @ P @ Az + Sm @ Ez  # (Ez'S)' = S Ez
        K = np.linalg.solve(H, G)
        P = Qj + Az.T @ P @ Az - G.T @ K
    z0 = np.concatenate([x0, u_prev])
    return np.asarray(u_prev) - K @ z0


# --------------------------------------------------------------------------- fixtures

T_SET, Q0, Q1 = 1073.15, 1.0e6, 0.8e6  # K, W, W (20 % drop in solar power)
DT, T_STEP, RATE, U_MIN, U_MAX = 5.0, 50.0, 0.05, 0.5, 8.0  # s, s, kg/s^2, kg/s, kg/s


def receiver_config(m0: float, **overrides) -> NMPCConfig:
    """A sensible NMPC for the toy receiver: 4 states, N = 24 (2 min look-ahead)."""
    settings = dict(
        dt=DT,
        horizon=24,
        tracked_outputs=[0],
        output_weight=[1.0],
        move_weight=[10.0],
        u_min=[U_MIN],
        u_max=[U_MAX],
        u_init=[m0],
        rate_max=[RATE],
        substeps=2,
        x_scale=[100.0] * 4,
        x_offset=[1000.0] * 4,
        u_scale=[1.0],
    )
    settings.update(overrides)
    return NMPCConfig(**settings)


@pytest.fixture(scope="module")
def receiver_case(toy_receiver):
    plant = toy_receiver()
    m0 = plant.flow_for(T_SET, Q0)
    return plant, m0, plant.steady_state(m0, Q0)


def run_receiver(plant, controller, x0, m0, *, n_steps=120, setpoint=T_SET, **kwargs):
    return run_closed_loop(
        plant,
        controller,
        x0,
        [m0],
        dt=DT,
        n_steps=n_steps,
        setpoint=setpoint,
        disturbance=step(T_STEP, Q0, Q1),
        plant_substeps=2,
        **kwargs,
    )


# --------------------------------------------------------------------------- LQ theory


def test_nmpc_satisfies_controller_protocol(first_order):
    cfg = NMPCConfig(
        dt=1.0,
        horizon=3,
        tracked_outputs=[0],
        output_weight=[1.0],
        move_weight=[1.0],
        u_min=[-1.0],
        u_max=[1.0],
        u_init=[0.0],
    )
    assert isinstance(NMPC(first_order(), cfg), Controller)


def test_unconstrained_nmpc_matches_riccati_lq_solution(linear_system):
    """3 states, 2 inputs, 2 outputs; bounds far away: NLP == finite-horizon LQ."""
    A = np.array([[-0.1, 0.05, 0.0], [0.0, -0.2, 0.1], [0.02, 0.0, -0.5]])
    B = np.array([[0.0, 0.1], [0.0, 0.3], [1.0, 0.0]])
    C = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    model = linear_system(A, B, C)
    Q, R, S, N, dt = np.array([2.0, 1.0]), np.array([0.5, 0.2]), np.array([0.1, 0.05]), 25, 0.5
    cfg = NMPCConfig(
        dt=dt,
        horizon=N,
        tracked_outputs=[0, 1],
        output_weight=Q,
        move_weight=R,
        input_weight=S,
        u_ref=[0.0, 0.0],
        u_min=[-1e3] * 2,
        u_max=[1e3] * 2,
        u_init=[0.4, -0.3],
        substeps=2,
    )
    mpc = NMPC(model, cfg)
    x0 = np.array([1.0, -0.5, 0.7])
    u_nmpc = mpc.step(0.0, C @ x0, np.zeros(2), x_hat=x0)
    Ad, Bd, _ = discrete_linear_maps(model, dt, 2)
    u_lq = riccati_first_move(Ad, Bd, C, x0=x0, u_prev=[0.4, -0.3], Q=Q, R=R, S=S, N=N)
    print(f"Riccati first move {u_lq}, NMPC {u_nmpc}, diff {np.abs(u_nmpc - u_lq).max():.2e}")
    assert mpc.solve_log[-1].success
    np.testing.assert_allclose(u_nmpc, u_lq, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("preview", [False, True])
def test_nmpc_tracking_with_disturbance_matches_batch_lq(first_order, preview):
    """Non-zero setpoint, input reference and disturbance (constant or previewed)."""
    model = first_order(2.0, 10.0)
    N, dt, r, x0, u_prev = 15, 1.0, 1.5, 0.1, 0.2
    cfg = NMPCConfig(
        dt=dt,
        horizon=N,
        tracked_outputs=[0],
        output_weight=[1.0],
        move_weight=[0.3],
        input_weight=[0.05],
        u_ref=[0.5],
        u_min=[-1e3],
        u_max=[1e3],
        u_init=[u_prev],
    )
    d = np.where(np.arange(N) >= 4, 0.9, 0.3)[:, None] if preview else np.array([0.3])
    mpc = NMPC(model, cfg)
    kwargs = {"d_preview": d} if preview else {"d_hat": d}
    u_nmpc = mpc.step(0.0, [x0], [r], x_hat=[x0], **kwargs)
    Ad, Bd, Ed = discrete_linear_maps(model, dt, 1)
    u_lq = lq_first_move(
        Ad,
        Bd,
        Ed,
        np.eye(1),
        x0=[x0],
        u_prev=[u_prev],
        r=[r],
        d=d,
        Q=np.ones(1),
        R=[0.3],
        S=[0.05],
        u_ref=[0.5],
        N=N,
    )
    np.testing.assert_allclose(u_nmpc, u_lq, rtol=1e-7, atol=1e-9)


def test_wrong_disturbance_gives_predicted_offset_and_true_one_removes_it(first_order):
    """Perfect model, true state; ``d_hat`` wrong by 1.0 -> offset; ``d_hat = d`` -> none.

    The unconstrained NMPC law is affine, ``u = a x + b u_prev + c``; with the
    plant ``x+ = A x + B u + E d`` the closed-loop equilibrium solves a 2x2
    linear system, which predicts the offset exactly.
    """
    model = first_order(2.0, 10.0)
    N, dt, r, d_true, d_wrong = 15, 1.0, 1.0, 1.0, 0.0
    cfg = NMPCConfig(
        dt=dt,
        horizon=N,
        tracked_outputs=[0],
        output_weight=[1.0],
        move_weight=[0.1],
        u_min=[-1e3],
        u_max=[1e3],
        u_init=[0.0],
    )
    Ad, Bd, Ed = discrete_linear_maps(model, dt, 1)
    A, B, E = Ad[0, 0], Bd[0, 0], Ed[0, 0]

    def law(x, u_prev):
        return lq_first_move(
            Ad,
            Bd,
            Ed,
            np.eye(1),
            x0=[x],
            u_prev=[u_prev],
            r=[r],
            d=[d_wrong],
            Q=np.ones(1),
            R=[0.1],
            S=[0.0],
            u_ref=[0.0],
            N=N,
        )[0]

    c = law(0.0, 0.0)
    a, b = law(1.0, 0.0) - c, law(0.0, 1.0) - c
    x_eq, _ = np.linalg.solve([[1 - A, -B], [-a, 1 - b]], [E * d_true, c])
    predicted_offset = x_eq - r

    def final_offset(known):
        res = run_closed_loop(
            model,
            NMPC(model, cfg),
            [0.0],
            [0.0],
            dt=dt,
            n_steps=150,
            setpoint=r,
            disturbance=d_true,
            known_disturbance=known,
            full_state_feedback=True,
        )
        return res.y_true[-1, 0] - r

    wrong, right = final_offset(d_wrong), final_offset(None)
    print(
        f"offset with wrong d_hat {wrong:.6f} (predicted {predicted_offset:.6f}), "
        f"true d_hat {right:.1e}"
    )
    assert abs(predicted_offset) > 0.1
    assert wrong == pytest.approx(predicted_offset, abs=1e-6)
    assert abs(right) < 1e-7


# --------------------------------------------------------------------------- nonlinear receiver


def test_zero_offset_with_perfect_model_and_known_disturbance(receiver_case):
    plant, m0, x0 = receiver_case
    mpc = NMPC(plant, receiver_config(m0))
    res = run_receiver(plant, mpc, x0, m0, full_state_feedback=True)
    assert all(rec.success for rec in mpc.solve_log)
    assert abs(res.y_true[-1, 0] - T_SET) < 1e-6
    assert res.u[-1, 0] == pytest.approx(plant.flow_for(T_SET, Q1), abs=1e-6)


def test_input_and_rate_bounds_hold_at_every_step(receiver_case):
    """Large setpoint moves drive the flow into its upper bound and its rate limit."""
    plant, m0, x0 = receiver_case
    u_max = 4.5
    assert plant.flow_for(973.15, Q0) > u_max  # the 973 K setpoint is unreachable
    mpc = NMPC(plant, receiver_config(m0, u_max=[u_max]))
    setpoint = piecewise_constant([20.0, 300.0], [T_SET, 973.15, 1123.15])
    res = run_receiver(plant, mpc, x0, m0, n_steps=110, setpoint=setpoint, full_state_feedback=True)
    u = res.u_cmd[:, 0]
    du = np.abs(np.diff(np.concatenate([[m0], u])))
    tol = 1e-7
    print(
        f"max bound violation {max(u.max() - u_max, U_MIN - u.min()):.1e} kg/s, "
        f"max rate violation {(du - RATE * DT).max():.1e} kg/s per sample"
    )
    assert U_MIN - tol <= u.min() and u.max() <= u_max + tol
    assert du.max() <= RATE * DT + tol
    # The constraints were genuinely active, so the check is not vacuous.
    assert u.max() == pytest.approx(u_max, abs=1e-6)
    assert du.max() == pytest.approx(RATE * DT, abs=1e-6)
    assert all(rec.success for rec in mpc.solve_log)


# IAE-optimal PID gains for exactly this disturbance (Nelder-Mead on log-gains from
# kp=0.2, ki=0.0066, kd=0.1; the local-optimality check below guards them).
PID_GAINS = (0.258, 0.01367, 0.682)  # kg/s/K, kg/s/(K s), kg/K


def test_nmpc_rejects_solar_step_better_than_tuned_pid(receiver_case):
    """20 % solar step on the rate-limited receiver: IAE of NMPC < IAE of the best PID.

    The NMPC sees the true state and the measured solar power; the PID is
    feedback-only (as at Sandia). A PID with static feedforward is reported too.
    """
    plant, m0, x0 = receiver_case

    def pid_iae(kp, ki, kd, feedforward=None):
        pid = PIDController(
            kp,
            ki,
            kd,
            DT,
            action="direct",
            bias=m0,
            u_min=U_MIN,
            u_max=U_MAX,
            rate_max=RATE,
            feedforward=feedforward,
        )
        res = run_receiver(plant, pid, x0, m0, actuator_rate_max=RATE)
        return integral_absolute_error(res.t, res.y_meas[:, 0], T_SET, T_STEP)

    iae_pid = pid_iae(*PID_GAINS)
    for i in range(3):  # "reasonably tuned": no +/-25 % change of one gain helps
        for factor in (0.75, 1.25):
            gains = list(PID_GAINS)
            gains[i] *= factor
            assert pid_iae(*gains) > 0.99 * iae_pid

    slope = (plant.flow_for(T_SET, Q1) - m0) / (Q1 - Q0)
    iae_pid_ff = pid_iae(*PID_GAINS, feedforward=lambda d: slope * (d[0] - Q0))

    mpc = NMPC(plant, receiver_config(m0))
    res = run_receiver(plant, mpc, x0, m0, actuator_rate_max=RATE, full_state_feedback=True)
    iae_mpc = integral_absolute_error(res.t, res.y_meas[:, 0], T_SET, T_STEP)
    solve = res.diagnostics["solve_time"]
    print(
        f"IAE [K s]: NMPC {iae_mpc:.1f}, PID {iae_pid:.1f}, "
        f"PID + static feedforward {iae_pid_ff:.1f}"
    )
    print(
        f"NMPC solve time (nx=4, N=24): mean {1e3 * solve.mean():.2f} ms, "
        f"worst {1e3 * solve.max():.2f} ms, "
        f"max iterations {res.diagnostics['iterations'].max():.0f}"
    )
    assert iae_mpc < iae_pid


def test_scaling_does_not_change_the_solution(receiver_case):
    plant, m0, x0 = receiver_case
    x_off = x0 + np.array([20.0, 10.0, -15.0, -15.0])
    u = {}
    for name, cfg in {
        "scaled": receiver_config(m0),
        "unscaled": receiver_config(m0, x_scale=None, x_offset=None, u_scale=None),
    }.items():
        mpc = NMPC(plant, cfg)
        u[name] = mpc.step(0.0, x_off[3:], [T_SET], x_hat=x_off, d_hat=[Q1])
        print(f"{name}: {mpc.solve_log[-1].iterations} IPOPT iterations, u0 = {u[name][0]:.8f}")
        assert mpc.solve_log[-1].success
    np.testing.assert_allclose(u["scaled"], u["unscaled"], atol=1e-6)


def test_soft_output_bound_trades_off_against_tracking(first_order):
    """Setpoint 2 above a soft bound 1: the steady output minimises the stage cost.

    At steady state ``Q (y - r)^2 + w2 (y - y_max)^2 + w1 (y - y_max)`` is minimised
    at ``y* = (2 Q r + 2 w2 y_max - w1) / (2 (Q + w2))`` (= 1.05 here).
    """
    model = first_order(2.0, 10.0)
    Q, w2, w1, r, y_max = 1.0, 9.0, 1.0, 2.0, 1.0
    cfg = NMPCConfig(
        dt=1.0,
        horizon=10,
        tracked_outputs=[0],
        output_weight=[Q],
        move_weight=[0.1],
        u_min=[-10.0],
        u_max=[10.0],
        u_init=[0.0],
        y_soft_max=[y_max],
        soft_weight=[w2],
        soft_weight_linear=[w1],
    )
    res = run_closed_loop(
        model,
        NMPC(model, cfg),
        [0.0],
        [0.0],
        dt=1.0,
        n_steps=150,
        setpoint=r,
        disturbance=0.0,
        full_state_feedback=True,
    )
    y_star = (2 * Q * r + 2 * w2 * y_max - w1) / (2 * (Q + w2))
    assert res.y_true[-1, 0] == pytest.approx(y_star, abs=1e-6)


# --------------------------------------------------------------------------- robustness


def test_nlp_is_built_once_and_warm_started(receiver_case, monkeypatch):
    plant, m0, x0 = receiver_case
    calls = []
    original = ca.nlpsol
    monkeypatch.setattr(ca, "nlpsol", lambda *a, **k: calls.append(1) or original(*a, **k))
    mpc = NMPC(plant, receiver_config(m0))
    run_receiver(plant, mpc, x0, m0, n_steps=30, full_state_feedback=True)
    assert len(calls) == 1
    iterations = [rec.iterations for rec in mpc.solve_log]
    # Before the step the shifted warm start is already optimal: few iterations.
    assert max(iterations[1:10]) <= iterations[0]


def test_solver_failure_holds_previous_input(first_order):
    cfg = NMPCConfig(
        dt=1.0,
        horizon=10,
        tracked_outputs=[0],
        output_weight=[1.0],
        move_weight=[0.1],
        u_min=[-5.0],
        u_max=[5.0],
        u_init=[0.3],
        ipopt_options={"max_iter": 0},
    )
    mpc = NMPC(first_order(), cfg)
    u = mpc.step(0.0, [0.0], [1.0], x_hat=[0.0], d_hat=[0.0])
    rec = mpc.solve_log[-1]
    assert u.tolist() == [0.3] and not rec.success and rec.status == "Maximum_Iterations_Exceeded"
    assert mpc.diagnostics()["success"] == 0.0


def test_nan_in_model_holds_previous_input():
    class Broken(ControlModelBase):
        state_names, input_names, output_names = ("x",), ("u",), ("y",)

        def rhs(self, x, u, d):
            xp = ops_for(x, u, d)
            return xp.concat([xp.sqrt(x[0] - 10.0) + u[0]])  # NaN for x < 10

        def output(self, x, u, d):
            return ops_for(x, u, d).concat([x[0]])

    cfg = NMPCConfig(
        dt=1.0,
        horizon=5,
        tracked_outputs=[0],
        output_weight=[1.0],
        move_weight=[0.1],
        u_min=[-5.0],
        u_max=[5.0],
        u_init=[-0.7],
    )
    mpc = NMPC(Broken(), cfg)
    u = mpc.step(0.0, [0.0], [1.0], x_hat=[0.0])
    assert u.tolist() == [-0.7] and not mpc.solve_log[-1].success
