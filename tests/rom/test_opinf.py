"""Operator inference: compact Kronecker algebra, FD derivatives, exact operator recovery."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pcsp.rom import (
    OpInfModel,
    compact_kron,
    compress_quadratic,
    expand_quadratic,
    fit_opinf,
    fit_pod,
    relative_error,
    time_derivative,
)

T = np.linspace(0.0, 5.0, 500)


def forcing(t: float) -> np.ndarray:
    return np.array([np.sin(2.0 * t)])


def test_compact_kron_has_each_product_once():
    q = np.array([2.0, 3.0, 5.0])
    np.testing.assert_array_equal(compact_kron(q), [4.0, 6.0, 9.0, 10.0, 15.0, 25.0])
    Q = np.random.default_rng(0).standard_normal((4, 7))
    np.testing.assert_allclose(compact_kron(Q)[:, 3], compact_kron(Q[:, 3]))


def test_expand_and_compress_are_consistent():
    rng = np.random.default_rng(1)
    r = 4
    Hc = rng.standard_normal((r, r * (r + 1) // 2))
    H = expand_quadratic(Hc)
    q = rng.standard_normal(r)
    np.testing.assert_allclose(H @ np.kron(q, q), Hc @ compact_kron(q), rtol=1e-13)
    np.testing.assert_allclose(compress_quadratic(H), Hc, rtol=1e-14)
    # The full operator is symmetric: H(q (x) p) = H(p (x) q).
    p = rng.standard_normal(r)
    np.testing.assert_allclose(H @ np.kron(q, p), H @ np.kron(p, q), rtol=1e-12)


@pytest.mark.parametrize("order", [2, 4, 6])
def test_time_derivative_converges_at_the_stated_order(order):
    # Non-uniform grid t = s + 0.1 sin(s) on purpose; error measured over all points
    # including the one-sided boundary stencils.
    errors = []
    for n in (100, 200):
        s = np.linspace(0.0, 2.0, n)
        t = s + 0.1 * np.sin(s)
        dq = time_derivative(np.vstack([np.sin(3 * t), np.exp(-t)]), t, order=order)
        errors.append(np.max(np.abs(dq - np.vstack([3 * np.cos(3 * t), -np.exp(-t)]))))
    assert np.log2(errors[0] / errors[1]) == pytest.approx(order, abs=0.35)


def lifted_data(true: OpInfModel, n_dof: int, n_traj: int, seed: int):
    """Trajectories of a reduced model embedded in R^n_dof: x = V q, dx/dt = V f(q, u)."""
    rng = np.random.default_rng(seed)
    V, _ = np.linalg.qr(rng.standard_normal((n_dof, true.rank)))
    X, dX, U = [], [], []
    for _ in range(n_traj):
        sol = solve_ivp(
            lambda t, q: true.rhs(q, forcing(t)),
            (T[0], T[-1]),
            0.5 * rng.standard_normal(true.rank),
            t_eval=T,
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
        )
        u = np.array([forcing(t) for t in T]).T
        X.append(V @ sol.y)
        dX.append(V @ true.rhs(sol.y, u))
        U.append(u)
    return np.hstack(X), np.hstack(dX), np.hstack(U), V


def quadratic_system(seed: int = 2) -> OpInfModel:
    """Stable forced quadratic system: damping + rotation, small quadratic coupling."""
    rng = np.random.default_rng(seed)
    r = 4
    skew = rng.standard_normal((r, r))
    return OpInfModel(
        c=0.1 * rng.standard_normal(r),
        A=-np.diag([1.0, 2.0, 3.0, 4.0]) + 0.5 * (skew - skew.T),
        H=0.3 * rng.standard_normal((r, r * (r + 1) // 2)),
        B=rng.standard_normal((r, 1)),
        rank=r,
    )


def rotate(true: OpInfModel, R: np.ndarray) -> dict[str, np.ndarray]:
    """Operators in coordinates q' = R q (R orthogonal): A' = R A R^T, H' = R H (R^T (x) R^T)."""
    H_full = R @ expand_quadratic(true.H) @ np.kron(R.T, R.T)
    return {
        "c": R @ true.c,
        "A": R @ true.A @ R.T,
        "H": compress_quadratic(H_full),
        "B": R @ true.B,
    }


def test_recovers_linear_system_operators_after_pod_projection():
    rng = np.random.default_rng(3)
    true = OpInfModel(
        c=None,
        A=-np.diag([0.5, 1.0, 2.0]) + np.triu(rng.standard_normal((3, 3)), 1),
        H=None,
        B=rng.standard_normal((3, 1)),
        rank=3,
    )
    X, dX, U, V = lifted_data(true, n_dof=150, n_traj=2, seed=4)
    pod = fit_pod(X, rank=3)
    model = fit_opinf(pod.project(X), pod.project(dX), U, terms="AB")
    R = pod.modes.T @ V
    assert np.linalg.norm(model.A - R @ true.A @ R.T) / np.linalg.norm(true.A) < 1e-12
    assert np.linalg.norm(model.B - R @ true.B) / np.linalg.norm(true.B) < 1e-12
    assert model.c is None and model.H is None


def test_recovers_quadratic_system_and_reproduces_trajectory():
    true = quadratic_system()
    X, dX, U, V = lifted_data(true, n_dof=200, n_traj=3, seed=5)
    pod = fit_pod(X, rank=4)
    Q = pod.project(X)
    model = fit_opinf(Q, pod.project(dX), U, terms="cAHB")
    expected = rotate(true, pod.modes.T @ V)
    for name in "cAHB":
        learned = getattr(model, name)
        # Measured: 6e-14 (c), 4e-15 (A), 1.3e-13 (H), 3e-14 (B); trajectory 8e-13.
        assert np.linalg.norm(learned - expected[name]) / np.linalg.norm(expected[name]) < 1e-11
    q_pred = model.predict(Q[:, 0], T, forcing, method="DOP853", rtol=1e-12, atol=1e-14)
    assert relative_error(Q[:, : T.size], q_pred) < 1e-9


def test_finite_difference_derivatives_recover_operators_at_their_order():
    true = quadratic_system()
    X, _, U, V = lifted_data(true, n_dof=200, n_traj=3, seed=5)
    pod = fit_pod(X, rank=4)
    Q = pod.project(X)
    expected = rotate(true, pod.modes.T @ V)
    errors = []
    for order in (2, 4, 6):
        dQ = np.hstack([time_derivative(q, T, order=order) for q in np.split(Q, 3, axis=1)])
        A = fit_opinf(Q, dQ, U, terms="cAHB").A
        errors.append(np.linalg.norm(A - expected["A"]) / np.linalg.norm(expected["A"]))
    # Measured: 8.6e-4, 6.2e-7, 5.7e-9 for h = 0.01.
    assert errors[1] < errors[0] / 50 and errors[2] < errors[1] / 50
    assert errors[2] < 1e-6


def test_regularisation_improves_learning_from_noisy_data():
    true = quadratic_system()
    X, _, U, V = lifted_data(true, n_dof=200, n_traj=3, seed=5)
    pod = fit_pod(X, rank=4)
    Q_clean = pod.project(X)
    H_true = rotate(true, pod.modes.T @ V)["H"]
    regs = (0.0, 0.1, 10.0)
    traj = np.empty((6, len(regs)))  # (noise realisation, regularisation)
    H_err = np.empty_like(traj)
    for seed in range(6):
        # 1 % white noise on the full states; derivatives by 2nd-order differences.
        noise = 1e-2 * np.abs(X).max() * np.random.default_rng(seed).standard_normal(X.shape)
        Q = pod.project(X + noise)
        dQ = np.hstack([time_derivative(q, T, order=2) for q in np.split(Q, 3, axis=1)])
        for j, reg in enumerate(regs):
            model = fit_opinf(Q, dQ, U, terms="cAHB", reg=reg)
            q_pred = model.predict(Q_clean[:, 0], T, forcing)
            traj[seed, j] = relative_error(Q_clean[:, : T.size], q_pred)
            H_err[seed, j] = np.linalg.norm(model.H - H_true) / np.linalg.norm(H_true)
    # Measured (means over seeds): H error 29 -> 2.0 -> 1.0; trajectory 2.6e-2 -> 1.8e-2
    # -> 0.78. Regularisation tames the poorly identifiable quadratic operator by an order
    # of magnitude, but the trajectory gain is modest (~1.5x) because those directions are
    # weakly excited; too much regularisation shrinks all operators towards zero.
    assert np.all(H_err[:, 1] < 0.2 * H_err[:, 0])
    assert traj[:, 1].mean() < 0.8 * traj[:, 0].mean()
    assert np.all(traj[:, 2] > 10 * traj[:, 1])


def test_invalid_term_choices_are_rejected():
    Q = np.random.default_rng(7).standard_normal((2, 20))
    with pytest.raises(ValueError, match="terms"):
        fit_opinf(Q, Q, terms="AX")
    with pytest.raises(ValueError, match="inputs U"):
        fit_opinf(Q, Q, terms="AB")
    with pytest.raises(ValueError, match="inputs U"):
        fit_opinf(Q, Q, np.ones((1, 20)), terms="A")
    with pytest.raises(ValueError, match="pass u"):
        fit_opinf(Q, Q, np.ones((1, 20)), terms="AB").rhs(Q[:, 0])
