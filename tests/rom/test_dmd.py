"""Exact DMD and DMDc on linear systems with known operators."""

import numpy as np
import pytest
from scipy.linalg import block_diag, expm

from pcsp.rom import fit_dmd, fit_dmdc, relative_error


def real_matrix_with_eigenvalues(eigs: list[complex], rng: np.random.Generator) -> np.ndarray:
    """Real matrix S J S^-1 with J block diagonal: 1x1 for real eigs, 2x2 rotations for pairs."""
    blocks = []
    for lam in eigs:
        if lam.imag == 0:
            blocks.append(np.array([[lam.real]]))
        else:  # the conjugate is implied
            blocks.append(np.array([[lam.real, lam.imag], [-lam.imag, lam.real]]))
    J = block_diag(*blocks)
    S = rng.standard_normal(J.shape)
    return S @ J @ np.linalg.inv(S)


def lifted_trajectory(A, n_dof, n_steps, rng):
    """Trajectory of x_{k+1} = A x_k embedded in R^n_dof by an orthonormal lift."""
    lift, _ = np.linalg.qr(rng.standard_normal((n_dof, A.shape[0])))
    x = rng.standard_normal(A.shape[0])
    states = [x]
    for _ in range(n_steps):
        x = A @ x
        states.append(x)
    return lift @ np.array(states).T


def test_dmd_recovers_eigenvalues_of_a_known_linear_system():
    rng = np.random.default_rng(0)
    pairs = [0.9 * np.exp(0.3j), 0.7 * np.exp(1.1j)]
    A = real_matrix_with_eigenvalues([0.95 + 0j, 0.8 + 0j, *pairs], rng)
    expected = np.array([0.95, 0.8, *pairs, *np.conj(pairs)])
    X = lifted_trajectory(A, n_dof=300, n_steps=30, rng=rng)
    dmd = fit_dmd(X)
    assert dmd.rank == 6  # numerical rank detects the true state dimension
    np.testing.assert_allclose(
        np.sort_complex(dmd.eigenvalues), np.sort_complex(expected), atol=1e-12
    )
    # Exact DMD modes are eigenvectors of A_full = Y X^+: A_full Phi = Phi Lambda.
    A_full = X[:, 1:] @ np.linalg.pinv(X[:, :-1])
    np.testing.assert_allclose(A_full @ dmd.modes, dmd.modes * dmd.eigenvalues, atol=1e-10)
    assert relative_error(X, dmd.predict(X[:, 0], 30)) < 1e-12
    assert dmd.is_stable()


def test_continuous_eigenvalues_of_a_sampled_ode():
    rng = np.random.default_rng(1)
    omegas = np.array([-0.5, -2.0, -0.1 + 3.0j, -0.1 - 3.0j])
    Ac = real_matrix_with_eigenvalues([-0.5 + 0j, -2.0 + 0j, -0.1 + 3.0j], rng)
    dt = 0.05
    X = lifted_trajectory(expm(Ac * dt), n_dof=50, n_steps=80, rng=rng)
    dmd = fit_dmd(X, rank=4)
    np.testing.assert_allclose(
        np.sort_complex(dmd.continuous_eigenvalues(dt)), np.sort_complex(omegas), atol=1e-9
    )


def test_stability_check_flags_an_unstable_system():
    rng = np.random.default_rng(2)
    A = real_matrix_with_eigenvalues([1.05 + 0j, 0.5 + 0j, 0.9 * np.exp(0.5j)], rng)
    dmd = fit_dmd(lifted_trajectory(A, n_dof=40, n_steps=25, rng=rng))
    assert dmd.spectral_radius() == pytest.approx(1.05, abs=1e-10)
    assert not dmd.is_stable()


def test_rank_beyond_numerical_rank_is_rejected():
    rng = np.random.default_rng(3)
    A = real_matrix_with_eigenvalues([0.9 + 0j, 0.5 + 0j], rng)
    X = lifted_trajectory(A, n_dof=20, n_steps=10, rng=rng)
    with pytest.raises(ValueError, match="numerical rank"):
        fit_dmd(X, rank=3)


def simulate_controlled(A, B, U, x0):
    states = [x0]
    for k in range(U.shape[1]):
        states.append(A @ states[-1] + B @ U[:, k])
    return np.array(states).T


def test_dmdc_recovers_known_state_and_input_matrices():
    rng = np.random.default_rng(4)
    A = 0.9 * np.linalg.qr(rng.standard_normal((5, 5)))[0]
    B = rng.standard_normal((5, 2))
    U = rng.standard_normal((2, 40))
    X = simulate_controlled(A, B, U, rng.standard_normal(5))
    model = fit_dmdc(X, U)
    A_id, B_id = model.full_operators()
    assert np.linalg.norm(A_id - A) / np.linalg.norm(A) < 1e-12
    assert np.linalg.norm(B_id - B) / np.linalg.norm(B) < 1e-12
    np.testing.assert_allclose(
        np.sort_complex(model.eigenvalues), np.sort_complex(np.linalg.eigvals(A)), atol=1e-12
    )
    assert model.is_stable()
    # Multi-step prediction under a new input sequence.
    U_new = rng.standard_normal((2, 30))
    x0 = rng.standard_normal(5)
    truth = simulate_controlled(A, B, U_new, x0)
    assert relative_error(truth, model.predict(x0, U_new)) < 1e-12


def test_dmdc_on_lifted_system_with_truncation_and_instability_flag():
    rng = np.random.default_rng(5)
    A = real_matrix_with_eigenvalues([1.02 + 0j, 0.6 + 0j, 0.8 * np.exp(0.7j)], rng)
    B = rng.standard_normal((4, 1))
    U = rng.standard_normal((1, 60))
    lift, _ = np.linalg.qr(rng.standard_normal((100, 4)))
    X = lift @ simulate_controlled(A, B, U, rng.standard_normal(4))
    model = fit_dmdc(X, U, rank_input=5, rank_output=4)
    np.testing.assert_allclose(
        np.sort_complex(model.eigenvalues), np.sort_complex(np.linalg.eigvals(A)), atol=1e-10
    )
    assert not model.is_stable()
