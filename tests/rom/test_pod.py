"""POD: orthonormality, Eckart-Young optimality, exact low-rank recovery, quadrature weights."""

import numpy as np
import pytest

from pcsp.rom import fit_pod, projection_error, rank_for_energy


def low_rank(n_dof: int, n_snap: int, rank: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_dof, rank)) @ rng.standard_normal((rank, n_snap))


@pytest.mark.parametrize("weighted", [False, True])
def test_basis_is_orthonormal_in_the_chosen_inner_product(weighted):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((120, 40))
    w = rng.uniform(0.1, 3.0, 120) if weighted else None
    pod = fit_pod(X, rank=15, weights=w)
    W = np.diag(w) if weighted else np.eye(120)
    np.testing.assert_allclose(pod.modes.T @ W @ pod.modes, np.eye(15), atol=1e-13)


@pytest.mark.parametrize("center", [False, True])
def test_reconstruction_error_equals_discarded_energy(center):
    # Eckart-Young: ||W^1/2 (X - X_r)||_F^2 = sum_{i>r} s_i^2, in the weighted norm.
    rng = np.random.default_rng(2)
    X = rng.standard_normal((80, 30)) @ np.diag(0.7 ** np.arange(30)) + 3.0
    w = rng.uniform(0.5, 2.0, 80)
    full = fit_pod(X, center=center, weights=w)
    s = full.singular_values
    for r in (1, 3, 8, 20):
        pod = full.truncate(r)
        residual = X - pod.reconstruct(pod.project(X))
        err2 = np.sum(w[:, None] * residual**2)
        assert err2 == pytest.approx(np.sum(s[r:] ** 2), rel=1e-10)
        # ... which is also sqrt(1 - E_r) as a relative error of the fluctuations.
        fluct2 = np.sum(w[:, None] * (X - pod.mean[:, None]) ** 2)
        e_r = pod.cumulative_energy()[r - 1]
        assert np.sqrt(err2 / fluct2) == pytest.approx(np.sqrt(1.0 - e_r), rel=1e-8)


def test_truncated_svd_is_optimal_against_random_subspaces():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((60, 25)) @ np.diag(0.8 ** np.arange(25))
    pod = fit_pod(X, rank=5)
    best = projection_error(pod, X)
    for _ in range(20):
        Q, _ = np.linalg.qr(rng.standard_normal((60, 5)))
        other = np.linalg.norm(X - Q @ (Q.T @ X)) / np.linalg.norm(X)
        assert other > best


def test_exact_recovery_of_data_of_known_rank():
    X = low_rank(200, 50, rank=5)
    pod = fit_pod(X, energy=1.0 - 1e-12)
    assert pod.rank == 5
    assert pod.singular_values[5] < 1e-12 * pod.singular_values[0]
    assert projection_error(pod, X) < 1e-13


def test_centering_removes_the_mean_direction():
    # Affine data mean + rank-3 fluctuations: 3 centred modes, but 4 uncentred ones.
    rng = np.random.default_rng(4)
    F = low_rank(100, 40, rank=3)
    X = rng.standard_normal(100)[:, None] + (F - F.mean(axis=1, keepdims=True))
    assert fit_pod(X, center=True, energy=1.0 - 1e-12).rank == 3
    assert fit_pod(X, center=False, energy=1.0 - 1e-12).rank == 4
    pod = fit_pod(X, center=True, rank=3)
    np.testing.assert_allclose(pod.project(pod.mean), 0.0, atol=1e-12)
    assert projection_error(pod, X) < 1e-13


def test_weights_make_pod_grid_independent_on_a_nonuniform_grid(heat_solution):
    # Weighted singular values approximate those of the continuous L2 operator, so they
    # agree between a fine uniform grid and a strongly stretched coarse grid. Without
    # weights (even rescaled by 1/sqrt(n)) the clustered cells are over-counted.
    t = np.linspace(0.002, 0.2, 100)

    def grid(n, stretch):
        faces = np.linspace(0.0, 1.0, n + 1) ** stretch  # cells cluster near x = 0
        return 0.5 * (faces[1:] + faces[:-1]), np.diff(faces)

    x_ref, w_ref = grid(2000, 1.0)
    s_ref = fit_pod(heat_solution(x_ref, t), weights=w_ref).singular_values[:4]
    x, w = grid(300, 3.0)
    X = heat_solution(x, t)
    s_weighted = fit_pod(X, weights=w).singular_values[:4]
    s_unweighted = fit_pod(X).singular_values[:4] / np.sqrt(x.size)
    # Measured: weighted 2.5e-6 (midpoint-rule error), unweighted 9-23 %.
    np.testing.assert_allclose(s_weighted, s_ref, rtol=1e-5)
    assert np.min(np.abs(s_unweighted / s_ref - 1.0)) > 0.05


def test_rank_for_energy():
    s = np.array([3.0, 2.0, 1.0, 0.0])  # energies 9, 4, 1 -> cumulative 9/14, 13/14, 1
    assert rank_for_energy(s, 0.5) == 1
    assert rank_for_energy(s, 13 / 14) == 2
    assert rank_for_energy(s, 0.95) == 3
    assert rank_for_energy(s, 1.0) == 3


def test_vector_and_matrix_projection_agree():
    X = low_rank(50, 20, rank=6)
    pod = fit_pod(X, rank=4, center=True)
    np.testing.assert_allclose(pod.project(X[:, 3]), pod.project(X)[:, 3], rtol=1e-12)
    a = pod.project(X)
    np.testing.assert_allclose(pod.reconstruct(a[:, 3]), pod.reconstruct(a)[:, 3], rtol=1e-12)


def test_invalid_arguments_are_rejected():
    X = low_rank(30, 10, rank=3)
    with pytest.raises(ValueError, match="either"):
        fit_pod(X, rank=2, energy=0.9)
    with pytest.raises(ValueError, match="positive"):
        fit_pod(X, weights=-np.ones(30))
    with pytest.raises(ValueError, match="rank"):
        fit_pod(X, rank=11)
    with pytest.raises(ValueError, match="rows"):
        fit_pod(X, rank=2).project(np.ones(29))
