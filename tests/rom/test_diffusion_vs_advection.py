"""Diffusion (analytic heat equation) versus advection (travelling sharp front).

The finding for the thesis: POD/linear-subspace ROMs converge exponentially for
diffusion-dominated fields but slowly for advection-dominated fields with moving
fronts (slow decay of the Kolmogorov n-width; Ohlberger & Rave 2016, Proc. ALGORITMY).
"""

import numpy as np
import pytest

from pcsp.rom import fit_opinf, fit_pod, projection_error, rank_for_energy, relative_error

X_GRID = (np.arange(200) + 0.5) / 200


def front(t: np.ndarray, speed: float = 1.0, width: float = 0.01) -> np.ndarray:
    """Pure advection u_t + c u_x = 0 of a tanh front of width 0.01 from z = 0.2 to 0.8."""
    return 0.5 * (1.0 - np.tanh((X_GRID[:, None] - 0.2 - speed * t[None, :]) / width))


@pytest.fixture
def heat_data(heat_solution):
    t = np.linspace(0.002, 0.2, 200)
    return t, heat_solution(X_GRID, t), heat_solution(X_GRID, t, derivative=True)


def test_heat_rom_error_decays_rapidly_and_tracks_projection_error(heat_data):
    # ROM: POD (uncentred) + operator inference dq/dt = A q from projected exact derivatives.
    t, X, dX = heat_data
    full = fit_pod(X)
    proj_errors, rom_errors = [], []
    for r in range(1, 9):
        pod = full.truncate(r)
        Q = pod.project(X)
        model = fit_opinf(Q, pod.project(dX), terms="A")
        q_pred = model.predict(Q[:, 0], t, rtol=1e-11, atol=1e-14)
        proj_errors.append(projection_error(pod, X))
        rom_errors.append(relative_error(X, pod.reconstruct(q_pred)))
        if r == 6:
            # The slowest learned eigenvalue is that of the PDE, -alpha pi^2.
            slowest = np.max(np.linalg.eigvals(model.A).real)
            assert slowest == pytest.approx(-(np.pi**2), rel=1e-8)
    proj, rom = np.array(proj_errors), np.array(rom_errors)
    # Measured: projection 8.3e-2 (r=1) ... 7.2e-9 (r=8); ROM/projection 1.01-1.05 for
    # r <= 6, then 1.2 and 3.0 at r = 7, 8 where the ROM error reaches a ~2e-8 floor
    # (least-squares closure of the truncated modes and the time integrator).
    assert np.all(rom >= proj * (1 - 1e-9))
    assert np.all(rom[:6] < 1.1 * proj[:6])
    assert np.all(np.diff(np.log(rom)) < -1.0)  # at least a factor e per added mode
    assert rom[-1] < 1e-7


def test_advection_needs_many_times_more_modes_than_diffusion(heat_data):
    _, X_heat, _ = heat_data
    X_front = front(np.linspace(0.0, 0.6, 200))  # same grid, same number of snapshots
    s_heat = fit_pod(X_heat).singular_values
    s_front = fit_pod(X_front).singular_values
    # Relative projection error sqrt(1 - E_r) <= 1e-3  <=>  E_r >= 1 - 1e-6.
    r_heat = rank_for_energy(s_heat, 1.0 - 1e-6)
    r_front = rank_for_energy(s_front, 1.0 - 1e-6)
    # Measured: 4 modes for diffusion, 63 for the front (34 vs 3 at 1e-2).
    assert r_heat <= 5
    assert r_front >= 10 * r_heat
    # Singular-value decay: by mode 11 the heat spectrum has fallen by 10 orders of
    # magnitude, the front by barely more than one.
    assert s_heat[10] / s_heat[0] < 1e-9
    assert s_front[10] / s_front[0] > 1e-2
