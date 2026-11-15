"""Slug-flow Graetz series against an independent finite-volume solution and known limits."""

import numpy as np
import pytest

from pcsp.models.hx import graetz

XI_POINTS = (0.002, 0.02, 0.1, 0.5, 2.0)


def _laplacian(n, bc):
    """Cell-centred FV Laplacian on 0 <= eta <= 1; symmetry at 0, wall condition at 1."""
    h = 1.0 / n
    A = np.diag(np.full(n, -2.0)) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    A[0, 0] = -1.0  # zero flux on the symmetry plane
    # Wall: theta = 0 half a cell beyond the last centre ("T"), or a prescribed flux ("H").
    A[-1, -1] = -3.0 if bc == "T" else -1.0
    return h, A / h**2


def fv_nusselt(n, xi, bc):
    """Local and mean Nu of the semi-discrete FV system, integrated exactly in xi (eigenmodes)."""
    h, A = _laplacian(n, bc)
    lam, V = np.linalg.eigh(A)
    if bc == "T":
        c = V.T @ np.ones(n)  # theta(eta, 0) = 1
        theta = V @ (c * np.exp(lam * xi))
        dtheta = V @ (c * lam * np.exp(lam * xi))
        theta_m, dtheta_m = h * theta.sum(), h * dtheta.sum()
        return 4.0 * -dtheta_m / theta_m, -4.0 / xi * np.log(theta_m)
    # "H": d theta/d xi = A theta + s, unit flux entering the wall cell, theta(eta, 0) = 0.
    src = np.zeros(n)
    src[-1] = 1.0 / h
    c = V.T @ src
    zero = np.abs(lam) < 1e-9 * np.max(np.abs(lam))
    lz = np.where(zero, 1.0, lam)
    theta = V @ np.where(zero, c * xi, c * np.expm1(lam * xi) / lz)
    theta_int = V @ np.where(zero, c * xi**2 / 2.0, c * (np.expm1(lam * xi) / lz**2 - xi / lz))
    # Wall value from the last centre and the known unit gradient (second order).
    delta = theta[-1] + h / 2.0 - h * theta.sum()
    delta_int = theta_int[-1] + xi * h / 2.0 - h * theta_int.sum()
    return 4.0 / delta, 4.0 * xi / delta_int


@pytest.mark.parametrize("bc", ["T", "H"])
def test_series_matches_finite_volume_with_second_order_convergence(bc):
    for xi in XI_POINTS:
        series = np.array([graetz.nusselt_local(xi, bc)[0], graetz.nusselt_mean(xi, bc)[0]])
        err = {n: np.array(fv_nusselt(n, xi, bc)) / series - 1.0 for n in (200, 400)}
        # Measured: |error| <= 6e-4 at n = 400 down to Gz^-1 = 1.25e-4 (xi = 0.002).
        assert np.all(np.abs(err[400]) < 1e-3), (xi, err[400])
        order = np.log2(np.abs(err[200] / err[400]))
        np.testing.assert_allclose(order, 2.0, atol=0.05)
        # Richardson extrapolation removes the O(h^2) term. The remainder (measured <= 1.7e-6,
        # largest in the thin boundary layer at xi = 0.002) is the FV's next-order term.
        richardson = (4.0 * err[400] - err[200]) / 3.0
        assert np.all(np.abs(richardson) < 5e-6), (xi, richardson)


def test_fully_developed_values_are_the_derived_asymptotes():
    # Derived in the module docstring: Nu_T = 4 lambda_1^2 = pi^2, Nu_H = 4 / (1/3) = 12.
    assert graetz.NU_FULLY_DEVELOPED["T"] == pytest.approx(np.pi**2, rel=1e-15)
    assert graetz.nusselt_local(50.0, "T")[0] == pytest.approx(np.pi**2, rel=1e-13)
    assert graetz.nusselt_local(50.0, "H")[0] == pytest.approx(12.0, rel=1e-13)
    # Mean Nu approaches the asymptote like 1/xi (inlet excess integrates to a constant).
    for bc in ("T", "H"):
        nu_inf = graetz.NU_FULLY_DEVELOPED[bc]
        excess = [(graetz.nusselt_mean(x, bc)[0] - nu_inf) * x for x in (100.0, 1000.0)]
        assert excess[0] == pytest.approx(excess[1], rel=1e-3)


def test_finite_volume_slowest_mode_gives_fully_developed_nu():
    # Independent route to pi^2: the slowest FV decay rate is lambda_1^2 = pi^2/4.
    _, A = _laplacian(400, "T")
    slowest = -np.max(np.linalg.eigvalsh(A))
    assert 4.0 * slowest == pytest.approx(np.pi**2, rel=1e-5)


@pytest.mark.parametrize("bc", ["T", "H"])
def test_short_entry_branch_joins_the_series(bc):
    below = graetz.XI_SHORT * (1.0 - 1e-12)
    for fn in (graetz.nusselt_local, graetz.nusselt_mean):
        assert fn(below, bc)[0] == pytest.approx(fn(graetz.XI_SHORT, bc)[0], rel=1e-10)


@pytest.mark.parametrize("bc", ["T", "H"])
def test_entry_region_ordering(bc):
    # Up to xi = 1.5: beyond, the local value equals the asymptote to round-off.
    xi = np.geomspace(1e-4, 1.5, 60)
    local = graetz.nusselt_local(xi, bc)
    mean = graetz.nusselt_mean(xi, bc)
    assert np.all(np.diff(local) < 0.0) and np.all(np.diff(mean) < 0.0)
    assert np.all(mean > local)
    assert np.all(local > graetz.NU_FULLY_DEVELOPED[bc] * (1 - 1e-12))
    # Uniform flux always gives the higher coefficient (12 > pi^2 asymptotically).
    other = "T" if bc == "H" else "H"
    if bc == "H":
        assert np.all(local > graetz.nusselt_local(xi, other))


def test_constant_flux_matches_adapa_eq4_and_eq11():
    # Adapa et al. (2024), eq. 4 with R_nw = 0, written in their variable Gz = D_h Pe / z.
    gz = np.array([20.0, 100.0, 1000.0])
    n = np.arange(1, 400)[:, None]
    bracket = 1 / 12 - 0.5 * np.sum(np.exp(-16 * n**2 * np.pi**2 / gz) / (n**2 * np.pi**2), axis=0)
    np.testing.assert_allclose(graetz.nusselt_local(16.0 / gz, "H"), 1.0 / bracket, rtol=1e-12)
    # Their eq. 11 (fully developed, resistance in series): Nu = (1/12 + R_nw / (4 R_p))^-1.
    b, k_eff, r_nw = 3e-3, 0.3, 5e-4
    h_series = 1.0 / (4.0 * b / (graetz.nusselt_local(100.0, "H")[0] * k_eff) + r_nw)
    assert h_series * 4.0 * b / k_eff == pytest.approx(1.0 / (1 / 12 + r_nw / (4 * b / k_eff)))


@pytest.mark.parametrize("bc", ["T", "H"])
def test_cell_average_is_consistent_with_mean(bc):
    edges = np.concatenate([[0.0], np.geomspace(1e-4, 5.0, 40)])
    cells = graetz.nusselt_cell(edges[:-1], edges[1:], bc)
    widths = np.diff(edges)
    if bc == "T":  # arithmetic mean of h over [0, xi_end]
        combined = np.sum(cells * widths) / edges[-1]
    else:  # mean of the resistance
        combined = edges[-1] / np.sum(widths / cells)
    assert combined == pytest.approx(graetz.nusselt_mean(edges[-1], bc)[0], rel=1e-12)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        graetz.nusselt_local(0.0)
    with pytest.raises(ValueError):
        graetz.nusselt_mean(1.0, "X")
    with pytest.raises(ValueError):
        graetz.nusselt_cell(0.2, 0.1)
