"""Shared correlations: hand-computed values and known physical limits."""

import casadi as ca
import numpy as np
import pytest

from pcsp.models import correlations as corr


def test_ergun_matches_hand_calculation():
    # u=0.1 m/s, d=1 mm, eps=0.4, air-like fluid:
    # viscous 150*1.8e-5*0.36*0.1/(0.064*1e-6) = 1518.75 Pa/m
    # inertial 1.75*1.2*0.6*0.01/(0.064*1e-3) = 196.875 Pa/m
    dp_dz = corr.ergun_pressure_gradient(u_s=0.1, d_p=1e-3, eps=0.4, rho_f=1.2, mu_f=1.8e-5)
    assert dp_dz == pytest.approx(1715.625, rel=1e-12)


def test_ergun_limits():
    kwargs = dict(d_p=1e-3, eps=0.4, rho_f=1.2, mu_f=1.8e-5)
    # Creeping flow: linear in velocity (Blake-Kozeny). Fast flow: quadratic (Burke-Plummer).
    slow = corr.ergun_pressure_gradient(u_s=2e-6, **kwargs) / corr.ergun_pressure_gradient(
        u_s=1e-6, **kwargs
    )
    fast = corr.ergun_pressure_gradient(u_s=200.0, **kwargs) / corr.ergun_pressure_gradient(
        u_s=100.0, **kwargs
    )
    assert slow == pytest.approx(2.0, rel=1e-4)
    assert fast == pytest.approx(4.0, rel=1e-2)


def test_reynolds_particle():
    assert corr.reynolds_particle(0.1, 1e-3, 1.2, 1.8e-5) == pytest.approx(6.6666667, rel=1e-6)


def test_nusselt_conduction_limits():
    assert corr.nu_ranz_marshall(0.0, 0.7) == pytest.approx(2.0)
    assert corr.nu_wakao_kaguei(0.0, 0.7) == pytest.approx(2.0)
    # Gunn recovers the isolated-sphere limit at eps = 1.
    assert corr.nu_gunn(0.0, 0.7, 1.0) == pytest.approx(2.0)


def test_gunn_approaches_ranz_marshall_for_a_single_sphere():
    Re, Pr = 100.0, 0.7
    assert corr.nu_gunn(Re, Pr, 1.0) == pytest.approx(corr.nu_ranz_marshall(Re, Pr), rel=0.15)


def test_packed_bed_nusselt_exceeds_single_sphere():
    Re, Pr = 100.0, 0.7
    assert corr.nu_gunn(Re, Pr, 0.4) > corr.nu_ranz_marshall(Re, Pr)
    assert corr.nu_wakao_kaguei(Re, Pr) > corr.nu_ranz_marshall(Re, Pr)


def test_zehner_schlunder_limits_and_bounds():
    k_f, eps = 0.05, 0.4
    # Equal conductivities: the bed is homogeneous.
    assert corr.k_eff_zehner_schlunder(k_f, k_f * (1 + 1e-9), eps) == pytest.approx(k_f, rel=1e-6)
    # Gas-solid bed: between the series and parallel (Wiener) bounds, increasing in k_s.
    k_s = np.array([1.0, 2.0, 5.0, 20.0])
    k_eff = corr.k_eff_zehner_schlunder(k_f, k_s, eps)
    series = 1.0 / (eps / k_f + (1 - eps) / k_s)
    parallel = eps * k_f + (1 - eps) * k_s
    assert np.all(k_eff > series) and np.all(k_eff < parallel)
    assert np.all(np.diff(k_eff) > 0)


def test_zehner_schlunder_magnitude_for_bauxite_in_air():
    # Sintered bauxite (k_s ~ 2 W/m/K) in hot air (k_f ~ 0.06 W/m/K), eps ~ 0.4:
    # stagnant beds of this kind measure a few tenths of W/m/K.
    k_eff = corr.k_eff_zehner_schlunder(0.06, 2.0, 0.4)
    assert 0.2 < k_eff < 0.6


def test_correlations_accept_casadi_symbols():
    k_s = ca.SX.sym("k_s")
    f = ca.Function("f", [k_s], [corr.k_eff_zehner_schlunder(0.06, k_s, 0.4)])
    assert float(f(2.0)) == pytest.approx(corr.k_eff_zehner_schlunder(0.06, 2.0, 0.4))
