"""Particle-side closures and the sCO2 channel correlation: hand values and limits."""

import numpy as np
import pytest

from pcsp.constants import SIGMA_SB
from pcsp.data.schema import to_jsonable
from pcsp.materials import air
from pcsp.materials.particles import CARBO_HSP_40_70
from pcsp.models.correlations import k_eff_zehner_schlunder
from pcsp.models.hx import (
    AdapaFlowingBed,
    BedClosure,
    ConstantBed,
    StagnantBed,
    YagiKuniiBed,
    albrecht_ho,
)
from pcsp.models.hx.channel import friction_factor_petukhov, nusselt_gnielinski
from pcsp.models.hx.closures import k_radiation_breitbach_barthels

PARTICLE = CARBO_HSP_40_70.with_(d_p=280e-6, packing_fraction=0.55, emissivity=0.9)


def test_closures_satisfy_protocol_and_serialise():
    closures = (ConstantBed(0.3, 1e-4), StagnantBed(), AdapaFlowingBed("CP 40/100"), YagiKuniiBed())
    for closure in closures:
        assert isinstance(closure, BedClosure)
        assert to_jsonable(closure)["name"] == closure.name
    with pytest.raises(ValueError):
        StagnantBed(name="something_else")
    with pytest.raises(ValueError):
        AdapaFlowingBed("HSP 16/30")


def test_breitbach_barthels_black_high_conductivity_limit():
    # e_r = 1, Lambda_s -> inf, eps = 0.4: factor = (1 - sqrt(0.6)) 0.4 + sqrt(0.6) (B + 1)/B,
    # B = 1.25 * 1.5^(10/9) = 1.96141 -> 0.090161 + 0.774597 * 1.50984 = 1.25968.
    T, d = 900.0, 1e-3
    k = k_radiation_breitbach_barthels(T, d, 0.4, 1.0, 1e12)
    assert k / (4 * SIGMA_SB * T**3 * d) == pytest.approx(1.25968, rel=1e-5)


def test_breitbach_barthels_trends():
    T = np.array([600.0, 900.0, 1200.0])
    k = k_radiation_breitbach_barthels(T, 300e-6, 0.45, 0.9, 1e9)
    # With a highly conducting solid the radiative conductivity scales with T^3 and d_p.
    np.testing.assert_allclose(k / T**3, k[0] / T[0] ** 3, rtol=1e-9)
    k_large = k_radiation_breitbach_barthels(T, 600e-6, 0.45, 0.9, 1e9)
    np.testing.assert_allclose(k_large / k, 2.0, rtol=1e-9)
    # Lower emissivity or a poorly conducting solid reduce it.
    assert np.all(k_radiation_breitbach_barthels(T, 300e-6, 0.45, 0.5, 1e9) < k)
    assert np.all(k_radiation_breitbach_barthels(T, 300e-6, 0.45, 0.9, 0.5) < k)


def test_stagnant_closure_composition():
    T = np.array([800.0, 950.0, 1050.0])
    no_rad = StagnantBed(radiation=False).k_eff(T, 5e-3, PARTICLE)
    np.testing.assert_allclose(
        no_rad, k_eff_zehner_schlunder(air.k(T), PARTICLE.k_s, PARTICLE.void_fraction), rtol=1e-14
    )
    k = StagnantBed().k_eff(T, 5e-3, PARTICLE)
    assert np.all(k > no_rad) and np.all(np.diff(k) > 0)
    # Albrecht & Ho (2019): bed conductivities of 0.2-0.6 W/m/K for these particles.
    assert np.all((k > 0.2) & (k < 0.6))
    np.testing.assert_allclose(
        StagnantBed().r_wall(T, 5e-3, PARTICLE), 0.085 * 280e-6 / air.k(T), rtol=1e-14
    )


def test_adapa_table7_values_at_650C():
    # Table 7, HSP 40/70: k = 1.5e-4 * 650 + 0.23 = 0.3275; D_air = 2e-2 * 650 + 22 = 35 um.
    closure = AdapaFlowingBed("HSP 40/70")
    T = np.array([650.0 + 273.15])
    assert closure.k_eff(T, 0.01, PARTICLE)[0] == pytest.approx(0.3275, rel=1e-12)
    assert closure.gap_thickness(T)[0] == pytest.approx(35e-6, rel=1e-12)
    assert closure.r_wall(T, 0.01, PARTICLE)[0] == pytest.approx(35e-6 / air.k(T)[0], rel=1e-12)
    # Velocity independent (Adapa et al. found no trend over 5-15 mm/s).
    assert closure.k_eff(T, 0.005, PARTICLE) == closure.k_eff(T, 0.015, PARTICLE)


def test_flowing_bed_has_larger_wall_resistance_than_stagnant_default():
    # Adapa et al.: the gap grows on the onset of flow (about 0.05 d_p stagnant to 0.08-0.12 d_p).
    T = np.array([650.0 + 273.15])
    hsp = CARBO_HSP_40_70.with_(d_p=404e-6)
    flowing = AdapaFlowingBed("HSP 40/70").r_wall(T, 0.01, hsp)
    stagnant = StagnantBed(gas_film_factor=0.05).r_wall(T, 0.0, hsp)
    assert flowing > stagnant


def test_yagi_kunii_closure_gives_the_conductivity_implied_by_albrecht_ho():
    # They quote fully developed particle-wall coefficients of 270.7 (uniform flux, Nu = 12)
    # and 222.65 W/m2/K (uniform temperature, Nu = pi^2) at 700 degC for a 6 mm gap
    # (hydraulic diameter 12 mm). Both imply k_eff = 0.2707 W/m/K.
    k = float(YagiKuniiBed().k_eff(700.0 + 273.15, 0.0, albrecht_ho.paper_particle()))
    assert 12.0 * k / 0.012 == pytest.approx(270.7, rel=2e-3)
    assert np.pi**2 * k / 0.012 == pytest.approx(222.65, rel=2e-3)


def test_yagi_kunii_near_wall_layer_hand_value_and_scaling():
    closure, T = YagiKuniiBed(), np.array([945.15])
    particle = albrecht_ho.paper_particle()
    k_g = air.k(T)[0]
    # Their eq. 20, flat wall: eps_w = 1 - 0.7293 * 0.6 = 0.56242; eq. 19 with phi_w = 0.118.
    k_nw = k_g * (0.56242 + 0.43758 / (0.236 + (2.0 / 3.0) * k_g / 2.0))
    assert closure.k_near_wall(T, particle)[0] == pytest.approx(k_nw, rel=1e-5)
    # Eq. 18: R = d_p / (2 k_nw). 8.5e-4 m2K/W here; the paper's Tables 2 and 3 imply 8.6e-4.
    r = closure.r_wall(T, 4e-3, particle)[0]
    assert r == pytest.approx(250e-6 / (2.0 * k_nw), rel=1e-5)
    assert r == pytest.approx(8.6e-4, rel=0.02)
    # Proportional to particle diameter, independent of velocity; no radiation in k_eff.
    large = albrecht_ho.paper_particle(750e-6)
    assert closure.r_wall(T, 4e-3, large)[0] == pytest.approx(3.0 * r, rel=1e-12)
    assert closure.r_wall(T, 0.0, particle)[0] == r
    assert closure.k_eff(T, 0.0, large)[0] == closure.k_eff(T, 0.0, particle)[0]
    with pytest.raises(ValueError):
        YagiKuniiBed(phi=0.0)


def test_gnielinski_hand_value_and_validity():
    # Re = 1e4, Pr = 0.7: f = (0.790 ln 1e4 - 1.64)^-2 = 0.031480;
    # Nu = (f/8)(Re - 1000) Pr / (1 + 12.7 sqrt(f/8) (Pr^(2/3) - 1)) = 29.817.
    assert friction_factor_petukhov(1e4) == pytest.approx(0.031480, rel=1e-4)
    assert nusselt_gnielinski(1e4, 0.7) == pytest.approx(29.817, rel=1e-4)
    with pytest.raises(ValueError, match="Re"):
        nusselt_gnielinski(2000.0, 0.7)
    with pytest.raises(ValueError, match="Pr"):
        nusselt_gnielinski(1e4, 0.1)
