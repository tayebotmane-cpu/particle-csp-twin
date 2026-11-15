"""Closures, property helpers, QoIs, dataset contract and cycle chaining."""

import json

import numpy as np
import pytest
from scipy.integrate import quad

from pcsp.data.schema import get_params, to_jsonable, validate
from pcsp.materials import air
from pcsp.materials.particles import CARBO_HSP_40_70
from pcsp.models import correlations as corr
from pcsp.models.storage import (
    PackedBedParams,
    Phase,
    gas_energy_density,
    gas_enthalpy,
    run_cycles,
    simulate,
    solid_enthalpy_density,
    solid_temperature,
    thermocline_thickness,
    volumetric_heat_transfer_coefficient,
)


def test_gas_enthalpy_and_energy_density_match_adaptive_quadrature(bed):
    T = np.array([300.0, 650.0, 1200.0, 1590.0])
    p = bed.with_(T_hot=1600.0)
    for Ti, h, e in zip(T, gas_enthalpy(p, T), gas_energy_density(p, T), strict=True):
        h_ref = quad(air.cp, p.T_cold, Ti, epsrel=1e-13)[0]
        e_ref = quad(lambda x: air.rho(x) * air.cp(x), p.T_cold, Ti, epsrel=1e-13)[0]
        assert h == pytest.approx(h_ref, rel=1e-13)
        assert e == pytest.approx(p.material.void_fraction * e_ref, rel=1e-12)


def test_solid_enthalpy_inverts_exactly_and_has_slope_cp(bed):
    p = bed.with_(material=CARBO_HSP_40_70)
    T = np.linspace(300.0, 1100.0, 9)
    H = solid_enthalpy_density(p, T)
    np.testing.assert_allclose(solid_temperature(p, H), T, rtol=1e-13)
    dH = (solid_enthalpy_density(p, T + 1e-3) - solid_enthalpy_density(p, T - 1e-3)) / 2e-3
    rho_s = (1 - p.material.void_fraction) * p.material.rho_p
    np.testing.assert_allclose(dH, rho_s * CARBO_HSP_40_70.cp(T), rtol=1e-8)


def test_h_v_is_wakao_kaguei_with_series_intraparticle_resistance(bed):
    T = 600.0
    mat = bed.material
    Re = bed.mass_flux * mat.d_p / air.mu(T)
    h = corr.nu_wakao_kaguei(Re, air.prandtl(T)) * air.k(T) / mat.d_p
    a_v = 6 * (1 - mat.void_fraction) / mat.d_p
    plain = volumetric_heat_transfer_coefficient(bed.with_(intraparticle_conduction=False), T)
    assert plain == pytest.approx(a_v * h, rel=1e-12)
    corrected = volumetric_heat_transfer_coefficient(bed, T)
    Bi = h * mat.d_p / (2 * mat.k_s)
    assert corrected == pytest.approx(a_v * h / (1 + Bi / 5), rel=1e-12)
    # A perfectly conducting particle has no internal resistance.
    big_k = bed.with_(material=mat.with_(k_s=1e12))
    assert volumetric_heat_transfer_coefficient(big_k, T) == pytest.approx(plain, rel=1e-9)


def test_thermocline_thickness_of_known_profiles():
    z = np.linspace(0.0, 2.0, 201)
    # Linear profile from 1 to 0: the 0.9 and 0.1 levels are 0.8 * 2 m apart.
    assert thermocline_thickness(z, 1.0 - z / 2.0) == pytest.approx(1.6, rel=1e-12)
    # A level that is not crossed inside the bed gives NaN.
    assert np.isnan(thermocline_thickness(z, np.ones_like(z)))


def test_pressure_drop_is_ergun_over_a_uniform_bed(bed):
    # Zero-time-change check: a bed already at T_hot, charged with T_hot gas.
    p = bed.with_(initial_T_s=bed.T_hot, duration=60.0, n_output=3)
    ds = simulate(p)
    rho = air.rho(p.T_hot)
    mat = p.material
    dp = p.length * corr.ergun_pressure_gradient(
        p.mass_flux / rho, mat.d_p, mat.void_fraction, rho, air.mu(p.T_hot)
    )
    np.testing.assert_allclose(ds.pressure_drop.values, dp, rtol=1e-12)
    np.testing.assert_allclose(ds.pumping_power.values, p.mass_flow * dp / rho, rtol=1e-12)
    assert float(ds.pumping_energy) == pytest.approx(p.mass_flow * dp / rho * 60.0, rel=1e-12)
    # Same mass flux in hotter gas: lower density, faster flow, larger drop.
    cold = simulate(bed.with_(duration=60.0, n_output=3, mode="discharge", initial_T_s=p.T_cold))
    assert float(ds.pressure_drop[-1]) > float(cold.pressure_drop[-1])


def test_capacity_and_efficiency_definitions(bed):
    p = bed.with_(property_mode="constant")
    ds = simulate(p)
    mat = p.material
    rho_f, cp_f = air.rho(p.T_prop), air.cp(p.T_prop)
    C = (1 - mat.void_fraction) * mat.rho_p * mat.cp_a + mat.void_fraction * rho_f * cp_f
    capacity = C * p.area * p.length * (p.T_hot - p.T_cold)
    assert float(ds.capacity) == pytest.approx(capacity, rel=1e-12)
    E_in = p.mass_flow * cp_f * (p.T_hot - p.T_cold) * p.duration
    assert float(ds.energy_in) == pytest.approx(E_in, rel=1e-9)
    dE = float(ds.energy_stored[-1] - ds.energy_stored[0])
    assert float(ds.phase_efficiency) == pytest.approx(dE / E_in, rel=1e-9)
    assert 0.0 < float(ds.phase_efficiency) <= 1.0


def test_dataset_follows_schema_and_records_params(bed):
    ds = simulate(bed)
    validate(ds)
    assert ds.attrs["component"] == "storage_bin" and ds.attrs["fidelity"] == "reduced"
    assert ds.T_f.dims == ("time", "z") and ds.T_s.shape == (bed.n_output, bed.n_cells)
    assert get_params(ds) == json.loads(json.dumps(to_jsonable(bed)))


def test_simulation_is_deterministic(bed):
    a, b = simulate(bed), simulate(bed)
    np.testing.assert_array_equal(a.T_s.values, b.T_s.values)
    np.testing.assert_array_equal(a.T_out.values, b.T_out.values)


def test_cycles_chain_states_and_conserve_energy(bed):
    phases = [Phase("charge", 1800.0, 0.4), Phase("discharge", 1800.0, 0.3)] * 2
    runs = run_cycles(bed.with_(U_wall=2.0), phases)
    assert [r.attrs["mode"] for r in runs] == ["charge", "discharge"] * 2
    for prev, nxt in zip(runs[:-1], runs[1:], strict=True):
        np.testing.assert_array_equal(nxt.T_s.values[0], prev.T_s.values[-1])
        assert float(nxt.energy_stored[0]) == pytest.approx(float(prev.energy_stored[-1]))
    # Energy over the whole sequence closes as well.
    dE = float(runs[-1].energy_stored[-1] - runs[0].energy_stored[0])
    net = sum(float(r.energy_in - r.energy_out - r.energy_loss) for r in runs)
    assert dE == pytest.approx(net, rel=1e-8, abs=1e-8 * float(runs[0].energy_in))
    # The recorded input of a later phase reproduces it exactly.
    again = simulate(PackedBedParams(**{**vars(bed.with_(U_wall=2.0)), **_phase_fields(runs[2])}))
    np.testing.assert_array_equal(again.T_s.values, runs[2].T_s.values)


def _phase_fields(ds):
    p = get_params(ds)
    keys = ("mode", "duration", "mass_flow", "initial_T_s", "initial_T_f")
    return {k: tuple(p[k]) if isinstance(p[k], list) else p[k] for k in keys}


@pytest.mark.parametrize(
    "change",
    [
        dict(T_hot=290.0),
        dict(mass_flow=-1.0),
        dict(mode="idle"),
        dict(initial_T_s=(300.0, 400.0)),
        dict(T_hot=2000.0),
    ],
)
def test_invalid_parameters_are_rejected(bed, change):
    with pytest.raises(ValueError):
        bed.with_(**change)
