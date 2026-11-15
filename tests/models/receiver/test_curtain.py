"""Steady curtain model: conservation, analytic limits, convergence and trends."""

import dataclasses
import json

import numpy as np
import pytest

from pcsp.constants import G
from pcsp.data.schema import get_params, validate
from pcsp.models.receiver.curtain import (
    FPRParams,
    aperture_fractions,
    outlet_temperature,
    release_velocity,
    simulate,
    temperature_from_enthalpy,
    two_flux_slab,
)

NOMINAL = FPRParams()


def eta(**changes) -> float:
    return float(simulate(dataclasses.replace(NOMINAL, **changes)).efficiency)


def T_out(**changes) -> float:
    return float(simulate(dataclasses.replace(NOMINAL, **changes)).T_out)


# --- energy conservation -------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 7, 50, 400])
def test_loss_breakdown_sums_to_incident_power(n):
    ds = simulate(dataclasses.replace(NOMINAL, n_segments=n))
    total = sum(float(ds[k]) for k in ("Q_absorbed", "Q_reflective", "Q_radiative"))
    total += float(ds.Q_advective) + float(ds.Q_wall)
    assert total == pytest.approx(NOMINAL.Q_in, rel=1e-12)
    assert abs(float(ds.energy_residual)) < 1e-10 * NOMINAL.Q_in


def test_absorbed_flux_equals_enthalpy_rise():
    ds = simulate(NOMINAL)
    area = NOMINAL.curtain_width * NOMINAL.fall_height / NOMINAL.n_segments
    absorbed = float(ds.q_absorbed.sum()) * area
    mat = NOMINAL.material
    enthalpy_rise = NOMINAL.mass_flow * (mat.enthalpy(float(ds.T_out)) - mat.enthalpy(NOMINAL.T_in))
    assert absorbed == pytest.approx(float(ds.Q_absorbed), rel=1e-12)
    assert enthalpy_rise == pytest.approx(float(ds.Q_absorbed), rel=1e-10)
    # The incident flux integrates to Q_in on any grid.
    assert float(ds.solar_flux.sum()) * area == pytest.approx(NOMINAL.Q_in, rel=1e-13)


def test_lossless_receiver_absorbs_everything():
    # No aperture escape, no advection, (numerically) perfect wall insulation: eta = 1.
    ds = simulate(
        dataclasses.replace(NOMINAL, aperture_escape_factor=0.0, h_adv=0.0, wall_conductivity=1e-30)
    )
    assert float(ds.efficiency) == pytest.approx(1.0, abs=1e-12)


def test_no_sun_particles_cool_by_exactly_their_losses():
    ds = simulate(dataclasses.replace(NOMINAL, Q_in=0.0))
    assert float(ds.T_out) < NOMINAL.T_in
    losses = float(ds.Q_radiative + ds.Q_advective + ds.Q_wall)
    assert float(ds.Q_reflective) == 0.0
    assert -float(ds.Q_absorbed) == pytest.approx(losses, rel=1e-12)


# --- kinematics and optics -----------------------------------------------------------------


def test_free_fall_kinematics_and_mass_conservation():
    ds = simulate(NOMINAL)
    v0 = release_velocity(NOMINAL, NOMINAL.mass_flow)
    y = ds.y.values
    np.testing.assert_allclose(ds.velocity.values, np.sqrt(v0**2 + 2 * G * y), rtol=1e-14)
    mat = NOMINAL.material
    flux = ds.solid_fraction * ds.thickness * ds.velocity * mat.rho_p * NOMINAL.curtain_width
    np.testing.assert_allclose(flux.values, NOMINAL.mass_flow, rtol=1e-12)
    # At the slot the curtain is a packed bed.
    assert float(ds.solid_fraction[0]) == pytest.approx(mat.packing_fraction, rel=1e-12)
    assert float(ds.residence_time) == pytest.approx(
        (np.sqrt(v0**2 + 2 * G * NOMINAL.fall_height) - v0) / G, rel=1e-12
    )


def test_two_flux_reduces_to_beer_law_without_scattering():
    tau = np.array([0.01, 0.3, 1.0, 4.0])
    R, T = two_flux_slab(tau, albedo=0.0, backscatter=0.5, path_factor=1.0)
    np.testing.assert_allclose(T, np.exp(-tau), rtol=1e-13)
    np.testing.assert_allclose(R, 0.0, atol=1e-15)
    _, T2 = two_flux_slab(tau, albedo=0.0, backscatter=0.5, path_factor=2.0)
    np.testing.assert_allclose(T2, np.exp(-2 * tau), rtol=1e-13)


def test_two_flux_opaque_limit_and_bounds():
    omega, b = 1.0 - 0.934, 0.5
    K_over_S = (1.0 - omega) / (b * omega)
    R_inf = 1.0 + K_over_S - np.sqrt(K_over_S**2 + 2.0 * K_over_S)
    R, T = two_flux_slab(np.array([200.0]), omega, b, 1.0)
    assert R[0] == pytest.approx(R_inf, rel=1e-12)
    assert T[0] < 1e-80
    tau = np.linspace(1e-6, 10.0, 200)
    R, T = two_flux_slab(tau, 0.3, 0.5, 2.0)
    assert np.all((R >= 0) & (T >= 0) & (R + T <= 1.0))
    assert np.all(np.diff(T) < 0) and np.all(np.diff(R) > 0)


def test_curtain_becomes_more_transparent_as_it_accelerates_and_thins():
    ds = simulate(NOMINAL)
    assert np.all(np.diff(ds.tau_solar.values) > 0)
    assert np.all(np.diff(ds.solid_fraction.values) < 0)
    # Kirchhoff: absorptance and emissivity of the curtain stay in (0, 1).
    for name in ("alpha_solar", "eps_curtain"):
        values = ds[name].values
        assert np.all((values > 0) & (values < 1))


def test_enthalpy_inverse_round_trip():
    T = np.linspace(300.0, 1400.0, 50)
    h = NOMINAL.material.enthalpy(T)
    np.testing.assert_allclose(temperature_from_enthalpy(NOMINAL.material, h), T, rtol=1e-13)


def test_aperture_fractions_integrate_to_one():
    for n in (1, 3, 18, 101):
        assert aperture_fractions(NOMINAL, n).sum() == pytest.approx(1.0, rel=1e-14)


# --- grid convergence ----------------------------------------------------------------------


def test_second_order_convergence_on_aperture_aligned_grids():
    # n = 18 * 2^k puts the aperture edges (0.625 m, 1.625 m) on segment edges.
    p = NOMINAL
    ref = outlet_temperature(p, 18 * 128, p.mass_flow, p.T_in, p.Q_in)
    errors = np.array(
        [abs(outlet_temperature(p, n, p.mass_flow, p.T_in, p.Q_in) - ref) for n in (18, 36, 72)]
    )
    orders = np.log2(errors[:-1] / errors[1:])
    np.testing.assert_allclose(orders, 2.0, atol=0.05)


def test_misaligned_grids_converge_to_the_same_answer():
    # Partially lit segments spoil the clean order but not the answer.
    p = NOMINAL
    ref = outlet_temperature(p, 18 * 128, p.mass_flow, p.T_in, p.Q_in)
    for n in (10, 25, 49, 101):
        assert abs(outlet_temperature(p, n, p.mass_flow, p.T_in, p.Q_in) - ref) < 0.05


# --- trends --------------------------------------------------------------------------------


def test_higher_mass_flow_cools_outlet_and_raises_efficiency_with_diminishing_returns():
    flows = [2.0, 4.0, 6.0, 8.0, 12.0, 16.0]
    temps = np.array([T_out(mass_flow=m) for m in flows])
    effs = np.array([eta(mass_flow=m) for m in flows])
    assert np.all(np.diff(temps) < 0)
    assert np.all(np.diff(effs) > 0)
    gains_per_kg = np.diff(effs) / np.diff(flows)
    assert np.all(np.diff(gains_per_kg) < 0)


def test_hotter_operation_lowers_efficiency():
    T_ins = [600.0, 700.0, 800.0, 900.0]
    temps = np.array([T_out(T_in=T) for T in T_ins])
    effs = np.array([eta(T_in=T) for T in T_ins])
    assert np.all(np.diff(temps) > 0)
    assert np.all(np.diff(effs) < 0)


def test_advective_loss_is_linear_in_h_adv_driving_force():
    ds = simulate(NOMINAL)
    area = NOMINAL.curtain_width * NOMINAL.fall_height / NOMINAL.n_segments
    T_seg = 0.5 * (ds.T_p.values[1:] + ds.T_p.values[:-1])
    approx = NOMINAL.h_adv * np.sum(T_seg - NOMINAL.T_amb) * area
    assert float(ds.Q_advective) == pytest.approx(approx, rel=1e-4)
    assert eta(h_adv=0.0) > eta() > eta(h_adv=80.0)


# --- dataset contract ----------------------------------------------------------------------


def test_dataset_follows_schema_and_records_params():
    ds = simulate(NOMINAL)
    validate(ds)
    assert ds.attrs["component"] == "receiver_fpr" and ds.attrs["fidelity"] == "reduced"
    params = get_params(ds)
    assert params["h_adv"] == NOMINAL.h_adv
    assert params["material"]["name"] == NOMINAL.material.name
    json.dumps(params)


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError, match="aperture"):
        FPRParams(aperture_top=2.0)
    with pytest.raises(ValueError, match="aperture_escape_factor"):
        FPRParams(aperture_escape_factor=1.5)
