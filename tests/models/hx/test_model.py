"""Moving-bed HX model: Graetz recovery, energy closure, limits, schema, validation point."""

import dataclasses
import time

import numpy as np
import pytest

from pcsp.data.schema import get_params, to_jsonable, validate
from pcsp.materials import sco2
from pcsp.models.hx import (
    AdapaFlowingBed,
    ConstantBed,
    MovingBedHXParams,
    albrecht_ho,
    graetz,
    simulate,
)
from pcsp.models.hx.channel import friction_factor_petukhov

CONSTANT_CP_PARTICLE = albrecht_ho.PARTICLE.with_(cp_a=1200.0, cp_b=0.0)


@pytest.fixture(scope="module")
def nominal():
    return simulate(albrecht_ho.nominal_params())


def _params(**changes) -> MovingBedHXParams:
    return dataclasses.replace(albrecht_ho.nominal_params(), **changes)


def test_dataset_follows_schema_and_records_parameters(nominal):
    validate(nominal)
    assert nominal.attrs["component"] == "hx_moving_bed"
    assert nominal.attrs["fidelity"] == "reduced"
    params = albrecht_ho.nominal_params()
    assert get_params(nominal) == to_jsonable(params)
    assert get_params(nominal)["closure"]["name"] == "stagnant_zs_bb_ss"
    assert nominal.sizes == {"z": 201, "z_cell": 200}


@pytest.mark.parametrize("closure", [None, AdapaFlowingBed("HSP 40/70")], ids=["stagnant", "adapa"])
def test_energy_balance_closes_with_real_properties(closure):
    ds = simulate(albrecht_ho.nominal_params(closure))
    assert abs(float(ds.energy_residual)) < 1e-10
    assert float(ds.heat_flux.sum()) * float(ds.heat_transfer_area) / 200 == pytest.approx(
        float(ds.duty), rel=1e-12
    )
    # Second law and sensible bounds.
    assert np.all(ds.T_particle.values > ds.T_co2.values)
    assert np.all(np.diff(ds.T_particle.values) < 0) and np.all(np.diff(ds.T_co2.values) < 0)
    assert 0.0 < float(ds.effectiveness) < 1.0


@pytest.mark.parametrize("bc", ["H", "T"])
def test_zero_resistance_recovers_pure_graetz(bc):
    # Constant k_eff and cp: alpha uniform, xi linear in z. With R_w = 0 the bed coefficient
    # averaged over the plate must equal the analytic mean Nusselt number at xi(L).
    k_eff = 0.3
    p = _params(closure=ConstantBed(k_eff, 0.0), particle=CONSTANT_CP_PARTICLE, wall_bc=bc)
    ds = simulate(p)
    b = 0.5 * p.channel_gap
    alpha = k_eff / (p.particle.rho_bulk * 1200.0)
    xi_end = alpha * p.height / (p.particle_velocity * b**2)
    nu_mean = graetz.nusselt_mean(xi_end, bc)[0]
    h = ds.h_bed.values
    combined = np.mean(h) if bc == "T" else 1.0 / np.mean(1.0 / h)
    assert combined == pytest.approx(nu_mean * k_eff / (4 * b), rel=1e-12)
    assert float(ds.inverse_graetz[-1]) == pytest.approx(xi_end / 16.0, rel=1e-12)


def test_more_near_wall_resistance_lowers_U():
    U = [
        float(simulate(_params(closure=ConstantBed(0.3, r))).U_mean)
        for r in (0.0, 1e-4, 5e-4, 2e-3)
    ]
    assert np.all(np.diff(U) < 0)
    # Vanishing resistance converges to the zero-resistance result.
    U_tiny = float(simulate(_params(closure=ConstantBed(0.3, 1e-12))).U_mean)
    assert U_tiny == pytest.approx(U[0], rel=1e-8)


def test_thinner_channel_raises_U_and_effectiveness():
    results = [simulate(_params(channel_gap=gap)) for gap in (8e-3, 6e-3, 4e-3)]
    U = [float(r.U_mean) for r in results]
    eff = [float(r.effectiveness) for r in results]
    assert np.all(np.diff(U) > 0) and np.all(np.diff(eff) > 0)


def test_flux_condition_gives_higher_U_than_wall_temperature_condition(nominal):
    assert float(simulate(_params(wall_bc="T")).U_mean) < float(nominal.U_mean)


def test_isothermal_pressure_drop_matches_darcy_weisbach():
    # A nearly insulating bed (k_eff = 1e-9 W/m/K; the entry singularity still passes ~1 W)
    # keeps the sCO2 within 0.1 K of its inlet state, so the pressure drop is
    # f L G^2 / (2 rho d) with properties straight from CoolProp.
    p = _params(closure=ConstantBed(1e-9, 0.0))
    ds = simulate(p)
    assert float(ds.T_co2[0]) == pytest.approx(p.T_co2_in, abs=0.1)
    state = sco2.coolprop_state(p.p_co2, p.T_co2_in)
    d = p.co2_channel_diameter
    m_channel = p.m_dot_co2 / p.n_co2_channels
    G = m_channel / (np.pi * d**2 / 4)
    Re = 4 * m_channel / (np.pi * d * state["mu"])
    expected = friction_factor_petukhov(Re) * p.height / d * G**2 / (2 * state["rho"])
    assert float(ds.dp_co2) == pytest.approx(expected, rel=1e-4)


def test_laminar_sco2_flow_is_rejected():
    with pytest.raises(ValueError, match="Gnielinski"):
        simulate(_params(m_dot_co2=0.01))


def test_parameter_validation():
    with pytest.raises(ValueError):
        _params(channel_gap=-1.0)
    with pytest.raises(ValueError):
        _params(T_particle_in=500.0)
    with pytest.raises(TypeError):
        _params(closure="stagnant")


def test_steady_solve_is_fast():
    params = albrecht_ho.nominal_params()
    simulate(params)  # builds the cached sCO2 table
    start = time.perf_counter()
    for _ in range(5):
        simulate(params)
    assert (time.perf_counter() - start) / 5 < 0.2  # measured ~8 ms on the development laptop


# --- Validation against Albrecht & Ho (2017/2019), single-bank 2-D model -------------------


def test_design_lmtd_matches_paper():
    # Solar Energy 178 (2019) 331, section 4.4: LMTD = 41.61 degC for the same four temperatures.
    assert albrecht_ho.design_lmtd() == pytest.approx(41.61, abs=0.005)


def test_nominal_point_measured_values(nominal):
    # Regression record of the comparison reported to the project (not tuned to the paper):
    # U = 234.7 W/m2/K and effectiveness 0.968 with the default stagnant closure ("H"),
    # against 144 W/m2/K and 0.914 (the design temperatures) in Albrecht & Ho.
    assert float(nominal.U_mean) == pytest.approx(234.7, rel=2e-3)
    assert float(nominal.effectiveness) == pytest.approx(0.9675, abs=1e-3)
    assert float(nominal.particle_velocity) == pytest.approx(4.56e-3, rel=5e-3)


def test_non_particle_resistance_agrees_with_albrecht_ho(nominal):
    # Their overall (144) and particle-wall (182) coefficients imply metal + sCO2 resistance
    # 1/144 - 1/182 = 1.45e-3 m2K/W. Ours (same channel geometry assumptions): 1.40e-3.
    ours = float(np.mean(1.0 / nominal.U_local - 1.0 / nominal.h_bed - nominal.r_near_wall))
    theirs = 1.0 / albrecht_ho.U_REPORTED - 1.0 / albrecht_ho.H_PARTICLE_WALL_REPORTED
    assert ours == pytest.approx(theirs, rel=0.1)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known discrepancy: particle-wall coefficient 337 W/m2/K here vs 182 in Albrecht & Ho "
        "(2017). Their bed closure inputs for that model are not available; with the same "
        "group's 2019 ZBS inputs they report ~313 W/m2/K particle-side for a similar geometry."
    ),
)
def test_overall_coefficient_within_30_percent_of_albrecht_ho(nominal):
    assert float(nominal.U_mean) == pytest.approx(albrecht_ho.U_REPORTED, rel=0.3)
