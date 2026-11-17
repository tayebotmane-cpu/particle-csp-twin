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
from pcsp.models.hx.conjugate import solve_conjugate

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


# --- Comparison with Albrecht & Ho, J. Sol. Energy Eng. 141 (2019) 031006 ------------------


def test_design_lmtd_matches_paper():
    # Solar Energy 178 (2019) 331, section 4.4: LMTD = 41.61 degC for the same four temperatures.
    assert albrecht_ho.design_lmtd() == pytest.approx(41.61, abs=0.005)


def test_generic_operating_point_regression(nominal):
    # The generic point of albrecht_ho.nominal_params (not the paper's case): default
    # stagnant closure with radiation, CARBO properties, 20 MPa.
    assert float(nominal.U_mean) == pytest.approx(234.7, rel=2e-3)
    assert float(nominal.effectiveness) == pytest.approx(0.9675, abs=1e-3)
    assert float(nominal.particle_velocity) == pytest.approx(4.56e-3, rel=5e-3)


def _particle_wall_mean(ds) -> float:
    """Mean particle-wall coefficient including the near-wall resistance (their eq. 5)."""
    return float(np.mean(1.0 / (1.0 / ds.h_bed + ds.r_near_wall)))


@pytest.fixture(scope="module")
def paper_2d():
    return solve_conjugate(albrecht_ho.paper_params(), n_y=40)


def test_two_dimensional_reference_is_grid_converged(paper_2d):
    coarse = solve_conjugate(albrecht_ho.paper_params(), n_y=20)
    assert coarse.U_mean == pytest.approx(paper_2d.U_mean, rel=2e-3)
    assert coarse.duty == pytest.approx(paper_2d.duty, rel=1e-4)
    # Energy balance of the two streams (sCO2 by its real-gas enthalpy).
    p = albrecht_ho.paper_params()
    co2 = sco2.isobar(p.p_co2, 300.0, 1100.0)
    q_co2 = p.m_dot_co2 * float(co2.h(paper_2d.T_co2_out) - co2.h(p.T_co2_in))
    assert q_co2 == pytest.approx(paper_2d.duty, rel=2e-4)


def test_developed_bed_nusselt_number_is_close_to_uniform_flux(paper_2d):
    # Past the thermal entry the wall heat flux falls slowly along the plate, so the bed
    # Nusselt number of the converged 2-D solution is 11.5: near the uniform-flux value 12,
    # not the uniform-temperature value pi^2 = 9.87. Measured: 11.50 to 11.56 over 0.2-1 m.
    developed = paper_2d.nu_bulk[paper_2d.x >= 0.2]
    assert developed.min() > 11.3 and developed.max() < 11.8


def test_reduced_model_agrees_with_two_dimensional_reference(paper_2d):
    # Same inputs, closure and mass flows. Measured (reduced "H" / 2-D): U 164.5 / 160.0,
    # particle-wall coefficient 217.6 / 210.5 W/m2/K, duty 6.006 / 5.983 kW.
    ds = simulate(albrecht_ho.paper_params())
    assert float(ds.duty) == pytest.approx(paper_2d.duty, rel=5e-3)
    assert float(ds.U_mean) == pytest.approx(paper_2d.U_mean, rel=0.04)
    assert _particle_wall_mean(ds) == pytest.approx(paper_2d.h_particle_wall_mean, rel=0.045)
    assert float(ds.T_particle_out) == pytest.approx(paper_2d.T_particle_out, abs=1.0)
    assert float(ds.T_co2_out) == pytest.approx(paper_2d.T_co2_out, abs=1.0)
    # The uniform-temperature wall condition is further from the reference, and below it.
    low = simulate(albrecht_ho.paper_params(wall_bc="T"))
    assert float(low.U_mean) < 0.93 * paper_2d.U_mean


def test_overall_coefficient_against_paper(paper_2d):
    # The paper reports 144 W/m2/K. With its printed inputs: reduced model 164.5 (+14 %),
    # converged 2-D solution 160.0 (+11 %). The difference from the paper is not explained
    # by anything printed in it; see the next test for what does reproduce its tables.
    ds = simulate(albrecht_ho.paper_params())
    assert float(ds.U_mean) == pytest.approx(164.5, rel=5e-3)
    assert float(ds.U_mean) == pytest.approx(albrecht_ho.U_REPORTED, rel=0.2)
    assert paper_2d.U_mean == pytest.approx(albrecht_ho.U_REPORTED, rel=0.15)


@pytest.mark.parametrize("case", ["nominal", "d_p_50um", "gap_3mm"])
def test_uniform_temperature_wall_condition_reproduces_paper_tables(case):
    # With the uniform-temperature Nusselt number the reduced model follows the paper's
    # Tables 2 and 3. Measured errors (U, particle-wall coefficient, duty): nominal +1.7,
    # +1.4, +3.4 %; 50 um particles +3.5, +2.5, +5.3 %; 3 mm plate spacing +1.4, +0.6, +3.3 %.
    ref = albrecht_ho.PAPER_CASES[case]
    params = albrecht_ho.at_design_temperatures(albrecht_ho.paper_params(case, wall_bc="T"))
    ds = simulate(params)
    assert float(ds.T_particle_out) == pytest.approx(albrecht_ho.T_PARTICLE_OUT, abs=1e-4)
    assert float(ds.T_co2_out) == pytest.approx(albrecht_ho.T_CO2_OUT, abs=1e-4)
    assert float(ds.U_area_mean) == pytest.approx(ref.U, rel=0.04)
    assert _particle_wall_mean(ds) == pytest.approx(ref.h_particle_wall, rel=0.03)
    assert float(ds.duty) == pytest.approx(ref.duty, rel=0.06)
    assert params.m_dot_particle == pytest.approx(ref.m_dot_particle, rel=0.06)
    assert params.m_dot_co2 == pytest.approx(ref.m_dot_co2, rel=0.06)


@pytest.mark.parametrize("case", ["d_p_750um", "gap_9mm"])
def test_two_paper_cases_have_laminar_sco2_flow_and_are_not_evaluated(case):
    # At the paper's own mass flows the sCO2 Reynolds number is 2110-2500 in these two
    # cases, below the range of the turbulent correlation, so the model refuses them.
    with pytest.raises(ValueError, match="Gnielinski"):
        simulate(albrecht_ho.paper_params(case))


def test_non_particle_resistance_agrees_with_paper():
    # Their overall (144) and particle-wall (182) coefficients imply a plate + sCO2
    # resistance of 1/144 - 1/182 = 1.45e-3 m2K/W. Reduced model, same case: 1.39e-3.
    ds = simulate(albrecht_ho.paper_params())
    ours = float(np.mean(1.0 / ds.U_local - 1.0 / ds.h_bed - ds.r_near_wall))
    theirs = 1.0 / albrecht_ho.U_REPORTED - 1.0 / albrecht_ho.H_PARTICLE_WALL_REPORTED
    assert ours == pytest.approx(theirs, rel=0.1)
