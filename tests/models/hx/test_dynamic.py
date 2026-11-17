"""Transient moving-bed HX model: protocol, backends, steady state against the 2-D conjugate
solution, energy conservation, step responses and bypass mixing."""

import casadi as ca
import numpy as np
import pytest

from pcsp.core import ControlModel, casadi_functions
from pcsp.models.hx import albrecht_ho
from pcsp.models.hx.conjugate import solve_conjugate
from pcsp.models.hx.dynamic import (
    FANG2022_RESPONSE_TIMES,
    MovingBedHXDynamic,
    fang2021_heat_capacity,
    fang2022_co2_mass_flow,
    fang2022_params,
    simulate_response,
    steady_state,
)


@pytest.fixture(scope="module")
def coarse():
    """Fang et al. base case on a controller-sized grid."""
    return MovingBedHXDynamic(fang2022_params(), n_z=20, n_y=4)


@pytest.fixture(scope="module")
def coarse_steady(coarse):
    u, d = coarse.nominal_inputs()
    return steady_state(coarse, u, d)


def _random_point(model, rng, bypass=0.3):
    u0, d0 = model.nominal_inputs()
    x = rng.uniform(830.0, 1040.0, model.nx)
    u = np.array([u0[0] * rng.uniform(0.5, 1.5), bypass])
    d = np.array([rng.uniform(1000.0, 1100.0), rng.uniform(780.0, 860.0), d0[2]])
    return x, u, d


def test_protocol_and_layout(coarse):
    assert isinstance(coarse, ControlModel)
    assert (coarse.nx, coarse.nu, coarse.nd, coarse.ny) == (20 * 5, 2, 3, 2)
    assert coarse.input_names == ("m_dot_particle", "bypass_fraction")
    assert coarse.disturbance_names == ("T_particle_in", "T_co2_in", "m_dot_co2")
    assert coarse.output_names == ("T_co2_mixed", "T_particle_out")
    assert coarse.state_names[0] == "T_bed[0,0]" and coarse.state_names[20] == "T_bed[0,1]"
    assert coarse.state_names[-1] == "T_plate[19]"


def test_polynomial_fits_are_accurate(coarse):
    errors = coarse.fit_errors
    for name in ("k_eff", "r_wall", "cp_co2", "mu_co2", "k_co2"):
        assert errors[name] < 1e-4, name
    assert errors["h_co2_abs_over_cp"] < 0.01  # enthalpy error below 0.01 K equivalent


@pytest.mark.parametrize("co2_htc", ["gnielinski", "laminar", 600.0])
def test_numpy_and_casadi_agree(co2_htc):
    model = MovingBedHXDynamic(fang2022_params(), n_z=6, n_y=3, co2_htc=co2_htc)
    f, h = casadi_functions(model)
    rng = np.random.default_rng(3)
    for bypass in (0.0, 0.4, 0.99):  # 0.99 is clipped to bypass_max
        x, u, d = _random_point(model, rng, bypass)
        np.testing.assert_allclose(
            np.asarray(f(x, u, d)).ravel(), model.rhs(x, u, d), rtol=1e-11, atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(h(x, u, d)).ravel(), model.output(x, u, d), rtol=1e-12
        )
    assert isinstance(
        model.rhs(ca.SX.sym("x", model.nx), ca.SX.sym("u", 2), np.array([1050.0, 823.0, 0.08])),
        ca.SX,
    )


def test_semi_discrete_energy_identity(coarse):
    # d/dt (stored energy) = P_particle - P_co2 holds for any state, to round-off.
    rng = np.random.default_rng(7)
    p = coarse.params
    for _ in range(5):
        x, u, d = _random_point(coarse, rng)
        dx = coarse.rhs(x, u, d)
        bed, plate = x[: coarse.n_y * coarse.n_z], coarse.plate(x)
        cell = p.particle.rho_bulk * p.width * coarse.dz * coarse.dy
        dE = 2.0 * (
            cell * np.sum(p.particle.cp(bed) * dx[: bed.size])
            + coarse.plate_heat_capacity * p.width * coarse.dz * np.sum(dx[bed.size :])
        )
        P_p, P_c = coarse.energy_rates(x, u, d)
        assert dE == pytest.approx(P_p - P_c, rel=1e-10, abs=1e-9 * abs(P_c))
        assert plate.size == coarse.n_z


def test_steady_state_converges_to_conjugate_solution():
    # Albrecht & Ho's design case. Same cross-channel grid (8 cells) in both models, so the
    # remaining difference is the first-order discretisation along the flow, which halves with
    # the cell size; Richardson extrapolation removes it.
    params = albrecht_ho.paper_params()
    ref = solve_conjugate(params, n_y=8)
    errors = []
    for n_z in (40, 80):
        model = MovingBedHXDynamic(params, n_z=n_z, n_y=8)
        ss = steady_state(model, *model.nominal_inputs())
        assert ss.residual < 1e-8
        errors.append(ss.y - [ref.T_co2_out, ref.T_particle_out])
    coarse_err, fine_err = errors
    assert np.all(np.abs(fine_err) < 1.2)  # K: -0.72 (sCO2), +1.00 (particles) at n_z = 80
    np.testing.assert_allclose(coarse_err / fine_err, 2.0, rtol=0.05)
    assert np.all(np.abs(2.0 * fine_err - coarse_err) < 0.1)  # K: 0.034 and 0.055


def test_steady_state_is_an_equilibrium_and_balances_energy(coarse, coarse_steady):
    ss = coarse_steady
    assert np.max(np.abs(coarse.rhs(ss.x, ss.u, ss.d))) < 1e-8
    P_p, P_c = coarse.energy_rates(ss.x, ss.u, ss.d)
    assert P_p == pytest.approx(P_c, rel=1e-9)
    assert ss.d[1] < ss.y[1] < ss.d[0] and ss.d[1] < ss.y[0] < ss.d[0]


def test_energy_is_conserved_through_a_transient(coarse, coarse_steady):
    u, d = coarse_steady.u, coarse_steady.d.copy()
    d[0] += 40.0
    u = u * [0.8, 1.0]
    traj = simulate_response(
        coarse, coarse_steady.x, u, d, np.linspace(0, 1500, 31), rtol=1e-9, atol=1e-9
    )
    stored = np.array([coarse.stored_energy(x) for x in traj.x])
    change = stored - stored[0]
    assert abs(change[-1]) > 1e5  # J: a real change of stored energy
    np.testing.assert_allclose(change, traj.energy_in, rtol=0, atol=1e-6 * abs(change[-1]))


@pytest.mark.parametrize(
    ("index", "delta", "sign"),
    [(0, 30.0, 1.0), (1, 30.0, 1.0), (2, 0.2, -1.0)],
    ids=["T_particle_in up", "T_co2_in up", "m_dot_co2 up 20 %"],
)
def test_single_input_steps_are_monotonic(coarse, coarse_steady, index, delta, sign):
    d = coarse_steady.d.copy()
    d[index] = d[index] * (1.0 + delta) if index == 2 else d[index] + delta
    t = np.linspace(0, 3000, 121)
    y = simulate_response(coarse, coarse_steady.x, coarse_steady.u, d, t).y
    final = steady_state(coarse, coarse_steady.u, d, x_guess=coarse_steady.x).y
    for k in range(2):
        steps = sign * np.diff(y[:, k])
        assert np.all(steps > -1e-6), coarse.output_names[k]
        assert sign * (final[k] - coarse_steady.y[k]) > 0.5
        assert y[-1, k] == pytest.approx(final[k], abs=0.05)


def test_bypass_mixing_is_an_enthalpy_balance(coarse, coarse_steady):
    x, u, d = coarse_steady.x, coarse_steady.u.copy(), coarse_steady.d
    table = coarse._co2_table
    for beta in (0.0, 0.25, 0.6):
        u[1] = beta
        T_hx = coarse.co2_hx_outlet(x, u, d)
        T_mix = coarse.output(x, u, d)[0]
        mixed = coarse.h_co2(T_mix)
        streams = (1.0 - beta) * coarse.h_co2(T_hx) + beta * coarse.h_co2(d[1])
        assert mixed == pytest.approx(streams, rel=0, abs=1e-6)  # J/kg, i.e. ~1e-9 K
        # against the CoolProp table: limited by the enthalpy fit (< 0.01 K)
        exact = (1.0 - beta) * table.h(T_hx) + beta * table.h(d[1])
        assert float(table.T_from_h(exact)) == pytest.approx(T_mix, abs=0.01)
        if beta == 0.0:
            assert T_mix == pytest.approx(T_hx, abs=1e-9)
    # the clip: bypass beyond bypass_max acts as bypass_max
    u[1] = 2.0
    y_big = coarse.output(x, u, d)
    u[1] = coarse.bypass_max
    np.testing.assert_allclose(y_big, coarse.output(x, u, d), rtol=1e-14)


def test_particle_outlet_is_a_mixing_cup_temperature(coarse, coarse_steady):
    x = coarse_steady.x.copy()
    last = [j * coarse.n_z + coarse.n_z - 1 for j in range(coarse.n_y)]
    x[last] = np.linspace(840.0, 900.0, coarse.n_y)
    T_out = coarse.output(x, coarse_steady.u, coarse_steady.d)[1]
    part = coarse.params.particle
    assert part.enthalpy(T_out) == pytest.approx(np.mean(part.enthalpy(x[last])), abs=1e-6)


def test_fang_reference_data():
    assert FANG2022_RESPONSE_TIMES["T_particle_in"] == {"T_co2_out": 338.0, "T_particle_out": 969.0}
    params = fang2022_params()
    # Bed properties printed in Fang et al. (2021): conductivity eq. 3, heat capacity eq. 2
    # (1082.6 J/kg/K at 823 K by hand), the latter as a power law within 1 % over 750-1150 K.
    assert params.closure.name == "bauxite_fit_thin_film"
    assert fang2021_heat_capacity(823.0) == pytest.approx(1082.6, abs=0.1)
    T = np.linspace(750.0, 1150.0, 41)
    np.testing.assert_allclose(params.particle.cp(T), fang2021_heat_capacity(T), rtol=0.01)
    assert params.particle.rho_bulk == pytest.approx(2000.0)
    # Plate of Fernandez-Torrijos et al. (2018), Table 1: 8238 kg/m3 x 468 J/kg/K x 1 mm.
    model = MovingBedHXDynamic(params, n_z=4, n_y=2)
    assert model.plate_heat_capacity == pytest.approx(3855.4, rel=1e-4)
    assert params.m_dot_particle == pytest.approx(0.06)  # 2000 * 0.003 * 0.01 * 1 m
    assert fang2022_co2_mass_flow(0.5) == pytest.approx(params.m_dot_co2)
    assert params.m_dot_co2 == pytest.approx(0.0769, rel=2e-3)  # 153.8 kg/m3 * 0.5 m/s * 1 mm


def test_input_validation():
    params = fang2022_params()
    with pytest.raises(ValueError):
        MovingBedHXDynamic(params, n_z=1)
    with pytest.raises(ValueError):
        MovingBedHXDynamic(params, co2_htc="turbulent")
    with pytest.raises(ValueError):
        MovingBedHXDynamic(params, T_range=(900.0, 1200.0))
    model = MovingBedHXDynamic(params, n_z=4, n_y=2)
    with pytest.raises(ValueError):
        model.check_range(np.full(model.nx, 1500.0))
