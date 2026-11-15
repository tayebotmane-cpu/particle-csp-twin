"""Verification of the packed-bed model: analytic solution, conservation, limits."""

import numpy as np
import pytest

from pcsp.materials.particles import CARBO_HSP_40_70
from pcsp.models.storage import (
    gas_properties,
    schumann_step_response,
    simulate,
    thermal_front_speed,
    volumetric_heat_transfer_coefficient,
)


def _theta(ds, name, params):
    return (ds[name].values - params.T_cold) / (params.T_hot - params.T_cold)


def _anzelius_errors(params):
    """L1 and max errors of theta_s, theta_f against the analytic solution at t_end."""
    ds = simulate(params)
    rho, cp, _, _ = gas_properties(params, params.T_prop)
    eps = params.material.void_fraction
    tf, ts = schumann_step_response(
        ds.z.values,
        params.duration,
        h_v=float(volumetric_heat_transfer_coefficient(params, params.T_prop)),
        mass_flux=params.mass_flux,
        cp_f=float(cp),
        fluid_capacity=float(eps * rho * cp),
        solid_capacity=(1 - eps) * params.material.rho_p * params.material.cp_a,
    )
    es = _theta(ds, "T_s", params)[-1] - ts
    ef = _theta(ds, "T_f", params)[-1] - tf
    return np.mean(np.abs(es)), np.max(np.abs(es)), np.mean(np.abs(ef))


def test_grid_convergence_to_anzelius_solution(bed):
    # Constant properties, Wakao-Kaguei h_v (bed NTU ~ 24); the front is mid-bed at t_end.
    # Tight time tolerance so the spatial error dominates.
    base = bed.with_(property_mode="constant", n_output=2, rtol=1e-9)
    cells = (25, 50, 100, 200)
    errs = np.array([_anzelius_errors(base.with_(n_cells=n)) for n in cells])
    order_l1 = np.log2(errs[:-1, 0] / errs[1:, 0])
    order_max = np.log2(errs[:-1, 1] / errs[1:, 1])
    # Measured: L1(theta_s) = 1.49e-3, 3.37e-4, 7.62e-5, 1.78e-5 (orders 2.14, 2.15, 2.10);
    # max(theta_s) = 3.08e-3 ... 3.84e-5 (orders 2.11, 2.13, 2.09).
    assert np.all(order_l1 > 1.9) and np.all(order_l1 < 2.2)
    assert np.all(order_max > 1.9)
    assert errs[-1, 0] < 2.5e-5  # L1, solid, 200 cells
    assert errs[-1, 1] < 5e-5  # max, solid, 200 cells
    assert errs[-1, 2] < 2.5e-5  # L1, gas, 200 cells


def _assert_closed(ds, rel):
    scale = max(float(ds.energy_in), float(ds.energy_out), abs(float(ds.energy_loss)))
    assert abs(float(ds.energy_balance_residual)) < rel * scale
    # The budget must also hold against stored energy recomputed from the fields.
    dE = float(ds.energy_stored[-1] - ds.energy_stored[0])
    rhs = float(ds.energy_in - ds.energy_out - ds.energy_loss)
    assert dE == pytest.approx(rhs, rel=rel, abs=rel * scale)


@pytest.mark.parametrize("mode", ["charge", "discharge"])
def test_energy_conservation_constant_properties(bed, mode):
    p = bed.with_(property_mode="constant", mode=mode, U_wall=5.0, solid_axial_conduction=True)
    ds = simulate(p)
    assert float(ds.energy_loss) > 0.0
    _assert_closed(ds, rel=1e-11)


@pytest.mark.parametrize("mode", ["charge", "discharge"])
def test_energy_conservation_variable_properties(bed, mode):
    # Power-law solid cp (sintered bauxite) exercises the nonlinear enthalpy inversion.
    p = bed.with_(
        material=CARBO_HSP_40_70.with_(d_p=0.005),
        mode=mode,
        U_wall=5.0,
        solid_axial_conduction=True,
    )
    ds = simulate(p)
    # Only the gas energy is a nonlinear function of the state, so the
    # residual is integration error on ~1e-4 of the energy: measured ~1e-9.
    _assert_closed(ds, rel=1e-7)


def test_large_h_v_collapses_to_single_front(bed):
    p = bed.with_(property_mode="constant", n_cells=200, n_output=11, duration=1000.0)
    u_th = thermal_front_speed(p)
    gaps = []
    for h_v in (1e6, 1e7):
        ds = simulate(p.with_(h_v=h_v))
        gaps.append(np.max(np.abs(_theta(ds, "T_f", p) - _theta(ds, "T_s", p))))
        theta = _theta(ds, "T_s", p)[-1]
        z = ds.z.values
        j = np.nonzero(theta <= 0.5)[0][0]
        z50 = z[j - 1] + (theta[j - 1] - 0.5) / (theta[j - 1] - theta[j]) * (z[j] - z[j - 1])
        assert z50 == pytest.approx(u_th * p.duration, abs=2 * p.length / p.n_cells)
    # The two phases merge as h_v grows (measured gap 2.6e-2 -> 3.3e-3).
    assert gaps[1] < 0.01
    assert gaps[1] < gaps[0] / 5.0


def test_zero_flow_leaves_bed_unchanged(bed):
    profile = tuple(np.linspace(bed.T_hot, bed.T_cold, bed.n_cells))
    ds = simulate(bed.with_(mass_flow=0.0, initial_T_s=profile, duration=36000.0))
    np.testing.assert_allclose(ds.T_s.values, np.broadcast_to(profile, ds.T_s.shape), atol=1e-9)
    np.testing.assert_allclose(ds.T_f.values, ds.T_s.values, atol=1e-9)
    assert float(ds.energy_in) == 0.0 and float(ds.energy_out) == 0.0
    assert np.all(ds.pressure_drop.values == 0.0)


def test_discharge_mirrors_charge_with_constant_properties(bed):
    p = bed.with_(property_mode="constant", rtol=1e-9)
    charge = simulate(p.with_(mode="charge"))
    discharge = simulate(p.with_(mode="discharge"))
    mirrored = p.T_hot + p.T_cold - charge.T_s.values[:, ::-1]
    np.testing.assert_allclose(discharge.T_s.values, mirrored, atol=1e-5)
    np.testing.assert_allclose(
        discharge.T_out.values, p.T_hot + p.T_cold - charge.T_out.values, atol=1e-5
    )


def test_long_charge_reaches_uniform_hot_bed(bed):
    ds = simulate(bed.with_(duration=6 * 3600.0))
    np.testing.assert_allclose(ds.T_s.values[-1], bed.T_hot, atol=1e-3)
    assert float(ds.T_out[-1]) == pytest.approx(bed.T_hot, abs=1e-3)
    assert float(ds.state_of_charge) == pytest.approx(1.0, abs=1e-6)
