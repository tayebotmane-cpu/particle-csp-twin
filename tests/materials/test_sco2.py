"""sCO2 isobar tables against direct CoolProp calls, including the pseudo-critical peak."""

import CoolProp.CoolProp as CP
import numpy as np
import pytest
from scipy.integrate import simpson

from pcsp.constants import R_UNIVERSAL
from pcsp.materials import sco2

PROPERTIES = ("cp", "rho", "mu", "k")


def _reference(p, T, key):
    return np.array([sco2.coolprop_state(p, Ti)[key] for Ti in T])


def test_constants_match_coolprop():
    assert CP.PropsSI("Tcrit", "CO2") == pytest.approx(sco2.T_CRITICAL, rel=1e-9)
    assert CP.PropsSI("pcrit", "CO2") == pytest.approx(sco2.P_CRITICAL, rel=1e-4)
    assert CP.PropsSI("M", "CO2") == pytest.approx(sco2.MOLAR_MASS, rel=1e-9)


@pytest.mark.parametrize("p", [8e6, 20e6])
def test_table_matches_coolprop_near_pseudocritical_peak(p):
    # 8 MPa has a sharp cp peak (~35 kJ/kg/K at ~308 K); 20 MPa a mild one (~349 K).
    table = sco2.isobar(p)
    T_pc = table.pseudocritical_temperature()
    T = np.random.default_rng(0).uniform(max(table.T_min, T_pc - 10.0), T_pc + 10.0, 150)
    h_ref = _reference(p, T, "h")
    assert np.max(np.abs(table.h(T) - h_ref)) < 1e-6 * np.mean(np.abs(h_ref))
    for key in PROPERTIES:
        rel_err = np.max(np.abs(getattr(table, key)(T) / _reference(p, T, key) - 1.0))
        assert rel_err < 1e-3, key


def test_pseudocritical_temperature_is_the_cp_maximum():
    table = sco2.isobar(8e6)
    T_pc = table.pseudocritical_temperature()
    cp_peak = sco2.coolprop_state(8e6, T_pc)["cp"]
    for dT in (-0.5, 0.5):
        assert sco2.coolprop_state(8e6, T_pc + dT)["cp"] < cp_peak
    assert sco2.T_CRITICAL < T_pc < 320.0


def test_table_matches_coolprop_in_heat_exchanger_range():
    p = 20e6
    table = sco2.isobar(p)
    T = np.random.default_rng(1).uniform(800.0, 1060.0, 100)
    h_ref = _reference(p, T, "h")
    assert np.max(np.abs(table.h(T) - h_ref)) < 1e-9 * np.mean(np.abs(h_ref))
    for key in PROPERTIES:
        assert np.max(np.abs(getattr(table, key)(T) / _reference(p, T, key) - 1.0)) < 1e-6, key


def test_inverse_enthalpy_round_trip():
    table = sco2.isobar(8e6)
    T = np.linspace(301.0, 1090.0, 997)
    np.testing.assert_allclose(table.T_from_h(table.h(T)), T, atol=1e-5)


def test_enthalpy_is_the_integral_of_cp():
    # Thermodynamic consistency of the table: h(T2) - h(T1) = int cp dT.
    table = sco2.isobar(8e6)
    T1, T2 = 300.0, 400.0
    T = np.linspace(T1, T2, 40001)
    integral = simpson(table.cp(T), x=T)
    assert table.h(T2) - table.h(T1) == pytest.approx(integral, rel=1e-4)


def test_ideal_gas_limit_at_low_pressure():
    table = sco2.SCO2Isobar(1e5, 600.0, 1000.0, n_nodes=50)
    T = np.array([600.0, 800.0, 1000.0])
    rho_ideal = 1e5 * sco2.MOLAR_MASS / (R_UNIVERSAL * T)
    np.testing.assert_allclose(table.rho(T), rho_ideal, rtol=1e-3)


def test_out_of_range_raises():
    table = sco2.isobar(20e6)
    with pytest.raises(ValueError, match="outside"):
        table.cp(1200.0)
    with pytest.raises(ValueError, match="outside"):
        table.T_from_h(table.h_max + 1.0)
