"""Closed-form air properties must track the CoolProp reference they were fitted to."""

import casadi as ca
import numpy as np
import pytest
from CoolProp.CoolProp import PropsSI

from pcsp.constants import P_ATM
from pcsp.materials import air

T_GRID = np.linspace(air.T_MIN, air.T_MAX, 55)


@pytest.mark.parametrize(
    ("func", "coolprop_key", "tolerance"),
    [(air.cp, "C", 0.002), (air.mu, "V", 0.005), (air.k, "L", 0.005), (air.rho, "D", 0.002)],
)
def test_property_matches_coolprop(func, coolprop_key, tolerance):
    reference = np.array([PropsSI(coolprop_key, "T", T, "P", P_ATM, "Air") for T in T_GRID])
    np.testing.assert_allclose(func(T_GRID), reference, rtol=tolerance)


def test_prandtl_is_physical():
    Pr = air.prandtl(T_GRID)
    assert np.all((Pr > 0.68) & (Pr < 0.75))


def test_properties_accept_casadi_symbols():
    T = ca.SX.sym("T")
    f = ca.Function("f", [T], [air.cp(T), air.mu(T), air.k(T), air.rho(T)])
    values = [float(v) for v in f(900.0)]
    expected = [air.cp(900.0), air.mu(900.0), air.k(900.0), air.rho(900.0)]
    np.testing.assert_allclose(values, expected, rtol=1e-13)
