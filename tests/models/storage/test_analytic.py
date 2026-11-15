"""The Anzelius-Schumann reference solution must itself satisfy the PDE it solves."""

import numpy as np
import pytest
from scipy.special import i0e

from pcsp.models.storage import j_function, schumann_step_response


def test_j_function_boundary_values():
    # Inlet (x = 0): J = 1. Fresh solid (y = 0): J = exp(-x).
    assert j_function(0.0, 7.0) == pytest.approx(1.0)
    assert j_function(3.0, 0.0) == pytest.approx(np.exp(-3.0), rel=1e-14)


@pytest.mark.parametrize(("x", "y"), [(3.0, 5.0), (50.0, 40.0), (200.0, 220.0), (1000.0, 900.0)])
def test_j_function_symmetry_identity(x, y):
    # J(x, y) + J(y, x) = 1 + exp(-x - y) I0(2 sqrt(xy)), a classical identity.
    rhs = 1.0 + np.exp(-((np.sqrt(x) - np.sqrt(y)) ** 2)) * i0e(2.0 * np.sqrt(x * y))
    assert j_function(x, y) + j_function(y, x) == pytest.approx(rhs, abs=1e-13)


@pytest.mark.parametrize(("xi", "eta"), [(2.0, 1.0), (20.0, 15.0), (150.0, 160.0)])
def test_solution_satisfies_dimensionless_schumann_equations(xi, eta):
    # d(theta_f)/d(xi) = theta_s - theta_f and d(theta_s)/d(eta) = theta_f - theta_s.
    def theta_f(a, b):
        return j_function(a, b)

    def theta_s(a, b):
        return 1.0 - j_function(b, a)

    h = 1e-4
    dtf_dxi = (theta_f(xi + h, eta) - theta_f(xi - h, eta)) / (2 * h)
    dts_deta = (theta_s(xi, eta + h) - theta_s(xi, eta - h)) / (2 * h)
    gap = theta_s(xi, eta) - theta_f(xi, eta)
    assert dtf_dxi == pytest.approx(gap, abs=1e-9)
    assert dts_deta == pytest.approx(-gap, abs=1e-9)


def test_dimensional_wrapper_delay_and_limits():
    kw = dict(h_v=1e4, mass_flux=0.5, cp_f=1000.0, fluid_capacity=300.0, solid_capacity=1.5e6)
    # Residence time at z = 1 m is C_f z / (G c_f) = 0.6 s: nothing has arrived before it.
    tf, ts = schumann_step_response(np.array([1.0]), 0.5, **kw)
    assert tf[0] == 0.0 and ts[0] == 0.0
    # Just after the gas front passes, the gas carries the exp(-xi) jump, the solid is still cold.
    tf, ts = schumann_step_response(np.array([1.0]), 0.6 + 1e-9, **kw)
    assert tf[0] == pytest.approx(np.exp(-20.0), rel=1e-6)
    assert ts[0] == pytest.approx(0.0, abs=1e-12)
    # Long after, the bed is fully charged.
    tf, ts = schumann_step_response(np.array([0.1, 1.0]), 1e6, **kw)
    np.testing.assert_allclose([tf, ts], 1.0, atol=1e-12)
