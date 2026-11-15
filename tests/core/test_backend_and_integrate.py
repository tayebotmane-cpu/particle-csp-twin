"""The NumPy and CasADi backends must give the same numbers for the same model."""

import casadi as ca
import numpy as np
import pytest

from pcsp.core import (
    CasadiOps,
    ControlModel,
    ControlModelBase,
    NumpyOps,
    casadi_discretize,
    casadi_functions,
    discretize,
    ops_for,
    rk4_step,
    simulate,
)


class CoolingTank(ControlModelBase):
    """Two lumps: a heated mass losing heat by radiation, and a lagging sensor."""

    state_names = ("T", "T_sensor")
    input_names = ("q",)
    disturbance_names = ("T_amb",)
    output_names = ("T_sensor",)

    def rhs(self, x, u, d):
        xp = ops_for(x, u, d)
        gains = xp.const([1.0e-9, 0.5])
        loss = gains[0] * (xp.maximum(x[0], d[0]) ** 4 - d[0] ** 4)
        return xp.concat([u[0] - loss, gains[1] * (x[0] - x[1])])

    def output(self, x, u, d):
        xp = ops_for(x, u, d)
        return xp.concat([x[1]])


@pytest.fixture
def model():
    return CoolingTank()


def test_model_satisfies_protocol_and_sizes(model):
    assert isinstance(model, ControlModel)
    assert (model.nx, model.nu, model.nd, model.ny) == (2, 1, 1, 1)


def test_ops_for_selects_backend():
    assert ops_for(np.zeros(2), 1.0) is NumpyOps
    assert ops_for(np.zeros(2), ca.SX.sym("u")) is CasadiOps


def test_numpy_and_casadi_rhs_agree(model):
    f, h = casadi_functions(model)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.uniform(300.0, 900.0, 2)
        u = rng.uniform(0.0, 50.0, 1)
        d = rng.uniform(280.0, 320.0, 1)
        np.testing.assert_allclose(np.array(f(x, u, d)).ravel(), model.rhs(x, u, d), rtol=1e-13)
        np.testing.assert_allclose(np.array(h(x, u, d)).ravel(), model.output(x, u, d), rtol=1e-13)


def test_numpy_and_casadi_discretisation_agree(model):
    step_np = discretize(model, dt=0.5, substeps=3)
    step_ca = casadi_discretize(model, dt=0.5, substeps=3)
    x, u, d = np.array([700.0, 650.0]), np.array([20.0]), np.array([300.0])
    np.testing.assert_allclose(np.array(step_ca(x, u, d)).ravel(), step_np(x, u, d), rtol=1e-13)


def test_rk4_is_fourth_order_on_linear_decay():
    def rhs(x, u, d):
        return -x

    x0 = np.array([1.0])
    errors = [
        abs(rk4_step(rhs, x0, None, None, dt=1.0, substeps=n)[0] - np.exp(-1.0)) for n in (16, 32)
    ]
    observed_order = np.log2(errors[0] / errors[1])
    assert observed_order == pytest.approx(4.0, abs=0.1)


def test_simulate_shapes_and_steady_state(model):
    n = 400
    u = np.full((n, 1), 10.0)
    d = np.full((n, 1), 300.0)
    X, Y = simulate(model, np.array([300.0, 300.0]), u, d, dt=1.0, substeps=2)
    assert X.shape == (n + 1, 2) and Y.shape == (n, 1)
    # Steady state: heating balances radiative loss, sensor reads the true value.
    T_ss = (10.0 / 1.0e-9 + 300.0**4) ** 0.25
    assert X[-1, 0] == pytest.approx(T_ss, rel=1e-6)
    assert X[-1, 1] == pytest.approx(T_ss, rel=1e-6)


def test_simulate_rejects_mismatched_inputs(model):
    with pytest.raises(ValueError, match="rows"):
        simulate(model, np.zeros(2), np.zeros((3, 1)), np.zeros((4, 1)), dt=1.0)
