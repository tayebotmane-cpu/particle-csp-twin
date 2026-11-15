"""Control-oriented FPR model: backend agreement, equilibrium and step response."""

import dataclasses

import casadi as ca
import numpy as np
import pytest

from pcsp.core import ControlModel, casadi_discretize, casadi_functions, discretize, simulate
from pcsp.models.receiver import FPRControlModel, FPRParams, steady_state
from pcsp.models.receiver import simulate as simulate_curtain

Q_NOMINAL = 739e3  # W, pre-perturbation power of Sandia's PID test (Mills et al. 2022, Fig. 9b)
T_IN = 823.15
T_TARGET = 973.15


@pytest.fixture(scope="module")
def model():
    return FPRControlModel(n_segments=20)


@pytest.fixture(scope="module")
def nominal(model):
    return steady_state(model, Q_NOMINAL, T_IN, T_TARGET)


def test_model_satisfies_protocol(model):
    assert isinstance(model, ControlModel)
    assert (model.nx, model.nu, model.nd, model.ny) == (3, 1, 2, 1)


@pytest.mark.parametrize("n_segments", [3, 20])
def test_numpy_and_casadi_agree(n_segments):
    m = FPRControlModel(n_segments=n_segments)
    f, h = casadi_functions(m)
    rng = np.random.default_rng(1)
    for _ in range(10):
        x = np.array([rng.uniform(1.0, 12.0), rng.uniform(700.0, 1100.0), rng.uniform(700, 1100)])
        u = rng.uniform(0.5, 12.0, 1)
        d = np.array([rng.uniform(0.0, 1.2e6), rng.uniform(550.0, 950.0)])
        np.testing.assert_allclose(np.array(f(x, u, d)).ravel(), m.rhs(x, u, d), rtol=1e-10)
        np.testing.assert_allclose(np.array(h(x, u, d)).ravel(), m.output(x, u, d), rtol=1e-12)


def test_casadi_jacobian_is_finite_with_physical_signs(model, nominal):
    x, u, d = ca.SX.sym("x", 3), ca.SX.sym("u", 1), ca.SX.sym("d", 2)
    xdot = model.rhs(x, u, d)
    jacobian = ca.Function("jacobian_xud", [x, u, d], [ca.jacobian(xdot, ca.vertcat(x, u, d))])
    J = np.array(jacobian(nominal.x, nominal.u, nominal.d))
    assert np.all(np.isfinite(J))
    d_Tmix = J[1]
    assert d_Tmix[0] < 0  # more flow -> colder curtain outlet
    assert d_Tmix[4] > 0  # more sun -> hotter
    assert d_Tmix[5] > 0  # hotter inlet -> hotter


def test_steady_state_matches_steady_curtain_model(model, nominal):
    assert np.max(np.abs(model.rhs(nominal.x, nominal.u, nominal.d))) < 1e-9
    assert nominal.x[2] == pytest.approx(T_TARGET, abs=1e-8)
    params = dataclasses.replace(
        model.params, mass_flow=float(nominal.u[0]), Q_in=Q_NOMINAL, T_in=T_IN
    )
    T_part1 = float(simulate_curtain(params).T_out)
    assert T_part1 == pytest.approx(nominal.x[1], abs=1e-9)


def test_unreachable_target_raises(model):
    with pytest.raises(ValueError, match="not reachable"):
        steady_state(model, 1e3, T_IN, T_TARGET)


def test_coarse_controller_model_converges_to_fine_plant():
    flows = {
        n: steady_state(FPRControlModel(n_segments=n), Q_NOMINAL, T_IN, T_TARGET).u[0]
        for n in (2, 5, 20, 80)
    }
    errors = [abs(flows[n] - flows[80]) for n in (2, 5, 20)]
    assert errors[0] > errors[1] > errors[2]


def test_open_loop_power_step_settles_on_new_equilibrium(model, nominal):
    # Sandia's PID test removed heliostats: 739 kW -> 610 kW at fixed gate.
    dt, n_steps = 1.0, 300
    u = np.tile(nominal.u, (n_steps, 1))
    d = np.tile([610e3, T_IN], (n_steps, 1))
    X, Y = simulate(model, nominal.x, u, d, dt=dt, substeps=1)
    T_new = model.equilibrium(float(nominal.u[0]), 610e3, T_IN).x[2]
    assert Y[-1, 0] == pytest.approx(T_new, abs=1e-3)
    # Three cascaded first-order lags: monotone, no overshoot.
    assert np.all(np.diff(Y[:, 0]) <= 1e-12)
    assert Y[-1, 0] < T_TARGET - 10.0
    # The recommended sample time (1 s, 1 RK4 substep) matches a 10x finer integration
    # over the transient (first minute).
    X_fine, _ = simulate(model, nominal.x, u[:60], d[:60], dt=dt, substeps=10)
    np.testing.assert_allclose(X[:61, 1:], X_fine[:, 1:], atol=0.02)


def test_numpy_and_casadi_discretisations_agree(model, nominal):
    step_np = discretize(model, dt=1.0, substeps=2)
    step_ca = casadi_discretize(model, dt=1.0, substeps=2)
    x = nominal.x + np.array([0.5, -4.0, 3.0])
    u, d = nominal.u * 0.9, np.array([610e3, T_IN])
    np.testing.assert_allclose(np.array(step_ca(x, u, d)).ravel(), step_np(x, u, d), rtol=1e-10)


def test_invalid_time_constants_are_rejected():
    with pytest.raises(ValueError, match="tau_mix"):
        FPRControlModel(FPRParams(), tau_mix=0.0)
