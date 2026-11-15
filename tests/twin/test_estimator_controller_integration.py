"""The estimation and control packages must work together through the protocols.

Scenario: a lumped heated particle stream whose solar input is not measured.
An EnKF on a disturbance-augmented (and deliberately mismatched) model
estimates the state and the solar power; the NMPC uses both. This is the
smallest version of the receiver experiment and guards the seams between the
closed-loop simulator, the estimators and the controllers.
"""

import numpy as np
import pytest

from pcsp.core import Controller, ControlModelBase, Estimator, discretize, ops_for
from pcsp.twin.control import NMPC, NMPCConfig, PIDController, regulation_metrics, run_closed_loop
from pcsp.twin.estimation import AugmentedModel, EnsembleKalmanFilter, ExtendedKalmanFilter

DT = 5.0
T_IN = 800.0
SETPOINT = 1000.0
MDOT_0 = 5.0
RATE_MAX = 0.1  # kg/s per second
X0 = np.array([SETPOINT, SETPOINT])


class HeatedStream(ControlModelBase):
    """``C dT/dt = Q_sun - mdot cp (T - T_in)`` read by a thermocouple with an 8 s lag."""

    state_names = ("T", "T_tc")
    input_names = ("mdot",)
    disturbance_names = ("Q_sun", "T_in")
    output_names = ("T_tc",)

    def __init__(self, cp: float = 1200.0, heat_capacity: float = 2.0e5) -> None:
        self.cp = cp
        self.heat_capacity = heat_capacity

    def rhs(self, x, u, d):
        xp = ops_for(x, u, d)
        heating = (d[0] - u[0] * self.cp * (x[0] - d[1])) / self.heat_capacity
        return xp.concat([heating, (x[0] - x[1]) / 8.0])

    def output(self, x, u, d):
        return ops_for(x, u, d).concat([x[1]])


def solar_step(t: float) -> np.ndarray:
    """20 % drop in solar power at t = 100 s; inlet temperature constant."""
    return np.array([1.2e6 if t < 100.0 else 0.96e6, T_IN])


def measured_disturbance(t: float) -> np.ndarray:
    """Only the inlet temperature is measured."""
    return np.array([T_IN])


LOOP = dict(
    dt=DT,
    n_steps=160,
    setpoint=SETPOINT,
    disturbance=solar_step,
    plant_substeps=4,
    noise_std=1.0,
    seed=3,
    actuator_rate_max=RATE_MAX,
)


def make_filter(kind: str, aug: AugmentedModel):
    z0 = aug.join(X0, np.array([1.2e6, T_IN]))
    Q = np.diag([0.5**2, 0.5**2, 2.0e4**2])
    R = np.array([[1.0]])
    P0 = np.diag([4.0, 4.0, 1.0e5**2])
    if kind == "ekf":
        return ExtendedKalmanFilter(aug, DT, Q, R, z0, P0, substeps=2)
    return EnsembleKalmanFilter(
        discretize(aug, DT, 2), aug.output, Q, R, z0, P0, 60, method=kind, inflation=1.02, seed=1
    )


@pytest.mark.parametrize("kind", ["sqrt", "stochastic", "ekf"])
def test_estimator_plus_nmpc_rejects_unmeasured_solar_step(kind):
    plant = HeatedStream()
    model = HeatedStream(cp=1150.0, heat_capacity=2.3e5)  # the controller's model is wrong
    aug = AugmentedModel(model, disturbances=("Q_sun",))
    estimator = make_filter(kind, aug)
    mpc = NMPC(
        model,
        NMPCConfig(
            dt=DT,
            horizon=20,
            tracked_outputs=[0],
            output_weight=[1.0],
            move_weight=[20.0],
            u_min=[1.0],
            u_max=[10.0],
            rate_max=[RATE_MAX],
            u_init=[MDOT_0],
            substeps=2,
            x_scale=[100.0, 100.0],
            x_offset=[1000.0, 1000.0],
        ),
    )
    assert isinstance(mpc, Controller) and isinstance(estimator, Estimator)

    measured = measured_disturbance(0.0)
    result = run_closed_loop(
        plant,
        mpc,
        X0,
        [MDOT_0],
        known_disturbance=measured_disturbance,
        estimator=estimator,
        estimate_to_state=lambda z: aug.split(z, measured)[0],
        estimate_to_disturbance=lambda z: aug.split(z, measured)[1],
        **LOOP,
    )

    # Every optimisation succeeded and the actuator limits were respected.
    assert np.all(result.diagnostics["success"] == 1.0)
    assert result.u.min() >= 1.0 and result.u.max() <= 10.0
    assert np.abs(np.diff(result.u[:, 0])).max() <= RATE_MAX * DT + 1e-7

    # The disturbance estimate absorbs the model error, so there is no offset
    # even though the controller's model has the wrong heat capacity.
    tail = result.y_true[-30:, 0]
    assert abs(tail.mean() - SETPOINT) < 0.5
    # The loop is at rest again: mass flow has dropped by about the 20 % step.
    assert result.u[-1, 0] == pytest.approx(0.8 * MDOT_0, rel=0.02)


def test_pid_baseline_runs_through_the_same_loop_and_is_reproducible():
    def make_pid():
        return PIDController(
            0.02, 0.0012, 0.0, DT, action="direct", bias=MDOT_0, u_min=1.0, u_max=10.0,
            rate_max=RATE_MAX,
        )  # fmt: skip

    first = run_closed_loop(HeatedStream(), make_pid(), X0, [MDOT_0], **LOOP)
    second = run_closed_loop(HeatedStream(), make_pid(), X0, [MDOT_0], **LOOP)
    np.testing.assert_array_equal(first.y_meas, second.y_meas)
    metrics = regulation_metrics(
        first.t, first.y_true[:, 0], first.r[:, 0], first.u, t_start=100.0, band=2.0
    )
    assert abs(first.y_true[-1, 0] - SETPOINT) < 1.0  # integral action removes the offset
    assert metrics.iae > 0.0
