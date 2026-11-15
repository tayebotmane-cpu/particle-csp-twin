"""Receiver control study: properties that must hold whatever the tuning.

The runs are short (a few minutes of plant time, a coarse plant and a short
NMPC horizon) so the file finishes in well under a minute; the full study is
``examples/receiver_control.py``.
"""

import dataclasses

import numpy as np
import pytest

from pcsp.core import Controller, Estimator
from pcsp.pipelines.receiver_control import (
    CONFIGURATIONS,
    EnKFConfig,
    ExperimentSetup,
    Instrumentation,
    ModelConfig,
    NMPCTuning,
    PlantConfig,
    Scenario,
    build_estimator,
    observability_report,
    run_monte_carlo,
    simulate_configuration,
    summarise,
    tune_pid,
)
from pcsp.twin.control import integral_absolute_error

SETUP = ExperimentSetup(
    scenario=Scenario(t_step=30.0, t_restore=200.0, t_end=210.0),
    plant=PlantConfig(n_segments=10),
    model=ModelConfig(upper_bound_segments=5),
    enkf=EnKFConfig(n_members=15),
    nmpc=NMPCTuning(horizon=15),
)
SEED = 7


@pytest.fixture(scope="module")
def runs():
    """One closed-loop run of every configuration on the same seed."""
    return {name: simulate_configuration(name, SETUP, SEED) for name in CONFIGURATIONS}


def test_configurations_implement_the_protocols(runs):
    for _, cfg in runs.values():
        assert isinstance(cfg.controller, Controller)
        if "estimator" in cfg.loop_kwargs:
            assert isinstance(cfg.loop_kwargs["estimator"], Estimator)


def test_every_configuration_respects_mass_flow_bounds_and_rate_limit(runs):
    ins = SETUP.instruments
    for name, (result, cfg) in runs.items():
        u = result.u[:, 0]
        assert u.min() >= ins.m_dot_min - 1e-9, name
        assert u.max() <= ins.m_dot_max + 1e-9, name
        assert np.abs(np.diff(u)).max() <= ins.rate_max * SETUP.scenario.dt + 1e-9, name
        # The controllers respect the limits themselves; the plant actuator clip is a backstop.
        np.testing.assert_allclose(result.u_cmd, result.u, atol=1e-6, err_msg=name)
        if cfg.nmpc is not None:
            assert all(rec.success for rec in cfg.nmpc.solve_log), name


def test_identical_seeds_give_identical_runs():
    short = dataclasses.replace(SETUP, scenario=dataclasses.replace(SETUP.scenario, t_end=50.0))
    first, _ = simulate_configuration("C", short, seed=3)
    second, _ = simulate_configuration("C", short, seed=3)
    for field in ("y_meas", "u", "x_est"):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    other, _ = simulate_configuration("A", short, seed=4)
    assert not np.array_equal(first.y_meas, other.y_meas)


def test_only_the_labelled_configurations_receive_plant_truth():
    """Plant power far from anything the controllers are told; check what they receive."""
    weird = dataclasses.replace(
        SETUP,
        scenario=dataclasses.replace(
            SETUP.scenario, Q_before=655_432.1, Q_after=588_888.8, t_step=10.0, t_end=30.0
        ),
    )
    true_powers = (655_432.1, 588_888.8)
    design = weird.model.design_power
    for name in ("A", "B_signal", "C", "D"):
        result, cfg = simulate_configuration(name, weird, SEED)
        assert not cfg.loop_kwargs.get("full_state_feedback", False), name
        # The only disturbance handed over is the measured inlet temperature.
        np.testing.assert_array_equal(
            result.d_hat, np.full((result.t.size, 1), weird.scenario.T_in)
        )
        for q in true_powers:
            assert not np.any(result.d_hat == q), name
        if result.x_hat is not None:
            # The state comes from the estimator, never from the plant.
            assert not np.any(np.all(result.x_hat[:, :3] == result.x[:, :3], axis=1)), name
            q0 = result.x_hat[0, cfg.power_estimate_index]
            assert abs(q0 - design) < abs(q0 - true_powers[0]), name
            assert not np.any(np.isin(result.x_hat[:, cfg.power_estimate_index], true_powers))
    # The idealised configurations do receive the truth, by construction.
    result, _ = simulate_configuration("B_true", weird, SEED)
    np.testing.assert_array_equal(result.d_hat[:, 0], result.d[:, 0])
    _, cfg = simulate_configuration("E", weird, SEED)
    assert cfg.loop_kwargs["full_state_feedback"]


@pytest.mark.parametrize("name", ["A", "C"])
def test_offset_is_removed_despite_model_mismatch(name):
    """Near-noise-free sensors isolate the offset from noise; the model mismatch remains."""
    quiet = dataclasses.replace(
        SETUP, instruments=Instrumentation(sigma_T=0.01, sigma_m_dot=1e-4, sigma_Q=10.0)
    )
    assert quiet.plant.h_adv != quiet.model.h_adv
    result, _ = simulate_configuration(name, quiet, SEED)
    sc = quiet.scenario
    plateau = (result.t >= sc.t_restore - 30.0) & (result.t < sc.t_restore)
    assert abs(np.mean(result.y_true[plateau, 0]) - sc.T_set) < 0.2


def test_upper_bound_is_not_worse_than_the_twin(runs):
    sc = SETUP.scenario

    def iae(name):
        result, _ = runs[name]
        return integral_absolute_error(result.t, result.y_true[:, 0], sc.T_set, sc.t_step)

    assert iae("E") <= iae("C")


def test_incident_power_is_observable_with_the_twin_sensors():
    rows = observability_report(SETUP)
    assert all(row.observable for row in rows)


def test_twin_power_estimate_follows_the_step(runs):
    """C's power estimate moves by about the true step (not to the true level: mismatch)."""
    _, aug = build_estimator("C", SETUP, seed=0)
    assert aug.state_names[-1] == "d:Q_in"
    result, _ = runs["C"]
    q_hat = result.x_est[:, -1]
    t, sc = result.t, SETUP.scenario
    before = q_hat[(t >= sc.t_step - 20.0) & (t < sc.t_step)].mean()
    after = q_hat[(t >= sc.t_restore - 20.0) & (t < sc.t_restore)].mean()
    true_step = sc.Q_after - sc.Q_before
    assert (after - before) == pytest.approx(true_step, rel=0.3)


def test_monte_carlo_and_summary_shapes():
    long = dataclasses.replace(SETUP, scenario=dataclasses.replace(SETUP.scenario, t_end=360.0))
    results = run_monte_carlo(long, configs=("A", "B_signal"), seeds=(0, 1))
    assert [(r.config, r.seed) for r in results] == [
        ("A", 0), ("A", 1), ("B_signal", 0), ("B_signal", 1),
    ]  # fmt: skip
    summary = summarise(results)
    assert set(summary) == {"A", "B_signal"}
    assert summary["A"].n_runs == 2 and 0.0 <= summary["A"].pass_rate <= 1.0
    assert summary["A"].mean["iae"] > 0.0
    assert np.isnan(summary["A"].mean["solve_time_mean"])


def test_pid_tuning_does_not_make_the_objective_worse():
    result = tune_pid(SETUP, seeds=(1,), max_evaluations=6)
    assert result.objective <= result.initial_objective
    assert min(result.gains.kp, result.gains.ki, result.gains.kd) > 0.0
