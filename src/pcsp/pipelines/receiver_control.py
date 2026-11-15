"""Outlet-temperature control of a falling-particle receiver: digital twin versus PID.

The experiment reproduces the published Sandia receiver control test (Mills et
al., *Energies* 15:1657, 2022): hold a particle outlet setpoint of 700 degC
with particles entering at 550 degC while the incident solar power is stepped
from 739 kW to 610 kW and later restored. The success criterion is an outlet
temperature within +/- 2 sigma <= 10 degC over any 5-minute window. The
manipulated variable is the particle mass flow through a slide gate.

Systems
-------
* **Plant** (the "truth"): :class:`~pcsp.models.receiver.FPRControlModel` at
  fine curtain resolution (50 segments) with parameters that differ from the
  controller's model (:class:`PlantConfig`), wrapped in
  :class:`InstrumentedReceiver`, which adds the sensors a real receiver has:
  outlet thermocouple, mass-flow meter and an imperfect incident-power signal.
* **Controller model**: the same class at coarse resolution (5 segments) with
  nominal parameters (:class:`ModelConfig`). Every estimator and controller
  except the labelled upper bound uses only this model.

Controller configurations (:data:`CONFIGURATIONS`)
--------------------------------------------------
``A``         PID on the measured outlet temperature (the Sandia baseline).
``B_true``    PID + static feedforward from the *true* incident power (idealised).
``B_signal``  PID + the same feedforward from the imperfect power signal.
``C``         Twin: EnKF on the disturbance-augmented controller model, fed by
              the thermocouple and the mass-flow meter, estimates the state
              and the incident power; NMPC on the controller model uses both.
``D``         As C, with the imperfect power signal fused in the EnKF as an
              extra measurement with its own lag and an estimated bias.
``E``         Upper bound: NMPC with the true plant parameters, the true plant
              state and the true disturbance (perfect information).

Information each configuration receives
----------------------------------------
All configurations see the same noisy measurement vector
``y = [T_out_meas, m_dot_meas, Q_signal]`` (one noise draw per seed, identical
across configurations) and the measured inlet temperature (taken as
noise-free; it is constant in this scenario). They also know the actuator
position at switch-on, ``u0``, which a real plant reads from the slide gate.
Only ``B_true`` (true power) and ``E`` (everything) receive plant truth, and
both are labelled as idealised.

Assumptions are labelled "(assumption)" in the dataclass docstrings; time
constants inherit the "assumption, unverified" label of
:mod:`pcsp.models.receiver.dynamic`.
"""

from __future__ import annotations

import dataclasses
import multiprocessing
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize

from pcsp.constants import T_ZERO_C
from pcsp.core.backend import ops_for
from pcsp.core.integrate import casadi_discretize
from pcsp.core.interfaces import ControlModel, ControlModelBase
from pcsp.models.receiver import FPRControlModel, FPRParams, steady_state
from pcsp.twin.control import (
    NMPC,
    ClosedLoopResult,
    NMPCConfig,
    PIDController,
    run_closed_loop,
    settling_time,
    total_variation,
    two_sigma_band,
)
from pcsp.twin.control.metrics import integral_absolute_error, peak_deviation
from pcsp.twin.estimation import AugmentedModel, EnsembleKalmanFilter, observability

CONFIGURATIONS = ("A", "B_true", "B_signal", "C", "D", "E")
"""All controller configurations, in reporting order."""

LABELS = {
    "A": "A: PID",
    "B_true": "B: PID + FF (true power, idealised)",
    "B_signal": "B: PID + FF (power signal)",
    "C": "C: twin (EnKF + NMPC)",
    "D": "D: twin + power signal",
    "E": "E: NMPC, perfect information",
}
"""Human-readable configuration names."""

NMPC_CONFIGURATIONS = ("C", "D", "E")
SANDIA_LIMIT = 10.0
"""Sandia success criterion: worst 5-minute 2-sigma band of the outlet temperature [K]."""

SANDIA_WINDOW = 300.0
"""Length of the Sandia evaluation window [s]."""

SETTLING_BAND = 5.0
"""Half-width of the settling band around the setpoint [K]."""

OFFSET_WINDOW = 120.0
"""Averaging window for the steady-state offset, ending at each power change [s]."""


# --------------------------------------------------------------------------- set-up
@dataclass(frozen=True)
class Scenario:
    """Operating point and incident-power profile (Mills et al. 2022 test).

    Attributes
    ----------
    Q_before : incident power before the step and after the restore [W] (739 kW, published)
    Q_after : incident power during the step [W] (610 kW, published: -17.5 %)
    T_in : particle inlet temperature [K] (550 degC, published "about 550 degC")
    T_set : outlet temperature setpoint [K] (700 degC, published)
    t_step : time of the power step [s]; the 300 s before it let estimators
        and integrators settle from their priors (metrics start here)
    t_restore : time the power is restored [s] (600 s after the step: two
        full Sandia windows at the low power)
    t_end : end of the run [s] (600 s after the restore)
    dt : sample period of all controllers and estimators [s]
    """

    Q_before: float = 739e3
    Q_after: float = 610e3
    T_in: float = 550.0 + T_ZERO_C
    T_set: float = 700.0 + T_ZERO_C
    t_step: float = 300.0
    t_restore: float = 900.0
    t_end: float = 1500.0
    dt: float = 1.0

    @property
    def n_steps(self) -> int:
        """Number of sample periods in the run."""
        return int(round(self.t_end / self.dt))

    def incident_power(self, t: float) -> float:
        """True solar power entering the aperture at time ``t`` [W]."""
        return self.Q_after if self.t_step <= t < self.t_restore else self.Q_before

    def disturbance(self, t: float) -> np.ndarray:
        """True plant disturbance ``[Q_in, T_in]`` at time ``t`` [W, K]."""
        return np.array([self.incident_power(t), self.T_in])

    def measured_inlet(self, t: float) -> np.ndarray:
        """Measured inlet temperature ``[T_in]`` [K] (noise-free, assumption)."""
        return np.array([self.T_in])

    def true_power(self, t: float) -> np.ndarray:
        """True incident power as a one-element vector [W] (idealised feedforward only)."""
        return np.array([self.incident_power(t)])


@dataclass(frozen=True)
class PlantConfig:
    """The virtual plant: how the "true" receiver differs from the controller model.

    Attributes
    ----------
    h_adv : advective-loss coefficient [W m^-2 K^-1]. 45.6 = 1.2 x the
        nominal 38 of González-Portillo et al. (CFD fit). 20 % more advective
        loss than modelled represents wind: CFD put the NSTTF advective loss at
        10-25 % of the incident power depending on wind (Mills et al. 2022,
        Sec. 3.4) (assumption: the size of the error).
    tau_gate, tau_mix, tau_tc : plant lags [s] (1.5, 14, 7: each 40-50 %
        slower than the model's nominal 1, 10, 5; the nominal values are
        themselves unverified assumptions, so the plant's are too)
    n_segments : curtain resolution of the plant
    power_signal_gain : multiplicative error of the incident-power signal [-]
        (1.08). Aperture power is not measured in real time; it is inferred
        from DNI and a heliostat-field efficiency model, whose error is of
        order 5-10 % (assumption).
    power_signal_lag : first-order lag of that signal [s] (15; DNI averaging,
        field-model update and acquisition latency, assumption). A pure delay
        cannot be written as an ODE state, so a lag stands in for it.
    """

    h_adv: float = 1.2 * 38.0
    tau_gate: float = 1.5
    tau_mix: float = 14.0
    tau_tc: float = 7.0
    n_segments: int = 50
    power_signal_gain: float = 1.08
    power_signal_lag: float = 15.0


@dataclass(frozen=True)
class ModelConfig:
    """The controller's model of the receiver (nominal, coarse).

    Attributes
    ----------
    h_adv : advective coefficient [W m^-2 K^-1] (38, nominal CFD fit)
    tau_gate, tau_mix, tau_tc : lags [s] (1, 10, 5: the module defaults,
        assumptions)
    n_segments : curtain resolution (5)
    design_power : the incident power the operator expects [W]; used only for
        estimator priors (739 kW, the published pre-step value)
    power_signal_lag : the lag the twin assumes for the power signal [s]
        (10, deliberately not the plant's 15)
    upper_bound_segments : curtain resolution of configuration E's model,
        which has the plant's true parameters (10: 50 would cost ~0.8 s per
        solve; the steady mass flow at 10 segments differs from 50 by ~0.15 %)
    """

    h_adv: float = 38.0
    tau_gate: float = 1.0
    tau_mix: float = 10.0
    tau_tc: float = 5.0
    n_segments: int = 5
    design_power: float = 739e3
    power_signal_lag: float = 10.0
    upper_bound_segments: int = 10


@dataclass(frozen=True)
class Instrumentation:
    """Sensor noise and actuator limits (shared by plant and all controllers).

    Attributes
    ----------
    sigma_T : outlet-thermocouple white noise [K] (1.5, assumption)
    sigma_m_dot : mass-flow measurement noise [kg s^-1] (0.05, ~2 % of the
        2.5 kg/s operating flow; assumption)
    sigma_Q : power-signal noise [W] (10 kW, ~1.5 %; assumption)
    m_dot_min, m_dot_max : mass-flow bounds [kg s^-1] (0.5: the curtain
        model's stability floor; 10: within the 1-11 kg/s of Sandia's tests)
    rate_max : slide-gate slew limit [kg s^-2] (0.1). The gate itself moves
        in < 1 s; the limit is a software slew limit protecting the particle
        conveying system (assumption). The 0.9 kg/s change the step requires
        then takes ~9 s, comparable to the lags.
    power_filter_time : low-pass filter on the power signal before it is used
        as PID feedforward [s]; chosen by :func:`tune_power_filter` (IAE on
        the controller model): 871, 696, 620, 637, 662, 683 K s for 0, 1, 3,
        5, 10, 20 s, so 3 s
    """

    sigma_T: float = 1.5
    sigma_m_dot: float = 0.05
    sigma_Q: float = 10e3
    m_dot_min: float = 0.5
    m_dot_max: float = 10.0
    rate_max: float = 0.1
    power_filter_time: float = 3.0

    @property
    def noise_std(self) -> np.ndarray:
        """Noise of ``[T_out_meas, m_dot_meas, Q_signal]`` [K, kg s^-1, W]."""
        return np.array([self.sigma_T, self.sigma_m_dot, self.sigma_Q])


@dataclass(frozen=True)
class EnKFConfig:
    """Ensemble Kalman filter settings of configurations C and D.

    Process noise is per sample (``dt = 1 s``) and given as standard
    deviations, keyed by augmented-state name; the same keys give the prior
    standard deviations. The incident-power random walk (4 kW per sample)
    trades detection speed against noise; it was chosen a priori, not tuned
    on the plant.
    """

    n_members: int = 40
    method: str = "sqrt"
    inflation: float = 1.0
    process_std: dict[str, float] = field(
        default_factory=lambda: {
            "m_dot": 0.01,
            "T_mix": 0.2,
            "T_tc": 0.05,
            "Q_signal_lag": 1e3,
            "d:Q_in": 4e3,
            "bias:Q_signal": 0.3e3,
        }
    )
    prior_std: dict[str, float] = field(
        default_factory=lambda: {
            "m_dot": 0.2,
            "T_mix": 5.0,
            "T_tc": 5.0,
            "Q_signal_lag": 50e3,
            "d:Q_in": 50e3,
            "bias:Q_signal": 50e3,
        }
    )


@dataclass(frozen=True)
class NMPCTuning:
    """NMPC horizon and weights (configurations C, D and E use the same).

    Attributes
    ----------
    horizon : prediction horizon [samples] (60 s, ~3x the plant's summed lags)
    output_weight : weight on the outlet-temperature error [K^-2]
    move_weight : weight on mass-flow moves [(kg/s)^-2]; chosen by
        :func:`tune_move_weight` (IAE on the controller model, the PID's
        objective) from 10-3000: 571, 539, 521, 521, 526, 536 K s for
        10, 30, 100, 300, 1000, 3000 - a flat optimum at 100-300
    max_iter, tol : IPOPT iteration limit and tolerance
    """

    horizon: int = 60
    output_weight: float = 1.0
    move_weight: float = 300.0
    max_iter: int = 100
    tol: float = 1e-6


@dataclass(frozen=True)
class PIDGains:
    """Parallel-form PID gains (``action="direct"``: more flow lowers temperature).

    The defaults are the result of :func:`tune_pid` on the default set-up
    (151 evaluations; tuning IAE 886 -> 577 K s from the SIMC start). The
    optimiser drove ``kd`` to ~3e-8 (derivative action only amplifies the
    thermocouple noise), so the tuned controller is a PI and ``kd`` is set
    to 0. Re-run after changing the controller model, sensors or scenario.
    """

    kp: float = 0.0222
    ki: float = 0.00270
    kd: float = 0.0


@dataclass(frozen=True)
class ExperimentSetup:
    """Everything that defines one experiment."""

    scenario: Scenario = field(default_factory=Scenario)
    plant: PlantConfig = field(default_factory=PlantConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    instruments: Instrumentation = field(default_factory=Instrumentation)
    enkf: EnKFConfig = field(default_factory=EnKFConfig)
    nmpc: NMPCTuning = field(default_factory=NMPCTuning)
    pid: PIDGains = field(default_factory=PIDGains)


# ------------------------------------------------------------------------- models
@dataclass(frozen=True)
class PowerSignalModel:
    """Incident-power signal ``Q_signal = gain * Q_lag``, ``dQ_lag/dt = (Q_in - Q_lag) / lag``."""

    gain: float
    lag: float


class InstrumentedReceiver(ControlModelBase):
    """A receiver model plus the sensors of a real plant.

    =========  ================  ================================  ======
    vector     name              meaning                           units
    =========  ================  ================================  ======
    x[0:3]     receiver states   ``m_dot``, ``T_mix``, ``T_tc``
    x[3]       ``Q_signal_lag``  lagged incident power (optional)  W
    y[0]       ``T_out_meas``    outlet thermocouple               K
    y[1]       ``m_dot_meas``    mass-flow meter                   kg s-1
    y[2]       ``Q_signal``      incident-power signal (optional)  W
    =========  ================  ================================  ======

    Inputs and disturbances are those of the wrapped receiver. Measurement
    noise is added by the closed-loop simulator, not here.
    """

    def __init__(self, receiver: FPRControlModel, power_signal: PowerSignalModel | None) -> None:
        self.receiver = receiver
        self.power_signal = power_signal
        extra_state = ("Q_signal_lag",) if power_signal is not None else ()
        extra_output = ("Q_signal",) if power_signal is not None else ()
        self.state_names = tuple(receiver.state_names) + extra_state
        self.input_names = tuple(receiver.input_names)
        self.disturbance_names = tuple(receiver.disturbance_names)
        self.output_names = ("T_out_meas", "m_dot_meas") + extra_output

    def rhs(self, x: Any, u: Any, d: Any) -> Any:
        """``dx/dt``: the receiver's, then the power-signal lag [W s^-1]."""
        xp = ops_for(x, u, d)
        n = self.receiver.nx
        dx = self.receiver.rhs(x[:n], u, d)
        if self.power_signal is None:
            return dx
        return xp.concat([dx, (d[0] - x[n]) / self.power_signal.lag])

    def output(self, x: Any, u: Any, d: Any) -> Any:
        """``[T_tc, m_dot]`` and, with a power signal, ``gain * Q_lag``."""
        xp = ops_for(x, u, d)
        parts = [x[2], x[0]]
        if self.power_signal is not None:
            parts.append(self.power_signal.gain * x[self.receiver.nx])
        return xp.concat(parts)


def build_plant(setup: ExperimentSetup) -> InstrumentedReceiver:
    """The virtual plant: fine curtain, mismatched parameters, all three sensors."""
    p = setup.plant
    receiver = FPRControlModel(
        dataclasses.replace(FPRParams(), h_adv=p.h_adv),
        n_segments=p.n_segments,
        tau_gate=p.tau_gate,
        tau_mix=p.tau_mix,
        tau_tc=p.tau_tc,
    )
    return InstrumentedReceiver(receiver, PowerSignalModel(p.power_signal_gain, p.power_signal_lag))


def build_controller_model(setup: ExperimentSetup) -> FPRControlModel:
    """The coarse, nominal receiver model used by every non-idealised configuration."""
    m = setup.model
    return FPRControlModel(
        dataclasses.replace(FPRParams(), h_adv=m.h_adv),
        n_segments=m.n_segments,
        tau_gate=m.tau_gate,
        tau_mix=m.tau_mix,
        tau_tc=m.tau_tc,
    )


def build_truth_model(setup: ExperimentSetup) -> InstrumentedReceiver:
    """Configuration E's model: the plant's parameters at ``upper_bound_segments``."""
    plant = build_plant(setup)
    receiver = FPRControlModel(
        plant.receiver.params,
        n_segments=setup.model.upper_bound_segments,
        tau_gate=plant.receiver.tau_gate,
        tau_mix=plant.receiver.tau_mix,
        tau_tc=plant.receiver.tau_tc,
    )
    return InstrumentedReceiver(receiver, plant.power_signal)


def plant_initial_condition(
    setup: ExperimentSetup, plant: InstrumentedReceiver
) -> tuple[np.ndarray, np.ndarray]:
    """Plant at rest on the setpoint at the initial power: ``(x0, u0)``."""
    sc = setup.scenario
    Q0 = sc.incident_power(0.0)
    ss = steady_state(plant.receiver, Q0, sc.T_in, sc.T_set)
    x0 = ss.x if plant.power_signal is None else np.concatenate([ss.x, [Q0]])
    return x0, ss.u.copy()


# ----------------------------------------------------------------------- adapters
class MeasurementSubset:
    """Estimator adapter that assimilates only some of the measured outputs.

    The plant reports ``[T_out_meas, m_dot_meas, Q_signal]``; configuration C
    must not use ``Q_signal``. This wrapper passes ``y[outputs]`` to the
    wrapped estimator and forwards everything else unchanged.
    """

    def __init__(self, estimator: EnsembleKalmanFilter, outputs: Sequence[int]) -> None:
        self.estimator = estimator
        self.outputs = np.asarray(outputs, dtype=int)

    def predict(self, u: np.ndarray, d: np.ndarray) -> None:
        self.estimator.predict(u, d)

    def update(
        self, y: np.ndarray, u: np.ndarray | None = None, d: np.ndarray | None = None
    ) -> None:
        self.estimator.update(np.asarray(y, dtype=float).ravel()[self.outputs], u, d)

    @property
    def x_hat(self) -> np.ndarray:
        return self.estimator.x_hat

    @property
    def P(self) -> np.ndarray:
        return self.estimator.P


def _pass_through(d: np.ndarray) -> float:
    """Feedforward callable for :class:`PIDController`: ``d_hat[0]`` is already the flow."""
    return float(d[0])


class FeedforwardPID:
    """PID plus static feedforward from an incident-power value.

    The feedforward is the steady-state mass-flow change that the controller
    model predicts for the observed power change since switch-on,
    ``m_ss(Q) - m_ss(Q_switch_on)``, so the loop starts bumplessly; feedback
    removes whatever the map and the signal get wrong. The term is added inside
    :class:`PIDController` (via its ``feedforward`` hook) so it is subject to
    the same saturation, rate limit and anti-windup as the feedback.

    Two signal-conditioning steps make the feedforward usable in practice:

    * The power is low-pass filtered, ``s_k = s_{k-1} + h / (T_f + h) (Q_k -
      s_{k-1})``, when ``filter_time = T_f > 0`` (for the noisy signal).
    * The feedforward is ramped at no more than the gate slew rate. Without
      this, :class:`PIDController`'s back-calculation (tracking time ``dt``
      for a PI) loads the integrator with every part of a feedforward step
      the rate limit clips, which cancels the step permanently (see the
      study report).

    Parameters
    ----------
    pid : feedback controller, built with ``feedforward=_pass_through``
    steady_flow : map from incident power [W] to steady mass flow [kg s^-1]
    signal_output : ``None`` reads the power from ``d_hat[0]`` (the idealised
        true power); an integer reads it from that entry of the measurement
    rate_max : ramp limit of the feedforward [kg s^-2]
    filter_time : time constant of the power filter [s] (0: no filter)
    """

    def __init__(
        self,
        pid: PIDController,
        steady_flow: Callable[[float], float],
        signal_output: int | None,
        rate_max: float,
        filter_time: float = 0.0,
    ) -> None:
        if pid.feedforward is None:
            raise ValueError("pid must be built with a feedforward hook")
        if rate_max <= 0.0 or filter_time < 0.0:
            raise ValueError("need rate_max > 0 and filter_time >= 0")
        self.pid = pid
        self.steady_flow = steady_flow
        self.signal_output = signal_output
        self.rate_max = rate_max
        self.filter_time = filter_time
        self.reset()

    def reset(self) -> None:
        self.pid.reset()
        self._flow_ref: float | None = None
        self._power: float | None = None
        self._ff = 0.0

    def step(
        self,
        t: float,
        y: np.ndarray,
        r: np.ndarray,
        x_hat: np.ndarray | None = None,
        d_hat: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.signal_output is None:
            if d_hat is None:
                raise ValueError("signal_output=None needs the power in d_hat")
            power = float(np.asarray(d_hat, dtype=float).ravel()[0])
        else:
            power = float(np.asarray(y, dtype=float).ravel()[self.signal_output])
        h = self.pid.dt
        if self._power is None:
            self._power = power
        self._power += h / (self.filter_time + h) * (power - self._power)
        flow = self.steady_flow(self._power)
        if self._flow_ref is None:
            self._flow_ref = flow
        step = self.rate_max * h
        self._ff += float(np.clip(flow - self._flow_ref - self._ff, -step, step))
        return self.pid.step(t, y, r, d_hat=np.array([self._ff]))

    def diagnostics(self) -> dict[str, float]:
        return self.pid.diagnostics()


class HoldInput:
    """Controller that holds a constant input (open-loop estimator tests)."""

    def __init__(self, u: np.ndarray) -> None:
        self.u = np.asarray(u, dtype=float).copy()

    def reset(self) -> None:
        """Nothing to reset."""

    def step(
        self,
        t: float,
        y: np.ndarray,
        r: np.ndarray,
        x_hat: np.ndarray | None = None,
        d_hat: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.u.copy()


def steady_flow_map(
    model: FPRControlModel, T_in: float, T_set: float, powers: np.ndarray
) -> Callable[[float], float]:
    """Steady mass flow [kg s^-1] holding ``T_set`` as a function of incident power [W].

    Tabulated on ``powers`` with the given model and linearly interpolated
    (clamped outside the table). Powers for which no flow in
    ``[0.5, 30] kg/s`` reaches ``T_set`` are left out.
    """
    table_Q, table_m = [], []
    for Q in np.asarray(powers, dtype=float):
        try:
            table_m.append(float(steady_state(model, Q, T_in, T_set).u[0]))
            table_Q.append(Q)
        except ValueError:
            continue
    if len(table_Q) < 2:
        raise ValueError("the setpoint is reachable at fewer than two tabulated powers")
    Qs, ms = np.array(table_Q), np.array(table_m)

    def flow(Q: float) -> float:
        return float(np.interp(Q, Qs, ms))

    return flow


# --------------------------------------------------------------- configurations
@dataclass
class LoopConfiguration:
    """A configuration ready for :func:`run_closed_loop`.

    Attributes
    ----------
    name : configuration key (see :data:`CONFIGURATIONS`)
    controller : the controller
    loop_kwargs : estimator, known disturbance and state maps for the loop
    nmpc : the NMPC inside ``controller`` (for solve statistics), if any
    power_estimate_index : index of the incident-power estimate in the state
        passed to the controller (C and D), else ``None``
    """

    name: str
    controller: Any
    loop_kwargs: dict[str, Any]
    nmpc: NMPC | None = None
    power_estimate_index: int | None = None


def _pid(setup: ExperimentSetup, u0: np.ndarray, feedforward: bool) -> PIDController:
    g, ins = setup.pid, setup.instruments
    return PIDController(
        g.kp,
        g.ki,
        g.kd,
        setup.scenario.dt,
        action="direct",
        bias=float(u0[0]),
        u_min=ins.m_dot_min,
        u_max=ins.m_dot_max,
        rate_max=ins.rate_max,
        feedforward=_pass_through if feedforward else None,
    )


def _nmpc(setup: ExperimentSetup, model: ControlModel, u0: np.ndarray) -> NMPC:
    """NMPC on a 4-state model ``[m_dot, T_mix, T_tc, power-like state]``."""
    sc, ins, tun = setup.scenario, setup.instruments, setup.nmpc
    if model.nx != 4:
        raise ValueError(f"expected a 4-state model, got {model.state_names}")
    config = NMPCConfig(
        dt=sc.dt,
        horizon=tun.horizon,
        tracked_outputs=[0],
        output_weight=[tun.output_weight],
        move_weight=[tun.move_weight],
        u_min=[ins.m_dot_min],
        u_max=[ins.m_dot_max],
        u_init=[float(u0[0])],
        rate_max=[ins.rate_max],
        x_scale=[1.0, 100.0, 100.0, 1e5],
        x_offset=[0.0, sc.T_set, sc.T_set, setup.model.design_power],
        ipopt_options={"max_iter": tun.max_iter, "tol": tun.tol, "mu_init": 1e-3},
    )
    return NMPC(model, config)


def build_estimator(
    name: str, setup: ExperimentSetup, seed: int
) -> tuple[MeasurementSubset | EnsembleKalmanFilter, AugmentedModel]:
    """EnKF of configuration ``"C"`` or ``"D"`` and the augmented model it runs on.

    * C: model outputs ``[T_out_meas, m_dot_meas]``; augmented state
      ``[m_dot, T_mix, T_tc, d:Q_in]``; the plant's ``Q_signal`` is dropped by
      :class:`MeasurementSubset`.
    * D: model adds the power signal with the twin's assumed lag and unit gain,
      and an additive random-walk bias on it; augmented state
      ``[m_dot, T_mix, T_tc, Q_signal_lag, d:Q_in, bias:Q_signal]``.

    The prior is the controller model's steady state on the setpoint at the
    design power - never the plant's state. The inlet temperature is the only
    disturbance passed in (measured).
    """
    if name not in ("C", "D"):
        raise ValueError(f"no estimator for configuration {name!r}")
    sc, cfg = setup.scenario, setup.enkf
    receiver = build_controller_model(setup)
    Q_design = setup.model.design_power
    ss = steady_state(receiver, Q_design, sc.T_in, sc.T_set)
    if name == "C":
        aug = AugmentedModel(InstrumentedReceiver(receiver, None), disturbances=("Q_in",))
        z0 = aug.join(ss.x, ss.d)
        R = np.diag(setup.instruments.noise_std[:2] ** 2)
    else:
        sensor = PowerSignalModel(gain=1.0, lag=setup.model.power_signal_lag)
        aug = AugmentedModel(
            InstrumentedReceiver(receiver, sensor),
            disturbances=("Q_in",),
            output_biases=("Q_signal",),
        )
        z0 = aug.join(np.concatenate([ss.x, [Q_design]]), ss.d)
        R = np.diag(setup.instruments.noise_std**2)
    Q = np.diag([cfg.process_std[n] ** 2 for n in aug.state_names])
    P0 = np.diag([cfg.prior_std[n] ** 2 for n in aug.state_names])
    lower = np.array([-np.inf if n.startswith("bias") else 0.0 for n in aug.state_names])
    F = casadi_discretize(aug, sc.dt)

    def transition(z: np.ndarray, u: np.ndarray, d: np.ndarray) -> np.ndarray:
        return np.asarray(F(z, u, d), dtype=float).ravel()

    enkf = EnsembleKalmanFilter(
        transition,
        aug.output,
        Q,
        R,
        z0,
        P0,
        cfg.n_members,
        method=cfg.method,
        inflation=cfg.inflation,
        lower=lower,
        seed=seed,
    )
    if name == "C":
        return MeasurementSubset(enkf, outputs=(0, 1)), aug
    return enkf, aug


def build_configuration(
    name: str, setup: ExperimentSetup, u0: np.ndarray, seed: int
) -> LoopConfiguration:
    """Controller (and estimator) of configuration ``name``; see the module docstring.

    ``u0`` is the actuator position at switch-on and ``seed`` seeds the
    estimator's own random generator.
    """
    sc = setup.scenario
    if name == "A":
        return LoopConfiguration(
            name, _pid(setup, u0, False), {"known_disturbance": sc.measured_inlet}
        )
    if name in ("B_true", "B_signal"):
        flow = steady_flow_map(
            build_controller_model(setup),
            sc.T_in,
            sc.T_set,
            np.linspace(0.4, 1.4, 41) * setup.model.design_power,
        )
        rate = setup.instruments.rate_max
        if name == "B_true":
            controller = FeedforwardPID(_pid(setup, u0, True), flow, None, rate)
            return LoopConfiguration(name, controller, {"known_disturbance": sc.true_power})
        controller = FeedforwardPID(
            _pid(setup, u0, True), flow, 2, rate, setup.instruments.power_filter_time
        )
        return LoopConfiguration(name, controller, {"known_disturbance": sc.measured_inlet})
    if name in ("C", "D"):
        estimator, aug = build_estimator(name, setup, seed)
        nmpc_model = AugmentedModel(build_controller_model(setup), disturbances=("Q_in",))
        names = aug.state_names
        index = np.array([names.index(n) for n in ("m_dot", "T_mix", "T_tc", "d:Q_in")])
        mpc = _nmpc(setup, nmpc_model, u0)

        def to_state(z: np.ndarray) -> np.ndarray:
            return np.asarray(z, dtype=float)[index]

        kwargs = {
            "known_disturbance": sc.measured_inlet,
            "estimator": estimator,
            "estimate_to_state": to_state,
        }
        return LoopConfiguration(name, mpc, kwargs, nmpc=mpc, power_estimate_index=3)
    if name == "E":
        mpc = _nmpc(setup, build_truth_model(setup), u0)
        return LoopConfiguration(name, mpc, {"full_state_feedback": True}, nmpc=mpc)
    raise ValueError(f"unknown configuration {name!r}; choose from {CONFIGURATIONS}")


def simulate_configuration(
    name: str, setup: ExperimentSetup, seed: int, plant: InstrumentedReceiver | None = None
) -> tuple[ClosedLoopResult, LoopConfiguration]:
    """One closed-loop run of configuration ``name`` with noise seed ``seed``.

    ``plant`` defaults to :func:`build_plant`; tuning passes the controller
    model instead (see :func:`tuning_setup`).
    """
    sc = setup.scenario
    plant = build_plant(setup) if plant is None else plant
    x0, u0 = plant_initial_condition(setup, plant)
    cfg = build_configuration(name, setup, u0, seed=seed + 100_000)
    result = run_closed_loop(
        plant,
        cfg.controller,
        x0,
        u0,
        dt=sc.dt,
        n_steps=sc.n_steps,
        setpoint=sc.T_set,
        disturbance=sc.disturbance,
        noise_std=setup.instruments.noise_std,
        seed=seed,
        actuator_rate_max=setup.instruments.rate_max,
        **cfg.loop_kwargs,
    )
    return result, cfg


# ------------------------------------------------------------------------ metrics
@dataclass(frozen=True)
class RunMetrics:
    """Metrics of one run, all from the first power step on.

    ``two_sigma_max`` uses the *measured* outlet temperature, as Sandia's
    criterion does; the other temperature metrics use the noise-free
    thermocouple signal (``y_true``) so they measure control, not noise.

    Attributes
    ----------
    two_sigma_max : worst sliding 5-minute ``2 sigma`` of the measured outlet [K]
    sandia_pass : ``two_sigma_max <= 10 K``
    peak_deviation : signed largest deviation from the setpoint [K]
    iae : integral absolute error [K s]
    settling_time : worst over the two power changes of the time to stay
        within +/- 5 K [s] (``inf`` if never)
    total_variation : mass-flow travel ``sum |u_{k+1} - u_k|`` [kg s^-1]
    offset : mean error over the 120 s before the restore or before the end,
        whichever is larger in magnitude [K]
    solve_time_mean, solve_time_max : NMPC wall time per step [s] (NaN for PID)
    failed_solves : NMPC steps where IPOPT failed and the input was held (0 for PID)
    """

    two_sigma_max: float
    sandia_pass: bool
    peak_deviation: float
    iae: float
    settling_time: float
    total_variation: float
    offset: float
    solve_time_mean: float
    solve_time_max: float
    failed_solves: int


@dataclass(frozen=True)
class Trajectory:
    """Time histories kept for plotting (one representative seed)."""

    t: np.ndarray
    T_meas: np.ndarray
    T_true: np.ndarray
    m_dot: np.ndarray
    Q_true: np.ndarray
    Q_signal: np.ndarray
    Q_estimate: np.ndarray | None


@dataclass(frozen=True)
class RunResult:
    """Metrics (and optionally the trajectory) of one configuration and seed."""

    config: str
    seed: int
    metrics: RunMetrics
    trajectory: Trajectory | None = None


def compute_metrics(
    result: ClosedLoopResult, scenario: Scenario, nmpc: NMPC | None = None
) -> RunMetrics:
    """:class:`RunMetrics` of a closed-loop result."""
    t, r = result.t, scenario.T_set
    y_meas, y_true, u = result.y_meas[:, 0], result.y_true[:, 0], result.u[:, 0]
    t0 = scenario.t_step
    _, band = two_sigma_band(t, y_meas, SANDIA_WINDOW, t_start=t0)
    settle = []
    for start, stop in ((scenario.t_step, scenario.t_restore), (scenario.t_restore, np.inf)):
        seg = (t >= start) & (t < stop)
        if np.any(seg):
            settle.append(settling_time(t[seg], y_true[seg], r, SETTLING_BAND, start))
    offsets = []
    for stop in (scenario.t_restore, scenario.t_end):
        win = (t >= stop - OFFSET_WINDOW) & (t < stop) & (t >= t0)
        if np.any(win):
            offsets.append(float(np.mean(y_true[win] - r)))
    offset = max(offsets, key=abs)
    if nmpc is not None:
        times = np.array([rec.solve_time for rec in nmpc.solve_log])
        failed = int(sum(not rec.success for rec in nmpc.solve_log))
        st_mean, st_max = float(times.mean()), float(times.max())
    else:
        st_mean = st_max = float("nan")
        failed = 0
    two_sigma = float(band.max())
    return RunMetrics(
        two_sigma_max=two_sigma,
        sandia_pass=two_sigma <= SANDIA_LIMIT,
        peak_deviation=peak_deviation(t, y_true, r, t0),
        iae=integral_absolute_error(t, y_true, r, t0),
        settling_time=float(max(settle)),
        total_variation=float(total_variation(u[t >= t0])[0]),
        offset=offset,
        solve_time_mean=st_mean,
        solve_time_max=st_max,
        failed_solves=failed,
    )


def run_scenario(
    name: str, setup: ExperimentSetup, seed: int, keep_trajectory: bool = False
) -> RunResult:
    """Simulate one configuration and seed and return its metrics."""
    result, cfg = simulate_configuration(name, setup, seed)
    trajectory = None
    if keep_trajectory:
        q_hat = None
        if cfg.power_estimate_index is not None and result.x_hat is not None:
            q_hat = result.x_hat[:, cfg.power_estimate_index].copy()
        trajectory = Trajectory(
            t=result.t,
            T_meas=result.y_meas[:, 0],
            T_true=result.y_true[:, 0],
            m_dot=result.u[:, 0],
            Q_true=result.d[:, 0],
            Q_signal=result.y_meas[:, 2],
            Q_estimate=q_hat,
        )
    return RunResult(name, seed, compute_metrics(result, setup.scenario, cfg.nmpc), trajectory)


def _run_task(task: tuple[str, ExperimentSetup, int, bool]) -> RunResult:
    return run_scenario(*task)


def run_monte_carlo(
    setup: ExperimentSetup,
    configs: Sequence[str] = CONFIGURATIONS,
    seeds: Sequence[int] = tuple(range(20)),
    workers: int = 1,
    trajectory_seed: int | None = None,
) -> list[RunResult]:
    """Run every configuration on every noise seed.

    Runs are independent, so with ``workers > 1`` they are spread over that
    many processes; each returns only its metrics (and the trajectory for
    ``trajectory_seed``). Memory is about ``workers`` x 0.8 GB at worst (an
    NMPC process peaks at 0.5-0.8 GB, a PID one at 0.2 GB). NMPC runs are
    submitted first for load balance.
    Results come back in ``(config, seed)`` order.
    """
    tasks = [(c, setup, int(s), s == trajectory_seed) for c in configs for s in seeds]
    return _run_tasks(tasks, workers)


def _run_tasks(
    tasks: list[tuple[str, ExperimentSetup, int, bool]], workers: int
) -> list[RunResult]:
    """Run :func:`run_scenario` on each task, NMPC runs first; results in task order."""
    order = sorted(range(len(tasks)), key=lambda i: tasks[i][0] not in NMPC_CONFIGURATIONS)
    if workers <= 1:
        out = {i: _run_task(tasks[i]) for i in order}
    else:
        # A fresh process per task returns each NMPC's memory (~0.5-0.8 GB) to the OS.
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx, max_tasks_per_child=1
        ) as pool:
            futures = {i: pool.submit(_run_task, tasks[i]) for i in order}
            out = {i: f.result() for i, f in futures.items()}
    return [out[i] for i in range(len(tasks))]


# ---------------------------------------------------------------------- summaries
SUMMARY_FIELDS = (
    "two_sigma_max",
    "peak_deviation",
    "iae",
    "settling_time",
    "total_variation",
    "offset",
    "solve_time_mean",
    "solve_time_max",
)


@dataclass(frozen=True)
class ConfigSummary:
    """Monte Carlo statistics of one configuration.

    ``mean`` and ``std`` (sample, ``ddof=1``) map each name in
    :data:`SUMMARY_FIELDS` to its statistic over seeds (``peak_deviation`` as a
    magnitude, since its sign varies between seeds); ``pass_rate`` is the
    fraction of seeds meeting the Sandia criterion; ``failed_solves`` the total
    over seeds; ``worst`` the maximum over seeds of ``two_sigma_max`` and of
    ``solve_time_max``.
    """

    config: str
    n_runs: int
    mean: dict[str, float]
    std: dict[str, float]
    pass_rate: float
    failed_solves: int
    worst: dict[str, float]


def summarise(results: Sequence[RunResult]) -> dict[str, ConfigSummary]:
    """Group run results by configuration and compute :class:`ConfigSummary`."""
    by_config: dict[str, list[RunMetrics]] = {}
    for res in results:
        by_config.setdefault(res.config, []).append(res.metrics)
    out = {}
    for name, runs in by_config.items():
        values = {f: np.array([getattr(m, f) for m in runs], dtype=float) for f in SUMMARY_FIELDS}
        # The sign of the worst excursion flips between seeds (step-down or restore),
        # so its magnitude is what is averaged.
        values["peak_deviation"] = np.abs(values["peak_deviation"])
        std = {f: float(v.std(ddof=1)) if v.size > 1 else 0.0 for f, v in values.items()}
        out[name] = ConfigSummary(
            config=name,
            n_runs=len(runs),
            mean={f: float(v.mean()) for f, v in values.items()},
            std=std,
            pass_rate=float(np.mean([m.sandia_pass for m in runs])),
            failed_solves=int(sum(m.failed_solves for m in runs)),
            worst={
                "two_sigma_max": float(values["two_sigma_max"].max()),
                "solve_time_max": float(values["solve_time_max"].max()),
            },
        )
    return out


def format_results_table(summaries: dict[str, ConfigSummary]) -> str:
    """Plain-text table: mean +/- std over seeds for every configuration."""

    def pm(s: ConfigSummary, f: str, scale: float = 1.0, digits: int = 1) -> str:
        m, sd = s.mean[f] * scale, s.std[f] * scale
        if not np.isfinite(m):
            return "-"
        return f"{m:.{digits}f} +/- {sd:.{digits}f}"

    head = (
        f"{'configuration':38s} {'2sigma_max [K]':>16s} {'pass':>5s} {'|peak| [K]':>15s} "
        f"{'IAE [K s]':>17s} {'settle [s]':>15s} {'TV [kg/s]':>14s} {'offset [K]':>15s} "
        f"{'solve mean/max [ms]':>20s} {'fail':>5s}"
    )
    lines = [head, "-" * len(head)]
    for name in CONFIGURATIONS:
        if name not in summaries:
            continue
        s = summaries[name]
        solve = "-"
        if np.isfinite(s.mean["solve_time_mean"]):
            solve = f"{1e3 * s.mean['solve_time_mean']:.0f} / {1e3 * s.worst['solve_time_max']:.0f}"
        lines.append(
            f"{LABELS[name]:38s} {pm(s, 'two_sigma_max', digits=2):>16s} "
            f"{100 * s.pass_rate:4.0f}% {pm(s, 'peak_deviation'):>15s} "
            f"{pm(s, 'iae', digits=0):>17s} "
            f"{pm(s, 'settling_time', digits=0):>15s} {pm(s, 'total_variation', digits=2):>14s} "
            f"{pm(s, 'offset', digits=2):>15s} {solve:>20s} {s.failed_solves:5d}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------- PID tuning
@dataclass(frozen=True)
class PIDTuningResult:
    """Outcome of :func:`tune_pid`."""

    gains: PIDGains
    initial_gains: PIDGains
    objective: float
    initial_objective: float
    n_evaluations: int


def tuning_setup(setup: ExperimentSetup) -> tuple[ExperimentSetup, InstrumentedReceiver]:
    """The set-up the PID is tuned on: the *controller model* as plant, step-down only.

    The PID is tuned with the same information as the twin: the coarse nominal
    model (plus a power-signal sensor with the twin's assumed lag and unit
    gain, which the PID does not use), the same sensor noise, the same
    actuator limits, and the same step, truncated at the restore.
    """
    sc = dataclasses.replace(setup.scenario, t_end=setup.scenario.t_restore)
    sensor = PowerSignalModel(gain=1.0, lag=setup.model.power_signal_lag)
    return dataclasses.replace(setup, scenario=sc), InstrumentedReceiver(
        build_controller_model(setup), sensor
    )


def pid_objective(
    gains: PIDGains, setup: ExperimentSetup, model: InstrumentedReceiver, seeds: Sequence[int]
) -> float:
    """Mean IAE after the step [K s] of configuration A with ``gains`` on ``model``."""
    return tuning_objective("A", dataclasses.replace(setup, pid=gains), model, seeds)


def tuning_objective(
    name: str, setup: ExperimentSetup, model: InstrumentedReceiver, seeds: Sequence[int]
) -> float:
    """Mean IAE after the step [K s] of configuration ``name`` with ``model`` as the plant."""
    sc = setup.scenario
    iae = []
    for seed in seeds:
        res, _ = simulate_configuration(name, setup, seed, plant=model)
        iae.append(integral_absolute_error(res.t, res.y_true[:, 0], sc.T_set, sc.t_step))
    return float(np.mean(iae))


def tune_pid(
    setup: ExperimentSetup,
    seeds: Sequence[int] = (1001, 1002),
    initial: PIDGains | None = None,
    max_evaluations: int = 150,
) -> PIDTuningResult:
    """Tune PID gains by minimising IAE on the controller model (Nelder-Mead in log gains).

    Objective: mean IAE of the noise-free thermocouple signal after the power
    step, over ``seeds`` (tuning seeds, disjoint from evaluation seeds), with
    measurement noise and the actuator limits active - so gains that amplify
    noise into gate chatter are penalised through the temperature error they
    cause. The plant is never used. The default initial guess is the SIMC
    rule (Skogestad 2003) for the controller model: gain ``-36 K/(kg/s)``,
    dominant lag 10 s, effective delay ~6 s, ``tau_c = 6 s`` gives
    ``Kp ~ 0.02 kg/s/K``, ``Ti = 10 s``; plus ``Td = 1 s``.
    """
    tune_set, model = tuning_setup(setup)
    start = initial if initial is not None else PIDGains(0.02, 0.002, 0.02)

    def unpack(v: np.ndarray) -> PIDGains:
        kp, ki, kd = 10.0 ** np.asarray(v, dtype=float)
        return PIDGains(float(kp), float(ki), float(kd))

    def cost(v: np.ndarray) -> float:
        return pid_objective(unpack(v), tune_set, model, seeds)

    v0 = np.log10([start.kp, start.ki, start.kd])
    j0 = cost(v0)
    res = minimize(
        cost,
        v0,
        method="Nelder-Mead",
        options={"maxfev": max_evaluations, "xatol": 0.01, "fatol": 0.5},
    )
    return PIDTuningResult(unpack(res.x), start, float(res.fun), j0, int(res.nfev) + 1)


def _move_weight_task(task: tuple[float, ExperimentSetup, tuple[int, ...]]) -> float:
    weight, setup, seeds = task
    tune_set, model = tuning_setup(setup)
    trial = dataclasses.replace(tune_set, nmpc=dataclasses.replace(setup.nmpc, move_weight=weight))
    return tuning_objective("C", trial, model, seeds)


def tune_move_weight(
    setup: ExperimentSetup,
    candidates: Sequence[float] = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0),
    seeds: Sequence[int] = (1001,),
    workers: int = 1,
) -> dict[float, float]:
    """IAE of configuration C for each NMPC move weight, on the controller model.

    The same objective and the same tuning plant as :func:`tune_pid` (the
    controller model with noise and limits, step-down only), so the PID and
    the twin are tuned on equal terms; the plant is never used. Returns
    ``{move_weight: mean IAE}``; pick the minimum.
    """
    tasks = [(float(w), setup, tuple(seeds)) for w in candidates]
    if workers <= 1:
        values = [_move_weight_task(t) for t in tasks]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx, max_tasks_per_child=1
        ) as pool:
            values = list(pool.map(_move_weight_task, tasks))
    return dict(zip((t[0] for t in tasks), values, strict=True))


def tune_power_filter(
    setup: ExperimentSetup,
    candidates: Sequence[float] = (0.0, 1.0, 3.0, 5.0, 10.0, 20.0),
    seeds: Sequence[int] = (1001, 1002),
) -> dict[float, float]:
    """IAE of configuration ``B_signal`` for each power-filter time [s], on the controller model.

    Same objective and tuning plant as :func:`tune_pid`; the tuning plant's
    power signal has the twin's assumed lag and no gain error. Returns
    ``{filter_time: mean IAE}``; pick the minimum.
    """
    tune_set, model = tuning_setup(setup)
    out = {}
    for tf in candidates:
        ins = dataclasses.replace(setup.instruments, power_filter_time=float(tf))
        trial = dataclasses.replace(tune_set, instruments=ins)
        out[float(tf)] = tuning_objective("B_signal", trial, model, seeds)
    return out


# ------------------------------------------------- observability and detection
@dataclass(frozen=True)
class ObservabilityRow:
    """Local observability of one estimator model at one operating point."""

    sensors: str
    power: float
    rank: int
    n_states: int
    condition_number: float
    smallest_singular_value: float

    @property
    def observable(self) -> bool:
        return self.rank == self.n_states


def observability_report(setup: ExperimentSetup) -> list[ObservabilityRow]:
    """Is the incident power observable on the augmented controller model?

    Three sensor sets: thermocouple only; thermocouple and mass flow (C);
    thermocouple, mass flow and power signal with a bias state (D). Each is
    analysed at the controller model's steady state on the setpoint at the
    pre- and post-step powers. States are scaled by typical ranges (0.1 kg/s,
    1 K, 1 K, 10 kW for powers and bias), outputs by their noise standard
    deviations, so the condition number compares signal-to-noise directly.
    """
    sc, ins = setup.scenario, setup.instruments
    receiver = build_controller_model(setup)
    sensor = PowerSignalModel(gain=1.0, lag=setup.model.power_signal_lag)
    models = {
        "T_out": AugmentedModel(receiver, disturbances=("Q_in",)),
        "T_out + m_dot (C)": AugmentedModel(
            InstrumentedReceiver(receiver, None), disturbances=("Q_in",)
        ),
        "T_out + m_dot + Q_signal (D)": AugmentedModel(
            InstrumentedReceiver(receiver, sensor),
            disturbances=("Q_in",),
            output_biases=("Q_signal",),
        ),
    }
    typical = {"m_dot": 0.1, "T_mix": 1.0, "T_tc": 1.0}
    noise = dict(zip(("T_out_meas", "m_dot_meas", "Q_signal"), ins.noise_std, strict=True))
    rows = []
    for label, aug in models.items():
        x_scale = np.array([typical.get(n, 1e4) for n in aug.state_names])
        y_scale = np.array([noise[n] for n in aug.output_names])
        for Q in (sc.Q_before, sc.Q_after):
            ss = steady_state(receiver, Q, sc.T_in, sc.T_set)
            x = ss.x if aug.model.nx == 3 else np.concatenate([ss.x, [Q]])
            z = aug.join(x, ss.d)
            res = observability(
                aug, z, ss.u, aug.measured_disturbances(ss.d), sc.dt,
                state_scale=x_scale, output_scale=y_scale,
            )  # fmt: skip
            rows.append(
                ObservabilityRow(
                    label,
                    Q,
                    res.rank,
                    res.n_states,
                    res.condition_number,
                    float(res.singular_values[-1]),
                )  # fmt: skip
            )
    return rows


@dataclass(frozen=True)
class DetectionResult:
    """Open-loop response of the power estimate to the step (mass flow held).

    Times are measured from the step [s]. ``t_detect`` is the first time the
    estimate leaves its pre-step mean by more than 3 pre-step standard
    deviations (towards the step); ``t50`` and ``t90`` are the times it has
    covered 50 % and 90 % of its own change (pre-step mean to the mean of the
    last 100 s). ``tc_t50`` and ``tc_t90`` give the same for the noise-free
    thermocouple, for comparison. ``final_error`` is the settled estimate minus
    the true post-step power [W]: with model mismatch the estimate converges to
    the power that makes the *model* match the measurements, not to the truth.
    """

    config: str
    seed: int
    t_detect: float
    t50: float
    t90: float
    tc_t50: float
    tc_t90: float
    final_error: float
    pre_step_error: float


def _first_crossing(t: np.ndarray, progress: np.ndarray, level: float) -> float:
    idx = np.flatnonzero(progress >= level)
    return float(t[idx[0]]) if idx.size else float("inf")


def estimator_detection(name: str, setup: ExperimentSetup, seed: int) -> DetectionResult:
    """Run the C or D estimator on the plant with the gate held, through the step-down only."""
    sc = dataclasses.replace(setup.scenario, t_end=setup.scenario.t_restore)
    plant = build_plant(setup)
    x0, u0 = plant_initial_condition(setup, plant)
    estimator, aug = build_estimator(name, setup, seed + 100_000)
    result = run_closed_loop(
        plant,
        HoldInput(u0),
        x0,
        u0,
        dt=sc.dt,
        n_steps=sc.n_steps,
        setpoint=sc.T_set,
        disturbance=sc.disturbance,
        noise_std=setup.instruments.noise_std,
        seed=seed,
        known_disturbance=sc.measured_inlet,
        estimator=estimator,
    )
    t = result.t
    q_hat = result.x_est[:, aug.state_names.index("d:Q_in")]
    pre = (t >= sc.t_step - 100.0) & (t < sc.t_step)
    post_win = t >= sc.t_end - 100.0
    after = t >= sc.t_step
    q_pre, q_pre_std, q_post = q_hat[pre].mean(), q_hat[pre].std(ddof=1), q_hat[post_win].mean()
    sign = np.sign(q_post - q_pre)
    progress = (q_hat[after] - q_pre) / (q_post - q_pre)
    t_after = t[after] - sc.t_step
    departed = np.flatnonzero(sign * (q_hat[after] - q_pre) > 3.0 * q_pre_std)
    y = result.y_true[:, 0]
    y_pre, y_post = y[pre].mean(), y[post_win].mean()
    tc_progress = (y[after] - y_pre) / (y_post - y_pre)
    return DetectionResult(
        config=name,
        seed=seed,
        t_detect=float(t_after[departed[0]]) if departed.size else float("inf"),
        t50=_first_crossing(t_after, progress, 0.5),
        t90=_first_crossing(t_after, progress, 0.9),
        tc_t50=_first_crossing(t_after, tc_progress, 0.5),
        tc_t90=_first_crossing(t_after, tc_progress, 0.9),
        final_error=float(q_post - sc.Q_after),
        pre_step_error=float(q_pre - sc.Q_before),
    )


# --------------------------------------------------------------- sensitivity
@dataclass(frozen=True)
class SensitivityPoint:
    """Monte Carlo summary at one plant-mismatch level or step size."""

    label: str
    h_adv_ratio: float
    step_fraction: float
    summaries: dict[str, ConfigSummary]


def run_sensitivity(
    setup: ExperimentSetup,
    h_adv_ratios: Sequence[float] = (1.0, 1.2, 1.5, 2.0),
    large_step_fraction: float = 0.30,
    configs: Sequence[str] = ("A", "C", "E"),
    seeds: Sequence[int] = tuple(range(10)),
    workers: int = 1,
) -> list[SensitivityPoint]:
    """Vary the plant's advective-loss error, then use a larger power step.

    ``h_adv_ratios`` are plant ``h_adv`` over the controller model's; the plant
    lags keep their mismatch, and the controllers (PID gains, NMPC weights,
    EnKF settings) are unchanged - nobody re-tunes for an error they do not
    know about. The large-step point uses ``Q_after = (1 - fraction) Q_before``
    at the default plant.
    """
    jobs: list[tuple[str, ExperimentSetup, float, float]] = []
    h_model = setup.model.h_adv
    sc = setup.scenario
    for ratio in h_adv_ratios:
        s = dataclasses.replace(
            setup, plant=dataclasses.replace(setup.plant, h_adv=ratio * h_model)
        )
        jobs.append((f"h_adv x{ratio:g}", s, ratio, 1.0 - sc.Q_after / sc.Q_before))
    big = dataclasses.replace(
        setup, scenario=dataclasses.replace(sc, Q_after=(1.0 - large_step_fraction) * sc.Q_before)
    )
    jobs.append(
        (
            f"{100 * large_step_fraction:.0f} % step",
            big,
            setup.plant.h_adv / h_model,
            large_step_fraction,
        )
    )
    tasks = [(c, s, int(seed), False) for _, s, _, _ in jobs for c in configs for seed in seeds]
    out = _run_tasks(tasks, workers)
    per_job = len(configs) * len(seeds)
    points = []
    for j, (label, _, ratio, frac) in enumerate(jobs):
        chunk = out[j * per_job : (j + 1) * per_job]
        points.append(SensitivityPoint(label, ratio, frac, summarise(chunk)))
    return points


def default_workers() -> int:
    """Worker processes for the Monte Carlo: at most 4 (memory, ~3 GB), at most CPUs - 2."""
    return max(1, min(4, (os.cpu_count() or 2) - 2))
