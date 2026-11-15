"""Feedback control for the digital twin: PID, nonlinear MPC, closed-loop simulation, metrics.

Everything here uses only the protocols in :mod:`pcsp.core.interfaces`, so the
same controllers run on a reduced physics model today and on a ROM or neural
surrogate later.
"""

from pcsp.twin.control.closed_loop import ClosedLoopResult, run_closed_loop
from pcsp.twin.control.metrics import (
    RegulationMetrics,
    integral_absolute_error,
    integral_squared_error,
    overshoot,
    peak_deviation,
    regulation_metrics,
    settling_time,
    total_variation,
    two_sigma_band,
)
from pcsp.twin.control.mpc import NMPC, NMPCConfig, SolveRecord
from pcsp.twin.control.pid import PIDController
from pcsp.twin.control.profiles import Profile, as_profile, constant, piecewise_constant, ramp, step

__all__ = [
    "NMPC",
    "ClosedLoopResult",
    "NMPCConfig",
    "PIDController",
    "Profile",
    "RegulationMetrics",
    "SolveRecord",
    "as_profile",
    "constant",
    "integral_absolute_error",
    "integral_squared_error",
    "overshoot",
    "peak_deviation",
    "piecewise_constant",
    "ramp",
    "regulation_metrics",
    "run_closed_loop",
    "settling_time",
    "step",
    "total_variation",
    "two_sigma_band",
]
