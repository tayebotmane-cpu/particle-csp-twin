"""Control-oriented dynamic model of the falling-particle receiver.

Design
------
Particles cross the irradiated cavity in well under a second (residence time
~0.6 s for the default 2.25 m fall, see ``simulate(...).residence_time``),
whereas the measured outlet temperature responds over many seconds. The
curtain is therefore treated as *quasi-steady*: at every instant its outlet
temperature is the steady 1-D solution of :mod:`pcsp.models.receiver.curtain`
(the same code, by explicit marching, at a chosen resolution). The slower
dynamics between the slide-gate command and the thermocouple reading are
three first-order lags:

.. math::
    \\dot{\\dot m} &= (u - \\dot m) / \\tau_{gate}
        && \\text{slide-gate actuator and discharge [kg s}^{-2}\\text{]} \\\\
    c_p(T_{mix})\\, \\dot T_{mix} &= \\big(h(T_{curtain}(\\dot m, Q_{in}, T_{in}))
        - h(T_{mix})\\big) / \\tau_{mix}
        && \\text{well-mixed outlet funnel [W kg}^{-1}\\text{]} \\\\
    \\dot T_{tc} &= (T_{mix} - T_{tc}) / \\tau_{tc}
        && \\text{thermocouple [K s}^{-1}\\text{]}

The funnel equation is the energy balance of a perfectly mixed hold-up of
mass ``M`` with throughput ``m_dot`` and residence time ``tau_mix = M /
m_dot``, written in enthalpy so that mixing is exact for temperature-dependent
cp; ``tau_mix`` is held constant (see assumptions). The output is
``y = T_tc``. In steady state ``T_tc = T_mix = T_curtain``, so the dynamic
model's equilibrium is exactly the Part 1 steady model.

Time constants
--------------
Sandia report that the slide gate "can move to any desired opening in less
than one second" and that the time to reach a desired outlet setpoint was
"less than one minute" with proportional control (Ho et al., SolarPACES
2020, citing Ho et al. 2018). Mills et al. (Energies 15:1657, 2022) describe
the outlet thermocouples as sitting in funnels designed to keep them
immersed "while minimizing lag", but no numerical lag is given in the
sources that could be accessed. The ES2021 control paper and AIP Conf. Proc.
2033/2126 papers (OSTI) could not be retrieved. Hence:

* ``tau_gate = 1 s`` - assumption, consistent with "< 1 s" gate travel;
* ``tau_mix = 10 s`` - assumption (funnel hold-up), unverified;
* ``tau_tc = 5 s`` - assumption (sheathed thermocouple immersed in a moving
  particle bed), unverified.

Their sum (~16 s) is consistent with a closed-loop settling time below one
minute, which is the only quantitative check available.

Numerics
--------
``rhs`` and ``output`` are backend-agnostic (NumPy or CasADi SX): explicit
marching, no implicit solves, no Python branching on states. The fastest
eigenvalue is ``-1/tau_gate``; classical RK4 is stable for ``|lambda| dt <
2.78``, so a sample time of 1 s with one RK4 substep is stable with margin.
"""

from __future__ import annotations

import dataclasses
from typing import Any, NamedTuple

import numpy as np
from scipy.optimize import brentq

from pcsp.core.backend import ops_for
from pcsp.core.interfaces import ControlModelBase
from pcsp.models.receiver.curtain import FPRParams, outlet_temperature


class SteadyState(NamedTuple):
    """An equilibrium ``rhs(x, u, d) = 0`` of :class:`FPRControlModel`."""

    x: np.ndarray
    u: np.ndarray
    d: np.ndarray


class FPRControlModel(ControlModelBase):
    """Falling-particle receiver for outlet-temperature control.

    =========  ==============  ===========================================  =======
    vector     name            meaning                                      units
    =========  ==============  ===========================================  =======
    x[0]       ``m_dot``       actual particle mass flow through curtain    kg s-1
    x[1]       ``T_mix``       bulk temperature in the outlet funnel        K
    x[2]       ``T_tc``        outlet thermocouple reading                  K
    u[0]       ``m_dot_cmd``   commanded mass flow (slide gate)             kg s-1
    d[0]       ``Q_in``        solar power entering the aperture            W
    d[1]       ``T_in``        particle inlet temperature                   K
    y[0]       ``T_out_meas``  measured particle outlet temperature         K
    =========  ==============  ===========================================  =======

    Parameters
    ----------
    params : curtain geometry, losses and material (operating-point fields
        ``mass_flow``, ``T_in`` and ``Q_in`` are ignored; they come from x, d)
    n_segments : curtain resolution. Fine (e.g. 50) for a plant, coarse (e.g.
        5-10) for a controller model.
    tau_gate, tau_mix, tau_tc : time constants [s] (assumptions, see module)
    """

    state_names = ("m_dot", "T_mix", "T_tc")
    input_names = ("m_dot_cmd",)
    disturbance_names = ("Q_in", "T_in")
    output_names = ("T_out_meas",)

    def __init__(
        self,
        params: FPRParams | None = None,
        n_segments: int = 20,
        tau_gate: float = 1.0,
        tau_mix: float = 10.0,
        tau_tc: float = 5.0,
    ) -> None:
        if n_segments < 1:
            raise ValueError(f"n_segments must be >= 1, got {n_segments}")
        for name, tau in (("tau_gate", tau_gate), ("tau_mix", tau_mix), ("tau_tc", tau_tc)):
            if tau <= 0.0:
                raise ValueError(f"{name} must be positive, got {tau}")
        base = params if params is not None else FPRParams()
        self.params = dataclasses.replace(base, n_segments=n_segments)
        self.n_segments = n_segments
        self.tau_gate = tau_gate
        self.tau_mix = tau_mix
        self.tau_tc = tau_tc

    def curtain_outlet(self, m_dot: Any, Q_in: Any, T_in: Any) -> Any:
        """Quasi-steady curtain outlet temperature [K] (Part 1 model)."""
        return outlet_temperature(self.params, self.n_segments, m_dot, T_in, Q_in)

    def rhs(self, x: Any, u: Any, d: Any) -> Any:
        """``dx/dt`` [kg s^-2, K s^-1, K s^-1]."""
        xp = ops_for(x, u, d)
        mat = self.params.material
        T_curtain = self.curtain_outlet(x[0], d[0], d[1])
        dm = (u[0] - x[0]) / self.tau_gate
        dT_mix = (mat.enthalpy(T_curtain) - mat.enthalpy(x[1])) / (self.tau_mix * mat.cp(x[1]))
        dT_tc = (x[1] - x[2]) / self.tau_tc
        return xp.concat([dm, dT_mix, dT_tc])

    def output(self, x: Any, u: Any, d: Any) -> Any:
        """Measured outlet temperature ``[T_tc]`` [K]."""
        xp = ops_for(x, u, d)
        return xp.concat([x[2]])

    def equilibrium(self, m_dot: float, Q_in: float, T_in: float) -> SteadyState:
        """Steady state for a given mass flow [kg s^-1], ``Q_in`` [W], ``T_in`` [K]."""
        T = float(self.curtain_outlet(m_dot, Q_in, T_in))
        return SteadyState(np.array([m_dot, T, T]), np.array([m_dot]), np.array([Q_in, T_in]))


def steady_state(
    model: FPRControlModel,
    Q_in: float,
    T_in: float,
    T_out: float,
    m_dot_bounds: tuple[float, float] = (0.5, 30.0),
) -> SteadyState:
    """Equilibrium that delivers outlet temperature ``T_out`` [K].

    Solves ``T_curtain(m_dot) = T_out`` for the mass flow [kg s^-1] by
    Brent's method on ``m_dot_bounds`` (the curtain outlet temperature falls
    monotonically with mass flow) and returns ``x``, ``u = [m_dot]`` and
    ``d = [Q_in, T_in]``. Raises ``ValueError`` if the target cannot be
    reached within the bounds.
    """
    lo, hi = m_dot_bounds

    def mismatch(m: float) -> float:
        return float(model.curtain_outlet(m, Q_in, T_in)) - T_out

    f_lo, f_hi = mismatch(lo), mismatch(hi)
    if f_lo * f_hi > 0.0:
        raise ValueError(
            f"T_out = {T_out:.1f} K is not reachable with m_dot in [{lo}, {hi}] kg/s "
            f"(outlet spans {f_hi + T_out:.1f}-{f_lo + T_out:.1f} K)"
        )
    m_dot = brentq(mismatch, lo, hi, xtol=1e-12, rtol=1e-14)
    return model.equilibrium(m_dot, Q_in, T_in)
