"""Turbulent forced convection and friction in the sCO2 channels.

Gnielinski (1976) correlation with the Petukhov (1970) smooth-tube friction
factor, using bulk properties (no property-ratio correction):

.. math::
    f_D = (0.790 \\ln Re - 1.64)^{-2}, \\qquad
    Nu = \\frac{(f_D/8)(Re - 1000) Pr}{1 + 12.7 \\sqrt{f_D/8}\\,(Pr^{2/3} - 1)} ,

with :math:`Re = \\rho u D / \\mu` and :math:`Nu = h D / k` on the channel
hydraulic diameter ``D``. Quoted validity: :math:`3000 \\le Re \\le 5\\times10^6`,
:math:`0.5 \\le Pr \\le 2000` (as stated in Incropera et al., Fundamentals of
Heat and Mass Transfer, eq. 8.62 and 8.21; not re-checked against the
original papers here). Outside that range :func:`nusselt_gnielinski` raises
rather than extrapolate.

The frictional pressure gradient is the Darcy-Weisbach law
:math:`-dp/dz = f_D \\, G^2 / (2 \\rho D)` with mass flux ``G``
[kg m^-2 s^-1].

References: Gnielinski, V., 1976. New equations for heat and mass transfer in
turbulent pipe and channel flow. Int. Chem. Eng. 16, 359-368. Petukhov, B.S.,
1970. Heat transfer and friction in turbulent pipe flow with variable physical
properties. Adv. Heat Transfer 6, 503-564.
"""

from __future__ import annotations

import numpy as np

RE_MIN = 3000.0
RE_MAX = 5e6
PR_MIN = 0.5
PR_MAX = 2000.0


def friction_factor_petukhov(Re: np.ndarray | float) -> np.ndarray:
    """Darcy friction factor of a smooth tube [-]."""
    return (0.790 * np.log(np.asarray(Re, dtype=float)) - 1.64) ** -2


def nusselt_gnielinski(Re: np.ndarray | float, Pr: np.ndarray | float) -> np.ndarray:
    """Fully developed turbulent Nusselt number [-]; raises outside the validity range."""
    Re = np.asarray(Re, dtype=float)
    Pr = np.asarray(Pr, dtype=float)
    if np.any(Re < RE_MIN) or np.any(Re > RE_MAX):
        raise ValueError(
            f"Gnielinski correlation needs {RE_MIN:g} <= Re <= {RE_MAX:g}; got Re in "
            f"[{np.min(Re):.4g}, {np.max(Re):.4g}]"
        )
    if np.any(Pr < PR_MIN) or np.any(Pr > PR_MAX):
        raise ValueError(f"Gnielinski correlation needs {PR_MIN} <= Pr <= {PR_MAX}")
    f8 = friction_factor_petukhov(Re) / 8.0
    return f8 * (Re - 1000.0) * Pr / (1.0 + 12.7 * np.sqrt(f8) * (Pr ** (2.0 / 3.0) - 1.0))
