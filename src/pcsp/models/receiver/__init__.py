"""Reduced-order falling-particle receiver (FPR) models.

* :mod:`.curtain` - steady 1-D curtain model, :func:`simulate`.
* :mod:`.dynamic` - control-oriented model (:class:`FPRControlModel`) and
  :func:`steady_state`.
* :mod:`.validation` - comparison with published Sandia on-sun tests.
"""

from pcsp.models.receiver.curtain import FPRParams, simulate
from pcsp.models.receiver.dynamic import FPRControlModel, SteadyState, steady_state

__all__ = ["FPRControlModel", "FPRParams", "SteadyState", "simulate", "steady_state"]
