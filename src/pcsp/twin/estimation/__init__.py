"""State and disturbance estimation for the digital twin.

* :class:`EnsembleKalmanFilter` - primary method; needs only forward simulations.
* :class:`KalmanFilter` - exact linear filter, the verification reference.
* :class:`ExtendedKalmanFilter` - nonlinear baseline with CasADi Jacobians.
* :class:`AugmentedModel` - turns unknown disturbances and output biases into
  random-walk states so any filter estimates them (offset-free MPC).
* :mod:`~pcsp.twin.estimation.diagnostics` - NEES/NIS consistency tests, RMSE
  and local observability analysis.

All filters implement :class:`pcsp.core.interfaces.Estimator` and additionally
expose ``innovation`` and ``innovation_cov`` after each update (for NIS).
"""

from pcsp.twin.estimation.augment import AugmentedModel
from pcsp.twin.estimation.diagnostics import (
    ObservabilityResult,
    chi2_bounds,
    nees,
    nis,
    observability,
    rmse,
)
from pcsp.twin.estimation.enkf import EnsembleKalmanFilter
from pcsp.twin.estimation.kalman import ExtendedKalmanFilter, KalmanFilter, kalman_update

__all__ = [
    "AugmentedModel",
    "EnsembleKalmanFilter",
    "ExtendedKalmanFilter",
    "KalmanFilter",
    "ObservabilityResult",
    "chi2_bounds",
    "kalman_update",
    "nees",
    "nis",
    "observability",
    "rmse",
]
