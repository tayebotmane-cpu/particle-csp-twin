"""Non-intrusive parametric ROM: POD + radial-basis-function interpolation.

Snapshot convention: a snapshot matrix has shape ``(n_dof, n_snapshots)``.

Time handling: space-time snapshots
-----------------------------------
Each training run ``k`` with parameters ``mu_k`` contributes **one** column: its
whole trajectory ``T(t_i, z_j)`` flattened to length ``n_dof = n_time * n_space``
(see :func:`pcsp.rom.snapshots.to_snapshots` with ``sample_dims="run"``). The
model is

    x(mu) ~= xbar + Phi a(mu),   a(mu) = RBF interpolant of {(mu_hat_k, a_k)},

with POD modes ``Phi`` of the space-time snapshot matrix (``rank <= n_runs``),
training coefficients ``a_k = Phi^T W (x_k - xbar)`` and inputs normalised to the
unit box of the training data, ``mu_hat = (mu - mu_min) / (mu_max - mu_min)``.
``mu_hat`` outside ``[0, 1]`` therefore means extrapolation.

Why not time as an extra RBF input? That approach needs one RBF centre per
(run, time) pair; for a few hundred runs times a few hundred output times the dense
RBF system has ~10^5 unknowns (tens of GB), and it mixes a physical time scale
with parameter scales in one isotropic kernel. Space-time snapshots keep the RBF
problem at ``n_runs`` centres in parameter space only, reproduce the training
trajectories exactly at the training time grid, and keep the temporal correlation
of each trajectory inside the modes. The price: all runs must share one output
time grid, and predictions are on that grid only (interpolate in time afterwards
if needed). With spatial weights ``w_z`` use ``weights=np.tile(w_z, n_time)``.

Kernel: the default thin-plate spline ``phi(r) = r^2 log r`` with a degree-1
polynomial tail has no shape parameter to tune and is conditionally positive
definite of order 2, so the interpolation system is uniquely solvable for
distinct centres (Wendland 2005, ch. 8). ``smoothing > 0`` gives a regularised
(non-interpolating) fit for noisy data.

References
----------
Bui-Thanh, Damodaran & Willcox (2004), "Aerodynamic data reconstruction and
inverse design using proper orthogonal decomposition", AIAA J. 42(8), 1505-1516.
Audouze, De Vuyst & Nair (2009), "Reduced-order modeling of parameterized PDEs
using time-space-parameter principal component analysis", IJNME 80(8), 1025-1057.
Wendland (2005), *Scattered Data Approximation*, Cambridge University Press.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import RBFInterpolator

from pcsp.rom.pod import POD, fit_pod


@dataclass(frozen=True)
class PODInterpolationROM:
    """POD + RBF interpolation surrogate. Build it with :meth:`fit`.

    Attributes
    ----------
    pod : the space-time POD basis (snapshot matrix shape ``(n_dof, n_runs)``).
    interpolator : maps normalised parameters ``(n, n_params)`` to coefficients ``(n, rank)``.
    param_min, param_range : per-parameter normalisation of the training box.
    """

    pod: POD
    interpolator: RBFInterpolator
    param_min: np.ndarray
    param_range: np.ndarray

    @classmethod
    def fit(
        cls,
        params: ArrayLike,
        snapshots: ArrayLike,
        *,
        rank: int | None = None,
        energy: float | None = None,
        center: bool = True,
        weights: ArrayLike | None = None,
        kernel: str = "thin_plate_spline",
        smoothing: float = 0.0,
        epsilon: float | None = None,
        degree: int | None = None,
    ) -> PODInterpolationROM:
        """Fit the surrogate.

        Parameters
        ----------
        params : ``(n_runs, n_params)`` parameter table, one row per training run.
        snapshots : ``(n_dof, n_runs)`` snapshot matrix; column ``k`` is the full
            (space-time) output of run ``k``.
        rank, energy, center, weights : passed to :func:`pcsp.rom.pod.fit_pod`.
        kernel, smoothing, epsilon, degree : passed to
            :class:`scipy.interpolate.RBFInterpolator` (``epsilon`` is required by
            kernels with a shape parameter, e.g. ``"gaussian"``).
        """
        P = _as_param_table(params)
        X = np.asarray(snapshots, dtype=float)
        if X.ndim != 2 or X.shape[1] != P.shape[0]:
            raise ValueError(f"snapshots must be (n_dof, n_runs={P.shape[0]}), got shape {X.shape}")
        p_min = P.min(axis=0)
        p_range = P.max(axis=0) - p_min
        if np.any(p_range == 0.0):
            raise ValueError("every parameter must vary across the training runs")

        pod = fit_pod(X, rank=rank, energy=energy, center=center, weights=weights)
        coeffs = pod.project(X)  # (rank, n_runs)
        interpolator = RBFInterpolator(
            (P - p_min) / p_range,
            coeffs.T,
            kernel=kernel,
            smoothing=smoothing,
            epsilon=epsilon,
            degree=degree,
        )
        return cls(pod=pod, interpolator=interpolator, param_min=p_min, param_range=p_range)

    def normalize(self, params: ArrayLike) -> np.ndarray:
        """Map ``(n, n_params)`` parameters to the training unit box.

        Values outside ``[0, 1]`` mean the query is an extrapolation.
        """
        P = _as_param_table(params)
        if P.shape[1] != self.param_min.size:
            raise ValueError(f"expected {self.param_min.size} parameters, got {P.shape[1]}")
        return (P - self.param_min) / self.param_range

    def coefficients(self, params: ArrayLike) -> np.ndarray:
        """Interpolated POD coefficients, shape ``(rank, n)``, for ``(n, n_params)`` parameters."""
        return self.interpolator(self.normalize(params)).T

    def predict(self, params: ArrayLike) -> np.ndarray:
        """Predicted snapshot matrix of shape ``(n_dof, n)`` for ``(n, n_params)`` parameters."""
        return self.pod.reconstruct(self.coefficients(params))


def _as_param_table(params: ArrayLike) -> np.ndarray:
    P = np.asarray(params, dtype=float)
    if P.ndim != 2:
        raise ValueError(f"params must be a (n_runs, n_params) table, got shape {P.shape}")
    return P
