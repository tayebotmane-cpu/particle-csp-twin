"""POD + RBF interpolation ROM on a smooth two-parameter family of space-time fields."""

import numpy as np
import pytest
import xarray as xr
from scipy.stats import qmc

from pcsp.rom import (
    PODInterpolationROM,
    from_snapshots,
    projection_error,
    relative_error,
    to_snapshots,
)

LO = np.array([0.5, 0.0])  # diffusivity alpha, mode amplitude a
HI = np.array([2.0, 1.0])
Z = (np.arange(64) + 0.5) / 64
TIME = np.linspace(0.0, 0.1, 40)


def family(params: np.ndarray) -> np.ndarray:
    """Exact heat solutions u(t, z; alpha, a) on [0, 1] with three Fourier modes.

    u = e^{-alpha pi^2 t} sin(pi z) + a e^{-4 alpha pi^2 t} sin(2 pi z)
        + a^2 / 2 e^{-9 alpha pi^2 t} sin(3 pi z),
    returned as space-time snapshots (n_time * n_z, n_runs), time-major rows.
    """
    columns = []
    for alpha, a in params:
        u = sum(
            amp * np.exp(-alpha * (k * np.pi) ** 2 * TIME)[:, None] * np.sin(k * np.pi * Z)[None, :]
            for k, amp in ((1, 1.0), (2, a), (3, 0.5 * a**2))
        )
        columns.append(u.ravel())
    return np.array(columns).T


def training_set(n: int) -> np.ndarray:
    """Nested design: the 4 box corners, then a Halton sequence (larger sets contain smaller)."""
    corners = np.array([[LO[0], LO[1]], [LO[0], HI[1]], [HI[0], LO[1]], [HI[0], HI[1]]])
    halton = qmc.scale(qmc.Halton(2, seed=0).random(n - 4), LO, HI)
    return np.vstack([corners, halton])


def held_out(n: int, margin: float, seed: int) -> np.ndarray:
    """Uniform points inside the box (margin=0) or in a band of relative width margin outside it."""
    rng = np.random.default_rng(seed)
    span = HI - LO
    if margin == 0.0:
        return LO + span * rng.random((n, 2))
    points = []
    while len(points) < n:
        p = LO - margin * span + (1 + 2 * margin) * span * rng.random(2)
        if np.any(p < LO) or np.any(p > HI):
            points.append(p)
    return np.array(points)


def test_training_runs_are_reproduced_exactly():
    P = training_set(20)
    X = family(P)
    rom = PODInterpolationROM.fit(P, X)
    assert relative_error(X, rom.predict(P)) < 1e-10


def test_held_out_error_is_small_and_decreases_with_samples():
    P_test = held_out(200, margin=0.0, seed=1)
    X_test = family(P_test)
    errors = []
    for n in (16, 32, 64, 128):
        P = training_set(n)
        rom = PODInterpolationROM.fit(P, family(P))
        errors.append(relative_error(X_test, rom.predict(P_test)))
        # The basis is not the bottleneck: the interpolation error dominates.
        assert projection_error(rom.pod, X_test) < 0.5 * errors[-1]
    # Measured: 1.3e-2, 5.5e-3, 1.5e-3, 1.3e-3 (projection error 9e-6 ... 3e-16).
    assert all(later < earlier for earlier, later in zip(errors, errors[1:], strict=False))
    assert errors[0] < 2e-2
    assert errors[-1] < 2e-3


def test_mild_extrapolation_is_worse_than_interpolation():
    P = training_set(64)
    rom = PODInterpolationROM.fit(P, family(P))
    P_in, P_out = held_out(200, 0.0, seed=2), held_out(200, 0.1, seed=3)
    assert np.all((rom.normalize(P_in) >= 0) & (rom.normalize(P_in) <= 1))
    assert np.all(np.any((rom.normalize(P_out) < 0) | (rom.normalize(P_out) > 1), axis=1))
    err_in = relative_error(family(P_in), rom.predict(P_in))
    err_out = relative_error(family(P_out), rom.predict(P_out))
    # Measured with 64 runs, 10 % band outside the box: interp 1.4e-3, extrap 2.0e-2 (~14x).
    assert err_out > 3.0 * err_in
    assert err_out < 5e-2


def test_xarray_workflow_round_trip():
    P = training_set(16)
    runs = family(P).T.reshape(16, TIME.size, Z.size)
    T = xr.DataArray(
        runs,
        dims=("run", "time", "z"),
        coords={
            "run": ("run", np.arange(16), {"units": "1", "long_name": "run"}),
            "time": ("time", TIME, {"units": "s", "long_name": "time"}),
            "z": ("z", Z, {"units": "m", "long_name": "height"}),
        },
    )
    X, layout = to_snapshots(T, sample_dims="run")
    rom = PODInterpolationROM.fit(P, X, energy=1.0 - 1e-10)
    P_new = held_out(3, 0.0, seed=4)
    pred = from_snapshots(rom.predict(P_new), layout, sample_coords={"run": [100, 101, 102]})
    assert pred.dims == ("run", "time", "z")
    truth = family(P_new).T.reshape(3, TIME.size, Z.size)
    assert relative_error(truth.ravel(), pred.values.ravel()) < 2e-2


def test_invalid_inputs_are_rejected():
    P = training_set(8)
    X = family(P)
    with pytest.raises(ValueError, match="vary"):
        PODInterpolationROM.fit(np.column_stack([P[:, 0], np.ones(8)]), X)
    with pytest.raises(ValueError, match="n_runs"):
        PODInterpolationROM.fit(P, X[:, :-1])
    rom = PODInterpolationROM.fit(P, X)
    with pytest.raises(ValueError, match="parameters"):
        rom.predict(np.ones((2, 3)))
