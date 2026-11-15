"""Shared fixtures for the data-layer tests: a small analytic toy model.

The toy model is a decaying travelling Gaussian pulse

    u(t, x) = amplitude * exp(-t / tau) * exp(-(x - x0 - speed * t)^2 / (2 w^2))

on a fixed grid. It is cheap, exact and has a per-run scalar (the peak value),
which is all the data layer needs. It must stay a module-level function so it
can be sent to worker processes.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest
import xarray as xr

from pcsp.data import Coord, ParameterSpace, Range, Var, new_dataset

TAU = 2.0
WIDTH = 0.05
X0 = 0.2


def pulse_model(params: dict[str, Any]) -> xr.Dataset:
    """Toy model. Raises for non-positive amplitude (a deliberately failing run).

    Optional parameter ``nx`` changes the spatial grid (to test grid mismatch).
    """
    speed = float(params["speed"])
    amplitude = float(params["amplitude"])
    if amplitude <= 0:
        raise ValueError(f"amplitude must be positive, got {amplitude}")
    t = np.linspace(0.0, 1.0, 11)
    x = np.linspace(0.0, 1.0, int(params.get("nx", 21)))
    u = (
        amplitude
        * np.exp(-t[:, None] / TAU)
        * np.exp(-((x[None, :] - X0 - speed * t[:, None]) ** 2) / (2 * WIDTH**2))
    )
    return new_dataset(
        component="benchmark",
        fidelity="reduced",
        model="toy-pulse",
        model_version="1.0",
        params=params,
        coords={
            "time": Coord(t, "s", "time"),
            "x": Coord(x, "m", "position"),
        },
        data_vars={
            "u": Var(("time", "x"), u, "1", "pulse amplitude"),
            "peak": Var((), u.max(), "1", "maximum of u over time and space"),
        },
    )


def worker_killing_model(params: dict[str, Any]) -> xr.Dataset:
    """Toy model whose process dies abruptly (like an OOM kill) when ``speed > 1``."""
    if params["speed"] > 1.0:
        os._exit(1)
    return pulse_model(params)


@pytest.fixture
def killing_model():
    return worker_killing_model


@pytest.fixture
def model():
    """The toy model (as a fixture, because conftest modules cannot be imported)."""
    return pulse_model


@pytest.fixture
def space() -> ParameterSpace:
    return ParameterSpace(
        {"speed": Range(0.1, 0.6), "amplitude": Range(0.5, 2.0, scale="log")},
        fixed={"model_tag": "pulse"},
    )


@pytest.fixture
def toy_dataset() -> xr.Dataset:
    return pulse_model({"speed": 0.3, "amplitude": 1.0})
