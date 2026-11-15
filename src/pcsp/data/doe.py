"""Design of experiments: which operating conditions to simulate.

A :class:`ParameterSpace` names the varied parameters (each a continuous
:class:`Range`, linear or logarithmic) and any fixed values. The samplers turn
it into a list of plain ``dict`` objects, one per run, ready to hand to
:func:`pcsp.data.campaign.run_campaign`::

    space = ParameterSpace(
        {"mass_flow": Range(1.0, 10.0), "dp": Range(1e-4, 1e-3, scale="log")},
        fixed={"T_in": 823.15},
    )
    samples = latin_hypercube(space, n=20, seed=1)
    # [{"mass_flow": 3.2..., "dp": 0.00041..., "T_in": 823.15}, ...]

Samplers with a ``seed`` return exactly the same samples for the same seed,
so a campaign can be regenerated (and resumed) from its seed alone.
"""

from __future__ import annotations

import copy
import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.stats import qmc

from pcsp.data.schema import to_jsonable


@dataclass(frozen=True)
class Range:
    """A continuous parameter range ``[low, high]``.

    ``scale="log"`` spreads samples uniformly in ``log(value)``, which is the
    right choice for quantities spanning orders of magnitude (particle
    diameters, permeabilities, Reynolds numbers). Requires ``low > 0``.
    """

    low: float
    high: float
    scale: Literal["linear", "log"] = "linear"

    def __post_init__(self) -> None:
        if not (math.isfinite(self.low) and math.isfinite(self.high)):
            raise ValueError(f"range bounds must be finite, got [{self.low}, {self.high}]")
        if not self.low < self.high:
            raise ValueError(f"range needs low < high, got [{self.low}, {self.high}]")
        if self.scale not in ("linear", "log"):
            raise ValueError(f"scale must be 'linear' or 'log', got {self.scale!r}")
        if self.scale == "log" and self.low <= 0:
            raise ValueError(f"a log-scaled range needs low > 0, got low={self.low}")

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        """Map values in ``[0, 1]`` onto the range."""
        if self.scale == "linear":
            return self.low + u * (self.high - self.low)
        log_low, log_high = math.log(self.low), math.log(self.high)
        values = np.exp(log_low + u * (log_high - log_low))
        # exp(log(x)) can be off by one ulp; keep the range ends exact.
        return np.where(u == 0.0, self.low, np.where(u == 1.0, self.high, values))


class ParameterSpace:
    """Named varied parameters (:class:`Range`) plus fixed values.

    Parameters
    ----------
    varied : name -> :class:`Range`. The order of names is the order of the
        sampler dimensions and of the grid loops in :func:`full_factorial`.
    fixed : name -> value, copied into every sample. Must be JSON-serialisable
        (the schema stores parameters as JSON).
    """

    def __init__(self, varied: Mapping[str, Range], fixed: Mapping[str, Any] | None = None):
        if not varied:
            raise ValueError("a parameter space needs at least one varied parameter")
        for name, rng in varied.items():
            if not isinstance(rng, Range):
                raise TypeError(f"varied parameter '{name}' must be a Range, got {rng!r}")
        fixed = dict(fixed or {})
        overlap = set(varied) & set(fixed)
        if overlap:
            raise ValueError(f"parameters cannot be both varied and fixed: {sorted(overlap)}")
        self.varied: dict[str, Range] = dict(varied)
        self.fixed: dict[str, Any] = to_jsonable(fixed)

    @property
    def names(self) -> list[str]:
        """Names of the varied parameters, in sampler-dimension order."""
        return list(self.varied)

    def __repr__(self) -> str:
        return f"ParameterSpace(varied={self.varied!r}, fixed={self.fixed!r})"

    def from_unit(self, unit_samples: np.ndarray) -> list[dict[str, Any]]:
        """Turn an ``(n, d)`` array of points in the unit cube into sample dicts."""
        unit_samples = np.asarray(unit_samples, dtype=float)
        if unit_samples.ndim != 2 or unit_samples.shape[1] != len(self.varied):
            raise ValueError(
                f"expected an array of shape (n, {len(self.varied)}), got {unit_samples.shape}"
            )
        columns = [rng.from_unit(unit_samples[:, j]) for j, rng in enumerate(self.varied.values())]
        return [self.sample([col[i] for col in columns]) for i in range(unit_samples.shape[0])]

    def sample(self, varied_values: Sequence[float]) -> dict[str, Any]:
        """One run's parameters: the varied values (in :attr:`names` order) plus a
        private copy of the fixed values."""
        varied = {name: float(v) for name, v in zip(self.varied, varied_values, strict=True)}
        return {**varied, **copy.deepcopy(self.fixed)}


def latin_hypercube(space: ParameterSpace, n: int, *, seed: int) -> list[dict[str, Any]]:
    """``n`` Latin-hypercube samples: each parameter's range is cut into ``n``
    equal-probability strata and every stratum is hit exactly once."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    sampler = qmc.LatinHypercube(d=len(space.varied), rng=np.random.default_rng(seed))
    return space.from_unit(sampler.random(n))


def sobol(
    space: ParameterSpace, n: int, *, seed: int, scramble: bool = True
) -> list[dict[str, Any]]:
    """``n`` points of a (scrambled) Sobol sequence.

    ``n`` must be a power of two: Sobol points only have their balance
    properties for those sizes (scipy warns otherwise). Use
    :func:`latin_hypercube` for arbitrary ``n``.
    """
    if n < 1 or n & (n - 1):
        raise ValueError(f"Sobol sampling needs n to be a power of two, got {n}")
    sampler = qmc.Sobol(d=len(space.varied), scramble=scramble, rng=np.random.default_rng(seed))
    return space.from_unit(sampler.random_base2(int(math.log2(n))))


def full_factorial(space: ParameterSpace, levels: int | Mapping[str, int]) -> list[dict[str, Any]]:
    """Every combination of evenly spaced levels (geometrically for log ranges).

    ``levels`` is one count for all parameters or a count per parameter. A
    parameter with 2 or more levels includes both ends of its range; with one
    level it sits at the (arithmetic or geometric) midpoint. The first
    parameter varies slowest.
    """
    counts = levels if isinstance(levels, Mapping) else dict.fromkeys(space.varied, levels)
    if set(counts) != set(space.varied):
        raise ValueError(
            f"levels must name exactly the varied parameters {space.names}, got {sorted(counts)}"
        )
    axes = []
    for name, rng in space.varied.items():
        k = counts[name]
        if k < 1:
            raise ValueError(f"parameter '{name}' needs at least one level, got {k}")
        unit = np.array([0.5]) if k == 1 else np.linspace(0.0, 1.0, k)
        axes.append(rng.from_unit(unit))
    return [space.sample(combo) for combo in itertools.product(*axes)]
