"""Train/validation/test splits over RUNS (operating conditions).

Snapshots from one run are strongly correlated: putting time steps of the same
run in both training and test sets measures interpolation in time, not
generalisation to new operating conditions. Every split here therefore assigns
whole runs, identified by their position along the ``run`` dimension of
:func:`pcsp.data.campaign.load_campaign`.

* :func:`random_split` - runs shuffled into train/val/test by fraction.
* :func:`holdout_split` - every run inside a region of parameter space
  (:class:`Threshold` or :class:`Box`) becomes the test set; use it to test
  extrapolation to conditions outside the training range.
* :func:`kfold_split` - k folds, each run in exactly one test fold.

:func:`check_split` / :func:`check_folds` raise if sets overlap or miss runs;
:func:`apply_split` cuts a stacked dataset into the three sets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import numpy as np
import xarray as xr

ParamTable = Mapping[str, Any] | xr.Dataset
"""Parameter values per run: a stacked campaign dataset or ``{name: array}``."""


class SplitError(ValueError):
    """A split is invalid (overlapping, out-of-range or missing runs)."""


class Split(NamedTuple):
    """Sorted integer run indices of each set. ``val`` may be empty."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def _indices(values: Any) -> np.ndarray:
    return np.sort(np.asarray(values, dtype=np.intp))


def check_split(split: Split, n_runs: int, *, cover: bool = True) -> None:
    """Raise :class:`SplitError` unless the sets are disjoint, in range and,
    if ``cover``, together contain every run ``0..n_runs-1`` exactly once.
    Train and test must not be empty."""
    problems = []
    for name, idx in split._asdict().items():
        if idx.size and (idx.min() < 0 or idx.max() >= n_runs):
            problems.append(f"{name} has indices outside 0..{n_runs - 1}")
        if np.unique(idx).size != idx.size:
            problems.append(f"{name} contains a run more than once")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = np.intersect1d(getattr(split, a), getattr(split, b))
        if shared.size:
            problems.append(f"runs {shared.tolist()} are in both {a} and {b}")
    if cover:
        missing = np.setdiff1d(np.arange(n_runs), np.concatenate(split))
        if missing.size:
            problems.append(f"runs {missing.tolist()} are in no set")
    for name in ("train", "test"):
        if getattr(split, name).size == 0:
            problems.append(f"{name} set is empty")
    if problems:
        raise SplitError("invalid split:\n  - " + "\n  - ".join(problems))


def check_folds(folds: Sequence[Split], n_runs: int) -> None:
    """Check every fold, and that each run is in exactly one fold's test set."""
    for fold in folds:
        check_split(fold, n_runs)
    counts = np.bincount(np.concatenate([f.test for f in folds]), minlength=n_runs)
    if not np.all(counts == 1):
        bad = np.flatnonzero(counts != 1).tolist()
        raise SplitError(f"runs {bad} are not in exactly one test fold")


def _counts(n_runs: int, val_fraction: float, test_fraction: float) -> tuple[int, int]:
    for name, f in (("val_fraction", val_fraction), ("test_fraction", test_fraction)):
        if not 0.0 <= f < 1.0:
            raise ValueError(f"{name} must be in [0, 1), got {f}")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1 to leave runs for training")
    return round(val_fraction * n_runs), round(test_fraction * n_runs)


def random_split(
    n_runs: int, *, val_fraction: float = 0.15, test_fraction: float = 0.15, seed: int
) -> Split:
    """Shuffle whole runs into train/val/test. Set sizes are the fractions of
    ``n_runs`` rounded to the nearest integer; training gets the rest."""
    n_val, n_test = _counts(n_runs, val_fraction, test_fraction)
    order = np.random.default_rng(seed).permutation(n_runs)
    split = Split(
        train=_indices(order[n_val + n_test :]),
        val=_indices(order[:n_val]),
        test=_indices(order[n_val : n_val + n_test]),
    )
    check_split(split, n_runs)
    return split


@dataclass(frozen=True)
class Threshold:
    """Runs whose parameter ``name`` is strictly above (or below) ``value``."""

    name: str
    value: float
    side: Literal["above", "below"] = "above"

    def __post_init__(self) -> None:
        if self.side not in ("above", "below"):
            raise ValueError(f"side must be 'above' or 'below', got {self.side!r}")

    def contains(self, params: ParamTable) -> np.ndarray:
        """Boolean mask over runs."""
        values = _column(params, self.name)
        return values > self.value if self.side == "above" else values < self.value


@dataclass(frozen=True)
class Box:
    """Runs inside (or, with ``inside=False``, outside) a box of closed intervals.

    ``bounds`` maps parameter names to ``(low, high)``; a run is inside if
    every named parameter lies in its interval. Use ``inside=False`` to hold out
    the edges of the design space and train on its centre.
    """

    bounds: Mapping[str, tuple[float, float]]
    inside: bool = True

    def __post_init__(self) -> None:
        if not self.bounds:
            raise ValueError("a Box needs at least one bounded parameter")
        for name, (low, high) in self.bounds.items():
            if not low <= high:
                raise ValueError(f"Box bound for '{name}' needs low <= high, got ({low}, {high})")

    def contains(self, params: ParamTable) -> np.ndarray:
        """Boolean mask over runs."""
        mask = np.ones_like(_column(params, next(iter(self.bounds))), dtype=bool)
        for name, (low, high) in self.bounds.items():
            values = _column(params, name)
            mask &= (values >= low) & (values <= high)
        return mask if self.inside else ~mask


def _column(params: ParamTable, name: str) -> np.ndarray:
    try:
        column = params[name]
    except KeyError:
        raise KeyError(
            f"no parameter '{name}' to split on; available: {sorted(map(str, params.keys()))}. "
            "Only parameters that vary across runs are coordinates of a stacked campaign."
        ) from None
    values = np.asarray(column)
    if values.ndim != 1:
        raise ValueError(f"parameter '{name}' must be 1-D over runs, got shape {values.shape}")
    return values


def holdout_split(
    params: ParamTable, region: Threshold | Box, *, val_fraction: float = 0.0, seed: int = 0
) -> Split:
    """Test set = every run in ``region``; the rest is shuffled into train/val.

    ``params`` is usually the stacked campaign dataset itself. ``val_fraction``
    is a fraction of the runs *outside* the region, so validation stays inside
    the training range. Raises if the region holds no run or every run.
    """
    in_region = region.contains(params)
    test = np.flatnonzero(in_region)
    rest = np.flatnonzero(~in_region)
    if test.size == 0:
        raise SplitError(f"no run lies in the hold-out region {region}")
    if rest.size == 0:
        raise SplitError(f"every run lies in the hold-out region {region}; nothing to train on")
    n_val, _ = _counts(rest.size, val_fraction, 0.0)
    rest = np.random.default_rng(seed).permutation(rest)
    split = Split(train=_indices(rest[n_val:]), val=_indices(rest[:n_val]), test=_indices(test))
    check_split(split, in_region.size)
    return split


def kfold_split(n_runs: int, k: int, *, seed: int) -> list[Split]:
    """``k`` folds over shuffled runs (sizes differ by at most one); fold ``i``
    tests on its runs and trains on all others. ``val`` is empty."""
    if not 2 <= k <= n_runs:
        raise ValueError(f"k must be between 2 and n_runs={n_runs}, got {k}")
    order = np.random.default_rng(seed).permutation(n_runs)
    folds = []
    for test in np.array_split(order, k):
        train = np.setdiff1d(order, test)
        folds.append(Split(train=_indices(train), val=_indices([]), test=_indices(test)))
    check_folds(folds, n_runs)
    return folds


def apply_split(ds: xr.Dataset, split: Split, *, dim: str = "run") -> dict[str, xr.Dataset]:
    """Select the runs of each set from a stacked campaign dataset.

    Returns ``{"train": ..., "val": ..., "test": ...}`` (``val`` may have zero
    runs). The split is checked against ``ds`` first.
    """
    if dim not in ds.dims:
        raise ValueError(f"dataset has no '{dim}' dimension; load it with load_campaign")
    check_split(split, ds.sizes[dim])
    return {name: ds.isel({dim: idx}) for name, idx in split._asdict().items()}
