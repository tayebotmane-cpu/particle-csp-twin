"""Packed-bed charging -> DoE campaign -> POD-interpolation surrogate, end to end.

The study asks how well a non-intrusive POD + RBF surrogate
(:class:`pcsp.rom.PODInterpolationROM`), trained on a Latin-hypercube campaign of
the packed-bed storage model (:func:`pcsp.models.storage.simulate`), predicts
the solid temperature field ``T_s(t, z)`` during charging for operating
conditions it has not seen, and how much faster it is than the model.

Pipeline
--------
1. :func:`parameter_space` / :func:`build_campaign` - varied ``mass_flow``,
   ``T_hot`` and particle diameter ``d_p``; every run stored with provenance and
   wall time by :func:`pcsp.data.run_campaign` (resumable).
2. :func:`make_splits` - split by *operating condition*, never by snapshot:
   ``"interpolation"`` (random runs) and ``"extrapolation"`` (every run with a
   mass flow above a quantile is held out; training lies entirely below it).
3. :func:`fit_surrogate` - one space-time snapshot per run, in one of two frames:

   * variant ``"A"``: physical time, ``t`` in ``[0, t_end]`` (same grid for every run);
   * variant ``"B"``: dimensionless time ``tau = t u_front / L`` on one common
     grid, where ``u_front`` is :func:`~pcsp.models.storage.thermal_front_speed`
     of that run, so the thermocline fronts of different runs are roughly
     aligned. A prediction is mapped back to physical time with the known
     ``t = tau L / u_front`` (linear interpolation in ``tau``).

4. :func:`evaluate` - per test run: relative L2 error, maximum absolute error,
   state-of-charge error, outlet-cell error, and the projection error of the
   basis (the best any model in that basis could do).

Fields are handled as the temperature rise ``T_s - T_cold`` [K] (``T_cold`` is
fixed in the study). A relative error of ``T_s`` in kelvin would be diluted by
the ~570 K offset; centred POD and the RBF interpolant are unchanged by the
shift.

Simulated duration. Variant B needs every run to cover the whole common ``tau``
grid ``[0, t_end u_max / L]``, so every run is simulated (on the same output
step as the study window) for ``t_end * u_max / u_min`` (``u`` over the
parameter box), roughly 3x the study window. Variant A only uses the first
``t_end`` of it. The manifest wall time is therefore that of the longer run;
:func:`measure_speed` also times the model on the study window alone so a
like-for-like speed-up can be quoted.

No leakage: the POD mean and basis, the input normalisation and the
interpolant are fitted on training runs only. The ``tau`` grid and the
simulated duration depend only on the parameter *box* (design information),
never on any simulated field.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import xarray as xr

from pcsp.data import (
    CampaignResult,
    ParameterSpace,
    Range,
    Split,
    Threshold,
    holdout_split,
    latin_hypercube,
    load_campaign,
    random_split,
    run_campaign,
)
from pcsp.data.schema import to_jsonable
from pcsp.materials.particles import CARBO_HSP_40_70, ParticleMaterial
from pcsp.models.storage import (
    PackedBedParams,
    simulate,
    solid_enthalpy_density,
    thermal_front_speed,
)
from pcsp.rom import PODInterpolationROM, projection_error, relative_error, time_full_vs_rom

Variant = Literal["A", "B"]
VARIANTS: tuple[Variant, ...] = ("A", "B")
VARIED = ("mass_flow", "T_hot", "d_p")
"""Varied parameters, in the column order of every parameter table."""

BED_MATERIAL = CARBO_HSP_40_70.with_(
    name="sintered-bauxite spheres (CARBO HSP bulk properties)",
    source=(
        "Bulk properties (rho_p, k_s, cp(T), packing) of CARBO HSP sintered bauxite "
        "(pcsp.materials.particles.CARBO_HSP_40_70), assumed to carry over to mm-size "
        "spheres of the same ceramic; d_p is set per run. " + CARBO_HSP_40_70.source
    ),
)
"""Bed solid. ``d_p`` is overridden by each run."""


# ---------------------------------------------------------------------------
# Configuration and grids
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StudyConfig:
    """Every choice of the study. Defaults are the full study.

    Attributes
    ----------
    n_runs, seed : Latin-hypercube size and seed (the seed also drives the splits)
    length, diameter : bed length and inner diameter [m]
    T_cold : initial bed and cold-end temperature, fixed [K]
    mass_flow, T_hot, d_p : varied ranges [kg s^-1], [K], [m]
    t_end : charging window of the study [s]
    n_time : output instants on ``[0, t_end]`` (variant A) and on the ``tau``
        grid (variant B), so both variants have the same snapshot length
    n_cells, rtol : model resolution and integrator tolerance
    interp_test_fraction : share of runs in the random (interpolation) test set
    extrap_quantile : runs with ``mass_flow`` above this quantile form the
        extrapolation test set
    energy : retained POD fluctuation energy that sets the rank of the main
        surrogates (training data only)
    ranks : ranks of the rank sweep (those above the training-set size are skipped)
    train_sizes : nested training-set sizes of the training-size sweep (the
        full training set is always added)
    n_timing_runs : test runs timed for the speed-up
    """

    n_runs: int = 120
    seed: int = 0
    length: float = 2.0
    diameter: float = 1.0
    T_cold: float = 573.15
    mass_flow: Range = Range(0.2, 0.7)
    T_hot: Range = Range(773.15, 1073.15)
    d_p: Range = Range(0.005, 0.025, scale="log")
    t_end: float = 7200.0
    n_time: int = 121
    n_cells: int = 100
    rtol: float = 1e-6
    interp_test_fraction: float = 0.2
    extrap_quantile: float = 0.75
    energy: float = 1.0 - 1e-6
    ranks: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96)
    train_sizes: tuple[int, ...] = (10, 20, 40, 60, 80)
    n_timing_runs: int = 5


@dataclass(frozen=True)
class TimeGrids:
    """Output grids shared by every run.

    Attributes
    ----------
    t : physical times of the study window ``[0, t_end]``, ``(n_time,)`` [s]
    tau : variant-B grid of ``tau = t u_front / L``, ``(n_time,)`` [-]
    t_sim : simulated output times, ``(n_sim,)`` [s]; ``t == t_sim[:n_time]``
    length : bed length [m]
    """

    t: np.ndarray
    tau: np.ndarray
    t_sim: np.ndarray
    length: float


def _fixed_params(config: StudyConfig, duration: float, n_output: int) -> dict[str, Any]:
    return {
        "base_material": to_jsonable(BED_MATERIAL),
        "length": config.length,
        "diameter": config.diameter,
        "T_cold": config.T_cold,
        "duration": duration,
        "n_cells": config.n_cells,
        "n_output": n_output,
        "rtol": config.rtol,
    }


def bed_params(sample: dict[str, Any]) -> PackedBedParams:
    """Packed-bed input for one campaign sample (variable-property charge)."""
    material = ParticleMaterial(**sample["base_material"]).with_(d_p=float(sample["d_p"]))
    return PackedBedParams(
        material=material,
        length=float(sample["length"]),
        diameter=float(sample["diameter"]),
        T_hot=float(sample["T_hot"]),
        T_cold=float(sample["T_cold"]),
        mass_flow=float(sample["mass_flow"]),
        duration=float(sample["duration"]),
        mode="charge",
        property_mode="variable",
        n_cells=int(sample["n_cells"]),
        n_output=int(sample["n_output"]),
        rtol=float(sample["rtol"]),
    )


def charge_model(sample: dict[str, Any]) -> xr.Dataset:
    """Campaign model: one charge of the bed (module level, so it can be pickled)."""
    return simulate(bed_params(sample))


def front_speed(sample: dict[str, Any]) -> float:
    """Thermal front speed ``u_front`` of one sample [m s^-1]."""
    return thermal_front_speed(bed_params(sample))


def front_speed_bounds(config: StudyConfig) -> tuple[float, float]:
    """Smallest and largest front speed over the parameter box [m s^-1].

    ``u_front`` grows with mass flow, does not depend on ``d_p`` and depends on
    ``T_hot`` through the property temperature, so it is evaluated at both mass
    flow bounds on a fine ``T_hot`` grid.
    """
    fixed = _fixed_params(config, duration=1.0, n_output=2)
    speeds = [
        front_speed({**fixed, "mass_flow": m, "T_hot": T, "d_p": config.d_p.low})
        for m in (config.mass_flow.low, config.mass_flow.high)
        for T in np.linspace(config.T_hot.low, config.T_hot.high, 21)
    ]
    return min(speeds), max(speeds)


def time_grids(config: StudyConfig) -> TimeGrids:
    """Study window, ``tau`` grid and simulated grid (see the module docstring)."""
    u_min, u_max = front_speed_bounds(config)
    dt = config.t_end / (config.n_time - 1)
    n_steps = math.ceil((config.n_time - 1) * u_max / u_min)
    t_sim = dt * np.arange(n_steps + 1)
    tau_max = config.t_end * u_max / config.length
    return TimeGrids(
        t=t_sim[: config.n_time].copy(),
        tau=np.linspace(0.0, tau_max, config.n_time),
        t_sim=t_sim,
        length=config.length,
    )


def parameter_space(config: StudyConfig) -> ParameterSpace:
    """Varied ``mass_flow``, ``T_hot``, ``d_p``; everything else fixed."""
    grids = time_grids(config)
    return ParameterSpace(
        {"mass_flow": config.mass_flow, "T_hot": config.T_hot, "d_p": config.d_p},
        fixed=_fixed_params(config, duration=float(grids.t_sim[-1]), n_output=grids.t_sim.size),
    )


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------


def build_campaign(
    config: StudyConfig, root: str | os.PathLike[str], *, max_workers: int = 1
) -> tuple[xr.Dataset, CampaignResult]:
    """Run (or resume) the campaign under ``root`` and load it stacked over runs."""
    samples = latin_hypercube(parameter_space(config), config.n_runs, seed=config.seed)
    result = run_campaign(charge_model, samples, root, max_workers=max_workers)
    return load_campaign(root), result


@dataclass(frozen=True)
class CampaignData:
    """The arrays of a loaded campaign that the study needs.

    Attributes
    ----------
    params : ``(n_run, 3)`` table in :data:`VARIED` order
    samples : the full parameter dict of each run
    rise : ``(n_run, n_sim, n_z)`` solid temperature rise ``T_s - T_cold`` [K]
    speed : ``(n_run,)`` thermal front speed [m s^-1]
    wall_seconds : ``(n_run,)`` model wall time from the manifest [s]
    param_hash : ``(n_run,)`` run identifiers
    z : ``(n_z,)`` cell centres from the hot end [m]
    grids : the shared time grids
    """

    params: np.ndarray
    samples: list[dict[str, Any]]
    rise: np.ndarray
    speed: np.ndarray
    wall_seconds: np.ndarray
    param_hash: np.ndarray
    z: np.ndarray
    grids: TimeGrids

    @property
    def n_runs(self) -> int:
        return self.params.shape[0]

    def subset(self, index: np.ndarray) -> CampaignData:
        """The runs at positions ``index``."""
        index = np.asarray(index, dtype=np.intp)
        return dataclasses.replace(
            self,
            params=self.params[index],
            samples=[self.samples[i] for i in index],
            rise=self.rise[index],
            speed=self.speed[index],
            wall_seconds=self.wall_seconds[index],
            param_hash=self.param_hash[index],
        )


def campaign_data(ds: xr.Dataset, config: StudyConfig) -> CampaignData:
    """Extract :class:`CampaignData` from a stacked campaign (checks the time grid)."""
    grids = time_grids(config)
    if ds.sizes["time"] != grids.t_sim.size or not np.allclose(ds["time"].values, grids.t_sim):
        raise ValueError("campaign time grid does not match the study configuration")
    fixed = json.loads(ds.attrs["fixed_params_json"])
    params = np.column_stack([ds[name].values.astype(float) for name in VARIED])
    samples = [{**fixed, **dict(zip(VARIED, row.tolist(), strict=True))} for row in params]
    return CampaignData(
        params=params,
        samples=samples,
        rise=ds["T_s"].transpose("run", "time", "z").values - config.T_cold,
        speed=np.array([front_speed(s) for s in samples]),
        wall_seconds=ds["wall_seconds"].values.astype(float),
        param_hash=ds["param_hash"].values,
        z=ds["z"].values,
        grids=grids,
    )


def make_splits(ds: xr.Dataset, config: StudyConfig) -> dict[str, Split]:
    """``"interpolation"`` (random runs) and ``"extrapolation"`` (highest mass flows)."""
    n = ds.sizes["run"]
    threshold = float(np.quantile(ds["mass_flow"].values, config.extrap_quantile))
    return {
        "interpolation": random_split(
            n, val_fraction=0.0, test_fraction=config.interp_test_fraction, seed=config.seed
        ),
        "extrapolation": holdout_split(ds, Threshold("mass_flow", threshold), seed=config.seed),
    }


# ---------------------------------------------------------------------------
# Frames: physical time (A) and front-aligned dimensionless time (B)
# ---------------------------------------------------------------------------


def _interp_rows(values: np.ndarray, grid: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Linear interpolation of ``values`` (rows on increasing ``grid``) at ``query``."""
    tol = 1e-9 * (grid[-1] - grid[0])
    if query.min() < grid[0] - tol or query.max() > grid[-1] + tol:
        raise ValueError(
            f"query range [{query.min():.6g}, {query.max():.6g}] is outside the grid "
            f"[{grid[0]:.6g}, {grid[-1]:.6g}]"
        )
    i = np.clip(np.searchsorted(grid, query, side="right") - 1, 0, grid.size - 2)
    w = np.clip((query - grid[i]) / (grid[i + 1] - grid[i]), 0.0, 1.0)[:, None]
    return (1.0 - w) * values[i] + w * values[i + 1]


def to_frame(data: CampaignData, variant: Variant) -> np.ndarray:
    """Fields of every run in the variant's frame, ``(n_run, n_time, n_z)`` [K]."""
    g = data.grids
    if variant == "A":
        return data.rise[:, : g.t.size].copy()
    return np.stack(
        [
            _interp_rows(field, g.t_sim, g.tau * g.length / u)
            for field, u in zip(data.rise, data.speed, strict=True)
        ]
    )


def to_physical(
    fields: np.ndarray, speed: np.ndarray, variant: Variant, grids: TimeGrids
) -> np.ndarray:
    """Map ``(n, n_time, n_z)`` fields from the variant's frame to ``t`` in ``[0, t_end]``."""
    if variant == "A":
        return fields
    return np.stack(
        [
            _interp_rows(f, grids.tau, grids.t * u / grids.length)
            for f, u in zip(fields, speed, strict=True)
        ]
    )


def _columns(fields: np.ndarray) -> np.ndarray:
    """``(n_run, n_time, n_z)`` -> space-time snapshot matrix ``(n_time * n_z, n_run)``."""
    return fields.reshape(fields.shape[0], -1).T


# ---------------------------------------------------------------------------
# Surrogate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThermoclineSurrogate:
    """A fitted surrogate of one variant. Build it with :func:`fit_surrogate`.

    Attributes
    ----------
    variant : ``"A"`` (physical time) or ``"B"`` (``tau`` frame)
    rom : the POD + RBF model in the variant's frame
    grids : shared time grids
    fixed : fixed parameters, needed to compute ``u_front`` of a query
    n_z : number of cells
    fit_seconds : wall time of fitting, including (for B) front speeds and resampling [s]
    """

    variant: Variant
    rom: PODInterpolationROM
    grids: TimeGrids
    fixed: dict[str, Any]
    n_z: int
    fit_seconds: float

    @property
    def rank(self) -> int:
        return self.rom.pod.rank

    def _shape(self, X: np.ndarray) -> np.ndarray:
        return X.T.reshape(X.shape[1], self.grids.t.size, self.n_z)

    def predict_native(self, params: np.ndarray) -> np.ndarray:
        """Temperature rise in the variant's own frame, ``(n, n_time, n_z)`` [K]."""
        return self._shape(self.rom.predict(params))

    def predict(self, params: np.ndarray) -> np.ndarray:
        """Temperature rise ``T_s - T_cold`` on ``t`` in ``[0, t_end]``, ``(n, n_time, n_z)`` [K].

        ``params`` is an ``(n, 3)`` table in :data:`VARIED` order.
        """
        params = np.atleast_2d(np.asarray(params, dtype=float))
        native = self.predict_native(params)
        if self.variant == "A":
            return native
        speed = np.array(
            [front_speed({**self.fixed, **dict(zip(VARIED, row, strict=True))}) for row in params]
        )
        return to_physical(native, speed, "B", self.grids)

    def project_physical(self, data: CampaignData) -> np.ndarray:
        """Best approximation of ``data`` in the basis, mapped to physical time [K]."""
        X = _columns(to_frame(data, self.variant))
        best = self._shape(self.rom.pod.reconstruct(self.rom.pod.project(X)))
        return to_physical(best, data.speed, self.variant, self.grids)


def fit_surrogate(
    train: CampaignData,
    variant: Variant,
    *,
    rank: int | None = None,
    energy: float | None = None,
) -> ThermoclineSurrogate:
    """Fit a centred POD + thin-plate-spline RBF surrogate on training runs only."""
    start = time.perf_counter()
    X = _columns(to_frame(train, variant))
    rom = PODInterpolationROM.fit(train.params, X, rank=rank, energy=energy, center=True)
    fixed = {k: v for k, v in train.samples[0].items() if k not in VARIED}
    return ThermoclineSurrogate(
        variant=variant,
        rom=rom,
        grids=train.grids,
        fixed=fixed,
        n_z=train.z.size,
        fit_seconds=time.perf_counter() - start,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorSummary:
    """Median, 90th percentile and worst (largest absolute) value over test runs."""

    median: float
    p90: float
    worst: float


def summarise(values: np.ndarray) -> ErrorSummary:
    a = np.abs(np.asarray(values, dtype=float))
    return ErrorSummary(float(np.median(a)), float(np.quantile(a, 0.9)), float(a.max()))


@dataclass(frozen=True)
class Evaluation:
    """Per-test-run errors of one surrogate (all arrays ``(n_test,)`` unless noted).

    Attributes
    ----------
    param_hash, params : the test runs and their ``(n_test, 3)`` parameters
    field_error : relative L2 error of the temperature rise on ``[0, t_end]`` [-]
    projection_error : the same for the best approximation in the basis, mapped
        to physical time [-]. For B this includes the ``tau`` resampling error
        and is not a strict lower bound of ``field_error``.
    native_field_error, native_projection_error : the two errors in the
        surrogate's own frame, where ``native_field_error >= native_projection_error``
        holds exactly
    max_abs_error : maximum absolute temperature error over ``(t, z)`` [K]
    soc_error : state of charge at ``t_end`` (solid enthalpy / capacity),
        predicted minus true [percentage points]
    outlet_error : maximum absolute error of the outlet-cell temperature history [K]
    distance_outside : how far outside the training box each run is: the largest
        excess of a normalised parameter beyond ``[0, 1]`` (0 inside) [-]
    truth, predicted : ``(n_test, n_time, n_z)`` temperature rise [K]
    """

    param_hash: np.ndarray
    params: np.ndarray
    field_error: np.ndarray
    projection_error: np.ndarray
    native_field_error: np.ndarray
    native_projection_error: np.ndarray
    max_abs_error: np.ndarray
    soc_error: np.ndarray
    outlet_error: np.ndarray
    distance_outside: np.ndarray
    truth: np.ndarray
    predicted: np.ndarray


def state_of_charge(sample: dict[str, Any], rise_end: np.ndarray) -> float:
    """Solid enthalpy of the end profile over its value at uniform ``T_hot`` [-]."""
    bed = bed_params(sample)
    H = solid_enthalpy_density(bed, bed.T_cold + rise_end)
    return float(np.mean(H) / solid_enthalpy_density(bed, np.array(bed.T_hot)))


def evaluate(surrogate: ThermoclineSurrogate, test: CampaignData) -> Evaluation:
    """Evaluate ``surrogate`` once on the test runs."""
    n_time = surrogate.grids.t.size
    truth = test.rise[:, :n_time]
    predicted = surrogate.predict(test.params)
    best = surrogate.project_physical(test)
    X_native = _columns(to_frame(test, surrogate.variant))
    pred_native = _columns(surrogate.predict_native(test.params))

    def per_run(approx: np.ndarray) -> np.ndarray:
        return relative_error(_columns(truth), _columns(approx), axis=0)

    soc_true = [state_of_charge(s, f[-1]) for s, f in zip(test.samples, truth, strict=True)]
    soc_pred = [state_of_charge(s, f[-1]) for s, f in zip(test.samples, predicted, strict=True)]
    unit = surrogate.rom.normalize(test.params)
    return Evaluation(
        param_hash=test.param_hash,
        params=test.params,
        field_error=per_run(predicted),
        projection_error=per_run(best),
        native_field_error=relative_error(X_native, pred_native, axis=0),
        native_projection_error=projection_error(surrogate.rom.pod, X_native, axis=0),
        max_abs_error=np.abs(predicted - truth).max(axis=(1, 2)),
        soc_error=100.0 * (np.array(soc_pred) - np.array(soc_true)),
        outlet_error=np.abs(predicted[:, :, -1] - truth[:, :, -1]).max(axis=1),
        distance_outside=np.maximum(0.0, np.maximum(-unit, unit - 1.0)).max(axis=1),
        truth=truth,
        predicted=predicted,
    )


@dataclass(frozen=True)
class SpeedResult:
    """Online cost of one prediction against the model.

    Attributes
    ----------
    rom_seconds : best-of-repeats time of one surrogate prediction (median over timed runs) [s]
    model_seconds_manifest : median manifest wall time of the test runs (simulated
        to ``t_sim[-1]``) [s]
    model_seconds_window : median measured time of the model run on ``[0, t_end]`` only [s]
    """

    rom_seconds: float
    model_seconds_manifest: float
    model_seconds_window: float

    @property
    def speedup_manifest(self) -> float:
        return self.model_seconds_manifest / self.rom_seconds

    @property
    def speedup_window(self) -> float:
        return self.model_seconds_window / self.rom_seconds


def measure_speed(
    surrogate: ThermoclineSurrogate, test: CampaignData, n_runs: int, *, repeats: int = 3
) -> SpeedResult:
    """Time one prediction against the model on the first ``n_runs`` test runs."""
    g = surrogate.grids
    timings = []
    for sample, row in zip(test.samples[:n_runs], test.params[:n_runs], strict=False):
        window = bed_params({**sample, "duration": float(g.t[-1]), "n_output": g.t.size})
        timings.append(
            time_full_vs_rom(
                lambda w=window: simulate(w),
                lambda r=row: surrogate.predict(r[None, :]),
                repeats=repeats,
            )
        )
    return SpeedResult(
        rom_seconds=float(np.median([t.rom_seconds for t in timings])),
        model_seconds_manifest=float(np.median(test.wall_seconds)),
        model_seconds_window=float(np.median([t.full_seconds for t in timings])),
    )


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sweep:
    """Median (and worst) held-out errors against a swept quantity (rank or n_train)."""

    x: np.ndarray
    rank: np.ndarray
    field_error: np.ndarray
    projection_error: np.ndarray
    worst_field_error: np.ndarray


def rank_sweep(
    train: CampaignData, test: CampaignData, variant: Variant, ranks: Sequence[int]
) -> Sweep:
    """Held-out errors of surrogates of fixed ranks (ranks above ``n_train`` are skipped)."""
    kept = [r for r in ranks if r <= train.n_runs]
    evals = [evaluate(fit_surrogate(train, variant, rank=r), test) for r in kept]
    return _sweep(np.array(kept), np.array(kept), evals)


def training_size_sweep(
    train: CampaignData,
    test: CampaignData,
    variant: Variant,
    sizes: Sequence[int],
    *,
    energy: float,
    seed: int,
) -> Sweep:
    """Held-out errors for nested training subsets (each contains the smaller ones).

    The rank of each fit follows the same energy criterion as the main surrogate.
    """
    order = np.random.default_rng(seed).permutation(train.n_runs)
    kept = sorted({n for n in sizes if 5 <= n < train.n_runs} | {train.n_runs})
    surrogates = [fit_surrogate(train.subset(order[:n]), variant, energy=energy) for n in kept]
    evals = [evaluate(s, test) for s in surrogates]
    return _sweep(np.array(kept), np.array([s.rank for s in surrogates]), evals)


def _sweep(x: np.ndarray, rank: np.ndarray, evals: list[Evaluation]) -> Sweep:
    return Sweep(
        x=x,
        rank=rank,
        field_error=np.array([np.median(e.field_error) for e in evals]),
        projection_error=np.array([np.median(e.projection_error) for e in evals]),
        worst_field_error=np.array([e.field_error.max() for e in evals]),
    )


# ---------------------------------------------------------------------------
# The whole study
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantResult:
    """Main surrogate of one variant on one split, with its evaluation and costs.

    ``campaign_seconds`` is the summed manifest wall time of the training runs;
    ``training_cost`` adds the fit time.
    """

    split: str
    variant: Variant
    n_train: int
    rank: int
    fit_seconds: float
    campaign_seconds: float
    evaluation: Evaluation
    speed: SpeedResult

    @property
    def training_cost(self) -> float:
        return self.campaign_seconds + self.fit_seconds


@dataclass(frozen=True)
class StudyResult:
    """Everything :func:`run_study` measured.

    Keys of ``results`` and ``rank_sweeps`` are ``(split, variant)``;
    ``spectra`` (singular values of the interpolation training set) and
    ``size_sweeps`` (on the interpolation split) are keyed by variant.
    """

    config: StudyConfig
    n_ok: int
    n_failed: int
    n_executed: int
    campaign_seconds: float
    splits: dict[str, Split]
    data: CampaignData
    results: dict[tuple[str, Variant], VariantResult]
    spectra: dict[Variant, np.ndarray]
    rank_sweeps: dict[tuple[str, Variant], Sweep]
    size_sweeps: dict[Variant, Sweep] = field(default_factory=dict)


def run_study(
    config: StudyConfig, root: str | os.PathLike[str], *, max_workers: int = 1
) -> StudyResult:
    """Campaign (resumed if present) -> splits -> surrogates A and B -> evaluation."""
    ds, campaign = build_campaign(config, root, max_workers=max_workers)
    data = campaign_data(ds, config)
    splits = make_splits(ds, config)
    results: dict[tuple[str, Variant], VariantResult] = {}
    rank_sweeps: dict[tuple[str, Variant], Sweep] = {}
    spectra: dict[Variant, np.ndarray] = {}
    size_sweeps: dict[Variant, Sweep] = {}
    for split_name, split in splits.items():
        train, test = data.subset(split.train), data.subset(split.test)
        for variant in VARIANTS:
            surrogate = fit_surrogate(train, variant, energy=config.energy)
            results[(split_name, variant)] = VariantResult(
                split=split_name,
                variant=variant,
                n_train=train.n_runs,
                rank=surrogate.rank,
                fit_seconds=surrogate.fit_seconds,
                campaign_seconds=float(train.wall_seconds.sum()),
                evaluation=evaluate(surrogate, test),
                speed=measure_speed(surrogate, test, config.n_timing_runs),
            )
            rank_sweeps[(split_name, variant)] = rank_sweep(train, test, variant, config.ranks)
            if split_name == "interpolation":
                spectra[variant] = surrogate.rom.pod.singular_values
                size_sweeps[variant] = training_size_sweep(
                    train, test, variant, config.train_sizes, energy=config.energy, seed=config.seed
                )
    return StudyResult(
        config=config,
        n_ok=len(campaign.ok),
        n_failed=len(campaign.failed),
        n_executed=len(campaign.executed),
        campaign_seconds=float(data.wall_seconds.sum()),
        splits=splits,
        data=data,
        results=results,
        spectra=spectra,
        rank_sweeps=rank_sweeps,
        size_sweeps=size_sweeps,
    )


def format_results_table(study: StudyResult) -> str:
    """Plain-text results table, one row per (split, variant)."""
    header = (
        f"{'split':<14}{'var':<4}{'n_tr':>5}{'rank':>5}{'proj med':>10}"
        f"{'ROM med':>10}{'ROM p90':>10}{'ROM max':>10}{'max|dT| K':>11}"
        f"{'SoC pp max':>11}{'outlet K max':>13}{'speed-up':>10}{'(window)':>10}"
        f"{'train s':>9}"
    )
    lines = [header, "-" * len(header)]
    for (split, variant), r in study.results.items():
        e = r.evaluation
        field_err = summarise(e.field_error)
        lines.append(
            f"{split:<14}{variant:<4}{r.n_train:>5}{r.rank:>5}"
            f"{np.median(e.projection_error):>10.2e}{field_err.median:>10.2e}"
            f"{field_err.p90:>10.2e}{field_err.worst:>10.2e}{e.max_abs_error.max():>11.1f}"
            f"{np.abs(e.soc_error).max():>11.2f}{e.outlet_error.max():>13.1f}"
            f"{r.speed.speedup_manifest:>10.0f}{r.speed.speedup_window:>10.0f}"
            f"{r.training_cost:>9.1f}"
        )
    return "\n".join(lines)
