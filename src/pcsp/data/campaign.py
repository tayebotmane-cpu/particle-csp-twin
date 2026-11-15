"""Simulation campaigns: run a model over a design of experiments, resumably.

On-disk layout of a campaign rooted at ``<root>``::

    <root>/
        manifest.json            one record per run (see :class:`RunRecord`)
        runs/
            <param_hash>.zarr    one schema dataset per successful run

``manifest.json`` is plain, human-readable JSON and is the source of truth: a
run counts as done only if its record says ``"status": "ok"`` and its dataset
exists. The manifest is rewritten atomically after every run, so killing a
campaign loses at most the runs in progress, and each dataset is itself
written atomically (:func:`pcsp.data.io.save`), so a crashed run never leaves
a half-written dataset under ``runs/``. Running the same campaign again skips
the finished runs and retries the failed ones.

The recorded ``wall_seconds`` is the time spent inside the model call only
(the compute cost for accuracy-versus-cost curves); ``save_seconds`` is the
storage overhead.

One campaign directory must not be written by two processes at once.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import pickle
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr

from pcsp.data import io
from pcsp.data.provenance import collect_provenance, stamp
from pcsp.data.schema import (
    canonical_json,
    get_params,
    param_hash,
    to_jsonable,
)

log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1
RUNS_DIR = "runs"

_PER_RUN_ATTRS = ("params_json", "param_hash", "created_utc")
"""Global attributes that identify a single run; replaced by coordinates when stacking."""

Model = Callable[[dict[str, Any]], xr.Dataset]
"""A model maps one run's parameters to a schema-valid dataset."""


@dataclass(frozen=True)
class RunRecord:
    """The manifest entry for one run.

    ``path`` is relative to the campaign root. ``wall_seconds`` is the model
    call alone; it is ``None`` if the worker process died before reporting.
    """

    param_hash: str
    params: dict[str, Any]
    status: Literal["ok", "failed"]
    path: str
    wall_seconds: float | None
    save_seconds: float | None = None
    error: str = ""
    traceback: str = ""
    finished_utc: str = ""


@dataclass(frozen=True)
class CampaignResult:
    """Outcome of :func:`run_campaign`.

    ``records`` has one entry per requested sample, in input order. ``executed``
    and ``skipped`` hold the ``param_hash`` of the runs that were run in this
    call and of those skipped because they had already succeeded.
    """

    records: list[RunRecord]
    executed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def ok(self) -> list[RunRecord]:
        return [r for r in self.records if r.status == "ok"]

    @property
    def failed(self) -> list[RunRecord]:
        return [r for r in self.records if r.status == "failed"]


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_path(h: str) -> str:
    return f"{RUNS_DIR}/{h}.zarr"


def _check_recorded_params(ds: xr.Dataset, params: dict[str, Any]) -> None:
    """Fail if the model recorded a different value for a parameter it was given.

    That would mean the dataset (and its ``param_hash``) describes a different
    operating condition than the manifest. Parameters the model records under
    other names (e.g. nested in a dataclass) are not checked.
    """
    recorded = get_params(ds)
    wrong = {
        k: (v, recorded[k])
        for k, v in to_jsonable(params).items()
        if k in recorded and recorded[k] != v
    }
    if wrong:
        details = ", ".join(f"{k}: given {g!r}, recorded {r!r}" for k, (g, r) in wrong.items())
        raise ValueError(f"model recorded different parameter values than it was given ({details})")


def _execute_run(
    model: Model, params: dict[str, Any], root: str, provenance: dict[str, str | int]
) -> RunRecord:
    """Run one sample and store its dataset. Never raises for model errors.

    Module-level so that it can be sent to worker processes.
    """
    h = param_hash(params)
    rel_path = _run_path(h)
    start = time.perf_counter()
    wall: float | None = None
    save_s: float | None = None
    try:
        ds = model(dict(params))
        wall = time.perf_counter() - start
        if not isinstance(ds, xr.Dataset):
            raise TypeError(f"model returned {type(ds).__name__}, expected xarray.Dataset")
        _check_recorded_params(ds, params)
        t_save = time.perf_counter()
        io.save(stamp(ds, provenance), Path(root) / rel_path, overwrite=True)
        save_s = time.perf_counter() - t_save
    except Exception as exc:
        if wall is None:
            wall = time.perf_counter() - start
        return RunRecord(
            param_hash=h,
            params=params,
            status="failed",
            path=rel_path,
            wall_seconds=wall,
            save_seconds=save_s,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            finished_utc=_utc_now(),
        )
    return RunRecord(
        param_hash=h,
        params=params,
        status="ok",
        path=rel_path,
        wall_seconds=wall,
        save_seconds=save_s,
        finished_utc=_utc_now(),
    )


def read_manifest(root: str | os.PathLike[str]) -> list[RunRecord]:
    """All run records of a campaign (ok and failed), in the order the samples were first given."""
    path = Path(root) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"no campaign manifest at '{path}'")
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
        if content["manifest_version"] != MANIFEST_VERSION:
            raise ValueError(f"unsupported manifest_version {content['manifest_version']!r}")
        return [RunRecord(**entry) for entry in content["runs"]]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"campaign manifest '{path}' is not readable: {exc}") from exc


def _write_manifest(root: Path, records: Mapping[str, RunRecord]) -> None:
    """Atomically replace the manifest with ``records``."""
    content = {
        "manifest_version": MANIFEST_VERSION,
        "runs": [dataclasses.asdict(r) for r in records.values()],
    }
    path = root / MANIFEST_NAME
    tmp = path.with_name(f".{MANIFEST_NAME}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run_campaign(
    model: Model,
    samples: Sequence[Mapping[str, Any]],
    root: str | os.PathLike[str],
    *,
    force: bool = False,
    max_workers: int = 1,
    provenance: dict[str, str | int] | None = None,
) -> CampaignResult:
    """Run ``model`` once per sample and store each result under ``root``.

    Parameters
    ----------
    model : ``params -> xarray.Dataset`` returning a schema-valid dataset. An
        exception raised by the model (or an invalid result) is recorded as a
        failed run; the campaign carries on.
    samples : one parameter dict per run, e.g. from :mod:`pcsp.data.doe`.
        Duplicate parameter sets are rejected.
    root : campaign directory (created if missing). Can hold runs from earlier
        calls; new samples are added to the same manifest.
    force : re-run samples that already succeeded (default: skip them).
    max_workers : number of worker processes. ``1`` (default) runs serially in
        this process. With more, ``model`` must be picklable (a module-level
        function or a ``functools.partial`` of one) and every worker holds one
        run in memory at a time, so size this to the available RAM.
    provenance : stamp to put on every run; collected once by default.
    """
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")
    root = Path(root)
    params_list = [to_jsonable(dict(s)) for s in samples]
    try:
        json.dumps(params_list, allow_nan=False)
    except ValueError as exc:
        raise ValueError(f"sample parameters must be finite numbers (no NaN/inf): {exc}") from exc
    hashes = [param_hash(p) for p in params_list]
    if len(set(hashes)) != len(hashes):
        dupes = sorted({h for h in hashes if hashes.count(h) > 1})
        raise ValueError(f"samples contain duplicate parameter sets (param_hash {dupes})")

    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / MANIFEST_NAME
    records = {r.param_hash: r for r in read_manifest(root)} if manifest_path.exists() else {}
    provenance = provenance if provenance is not None else collect_provenance()

    todo: list[dict[str, Any]] = []
    skipped: list[str] = []
    for h, params in zip(hashes, params_list, strict=True):
        previous = records.get(h)
        done = previous is not None and previous.status == "ok" and (root / previous.path).exists()
        if done and not force:
            skipped.append(h)
        else:
            todo.append(params)
    log.info("campaign %s: %d to run, %d already done", root, len(todo), len(skipped))

    # Manifest order: earlier runs first, then this call's samples in input order
    # (not completion order, which varies between parallel runs).
    order = [*records, *(h for h in hashes if h not in records)]

    def record(result: RunRecord) -> None:
        records[result.param_hash] = result
        _write_manifest(root, {h: records[h] for h in order if h in records})
        if result.status == "failed":
            log.warning("run %s failed: %s", result.param_hash, result.error)

    if max_workers == 1 or not todo:
        for params in todo:
            record(_execute_run(model, params, str(root), provenance))
    else:
        _run_parallel(model, todo, root, provenance, max_workers, record)

    return CampaignResult(
        records=[records[h] for h in hashes],
        executed=[param_hash(p) for p in todo],
        skipped=skipped,
    )


def _run_parallel(
    model: Model,
    todo: list[dict[str, Any]],
    root: Path,
    provenance: dict[str, str | int],
    max_workers: int,
    record: Callable[[RunRecord], None],
) -> None:
    """Run ``todo`` in a process pool, recording each run as it finishes."""
    try:
        pickle.dumps(model)
    except Exception as exc:
        raise TypeError(
            f"max_workers={max_workers} needs a picklable model (a module-level function or "
            f"functools.partial of one); {model!r} is not: {exc}"
        ) from exc
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures: dict[Future[RunRecord], dict[str, Any]] = {
            pool.submit(_execute_run, model, params, str(root), provenance): params
            for params in todo
        }
        for future in as_completed(futures):
            params = futures[future]
            try:
                result = future.result()
            except BrokenProcessPool as exc:
                # The worker died (killed, out of memory, segfault): no record came back.
                result = RunRecord(
                    param_hash=param_hash(params),
                    params=params,
                    status="failed",
                    path=_run_path(param_hash(params)),
                    wall_seconds=None,
                    error=f"worker process died before finishing (e.g. out of memory): {exc}",
                    finished_utc=_utc_now(),
                )
            record(result)


def _grid_mismatch(ref: xr.Dataset, other: xr.Dataset) -> str | None:
    """Describe how ``other``'s grid or variables differ from ``ref`` (None if same)."""
    if set(ref.coords) != set(other.coords):
        return f"coordinates {sorted(map(str, other.coords))} vs {sorted(map(str, ref.coords))}"
    for name in ref.coords:
        a, b = ref.coords[name], other.coords[name]
        if a.shape != b.shape:
            return f"coordinate '{name}' has shape {b.shape} vs {a.shape}"
        if not a.identical(b):
            return f"coordinate '{name}' has different values or attributes"
    if set(ref.data_vars) != set(other.data_vars):
        return f"variables {sorted(map(str, other.data_vars))} vs {sorted(map(str, ref.data_vars))}"
    for name in ref.data_vars:
        if ref[name].dims != other[name].dims:
            return f"variable '{name}' has dims {other[name].dims} vs {ref[name].dims}"
    return None


def _coord_value(value: Any) -> Any:
    """Scalars stay as they are; lists/dicts become canonical JSON text."""
    return value if isinstance(value, (str, int, float, bool)) else canonical_json(value)


def load_campaign(
    root: str | os.PathLike[str], *, param_hashes: Sequence[str] | None = None
) -> xr.Dataset:
    """Load the successful runs of a campaign stacked along a new ``run`` dimension.

    The result has

    * every data variable with a leading ``run`` dimension; per-run scalars
      (0-d variables) become 1-D arrays over ``run``;
    * coordinates ``run`` (0..n-1), ``param_hash`` and one per *varied* input
      parameter, all along ``run``;
    * a data variable ``wall_seconds`` (model compute time per run);
    * the global attributes shared by all runs, plus ``fixed_params_json``
      (canonical JSON of the parameters that are equal in every run). An
      attribute that differs between runs - e.g. ``git_sha`` when a campaign
      was resumed after a new commit - becomes a coordinate along ``run``
      instead, so mixed provenance is never hidden.

    Runs are in manifest order. All runs must share the same coordinates
    (grid); otherwise a ``ValueError`` explains the options. Nothing is
    interpolated. The stacked view is for analysis and training: it has string
    coordinates, so it is not itself a schema dataset to be archived.

    Parameters
    ----------
    param_hashes : load only these runs (e.g. a subset sharing one grid).
    """
    root = Path(root)
    records = [r for r in read_manifest(root) if r.status == "ok"]
    if param_hashes is not None:
        wanted = set(param_hashes)
        unknown = wanted - {r.param_hash for r in records}
        if unknown:
            raise ValueError(f"no successful runs with param_hash {sorted(unknown)} in '{root}'")
        records = [r for r in records if r.param_hash in wanted]
    if not records:
        raise ValueError(f"campaign '{root}' has no successful runs to load")
    datasets = [io.load(root / r.path) for r in records]

    mismatches = []
    for r, ds in zip(records[1:], datasets[1:], strict=True):
        reason = _grid_mismatch(datasets[0], ds)
        if reason:
            mismatches.append(f"run {r.param_hash}: {reason}")
    if mismatches:
        raise ValueError(
            f"cannot stack campaign '{root}': runs do not share one grid (compared with run "
            f"{records[0].param_hash}):\n  - " + "\n  - ".join(mismatches) + "\nStacking would "
            "need interpolation, which is not done silently. Options: (1) re-run the campaign "
            "on a common grid; (2) stack groups of runs that share a grid with "
            "load_campaign(root, param_hashes=[...]); (3) load runs individually with "
            "pcsp.data.load(root / record.path) (records from read_manifest) and interpolate "
            "them explicitly onto a grid of your choice with Dataset.interp."
        )

    stacked = xr.concat(
        [ds.drop_attrs(deep=False) for ds in datasets],
        dim="run",
        data_vars="all",
        coords="minimal",
        compat="identical",
        join="exact",
        combine_attrs="drop",
    )
    keysets = {frozenset(r.params) for r in records}
    if len(keysets) > 1:
        raise ValueError(
            f"runs in '{root}' do not all have the same parameter names: "
            f"{sorted(sorted(k) for k in keysets)}"
        )

    per_run: dict[str, xr.Variable] = {}

    def add(name: str, values: Sequence[Any], long_name: str, units: str | None = None) -> None:
        if name in per_run or name in stacked.variables:
            raise ValueError(
                f"cannot add per-run coordinate '{name}' to the stacked campaign: the name is "
                "already used by a parameter, provenance attribute or dataset variable"
            )
        attrs = (
            {"long_name": long_name} if units is None else {"units": units, "long_name": long_name}
        )
        per_run[name] = xr.Variable("run", np.array(values), attrs=attrs)

    add("run", np.arange(len(records)), "run index", units="1")
    add("param_hash", [r.param_hash for r in records], "hash of the run's input parameters")
    fixed_params = {}
    for name in records[0].params:
        values = [_coord_value(r.params[name]) for r in records]
        if len({canonical_json(v) for v in values}) > 1:
            add(name, values, f"input parameter {name}")
        else:
            fixed_params[name] = records[0].params[name]

    attrs: dict[str, Any] = {"fixed_params_json": canonical_json(fixed_params)}
    for key in sorted({k for ds in datasets for k in ds.attrs} - set(_PER_RUN_ATTRS)):
        values = [ds.attrs.get(key, "") for ds in datasets]
        if all(v == values[0] for v in values):
            attrs[key] = values[0]
        else:
            add(key, values, f"global attribute '{key}' of each run")

    if "wall_seconds" in per_run or "wall_seconds" in stacked.variables:
        raise ValueError("'wall_seconds' is reserved for the per-run compute time")
    stacked = stacked.assign_coords(per_run)
    stacked["wall_seconds"] = xr.Variable(
        "run",
        np.array([r.wall_seconds for r in records], dtype=float),
        attrs={"units": "s", "long_name": "model wall-clock time"},
    )
    stacked.attrs = dict(sorted(attrs.items()))
    return stacked
