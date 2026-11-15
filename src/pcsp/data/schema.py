"""The canonical dataset schema.

Every simulation or experiment that enters the project - reduced model, CFD,
CFD-DEM, measured data - is stored as one :class:`xarray.Dataset` that obeys
the rules below, so that everything downstream (ROMs, surrogates, the twin)
can consume any fidelity through one interface.

Rules
-----
1. SI units. Temperatures in kelvin. Time in seconds on a coordinate ``time``.
2. Every coordinate and data variable carries ``units`` and ``long_name``
   attributes. Use ``"1"`` for dimensionless quantities.
3. Every dimension has a coordinate variable.
4. Global attributes identify what produced the data (see
   :data:`REQUIRED_ATTRS`): component, fidelity, model name and version, the
   full input parameter set as canonical JSON, and a hash of those parameters.
5. Attribute values are ``str``, ``int`` or ``float`` only, so datasets
   round-trip through both Zarr and netCDF/HDF5.

Build datasets with :func:`new_dataset`; check them with :func:`validate`.
Provenance stamps (git SHA etc., see :data:`PROVENANCE_ATTRS`) are added by
:mod:`pcsp.data.provenance` when a dataset is archived.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import xarray as xr

SCHEMA_VERSION = "0.1.0"

COMPONENTS = ("receiver_fpr", "hx_moving_bed", "storage_bin", "benchmark")
"""Plant components a dataset can describe."""

FIDELITIES = ("reduced", "rom", "surrogate", "cfd", "cfd_dem", "dem", "experiment")
"""Rungs of the fidelity ladder a dataset can come from."""

REQUIRED_ATTRS = (
    "pcsp_schema_version",
    "component",
    "fidelity",
    "model",
    "model_version",
    "params_json",
    "param_hash",
    "created_utc",
)

PROVENANCE_ATTRS = ("git_sha", "git_dirty", "pcsp_version")
"""Required once a dataset is archived (``validate(require_provenance=True)``)."""

OPTIONAL_PROVENANCE_ATTRS = ("container_digest", "hostname")

_NON_SI_TEMPERATURE_UNITS = frozenset({"degC", "C", "°C", "celsius", "degF", "F"})
_ATTR_TYPES = (str, int, float)


class SchemaError(ValueError):
    """A dataset violates the canonical schema. The message lists every problem."""


class Coord(NamedTuple):
    """A coordinate: 1-D values plus mandatory metadata."""

    values: Any
    units: str
    long_name: str


class Var(NamedTuple):
    """A data variable. ``dims=()`` declares a scalar quantity of interest."""

    dims: tuple[str, ...] | str
    values: Any
    units: str
    long_name: str


def to_jsonable(obj: Any) -> Any:
    """Convert parameters to plain JSON types.

    Handles dataclasses, mappings, sequences, NumPy scalars/arrays and paths.
    Anything else (callables, arbitrary objects) raises ``TypeError``: a
    parameter that cannot be written down cannot be reproduced.
    """
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    raise TypeError(f"parameter of type {type(obj).__name__} is not JSON-serialisable: {obj!r}")


def canonical_json(params: Any) -> str:
    """Deterministic JSON text: sorted keys, no insignificant whitespace."""
    return json.dumps(to_jsonable(params), sort_keys=True, separators=(",", ":"))


def param_hash(params: Any) -> str:
    """Short stable identifier of a parameter set (first 16 hex of SHA-256)."""
    return hashlib.sha256(canonical_json(params).encode("utf-8")).hexdigest()[:16]


def new_dataset(
    *,
    component: str,
    fidelity: str,
    model: str,
    model_version: str,
    params: Any,
    coords: Mapping[str, Coord],
    data_vars: Mapping[str, Var],
    attrs: Mapping[str, str | int | float] | None = None,
) -> xr.Dataset:
    """Build a schema-conforming dataset.

    Parameters
    ----------
    component : one of :data:`COMPONENTS`
    fidelity : one of :data:`FIDELITIES`
    model : name of the model or experiment that produced the data
    model_version : version string of that model
    params : complete input parameter set (dataclass or mapping). Stored as
        canonical JSON in ``attrs["params_json"]`` and hashed.
    coords : coordinate name -> :class:`Coord`
    data_vars : variable name -> :class:`Var`
    attrs : extra global attributes (``str``/``int``/``float`` values)

    The result is validated before it is returned.
    """
    xr_coords = {
        name: xr.Variable(
            (name,),
            np.asarray(c.values),
            attrs={"units": c.units, "long_name": c.long_name},
        )
        for name, c in coords.items()
    }
    xr_vars = {}
    for name, v in data_vars.items():
        dims = (v.dims,) if isinstance(v.dims, str) else tuple(v.dims)
        xr_vars[name] = xr.Variable(
            dims, np.asarray(v.values), attrs={"units": v.units, "long_name": v.long_name}
        )
    global_attrs: dict[str, str | int | float] = dict(attrs or {})
    clash = set(global_attrs) & set(REQUIRED_ATTRS)
    if clash:
        raise SchemaError(f"attrs may not override reserved attributes: {sorted(clash)}")
    global_attrs.update(
        pcsp_schema_version=SCHEMA_VERSION,
        component=component,
        fidelity=fidelity,
        model=model,
        model_version=model_version,
        params_json=canonical_json(params),
        param_hash=param_hash(params),
        created_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    ds = xr.Dataset(data_vars=xr_vars, coords=xr_coords, attrs=global_attrs)
    validate(ds)
    return ds


def get_params(ds: xr.Dataset) -> dict[str, Any]:
    """Return the input parameters recorded in a dataset."""
    return json.loads(ds.attrs["params_json"])


def validate(ds: xr.Dataset, *, require_provenance: bool = False) -> None:
    """Raise :class:`SchemaError` listing every violation; return ``None`` if clean."""
    problems: list[str] = []

    for key in REQUIRED_ATTRS:
        if key not in ds.attrs:
            problems.append(f"missing global attribute '{key}'")
    if require_provenance:
        for key in PROVENANCE_ATTRS:
            if key not in ds.attrs:
                problems.append(f"missing provenance attribute '{key}'")
    for key, value in ds.attrs.items():
        if isinstance(value, bool) or not isinstance(value, _ATTR_TYPES):
            problems.append(
                f"global attribute '{key}' has type {type(value).__name__}; "
                "only str, int and float round-trip through Zarr and netCDF"
            )

    component = ds.attrs.get("component")
    if component is not None and component not in COMPONENTS:
        problems.append(f"component '{component}' is not one of {COMPONENTS}")
    fidelity = ds.attrs.get("fidelity")
    if fidelity is not None and fidelity not in FIDELITIES:
        problems.append(f"fidelity '{fidelity}' is not one of {FIDELITIES}")

    params_json = ds.attrs.get("params_json")
    if isinstance(params_json, str):
        try:
            params = json.loads(params_json)
        except json.JSONDecodeError as exc:
            problems.append(f"params_json is not valid JSON: {exc}")
        else:
            if not isinstance(params, dict):
                problems.append("params_json must encode a JSON object")
            elif "param_hash" in ds.attrs and ds.attrs["param_hash"] != param_hash(params):
                problems.append("param_hash does not match params_json")

    for dim in ds.dims:
        if dim not in ds.coords:
            problems.append(f"dimension '{dim}' has no coordinate variable")

    for name, variable in {**ds.coords, **ds.data_vars}.items():
        for key in ("units", "long_name"):
            value = variable.attrs.get(key)
            if not isinstance(value, str) or not value:
                problems.append(f"'{name}' is missing a non-empty '{key}' attribute")
        if variable.attrs.get("units") in _NON_SI_TEMPERATURE_UNITS:
            problems.append(
                f"'{name}' uses units '{variable.attrs['units']}'; store temperatures in K"
            )
        if variable.dtype.kind not in "fiub":
            problems.append(f"'{name}' has non-numeric dtype {variable.dtype}")

    if "time" in ds.coords:
        time = ds.coords["time"]
        if time.attrs.get("units") != "s":
            problems.append("coordinate 'time' must have units 's'")
        if time.ndim == 1 and time.size > 1 and not np.all(np.diff(time.values) > 0):
            problems.append("coordinate 'time' must be strictly increasing")

    if problems:
        raise SchemaError("dataset violates the pcsp schema:\n  - " + "\n  - ".join(problems))
