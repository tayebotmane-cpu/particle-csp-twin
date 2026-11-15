"""Convert between :class:`xarray.DataArray` fields and snapshot matrices.

Snapshot convention: a snapshot matrix has shape ``(n_dof, n_snapshots)``.

The caller says which dimensions index *snapshots* (``sample_dims``); every other
dimension is flattened into the degrees of freedom (rows), in C order of the
remaining dimensions as they appear in the array. The three uses in this
package are, for ``T`` with dims ``(run, time, z)``:

* ``to_snapshots(T, sample_dims=("run", "time"))`` -> ``(n_z, n_run * n_time)``:
  one column per state, for POD of states, DMD or operator inference
  (columns ordered run-major: all times of run 0, then run 1, ...);
* ``to_snapshots(T, sample_dims="run")`` -> ``(n_time * n_z, n_run)``:
  one *space-time* column per run, for :class:`pcsp.rom.interp.PODInterpolationROM`
  (rows ordered time-major: all ``z`` at ``time[0]``, then ``time[1]``, ...);
* for a single run with dims ``(time, z)``: ``sample_dims="time"`` -> ``(n_z, n_time)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class SnapshotLayout:
    """Everything needed to rebuild a DataArray from a snapshot matrix.

    ``dof_dims``/``sample_dims`` are dimension names, ``shape`` maps every dimension
    to its length and ``coords`` holds the original coordinate variables (with their
    ``units``/``long_name`` attributes).
    """

    dof_dims: tuple[str, ...]
    sample_dims: tuple[str, ...]
    shape: Mapping[str, int]
    coords: Mapping[str, xr.Variable]
    name: Any
    attrs: Mapping[str, Any]

    @property
    def n_dof(self) -> int:
        return int(np.prod([self.shape[d] for d in self.dof_dims]))


def to_snapshots(
    da: xr.DataArray, *, sample_dims: str | Sequence[str]
) -> tuple[np.ndarray, SnapshotLayout]:
    """Flatten ``da`` into a snapshot matrix of shape ``(n_dof, n_snapshots)``.

    ``sample_dims`` are the dimensions whose combinations become columns; all other
    dimensions become rows. Returns the matrix and the layout for
    :func:`from_snapshots`.
    """
    sample = (sample_dims,) if isinstance(sample_dims, str) else tuple(sample_dims)
    missing = [d for d in sample if d not in da.dims]
    if missing:
        raise ValueError(f"sample dims {missing} not in {da.dims}")
    dof = tuple(str(d) for d in da.dims if d not in sample)
    if not dof:
        raise ValueError("at least one dimension must remain as degrees of freedom")
    values = da.transpose(*sample, *dof).values
    n_samples = int(np.prod(values.shape[: len(sample)]))
    X = values.reshape(n_samples, -1).T.astype(float)
    layout = SnapshotLayout(
        dof_dims=dof,
        sample_dims=sample,
        shape={str(d): int(n) for d, n in da.sizes.items()},
        coords={str(k): v.variable for k, v in da.coords.items() if k in da.dims},
        name=da.name,
        attrs=dict(da.attrs),
    )
    return X, layout


def from_snapshots(
    X: ArrayLike,
    layout: SnapshotLayout,
    *,
    sample_coords: Mapping[str, ArrayLike | xr.Variable] | None = None,
) -> xr.DataArray:
    """Inverse of :func:`to_snapshots` for a snapshot matrix of shape ``(n_dof, n_snapshots)``.

    The number of rows must equal the layout's ``n_dof``. If the number of columns
    differs from the original (e.g. ROM predictions at new parameters), the layout
    must have exactly one sample dimension and ``sample_coords`` should give its new
    coordinate values (otherwise ``0, 1, 2, ...`` with ``units="1"`` is used).
    Dimensions are returned in the order ``(*sample_dims, *dof_dims)``.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] != layout.n_dof:
        raise ValueError(f"X must have shape ({layout.n_dof}, n_snapshots), got {X.shape}")
    coords = {d: layout.coords[d] for d in layout.dof_dims if d in layout.coords}
    sample_shape = [layout.shape[d] for d in layout.sample_dims]
    new_coords = dict(sample_coords or {})

    if X.shape[1] == int(np.prod(sample_shape)) and not new_coords:
        coords.update({d: layout.coords[d] for d in layout.sample_dims if d in layout.coords})
    else:
        if len(layout.sample_dims) != 1:
            raise ValueError(
                "column count differs from the original layout; this is only supported "
                "with a single sample dimension"
            )
        dim = layout.sample_dims[0]
        sample_shape = [X.shape[1]]
        values = new_coords.get(dim, np.arange(X.shape[1]))
        if isinstance(values, xr.Variable):
            coords[dim] = values
        else:
            # Keep the original coordinate's units/long_name when new values are given.
            if dim in new_coords and dim in layout.coords:
                attrs = dict(layout.coords[dim].attrs)
            else:
                attrs = {"units": "1", "long_name": f"{dim} index"}
            coords[dim] = xr.Variable((dim,), np.asarray(values), attrs=attrs)

    dof_shape = [layout.shape[d] for d in layout.dof_dims]
    values = X.T.reshape(*sample_shape, *dof_shape)
    return xr.DataArray(
        values,
        dims=(*layout.sample_dims, *layout.dof_dims),
        coords=coords,
        name=layout.name,
        attrs=dict(layout.attrs),
    )
