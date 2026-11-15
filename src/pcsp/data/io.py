"""Save and load schema datasets as Zarr (default) or netCDF-4/HDF5.

The format is chosen from the path suffix:

* ``*.zarr`` - a Zarr v3 directory store. The default for project data.
* ``*.nc``   - a single netCDF-4/HDF5 file written with the ``h5netcdf`` engine,
  for exchange with tools that expect netCDF (ParaView, MATLAB, ...).

Storage defaults (all lossless, so a round trip is bit-identical)
-----------------------------------------------------------------
* Compression: Zarr uses Blosc-Zstd level 3 with byte shuffle; netCDF uses
  gzip level 4 with the HDF5 shuffle filter (gzip is the only compressor every
  netCDF reader is guaranteed to have).
* Chunking: each variable is chunked along its *first* dimension only (by
  convention ``time``), with every other dimension kept whole, aiming at
  about 8 MiB per chunk. Small arrays therefore get exactly one chunk and
  large transient fields can still be read a few time steps at a time.
  0-d variables (scalar quantities of interest) are stored uncompressed.
* No ``_FillValue`` is written, so NaN stays NaN and integer data is never
  turned into floats on reading.
* Zarr metadata is *not* consolidated. Consolidated metadata is not part of
  the Zarr v3 specification (zarr warns about it) and only speeds up opening
  stores on high-latency cloud storage, which this project does not use.
  Writing and reading with ``consolidated=False`` avoids the warning without
  suppressing anything.

Writes are atomic with respect to crashes of the writing process: data is
written to a hidden temporary sibling (``.<name>.tmp-<random>``) and renamed
into place only once complete, so the final name either holds a complete
dataset or nothing. (Durability against power loss would additionally need
fsync, which is not done.)
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from zarr.codecs import BloscCodec

from pcsp.data.provenance import is_stamped, stamp
from pcsp.data.schema import validate

CHUNK_TARGET_BYTES = 8 * 2**20
"""Approximate size of one chunk along the first dimension (8 MiB)."""

_ZARR_COMPRESSOR = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
_NETCDF_COMPRESSION = {"compression": "gzip", "compression_opts": 4, "shuffle": True}

# Never let xarray reinterpret our numbers: a variable whose units happen to be
# "seconds" or "days" would otherwise be decoded into timedelta64/datetime64.
_OPEN_KWARGS: dict[str, Any] = {"decode_times": False, "decode_timedelta": False}


def _storage_format(path: Path) -> str:
    if path.suffix == ".zarr":
        return "zarr"
    if path.suffix == ".nc":
        return "netcdf"
    raise ValueError(
        f"cannot infer storage format from '{path}': use a '.zarr' directory (default) "
        "or a '.nc' netCDF-4/HDF5 file"
    )


def _chunk_shape(variable: xr.Variable) -> tuple[int, ...]:
    """Whole array, except the first dimension is split into ~8 MiB chunks."""
    shape = variable.shape
    bytes_per_slice = variable.dtype.itemsize * int(np.prod(shape[1:], dtype=np.int64))
    first = max(1, min(shape[0], CHUNK_TARGET_BYTES // max(1, bytes_per_slice)))
    return (first, *shape[1:])


def _encoding(ds: xr.Dataset, fmt: str) -> dict[str, dict[str, Any]]:
    encoding: dict[str, dict[str, Any]] = {}
    for name, variable in ds.variables.items():
        enc: dict[str, Any] = {"_FillValue": None}
        if variable.ndim > 0 and variable.size > 0:
            chunks = _chunk_shape(variable)
            if fmt == "zarr":
                enc.update(chunks=chunks, compressors=(_ZARR_COMPRESSOR,))
            else:
                enc.update(chunksizes=chunks, **_NETCDF_COMPRESSION)
        encoding[str(name)] = enc
    return encoding


def _without_encoding(ds: xr.Dataset) -> xr.Dataset:
    """Shallow copy with storage encodings dropped (they are format-specific)."""
    out = ds.copy(deep=False)
    for variable in out.variables.values():
        variable.encoding = {}
    out.encoding = {}
    return out


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def save(
    ds: xr.Dataset,
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    provenance: dict[str, str | int] | None = None,
) -> Path:
    """Validate, stamp (if needed) and atomically write ``ds`` to ``path``.

    Parameters
    ----------
    ds : a schema dataset. If it has no provenance stamp yet it is stamped
        (with ``provenance`` if given, otherwise collected now); an existing
        stamp is kept, because it describes the code that produced the data.
    path : ``*.zarr`` (directory) or ``*.nc`` (file). Parent directories are
        created.
    overwrite : replace an existing dataset at ``path``. Without it an existing
        path raises ``FileExistsError``.

    Returns the path written. Nothing is written if validation fails.
    """
    path = Path(path)
    fmt = _storage_format(path)
    if not is_stamped(ds):
        ds = stamp(ds, provenance)
    validate(ds, require_provenance=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"'{path}' already exists; pass overwrite=True to replace it")

    ds = _without_encoding(ds)
    path.parent.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex[:12]
    tmp = path.with_name(f".{path.name}.tmp-{tag}")
    try:
        if fmt == "zarr":
            ds.to_zarr(
                tmp, mode="w-", zarr_format=3, consolidated=False, encoding=_encoding(ds, fmt)
            )
        else:
            ds.to_netcdf(tmp, engine="h5netcdf", encoding=_encoding(ds, fmt))
        _move_into_place(tmp, path, overwrite=overwrite, tag=tag)
    except BaseException:
        _remove(tmp)
        raise
    return path


def _move_into_place(tmp: Path, path: Path, *, overwrite: bool, tag: str) -> None:
    """Rename the finished temporary ``tmp`` to ``path``."""
    if not path.exists() and not path.is_symlink():
        # os.rename would silently replace an empty directory, so check first.
        os.rename(tmp, path)
        return
    if not overwrite:
        raise FileExistsError(f"'{path}' appeared while writing; not overwriting it")
    if tmp.is_file() and path.is_file():
        os.replace(tmp, path)  # atomic for files
        return
    # A directory cannot be atomically replaced: move the old one aside first,
    # so the final name only ever points at a complete dataset (or briefly at
    # nothing), then delete the old copy.
    old = path.with_name(f".{path.name}.old-{tag}")
    os.rename(path, old)
    os.rename(tmp, path)
    _remove(old)


def _plain(value: Any) -> Any:
    """NumPy scalars (h5netcdf returns attributes as these) to Python scalars."""
    return value.item() if isinstance(value, np.generic) else value


def load(path: str | os.PathLike[str]) -> xr.Dataset:
    """Read a dataset written by :func:`save` fully into memory and validate it.

    The file is closed before returning. Raises ``FileNotFoundError`` if
    ``path`` does not exist and :class:`~pcsp.data.SchemaError` if the stored
    data does not satisfy the schema including provenance.
    """
    path = Path(path)
    fmt = _storage_format(path)
    if not path.exists():
        raise FileNotFoundError(f"no dataset at '{path}'")
    if fmt == "zarr":
        opened = xr.open_dataset(
            path, engine="zarr", consolidated=False, chunks=None, **_OPEN_KWARGS
        )
    else:
        opened = xr.open_dataset(path, engine="h5netcdf", **_OPEN_KWARGS)
    with opened:
        ds = _without_encoding(opened.load())
    ds.attrs = {k: _plain(v) for k, v in ds.attrs.items()}
    for variable in ds.variables.values():
        variable.attrs = {k: _plain(v) for k, v in variable.attrs.items()}
    validate(ds, require_provenance=True)
    return ds
