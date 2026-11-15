import numpy as np
import pytest
import xarray as xr

from pcsp.data import Coord, SchemaError, Var, is_stamped, load, new_dataset, save, stamp
from pcsp.data import io as pio

PROVENANCE = {"git_sha": "abc123", "git_dirty": 0, "pcsp_version": "0.1.0"}

pytestmark = pytest.mark.filterwarnings("error")  # e.g. zarr's consolidated-metadata warning


@pytest.fixture
def awkward_dataset():
    """Values that lossy or over-eager encoders get wrong."""
    return new_dataset(
        component="benchmark",
        fidelity="reduced",
        model="io-test",
        model_version="0",
        params={"n": 3, "label": "a"},
        coords={
            "time": Coord(np.array([0.0, 0.5, 1.0]), "s", "time"),
            "cell": Coord(np.arange(4, dtype=np.int32), "1", "cell index"),
        },
        data_vars={
            "T": Var(("time", "cell"), np.array([[np.nan, 1e-300, -0.0, 3.3]] * 3), "K", "temp"),
            "q32": Var("cell", np.array([1.5, 2.5, np.inf, 4.5], dtype=np.float32), "W", "q"),
            "count": Var("cell", np.array([-1, 0, 7, 2**40]), "1", "count"),
            "residence": Var("cell", np.ones(4), "seconds", "would decode as timedelta"),
            "efficiency": Var((), 0.875, "1", "scalar"),
        },
        attrs={"n_cells": 4, "dx": 0.25},
    )


def _assert_bit_identical(a: xr.Dataset, b: xr.Dataset) -> None:
    xr.testing.assert_identical(a, b)
    for name, variable in a.variables.items():
        assert variable.dtype == b[name].dtype, name
        assert variable.values.tobytes() == b[name].values.tobytes(), name
    for key, value in a.attrs.items():
        assert type(b.attrs[key]) is type(value), key


@pytest.mark.parametrize("suffix", [".zarr", ".nc"])
def test_round_trip_is_bit_identical(tmp_path, awkward_dataset, suffix):
    ds = stamp(awkward_dataset, PROVENANCE)
    path = save(ds, tmp_path / f"run{suffix}")
    _assert_bit_identical(ds, load(path))


def test_round_trip_across_formats(tmp_path, awkward_dataset):
    # Encodings left over from one format must not leak into the other.
    ds = stamp(awkward_dataset, PROVENANCE)
    via_zarr = load(save(ds, tmp_path / "a.zarr"))
    via_nc = load(save(via_zarr, tmp_path / "b.nc"))
    _assert_bit_identical(ds, load(save(via_nc, tmp_path / "c.zarr")))


def test_save_stamps_unstamped_and_keeps_existing_stamp(tmp_path, toy_dataset):
    assert not is_stamped(toy_dataset)
    loaded = load(save(toy_dataset, tmp_path / "fresh.zarr", provenance=PROVENANCE))
    assert loaded.attrs["git_sha"] == "abc123"
    original = stamp(toy_dataset, {**PROVENANCE, "git_sha": "producer"})
    loaded = load(save(original, tmp_path / "kept.zarr", provenance=PROVENANCE))
    assert loaded.attrs["git_sha"] == "producer"


@pytest.mark.parametrize("suffix", [".zarr", ".nc"])
def test_refuses_to_overwrite_unless_asked(tmp_path, toy_dataset, suffix):
    path = tmp_path / f"run{suffix}"
    save(toy_dataset, path, provenance=PROVENANCE)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        save(toy_dataset, path, provenance=PROVENANCE)
    newer = stamp(toy_dataset, {**PROVENANCE, "git_sha": "newer"})
    save(newer, path, overwrite=True)
    assert load(path).attrs["git_sha"] == "newer"
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]  # no leftovers


def test_invalid_dataset_writes_nothing(tmp_path, toy_dataset):
    bad = toy_dataset.copy()
    bad["u"].attrs["units"] = "degC"
    with pytest.raises(SchemaError, match="store temperatures in K"):
        save(bad, tmp_path / "bad.zarr", provenance=PROVENANCE)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("suffix", [".zarr", ".nc"])
def test_interrupted_write_leaves_previous_dataset_intact(
    tmp_path, toy_dataset, monkeypatch, suffix
):
    path = tmp_path / f"run{suffix}"
    save(stamp(toy_dataset, PROVENANCE), path)
    real_to_zarr, real_to_netcdf = xr.Dataset.to_zarr, xr.Dataset.to_netcdf

    def crash_after_partial_write(real):
        def write(self, target, *args, **kwargs):
            kwargs.pop("encoding")
            real(self.drop_vars("u"), target, *args, **kwargs)  # something lands on disk...
            raise KeyboardInterrupt  # ...then the process is interrupted

        return write

    monkeypatch.setattr(xr.Dataset, "to_zarr", crash_after_partial_write(real_to_zarr))
    monkeypatch.setattr(xr.Dataset, "to_netcdf", crash_after_partial_write(real_to_netcdf))
    newer = stamp(toy_dataset, {**PROVENANCE, "git_sha": "newer"})
    with pytest.raises(KeyboardInterrupt):
        save(newer, path, overwrite=True)
    with pytest.raises(KeyboardInterrupt):
        save(newer, tmp_path / f"new{suffix}")
    monkeypatch.undo()

    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]  # temp copies removed
    survivor = load(path)
    assert survivor.attrs["git_sha"] == "abc123" and "u" in survivor


def test_load_rejects_data_without_provenance(tmp_path, toy_dataset):
    path = tmp_path / "raw.zarr"
    toy_dataset.to_zarr(path, consolidated=False)
    with pytest.raises(SchemaError, match="git_sha"):
        load(path)


def test_clear_errors_for_bad_paths(tmp_path, toy_dataset):
    with pytest.raises(ValueError, match="'.zarr' directory"):
        save(toy_dataset, tmp_path / "run.h5", provenance=PROVENANCE)
    with pytest.raises(FileNotFoundError, match="no dataset"):
        load(tmp_path / "missing.zarr")


def test_chunking_splits_only_the_first_dimension():
    small = xr.Variable(("time", "x"), np.zeros((10, 20)))
    assert pio._chunk_shape(small) == (10, 20)
    big = xr.Variable(("time", "x"), np.zeros((1000, 2**12)))  # 32 MiB of float64
    assert pio._chunk_shape(big) == (pio.CHUNK_TARGET_BYTES // (8 * 2**12), 2**12)


def test_stored_chunks_and_compression(tmp_path, toy_dataset):
    path = save(toy_dataset, tmp_path / "run.zarr", provenance=PROVENANCE)
    with xr.open_dataset(path, engine="zarr", consolidated=False, chunks=None) as stored:
        enc = stored["u"].encoding
    assert enc["chunks"] == toy_dataset["u"].shape
    assert str(enc["compressors"][0].cname) == "zstd"
    assert not (path / "zarr.json").read_text().count('consolidated_metadata": {')
