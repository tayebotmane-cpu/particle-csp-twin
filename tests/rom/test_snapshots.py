"""DataArray <-> snapshot matrix conversion: shapes, ordering, round trips, metadata."""

import numpy as np
import pytest
import xarray as xr

from pcsp.rom import from_snapshots, to_snapshots


@pytest.fixture
def field() -> xr.DataArray:
    rng = np.random.default_rng(0)
    return xr.DataArray(
        rng.standard_normal((3, 4, 5)),
        dims=("run", "time", "z"),
        coords={
            "run": ("run", [10, 11, 12], {"units": "1", "long_name": "run id"}),
            "time": ("time", [0.0, 1.0, 2.0, 3.0], {"units": "s", "long_name": "time"}),
            "z": ("z", np.linspace(0, 1, 5), {"units": "m", "long_name": "height"}),
        },
        name="T",
        attrs={"units": "K", "long_name": "temperature"},
    )


def test_state_snapshots_have_one_column_per_run_and_time(field):
    X, layout = to_snapshots(field, sample_dims=("run", "time"))
    assert X.shape == (5, 12)
    # Run-major column order: column 4 * run + time.
    np.testing.assert_array_equal(X[:, 4 * 2 + 1], field.isel(run=2, time=1).values)
    xr.testing.assert_identical(from_snapshots(X, layout), field)


def test_space_time_snapshots_have_one_column_per_run(field):
    X, layout = to_snapshots(field, sample_dims="run")
    assert X.shape == (20, 3)
    # Time-major row order: row 5 * time + z.
    assert X[5 * 3 + 2, 1] == field.isel(run=1, time=3, z=2).item()
    xr.testing.assert_identical(from_snapshots(X, layout), field)


def test_single_run_round_trip_and_transposed_input(field):
    one = field.isel(run=0, drop=True)
    X, layout = to_snapshots(one.transpose("z", "time"), sample_dims="time")
    assert X.shape == (5, 4)
    xr.testing.assert_identical(from_snapshots(X, layout), one)


def test_new_columns_take_new_coordinates_with_original_metadata(field):
    X, layout = to_snapshots(field, sample_dims="run")
    out = from_snapshots(X[:, :2], layout, sample_coords={"run": [99, 100]})
    assert out.sizes == {"run": 2, "time": 4, "z": 5}
    np.testing.assert_array_equal(out["run"].values, [99, 100])
    assert out["run"].attrs == field["run"].attrs
    assert out["z"].attrs["units"] == "m"


def test_mismatched_shapes_are_rejected(field):
    X, layout = to_snapshots(field, sample_dims=("run", "time"))
    with pytest.raises(ValueError, match="shape"):
        from_snapshots(X[:-1], layout)
    with pytest.raises(ValueError, match="single sample dimension"):
        from_snapshots(X[:, :5], layout)
    with pytest.raises(ValueError, match="not in"):
        to_snapshots(field, sample_dims="cycle")
