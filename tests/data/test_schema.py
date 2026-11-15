import dataclasses
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pcsp.data import Coord, SchemaError, Var, get_params, new_dataset, param_hash, validate
from pcsp.data.schema import canonical_json, to_jsonable


@dataclasses.dataclass(frozen=True)
class Params:
    mass_flow: float = 5.0
    T_in: float = 823.15
    grid: tuple[int, ...] = (10, 20)


def make_dataset(**overrides):
    kwargs = dict(
        component="storage_bin",
        fidelity="reduced",
        model="unit-test",
        model_version="0.0.1",
        params=Params(),
        coords={
            "time": Coord(np.arange(4.0), "s", "time"),
            "z": Coord(np.linspace(0.0, 1.0, 3), "m", "axial position"),
        },
        data_vars={
            "T_s": Var(("time", "z"), np.full((4, 3), 600.0), "K", "solid temperature"),
            "efficiency": Var((), 0.9, "1", "storage efficiency"),
        },
    )
    kwargs.update(overrides)
    return new_dataset(**kwargs)


def test_new_dataset_is_valid_and_self_describing():
    ds = make_dataset()
    assert ds.attrs["component"] == "storage_bin"
    assert ds["T_s"].attrs == {"units": "K", "long_name": "solid temperature"}
    assert ds["efficiency"].ndim == 0
    assert get_params(ds) == {"mass_flow": 5.0, "T_in": 823.15, "grid": [10, 20]}
    assert ds.attrs["param_hash"] == param_hash(Params())


def test_param_hash_is_order_independent_and_value_sensitive():
    assert param_hash({"a": 1.0, "b": 2}) == param_hash({"b": 2, "a": 1.0})
    assert param_hash({"a": 1.0}) != param_hash({"a": 1.0000001})
    assert param_hash(Params()) == param_hash(dataclasses.asdict(Params()))


def test_to_jsonable_handles_numpy_paths_and_rejects_callables():
    obj = {"n": np.int64(3), "x": np.array([1.0, 2.0]), "p": Path("a/b"), "t": (1, 2)}
    assert to_jsonable(obj) == {"n": 3, "x": [1.0, 2.0], "p": "a/b", "t": [1, 2]}
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    with pytest.raises(TypeError, match="not JSON-serialisable"):
        to_jsonable({"f": lambda x: x})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (dict(component="boiler"), "component 'boiler'"),
        (dict(fidelity="guess"), "fidelity 'guess'"),
        (
            dict(data_vars={"T": Var(("z",), np.ones(3), "degC", "temperature")}),
            "store temperatures in K",
        ),
        (dict(data_vars={"T": Var(("z",), np.ones(3), "", "temperature")}), "'units'"),
        (dict(data_vars={"T": Var(("y",), np.ones(3), "K", "temperature")}), "dimension 'y'"),
        (
            dict(coords={"time": Coord([0.0, 2.0, 1.0], "s", "time")}, data_vars={}),
            "strictly increasing",
        ),
        (dict(coords={"time": Coord([0.0, 1.0], "min", "time")}, data_vars={}), "units 's'"),
        (dict(attrs={"flag": True}), "type bool"),
    ],
)
def test_violations_are_rejected(overrides, message):
    with pytest.raises(SchemaError, match=message):
        make_dataset(**overrides)


def test_reserved_attributes_cannot_be_overridden():
    with pytest.raises(SchemaError, match="reserved"):
        make_dataset(attrs={"component": "receiver_fpr"})


def test_all_problems_are_reported_together():
    ds = make_dataset()
    del ds.attrs["model"]
    ds["T_s"].attrs.pop("units")
    with pytest.raises(SchemaError) as excinfo:
        validate(ds)
    assert "'model'" in str(excinfo.value) and "'units'" in str(excinfo.value)


def test_tampered_parameters_are_detected():
    ds = make_dataset()
    ds.attrs["params_json"] = '{"mass_flow": 99.0}'
    with pytest.raises(SchemaError, match="param_hash does not match"):
        validate(ds)


def test_provenance_is_required_only_when_asked():
    ds = make_dataset()
    validate(ds)
    with pytest.raises(SchemaError, match="git_sha"):
        validate(ds, require_provenance=True)
    ds.attrs.update(git_sha="abc123", git_dirty=0, pcsp_version="0.1.0")
    validate(ds, require_provenance=True)


@pytest.mark.parametrize("engine", ["zarr", "h5netcdf"])
def test_round_trip_through_storage(tmp_path, engine):
    ds = make_dataset()
    if engine == "zarr":
        path = tmp_path / "ds.zarr"
        ds.to_zarr(path, consolidated=False)
        loaded = xr.open_zarr(path, consolidated=False).load()
    else:
        path = tmp_path / "ds.nc"
        ds.to_netcdf(path, engine="h5netcdf")
        loaded = xr.open_dataset(path, engine="h5netcdf").load()
    validate(loaded)
    xr.testing.assert_identical(loaded, ds)
