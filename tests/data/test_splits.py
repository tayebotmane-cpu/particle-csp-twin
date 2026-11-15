import numpy as np
import pytest
import xarray as xr

from pcsp.data import (
    Box,
    Split,
    SplitError,
    Threshold,
    apply_split,
    check_folds,
    check_split,
    holdout_split,
    kfold_split,
    random_split,
)


def _split(train, val, test):
    return Split(*(np.array(x, dtype=np.intp) for x in (train, val, test)))


@pytest.fixture
def params():
    """A 5 x 5 grid of operating conditions, as a plain mapping of run columns."""
    speed, amplitude = np.meshgrid(np.linspace(0.1, 0.5, 5), np.linspace(1.0, 3.0, 5))
    return {"speed": speed.ravel(), "amplitude": amplitude.ravel()}


def test_random_split_is_a_reproducible_partition_of_runs():
    split = random_split(20, val_fraction=0.2, test_fraction=0.25, seed=7)
    check_split(split, 20)
    assert (split.train.size, split.val.size, split.test.size) == (11, 4, 5)
    again = random_split(20, val_fraction=0.2, test_fraction=0.25, seed=7)
    assert all(np.array_equal(a, b) for a, b in zip(split, again, strict=True))
    other = random_split(20, val_fraction=0.2, test_fraction=0.25, seed=8)
    assert not np.array_equal(split.test, other.test)


def test_threshold_holds_out_everything_strictly_above(params):
    split = holdout_split(params, Threshold("speed", 0.4), val_fraction=0.25, seed=0)
    check_split(split, 25)
    assert np.all(params["speed"][split.test] > 0.4)
    trained_on = np.concatenate([split.train, split.val])
    assert np.all(params["speed"][trained_on] <= 0.4)
    assert (split.test.size, split.val.size) == (5, 5)  # val is 25 % of the 20 runs left


def test_threshold_below(params):
    split = holdout_split(params, Threshold("amplitude", 1.5, side="below"))
    assert np.all(params["amplitude"][split.test] < 1.5) and split.test.size == 5
    assert split.val.size == 0


def test_box_inside_and_outside(params):
    box = {"speed": (0.2, 0.4), "amplitude": (1.5, 2.5)}  # closed intervals: 3 x 3 runs
    inside = holdout_split(params, Box(box))
    outside = holdout_split(params, Box(box, inside=False))
    assert inside.test.size == 9 and outside.test.size == 16
    assert np.array_equal(inside.test, outside.train)  # train on the centre, test on the edges


def test_holdout_needs_runs_on_both_sides(params):
    with pytest.raises(SplitError, match="no run"):
        holdout_split(params, Threshold("speed", 10.0))
    with pytest.raises(SplitError, match="every run"):
        holdout_split(params, Threshold("speed", 0.0))
    with pytest.raises(KeyError, match="available"):
        holdout_split(params, Threshold("mass_flow", 1.0))


def test_holdout_accepts_a_stacked_dataset(params):
    ds = xr.Dataset(coords={name: ("run", values) for name, values in params.items()})
    split = holdout_split(ds, Threshold("speed", 0.4))
    np.testing.assert_array_equal(split.test, np.flatnonzero(params["speed"] > 0.4))


def test_kfold_tests_every_run_exactly_once():
    folds = kfold_split(11, 4, seed=0)
    check_folds(folds, 11)
    assert sorted(f.test.size for f in folds) == [2, 3, 3, 3]
    with pytest.raises(ValueError, match="between 2"):
        kfold_split(3, 4, seed=0)


@pytest.mark.parametrize(
    ("split", "message"),
    [
        (_split([0, 1, 2], [], [2, 3]), r"runs \[2\] are in both train and test"),
        (_split([0, 1], [2], []), "test set is empty"),
        (_split([0, 1], [], [2]), r"runs \[3\] are in no set"),
        (_split([0, 1, 2], [], [3, 4]), "outside 0..3"),
        (_split([0, 0, 1, 2], [], [3]), "more than once"),
    ],
)
def test_check_split_catches_leaks_and_gaps(split, message):
    with pytest.raises(SplitError, match=message):
        check_split(split, 4)


def test_check_folds_catches_a_run_tested_twice():
    folds = [_split([2, 3], [], [0, 1]), _split([0, 3], [], [1, 2])]
    with pytest.raises(SplitError, match="exactly one test fold"):
        check_folds(folds, 4)


def test_apply_split_selects_whole_runs():
    ds = xr.Dataset(
        {"u": (("run", "time"), np.arange(12.0).reshape(4, 3))},
        coords={"run": np.arange(4), "time": [0.0, 1.0, 2.0]},
    )
    parts = apply_split(ds, _split([0, 3], [1], [2]))
    assert parts["train"]["run"].values.tolist() == [0, 3]
    assert parts["test"]["u"].values.tolist() == [[6.0, 7.0, 8.0]]
    assert parts["val"].sizes == {"run": 1, "time": 3}
    with pytest.raises(SplitError):
        apply_split(ds, _split([0], [1], [2]))  # run 3 missing
