import math

import numpy as np
import pytest

from pcsp.data import ParameterSpace, Range, full_factorial, latin_hypercube, param_hash, sobol


def _column(samples, name):
    return np.array([s[name] for s in samples])


@pytest.mark.parametrize("sampler", [latin_hypercube, sobol])
def test_samplers_are_reproducible_from_seed(space, sampler):
    a, b, c = sampler(space, 8, seed=3), sampler(space, 8, seed=3), sampler(space, 8, seed=4)
    assert a == b
    assert [param_hash(s) for s in a] != [param_hash(s) for s in c]


@pytest.mark.parametrize("sampler", [latin_hypercube, sobol])
def test_samples_are_plain_dicts_inside_bounds_with_fixed_values(space, sampler):
    samples = sampler(space, 16, seed=0)
    assert len(samples) == 16
    for s in samples:
        assert set(s) == {"speed", "amplitude", "model_tag"}
        assert type(s["speed"]) is float and s["model_tag"] == "pulse"
        assert 0.1 <= s["speed"] <= 0.6 and 0.5 <= s["amplitude"] <= 2.0


def test_latin_hypercube_hits_every_stratum_once(space):
    n = 10
    samples = latin_hypercube(space, n, seed=1)
    speed_strata = np.floor((_column(samples, "speed") - 0.1) / 0.5 * n)
    log_amp = np.log(_column(samples, "amplitude"))
    amp_strata = np.floor((log_amp - math.log(0.5)) / (math.log(2.0) - math.log(0.5)) * n)
    assert sorted(speed_strata) == list(range(n))
    assert sorted(amp_strata) == list(range(n))  # stratified in log space


def test_sobol_requires_power_of_two(space):
    with pytest.raises(ValueError, match="power of two"):
        sobol(space, 10, seed=0)


def test_full_factorial_grid(space):
    samples = full_factorial(space, {"speed": 3, "amplitude": 2})
    assert len(samples) == 6
    assert _column(samples, "speed").tolist() == [0.1, 0.1, 0.35, 0.35, 0.6, 0.6]  # slowest
    assert _column(samples, "amplitude").tolist() == [0.5, 2.0] * 3  # exact range ends
    log_levels = full_factorial(ParameterSpace({"d": Range(1e-4, 1e-2, scale="log")}), 3)
    assert _column(log_levels, "d") == pytest.approx([1e-4, 1e-3, 1e-2], rel=1e-12)
    midpoint = full_factorial(ParameterSpace({"d": Range(1.0, 100.0, scale="log")}), 1)
    assert midpoint[0]["d"] == pytest.approx(10.0)


def test_fixed_values_are_independent_copies():
    space = ParameterSpace({"a": Range(0.0, 1.0)}, fixed={"grid": [10, 20]})
    samples = full_factorial(space, 2)
    samples[0]["grid"].append(30)
    assert samples[1]["grid"] == [10, 20]
    assert space.fixed["grid"] == [10, 20]


@pytest.mark.parametrize(
    ("make", "error", "message"),
    [
        (lambda: Range(1.0, 1.0), ValueError, "low < high"),
        (lambda: Range(0.0, 1.0, scale="log"), ValueError, "low > 0"),
        (lambda: Range(0.0, math.inf), ValueError, "finite"),
        (lambda: ParameterSpace({}), ValueError, "at least one"),
        (lambda: ParameterSpace({"a": (0, 1)}), TypeError, "must be a Range"),
        (lambda: ParameterSpace({"a": Range(0, 1)}, fixed={"a": 2}), ValueError, "both"),
        (lambda: ParameterSpace({"a": Range(0, 1)}, fixed={"f": print}), TypeError, "JSON"),
    ],
)
def test_invalid_spaces_are_rejected(make, error, message):
    with pytest.raises(error, match=message):
        make()


def test_full_factorial_levels_must_match_parameters(space):
    with pytest.raises(ValueError, match="exactly the varied parameters"):
        full_factorial(space, {"speed": 3})
