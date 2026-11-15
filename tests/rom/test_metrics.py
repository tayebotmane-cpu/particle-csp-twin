"""Error measures and timing utility."""

import time

import numpy as np
import pytest

from pcsp.rom import fit_pod, projection_error, relative_error, time_full_vs_rom


def test_relative_error_hand_calculation():
    X = np.array([[3.0, 0.0], [4.0, 1.0]])
    Y = np.array([[3.0, 0.0], [4.0, 0.0]])
    # ||X - Y||_F = 1, ||X||_F = sqrt(26); per column: 0/5 and 1/1.
    assert relative_error(X, Y) == pytest.approx(1.0 / np.sqrt(26.0), rel=1e-14)
    np.testing.assert_allclose(relative_error(X, Y, axis=0), [0.0, 1.0], atol=1e-15)
    # Weights w = (1, 4): ||X||_W^2 = 9 + 64 + 4 = 77, ||X - Y||_W^2 = 4.
    assert relative_error(X, Y, weights=[1.0, 4.0]) == pytest.approx(2.0 / np.sqrt(77.0))


def test_projection_error_of_training_data_is_discarded_energy():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 20)) @ np.diag(0.6 ** np.arange(20))
    pod = fit_pod(X, rank=4)
    s = pod.singular_values
    expected = np.sqrt(np.sum(s[4:] ** 2) / np.sum(s**2))
    assert projection_error(pod, X) == pytest.approx(expected, rel=1e-10)


def test_timing_reports_speedup_of_the_faster_callable():
    timing = time_full_vs_rom(lambda: time.sleep(0.02), lambda: time.sleep(0.001), repeats=3)
    assert timing.full_seconds >= 0.02
    assert timing.rom_seconds >= 0.001
    assert timing.speedup == pytest.approx(timing.full_seconds / timing.rom_seconds)
    assert timing.speedup > 2.0
