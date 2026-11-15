"""Metrics against closed-form values for textbook responses."""

import numpy as np
import pytest

from pcsp.twin.control import (
    integral_absolute_error,
    integral_squared_error,
    overshoot,
    peak_deviation,
    regulation_metrics,
    settling_time,
    total_variation,
    two_sigma_band,
)

TAU = 10.0
T = np.arange(0.0, 200.0 + 1e-9, 0.01)
Y_FIRST_ORDER = 1.0 - np.exp(-T / TAU)


def test_integral_errors_of_first_order_step():
    """``int_0^inf e^{-t/tau} dt = tau`` and ``int_0^inf e^{-2t/tau} dt = tau / 2``."""
    assert integral_absolute_error(T, Y_FIRST_ORDER, 1.0) == pytest.approx(TAU, rel=1e-5)
    assert integral_squared_error(T, Y_FIRST_ORDER, 1.0) == pytest.approx(TAU / 2, rel=1e-5)
    # From t_start = tau on: tau * e^{-1}.
    assert integral_absolute_error(T, Y_FIRST_ORDER, 1.0, t_start=TAU) == pytest.approx(
        TAU * np.exp(-1.0), rel=1e-5
    )


def test_settling_time_and_peak_of_first_order_step():
    """2 % settling time of ``1 - e^{-t/tau}`` is ``tau ln 50``, to one sample."""
    assert settling_time(T, Y_FIRST_ORDER, 1.0, band=0.02) == pytest.approx(
        TAU * np.log(50), abs=0.01
    )
    assert settling_time(T, Y_FIRST_ORDER, 1.0, band=1e-12) == np.inf
    assert peak_deviation(T, Y_FIRST_ORDER, 1.0) == -1.0
    assert overshoot(Y_FIRST_ORDER, 0.0, 1.0) == 0.0


def test_overshoot_of_underdamped_second_order():
    """``M_p = exp(-pi zeta / sqrt(1 - zeta^2))`` = 16.3 % for ``zeta = 0.5``."""
    zeta, t = 0.5, np.arange(0.0, 20.0, 1e-3)
    wd = np.sqrt(1 - zeta**2)
    y = 1 - np.exp(-zeta * t) * (np.cos(wd * t) + zeta / wd * np.sin(wd * t))
    assert overshoot(y, 0.0, 1.0) == pytest.approx(100 * np.exp(-np.pi * zeta / wd), rel=1e-5)
    assert overshoot(-y, 0.0, -1.0) == pytest.approx(100 * np.exp(-np.pi * zeta / wd), rel=1e-5)


def test_two_sigma_band_of_a_sine():
    """Over whole periods a sine of amplitude A has ``2 sigma = sqrt(2) A``.

    A 300 s window at 1 s sampling holds 301 samples; aligned windows over five
    60 s periods give exactly ``sum sin^2 = 150``, i.e. ``var = A^2 / 2`` with ddof = 1.
    """
    A, t = 4.0, np.arange(0.0, 901.0)
    y = 1073.0 + A * np.sin(2 * np.pi * t / 60.0)
    t_end, fixed = two_sigma_band(t, y, window=300.0, stride=300.0)
    np.testing.assert_allclose(t_end, [300.0, 600.0, 900.0])
    np.testing.assert_allclose(fixed, np.sqrt(2) * A, rtol=1e-12)
    t_end, sliding = two_sigma_band(t, y, window=300.0)
    assert t_end.size == 601
    np.testing.assert_allclose(sliding, np.sqrt(2) * A, rtol=0.01)


def test_total_variation():
    np.testing.assert_allclose(total_variation(np.array([0.0, 1.0, 0.5, 2.0])), [3.0])
    u = np.array([[0.0, 1.0], [1.0, 1.0], [0.0, 3.0]])
    np.testing.assert_allclose(total_variation(u), [2.0, 2.0])


def test_regulation_metrics_bundle_is_consistent():
    u = np.exp(-T / TAU)[:, None]
    m = regulation_metrics(T, Y_FIRST_ORDER, 1.0, u, t_start=5.0, band=0.02, window=100.0)
    assert m.iae == integral_absolute_error(T, Y_FIRST_ORDER, 1.0, 5.0)
    assert m.settling_time == settling_time(T, Y_FIRST_ORDER, 1.0, 0.02, 5.0)
    assert m.peak_deviation == pytest.approx(-np.exp(-0.5))
    np.testing.assert_allclose(m.total_variation, [np.exp(-0.5) - np.exp(-20.0)], rtol=1e-9)
    assert m.two_sigma_max == two_sigma_band(T, Y_FIRST_ORDER, 100.0, t_start=5.0)[1].max()
