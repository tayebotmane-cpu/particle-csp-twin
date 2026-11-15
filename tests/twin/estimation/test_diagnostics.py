"""Consistency statistics and observability on cases with known answers."""

import numpy as np
import pytest

from pcsp.core import ControlModelBase, ops_for
from pcsp.twin.estimation import chi2_bounds, nees, nis, observability, rmse


def test_nees_by_hand_and_batched():
    P = np.diag([4.0, 1.0])
    assert nees(np.array([2.0, 1.0]), P) == pytest.approx(2.0)  # 4/4 + 1/1
    errors = np.array([[2.0, 0.0], [0.0, 3.0]])
    np.testing.assert_allclose(nees(errors, np.stack([P, P])), [1.0, 9.0])
    assert nis(np.array([3.0]), np.array([[9.0]])) == pytest.approx(1.0)


def test_chi2_bounds_match_tabulated_values():
    # 95 % two-sided interval for one χ²₂ sample.
    assert chi2_bounds(2, 1, 0.95) == pytest.approx((0.0506, 7.378), abs=1e-3)
    # Bar-Shalom et al. (2001), sec. 5.4.2: 50 runs, n = 2 -> [1.5, 2.6].
    lo, hi = chi2_bounds(2, 50, 0.95)
    assert (round(lo, 1), round(hi, 1)) == (1.5, 2.6)


def test_rmse():
    est = np.array([[1.0, 0.0], [3.0, 0.0]])
    truth = np.zeros((2, 2))
    np.testing.assert_allclose(rmse(est, truth), [np.sqrt(5.0), 0.0])
    assert rmse(est, truth, axis=None) == pytest.approx(np.sqrt(2.5))


class DoubleIntegrator(ControlModelBase):
    """ẍ = u, with either the position or the velocity measured."""

    state_names = ("pos", "vel")
    input_names = ("u",)

    def __init__(self, measure: str) -> None:
        self.output_names = (measure,)
        self._k = self.state_names.index(measure)

    def rhs(self, x, u, d):
        xp = ops_for(x, u, d)
        return xp.concat([x[1], u[0]])

    def output(self, x, u, d):
        xp = ops_for(x, u, d)
        return xp.concat([x[self._k]])


def test_observability_of_double_integrator():
    args = (np.zeros(2), np.zeros(1), np.zeros(0), 0.1)
    pos = observability(DoubleIntegrator("pos"), *args)
    assert pos.observable and pos.rank == 2
    # O = [[1, 0], [1, 0.1]] exactly (RK4 is exact for this system).
    O = np.array([[1.0, 0.0], [1.0, 0.1]])
    np.testing.assert_allclose(pos.singular_values, np.linalg.svd(O, compute_uv=False))
    assert pos.condition_number == pytest.approx(np.linalg.cond(O))

    vel = observability(DoubleIntegrator("vel"), *args)
    assert not vel.observable and vel.rank == 1
    np.testing.assert_allclose(np.abs(vel.unobservable_directions[:, 0]), [1.0, 0.0])


def test_state_scaling_changes_conditioning_not_rank():
    model = DoubleIntegrator("pos")
    args = (np.zeros(2), np.zeros(1), np.zeros(0), 0.1)
    plain = observability(model, *args)
    scaled = observability(model, *args, state_scale=np.array([1.0, 10.0]))
    assert scaled.rank == plain.rank == 2
    # Velocity in units of 10: column 2 of O scales by 10, making it better conditioned.
    O = np.array([[1.0, 0.0], [1.0, 1.0]])
    assert scaled.condition_number == pytest.approx(np.linalg.cond(O))
