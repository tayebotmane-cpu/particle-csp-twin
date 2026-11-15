"""Counterflow enthalpy-balance solver against the analytic effectiveness-NTU solution."""

import numpy as np
import pytest

from pcsp.models.hx import effectiveness_ntu_counterflow, solve_counterflow

T_HOT_IN, T_COLD_IN = 1000.0, 800.0


def _solve_constant(n, m_hot, m_cold, cp_hot, cp_cold, ua_total):
    sol = solve_counterflow(
        n,
        m_hot,
        m_cold,
        lambda T: cp_hot * T,
        lambda T: cp_hot + 0.0 * T,
        lambda T: cp_cold * T,
        lambda T: cp_cold + 0.0 * T,
        T_HOT_IN,
        T_COLD_IN,
        lambda Th, Tc: np.full(n, ua_total / n),
    )
    c_hot, c_cold = m_hot * cp_hot, m_cold * cp_cold
    q = c_hot * (T_HOT_IN - sol.T_hot[-1])
    eff = q / (min(c_hot, c_cold) * (T_HOT_IN - T_COLD_IN))
    exact = effectiveness_ntu_counterflow(
        ua_total / min(c_hot, c_cold), min(c_hot, c_cold) / max(c_hot, c_cold)
    )
    return sol, eff, exact, q


@pytest.mark.parametrize(
    ("m_hot", "m_cold"), [(1.0, 0.6), (0.6, 1.0), (1.0, 0.25)], ids=["hot-max", "hot-min", "Cr0.25"]
)
def test_matches_effectiveness_ntu_with_second_order_convergence(m_hot, m_cold):
    errors = []
    for n in (25, 50, 100, 200):
        _, eff, exact, _ = _solve_constant(n, m_hot, m_cold, 1000.0, 1000.0, 3000.0)
        errors.append(abs(eff - exact))
    order = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    # Measured orders approach 2 from below (Cr = 0.25: 1.968, 1.992, 1.998).
    np.testing.assert_allclose(order, 2.0, atol=0.05)
    assert order[-1] == pytest.approx(2.0, abs=0.005)
    assert errors[-1] < 2e-6


def test_balanced_exchanger_is_exact():
    # Cr = 1: linear temperature profiles, which the trapezoidal rule integrates exactly.
    _, eff, exact, _ = _solve_constant(20, 1.0, 1.0, 1000.0, 1000.0, 5000.0)
    assert exact == pytest.approx(5.0 / 6.0)
    assert eff == pytest.approx(exact, abs=1e-13)


def test_energy_balance_and_cell_duties_close():
    sol, _, _, q = _solve_constant(100, 1.0, 0.6, 1100.0, 1250.0, 3000.0)
    q_cold = 0.6 * 1250.0 * (sol.T_cold[0] - T_COLD_IN)
    assert q_cold == pytest.approx(q, rel=1e-12)
    assert np.sum(sol.Q_cells) == pytest.approx(q, rel=1e-12)
    # Second law: the hot stream is everywhere hotter than the cold stream.
    assert np.all(sol.T_hot > sol.T_cold)


def test_zero_ntu_and_infinite_ntu_limits():
    assert effectiveness_ntu_counterflow(0.0, 0.5) == 0.0
    assert effectiveness_ntu_counterflow(200.0, 0.5) == pytest.approx(1.0)
    assert effectiveness_ntu_counterflow(200.0, 1.0) == pytest.approx(200.0 / 201.0)


def test_rejects_wrong_inlet_order():
    with pytest.raises(ValueError, match="hotter"):
        solve_counterflow(
            10, 1.0, 1.0, np.abs, np.abs, np.abs, np.abs, 500.0, 600.0, lambda a, b: np.ones(10)
        )
