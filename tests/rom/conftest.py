"""Analytic benchmark fields shared by the ROM tests."""

from collections.abc import Callable

import numpy as np
import pytest


def _heat_series(x: np.ndarray, t: np.ndarray, alpha: float, derivative: bool) -> np.ndarray:
    """Fourier series of u_t = alpha u_xx on [0, 1], u(0) = u(1) = 0, u(x, 0) = 1.

    u(x, t) = sum_{n odd} 4 / (n pi) sin(n pi x) exp(-alpha (n pi)^2 t). 4000 odd
    terms: for t >= 2e-3 the truncation error is below exp(-alpha pi^2 8000^2 t),
    i.e. negligible. Returns the snapshot matrix (n_x, n_t) of u or of du/dt.
    """
    n = np.arange(1, 8000, 2)
    k2 = (n * np.pi) ** 2
    b = 4.0 / (n * np.pi)
    if derivative:
        b = -alpha * k2 * b
    return (np.sin(np.pi * np.outer(x, n)) * b) @ np.exp(-alpha * k2[:, None] * t[None, :])


@pytest.fixture
def heat_solution() -> Callable[..., np.ndarray]:
    """``heat_solution(x, t, alpha=1.0, derivative=False)`` -> snapshots ``(n_x, n_t)``."""

    def solution(
        x: np.ndarray, t: np.ndarray, alpha: float = 1.0, derivative: bool = False
    ) -> np.ndarray:
        return _heat_series(np.asarray(x), np.asarray(t), alpha, derivative)

    return solution
