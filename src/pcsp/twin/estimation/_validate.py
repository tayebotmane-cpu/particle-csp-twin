"""Argument checks shared by the filters."""

from __future__ import annotations

import numpy as np


def resolve_ud(
    u: np.ndarray | None,
    d: np.ndarray | None,
    last: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Inputs for the output map: explicit arguments win, else the last predict's."""
    if u is not None and d is not None:
        return np.asarray(u, dtype=float), np.asarray(d, dtype=float)
    if last is None:
        raise RuntimeError("update() before any predict(): pass u and d explicitly")
    return (
        last[0] if u is None else np.asarray(u, dtype=float),
        last[1] if d is None else np.asarray(d, dtype=float),
    )


def check_vec(v: np.ndarray, n: int, name: str) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    if v.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {v.shape}")
    return v.copy()


def check_cov(M: np.ndarray, n: int, name: str) -> np.ndarray:
    M = np.atleast_2d(np.asarray(M, dtype=float))
    if M.shape != (n, n):
        raise ValueError(f"{name} must have shape ({n}, {n}), got {M.shape}")
    if not np.allclose(M, M.T, rtol=1e-10, atol=0.0):
        raise ValueError(f"{name} must be symmetric")
    return M.copy()
