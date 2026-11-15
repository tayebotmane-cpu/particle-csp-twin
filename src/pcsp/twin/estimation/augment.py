"""Disturbance and output-bias augmentation for offset-free estimation and control.

Given a model ``ẋ = f(x, u, d)``, ``y = h(x, u, d)``, split the disturbance
vector into channels ``d_e`` to be *estimated* and channels ``d_m`` that stay
*measured* (passed in as before), and optionally add constant output biases
``b`` to selected outputs. The augmented model has state ``z = [x; d_e; b]``:

.. math::

    \\dot x = f(x, u, d(d_e, d_m)), \\qquad \\dot d_e = 0, \\qquad \\dot b = 0, \\\\
    y = h(x, u, d(d_e, d_m)) + S b

where ``d(d_e, d_m)`` re-assembles the disturbance vector in the original
model's order and ``S`` places each bias on its output. The zero dynamics plus
the process noise that the filter assigns to ``d_e`` and ``b`` make them
random walks, ``d_e,k+1 = d_e,k + w_k``. A filter run on the augmented model
therefore tracks slowly varying unknown inputs, and an MPC fed with the
estimated ``d_e`` and ``b`` removes steady-state offset provided the augmented
system is observable (check with :func:`pcsp.twin.estimation.observability`) and
the number of augmented states does not exceed the number of measurements
(Muske & Badgwell, 2002; Pannocchia & Rawlings, 2003).

References
----------
* K. R. Muske, T. A. Badgwell (2002), "Disturbance modeling for offset-free
  linear model predictive control", *J. Process Control* 12, 617-632.
* G. Pannocchia, J. B. Rawlings (2003), "Disturbance models for offset-free
  model-predictive control", *AIChE J.* 49(2), 426-437.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from pcsp.core.backend import ops_for
from pcsp.core.interfaces import ControlModel, ControlModelBase


class AugmentedModel(ControlModelBase):
    """A :class:`ControlModel` whose state is ``z = [x; d_e; b]`` (see module docstring).

    Backend-agnostic like the wrapped model: NumPy in, NumPy out; CasADi in,
    CasADi out.

    Parameters
    ----------
    model : the physical model
    disturbances : names of the disturbance channels to estimate as states
    output_biases : names of the outputs that receive an additive bias state

    Layouts
    -------
    * ``state_names``: the model's states, then ``"d:<name>"`` for each
      estimated disturbance, then ``"bias:<name>"`` for each biased output.
    * ``disturbance_names``: the model's disturbances that are *not* estimated,
      in their original order (still supplied by the caller as ``d``).
    * ``input_names`` and ``output_names``: unchanged.
    """

    def __init__(
        self,
        model: ControlModel,
        disturbances: Sequence[str] = (),
        output_biases: Sequence[str] = (),
    ) -> None:
        _check_names(disturbances, model.disturbance_names, "disturbance")
        _check_names(output_biases, model.output_names, "output")
        self.model = model
        self.estimated_disturbances = tuple(disturbances)
        self.biased_outputs = tuple(output_biases)

        nx, ne = model.nx, len(self.estimated_disturbances)
        self.state_names = (
            tuple(model.state_names)
            + tuple(f"d:{n}" for n in self.estimated_disturbances)
            + tuple(f"bias:{n}" for n in self.biased_outputs)
        )
        self.input_names = tuple(model.input_names)
        self.disturbance_names = tuple(
            n for n in model.disturbance_names if n not in self.estimated_disturbances
        )
        self.output_names = tuple(model.output_names)

        #: Indices into the augmented state ``z``.
        self.physical_state_indices = tuple(range(nx))
        self.disturbance_state_indices = tuple(range(nx, nx + ne))
        self.bias_state_indices = tuple(range(nx + ne, self.nx))

        # For each original disturbance channel: ("z", index into z) or
        # ("d", index into the reduced measured-disturbance vector).
        self._d_source: list[tuple[str, int]] = []
        for name in model.disturbance_names:
            if name in self.estimated_disturbances:
                self._d_source.append(("z", nx + self.estimated_disturbances.index(name)))
            else:
                self._d_source.append(("d", self.disturbance_names.index(name)))
        # For each output: index into z of its bias state, or None.
        self._bias_of_output = [
            nx + ne + self.biased_outputs.index(n) if n in self.biased_outputs else None
            for n in model.output_names
        ]

    # ------------------------------------------------------- ControlModel API
    def rhs(self, z: Any, u: Any, d: Any) -> Any:
        xp = ops_for(z, u, d)
        x_dot = self.model.rhs(z[: self.model.nx], u, self._full_disturbance(z, d))
        n_aug = self.nx - self.model.nx
        if n_aug == 0:
            return x_dot
        return xp.concat([x_dot, xp.const(np.zeros(n_aug))])

    def output(self, z: Any, u: Any, d: Any) -> Any:
        xp = ops_for(z, u, d)
        y = self.model.output(z[: self.model.nx], u, self._full_disturbance(z, d))
        if not self.biased_outputs:
            return y
        return xp.concat(
            [y[i] if j is None else y[i] + z[j] for i, j in enumerate(self._bias_of_output)]
        )

    def _full_disturbance(self, z: Any, d: Any) -> Any:
        """Disturbance vector in the wrapped model's layout."""
        if self.model.nd == 0:
            return d
        xp = ops_for(z, d)
        return xp.concat([z[i] if src == "z" else d[i] for src, i in self._d_source])

    # ------------------------------------------------------ numerical helpers
    def split(
        self, z: np.ndarray, d_measured: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split an augmented estimate into ``(x_hat, d_hat)`` for the original model.

        ``d_hat`` has the original model's disturbance layout: estimated
        channels come from ``z``, the others from ``d_measured`` (layout
        :attr:`disturbance_names`; may be omitted if every channel is estimated).
        Output biases are not part of ``d_hat``; use :meth:`output_bias`.
        """
        z = self._check_z(z)
        if d_measured is None:
            if self.disturbance_names:
                raise ValueError(f"d_measured is required for {self.disturbance_names}")
            d_measured = np.zeros(0)
        d_measured = np.asarray(d_measured, dtype=float).ravel()
        if d_measured.size != self.nd:
            raise ValueError(f"d_measured must have {self.nd} entries, got {d_measured.size}")
        return z[: self.model.nx].copy(), np.asarray(
            self._full_disturbance(z, d_measured), dtype=float
        ).reshape(self.model.nd)

    def output_bias(self, z: np.ndarray) -> np.ndarray:
        """Full output-bias vector, shape ``(ny,)``; zero for unbiased outputs."""
        z = self._check_z(z)
        return np.array([0.0 if j is None else z[j] for j in self._bias_of_output])

    def join(self, x: np.ndarray, d: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
        """Build ``z`` from a physical state, a full disturbance vector and biases.

        The inverse of :meth:`split`. ``d`` has the original model's layout;
        ``bias`` the full output layout (``None`` means zero bias).
        """
        x = np.asarray(x, dtype=float).ravel()
        d = np.asarray(d, dtype=float).ravel()
        if x.size != self.model.nx or d.size != self.model.nd:
            raise ValueError("x and d must have the original model's sizes")
        bias = np.zeros(self.ny) if bias is None else np.asarray(bias, dtype=float).ravel()
        idx = self.model.disturbance_names.index
        out_idx = self.model.output_names.index
        return np.concatenate(
            [
                x,
                [d[idx(n)] for n in self.estimated_disturbances],
                [bias[out_idx(n)] for n in self.biased_outputs],
            ]
        )

    def measured_disturbances(self, d: np.ndarray) -> np.ndarray:
        """Select, from a full disturbance vector, the channels this model still takes as ``d``."""
        d = np.asarray(d, dtype=float).ravel()
        idx = self.model.disturbance_names.index
        return np.array([d[idx(n)] for n in self.disturbance_names], dtype=float)

    def _check_z(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float).ravel()
        if z.size != self.nx:
            raise ValueError(f"augmented state must have {self.nx} entries, got {z.size}")
        return z


def _check_names(chosen: Sequence[str], available: Sequence[str], kind: str) -> None:
    unknown = [n for n in chosen if n not in available]
    if unknown:
        raise ValueError(f"unknown {kind} name(s) {unknown}; available: {tuple(available)}")
    if len(set(chosen)) != len(chosen):
        raise ValueError(f"duplicate {kind} names in {tuple(chosen)}")
