"""Non-intrusive reduced-order modelling (NumPy/SciPy only).

Snapshot convention throughout: a snapshot matrix has shape ``(n_dof, n_snapshots)``;
each column is one full field (or, for :class:`PODInterpolationROM`, one full
space-time trajectory).

* :mod:`pcsp.rom.pod` - Proper Orthogonal Decomposition (weighted, centred, truncated).
* :mod:`pcsp.rom.interp` - POD + RBF interpolation parametric surrogate.
* :mod:`pcsp.rom.dmd` - exact DMD and DMD with control.
* :mod:`pcsp.rom.opinf` - operator inference for polynomial reduced models.
* :mod:`pcsp.rom.metrics` - relative errors, projection error, timing.
* :mod:`pcsp.rom.snapshots` - ``xarray.DataArray`` <-> snapshot matrix.
"""

from pcsp.rom.dmd import DMD, DMDc, fit_dmd, fit_dmdc
from pcsp.rom.interp import PODInterpolationROM
from pcsp.rom.metrics import Timing, projection_error, relative_error, time_full_vs_rom
from pcsp.rom.opinf import (
    OpInfModel,
    compact_kron,
    compress_quadratic,
    expand_quadratic,
    fit_opinf,
    time_derivative,
)
from pcsp.rom.pod import POD, fit_pod, rank_for_energy
from pcsp.rom.snapshots import SnapshotLayout, from_snapshots, to_snapshots

__all__ = [
    "DMD",
    "POD",
    "DMDc",
    "OpInfModel",
    "PODInterpolationROM",
    "SnapshotLayout",
    "Timing",
    "compact_kron",
    "compress_quadratic",
    "expand_quadratic",
    "fit_dmd",
    "fit_dmdc",
    "fit_opinf",
    "fit_pod",
    "from_snapshots",
    "projection_error",
    "rank_for_energy",
    "relative_error",
    "time_derivative",
    "time_full_vs_rom",
    "to_snapshots",
]
