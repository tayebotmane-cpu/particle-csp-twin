"""Moving packed-bed particle-to-sCO2 heat exchanger (shell-and-plate), steady reduced model.

Particle side: slug-flow Graetz solution (:mod:`.graetz`) with pluggable
closures for bed conductivity and near-wall resistance (:mod:`.closures`).
sCO2 side: Gnielinski correlation with real-gas properties (:mod:`.channel`,
:mod:`pcsp.materials.sco2`). The 1-D counterflow enthalpy balances are solved
in :mod:`.counterflow`; :func:`simulate` returns a schema-conforming dataset.
"""

from pcsp.models.hx import albrecht_ho, graetz
from pcsp.models.hx.closures import AdapaFlowingBed, BedClosure, ConstantBed, StagnantBed
from pcsp.models.hx.counterflow import effectiveness_ntu_counterflow, solve_counterflow
from pcsp.models.hx.model import MovingBedHXParams, local_coefficients, simulate, solve

__all__ = [
    "AdapaFlowingBed",
    "BedClosure",
    "ConstantBed",
    "MovingBedHXParams",
    "StagnantBed",
    "albrecht_ho",
    "effectiveness_ntu_counterflow",
    "graetz",
    "local_coefficients",
    "simulate",
    "solve",
    "solve_counterflow",
]
