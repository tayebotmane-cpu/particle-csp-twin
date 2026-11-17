"""Moving packed-bed particle-to-sCO2 heat exchanger (shell-and-plate).

Particle side: slug-flow Graetz solution (:mod:`.graetz`) with pluggable
closures for bed conductivity and near-wall resistance (:mod:`.closures`).
sCO2 side: Gnielinski correlation with real-gas properties (:mod:`.channel`,
:mod:`pcsp.materials.sco2`). The 1-D counterflow enthalpy balances are solved
in :mod:`.counterflow`; :func:`simulate` returns a schema-conforming dataset.
:mod:`.conjugate` solves the same physics in two dimensions, without the
Nusselt-number approximation, as a reference for the reduced model.
:mod:`.dynamic` adds time (bed and plate heat capacity) for response and
control studies.
"""

from pcsp.models.hx import albrecht_ho, conjugate, graetz
from pcsp.models.hx.closures import (
    AdapaFlowingBed,
    BauxiteFitBed,
    BedClosure,
    ConstantBed,
    StagnantBed,
    YagiKuniiBed,
)
from pcsp.models.hx.counterflow import effectiveness_ntu_counterflow, solve_counterflow
from pcsp.models.hx.dynamic import MovingBedHXDynamic
from pcsp.models.hx.model import MovingBedHXParams, local_coefficients, simulate, solve

__all__ = [
    "AdapaFlowingBed",
    "BauxiteFitBed",
    "BedClosure",
    "ConstantBed",
    "MovingBedHXDynamic",
    "MovingBedHXParams",
    "StagnantBed",
    "YagiKuniiBed",
    "albrecht_ho",
    "conjugate",
    "effectiveness_ntu_counterflow",
    "graetz",
    "local_coefficients",
    "simulate",
    "solve",
    "solve_counterflow",
]
