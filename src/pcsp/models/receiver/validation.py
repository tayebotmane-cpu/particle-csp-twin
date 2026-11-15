"""Comparison of the steady curtain model with Sandia on-sun receiver tests.

Data
----
The intended validation anchor, the 26 pseudo-steady cases of Mills,
Albrecht, González-Portillo & Ho, *Model validation of falling particle
receivers with on-sun experiments*, AIP Conf. Proc. 2815, 030012 (2023),
doi:10.1063/5.0149777, is only available as full text from OSTI
(https://www.osti.gov/servlets/purl/1890598), which refused connections from
the development machine; the publisher page shows only the abstract. Its
table was therefore **not** transcribed and no number from it is used.

The data used instead is TABLE 2 of Ho et al., *Receiver Design and On-Sun
Testing for G3P3-USA*, SolarPACES 2020 (47 tests on the same 1 MWt NSTTF
receiver, Aug-Sep 2020), transcribed in
``data/sandia_g3p3_2020_onsun_tests.csv`` with its full citation. This is a
weaker check than intended, for three reasons that the report must state:

1. All tests had 1 or 2 catch-and-release troughs ("StAIR") and a
   reduced-volume cavity partition. The free-fall model cannot represent the
   troughs (which raise opacity and residence time) or the partition (which
   cut advective losses by ~40 % in ground tests, same paper).
2. ``Q_in`` is not tabulated. It is recovered from the paper's own Eq. 1 and
   Eq. 2, ``Q_in = m_dot * dh_Sandia(T_in, T_out) / eta``, with Sandia's cp fit
   ``cp = 148.2 T^0.3093``. This reproduces the measured ``Q_in`` up to the
   rounding of the table (eta to 1 %, temperatures to 1 degC).
3. Wind (0.9-7.8 m/s in the table) is not modelled.

Rows with ``T_out <= T_in`` (8 rows on 2020-09-24) contradict their own
efficiency in the source and are excluded.

Calibration
-----------
One parameter, the advective coefficient ``h_adv``, is fitted by least
squares on the outlet temperature (optionally the efficiency) of a
calibration subset (every other usable case in table order) and scored on
the held-out remainder; see :func:`validation_report`.

Efficiency comparisons carry a known systematic offset: the model's eta uses
the package cp correlation (``ParticleMaterial.cp``), the experimental eta
Sandia's, and the two differ by several per cent over 400-800 degC.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from pcsp.constants import T_ZERO_C
from pcsp.models.receiver.curtain import FPRParams, simulate

DATA_FILE = Path(__file__).parent / "data" / "sandia_g3p3_2020_onsun_tests.csv"

SANDIA_CP_A = 148.2
SANDIA_CP_B = 0.3093
"""Sandia DSC fit for CARBO HSP 40/70, ``cp = 148.2 T[K]^0.3093`` J/kg/K (Ho et al. 2020, Eq. 2)."""


def sandia_enthalpy_rise(T_in: float, T_out: float) -> float:
    """``integral(cp dT)`` [J kg^-1] with Sandia's cp fit, as in Ho et al. (2020) Eq. 1."""
    n = 1.0 + SANDIA_CP_B
    return SANDIA_CP_A / n * (T_out**n - T_in**n)


@dataclass(frozen=True)
class OnSunCase:
    """One on-sun test, in SI units (temperatures in K, powers in W)."""

    date: str
    Q_in: float
    mass_flow: float
    T_in: float
    T_out: float
    efficiency: float
    T_amb: float
    wind_speed: float
    n_stairs: int
    peak_irradiance: float


def load_cases(path: Path = DATA_FILE) -> list[OnSunCase]:
    """Read the usable tests (``T_out > T_in``) and derive ``Q_in`` from Eq. 1."""
    table = pd.read_csv(path, comment="#")
    cases = []
    for row in table.itertuples(index=False):
        if row.T_out_C <= row.T_in_C:
            continue
        T_in, T_out = row.T_in_C + T_ZERO_C, row.T_out_C + T_ZERO_C
        eta = row.efficiency_pct / 100.0
        cases.append(
            OnSunCase(
                date=str(row.date),
                Q_in=row.mass_flow_kg_s * sandia_enthalpy_rise(T_in, T_out) / eta,
                mass_flow=float(row.mass_flow_kg_s),
                T_in=T_in,
                T_out=T_out,
                efficiency=eta,
                T_amb=row.T_amb_C + T_ZERO_C,
                wind_speed=float(row.wind_speed_m_s),
                n_stairs=int(row.n_stairs),
                peak_irradiance=row.peak_irradiance_kW_m2 * 1e3,
            )
        )
    return cases


def predict(case: OnSunCase, params: FPRParams) -> tuple[float, float]:
    """Model outlet temperature [K] and efficiency [-] at the case's operating point."""
    p = dataclasses.replace(
        params, mass_flow=case.mass_flow, T_in=case.T_in, Q_in=case.Q_in, T_amb=case.T_amb
    )
    ds = simulate(p)
    return float(ds.T_out), float(ds.efficiency)


def error_metrics(cases: list[OnSunCase], params: FPRParams) -> dict[str, float]:
    """Bias (model - experiment) and RMSE of outlet temperature [K] and efficiency [-]."""
    predictions = np.array([predict(c, params) for c in cases])
    dT = predictions[:, 0] - np.array([c.T_out for c in cases])
    deta = predictions[:, 1] - np.array([c.efficiency for c in cases])
    return {
        "n_cases": len(cases),
        "T_out_bias": float(np.mean(dT)),
        "T_out_rmse": float(np.sqrt(np.mean(dT**2))),
        "eta_bias": float(np.mean(deta)),
        "eta_rmse": float(np.sqrt(np.mean(deta**2))),
    }


def split_cases(cases: list[OnSunCase]) -> tuple[list[OnSunCase], list[OnSunCase]]:
    """Calibration set (even positions in table order) and held-out set (odd positions)."""
    return cases[0::2], cases[1::2]


def calibrate_h_adv(
    cases: list[OnSunCase],
    params: FPRParams,
    objective: str = "T_out",
    bounds: tuple[float, float] = (0.0, 150.0),
) -> float:
    """``h_adv`` [W m^-2 K^-1] minimising a sum of squared errors on ``cases``.

    ``objective="T_out"`` (default) fits the outlet temperature, which the
    thermocouples measure directly. ``objective="efficiency"`` fits eta; in
    this data set eta is ill-conditioned for low-power tests with a
    temperature rise of a few kelvin, which then dominate the fit.
    """
    column = {"T_out": 0, "efficiency": 1}[objective]

    def sse(h_adv: float) -> float:
        p = dataclasses.replace(params, h_adv=h_adv)
        return sum((predict(c, p)[column] - (c.T_out, c.efficiency)[column]) ** 2 for c in cases)

    result = minimize_scalar(sse, bounds=bounds, method="bounded", options={"xatol": 1e-3})
    return float(result.x)


def validation_report(
    params: FPRParams | None = None, objective: str = "T_out"
) -> dict[str, object]:
    """Uncalibrated and calibrated error metrics on calibration and held-out cases."""
    base = params if params is not None else FPRParams()
    cases = load_cases()
    calib, held_out = split_cases(cases)
    h_fit = calibrate_h_adv(calib, base, objective)
    fitted = dataclasses.replace(base, h_adv=h_fit)
    return {
        "objective": objective,
        "h_adv_nominal": base.h_adv,
        "h_adv_calibrated": h_fit,
        "uncalibrated_all": error_metrics(cases, base),
        "uncalibrated_calibration_set": error_metrics(calib, base),
        "uncalibrated_held_out": error_metrics(held_out, base),
        "calibrated_calibration_set": error_metrics(calib, fitted),
        "calibrated_held_out": error_metrics(held_out, fitted),
    }
