"""Validation data handling and the h_adv calibration.

These tests check that the published data are read faithfully and that the
calibration does what it claims. The model-vs-experiment numbers themselves
are *reported* (``validation_report``), not asserted against a tolerance.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pcsp.models.receiver.curtain import FPRParams
from pcsp.models.receiver.validation import (
    DATA_FILE,
    OnSunCase,
    calibrate_h_adv,
    error_metrics,
    load_cases,
    predict,
    sandia_enthalpy_rise,
    split_cases,
    validation_report,
)


def test_csv_carries_citation_and_full_table():
    header = [line for line in DATA_FILE.read_text().splitlines() if line.startswith("#")]
    assert any("Receiver Design and On-Sun Testing for G3P3-USA" in line for line in header)
    table = pd.read_csv(DATA_FILE, comment="#")
    assert len(table) == 47
    # Spot checks against the printed table (first, a middle and the last row).
    assert table.iloc[0].tolist()[1:] == [680, 2.38, 545, 629, 42, 31, 3.2, 251, 2]
    assert table.iloc[29].tolist()[1:] == [680, 7.25, 435, 502, 86, 27, 4.3, 292, 1]
    assert table.iloc[-1].tolist()[1:] == [620, 7.39, 479, 534, 83, 28, 3.4, 320, 1]


def test_only_self_contradictory_rows_are_excluded():
    table = pd.read_csv(DATA_FILE, comment="#")
    excluded = table[table.T_out_C <= table.T_in_C]
    assert len(excluded) == 8 and set(excluded.date) == {"2020-09-24"}
    assert len(load_cases()) == 47 - 8


def test_derived_incident_power_reproduces_reported_efficiency():
    for case in load_cases():
        eta = case.mass_flow * sandia_enthalpy_rise(case.T_in, case.T_out) / case.Q_in
        assert eta == pytest.approx(case.efficiency, rel=1e-12)
        assert 5e4 < case.Q_in < 1.2e6  # a 1 MWt receiver


def test_sandia_enthalpy_matches_their_integrated_form():
    # Ho et al. (2020) Eq. 1 prints the integral as 113.2 (T_out^1.3093 - T_in^1.3093).
    T_in, T_out = 857.15, 965.15
    printed = 113.2 * (T_out**1.3093 - T_in**1.3093)
    assert sandia_enthalpy_rise(T_in, T_out) == pytest.approx(printed, rel=1e-4)


def test_calibration_recovers_a_known_coefficient_from_synthetic_data():
    truth = dataclasses.replace(FPRParams(n_segments=18), h_adv=25.0)
    synthetic = []
    for c in load_cases()[::5]:
        T_out, eta = predict(c, truth)
        synthetic.append(dataclasses.replace(c, T_out=T_out, efficiency=eta))
    start = FPRParams(n_segments=18)
    assert calibrate_h_adv(synthetic, start, "T_out") == pytest.approx(25.0, abs=0.01)
    assert calibrate_h_adv(synthetic, start, "efficiency") == pytest.approx(25.0, abs=0.01)


def test_split_is_disjoint_and_complete():
    cases = load_cases()
    calib, held_out = split_cases(cases)
    assert len(calib) + len(held_out) == len(cases)
    assert not {id(c) for c in calib} & {id(c) for c in held_out}


def test_error_metrics_on_a_perfect_prediction_are_zero():
    p = FPRParams(n_segments=18)
    perfect: list[OnSunCase] = []
    for c in load_cases()[:4]:
        T_out, eta = predict(c, p)
        perfect.append(dataclasses.replace(c, T_out=T_out, efficiency=eta))
    m = error_metrics(perfect, p)
    assert m["T_out_rmse"] == 0.0 and m["eta_rmse"] == 0.0


def test_validation_report_calibration_does_not_worsen_its_own_fit():
    report = validation_report()
    calibrated = report["calibrated_calibration_set"]["T_out_rmse"]
    uncalibrated = report["uncalibrated_calibration_set"]["T_out_rmse"]
    assert calibrated <= uncalibrated + 1e-9
    assert 0.0 < report["h_adv_calibrated"] < 150.0
    for key in ("uncalibrated_all", "uncalibrated_held_out", "calibrated_held_out"):
        assert all(np.isfinite(v) for v in report[key].values())
