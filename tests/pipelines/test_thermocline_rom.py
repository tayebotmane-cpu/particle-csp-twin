"""Whole thermocline-surrogate pipeline on a tiny, coarse campaign.

The assertions are properties that must hold whatever the tuning: no leakage
between training and test runs, a genuine extrapolation split, ROM error never
below the projection error of its basis, exact reproduction of training runs,
more training runs helping, and resumable campaigns.
"""

import json

import numpy as np
import pytest

from pcsp.data import read_manifest
from pcsp.pipelines.thermocline_rom import (
    VARIANTS,
    StudyConfig,
    build_campaign,
    evaluate,
    fit_surrogate,
    format_results_table,
    run_study,
    time_grids,
)

TINY = StudyConfig(
    n_runs=14,
    n_cells=30,
    n_time=31,
    rtol=1e-4,
    ranks=(1, 2, 4, 8),
    train_sizes=(5,),
    n_timing_runs=1,
)


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return tmp_path_factory.mktemp("thermocline_campaign")


@pytest.fixture(scope="module")
def study(root):
    return run_study(TINY, root)


def test_campaign_ran_completely(study):
    assert study.n_ok == TINY.n_runs and study.n_failed == 0
    assert study.n_executed == TINY.n_runs
    assert np.all(study.data.wall_seconds > 0.0)
    assert "interpolation" in format_results_table(study)


def test_every_run_covers_the_study_window_and_the_tau_grid(study):
    grids = time_grids(TINY)
    np.testing.assert_array_equal(grids.t, grids.t_sim[: TINY.n_time])
    assert grids.t[-1] == pytest.approx(TINY.t_end)
    reached_tau = grids.t_sim[-1] * study.data.speed / TINY.length
    assert np.all(reached_tau >= grids.tau[-1])


@pytest.mark.parametrize("split_name", ["interpolation", "extrapolation"])
def test_no_test_run_is_in_the_training_set(study, split_name):
    split = study.splits[split_name]
    train_hashes = set(study.data.param_hash[split.train])
    test_hashes = set(study.data.param_hash[split.test])
    assert train_hashes.isdisjoint(test_hashes)
    assert len(train_hashes) + len(test_hashes) == TINY.n_runs
    for variant in VARIANTS:
        result = study.results[(split_name, variant)]
        assert set(result.evaluation.param_hash) == test_hashes
        assert result.n_train == len(train_hashes)


def test_extrapolation_conditions_lie_outside_the_training_range(study):
    split = study.splits["extrapolation"]
    mass_flow = study.data.params[:, 0]
    assert mass_flow[split.test].min() > mass_flow[split.train].max()
    for variant in VARIANTS:
        assert np.all(study.results[("extrapolation", variant)].evaluation.distance_outside > 0.0)


def test_rom_error_is_never_below_projection_error(study):
    for result in study.results.values():
        e = result.evaluation
        assert np.all(e.native_field_error >= e.native_projection_error * (1.0 - 1e-9))
    for sweep in study.rank_sweeps.values():
        # Medians of per-run errors keep the ordering only approximately; the
        # physical-frame floor of B also contains the tau resampling error.
        assert np.all(sweep.field_error >= 0.5 * sweep.projection_error)


@pytest.mark.parametrize("variant", VARIANTS)
def test_training_runs_are_reproduced_at_projection_error(study, variant):
    train = study.data.subset(study.splits["interpolation"].train)
    # A truncated basis, so that the projection error is well above round-off.
    surrogate = fit_surrogate(train, variant, rank=4)
    e = evaluate(surrogate, train)
    assert np.all(e.native_projection_error > 1e-6)
    np.testing.assert_allclose(
        e.native_field_error, e.native_projection_error, rtol=1e-6, atol=1e-12
    )
    np.testing.assert_allclose(e.field_error, e.projection_error, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("variant", VARIANTS)
def test_interpolation_error_falls_when_training_runs_are_added(study, variant):
    sweep = study.size_sweeps[variant]
    assert sweep.x[0] < sweep.x[-1]
    assert sweep.field_error[-1] < sweep.field_error[0]


def test_campaign_resumes_without_rerunning_finished_simulations(study, root):
    before = {r.param_hash: r.finished_utc for r in read_manifest(root)}
    ds, again = build_campaign(TINY, root)
    assert again.executed == [] and len(again.skipped) == TINY.n_runs
    assert {r.param_hash: r.finished_utc for r in read_manifest(root)} == before
    assert ds.sizes["run"] == TINY.n_runs
    assert json.loads(ds.attrs["fixed_params_json"])["n_cells"] == TINY.n_cells
