import json
import shutil

import numpy as np
import pytest
import xarray as xr

from pcsp.data import load, load_campaign, param_hash, read_manifest, run_campaign

PROVENANCE = {"git_sha": "abc123", "git_dirty": 0, "pcsp_version": "0.1.0"}

SAMPLES = [
    {"speed": 0.1, "amplitude": 1.0},
    {"speed": 0.3, "amplitude": -1.0},  # the toy model rejects this one
    {"speed": 0.5, "amplitude": 2.0},
]


class CountingModel:
    """Wraps a model and records which parameter sets it was called with."""

    def __init__(self, model):
        self.model = model
        self.calls = []

    def __call__(self, params):
        self.calls.append(param_hash(params))
        return self.model(params)


def test_runs_are_stored_and_failures_recorded_without_stopping(tmp_path, model):
    result = run_campaign(model, SAMPLES, tmp_path, provenance=PROVENANCE)
    assert [r.status for r in result.records] == ["ok", "failed", "ok"]

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    entries = {e["param_hash"]: e for e in manifest["runs"]}
    failed = entries[param_hash(SAMPLES[1])]
    assert failed["error"] == "ValueError: amplitude must be positive, got -1.0"
    assert "Traceback" in failed["traceback"]
    assert not (tmp_path / failed["path"]).exists()

    for sample in (SAMPLES[0], SAMPLES[2]):
        entry = entries[param_hash(sample)]
        assert entry["params"] == sample and entry["wall_seconds"] > 0
        assert entry["path"] == f"runs/{param_hash(sample)}.zarr"
        stored = load(tmp_path / entry["path"])
        assert stored.attrs["git_sha"] == "abc123"
        xr.testing.assert_identical(stored.drop_attrs(), model(sample).drop_attrs())
    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == sorted(
        f"{param_hash(s)}.zarr" for s in (SAMPLES[0], SAMPLES[2])
    )


def test_resume_reruns_only_unfinished_work(tmp_path, model):
    run_campaign(model, SAMPLES, tmp_path, provenance=PROVENANCE)
    created = {
        r.param_hash: load(tmp_path / r.path).attrs["created_utc"]
        for r in read_manifest(tmp_path)
        if r.status == "ok"
    }

    counting = CountingModel(model)
    result = run_campaign(counting, SAMPLES, tmp_path, provenance=PROVENANCE)
    assert counting.calls == [param_hash(SAMPLES[1])]  # only the failed run is retried
    assert result.skipped == [param_hash(SAMPLES[0]), param_hash(SAMPLES[2])]
    for h, stamp_time in created.items():
        assert load(tmp_path / f"runs/{h}.zarr").attrs["created_utc"] == stamp_time

    counting = CountingModel(model)
    run_campaign(counting, SAMPLES, tmp_path, force=True, provenance=PROVENANCE)
    assert counting.calls == [param_hash(s) for s in SAMPLES]


def test_run_marked_ok_but_missing_on_disk_is_rerun(tmp_path, model):
    run_campaign(model, SAMPLES[:1], tmp_path, provenance=PROVENANCE)
    record = read_manifest(tmp_path)[0]
    shutil.rmtree(tmp_path / record.path)
    counting = CountingModel(model)
    run_campaign(counting, SAMPLES[:1], tmp_path, provenance=PROVENANCE)
    assert counting.calls == [record.param_hash]


def test_interrupted_campaign_keeps_finished_runs_and_resumes(tmp_path, model):
    samples = [{"speed": s, "amplitude": 1.0} for s in (0.1, 0.2, 0.3, 0.4)]

    def dies_on_third_run(params):
        if params["speed"] == 0.3:
            raise KeyboardInterrupt  # e.g. Ctrl-C or the job being killed
        return model(params)

    with pytest.raises(KeyboardInterrupt):
        run_campaign(dies_on_third_run, samples, tmp_path, provenance=PROVENANCE)
    assert [r.param_hash for r in read_manifest(tmp_path)] == [param_hash(s) for s in samples[:2]]
    assert len(list((tmp_path / "runs").iterdir())) == 2

    counting = CountingModel(model)
    run_campaign(counting, samples, tmp_path, provenance=PROVENANCE)
    assert counting.calls == [param_hash(s) for s in samples[2:]]
    assert load_campaign(tmp_path).sizes["run"] == 4


def test_invalid_model_output_is_a_failed_run_with_nothing_stored(tmp_path, model):
    def wrong_params(params):
        return model({**params, "speed": 0.9})

    def not_a_dataset(params):
        return np.zeros(3)

    result = run_campaign(wrong_params, SAMPLES[:1], tmp_path / "a", provenance=PROVENANCE)
    assert "recorded different parameter values" in result.failed[0].error
    result = run_campaign(not_a_dataset, SAMPLES[:1], tmp_path / "b", provenance=PROVENANCE)
    assert "expected xarray.Dataset" in result.failed[0].error
    assert not (tmp_path / "a" / "runs").exists() and not (tmp_path / "b" / "runs").exists()


def test_duplicate_samples_are_rejected(tmp_path, model):
    with pytest.raises(ValueError, match="duplicate"):
        run_campaign(model, [SAMPLES[0], dict(SAMPLES[0])], tmp_path)


def test_parallel_run_matches_serial(tmp_path, model):
    samples = [{"speed": s, "amplitude": 1.0} for s in (0.1, 0.2, 0.3)] + [SAMPLES[1]]
    serial = run_campaign(model, samples, tmp_path / "serial", provenance=PROVENANCE)
    parallel = run_campaign(model, samples, tmp_path / "par", max_workers=2, provenance=PROVENANCE)
    assert [r.status for r in parallel.records] == [r.status for r in serial.records]
    a, b = load_campaign(tmp_path / "serial"), load_campaign(tmp_path / "par")
    xr.testing.assert_identical(a.drop_vars("wall_seconds"), b.drop_vars("wall_seconds"))


def test_dead_worker_is_recorded_and_store_stays_loadable(tmp_path, killing_model, model):
    samples = [{"speed": 0.1, "amplitude": 1.0}, {"speed": 2.0, "amplitude": 1.0}]
    result = run_campaign(killing_model, samples, tmp_path, max_workers=2, provenance=PROVENANCE)
    killed = next(r for r in result.records if r.params["speed"] == 2.0)
    assert killed.status == "failed" and "worker process died" in killed.error
    assert not (tmp_path / killed.path).exists()
    # The killed run (and any run the broken pool abandoned) is retried on resume.
    resumed = run_campaign(model, samples[:1], tmp_path, provenance=PROVENANCE)
    assert resumed.ok and load_campaign(tmp_path).sizes["run"] == 1


def test_non_finite_parameters_are_rejected_up_front(tmp_path, model):
    with pytest.raises(ValueError, match="finite"):
        run_campaign(model, [{"speed": float("nan"), "amplitude": 1.0}], tmp_path)
    assert not (tmp_path / "manifest.json").exists()


def test_parallel_needs_a_picklable_model(tmp_path, model):
    with pytest.raises(TypeError, match="picklable"):
        run_campaign(lambda p: model(p), SAMPLES[:2], tmp_path, max_workers=2)


def test_load_campaign_stacks_runs(tmp_path, model):
    samples = [{"speed": s, "amplitude": a, "tag": "x"} for s, a in [(0.1, 1.0), (0.2, 1.5)]]
    run_campaign(model, samples + [{"speed": 0.3, "amplitude": -1.0, "tag": "x"}], tmp_path)
    ds = load_campaign(tmp_path)

    assert ds["u"].dims == ("run", "time", "x") and ds.sizes["run"] == 2
    assert ds["peak"].dims == ("run",)  # 0-d per-run scalar -> 1-D over run
    assert ds["speed"].values.tolist() == [0.1, 0.2]
    assert ds["amplitude"].values.tolist() == [1.0, 1.5]
    assert ds["param_hash"].values.tolist() == [param_hash(s) for s in samples]
    assert "tag" not in ds.coords and json.loads(ds.attrs["fixed_params_json"]) == {"tag": "x"}
    assert ds["wall_seconds"].attrs["units"] == "s" and np.all(ds["wall_seconds"] > 0)
    assert "params_json" not in ds.attrs and ds.attrs["model"] == "toy-pulse"
    for i, s in enumerate(samples):
        np.testing.assert_array_equal(ds["u"].isel(run=i).values, model(s)["u"].values)


def test_differing_provenance_becomes_a_run_coordinate(tmp_path, model):
    run_campaign(model, SAMPLES[:1], tmp_path, provenance=PROVENANCE)
    run_campaign(model, SAMPLES[2:], tmp_path, provenance={**PROVENANCE, "git_sha": "def456"})
    ds = load_campaign(tmp_path)
    assert ds["git_sha"].values.tolist() == ["abc123", "def456"]
    assert "git_sha" not in ds.attrs and ds.attrs["git_dirty"] == 0


def test_mismatched_grids_fail_with_options(tmp_path, model):
    fine, coarse = (
        {"speed": 0.1, "amplitude": 1.0, "nx": 21},
        {"speed": 0.2, "amplitude": 1.0, "nx": 11},
    )
    run_campaign(model, [fine, coarse], tmp_path, provenance=PROVENANCE)
    with pytest.raises(ValueError, match="do not share one grid") as excinfo:
        load_campaign(tmp_path)
    message = str(excinfo.value)
    assert "coordinate 'x'" in message and "interpolat" in message and "param_hashes" in message
    subset = load_campaign(tmp_path, param_hashes=[param_hash(fine)])
    assert subset.sizes["run"] == 1


def test_load_campaign_without_successful_runs_fails_clearly(tmp_path, model):
    run_campaign(model, SAMPLES[1:2], tmp_path, provenance=PROVENANCE)
    with pytest.raises(ValueError, match="no successful runs"):
        load_campaign(tmp_path)
    with pytest.raises(FileNotFoundError, match="manifest"):
        load_campaign(tmp_path / "elsewhere")
