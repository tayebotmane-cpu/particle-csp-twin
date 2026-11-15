"""Sample -> run campaign (one failure) -> resume -> load stacked -> hold-out split."""

import numpy as np

from pcsp.data import (
    Threshold,
    apply_split,
    check_split,
    holdout_split,
    latin_hypercube,
    load_campaign,
    param_hash,
    read_manifest,
    run_campaign,
)
from pcsp.data import provenance as prov


def test_campaign_pipeline(tmp_path, model, space, monkeypatch):
    monkeypatch.setenv(prov.CONTAINER_DIGEST_ENV, "sha256:feedbeef")
    samples = latin_hypercube(space, 12, seed=42)
    bad = {**samples[0], "amplitude": -1.0}  # deliberately failing operating condition
    samples.append(bad)
    root = tmp_path / "campaign"

    first = run_campaign(model, samples, root)
    assert len(first.ok) == 12 and [r.param_hash for r in first.failed] == [param_hash(bad)]

    calls = []

    def counting_model(params):
        calls.append(param_hash(params))
        return model(params)

    resumed = run_campaign(counting_model, samples, root)
    assert calls == [param_hash(bad)]  # finished work is not repeated
    assert len(resumed.skipped) == 12 and len(read_manifest(root)) == 13

    ds = load_campaign(root)
    assert ds.sizes["run"] == 12
    assert ds["u"].dims == ("run", "time", "x") and ds["peak"].dims == ("run",)
    ok_samples = samples[:-1]
    assert ds["param_hash"].values.tolist() == [param_hash(s) for s in ok_samples]
    np.testing.assert_array_equal(ds["speed"].values, [s["speed"] for s in ok_samples])
    np.testing.assert_array_equal(ds["amplitude"].values, [s["amplitude"] for s in ok_samples])
    assert "model_tag" not in ds.coords  # fixed parameter, not a coordinate
    expected = prov.collect_provenance()
    for key in ("git_sha", "git_dirty", "pcsp_version", "container_digest"):
        assert ds.attrs[key] == expected[key]
    assert ds.attrs["container_digest"] == "sha256:feedbeef"
    assert np.all(ds["wall_seconds"].values > 0)

    # Hold out the fastest quarter of the operating conditions.
    threshold = float(np.quantile(ds["speed"].values, 0.75))
    split = holdout_split(ds, Threshold("speed", threshold), val_fraction=0.2, seed=0)
    check_split(split, ds.sizes["run"])
    parts = apply_split(ds, split)
    assert sum(p.sizes["run"] for p in parts.values()) == 12
    assert parts["test"].sizes["run"] == 3 and parts["val"].sizes["run"] == 2
    # The test conditions are an extrapolation: outside the range seen in training.
    assert parts["test"]["speed"].min() > parts["train"]["speed"].max()
    assert parts["test"]["speed"].min() > parts["val"]["speed"].max()
    assert set(parts["test"]["param_hash"].values).isdisjoint(parts["train"]["param_hash"].values)
