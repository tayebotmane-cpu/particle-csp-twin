import re
import shutil
import subprocess

import pytest

import pcsp
from pcsp.data import collect_provenance, is_stamped, stamp, validate
from pcsp.data import provenance as prov

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def fake_checkout(tmp_path, monkeypatch):
    """A throw-away git repo with a committed __init__.py posing as the package."""
    _git(tmp_path, "init", "-q")
    (tmp_path / "__init__.py").write_text("")
    _git(tmp_path, "add", "__init__.py")
    _git(tmp_path, "commit", "-q", "-m", "init")
    monkeypatch.setattr(prov, "_PACKAGE_DIR", tmp_path)
    return tmp_path


def test_record_has_schema_keys_and_package_version(monkeypatch):
    monkeypatch.delenv(prov.CONTAINER_DIGEST_ENV, raising=False)
    record = collect_provenance()
    assert set(record) == {"git_sha", "git_dirty", "pcsp_version"}
    assert record["pcsp_version"] == pcsp.__version__
    assert record["git_dirty"] in (-1, 0, 1)


@needs_git
def test_sha_and_dirty_flag_follow_the_checkout(fake_checkout):
    head = subprocess.run(
        ["git", "-C", str(fake_checkout), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    record = collect_provenance()
    assert record["git_sha"] == head and re.fullmatch(r"[0-9a-f]{40}", head)
    assert record["git_dirty"] == 0
    (fake_checkout / "__init__.py").write_text("x = 1\n")
    assert collect_provenance()["git_dirty"] == 1


@needs_git
def test_untracked_files_count_as_dirty(fake_checkout):
    (fake_checkout / "new_module.py").write_text("")
    assert collect_provenance()["git_dirty"] == 1


def test_outside_a_git_checkout_is_unknown_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(prov, "_PACKAGE_DIR", tmp_path)
    record = collect_provenance()
    assert record["git_sha"] == prov.UNKNOWN
    assert record["git_dirty"] == prov.GIT_DIRTY_UNKNOWN


@needs_git
def test_package_not_tracked_by_the_enclosing_repo_is_unknown(tmp_path, monkeypatch):
    # e.g. an installed copy inside a virtualenv that sits inside some other repository
    _git(tmp_path, "init", "-q")
    (tmp_path / "__init__.py").write_text("")
    monkeypatch.setattr(prov, "_PACKAGE_DIR", tmp_path)
    assert collect_provenance()["git_sha"] == prov.UNKNOWN


def test_missing_git_executable_is_unknown_not_an_error(monkeypatch):
    monkeypatch.setenv("PATH", "")
    record = collect_provenance()
    assert record["git_sha"] == prov.UNKNOWN
    assert record["git_dirty"] == prov.GIT_DIRTY_UNKNOWN


def test_container_digest_comes_from_environment(monkeypatch):
    monkeypatch.setenv(prov.CONTAINER_DIGEST_ENV, "  sha256:abc123  ")
    assert collect_provenance()["container_digest"] == "sha256:abc123"
    monkeypatch.setenv(prov.CONTAINER_DIGEST_ENV, "")
    assert "container_digest" not in collect_provenance()


def test_hostname_is_off_by_default():
    assert "hostname" not in collect_provenance()
    assert collect_provenance(include_hostname=True)["hostname"]


def test_stamp_makes_dataset_archivable_without_mutating_input(toy_dataset):
    stamped = stamp(toy_dataset, {"git_sha": "abc", "git_dirty": 0, "pcsp_version": "0.1.0"})
    validate(stamped, require_provenance=True)
    assert is_stamped(stamped) and not is_stamped(toy_dataset)
    assert stamped.attrs["param_hash"] == toy_dataset.attrs["param_hash"]


def test_restamping_replaces_the_whole_previous_stamp(toy_dataset):
    first = stamp(
        toy_dataset,
        {"git_sha": "old", "git_dirty": 1, "pcsp_version": "0.0.1", "container_digest": "d1"},
    )
    second = stamp(first, {"git_sha": "new", "git_dirty": 0, "pcsp_version": "0.1.0"})
    assert second.attrs["git_sha"] == "new"
    assert "container_digest" not in second.attrs  # no mixing of two stamps


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"git_sha": "a", "git_dirty": 0}, "missing required keys"),
        ({"git_sha": "a", "git_dirty": 0, "pcsp_version": "1", "user": "me"}, "unexpected keys"),
    ],
)
def test_stamp_rejects_malformed_records(toy_dataset, record, message):
    with pytest.raises(ValueError, match=message):
        stamp(toy_dataset, record)
