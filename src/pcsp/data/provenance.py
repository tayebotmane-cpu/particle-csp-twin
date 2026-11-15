"""Provenance stamps: record *what code* produced a dataset.

The schema (:mod:`pcsp.data.schema`) already stores the full input parameters
(``params_json``) and the model name/version. This module adds the rest of
the reproducibility record when a dataset is archived:

=====================  =========================================================
``git_sha``            commit of the repository this package was imported from,
                       or ``"unknown"`` (not a git checkout, git not installed)
``git_dirty``          ``1`` if the working tree had uncommitted changes
                       (including untracked, non-ignored files), ``0`` if clean,
                       ``-1`` if unknown
``pcsp_version``       :data:`pcsp.__version__`
``container_digest``   optional; taken from the environment variable
                       ``PCSP_CONTAINER_DIGEST`` when it is set and non-empty
``hostname``           optional and OFF by default (it can identify a person)
=====================  =========================================================

Collecting provenance never raises: anything that cannot be determined is
recorded explicitly as unknown, so a missing ``git`` never blocks a run.
"""

from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path

import xarray as xr

import pcsp
from pcsp.data.schema import OPTIONAL_PROVENANCE_ATTRS, PROVENANCE_ATTRS

UNKNOWN = "unknown"
"""Value of ``git_sha`` when the commit cannot be determined."""

GIT_DIRTY_UNKNOWN = -1
"""Value of ``git_dirty`` when the tree state cannot be determined."""

CONTAINER_DIGEST_ENV = "PCSP_CONTAINER_DIGEST"

_PACKAGE_DIR = Path(pcsp.__file__).resolve().parent
_GIT_TIMEOUT_S = 10.0


def _git(*args: str) -> str | None:
    """Run ``git <args>`` in the package directory; ``None`` on any failure."""
    try:
        result = subprocess.run(
            ["git", "-C", str(_PACKAGE_DIR), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout


def _git_state() -> tuple[str, int]:
    """Return ``(sha, dirty)`` for the checkout this package lives in.

    The package must be *tracked* by the repository that git finds. This stops
    an installed copy (e.g. in a virtualenv that happens to sit inside some
    other repository) from being stamped with an unrelated commit.
    """
    if _git("ls-files", "--error-unmatch", "__init__.py") is None:
        return UNKNOWN, GIT_DIRTY_UNKNOWN
    sha = _git("rev-parse", "HEAD")
    if sha is None or not sha.strip():
        return UNKNOWN, GIT_DIRTY_UNKNOWN
    status = _git("status", "--porcelain")
    dirty = GIT_DIRTY_UNKNOWN if status is None else int(bool(status.strip()))
    return sha.strip(), dirty


def collect_provenance(*, include_hostname: bool = False) -> dict[str, str | int]:
    """Collect a provenance record whose keys are the schema's provenance attributes.

    Parameters
    ----------
    include_hostname : also record ``socket.gethostname()``. Off by default for
        privacy; datasets may be shared outside the group.
    """
    sha, dirty = _git_state()
    record: dict[str, str | int] = {
        "git_sha": sha,
        "git_dirty": dirty,
        "pcsp_version": pcsp.__version__,
    }
    digest = os.environ.get(CONTAINER_DIGEST_ENV, "").strip()
    if digest:
        record["container_digest"] = digest
    if include_hostname:
        record["hostname"] = socket.gethostname()
    return record


def is_stamped(ds: xr.Dataset) -> bool:
    """True if every required provenance attribute is present."""
    return all(key in ds.attrs for key in PROVENANCE_ATTRS)


def stamp(
    ds: xr.Dataset,
    provenance: dict[str, str | int] | None = None,
    *,
    include_hostname: bool = False,
) -> xr.Dataset:
    """Return a shallow copy of ``ds`` carrying a provenance stamp.

    Any previous stamp (required and optional attributes) is removed first, so
    the result never mixes two stamps. ``ds`` itself is not modified.

    Parameters
    ----------
    provenance : a record from :func:`collect_provenance`. Pass one in to
        collect once and stamp many datasets; by default it is collected now.
    include_hostname : forwarded to :func:`collect_provenance` when
        ``provenance`` is not given.
    """
    if provenance is None:
        provenance = collect_provenance(include_hostname=include_hostname)
    allowed = set(PROVENANCE_ATTRS) | set(OPTIONAL_PROVENANCE_ATTRS)
    unexpected = set(provenance) - allowed
    if unexpected:
        raise ValueError(
            f"provenance has unexpected keys {sorted(unexpected)}; allowed: {sorted(allowed)}"
        )
    missing = set(PROVENANCE_ATTRS) - set(provenance)
    if missing:
        raise ValueError(f"provenance is missing required keys {sorted(missing)}")
    out = ds.copy(deep=False)
    out.attrs = {k: v for k, v in ds.attrs.items() if k not in allowed}
    out.attrs.update(provenance)
    return out
