"""Shared pytest fixtures for the ffrprep test suite.

Two fixtures back the integration and system tiers:

* :func:`bids_dataset` (session-scoped) — returns a BIDS root containing
  subjects ``03`` and ``21``. If ``FFRPREP_TEST_CACHE`` is set and already
  contains the required subjects, the download step is skipped; otherwise
  :func:`ffrprep.datasets.download_raw_data` is invoked once per session.

* :func:`bids_workspace` (function-scoped) — mirrors the dataset into a
  per-test temp directory by symlinking each ``sub-*`` directory and
  copying the BIDS metadata. The pipeline writes derivatives into the
  workspace, leaving the cached source untouched.

Note on cache reuse: :func:`download_raw_data` removes the per-subject zip
after extraction, so it would re-download on every call if the cache was
non-empty but the zip absent. The fixture short-circuits that by checking
for already-extracted subject directories before invoking the downloader.
"""
import os
import shutil
from pathlib import Path

import pytest

from ffrprep.datasets import download_raw_data


REQUIRED_SUBJECTS = ["03", "21"]
BIDS_METADATA_FILES = (
    "dataset_description.json",
    "participants.json",
    "participants.tsv",
    "README",
)


def _all_subjects_present(bids_root, subjects):
    """Return True iff bids_root has every requested subject extracted."""
    if not bids_root.is_dir():
        return False
    return all((bids_root / f"sub-{sub}").is_dir() for sub in subjects)


def _resolve_cache_dir(tmp_path_factory):
    """Pick the dataset cache: env var if set, else a session tmp dir."""
    cache_env = os.environ.get("FFRPREP_TEST_CACHE")
    if cache_env:
        cache_dir = Path(cache_env)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    return Path(tmp_path_factory.mktemp("ffrprep_test_data"))


@pytest.fixture(scope="session")
def bids_dataset(tmp_path_factory):
    """Path to a BIDS root with subjects 03 and 21 available on disk."""
    cache_dir = _resolve_cache_dir(tmp_path_factory)
    bids_root = cache_dir / "ffrprep_raw_data"
    if _all_subjects_present(bids_root, REQUIRED_SUBJECTS):
        return bids_root
    return Path(
        download_raw_data(subjects=REQUIRED_SUBJECTS, dataset_path=str(cache_dir))
    )


@pytest.fixture
def bids_workspace(bids_dataset, tmp_path):
    """Per-test writable BIDS root mirroring ``bids_dataset`` via symlinks."""
    workspace = tmp_path / "bids"
    workspace.mkdir()
    for name in BIDS_METADATA_FILES:
        src = bids_dataset / name
        if src.exists():
            shutil.copy(src, workspace / name)
    for subject in REQUIRED_SUBJECTS:
        (workspace / f"sub-{subject}").symlink_to(bids_dataset / f"sub-{subject}")
    return workspace
