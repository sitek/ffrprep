"""Integration tests against the downloaded BIDS dataset.

Each test depends on the ``bids_dataset`` session-scoped fixture (defined in
``conftest.py``). The fixture downloads at least subjects ``03`` and ``21``
via :func:`ffrprep.datasets.download_raw_data` and returns the BIDS root.

Run inside the container::

    docker run --rm \\
      -v /path/to/data:/cache \\
      -e FFRPREP_TEST_CACHE=/cache \\
      ffrprep:local test ffrprep/tests/test_integration_data.py -v

Set ``FFRPREP_TEST_CACHE`` to a stable host path mounted into the container
to avoid re-downloading between sessions.

Note: the OSF dataset has no ``ses-*`` level, so the multi-session axis is
covered by a single ``pytest.skip`` rather than by parametrisation.
"""
from pathlib import Path

import mne
import pytest

from ffrprep.preproc import get_participants, get_sessions_tasks_runs


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Dataset shape on disk
# ---------------------------------------------------------------------------

def test_dataset_root_exists(bids_dataset):
    assert Path(bids_dataset).is_dir()


def test_dataset_metadata_present(bids_dataset):
    root = Path(bids_dataset)
    assert (root / "dataset_description.json").is_file()
    assert (root / "participants.tsv").is_file()


@pytest.mark.parametrize("subject", ["03", "21"])
def test_subject_eeg_directory_present(bids_dataset, subject):
    assert (Path(bids_dataset) / f"sub-{subject}" / "eeg").is_dir()


# ---------------------------------------------------------------------------
# Subject queries — single and multiple
# ---------------------------------------------------------------------------

def test_get_participants_single_subject(bids_dataset):
    assert get_participants(str(bids_dataset), ["03"]) == ["03"]


def test_get_participants_multiple_subjects(bids_dataset):
    result = get_participants(str(bids_dataset), ["03", "21"])
    assert sorted(result) == ["03", "21"]


def test_get_participants_unfiltered_includes_targets(bids_dataset):
    result = get_participants(str(bids_dataset))
    assert {"03", "21"}.issubset(set(result))


# ---------------------------------------------------------------------------
# Run queries — single and multiple
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "subject,task,min_runs",
    [
        ("03", "active", 3),
        ("03", "passive", 3),
        ("21", "active", 4),
        ("21", "passive", 4),
    ],
)
def test_minimum_runs_per_task_on_disk(bids_dataset, subject, task, min_runs):
    eeg_dir = Path(bids_dataset) / f"sub-{subject}" / "eeg"
    bdf_files = sorted(eeg_dir.glob(f"sub-{subject}_task-{task}_run-*_eeg.bdf"))
    assert len(bdf_files) >= min_runs


def test_get_sessions_tasks_runs_exposes_multiple_runs(bids_dataset):
    metadata = get_sessions_tasks_runs(str(bids_dataset), "03")
    assert len(metadata["runs"]) > 1


def test_get_sessions_tasks_runs_exposes_both_tasks(bids_dataset):
    metadata = get_sessions_tasks_runs(str(bids_dataset), "03")
    assert {"active", "passive"}.issubset(set(metadata["tasks"]))


# ---------------------------------------------------------------------------
# Session axis — dataset has no ses-* level, so just document the gap
# ---------------------------------------------------------------------------

def test_get_sessions_tasks_runs_session_axis(bids_dataset):
    metadata = get_sessions_tasks_runs(str(bids_dataset), "03")
    if metadata["sessions"] == [None]:
        pytest.skip("OSF dataset has no ses-* level — multi-session axis untested")
    assert len(metadata["sessions"]) >= 2


# ---------------------------------------------------------------------------
# Raw EEG file is loadable via mne
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("subject", ["03", "21"])
def test_raw_bdf_loads_with_mne(bids_dataset, subject):
    eeg_dir = Path(bids_dataset) / f"sub-{subject}" / "eeg"
    bdf = sorted(eeg_dir.glob("*_eeg.bdf"))[0]
    raw = mne.io.read_raw_bdf(str(bdf), preload=False, verbose="ERROR")
    assert raw.info["sfreq"] > 0
    assert len(raw.ch_names) > 0
