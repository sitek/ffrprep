"""Unit tests for ffrprep.datasets download functions."""

import pytest
import shutil
from pathlib import Path
from ffrprep.datasets import (download_example_data, download_raw_data,
                              download_epoch_data)


@pytest.fixture
def test_dataset_path(tmp_path):
    """Fixture to provide a temporary dataset path for testing."""
    return tmp_path / "test_dataset"


@pytest.fixture
def expected_raw_files():
    """Return expected files after downloading raw data (1 subject)."""
    return [
        "ffrprep_raw_data/dataset_description.json",
        "ffrprep_raw_data/participants.json",
        "ffrprep_raw_data/participants.tsv",
        "ffrprep_raw_data/README",
        "ffrprep_raw_data/sub-21/eeg/sub-21_task-active_run-1_channels.tsv",
        "ffrprep_raw_data/sub-21/eeg/sub-21_task-active_run-1_eeg.bdf",
        "ffrprep_raw_data/sub-21/eeg/sub-21_task-active_run-1_eeg.json",
        "ffrprep_raw_data/sub-21/eeg/sub-21_task-active_run-1_events.json",
        "ffrprep_raw_data/sub-21/eeg/sub-21_task-active_run-1_events.tsv",
        "ffrprep_raw_data/sub-21/eeg/sub-21_task-passive_run-1_channels.tsv",
        "ffrprep_raw_data/sub-21/eeg/sub-21_task-passive_run-1_eeg.bdf",
        "ffrprep_raw_data/sub-21/eeg/sub-21_task-passive_run-1_eeg.json",
        "ffrprep_raw_data/sub-21/eeg/sub-21_task-passive_run-1_events.json",
        "ffrprep_raw_data/sub-21/eeg/sub-21_task-passive_run-1_events.tsv",
        "ffrprep_raw_data/sub-21/sub-21_scans.tsv",
    ]


@pytest.fixture
def available_raw_subjects():
    """Available subjects for raw data downloads."""
    return ["03", "05", "09", "11", "15", "21"]


@pytest.fixture
def available_epoch_subjects():
    """Available subjects for epoched data downloads."""
    return ["03", "15", "21", "30"]


def test_download_example_data_basic(test_dataset_path):
    """Test basic example data download functionality."""
    # Call the function to download example data (1 subject)
    data_path = download_example_data(test_dataset_path)

    # Basic assertions
    assert data_path is not None, "Download should return a path"
    assert isinstance(data_path, Path), "Returned path should be Path object"
    assert data_path.exists(), "Downloaded data directory should exist"
    expected_name = "ffrprep_raw_data"
    assert data_path.name == expected_name, "Should use raw data structure"

    # Clean up
    if data_path.exists():
        shutil.rmtree(data_path)


@pytest.mark.parametrize("n_subjects", [1, 2, 3])
def test_download_raw_data_by_number(test_dataset_path, n_subjects):
    """Test downloading raw data by specifying number of subjects."""
    data_path = download_raw_data(subjects=n_subjects,
                                  dataset_path=test_dataset_path)

    # Verify basic structure
    assert data_path is not None, "Download should return a path"
    assert data_path.exists(), "Downloaded data directory should exist"
    assert data_path.name == "ffrprep_raw_data", "Should create raw data dir"

    # Clean up
    if data_path.exists():
        shutil.rmtree(data_path)


@pytest.mark.parametrize("subject_list", [["03"], ["21"], ["03", "21"],
                                          ["05", "11", "15"]])
def test_download_raw_data_by_list(test_dataset_path, subject_list):
    """Test downloading raw data by specifying subject list."""
    data_path = download_raw_data(subjects=subject_list,
                                  dataset_path=test_dataset_path)

    # Verify basic structure
    assert data_path is not None, "Download should return a path"
    assert data_path.exists(), "Downloaded data directory should exist"

    # Clean up
    if data_path.exists():
        shutil.rmtree(data_path)


def test_download_raw_data_invalid_subjects(test_dataset_path,
                                            available_raw_subjects):
    """Test raw data download with invalid subject IDs."""
    # Test with completely invalid subjects
    invalid_subjects = ["99", "100"]

    with pytest.raises(ValueError, match="No valid subjects specified"):
        download_raw_data(subjects=invalid_subjects,
                          dataset_path=test_dataset_path)

    # Test with mix of valid and invalid subjects
    mixed_subjects = ["03", "99", "21"]
    data_path = download_raw_data(subjects=mixed_subjects,
                                  dataset_path=test_dataset_path)

    # Should succeed with valid subjects only
    assert data_path is not None
    assert data_path.exists()

    # Clean up
    if data_path.exists():
        shutil.rmtree(data_path)


def test_download_raw_data_too_many_subjects(test_dataset_path):
    """Test raw data download requesting more subjects than available."""
    # Request more subjects than available
    data_path = download_raw_data(subjects=10, dataset_path=test_dataset_path)

    # Should succeed with warning and download all available
    assert data_path is not None
    assert data_path.exists()

    # Clean up
    if data_path.exists():
        shutil.rmtree(data_path)


def test_download_raw_data_invalid_input_type(test_dataset_path):
    """Test raw data download with invalid input type."""
    with pytest.raises(TypeError, match="subjects must be an integer or list"):
        download_raw_data(subjects="invalid_string",
                          dataset_path=test_dataset_path)


@pytest.mark.parametrize("n_subjects", [1, 2])
def test_download_epoch_data_by_number(test_dataset_path, n_subjects):
    """Test downloading epoched data by specifying number of subjects."""
    data_path = download_epoch_data(subjects=n_subjects,
                                    dataset_path=test_dataset_path)

    # Verify BIDS derivatives structure
    assert data_path is not None, "Download should return a path"
    assert data_path.exists(), "Downloaded data directory should exist"
    assert data_path.name == "epochs", "Should be epochs directory"
    assert "derivatives" in str(data_path), "Should be in derivatives folder"

    # Clean up - remove the entire raw data directory
    base_path = data_path.parent.parent  # Go up to ffrprep_raw_data
    if base_path.exists() and base_path.name == "ffrprep_raw_data":
        shutil.rmtree(base_path)


@pytest.mark.parametrize("subject_list", [["03"], ["21"], ["03", "30"]])
def test_download_epoch_data_by_list(test_dataset_path, subject_list):
    """Test downloading epoched data by specifying subject list."""
    data_path = download_epoch_data(subjects=subject_list,
                                    dataset_path=test_dataset_path)

    # Verify BIDS derivatives structure
    assert data_path is not None, "Download should return a path"
    assert data_path.exists(), "Downloaded data directory should exist"

    # Check that subject directories would be created
    for subject in subject_list:
        subject_dir = data_path / f"sub-{subject}"
        # Note: directories might not exist if OSF IDs are placeholders
        print(f"Would create directory: {subject_dir}")

    # Clean up
    base_path = data_path.parent.parent  # Go up to ffrprep_raw_data
    if base_path.exists() and base_path.name == "ffrprep_raw_data":
        shutil.rmtree(base_path)


def test_download_epoch_data_invalid_subjects(test_dataset_path):
    """Test epoched data download with invalid subject IDs."""
    # Test with completely invalid subjects
    invalid_subjects = ["99", "100"]

    with pytest.raises(ValueError, match="No valid subjects specified"):
        download_epoch_data(subjects=invalid_subjects,
                            dataset_path=test_dataset_path)


def test_download_epoch_data_bids_structure(test_dataset_path):
    """Test that epoched data follows proper BIDS derivatives structure."""
    data_path = download_epoch_data(subjects=["03"],
                                    dataset_path=test_dataset_path)

    # Verify BIDS derivatives structure
    base_path = data_path.parent.parent  # ffrprep_raw_data
    derivatives_path = base_path / "derivatives"
    epochs_path = derivatives_path / "epochs"

    assert derivatives_path.exists(), "Derivatives directory should exist"
    assert epochs_path.exists(), "Epochs directory should exist"
    assert data_path == epochs_path, "Should return epochs path"

    # Clean up
    if base_path.exists():
        shutil.rmtree(base_path)


def test_all_functions_return_path_objects(test_dataset_path):
    """Test that all download functions return Path objects."""
    # Test download_example_data
    path1 = download_example_data(test_dataset_path)
    assert isinstance(path1, Path), "download_example_data should return Path"

    # Test download_raw_data
    path2 = download_raw_data(subjects=1, dataset_path=test_dataset_path)
    assert isinstance(path2, Path), "download_raw_data should return Path"

    # Test download_epoch_data
    path3 = download_epoch_data(subjects=1, dataset_path=test_dataset_path)
    assert isinstance(path3, Path), "download_epoch_data should return Path"

    # Clean up
    for path in [path1, path2]:
        if path and path.exists():
            shutil.rmtree(path)

    # Clean up epoch data (different structure)
    if path3 and path3.exists():
        base_path = path3.parent.parent
        if base_path.name == "ffrprep_raw_data":
            shutil.rmtree(base_path)


def test_default_parameters(test_dataset_path):
    """Test that functions work with default parameters."""
    # Should work with minimal arguments
    path1 = download_example_data()
    path2 = download_raw_data()
    path3 = download_epoch_data()

    # All should return valid paths
    assert path1 is not None
    assert path2 is not None
    assert path3 is not None

    # Clean up - these will be in current directory
    for path in [path1, path2]:
        if path and path.exists():
            shutil.rmtree(path)

    if path3 and path3.exists():
        base_path = path3.parent.parent
        if base_path.name == "ffrprep_raw_data":
            shutil.rmtree(base_path)


# ---------------------------------------------------------------------------
# download_stimuli + --with-stimuli wiring (mocked, no OSF traffic)
# ---------------------------------------------------------------------------


class _NullZip:
    """Minimal stand-in for zipfile.ZipFile in with_stimuli tests."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extractall(self, *args, **kwargs):
        pass


def test_download_stimuli_writes_to_root_stimuli_dir(tmp_path, monkeypatch):
    """download_stimuli puts every OSF-fetched file under <dataset>/stimuli/.

    Injects a known ``stim_urls`` mapping so the test stays decoupled
    from whatever real OSF IDs the module ships with; verifies that
    every entry is routed to ``_download_single_file`` with the
    BIDS-spec ``<dataset>/ffrprep_raw_data/stimuli`` destination.
    """
    from ffrprep.datasets import download_stimuli

    captured = []

    def fake_download_single_file(osf_url, dest_path, save_name=None):
        captured.append({
            "osf_url": osf_url,
            "dest_path": Path(dest_path),
            "save_name": save_name,
        })
        target = Path(dest_path) / (save_name or "downloaded.bin")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
        return target

    monkeypatch.setattr(
        "ffrprep.datasets._download_single_file",
        fake_download_single_file,
    )
    # Inject a known non-placeholder mapping so the loop fires once
    # regardless of what the production ``STIM_URLS`` constant
    # currently contains. (Real OSF IDs land later via a follow-up
    # commit.)
    monkeypatch.setattr(
        "ffrprep.datasets.STIM_URLS",
        {"fixture_stim.wav": "fakeosfid"},
    )

    stim_path = download_stimuli(dataset_path=tmp_path)

    assert stim_path is not None, "download_stimuli should return a path"
    assert isinstance(stim_path, Path), "Returned path should be Path"
    assert stim_path == tmp_path / "ffrprep_raw_data" / "stimuli", (
        "Stimuli must land at <dataset>/ffrprep_raw_data/stimuli per BIDS"
    )
    assert stim_path.exists(), "stimuli directory must be created"
    assert captured, "At least one stimulus file should be downloaded"
    for entry in captured:
        assert entry["dest_path"] == stim_path, (
            f"Every download routed to {stim_path}; got {entry['dest_path']}"
        )


def test_download_stimuli_default_dataset_path_is_cwd(monkeypatch, tmp_path):
    """When ``dataset_path=None``, files land under cwd/ffrprep_raw_data/stimuli."""
    from ffrprep.datasets import download_stimuli

    monkeypatch.chdir(tmp_path)

    def fake_download_single_file(osf_url, dest_path, save_name=None):
        target = Path(dest_path) / (save_name or "stub.bin")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
        return target

    monkeypatch.setattr(
        "ffrprep.datasets._download_single_file",
        fake_download_single_file,
    )

    stim_path = download_stimuli()
    # When dataset_path=None the helper builds a path off os.curdir
    # (matching the existing download_raw_data convention), so the
    # return may be relative; compare resolved forms.
    expected = tmp_path / "ffrprep_raw_data" / "stimuli"
    assert stim_path.resolve() == expected.resolve()
    assert stim_path.exists()


def test_download_example_data_default_with_stimuli_false(monkeypatch, tmp_path):
    """``download_example_data`` does NOT download stimuli by default."""
    from ffrprep.datasets import download_example_data

    stim_called = {"n": 0}

    def fake_download_raw_data(subjects=1, dataset_path=None, with_stimuli=False):
        return Path(dataset_path or tmp_path) / "ffrprep_raw_data"

    def fake_download_stimuli(dataset_path=None):
        stim_called["n"] += 1
        return Path("ignored")

    monkeypatch.setattr(
        "ffrprep.datasets.download_raw_data",
        fake_download_raw_data,
    )
    monkeypatch.setattr(
        "ffrprep.datasets.download_stimuli",
        fake_download_stimuli,
    )

    download_example_data(dataset_path=tmp_path)
    assert stim_called["n"] == 0, (
        "download_example_data must NOT fetch stimuli by default"
    )


def test_download_example_data_with_stimuli_true_calls_download_stimuli(
    monkeypatch, tmp_path,
):
    """``with_stimuli=True`` triggers exactly one ``download_stimuli`` call."""
    from ffrprep.datasets import download_example_data

    stim_calls = []

    def fake_download_raw_data(subjects=1, dataset_path=None, with_stimuli=False):
        return Path(dataset_path or tmp_path) / "ffrprep_raw_data"

    def fake_download_stimuli(dataset_path=None):
        stim_calls.append(dataset_path)
        return Path("stub")

    monkeypatch.setattr(
        "ffrprep.datasets.download_raw_data",
        fake_download_raw_data,
    )
    monkeypatch.setattr(
        "ffrprep.datasets.download_stimuli",
        fake_download_stimuli,
    )

    download_example_data(dataset_path=tmp_path, with_stimuli=True)
    assert len(stim_calls) == 1, (
        "with_stimuli=True must call download_stimuli exactly once"
    )
    assert stim_calls[0] == tmp_path, (
        "download_stimuli should receive the same dataset_path"
    )


def test_download_raw_data_with_stimuli_true_calls_download_stimuli(
    monkeypatch, tmp_path,
):
    """``download_raw_data(..., with_stimuli=True)`` also triggers stimuli."""
    from ffrprep.datasets import download_raw_data

    stim_calls = []

    def fake_download_single_file(osf_url, dest_path, save_name=None):
        target = Path(dest_path) / (save_name or "stub.bin")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
        return target

    def fake_download_stimuli(dataset_path=None):
        stim_calls.append(dataset_path)
        return Path("stub")

    monkeypatch.setattr(
        "ffrprep.datasets._download_single_file",
        fake_download_single_file,
    )
    monkeypatch.setattr(
        "ffrprep.datasets.download_stimuli",
        fake_download_stimuli,
    )
    # Skip the unzip step — fake_download_single_file just touches a
    # plain file rather than a real zip, so patch ZipFile to a no-op.
    monkeypatch.setattr(
        "ffrprep.datasets.zipfile.ZipFile",
        lambda *a, **kw: _NullZip(),
    )
    # Suppress the post-unzip remove of the (non-existent) zip.
    monkeypatch.setattr("ffrprep.datasets.os.remove", lambda *a, **kw: None)

    download_raw_data(
        subjects=["21"], dataset_path=tmp_path, with_stimuli=True,
    )
    assert len(stim_calls) == 1
    assert stim_calls[0] == tmp_path


# ---------------------------------------------------------------------------
# STIM_URLS + events.tsv augmentation (example-dataset specific)
# ---------------------------------------------------------------------------

def test_stim_urls_contains_pol1_and_pol2():
    """The production STIM_URLS dict carries both example-dataset polarities.

    Regression-lock so a future edit doesn't silently empty the dict
    again, leaving ``--with-stimuli`` with nothing to download.
    """
    from ffrprep.datasets import STIM_URLS

    assert "Da_Stimulus_44100Hz_pol1.wav" in STIM_URLS
    assert "Da_Stimulus_44100Hz_pol2.wav" in STIM_URLS
    for filename, osf_id in STIM_URLS.items():
        assert isinstance(osf_id, str) and len(osf_id) > 0, (
            f"STIM_URLS[{filename!r}] must hold a non-empty OSF id"
        )


def _write_events_tsv(path, rows, columns=("onset", "duration", "trial_type", "value")):
    """Helper: write a minimal BIDS events.tsv at `path`.

    ``rows`` is a list of dicts keyed by column name.
    """
    import pandas as pd
    df = pd.DataFrame(rows, columns=list(columns))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def test_augment_events_with_stim_file_adds_column(tmp_path):
    """Walks every events.tsv under bids_root and adds a stim_file column."""
    from ffrprep.datasets import _augment_events_with_stim_file

    bids_root = tmp_path / "ds"
    events_path = bids_root / "sub-03" / "eeg" / "sub-03_task-active_run-1_events.tsv"
    _write_events_tsv(events_path, [
        {"onset": 1.0, "duration": 0.17, "trial_type": "positive", "value": 1},
        {"onset": 2.0, "duration": 0.17, "trial_type": "negative", "value": 2},
    ])

    _augment_events_with_stim_file(bids_root, mapping={
        "positive": "stimuli/da_positive.wav",
        "negative": "stimuli/da_negative.wav",
    })

    import pandas as pd
    df = pd.read_csv(events_path, sep="\t")
    assert "stim_file" in df.columns
    assert df.loc[df["trial_type"] == "positive", "stim_file"].iloc[0] == (
        "stimuli/da_positive.wav"
    )
    assert df.loc[df["trial_type"] == "negative", "stim_file"].iloc[0] == (
        "stimuli/da_negative.wav"
    )


def test_augment_events_with_stim_file_skips_when_column_present(tmp_path):
    """An existing stim_file column is preserved — the helper is idempotent."""
    from ffrprep.datasets import _augment_events_with_stim_file

    bids_root = tmp_path / "ds"
    events_path = bids_root / "sub-03" / "eeg" / "sub-03_task-active_run-1_events.tsv"
    _write_events_tsv(events_path, [
        {"onset": 1.0, "duration": 0.17, "trial_type": "positive", "value": 1,
         "stim_file": "stimuli/custom.wav"},
    ], columns=("onset", "duration", "trial_type", "value", "stim_file"))

    _augment_events_with_stim_file(bids_root, mapping={
        "positive": "stimuli/da_positive.wav",
    })

    import pandas as pd
    df = pd.read_csv(events_path, sep="\t")
    assert df["stim_file"].iloc[0] == "stimuli/custom.wav", (
        "existing stim_file values must not be overwritten"
    )


def test_augment_events_with_stim_file_leaves_unknown_trial_types_as_na(tmp_path):
    """Trial types not in the mapping table get an n/a stim_file entry."""
    from ffrprep.datasets import _augment_events_with_stim_file

    bids_root = tmp_path / "ds"
    events_path = bids_root / "sub-03" / "eeg" / "sub-03_task-active_run-1_events.tsv"
    _write_events_tsv(events_path, [
        {"onset": 1.0, "duration": 0.17, "trial_type": "positive", "value": 1},
        {"onset": 2.0, "duration": 0.17, "trial_type": "other", "value": 3},
    ])

    _augment_events_with_stim_file(bids_root, mapping={
        "positive": "stimuli/da_positive.wav",
    })

    import pandas as pd
    df = pd.read_csv(events_path, sep="\t")
    assert df.loc[df["trial_type"] == "positive", "stim_file"].iloc[0] == (
        "stimuli/da_positive.wav"
    )
    # Unknown trial_types must surface as the BIDS sentinel "n/a"
    # (pandas reads it back as NaN, which we then normalise).
    val = df.loc[df["trial_type"] == "other", "stim_file"].iloc[0]
    assert val == "n/a" or (isinstance(val, float) and pd.isna(val)), (
        f"unknown trial_type stim_file should be n/a / NaN; got {val!r}"
    )


def test_augment_events_with_stim_file_walks_multiple_subjects(tmp_path):
    """Walks every sub-*/eeg/*_events.tsv, not just one fixture file."""
    from ffrprep.datasets import _augment_events_with_stim_file

    bids_root = tmp_path / "ds"
    paths = [
        bids_root / "sub-03" / "eeg" / "sub-03_task-active_run-1_events.tsv",
        bids_root / "sub-03" / "eeg" / "sub-03_task-passive_run-1_events.tsv",
        bids_root / "sub-21" / "eeg" / "sub-21_task-active_run-1_events.tsv",
    ]
    for p in paths:
        _write_events_tsv(p, [
            {"onset": 1.0, "duration": 0.17, "trial_type": "positive", "value": 1},
        ])

    _augment_events_with_stim_file(bids_root, mapping={
        "positive": "stimuli/da_positive.wav",
    })

    import pandas as pd
    for p in paths:
        df = pd.read_csv(p, sep="\t")
        assert "stim_file" in df.columns, f"{p} missing stim_file column"


def test_download_stimuli_invokes_augment_events(tmp_path, monkeypatch):
    """``download_stimuli`` augments events.tsv after writing the wav files."""
    from ffrprep.datasets import download_stimuli

    # Build a fake bids tree the helper can walk
    bids_root = tmp_path / "ffrprep_raw_data"
    events_path = bids_root / "sub-03" / "eeg" / "sub-03_task-active_run-1_events.tsv"
    _write_events_tsv(events_path, [
        {"onset": 1.0, "duration": 0.17, "trial_type": "positive", "value": 1},
    ])

    def fake_download_single_file(osf_url, dest_path, save_name=None):
        target = Path(dest_path) / (save_name or "stub.bin")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
        return target

    monkeypatch.setattr(
        "ffrprep.datasets._download_single_file",
        fake_download_single_file,
    )
    monkeypatch.setattr(
        "ffrprep.datasets.STIM_URLS",
        {"Da_Stimulus_44100Hz_pol1.wav": "fake1"},
    )

    download_stimuli(dataset_path=tmp_path)

    import pandas as pd
    df = pd.read_csv(events_path, sep="\t")
    assert "stim_file" in df.columns, (
        "download_stimuli must augment events.tsv as part of its workflow"
    )
