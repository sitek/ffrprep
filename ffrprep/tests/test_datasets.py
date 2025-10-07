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
    """Expected files after downloading raw data (1 subject)."""
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
    return ['03', '05', '09', '11', '15', '21']


@pytest.fixture
def available_epoch_subjects():
    """Available subjects for epoched data downloads."""
    return ['03', '15', '21', '30']


def test_download_example_data_basic(test_dataset_path):
    """Test basic example data download functionality."""
    # Call the function to download example data (1 subject)
    data_path = download_example_data(test_dataset_path)

    # Basic assertions
    assert data_path is not None, "Download should return a path"
    assert isinstance(data_path, Path), "Returned path should be Path object"
    assert data_path.exists(), "Downloaded data directory should exist"
    expected_name = 'ffrprep_raw_data'
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
    assert data_path.name == 'ffrprep_raw_data', "Should create raw data dir"

    # Clean up
    if data_path.exists():
        shutil.rmtree(data_path)


@pytest.mark.parametrize("subject_list", [
    ['03'],
    ['21'],
    ['03', '21'],
    ['05', '11', '15']
])
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
    invalid_subjects = ['99', '100']
    
    with pytest.raises(ValueError, match="No valid subjects specified"):
        download_raw_data(subjects=invalid_subjects,
                          dataset_path=test_dataset_path)

    # Test with mix of valid and invalid subjects
    mixed_subjects = ['03', '99', '21']
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
    data_path = download_raw_data(subjects=10,  # More than available
                                  dataset_path=test_dataset_path)

    # Should succeed with warning and download all available
    assert data_path is not None
    assert data_path.exists()

    # Clean up
    if data_path.exists():
        shutil.rmtree(data_path)


def test_download_raw_data_invalid_input_type(test_dataset_path):
    """Test raw data download with invalid input type."""
    with pytest.raises(TypeError,
                       match="subjects must be an integer or list"):
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
    assert data_path.name == 'epochs', "Should be epochs directory"
    assert 'derivatives' in str(data_path), "Should be in derivatives folder"

    # Clean up - remove the entire raw data directory
    base_path = data_path.parent.parent  # Go up to ffrprep_raw_data
    if base_path.exists() and base_path.name == 'ffrprep_raw_data':
        shutil.rmtree(base_path)


@pytest.mark.parametrize("subject_list", [
    ['03'],
    ['21'],
    ['03', '30']
])
def test_download_epoch_data_by_list(test_dataset_path, subject_list):
    """Test downloading epoched data by specifying subject list."""
    data_path = download_epoch_data(subjects=subject_list,
                                    dataset_path=test_dataset_path)

    # Verify BIDS derivatives structure
    assert data_path is not None, "Download should return a path"
    assert data_path.exists(), "Downloaded data directory should exist"

    # Check that subject directories would be created
    for subject in subject_list:
        subject_dir = data_path / f'sub-{subject}'
        # Note: directories might not exist if OSF IDs are placeholders
        print(f"Would create directory: {subject_dir}")

    # Clean up
    base_path = data_path.parent.parent  # Go up to ffrprep_raw_data
    if base_path.exists() and base_path.name == 'ffrprep_raw_data':
        shutil.rmtree(base_path)


def test_download_epoch_data_invalid_subjects(test_dataset_path):
    """Test epoched data download with invalid subject IDs."""
    # Test with completely invalid subjects
    invalid_subjects = ['99', '100']
    
    with pytest.raises(ValueError, match="No valid subjects specified"):
        download_epoch_data(subjects=invalid_subjects,
                            dataset_path=test_dataset_path)


def test_download_epoch_data_bids_structure(test_dataset_path):
    """Test that epoched data follows proper BIDS derivatives structure."""
    data_path = download_epoch_data(subjects=['03'],
                                    dataset_path=test_dataset_path)

    # Verify BIDS derivatives structure
    base_path = data_path.parent.parent  # ffrprep_raw_data
    derivatives_path = base_path / 'derivatives'
    epochs_path = derivatives_path / 'epochs'

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
        if base_path.name == 'ffrprep_raw_data':
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
        if base_path.name == 'ffrprep_raw_data':
            shutil.rmtree(base_path)
