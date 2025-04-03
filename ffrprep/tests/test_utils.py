import pytest
import shutil
from ffrprep.datasets import download_unzip_exp_data
from ffrprep.utils import validate_input_dir


@pytest.fixture
def osf_url():
    # Provide a valid OSF URL for testing
    return "https://osf.io/download/kap3z"


@pytest.fixture
def expected_files():
    # List of expected files after unzipping
    return [
        "ffrprep_dataset/dataset_description.json",
        "ffrprep_dataset/participants.json",
        "ffrprep_dataset/participants.tsv",
        "ffrprep_dataset/README",
        "ffrprep_dataset/sub-21/eeg/sub-21_task-active_run-1_channels.tsv",
        "ffrprep_dataset/sub-21/eeg/sub-21_task-active_run-1_eeg.bdf",
        "ffrprep_dataset/sub-21/eeg/sub-21_task-active_run-1_eeg.json",
        "ffrprep_dataset/sub-21/eeg/sub-21_task-active_run-1_events.json",
        "ffrprep_dataset/sub-21/eeg/sub-21_task-active_run-1_events.tsv",
        "ffrprep_dataset/sub-21/eeg/sub-21_task-active_run-2_channels.tsv",
        "ffrprep_dataset/sub-21/eeg/sub-21_task-active_run-2_eeg.bdf",
        "ffrprep_dataset/sub-21/eeg/sub-21_task-active_run-2_eeg.json",
        "ffrprep_dataset/sub-21/eeg/sub-21_task-active_run-2_events.json",
        "ffrprep_dataset/sub-21/eeg/sub-21_task-active_run-2_events.tsv",
        "ffrprep_dataset/sub-21/eeg/sub-21_task-passive_run-1_channels.tsv",
        "ffrprep_dataset/sub-21/eeg/sub-21_task-passive_run-1_eeg.bdf",
        "ffrprep_dataset/sub-21/eeg/sub-21_task-passive_run-1_eeg.json",
        "ffrprep_dataset/sub-21/eeg/sub-21_task-passive_run-1_events.json",
        "ffrprep_dataset/sub-21/eeg/sub-21_task-passive_run-1_events.tsv",
        "ffrprep_dataset/sub-21/sub-21_scans.tsv",
    ]


def test_bids_validator(osf_url, expected_files, tmp_path):

    # Use temporary directory for testing
    dataset_path = tmp_path / "test_dataset"

    # Set the execution environment to test
    exec_env = 'test'

    # Set the participant label
    participants = ['21']

    # Call the function to download and unzip the data
    download_unzip_exp_data(osf_url, dataset_path)

    # call the validator function
    validate_input_dir(exec_env,
                       dataset_path.joinpath('ffrprep_dataset'),
                       participants)

    # Clean up the downloaded files after the test
    shutil.rmtree(dataset_path)
