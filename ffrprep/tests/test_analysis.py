import shutil
import pytest
import mne
from ffrprep.datasets import download_unzip_exp_data
from ffrprep.preproc import load_data, reference_data
from ffrprep.preproc import filter_data, epoch_data, preproc_pipeline
from ffrprep.preproc import make_evoked
from ffrprep.analysis import compute_power, rms_snr, autocorrelation

@pytest.fixture
def osf_url():
    # Provide a valid OSF URL for testing
    return "https://osf.io/download/kap3z"


def test_compute_power(osf_url, tmp_path):
    # Use temporary directory for testing
    dataset_path = tmp_path / "test_dataset"

    # Download and unzip the data
    data_path = download_unzip_exp_data(osf_url, dataset_path)

    eeg_data = load_data(bids_root=data_path,
                         sub_label='21',
                         session_label=None,
                         task_label='passive',
                         run_label=1,)

    epochs = epoch_data(eeg_data, baseline=[-0.05, 0.2])
    evoked = make_evoked(epochs=epochs, by_event_type=True)

    # Test function with defaults
    compute_power(evoked)


def test_rms_snr(osf_url, tmp_path):
    # Use temporary directory for testing
    dataset_path = tmp_path / "test_dataset"

    # Download and unzip the data
    data_path = download_unzip_exp_data(osf_url, dataset_path)

    eeg_data = load_data(bids_root=data_path,
                         sub_label='21',
                         session_label=None,
                         task_label='passive',
                         run_label=1,)

    epochs = epoch_data(eeg_data, baseline=[-0.05, 0.2])
    evoked = make_evoked(epochs=epochs, by_event_type=True)

    # Test function with defaults
    rms_snr(evoked)


def test_autocorrelation(osf_url, tmp_path):
    # Use temporary directory for testing
    dataset_path = tmp_path / "test_dataset"

    # Download and unzip the data
    data_path = download_unzip_exp_data(osf_url, dataset_path)

    eeg_data = load_data(bids_root=data_path,
                         sub_label='21',
                         session_label=None,
                         task_label='passive',
                         run_label=1,)

    epochs = epoch_data(eeg_data, baseline=[-0.05, 0.2])
    evoked = make_evoked(epochs=epochs, by_event_type=True)

    # Test function with defaults
    autocorrelation(evoked)
