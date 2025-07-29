import shutil
import pytest
from ffrprep.datasets import download_unzip_exp_data
from ffrprep.preproc import load_data, filter_data, epoch_data
from ffrprep.classifier import classify_data

@pytest.fixture
def osf_url():
    # Provide a valid OSF URL for testing
    return "https://osf.io/download/kap3z"

def test_classify_data(osf_url, tmp_path):
    dataset_path = tmp_path / "test_dataset"

    # Download and unzip the data
    data_path = download_unzip_exp_data(osf_url, dataset_path)

    data, bids_path = load_data(bids_root=data_path,
                                sub_label='21',
                                session_label=None,
                                task_label='passive',
                                run_label=1)
    
    # filter_data(eeg_data=data, high_pass=80, low_pass=1000)
    epoched_data = epoch_data(eeg_data=data, baseline=[-500.0, 0.5], tmin=-0.200, tmax=2.000, verbose='WARNING')
    print(epoched_data)

    epochs = epoched_data[0]
    tmin, tmax = epoched_data[1]
    classify_data(epochs=epochs, min_freq=8.0, max_freq=20.0, n_freqs=6, n_cycles=10, tmin=tmin, tmax=tmax)

    # Clean up the downloaded files after the test
    shutil.rmtree(dataset_path)
