import shutil
import pytest
from ffrprep.datasets import download_unzip_exp_data
from ffrprep.preproc import load_data, reference_data, filter_data, epoch_data, preproc_pipeline


@pytest.fixture
def osf_url():
    # Provide a valid OSF URL for testing
    return "https://osf.io/download/kap3z"


def test_load_data(osf_url, tmp_path):
    # Use temporary directory for testing
    dataset_path = tmp_path / "test_dataset"

    # Download and unzip the data
    data_path = download_unzip_exp_data(osf_url, dataset_path)

    # Call the function to load correct data
    load_data(bids_root=data_path,
                                sub_label='21',
                                session_label=None,
                                task_label='passive',
                                run_label=1,)

    # Should fail if not given specific EEG file
    try:
        load_data(bids_root=dataset_path)
    except FileNotFoundError:
        print('BIDSPath must point to a specific EEG file')

    # Clean up the downloaded files after the test
    shutil.rmtree(dataset_path)


def test_reference_data(osf_url, tmp_path):
    # Use temporary directory for testing
    dataset_path = tmp_path / "test_dataset"

    # Download and unzip the data
    data_path = download_unzip_exp_data(osf_url, dataset_path)

    # Call the function to load correct data
    data, _ = load_data(bids_root=data_path,
                                sub_label='21',
                                session_label=None,
                                task_label='passive',
                                run_label=1,)

    # Call the function to re-reference the data
    reference_data(data, ref_channels=None)

    # Should fail if not given existing EEG file
    try:
        reference_data(data, ref_channels='fake_channel')
    except ValueError:
        print('Given reference channel does not exist')

    # Clean up the downloaded files after the test
    shutil.rmtree(dataset_path)


def test_filter_data(osf_url, tmp_path):
    # Use temporary directory for testing
    dataset_path = tmp_path / "test_dataset"

    # Download and unzip the data
    data_path = download_unzip_exp_data(osf_url, dataset_path)

    # Call the function to load correct data
    data, _ = load_data(bids_root=data_path,
                                sub_label='21',
                                session_label=None,
                                task_label='passive',
                                run_label=1,)

    # Call the function to filter the data
    filter_data(data, high_pass=80, low_pass=1000)

    # Should fail if low-pass (upper) frequency is above
    # the Nyquist frequency of the EEG data
    try:
        filter_data(data, low_pass=99999)
    except ValueError:
        print('Upper edge frequency of the filter must be below '
              'the Nyquist frequency of the EEG data')

    # Clean up the downloaded files after the test
    shutil.rmtree(dataset_path)


def test_epoch_data(osf_url, tmp_path):
    # Use temporary directory for testing
    dataset_path = tmp_path / "test_dataset"

    # Download and unzip the data
    data_path = download_unzip_exp_data(osf_url, dataset_path)
    data, _ = load_data(bids_root=data_path,
                                sub_label='21',
                                session_label=None,
                                task_label='passive',
                                run_label=1)

    # Using both a baseline window and a baseline float should both be accepted
    epoch_data(eeg_data=data, picks='Cz', baseline=-0.05, verbose='WARNING')
    epoch_data(eeg_data=data, picks='Cz', baseline=[-0.05, 0.2], verbose='WARNING')

    # An invalid baseline should result in error
    with pytest.raises(ValueError) as e:
        epoch_data(eeg_data=data, picks='Cz', baseline=999.0, verbose='WARNING')
        print("Successfully caught error: " + e)

    # Clean up the downloaded files after the test
    shutil.rmtree(dataset_path)


def test_preproc_pipeline(osf_url, tmp_path):
    # Use temporary directory for testing
    dataset_path = tmp_path / "test_dataset"

    # Download and unzip the data
    data_path = download_unzip_exp_data(osf_url, dataset_path)
    data = preproc_pipeline(bids_root=data_path,
                                baseline=-0.05,
                                sub_label='21',
                                session_label=None,
                                task_label='passive',
                                run_label=1,
                                ref_channels=['M1'],
                                picks='Cz',
                                high_pass=80,
                                low_pass=2000,
                                verbose=False)
    
    print(data[0])

    # Clean up the downloaded files after the test
    shutil.rmtree(dataset_path)