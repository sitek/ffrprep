from pathlib import Path
import pytest
import shutil
from ffrprep.datasets import download_unzip_exp_data
from ffrprep.tests.test_datasets import osf_url
from ffrprep.preproc import load_data

def test_load_data(tmp_path):
    # Use temporary directory for testing
    dataset_path = tmp_path / "test_dataset"

    # Download and unzip the data
    download_unzip_exp_data(osf_url, dataset_path)

    # Call the function to load correct data
    data, bids_path = load_data(bids_root=dataset_path, 
                                sub_label=21, 
                                session_label=None, 
                                task_label='passive', 
                                run_label=1,)

    # Should fail if not given specific EEG file
    try:
        data, bids_path = load_data(bids_root=dataset_path)
    except FileNotFoundError:
        print('BIDSPath must point to a specific EEG file')

    # Clean up the downloaded files after the test
    shutil.rmtree(dataset_path)
