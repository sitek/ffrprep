"""Unit tests for ffrprep utility functions."""
from pathlib import Path

import pytest

from ffrprep.datasets import download_example_data
from ffrprep.utils import validate_input_dir


@pytest.mark.integration
def test_validate_input_dir_rejects_unknown_participant(bids_dataset):
    """Validation should raise RuntimeError when a requested participant
    label is not present in the dataset."""
    with pytest.raises(RuntimeError):
        validate_input_dir("test", Path(bids_dataset), ["99"])


def test_download_example_data_returns_bids_root(tmp_path):
    """download_example_data returns a path to the standard
    ``ffrprep_raw_data`` BIDS root directory."""
    data_path = download_example_data(tmp_path / "test_dataset")

    assert data_path is not None
    assert data_path.exists()
    assert data_path.name == "ffrprep_raw_data"
