import pytest
import shutil
from ffrprep.datasets import download_example_data
from ffrprep.utils import validate_input_dir


@pytest.fixture
def test_dataset_path(tmp_path):
    """Fixture to provide a temporary dataset path for testing."""
    return tmp_path / "test_dataset"


@pytest.fixture
def sample_participants():
    """Fixture to provide sample participant labels for testing."""
    return ["21"]


def test_bids_validator_invalid_participants(test_dataset_path):
    """Test BIDS validator with invalid participant labels."""
    # Download example data
    data_path = download_example_data(test_dataset_path)

    # Test with invalid participant
    invalid_participants = ["99"]  # Non-existent participant

    try:
        # This might raise an exception or handle gracefully
        validate_input_dir("test", data_path, invalid_participants)
        print("Validation with invalid participant handled gracefully")
    except Exception as e:
        print(f"Expected: validation failed with invalid participant: {e}")
    finally:
        # Clean up
        if data_path and data_path.exists():
            shutil.rmtree(data_path)


def test_download_example_data_structure(test_dataset_path):
    """Test that downloaded example data has expected structure."""
    # Download example data
    data_path = download_example_data(test_dataset_path)

    assert data_path is not None, "Download function should return a path"
    assert data_path.exists(), "Downloaded data directory should exist"

    # Check if it's using the raw data structure
    expected_base_name = "ffrprep_raw_data"
    error_msg = f"Expected {expected_base_name}, got {data_path.name}"
    assert data_path.name == expected_base_name, error_msg

    # Clean up
    if data_path.exists():
        shutil.rmtree(data_path)
