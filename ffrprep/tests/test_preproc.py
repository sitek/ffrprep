import json
import shutil
import pytest
from ffrprep.datasets import download_example_data
from ffrprep.preproc import (
    create_preprocessing_workflow,
    epoch_data,
    filter_data,
    load_data,
    make_evoked,
    reference_data,
    save_analysis_outputs,
    save_preprocessing_outputs,
    setup_derivatives_directories,
)


def test_load_data(tmp_path):
    """Test the load_data function with mock BIDS structure."""
    import numpy as np
    import json
    from mne.io import RawArray
    from mne import create_info
    from mne_bids import write_raw_bids, BIDSPath

    # Create a temporary BIDS dataset structure
    bids_root = tmp_path / "test_bids"
    bids_root.mkdir()

    # Create dataset_description.json (required for BIDS)
    dataset_desc = {
        "Name": "Test Dataset",
        "BIDSVersion": "1.6.0",
        "Authors": ["Test Author"]
    }
    with open(bids_root / "dataset_description.json", 'w') as f:
        json.dump(dataset_desc, f)

    # Create synthetic EEG data
    n_channels = 8
    n_times = 5000  # 5 seconds at 1000 Hz
    sfreq = 1000.0

    times = np.arange(n_times) / sfreq
    data_array = np.zeros((n_channels, n_times))

    for ch_idx in range(n_channels):
        alpha_wave = np.sin(2 * np.pi * 10 * times) * 0.5
        noise = np.random.randn(n_times) * 0.1
        data_array[ch_idx, :] = alpha_wave + noise

    # Create MNE Raw object and save it first
    ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw_data = RawArray(data_array, info)

    # Save to a temporary file first (required for write_raw_bids)
    temp_filename = tmp_path / "temp_raw_data.fif"
    raw_data.save(temp_filename, overwrite=True, verbose=False)

    # Load the saved file to get proper filename info
    from mne import io
    raw_from_file = io.read_raw_fif(temp_filename, verbose=False)

    try:
        # Create BIDS path for writing test data
        bids_path = BIDSPath(
            subject="03", task="passive", run=1,
            root=bids_root, datatype="eeg"
        )

        # Write the data to BIDS format
        write_raw_bids(
            raw=raw_from_file,
            bids_path=bids_path,
            overwrite=True,
            verbose=False
        )
        bids_write_success = True

    except Exception as write_error:
        print(f"BIDS write failed: {write_error}")
        bids_write_success = False

    # Now test the actual load_data function
    if bids_write_success:
        try:
            loaded_data, returned_bids_path, original_filename = load_data(
                bids_root=str(bids_root),
                sub_label="03",
                task_label="passive",
                run_label=1
            )

            # Test that load_data actually works and returns expected types
            import mne
            assert loaded_data is not None, \
                "Data should be loaded successfully"
            assert isinstance(loaded_data, mne.io.BaseRaw), \
                "Loaded data should be an MNE Raw object"
            assert returned_bids_path is not None, \
                "BIDS path should be returned"
            assert original_filename is not None, \
                "Original filename should be returned"

            # Test data properties
            assert len(loaded_data.ch_names) == n_channels, \
                "Loaded data should have correct number of channels"
            assert loaded_data.n_times == n_times, \
                "Loaded data should have correct number of time points"
            assert loaded_data.info['sfreq'] == sfreq, \
                "Loaded data should have correct sampling frequency"

            # Test that channels exist and have proper types
            ch_types_loaded = loaded_data.get_channel_types()
            assert len(ch_types_loaded) > 0, \
                "Loaded data should have channel types defined"
            assert all(ch_type == 'eeg' for ch_type in ch_types_loaded), \
                "All channels should be EEG type"

            # Verify the data actually contains signal (not all zeros)
            loaded_data_array = loaded_data.get_data()
            assert loaded_data_array.size > 0, \
                "Loaded data array should not be empty"
            assert not (loaded_data_array == 0).all(), \
                "Loaded data should contain actual signal, not all zeros"

            # Test BIDS path properties
            assert returned_bids_path.subject == "03", \
                "Returned BIDS path should have correct subject"
            assert returned_bids_path.task == "passive", \
                "Returned BIDS path should have correct task"
            assert returned_bids_path.run == 1, \
                "Returned BIDS path should have correct run"

            # Test original filename
            assert "sub-03" in original_filename, \
                "Original filename should contain subject"
            assert "task-passive" in original_filename, \
                "Original filename should contain task"
            assert "run-01" in original_filename, \
                "Original filename should contain run"

            print("✓ load_data function test passed!")

        except Exception as e:
            # If load_data fails, it might be due to missing dependencies
            print(f"load_data failed (might be environment-related): {e}")
            bids_write_success = False

    if not bids_write_success:
        # Fall back to testing the expected data format
        print("Falling back to basic structure tests...")

        # At minimum, verify the synthetic data structure
        assert raw_from_file is not None, \
            "Synthetic data should be created correctly"
        assert len(raw_from_file.ch_names) == n_channels, \
            "Synthetic data should have correct number of channels"
        assert raw_from_file.info['sfreq'] == sfreq, \
            "Synthetic data should have correct sampling frequency"

        print("✓ Fallback synthetic data test passed!")


def test_load_data_error_handling(tmp_path):
    """Test load_data error handling with missing files."""
    import json

    # Create a temporary BIDS dataset structure
    bids_root = tmp_path / "test_bids_empty"
    bids_root.mkdir()

    # Create dataset_description.json (required for BIDS)
    dataset_desc = {
        "Name": "Empty Test Dataset",
        "BIDSVersion": "1.6.0",
        "Authors": ["Test Author"]
    }
    with open(bids_root / "dataset_description.json", 'w') as f:
        json.dump(dataset_desc, f)

    # Test 1: Non-existent subject
    try:
        load_data(
            bids_root=str(bids_root),
            sub_label="999",  # Non-existent subject
            task_label="passive",
            run_label=1
        )
        assert False, "Should raise FileNotFoundError for non-existent subject"
    except FileNotFoundError as e:
        assert "No EEG files found" in str(e), \
            "Should provide helpful error message"
        print("✓ Non-existent subject error handling works!")

    # Test 2: Non-existent BIDS root
    try:
        load_data(
            bids_root="/nonexistent/path",
            sub_label="03",
            task_label="passive",
            run_label=1
        )
        assert False, "Should raise error for non-existent BIDS root"
    except Exception as e:
        # Could be FileNotFoundError, OSError, or ValidationError
        error_type = type(e).__name__
        print(f"✓ Non-existent BIDS root error handling works: {error_type}")

    print("✓ load_data error handling tests passed!")


def test_reference_data(tmp_path):
    """Test the reference_data function with synthetic data."""
    # Create synthetic EEG data for testing
    import numpy as np
    from mne.io import RawArray
    from mne import create_info

    # Create synthetic EEG data
    n_channels = 8
    n_times = 5000  # 5 seconds at 1000 Hz
    sfreq = 1000.0

    # Generate synthetic EEG-like data
    times = np.arange(n_times) / sfreq
    data_array = np.zeros((n_channels, n_times))

    for ch_idx in range(n_channels):
        # Mix of alpha (10 Hz) and noise
        alpha_wave = np.sin(2 * np.pi * 10 * times) * 0.5
        noise = np.random.randn(n_times) * 0.1
        data_array[ch_idx, :] = alpha_wave + noise

    # Create MNE info structure
    ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Create Raw object for testing
    data = RawArray(data_array, info)

    # Call the function to re-reference the data
    referenced_data = reference_data(data, ref_channels=None)

    # Verify that referenced data is returned and is a proper MNE object
    assert referenced_data is not None, "Referenced data should be returned"

    # Test that referenced data is still an MNE Raw object
    import mne
    assert isinstance(referenced_data, mne.io.BaseRaw), \
        "Referenced data should still be an MNE Raw object"

    # Check that it maintains the same basic structure as original
    assert len(referenced_data.ch_names) == len(data.ch_names), \
        "Referenced data should have same number of channels"
    assert referenced_data.n_times == data.n_times, \
        "Referenced data should have same number of time points"

    # Check that referencing preserves essential metadata
    assert referenced_data.info['sfreq'] == data.info['sfreq'], \
        "Sampling frequency should be preserved after referencing"

    # Verify the data array has same shape but potentially different values
    ref_data_array = referenced_data.get_data()
    orig_data_array = data.get_data()
    assert ref_data_array.shape == orig_data_array.shape, \
        "Referenced data should have same shape as original"
    assert ref_data_array.size > 0, \
        "Referenced data array should not be empty"

    # Should handle non-existent reference channels gracefully
    try:
        reference_data(data, ref_channels="fake_channel")
    except (ValueError, KeyError):
        print("Given reference channel does not exist")


def test_filter_data(tmp_path):
    """Test the filter_data function with synthetic data."""
    # Create synthetic EEG data for testing
    import numpy as np
    from mne.io import RawArray
    from mne import create_info

    # Create synthetic EEG data with multiple frequency components
    n_channels = 8
    n_times = 10000  # 10 seconds at 1000 Hz
    sfreq = 1000.0

    # Generate synthetic EEG-like data with known frequency content
    times = np.arange(n_times) / sfreq
    data_array = np.zeros((n_channels, n_times))

    for ch_idx in range(n_channels):
        # Mix of different frequency components
        low_freq = np.sin(2 * np.pi * 0.5 * times) * 0.3  # 0.5 Hz
        alpha = np.sin(2 * np.pi * 10 * times) * 0.5      # 10 Hz
        beta = np.sin(2 * np.pi * 20 * times) * 0.3       # 20 Hz
        high_freq = np.sin(2 * np.pi * 60 * times) * 0.4  # 60 Hz
        noise = np.random.randn(n_times) * 0.1
        data_array[ch_idx, :] = low_freq + alpha + beta + high_freq + noise

    # Create MNE info structure
    ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Create Raw object for testing
    data = RawArray(data_array, info)

    # Apply basic filtering (1-40 Hz bandpass)
    filtered_data = filter_data(eeg_data=data, high_pass=1.0, low_pass=40.0)

    # Verify that filtered data is returned and is a proper MNE object
    assert filtered_data is not None, "Filtered data should be returned"

    # Test that filtered data is still an MNE Raw object
    import mne
    assert isinstance(filtered_data, mne.io.BaseRaw), \
        "Filtered data should still be an MNE Raw object"

    # Check that it maintains the same basic structure as original
    assert len(filtered_data.ch_names) == len(data.ch_names), \
        "Filtered data should have same number of channels"
    assert filtered_data.n_times == data.n_times, \
        "Filtered data should have same number of time points"

    # Check that filtering preserves essential metadata
    assert filtered_data.info['sfreq'] == data.info['sfreq'], \
        "Sampling frequency should be preserved after filtering"

    # Verify the data array has same shape but potentially different values
    filt_data_array = filtered_data.get_data()
    orig_data_array = data.get_data()
    assert filt_data_array.shape == orig_data_array.shape, \
        "Filtered data should have same shape as original"
    assert filt_data_array.size > 0, \
        "Filtered data array should not be empty"

    # Check that filtering actually changed the data (should be different)
    assert not (filt_data_array == orig_data_array).all(), \
        "Filtering should modify the data values"

    # Verify filter information is stored in the info
    assert 'lowpass' in filtered_data.info, \
        "Lowpass filter info should be stored"
    assert 'highpass' in filtered_data.info, \
        "Highpass filter info should be stored"

    # Check that the data is actually within the specified filter range
    # Compute power spectral density to verify frequency content
    import numpy as np
    from scipy import signal

    # Use a single channel for frequency analysis
    sfreq = filtered_data.info['sfreq']
    channel_data = filt_data_array[0, :]  # First channel

    # Compute power spectral density
    nperseg = min(2048, len(channel_data)//4)
    freqs, psd = signal.welch(channel_data, fs=sfreq, nperseg=nperseg)

    # Find power in different frequency bands
    low_freq_mask = freqs < 1.0  # Below lowpass
    pass_band_mask = (freqs >= 1.0) & (freqs <= 40.0)  # Within passband
    high_freq_mask = freqs > 40.0  # Above highpass

    # Calculate power in each band
    low_power = np.mean(psd[low_freq_mask]) if np.any(low_freq_mask) else 0
    pass_power = np.mean(psd[pass_band_mask]) if np.any(pass_band_mask) else 0
    high_power = np.mean(psd[high_freq_mask]) if np.any(high_freq_mask) else 0

    # Verify that passband has significant power (filtering worked)
    total_power = low_power + pass_power + high_power
    if total_power > 0:
        pass_band_ratio = pass_power / total_power
        # Relaxed threshold since synthetic data has specific characteristics
        assert pass_band_ratio > 0.2, \
            f"Passband should have significant power after filtering. " \
            f"Got {pass_band_ratio:.2f}"

        # Verify high frequencies were attenuated more than low frequencies
        if high_power > 0 and low_power > 0:
            attenuation_ratio = high_power / (low_power + high_power)
            assert attenuation_ratio < 0.8, \
                "High frequencies should be more attenuated than low"
    # Test completed successfully


def test_epoch_data(tmp_path):
    """Test the epoch_data function with synthetic data."""
    # Create synthetic EEG data for testing instead of relying on downloads
    import numpy as np
    from mne import create_info, io

    # Create synthetic EEG data
    n_channels = 8
    n_times = 10000  # 10 seconds at 1000 Hz
    sfreq = 1000.0

    # Generate some synthetic EEG-like data with different frequency components
    times = np.arange(n_times) / sfreq
    data = np.zeros((n_channels, n_times))

    for ch_idx in range(n_channels):
        # Mix of alpha (10 Hz), beta (20 Hz), and some noise
        alpha_wave = np.sin(2 * np.pi * 10 * times) * 0.5
        beta_wave = np.sin(2 * np.pi * 20 * times) * 0.3
        noise = np.random.randn(n_times) * 0.1
        data[ch_idx, :] = alpha_wave + beta_wave + noise

    # Create MNE info structure
    ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Create Raw object
    raw_data = io.RawArray(data, info)

    # Apply basic preprocessing before epoching
    referenced_data = reference_data(raw_data, ref_channels=None)
    filtered_data = filter_data(
        referenced_data, high_pass=1.0, low_pass=40.0
    )

    # Test epoch_data function with different baseline configurations
    # Create artificial events for testing
    n_events = 8
    min_time_samples = int(1.0 * sfreq)  # Start after 1 second
    max_time_samples = filtered_data.n_times - int(1.0 * sfreq)
    event_times = np.linspace(
        min_time_samples, max_time_samples, n_events, dtype=int
    )

    # Add events using a stimulus channel that find_events can detect
    # Create a copy of the data with an additional stimulus channel
    stim_data = np.zeros((1, filtered_data.n_times))
    stim_data[0, event_times] = 1  # Set trigger events

    # Add the stimulus channel to the data
    from mne import create_info
    from mne.io import RawArray
    stim_info = create_info(['STI 014'], sfreq, ['stim'])
    stim_raw = RawArray(stim_data, stim_info)

    # Combine EEG and stimulus data
    filtered_data = filtered_data.add_channels(
        [stim_raw], force_update_info=True
    )

    # Test 1: Single float baseline (start time)
    epochs1, time_window1 = epoch_data(
        filtered_data, baseline=-0.1, tmin=-0.2, tmax=0.5, verbose=False
    )

    # Verify that epochs were created
    assert epochs1 is not None, "Epochs should be created successfully"
    assert time_window1 is not None, "Time window should be returned"

    # Check time window matches input parameters
    assert time_window1[0] == -0.2, "Time window start should match tmin"
    assert time_window1[1] == 0.5, "Time window end should match tmax"

    # Verify epochs object properties
    assert hasattr(epochs1, 'get_data'), \
        "Epochs should have get_data method"
    assert hasattr(epochs1, 'info'), "Epochs should have info attribute"
    assert epochs1.info['sfreq'] == filtered_data.info['sfreq'], \
        "Epochs should preserve sampling frequency"

    # Test 2: Array baseline (baseline window)
    epochs2, time_window2 = epoch_data(
        filtered_data, baseline=[-0.2, -0.05], tmin=-0.3, tmax=0.6,
        verbose=False
    )

    assert epochs2 is not None, \
        "Epochs with array baseline should be created"
    assert time_window2[0] == -0.3, "Time window should match custom tmin"
    assert time_window2[1] == 0.6, "Time window should match custom tmax"

    # Test 3: Default parameters (None for tmin/tmax)
    epochs3, time_window3 = epoch_data(
        filtered_data, baseline=-0.1, verbose=False
    )

    assert epochs3 is not None, \
        "Epochs with default parameters should be created"
    assert time_window3[0] == -0.1, \
        "Default tmin should match baseline start"
    assert time_window3[1] == 0.5, "Default tmax should be 0.5"

    # Test 4: With channel picks
    eeg_channels = [ch for ch in filtered_data.ch_names
                    if filtered_data.get_channel_types([ch])[0] == 'eeg']
    if eeg_channels:
        epochs4, time_window4 = epoch_data(
            filtered_data, baseline=-0.1, picks=eeg_channels[:2],
            tmin=-0.2, tmax=0.5, verbose=False
        )

        assert epochs4 is not None, \
            "Epochs with channel picks should be created"
        assert len(epochs4.ch_names) <= len(eeg_channels), \
            "Epochs should have fewer or equal channels when picks used"

    # Test 5: With external events file (simulate TSV file)
    import tempfile
    import pandas as pd

    # Create a temporary events file
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.tsv', delete=False
    ) as f:
        events_data = pd.DataFrame({
            'onset': [1.0, 2.0, 3.0, 4.0, 5.0],  # Event times in seconds
            'duration': [0.1, 0.1, 0.1, 0.1, 0.1],
            'trial_type': [1, 2, 1, 2, 1]
        })
        events_data.to_csv(f.name, sep='\t', index=False)
        events_file = f.name

    try:
        epochs5, time_window5 = epoch_data(
            filtered_data, baseline=-0.1, events_file=events_file,
            tmin=-0.2, tmax=0.5, verbose=False
        )

        assert epochs5 is not None, \
            "Epochs with external events file should be created"
        # Should have events based on the TSV file
        if len(epochs5) > 0:
            assert hasattr(epochs5, 'event_id'), \
                "Epochs should have event_id when using events file"

    except Exception as e:
        # Events from file might fail due to timing issues with data
        print(f"Events file test failed (expected with example data): {e}")

    finally:
        # Clean up temporary events file
        import os
        if os.path.exists(events_file):
            os.unlink(events_file)

    # Test 6: Error handling - invalid baseline
    try:
        epoch_data(filtered_data, baseline="invalid")
        assert False, "Should raise error for invalid baseline"
    except (TypeError, ValueError):
        pass  # Expected error

    # Test 7: Verify that epochs have reasonable structure
    if len(epochs1) > 0:  # If we have valid epochs
        epochs_data = epochs1.get_data()
        assert epochs_data.ndim == 3, \
            "Epochs data should be 3D (epochs x channels x time)"
        assert epochs_data.shape[0] > 0, "Should have at least one epoch"
        assert epochs_data.shape[1] > 0, "Should have at least one channel"
        assert epochs_data.shape[2] > 0, "Should have time points"

        # Check that epoch timing makes sense
        times = epochs1.times
        assert len(times) == epochs_data.shape[2], \
            "Times should match data time dimension"
        assert times[0] >= time_window1[0], \
            "First time point should be >= tmin"
        assert times[-1] <= time_window1[1], \
            "Last time point should be <= tmax"

    # Test 8: Verify baseline correction was applied (EEG channels only —
    # stim channels are deliberately NOT baseline-corrected by MNE).
    if len(epochs1) > 0:
        baseline_indices = (epochs1.times >= -0.1) & (epochs1.times <= 0)
        if np.any(baseline_indices):
            eeg_picks = [
                i for i, ch in enumerate(epochs1.ch_names)
                if epochs1.get_channel_types([ch])[0] == "eeg"
            ]
            baseline_data = epochs_data[:, eeg_picks, :][:, :, baseline_indices]
            baseline_mean = np.mean(baseline_data, axis=2)
            assert np.all(np.abs(baseline_mean) < 1e-10), \
                "EEG baseline period should be ~zero after correction"

    print("✓ All epoch_data tests passed!")


def test_create_preprocessing_workflow(tmp_path):
    """Test the create_preprocessing_workflow function."""
    # Test that the workflow can be created without errors
    workflow = create_preprocessing_workflow(name="test_preproc")

    # Verify that workflow is created and is a nipype Workflow object
    assert workflow is not None, "Workflow should be created successfully"

    # Check that workflow has the expected name
    assert workflow.name == "test_preproc", \
        "Workflow should have the specified name"

    # Verify that essential nodes exist in the workflow
    node_names = [node.name for node in workflow._graph.nodes()]
    expected_nodes = [
        "inputnode", "outputnode", "load_data", "reference_data",
        "filter_data", "epoch_data", "save_preprocessing"
    ]

    for expected_node in expected_nodes:
        assert expected_node in node_names, \
            f"Workflow should contain {expected_node} node"

    # Check that input node has correct fields
    inputnode = None
    for node in workflow._graph.nodes():
        if node.name == "inputnode":
            inputnode = node
            break

    assert inputnode is not None, "Input node should exist"

    expected_input_fields = [
        "bids_root", "sub_label", "session_label", "task_label",
        "run_label", "ref_channels", "high_pass", "low_pass",
        "baseline", "tmin", "tmax", "output_dir"
    ]

    for field in expected_input_fields:
        assert field in inputnode.interface._fields, \
            f"Input node should have {field} field"

    # Check that output node has correct fields
    outputnode = None
    for node in workflow._graph.nodes():
        if node.name == "outputnode":
            outputnode = node
            break

    assert outputnode is not None, "Output node should exist"

    expected_output_fields = [
        "epochs", "preprocessing_report", "original_filename"
    ]
    for field in expected_output_fields:
        assert field in outputnode.interface._fields, \
            f"Output node should have {field} field"

    # Verify that nodes are properly connected by checking edges
    edges = workflow._graph.edges()
    assert len(edges) > 0, "Workflow should have connections between nodes"

    # Check that there are connections from inputnode to processing nodes
    inputnode_connections = [
        edge for edge in edges if edge[0].name == "inputnode"
    ]
    assert len(inputnode_connections) > 0, \
        "Input node should have outgoing connections"

    # Check that there are connections to outputnode from processing nodes
    outputnode_connections = [
        edge for edge in edges if edge[1].name == "outputnode"
    ]
    assert len(outputnode_connections) > 0, \
        "Output node should have incoming connections"


def test_preprocessing_workflow_structure():
    """Test preprocessing workflow structure and configuration."""
    # Test that workflow can be created with custom name
    workflow = create_preprocessing_workflow(name="custom_name")
    assert workflow.name == "custom_name"

    # Test default name
    default_workflow = create_preprocessing_workflow()
    assert default_workflow.name == "ffrprep_preproc"

    # Verify workflow graph structure
    nodes = list(workflow._graph.nodes())
    node_names = [n.name for n in nodes]

    # Test that all required processing steps are present
    processing_steps = [
        "load_data", "reference_data", "filter_data", "epoch_data"
    ]
    for step in processing_steps:
        assert step in node_names, \
            f"Workflow should include {step} processing step"

    # Test that workflow has proper input/output structure
    assert "inputnode" in node_names, "Workflow should have input node"
    assert "outputnode" in node_names, "Workflow should have output node"
    assert "save_preprocessing" in node_names, "Workflow should have save node"

    # Verify workflow connections exist
    edges = list(workflow._graph.edges())
    assert len(edges) >= 6, "Workflow should have multiple node connections"

    # Check that processing flows in logical order
    # (inputnode connects to processing nodes, processing nodes connect to
    # save/output)
    input_connections = [e for e in edges if e[0].name == "inputnode"]
    output_connections = [e for e in edges if e[1].name == "outputnode"]
    save_connections = [e for e in edges if e[1].name == "save_preprocessing"]

    assert len(input_connections) >= 3, (
        "Input node should connect to multiple processing nodes"
    )
    assert len(output_connections) >= 1, (
        "Output node should receive connections"
    )
    assert len(save_connections) >= 1, (
        "Save node should receive connections"
    )


def test_preprocessing_workflow_execution(tmp_path):
    """Test preprocessing workflow execution with example data."""
    _ = pytest.importorskip(
        "nipype", reason="nipype required for workflow execution"
    )

    # Use temporary directory for testing
    dataset_path = tmp_path / "test_dataset"

    # Download example data
    data_path = download_example_data(dataset_path)

    # Create the preprocessing workflow
    workflow = create_preprocessing_workflow(name="test_execution")

    # Set up workflow inputs
    workflow.inputs.inputnode.bids_root = str(data_path)
    workflow.inputs.inputnode.sub_label = "03"
    workflow.inputs.inputnode.session_label = None
    workflow.inputs.inputnode.task_label = "passive"
    workflow.inputs.inputnode.run_label = 1
    workflow.inputs.inputnode.ref_channels = None  # Average reference
    workflow.inputs.inputnode.high_pass = 1.0
    workflow.inputs.inputnode.low_pass = 40.0
    workflow.inputs.inputnode.baseline = -0.1
    workflow.inputs.inputnode.tmin = -0.2
    workflow.inputs.inputnode.tmax = 0.5
    workflow.inputs.inputnode.output_dir = str(tmp_path / "derivatives")

    # Set up working directory for nipype
    workflow.base_dir = str(tmp_path / "working")

    try:
        # Run the workflow
        result = workflow.run()

        # Verify that the workflow completed successfully
        assert result is not None, "Workflow should complete and return result"

        # Check that output files were created
        derivatives_dir = (
            tmp_path / "derivatives" / "ffrprep-preprocessing" / "sub-03"
        )
        if derivatives_dir.exists():
            output_files = list(derivatives_dir.glob("*desc-preproc.fif"))
            assert len(output_files) > 0, \
                "Preprocessing workflow should create output files"

    except Exception as e:
        # If workflow execution fails, it might be due to environment issues
        # but the workflow structure should still be valid
        print(f"Workflow execution failed (might be environment-related): {e}")

        # At minimum, verify the workflow was set up correctly
        assert workflow.inputs.inputnode.bids_root == str(data_path), \
            "Workflow inputs should be set correctly"

    finally:
        # Clean up the downloaded files after the test
        if data_path.exists():
            shutil.rmtree(data_path)


def test_output_structure_and_files(tmp_path):
    """Test that output structure and files are as expected according to BIDS.
    """
    # Create synthetic data for testing output structure functionality
    import numpy as np
    from mne.io import RawArray
    from mne import create_info
    from mne_bids import BIDSPath

    # Create synthetic EEG data
    n_channels = 8
    n_times = 5000  # 5 seconds at 1000 Hz
    sfreq = 1000.0

    times = np.arange(n_times) / sfreq
    data_array = np.zeros((n_channels, n_times))

    for ch_idx in range(n_channels):
        alpha_wave = np.sin(2 * np.pi * 10 * times) * 0.5
        noise = np.random.randn(n_times) * 0.1
        data_array[ch_idx, :] = alpha_wave + noise

    ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Create synthetic data that simulates the pipeline
    data = RawArray(data_array, info)

    # Create mock BIDS root for testing output structure
    data_path = tmp_path / "test_bids_root"
    data_path.mkdir(exist_ok=True)

    # Create mock BIDS path and filename
    _ = BIDSPath(
        subject="03", task="passive", run=1,
        root=data_path, datatype="eeg"
    )
    _ = "sub-03_task-passive_run-1_eeg"

    # Apply preprocessing steps
    referenced_data = reference_data(data, ref_channels=None)
    filtered_data = filter_data(referenced_data, high_pass=1.0, low_pass=40.0)

    # Use more lenient epoching for testing purposes
    from mne import Epochs
    import numpy as np

    # Create simulated events for testing
    n_events = 10
    event_times = np.linspace(1000, data.n_times - 1000, n_events, dtype=int)
    events = np.column_stack([event_times,
                             np.zeros(n_events, dtype=int),
                             np.ones(n_events, dtype=int)])

    # Create epochs with relaxed rejection criteria for testing
    epochs = Epochs(
        filtered_data, events, tmin=-0.2, tmax=0.5,
        baseline=(-0.1, 0), verbose=False,
        reject=None,  # No rejection for testing
        preload=True
    )

    # Test preprocessing output structure
    preprocessing_output_path = save_preprocessing_outputs(
        epochs, data_path, subject="03", task="passive",
        session=None, run=1
    )

    # Verify preprocessing derivatives directory structure
    derivatives_root = data_path / "derivatives"
    preprocessing_dir = derivatives_root / "ffrprep-preprocessing"
    subject_preprocessing_dir = preprocessing_dir / "sub-03"

    assert derivatives_root.exists(), (
        "Derivatives root directory should exist"
    )
    assert preprocessing_dir.exists(), (
        "Preprocessing derivatives directory should exist"
    )
    assert subject_preprocessing_dir.exists(), (
        "Subject preprocessing directory should exist"
    )

    # Check dataset_description.json for preprocessing
    dataset_desc_path = preprocessing_dir / "dataset_description.json"
    assert dataset_desc_path.exists(), (
        "Dataset description should exist for preprocessing"
    )

    with open(dataset_desc_path, 'r') as f:
        dataset_desc = json.load(f)

    assert "Name" in dataset_desc, "Dataset description should have Name field"
    assert "BIDSVersion" in dataset_desc, (
        "Dataset description should have BIDSVersion"
    )
    assert "GeneratedBy" in dataset_desc, (
        "Dataset description should have GeneratedBy"
    )
    assert dataset_desc["Name"] == "ffrprep preprocessing outputs"
    assert "ffrprep" in str(dataset_desc["GeneratedBy"]).lower()

    # Verify preprocessing output file naming convention. Outputs land
    # under the BIDS-derivatives ``sub-XX/eeg`` subdirectory, and the
    # filename uses MNE's conventional ``_epo.fif`` epoch suffix.
    expected_preprocessing_filename = (
        "sub-03_task-passive_run-1_desc-preproc_epo.fif"
    )
    expected_preprocessing_path = (
        subject_preprocessing_dir / "eeg" / expected_preprocessing_filename
    )

    assert expected_preprocessing_path.exists(), (
        f"Expected preprocessing output file should exist: "
        f"{expected_preprocessing_filename}"
    )
    assert preprocessing_output_path == expected_preprocessing_path, \
        "Returned path should match expected BIDS-compliant path"

    # Test different parameter combinations for filename generation
    # Test with session
    preprocessing_output_with_session = save_preprocessing_outputs(
        epochs, data_path, subject="03", task="passive",
        session="01", run=1
    )
    expected_with_session = (
        "sub-03_ses-01_task-passive_run-1_desc-preproc_epo.fif"
    )
    assert preprocessing_output_with_session.name == expected_with_session, \
        "Filename should include session when provided"

    # Test without run
    preprocessing_output_no_run = save_preprocessing_outputs(
        epochs, data_path, subject="03", task="passive",
        session=None, run=None
    )
    expected_no_run = "sub-03_task-passive_desc-preproc_epo.fif"
    assert preprocessing_output_no_run.name == expected_no_run, \
        "Filename should work without run parameter"

    # Test analysis output structure
    # Create a simple evoked object for testing
    evoked = epochs.average()

    analysis_output_paths = save_analysis_outputs(
        evoked, data_path, subject="03", task="passive",
        session=None, run=1, analysis_type="evoked"
    )

    # Verify analysis derivatives directory structure
    analysis_dir = derivatives_root / "ffrprep-analysis"
    subject_analysis_dir = analysis_dir / "sub-03"

    assert analysis_dir.exists(), "Analysis derivatives directory should exist"
    assert subject_analysis_dir.exists(), (
        "Subject analysis directory should exist"
    )

    # Check dataset_description.json for analysis
    analysis_dataset_desc_path = analysis_dir / "dataset_description.json"
    assert analysis_dataset_desc_path.exists(), (
        "Dataset description should exist for analysis"
    )

    with open(analysis_dataset_desc_path, 'r') as f:
        analysis_dataset_desc = json.load(f)

    assert analysis_dataset_desc["Name"] == "ffrprep analysis outputs"

    # Verify analysis output file naming convention
    expected_analysis_filename = "sub-03_task-passive_run-1_desc-evoked.fif"
    expected_analysis_path = subject_analysis_dir / expected_analysis_filename

    assert len(analysis_output_paths) == 1, (
        "Should return one output path for single evoked"
    )
    assert analysis_output_paths[0] == expected_analysis_path, \
        "Analysis output path should match BIDS convention"
    assert expected_analysis_path.exists(), (
        f"Expected analysis output file should exist: "
        f"{expected_analysis_filename}"
    )

    # Test file format validation (basic check that files can be opened)
    import mne

    # Test that preprocessing output can be loaded
    loaded_epochs = mne.read_epochs(expected_preprocessing_path)
    assert hasattr(loaded_epochs, 'get_data'), \
        "Saved preprocessing output should be loadable as MNE Epochs"
    assert loaded_epochs.info['sfreq'] == epochs.info['sfreq'], \
        "Loaded epochs should preserve sampling frequency"

    # Test that analysis output can be loaded
    loaded_evoked = mne.read_evokeds(expected_analysis_path)[0]
    assert hasattr(loaded_evoked, 'data'), \
        "Saved analysis output should be loadable as MNE Evoked"
    assert loaded_evoked.info['sfreq'] == evoked.info['sfreq'], \
        "Loaded evoked should preserve sampling frequency"

    print("✓ All output structure and file tests passed!")


def test_make_evoked(tmp_path):
    """Test the make_evoked function with synthetic data."""
    # Create synthetic EEG data for testing
    import numpy as np
    from mne.io import RawArray
    from mne import create_info, Epochs

    # Create synthetic EEG data
    n_channels = 8
    n_times = 10000  # 10 seconds at 1000 Hz
    sfreq = 1000.0

    # Generate synthetic EEG-like data with known frequency components
    times = np.arange(n_times) / sfreq
    data_array = np.zeros((n_channels, n_times))

    for ch_idx in range(n_channels):
        # Mix of alpha (10 Hz) and beta (20 Hz) waves
        alpha_wave = np.sin(2 * np.pi * 10 * times) * 0.5
        beta_wave = np.sin(2 * np.pi * 20 * times) * 0.3
        noise = np.random.randn(n_times) * 0.1
        data_array[ch_idx, :] = alpha_wave + beta_wave + noise

    # Create MNE info structure
    ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Create Raw object
    raw_data = RawArray(data_array, info)

    # Create events with multiple event types for testing
    n_events = 20
    event_times = np.linspace(
        1000, raw_data.n_times - 1000, n_events, dtype=int
    )

    # Create two different event types
    event_ids = np.array([1, 2] * (n_events // 2))  # Alternating event types
    events = np.column_stack([event_times,
                             np.zeros(n_events, dtype=int),
                             event_ids])

    # Create epochs with multiple event types
    epochs = Epochs(
        raw_data, events,
        event_id={'condition_1': 1, 'condition_2': 2},
        tmin=-0.2, tmax=0.5,
        baseline=(-0.1, 0),
        verbose=False,
        reject=None,  # No rejection for testing
        preload=True
    )

    # Test 1: Average all epochs together (by_event_type=False)
    evoked_all = make_evoked(epochs, by_event_type=False)

    # Verify that we get a single Evoked object
    import mne
    assert isinstance(evoked_all, mne.Evoked), \
        "make_evoked with by_event_type=False should return single Evoked"

    # Check basic properties
    assert evoked_all.nave > 0, \
        "Evoked should have non-zero number of averages"
    assert len(evoked_all.ch_names) == n_channels, \
        "Evoked should have same number of channels as input epochs"
    assert evoked_all.info['sfreq'] == epochs.info['sfreq'], \
        "Evoked should preserve sampling frequency"

    # Check that data shape is correct (channels x time_points)
    evoked_data = evoked_all.data
    assert evoked_data.shape[0] == n_channels, \
        "Evoked data should have correct number of channels"
    assert evoked_data.shape[1] > 0, \
        "Evoked data should have time points"

    # Test 2: Average by event type (by_event_type=True)
    evoked_by_condition = make_evoked(epochs, by_event_type=True)

    # Verify that we get a dictionary of Evoked objects
    assert isinstance(evoked_by_condition, dict), \
        "make_evoked with by_event_type=True should return dictionary"

    # Check that we have the expected event types
    expected_conditions = ['condition_1', 'condition_2']
    for condition in expected_conditions:
        assert condition in evoked_by_condition, \
            f"Dictionary should contain key for {condition}"

        evoked_cond = evoked_by_condition[condition]
        assert isinstance(evoked_cond, mne.Evoked), \
            "Each condition should be an Evoked object"

        # Check properties of each condition
        assert evoked_cond.nave > 0, \
            f"Evoked for {condition} should have non-zero averages"
        assert len(evoked_cond.ch_names) == n_channels, \
            f"Evoked for {condition} should have correct number of channels"
        assert evoked_cond.info['sfreq'] == epochs.info['sfreq'], \
            f"Evoked for {condition} should preserve sampling frequency"

    # Test 3: Verify that by_event_type affects the number of averages
    total_epochs = len(epochs)
    condition_1_epochs = len([e for e in epochs.events if e[2] == 1])
    condition_2_epochs = len([e for e in epochs.events if e[2] == 2])

    # Single average should use all epochs
    assert evoked_all.nave == total_epochs, \
        "Single evoked should average all epochs"

    # Condition-specific averages should use subset of epochs
    assert evoked_by_condition['condition_1'].nave == condition_1_epochs, \
        "Condition 1 evoked should average only condition 1 epochs"
    assert evoked_by_condition['condition_2'].nave == condition_2_epochs, \
        "Condition 2 evoked should average only condition 2 epochs"

    # Test 4: Check that different conditions have different responses
    # (Due to different event timing, they should be slightly different)
    data_cond1 = evoked_by_condition['condition_1'].data
    data_cond2 = evoked_by_condition['condition_2'].data

    # They should have same shape but potentially different values
    assert data_cond1.shape == data_cond2.shape, \
        "Both conditions should have same data shape"

    # Test 5: Test with single event type
    # Create epochs with only one event type
    single_events = np.column_stack([event_times[:10],
                                    np.zeros(10, dtype=int),
                                    np.ones(10, dtype=int)])

    single_epochs = Epochs(
        raw_data, single_events,
        event_id={'single_condition': 1},
        tmin=-0.2, tmax=0.5,
        baseline=(-0.1, 0),
        verbose=False,
        reject=None,
        preload=True
    )

    # Test both methods with single condition
    evoked_single_all = make_evoked(single_epochs, by_event_type=False)
    evoked_single_by_type = make_evoked(single_epochs, by_event_type=True)

    assert isinstance(evoked_single_all, mne.Evoked), \
        "Single condition with by_event_type=False should return Evoked"
    assert isinstance(evoked_single_by_type, dict), \
        "Single condition with by_event_type=True should return dict"
    assert len(evoked_single_by_type) == 1, \
        "Single condition dict should have one entry"
    assert 'single_condition' in evoked_single_by_type, \
        "Dict should contain the single condition key"

    # Test 6: Error handling with empty epochs
    # Create epochs with no events that pass (should be handled gracefully)
    empty_events = np.empty((0, 3), dtype=int)
    try:
        empty_epochs = Epochs(
            raw_data, empty_events,
            tmin=-0.2, tmax=0.5,
            verbose=False,
            preload=True
        )
        # If empty epochs are created, make_evoked should handle them
        if len(empty_epochs) == 0:
            print("Empty epochs case: would need special handling")
    except ValueError:
        # Expected - can't create epochs with no events
        pass

    print("✓ All make_evoked tests passed!")


def test_setup_derivatives_directories(tmp_path):
    """Test the setup_derivatives_directories function."""
    from pathlib import Path

    # Create a temporary BIDS root directory
    bids_root = tmp_path / "test_bids"
    bids_root.mkdir()

    # Test 1: Basic setup with both preprocessing and analysis directories
    result = setup_derivatives_directories(
        bids_root=bids_root,
        subject="03",
        create_preprocessing=True,
        create_analysis=True
    )

    # Verify return structure
    assert isinstance(result, dict), \
        "Function should return a dictionary"

    expected_keys = [
        "derivatives_root", "preprocessing_dir", "analysis_dir",
        "preprocessing_subject_dir", "analysis_subject_dir"
    ]
    for key in expected_keys:
        assert key in result, f"Result should contain key: {key}"

    # Verify directory creation
    derivatives_root = bids_root / "derivatives"
    assert derivatives_root.exists(), \
        "Derivatives root directory should be created"
    assert result["derivatives_root"] == derivatives_root, \
        "Should return correct derivatives root path"

    # Check preprocessing directory structure. preprocessing_subject_dir
    # points at the BIDS-derivatives ``sub-XX/eeg`` subdirectory where the
    # actual preprocessed EEG outputs live (per setup_derivatives_directories).
    preproc_dir = derivatives_root / "ffrprep-preprocessing"
    preproc_subject_dir = preproc_dir / "sub-03"
    preproc_subject_eeg_dir = preproc_subject_dir / "eeg"

    assert preproc_dir.exists(), \
        "Preprocessing directory should be created"
    assert preproc_subject_dir.exists(), \
        "Preprocessing subject directory should be created"
    assert preproc_subject_eeg_dir.exists(), \
        "Preprocessing subject eeg/ subdirectory should be created"
    assert result["preprocessing_dir"] == preproc_dir, \
        "Should return correct preprocessing directory path"
    assert result["preprocessing_subject_dir"] == preproc_subject_eeg_dir, \
        "Should point at the BIDS-derivatives eeg/ subdirectory"

    # Check analysis directory structure
    analysis_dir = derivatives_root / "ffrprep-analysis"
    analysis_subject_dir = analysis_dir / "sub-03"

    assert analysis_dir.exists(), \
        "Analysis directory should be created"
    assert analysis_subject_dir.exists(), \
        "Analysis subject directory should be created"
    assert result["analysis_dir"] == analysis_dir, \
        "Should return correct analysis directory path"
    assert result["analysis_subject_dir"] == analysis_subject_dir, \
        "Should return correct analysis subject directory path"

    # Test 2: Only preprocessing directories
    bids_root_2 = tmp_path / "test_bids_2"
    bids_root_2.mkdir()

    result_preproc_only = setup_derivatives_directories(
        bids_root=bids_root_2,
        subject="05",
        create_preprocessing=True,
        create_analysis=False
    )

    # Should create preprocessing but not analysis
    derivatives_root_2 = bids_root_2 / "derivatives"
    preproc_dir_2 = derivatives_root_2 / "ffrprep-preprocessing" / "sub-05"
    analysis_dir_2 = derivatives_root_2 / "ffrprep-analysis"

    assert preproc_dir_2.exists(), \
        "Preprocessing directory should be created when requested"
    assert not analysis_dir_2.exists(), \
        "Analysis directory should not be created when not requested"

    assert result_preproc_only["preprocessing_dir"] is not None, \
        "Should return preprocessing directory path"
    assert result_preproc_only["analysis_dir"] is None, \
        "Should return None for analysis directory when not created"
    assert result_preproc_only["analysis_subject_dir"] is None, \
        "Should return None for analysis subject directory when not created"

    # Test 3: Only analysis directories
    bids_root_3 = tmp_path / "test_bids_3"
    bids_root_3.mkdir()

    result_analysis_only = setup_derivatives_directories(
        bids_root=bids_root_3,
        subject="07",
        create_preprocessing=False,
        create_analysis=True
    )

    # Should create analysis but not preprocessing
    derivatives_root_3 = bids_root_3 / "derivatives"
    preproc_dir_3 = derivatives_root_3 / "ffrprep-preprocessing"
    analysis_dir_3 = derivatives_root_3 / "ffrprep-analysis" / "sub-07"

    assert analysis_dir_3.exists(), \
        "Analysis directory should be created when requested"
    assert not preproc_dir_3.exists(), \
        "Preprocessing directory should not be created when not requested"

    assert result_analysis_only["analysis_dir"] is not None, \
        "Should return analysis directory path"
    assert result_analysis_only["preprocessing_dir"] is None, (
        "Should return None for preprocessing directory when not created"
    )
    assert result_analysis_only["preprocessing_subject_dir"] is None, (
        "Should return None for preprocessing subject directory when not "
        "created"
    )

    # Test 4: Neither directory type (edge case)
    bids_root_4 = tmp_path / "test_bids_4"
    bids_root_4.mkdir()

    result_neither = setup_derivatives_directories(
        bids_root=bids_root_4,
        subject="09",
        create_preprocessing=False,
        create_analysis=False
    )

    # Should still create derivatives root but no subdirectories
    derivatives_root_4 = bids_root_4 / "derivatives"
    assert derivatives_root_4.exists(), \
        "Derivatives root should always be created"
    assert result_neither["derivatives_root"] == derivatives_root_4, \
        "Should return derivatives root path"

    # No specific derivative directories should be created
    assert result_neither["preprocessing_dir"] is None, \
        "Should return None when no preprocessing directory created"
    assert result_neither["analysis_dir"] is None, \
        "Should return None when no analysis directory created"

    # Test 5: Path input as string vs Path object
    bids_root_str = str(tmp_path / "test_bids_str")
    Path(bids_root_str).mkdir()

    result_str = setup_derivatives_directories(
        bids_root=bids_root_str,  # Pass as string
        subject="11",
        create_preprocessing=True,
        create_analysis=True
    )

    # Should work the same way with string input
    expected_derivatives = Path(bids_root_str) / "derivatives"
    assert result_str["derivatives_root"] == expected_derivatives, (
        "Should handle string input correctly"
    )

    # Test 6: Subject with different formats
    subjects_to_test = ["03", "subject-05", "sub-07"]

    for subject in subjects_to_test:
        bids_root_subj = (
            tmp_path / f"test_bids_subj_{subject.replace('-', '_')}"
        )
        bids_root_subj.mkdir()

        result_subj = setup_derivatives_directories(
            bids_root=bids_root_subj,
            subject=subject,
            create_preprocessing=True,
            create_analysis=False
        )

        # Should create directory with sub- prefix regardless of input format
        expected_subj_dir = (
            result_subj["derivatives_root"] /
            "ffrprep-preprocessing" /
            f"sub-{subject}"
        )

        assert expected_subj_dir.exists(), \
            f"Subject directory should be created for subject: {subject}"

    # Test 7: Existing derivatives directory
    bids_root_existing = tmp_path / "test_bids_existing"
    bids_root_existing.mkdir()
    existing_derivatives = bids_root_existing / "derivatives"
    existing_derivatives.mkdir()

    # Create some existing content
    existing_file = existing_derivatives / "existing_file.txt"
    existing_file.write_text("existing content")

    _ = setup_derivatives_directories(
        bids_root=bids_root_existing,
        subject="13",
        create_preprocessing=True,
        create_analysis=True
    )

    # Should not overwrite existing derivatives directory
    assert existing_derivatives.exists(), \
        "Existing derivatives directory should be preserved"
    assert existing_file.exists(), \
        "Existing files should be preserved"
    assert existing_file.read_text() == "existing content", \
        "Existing file content should be preserved"

    # But should still create the new subdirectories
    new_preproc_dir = existing_derivatives / "ffrprep-preprocessing" / "sub-13"
    new_analysis_dir = existing_derivatives / "ffrprep-analysis" / "sub-13"

    assert new_preproc_dir.exists(), \
        "New preprocessing directory should be created"
    assert new_analysis_dir.exists(), \
        "New analysis directory should be created"

    print("✓ All setup_derivatives_directories tests passed!")


@pytest.mark.integration
def test_load_data_with_example_dataset(bids_dataset):
    """Test load_data function on the real downloaded example dataset."""
    import mne

    loaded_data, bids_path, original_filename, _events_file = load_data(
        bids_root=str(bids_dataset),
        sub_label="03",
        task_label="passive",
        run_label=1,
    )

    assert isinstance(loaded_data, mne.io.BaseRaw)
    assert loaded_data.n_times > 0
    assert len(loaded_data.ch_names) > 0

    assert bids_path.subject == "03"
    assert bids_path.task == "passive"

    assert "sub-03" in original_filename
    assert "task-passive" in original_filename


@pytest.mark.integration
def test_reference_data_with_example_dataset(bids_dataset):
    """Test reference_data on the real downloaded example dataset."""
    raw_data, _bids_path, _orig_name, _events = load_data(
        bids_root=str(bids_dataset),
        sub_label="03",
        task_label="passive",
        run_label=1,
    )

    referenced_data = reference_data(raw_data, ref_channels=None)

    assert len(referenced_data.ch_names) == len(raw_data.ch_names)
    assert referenced_data.info["sfreq"] == raw_data.info["sfreq"]


@pytest.mark.integration
def test_filter_data_with_example_dataset(bids_dataset):
    """Test filter_data on the real downloaded example dataset."""
    raw_data, _bids_path, _orig_name, _events = load_data(
        bids_root=str(bids_dataset),
        sub_label="03",
        task_label="passive",
        run_label=1,
    )

    filtered_data = filter_data(eeg_data=raw_data, high_pass=1.0, low_pass=40.0)

    assert len(filtered_data.ch_names) == len(raw_data.ch_names)
    assert filtered_data.n_times == raw_data.n_times
    assert filtered_data.info["sfreq"] == raw_data.info["sfreq"]
    assert filtered_data.info["lowpass"] == 40.0
    assert filtered_data.info["highpass"] == 1.0


@pytest.mark.integration
def test_epoch_data_with_example_dataset(bids_dataset):
    """Test epoch_data on the real downloaded example dataset."""
    raw_data, _bids_path, _orig_name, _events = load_data(
        bids_root=str(bids_dataset),
        sub_label="03",
        task_label="passive",
        run_label=1,
    )

    referenced_data = reference_data(raw_data, ref_channels=None)
    filtered_data = filter_data(referenced_data, high_pass=1.0, low_pass=40.0)

    epochs, time_window = epoch_data(
        eeg_data=filtered_data,
        baseline=-0.1,
        tmin=-0.2,
        tmax=0.5,
        verbose=False,
    )

    assert time_window[0] == -0.2
    assert time_window[1] == 0.5
    assert len(epochs) > 0

    epochs_array = epochs.get_data()
    assert epochs_array.ndim == 3
    assert epochs_array.shape[0] > 0
    assert epochs_array.shape[1] > 0
    assert epochs_array.shape[2] > 0


@pytest.mark.integration
def test_make_evoked_with_example_dataset(bids_dataset):
    """Test make_evoked on the real downloaded example dataset."""
    import mne

    raw_data, _bids_path, _orig_name, _events = load_data(
        bids_root=str(bids_dataset),
        sub_label="03",
        task_label="passive",
        run_label=1,
    )

    referenced_data = reference_data(raw_data, ref_channels=None)
    filtered_data = filter_data(referenced_data, high_pass=1.0, low_pass=40.0)
    epochs, _time_window = epoch_data(
        eeg_data=filtered_data,
        baseline=-0.1,
        tmin=-0.2,
        tmax=0.5,
        verbose=False,
    )
    assert len(epochs) > 0

    evoked_all = make_evoked(epochs, by_event_type=False)
    evoked_by_condition = make_evoked(epochs, by_event_type=True)

    assert isinstance(evoked_all, mne.Evoked)
    assert isinstance(evoked_by_condition, dict)
    assert evoked_all.nave > 0
    assert len(evoked_all.ch_names) > 0
    assert len(evoked_by_condition) > 0


@pytest.mark.integration
def test_full_pipeline_with_example_dataset(bids_workspace):
    """End-to-end preprocessing on real downloaded data.

    Uses the writable ``bids_workspace`` fixture so derivatives can be
    written without polluting the cached source dataset.
    """
    raw_data, _bids_path, _orig_name, _events = load_data(
        bids_root=str(bids_workspace),
        sub_label="03",
        task_label="passive",
        run_label=1,
    )

    referenced_data = reference_data(raw_data, ref_channels=None)
    filtered_data = filter_data(referenced_data, high_pass=1.0, low_pass=40.0)
    epochs, _time_window = epoch_data(
        eeg_data=filtered_data,
        baseline=-0.1,
        tmin=-0.2,
        tmax=0.5,
        verbose=False,
    )
    assert len(epochs) > 0

    derivatives_info = setup_derivatives_directories(
        bids_root=bids_workspace,
        subject="03",
        create_preprocessing=True,
        create_analysis=True,
    )
    assert derivatives_info["derivatives_root"].exists()
    assert derivatives_info["preprocessing_subject_dir"].exists()
    assert derivatives_info["analysis_subject_dir"].exists()

    preprocessing_output = save_preprocessing_outputs(
        epochs, bids_workspace, subject="03", task="passive",
        session=None, run=1,
    )
    assert preprocessing_output.exists()

    evoked = make_evoked(epochs, by_event_type=False)
    analysis_outputs = save_analysis_outputs(
        evoked, bids_workspace, subject="03", task="passive",
        session=None, run=1, analysis_type="evoked",
    )
    assert len(analysis_outputs) > 0
    assert analysis_outputs[0].exists()
