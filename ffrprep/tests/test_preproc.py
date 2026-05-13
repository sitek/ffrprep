import json
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

    # Write the data to BIDS format. Failures here propagate so a real
    # BIDS-write regression surfaces immediately.
    bids_path = BIDSPath(
        subject="03", task="passive", run=1,
        root=bids_root, datatype="eeg",
    )
    write_raw_bids(
        raw=raw_from_file,
        bids_path=bids_path,
        overwrite=True,
        verbose=False,
    )

    # load_data returns (raw, bids_path, original_filename, events_file)
    loaded_data, returned_bids_path, original_filename, _events_file = load_data(
        bids_root=str(bids_root),
        sub_label="03",
        task_label="passive",
        run_label=1,
    )

    import mne
    assert isinstance(loaded_data, mne.io.BaseRaw)
    assert returned_bids_path is not None
    assert original_filename is not None

    # Data properties
    assert len(loaded_data.ch_names) == n_channels
    assert loaded_data.n_times == n_times
    assert loaded_data.info["sfreq"] == sfreq

    ch_types_loaded = loaded_data.get_channel_types()
    assert len(ch_types_loaded) > 0
    assert all(ch_type == "eeg" for ch_type in ch_types_loaded)

    # Data is non-empty and contains the synthetic signal
    loaded_data_array = loaded_data.get_data()
    assert loaded_data_array.size > 0
    assert not (loaded_data_array == 0).all()

    # BIDS path properties. pybids returns ``run`` as a string ("1"),
    # not an int, in current versions.
    assert returned_bids_path.subject == "03"
    assert returned_bids_path.task == "passive"
    assert str(returned_bids_path.run) == "1"

    assert "sub-03" in original_filename
    assert "task-passive" in original_filename
    # pybids no longer zero-pads run identifiers in filenames.
    assert "run-1" in original_filename


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

    # Test 1: Non-existent subject — load_data should raise
    # FileNotFoundError with a helpful message.
    with pytest.raises(FileNotFoundError, match="No EEG files found"):
        load_data(
            bids_root=str(bids_root),
            sub_label="999",
            task_label="passive",
            run_label=1,
        )

    # Test 2: Non-existent BIDS root — pybids/mne_bids surfaces this as
    # one of FileNotFoundError, OSError, or ValueError depending on
    # which layer hits it first.
    with pytest.raises((FileNotFoundError, OSError, ValueError)):
        load_data(
            bids_root="/nonexistent/path",
            sub_label="03",
            task_label="passive",
            run_label=1,
        )


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

    # A non-existent reference channel must surface as ValueError or
    # KeyError from MNE's channel resolution.
    with pytest.raises((ValueError, KeyError)):
        reference_data(data, ref_channels="fake_channel")


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

    # Test 5: With external events file (simulate TSV file). Use tmp_path
    # so the file is cleaned up automatically — no need for unlink.
    import pandas as pd

    events_file = tmp_path / "events.tsv"
    pd.DataFrame({
        "onset": [1.0, 2.0, 3.0, 4.0, 5.0],  # Event times in seconds
        "duration": [0.1, 0.1, 0.1, 0.1, 0.1],
        "trial_type": [1, 2, 1, 2, 1],
    }).to_csv(events_file, sep="\t", index=False)

    epochs5, _time_window5 = epoch_data(
        filtered_data, baseline=-0.1, events_file=str(events_file),
        tmin=-0.2, tmax=0.5, verbose=False,
    )
    assert epochs5 is not None
    if len(epochs5) > 0:
        assert hasattr(epochs5, "event_id")

    # Test 6: Invalid baseline must raise TypeError or ValueError
    with pytest.raises((TypeError, ValueError)):
        epoch_data(filtered_data, baseline="invalid")

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


def _make_raw_with_string_trial_types(tmp_path):
    """Synthetic Raw + events.tsv whose trial_type column is string-valued.

    Three trial types ("Pos", "Neg", "Other") with two onsets each
    (six events total). Used by the trial-types subset tests below to
    exercise the new ``trial_types=`` parameter on :func:`epoch_data`.
    Returns ``(raw, events_path)``.
    """
    import numpy as np
    import pandas as pd
    from mne import create_info
    from mne.io import RawArray

    sfreq = 1000.0
    n_times = 12000
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, n_times)) * 1e-6
    info = create_info(
        ch_names=[f"EEG{i:03d}" for i in range(4)],
        sfreq=sfreq,
        ch_types=["eeg"] * 4,
    )
    raw = RawArray(data, info)

    onsets = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    trial_types = ["Pos", "Neg", "Other", "Pos", "Neg", "Other"]
    events_path = tmp_path / "sub-01_task-active_events.tsv"
    pd.DataFrame({
        "onset": onsets,
        "duration": [0.1] * len(onsets),
        "trial_type": trial_types,
    }).to_csv(events_path, sep="\t", index=False)
    return raw, events_path


def test_epoch_data_trial_types_none_keeps_all_types(tmp_path):
    """Default ``trial_types=None`` keeps every trial type from events.tsv."""
    raw, events_path = _make_raw_with_string_trial_types(tmp_path)
    epochs, _ = epoch_data(
        raw, baseline=[-0.05, 0.0],
        events_file=str(events_path),
        tmin=-0.05, tmax=0.1, verbose=False,
    )
    assert set(epochs.event_id.keys()) == {"Pos", "Neg", "Other"}


def test_epoch_data_trial_types_filters_to_subset(tmp_path):
    """A subset list narrows event_id to exactly that subset."""
    raw, events_path = _make_raw_with_string_trial_types(tmp_path)
    epochs, _ = epoch_data(
        raw, baseline=[-0.05, 0.0],
        events_file=str(events_path),
        tmin=-0.05, tmax=0.1, verbose=False,
        trial_types=["Pos", "Neg"],
    )
    assert set(epochs.event_id.keys()) == {"Pos", "Neg"}
    # Each retained type contributed two onsets -> four epochs total.
    assert len(epochs) == 4


def test_epoch_data_trial_types_single_value(tmp_path):
    """A single-element list yields a single-condition Epochs object."""
    raw, events_path = _make_raw_with_string_trial_types(tmp_path)
    epochs, _ = epoch_data(
        raw, baseline=[-0.05, 0.0],
        events_file=str(events_path),
        tmin=-0.05, tmax=0.1, verbose=False,
        trial_types=["Pos"],
    )
    assert set(epochs.event_id.keys()) == {"Pos"}
    assert len(epochs) == 2


def test_epoch_data_trial_types_unknown_raises(tmp_path):
    """An unknown trial-type name raises ValueError (no silent empty Epochs)."""
    raw, events_path = _make_raw_with_string_trial_types(tmp_path)
    with pytest.raises(ValueError, match="trial_types"):
        epoch_data(
            raw, baseline=[-0.05, 0.0],
            events_file=str(events_path),
            tmin=-0.05, tmax=0.1, verbose=False,
            trial_types=["Nonexistent"],
        )


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

    # Run the workflow. Failures propagate so workflow regressions surface
    # immediately rather than getting masked as "environment issues".
    result = workflow.run()
    assert result is not None

    # save_preprocessing_outputs derives derivatives from ``bids_root``,
    # not from the workflow's output_dir input — outputs land under the
    # downloaded dataset's own derivatives/ subdirectory.
    from pathlib import Path

    derivatives_dir = (
        Path(data_path) / "derivatives" / "ffrprep-preprocessing" / "sub-03" / "eeg"
    )
    assert derivatives_dir.exists()
    # Glob matches both the bare _desc-preproc_epo.fif (split=False) and
    # the per-trial-type _desc-preproc{Cond}_epo.fif (split=True default)
    # naming, so this smoke test stays robust to the split-by-trial-type
    # default.
    output_files = list(derivatives_dir.glob("*desc-preproc*_epo.fif"))
    assert len(output_files) > 0


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


def test_save_preprocessing_outputs_persists_rejection_metadata(tmp_path):
    """Sidecar must carry pre-rejection counts + reject thresholds.

    Without these, downstream consumers (the report builder) have to
    back-calculate the rejected count from ``events.tsv`` row count
    minus the saved ``EpochCount``. That back-calc breaks whenever the
    epoching uses an event-id filter, when events.tsv is missing, or
    when the user is just inspecting the sidecar by hand. Persisting
    the metadata at save time makes the sidecar self-describing.
    """
    import numpy as np
    from mne import Epochs, create_info
    from mne.io import RawArray

    # Synthesize 6 s of clean data on 4 channels at 1 kHz, then inject
    # a high-amplitude excursion in one channel during the window of
    # the third event so amplitude-based rejection drops exactly that
    # epoch.
    sfreq = 1000.0
    n_channels = 4
    n_times = int(sfreq * 6)
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1e-6, size=(n_channels, n_times))
    spike_start = int(sfreq * 3.0)
    data[0, spike_start:spike_start + 100] = 1e-3  # 1 mV >> 75 µV reject

    info = create_info(
        ch_names=["Cz", "F3", "F4", "Pz"],
        sfreq=sfreq,
        ch_types=["eeg"] * n_channels,
    )
    raw = RawArray(data, info, verbose=False)

    n_events = 5
    event_samples = np.linspace(500, n_times - 500, n_events, dtype=int)
    events = np.column_stack([
        event_samples,
        np.zeros(n_events, dtype=int),
        np.ones(n_events, dtype=int),
    ])

    reject_thresholds = {"eeg": 7.5e-5}
    epochs = Epochs(
        raw, events, tmin=-0.04, tmax=0.4, baseline=None,
        reject=reject_thresholds, preload=True, verbose=False,
    )
    epochs.drop_bad()

    n_total = len(epochs.drop_log)
    n_accepted = len(epochs)
    n_rejected = n_total - n_accepted
    assert n_rejected >= 1, (
        "fixture didn't actually reject anything — adjust the spike "
        "or the threshold"
    )

    bids_root = tmp_path / "bids"
    bids_root.mkdir()
    out_path = save_preprocessing_outputs(
        epochs, bids_root, subject="01", task="active", run=1,
    )

    sidecar_path = out_path.with_suffix(".json")
    with open(sidecar_path) as f:
        sidecar = json.load(f)

    assert sidecar["EpochCount"] == n_accepted
    assert sidecar["EpochCountTotal"] == n_total, \
        "sidecar must record the pre-rejection epoch count"
    assert sidecar["EpochCountRejected"] == n_rejected, \
        "sidecar must record the count of rejected epochs"
    assert sidecar["RejectionThresholds"] == {"eeg": 7.5e-5}, \
        "sidecar must record the reject thresholds that were applied"


def test_save_preprocessing_outputs_persists_baseline(tmp_path):
    """The preproc sidecar must record the baseline window from the input Epochs.

    save_preprocessing_outputs reconstructs the input as
    ``mne.EpochsArray`` before writing the .fif (to avoid round-trip
    edge cases), and the EpochsArray constructor doesn't carry a
    baseline argument, so the on-disk Epochs has ``baseline=None``
    when reloaded. Without persisting the window in the BIDS sidecar,
    downstream consumers (the analysis worker) can't restore
    ``epochs.baseline`` post-load, and the chain leading to the
    analysis report's RMS SNR stays broken.
    """
    import numpy as np
    from mne import Epochs, create_info
    from mne.io import RawArray

    sfreq = 1000.0
    n_channels = 2
    n_times = int(sfreq * 5)
    rng = np.random.default_rng(4)
    data = rng.normal(0, 1e-6, size=(n_channels, n_times))
    info = create_info(["Cz", "F3"], sfreq=sfreq, ch_types=["eeg"] * 2)
    raw = RawArray(data, info, verbose=False)

    n_events = 3
    event_samples = np.linspace(500, n_times - 500, n_events, dtype=int)
    events = np.column_stack([
        event_samples,
        np.zeros(n_events, dtype=int),
        np.ones(n_events, dtype=int),
    ])

    baseline = (-0.04, 0.0)
    epochs = Epochs(
        raw, events, tmin=-0.04, tmax=0.4, baseline=baseline,
        preload=True, verbose=False,
    )
    assert epochs.baseline == baseline, (
        "fixture: Epochs constructor should carry baseline metadata"
    )

    bids_root = tmp_path / "bids"
    bids_root.mkdir()

    out_path = save_preprocessing_outputs(
        epochs, bids_root, subject="01", task="active", run=1,
    )

    sidecar_path = out_path.with_suffix(".json")
    with open(sidecar_path) as f:
        sidecar = json.load(f)

    assert "Baseline" in sidecar, (
        "preprocessing sidecar must record the baseline window from "
        "epochs.baseline; otherwise the analysis worker has nothing "
        "to restore from after load"
    )
    assert tuple(sidecar["Baseline"]) == baseline


def _two_condition_epochs_dict(tmp_path):
    """Build a ``{condition: Epochs}`` dict for two trial types.

    Returns ``({"Pos": epochs_pos, "Neg": epochs_neg}, bids_root)``.
    Two trial types ("Pos", "Neg") with three events each. Used by the
    per-condition save tests below.
    """
    import numpy as np
    from mne import Epochs, create_info
    from mne.io import RawArray

    sfreq = 1000.0
    n_channels = 2
    n_times = int(sfreq * 8)
    rng = np.random.default_rng(7)
    data = rng.normal(0, 1e-6, size=(n_channels, n_times))
    info = create_info(["Cz", "F3"], sfreq=sfreq, ch_types=["eeg"] * 2)
    raw = RawArray(data, info, verbose=False)

    onsets = [1000, 2000, 3000, 4000, 5000, 6000]
    codes = [1, 2, 1, 2, 1, 2]
    events = np.column_stack([
        np.array(onsets, dtype=int),
        np.zeros(len(onsets), dtype=int),
        np.array(codes, dtype=int),
    ])
    event_id = {"Pos": 1, "Neg": 2}
    epochs = Epochs(
        raw, events, event_id=event_id,
        tmin=-0.04, tmax=0.4, baseline=(-0.04, 0.0),
        preload=True, verbose=False,
    )
    bids_root = tmp_path / "bids"
    bids_root.mkdir()
    return {"Pos": epochs["Pos"], "Neg": epochs["Neg"]}, bids_root


def test_save_preprocessing_outputs_dict_returns_list_of_paths(tmp_path):
    """Dict input → list of Paths, one per condition."""
    epochs_dict, bids_root = _two_condition_epochs_dict(tmp_path)
    out_paths = save_preprocessing_outputs(
        epochs_dict, bids_root, subject="01", task="active", run=1,
    )
    assert isinstance(out_paths, list), (
        "dict input must yield a list of paths (one per condition)"
    )
    assert len(out_paths) == 2


def test_save_preprocessing_outputs_dict_writes_per_condition_files(tmp_path):
    """Each per-condition file uses the ``_desc-preproc{Cond}_epo.fif`` pattern."""
    epochs_dict, bids_root = _two_condition_epochs_dict(tmp_path)
    out_paths = save_preprocessing_outputs(
        epochs_dict, bids_root, subject="01", task="active", run=1,
    )
    names = sorted(p.name for p in out_paths)
    assert names == [
        "sub-01_task-active_run-1_desc-preprocNeg_epo.fif",
        "sub-01_task-active_run-1_desc-preprocPos_epo.fif",
    ]
    for p in out_paths:
        assert p.exists(), f"per-condition .fif must exist: {p}"


def test_save_preprocessing_outputs_per_condition_sidecar_has_condition(tmp_path):
    """Each per-condition sidecar carries a ``Condition: <name>`` field."""
    epochs_dict, bids_root = _two_condition_epochs_dict(tmp_path)
    out_paths = save_preprocessing_outputs(
        epochs_dict, bids_root, subject="01", task="active", run=1,
    )
    seen = {}
    for p in out_paths:
        with open(p.with_suffix(".json")) as f:
            sidecar = json.load(f)
        assert "Condition" in sidecar, (
            f"per-condition sidecar must record the trial type: {p.name}"
        )
        seen[sidecar["Condition"]] = p.name
    assert set(seen.keys()) == {"Pos", "Neg"}


def test_save_preprocessing_outputs_scalar_input_unchanged(tmp_path):
    """Scalar Epochs input keeps today's behavior: single Path return."""
    from pathlib import Path

    epochs_dict, bids_root = _two_condition_epochs_dict(tmp_path)
    # Stack the two condition slices back into a single Epochs to feed
    # the scalar code path, exercising the unchanged contract.
    import mne

    combined = mne.concatenate_epochs(
        [epochs_dict["Pos"], epochs_dict["Neg"]],
    )
    out_path = save_preprocessing_outputs(
        combined, bids_root, subject="01", task="active", run=2,
    )
    assert isinstance(out_path, Path), (
        "scalar Epochs input must keep returning a single Path"
    )
    assert out_path.name == "sub-01_task-active_run-2_desc-preproc_epo.fif"


def _two_condition_epochs_combined(tmp_path):
    """Build a single multi-event Epochs object + bids_root for split tests.

    Mirrors :func:`_two_condition_epochs_dict` but returns the combined
    Epochs (event_id={"Pos": 1, "Neg": 2}) instead of pre-split slices,
    so the splitting logic itself is exercised by the test.
    """
    import numpy as np
    from mne import Epochs, create_info
    from mne.io import RawArray

    sfreq = 1000.0
    n_channels = 2
    n_times = int(sfreq * 8)
    rng = np.random.default_rng(17)
    data = rng.normal(0, 1e-6, size=(n_channels, n_times))
    info = create_info(["Cz", "F3"], sfreq=sfreq, ch_types=["eeg"] * 2)
    raw = RawArray(data, info, verbose=False)

    onsets = [1000, 2000, 3000, 4000, 5000, 6000]
    codes = [1, 2, 1, 2, 1, 2]
    events = np.column_stack([
        np.array(onsets, dtype=int),
        np.zeros(len(onsets), dtype=int),
        np.array(codes, dtype=int),
    ])
    epochs = Epochs(
        raw, events, event_id={"Pos": 1, "Neg": 2},
        tmin=-0.04, tmax=0.4, baseline=(-0.04, 0.0),
        preload=True, verbose=False,
    )
    bids_root = tmp_path / "bids"
    bids_root.mkdir()
    return epochs, bids_root


def test_save_preprocessing_node_split_writes_per_condition_files(tmp_path):
    """``split_by_trial_type=True`` splits the Epochs and writes per-cond files."""
    from pathlib import Path

    from ffrprep.preproc import save_preprocessing_node

    epochs, bids_root = _two_condition_epochs_combined(tmp_path)
    output_paths = save_preprocessing_node(
        epochs, str(bids_root), "01", task="active", run=1,
        split_by_trial_type=True,
    )
    assert isinstance(output_paths, list)
    names = sorted(Path(p).name for p in output_paths)
    assert names == [
        "sub-01_task-active_run-1_desc-preprocNeg_epo.fif",
        "sub-01_task-active_run-1_desc-preprocPos_epo.fif",
    ]
    for p in output_paths:
        assert Path(p).exists()


def test_save_preprocessing_node_no_split_writes_bare_file(tmp_path):
    """``split_by_trial_type=False`` writes one bare ``_desc-preproc_epo.fif``."""
    from pathlib import Path

    from ffrprep.preproc import save_preprocessing_node

    epochs, bids_root = _two_condition_epochs_combined(tmp_path)
    output_path = save_preprocessing_node(
        epochs, str(bids_root), "01", task="active", run=1,
        split_by_trial_type=False,
    )
    assert isinstance(output_path, str)
    assert Path(output_path).name == "sub-01_task-active_run-1_desc-preproc_epo.fif"


def test_save_preprocessing_node_split_default_true(tmp_path):
    """Default ``split_by_trial_type=True`` matches the CLI default."""
    from ffrprep.preproc import save_preprocessing_node

    epochs, bids_root = _two_condition_epochs_combined(tmp_path)
    output_paths = save_preprocessing_node(
        epochs, str(bids_root), "01", task="active", run=1,
    )
    assert isinstance(output_paths, list), (
        "default save_preprocessing_node must split (matches CLI default)"
    )
    assert len(output_paths) == 2


def test_save_preprocessing_node_split_single_condition(tmp_path):
    """A single-event Epochs still emits a per-condition file under split=True."""
    from pathlib import Path

    import numpy as np
    from mne import Epochs, create_info
    from mne.io import RawArray

    from ffrprep.preproc import save_preprocessing_node

    sfreq = 1000.0
    rng = np.random.default_rng(19)
    data = rng.normal(0, 1e-6, size=(2, int(sfreq * 5)))
    info = create_info(["Cz", "F3"], sfreq=sfreq, ch_types=["eeg"] * 2)
    raw = RawArray(data, info, verbose=False)
    events = np.column_stack([
        np.array([1000, 2000, 3000], dtype=int),
        np.zeros(3, dtype=int),
        np.ones(3, dtype=int),
    ])
    epochs = Epochs(
        raw, events, event_id={"Pos": 1},
        tmin=-0.04, tmax=0.4, baseline=(-0.04, 0.0),
        preload=True, verbose=False,
    )
    bids_root = tmp_path / "bids"
    bids_root.mkdir()
    output_paths = save_preprocessing_node(
        epochs, str(bids_root), "01", task="active", run=1,
        split_by_trial_type=True,
    )
    assert isinstance(output_paths, list)
    assert len(output_paths) == 1
    assert Path(output_paths[0]).name == (
        "sub-01_task-active_run-1_desc-preprocPos_epo.fif"
    )


def _two_evoked_dict_for_save_analysis(tmp_path):
    """Build ``({Pos, Neg} evoked dict, bids_root)`` for the analysis tests."""
    import numpy as np
    from mne import Epochs, create_info
    from mne.io import RawArray

    sfreq = 1000.0
    rng = np.random.default_rng(13)
    data = rng.normal(0, 1e-6, size=(2, int(sfreq * 8)))
    info = create_info(["Cz", "F3"], sfreq=sfreq, ch_types=["eeg"] * 2)
    raw = RawArray(data, info, verbose=False)
    onsets = [1000, 2000, 3000, 4000, 5000, 6000]
    codes = [1, 2, 1, 2, 1, 2]
    events = np.column_stack([
        np.array(onsets, dtype=int),
        np.zeros(len(onsets), dtype=int),
        np.array(codes, dtype=int),
    ])
    epochs = Epochs(
        raw, events, event_id={"Pos": 1, "Neg": 2},
        tmin=-0.04, tmax=0.4, baseline=(-0.04, 0.0),
        preload=True, verbose=False,
    )
    bids_root = tmp_path / "bids"
    bids_root.mkdir()
    return {
        "Pos": epochs["Pos"].average(),
        "Neg": epochs["Neg"].average(),
    }, bids_root


def test_save_analysis_outputs_structured_per_type_files(tmp_path):
    """``{"by_type": {...}}`` writes one ``_desc-evoked{Cond}.fif`` per type."""
    evokeds, bids_root = _two_evoked_dict_for_save_analysis(tmp_path)
    out_paths = save_analysis_outputs(
        {"by_type": evokeds}, bids_root,
        subject="01", task="active", run=1,
    )
    names = sorted(p.name for p in out_paths)
    assert names == [
        "sub-01_task-active_run-1_desc-evokedNeg.fif",
        "sub-01_task-active_run-1_desc-evokedPos.fif",
    ]


def test_save_analysis_outputs_structured_combined_file(tmp_path):
    """``{"combined": Evoked}`` writes one bare ``_desc-evoked.fif`` file."""
    evokeds, bids_root = _two_evoked_dict_for_save_analysis(tmp_path)
    combined = evokeds["Pos"]  # any single Evoked stands in here
    out_paths = save_analysis_outputs(
        {"combined": combined}, bids_root,
        subject="01", task="active", run=1,
    )
    assert len(out_paths) == 1
    assert out_paths[0].name == "sub-01_task-active_run-1_desc-evoked.fif"


def test_save_analysis_outputs_structured_diff_file(tmp_path):
    """``{"diff": {(A, B): Evoked}}`` writes ``_desc-evokedDiff{A}Vs{B}.fif``."""
    evokeds, bids_root = _two_evoked_dict_for_save_analysis(tmp_path)
    out_paths = save_analysis_outputs(
        {"diff": {("Pos", "Neg"): evokeds["Pos"]}}, bids_root,
        subject="01", task="active", run=1,
    )
    assert len(out_paths) == 1
    assert out_paths[0].name == (
        "sub-01_task-active_run-1_desc-evokedDiffPosVsNeg.fif"
    )


def test_save_analysis_outputs_structured_full_payload(tmp_path):
    """All three sub-keys → 2 per-type + 1 combined + 1 diff = 4 files."""
    evokeds, bids_root = _two_evoked_dict_for_save_analysis(tmp_path)
    out_paths = save_analysis_outputs(
        {
            "by_type": evokeds,
            "combined": evokeds["Pos"],
            "diff": {("Pos", "Neg"): evokeds["Pos"]},
        },
        bids_root, subject="01", task="active", run=1,
    )
    assert len(out_paths) == 4


def test_save_analysis_outputs_per_type_sidecar_has_condition(tmp_path):
    """Per-type sidecar carries ``Condition: <name>``."""
    evokeds, bids_root = _two_evoked_dict_for_save_analysis(tmp_path)
    out_paths = save_analysis_outputs(
        {"by_type": evokeds}, bids_root,
        subject="01", task="active", run=1,
    )
    for p in out_paths:
        with open(p.with_suffix(".json")) as f:
            sidecar = json.load(f)
        assert "Condition" in sidecar


def test_save_analysis_outputs_combined_sidecar_no_condition(tmp_path):
    """Combined sidecar omits ``Condition`` (it's an across-types average)."""
    evokeds, bids_root = _two_evoked_dict_for_save_analysis(tmp_path)
    out_paths = save_analysis_outputs(
        {"combined": evokeds["Pos"]}, bids_root,
        subject="01", task="active", run=1,
    )
    with open(out_paths[0].with_suffix(".json")) as f:
        sidecar = json.load(f)
    assert "Condition" not in sidecar


def test_save_analysis_outputs_diff_sidecar_has_difference_of(tmp_path):
    """Diff sidecar carries ``DifferenceOf: [A, B]``."""
    evokeds, bids_root = _two_evoked_dict_for_save_analysis(tmp_path)
    out_paths = save_analysis_outputs(
        {"diff": {("Pos", "Neg"): evokeds["Pos"]}}, bids_root,
        subject="01", task="active", run=1,
    )
    with open(out_paths[0].with_suffix(".json")) as f:
        sidecar = json.load(f)
    assert sidecar.get("DifferenceOf") == ["Pos", "Neg"]


def test_save_analysis_outputs_scalar_input_unchanged(tmp_path):
    """Scalar Evoked input keeps today's contract: bare ``_desc-evoked.fif``."""
    evokeds, bids_root = _two_evoked_dict_for_save_analysis(tmp_path)
    out_paths = save_analysis_outputs(
        evokeds["Pos"], bids_root,
        subject="01", task="active", run=1,
    )
    assert len(out_paths) == 1
    assert out_paths[0].name == "sub-01_task-active_run-1_desc-evoked.fif"


def test_save_analysis_outputs_persists_baseline(tmp_path):
    """The analysis sidecar must record the baseline window.

    MNE's Evoked.save() does not write ``evoked.baseline`` into the
    .fif on disk, so after a load round-trip the attribute is None.
    This drops the RMS SNR row from the analysis report (which gates
    on ``evoked.baseline is not None``). Persisting the window in the
    BIDS sidecar lets downstream consumers restore the attribute via
    ``evoked.apply_baseline(...)`` post-load.
    """
    import numpy as np
    from mne import Epochs, create_info
    from mne.io import RawArray

    sfreq = 1000.0
    n_channels = 2
    n_times = int(sfreq * 5)
    rng = np.random.default_rng(1)
    data = rng.normal(0, 1e-6, size=(n_channels, n_times))
    info = create_info(["Cz", "F3"], sfreq=sfreq, ch_types=["eeg", "eeg"])
    raw = RawArray(data, info, verbose=False)

    n_events = 3
    event_samples = np.linspace(500, n_times - 500, n_events, dtype=int)
    events = np.column_stack([
        event_samples,
        np.zeros(n_events, dtype=int),
        np.ones(n_events, dtype=int),
    ])

    baseline = (-0.04, 0.0)
    epochs = Epochs(
        raw, events, tmin=-0.04, tmax=0.4, baseline=baseline,
        preload=True, verbose=False,
    )
    evoked = epochs.average()
    assert evoked.baseline == baseline, (
        "fixture: average() should propagate baseline metadata"
    )

    bids_root = tmp_path / "bids"
    bids_root.mkdir()
    out_paths = save_analysis_outputs(
        evoked, bids_root, subject="01", task="active", run=1,
    )

    sidecar_path = out_paths[0].with_suffix(".json")
    with open(sidecar_path) as f:
        sidecar = json.load(f)

    assert "Baseline" in sidecar, \
        "sidecar must record the baseline window from evoked.baseline"
    assert tuple(sidecar["Baseline"]) == baseline, \
        "sidecar Baseline must match evoked.baseline exactly"


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

    # Test 6: Empty events array — MNE's Epochs raises ValueError when
    # asked to construct epochs from zero events.
    empty_events = np.empty((0, 3), dtype=int)
    with pytest.raises(ValueError):
        Epochs(
            raw_data, empty_events,
            tmin=-0.2, tmax=0.5,
            verbose=False,
            preload=True,
        )


def _two_condition_epochs_for_evoked_helpers():
    """Build a two-condition Epochs object for the combined/diff tests."""
    import numpy as np
    from mne import Epochs, create_info
    from mne.io import RawArray

    sfreq = 1000.0
    n_channels = 2
    n_times = int(sfreq * 8)
    rng = np.random.default_rng(11)
    data = rng.normal(0, 1e-6, size=(n_channels, n_times))
    info = create_info(["Cz", "F3"], sfreq=sfreq, ch_types=["eeg"] * 2)
    raw = RawArray(data, info, verbose=False)

    onsets = [1000, 2000, 3000, 4000, 5000, 6000]
    codes = [1, 2, 1, 2, 1, 2]
    events = np.column_stack([
        np.array(onsets, dtype=int),
        np.zeros(len(onsets), dtype=int),
        np.array(codes, dtype=int),
    ])
    epochs = Epochs(
        raw, events, event_id={"Pos": 1, "Neg": 2},
        tmin=-0.04, tmax=0.4, baseline=(-0.04, 0.0),
        preload=True, verbose=False,
    )
    return epochs


def test_make_combined_evoked_returns_single_evoked_with_comment():
    """make_combined_evoked returns one Evoked tagged ``combined``."""
    import mne

    from ffrprep.preproc import make_combined_evoked

    epochs = _two_condition_epochs_for_evoked_helpers()
    combined = make_combined_evoked(epochs)
    assert isinstance(combined, mne.Evoked)
    assert combined.comment == "combined"


def test_make_combined_evoked_averages_across_all_events():
    """Combined evoked equals an unconditional ``epochs.average()``."""
    import numpy as np

    from ffrprep.preproc import make_combined_evoked

    epochs = _two_condition_epochs_for_evoked_helpers()
    combined = make_combined_evoked(epochs)
    expected = epochs.average()
    np.testing.assert_array_equal(combined.data, expected.data)
    assert combined.nave == expected.nave


def test_make_difference_evokeds_two_types_auto_pair():
    """Two-type dict with ``pairs=None`` auto-builds the single A-B pair."""
    from ffrprep.preproc import make_difference_evokeds, make_evoked

    epochs = _two_condition_epochs_for_evoked_helpers()
    evoked_dict = make_evoked(epochs, by_event_type=True)
    diffs = make_difference_evokeds(evoked_dict, pairs=None)
    assert set(diffs.keys()) == {("Pos", "Neg")}


def test_make_difference_evokeds_three_types_no_pairs_returns_empty():
    """Three-type dict + ``pairs=None`` → empty dict (must opt in explicitly)."""
    from ffrprep.preproc import make_difference_evokeds

    fake_evokeds = {"A": object(), "B": object(), "C": object()}
    diffs = make_difference_evokeds(fake_evokeds, pairs=None)
    assert diffs == {}


def test_make_difference_evokeds_explicit_pairs():
    """Explicit pairs are honored and emitted as tuple-keyed entries."""
    from ffrprep.preproc import make_difference_evokeds, make_evoked

    epochs = _two_condition_epochs_for_evoked_helpers()
    evoked_dict = make_evoked(epochs, by_event_type=True)
    diffs = make_difference_evokeds(
        evoked_dict, pairs=[("Pos", "Neg"), ("Neg", "Pos")],
    )
    assert set(diffs.keys()) == {("Pos", "Neg"), ("Neg", "Pos")}


def test_make_difference_evokeds_data_matches_subtraction():
    """Diff data equals ``A.data - B.data`` to numerical precision."""
    import numpy as np

    from ffrprep.preproc import make_difference_evokeds, make_evoked

    epochs = _two_condition_epochs_for_evoked_helpers()
    evoked_dict = make_evoked(epochs, by_event_type=True)
    diffs = make_difference_evokeds(evoked_dict, pairs=[("Pos", "Neg")])
    diff = diffs[("Pos", "Neg")]
    expected = evoked_dict["Pos"].data - evoked_dict["Neg"].data
    np.testing.assert_allclose(diff.data, expected, rtol=1e-10, atol=1e-15)


def test_make_difference_evokeds_comment_encodes_pair():
    """Each diff Evoked's ``.comment`` carries the pair so reports can label."""
    from ffrprep.preproc import make_difference_evokeds, make_evoked

    epochs = _two_condition_epochs_for_evoked_helpers()
    evoked_dict = make_evoked(epochs, by_event_type=True)
    diffs = make_difference_evokeds(evoked_dict, pairs=[("Pos", "Neg")])
    assert diffs[("Pos", "Neg")].comment == "diff_PosVsNeg"


def test_make_difference_evokeds_unknown_key_raises():
    """A pair referencing a missing trial type raises KeyError-style error."""
    from ffrprep.preproc import make_difference_evokeds, make_evoked

    epochs = _two_condition_epochs_for_evoked_helpers()
    evoked_dict = make_evoked(epochs, by_event_type=True)
    with pytest.raises((KeyError, ValueError)):
        make_difference_evokeds(evoked_dict, pairs=[("Pos", "Missing")])


def test_save_preprocessing_outputs_honors_output_dir(tmp_path):
    """save_preprocessing_outputs must write under the explicit output_dir.

    Regression test for the deeper layer of the silent-fail bug:
    even after setup_derivatives_directories was taught to honor
    output_dir, save_preprocessing_outputs internally re-derived its
    save location via setup_derivatives_directories WITHOUT the
    output_dir kwarg, so the workflow's save node always wrote to
    bids_root/derivatives regardless of what the CLI requested.
    """
    import numpy as np
    from mne import Epochs, create_info
    from mne.io import RawArray

    sfreq = 1000.0
    n_channels = 2
    n_times = int(sfreq * 5)
    rng = np.random.default_rng(2)
    data = rng.normal(0, 1e-6, size=(n_channels, n_times))
    info = create_info(["Cz", "F3"], sfreq=sfreq, ch_types=["eeg"] * 2)
    raw = RawArray(data, info, verbose=False)

    n_events = 3
    event_samples = np.linspace(500, n_times - 500, n_events, dtype=int)
    events = np.column_stack([
        event_samples,
        np.zeros(n_events, dtype=int),
        np.ones(n_events, dtype=int),
    ])

    epochs = Epochs(
        raw, events, tmin=-0.04, tmax=0.4, baseline=None,
        preload=True, verbose=False,
    )

    bids_root = tmp_path / "bids"
    bids_root.mkdir()
    custom_out = tmp_path / "custom_outputs"

    out_path = save_preprocessing_outputs(
        epochs, bids_root, subject="01", task="active", run=1,
        output_dir=custom_out,
    )

    expected = (
        custom_out / "ffrprep-preprocessing" / "sub-01" / "eeg"
        / "sub-01_task-active_run-1_desc-preproc_epo.fif"
    )
    assert out_path == expected, f"Expected {expected}, got {out_path}"
    assert out_path.exists(), "epoched .fif must exist at the explicit output_dir"
    assert out_path.with_suffix(".json").exists(), "sidecar must exist alongside"
    assert not (bids_root / "derivatives").exists(), (
        "When output_dir is supplied, the legacy bids_root/derivatives "
        "tree must not be created"
    )


def test_save_analysis_outputs_honors_output_dir(tmp_path):
    """save_analysis_outputs must write under the explicit output_dir.

    Same deeper bug as the preprocessing-side: setup_derivatives_directories
    was called internally without forwarding the user's output_dir,
    silently dropping the explicit destination.
    """
    import numpy as np
    from mne import Epochs, create_info
    from mne.io import RawArray

    sfreq = 1000.0
    n_channels = 2
    n_times = int(sfreq * 5)
    rng = np.random.default_rng(3)
    data = rng.normal(0, 1e-6, size=(n_channels, n_times))
    info = create_info(["Cz", "F3"], sfreq=sfreq, ch_types=["eeg"] * 2)
    raw = RawArray(data, info, verbose=False)

    n_events = 3
    event_samples = np.linspace(500, n_times - 500, n_events, dtype=int)
    events = np.column_stack([
        event_samples,
        np.zeros(n_events, dtype=int),
        np.ones(n_events, dtype=int),
    ])

    epochs = Epochs(
        raw, events, tmin=-0.04, tmax=0.4, baseline=(-0.04, 0.0),
        preload=True, verbose=False,
    )
    evoked = epochs.average()

    bids_root = tmp_path / "bids"
    bids_root.mkdir()
    custom_out = tmp_path / "custom_outputs"

    out_paths = save_analysis_outputs(
        evoked, bids_root, subject="01", task="active", run=1,
        output_dir=custom_out,
    )

    expected = (
        custom_out / "ffrprep-analysis" / "sub-01"
        / "sub-01_task-active_run-1_desc-evoked.fif"
    )
    assert out_paths[0] == expected, f"Expected {expected}, got {out_paths[0]}"
    assert out_paths[0].exists(), "evoked .fif must exist at the explicit output_dir"
    assert out_paths[0].with_suffix(".json").exists(), "sidecar must exist alongside"
    assert not (bids_root / "derivatives").exists(), (
        "When output_dir is supplied, the legacy bids_root/derivatives "
        "tree must not be created"
    )


def test_setup_derivatives_directories_honors_output_dir(tmp_path):
    """When ``output_dir`` is given, the derivatives root must point there.

    Regression test for a silent failure: prior to the fix
    ``setup_derivatives_directories`` hardcoded ``bids_root /
    "derivatives"`` and ignored any user-supplied output directory,
    so the BIDS-App's second positional argument (``output_dir``)
    was effectively cosmetic. Outputs always landed at
    ``bids_root/derivatives/`` regardless of where the user asked
    them to go, silently colliding with prior runs.
    """
    bids_root = tmp_path / "test_bids"
    bids_root.mkdir()
    custom_out = tmp_path / "custom_outputs"

    result = setup_derivatives_directories(
        bids_root=bids_root,
        subject="03",
        output_dir=custom_out,
    )

    assert result["derivatives_root"] == custom_out, (
        "derivatives_root must equal the explicit output_dir, not "
        "bids_root/derivatives"
    )
    assert result["preprocessing_dir"] == custom_out / "ffrprep-preprocessing"
    assert (
        result["preprocessing_subject_dir"]
        == custom_out / "ffrprep-preprocessing" / "sub-03" / "eeg"
    )
    assert (
        result["analysis_subject_dir"] == custom_out / "ffrprep-analysis" / "sub-03"
    )

    # Sanity: the legacy bids_root/derivatives location was NOT used.
    assert not (bids_root / "derivatives").exists(), (
        "When output_dir is supplied, the legacy bids_root/derivatives "
        "tree must not be created"
    )


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

    # Test 2: Only preprocessing directories.
    # The create_* flags only control mkdir behaviour; paths are always
    # resolvable so cross-stage callers can locate inputs/outputs from
    # the *other* stage without re-deriving the canonical layout.
    bids_root_2 = tmp_path / "test_bids_2"
    bids_root_2.mkdir()

    result_preproc_only = setup_derivatives_directories(
        bids_root=bids_root_2,
        subject="05",
        create_preprocessing=True,
        create_analysis=False
    )

    derivatives_root_2 = bids_root_2 / "derivatives"
    preproc_subject_dir_2 = derivatives_root_2 / "ffrprep-preprocessing" / "sub-05"
    analysis_dir_2 = derivatives_root_2 / "ffrprep-analysis"

    assert preproc_subject_dir_2.exists(), \
        "Preprocessing directory should be created when requested"
    assert not analysis_dir_2.exists(), \
        "Analysis directory should not be created when not requested"

    expected_analysis_subject_dir = analysis_dir_2 / "sub-05"
    assert result_preproc_only["analysis_dir"] == analysis_dir_2, \
        "Analysis dir path is always resolved, even when not created"
    assert result_preproc_only["analysis_subject_dir"] == expected_analysis_subject_dir, \
        "Analysis subject dir path is always resolved, even when not created"

    # Test 3: Only analysis directories.
    # This is the analysis-only run path: analysis stage needs to read
    # existing preproc outputs from the canonical preproc location, so
    # preprocessing_subject_dir must still be a usable Path even though
    # mkdir is skipped.
    bids_root_3 = tmp_path / "test_bids_3"
    bids_root_3.mkdir()

    result_analysis_only = setup_derivatives_directories(
        bids_root=bids_root_3,
        subject="07",
        create_preprocessing=False,
        create_analysis=True
    )

    derivatives_root_3 = bids_root_3 / "derivatives"
    preproc_dir_3 = derivatives_root_3 / "ffrprep-preprocessing"
    analysis_subject_dir_3 = derivatives_root_3 / "ffrprep-analysis" / "sub-07"

    assert analysis_subject_dir_3.exists(), \
        "Analysis directory should be created when requested"
    assert not preproc_dir_3.exists(), \
        "Preprocessing directory should not be created when not requested"

    expected_preproc_subject_dir = preproc_dir_3 / "sub-07" / "eeg"
    assert result_analysis_only["preprocessing_dir"] == preproc_dir_3, \
        "Preprocessing dir path is always resolved, even when not created"
    assert result_analysis_only["preprocessing_subject_dir"] == expected_preproc_subject_dir, \
        "Preprocessing subject dir path is always resolved, even when not created"

    # Test 4: Neither directory type (edge case).
    # Paths still resolve so callers can introspect canonical layout.
    bids_root_4 = tmp_path / "test_bids_4"
    bids_root_4.mkdir()

    result_neither = setup_derivatives_directories(
        bids_root=bids_root_4,
        subject="09",
        create_preprocessing=False,
        create_analysis=False
    )

    derivatives_root_4 = bids_root_4 / "derivatives"
    assert derivatives_root_4.exists(), \
        "Derivatives root should always be created"
    assert result_neither["derivatives_root"] == derivatives_root_4

    expected_preproc_dir_4 = derivatives_root_4 / "ffrprep-preprocessing"
    expected_analysis_dir_4 = derivatives_root_4 / "ffrprep-analysis"
    assert result_neither["preprocessing_dir"] == expected_preproc_dir_4, \
        "Preprocessing dir path is always resolved, even when not created"
    assert result_neither["analysis_dir"] == expected_analysis_dir_4, \
        "Analysis dir path is always resolved, even when not created"
    assert not expected_preproc_dir_4.exists(), \
        "Preprocessing dir should not be created when create_preprocessing=False"
    assert not expected_analysis_dir_4.exists(), \
        "Analysis dir should not be created when create_analysis=False"

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
