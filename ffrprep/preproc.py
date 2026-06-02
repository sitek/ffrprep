from mne_bids import BIDSPath, read_raw_bids
from mne import Epochs
from bids import BIDSLayout


def load_data(bids_root=None, sub_label=None, session_label=None,
              task_label=None, run_label=None):
    """
    Identify and load EEG data from a BIDS directory using pybids for querying.

    This function uses pybids to robustly query the BIDS dataset and find
    the appropriate EEG files, then uses MNE-BIDS to load the data.

    Parameters
    ----------
    bids_root : string
        The top-level directory of the BIDS dataset.
    sub_label : string
        Subject label of the specific participant whose data is to be loaded.
        Default = None.
    session_label : string
        Session label of the specific session to be loaded.
        Default = None.
    task_label : string
        Task label of the specific task to be loaded.
        Default = None.
    run_label : string
        Run label of the specific run to be loaded.
        Default = None.

    Returns
    -------
    data : MNE `data` object
        MNE `data` object containing EEG data and metadata.
    bids_path : MNE-BIDS `BidsPath` object
        MNE-BIDS `BidsPath` object containing EEG filepaths.

    Examples
    --------
    Load an EEG-BIDS file.

    >>> data = load_data(bids_root=bids_root,
                         sub_label=sub_label,
                         run_label=run_label)
    """
    # Create BIDSLayout for robust querying
    layout = BIDSLayout(bids_root, validate=False)

    # Build query parameters, filtering out None values
    query_params = {
        "subject": sub_label,
        "datatype": "eeg",
        "extension": ".edf",  # Common EEG format, adjust as needed
    }

    # Add optional parameters if provided
    if session_label:
        query_params["session"] = session_label
    if task_label:
        query_params["task"] = task_label
    if run_label:
        query_params["run"] = run_label

    # Query for EEG files using pybids
    eeg_files = layout.get(**query_params)

    if not eeg_files:
        # Try alternative extensions if .edf not found
        for ext in [".bdf", ".vhdr", ".fif", ".set"]:
            query_params["extension"] = ext
            eeg_files = layout.get(**query_params)
            if eeg_files:
                print(f"Found {len(eeg_files)} EEG files with extension {ext}")
                break

    if not eeg_files:
        # Provide helpful error message with available options
        all_eeg_files = layout.get(subject=sub_label, datatype="eeg")
        if all_eeg_files:
            available_entities = set()
            for f in all_eeg_files:
                entities = layout.parse_file_entities(f.path)
                available_entities.add(
                    f"task-{entities.get('task', 'N/A')}, "
                    f"run-{entities.get('run', 'N/A')}, "
                    f"ses-{entities.get('session', 'N/A')}"
                )
            raise FileNotFoundError(
                f"No EEG files found for subject {sub_label} with the "
                f"specified parameters. Available files have: "
                f"{'; '.join(available_entities)}"
            )
        else:
            raise FileNotFoundError(
                f"No EEG files found for subject {sub_label} "
                f"in {bids_root}"
            )

    # Use the first file if multiple matches (warn if multiple)  # noqa: E501
    eeg_file = eeg_files[0]
    if len(eeg_files) > 1:
        print(f"Warning: Multiple files found, using: {eeg_file.filename}")
        print(f"Available files: {[f.filename for f in eeg_files]}")

    # Extract BIDS entities from the found file
    entities = layout.parse_file_entities(eeg_file.path)

    # Create a BIDS path object using the found file's entities
    bids_path = BIDSPath(
        subject=entities.get("subject"),
        session=entities.get("session"),
        task=entities.get("task"),
        run=entities.get("run"),
        root=bids_root,
        datatype="eeg",
    )

    # Read the raw data from the BIDS structure
    data = read_raw_bids(bids_path=bids_path, verbose=False)

    # Load the data into memory for processing
    data = data.load_data()

    # Store original filename information for derivatives naming
    from pathlib import Path

    original_filename = Path(eeg_file.filename).stem  # Remove extension

    # Return data, bids_path, and original filename info
    return data, bids_path, original_filename


def get_participants(bids_root, participant_label=None):
    """
    Get list of participants from BIDS dataset using pybids.

    Parameters
    ----------
    bids_root : str or pathlib.Path
        Path to the BIDS dataset root directory.
    participant_label : list of str, optional
        List of participant labels to filter. If None, returns all participants
        with EEG data.

    Returns
    -------
    participants : list of str
        List of participant labels (without 'sub-' prefix).

    Examples
    --------
    Get all participants with EEG data:

    >>> participants = get_participants('/path/to/bids')

    Get specific participants:

    >>> participants = get_participants('/path/to/bids', ['01', '02'])
    """
    # Create BIDSLayout for querying
    layout = BIDSLayout(bids_root, validate=False)

    # Query for all participants with EEG data
    eeg_participants = layout.get_subjects(datatype="eeg")

    if participant_label:
        # Filter to requested participants
        available_participants = [
            p for p in participant_label if p in eeg_participants
        ]

        # Warn about missing participants
        missing_participants = [
            p for p in participant_label if p not in eeg_participants
        ]
        if missing_participants:
            print(
                f"Warning: Participants {missing_participants} not found "
                f"or have no EEG data in {bids_root}"
            )

        return available_participants
    else:
        return eeg_participants


def get_sessions_tasks_runs(bids_root, subject):
    """
    Get available sessions, tasks, and runs for a specific participant.

    Parameters
    ----------
    bids_root : str or pathlib.Path
        Path to the BIDS dataset root directory.
    subject : str
        Subject label (without 'sub-' prefix).

    Returns
    -------
    metadata : dict
        Dictionary containing available sessions, tasks,
        and runs for the subject.

    Examples
    --------
    Get metadata for a subject:

    >>> metadata = get_sessions_tasks_runs('/path/to/bids', '01')
    """
    layout = BIDSLayout(bids_root, validate=False)

    # Query for sessions
    sessions = layout.get_sessions(subject=subject, datatype="eeg")

    # Query for tasks
    tasks = layout.get_tasks(subject=subject, datatype="eeg")

    # Query for runs
    runs = layout.get_runs(subject=subject, datatype="eeg")

    return {
        "sessions": sessions if sessions else [None],
        "tasks": tasks if tasks else [None],
        "runs": runs if runs else [None],
    }


def reference_data(eeg_data=None, ref_channels=None):
    """
    Re-reference the provided EEG data object.

    Parameters
    ----------
    data : MNE `data` object
        MNE `data` object containing EEG data and metadata.
    ref_channels : list
        Channels to be used as reference. If more than one channel
        in list, the average of the channels in `ref_channels` will
        be used as the reference. If `None`, all channels will be
        averaged as the reference.
        Default = None.

    Returns
    -------
    referenced_eeg : MNE `data` object
        MNE `data` object containing EEG data and metadata referenced
        to new provided channels.

    Examples
    --------
    Re-reference an EEG data object to the average of all channels.

    >>> referenced_data = reference_data(eeg_data, ref_channels=None)

    Re-reference an EEG data object to the average of specific channels.

    >>> referenced_data = reference_data(eeg_data, ref_channels=['M1', 'M2'])

    Re-reference an EEG data object to a specific channel.

    >>> referenced_data = reference_data(eeg_data, ref_channels=['M1'])
    """
    # Apply average reference by default (all channels averaged)
    if ref_channels is None:
        eeg_data.set_eeg_reference()
    # If single channel provided as a string, convert to list
    elif isinstance(ref_channels, str):
        eeg_data.set_eeg_reference(ref_channels=[ref_channels])
    # If list of channels is provided, use them directly
    else:
        eeg_data.set_eeg_reference(ref_channels=ref_channels)

    # Return the re-referenced data
    return eeg_data


def filter_data(eeg_data=None, high_pass=None, low_pass=None):
    """
    Apply frequency filters to the provided EEG data object.

    Parameters
    ----------
    eeg_data : MNE `data` object
        MNE `data` object containing EEG data and metadata.
    high_pass : float
        Lower passband edge (in Hertz).
        Default = None.
    low_pass : float
        Upper passband edge (in Hertz).
        Default = None.

    Returns
    -------
    filtered_eeg : MNE `data` object
        EEG data filtered to given frequency range.

    Examples
    --------
    Filter an EEG data object with a band-pass filter (1-40 Hz).

    >>> filtered_data = filter_data(eeg_data, high_pass=1.0, low_pass=40.0)

    Filter an EEG data object with a high-pass filter (0.1 Hz).

    >>> filtered_data = filter_data(eeg_data, high_pass=0.1)

    Filter an EEG data object with a low-pass filter (100 Hz).

    >>> filtered_data = filter_data(eeg_data, low_pass=100.0)
    """
    # Create a copy of the data to avoid modifying the original
    filtered_eeg = eeg_data.copy()

    # Apply filtering using MNE's filter method
    # l_freq is low-frequency cutoff (high-pass), h_freq is high-frequency cutoff (low-pass)  # noqa: E501
    filtered_eeg.filter(l_freq=high_pass, h_freq=low_pass)

    # Return the filtered data
    return filtered_eeg


def epoch_data(eeg_data, baseline, events_file=None, picks=None, tmin=None,
               tmax=None, verbose=True):
    """
    Epoch the provided EEG data object based on events.

    Parameters
    ----------
    eeg_data : MNE `data` object
        MNE `data` object containing EEG data and metadata.
    baseline : float or 1-D array
        In seconds: start of baseline period (if float);
        baseline window (if array).
    events_file : str, optional
        Path to events file. If provided, events will be loaded from file.
        If None, events will be found from the data annotations or triggers.
    picks : list, optional
        Channels to include in epochs.
    tmin : float, optional
        Start time before event (in seconds).
    tmax : float, optional
        End time after event (in seconds).
    verbose : bool
        Whether to print verbose output.

    Returns
    -------
    epoched_data : MNE `epoch` object
        MNE `epoch` object containing time-locked epochs.

    Examples
    --------
    Epoch an EEG data object with a baseline time start
    of 50 ms before event onset.

    >>> epoched_data = epoch_data(eeg_data, baseline=-0.05)

    Epoch an EEG data object with a baseline window
    of 200 to 50 ms before event onset.

    >>> epoched_data = epoch_data(eeg_data, baseline=[-0.2, -0.05])

    Load events from external file:

    >>> epoched_data = epoch_data(eeg_data, baseline=[-0.2, -0.05],
    ...                          events_file='/path/to/events.tsv')

    """

    # Load events from file or extract from data
    if events_file:
        # Load events from external file (e.g., BIDS events.tsv)
        import pandas as pd
        import numpy as np

        # Read events file
        events_df = pd.read_csv(events_file, sep='\t')

        # Convert onset times to sample indices
        sfreq = eeg_data.info['sfreq']
        onset_samples = (events_df['onset'] * sfreq).astype(int)

        # Create MNE events array [sample, prev_sample, event_id]
        events = np.column_stack([
            onset_samples,
            np.zeros(len(onset_samples), dtype=int),  # previous sample
            events_df.get('trial_type', 1).astype(int)  # event codes
        ])

        # Create event dictionary from trial types
        if 'trial_type' in events_df.columns:
            unique_types = events_df['trial_type'].unique()
            event_dict = {
                str(trial_type): idx + 1
                for idx, trial_type in enumerate(unique_types)
            }
        else:
            event_dict = {'event': 1}
    else:
        # Try to find events from data annotations or triggers
        try:
            from mne import find_events
            events = find_events(eeg_data, verbose=verbose)
            event_dict = None  # Will use default event IDs
        except Exception:
            # If no events found, create dummy events for demonstration
            # This should be replaced with proper event detection
            events = None
            event_dict = None

    # Define baseline window (in seconds)
    if isinstance(baseline, float):
        # If single float provided, use it as start time with 0 as end time
        baseline_window = (baseline, 0)
    else:
        # If array provided, use it directly as baseline window
        baseline_window = tuple(baseline)

    # Define epoch time window (in seconds)
    if not tmin:
        # If no tmin specified, start from baseline start time
        tmin = baseline_window[0]
    if not tmax:
        # Default epoch end time is 0.5 seconds after event
        tmax = 0.5

    # Create epochs with specified parameters
    epoched_data = Epochs(
        eeg_data,
        events=events,
        event_id=event_dict,
        on_missing="warn",  # Warn about missing events
        picks=picks,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline_window,
        verbose=verbose,
        reject=dict(eeg=75e-6),
    ).drop_bad()  # Reject bad epochs  # noqa: E501

    return epoched_data, (tmin, tmax)


def make_evoked(epochs, by_event_type: bool):
    """
    Create evoked responses by averaging epochs.

    This function computes the average response across all epochs,
    optionally grouping by event type to create separate evoked
    responses for different experimental conditions.

    Parameters
    ----------
    epochs : mne.Epochs
        MNE `Epochs` object containing time-locked epochs.
    by_event_type : bool
        Whether to sort epochs by event type when averaging.
        If True, creates separate evoked responses for each event type.
        If False, averages all epochs together.

    Returns
    -------
    evoked : mne.Evoked or dict of mne.Evoked
        If by_event_type is False: single MNE `Evoked` object containing
        the average across all epochs.
        If by_event_type is True: dictionary with event names as keys and
        corresponding `Evoked` objects as values.

    Examples
    --------
    Average all epochs together:

    >>> evoked_data = make_evoked(epoched_data, by_event_type=False)

    Create separate evoked responses for each event type:

    >>> evoked_by_condition = make_evoked(epoched_data, by_event_type=True)
    """
    if by_event_type:
        # Create separate evoked responses for each event type
        evoked = dict()
        for event_name, event_id in epochs.event_id.items():
            epochs_subset = epochs[event_name]
            evoked[event_name] = epochs_subset.average()
    else:
        # Average all epochs together
        evoked = epochs.average()

    # Return the averaged epochs (evoked responses)
    return evoked


def create_preprocessing_workflow(name="ffrprep_preproc"):
    """
    Create nipype workflow for FFR preprocessing.

    Parameters
    ----------
    name : str
        Name of the workflow

    Returns
    -------
    workflow : nipype.Workflow
        Preprocessing workflow
    """
    from nipype import Workflow, Node, Function
    from nipype.interfaces import utility as niu

    workflow = Workflow(name=name)

    # Input node to specify subject parameters
    inputnode = Node(
        niu.IdentityInterface(
            fields=[
                "bids_root",
                "sub_label",
                "session_label",
                "task_label",
                "run_label",
                "ref_channels",
                "high_pass",
                "low_pass",
                "baseline",
                "tmin",
                "tmax",
                "output_dir",
            ]
        ),
        name="inputnode",
    )

    # Output node to collect results
    outputnode = Node(
        niu.IdentityInterface(fields=["epochs", "preprocessing_report",
                                      "original_filename"]), name="outputnode"
    )

    # Load data node
    load_node = Node(
        Function(
            input_names=["bids_root", "sub_label", "session_label",
                         "task_label", "run_label"],
            output_names=["eeg_data", "bids_path", "original_filename"],
            function=load_data,
        ),
        name="load_data",
    )

    # Reference data node
    reference_node = Node(
        Function(
            input_names=["eeg_data", "ref_channels"],
            output_names=["referenced_data"], function=reference_data
        ),
        name="reference_data",
    )

    # Return the epoched data
    return preprocessed_data

