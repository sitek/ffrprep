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

    # Filter data node
    filter_node = Node(
        Function(
            input_names=["eeg_data", "high_pass", "low_pass"],
            output_names=["filtered_data"], function=filter_data
        ),
        name="filter_data",
    )

    # Epoch data node
    epoch_node = Node(
        Function(
            input_names=["eeg_data", "baseline", "tmin", "tmax"],
            output_names=["epochs", "time_window"],
            function=epoch_data,
        ),
        name="epoch_data",
    )

    # Save preprocessing outputs node
    save_node = Node(
        Function(
            input_names=["epochs", "bids_root", "sub_label",
                         "original_filename"],
            output_names=["output_path"],
            function=save_preprocessing_node,
        ),
        name="save_preprocessing",
    )

    # Connect workflow nodes
    workflow.connect(
        [
            # Connect inputs to load node
            (
                inputnode,
                load_node,
                [
                    ("bids_root", "bids_root"),
                    ("sub_label", "sub_label"),
                    ("session_label", "session_label"),
                    ("task_label", "task_label"),
                    ("run_label", "run_label"),
                ],
            ),
            # Connect load to reference
            (load_node, reference_node, [("eeg_data", "eeg_data")]),
            (inputnode, reference_node, [("ref_channels", "ref_channels")]),
            # Connect reference to filter
            (reference_node, filter_node, [("referenced_data", "eeg_data")]),
            (inputnode, filter_node, [("high_pass", "high_pass"),
                                      ("low_pass", "low_pass")]),
            # Connect filter to epoch
            (filter_node, epoch_node, [("filtered_data", "eeg_data")]),
            (inputnode, epoch_node, [("baseline", "baseline"),
                                     ("tmin", "tmin"), ("tmax", "tmax")]),
            # Connect to save node
            (epoch_node, save_node, [("epochs", "epochs")]),
            (
                inputnode,
                save_node,
                [("bids_root", "bids_root"), ("sub_label", "sub_label")],
            ),
            (load_node, save_node, [("original_filename",
                                     "original_filename")]),
            # Connect to output
            (epoch_node, outputnode, [("epochs", "epochs")]),
            (load_node, outputnode, [("original_filename",
                                      "original_filename")]),
        ]
    )

    return workflow


def create_analysis_workflow(name="ffrprep_analysis"):
    """
    Create nipype workflow for FFR analysis.

    Parameters
    ----------
    name : str
        Name of the workflow

    Returns
    -------
    workflow : nipype.Workflow
        Analysis workflow
    """
    from nipype import Workflow, Node, Function
    from nipype.interfaces import utility as niu

    workflow = Workflow(name=name)

    # Input node
    inputnode = Node(
        niu.IdentityInterface(
            fields=["epochs", "by_event_type", "bids_root", "subject",
                    "output_dir", "original_filename"]
        ),
        name="inputnode",
    )

    # Output node
    outputnode = Node(niu.IdentityInterface(fields=["evoked",
                                                    "analysis_report"]),
                      name="outputnode")

    # Make evoked node
    evoked_node = Node(
        Function(input_names=["epochs", "by_event_type"],
                 output_names=["evoked"], function=make_evoked),
        name="make_evoked",
    )

    # Save analysis outputs node
    save_analysis_node_func = Node(
        Function(
            input_names=["evoked", "bids_root", "subject",
                         "original_filename"],
            output_names=["output_paths"],
            function=save_analysis_node,
        ),
        name="save_analysis",
    )

    # Connect workflow
    workflow.connect(
        [
            (inputnode, evoked_node, [("epochs", "epochs"),
                                      ("by_event_type", "by_event_type")]),
            # Connect to save node
            (evoked_node, save_analysis_node_func, [("evoked", "evoked")]),
            (
                inputnode,
                save_analysis_node_func,
                [
                    ("bids_root", "bids_root"),
                    ("subject", "subject"),
                    ("original_filename", "original_filename"),
                ],
            ),
            # Connect to output
            (evoked_node, outputnode, [("evoked", "evoked")]),
        ]
    )

    return workflow


def setup_derivatives_directories(bids_root, subject,
                                  create_preprocessing=True,
                                  create_analysis=True):
    """
    Set up BIDS derivatives directories for ffrprep outputs.

    Parameters
    ----------
    bids_root : str or pathlib.Path
        Path to the BIDS dataset root directory.
    subject : str
        Subject label (without 'sub-' prefix).
    create_preprocessing : bool
        Whether to create preprocessing derivatives directory.
    create_analysis : bool
        Whether to create analysis derivatives directory.

    Returns
    -------
    derivatives_info : dict
        Dictionary containing paths to derivatives directories.
    """
    from pathlib import Path

    bids_root = Path(bids_root)
    derivatives_root = bids_root / "derivatives"

    # Create main derivatives directory if it doesn't exist
    derivatives_root.mkdir(exist_ok=True)

    # Set up preprocessing derivatives
    preproc_dir = None
    if create_preprocessing:
        preproc_dir = derivatives_root / "ffrprep-preprocessing"
        preproc_subject_dir = preproc_dir / f"sub-{subject}"
        preproc_subject_dir.mkdir(parents=True, exist_ok=True)

    # Set up analysis derivatives
    analysis_dir = None
    if create_analysis:
        analysis_dir = derivatives_root / "ffrprep-analysis"
        analysis_subject_dir = analysis_dir / f"sub-{subject}"
        analysis_subject_dir.mkdir(parents=True, exist_ok=True)

    return {
        "derivatives_root": derivatives_root,
        "preprocessing_dir": preproc_dir,
        "analysis_dir": analysis_dir,
        "preprocessing_subject_dir": (
            preproc_dir / f"sub-{subject}" if preproc_dir else None
        ),
        "analysis_subject_dir": (
            analysis_dir / f"sub-{subject}" if analysis_dir else None
        ),
    }


def check_preprocessing_exists(bids_root, subject):
    """
    Check if preprocessing outputs exist for a given subject.

    Parameters
    ----------
    bids_root : str or pathlib.Path
        Path to the BIDS dataset root directory.
    subject : str
        Subject label (without 'sub-' prefix).

    Returns
    -------
    exists : bool
        Whether preprocessing outputs exist for the subject.
    preproc_files : list
        List of found preprocessing files.
    """
    from pathlib import Path

    bids_root = Path(bids_root)
    preproc_dir = (
        bids_root
        / "derivatives"
        / "ffrprep-preprocessing"
        / f"sub-{subject}"
    )

    if not preproc_dir.exists():
        return False, []

    # Look for common preprocessing output files
    preproc_patterns = [
        "*_desc-preproc.fif",  # New naming convention
        f"sub-{subject}_*_desc-preproc.fif",  # With subject prefix
        f"sub-{subject}_*_epo.fif",  # Legacy epoched data
        f"sub-{subject}_*_epochs.fif",  # Legacy alternative epoch naming
        f"sub-{subject}_*_preproc.fif",  # Legacy preprocessed continuous data
    ]

    found_files = []
    for pattern in preproc_patterns:
        found_files.extend(list(preproc_dir.glob(pattern)))

    return len(found_files) > 0, found_files


def save_preprocessing_outputs(epochs, bids_root, subject, task,
                               session=None, run=None):
    """
    Save preprocessing outputs to BIDS derivatives structure.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched data to save.
    bids_root : str or pathlib.Path
        Path to the BIDS dataset root directory.
    subject : str
        Subject label (without 'sub-' prefix).
    task : str
        Task label (without 'task-' prefix). Required for BIDS compliance.
    session : str, optional
        Session label (without 'ses-' prefix).
    run : str or int, optional
        Run label (without 'run-' prefix).

    Returns
    -------
    output_path : pathlib.Path
        Path to the saved epochs file.
    """

    # Set up derivatives directory
    derivatives_info = setup_derivatives_directories(
        bids_root, subject, create_preprocessing=True, create_analysis=False
    )

    # Build BIDS-compliant filename with task (required)
    # and session/run (optional)
    filename_parts = [f"sub-{subject}"]

    # Session is optional
    if session:
        filename_parts.append(f"ses-{session}")

    # Task is required for BIDS compliance
    filename_parts.append(f"task-{task}")

    # Run is optional
    if run:
        filename_parts.append(f"run-{run}")

    # Add descriptor and extension
    filename_parts.append("desc-preproc.fif")
    filename = "_".join(filename_parts)

    output_path = derivatives_info["preprocessing_subject_dir"] / filename

    # Save epochs
    epochs.save(output_path, overwrite=True)

    # Create dataset_description.json if it doesn't exist
    preprocessing_dir = derivatives_info["preprocessing_dir"]
    dataset_desc_path = preprocessing_dir / "dataset_description.json"
    if not dataset_desc_path.exists():
        import json

        dataset_desc = {
            "Name": "ffrprep preprocessing outputs",
            "BIDSVersion": "1.6.0",
            "GeneratedBy": [
                {
                    "Name": "ffrprep",
                    "Description": (
                        "Frequency-following response preprocessing "
                        "pipeline"
                    ),
                }
            ],
        }
        with open(dataset_desc_path, "w") as f:
            json.dump(dataset_desc, f, indent=2)

    return output_path


def load_preprocessing_outputs(bids_root, subject, original_filename=None):
    """
    Load preprocessing outputs from BIDS derivatives structure.

    Parameters
    ----------
    bids_root : str or pathlib.Path
        Path to the BIDS dataset root directory.
    subject : str
        Subject label (without 'sub-' prefix).
    original_filename : str, optional
        Original filename base to look for. If None, finds any
        preprocessing file for the subject.

    Returns
    -------
    epochs : mne.Epochs
        Loaded epoched data.
    """
    from pathlib import Path
    import importlib

    # Lazy import of mne with a helpful error if it's not installed.
    try:
        mne = importlib.import_module("mne")
    except ImportError as exc:
        raise ImportError(
            "The 'mne' package is required to load preprocessing outputs. "
            "Install it with 'pip install mne'"
        ) from exc

    bids_root = Path(bids_root)
    preproc_dir = (
        bids_root
        / "derivatives"
        / "ffrprep-preprocessing"
        / f"sub-{subject}"
    )

    if original_filename:
        # Look for file with specific original filename
        expected_filename = f"{original_filename}_desc-preproc.fif"
        expected_path = preproc_dir / expected_filename

        if expected_path.exists():
            return mne.read_epochs(expected_path)

    # Try to find any preprocessing file for the subject
    epoch_patterns = [
        "*desc-preproc.fif",  # New naming convention
        f"sub-{subject}_*desc-preproc.fif",  # Fallback with subject
        f"sub-{subject}_*_epo.fif",  # Legacy pattern
    ]

    for pattern in epoch_patterns:
        epoch_files = list(preproc_dir.glob(pattern))
        if epoch_files:
            if len(epoch_files) > 1:
                print(
                    f"Warning: Multiple files found, using "
                    f"{epoch_files[0].name}"
                )
            return mne.read_epochs(epoch_files[0])

    raise FileNotFoundError(
        f"No preprocessing outputs found for subject {subject} in "
        f"{preproc_dir}"
    )


def save_analysis_outputs(evoked, bids_root, subject, task,
                          session=None, run=None, analysis_type="evoked"):
    """
    Save analysis outputs to BIDS derivatives structure.

    Parameters
    ----------
    evoked : mne.Evoked or dict of mne.Evoked
        Evoked data to save.
    bids_root : str or pathlib.Path
        Path to the BIDS dataset root directory.
    subject : str
        Subject label (without 'sub-' prefix).
    task : str
        Task label (without 'task-' prefix). Required for BIDS compliance.
    session : str, optional
        Session label (without 'ses-' prefix).
    run : str or int, optional
        Run label (without 'run-' prefix).
    analysis_type : str
        Type of analysis output ('evoked', 'spectrum', etc.).

    Returns
    -------
    output_paths : list of pathlib.Path
        Paths to the saved files.
    """

    # Set up derivatives directory
    derivatives_info = setup_derivatives_directories(
        bids_root, subject, create_preprocessing=False, create_analysis=True
    )

    output_paths = []

    # Build base filename with task (required) and session/run (optional)
    filename_base_parts = [f"sub-{subject}"]

    # Session is optional
    if session:
        filename_base_parts.append(f"ses-{session}")

    # Task is required for BIDS compliance
    filename_base_parts.append(f"task-{task}")

    # Run is optional
    if run:
        filename_base_parts.append(f"run-{run}")

    filename_base = "_".join(filename_base_parts)

    if isinstance(evoked, dict):
        # Multiple conditions - save each separately
        for condition, evoked_data in evoked.items():
            # Format condition name with proper capitalization
            condition_formatted = str(condition).capitalize()
            filename = (
                f"{filename_base}_desc-{analysis_type}"
                f"{condition_formatted}.fif"
            )
            output_path = derivatives_info["analysis_subject_dir"] / filename
            evoked_data.save(output_path)
            output_paths.append(output_path)
    else:
        # Single evoked response
        filename = f"{filename_base}_desc-{analysis_type}.fif"
        output_path = derivatives_info["analysis_subject_dir"] / filename
        evoked.save(output_path)
        output_paths.append(output_path)

    # Create dataset_description.json if it doesn't exist
    dataset_desc_path = (
        derivatives_info["analysis_dir"] / "dataset_description.json"
    )
    if not dataset_desc_path.exists():
        import json

        dataset_desc = {
            "Name": "ffrprep analysis outputs",
            "BIDSVersion": "1.6.0",
            "GeneratedBy": [
                {"Name": "ffrprep", "Description": "Frequency-following response analysis pipeline"}  # noqa: E501
            ],
        }
        with open(dataset_desc_path, "w") as f:
            json.dump(dataset_desc, f, indent=2)

    return output_paths


def save_preprocessing_node(epochs, bids_root, subject, task,
                            session=None, run=None):
    """
    Nipype-compatible function to save preprocessing outputs.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched data to save.
    bids_root : str
        Path to the BIDS dataset root directory.
    subject : str
        Subject label (without 'sub-' prefix).
    task : str
        Task label (without 'task-' prefix). Required for BIDS compliance.
    session : str, optional
        Session label (without 'ses-' prefix).
    run : str or int, optional
        Run label (without 'run-' prefix).

    Returns
    -------
    output_path : str
        Path to the saved epochs file.
    """
    output_path = save_preprocessing_outputs(
        epochs, bids_root, subject, task, session, run
    )
    return str(output_path)


def save_analysis_node(evoked, bids_root, subject, task,
                       session=None, run=None, analysis_type="evoked"):
    """
    Nipype-compatible function to save analysis outputs.

    Parameters
    ----------
    evoked : mne.Evoked or dict of mne.Evoked
        Evoked data to save.
    bids_root : str
        Path to the BIDS dataset root directory.
    subject : str
        Subject label (without 'sub-' prefix).
    task : str
        Task label (without 'task-' prefix). Required for BIDS compliance.
    session : str, optional
        Session label (without 'ses-' prefix).
    run : str or int, optional
        Run label (without 'run-' prefix).
    analysis_type : str
        Type of analysis output ('evoked', 'spectrum', etc.).

    Returns
    -------
    output_paths : list of str
        Paths to the saved files.
    """
    output_paths = save_analysis_outputs(
        evoked, bids_root, subject, task, session, run, analysis_type
    )
    return [str(p) for p in output_paths]
