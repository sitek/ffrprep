"""
preproc module for ffrprep.

Local imports are used inside functions so that Nipype Function nodes
executed in separate processes have the necessary imports available at
runtime. Avoid top-level imports for packages that are imported inside
functions to prevent redefinition and lint warnings.
"""


def load_data(bids_root=None, sub_label=None, session_label=None, task_label=None, run_label=None):
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
    # Local imports keep these names resolvable inside Nipype Function
    # nodes (which run in fresh subprocesses). pybids and mne-bids are
    # declared as required dependencies in pyproject.toml, so a plain
    # import is always safe here.
    from bids import BIDSLayout
    from mne_bids import BIDSPath, read_raw_bids

    # Create BIDSLayout for robust querying
    layout = BIDSLayout(bids_root, validate=False)

    # Build base query parameters, filtering out None values. Extension is
    # discovered separately below so we don't bake .edf into the per-run
    # lookup (which would silently drop .bdf/.vhdr/.fif/.set datasets).
    query_params = {
        "subject": sub_label,
        "datatype": "eeg",
    }
    if session_label:
        query_params["session"] = session_label
    if task_label:
        query_params["task"] = task_label

    # Discover the file extension actually present in the dataset by
    # probing without the run filter. This must happen before any
    # run-aware query so per-run lookups use the correct extension.
    discovered_ext = None
    for ext in [".edf", ".bdf", ".vhdr", ".fif", ".set"]:
        probe_qp = dict(query_params)
        probe_qp["extension"] = ext
        if layout.get(**probe_qp):
            discovered_ext = ext
            break
    if discovered_ext is None:
        raise FileNotFoundError(
            f"No EEG files found for subject {sub_label} in {bids_root} "
            f"(no .edf/.bdf/.vhdr/.fif/.set under the requested "
            f"task/session)."
        )
    query_params["extension"] = discovered_ext
    print(f"Using EEG file extension: {discovered_ext}")

    # Support run_label being a single value or a list of values. When a
    # list is provided, build a combined list of matching files across the
    # requested runs (used for selective concatenation).
    eeg_files = []
    if isinstance(run_label, (list, tuple)):
        for rl in run_label:
            qp = dict(query_params)
            if rl:
                qp["run"] = rl
            found = layout.get(**qp)
            if found:
                # Prefer the first match for the requested run
                eeg_files.append(found[0])
            else:
                print(f"Warning: No EEG file found for run {rl} (subject {sub_label})")
    else:
        if run_label:
            query_params["run"] = run_label
        eeg_files = layout.get(**query_params)

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
            raise FileNotFoundError(f"No EEG files found for subject {sub_label} " f"in {bids_root}")

    # If multiple files found and no specific run was requested, or a list
    # of runs was provided, read and concatenate the selected runs for
    # that subject/task. If a single run_label was specified, fall back
    # to the single-run behavior.
    if len(eeg_files) > 1 and (run_label is None or isinstance(run_label, (list, tuple))):
        print(f"Multiple runs found ({len(eeg_files)}). Concatenating all runs.")
        raws = []
        # Keep entities from the first file for naming/metadata
        first_entities = None
        for f in eeg_files:
            entities_f = layout.parse_file_entities(f.path)
            if first_entities is None:
                first_entities = entities_f

            bp = BIDSPath(
                subject=entities_f.get("subject"),
                session=entities_f.get("session"),
                task=entities_f.get("task"),
                run=entities_f.get("run"),
                root=bids_root,
                datatype="eeg",
            )
            raw = read_raw_bids(bids_path=bp, verbose=False)
            raws.append(raw)

        # Concatenate raws into a single Raw object. mne>=1.0 exposes
        # concatenate_raws at the package top-level; pyproject pins mne>=1.9.
        from mne import concatenate_raws

        data = concatenate_raws(raws)

        # Load into memory
        data = data.load_data()

        # Build a conservative original filename (subject+task) for outputs
        original_filename = f"sub-{first_entities.get('subject')}_" f"task-{first_entities.get('task')}"

        # Create a BIDSPath without run for metadata purposes
        bids_path = BIDSPath(
            subject=first_entities.get("subject"),
            session=first_entities.get("session"),
            task=first_entities.get("task"),
            root=bids_root,
            datatype="eeg",
        )
    else:
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

    # Try to find an accompanying events.tsv file for this recording via
    # the pybids layout. Prefer a file that matches subject/task/run.
    #
    # Important: when concatenating multiple runs, do NOT return a single
    # per-run events.tsv path — it would only describe one run's events
    # while the data spans them all. Return None and let downstream
    # epoching pull events from the concatenated raw's annotations
    # (read_raw_bids attaches each run's events as annotations and
    # concatenate_raws preserves them with correct time offsets).
    if isinstance(run_label, (list, tuple)) or (run_label is None and len(eeg_files) > 1):
        events_file = None
    else:
        ev_qp = {"subject": sub_label, "suffix": "events", "extension": ".tsv"}
        if session_label:
            ev_qp["session"] = session_label
        if task_label:
            ev_qp["task"] = task_label
        if run_label:
            ev_qp["run"] = run_label
        ev_found = layout.get(**ev_qp)
        events_file = ev_found[0].path if ev_found else None

    # Store original filename information for derivatives naming. If
    # original_filename was already set (e.g., when concatenating runs),
    # keep it. Otherwise derive it from the single-file eeg_file.
    if "original_filename" not in locals():
        from pathlib import Path

        original_filename = Path(eeg_file.filename).stem  # Remove extension

    # Return data, bids_path, original filename info, and optional events file
    return data, bids_path, original_filename, events_file


def load_data_to_fif(
    bids_root=None, sub_label=None, session_label=None, task_label=None, run_label=None, output_dir=None
):
    """
    Find BIDS raw file(s), read them and save a single FIF copy to the
    derivatives directory. Returns the path to the saved FIF file along
    with the BIDSPath and original filename base.

    This helper is intended for disk-backed workflows where we avoid
    passing large Raw objects between Nipype nodes.
    """
    # Local imports for nipype subprocess context (deps guaranteed by
    # pyproject.toml).
    from bids import BIDSLayout
    from mne_bids import BIDSPath, read_raw_bids

    layout = BIDSLayout(bids_root, validate=False)

    # Build base query params and support run_label as list for selective
    # concatenation.
    base_qp = {"subject": sub_label, "datatype": "eeg", "extension": ".edf"}
    if session_label:
        base_qp["session"] = session_label
    if task_label:
        base_qp["task"] = task_label

    eeg_files = []
    if isinstance(run_label, (list, tuple)):
        # Collect one file per requested run (warn if some runs missing)
        for rl in run_label:
            qp = dict(base_qp)
            if rl:
                qp["run"] = rl
            found = layout.get(**qp)
            if not found:
                # Try alternative extensions for this run
                for ext in [".bdf", ".vhdr", ".fif", ".set"]:
                    qp["extension"] = ext
                    found = layout.get(**qp)
                    if found:
                        break
            if found:
                eeg_files.append(found[0])
            else:
                print(f"Warning: No EEG file found for run {rl} (subject {sub_label})")
    else:
        qp = dict(base_qp)
        if run_label:
            qp["run"] = run_label
        eeg_files = layout.get(**qp)
        if not eeg_files:
            for ext in [".bdf", ".vhdr", ".fif", ".set"]:
                qp["extension"] = ext
                eeg_files = layout.get(**qp)
                if eeg_files:
                    break

    if not eeg_files:
        raise FileNotFoundError("No EEG files found for disk-backed load")

    # If multiple files were collected (e.g., selective runs), read each
    # and concatenate them into a single Raw object before saving.
    if len(eeg_files) > 1:
        raws = []
        first_entities = None
        for f in eeg_files:
            entities_f = layout.parse_file_entities(f.path)
            if first_entities is None:
                first_entities = entities_f
            bp = BIDSPath(
                subject=entities_f.get("subject"),
                session=entities_f.get("session"),
                task=entities_f.get("task"),
                run=entities_f.get("run"),
                root=bids_root,
                datatype="eeg",
            )
            raws.append(read_raw_bids(bids_path=bp, verbose=False))

        from mne import concatenate_raws

        raw = concatenate_raws(raws)
        # Build conservative original filename base
        original_filename = f"sub-{first_entities.get('subject')}_task-{first_entities.get('task')}"
        entities = first_entities
    else:
        # Choose first file (if multiple) and read then save to FIF
        eeg_file = eeg_files[0]
        entities = layout.parse_file_entities(eeg_file.path)
        bp = BIDSPath(
            subject=entities.get("subject"),
            session=entities.get("session"),
            task=entities.get("task"),
            run=entities.get("run"),
            root=bids_root,
            datatype="eeg",
        )

        raw = read_raw_bids(bids_path=bp, verbose=False)

        # Build conservative original filename base
        original_filename = f"sub-{entities.get('subject')}_task-{entities.get('task')}"

    # Save to derivatives subject folder under output_dir
    from pathlib import Path

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    parts = [f"sub-{entities.get('subject')}"]
    if entities.get("session"):
        parts.append(f"ses-{entities.get('session')}")
    parts.append(f"task-{entities.get('task')}")
    if entities.get("run"):
        parts.append(f"run-{entities.get('run')}")
    parts.append("desc-loaded_raw.fif")
    fname = "_".join(parts)
    outpath = outdir / fname

    # Save as FIF
    raw.save(outpath, overwrite=True)

    # Try to find an accompanying events.tsv file. Same caveat as the
    # in-memory load_data: when concatenating runs, return None so
    # downstream epoching uses concatenated annotations rather than a
    # single per-run events.tsv (which would only describe one run).
    if isinstance(run_label, (list, tuple)) or (run_label is None and len(eeg_files) > 1):
        events_file = None
    else:
        ev_qp = {"subject": sub_label, "suffix": "events", "extension": ".tsv"}
        if session_label:
            ev_qp["session"] = session_label
        if task_label:
            ev_qp["task"] = task_label
        if run_label:
            ev_qp["run"] = run_label
        ev_found = layout.get(**ev_qp)
        events_file = ev_found[0].path if ev_found else None

    # Return path, a minimal BIDSPath-like object, filename base, and events file
    return str(outpath), bp, original_filename, events_file


def save_raw_helper(eeg_data, subject, task=None, session=None, run=None, desc="referenced", output_dir=None):
    """
    Save an MNE Raw object to a FIF file under the derivatives subject dir
    with a descriptor (e.g., 'referenced', 'filtered'). Returns the path
    to the saved file.
    """
    from pathlib import Path

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    filename_parts = [f"sub-{subject}"]
    # session may be None
    if session:
        filename_parts.append(f"ses-{session}")
    filename_parts.append(f"task-{task}")
    # Skip the run token for concatenated runs (run is a list/tuple) — the
    # output is one merged file with no single run identifier.
    if run and not isinstance(run, (list, tuple)):
        filename_parts.append(f"run-{run}")
    filename_parts.append(f"desc-{desc}_raw.fif")
    filename = "_".join(filename_parts)

    output_path = outdir / filename
    # Save Raw
    eeg_data.save(output_path, overwrite=True)
    return str(output_path)


def reference_raw_file(input_raw_path, ref_channels, output_dir, subject=None, task=None, session=None, run=None):
    """
    Load a raw FIF file, re-reference, and save a referenced FIF. Returns
    the saved path.
    """
    import mne
    from pathlib import Path

    raw = mne.io.read_raw_fif(input_raw_path, preload=True, verbose=False)
    # Local imports keep these names resolvable inside Nipype Function
    # subprocesses (where module-level globals may not be bound).
    from ffrprep.preproc import reference_data as _reference_data
    from ffrprep.preproc import save_raw_helper as _save_raw_helper

    referenced = _reference_data(raw, ref_channels)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Derive simple naming from file path if possible
    # Use save_raw_helper to keep naming consistent
    # Attempt to extract subject/task/run from filename
    import re

    m = re.search(r"sub-([^_]+)", Path(input_raw_path).name)
    subj = m.group(1) if m else subject
    m2 = re.search(r"task-([^_]+)", Path(input_raw_path).name)
    t = m2.group(1) if m2 else task
    m3 = re.search(r"ses-([^_]+)", Path(input_raw_path).name)
    ses = m3.group(1) if m3 else session
    m4 = re.search(r"run-([^_]+)", Path(input_raw_path).name)
    r = m4.group(1) if m4 else run

    return _save_raw_helper(referenced, subj, task=t, session=ses, run=r, desc="referenced", output_dir=outdir)


def filter_raw_file(
    input_raw_path, high_pass, low_pass, output_dir, subject=None, task=None, session=None, run=None
):
    """
    Load a raw FIF file, apply filtering, save filtered FIF. Returns path.
    """
    import mne
    from pathlib import Path

    raw = mne.io.read_raw_fif(input_raw_path, preload=True, verbose=False)
    # Local imports keep these names resolvable inside Nipype Function
    # subprocesses (where module-level globals may not be bound).
    from ffrprep.preproc import filter_data as _filter_data
    from ffrprep.preproc import save_raw_helper as _save_raw_helper

    filtered = _filter_data(raw, high_pass=high_pass, low_pass=low_pass)

    # Derive identifiers from filename as in reference_raw_file
    import re

    m = re.search(r"sub-([^_]+)", Path(input_raw_path).name)
    subj = m.group(1) if m else subject
    m2 = re.search(r"task-([^_]+)", Path(input_raw_path).name)
    t = m2.group(1) if m2 else task
    m3 = re.search(r"ses-([^_]+)", Path(input_raw_path).name)
    ses = m3.group(1) if m3 else session
    m4 = re.search(r"run-([^_]+)", Path(input_raw_path).name)
    r = m4.group(1) if m4 else run

    return _save_raw_helper(filtered, subj, task=t, session=ses, run=r, desc="filtered", output_dir=output_dir)


def load_raw_for_epoch(input_raw_path):
    """
    Load a FIF raw file and return the MNE Raw object (used before epoching).
    """
    import mne

    raw = mne.io.read_raw_fif(input_raw_path, preload=True, verbose=False)
    return raw


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
    # Local import for nipype subprocess context.
    from bids import BIDSLayout

    layout = BIDSLayout(bids_root, validate=False)

    # Query for all participants with EEG data
    eeg_participants = layout.get_subjects(datatype="eeg")

    if participant_label:
        # Filter to requested participants
        available_participants = [p for p in participant_label if p in eeg_participants]

        # Warn about missing participants
        missing_participants = [p for p in participant_label if p not in eeg_participants]
        if missing_participants:
            print(f"Warning: Participants {missing_participants} not found " f"or have no EEG data in {bids_root}")

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
    # Local import for nipype subprocess context.
    from bids import BIDSLayout

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


def epoch_data(
    eeg_data,
    baseline,
    events_file=None,
    picks=None,
    tmin=None,
    tmax=None,
    reject=None,
    on_missing="warn",
    event_id=None,
    derivatives_root=None,
    subject=None,
    task=None,
    run=None,
    trial_types=None,
    verbose=True,
):
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

    # Local import for nipype subprocess context.
    from mne import Epochs

    # Load events from file or extract from data
    if events_file:
        # Load events from external file (e.g., BIDS events.tsv)
        import pandas as pd
        import numpy as np

        # Read events file
        events_df = pd.read_csv(events_file, sep="\t")

        # Convert onset times to sample indices
        sfreq = eeg_data.info["sfreq"]
        onset_samples = (events_df["onset"] * sfreq).astype(int)
        # Determine event codes robustly. BIDS `trial_type` is often a
        # string label (e.g., 'positive') — map labels to integer event
        # codes. Support user-provided `event_id` mapping, numeric `value`
        # or `event_id` columns, and fall back to a default code of 1.
        codes = None
        event_dict = None

        def _coerce_int(val):
            """Return ``val`` as int when it cleanly represents an integer.

            Handles ints, numpy ints, strings of digits (with optional
            leading minus sign), and floats whose textual form is
            integer-only (``"1"``, ``"1.0"``). Returns ``None`` for
            anything else, including booleans and non-integer floats.
            """
            if isinstance(val, (int, np.integer)) and not isinstance(val, bool):
                return int(val)
            s = str(val).strip()
            if not s:
                return None
            if s.lstrip("-").isdigit():
                return int(s)
            if s.lstrip("-").replace(".", "", 1).isdigit():
                f = float(s)
                if f.is_integer():
                    return int(f)
            return None

        # If caller provided an explicit event_id mapping, use it to map
        # trial_type values (or numeric values) to codes.
        if event_id is not None:
            mapping = {str(k): int(v) for k, v in event_id.items()}
            if "trial_type" in events_df.columns:
                codes = [mapping.get(str(v), _coerce_int(v) or 0) for v in events_df["trial_type"]]
            elif "value" in events_df.columns:
                codes = [mapping.get(str(v), _coerce_int(v) or 0) for v in events_df["value"]]
            elif "event_id" in events_df.columns:
                codes = [mapping.get(str(v), _coerce_int(v) or 0) for v in events_df["event_id"]]
            else:
                codes = [_coerce_int(v) or 1 for v in events_df.index]
            event_dict = mapping
        else:
            # No user mapping: create mapping from trial_type strings if present
            if "trial_type" in events_df.columns:
                unique_types = list(pd.unique(events_df["trial_type"]))
                event_dict = {str(t): idx + 1 for idx, t in enumerate(unique_types)}
                codes = [event_dict.get(str(v)) for v in events_df["trial_type"]]
            elif "value" in events_df.columns:
                # Numeric event values (already ints)
                codes = events_df["value"].apply(_coerce_int).fillna(1).astype(int).tolist()
            elif "event_id" in events_df.columns:
                codes = events_df["event_id"].apply(_coerce_int).fillna(1).astype(int).tolist()
            else:
                codes = [1] * len(events_df)

        # Ensure codes is a numpy int array
        codes = np.array(codes, dtype=int)

        # Create MNE events array [sample, prev_sample, event_id]
        events = np.column_stack(
            [
                onset_samples,
                np.zeros(len(onset_samples), dtype=int),  # previous sample
                codes,
            ]
        )
    else:
        # First, try to locate stimtrack-derived events in the derivatives
        # directory if a derivatives_root was provided. This is common when
        # a separate stimulus-tracking pipeline writes event files we can
        # reuse for epoching. Failures inside the stimtrack branch
        # propagate — a malformed stimtrack file is a real error to fix
        # upstream, not something to silently mask.
        events = None
        event_dict = None
        stim_matches = []
        if derivatives_root:
            import os
            from glob import glob

            deriv_dir = str(derivatives_root)
            stim_dir = os.path.join(deriv_dir, "events-stimtrack")
            if os.path.isdir(stim_dir):
                subj = str(subject) if subject is not None else ""
                task_k = str(task) if task is not None else ""
                # run may be a list (when concatenating) or a scalar
                if isinstance(run, (list, tuple)):
                    run_pat = "*"
                else:
                    run_pat = str(run) if run is not None else "*"
                pattern = os.path.join(stim_dir, f"*{subj}*{task_k}*{run_pat}_stimtrack_events.tsv")
                stim_matches = sorted(glob(pattern))

        if stim_matches:
            import pandas as pd
            import mne

            events_fpath = stim_matches[0]
            events_df = pd.read_csv(events_fpath, sep="\t")
            # Duration here set to 0.170s as in user example
            annot = mne.Annotations(
                onset=events_df["onset"].tolist(),
                duration=[0.170] * len(events_df),
                description=events_df["type"].astype(str).tolist(),
            )
            eeg_data.set_annotations(annot)
            events, event_dict = mne.events_from_annotations(eeg_data)
        else:
            # Prefer existing annotations on the raw — read_raw_bids attaches
            # each run's events.tsv as annotations, and concatenate_raws
            # preserves them with correct time offsets across runs. Fall
            # back to stim-channel find_events only if no annotations are
            # present (e.g., a non-BIDS Raw with just trigger codes).
            import mne

            existing_ann = getattr(eeg_data, "annotations", None)
            if existing_ann is not None and len(existing_ann) > 0:
                events, event_dict = mne.events_from_annotations(eeg_data, verbose=verbose)
                # User-provided mapping wins if supplied
                if event_id is not None:
                    event_dict = event_id
            else:
                from mne import find_events

                events = find_events(eeg_data, verbose=verbose)
                event_dict = event_id if event_id is not None else None

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

    # Create epochs with specified parameters. Use provided reject criteria
    # if available. If reject is None, no automatic amplitude-based
    # rejection is applied.
    # Use user-specified event_id if provided, otherwise fall back to
    # the event_dict discovered/constructed above.
    chosen_event_id = event_id if event_id is not None else event_dict

    # Narrow chosen_event_id to the user-requested subset of trial
    # types. Validate up front so a typo surfaces as an actionable
    # error rather than as silently empty epochs downstream.
    if trial_types is not None:
        if chosen_event_id is None:
            raise ValueError(
                "trial_types was supplied but no event_id mapping is "
                "available; cannot subset trials by name."
            )
        requested = list(trial_types)
        unknown = [t for t in requested if t not in chosen_event_id]
        if unknown:
            raise ValueError(
                "trial_types contains entries not present in the "
                f"discovered event_id mapping: {unknown}. Known: "
                f"{sorted(chosen_event_id.keys())}."
            )
        chosen_event_id = {k: chosen_event_id[k] for k in requested}

    epoched_data = Epochs(
        eeg_data,
        events=events,
        event_id=chosen_event_id,
        on_missing=on_missing,
        picks=picks,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline_window,
        verbose=verbose,
        reject=reject,
    )

    # drop_bad() will apply rejection heuristics and return the filtered
    # Epochs object; keep this separate for clarity and easier debugging.
    epoched_data = epoched_data.drop_bad()

    return epoched_data, (tmin, tmax)


def make_combined_evoked(epochs):
    """Average across every event in ``epochs``, ignoring event_id groups.

    Sets ``.comment = "combined"`` so downstream reports can label this
    section distinctly from the per-condition and difference outputs
    that share the same (subject, task, run) group.

    Parameters
    ----------
    epochs : mne.Epochs
        Source epochs. May contain multiple event types.

    Returns
    -------
    evoked : mne.Evoked
        Single Evoked across all events.
    """
    evoked = epochs.average()
    evoked.comment = "combined"
    return evoked


def make_difference_evokeds(evoked_dict, pairs=None):
    """Compute ``A - B`` Evokeds for one or more trial-type pairs.

    Parameters
    ----------
    evoked_dict : dict[str, mne.Evoked]
        Per-trial-type Evoked outputs, typically the dict returned by
        ``make_evoked(epochs, by_event_type=True)``.
    pairs : list[tuple[str, str]] | None
        Pairs to subtract. When None, the single ``(A, B)`` pair is
        auto-built from insertion order if and only if the dict has
        exactly two entries; otherwise an empty dict is returned and
        the caller is expected to opt in explicitly via ``pairs``.

    Returns
    -------
    diffs : dict[tuple[str, str], mne.Evoked]
        Tuple-keyed dict. Each Evoked has
        ``.comment = "diff_{A}Vs{B}"`` so reports can label by pair.

    Raises
    ------
    KeyError
        If any requested pair references a trial type missing from
        ``evoked_dict``.
    """
    import mne as _mne

    if pairs is None:
        if len(evoked_dict) != 2:
            return {}
        keys = list(evoked_dict.keys())
        pairs = [(keys[0], keys[1])]

    diffs = {}
    for a, b in pairs:
        if a not in evoked_dict or b not in evoked_dict:
            raise KeyError(
                f"Difference pair ({a!r}, {b!r}) references a trial "
                f"type missing from evoked_dict (have: "
                f"{sorted(evoked_dict.keys())})."
            )
        diff = _mne.combine_evoked(
            [evoked_dict[a], evoked_dict[b]], weights=[1, -1],
        )
        diff.comment = f"diff_{a}Vs{b}"
        diffs[(a, b)] = diff
    return diffs


def make_evoked(epochs, by_event_type: bool = True):
    """
    Create evoked responses by averaging epochs.

    Parameters
    ----------
    epochs : mne.Epochs
        MNE `Epochs` object containing time-locked epochs.
    by_event_type : bool, default=True
        If True (default), return a dict with one Evoked per event type
        (each Evoked's ``.comment`` is set to the event name so
        downstream reports show meaningful condition labels).
        If False, average all epochs together into a single Evoked.

    Returns
    -------
    evoked : mne.Evoked or dict of mne.Evoked
        Single MNE `Evoked` when ``by_event_type=False``; otherwise a
        dict keyed by event name.
    """
    if by_event_type:
        evoked = dict()
        for event_name, event_id in epochs.event_id.items():
            epochs_subset = epochs[event_name]
            ev = epochs_subset.average()
            ev.comment = event_name
            evoked[event_name] = ev
    else:
        evoked = epochs.average()

    return evoked


def build_analysis_payload(epochs, by_event_type=True, difference_pairs=None):
    """Compute the structured ``{by_type, combined, diff}`` analysis shape.

    Always emits a ``combined`` Evoked. When ``by_event_type`` is True,
    also emits per-trial-type Evokeds and (when applicable) one or more
    difference Evokeds. Difference computation follows the rules of
    :func:`make_difference_evokeds`: auto-paired only when there are
    exactly two trial types and ``difference_pairs is None``; for >2
    types the caller must opt in via ``difference_pairs``.

    Parameters
    ----------
    epochs : mne.Epochs
        Source epochs (typically loaded from a single per-condition
        preprocessing output, or stacked across conditions).
    by_event_type : bool, default True
        Emit per-trial-type and difference outputs in addition to the
        combined Evoked.
    difference_pairs : list[tuple[str, str]] | None
        Forwarded to :func:`make_difference_evokeds`.

    Returns
    -------
    payload : dict
        A subset of ``{"by_type", "combined", "diff"}`` populated based
        on ``by_event_type`` and the pair-discovery rules above.
    """
    payload = {"combined": make_combined_evoked(epochs)}
    if by_event_type:
        by_type = make_evoked(epochs, by_event_type=True)
        payload["by_type"] = by_type
        diffs = make_difference_evokeds(by_type, pairs=difference_pairs)
        if diffs:
            payload["diff"] = diffs
    return payload


def create_preprocessing_workflow(name="ffrprep_preproc", disk_backed=False):
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
    # Avoid nipype/etelemetry network checks when constructing workflows
    # in restricted environments (CI, isolated shells, or tests). Set
    # these env vars here so callers that simply import or instantiate
    # workflows don't hang attempting to contact external services.
    import os

    os.environ.setdefault("ETEL_NO_CHECK", "1")
    os.environ.setdefault("NIPYPE_NO_ET", "1")

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
                "reject",
                "ref_channels",
                "high_pass",
                "low_pass",
                "baseline",
                "tmin",
                "tmax",
                "derivatives_root",
                "picks",
                "on_missing",
                "event_id",
                "events_file",
                "trial_types",
                "split_by_trial_type",
                "output_dir",
            ]
        ),
        name="inputnode",
    )

    # Output node to collect results
    outputnode = Node(
        niu.IdentityInterface(fields=["epochs", "preprocessing_report", "original_filename"]), name="outputnode"
    )

    # Load data node: choose file-based loader when disk_backed requested
    if disk_backed:
        load_node = Node(
            Function(
                input_names=["bids_root", "sub_label", "session_label", "task_label", "run_label", "output_dir"],
                output_names=["eeg_fif_path", "bids_path", "original_filename", "events_file"],
                function=load_data_to_fif,
            ),
            name="load_data_to_fif",
        )
    else:
        load_node = Node(
            Function(
                input_names=["bids_root", "sub_label", "session_label", "task_label", "run_label"],
                output_names=["eeg_data", "bids_path", "original_filename", "events_file"],
                function=load_data,
            ),
            name="load_data",
        )

    # Reference data node
    if disk_backed:
        # operate on FIF file paths and write referenced FIF
        reference_node = Node(
            Function(
                input_names=["input_raw_path", "ref_channels", "output_dir", "subject", "task", "session", "run"],
                output_names=["referenced_raw_path"],
                function=reference_raw_file,
            ),
            name="reference_file",
        )
    else:
        reference_node = Node(
            Function(
                input_names=["eeg_data", "ref_channels"], output_names=["referenced_data"], function=reference_data
            ),
            name="reference_data",
        )

    # Filter data node
    if disk_backed:
        filter_node = Node(
            Function(
                input_names=[
                    "input_raw_path",
                    "high_pass",
                    "low_pass",
                    "output_dir",
                    "subject",
                    "task",
                    "session",
                    "run",
                ],
                output_names=["filtered_raw_path"],
                function=filter_raw_file,
            ),
            name="filter_file",
        )
    else:
        filter_node = Node(
            Function(
                input_names=["eeg_data", "high_pass", "low_pass"],
                output_names=["filtered_data"],
                function=filter_data,
            ),
            name="filter_data",
        )

    # Epoch data node
    if disk_backed:
        # First load filtered FIF into memory, then epoch
        load_for_epoch = Node(
            Function(
                input_names=["input_raw_path"],
                output_names=["eeg_data"],
                function=load_raw_for_epoch,
            ),
            name="load_for_epoch",
        )

        epoch_node = Node(
            Function(
                input_names=[
                    "eeg_data",
                    "baseline",
                    "tmin",
                    "tmax",
                    "reject",
                    "events_file",
                    "picks",
                    "on_missing",
                    "event_id",
                    "derivatives_root",
                    "subject",
                    "task",
                    "run",
                    "trial_types",
                ],
                output_names=["epochs", "time_window"],
                function=epoch_data,
            ),
            name="epoch_data",
        )
    else:
        epoch_node = Node(
            Function(
                input_names=[
                    "eeg_data",
                    "baseline",
                    "tmin",
                    "tmax",
                    "reject",
                    "events_file",
                    "picks",
                    "on_missing",
                    "event_id",
                    "derivatives_root",
                    "subject",
                    "task",
                    "run",
                    "trial_types",
                ],
                output_names=["epochs", "time_window"],
                function=epoch_data,
            ),
            name="epoch_data",
        )

    # Save preprocessing outputs node
    # Note: save_preprocessing_node accepts an optional original_filename and
    # will attempt to infer `task` from that if `task` is not provided.
    save_node = Node(
        Function(
            input_names=[
                "epochs",
                "bids_root",
                "subject",
                "task",
                "original_filename",
                "session",
                "run",
                "output_dir",
                "split_by_trial_type",
            ],
            output_names=["output_path"],
            function=save_preprocessing_node,
        ),
        name="save_preprocessing",
    )

    # (Report generation is handled by the CLI using the canonical
    # `ffrprep.reports` module; no in-workflow report node is created here.)

    # Connect workflow nodes (different wiring for disk-backed mode)
    if disk_backed:
        workflow.connect(
            [
                # Input -> load_data_to_fif
                (
                    inputnode,
                    load_node,
                    [
                        ("bids_root", "bids_root"),
                        ("sub_label", "sub_label"),
                        ("session_label", "session_label"),
                        ("task_label", "task_label"),
                        ("run_label", "run_label"),
                        ("output_dir", "output_dir"),
                    ],
                ),
                # load -> reference file (takes eeg_fif_path)
                (load_node, reference_node, [("eeg_fif_path", "input_raw_path")]),
                (
                    inputnode,
                    reference_node,
                    [
                        ("ref_channels", "ref_channels"),
                        ("output_dir", "output_dir"),
                        ("sub_label", "subject"),
                        ("task_label", "task"),
                        ("session_label", "session"),
                        ("run_label", "run"),
                    ],
                ),
                # reference -> filter (file paths)
                (reference_node, filter_node, [("referenced_raw_path", "input_raw_path")]),
                (
                    inputnode,
                    filter_node,
                    [
                        ("high_pass", "high_pass"),
                        ("low_pass", "low_pass"),
                        ("output_dir", "output_dir"),
                        ("sub_label", "subject"),
                        ("task_label", "task"),
                        ("session_label", "session"),
                        ("run_label", "run"),
                    ],
                ),
                # filter -> load_for_epoch -> epoch
                (filter_node, load_for_epoch, [("filtered_raw_path", "input_raw_path")]),
                (load_for_epoch, epoch_node, [("eeg_data", "eeg_data")]),
                # Route events_file from the loader directly into the epoch
                # node. Do NOT connect loader -> inputnode as that would
                # create a circular dependency (inputnode -> load_node ->
                # inputnode). Keep inputnode -> epoch_node for other inputs
                # but omit events_file here.
                (load_node, epoch_node, [("events_file", "events_file")]),
                (
                    inputnode,
                    epoch_node,
                    [
                        ("baseline", "baseline"),
                        ("tmin", "tmin"),
                        ("tmax", "tmax"),
                        ("reject", "reject"),
                        ("picks", "picks"),
                        ("on_missing", "on_missing"),
                        ("event_id", "event_id"),
                        ("trial_types", "trial_types"),
                        ("derivatives_root", "derivatives_root"),
                        ("sub_label", "subject"),
                        ("task_label", "task"),
                        ("run_label", "run"),
                    ],
                ),
                # epoch -> save
                (epoch_node, save_node, [("epochs", "epochs")]),
                (
                    inputnode,
                    save_node,
                    [
                        ("bids_root", "bids_root"),
                        ("sub_label", "subject"),
                        ("task_label", "task"),
                        ("session_label", "session"),
                        ("run_label", "run"),
                        # derivatives_root is the BIDS-App output_dir
                        # (e.g. /data/derivatives_concat) — the save
                        # function uses it as the derivatives root and
                        # constructs the per-subject path internally.
                        # The inputnode's "output_dir" field carries
                        # the SUBJECT-level eeg dir for intermediate
                        # disk-backed saves; that's a different value.
                        ("derivatives_root", "output_dir"),
                        ("split_by_trial_type", "split_by_trial_type"),
                    ],
                ),
                (load_node, save_node, [("original_filename", "original_filename")]),
                (load_node, outputnode, [("events_file", "events_file")]),
                # outputs
                (epoch_node, outputnode, [("epochs", "epochs")]),
                (load_node, outputnode, [("original_filename", "original_filename")]),
            ]
        )
    else:
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
                (load_node, epoch_node, [("events_file", "events_file")]),
                (inputnode, reference_node, [("ref_channels", "ref_channels")]),
                # Connect reference to filter
                (reference_node, filter_node, [("referenced_data", "eeg_data")]),
                (inputnode, filter_node, [("high_pass", "high_pass"), ("low_pass", "low_pass")]),
                # Connect filter to epoch
                (filter_node, epoch_node, [("filtered_data", "eeg_data")]),
                (
                    inputnode,
                    epoch_node,
                    [
                        ("baseline", "baseline"),
                        ("tmin", "tmin"),
                        ("tmax", "tmax"),
                        ("reject", "reject"),
                        ("picks", "picks"),
                        ("on_missing", "on_missing"),
                        ("event_id", "event_id"),
                        ("trial_types", "trial_types"),
                        ("derivatives_root", "derivatives_root"),
                        ("sub_label", "subject"),
                        ("task_label", "task"),
                        ("run_label", "run"),
                    ],
                ),
                # Connect to save node
                (epoch_node, save_node, [("epochs", "epochs")]),
                (
                    inputnode,
                    save_node,
                    [
                        ("bids_root", "bids_root"),
                        ("sub_label", "subject"),
                        ("task_label", "task"),
                        ("session_label", "session"),
                        ("run_label", "run"),
                        # derivatives_root is the BIDS-App output_dir
                        # (e.g. /data/derivatives_concat) — the save
                        # function uses it as the derivatives root and
                        # constructs the per-subject path internally.
                        # The inputnode's "output_dir" field carries
                        # the SUBJECT-level eeg dir for intermediate
                        # disk-backed saves; that's a different value.
                        ("derivatives_root", "output_dir"),
                        ("split_by_trial_type", "split_by_trial_type"),
                    ],
                ),
                (
                    load_node,
                    save_node,
                    [("original_filename", "original_filename")],
                ),
                # Connect to output
                (epoch_node, outputnode, [("epochs", "epochs")]),
                (load_node, outputnode, [("original_filename", "original_filename")]),
            ]
        )
        # (No in-workflow reporting node for in-memory mode either.)

    return workflow


def create_analysis_workflow(name="ffrprep_analysis"):
    """
    Build a nipype workflow that averages epochs into evoked responses
    and saves them to BIDS-derivatives.

    Parameters
    ----------
    name : str
        Workflow name. Default: "ffrprep_analysis".

    Returns
    -------
    workflow : nipype.Workflow
        Nipype workflow with an inputnode (fields: epochs, by_event_type,
        difference_pairs, bids_root, subject, output_dir, derivatives_root,
        original_filename) and an outputnode (fields: evoked,
        analysis_report). The inputnode's ``evoked`` output carries
        the structured ``{by_type, combined, diff}`` payload built
        by :func:`build_analysis_payload`.
    """
    from nipype import Workflow, Node, Function
    from nipype.interfaces import utility as niu

    workflow = Workflow(name=name)

    inputnode = Node(
        niu.IdentityInterface(
            fields=[
                "epochs", "by_event_type", "difference_pairs",
                "bids_root", "subject",
                "output_dir", "derivatives_root", "original_filename",
            ]
        ),
        name="inputnode",
    )

    outputnode = Node(niu.IdentityInterface(fields=["evoked", "analysis_report"]), name="outputnode")

    build_payload_node = Node(
        Function(
            input_names=["epochs", "by_event_type", "difference_pairs"],
            output_names=["evoked"],
            function=build_analysis_payload,
        ),
        name="build_analysis_payload",
    )

    save_analysis_node_func = Node(
        Function(
            input_names=[
                "evoked", "bids_root", "subject", "original_filename",
                "output_dir",
            ],
            output_names=["output_paths"],
            function=save_analysis_node,
        ),
        name="save_analysis",
    )

    workflow.connect(
        [
            (
                inputnode, build_payload_node,
                [
                    ("epochs", "epochs"),
                    ("by_event_type", "by_event_type"),
                    ("difference_pairs", "difference_pairs"),
                ],
            ),
            (build_payload_node, save_analysis_node_func, [("evoked", "evoked")]),
            (
                inputnode,
                save_analysis_node_func,
                [
                    ("bids_root", "bids_root"),
                    ("subject", "subject"),
                    ("original_filename", "original_filename"),
                    # derivatives_root is the BIDS-App output_dir; the
                    # save node uses it as the derivatives root and
                    # constructs the per-subject path internally.
                    ("derivatives_root", "output_dir"),
                ],
            ),
            (build_payload_node, outputnode, [("evoked", "evoked")]),
        ]
    )

    return workflow


def setup_derivatives_directories(
    bids_root, subject,
    create_preprocessing=True, create_analysis=True,
    output_dir=None,
):
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
    output_dir : str or pathlib.Path, optional
        Explicit destination root for the ffrprep-preprocessing and
        ffrprep-analysis subtrees. When omitted, falls back to the
        BIDS-conventional ``bids_root / "derivatives"``. The CLI
        passes its second positional argument here so users can
        redirect outputs to a separate path (e.g. to keep different
        runs from clobbering each other's reports / logs).

    Returns
    -------
    derivatives_info : dict
        Dictionary containing paths to derivatives directories.
    """
    from pathlib import Path

    bids_root = Path(bids_root)
    if output_dir is not None:
        derivatives_root = Path(output_dir)
    else:
        derivatives_root = bids_root / "derivatives"
    derivatives_root.mkdir(parents=True, exist_ok=True)

    # Always resolve canonical paths so callers can locate inputs/outputs
    # of the *other* stage (e.g. analysis-only mode reads existing
    # preprocessing outputs from preprocessing_subject_dir). The
    # create_* flags only control whether the directory is materialized.
    preproc_dir = derivatives_root / "ffrprep-preprocessing"
    preproc_subject_eeg_dir = preproc_dir / f"sub-{subject}" / "eeg"
    if create_preprocessing:
        preproc_subject_eeg_dir.mkdir(parents=True, exist_ok=True)

    analysis_dir = derivatives_root / "ffrprep-analysis"
    analysis_subject_dir = analysis_dir / f"sub-{subject}"
    if create_analysis:
        analysis_subject_dir.mkdir(parents=True, exist_ok=True)

    return {
        "derivatives_root": derivatives_root,
        "preprocessing_dir": preproc_dir,
        "analysis_dir": analysis_dir,
        "preprocessing_subject_dir": preproc_subject_eeg_dir,
        "analysis_subject_dir": analysis_subject_dir,
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
    preproc_dir = bids_root / "derivatives" / "ffrprep-preprocessing" / f"sub-{subject}"

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


def _build_preproc_filename(subject, session, task, run, condition=None):
    """Build the BIDS basename for a saved preprocessing epochs file.

    ``condition`` (when given) is appended to the ``desc-preproc``
    segment as e.g. ``desc-preprocPos`` so per-trial-type outputs
    sit next to each other in the derivatives directory.
    """
    parts = [f"sub-{subject}"]
    if session:
        parts.append(f"ses-{session}")
    parts.append(f"task-{task}")
    # Run token is dropped for concatenated runs (the merged output has
    # no single run identifier).
    if run and not isinstance(run, (list, tuple)):
        parts.append(f"run-{run}")
    desc = "desc-preproc"
    if condition is not None:
        desc += str(condition).capitalize()
    parts.append(f"{desc}_epo.fif")
    return "_".join(parts)


def _write_preproc_dataset_description(preprocessing_dir):
    """Write ``dataset_description.json`` at the preprocessing root once."""
    import json as _json

    dataset_desc_path = preprocessing_dir / "dataset_description.json"
    if dataset_desc_path.exists():
        return
    dataset_desc = {
        "Name": "ffrprep preprocessing outputs",
        "BIDSVersion": "1.6.0",
        "GeneratedBy": [
            {
                "Name": "ffrprep",
                "Description": "Frequency-following response preprocessing pipeline",
            }
        ],
    }
    with open(dataset_desc_path, "w") as f:
        _json.dump(dataset_desc, f, indent=2)


def _save_one_preproc_epochs(
    epochs, output_path, subject, task, session, run, condition=None,
):
    """Persist a single Epochs object + matching JSON sidecar.

    Reused by :func:`save_preprocessing_outputs` for both the scalar
    and the per-condition (dict input) code paths. ``condition``, when
    given, is recorded in the sidecar's ``Condition`` field.
    """
    import json as _json

    if len(epochs) == 0:
        raise ValueError(
            "Cannot save preprocessing outputs: the epochs object is "
            "empty. All candidate epochs were rejected. Common causes: "
            "trigger channels (e.g. Erg1) are typed as 'eeg' rather "
            "than 'stim' and exceed --reject-eeg every epoch. Fixes: "
            "pass --no-auto-reject, raise --reject-eeg, or correct the "
            "channel types in the source dataset."
        )

    # Capture rejection + baseline metadata from the original Epochs
    # object before reconstruction. drop_log, .reject thresholds, and
    # the baseline window are all lost when EpochsArray is constructed
    # below (the constructor doesn't take any of them).
    n_total_epochs = len(epochs.drop_log)
    n_accepted = len(epochs)
    n_rejected_epochs = n_total_epochs - n_accepted
    reject_thresholds = getattr(epochs, "reject", None)
    baseline_window = getattr(epochs, "baseline", None)

    import mne as _mne

    data = epochs.get_data()
    info = epochs.info.copy()
    tmin = getattr(epochs, "tmin", None)

    new_epochs = _mne.EpochsArray(data, info, tmin=tmin)
    new_epochs.save(output_path, overwrite=True)

    filename = output_path.name
    raw_basename = filename.split("_desc-")[0] + "_eeg.bdf"
    sidecar = {
        "Description": "FFR preprocessed epochs (referenced, filtered, baseline-corrected).",
        "GeneratedBy": [
            {
                "Name": "ffrprep",
                "Description": "Frequency-following response preprocessing pipeline",
            }
        ],
        "Sources": [f"bids:raw:sub-{subject}/eeg/{raw_basename}"],
        "RawSources": [f"sub-{subject}/eeg/{raw_basename}"],
        "TaskName": task,
        "SamplingFrequency": float(info["sfreq"]),
        "EpochCount": int(len(new_epochs)),
        "EpochCountTotal": int(n_total_epochs),
        "EpochCountRejected": int(n_rejected_epochs),
        "EpochTmin": float(new_epochs.tmin),
        "EpochTmax": float(new_epochs.tmax),
        "Channels": list(new_epochs.ch_names),
        "Filtering": {
            "HighpassFilterFrequency": (
                float(info["highpass"]) if info.get("highpass") is not None else None
            ),
            "LowpassFilterFrequency": (
                float(info["lowpass"]) if info.get("lowpass") is not None else None
            ),
        },
    }
    if reject_thresholds:
        sidecar["RejectionThresholds"] = {
            k: float(v) for k, v in reject_thresholds.items()
        }
    if baseline_window is not None:
        sidecar["Baseline"] = [
            float(baseline_window[0]), float(baseline_window[1]),
        ]
    if session is not None:
        sidecar["Session"] = str(session)
    if run is not None:
        if isinstance(run, (list, tuple)):
            sidecar["ConcatenatedRuns"] = [str(r) for r in run]
        else:
            sidecar["Run"] = str(run)
    if condition is not None:
        sidecar["Condition"] = str(condition)

    with open(output_path.with_suffix(".json"), "w") as f:
        _json.dump(sidecar, f, indent=2)


def save_preprocessing_outputs(epochs, bids_root, subject, task, session=None, run=None, output_dir=None):
    """
    Save preprocessing outputs to BIDS derivatives structure.

    Parameters
    ----------
    epochs : mne.Epochs or dict[str, mne.Epochs]
        Epoched data to save. When a dict is passed (one key per
        trial type), one BIDS file is written per condition with
        the ``_desc-preproc{Condition}_epo.fif`` filename pattern,
        and the per-file sidecar carries a ``Condition`` field.
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
    output_path : pathlib.Path or list[pathlib.Path]
        Single Path for scalar Epochs input; list of Paths (one per
        condition) for dict input.
    """
    derivatives_info = setup_derivatives_directories(
        bids_root, subject, create_preprocessing=True, create_analysis=False,
        output_dir=output_dir,
    )
    subject_dir = derivatives_info["preprocessing_subject_dir"]
    _write_preproc_dataset_description(derivatives_info["preprocessing_dir"])

    if isinstance(epochs, dict):
        out_paths = []
        for condition, epochs_obj in epochs.items():
            filename = _build_preproc_filename(
                subject, session, task, run, condition=condition,
            )
            output_path = subject_dir / filename
            _save_one_preproc_epochs(
                epochs_obj, output_path,
                subject, task, session, run,
                condition=condition,
            )
            out_paths.append(output_path)
        return out_paths

    filename = _build_preproc_filename(subject, session, task, run)
    output_path = subject_dir / filename
    _save_one_preproc_epochs(
        epochs, output_path, subject, task, session, run,
    )
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
    import mne

    bids_root = Path(bids_root)
    preproc_dir = bids_root / "derivatives" / "ffrprep-preprocessing" / f"sub-{subject}"

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
                print(f"Warning: Multiple files found, using " f"{epoch_files[0].name}")
            return mne.read_epochs(epoch_files[0])

    raise FileNotFoundError(f"No preprocessing outputs found for subject {subject} in " f"{preproc_dir}")


_STRUCTURED_PAYLOAD_KEYS = {"by_type", "combined", "diff"}


def _normalize_evoked_payload(evoked):
    """Coerce ``evoked`` into the structured ``{by_type, combined, diff}`` shape.

    A scalar Evoked is treated as the ``combined`` output. A dict whose
    keys are exactly the structured-payload keys is returned as-is.
    Any other dict shape (legacy ``{condition_name: Evoked}``) is
    rejected with a clear error so callers migrate.
    """
    if isinstance(evoked, dict):
        keys = set(evoked.keys())
        if keys.issubset(_STRUCTURED_PAYLOAD_KEYS):
            return evoked
        raise ValueError(
            "save_analysis_outputs no longer accepts the legacy "
            "{condition: Evoked} dict shape. Wrap per-condition "
            'evokeds in {"by_type": <dict>} (and optionally add '
            '"combined" / "diff" entries).'
        )
    return {"combined": evoked}


def _make_analysis_filename_base(subject, task, session, run):
    """Build the BIDS basename (no desc segment) for analysis outputs."""
    parts = [f"sub-{subject}"]
    if session:
        parts.append(f"ses-{session}")
    parts.append(f"task-{task}")
    if run and not isinstance(run, (list, tuple)):
        parts.append(f"run-{run}")
    return "_".join(parts)


def _write_evoked_sidecar(
    evoked_obj, evoked_path, task, analysis_type,
    session=None, run=None, condition=None, difference_of=None,
):
    """Write a BIDS-derivatives JSON sidecar for an evoked .fif file.

    ``condition`` (per-type) and ``difference_of`` (a 2-tuple/list of
    condition names) are mutually exclusive. The combined output omits
    both.
    """
    import json

    sidecar = {
        "Description": "FFR evoked response (averaged epochs).",
        "GeneratedBy": [
            {
                "Name": "ffrprep",
                "Description": "Frequency-following response analysis pipeline",
            }
        ],
        "TaskName": task,
        "AnalysisType": analysis_type,
        "SamplingFrequency": float(evoked_obj.info["sfreq"]),
        "AverageCount": int(getattr(evoked_obj, "nave", 0)),
        "Tmin": float(evoked_obj.tmin),
        "Tmax": float(evoked_obj.tmax),
        "Channels": list(evoked_obj.ch_names),
    }
    # Evoked.save() does NOT write evoked.baseline into the .fif, so
    # without this field downstream consumers see baseline=None after a
    # load round-trip and lose access to baseline-anchored metrics
    # (e.g. RMS SNR).
    baseline = getattr(evoked_obj, "baseline", None)
    if baseline is not None:
        sidecar["Baseline"] = [float(baseline[0]), float(baseline[1])]
    if session is not None:
        sidecar["Session"] = str(session)
    if run is not None:
        if isinstance(run, (list, tuple)):
            sidecar["ConcatenatedRuns"] = [str(r) for r in run]
        else:
            sidecar["Run"] = str(run)
    if condition is not None:
        sidecar["Condition"] = str(condition)
    if difference_of is not None:
        sidecar["DifferenceOf"] = [str(difference_of[0]), str(difference_of[1])]
    with open(evoked_path.with_suffix(".json"), "w") as f:
        json.dump(sidecar, f, indent=2)


def save_analysis_outputs(
    evoked, bids_root, subject, task,
    session=None, run=None, analysis_type="evoked", output_dir=None,
):
    """
    Save analysis outputs to BIDS derivatives structure.

    Parameters
    ----------
    evoked : mne.Evoked or dict
        Either a single Evoked (treated as the combined output) or a
        structured payload with keys drawn from
        ``{"by_type", "combined", "diff"}`` (any subset).
        ``by_type`` is ``dict[str, Evoked]`` keyed by trial-type name;
        ``combined`` is a single ``mne.Evoked``; ``diff`` is
        ``dict[tuple[str, str], Evoked]`` keyed by ``(A, B)``.
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
        Paths to the saved files. Per-type entries use
        ``_desc-{analysis_type}{Cond}.fif``; combined uses
        ``_desc-{analysis_type}.fif``; diff uses
        ``_desc-{analysis_type}Diff{A}Vs{B}.fif``.
    """
    payload = _normalize_evoked_payload(evoked)

    derivatives_info = setup_derivatives_directories(
        bids_root, subject, create_preprocessing=False, create_analysis=True,
        output_dir=output_dir,
    )
    subject_dir = derivatives_info["analysis_subject_dir"]
    filename_base = _make_analysis_filename_base(subject, task, session, run)
    output_paths = []

    by_type = payload.get("by_type") or {}
    for condition, evoked_obj in by_type.items():
        cond = str(condition).capitalize()
        path = subject_dir / f"{filename_base}_desc-{analysis_type}{cond}.fif"
        evoked_obj.save(path)
        _write_evoked_sidecar(
            evoked_obj, path, task, analysis_type,
            session=session, run=run, condition=condition,
        )
        output_paths.append(path)

    combined = payload.get("combined")
    if combined is not None:
        path = subject_dir / f"{filename_base}_desc-{analysis_type}.fif"
        combined.save(path)
        _write_evoked_sidecar(
            combined, path, task, analysis_type,
            session=session, run=run,
        )
        output_paths.append(path)

    diff = payload.get("diff") or {}
    for (a, b), evoked_obj in diff.items():
        a_cap, b_cap = str(a).capitalize(), str(b).capitalize()
        path = subject_dir / (
            f"{filename_base}_desc-{analysis_type}Diff{a_cap}Vs{b_cap}.fif"
        )
        evoked_obj.save(path)
        _write_evoked_sidecar(
            evoked_obj, path, task, analysis_type,
            session=session, run=run, difference_of=(a, b),
        )
        output_paths.append(path)

    import json as _json

    dataset_desc_path = derivatives_info["analysis_dir"] / "dataset_description.json"
    if not dataset_desc_path.exists():
        dataset_desc = {
            "Name": "ffrprep analysis outputs",
            "BIDSVersion": "1.6.0",
            "GeneratedBy": [
                {"Name": "ffrprep", "Description": "Frequency-following response analysis pipeline"}  # noqa: E501
            ],
        }
        with open(dataset_desc_path, "w") as f:
            _json.dump(dataset_desc, f, indent=2)

    return output_paths


def save_preprocessing_node(
    epochs, bids_root, subject, task=None, original_filename=None,
    session=None, run=None, output_dir=None, split_by_trial_type=True,
):
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
    split_by_trial_type : bool, default True
        When True (default) split ``epochs`` by ``event_id`` and write
        one ``_desc-preproc{Cond}_epo.fif`` per trial type; the return
        becomes a list of paths. When False, fall back to a single
        ``_desc-preproc_epo.fif`` file.

    Returns
    -------
    output_path : str or list[str]
        Single path under ``split_by_trial_type=False``; list of paths
        (one per trial type) under ``split_by_trial_type=True``.
    """
    # If task is not provided, try to infer it from the original filename
    if task is None and original_filename:
        import re

        m = re.search(r"task-([^_]+)", original_filename)
        if m:
            task = m.group(1)

    if task is None:
        raise ValueError(
            "Task label is required to save preprocessing outputs; "
            "provide a task_label input or ensure original_filename "
            "contains 'task-<label>'."
        )

    # Always go through the imported module so this works whether the call
    # happens in the host process (where save_preprocessing_outputs is in
    # globals) or inside a Nipype Function subprocess (where the
    # module-level name may not be bound).
    from importlib import import_module

    mod = import_module("ffrprep.preproc")
    # When split_by_trial_type=True, fan the multi-event Epochs out
    # into a {condition: epochs[condition]} dict so save_preprocessing_outputs
    # writes one BIDS-derivative file per trial type. event_id ordering
    # determines the on-disk file order.
    if split_by_trial_type and getattr(epochs, "event_id", None):
        epochs_arg = {
            cond: epochs[cond] for cond in epochs.event_id
        }
    else:
        epochs_arg = epochs

    output_path = mod.save_preprocessing_outputs(
        epochs_arg, bids_root, subject, task, session, run,
        output_dir=output_dir,
    )

    if isinstance(output_path, list):
        return [str(p) for p in output_path]
    return str(output_path)


def save_analysis_node(evoked, bids_root, subject, original_filename, analysis_type="evoked", output_dir=None):
    """
    Nipype-compatible wrapper that saves analysis outputs.

    Parameters
    ----------
    evoked : mne.Evoked or dict of mne.Evoked
        Evoked data to save.
    bids_root : str
        Path to the BIDS dataset root directory.
    subject : str
        Subject label (without 'sub-' prefix).
    original_filename : str
        BIDS basename of the source preprocessing output (e.g.
        ``sub-03_task-active_run-1_desc-preproc_epo``). The task,
        session, and run identifiers are parsed from this string.
    analysis_type : str
        Type of analysis output ('evoked', 'spectrum', etc.).

    Returns
    -------
    output_paths : list of str
        Paths to the saved files.
    """
    import re
    from importlib import import_module

    name = str(original_filename)
    task_match = re.search(r"task-([^_]+)", name)
    session_match = re.search(r"ses-([^_]+)", name)
    run_match = re.search(r"run-([^_]+)", name)
    if task_match is None:
        raise ValueError(
            f"Cannot infer task from original_filename {name!r}; expected "
            f"a BIDS basename containing 'task-<label>'."
        )
    task = task_match.group(1)
    session = session_match.group(1) if session_match else None
    run = run_match.group(1) if run_match else None

    # Always go through the imported module to handle both host and Nipype
    # Function subprocess execution (where module-level names may not be
    # bound in globals).
    mod = import_module("ffrprep.preproc")
    output_paths = mod.save_analysis_outputs(
        evoked, bids_root, subject, task, session, run, analysis_type,
        output_dir=output_dir,
    )
    return [str(p) for p in output_paths]
