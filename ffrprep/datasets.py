import os
import shutil
from pathlib import Path
import requests
import zipfile
from tqdm import tqdm


def _download_single_file(osf_url, output_path, save_name=None):
    """
    Helper function to download a single file from OSF.

    Parameters
    ----------
    osf_url : string
        The URL of the OSF file to download.
    output_path : Path
        Path where the file will be saved.
    save_name : str or None
        Optional filename to save the downloaded file as. If None, the
        filename will be inferred from the URL (which may be an OSF id).

    Returns
    -------
    Path
        Path to the downloaded file.
    """
    # Determine the file name to save as
    if save_name:
        file_name = save_name
    else:
        # Fallback: get the file name from the URL (may be an OSF id)
        file_name = osf_url.split("/")[-1]

    file_path = output_path / file_name

    # if the file does not already exist, download it from osf
    if not file_path.exists():
        # provide a little update message
        print(f"Downloading {file_name} to {file_path}")

        # download and save the file, updating the user on download progress
        with requests.get(osf_url, stream=True) as file:
            # get total size of file for download updates
            file_size = int(file.headers.get("Content-Length", 0))

            # implement progress bar via tqdm
            desc = f"Downloading {file_name}"
            with tqdm.wrapattr(file.raw, "read", total=file_size, desc=desc) as raw:
                # save the output to the file specified before
                with open(file_path, "wb") as output:
                    shutil.copyfileobj(raw, output)
    else:
        print(f"File {file_name} already exists at {file_path}")

    return file_path


def download_example_data(dataset_path=None, with_stimuli=False):
    """
    Download example EEG data (1 subject) for testing and tutorials.

    Parameters
    ----------
    dataset_path : string
        Path where the files will be saved. If None, the files will be saved
        in the current working directory. Default = None.
    with_stimuli : bool
        When True, additionally download the BIDS ``/stimuli/`` directory
        used by stimulus-aware analyses (e.g. corr_stim_to_resp). When
        False (default), only the EEG data is fetched.

    Returns
    -------
    dataset_path : Path
        Path to the directory containing the downloaded data.

    Examples
    --------
    Download example data without specifying a path.

    >>> download_example_data()

    Download example data, specifying a path.

    >>> download_example_data(dataset_path='/home/user/Desktop')
    """
    # Download 1 subject as example data
    path = download_raw_data(subjects=1, dataset_path=dataset_path)
    if with_stimuli:
        download_stimuli(dataset_path=dataset_path)
    return path


def download_raw_data(subjects=1, dataset_path=None, with_stimuli=False):
    """
    Download raw EEG data for specified subjects from OSF.

    Parameters
    ----------
    subjects : int or list
        Number of subjects to download (int) or specific subject IDs (list).
        If int, downloads the first N subjects. If list, downloads specific
        subjects by ID (e.g., ['03', '21']). Default = 1.
    dataset_path : string
        Path where the files will be saved. If None, the files will be saved
        in the current working directory. Default = None.
    with_stimuli : bool
        When True, additionally download the BIDS ``/stimuli/`` directory
        used by stimulus-aware analyses (e.g. corr_stim_to_resp). When
        False (default), only the EEG data is fetched.

    Returns
    -------
    dataset_path : Path
        Path to the directory containing the downloaded data.

    Examples
    --------
    Download data for 3 subjects without specifying a path.

    >>> download_raw_data(subjects=3)

    Download data for 5 subjects, specifying a path.

    >>> download_raw_data(subjects=5, dataset_path='/home/user/Desktop')

    Download specific subjects by ID.

    >>> download_raw_data(subjects=['03', '21'],
    ...                   dataset_path='/home/user/Desktop')
    """
    # Available subjects based on the OSF structure
    available_subjects = ["03", "05", "09", "11", "15", "21"]

    # Handle different input types for subjects parameter
    if isinstance(subjects, int):
        # If integer, select first N subjects
        n_subjects = subjects
        if n_subjects > len(available_subjects):
            warning_msg = (
                f"Warning: Only {len(available_subjects)} subjects "
                "available. Downloading all available subjects."
            )
            print(warning_msg)
            n_subjects = len(available_subjects)
        subjects_to_download = available_subjects[:n_subjects]
    elif isinstance(subjects, list):
        # If list, validate and use specific subject IDs
        invalid_subjects = [s for s in subjects if s not in available_subjects]
        if invalid_subjects:
            warning_msg = (
                f"Warning: Subjects {invalid_subjects} not " f"available. Available: {available_subjects}"
            )
            print(warning_msg)
        subjects_to_download = [s for s in subjects if s in available_subjects]
        if not subjects_to_download:
            raise ValueError("No valid subjects specified")
    else:
        raise TypeError("subjects must be an integer or list of subject IDs")

    # Set up directory structure
    if dataset_path is None:
        path = Path(os.curdir) / "ffrprep_raw_data"
    else:
        path = Path(dataset_path) / "ffrprep_raw_data"

    if not path.exists():
        os.makedirs(path)

    # Base OSF URL for the raw data
    base_url = "https://osf.io/download/"

    # Download metadata files first
    metadata_files = {
        "dataset_description.json": "8dp94",  # Update with actual OSF IDs
        "participants.json": "3y27d",
        "participants.tsv": "8pwfq",
        "README": "je2bf",
    }

    print("Downloading metadata files...")
    for filename, osf_id in metadata_files.items():
        if osf_id != "participants_json_id":  # Skip placeholder IDs
            osf_url = f"{base_url}{osf_id}"
            # Save the metadata file using the intended filename (the dict key)
            _download_single_file(osf_url, path, save_name=filename)

    # Download subject data
    subject_urls = {
        "03": "x2kd6",
        "05": "s8wu6",
        "09": "xj7pd",
        "11": "4ge3d",
        "15": "kqhux",
        "21": "hdc8f",
        "23": "9rnp6",
        "25": "dgfsp",
        "27": "6enym",
        "29": "9k24c",
        "31": "npywc",
        "33": "dh23w",
    }

    print(f"Downloading data for {len(subjects_to_download)} subjects...")
    for subject in subjects_to_download:
        placeholder_id = f"sub{subject}_osf_id"
        # Skip placeholders
        if subject in subject_urls and subject_urls[subject] != placeholder_id:
            osf_url = f"{base_url}{subject_urls[subject]}"
            zip_path = _download_single_file(osf_url, path)

            # Unzip the subject data
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(path)

            # Remove the zip file
            os.remove(zip_path)

    # Clean up __MACOSX folder if it exists
    macosx_path = path / "__MACOSX"
    if macosx_path.exists():
        shutil.rmtree(macosx_path, ignore_errors=True)

    if with_stimuli:
        download_stimuli(dataset_path=dataset_path)

    return path


# Module-level mapping of stimulus filename -> OSF download ID.
# Empty until real OSF IDs land. The constant is lifted to module
# scope so tests can monkeypatch it without a special seam.
STIM_URLS = {
    # "<filename>.wav": "<osf_id>",
    # Populate once OSF stimulus IDs are available from the dataset
    # maintainer.
}


def download_stimuli(dataset_path=None):
    """
    Download FFR stimulus files into a BIDS-compliant ``stimuli/`` directory.

    Stimulus files are referenced by the ``stim_file`` column of each
    BIDS ``events.tsv`` and consumed by stimulus-aware analyses (e.g.
    :func:`ffrprep.analysis.corr_stim_to_resp`). They are shared across
    subjects, so they land at ``<dataset>/ffrprep_raw_data/stimuli/``
    per the BIDS specification.

    Parameters
    ----------
    dataset_path : string, optional
        Path where the files will be saved. If None, the files will be
        saved in the current working directory.

    Returns
    -------
    stim_dir : Path
        Path to the ``stimuli/`` directory under the dataset root.
    """
    if dataset_path is None:
        base = Path(os.curdir) / "ffrprep_raw_data"
    else:
        base = Path(dataset_path) / "ffrprep_raw_data"
    stim_dir = base / "stimuli"
    stim_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://osf.io/download/"
    for filename, osf_id in STIM_URLS.items():
        osf_url = f"{base_url}{osf_id}"
        _download_single_file(osf_url, stim_dir, save_name=filename)

    return stim_dir


def download_epoch_data(subjects=1, dataset_path=None):
    """
    Download epoched EEG data for specified subjects from OSF.

    Data will be organized following BIDS derivatives structure within
    the raw dataset directory.

    Parameters
    ----------
    subjects : int or list
        Number of subjects to download (int) or specific subject IDs (list).
        If int, downloads the first N subjects. If list, downloads specific
        subjects by ID (e.g., ['03', '21']). Default = 1.
    dataset_path : string
        Path where the files will be saved. If None, the files will be saved
        in the current working directory. Default = None.

    Returns
    -------
    dataset_path : Path
        Path to the directory containing the downloaded data.

    Examples
    --------
    Download epoched data for 2 subjects without specifying a path.

    >>> download_epoch_data(subjects=2)

    Download epoched data for 4 subjects, specifying a path.

    >>> download_epoch_data(subjects=4, dataset_path='/home/user/Desktop')

    Download specific subjects by ID.

    >>> download_epoch_data(subjects=['03', '21'],
    ...                     dataset_path='/home/user/Desktop')
    """
    # Available subjects for epoched data (based on OSF structure)
    available_subjects = ["03", "15", "21", "30"]

    # Handle different input types for subjects parameter
    if isinstance(subjects, int):
        # If integer, select first N subjects
        n_subjects = subjects
        if n_subjects > len(available_subjects):
            warning_msg = (
                f"Warning: Only {len(available_subjects)} subjects "
                "available. Downloading all available subjects."
            )
            print(warning_msg)
            n_subjects = len(available_subjects)
        subjects_to_download = available_subjects[:n_subjects]
    elif isinstance(subjects, list):
        # If list, validate and use specific subject IDs
        invalid_subjects = [s for s in subjects if s not in available_subjects]
        if invalid_subjects:
            warning_msg = (
                f"Warning: Subjects {invalid_subjects} not " f"available. Available: {available_subjects}"
            )
            print(warning_msg)
        subjects_to_download = [s for s in subjects if s in available_subjects]
        if not subjects_to_download:
            raise ValueError("No valid subjects specified")
    else:
        raise TypeError("subjects must be an integer or list of subject IDs")

    # Set up BIDS derivatives directory structure within raw data
    if dataset_path is None:
        base_path = Path(os.curdir) / "ffrprep_raw_data"
    else:
        base_path = Path(dataset_path) / "ffrprep_raw_data"

    # Create derivatives/epochs directory
    derivatives_path = base_path / "derivatives" / "epochs"
    if not derivatives_path.exists():
        os.makedirs(derivatives_path)

    # Base OSF URL for epoched data
    base_url = "https://osf.io/download/"

    # Epoched data OSF IDs (update with actual IDs from your OSF project)
    epoch_files = {
        "03": {"active": "qtm3s", "passive": "rvnfw"},
        "15": {"active": "8wykn", "passive": "k3zcn"},
        "21": {"active": "pvrs3", "passive": "9yvp4"},
        "30": {"active": "ub4cz", "passive": "rbm74"},
    }

    n_subs = len(subjects_to_download)
    print(f"Downloading epoched data for {n_subs} subjects...")

    for subject in subjects_to_download:
        # Create subject-specific directories following BIDS derivatives
        subject_dir = derivatives_path / f"sub-{subject}"
        eeg_dir = subject_dir / "eeg"
        if not eeg_dir.exists():
            os.makedirs(eeg_dir)

        print(f"Processing subject {subject}...")

        # Download both active and passive task epochs for each subject
        for task in ["active", "passive"]:
            if subject in epoch_files:
                osf_id = epoch_files[subject][task]
                if osf_id.startswith("osf_id_"):  # Skip placeholder IDs
                    print(f"  Skipping {task} task - OSF ID not yet provided")
                    continue

                osf_url = f"{base_url}{osf_id}"
                filename = f"sub-{subject}_task-{task}_run-all_" f"event-stimtrack_epochs.fif"
                file_path = eeg_dir / filename

                # Download the file directly to the BIDS structure
                if not file_path.exists():
                    print(f"  Downloading {filename}...")
                    with requests.get(osf_url, stream=True) as file:
                        file_size = int(file.headers.get("Content-Length", 0))
                        desc = f"Downloading {filename}"
                        with tqdm.wrapattr(file.raw, "read", total=file_size, desc=desc) as raw:
                            with open(file_path, "wb") as output:
                                shutil.copyfileobj(raw, output)
                else:
                    print(f"  File {filename} already exists")

    print("Note: Update epoch_files dictionary with actual OSF IDs")

    # Clean up __MACOSX folder if it exists
    macosx_path = base_path / "__MACOSX"
    if macosx_path.exists():
        shutil.rmtree(macosx_path, ignore_errors=True)

    return derivatives_path
