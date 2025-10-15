"""
Preprocessing utilities for EEG data in BIDS format.

Includes loading, referencing, filtering, epoching,
and evoked response calculation.
"""

from mne_bids import BIDSPath, read_raw_bids


def load_data(bids_root=None,
              sub_label=None,
              session_label=None,
              task_label=None,
              run_label=None):
    """
    Identify an EEG filepath in a BIDS directory and load it.

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
    bids_path = BIDSPath(subject=sub_label,
                         session=session_label,
                         task=task_label,
                         run=run_label,
                         root=bids_root,
                         datatype='eeg')

    data = read_raw_bids(bids_path=bids_path, verbose=False)
    data = data.load_data()

    return data, bids_path


def reference_data(eeg_data=None,
                   ref_channels=None):
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
    # apply average reference by default
    if ref_channels is None:
        eeg_data.set_eeg_reference()
    # if single channel provided as a string
    elif isinstance(ref_channels, str):
        eeg_data.set_eeg_reference(ref_channels=[ref_channels])
    # if list of channels is provided
    else:
        eeg_data.set_eeg_reference(ref_channels=ref_channels)

    return eeg_data


def filter_data(eeg_data=None,
                high_pass=None,
                low_pass=None):
    """
    Re-reference the provided EEG data object.

    Parameters
    ----------
    data : MNE `data` object
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
        to given frequency range.

    Examples
    --------
    Filter an EEG data object with a band-pass filter.

    >>> filtered_data = filter_data(eeg_data, high_pass=50, low_pass=2000)

    Filter an EEG data object with a high-pass filter.

    >>> filtered_data = filter_data(eeg_data, high_pass=50)

        Filter an EEG data object with a low-pass filter.

    >>> filtered_data = filter_data(eeg_data, low_pass=2000)
    """
    eeg_data.filter(l_freq=high_pass,
                    h_freq=low_pass)

    return eeg_data


def epoch_data(eeg_data,
               baseline,
               events_file=None,
               picks=None,
               epoch_window=None,
               tmin=None,
               tmax=None,
               verbose=True):
    """
    Epoch the provided EEG data object based on events.

    Parameters
    ----------
    data : MNE `data` object
        MNE `data` object containing EEG data and metadata.
    baseline : float or 1-D array
        In seconds: start of baseline period (if float);
        baseline window (if array).

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

    """
    from mne import Epochs

    events = None
    event_dict = None

    # define baseline window (in seconds)
    if isinstance(baseline, float):
        baseline_window = [baseline, 0]
    else:
        baseline_window = baseline

    # define epoch time window (in seconds)
    if not tmin:
        tmin = baseline_window[0]
    if not tmax:
        tmax = 0.5

    epoched_data = Epochs(eeg_data,
                          events=events,
                          event_id=event_dict,
                          on_missing='warn',
                          picks=picks,
                          tmin=tmin, tmax=tmax,
                          baseline=baseline_window,
                          verbose=verbose,
                          reject=dict(eeg=75e-6)).drop_bad()

    return epoched_data, (tmin, tmax)


# use mne.Report for all functions, created evoked object
def preproc_pipeline(bids_root,
                     baseline,
                     sub_label=None,
                     session_label=None,
                     task_label=None,
                     run_label=None,
                     ref_channels=None,
                     high_pass=None,
                     low_pass=None,
                     events_file=None,
                     picks=None,
                     epoch_window=None,
                     tmin=None,
                     tmax=None,
                     verbose=True):
    """
    Preprocess raw EEG data.

    Parameters
    ----------
    data : MNE `data` object
        MNE `data` object containing EEG data and metadata.

    Returns
    -------
    preprocessed_data : MNE `XXX` object
        MNE `XXX` object.

    Examples
    --------
    Preprocess EEG data.

    >>> preprocessed_data = preproc_pipeline()


    """
    # Load EEG data from bids_root path, with optional specification through
    # parameters
    # Path variable is stored for future usage
    eeg_data, _ = load_data(bids_root=bids_root,
                            sub_label=sub_label,
                            session_label=session_label,
                            task_label=task_label,
                            run_label=run_label)

    # Reference the given EEG data to specific channels
    referenced_eeg_data = reference_data(eeg_data, ref_channels=ref_channels)

    # Filter the referenced EEG data using the given band-pass
    filtered_eeg_data = filter_data(referenced_eeg_data, high_pass=high_pass,
                                    low_pass=low_pass)

    # Epoch the filtered data using the given baseline
    preprocessed_data = epoch_data(filtered_eeg_data, baseline,
                                   events_file=events_file,
                                   picks=picks, epoch_window=epoch_window,
                                   tmin=tmin, tmax=tmax, verbose=verbose)

    # Return the epoched data
    return preprocessed_data


def make_evoked(epochs, by_event_type: bool):
    """
    Make an estimate of all epochs.

    Parameters
    ----------
    epochs : MNE `Epochs` object
        MNE `Epochs` object containing time-locked epochs.
    epochs : Boolean
        Boolean containing whether to sort epochs by event type

    epochs : Boolean
        Boolean containing whether to sort epochs by event type

    Returns
    -------
    evoked : MNE `Evoked` object
        MNE `Evoked` object containing the average epoch.
    Examples
    --------
    Evoke an EEG Epochs object.
    >>> evoked_data = make_evoked(epoched_data)
    """
    # average the epochs
    evoked = epochs.average(by_event_type)

    # return the averaged epochs
    return evoked
