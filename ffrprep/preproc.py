from mne_bids import BIDSPath, read_raw_bids


def load_data(bids_root=None,
              sub_label=None,
              session_label=None,
              task_label=None,
              run_label=None,):
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
