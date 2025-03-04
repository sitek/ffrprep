#import os
#import mne
#import mne_bids
#import numpy as np
#import matplotlib.pyplot as plt

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

    >>> data = load_data(bids_root=bids_root, sub_label=sub_label, run_label=run_label)

    """

    bids_path = BIDSPath(subject=sub_label, 
                         session=session_label,
                         task=task_label, 
                         run=run_label,
                         root=bids_root, 
                         datatype='eeg', )
    
    data = read_raw_bids(bids_path=bids_path, verbose=False)
    data = data.load_data()

    return data, bids_path