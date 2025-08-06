import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

from mne import create_info
from mne.decoding import CSP
from mne.time_frequency import AverageTFRArray


def classify_data(epochs,
                  n_freqs,
                  n_cycles,
                  min_freq,
                  max_freq,
                  tmin,
                  tmax):
    """
    Use cross-validation to score model performance across frequency and time
    ranges.

    Parameters
    ----------
    epochs : mne.Epochs object
        MNE Epochs object created from epoching raw MNE data.
    n_freqs : int
        Number of bins to separate frequencies into for analysis.
    n_cycles : int
        Number of bins to separate time into for analysis.
    min_freq : float
        Lower bound for frequency in evaluation.
    max_freq : float
        Upper bound for frequency in evaluation.
    tmin : float
        Lower bound for time in evaluation.
    tmax : float
        Upper bound for time in evaluation.

    Returns
    -------
    av_tfr : average nic

    Examples
    --------
    Evaluate the model performance on the given Epochs data object
    with the given time and frequency ranges.

    >>> classify_data(epochs=epochs,
                      n_freqs=6,
                      n_cycles=10,
                      min_freq=8.0,
                      max_freq=20.0,
                      tmin=tmin,
                      tmax=tmax)

    """

    # Use classifier from sci-kit learn
    clf = make_pipeline(CSP(n_components=4, reg=None, log=True,
                            norm_trace=False), LinearDiscriminantAnalysis())

    sfreq = epochs.info["sfreq"]
    n_splits = 3  # for cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Instantiate label encoder
    le = LabelEncoder()

    freqs = np.linspace(min_freq, max_freq, n_freqs)  # assemble frequencies
    freq_ranges = list(zip(freqs[:-1], freqs[1:]))  # make freqs list of tuples

    # initialize scores
    freq_scores = np.zeros((n_freqs - 1,))

    # Loop through each frequency range
    for freq, (fmin, fmax) in enumerate(freq_ranges):
        y = le.fit_transform(epochs.events[:, 2])

        X = epochs.get_data(copy=False)

        # Save mean scores over folds for each frequency and time window
        freq_scores[freq] = np.mean(
            cross_val_score(estimator=clf, X=X, y=y, scoring="roc_auc", cv=cv),
            axis=0)

    # Infer window spacing from the max freq and number of cycles to avoid gaps
    window_spacing = n_cycles / np.max(freqs) / 2.0
    centered_w_times = np.arange(tmin, tmax, window_spacing)[1:]
    n_windows = len(centered_w_times)
    # init scores
    tf_scores = np.zeros((n_freqs - 1, n_windows))

    # Loop through each section of frequency range
    for freq, (fmin, fmax) in enumerate(freq_ranges):
        # Infer window size based on the frequency being used
        w_size = n_cycles / ((fmax + fmin) / 2.0)  # in seconds
        y = le.fit_transform(epochs.events[:, 2])

        # loop through every section of time
        for t, w_time in enumerate(centered_w_times):
            # Center the min and max of the window
            w_tmin = w_time - w_size / 2.0
            w_tmax = w_time + w_size / 2.0

            # Crop data into time-window of interest
            X = epochs.get_data(tmin=w_tmin, tmax=w_tmax, copy=False)

            # Save mean scores using roc auc over folds for each frequency and
            # time window
            tf_scores[freq, t] = np.mean(
                cross_val_score(estimator=clf, X=X, y=y, scoring="roc_auc",
                                cv=cv), axis=0)

    # Set up time frequency object for graphing
    av_tfr = AverageTFRArray(
        info=create_info(["freq"], sfreq),
        data=tf_scores[np.newaxis, :],
        times=centered_w_times,
        freqs=freqs[1:],
        nave=1,
    )

    chance = np.mean(y)  # set chance level to white in the plot
    av_tfr.plot(
        [0], vlim=(chance, None), title="Time-Frequency Decoding Scores",
        cmap=plt.cm.Reds)

    return av_tfr
