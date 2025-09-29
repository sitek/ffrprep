import numpy as np
from numpy import mean, sqrt, square
import mne
import statsmodels as sm


def compute_power(avg_evoked, f_low=90, f_high=110, t_low=0.1, t_high=0.2):
    """
    Compute the average oscillatory power of a given frequency band.

    Parameters
    ----------
    avg_evoked : mne.Evoked object
        Evoked object to calculate the average power of.
    f_low : float
        Lower bound of the frequency band of interest (Hz).
    f_high : float
        Upper bound of the frequency band of interest (Hz).
    t_low : float
        Start of the time window of interest (in seconds).
    t_high : float
        End of the time window of interest (in seconds).

    Returns
    -------
    average_power : float
        Value representing the mean power of the first channel specified
        within the given time and frequency ranges.

    Examples
    --------

    Compute average gamma-band (90-110 Hz) power between 100-200 ms

    >>> power_val = compute_power(evoked, f_low=90, f_high=110,
    ...                           t_low=0.1, t_high=0.2)
    >>> print(power_val)
    2.347e-10

    """
    # Compute the spectrogram (time-frequency representation)
    power = mne.time_frequency.tfr_multitaper(avg_evoked,
                                              freqs=np.arange
                                              (f_low, f_high, 1),
                                              n_cycles=2,
                                              time_bandwidth=4.0,
                                              average=True,
                                              return_itc=False)

    # Select the time and frequency bands
    time_mask = (power.times >= t_low) & (power.times <= t_high)
    freq_mask = (power.freqs >= f_low) & (power.freqs <= f_high)

    # Extract the power within the specified bands
    selected_power = power.data[:, freq_mask, :][:, :, time_mask]

    # Average power over the selected time and frequency bands
    average_power = selected_power.mean(axis=2).mean(axis=1)[0]
    return average_power


def rms_snr(evoked, response_lower=0.100, response_upper=0.200):
    """
    Compute RMS-based signal-to-noise ratio (SNR) for an Evoked response.

    Parameters
    ----------
    evoked : mne.Evoked
        Evoked object containing averaged M/EEG data.
    response_lower : float, default=0.100
        Start time (in seconds) of the response window, relative to the Evoked
        time axis.
    response_upper : float, default=0.200
        End time (in seconds) of the response window, relative to the Evoked
        time axis.

    Returns
    -------
    rms_snr : float
        The RMS SNR computed as:
        RMS(response window) / RMS(baseline window)
        for the first channel in `evoked.data`.

    Example
    -------
    Compute RMS SNR for a 100-200 ms response window

    >>> snr = rms_snr(evoked, response_lower=0.100, response_upper=0.200)
    >>> float(np.round(snr, 3))
    3.412
    """

    baseline_ind_bounds = evoked.time_as_index(evoked.baseline)
    response_ind_bounds = evoked.time_as_index(
        [response_lower, response_upper])

    evoked_baseline = evoked.data[0,
                                  baseline_ind_bounds[0]:
                                  baseline_ind_bounds[1]]
    evoked_response = evoked.data[0,
                                  response_ind_bounds[0]:
                                  response_ind_bounds[1]]
    rms_baseline = sqrt(mean(square(evoked_baseline)))
    rms_response = sqrt(mean(square(evoked_response)))

    rms_snr = rms_response / rms_baseline
    return rms_snr


def autocorrelation(evoked):
    acf, confidence_interval = sm.tsa.stattools.acf(
        evoked,
        nlags=len(evoked) - 1,
        alpha=0.05
    )

    return acf, confidence_interval
