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


def compute_pitch_and_conf(evoked,
                           win_dur=0.040,
                           hop_dur=0.010,
                           fmin=60,
                           fmax=200,
                           strength_thresh=0.15,
                           smooth_k=3):
    """
    Compute the pitch (f0) and confidence metrics for an Evoked response.

    Parameters
    ----------
    evoked : mne.Evoked
        Evoked object containing averaged M/EEG data.
    win_dur : float, default=0.040
        Duration of the sliding window (in seconds) used for pitch estimation.
    hop_dur : float, default=0.010
        Step size (in seconds) for sliding the window across the signal.
    fmin : float, default=60
        Minimum frequency (in Hz) to consider for pitch estimation.
    fmax : float, default=200
        Maximum frequency (in Hz) to consider for pitch estimation.
    strength_thresh : float, default=0.15
        Minimum normalized autocorrelation strength required to consider
        a pitch estimate valid.
    smooth_k : int, default=3
        Kernel size for smoothing the pitch estimates (in number of windows).

    Returns
    -------
    dict
        A dictionary with the following keys:
        - 'times':
            Array of time points corresponding to the center of each window.
        - 'pitch_hz':
            Array of pitch estimates (in Hz) for each window.
        - 'pitch_hz_smooth':
            Smoothed pitch estimates (in Hz) for each window.
        - 'peak_strength':
            Array of normalized autocorrelation strengths for each window.
        - 'conf_rmax':
            Array of maximum normalized autocorrelation values for each window.
        - 'conf_pnr':
            Array of pitch-to-noise ratios for each window.
        - 'conf_z':
            Array of z-scores for the maximum autocorrelation values
            for each window.

    Examples
    --------
    Compute pitch and confidence metrics for an Evoked response
    with default parameters.

    >>> results = compute_pitch_and_conf(evoked)
    >>> print(results['times'])
    [0.02 0.03 0.04 ... 0.98 0.99 1.00]
    >>> print(results['pitch_hz'])
    [120.0 125.0 130.0 ... 110.0 115.0 120.0]
    >>> print(results['conf_rmax'])
    [0.8 0.85 0.9 ... 0.75 0.8 0.82]
    """
    from scipy.signal import find_peaks

    signal = evoked.data[0]
    times = evoked.times
    if hasattr(evoked, 'info'):
        sfreq = evoked.info['sfreq']
    else:
        sfreq = 1.0 / np.diff(times)[0]

    # Define window, hop sizes, and lags in samples
    win_samps = max(3, int(round(win_dur * sfreq)))
    hop_samps = max(1, int(round(hop_dur * sfreq)))
    lag_min = max(1, int(round(sfreq / fmax)))
    lag_max = int(round(sfreq / fmin))

    # Initialize output lists
    pitch_times, pitch_hz, peak_strength = [], [], []
    rmax_list, pnr_list, z_list = [], [], []

    # Compute pitch (autocorr) and confidence per window (single pass)
    for start in range(0, len(signal) - win_samps + 1, hop_samps):
        # extract windowed signal and normalize
        w = signal[start:start + win_samps].copy()
        w = w - np.mean(w)

        # compute autocorrelation
        ac_full = np.correlate(w, w, mode='full')
        ac = ac_full[ac_full.size // 2:]  # keep positive lags

        # time stamp for this window
        center_time = times[start + win_samps // 2]
        pitch_times.append(center_time)

        # guard for too-short autocorrelation / invalid lag range
        if ac.size <= lag_min:
            pitch_hz.append(np.nan)
            peak_strength.append(0.0)
            rmax_list.append(np.nan)
            pnr_list.append(np.nan)
            z_list.append(np.nan)
            continue

        # normalize autocorrelation and search for peaks in desired lag range
        norm_ac = ac / (ac[0] if ac[0] != 0 else 1.0)
        search = norm_ac[lag_min: min(lag_max + 1, len(norm_ac))]

        # if no peaks found
        if search.size == 0:
            pitch_hz.append(np.nan)
            peak_strength.append(0.0)
            rmax_list.append(np.nan)
            pnr_list.append(np.nan)
            z_list.append(np.nan)
            continue

        # main peak and strength
        idx_rel = int(np.argmax(search))
        strength = float(search[idx_rel])
        lag = idx_rel + lag_min
        if strength >= strength_thresh:
            pitch = float(sfreq / lag)
        else:
            pitch = np.nan

        pitch_hz.append(pitch)
        peak_strength.append(strength)

        # confidence metrics
        rmax = strength

        # Find next highest peak excluding the main peak
        peaks, _ = find_peaks(search)
        if peaks.size == 0:
            sorted_idx = np.argsort(search)
            if sorted_idx.size >= 2:
                next_max = float(search[sorted_idx[-2]])
            else:
                next_max = 0.0
        else:
            peak_vals = search[peaks]
            mask_other = peaks != idx_rel
            other_vals = peak_vals[mask_other]
            next_max = float(other_vals.max()) if other_vals.size > 0 else 0.0

        # pitch-to-noise ratio and z-score
        pnr = (rmax / (next_max + 1e-12)) if next_max > 0 else np.inf
        search_mean = float(np.mean(search))
        search_std = float(np.std(search)) if float(np.std(search)) > 0 else 1e-12
        z = (rmax - search_mean) / search_std

        rmax_list.append(rmax)
        pnr_list.append(pnr)
        z_list.append(z)

    pitch_times = np.array(pitch_times)
    pitch_hz = np.array(pitch_hz)
    peak_strength = np.array(peak_strength)

    # smoothing that ignores NaNs
    def moving_avg_ignore_nan(x, k=3):
        mask = ~np.isnan(x)
        x0 = np.where(mask, x, 0.0)
        num = np.convolve(x0, np.ones(k), mode='same')
        den = np.convolve(mask.astype(float), np.ones(k), mode='same')
        den[den == 0] = np.nan
        return num / den

    pitch_hz_smooth = moving_avg_ignore_nan(pitch_hz, k=smooth_k)

    # Prepare output arrays
    rmax_arr = np.array(rmax_list)
    pnr_arr = np.array(pnr_list)
    z_arr = np.array(z_list)

    return {
        'times': pitch_times,
        'pitch_hz': pitch_hz,
        'pitch_hz_smooth': pitch_hz_smooth,
        'peak_strength': peak_strength,
        'conf_rmax': rmax_arr,
        'conf_pnr': pnr_arr,
        'conf_z': z_arr,
    }


def plot_pitch_and_conf(results):
    """
    Plot a smoothed pitch track together with associated confidence measures.

    This function visualizes the output of compute_pitch_and_conf() by
    creating a two-row matplotlib figure:
    the top row shows the smoothed pitch track (in Hz)
    with optional per-frame coloring by a confidence metric, and the bottom row
    shows one or more confidence time-series (rmax, z-score, pnr, etc.).

    Parameters
    ----------
    results : dict
        Dictionary as returned by compute_pitch_and_conf(). The dictionary must
        contain at least the following keys:
          - 'times' : array-like
              1-D array of time stamps in seconds for the pitch track.
          - 'pitch_hz_smooth' : array-like
              1-D array of smoothed pitch values in Hz.
              Use NaN for unvoiced frames.
        The dictionary may also include the following optional entries:
          - 'peak_strength' : array-like or None
              Per-frame peak strength values that can be used as an alternate
              confidence measure for coloring the pitch scatter.
          - 'conf_rmax' : array-like or None
              Preferred confidence metric (e.g., correlation maximum).
              If present and aligned with the pitch/times vector,
              it will be used to color the
              pitch points and plotted in the bottom panel.
          - 'conf_z' : array-like or None
              Z-score confidence values (plotted in bottom panel when present).
          - 'conf_pnr' : array-like or None
              PNR (pitch-to-noise ratio) confidence values (plotted in bottom).

    Returns
    -------
    None
        Displays a matplotlib figure. The top subplot shows the smoothed pitch
        (points colored by conf_rmax if available, otherwise by peak_strength;
        falls back to a simple line if no color values are available).
        The bottom subplot shows available confidence traces aligned in time.
        The function does not return any value.

    Raises
    ------
    TypeError
        If `results` is not a dict.
    ValueError
        If required keys ('times' and 'pitch_hz_smooth') are missing or None.

    Notes
    -----
    - The function uses numpy.asarray() to coerce array-like inputs
      into arrays and masks NaN values in the pitch track
      so unvoiced frames are omitted from the scatter/line points.
    - If confidence arrays have a different length than the pitch array,
      the code will attempt to align them against the provided time vector;
      otherwise they are used as-is.
    - Requires matplotlib.pyplot (imported as plt in the function)
      and numpy (np) to be available in the module scope.

    Example
    -------
    # Assuming compute_pitch_and_conf() returns the expected dict:
    results = compute_pitch_and_conf(audio_chunk)
    plot_pitch_and_conf(results)
    """
    import matplotlib.pyplot as plt

    # basic validation:
    # expect a dict like the output of compute_pitch_and_conf()
    if not isinstance(results, dict):
        raise TypeError("results must be a dict as returned"
                        " by compute_pitch_and_conf()")

    # require at least these keys
    required_keys = ('times', 'pitch_hz_smooth')
    missing = [k for k in required_keys if k not in results]
    if missing:
        raise ValueError(f"results is missing required keys: {missing}")

    # expect arrays (or array-like) for core entries
    times = results.get('times')
    pitch_smooth = results.get('pitch_hz_smooth')
    peak_strength = results.get('peak_strength')
    conf_rmax = results.get('conf_rmax')
    conf_z = results.get('conf_z')
    conf_pnr = results.get('conf_pnr')

    if times is None or pitch_smooth is None:
        raise ValueError("results must contain 'times' and 'pitch_hz_smooth'.")

    times = np.asarray(times)
    pitch_smooth = np.asarray(pitch_smooth)

    # mask valid pitch values
    mask = (~np.isnan(pitch_smooth))

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 6),
                                         gridspec_kw={'height_ratios': [3, 1]})

    # top: pitch track colored by confidence
    color_vals = None
    color_label = None
    if conf_rmax is not None:
        conf_rmax = np.asarray(conf_rmax)
    if peak_strength is not None:
        peak_strength = np.asarray(peak_strength)

    if (conf_rmax is not None and
            (conf_rmax.size == mask.sum() or
             conf_rmax.size == len(times))):
        # try align lengths (conf arrays are usually same length as times)
        try:
            color_vals = conf_rmax
            color_label = 'conf_rmax'
        except Exception:
            color_vals = None

    if color_vals is None and peak_strength is not None:
        color_vals = peak_strength
        color_label = 'peak_strength'

    if color_vals is not None:
        # handle cases where color_vals length differs from pitch_smooth
        if color_vals.shape == pitch_smooth.shape:
            c = color_vals[mask]
        elif color_vals.shape == times.shape:
            c = color_vals[mask]
        else:
            c = color_vals
        sc = ax_top.scatter(times[mask], pitch_smooth[mask], c=c,
                            cmap='viridis', s=40, edgecolor='k', lw=0.3)
        cbar = fig.colorbar(sc, ax=ax_top, pad=0.01)
        cbar.set_label(color_label)
    else:
        ax_top.plot(times[mask], pitch_smooth[mask], '-o', markersize=4)

    ax_top.plot(times, pitch_smooth, linestyle='-', color='0.2', alpha=0.4)
    ax_top.set_ylabel('Pitch (Hz)')
    ax_top.set_title('Pitch track (smoothed)')
    ax_top.grid(True)

    # bottom: confidence time-series (rmax, z, pnr if present)
    conf_times = times

    if conf_rmax is not None:
        ax_bot.plot(conf_times, conf_rmax, label='rmax', color='C0')
    if conf_z is not None:
        conf_z = np.asarray(conf_z)
        ax_bot.plot(conf_times, conf_z, label='z-score', color='C1', alpha=0.8)
    if conf_pnr is not None:
        conf_pnr = np.asarray(conf_pnr)
        ax_bot.plot(conf_times, conf_pnr, label='pnr', color='C2', alpha=0.6)

    ax_bot.axhline(0.15, color='k', ls='--', alpha=0.5)
    ax_bot.set_xlim(ax_top.get_xlim())
    ax_bot.set_xlabel('Time (s)')
    ax_bot.set_ylabel('Confidence')
    ax_bot.legend(loc='upper right')
    ax_bot.grid(True)

    fig.tight_layout()
    plt.show()
