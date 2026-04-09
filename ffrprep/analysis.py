import numpy as np
from numpy import mean, sqrt, square
import mne
import statsmodels as sm
from scipy import signal
import matplotlib.pyplot as plt


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

def compute_phase_consistency(
    epochs_A,
    epochs_B,
    chunksize=0.04,
    overlap=0.036,
    freqcap=2000
):
    """
    Compute phase consistency from FFR epochs.
    
    The function computes phase consistency for:
    - Polarity A (from epochs_A)
    - Polarity B (from epochs_B)  
    - ADD polarity: (A + B) / 2 (computed from phase vectors)
    - SUB polarity: (A - B) / 2 (computed from phase vectors)
    
    Parameters
    ----------
    epochs_A : mne.Epochs
        Individual epochs for polarity A
    epochs_B : mne.Epochs
        Individual epochs for polarity B
    chunksize : float
        Analysis window size in seconds (default: 0.04 = 40 ms)
    overlap : float
        Window overlap in seconds (default: 0.036 = 36 ms, gives ~4 ms step)
    freqcap : int
        Maximum frequency in Hz (default: 2000)
        
    Returns
    -------
    phasecon : dict
        Dictionary with keys 'A', 'B', 'add', 'sub' and phase consistency arrays (freq x time) as values
    xaxis : ndarray
        Time axis in milliseconds
    yaxis : ndarray
        Frequency axis in Hz
    numsweeps : int
        Number of sweeps used (minimum of A and B)
    
    Examples
    --------
    >>> phasecon, xaxis, yaxis, numsweeps = compute_phase_consistency_minimal(
    ...     epochs_A, epochs_B, chunksize=0.04, overlap=0.036
    ... )
    >>> 
    >>> # Plot
    >>> plot_phase_consistency(phasecon, xaxis, yaxis)
    """
    
    # Get sampling info from epochs
    sfreq = epochs_A.info['sfreq']
    tmin = epochs_A.times[0]
    tmax = epochs_A.times[-1]
    
    # Convert parameters to samples
    chunksizepts = int(np.round(chunksize * sfreq))
    overlappts = int(np.round(overlap * sfreq))
    
    # Create Hann window
    ramp = signal.windows.hann(chunksizepts)
    
    # Get epoch data for A and B
    data_A = epochs_A.get_data(picks='eeg')
    data_B = epochs_B.get_data(picks='eeg')
    
    # Average across channels if multiple
    if data_A.shape[1] > 1:
        print(f"Averaging across {data_A.shape[1]} channels")
        data_A = np.mean(data_A, axis=1)  # Shape: (n_epochs, n_times)
        data_B = np.mean(data_B, axis=1)
    else:
        data_A = data_A[:, 0, :]
        data_B = data_B[:, 0, :]
    
    # Determine minimum number of sweeps
    numsweeps = min(len(epochs_A), len(epochs_B))
    data_A = data_A[:numsweeps, :]
    data_B = data_B[:numsweeps, :]
    
    n_epochs, n_samples = data_A.shape
    
    # Calculate sliding window positions
    segstart = np.arange(0, n_samples - chunksizepts, chunksizepts - overlappts)
    segstop = segstart + chunksizepts
    
    # Tile the window for all epochs
    bigramp = np.tile(ramp[:, np.newaxis], (1, n_epochs))
    
    # Initialize phase info for A and B: (freq, epochs, time_windows)
    phaseinfo_A = np.zeros((freqcap + 1, n_epochs, len(segstart)), dtype=complex)
    phaseinfo_B = np.zeros((freqcap + 1, n_epochs, len(segstart)), dtype=complex)
    
    # Process each time window
    print(f"Processing {len(segstart)} time windows, {n_epochs} epochs each for A and B")
    for s in range(len(segstart)):
        if s % 20 == 0:
            print(f"  Window {s+1}/{len(segstart)}")
        
        # Extract segments from all epochs
        segment_A = data_A[:, segstart[s]:segstop[s]].T  # Shape: (time, epochs)
        segment_B = data_B[:, segstart[s]:segstop[s]].T
        
        # Detrend (baseline to 0)
        segment_A_detr = signal.detrend(segment_A, axis=0, type='constant')
        segment_B_detr = signal.detrend(segment_B, axis=0, type='constant')
        
        # Apply Hann window
        segment_A_windowed = segment_A_detr * bigramp
        segment_B_windowed = segment_B_detr * bigramp
        
        # Compute FFT with 1 Hz resolution
        fft_A = np.fft.fft(segment_A_windowed, n=int(sfreq), axis=0)
        fft_B = np.fft.fft(segment_B_windowed, n=int(sfreq), axis=0)
        
        # Keep only frequencies up to freqcap
        fft_A = fft_A[:freqcap + 1, :]
        fft_B = fft_B[:freqcap + 1, :]
        
        # Normalize to get phase only (discard amplitude)
        fft_mag_A = np.abs(fft_A)
        fft_mag_B = np.abs(fft_B)
        phaseinfo_A[:, :, s] = fft_A / (fft_mag_A + 1e-10)
        phaseinfo_B[:, :, s] = fft_B / (fft_mag_B + 1e-10)
    
    # Average across epochs (complex average) for A and B
    phase_avg_A = np.mean(phaseinfo_A, axis=1)  # Shape: (freq, time_windows)
    phase_avg_B = np.mean(phaseinfo_B, axis=1)
    
    # Compute ADD and SUB from the averaged phase vectors
    phase_avg_add = (phase_avg_A + phase_avg_B) / 2
    phase_avg_sub = (phase_avg_A - phase_avg_B) / 2
    
    # Phase consistency is the absolute value
    phasecon = {
        'A': np.abs(phase_avg_A),
        'B': np.abs(phase_avg_B),
        'add': np.abs(phase_avg_add),
        'sub': np.abs(phase_avg_sub)
    }
    
    # Create axes
    xaxis = 1000 * np.linspace(
        tmin + (chunksize / 2),
        tmax - (chunksize / 2),
        len(segstart)
    )
    yaxis = np.arange(0, freqcap + 1)
    
    print(f"\nPhase consistency computed!")
    print(f"  Polarities: A, B, add, sub")
    print(f"  Number of sweeps: {numsweeps}")
    print(f"  Time axis: {xaxis[0]:.1f} to {xaxis[-1]:.1f} ms ({len(xaxis)} points)")
    print(f"  Freq axis: {yaxis[0]} to {yaxis[-1]} Hz ({len(yaxis)} points)")
    
    return phasecon, xaxis, yaxis, numsweeps


def plot_phase_consistency(
    phasecon,
    xaxis,
    yaxis,
    pol_names=None,
    vmax=0.14,
    ylim=None,
    figsize=(12, 8),
    cmap='viridis'
):
    """
    Plot phase consistency matrices.
    
    Parameters
    ----------
    phasecon : dict
        Dictionary of phase consistency arrays
    xaxis : ndarray
        Time axis in ms
    yaxis : ndarray
        Frequency axis in Hz
    pol_names : list, optional
        Order of polarities to plot. If None, uses dict order
    vmax : float
        Maximum value for colormap (default: 0.14)
    ylim : tuple, optional
        Frequency limits (min_freq, max_freq)
    figsize : tuple
        Figure size (default: (12, 8))
    cmap : str
        Colormap name (default: 'viridis')
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    
    if pol_names is None:
        pol_names = list(phasecon.keys())
    
    n_pols = len(pol_names)
    
    # Determine subplot layout
    if n_pols <= 2:
        nrows, ncols = 1, n_pols
    elif n_pols <= 4:
        nrows, ncols = 2, 2
    else:
        nrows = int(np.ceil(n_pols / 3))
        ncols = 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_pols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, pol_name in enumerate(pol_names):
        if pol_name not in phasecon:
            continue
        
        ax = axes[idx]
        
        im = ax.imshow(
            phasecon[pol_name],
            aspect='auto',
            origin='lower',
            extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
            vmin=0,
            vmax=vmax,
            cmap=cmap
        )
        
        if ylim is not None:
            ax.set_ylim(ylim)
        
        ax.set_title(pol_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        
        plt.colorbar(im, ax=ax, label='Phase Consistency')
    
    # Hide extra subplots
    for idx in range(len(pol_names), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_phase_consistency_masked(
    phasecon,
    xaxis,
    yaxis,
    numsweeps,
    alpha=0.01,
    pol_names=None,
    vmax=0.14,
    ylim=None,
    figsize=(12, 8)
):
    """
    Plot phase consistency with significance masking.
    
    Parameters
    ----------
    phasecon : dict
        Dictionary of phase consistency arrays
    xaxis : ndarray
        Time axis in ms
    yaxis : ndarray
        Frequency axis in Hz
    numsweeps : int
        Number of sweeps used in the analysis
    alpha : float
        Significance level (default: 0.01)
    pol_names : list, optional
        Order of polarities to plot
    vmax : float
        Maximum value for colormap
    ylim : tuple, optional
        Frequency limits
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    
    if pol_names is None:
        pol_names = list(phasecon.keys())
    
    # Calculate cutoff (same for all polarities)
    cutoff = np.sqrt(-np.log(alpha) / numsweeps)
    print(f"Significance cutoff (α={alpha}, n={numsweeps}): {cutoff:.4f}")
    
    # Create masked version
    maskedphasecon = {}
    for pol_name in pol_names:
        if pol_name not in phasecon:
            continue
        
        # Mask
        masked = phasecon[pol_name].copy()
        masked[masked < cutoff] = 0
        maskedphasecon[pol_name] = masked
    
    # Create colormap with black for zero
    cmap = plt.cm.viridis.copy()
    cmap.set_under('black')
    
    n_pols = len(pol_names)
    
    # Determine subplot layout
    if n_pols <= 2:
        nrows, ncols = 1, n_pols
    elif n_pols <= 4:
        nrows, ncols = 2, 2
    else:
        nrows = int(np.ceil(n_pols / 3))
        ncols = 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_pols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, pol_name in enumerate(pol_names):
        if pol_name not in maskedphasecon:
            continue
        
        ax = axes[idx]
        
        im = ax.imshow(
            maskedphasecon[pol_name],
            aspect='auto',
            origin='lower',
            extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
            vmin=cutoff,  # Values below are 'under' (black)
            vmax=vmax,
            cmap=cmap
        )
        
        if ylim is not None:
            ax.set_ylim(ylim)
        
        ax.set_title(f"{pol_name} (p < {alpha})", fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        
        plt.colorbar(im, ax=ax, label='Phase Consistency')
    
    # Hide extra subplots
    for idx in range(len(pol_names), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig

def corr_stim_to_resp(stim, resp, sfreq):
    minimum_length = min(len(stim), len(resp))
    stim = stim[:minimum_length, :]
    resp = resp[:minimum_length, :]

    corrs = signal.correlate(resp, stim, mode="full")
    corrs = corrs / (np.std(stim) * np.std(resp) * len(stim))

    lag = np.arange(-len(stim) + 1, len(resp))
    lag_milliseconds = lag / sfreq * 1000
    peak_n = np.argmax(corrs)
    peak_corr = corrs[peak_n]
    peak_lag = lag_milliseconds[peak_n]

    return peak_corr, peak_lag
