"""Numerical correctness tests for analysis functions.

These tests build synthetic Evoked responses with known frequency content
and assert that the analysis functions return values consistent with that
content. Synthetic data with a known answer is the strongest way to catch
silent regressions where the function still returns something but the
something is wrong.
"""
import numpy as np
import pytest

import mne

from ffrprep.analysis import autocorrelation, compute_power, rms_snr


SFREQ = 1000.0
TMIN = -0.2
TMAX = 0.5
SIGNAL_FREQ_HZ = 100.0


def _make_sine_evoked(freq_hz, sfreq=SFREQ, tmin=TMIN, tmax=TMAX, ch_name="Cz"):
    """Build a single-channel mne.Evoked carrying a pure sine wave."""
    n_samples = int(round((tmax - tmin) * sfreq)) + 1
    times = np.linspace(tmin, tmax, n_samples)
    data = np.sin(2 * np.pi * freq_hz * times)[np.newaxis, :]
    info = mne.create_info(ch_names=[ch_name], sfreq=sfreq, ch_types="eeg")
    return mne.EvokedArray(data, info, tmin=tmin, nave=1, verbose="ERROR")


def _make_evoked_with_response(
    noise_amp=0.01, signal_amp=1.0, sfreq=SFREQ, tmin=TMIN, tmax=TMAX,
    response_lower=0.1, response_upper=0.2, freq_hz=SIGNAL_FREQ_HZ, seed=42,
):
    """Evoked with low-amplitude noise everywhere + sine in response window.

    Used for rms_snr correctness: the response window has a much larger
    signal than the baseline, so SNR should be well above 1.
    """
    n_samples = int(round((tmax - tmin) * sfreq)) + 1
    times = np.linspace(tmin, tmax, n_samples)

    rng = np.random.default_rng(seed)
    data = rng.normal(scale=noise_amp, size=n_samples)

    response_mask = (times >= response_lower) & (times <= response_upper)
    data[response_mask] += signal_amp * np.sin(2 * np.pi * freq_hz * times[response_mask])

    info = mne.create_info(ch_names=["Cz"], sfreq=sfreq, ch_types="eeg")
    return mne.EvokedArray(
        data[np.newaxis, :], info, tmin=tmin, nave=1,
        baseline=(tmin, 0.0), verbose="ERROR",
    )


def test_compute_power_in_band_dominates_out_of_band():
    """compute_power should report higher power inside the signal's band."""
    evoked = _make_sine_evoked(freq_hz=SIGNAL_FREQ_HZ)

    # Band centered on the synthetic signal
    p_match = compute_power(
        evoked, f_low=90, f_high=110, t_low=0.1, t_high=0.2,
    )

    # Band well above the signal — should be near zero
    p_off = compute_power(
        evoked, f_low=200, f_high=300, t_low=0.1, t_high=0.2,
    )

    assert np.isfinite(p_match), "in-band power should be a finite number"
    assert np.isfinite(p_off), "out-of-band power should be a finite number"
    assert p_match > 0
    # In-band power must materially dominate out-of-band. The exact ratio
    # depends on multitaper spectral spread (n_cycles=2, time_bandwidth=4.0
    # bleed energy across nearby frequencies); empirically a pure 100 Hz
    # sine over 700 ms gives about 2.5–3x dominance. The 2x floor catches
    # any regression that would let out-of-band power equal or exceed
    # in-band, which would mean the function is no longer band-selective.
    assert p_match > 2 * p_off, (
        f"Expected in-band power to dominate out-of-band by >2x; "
        f"got p_match={p_match!r}, p_off={p_off!r}"
    )


def test_rms_snr_response_window_dominates_baseline():
    """rms_snr should report SNR well above 1 when the response amplitude
    far exceeds the baseline noise level."""
    evoked = _make_evoked_with_response(noise_amp=0.01, signal_amp=1.0)
    snr = rms_snr(evoked, response_lower=0.1, response_upper=0.2)

    assert np.isfinite(snr)
    # Response RMS ≈ 1/sqrt(2) ≈ 0.707; baseline RMS ≈ 0.01 → SNR ≈ 70.
    # Floor at 5 keeps the assertion robust to RNG variation.
    assert snr > 5, f"Expected SNR much greater than 1; got {snr!r}"


def test_autocorrelation_periodic_signal_peaks_at_period():
    """ACF of a pure 100 Hz sine at sfreq=1000 should peak at lag=10 samples
    (one full period) and be near -1 at lag=5 samples (half period)."""
    evoked = _make_sine_evoked(freq_hz=SIGNAL_FREQ_HZ)
    acf, _ci = autocorrelation(evoked)

    # ACF at lag 0 is 1 by definition
    assert acf[0] == pytest.approx(1.0, abs=1e-9)
    # One full period later → strongly correlated
    assert acf[10] > 0.9, f"ACF at lag=10 (one period) was {acf[10]!r}"
    # Half period later → strongly anti-correlated
    assert acf[5] < -0.9, f"ACF at lag=5 (half period) was {acf[5]!r}"


# ---------------------------------------------------------------------------
# _xcorr_normalized: shared cross-correlation building block
# ---------------------------------------------------------------------------

def test_xcorr_normalized_returns_corrs_and_lags():
    """``_xcorr_normalized`` returns the full normalized cross-correlation + lag axis."""
    from ffrprep.analysis import _xcorr_normalized

    rng = np.random.default_rng(0)
    a = rng.standard_normal(256)
    b = rng.standard_normal(256)
    sfreq = 1000.0

    corrs, lag_ms = _xcorr_normalized(a, b, sfreq)
    # signal.correlate(b, a, mode="full") gives 2*N - 1 lags
    assert len(corrs) == 2 * 256 - 1
    assert len(lag_ms) == len(corrs)
    # Lag axis is monotonic and centred on 0
    assert lag_ms[0] < 0 < lag_ms[-1]
    assert abs(lag_ms[0] + lag_ms[-1]) < 1e-6, (
        "lag axis should be symmetric around 0 for equal-length inputs"
    )


def test_xcorr_normalized_peak_matches_corr_stim_to_resp():
    """Peak of ``_xcorr_normalized`` matches the scalars from corr_stim_to_resp.

    ``corr_stim_to_resp`` returns ``(corrs[argmax], lag_ms[argmax])``
    from the shared helper; regression-lock the equality.
    """
    from ffrprep.analysis import _xcorr_normalized, corr_stim_to_resp

    rng = np.random.default_rng(1)
    stim = rng.standard_normal(512)
    resp = rng.standard_normal(512)
    sfreq = 2000.0

    corrs, lag_ms = _xcorr_normalized(stim, resp, sfreq)
    peak_idx = int(np.argmax(corrs))
    expected_peak_r = corrs[peak_idx]
    expected_peak_lag = lag_ms[peak_idx]

    peak_r, peak_lag = corr_stim_to_resp(stim, resp, sfreq)
    assert peak_r == expected_peak_r
    assert peak_lag == expected_peak_lag


def _pairwise_pearson_reference(data):
    """The original O(n**2) scipy.stats.pearsonr loop, kept as the oracle."""
    from scipy.stats import pearsonr

    r_vals = []
    for i in range(data.shape[0]):
        for j in range(i + 1, data.shape[0]):
            r_vals.append(pearsonr(data[i], data[j])[0])
    return np.array(r_vals)


def _make_epochs(n_epochs, n_channels=1, n_times=200, seed=0, sfreq=1000.0):
    rng = np.random.default_rng(seed)
    shared = np.sin(2 * np.pi * 100.0 * np.arange(n_times) / sfreq)
    data = shared + rng.normal(scale=2.0, size=(n_epochs, n_channels, n_times))
    info = mne.create_info(
        ch_names=[f"E{i}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg",
    )
    return mne.EpochsArray(data * 1e-6, info, tmin=0.0, verbose="ERROR")


@pytest.mark.parametrize("n_channels", [1, 3])
def test_response_consistency_matches_pairwise_pearson_loop(n_channels):
    """Vectorized np.corrcoef must reproduce the pairwise pearsonr loop."""
    from ffrprep.analysis import response_consistency

    epochs = _make_epochs(n_epochs=15, n_channels=n_channels, seed=3)
    mean_r, r_vals = response_consistency(epochs)

    data = epochs.get_data(picks="eeg")
    data = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0, :]
    expected = _pairwise_pearson_reference(data)

    assert r_vals.shape == (15 * 14 // 2,)
    np.testing.assert_allclose(r_vals, expected, rtol=0, atol=1e-10)
    assert mean_r == pytest.approx(expected.mean(), abs=1e-10)


def test_response_consistency_honors_time_window():
    from ffrprep.analysis import response_consistency

    epochs = _make_epochs(n_epochs=8, seed=5)
    full_mean, _ = response_consistency(epochs)
    cropped_mean, r_vals = response_consistency(epochs, tmin=0.05, tmax=0.15)

    data = epochs.copy().crop(tmin=0.05, tmax=0.15).get_data(picks="eeg")[:, 0, :]
    np.testing.assert_allclose(r_vals, _pairwise_pearson_reference(data), atol=1e-10)
    assert cropped_mean != pytest.approx(full_mean)


def test_response_consistency_single_epoch_returns_nan_and_empty():
    from ffrprep.analysis import response_consistency

    epochs = _make_epochs(n_epochs=1)
    with pytest.warns(RuntimeWarning):
        mean_r, r_vals = response_consistency(epochs)
    assert r_vals.size == 0
    assert np.isnan(mean_r)


# ---------------------------------------------------------------------------
# harmonic_amplitudes / stim_to_resp_xcorr / wav + resampling helpers
# ---------------------------------------------------------------------------

def _harmonic_evoked(amps_uv, f0=100.0, sfreq=16384.0, tmin=-0.04, tmax=0.213, phase=0.3):
    """Evoked (Volts) made of sines at k*f0 with given amplitudes in microvolts."""
    n = int(round((tmax - tmin) * sfreq)) + 1
    times = tmin + np.arange(n) / sfreq
    data = np.zeros(n)
    for k, amp in enumerate(amps_uv, start=1):
        data += amp * 1e-6 * np.sin(2 * np.pi * k * f0 * times + phase * k)
    info = mne.create_info(["Cz"], sfreq, "eeg")
    return mne.EvokedArray(data[np.newaxis, :], info, tmin=tmin, nave=1, verbose="ERROR")


def _reference_harmonic_amplitudes(evoked, f0, n_harmonics, bin_hz, tmin, tmax, pad_pow):
    """Straight-line restatement of the published MATLAB recipe (dataFFT.m)."""
    sfreq = evoked.info["sfreq"]
    # Nearest-sample, inclusive window (MATLAB-style index lookup).
    first = int(round((tmin - evoked.tmin) * sfreq))
    last = int(round((tmax - evoked.tmin) * sfreq))
    x = evoked.data[0][first:last + 1] * 1e6
    length = x.size
    nfft = 2 ** (int(np.ceil(np.log2(length))) + pad_pow)
    amp = 2 * np.abs(np.fft.fft(x, nfft) / length)
    freqs = np.arange(nfft) * sfreq / nfft
    out = []
    for k in range(1, n_harmonics + 1):
        band = (freqs >= k * f0 - bin_hz / 2) & (freqs <= k * f0 + bin_hz / 2)
        out.append(amp[band].mean())
    return np.array(out)


def test_harmonic_amplitudes_matches_reference_recipe():
    from ffrprep.analysis import harmonic_amplitudes

    evoked = _harmonic_evoked([1.0, 0.5, 0.25, 0.1, 0.05])
    result = harmonic_amplitudes(evoked)
    expected = _reference_harmonic_amplitudes(
        evoked, f0=100.0, n_harmonics=10, bin_hz=60.0, tmin=0.06, tmax=0.18, pad_pow=2,
    )
    np.testing.assert_allclose(result["harmonics"], expected, rtol=1e-12)
    assert result["f0"] == pytest.approx(expected[0])
    assert result["upper_harmonics"] == pytest.approx(expected[1:].sum())


def test_harmonic_amplitudes_scales_linearly_and_reports_microvolts():
    from ffrprep.analysis import harmonic_amplitudes

    base = harmonic_amplitudes(_harmonic_evoked([1.0, 0.5]))
    doubled = harmonic_amplitudes(_harmonic_evoked([2.0, 1.0]))
    assert doubled["f0"] == pytest.approx(2 * base["f0"])
    assert doubled["upper_harmonics"] == pytest.approx(2 * base["upper_harmonics"])
    # A 1 uV tone yields a microvolt-scale (not volt-scale) amplitude.
    assert 1e-3 < base["f0"] < 1.0

    single_uv = harmonic_amplitudes(_harmonic_evoked([1.0]))
    single_volts = harmonic_amplitudes(_harmonic_evoked([1.0]), unit_scale=1.0)
    assert single_volts["f0"] == pytest.approx(single_uv["f0"] * 1e-6)


def test_harmonic_amplitudes_separates_fundamental_from_upper_harmonics():
    from ffrprep.analysis import harmonic_amplitudes

    fundamental_only = harmonic_amplitudes(_harmonic_evoked([1.0]))
    harmonics_only = harmonic_amplitudes(_harmonic_evoked([0.0, 1.0, 1.0, 1.0]))
    assert fundamental_only["f0"] > fundamental_only["upper_harmonics"] * 5
    assert harmonics_only["upper_harmonics"] > harmonics_only["f0"] * 5


def test_harmonic_amplitudes_validates_inputs():
    from ffrprep.analysis import harmonic_amplitudes

    low_rate = _harmonic_evoked([1.0], sfreq=1000.0)
    with pytest.raises(ValueError, match="Nyquist"):
        harmonic_amplitudes(low_rate)
    with pytest.raises(ValueError, match="fewer than 2 samples"):
        harmonic_amplitudes(_harmonic_evoked([1.0]), tmin=0.1, tmax=0.1)


def test_stim_to_resp_xcorr_recovers_delay_and_matches_numpy_reference():
    from ffrprep.analysis import stim_to_resp_xcorr

    rng = np.random.default_rng(4)
    sfreq = 16384.0
    stim = rng.normal(size=1967)
    delay = 164  # samples (~10 ms)
    resp = np.concatenate([np.zeros(delay), stim])[: stim.size] + 0.05 * rng.normal(size=stim.size)

    r, z, lag_ms = stim_to_resp_xcorr(stim, resp, sfreq)

    full = np.correlate(resp, stim, mode="full") / np.sqrt(np.sum(stim**2) * np.sum(resp**2))
    assert r == pytest.approx(full.max())
    assert z == pytest.approx(np.arctanh(r))
    assert lag_ms == pytest.approx(delay / sfreq * 1000.0)


def test_stim_to_resp_xcorr_does_not_remove_the_mean():
    """'coeff' normalization (MATLAB xcorr) keeps DC: not a Pearson r."""
    from ffrprep.analysis import stim_to_resp_xcorr

    stim = np.ones(200)
    resp = np.ones(200)
    r, _, lag_ms = stim_to_resp_xcorr(stim, resp, 1000.0)
    assert r == pytest.approx(1.0)
    assert lag_ms == pytest.approx(0.0)


def test_stim_to_resp_xcorr_lag_range_restricts_search():
    from ffrprep.analysis import stim_to_resp_xcorr

    rng = np.random.default_rng(6)
    sfreq = 16384.0
    stim = rng.normal(size=1500)
    resp = np.concatenate([np.zeros(164), stim])[:1500]  # true lag ~ 10 ms
    r_all, _, lag_all = stim_to_resp_xcorr(stim, resp, sfreq)
    r_win, _, lag_win = stim_to_resp_xcorr(stim, resp, sfreq, lag_range_ms=(0.0, 5.0))
    assert lag_all == pytest.approx(164 / sfreq * 1000)
    assert 0.0 <= lag_win <= 5.0 + 1e-6
    assert r_win < r_all
    # Window containing the true lag finds it again.
    r_in, _, lag_in = stim_to_resp_xcorr(stim, resp, sfreq, lag_range_ms=(6.9, 10.9))
    assert lag_in == pytest.approx(lag_all)
    assert r_in == pytest.approx(r_all)


def test_stim_to_resp_xcorr_degenerate_inputs():
    from ffrprep.analysis import stim_to_resp_xcorr

    r, z, lag = stim_to_resp_xcorr(np.zeros(50), np.ones(50), 1000.0)
    assert np.isnan(r) and np.isnan(z) and np.isnan(lag)
    with pytest.raises(ValueError):
        stim_to_resp_xcorr([], [1.0], 1000.0)
    r, _, _ = stim_to_resp_xcorr(np.ones(10), np.ones(10), 1000.0, lag_range_ms=(500.0, 600.0))
    assert np.isnan(r)


def test_load_wav_mono_and_resample_signal(tmp_path):
    from scipy.io import wavfile

    from ffrprep.analysis import load_wav_mono, resample_signal

    sfreq_in = 24414
    t = np.arange(4151) / sfreq_in
    left = (np.sin(2 * np.pi * 100 * t) * 20000).astype(np.int16)
    right = (np.sin(2 * np.pi * 100 * t) * 10000).astype(np.int16)
    path = tmp_path / "stim.wav"
    wavfile.write(path, sfreq_in, np.column_stack([left, right]))

    data, sfreq = load_wav_mono(path)
    assert sfreq == sfreq_in
    assert data.ndim == 1
    np.testing.assert_allclose(data, (left.astype(float) + right.astype(float)) / 2)

    out = resample_signal(data, sfreq_in, 16384.0)
    assert out.size == pytest.approx(data.size * 16384 / sfreq_in, abs=2)
    same = resample_signal(data, 16384.0, 16384.0)
    np.testing.assert_array_equal(same, np.asarray(data, dtype=float))
    # A 100 Hz tone stays a 100 Hz tone after resampling.
    spectrum = np.abs(np.fft.rfft(out))
    peak_hz = np.fft.rfftfreq(out.size, 1 / 16384.0)[np.argmax(spectrum)]
    assert peak_hz == pytest.approx(100.0, abs=3.0)
