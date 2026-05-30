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
