"""Tests for the ffrprep group-level aggregation step (ffrprep.group).

Functions only — no test classes (CLAUDE.md). Mocks via unittest.mock.patch
work the same on free functions as on methods.
"""
import json
from types import SimpleNamespace

import matplotlib
import mne
import numpy as np
import pytest

from ffrprep.group import (
    compute_grand_average,
    compute_subject_metrics,
    discover_group_inputs,
    run_group_level,
    save_group_metrics,
    save_group_outputs,
)

# Use non-interactive backend for testing
matplotlib.use("Agg")


def _make_evoked(seed, n_times=500, sfreq=1000.0, baseline=(-0.04, 0.0)):
    """A single-channel Evoked long enough for compute_power's TFR wavelets."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 1e-6, size=(1, n_times))
    info = mne.create_info(ch_names=["Cz"], sfreq=sfreq, ch_types=["eeg"])
    evoked = mne.EvokedArray(data, info, tmin=-0.04, nave=20, verbose=False)
    evoked.baseline = baseline
    return evoked


def _make_epochs(seed, n_epochs=10, n_times=500, sfreq=1000.0):
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 1e-6, size=(n_epochs, 1, n_times))
    info = mne.create_info(ch_names=["Cz"], sfreq=sfreq, ch_types=["eeg"])
    return mne.EpochsArray(data, info, tmin=-0.04, verbose=False)


def _write_analysis_derivatives(root, subject, task="ffr", run="1", diff=True):
    """Write a minimal ffrprep-analysis/sub-<subject>/ tree for one (task, run)."""
    subject_dir = root / "ffrprep-analysis" / f"sub-{subject}"
    subject_dir.mkdir(parents=True, exist_ok=True)
    base = f"sub-{subject}_task-{task}_run-{run}"

    evoked = _make_evoked(seed=int(subject))
    evoked.save(subject_dir / f"{base}_desc-evoked.fif", overwrite=True)

    if diff:
        diff_evoked = _make_evoked(seed=int(subject) + 100)
        diff_evoked.save(
            subject_dir / f"{base}_desc-evokedDiffPositiveVsNegative.fif", overwrite=True
        )
    return subject_dir


def _write_preproc_epochs(root, subject, task="ffr", run="1", condition="Positive"):
    subject_dir = root / "ffrprep-preprocessing" / f"sub-{subject}" / "eeg"
    subject_dir.mkdir(parents=True, exist_ok=True)
    epochs = _make_epochs(seed=int(subject) + 200)
    path = subject_dir / f"sub-{subject}_task-{task}_run-{run}_desc-preproc{condition}_epo.fif"
    epochs.save(path, overwrite=True)
    return path


# ---------------------------------------------------------------------------
# discover_group_inputs
# ---------------------------------------------------------------------------

def test_discover_group_inputs_groups_by_task_session_run(tmp_path):
    analysis_dir = tmp_path / "ffrprep-analysis"
    for subject in ["01", "02"]:
        subject_dir = analysis_dir / f"sub-{subject}"
        subject_dir.mkdir(parents=True)
        (subject_dir / f"sub-{subject}_task-ffr_run-1_desc-evoked.fif").touch()
        (subject_dir / f"sub-{subject}_task-ffr_run-1_desc-evokedDiffPositiveVsNegative.fif").touch()
        # A per-condition evoked file, which should NOT be picked up as
        # "combined" or "diff".
        (subject_dir / f"sub-{subject}_task-ffr_run-1_desc-evokedPositive.fif").touch()

    inputs = discover_group_inputs(tmp_path)

    assert set(inputs["evoked"].keys()) == {("ffr", None, "1")}
    assert set(inputs["evoked"][("ffr", None, "1")].keys()) == {"01", "02"}

    assert set(inputs["diff"].keys()) == {("ffr", None, "1", "Positive", "Negative")}


def test_discover_group_inputs_filters_by_participant_and_task(tmp_path):
    analysis_dir = tmp_path / "ffrprep-analysis"
    for subject, task in [("01", "ffr"), ("02", "ffr"), ("03", "click")]:
        subject_dir = analysis_dir / f"sub-{subject}"
        subject_dir.mkdir(parents=True)
        (subject_dir / f"sub-{subject}_task-{task}_desc-evoked.fif").touch()

    inputs = discover_group_inputs(tmp_path, participant_label=["01", "02"], task=["ffr"])

    assert set(inputs["evoked"][("ffr", None, None)].keys()) == {"01", "02"}


def test_discover_group_inputs_groups_preprocessing_epochs_by_subject(tmp_path):
    _write_preproc_epochs(tmp_path, "01", condition="Positive")
    _write_preproc_epochs(tmp_path, "01", condition="Negative")
    _write_preproc_epochs(tmp_path, "02", condition="Positive")

    inputs = discover_group_inputs(tmp_path)

    epochs = inputs["epochs"][("ffr", None, "1")]
    assert len(epochs["01"]) == 2
    assert len(epochs["02"]) == 1


# ---------------------------------------------------------------------------
# compute_grand_average / compute_subject_metrics
# ---------------------------------------------------------------------------

def test_compute_grand_average(tmp_path):
    paths = {}
    for subject in ["01", "02", "03"]:
        evoked = _make_evoked(seed=int(subject))
        path = tmp_path / f"sub-{subject}_evoked.fif"
        evoked.save(path, overwrite=True)
        paths[subject] = path

    grand_average, subjects = compute_grand_average(paths)

    assert subjects == ["01", "02", "03"]
    assert isinstance(grand_average, mne.Evoked)
    assert grand_average.data.shape == (1, 500)


def test_compute_subject_metrics_includes_response_consistency(tmp_path):
    evoked_paths = {}
    epochs_paths = {}
    for subject in ["01", "02"]:
        evoked = _make_evoked(seed=int(subject))
        evoked_path = tmp_path / f"sub-{subject}_evoked.fif"
        evoked.save(evoked_path, overwrite=True)
        evoked_paths[subject] = evoked_path

        epochs = _make_epochs(seed=int(subject))
        epochs_path = tmp_path / f"sub-{subject}_epo.fif"
        epochs.save(epochs_path, overwrite=True)
        epochs_paths[subject] = [epochs_path]

    metrics = compute_subject_metrics(evoked_paths, epochs_paths)

    assert list(metrics["subject"]) == ["01", "02"]
    assert metrics["rms_snr"].notna().all()
    assert metrics["band_power_90_110hz"].notna().all()
    assert metrics["response_consistency"].notna().all()


def test_compute_subject_metrics_without_epochs_leaves_consistency_null(tmp_path):
    evoked = _make_evoked(seed=1)
    evoked_path = tmp_path / "sub-01_evoked.fif"
    evoked.save(evoked_path, overwrite=True)

    metrics = compute_subject_metrics({"01": evoked_path})

    assert metrics.loc[0, "response_consistency"] is None


def test_compute_subject_metrics_skips_response_consistency_below_10_trials(tmp_path):
    """Matches the participant-level report's gate (reports.build_epoch_section):
    below 10 trials the pairwise-correlation estimate is unstable."""
    evoked = _make_evoked(seed=1)
    evoked_path = tmp_path / "sub-01_evoked.fif"
    evoked.save(evoked_path, overwrite=True)

    epochs = _make_epochs(seed=1, n_epochs=8)
    epochs_path = tmp_path / "sub-01_epo.fif"
    epochs.save(epochs_path, overwrite=True)

    metrics = compute_subject_metrics({"01": evoked_path}, {"01": [epochs_path]})

    assert metrics.loc[0, "response_consistency"] is None


# ---------------------------------------------------------------------------
# save_group_outputs / save_group_metrics
# ---------------------------------------------------------------------------

def test_save_group_outputs_writes_evoked_and_sidecar(tmp_path):
    grand_average = _make_evoked(seed=1)
    evoked_path = save_group_outputs(tmp_path, "ffr", None, "1", grand_average, ["01", "02"])

    assert evoked_path.exists()
    sidecar = json.loads(evoked_path.with_suffix(".json").read_text())
    assert sidecar["Subjects"] == ["01", "02"]
    assert sidecar["NumberOfSubjects"] == 2
    assert sidecar["Run"] == "1"
    assert (tmp_path / "ffrprep-group" / "dataset_description.json").exists()


def test_save_group_metrics_writes_tsv(tmp_path):
    import pandas as pd

    metrics = pd.DataFrame({"subject": ["01", "02"], "rms_snr": [3.1, 2.9]})
    metrics_path = save_group_metrics(tmp_path, "ffr", None, "1", metrics)

    assert metrics_path.exists()
    assert metrics_path.read_text().startswith("subject\trms_snr")


# ---------------------------------------------------------------------------
# run_group_level (end-to-end over a synthetic derivatives tree)
# ---------------------------------------------------------------------------

def test_run_group_level_writes_grand_average_metrics_and_report(tmp_path, capsys):
    for subject in ["01", "02", "03"]:
        _write_analysis_derivatives(tmp_path, subject)
        _write_preproc_epochs(tmp_path, subject, condition="Positive")

    args = SimpleNamespace(
        output_dir=tmp_path, participant_label=None, task=None, response_window=(0.100, 0.200),
    )

    written = run_group_level(args)

    group_dir = tmp_path / "ffrprep-group"
    assert (group_dir / "task-ffr_run-1_desc-grandAverage_ave.fif").exists()
    assert (group_dir / "task-ffr_run-1_desc-grandAverageDiffPositiveVsNegative_ave.fif").exists()
    assert (group_dir / "task-ffr_run-1_metrics.tsv").exists()
    assert (group_dir / "group_report.html").exists()
    assert all(path.exists() for path in written)


def test_run_group_level_skips_task_run_with_fewer_than_two_subjects(tmp_path, capsys):
    _write_analysis_derivatives(tmp_path, "01", diff=False)

    args = SimpleNamespace(
        output_dir=tmp_path, participant_label=None, task=None, response_window=(0.100, 0.200),
    )

    written = run_group_level(args)

    assert written == []
    assert not (tmp_path / "ffrprep-group").exists()
    assert "only 1 subject(s) found" in capsys.readouterr().out


def test_run_group_level_with_no_derivatives_prints_message_and_returns_empty(tmp_path, capsys):
    args = SimpleNamespace(
        output_dir=tmp_path, participant_label=None, task=None, response_window=(0.100, 0.200),
    )

    written = run_group_level(args)

    assert written == []
    assert "No participant-level ffrprep-analysis derivatives found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# per-type evokeds, polarity sum, harmonic / stimulus metrics, covariates
# ---------------------------------------------------------------------------

DA_SFREQ = 8192.0
DA_TMIN, DA_TMAX = -0.04, 0.213


def _chirp(t):
    """A 170 ms 100->120 Hz glide with a smooth envelope (unique xcorr peak)."""
    inside = (t >= 0) & (t <= 0.17)
    envelope = np.sin(np.pi * np.clip(t, 0, 0.17) / 0.17) ** 2
    phase = 2 * np.pi * (100.0 * t + 0.5 * (20.0 / 0.17) * t ** 2)
    return np.where(inside, np.sin(phase) * envelope, 0.0)


def _da_evoked(data_v, nave=3000):
    info = mne.create_info(["Cz"], DA_SFREQ, ch_types=["eeg"])
    evoked = mne.EvokedArray(
        data_v[np.newaxis, :], info, tmin=DA_TMIN, nave=nave, verbose=False,
    )
    evoked.baseline = (DA_TMIN, 0.0)
    return evoked


def _da_times():
    n = int(round((DA_TMAX - DA_TMIN) * DA_SFREQ)) + 1
    return DA_TMIN + np.arange(n) / DA_SFREQ


def _write_da_subject(root, subject, amplitude_uv=1.0, delay_s=0.012, seed=0):
    """Combined + two per-polarity evokeds; polarity artifact cancels in the sum."""
    rng = np.random.default_rng(seed)
    times = _da_times()
    signal = amplitude_uv * 1e-6 * _chirp(times - delay_s)
    artifact = 5e-6 * rng.normal(size=times.size)
    pol1, pol2 = signal + artifact, signal - artifact

    subject_dir = root / "ffrprep-analysis" / f"sub-{subject}"
    subject_dir.mkdir(parents=True, exist_ok=True)
    base = f"sub-{subject}_task-da_run-01"
    _da_evoked(pol1).save(subject_dir / f"{base}_desc-evokedDaPol1.fif", overwrite=True)
    _da_evoked(pol2).save(subject_dir / f"{base}_desc-evokedDaPol2.fif", overwrite=True)
    _da_evoked((pol1 + pol2) / 2, nave=6000).save(
        subject_dir / f"{base}_desc-evoked.fif", overwrite=True,
    )
    return signal


def _write_stimulus_wav(path, sfreq=24414):
    from scipy.io import wavfile

    t = np.arange(int(round(0.213 * sfreq))) / sfreq
    wavfile.write(path, sfreq, (_chirp(t) * 20000).astype(np.int16))


def test_discover_group_inputs_collects_per_type_evoked(tmp_path):
    for subject in ("01", "02"):
        _write_da_subject(tmp_path, subject, seed=int(subject))
    inputs = discover_group_inputs(tmp_path)

    key = ("da", None, "01")
    assert set(inputs["by_type"][key]["01"]) == {"DaPol1", "DaPol2"}
    assert set(inputs["by_type"][key].keys()) == {"01", "02"}
    # the combined file is still the "evoked" entry, not a per-type one
    assert set(inputs["evoked"][key].keys()) == {"01", "02"}
    assert inputs["diff"] == {}


def test_load_polarity_sum_is_the_plain_sum_and_needs_two_types(tmp_path):
    from ffrprep.group import _load_polarity_sum

    _write_da_subject(tmp_path, "01", seed=3)
    paths = discover_group_inputs(tmp_path)["by_type"][("da", None, "01")]["01"]
    summed = _load_polarity_sum(paths)

    first = mne.read_evokeds(str(paths["DaPol1"]), condition=0, verbose=False)
    second = mne.read_evokeds(str(paths["DaPol2"]), condition=0, verbose=False)
    np.testing.assert_allclose(summed.data, first.data + second.data, atol=1e-15)
    assert summed.baseline is not None
    assert _load_polarity_sum({"DaPol1": paths["DaPol1"]}) is None
    assert _load_polarity_sum(None) is None


def test_compute_subject_metrics_polarity_sum_harmonics_and_stimulus(tmp_path):
    from ffrprep.analysis import harmonic_amplitudes

    signal = _write_da_subject(tmp_path, "01", amplitude_uv=1.0, delay_s=0.012, seed=1)
    _write_da_subject(tmp_path, "02", amplitude_uv=2.0, delay_s=0.012, seed=2)
    stim_path = tmp_path / "stim.wav"
    _write_stimulus_wav(stim_path)

    inputs = discover_group_inputs(tmp_path)
    key = ("da", None, "01")
    metrics = compute_subject_metrics(
        inputs["evoked"][key],
        by_type_paths=inputs["by_type"][key],
        response_window=(0.0, 0.213),
        harmonics={"f0": 100.0, "n_harmonics": 10, "bin_hz": 60.0, "tmin": 0.06, "tmax": 0.18},
        stimulus={
            "path": stim_path, "stim_window": (0.05, 0.17), "resp_window": (0.06, 0.18),
        },
        n_trials_presented=6000,
    )

    assert list(metrics["subject"]) == ["01", "02"]
    assert list(metrics["n_avg"]) == [6000, 6000]
    assert metrics["usable_pct"].tolist() == [100.0, 100.0]
    for column in ("rms_snr_polarity_sum", "f0_uv", "upper_harmonics_uv",
                   "stim2resp_r", "stim2resp_z", "stim2resp_lag_ms"):
        assert metrics[column].notna().all(), column

    # harmonics come from 2 * signal (the sum of the two polarity averages)
    expected = harmonic_amplitudes(_da_evoked(2 * signal))
    assert metrics.loc[0, "f0_uv"] == pytest.approx(expected["f0"], rel=1e-6)
    assert metrics.loc[0, "upper_harmonics_uv"] == pytest.approx(expected["upper_harmonics"], rel=1e-6)
    # a 2x larger response gives 2x larger spectral amplitudes
    assert metrics.loc[1, "f0_uv"] == pytest.approx(2 * metrics.loc[0, "f0_uv"], rel=1e-6)
    # stimulus following: strong correlation, z = atanh(r), lag = delay - 10 ms window offset
    assert metrics.loc[0, "stim2resp_r"] > 0.9
    assert metrics.loc[0, "stim2resp_z"] == pytest.approx(np.arctanh(metrics.loc[0, "stim2resp_r"]))
    assert metrics.loc[0, "stim2resp_lag_ms"] == pytest.approx(2.0, abs=0.3)


def test_compute_subject_metrics_lag_limited_variant(tmp_path):
    _write_da_subject(tmp_path, "01", delay_s=0.012, seed=1)
    stim_path = tmp_path / "stim.wav"
    _write_stimulus_wav(stim_path)
    inputs = discover_group_inputs(tmp_path)
    key = ("da", None, "01")

    metrics = compute_subject_metrics(
        {"01": inputs["evoked"][key]["01"]},
        by_type_paths=inputs["by_type"][key],
        stimulus={
            "path": stim_path, "stim_window": (0.05, 0.17), "resp_window": (0.06, 0.18),
            "lag_range_ms": (6.9, 10.9), "lag_resp_window": (0.05, 0.20),
        },
    )
    # The lag-limited variant is confined to its window (the true 12 ms delay lies outside)...
    assert 6.9 <= metrics.loc[0, "stim2resp_lim_lag_ms"] <= 10.9
    # ...while the primary measure searches ALL lags: its windows are offset by 10 ms, so
    # the 12 ms delay appears at a 2 ms array lag, outside the restriction, with a
    # strictly higher correlation than anything inside the 6.9-10.9 ms window.
    assert metrics.loc[0, "stim2resp_lag_ms"] == pytest.approx(2.0, abs=0.3)
    assert metrics.loc[0, "stim2resp_r"] > metrics.loc[0, "stim2resp_lim_r"] + 0.05


def test_polarity_sum_metrics_are_nan_without_two_types(tmp_path, capsys):
    _write_da_subject(tmp_path, "01", seed=1)
    inputs = discover_group_inputs(tmp_path)
    key = ("da", None, "01")
    only_one = {"01": {"DaPol1": inputs["by_type"][key]["01"]["DaPol1"]}}

    metrics = compute_subject_metrics(
        {"01": inputs["evoked"][key]["01"]},
        by_type_paths=only_one,
        harmonics={"f0": 100.0},
    )
    assert np.isnan(metrics.loc[0, "f0_uv"])
    assert np.isnan(metrics.loc[0, "rms_snr_polarity_sum"])
    assert "exactly two per-trial-type" in capsys.readouterr().out


def test_metrics_without_new_options_keep_original_columns(tmp_path):
    _write_da_subject(tmp_path, "01", seed=1)
    inputs = discover_group_inputs(tmp_path)
    key = ("da", None, "01")
    metrics = compute_subject_metrics({"01": inputs["evoked"][key]["01"]})
    assert list(metrics.columns) == [
        "subject", "n_avg", "rms_snr", "band_power_90_110hz", "response_consistency",
    ]


def test_merge_covariates_joins_on_participant_id_without_overwriting(tmp_path, capsys):
    import pandas as pd

    from ffrprep.group import merge_covariates

    metrics = pd.DataFrame({"subject": ["bu001", "bu002", "zz999"], "f0_uv": [0.1, 0.2, 0.3]})
    participants = tmp_path / "participants.tsv"
    participants.write_text(
        "participant_id\tsite\tgroup\tage\nsub-bu001\tBU\tMus\t20\nsub-bu002\tBU\tNMus\t33\n"
    )
    phenotype = tmp_path / "pheno.tsv"
    phenotype.write_text("participant_id\tage\tdp\nsub-bu001\t99\t1.5\nsub-bu002\t99\t0.5\n")

    merged = merge_covariates(metrics, [participants, phenotype, tmp_path / "missing.tsv"])

    assert list(merged["group"][:2]) == ["Mus", "NMus"]
    assert pd.isna(merged.loc[2, "group"])  # subject without a covariate row
    assert list(merged["age"][:2]) == [20, 33]  # first file wins, not overwritten by pheno
    assert list(merged["dp"][:2]) == [1.5, 0.5]
    assert "covariate file not found" in capsys.readouterr().out


def test_run_group_level_end_to_end_with_harmonics_stimulus_and_covariates(tmp_path):
    import pandas as pd

    bids_dir = tmp_path / "bids"
    bids_dir.mkdir()
    (bids_dir / "participants.tsv").write_text(
        "participant_id\tsite\tgroup\n"
        "sub-01\tBU\tMus\nsub-02\tBU\tNMus\nsub-03\tCMU\tVar\n"
    )
    out = tmp_path / "out"
    for i, subject in enumerate(["01", "02", "03"], start=1):
        _write_da_subject(out, subject, amplitude_uv=float(i), seed=i)
    stim_path = tmp_path / "stim.wav"
    _write_stimulus_wav(stim_path)

    args = SimpleNamespace(
        output_dir=out, bids_dir=bids_dir, participant_label=None, task=None,
        response_window=(0.0, 0.213), f0=100.0, n_harmonics=10, harmonic_bin_hz=60.0,
        harmonic_window=(0.06, 0.18), stimulus=stim_path,
        xcorr_stim_window=(0.05, 0.17), xcorr_resp_window=(0.06, 0.18),
        xcorr_lag_range=None, xcorr_lag_resp_window=None,
        n_trials_presented=6000, covariates=None,
    )
    written = run_group_level(args)

    group_dir = out / "ffrprep-group"
    table = pd.read_csv(group_dir / "task-da_run-01_metrics.tsv", sep="\t", dtype={"subject": str})
    assert list(table["subject"]) == ["01", "02", "03"]
    assert list(table["group"]) == ["Mus", "NMus", "Var"]
    assert list(table["site"]) == ["BU", "BU", "CMU"]
    for column in ("f0_uv", "upper_harmonics_uv", "stim2resp_z", "usable_pct"):
        assert table[column].notna().all(), column
    assert table["f0_uv"].is_monotonic_increasing  # amplitudes 1, 2, 3 uV
    assert (group_dir / "group_report.html").exists()
    assert all(path.exists() for path in written)


def test_metrics_section_uses_histograms_for_large_cohorts():
    import pandas as pd

    from ffrprep.reports import build_metrics_table_section

    rng = np.random.default_rng(0)
    big = pd.DataFrame({
        "subject": [f"{i:03d}" for i in range(80)],
        "n_avg": 6000,
        "f0_uv": rng.normal(0.14, 0.03, 80),
        "upper_harmonics_uv": rng.normal(0.2, 0.05, 80),
    })
    small = big.head(5)

    assert build_metrics_table_section(big, section_id="a", title="t")["figures"][0]["title"] == (
        "Metric distributions"
    )
    assert build_metrics_table_section(small, section_id="b", title="t")["figures"][0]["title"] == (
        "Per-subject metrics"
    )
    assert "f0 uv" in build_metrics_table_section(big, section_id="a", title="t")["summary"]
