"""Tests for the ffrprep group-level aggregation step (ffrprep.group).

Functions only — no test classes (CLAUDE.md). Mocks via unittest.mock.patch
work the same on free functions as on methods.
"""
import json
from types import SimpleNamespace

import matplotlib
import mne
import numpy as np

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
