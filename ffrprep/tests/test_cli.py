"""Tests for the ffrprep BIDS-App CLI.

Functions only — no test classes (CLAUDE.md). Mocks via unittest.mock.patch
work the same on free functions as on methods.
"""
import argparse
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ffrprep.ffrprep_cli import (
    _resolve_condition_labels,
    _restore_epochs_baseline,
    get_parser,
    parse_baseline,
    parse_ref_channels,
    run_ffrprep,
)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def test_parser_creation():
    parser = get_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    assert "ffrprep" in parser.description


def test_required_arguments():
    parser = get_parser()
    args = parser.parse_args([
        "/path/to/bids", "/path/to/output", "participant",
    ])
    assert args.bids_dir == Path("/path/to/bids")
    assert args.output_dir == Path("/path/to/output")
    assert args.analysis_level == "participant"


def test_optional_arguments_defaults():
    """Default values for optional arguments.

    --ref_channels and --baseline were converted from string options to
    nargs="+"/nargs=2 in the CLI harmonization pass; default is None for
    ref_channels (treated as "average" downstream) and a [start, end]
    float list for baseline.
    """
    parser = get_parser()
    args = parser.parse_args([
        "/path/to/bids", "/path/to/output", "participant",
    ])
    assert args.stage == "both"
    assert args.ref_channels is None
    assert args.high_pass == 1.0
    assert args.low_pass == 40.0
    assert args.baseline == [-0.2, 0.0]
    assert args.tmin == -0.2
    assert args.tmax == 0.6
    assert args.n_procs == 1
    # Default flipped: per-trial-type outputs are emitted out of the box
    # for the canonical FFR setup. --no-split-by-trial-type opts out.
    assert args.split_by_trial_type is True
    assert args.trial_types is None
    assert args.difference_pairs is None
    assert args.skip_bids_validation is False
    assert args.participant_label is None
    assert args.work_dir is None


def test_analysis_level_choices():
    parser = get_parser()
    assert parser.parse_args(["/bids", "/output", "participant"]).analysis_level == "participant"
    assert parser.parse_args(["/bids", "/output", "group"]).analysis_level == "group"
    with pytest.raises(SystemExit):
        parser.parse_args(["/bids", "/output", "invalid"])


def test_stage_choices():
    parser = get_parser()
    for stage in ["preprocessing", "analysis", "both"]:
        args = parser.parse_args(["/bids", "/output", "participant", "--stage", stage])
        assert args.stage == stage
    with pytest.raises(SystemExit):
        parser.parse_args(["/bids", "/output", "participant", "--stage", "invalid"])


def test_participant_label_multiple():
    parser = get_parser()
    args = parser.parse_args([
        "/bids", "/output", "participant",
        "--participant_label", "01", "02", "03",
    ])
    assert args.participant_label == ["01", "02", "03"]


def test_preprocessing_parameters():
    """ref_channels uses nargs="+", so a single channel still arrives as a list."""
    parser = get_parser()
    args = parser.parse_args(["/bids", "/output", "participant", "--ref_channels", "Cz"])
    assert args.ref_channels == ["Cz"]

    args = parser.parse_args(["/bids", "/output", "participant", "--high_pass", "0.5"])
    assert args.high_pass == 0.5

    args = parser.parse_args(["/bids", "/output", "participant", "--low_pass", "30.0"])
    assert args.low_pass == 30.0


def test_analysis_parameters_split_default_true():
    """Default is to split outputs by trial type."""
    parser = get_parser()
    args = parser.parse_args(["/bids", "/output", "participant"])
    assert args.split_by_trial_type is True


def test_no_split_by_trial_type_opt_out():
    """--no-split-by-trial-type forces a single combined output."""
    parser = get_parser()
    args = parser.parse_args([
        "/bids", "/output", "participant",
        "--no-split-by-trial-type",
    ])
    assert args.split_by_trial_type is False


def test_by_event_type_deprecated_alias():
    """--by_event_type is preserved as a deprecated alias.

    Both the long-form ``--split-by-trial-type`` and the legacy
    ``--by_event_type`` resolve to the same destination so existing
    user scripts keep working for one release.
    """
    parser = get_parser()
    args = parser.parse_args([
        "/bids", "/output", "participant", "--by_event_type",
    ])
    assert args.split_by_trial_type is True


def test_trial_types_subset():
    """--trial-types accepts one or more trial-type tokens."""
    parser = get_parser()
    args = parser.parse_args([
        "/bids", "/output", "participant",
        "--trial-types", "Pos", "Neg",
    ])
    assert args.trial_types == ["Pos", "Neg"]

    args = parser.parse_args([
        "/bids", "/output", "participant",
        "--trial-types", "Pos",
    ])
    assert args.trial_types == ["Pos"]


def test_difference_pairs_argument():
    """--difference-pairs accepts ``A:B`` tokens, parsed downstream."""
    parser = get_parser()
    args = parser.parse_args([
        "/bids", "/output", "participant",
        "--difference-pairs", "Pos:Neg", "Tone1:Tone2",
    ])
    # Stored as raw string list at parse time; parse_difference_pairs
    # converts to list of tuples.
    assert args.difference_pairs == ["Pos:Neg", "Tone1:Tone2"]


def test_general_options():
    parser = get_parser()
    args = parser.parse_args([
        "/bids", "/output", "participant",
        "--skip_bids_validation",
        "--n_procs", "4",
    ])
    assert args.skip_bids_validation is True
    assert args.n_procs == 4


def test_version_argument():
    parser = get_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--version"])
    assert excinfo.value.code == 0


# ---------------------------------------------------------------------------
# parse_baseline
# ---------------------------------------------------------------------------

def test_parse_baseline_comma_separated():
    assert parse_baseline("-0.2,0.0") == (-0.2, 0.0)
    assert parse_baseline("-0.1,0.1") == (-0.1, 0.1)
    assert parse_baseline("0,0.5") == (0.0, 0.5)


def test_parse_baseline_single_value():
    assert parse_baseline("-0.2") == (-0.2, 0.0)
    assert parse_baseline("0.1") == (0.1, 0.0)


def test_parse_baseline_invalid_format():
    with pytest.raises(ValueError):
        parse_baseline("invalid")
    with pytest.raises(ValueError):
        parse_baseline("0.1,invalid")


# ---------------------------------------------------------------------------
# parse_ref_channels
# ---------------------------------------------------------------------------

def test_parse_ref_channels_average():
    assert parse_ref_channels("average") is None
    assert parse_ref_channels("AVERAGE") is None
    assert parse_ref_channels("Average") is None


def test_parse_ref_channels_single():
    assert parse_ref_channels("Cz") == "Cz"
    assert parse_ref_channels("TP9") == "TP9"


def test_parse_ref_channels_multiple():
    assert parse_ref_channels("TP9,TP10") == ["TP9", "TP10"]
    assert parse_ref_channels("Cz,FCz,CPz") == ["Cz", "FCz", "CPz"]


def test_parse_ref_channels_empty_string():
    """An empty string normalizes to None (no reference channel given)."""
    assert parse_ref_channels("") is None


# ---------------------------------------------------------------------------
# run_ffrprep — mocked end-to-end behavior
# ---------------------------------------------------------------------------

@patch("ffrprep.ffrprep_cli.reports")
@patch("mne.read_epochs")
@patch("ffrprep.ffrprep_cli.get_parser")
@patch("ffrprep.ffrprep_cli.validate_input_dir")
@patch("ffrprep.ffrprep_cli.get_participants")
@patch("ffrprep.ffrprep_cli.get_sessions_tasks_runs")
@patch("ffrprep.ffrprep_cli.setup_derivatives_directories")
@patch("ffrprep.ffrprep_cli.create_preprocessing_workflow")
@patch("ffrprep.ffrprep_cli.create_analysis_workflow")
def test_run_ffrprep_both_stages(
    mock_analysis_wf, mock_preproc_wf, mock_setup_dirs,
    mock_get_sessions, mock_get_participants, mock_validate, mock_parser,
    mock_read_epochs, mock_reports,
):
    """Run the full preprocessing+analysis path with mocked dependencies."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_args = MagicMock()
        mock_args.bids_dir = Path(tmp_dir) / "bids"
        mock_args.output_dir = Path(tmp_dir) / "output"
        mock_args.analysis_level = "participant"
        mock_args.stage = "both"
        mock_args.skip_bids_validation = False
        mock_args.participant_label = None
        mock_args.baseline = "-0.2,0"
        mock_args.ref_channels = "average"
        mock_args.high_pass = 1.0
        mock_args.low_pass = 40.0
        mock_args.tmin = -0.2
        mock_args.tmax = 0.6
        mock_args.by_event_type = False
        mock_args.work_dir = None

        # The harmonized CLI reads many additional flags via args.X.
        # MagicMock() returns truthy MagicMock objects for unset attrs,
        # which would short-circuit boolean checks like `if args.no_filter:`
        # and `if args.task:` in unexpected ways. Set them all explicitly.
        mock_args.task = None
        mock_args.run = None
        mock_args.concat_runs = False
        mock_args.no_filter = False
        mock_args.no_auto_reject = False
        mock_args.save_each_node = False
        mock_args.events_file = None
        mock_args.picks = None
        mock_args.event_id = None
        mock_args.reject_eeg = 75e-6
        mock_args.on_missing = "warn"
        mock_args.l_freq = None
        mock_args.h_freq = None
        mock_args.n_procs = 1

        mock_parser.return_value.parse_args.return_value = mock_args
        mock_get_participants.return_value = ["01"]
        mock_get_sessions.return_value = {
            "sessions": [None],
            "tasks": ["passive"],
            "runs": [1],
        }

        derivatives_root = Path(tmp_dir) / "derivatives"
        preprocessing_subject_dir = (
            derivatives_root / "ffrprep-preprocessing" / "sub-01"
        )
        # Create the directory and a placeholder .fif so the CLI's
        # file-presence guard (which globs for *_desc-preproc_epo.fif
        # after preprocessing returns) finds something. The workflow is
        # mocked so it wouldn't actually write the file.
        preprocessing_subject_dir.mkdir(parents=True, exist_ok=True)
        (
            preprocessing_subject_dir
            / "sub-01_task-passive_run-1_desc-preproc_epo.fif"
        ).touch()
        mock_setup_dirs.return_value = {
            "derivatives_root": derivatives_root,
            "preprocessing_dir": derivatives_root / "ffrprep-preprocessing",
            "preprocessing_subject_dir": preprocessing_subject_dir,
            "analysis_dir": derivatives_root / "ffrprep-analysis",
            "analysis_subject_dir": (
                derivatives_root / "ffrprep-analysis" / "sub-01"
            ),
        }

        mock_preproc_workflow = MagicMock()
        mock_preproc_wf.return_value = mock_preproc_workflow
        mock_analysis_workflow = MagicMock()
        mock_analysis_wf.return_value = mock_analysis_workflow

        run_ffrprep()

        mock_validate.assert_called_once()
        mock_get_participants.assert_called_once_with(
            str(Path(tmp_dir) / "bids"), None,
        )
        mock_setup_dirs.assert_called_once()
        mock_preproc_wf.assert_called_once()
        mock_analysis_wf.assert_called_once()
        mock_preproc_workflow.run.assert_called_once()
        mock_analysis_workflow.run.assert_called_once()


@patch("ffrprep.ffrprep_cli.get_parser")
@patch("ffrprep.ffrprep_cli.get_participants")
def test_run_ffrprep_no_participants(mock_get_participants, mock_parser):
    """Behavior when no participants are found."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_args = MagicMock()
        mock_args.bids_dir = Path(tmp_dir) / "bids"
        mock_args.output_dir = Path(tmp_dir) / "output"
        mock_args.analysis_level = "participant"
        mock_args.skip_bids_validation = True
        mock_args.participant_label = ["99"]

        mock_parser.return_value.parse_args.return_value = mock_args
        mock_get_participants.return_value = []

        with patch("builtins.print") as mock_print:
            run_ffrprep()

        mock_print.assert_any_call("No participants found matching ['99']")


@patch("ffrprep.ffrprep_cli.get_parser")
def test_run_ffrprep_group_level_not_supported(mock_parser):
    """Group level analysis returns early with a helpful message."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_args = MagicMock()
        mock_args.bids_dir = Path(tmp_dir) / "bids"
        mock_args.output_dir = Path(tmp_dir) / "output"
        mock_args.analysis_level = "group"

        mock_parser.return_value.parse_args.return_value = mock_args

        with patch("builtins.print") as mock_print:
            run_ffrprep()

        mock_print.assert_any_call(
            "Currently only participant-level analysis is supported."
        )


@patch("ffrprep.ffrprep_cli.get_parser")
@patch("ffrprep.ffrprep_cli.get_participants")
@patch("ffrprep.ffrprep_cli.setup_derivatives_directories")
def test_run_ffrprep_analysis_missing_preprocessing(
    mock_setup_dirs, mock_get_participants, mock_parser,
):
    """Analysis-only mode when no preprocessing outputs exist on disk.

    The CLI now globs ``preprocessing_subject_dir`` for ``*_desc-preproc_epo.fif``
    files and prints an actionable error if none are found, instead of
    calling a separate ``check_preprocessing_exists`` helper.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_args = MagicMock()
        mock_args.bids_dir = Path(tmp_dir) / "bids"
        mock_args.output_dir = Path(tmp_dir) / "output"
        mock_args.analysis_level = "participant"
        mock_args.stage = "analysis"
        mock_args.skip_bids_validation = True
        mock_args.participant_label = None

        mock_parser.return_value.parse_args.return_value = mock_args
        mock_get_participants.return_value = ["01"]

        # Real (empty) directories so the CLI's glob returns nothing.
        derivatives_root = Path(tmp_dir) / "derivatives"
        preproc_subject_dir = (
            derivatives_root / "ffrprep-preprocessing" / "sub-01" / "eeg"
        )
        preproc_subject_dir.mkdir(parents=True)
        mock_setup_dirs.return_value = {
            "derivatives_root": derivatives_root,
            "preprocessing_dir": derivatives_root / "ffrprep-preprocessing",
            "preprocessing_subject_dir": preproc_subject_dir,
            "analysis_dir": derivatives_root / "ffrprep-analysis",
            "analysis_subject_dir": (
                derivatives_root / "ffrprep-analysis" / "sub-01"
            ),
        }

        with patch("builtins.print") as mock_print:
            run_ffrprep()

        mock_print.assert_any_call(
            "ERROR: No preprocessing outputs found for subject 01."
        )


@patch.dict(os.environ, {"IS_DOCKER": "1"})
@patch("ffrprep.ffrprep_cli.get_parser")
@patch("ffrprep.ffrprep_cli.validate_input_dir")
@patch("ffrprep.ffrprep_cli.get_participants")
@patch("ffrprep.ffrprep_cli.Path")
def test_run_ffrprep_docker_environment(
    mock_path_class, mock_get_participants, mock_validate, mock_parser,
):
    """Docker execution environment detection."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_args = MagicMock()
        mock_args.bids_dir = Path(tmp_dir) / "bids"
        mock_args.output_dir = Path(tmp_dir) / "output"
        mock_args.analysis_level = "participant"
        mock_args.skip_bids_validation = False
        mock_args.participant_label = None

        mock_parser.return_value.parse_args.return_value = mock_args
        mock_get_participants.return_value = []

        mock_cgroup_path = MagicMock()
        mock_cgroup_path.exists.return_value = True
        mock_cgroup_path.read_text.return_value = "docker"
        mock_path_class.return_value = mock_cgroup_path

        run_ffrprep()

        mock_validate.assert_called_once_with(
            "docker", Path(tmp_dir) / "bids", None,
        )


# ---------------------------------------------------------------------------
# CLI integration smoke
# ---------------------------------------------------------------------------

def test_cli_import():
    """Verify the CLI module exposes its public surface."""
    from ffrprep import ffrprep_cli
    assert hasattr(ffrprep_cli, "get_parser")
    assert hasattr(ffrprep_cli, "run_ffrprep")
    assert hasattr(ffrprep_cli, "parse_baseline")
    assert hasattr(ffrprep_cli, "parse_ref_channels")


def test_parser_comprehensive():
    """Parser handles a comprehensive set of arguments."""
    parser = get_parser()

    args = parser.parse_args([
        "/path/to/bids", "/path/to/output", "participant",
    ])
    assert args.bids_dir == Path("/path/to/bids")
    assert args.output_dir == Path("/path/to/output")
    assert args.analysis_level == "participant"

    args = parser.parse_args([
        "/path/to/bids", "/path/to/output", "participant",
        "--participant_label", "01", "02",
        "--stage", "preprocessing",
        "--ref_channels", "TP9,TP10",
        "--no-split-by-trial-type",
        "--skip_bids_validation",
        "--n_procs", "8",
    ])
    assert args.participant_label == ["01", "02"]
    assert args.stage == "preprocessing"
    # ref_channels uses nargs="+"; comma-separated tokens are kept as a
    # single list element here (parse_ref_channels later splits them).
    assert args.ref_channels == ["TP9,TP10"]
    assert args.split_by_trial_type is False
    assert args.skip_bids_validation is True
    assert args.n_procs == 8


# ---------------------------------------------------------------------------
# _resolve_condition_labels: events.tsv trial_type lookup for evoked titles
# ---------------------------------------------------------------------------

def _write_events_tsv(path, rows):
    """Write a minimal events.tsv with the given rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for row in rows for k in row.keys()})
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(c, "n/a")) for c in cols) + "\n")


def test_resolve_condition_labels_single_id(tmp_path):
    """A single numeric event id resolves to its trial_type."""
    events_fpath = tmp_path / "events.tsv"
    _write_events_tsv(events_fpath, [
        {"onset": 0.5, "duration": 0.17, "value": 1, "trial_type": "deviant"},
        {"onset": 1.0, "duration": 0.17, "value": 1, "trial_type": "deviant"},
        {"onset": 1.5, "duration": 0.17, "value": 2, "trial_type": "standard"},
    ])
    assert _resolve_condition_labels("1", events_fpath) == "deviant"
    assert _resolve_condition_labels("2", events_fpath) == "standard"


def test_resolve_condition_labels_comma_separated(tmp_path):
    """A comma-separated list resolves each id independently."""
    events_fpath = tmp_path / "events.tsv"
    _write_events_tsv(events_fpath, [
        {"onset": 0.5, "duration": 0.17, "value": 1, "trial_type": "deviant"},
        {"onset": 1.5, "duration": 0.17, "value": 2, "trial_type": "standard"},
    ])
    out = _resolve_condition_labels("1, 2", events_fpath)
    # Order preserved as in input; both translated.
    assert out == "deviant, standard"


def test_resolve_condition_labels_unknown_id_passes_through(tmp_path):
    """Ids without a row in events.tsv come through unchanged."""
    events_fpath = tmp_path / "events.tsv"
    _write_events_tsv(events_fpath, [
        {"onset": 0.5, "duration": 0.17, "value": 1, "trial_type": "deviant"},
    ])
    assert _resolve_condition_labels("99", events_fpath) == "99"


def test_resolve_condition_labels_missing_events_file(tmp_path):
    """When events.tsv is absent, return the original event-id string."""
    missing = tmp_path / "no-such-events.tsv"
    assert _resolve_condition_labels("1", missing) == "1"


def test_resolve_condition_labels_missing_trial_type_column(tmp_path):
    """When events.tsv lacks a trial_type column, return the input."""
    events_fpath = tmp_path / "events.tsv"
    _write_events_tsv(events_fpath, [
        {"onset": 0.5, "duration": 0.17, "value": 1},
    ])
    assert _resolve_condition_labels("1", events_fpath) == "1"


def test_resolve_condition_labels_empty_input(tmp_path):
    """Empty / falsy input returns unchanged (caller's None / '' default)."""
    events_fpath = tmp_path / "events.tsv"
    _write_events_tsv(events_fpath, [
        {"onset": 0.5, "duration": 0.17, "value": 1, "trial_type": "deviant"},
    ])
    assert _resolve_condition_labels("", events_fpath) == ""
    assert _resolve_condition_labels(None, events_fpath) is None


# ---------------------------------------------------------------------------
# _restore_epochs_baseline: re-apply baseline from preproc sidecar after load
# ---------------------------------------------------------------------------

def _make_synthetic_epochs(baseline=None):
    """Build a tiny in-memory mne.Epochs object for baseline tests."""
    import numpy as np
    import mne
    sfreq = 1000.0
    n_channels = 2
    n_times = int(sfreq * 5)
    rng = np.random.default_rng(7)
    data = rng.normal(0, 1e-6, size=(n_channels, n_times))
    info = mne.create_info(["Cz", "F3"], sfreq=sfreq, ch_types=["eeg"] * 2)
    raw = mne.io.RawArray(data, info, verbose=False)
    n_events = 3
    event_samples = np.linspace(500, n_times - 500, n_events, dtype=int)
    events = np.column_stack([
        event_samples,
        np.zeros(n_events, dtype=int),
        np.ones(n_events, dtype=int),
    ])
    return mne.Epochs(
        raw, events, tmin=-0.04, tmax=0.4, baseline=baseline,
        preload=True, verbose=False,
    )


def _write_sidecar(path, payload):
    """Write a JSON sidecar with the given payload."""
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_restore_epochs_baseline_reapplies_window_from_sidecar(tmp_path):
    """When the sidecar has a Baseline window and epochs.baseline is None,
    the helper applies it back."""
    epochs = _make_synthetic_epochs(baseline=None)
    assert epochs.baseline is None

    sidecar = tmp_path / "sidecar.json"
    _write_sidecar(sidecar, {"Baseline": [-0.04, 0.0]})

    restored = _restore_epochs_baseline(epochs, sidecar)
    assert restored.baseline == (-0.04, 0.0)


def test_restore_epochs_baseline_no_op_when_already_set(tmp_path):
    """When epochs.baseline is already set, the helper leaves it alone."""
    epochs = _make_synthetic_epochs(baseline=(-0.04, 0.0))
    assert epochs.baseline == (-0.04, 0.0)

    sidecar = tmp_path / "sidecar.json"
    _write_sidecar(sidecar, {"Baseline": [-0.1, 0.0]})  # different window

    restored = _restore_epochs_baseline(epochs, sidecar)
    # Helper must not overwrite an already-set baseline.
    assert restored.baseline == (-0.04, 0.0)


def test_restore_epochs_baseline_no_op_when_sidecar_missing(tmp_path):
    """No sidecar file → epochs come back unchanged (still baseline=None)."""
    epochs = _make_synthetic_epochs(baseline=None)
    missing = tmp_path / "no-such.json"
    restored = _restore_epochs_baseline(epochs, missing)
    assert restored.baseline is None


def test_restore_epochs_baseline_no_op_when_field_absent(tmp_path):
    """Sidecar present but no Baseline key → leave epochs unchanged."""
    epochs = _make_synthetic_epochs(baseline=None)
    sidecar = tmp_path / "sidecar.json"
    _write_sidecar(sidecar, {"EpochCount": 3})  # no Baseline key
    restored = _restore_epochs_baseline(epochs, sidecar)
    assert restored.baseline is None


# ---------------------------------------------------------------------------
# _find_raw_paths_for_section: locate raw EEG files for a (task, runs) combo
# ---------------------------------------------------------------------------

from ffrprep.ffrprep_cli import _find_raw_paths_for_section  # noqa: E402


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_find_raw_paths_for_section_single_run(tmp_path):
    """A single-run identifier returns one path when the file exists."""
    eeg_dir = tmp_path / "sub-03" / "eeg"
    f = eeg_dir / "sub-03_task-active_run-1_eeg.bdf"
    _touch(f)
    paths = _find_raw_paths_for_section(eeg_dir, "03", "active", "1")
    assert paths == [f]


def test_find_raw_paths_for_section_concat_runs_list(tmp_path):
    """A list of run ids returns the matching paths in order."""
    eeg_dir = tmp_path / "sub-03" / "eeg"
    f1 = eeg_dir / "sub-03_task-active_run-1_eeg.bdf"
    f2 = eeg_dir / "sub-03_task-active_run-2_eeg.bdf"
    f3 = eeg_dir / "sub-03_task-active_run-3_eeg.bdf"
    for f in (f1, f2, f3):
        _touch(f)
    paths = _find_raw_paths_for_section(eeg_dir, "03", "active", ["1", "2", "3"])
    assert paths == [f1, f2, f3]


def test_find_raw_paths_for_section_skips_missing(tmp_path):
    """Missing files are silently skipped; the caller decides what to do."""
    eeg_dir = tmp_path / "sub-03" / "eeg"
    f1 = eeg_dir / "sub-03_task-active_run-1_eeg.bdf"
    f3 = eeg_dir / "sub-03_task-active_run-3_eeg.bdf"
    _touch(f1)
    _touch(f3)
    # run-2 missing
    paths = _find_raw_paths_for_section(eeg_dir, "03", "active", ["1", "2", "3"])
    assert paths == [f1, f3]


def test_find_raw_paths_for_section_extension_fallback(tmp_path):
    """When .bdf isn't present, .edf / .fif are tried in order."""
    eeg_dir = tmp_path / "sub-03" / "eeg"
    edf = eeg_dir / "sub-03_task-active_run-1_eeg.edf"
    _touch(edf)
    paths = _find_raw_paths_for_section(eeg_dir, "03", "active", "1")
    assert paths == [edf]


def test_find_raw_paths_for_section_none_when_run_is_none(tmp_path):
    """When the run identifier is None and no list is given, returns []."""
    eeg_dir = tmp_path / "sub-03" / "eeg"
    paths = _find_raw_paths_for_section(eeg_dir, "03", "active", None)
    assert paths == []


# ---------------------------------------------------------------------------
# parse_difference_pairs
# ---------------------------------------------------------------------------

def test_parse_difference_pairs_returns_list_of_tuples():
    """``A:B,C:D`` → ``[("A","B"), ("C","D")]``."""
    from ffrprep.ffrprep_cli import parse_difference_pairs

    pairs = parse_difference_pairs(["Pos:Neg", "Tone1:Tone2"])
    assert pairs == [("Pos", "Neg"), ("Tone1", "Tone2")]


def test_parse_difference_pairs_none_passthrough():
    """A None argument round-trips as None (no diff requested)."""
    from ffrprep.ffrprep_cli import parse_difference_pairs

    assert parse_difference_pairs(None) is None


def test_parse_difference_pairs_empty_list_returns_none():
    """An empty list normalizes to None."""
    from ffrprep.ffrprep_cli import parse_difference_pairs

    assert parse_difference_pairs([]) is None


def test_parse_difference_pairs_strips_whitespace():
    """Whitespace around tokens is tolerated."""
    from ffrprep.ffrprep_cli import parse_difference_pairs

    assert parse_difference_pairs([" Pos : Neg "]) == [("Pos", "Neg")]


def test_parse_difference_pairs_rejects_missing_colon():
    """A token without ``:`` is malformed and raises."""
    from ffrprep.ffrprep_cli import parse_difference_pairs

    with pytest.raises(argparse.ArgumentTypeError):
        parse_difference_pairs(["PosNeg"])


def test_parse_difference_pairs_rejects_self_pair():
    """``A:A`` is a degenerate diff and must be rejected."""
    from ffrprep.ffrprep_cli import parse_difference_pairs

    with pytest.raises(argparse.ArgumentTypeError):
        parse_difference_pairs(["Pos:Pos"])


def test_parse_difference_pairs_rejects_empty_side():
    """``A:`` or ``:B`` is malformed and raises."""
    from ffrprep.ffrprep_cli import parse_difference_pairs

    with pytest.raises(argparse.ArgumentTypeError):
        parse_difference_pairs(["Pos:"])
    with pytest.raises(argparse.ArgumentTypeError):
        parse_difference_pairs([":Neg"])
