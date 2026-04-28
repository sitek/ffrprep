"""System tests: invoke the ffrprep BIDS-App on real downloaded data.

These tests are slow (minutes per subject/run) and are gated behind the
``system`` marker so they do not run by default. They depend on two
fixtures from ``conftest.py``:

* ``bids_dataset`` — session-scoped, returns a read-only BIDS root that
  has subjects ``03`` and ``21`` available (downloaded once per session).
* ``bids_workspace`` — function-scoped, returns a writable BIDS root
  (subjects symlinked, metadata copied) into which derivatives may be
  written without polluting the cached dataset.

Run inside the container::

    docker run --rm \\
      -v /path/to/data:/cache \\
      -e FFRPREP_TEST_CACHE=/cache \\
      ffrprep:local test -m system -v

Each test invokes the CLI as a subprocess (the entrypoint has already
activated the .venv, so ``ffrprep`` is on PATH) to keep nipype workflow
state from leaking between tests.
"""
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.system


# FFR-specific defaults mirroring the docker_test.sh template.
_FFR_PREPROC_ARGS = [
    "--stage", "preprocessing",
    "--ref_channels", "M1", "M2",
    "--l_freq", "65",
    "--h_freq", "2000",
    "--picks", "Cz",
    "--tmin", "-0.04",
    "--tmax", "0.4",
    "--baseline", "-0.04", "0",
    "--reject-eeg", "0.000075",
    "--on-missing", "warn",
    "--n_procs", "1",
    "--skip_bids_validation",
]


def _run_ffrprep(bids_root, output_dir, work_dir, *extra_args):
    """Invoke ``ffrprep`` as a subprocess and return CompletedProcess.

    The dispatcher entrypoint has already activated /home/ffrprep/.venv,
    so ``ffrprep`` is on PATH inside the container without ``uv run``.
    """
    cmd = [
        "ffrprep",
        str(bids_root), str(output_dir), "participant",
        "--work_dir", str(work_dir),
        *_FFR_PREPROC_ARGS,
        *extra_args,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def _derivatives_root(workspace):
    return Path(workspace) / "derivatives" / "ffrprep-preprocessing"


# ---------------------------------------------------------------------------
# Single subject — single run, single task
# ---------------------------------------------------------------------------

def test_single_subject_single_run(bids_workspace, tmp_path):
    out_dir = tmp_path / "out"
    work_dir = tmp_path / "work"
    _run_ffrprep(
        bids_workspace, out_dir, work_dir,
        "--participant_label", "03",
        "--task", "active",
        "--run", "1",
    )
    sub_deriv = _derivatives_root(bids_workspace) / "sub-03"
    assert sub_deriv.is_dir()
    assert any(sub_deriv.rglob("*.fif"))


# ---------------------------------------------------------------------------
# Single subject — multiple runs of the same task
# ---------------------------------------------------------------------------

def test_single_subject_multiple_runs(bids_workspace, tmp_path):
    out_dir = tmp_path / "out"
    work_dir = tmp_path / "work"
    _run_ffrprep(
        bids_workspace, out_dir, work_dir,
        "--participant_label", "03",
        "--task", "active",
        "--run", "1", "2",
    )
    sub_deriv = _derivatives_root(bids_workspace) / "sub-03"
    fifs = sorted(sub_deriv.rglob("sub-03_task-active_run-*.fif"))
    assert len(fifs) >= 2


# ---------------------------------------------------------------------------
# Single subject — multiple tasks
# ---------------------------------------------------------------------------

def test_single_subject_multiple_tasks(bids_workspace, tmp_path):
    out_dir = tmp_path / "out"
    work_dir = tmp_path / "work"
    _run_ffrprep(
        bids_workspace, out_dir, work_dir,
        "--participant_label", "03",
        "--task", "active", "passive",
        "--run", "1",
    )
    sub_deriv = _derivatives_root(bids_workspace) / "sub-03"
    active_fifs = list(sub_deriv.rglob("sub-03_task-active_run-*.fif"))
    passive_fifs = list(sub_deriv.rglob("sub-03_task-passive_run-*.fif"))
    assert active_fifs, "missing derivatives for task-active"
    assert passive_fifs, "missing derivatives for task-passive"


# ---------------------------------------------------------------------------
# Multiple subjects — one run, one task per subject
# ---------------------------------------------------------------------------

def test_multiple_subjects_single_run(bids_workspace, tmp_path):
    out_dir = tmp_path / "out"
    work_dir = tmp_path / "work"
    _run_ffrprep(
        bids_workspace, out_dir, work_dir,
        "--participant_label", "03", "21",
        "--task", "active",
        "--run", "1",
    )
    base = _derivatives_root(bids_workspace)
    assert (base / "sub-03").is_dir()
    assert (base / "sub-21").is_dir()


# ---------------------------------------------------------------------------
# CLI flag coverage — --concat-runs and --save-each-node
# ---------------------------------------------------------------------------

def test_concat_runs_single_subject(bids_workspace, tmp_path):
    """``--concat-runs`` runs a single workflow per task with runs merged.

    With concatenation, the per-run loop is bypassed and the resulting
    derivative file should not carry a ``run-*`` token.
    """
    out_dir = tmp_path / "out"
    work_dir = tmp_path / "work"
    _run_ffrprep(
        bids_workspace, out_dir, work_dir,
        "--participant_label", "03",
        "--task", "active",
        "--concat-runs",
    )
    sub_deriv = _derivatives_root(bids_workspace) / "sub-03"
    fifs = sorted(sub_deriv.rglob("sub-03_task-active*.fif"))
    assert fifs, "concat-runs produced no derivatives"
    # No per-run output should appear when runs are concatenated
    per_run = [f for f in fifs if "_run-" in f.name]
    assert not per_run, (
        f"--concat-runs should suppress per-run derivative files; got {per_run}"
    )


def test_save_each_node_writes_intermediates(bids_workspace, tmp_path):
    """``--save-each-node`` should leave intermediate node outputs on disk."""
    out_dir = tmp_path / "out"
    work_dir = tmp_path / "work"
    _run_ffrprep(
        bids_workspace, out_dir, work_dir,
        "--participant_label", "03",
        "--task", "active",
        "--run", "1",
        "--save-each-node",
    )
    # With disk-backed mode, multiple .fif files representing intermediate
    # preprocessing steps (referenced, filtered, epoched) should appear in
    # the work dir or under derivatives. We assert the count is materially
    # larger than the single output produced without --save-each-node.
    intermediate = list(work_dir.rglob("*.fif"))
    intermediate += list(_derivatives_root(bids_workspace).rglob("*.fif"))
    assert len(intermediate) >= 3, (
        f"--save-each-node should leave at least three intermediate .fif "
        f"files on disk; found {len(intermediate)}: {intermediate}"
    )


# ---------------------------------------------------------------------------
# Session axis — dataset has no ses-* level
# ---------------------------------------------------------------------------

def test_session_axis_skipped_when_dataset_is_single_session(bids_workspace):
    sessions = sorted((Path(bids_workspace) / "sub-03").glob("ses-*"))
    if not sessions:
        pytest.skip("OSF dataset has no ses-* level — multi-session axis untested")
    assert len(sessions) >= 2
