"""Tests for the parallel-dispatch infrastructure in ffrprep_cli.

These tests cover the dispatcher (`_dispatch`) and the helper functions
that prepare its inputs (`_snapshot_args`, `_snapshot_deriv`,
`_make_*_payload`, `_setup_worker_log`). They do NOT exercise the full
preprocessing / analysis workflows — those are covered by end-to-end
docker runs.

Workers passed to `_dispatch` here are trivial top-level functions (must
be picklable for ProcessPoolExecutor) so we can assert dispatcher
semantics independently of the nipype workflow.
"""

import argparse
import logging
from pathlib import Path

import pytest

from ffrprep.ffrprep_cli import (
    IterationResult,
    _dispatch,
    _make_analysis_payload,
    _make_concat_payload,
    _make_preproc_payload,
    _setup_worker_log,
    _snapshot_args,
    _snapshot_deriv,
)


# ----- Top-level worker functions used as test fixtures.
#       Must be top-level so ProcessPoolExecutor can pickle them.


def _ok_worker(payload):
    """Returns success for any payload."""
    return IterationResult(
        identifier=payload["identifier"],
        duration_s=0.0,
    )


def _fail_task_X(payload):
    """Top-level (picklable) worker that raises on identifier 'task-X'."""
    if payload["identifier"] == "task-X":
        raise RuntimeError("injected failure for task-X")
    return IterationResult(
        identifier=payload["identifier"],
        duration_s=0.0,
    )


# ----- IterationResult dataclass


def test_iteration_result_defaults():
    """IterationResult has sane defaults so partial construction works."""
    r = IterationResult(identifier="task-1")
    assert r.identifier == "task-1"
    assert r.output_files == []
    assert r.log_path == ""
    assert r.duration_s == 0.0


# ----- _snapshot_args / _snapshot_deriv


def test_snapshot_args_stringifies_paths(tmp_path):
    """Path-typed values must be stringified at the boundary so workers
    don't accidentally couple to pathlib.Path types."""
    args = argparse.Namespace(
        bids_dir=tmp_path / "bids",
        n_procs=4,
        save_each_node=False,
        picks="Cz",
        baseline=None,
    )
    snap = _snapshot_args(args)
    assert snap["bids_dir"] == str(tmp_path / "bids")
    assert isinstance(snap["bids_dir"], str)
    assert snap["n_procs"] == 4
    assert snap["picks"] == "Cz"


def test_snapshot_deriv_stringifies_paths(tmp_path):
    deriv = {
        "derivatives_root": tmp_path / "deriv",
        "preprocessing_dir": tmp_path / "deriv" / "ffrprep-preprocessing",
        "preprocessing_subject_dir": tmp_path / "deriv" / "ffrprep-preprocessing" / "sub-01" / "eeg",
        "analysis_dir": tmp_path / "deriv" / "ffrprep-analysis",
        "analysis_subject_dir": tmp_path / "deriv" / "ffrprep-analysis" / "sub-01",
    }
    snap = _snapshot_deriv(deriv)
    for key, value in snap.items():
        assert isinstance(value, str), f"{key} is not a string: {type(value)}"


# ----- _make_*_payload


def test_make_preproc_payload_per_run_layout(tmp_path):
    """Per-(task, run) payload picks task-X/run-Y subdirs in the work_dir."""
    args_snap = {
        "bids_dir": str(tmp_path / "bids"),
        "tmin": -0.04,
        "tmax": 0.4,
        "picks": None,
        "on_missing": "warn",
        "event_id": None,
        "events_file": None,
        "save_each_node": False,
        "work_dir": None,
    }
    deriv_snap = _snapshot_deriv({
        "derivatives_root": tmp_path / "deriv",
        "preprocessing_dir": tmp_path / "deriv" / "ffrprep-preprocessing",
        "preprocessing_subject_dir": tmp_path / "deriv" / "ffrprep-preprocessing" / "sub-01" / "eeg",
        "analysis_dir": tmp_path / "deriv" / "ffrprep-analysis",
        "analysis_subject_dir": tmp_path / "deriv" / "ffrprep-analysis" / "sub-01",
    })
    payload = _make_preproc_payload(
        args_snap, deriv_snap, "01", "active", "1",
        ["M1", "M2"], 65.0, 2000.0, (-0.04, 0.0), {"eeg": 7.5e-5},
    )
    assert payload["kind"] == "per_run"
    assert payload["identifier"] == "task-active_run-1"
    assert payload["task_label"] == "active"
    assert payload["run_label"] == "1"
    assert "task-active" in payload["work_dir"]
    assert "run-1" in payload["work_dir"]
    assert payload["ref_channels"] == ["M1", "M2"]


def test_make_concat_payload_task_level_workdir(tmp_path):
    """Concat payload's work_dir is per-task (not shared across tasks)
    so concurrent workers don't collide on nipype scratch."""
    args_snap = {
        "bids_dir": str(tmp_path / "bids"),
        "tmin": -0.04,
        "tmax": 0.4,
        "picks": None,
        "on_missing": "warn",
        "event_id": None,
        "events_file": None,
        "save_each_node": False,
        "work_dir": None,
    }
    deriv_snap = _snapshot_deriv({
        "derivatives_root": tmp_path / "deriv",
        "preprocessing_dir": tmp_path / "deriv" / "ffrprep-preprocessing",
        "preprocessing_subject_dir": tmp_path / "deriv" / "ffrprep-preprocessing" / "sub-01" / "eeg",
        "analysis_dir": tmp_path / "deriv" / "ffrprep-analysis",
        "analysis_subject_dir": tmp_path / "deriv" / "ffrprep-analysis" / "sub-01",
    })
    payload_active = _make_concat_payload(
        args_snap, deriv_snap, "01", "active", None,
        ["M1", "M2"], 65.0, 2000.0, (-0.04, 0.0), {"eeg": 7.5e-5},
    )
    payload_passive = _make_concat_payload(
        args_snap, deriv_snap, "01", "passive", None,
        ["M1", "M2"], 65.0, 2000.0, (-0.04, 0.0), {"eeg": 7.5e-5},
    )
    assert payload_active["kind"] == "concat"
    assert payload_active["work_dir"] != payload_passive["work_dir"], (
        "concat workers must not share a work_dir across tasks"
    )
    assert "task-active" in payload_active["work_dir"]
    assert "task-passive" in payload_passive["work_dir"]


def test_make_analysis_payload_keys(tmp_path):
    args_snap = {
        "bids_dir": str(tmp_path / "bids"),
        "by_event_type": False,
        "work_dir": None,
    }
    deriv_snap = _snapshot_deriv({
        "derivatives_root": tmp_path / "deriv",
        "preprocessing_dir": tmp_path / "deriv" / "ffrprep-preprocessing",
        "preprocessing_subject_dir": tmp_path / "deriv" / "ffrprep-preprocessing" / "sub-01" / "eeg",
        "analysis_dir": tmp_path / "deriv" / "ffrprep-analysis",
        "analysis_subject_dir": tmp_path / "deriv" / "ffrprep-analysis" / "sub-01",
    })
    preproc_file = tmp_path / "sub-01_task-active_run-1_desc-preproc_epo.fif"
    payload = _make_analysis_payload(args_snap, deriv_snap, "01", preproc_file)
    assert payload["preproc_file"] == str(preproc_file)
    assert "analysis_sub-01_task-active_run-1_desc-preproc_epo" in payload["identifier"]
    assert "sub-01_task-active_run-1_desc-preproc_epo" in payload["work_dir"]


# ----- _setup_worker_log


def test_setup_worker_log_writes_to_named_file(tmp_path):
    """The per-iteration log file is created at <work_dir>/<identifier>.log."""
    log_path = _setup_worker_log(tmp_path / "wf", "task-active_run-1")
    assert Path(log_path).exists()
    assert Path(log_path).name == "task-active_run-1.log"
    logging.getLogger().info("hello from test")
    # Detach the handler so subsequent tests don't accumulate.
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_ffrprep_worker_handler", False):
            root.removeHandler(h)
            h.close()
    contents = Path(log_path).read_text()
    assert "hello from test" in contents
    assert "worker iteration started: task-active_run-1" in contents


# ----- _dispatch


def test_dispatch_n_procs_1_in_process_fast_path():
    """With n_procs=1 the dispatcher runs workers in-process — no
    spawn overhead, no executor; results come back in submission order."""
    payloads = [{"identifier": f"task-{i}"} for i in range(3)]
    results = _dispatch(_ok_worker, payloads, n_procs=1, kind_label="test")
    assert len(results) == 3
    assert [r.identifier for r in results] == ["task-0", "task-1", "task-2"]


def test_dispatch_empty_payloads():
    results = _dispatch(_ok_worker, [], n_procs=4, kind_label="test")
    assert results == []


def test_dispatch_n_procs_2_across_processes():
    """With n_procs >= 2 the dispatcher uses a ProcessPoolExecutor with
    the spawn start method; results may arrive out of submission order
    but every payload completes."""
    payloads = [{"identifier": f"task-{i}"} for i in range(4)]
    results = _dispatch(_ok_worker, payloads, n_procs=2, kind_label="test")
    assert len(results) == 4
    identifiers = sorted(r.identifier for r in results)
    assert identifiers == ["task-0", "task-1", "task-2", "task-3"]


def test_dispatch_propagates_worker_exception_in_process():
    """Fail-fast contract for n_procs=1: a raising worker propagates
    its exception up out of `_dispatch`."""
    payloads = [
        {"identifier": "task-A"},
        {"identifier": "task-X"},  # raises
        {"identifier": "task-B"},  # never reached
    ]
    with pytest.raises(RuntimeError, match="injected failure for task-X"):
        _dispatch(_fail_task_X, payloads, n_procs=1, kind_label="test")


def test_dispatch_propagates_worker_exception_across_processes():
    """Fail-fast contract for n_procs>=2: a worker raising inside the
    pool propagates its exception up out of `_dispatch` via
    `fut.result()`."""
    payloads = [
        {"identifier": "task-A"},
        {"identifier": "task-X"},  # raises
        {"identifier": "task-B"},
    ]
    with pytest.raises(RuntimeError, match="injected failure for task-X"):
        _dispatch(_fail_task_X, payloads, n_procs=2, kind_label="test")
