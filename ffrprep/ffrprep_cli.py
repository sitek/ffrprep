import argparse
import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from ffrprep.utils import validate_input_dir
from ffrprep.preproc import (
    create_preprocessing_workflow,
    create_analysis_workflow,
    get_participants,
    get_sessions_tasks_runs,
    setup_derivatives_directories,
)
import ffrprep.reports as reports
from importlib.metadata import version as _pkg_version
import re


@dataclass
class IterationResult:
    """Result returned by a successful per-(task, run) worker.

    Workers raise on failure; the dispatcher does not catch. So every
    ``IterationResult`` produced represents a successful iteration.
    """

    identifier: str
    output_files: list = field(default_factory=list)
    log_path: str = ""
    duration_s: float = 0.0


def _propagate_run_provenance(preproc_file, analysis_dir):
    """Copy ConcatenatedRuns / Run from a preproc sidecar to its analysis siblings.

    The analysis stage only sees the preprocessing .fif filename; for
    concatenated-run inputs the filename has no ``_run-N`` token so the
    downstream save can't know it represents multiple runs. This helper
    reads the preproc sidecar and writes the same provenance fields into
    every analysis sidecar that came from this preproc file.
    """
    import json

    preproc_sidecar = preproc_file.with_suffix(".json")
    if not preproc_sidecar.exists():
        return
    with open(preproc_sidecar) as f:
        preproc_meta = json.load(f)
    fields = {}
    if "ConcatenatedRuns" in preproc_meta:
        fields["ConcatenatedRuns"] = preproc_meta["ConcatenatedRuns"]
    if "Run" in preproc_meta and "Run" not in fields:
        fields["Run"] = preproc_meta["Run"]
    if not fields:
        return

    # The analysis sidecar(s) for this preproc file share its base
    # stem — everything before the ``_desc-preproc`` segment. Using
    # partition() handles both the bare ``_desc-preproc_epo`` and
    # per-trial-type ``_desc-preproc{Cond}_epo`` filenames.
    base_stem, _, _ = preproc_file.stem.partition("_desc-preproc")
    for analysis_sidecar in analysis_dir.glob(f"{base_stem}_desc-*.json"):
        with open(analysis_sidecar) as f:
            data = json.load(f)
        data.update(fields)
        with open(analysis_sidecar, "w") as f:
            json.dump(data, f, indent=2)


def _setup_subject_log(derivatives_info, subject, stage):
    """Tee stdout-style run information into a per-subject log file.

    Creates ``sub-<id>_<descriptor>.log`` next to the subject's outputs
    for the requested stage. The file captures the CLI invocation,
    ffrprep version, timestamp, and any subsequent log messages emitted
    via the root or nipype loggers. A handler is attached per call so
    each subject ends up with its own log file.

    For "preprocessing" or "both" stages, the log lives in the
    preprocessing subject dir; for "analysis", in the analysis subject
    dir. The descriptor in the filename matches.
    """
    if stage in ("preprocessing", "both"):
        log_dir = derivatives_info.get("preprocessing_subject_dir")
        descriptor = "preprocessing"
    else:
        log_dir = derivatives_info.get("analysis_subject_dir")
        descriptor = "analysis"
    if log_dir is None:
        return
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"sub-{subject}_{descriptor}.log"

    # Clean up the legacy log filename (pre-descriptor naming) if present.
    legacy_log_path = log_dir / f"sub-{subject}_ffrprep.log"
    if legacy_log_path.exists() and legacy_log_path != log_path:
        legacy_log_path.unlink()

    # Detach any per-subject handler left over from a previous subject in
    # the same process (otherwise its log file keeps capturing this
    # subject's messages too — log bleed between subjects).
    root_logger = logging.getLogger()
    for existing in list(root_logger.handlers):
        if getattr(existing, "_ffrprep_subject_handler", False):
            root_logger.removeHandler(existing)
            existing.close()

    handler = logging.FileHandler(log_path, mode="w")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    )
    # Tag so the next call can find and remove it.
    handler._ffrprep_subject_handler = True

    # Attach the handler only to the root logger. nipype's logger
    # propagates to root by default, so the messages get captured exactly
    # once. (Adding the handler to both root and nipype causes duplicate
    # log lines.)
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)

    root_logger.info("ffrprep version: %s", _pkg_version("ffrprep"))
    root_logger.info("invocation: %s", " ".join(sys.argv))
    root_logger.info("started at: %s", time.strftime("%Y-%m-%d %H:%M:%S"))
    root_logger.info("derivatives root: %s", derivatives_info.get("derivatives_root"))
    print(f"Run log: {log_path}")


def _setup_worker_log(work_dir, identifier):
    """Attach a per-iteration FileHandler to the worker's root logger.

    Workers spawned by ProcessPoolExecutor inherit the parent's log
    handlers under ``fork`` start method (under ``spawn`` they don't,
    but we defensively detach any inherited per-subject handler anyway
    so worker writes never reach the parent's subject log via a stale
    file handle).
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / f"{identifier}.log"

    root_logger = logging.getLogger()
    for existing in list(root_logger.handlers):
        if getattr(existing, "_ffrprep_subject_handler", False):
            root_logger.removeHandler(existing)
            existing.close()

    handler = logging.FileHandler(log_path, mode="w")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    )
    handler._ffrprep_worker_handler = True
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    root_logger.info("worker iteration started: %s", identifier)
    return log_path


def _snapshot_args(args):
    """Pickleable plain-dict snapshot of an argparse Namespace.

    Path-typed values are stringified at the boundary so workers don't
    accidentally couple to the parser. The snapshot is the worker
    contract; anything not in here is unavailable inside the worker.
    """
    out = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def _snapshot_deriv(derivatives_info):
    """Pickleable plain-dict snapshot of `setup_derivatives_directories`."""
    return {k: (str(v) if isinstance(v, Path) else v)
            for k, v in derivatives_info.items()}


def _make_preproc_payload(args_snap, deriv_snap, subject, task_label, run_label,
                          ref_channels, effective_l, effective_h, baseline,
                          reject_value):
    """Assemble the input dict for one per-(task, run) preprocessing worker."""
    if args_snap.get("work_dir"):
        work_dir = (
            Path(args_snap["work_dir"])
            / f"sub-{subject}"
            / "preprocessing"
            / (f"task-{task_label}" if task_label is not None else "task-None")
            / (f"run-{run_label}" if run_label is not None else "single")
        )
    else:
        work_dir = (
            Path(deriv_snap["preprocessing_dir"])
            / "work"
            / f"sub-{subject}"
            / (f"task-{task_label}" if task_label is not None else "task-None")
            / (f"run-{run_label}" if run_label is not None else "single")
        )
    identifier = (
        f"task-{task_label or 'None'}_run-{run_label if run_label is not None else 'single'}"
    )
    return {
        "kind": "per_run",
        "identifier": identifier,
        "subject": subject,
        "task_label": task_label,
        "run_label": run_label,
        "work_dir": str(work_dir),
        "bids_root": args_snap["bids_dir"],
        "ref_channels": ref_channels,
        "high_pass": effective_l,
        "low_pass": effective_h,
        "baseline": baseline,
        "tmin": args_snap.get("tmin"),
        "tmax": args_snap.get("tmax"),
        "reject": reject_value,
        "picks": parse_picks(args_snap.get("picks")),
        "on_missing": args_snap.get("on_missing", "warn"),
        "event_id": parse_event_id(args_snap.get("event_id")),
        "trial_types": args_snap.get("trial_types"),
        "split_by_trial_type": bool(
            args_snap.get("split_by_trial_type", True)
        ),
        "events_file": args_snap.get("events_file"),
        "derivatives_root": str(deriv_snap.get("derivatives_root", "")),
        "output_dir": str(deriv_snap["preprocessing_subject_dir"]),
        "save_each_node": bool(args_snap.get("save_each_node")),
    }


def _make_concat_payload(args_snap, deriv_snap, subject, task_label, runs,
                         ref_channels, effective_l, effective_h, baseline,
                         reject_value):
    """Assemble the input dict for one per-task (concat-runs) worker.

    `runs` is the explicit list of runs the user passed via ``--run``,
    or ``None`` to let the loader concatenate every run available for
    the task. The work_dir gets a per-task suffix so concurrent
    per-task workers don't share nipype scratch.
    """
    if args_snap.get("work_dir"):
        work_dir = (
            Path(args_snap["work_dir"])
            / f"sub-{subject}"
            / "preprocessing"
            / (f"task-{task_label}" if task_label is not None else "task-None")
        )
    else:
        work_dir = (
            Path(deriv_snap["preprocessing_dir"])
            / "work"
            / f"sub-{subject}"
            / (f"task-{task_label}" if task_label is not None else "task-None")
        )
    identifier = f"task-{task_label or 'None'}_concat"
    return {
        "kind": "concat",
        "identifier": identifier,
        "subject": subject,
        "task_label": task_label,
        "run_label": runs,  # list when user supplied --run; else None
        "work_dir": str(work_dir),
        "bids_root": args_snap["bids_dir"],
        "ref_channels": ref_channels,
        "high_pass": effective_l,
        "low_pass": effective_h,
        "baseline": baseline,
        "tmin": args_snap.get("tmin"),
        "tmax": args_snap.get("tmax"),
        "reject": reject_value,
        "picks": parse_picks(args_snap.get("picks")),
        "on_missing": args_snap.get("on_missing", "warn"),
        "event_id": parse_event_id(args_snap.get("event_id")),
        "trial_types": args_snap.get("trial_types"),
        "split_by_trial_type": bool(
            args_snap.get("split_by_trial_type", True)
        ),
        "events_file": args_snap.get("events_file"),
        "derivatives_root": str(deriv_snap.get("derivatives_root", "")),
        "output_dir": str(deriv_snap["preprocessing_subject_dir"]),
        "save_each_node": bool(args_snap.get("save_each_node")),
    }


def _collect_analysis_groups(eeg_dir):
    """Group preprocessing-output files by (subject, task, session, run).

    Globs ``*_desc-preproc*_epo.fif`` so both the bare combined output
    and the per-trial-type split outputs are picked up. The grouping
    key is the BIDS basename up to (but not including) the
    ``_desc-preproc`` segment, so all per-condition slices for the
    same (task, run) tuple end up in the same group.

    Parameters
    ----------
    eeg_dir : pathlib.Path
        Per-subject ``eeg/`` directory under the preprocessing
        derivatives root.

    Returns
    -------
    groups : list[dict]
        One entry per (subject, task, session, run) tuple. Each entry
        carries ``identifier`` (the shared BIDS basename) and
        ``preproc_files`` (a sorted list of Paths). Empty list when no
        preprocessing outputs are present.
    """
    eeg_dir = Path(eeg_dir)
    files = sorted(eeg_dir.glob("*_desc-preproc*_epo.fif"))
    groups_map = {}
    for fpath in files:
        identifier, _, _ = fpath.name.partition("_desc-preproc")
        if identifier not in groups_map:
            groups_map[identifier] = {
                "identifier": identifier,
                "preproc_files": [],
            }
        groups_map[identifier]["preproc_files"].append(fpath)
    return list(groups_map.values())


def _condition_from_preproc_filename(name):
    """Extract the trial-type token from a per-condition preproc filename.

    ``sub-01_task-active_run-1_desc-preprocPos_epo.fif`` → ``"Pos"``.
    ``sub-01_task-active_run-1_desc-preproc_epo.fif`` → ``""`` (combined,
    no per-type suffix).
    """
    _, _, after = name.partition("_desc-preproc")
    # ``after`` is something like ``"Pos_epo.fif"`` or ``"_epo.fif"``.
    after = after.split("_epo.fif", 1)[0]
    return after  # empty string for the combined / un-split case


def _collect_evoked_groups(analysis_dir):
    """Group analysis-output files by (subject, task, session, run).

    Globs ``*_desc-evoked*.fif`` so per-trial-type, combined, and
    difference outputs are all picked up. Files are grouped by the
    BIDS basename up to ``_desc-evoked``, so the per-type, combined,
    and diff slices for one (task, run) tuple end up in the same
    group. Each group also exposes the parsed ``task`` and ``run``
    tokens for downstream report-section labelling.

    Parameters
    ----------
    analysis_dir : pathlib.Path
        Per-subject directory under the analysis-derivatives root.

    Returns
    -------
    groups : list[dict]
        One entry per (subject, task, session, run) tuple. Keys:
        ``identifier``, ``task``, ``run``, ``evoked_files`` (sorted
        list of Paths). Empty list when no matching files exist.
    """
    analysis_dir = Path(analysis_dir)
    files = sorted(analysis_dir.glob("*_desc-evoked*.fif"))
    groups_map = {}
    for fpath in files:
        identifier, _, _ = fpath.name.partition("_desc-evoked")
        if identifier not in groups_map:
            m_task = re.search(r"task-([^_]+)", identifier)
            m_run = re.search(r"run-([^_]+)", identifier)
            groups_map[identifier] = {
                "identifier": identifier,
                "task": m_task.group(1) if m_task else None,
                "run": m_run.group(1) if m_run else None,
                "evoked_files": [],
            }
        groups_map[identifier]["evoked_files"].append(fpath)
    return list(groups_map.values())


def _make_analysis_payload(args_snap, deriv_snap, subject, group):
    """Assemble the input dict for one per-(task, run) analysis worker.

    ``group`` is a dict from :func:`_collect_analysis_groups`, carrying
    the shared identifier and a list of one or more per-condition
    preprocessing-output files for the (task, run) tuple.
    """
    identifier = group["identifier"]
    preproc_files = [str(p) for p in group["preproc_files"]]
    if args_snap.get("work_dir"):
        work_dir = (
            Path(args_snap["work_dir"])
            / f"sub-{subject}"
            / "analysis"
            / identifier
        )
    else:
        work_dir = (
            Path(deriv_snap["analysis_dir"])
            / "work"
            / f"sub-{subject}"
            / identifier
        )
    return {
        "identifier": f"analysis_{identifier}",
        "subject": subject,
        "preproc_files": preproc_files,
        "original_filename": identifier,
        "work_dir": str(work_dir),
        "bids_root": args_snap["bids_dir"],
        "by_event_type": bool(args_snap.get("split_by_trial_type", True)),
        "difference_pairs": parse_difference_pairs(
            args_snap.get("difference_pairs"),
        ),
        "analysis_subject_dir": str(deriv_snap["analysis_subject_dir"]),
        "derivatives_root": str(deriv_snap.get("derivatives_root", "")),
    }


def _build_preproc_workflow(payload):
    """Build a fresh preprocessing workflow with inputs set from `payload`.

    Handles both the per-(task, run) and concat-runs payloads — the
    only difference between them is the value of ``run_label`` (single
    value, list, or None) which is wired through unchanged.
    """
    wf = create_preprocessing_workflow(disk_backed=payload["save_each_node"])
    wf.base_dir = payload["work_dir"]

    in_node = wf.get_node("inputnode")
    target = in_node.inputs if in_node is not None else wf.inputs.inputnode

    target.bids_root = payload["bids_root"]
    target.sub_label = payload["subject"]
    target.run_label = payload["run_label"]
    target.task_label = payload["task_label"]
    target.ref_channels = payload["ref_channels"]
    target.high_pass = payload["high_pass"]
    target.low_pass = payload["low_pass"]
    target.baseline = payload["baseline"]
    target.tmin = payload["tmin"]
    target.tmax = payload["tmax"]
    target.reject = payload["reject"]
    target.picks = payload["picks"]
    target.on_missing = payload["on_missing"]
    target.event_id = payload["event_id"]
    target.trial_types = payload.get("trial_types")
    target.split_by_trial_type = payload.get("split_by_trial_type", True)
    if payload.get("events_file") is not None:
        target.events_file = payload["events_file"]
    target.derivatives_root = payload["derivatives_root"]
    target.output_dir = payload["output_dir"]
    return wf


def _find_raw_paths_for_section(eeg_dir, subject, task, runs):
    """Locate raw EEG files for a (task, runs) combination.

    `runs` is either a single run identifier (string), a list/tuple
    of run identifiers (concat-runs case, in which case the caller is
    expected to concatenate the loaded raws), or ``None``. The first
    matching extension among .bdf, .edf, .fif is used per run; missing
    runs are silently skipped (the caller decides how to handle a
    partial result). Returns a list of :class:`pathlib.Path`.
    """
    if runs is None:
        return []
    if not isinstance(runs, (list, tuple)):
        runs = [runs]
    paths = []
    for r in runs:
        if r is None:
            continue
        for ext in (".bdf", ".edf", ".fif"):
            candidate = eeg_dir / f"sub-{subject}_task-{task}_run-{r}_eeg{ext}"
            if candidate.exists():
                paths.append(candidate)
                break
    return paths


def _read_raw_any(fpath):
    """Read an MNE-supported raw EEG file by extension."""
    import mne

    suffix = fpath.suffix.lower()
    if suffix == ".bdf":
        return mne.io.read_raw_bdf(str(fpath), preload=True, verbose=False)
    if suffix == ".edf":
        return mne.io.read_raw_edf(str(fpath), preload=True, verbose=False)
    if suffix == ".fif":
        return mne.io.read_raw_fif(str(fpath), preload=True, verbose=False)
    raise ValueError(f"Unsupported raw EEG file extension: {suffix}")


def _rejection_summary(epo_fpath, events_fpath, n_accepted):
    """Pre-rejection / accepted / rejected counts for an Epochs section.

    Prefers the authoritative pre-rejection counts written to the BIDS
    sidecar at preprocessing time (``EpochCountTotal`` /
    ``EpochCountRejected`` / ``RejectionThresholds``). Falls back to
    back-calculating from ``events.tsv`` row count vs ``EpochCount``
    when the sidecar predates the metadata change. Returns an ordered
    dict suitable for ``extra_summary=`` on
    :func:`reports.build_epoch_section`.
    """
    import json

    import pandas as pd

    sidecar = epo_fpath.with_suffix(".json")
    meta = {}
    if sidecar.exists():
        meta = json.loads(sidecar.read_text())
        n_accepted = meta.get("EpochCount", n_accepted)

    # Preferred path: sidecar carries the counts directly.
    if "EpochCountTotal" in meta and "EpochCountRejected" in meta:
        n_total = int(meta["EpochCountTotal"])
        n_rejected = int(meta["EpochCountRejected"])
        rate = (100.0 * n_rejected / n_total) if n_total else 0.0
        row = {
            "Epochs (total / accepted / rejected)":
                f"{n_total} / {n_accepted} / {n_rejected}",
            "Rejection rate": f"{rate:.1f} %",
        }
        thresholds = meta.get("RejectionThresholds")
        if thresholds:
            row["Rejection thresholds"] = ", ".join(
                f"{k}: {v:.2e}" for k, v in thresholds.items()
            )
        return row

    # Fallback: derive total from events.tsv row count.
    if not events_fpath.exists():
        return {"Epochs accepted": str(n_accepted)}

    n_total = len(pd.read_csv(events_fpath, sep="\t"))
    if n_total < n_accepted:
        return {"Epochs accepted": str(n_accepted)}

    n_rejected = n_total - n_accepted
    rate = (100.0 * n_rejected / n_total) if n_total else 0.0
    return {
        "Epochs (total / accepted / rejected)":
            f"{n_total} / {n_accepted} / {n_rejected}",
        "Rejection rate": f"{rate:.1f} %",
    }


def _resolve_condition_labels(event_ids, events_fpath):
    """Translate evoked.comment-style event-id strings to trial_type names.

    `event_ids` is the value of ``evoked.comment`` — typically a single
    numeric event id like ``"1"`` (default-mode averaging produces a
    single id per Evoked when only one event_id is present), or a
    comma-separated list like ``"1, 2"`` when multiple ids were
    averaged together. When the BIDS ``events.tsv`` carries both a
    ``value`` and a ``trial_type`` column, this helper joins on
    ``value`` and substitutes the corresponding ``trial_type`` name
    (so ``"1"`` becomes ``"deviant"`` etc.). Falls back to the
    original string when the events file is missing, lacks a
    ``trial_type`` column, or has no row matching the requested id.
    """
    if not event_ids:
        return event_ids
    if not events_fpath.exists():
        return event_ids

    import pandas as pd

    events = pd.read_csv(events_fpath, sep="\t")
    if "value" not in events.columns or "trial_type" not in events.columns:
        return event_ids

    # value→trial_type mapping (first occurrence wins on duplicates).
    mapping = {}
    for value, trial_type in zip(
        events["value"].astype(str), events["trial_type"].astype(str)
    ):
        if value not in mapping:
            mapping[value] = trial_type

    parts = [p.strip() for p in str(event_ids).split(",")]
    translated = [mapping.get(p, p) for p in parts]
    return ", ".join(translated)


def _restore_epochs_baseline(epochs, sidecar_path):
    """Re-apply ``epochs.baseline`` from the BIDS sidecar's Baseline field.

    ``mne.EpochsArray`` (used inside ``save_preprocessing_outputs``)
    doesn't take a ``baseline`` constructor argument, so the saved
    .fif's baseline metadata round-trips as ``None``. The
    preprocessing sidecar carries a ``Baseline`` field for this
    reason; restoring it here puts ``epochs.baseline`` back so that
    downstream consumers (analysis-side averaging, the report's
    RMS SNR row) keep working.

    ``Epochs.apply_baseline`` is idempotent on already-baselined data:
    it sets the metadata attribute and re-applies the correction
    (which subtracts ~zero from data already centered).
    """
    import json

    if epochs.baseline is not None:
        return epochs
    if not sidecar_path.exists():
        return epochs
    meta = json.loads(sidecar_path.read_text())
    baseline = meta.get("Baseline")
    if baseline is None:
        return epochs
    epochs.apply_baseline(tuple(baseline))
    return epochs


def _build_overview(args, subject, source_files, stage_label):
    """Build the overview dict for the report.

    The summary table lists the BIDS dataset path, subject, stage,
    tasks and runs found in the source files, output count, ffrprep
    version, and a render timestamp. The command field carries the
    full CLI invocation as ``sys.argv``.
    """
    tasks = sorted({
        m.group(1)
        for f in source_files
        for m in [re.search(r"task-([^_]+)", f.name)]
        if m
    })
    runs = sorted({
        m.group(1)
        for f in source_files
        for m in [re.search(r"run-([^_]+)", f.name)]
        if m
    })

    summary = {
        "BIDS dataset": str(args.bids_dir),
        "Subject": f"sub-{subject}",
        "Stage": stage_label,
        "Tasks": ", ".join(tasks) if tasks else "—",
        "Runs": ", ".join(runs) if runs else "—",
        f"{stage_label.capitalize()} outputs": str(len(source_files)),
        "ffrprep version": _pkg_version("ffrprep"),
        "Report rendered": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return {"summary": summary, "command": " ".join(sys.argv)}


def _build_preproc_report(args, derivatives_info, subject):
    """Render the single-file HTML preprocessing report.

    Discovers preprocessing outputs via :func:`_collect_analysis_groups`,
    which groups per-trial-type and combined files by
    (task, session, run). For each group: one Raw section (from the
    original BIDS .bdf/.edf for the task+run, concatenating across
    runs when the source is concat-runs) plus one Epoched section
    per saved file (so split-by-trial-type runs surface a separate
    Epoched section per trial type).
    """
    import json

    import mne

    print("\n" + "=" * 60)
    print("Generating preprocessing report...")
    print("=" * 60)

    bids_root = Path(args.bids_dir)
    preproc_dir = Path(derivatives_info["preprocessing_subject_dir"])
    eeg_dir = bids_root / f"sub-{subject}" / "eeg"

    groups_meta = _collect_analysis_groups(preproc_dir)
    if not groups_meta:
        print(f"No preprocessing outputs found for sub-{subject}; "
              "skipping report.")
        return

    groups = []
    all_files = []
    for grp in groups_meta:
        identifier = grp["identifier"]
        m_task = re.search(r"task-([^_]+)", identifier)
        m_run = re.search(r"run-([^_]+)", identifier)
        task = m_task.group(1) if m_task else None
        run = m_run.group(1) if m_run else None

        # Concat-runs metadata lives in any of the per-condition
        # sidecars in the group (they all share the same source runs).
        first_file = grp["preproc_files"][0]
        first_sidecar = first_file.with_suffix(".json")
        sidecar_meta = (
            json.loads(first_sidecar.read_text())
            if first_sidecar.exists() else {}
        )
        runs_for_section = sidecar_meta.get("ConcatenatedRuns") or [run]

        # Locate the original raw file(s). For concat-runs we load +
        # concatenate so the Raw QA plot reflects what the workflow
        # actually processed.
        raw_paths = _find_raw_paths_for_section(
            eeg_dir, subject, task, runs_for_section,
        )

        first_run = next(
            (r for r in runs_for_section if r is not None), None
        )
        events_fpath = (
            eeg_dir / f"sub-{subject}_task-{task}_run-{first_run}_events.tsv"
            if first_run is not None
            else eeg_dir / f"sub-{subject}_task-{task}_events.tsv"
        )

        is_concat = len(runs_for_section) > 1
        section_run_token = "concat" if is_concat else (run or "single")
        raw_title = (
            f"Raw (concatenated runs: {', '.join(runs_for_section)})"
            if is_concat else "Raw"
        )

        sections = []
        if raw_paths:
            if len(raw_paths) == 1:
                print(f"  loading raw  for sub-{subject} task-{task} "
                      f"{section_run_token}")
                raw = _read_raw_any(raw_paths[0])
            else:
                print(f"  loading {len(raw_paths)} raws for sub-{subject} "
                      f"task-{task} (concat)")
                raws = [_read_raw_any(p) for p in raw_paths]
                raw = mne.concatenate_raws(raws)
            sections.append(reports.build_raw_section(
                raw,
                section_id=f"raw-{task}-{section_run_token}",
                title=raw_title,
                label="Raw",
                events_fpath=str(events_fpath) if events_fpath.exists() else None,
            ))

        # One Epoched section per file in the group — for
        # split-by-trial-type each per-condition file produces its
        # own section labelled with the trial type.
        for epo_fpath in grp["preproc_files"]:
            all_files.append(epo_fpath)
            condition = _condition_from_preproc_filename(epo_fpath.name)
            section_suffix = (
                f"-{condition.lower()}" if condition else ""
            )
            section_title = (
                f"Epoched ({condition})" if condition else "Epoched"
            )
            print(
                f"  loading epoch for sub-{subject} task-{task} "
                f"{section_run_token} ({condition or 'combined'})"
            )
            epochs = mne.read_epochs(
                str(epo_fpath), preload=True, verbose=False,
            )
            extra = _rejection_summary(
                epo_fpath, events_fpath, len(epochs),
            )
            sections.append(reports.build_epoch_section(
                epochs,
                section_id=(
                    f"epoched-{task}-{section_run_token}{section_suffix}"
                ),
                title=section_title,
                extra_summary=extra,
            ))

        groups.append(reports.make_group(task=task, run=run, sections=sections))

    overview = _build_overview(args, subject, all_files, "preprocessing")

    out_path = reports.build_subject_report(
        bids_root=str(bids_root),
        subject=subject,
        out_dir=str(preproc_dir),
        groups=groups,
        overview=overview,
    )
    print(f"\n{'=' * 60}")
    print(f"Preprocessing report written to: {out_path}")
    print(f"{'=' * 60}")


def _build_analysis_report(args, derivatives_info, subject):
    """Render the single-file HTML analysis report.

    Discovers analysis outputs via :func:`_collect_evoked_groups`,
    which groups per-trial-type, combined, and difference files by
    (task, session, run). Builds one section per file (one Evoked
    per file, since save_analysis_outputs writes one .fif per
    condition / combined / pair). Wraps each group via
    :func:`reports.make_group` and renders via
    :func:`reports.build_analysis_report`.
    """
    import json

    import mne

    print("\n" + "=" * 60)
    print("Generating analysis report...")
    print("=" * 60)

    bids_root = Path(args.bids_dir)
    analysis_dir = Path(derivatives_info["analysis_subject_dir"])

    groups_meta = _collect_evoked_groups(analysis_dir)
    if not groups_meta:
        print(f"No analysis outputs found for sub-{subject}; "
              "skipping report.")
        return

    groups = []
    all_files = []
    for grp in groups_meta:
        task = grp["task"]
        run = grp["run"]
        sections = []
        events_fpath = (
            bids_root / f"sub-{subject}" / "eeg"
            / (
                f"sub-{subject}_task-{task}_run-{run}_events.tsv"
                if run is not None
                else f"sub-{subject}_task-{task}_events.tsv"
            )
        )
        for idx, evo_fpath in enumerate(grp["evoked_files"]):
            all_files.append(evo_fpath)
            print(
                f"  loading evoked for sub-{subject} task-{task} "
                f"run-{run} ({evo_fpath.name})"
            )
            evoked_list = mne.read_evokeds(str(evo_fpath), verbose=False)
            # Restore evoked.baseline from the BIDS sidecar — MNE's
            # Evoked.save() doesn't write the baseline window into
            # the .fif, so without this restore RMS SNR (which gates
            # on evoked.baseline is not None) would silently
            # disappear from the report.
            evo_sidecar = evo_fpath.with_suffix(".json")
            if evo_sidecar.exists():
                evo_meta = json.loads(evo_sidecar.read_text())
                baseline = evo_meta.get("Baseline")
                if baseline is not None:
                    for ev in evoked_list:
                        if ev.baseline is None:
                            ev.apply_baseline(tuple(baseline))
            for ev_idx, evoked in enumerate(evoked_list):
                raw_cond = evoked.comment or f"condition-{ev_idx}"
                cond = _resolve_condition_labels(raw_cond, events_fpath)
                sections.append(reports.build_evoked_section(
                    evoked,
                    section_id=f"evoked-{task}-{run}-{idx}-{ev_idx}",
                    title=f"Evoked ({cond})",
                    label="Evoked",
                ))
        groups.append(reports.make_group(task=task, run=run, sections=sections))

    overview = _build_overview(args, subject, all_files, "analysis")

    out_path = reports.build_analysis_report(
        bids_root=str(bids_root),
        subject=subject,
        out_dir=str(analysis_dir),
        groups=groups,
        overview=overview,
    )
    print(f"Analysis report written to: {out_path}")


def _check_preproc_output_or_raise(payload):
    """Confirm at least one preprocessing output landed on disk.

    nipype caches by input hash and reports "Cached, collecting
    precomputed outputs" without checking that the recorded output
    file still exists. If a previous run was cleaned up but the work
    cache wasn't, the workflow silently "succeeds" without writing
    anything. Catch that here with an actionable error.

    The glob matches both the bare ``_desc-preproc_epo.fif`` (under
    ``--no-split-by-trial-type``) and per-trial-type
    ``_desc-preproc{Cond}_epo.fif`` (default) variants.

    Returns the matched output file paths.
    """
    expected_dir = Path(payload["output_dir"])
    subject = payload["subject"]
    task_label = payload["task_label"]
    candidates = sorted(expected_dir.glob(
        f"sub-{subject}_*task-{task_label}_*desc-preproc*_epo.fif"
    ))
    if payload["kind"] == "per_run":
        run_label = payload["run_label"]
        if run_label is not None:
            candidates = [c for c in candidates if f"run-{run_label}" in c.name]
        marker = f"run-{run_label}" if run_label is not None else "single"
    else:  # concat
        candidates = [c for c in candidates if "_run-" not in c.name]
        marker = "concat"
    if not candidates:
        raise FileNotFoundError(
            f"Preprocessing reported success but no _desc-preproc*_epo.fif "
            f"appeared in {expected_dir} for sub-{subject}, task-{task_label}, "
            f"{marker}. Most likely cause: stale nipype cache pointing at a "
            f"previously-deleted output. Wipe the work directory "
            f"({payload['work_dir']}) and re-run."
        )
    return candidates


def _preproc_iteration(payload):
    """Worker entry point — process one (task, run) preprocessing iteration.

    Raises on any failure; the dispatcher propagates the exception up
    to the CLI entry point and the run aborts. Returns an
    :class:`IterationResult` on success.
    """
    t0 = time.time()
    work_dir = Path(payload["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = _setup_worker_log(work_dir, payload["identifier"])
    wf = _build_preproc_workflow(payload)
    wf.run(plugin="Linear")
    outs = _check_preproc_output_or_raise(payload)
    return IterationResult(
        identifier=payload["identifier"],
        output_files=[str(p) for p in outs],
        log_path=str(log_path),
        duration_s=time.time() - t0,
    )


def _concat_iteration(payload):
    """Worker entry point — process one task in concat-runs mode.

    Same shape as :func:`_preproc_iteration`; the only material
    difference is that ``payload['run_label']`` is a list (or None)
    instead of a single value, which the loader concatenates inside the
    workflow. Within a task, runs are inherently serial because of the
    shared work_dir and concatenation semantics.
    """
    t0 = time.time()
    work_dir = Path(payload["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = _setup_worker_log(work_dir, payload["identifier"])
    wf = _build_preproc_workflow(payload)
    wf.run(plugin="Linear")
    outs = _check_preproc_output_or_raise(payload)
    return IterationResult(
        identifier=payload["identifier"],
        output_files=[str(p) for p in outs],
        log_path=str(log_path),
        duration_s=time.time() - t0,
    )


def _analysis_iteration(payload):
    """Worker entry point — analyze one (subject, task, run) group.

    Reads every per-condition preprocessing file in the group, stacks
    them via ``mne.concatenate_epochs`` to recover the union view
    (event_id is preserved across the concat), restores
    ``epochs.baseline`` from the first file's sidecar, and drives the
    analysis workflow once. Epochs objects are not pickle-friendly
    across the executor boundary; file paths are.
    """
    import mne

    t0 = time.time()
    work_dir = Path(payload["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = _setup_worker_log(work_dir, payload["identifier"])

    preproc_files = [Path(p) for p in payload["preproc_files"]]
    if len(preproc_files) == 1:
        epochs = mne.read_epochs(
            str(preproc_files[0]), preload=True, verbose=False,
        )
    else:
        per_condition = [
            mne.read_epochs(str(p), preload=True, verbose=False)
            for p in preproc_files
        ]
        epochs = mne.concatenate_epochs(per_condition)

    # Restore epochs.baseline from the first file's sidecar. All
    # per-condition files in a group share the same baseline window
    # (preproc applies it once before splitting), so reading just one
    # is sufficient. EpochsArray (used in save_preprocessing_outputs)
    # doesn't carry baseline metadata, so without this restore the
    # downstream evoked.baseline would be None and the analysis
    # sidecar would silently drop its Baseline field, breaking the
    # report's RMS SNR row.
    epochs = _restore_epochs_baseline(
        epochs, preproc_files[0].with_suffix(".json"),
    )

    analysis_wf = create_analysis_workflow()
    analysis_wf.base_dir = payload["work_dir"]
    analysis_wf.inputs.inputnode.epochs = epochs
    analysis_wf.inputs.inputnode.by_event_type = payload["by_event_type"]
    analysis_wf.inputs.inputnode.difference_pairs = payload.get("difference_pairs")
    analysis_wf.inputs.inputnode.bids_root = payload["bids_root"]
    analysis_wf.inputs.inputnode.subject = payload["subject"]
    analysis_wf.inputs.inputnode.original_filename = payload["original_filename"]
    analysis_wf.inputs.inputnode.output_dir = payload["analysis_subject_dir"]
    # derivatives_root is the BIDS-App output_dir (e.g.
    # /data/derivatives_concat). save_analysis_outputs uses it as the
    # derivatives root and constructs the per-subject path internally.
    analysis_wf.inputs.inputnode.derivatives_root = payload["derivatives_root"]
    analysis_wf.run(plugin="Linear")

    # Propagate provenance from each per-condition source file's sidecar
    # into the corresponding analysis-output sidecars. The first file
    # carries the canonical (task, session, run) tokens used by the
    # rest of the group.
    for preproc_file in preproc_files:
        _propagate_run_provenance(
            preproc_file=preproc_file,
            analysis_dir=Path(payload["analysis_subject_dir"]),
        )
    return IterationResult(
        identifier=payload["identifier"],
        log_path=str(log_path),
        duration_s=time.time() - t0,
    )


def _dispatch(worker_fn, payloads, n_procs, kind_label):
    """Run `worker_fn` over `payloads` in a ProcessPoolExecutor.

    Returns the list of :class:`IterationResult` for successful
    iterations. Worker exceptions propagate up and abort the run
    (fail-fast); the dispatcher does not catch.
    """
    log = logging.getLogger()
    if not payloads:
        log.info("No %s iterations to dispatch.", kind_label)
        return []

    n_workers = max(1, min(int(n_procs or 1), len(payloads)))
    if n_workers == 1:
        # In-process fast path — no executor overhead, no spawn cost.
        results = []
        for p in payloads:
            r = worker_fn(p)
            results.append(r)
            log.info(
                "[%s %s] OK (%.1fs) log=%s",
                kind_label, r.identifier, r.duration_s, r.log_path,
            )
        return results

    ctx = multiprocessing.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futures = {ex.submit(worker_fn, p): p["identifier"] for p in payloads}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            log.info(
                "[%s %s] OK (%.1fs) log=%s",
                kind_label, r.identifier, r.duration_s, r.log_path,
            )
    return results


# Define parser to collect required inputs
def get_parser():
    """Create and return an argument parser for BIDS-App."""

    # get version
    __version__ = _pkg_version("ffrprep")

    # define parser description
    parser = argparse.ArgumentParser(
        description=("ffrprep: A BIDS-App for standardized FFR preprocessing " "and analysis"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # add parser argument for version
    parser.add_argument("-v", "--version", action="version", version="ffrprep version {}".format(__version__))

    # BIDS-App standard arguments
    parser.add_argument(
        "bids_dir",
        action="store",
        type=Path,
        help=("The directory with the input dataset " "formatted according to the BIDS standard."),
    )
    parser.add_argument(
        "output_dir",
        action="store",
        type=Path,
        help="The directory where the output files "
        "should be stored. If you are running group level analysis "
        "this folder should be prepopulated with the results of the "
        "participant level analysis.",
    )
    parser.add_argument(
        "analysis_level",
        choices=["participant", "group"],
        help="Level of the analysis that will be performed. "
        "Multiple participant level analyses can be run independently "
        "(in parallel) using the same output_dir.",
    )

    # Participant selection
    parser.add_argument(
        "--participant_label",
        help=(
            "The label(s) of the participant(s) that should "
            "be processed. The label corresponds to "
            "sub-<participant_label> from the BIDS spec "
            '(so it does not include "sub-"). '
            "Multiple participants can be specified with "
            "a space-separated list."
        ),
        nargs="+",
    )

    # Processing stage control
    parser.add_argument(
        "--stage",
        choices=["preprocessing", "analysis", "both"],
        default="both",
        help=("Processing stage to run: preprocessing only, " "analysis only, or both stages sequentially."),
    )

    # Preprocessing parameters
    preproc_group = parser.add_argument_group("Preprocessing options")
    preproc_group.add_argument(
        "--ref_channels",
        help=(
            "Reference channel(s) for re-referencing. Provide space-separated "
            "channel names (e.g. --ref_channels M1 M2). Use 'average' for "
            "average reference. Comma-separated single-argument styles are "
            "also accepted for backward compatibility (e.g. 'M1,M2')."
        ),
        nargs="+",
        type=str,
        default=None,
    )
    preproc_group.add_argument(
        "--high_pass", type=float, help="High-pass filter cutoff frequency in Hz.", default=1.0
    )
    preproc_group.add_argument(
        "--low_pass", type=float, help="Low-pass filter cutoff frequency in Hz.", default=40.0
    )
    # New MNE-style arguments (l_freq/h_freq) to make mapping explicit.
    # These override the legacy --high_pass/--low_pass when provided.
    preproc_group.add_argument(
        "--l_freq",
        type=float,
        help=(
            "Lower-pass edge in Hz (MNE name: l_freq). "
            "If provided, overrides --high_pass. Use 'None' via --no-filter to disable."
        ),
        default=None,
    )
    preproc_group.add_argument(
        "--h_freq",
        type=float,
        help=(
            "Upper-pass edge in Hz (MNE name: h_freq). "
            "If provided, overrides --low_pass. Use 'None' via --no-filter to disable."
        ),
        default=None,
    )
    preproc_group.add_argument(
        "--no-filter",
        action="store_true",
        help=(
            "Disable filtering entirely (equivalent to l_freq=None and h_freq=None). "
            "When set, any provided l_freq/h_freq/high_pass/low_pass are ignored."
        ),
        default=False,
    )
    preproc_group.add_argument(
        "--baseline",
        help=(
            "Baseline correction period. Provide two numbers: START END "
            "in seconds (e.g. --baseline -0.2 0 for -200ms to 0ms)."
        ),
        nargs=2,
        type=float,
        metavar=("START", "END"),
        default=[-0.2, 0.0],
    )
    preproc_group.add_argument(
        "--picks",
        help=(
            "Channels to include in epochs. Comma-separated list or single "
            "channel name (e.g. 'Cz' or 'Cz,Fz'). If not provided, all "
            "channels are considered."
        ),
        default=None,
    )
    preproc_group.add_argument(
        "--on-missing",
        choices=["warn", "raise", "ignore"],
        default="warn",
        help=(
            "Behavior when events referenced by event_id are missing: "
            "warn, raise, or ignore. Matches MNE's on_missing option."
        ),
    )
    preproc_group.add_argument(
        "--event-id",
        help=(
            'Event id mapping. Provide a JSON string like \'{"A":1,"B":2}\' '
            "or comma-separated pairs like 'A:1,B:2'. If not provided, "
            "event ids will be inferred from events file or annotations."
        ),
        default=None,
    )
    preproc_group.add_argument(
        "--events-file",
        help=(
            "Optional path to an events.tsv file to use for epoching. "
            "If not provided, events will be inferred from annotations."
        ),
        default=None,
    )
    preproc_group.add_argument(
        "--tmin", type=float, help="Start time of epochs relative to event onset (s).", default=-0.2
    )
    preproc_group.add_argument(
        "--tmax", type=float, help="End time of epochs relative to event onset (s).", default=0.6
    )
    preproc_group.add_argument(
        "--reject-eeg",
        type=float,
        help=(
            "Peak-to-peak rejection threshold for EEG channels in Volts. "
            "Set to 0 to disable automatic rejection."
        ),
        default=75e-6,
    )
    preproc_group.add_argument(
        "--no-auto-reject",
        action="store_true",
        help=("Disable automatic amplitude-based epoch rejection."),
        default=False,
    )
    preproc_group.add_argument(
        "--concat-runs",
        action="store_true",
        help=(
            "Concatenate multiple runs for a subject and process them as a "
            "single recording. By default runs are processed separately."
        ),
        default=False,
    )
    preproc_group.add_argument(
        "--save-each-node",
        action="store_true",
        help=(
            "Write intermediate outputs to disk after each preprocessing "
            "step (disk-backed mode). This reduces peak memory at the "
            "expense of increased I/O and runtime."
        ),
        default=False,
    )
    preproc_group.add_argument(
        "--run",
        dest="run",
        nargs="+",
        help=(
            "Run label(s) to process for the participant (without the 'run-' "
            "prefix). If not provided, all runs found for the subject will "
            "be processed. Multiple runs can be provided as space-separated "
            "values, or as a single comma-separated string (e.g. '1 2' or "
            "'1,2')."
        ),
    )
    preproc_group.add_argument(
        "--task",
        dest="task",
        nargs="+",
        help=(
            "Task label(s) to process for the participant (without the 'task-' "
            "prefix). If not provided, all tasks found for the subject will "
            "be processed. Multiple tasks can be provided as space-separated "
            "values, or as a single comma-separated string (e.g. 'active passive')."
        ),
    )

    # Analysis parameters
    analysis_group = parser.add_argument_group("Analysis options")
    analysis_group.add_argument(
        "--split-by-trial-type",
        dest="split_by_trial_type",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Emit per-trial-type epoched and evoked outputs. "
            "Pass --no-split-by-trial-type to fall back to a single "
            "combined output per (subject, task, run)."
        ),
    )
    # Deprecated alias for --split-by-trial-type. Kept for one release
    # so existing user scripts don't break; new scripts should use
    # --split-by-trial-type / --no-split-by-trial-type.
    analysis_group.add_argument(
        "--by_event_type",
        dest="split_by_trial_type",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    analysis_group.add_argument(
        "--trial-types",
        dest="trial_types",
        nargs="+",
        default=None,
        help=(
            "Restrict per-trial-type outputs to this subset of trial "
            "types (matched against the events.tsv ``trial_type`` "
            "column). If omitted, all trial types are emitted."
        ),
    )
    analysis_group.add_argument(
        "--difference-pairs",
        dest="difference_pairs",
        nargs="+",
        default=None,
        help=(
            "Difference evokeds to compute, given as ``A:B`` tokens "
            "(e.g. Pos:Neg). For exactly two trial types the diff is "
            "auto-computed; this flag is required to opt in to diffs "
            "when there are three or more trial types."
        ),
    )

    # General options
    parser.add_argument(
        "--skip_bids_validation",
        action="store_true",
        help=("Assume the input dataset is BIDS compliant " "and skip the validation."),
    )
    parser.add_argument(
        "--n_procs", type=int, default=1,
        help=(
            "Number of parallel (task, run) workers per subject. "
            "Each worker runs its own preprocessing or analysis "
            "workflow with the Linear plugin. Memory footprint scales "
            "linearly with this value. A failure in any worker aborts "
            "the run. Default: 1 (sequential). For multi-node / cluster "
            "scaling, run one ffrprep invocation per subject (e.g. via "
            "slurm array or GNU parallel) and use --n_procs to control "
            "intra-subject parallelism."
        ),
    )
    parser.add_argument("--work_dir", type=Path, help="Path where intermediate results should be " "stored.")

    return parser


def parse_baseline(baseline_str):
    """Parse baseline input to a (start, end) tuple of floats.

    Accepts:
    - a list/tuple of two floats (from argparse with nargs=2)
    - a comma-separated string like '-0.2,0'
    - a single numeric string which is interpreted as (value, 0.0)
    """
    # If argparse provided two floats (nargs=2), just return them
    if isinstance(baseline_str, (list, tuple)) and len(baseline_str) == 2:
        return float(baseline_str[0]), float(baseline_str[1])

    # If a comma-separated string was provided
    if isinstance(baseline_str, str) and "," in baseline_str:
        start, end = baseline_str.split(",")
        return float(start), float(end)

    # Otherwise treat as single numeric value (start) with 0 as end
    return float(baseline_str), 0.0


def parse_ref_channels(ref_str):
    """Parse reference channels string."""
    if not ref_str:
        return None

    if isinstance(ref_str, (list, tuple)):
        # Already a sequence; ensure items are stripped and handle
        # comma-separated tokens inside any element for backward
        # compatibility (e.g. ['M1,M2'] -> ['M1','M2']).
        out = []
        for item in ref_str:
            if isinstance(item, str) and "," in item:
                out.extend([c.strip() for c in item.split(",") if c.strip()])
            else:
                out.append(str(item).strip())
        return out

    s = str(ref_str)
    if s.lower() == "average":
        return None  # Average reference
    if "," in s:
        # Split and strip whitespace around channel names
        return [c.strip() for c in s.split(",") if c.strip()]
    # Single channel name; strip whitespace
    return s.strip()


def parse_picks(picks_str):
    """Parse picks string into list or return None."""
    if not picks_str:
        return None
    if isinstance(picks_str, (list, tuple)):
        return picks_str
    # comma-separated
    return [p.strip() for p in str(picks_str).split(",") if p.strip()]


def parse_difference_pairs(values):
    """Parse ``--difference-pairs`` tokens into a list of ``(A, B)`` tuples.

    Accepts a list of ``"A:B"`` strings (one pair per token). Returns
    ``None`` when ``values`` is None or an empty list. Raises
    :class:`argparse.ArgumentTypeError` for malformed tokens (no ``:``,
    empty side, or ``A == B``).
    """
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]
    if len(values) == 0:
        return None

    pairs = []
    for raw in values:
        token = str(raw).strip()
        if ":" not in token:
            raise argparse.ArgumentTypeError(
                f"--difference-pairs token {raw!r} is missing ':'; "
                "expected A:B (e.g. Pos:Neg)."
            )
        a, b = token.split(":", 1)
        a = a.strip()
        b = b.strip()
        if not a or not b:
            raise argparse.ArgumentTypeError(
                f"--difference-pairs token {raw!r} has an empty side; "
                "both A and B must be non-empty."
            )
        if a == b:
            raise argparse.ArgumentTypeError(
                f"--difference-pairs token {raw!r} pairs a trial type "
                "with itself; A and B must differ."
            )
        pairs.append((a, b))
    return pairs


def parse_event_id(event_id_str):
    """Parse event-id mapping. Accept JSON string or key:val pairs.

    Examples:
      '{"A": 1, "B": 2}'  OR  'A:1,B:2'
    """
    if not event_id_str:
        return None
    import json

    # JSON form first — detect by leading '{' (after stripping whitespace).
    s = str(event_id_str).strip()
    if s.startswith("{"):
        return json.loads(s)

    # Otherwise: key:val comma-separated format.
    mapping = {}
    for part in s.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        v = v.strip()
        # Coerce to int when the token looks like an integer literal.
        if v.lstrip("-").isdigit():
            v = int(v)
        mapping[k] = v
    return mapping if mapping else None


def run_ffrprep():
    """Run the FFR processing pipeline as a BIDS-App."""
    # Get arguments from parser
    args = get_parser().parse_args()

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set computational environment
    if os.getenv("IS_DOCKER"):
        exec_env = "singularity"
        cgroup = Path("/proc/1/cgroup")
        if cgroup.exists() and "docker" in cgroup.read_text():
            exec_env = "docker"
    else:
        exec_env = "local"

    # Validate input data
    if args.skip_bids_validation:
        print("Input data will not be checked for BIDS compliance.")
    else:
        print("Making sure the input data is BIDS compliant " "(warnings can be ignored in most cases).")
        validate_input_dir(exec_env, args.bids_dir, args.participant_label)

    # Only run participant-level analysis for now
    if args.analysis_level != "participant":
        print("Currently only participant-level analysis is supported.")
        return

    # Parse processing parameters
    baseline = parse_baseline(args.baseline)
    ref_channels = parse_ref_channels(args.ref_channels)
    # Determine rejection criteria: allow disabling automatic rejection via
    # --no-auto-reject, otherwise use the provided --reject-eeg threshold.
    if args.no_auto_reject:
        reject_value = None
    else:
        reject_value = {"eeg": float(args.reject_eeg)}

    # Determine filtering parameters. Support new MNE-style l_freq/h_freq
    # CLI flags while preserving backward compatibility with
    # --high_pass/--low_pass. --no-filter forces both to None.
    if args.no_filter:
        effective_l = None
        effective_h = None
        print("Filtering disabled by --no-filter; l_freq/h_freq set to None.")
    else:
        # Prefer explicit MNE-style args when provided
        effective_l = args.l_freq if getattr(args, "l_freq", None) is not None else args.high_pass
        effective_h = args.h_freq if getattr(args, "h_freq", None) is not None else args.low_pass
        print(f"Using filter settings: l_freq={effective_l}, h_freq={effective_h}")

    # Get participant labels using pybids for robust querying
    print("Discovering participants using pybids...")
    subjects = get_participants(str(args.bids_dir), args.participant_label)

    if not subjects:
        if args.participant_label:
            print(f"No participants found matching {args.participant_label}")
        else:
            print("No participants with EEG data found in the dataset")
        return

    print(f"Processing subjects: {subjects}")

    # Process each subject
    for subject in subjects:
        print(f"\n{'='*60}")
        print(f"Processing subject: sub-{subject}")
        print(f"{'='*60}")

        # Set up derivatives directories. The user-supplied
        # args.output_dir (the second positional CLI argument) is the
        # destination root for ffrprep-preprocessing/ and
        # ffrprep-analysis/. When omitted, setup_derivatives_directories
        # falls back to the BIDS-conventional bids_root/derivatives/.
        derivatives_info = setup_derivatives_directories(
            args.bids_dir,
            subject,
            create_preprocessing=args.stage in ["preprocessing", "both"],
            create_analysis=args.stage in ["analysis", "both"],
            output_dir=args.output_dir,
        )

        print(f"Derivatives will be stored in: " f"{derivatives_info['derivatives_root']}")

        # Persist a structured run log next to the subject's derivatives so
        # there is a record of inputs/outputs/computing logs after the run.
        # The handler is added per-subject so each subject gets its own log
        # file in its own derivatives directory.
        _setup_subject_log(derivatives_info, subject, args.stage)

        # Picklable snapshots for ProcessPoolExecutor workers — every
        # value needed by a worker must come through these dicts.
        args_snap = _snapshot_args(args)
        deriv_snap = _snapshot_deriv(derivatives_info)

        # Create workflow based on stage
        if args.stage in ["preprocessing", "both"]:
            print("\n" + "=" * 60)
            print("Running preprocessing workflow...")
            print("=" * 60)

            # Determine runs for this subject. Default behavior: process
            # runs separately (one workflow run per run label). If the user
            # asked to concatenate runs via `--concat-runs`, run once
            # with run_label=None and let `load_data()` perform
            # concatenation.
            meta = get_sessions_tasks_runs(str(args.bids_dir), subject)
            # Allow overriding runs via CLI --run. Accept space-separated
            # values or a single comma-separated string per argument.
            if args.run:
                # Flatten any comma-separated entries
                provided = []
                for entry in args.run:
                    if isinstance(entry, str) and "," in entry:
                        provided.extend([r.strip() for r in entry.split(",") if r.strip()])
                    else:
                        provided.append(entry)
                runs = provided
            else:
                runs = meta.get("runs", [None])

            # Allow overriding tasks via CLI --task (same parsing rules)
            if args.task:
                provided_tasks = []
                for entry in args.task:
                    if isinstance(entry, str) and "," in entry:
                        provided_tasks.extend([t.strip() for t in entry.split(",") if t.strip()])
                    else:
                        provided_tasks.append(entry)
                tasks = provided_tasks
            else:
                tasks = meta.get("tasks", [None])

            # Validate requested tasks and runs against dataset metadata
            valid_tasks = [t for t in meta.get("tasks", []) if t is not None]
            valid_runs = [r for r in meta.get("runs", []) if r is not None]

            if args.task:
                # Identify missing/invalid task entries
                missing_tasks = [t for t in tasks if t not in valid_tasks]
                # If the user explicitly requested tasks but none are valid,
                # this is a user error: fail loudly so the caller can fix the CLI
                # invocation. If some are valid, warn and proceed with the
                # subset that exists.
                if missing_tasks and len(missing_tasks) == len(tasks):
                    print(f"ERROR: None of the requested task(s) {tasks} were found for subject sub-{subject}.")
                    if valid_tasks:
                        print(f"Available tasks for this subject: {valid_tasks}")
                    else:
                        print("No tasks available for this subject in the dataset.")
                    sys.exit(2)
                if missing_tasks:
                    print(
                        f"Warning: Requested task(s) {missing_tasks} not found for "
                        f"subject sub-{subject}; they will be ignored."
                    )
                    tasks = [t for t in tasks if t in valid_tasks]
                if not tasks:
                    # If tasks ended up empty after filtering, skip this subject
                    print(f"No valid tasks to process for subject sub-{subject}; skipping.")
                    continue

            if args.run:
                # If the dataset doesn't list any runs but the user explicitly
                # requested runs, treat as an error (likely CLI/dataset mismatch).
                if not valid_runs:
                    print(
                        f"ERROR: No runs available for subject sub-{subject} in "
                        f"dataset but --run was provided ({runs})."
                    )
                    sys.exit(2)

                missing_runs = [r for r in runs if r not in valid_runs]
                if missing_runs and len(missing_runs) == len(runs):
                    # User requested runs but none exist
                    print(f"ERROR: None of the requested run(s) {runs} were found for subject sub-{subject}.")
                    print(f"Available runs for this subject: {valid_runs}")
                    sys.exit(2)
                if missing_runs:
                    print(
                        f"Warning: Requested run(s) {missing_runs} not found for "
                        f"subject sub-{subject}; they will be ignored."
                    )
                    runs = [r for r in runs if r in valid_runs]
                if not runs:
                    print(f"No valid runs to process for subject sub-{subject}; skipping.")
                    continue

            # If concatenation requested and specific runs provided, we'll
            # pass the selected run list into the loader so only those runs
            # are concatenated. If no runs provided, all runs for the task
            # will be concatenated.

            if args.concat_runs:
                # One worker per task; runs within a task remain serial
                # (concatenation semantics demand it).
                payloads = [
                    _make_concat_payload(
                        args_snap, deriv_snap, subject, task_label,
                        runs if args.run else None,
                        ref_channels, effective_l, effective_h,
                        baseline, reject_value,
                    )
                    for task_label in tasks
                ]
                _dispatch(_concat_iteration, payloads,
                          n_procs=args.n_procs, kind_label="concat")
                _build_preproc_report(args, derivatives_info, subject)
            else:
                # One worker per (task, run) — fully independent iterations.
                payloads = [
                    _make_preproc_payload(
                        args_snap, deriv_snap, subject, task_label, run,
                        ref_channels, effective_l, effective_h,
                        baseline, reject_value,
                    )
                    for task_label in tasks
                    for run in runs
                ]
                _dispatch(_preproc_iteration, payloads,
                          n_procs=args.n_procs, kind_label="preproc")
                _build_preproc_report(args, derivatives_info, subject)

        if args.stage in ["analysis", "both"]:
            print("\n" + "=" * 60)
            print("Running analysis workflow...")
            print("=" * 60)

            # Group preprocessing outputs by (task, session, run). With
            # split-by-trial-type=True a single (task, run) tuple yields
            # multiple per-condition files that are stacked back together
            # inside the worker before driving the analysis workflow once.
            preproc_subject_dir = Path(derivatives_info["preprocessing_subject_dir"])
            groups = _collect_analysis_groups(preproc_subject_dir)

            if not groups:
                print(f"ERROR: No preprocessing outputs found for subject {subject}.")
                print(f"Expected location: {preproc_subject_dir}")
                print("Please run preprocessing stage first or use 'both' stage.")
                continue
            n_files = sum(len(g["preproc_files"]) for g in groups)
            print(
                f"Found {n_files} preprocessing output(s) in "
                f"{len(groups)} (task, run) group(s) to analyze."
            )

            # One worker per (task, run) group — fully independent.
            payloads = [
                _make_analysis_payload(args_snap, deriv_snap, subject, g)
                for g in groups
            ]
            _dispatch(_analysis_iteration, payloads,
                      n_procs=args.n_procs, kind_label="analysis")
            print(f"Analysis completed. Outputs saved to: {derivatives_info['analysis_subject_dir']}")
            _build_analysis_report(args, derivatives_info, subject)

    print("\n" + "=" * 60)
    print("ffrprep processing completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_ffrprep()
