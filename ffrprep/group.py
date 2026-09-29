"""
group module for ffrprep.

Aggregates already-computed participant-level derivatives
(``ffrprep-analysis/sub-*/`` and ``ffrprep-preprocessing/sub-*/eeg/``)
into group-level summaries: a grand-average evoked response per (task,
session, run), a per-subject scalar-metrics table, and a group HTML
report. This mirrors the aggregation-only scope other BIDS Apps use for
their "group" level (e.g. MRIQC's group report): it summarizes what
participant-level ffrprep already computed, it does not run any
group-level statistics.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import mne
import pandas as pd

import ffrprep.reports as reports
from ffrprep.analysis import compute_power, response_consistency, rms_snr

# Matches combined/diff evoked filenames written by
# preproc.save_analysis_outputs(), e.g.
# "sub-01_ses-1_task-ffr_run-1_desc-evoked.fif" or
# "sub-01_task-ffr_desc-evokedDiffPositiveVsNegative.fif".
_EVOKED_RE = re.compile(
    r"^sub-(?P<subject>[^_]+)"
    r"(?:_ses-(?P<session>[^_]+))?"
    r"_task-(?P<task>[^_]+)"
    r"(?:_run-(?P<run>[^_]+))?"
    r"_desc-(?P<desc>[A-Za-z0-9]+)\.fif$"
)
_DIFF_DESC_RE = re.compile(r"^evokedDiff([A-Za-z0-9]+)Vs([A-Za-z0-9]+)$")

# Matches preprocessing epochs filenames written by
# preproc.save_preprocessing_outputs(), e.g.
# "sub-01_task-ffr_run-1_desc-preprocPositive_epo.fif" or the combined
# "sub-01_task-ffr_run-1_desc-preproc_epo.fif".
_PREPROC_EPO_RE = re.compile(
    r"^sub-(?P<subject>[^_]+)"
    r"(?:_ses-(?P<session>[^_]+))?"
    r"_task-(?P<task>[^_]+)"
    r"(?:_run-(?P<run>[^_]+))?"
    r"_desc-preproc(?P<condition>[A-Za-z0-9]*)_epo\.fif$"
)


def discover_group_inputs(derivatives_root, participant_label=None, task=None):
    """
    Discover participant-level derivatives to aggregate at the group level.

    Parameters
    ----------
    derivatives_root : str or pathlib.Path
        Root directory containing ``ffrprep-analysis/`` and
        ``ffrprep-preprocessing/``, as produced by
        :func:`ffrprep.preproc.setup_derivatives_directories`.
    participant_label : list of str, optional
        Restrict to these subject labels (without the ``sub-`` prefix).
        If None, all subjects found under ``ffrprep-analysis/`` are used.
    task : list of str, optional
        Restrict to these task labels. If None, all tasks found are used.

    Returns
    -------
    inputs : dict
        Dictionary with keys ``"evoked"``, ``"diff"``, and
        ``"epochs"``. ``"evoked"`` and ``"epochs"`` map a
        ``(task, session, run)`` tuple to ``{subject: Path}``
        (``{subject: [Path, ...]}`` for ``"epochs"``); ``"diff"``
        maps a ``(task, session, run, condition_a, condition_b)``
        tuple to ``{subject: Path}``. ``session`` and ``run`` are
        None when not present in a filename.
    """
    derivatives_root = Path(derivatives_root)
    analysis_dir = derivatives_root / "ffrprep-analysis"
    preproc_dir = derivatives_root / "ffrprep-preprocessing"

    evoked = defaultdict(dict)
    diff = defaultdict(dict)
    for path in sorted(analysis_dir.glob("sub-*/*.fif")):
        match = _EVOKED_RE.match(path.name)
        if not match:
            continue
        fields = match.groupdict()
        subject = fields["subject"]
        if participant_label and subject not in participant_label:
            continue
        task_label = fields["task"]
        if task and task_label not in task:
            continue
        key = (task_label, fields["session"], fields["run"])
        desc = fields["desc"]
        if desc == "evoked":
            evoked[key][subject] = path
            continue
        diff_match = _DIFF_DESC_RE.match(desc)
        if diff_match:
            a, b = diff_match.groups()
            diff[key + (a, b)][subject] = path

    epochs = defaultdict(lambda: defaultdict(list))
    for path in sorted(preproc_dir.glob("sub-*/eeg/*.fif")):
        match = _PREPROC_EPO_RE.match(path.name)
        if not match:
            continue
        fields = match.groupdict()
        subject = fields["subject"]
        if participant_label and subject not in participant_label:
            continue
        task_label = fields["task"]
        if task and task_label not in task:
            continue
        key = (task_label, fields["session"], fields["run"])
        epochs[key][subject].append(path)

    return {
        "evoked": dict(evoked),
        "diff": dict(diff),
        "epochs": {key: dict(value) for key, value in epochs.items()},
    }


def _read_evoked_with_baseline(path):
    """Load a saved Evoked and restore ``.baseline`` from its sidecar.

    MNE's ``Evoked.save()`` doesn't write the baseline window into the
    ``.fif``, so a bare ``mne.read_evokeds()`` always comes back with
    ``baseline=None`` — which would silently drop RMS SNR everywhere
    downstream (both here and in ``ffrprep.reports.build_evoked_section``,
    which both gate on ``evoked.baseline is not None``). Same fix as
    ``ffrprep_cli._build_analysis_report`` applies at participant level.
    """
    evoked = mne.read_evokeds(str(path), condition=0)
    sidecar_path = Path(path).with_suffix(".json")
    if evoked.baseline is None and sidecar_path.exists():
        baseline = json.loads(sidecar_path.read_text()).get("Baseline")
        if baseline is not None:
            evoked.apply_baseline(tuple(baseline))
    return evoked


def compute_grand_average(evoked_paths):
    """
    Load per-subject Evoked files and compute their grand average.

    Parameters
    ----------
    evoked_paths : dict[str, pathlib.Path]
        Mapping of subject label to the Evoked ``.fif`` file to include.

    Returns
    -------
    grand_average : mne.Evoked
    subjects : list of str
        Subject labels included, sorted, in the order averaged.
    """
    subjects = sorted(evoked_paths)
    evokeds = [_read_evoked_with_baseline(evoked_paths[subject]) for subject in subjects]
    grand_average = mne.grand_average(evokeds)
    return grand_average, subjects


def compute_subject_metrics(evoked_paths, epochs_paths=None, response_window=(0.100, 0.200)):
    """
    Compute per-subject scalar FFR metrics from saved derivatives.

    Recomputes the same scalar metrics ``ffrprep.reports`` shows in the
    participant-level report (RMS SNR, band power) directly from each
    subject's saved combined Evoked, plus trial-to-trial response
    consistency from the saved preprocessing Epochs when available.
    Nothing here is computed from raw data; this only aggregates
    already-computed participant-level derivatives.

    Parameters
    ----------
    evoked_paths : dict[str, pathlib.Path]
        Mapping of subject label to combined Evoked ``.fif`` path.
    epochs_paths : dict[str, list of pathlib.Path], optional
        Mapping of subject label to that subject's preprocessing Epochs
        ``.fif`` path(s) for the same (task, session, run). When a
        subject has multiple (per-trial-type) epochs files, response
        consistency is averaged across them.
    response_window : tuple of (float, float)
        Response window for RMS SNR / band power, matching the
        participant-level ``--response-window`` default.

    Returns
    -------
    metrics : pandas.DataFrame
        One row per subject with columns ``subject``, ``n_avg``,
        ``rms_snr``, ``band_power_90_110hz``, ``response_consistency``.
    """
    epochs_paths = epochs_paths or {}
    resp_lower, resp_upper = response_window
    rows = []
    for subject in sorted(evoked_paths):
        evoked = _read_evoked_with_baseline(evoked_paths[subject])
        row = {
            "subject": subject,
            "n_avg": int(evoked.nave),
            "rms_snr": None,
            "band_power_90_110hz": compute_power(
                evoked, f_low=90, f_high=110, t_low=resp_lower, t_high=resp_upper,
            ),
            "response_consistency": None,
        }
        if evoked.baseline is not None:
            row["rms_snr"] = rms_snr(evoked, response_lower=resp_lower, response_upper=resp_upper)

        consistencies = []
        for epochs_path in epochs_paths.get(subject, []):
            epochs = mne.read_epochs(str(epochs_path))
            # Matches the participant-level report's gate
            # (ffrprep.reports.build_epoch_section): below 10 trials
            # the pairwise-correlation estimate is unstable.
            if len(epochs) < 10:
                continue
            mean_r, _ = response_consistency(epochs)
            consistencies.append(mean_r)
        if consistencies:
            row["response_consistency"] = sum(consistencies) / len(consistencies)

        rows.append(row)
    return pd.DataFrame(rows)


def _sort_key(key):
    """Sort key for (task, session, run, ...) tuples that may mix None
    with strings — plain tuple comparison raises TypeError in that case.
    """
    return tuple((v is None, "" if v is None else str(v)) for v in key)


def _basename(task, session, run):
    parts = [f"task-{task}"]
    if session:
        parts.insert(0, f"ses-{session}")
    if run:
        parts.append(f"run-{run}")
    return "_".join(parts)


def save_group_outputs(derivatives_root, task, session, run, grand_average, subjects, desc="grandAverage"):
    """
    Save a grand-average Evoked and its provenance sidecar under
    ``derivatives_root/ffrprep-group/``.

    Parameters
    ----------
    derivatives_root : str or pathlib.Path
    task, session, run : str or None
    grand_average : mne.Evoked
    subjects : list of str
        Subject labels that contributed to the grand average.
    desc : str
        BIDS ``desc-`` entity for the grand-average file.

    Returns
    -------
    evoked_path : pathlib.Path
    """
    derivatives_root = Path(derivatives_root)
    group_dir = derivatives_root / "ffrprep-group"
    group_dir.mkdir(parents=True, exist_ok=True)
    _write_group_dataset_description(group_dir)

    base = _basename(task, session, run)
    evoked_path = group_dir / f"{base}_desc-{desc}_ave.fif"
    grand_average.save(evoked_path, overwrite=True)

    sidecar = {
        "Description": "Grand-average FFR evoked response across subjects.",
        "GeneratedBy": [
            {"Name": "ffrprep", "Description": "Frequency-following response analysis pipeline"}
        ],
        "TaskName": task,
        "Subjects": subjects,
        "NumberOfSubjects": len(subjects),
        "SamplingFrequency": float(grand_average.info["sfreq"]),
        "AverageCount": int(grand_average.nave),
    }
    if session is not None:
        sidecar["Session"] = str(session)
    if run is not None:
        sidecar["Run"] = str(run)
    with open(evoked_path.with_suffix(".json"), "w") as f:
        json.dump(sidecar, f, indent=2)

    return evoked_path


def save_group_metrics(derivatives_root, task, session, run, metrics):
    """Save a per-subject metrics table to a group-level TSV file."""
    derivatives_root = Path(derivatives_root)
    group_dir = derivatives_root / "ffrprep-group"
    group_dir.mkdir(parents=True, exist_ok=True)
    _write_group_dataset_description(group_dir)

    base = _basename(task, session, run)
    metrics_path = group_dir / f"{base}_metrics.tsv"
    metrics.to_csv(metrics_path, sep="\t", index=False)
    return metrics_path


def _write_group_dataset_description(group_dir):
    dataset_desc_path = group_dir / "dataset_description.json"
    if dataset_desc_path.exists():
        return
    dataset_desc = {
        "Name": "ffrprep group-level outputs",
        "BIDSVersion": "1.6.0",
        "GeneratedBy": [
            {"Name": "ffrprep", "Description": "Frequency-following response analysis pipeline"}
        ],
    }
    with open(dataset_desc_path, "w") as f:
        json.dump(dataset_desc, f, indent=2)


def run_group_level(args):
    """
    Run the group-level aggregation step (BIDS-App ``analysis_level=group``).

    Discovers participant-level ``ffrprep-analysis``/``ffrprep-preprocessing``
    derivatives under ``args.output_dir``, computes a grand-average evoked
    response and a per-subject scalar-metrics table for each (task,
    session, run) with at least two contributing subjects, writes them
    plus a group HTML report to ``args.output_dir/ffrprep-group/``, and
    returns the list of output paths written.

    This step only aggregates outputs participant-level ffrprep has
    already computed; it does not run any group-level statistics.
    """
    output_dir = Path(args.output_dir)
    response_window = tuple(getattr(args, "response_window", None) or (0.100, 0.200))
    participant_label = getattr(args, "participant_label", None)
    task_filter = getattr(args, "task", None)

    inputs = discover_group_inputs(output_dir, participant_label=participant_label, task=task_filter)

    if not inputs["evoked"]:
        print(
            f"No participant-level ffrprep-analysis derivatives found under {output_dir}. "
            "Run participant-level ffrprep first."
        )
        return []

    written = []
    report_groups = []
    for (task, session, run), evoked_paths in sorted(inputs["evoked"].items(), key=lambda kv: _sort_key(kv[0])):
        label = _basename(task, session, run)
        if len(evoked_paths) < 2:
            print(f"Skipping {label}: only {len(evoked_paths)} subject(s) found, need at least 2 to aggregate.")
            continue

        print(f"Aggregating {label} across {len(evoked_paths)} subjects")
        grand_average, subjects = compute_grand_average(evoked_paths)
        evoked_path = save_group_outputs(output_dir, task, session, run, grand_average, subjects)
        written.append(evoked_path)

        epochs_paths = inputs["epochs"].get((task, session, run), {})
        metrics = compute_subject_metrics(evoked_paths, epochs_paths, response_window=response_window)
        metrics_path = save_group_metrics(output_dir, task, session, run, metrics)
        written.append(metrics_path)

        sections = [
            reports.build_evoked_section(
                grand_average,
                section_id=f"{label}-grand-average",
                title="Grand-average evoked response",
                label=f"N={len(subjects)} subjects",
                response_window=response_window,
            ),
            reports.build_metrics_table_section(
                metrics,
                section_id=f"{label}-metrics",
                title="Per-subject metrics",
            ),
        ]

        for (d_task, d_session, d_run, cond_a, cond_b), diff_paths in sorted(
            inputs["diff"].items(), key=lambda kv: _sort_key(kv[0])
        ):
            if (d_task, d_session, d_run) != (task, session, run) or len(diff_paths) < 2:
                continue
            diff_average, diff_subjects = compute_grand_average(diff_paths)
            diff_desc = f"grandAverageDiff{cond_a}Vs{cond_b}"
            written.append(
                save_group_outputs(output_dir, task, session, run, diff_average, diff_subjects, desc=diff_desc)
            )
            sections.append(
                reports.build_evoked_section(
                    diff_average,
                    section_id=f"{label}-diff-{cond_a}-vs-{cond_b}",
                    title=f"Grand-average difference ({cond_a} vs {cond_b})",
                    label=f"N={len(diff_subjects)} subjects",
                    response_window=response_window,
                )
            )

        report_groups.append(reports.make_group(sections=sections, task=task, session=session, run=run))

    if report_groups:
        report_path = reports.build_group_report(output_dir / "ffrprep-group", groups=report_groups)
        written.append(Path(report_path))
    else:
        print("No task/run had at least 2 subjects to aggregate; no group outputs were written.")

    return written
