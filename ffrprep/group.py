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
from ffrprep.analysis import (
    compute_power,
    harmonic_amplitudes,
    load_wav_mono,
    resample_signal,
    response_consistency,
    rms_snr,
    stim_to_resp_xcorr,
)

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
_PER_TYPE_DESC_RE = re.compile(r"^evoked(?P<label>[A-Za-z0-9]+)$")

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
        Dictionary with keys ``"evoked"``, ``"diff"``, ``"by_type"``
        and ``"epochs"``. ``"evoked"`` and ``"epochs"`` map a
        ``(task, session, run)`` tuple to ``{subject: Path}``
        (``{subject: [Path, ...]}`` for ``"epochs"``); ``"diff"``
        maps a ``(task, session, run, condition_a, condition_b)``
        tuple to ``{subject: Path}``; ``"by_type"`` maps a
        ``(task, session, run)`` tuple to
        ``{subject: {condition_label: Path}}`` for the per-trial-type
        evoked files. ``session`` and ``run`` are None when not present
        in a filename.
    """
    derivatives_root = Path(derivatives_root)
    analysis_dir = derivatives_root / "ffrprep-analysis"
    preproc_dir = derivatives_root / "ffrprep-preprocessing"

    evoked = defaultdict(dict)
    diff = defaultdict(dict)
    by_type = defaultdict(lambda: defaultdict(dict))
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
            continue
        type_match = _PER_TYPE_DESC_RE.match(desc)
        if type_match:
            by_type[key][subject][type_match.group("label")] = path

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
        "by_type": {
            key: {subject: dict(conds) for subject, conds in subjects.items()}
            for key, subjects in by_type.items()
        },
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
    _harmonize_channel_names(evokeds, subjects)
    grand_average = mne.grand_average(evokeds)
    return grand_average, subjects


def _harmonize_channel_names(evokeds, subjects):
    """Give single-channel Evokeds a common channel name (in place).

    Multi-site FFR cohorts often record one channel that different sites
    label differently (e.g. ``A32`` vs ``Cz``); ``mne.grand_average``
    refuses to combine those. When every Evoked has exactly one channel
    the name carries no information, so all are renamed to the most
    common one. Differing names across multi-channel Evokeds cannot be
    aligned automatically and raise a ``ValueError``.
    """
    names = [tuple(ev.ch_names) for ev in evokeds]
    if len(set(names)) <= 1:
        return
    if all(len(n) == 1 for n in names):
        counts = {}
        for (name,) in names:
            counts[name] = counts.get(name, 0) + 1
        target = max(sorted(counts), key=counts.get)
        print(
            f"Note: single-channel evokeds use different channel names {sorted(counts)}; "
            f"treating them as the same channel and renaming to '{target}'."
        )
        for ev in evokeds:
            if ev.ch_names[0] != target:
                ev.rename_channels({ev.ch_names[0]: target})
        return
    first = names[0]
    for subject, name in zip(subjects, names):
        if name != first:
            raise ValueError(
                f"sub-{subject} has channels {list(name)} but sub-{subjects[0]} has "
                f"{list(first)}; grand averaging needs identical channel sets."
            )


def _load_polarity_sum(cond_paths):
    """Sum of a subject's two per-trial-type Evokeds (or None).

    FFR analyses that suppress the cochlear microphonic / stimulus
    artifact add the responses to the two stimulus polarities (a **sum**
    of the per-polarity averages, not a trial-weighted mean). Returns
    None unless exactly two per-type files are available.
    """
    if not cond_paths or len(cond_paths) != 2:
        return None
    first, second = (_read_evoked_with_baseline(cond_paths[c]) for c in sorted(cond_paths))
    summed = mne.combine_evoked([first, second], weights=[1, 1])
    summed.baseline = first.baseline
    return summed


def _window(array, sfreq, tmin, tmax):
    """Nearest-sample, inclusive ``tmin``-``tmax`` slice of a signal that starts at t=0."""
    return array[int(round(tmin * sfreq)): int(round(tmax * sfreq)) + 1]


def _stim_to_resp_columns(summed, stim, spec, prefix="stim2resp"):
    """r / z / lag columns for one stimulus-to-response window definition."""
    sfreq = float(summed.info["sfreq"])
    stim_tmin, stim_tmax = spec["stim_window"]
    resp_tmin, resp_tmax = spec["resp_window"]
    stim_seg = _window(stim, sfreq, stim_tmin, stim_tmax)
    resp_seg = summed.copy().crop(tmin=resp_tmin, tmax=resp_tmax).data[0]
    r, z, lag_ms = stim_to_resp_xcorr(
        stim_seg, resp_seg, sfreq, lag_range_ms=spec.get("lag_range_ms"),
    )
    return {f"{prefix}_r": r, f"{prefix}_z": z, f"{prefix}_lag_ms": lag_ms}


def compute_subject_metrics(
    evoked_paths,
    epochs_paths=None,
    response_window=(0.100, 0.200),
    by_type_paths=None,
    harmonics=None,
    stimulus=None,
    n_trials_presented=None,
):
    """
    Compute per-subject scalar FFR metrics from saved derivatives.

    Recomputes the same scalar metrics ``ffrprep.reports`` shows in the
    participant-level report (RMS SNR, band power) directly from each
    subject's saved combined Evoked, plus trial-to-trial response
    consistency from the saved preprocessing Epochs when available.
    Nothing here is computed from raw data; this only aggregates
    already-computed participant-level derivatives.

    Optionally (``harmonics`` / ``stimulus``) it also derives spectral and
    stimulus-following measures from the **sum of the two per-trial-type
    (polarity) averages**, the response commonly analyzed in FFR studies.

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
    by_type_paths : dict[str, dict[str, pathlib.Path]], optional
        ``{subject: {condition_label: per-type Evoked path}}``. Required
        for the polarity-sum measures; a subject needs exactly two
        conditions, otherwise those columns are NaN.
    harmonics : dict, optional
        Keyword arguments for :func:`ffrprep.analysis.harmonic_amplitudes`
        (``f0``, ``n_harmonics``, ``bin_hz``, ``tmin``, ``tmax``). Adds
        ``rms_snr_polarity_sum``, ``f0_uv`` and ``upper_harmonics_uv``.
    stimulus : dict, optional
        ``{"path": wav, "stim_window": (t0, t1), "resp_window": (t0, t1)}``
        plus optionally ``"lag_range_ms": (lo, hi)`` and
        ``"lag_resp_window": (t0, t1)``. Adds ``stim2resp_r``,
        ``stim2resp_z``, ``stim2resp_lag_ms`` (and ``stim2resp_lim_*`` when
        a lag range is given). The stimulus is resampled to the EEG rate.
    n_trials_presented : int, optional
        Trials presented per recording; when given, ``usable_pct`` =
        100 * kept trials / presented is added.

    Returns
    -------
    metrics : pandas.DataFrame
        One row per subject with columns ``subject``, ``n_avg``,
        ``rms_snr``, ``band_power_90_110hz``, ``response_consistency`` and
        the optional columns described above.
    """
    epochs_paths = epochs_paths or {}
    by_type_paths = by_type_paths or {}
    resp_lower, resp_upper = response_window
    use_sum = harmonics is not None or stimulus is not None
    stim_cache = {}
    rows = []
    for subject in sorted(evoked_paths):
        evoked = _read_evoked_with_baseline(evoked_paths[subject])
        row = {"subject": subject, "n_avg": int(evoked.nave)}
        if n_trials_presented:
            row["usable_pct"] = 100.0 * int(evoked.nave) / float(n_trials_presented)
        row.update({
            "rms_snr": None,
            "band_power_90_110hz": compute_power(
                evoked, f_low=90, f_high=110, t_low=resp_lower, t_high=resp_upper,
            ),
            "response_consistency": None,
        })
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

        if use_sum:
            summed = _load_polarity_sum(by_type_paths.get(subject))
            if summed is None:
                print(
                    f"Warning: sub-{subject} does not have exactly two per-trial-type "
                    "evoked files; polarity-sum metrics are left empty."
                )
            row.update(_polarity_sum_metrics(
                summed, harmonics, stimulus, response_window, stim_cache,
            ))

        rows.append(row)
    return pd.DataFrame(rows)


def _polarity_sum_metrics(summed, harmonics, stimulus, response_window, stim_cache):
    """Columns derived from the polarity-summed response (NaN when unavailable)."""
    nan = float("nan")
    columns = {"rms_snr_polarity_sum": nan}
    if harmonics is not None:
        columns.update({"f0_uv": nan, "upper_harmonics_uv": nan})
    if stimulus is not None:
        columns.update({"stim2resp_r": nan, "stim2resp_z": nan, "stim2resp_lag_ms": nan})
        if stimulus.get("lag_range_ms") is not None:
            columns.update({
                "stim2resp_lim_r": nan, "stim2resp_lim_z": nan, "stim2resp_lim_lag_ms": nan,
            })
    if summed is None:
        return columns

    if summed.baseline is not None:
        columns["rms_snr_polarity_sum"] = rms_snr(
            summed, response_lower=response_window[0], response_upper=response_window[1],
        )
    if harmonics is not None:
        result = harmonic_amplitudes(summed, **harmonics)
        columns["f0_uv"] = result["f0"]
        columns["upper_harmonics_uv"] = result["upper_harmonics"]
    if stimulus is not None:
        sfreq = float(summed.info["sfreq"])
        if sfreq not in stim_cache:
            wav, wav_sfreq = load_wav_mono(stimulus["path"])
            stim_cache[sfreq] = resample_signal(wav, wav_sfreq, sfreq)
        stim = stim_cache[sfreq]
        # The primary measure searches ALL lags; ``lag_range_ms`` belongs to
        # the separate lag-limited ("lim") variant below.
        primary_spec = {
            "stim_window": stimulus["stim_window"],
            "resp_window": stimulus["resp_window"],
        }
        columns.update(_stim_to_resp_columns(summed, stim, primary_spec))
        if stimulus.get("lag_range_ms") is not None:
            lim_spec = {
                "stim_window": stimulus["stim_window"],
                "resp_window": stimulus.get("lag_resp_window", stimulus["resp_window"]),
                "lag_range_ms": stimulus["lag_range_ms"],
            }
            columns.update(_stim_to_resp_columns(summed, stim, lim_spec, prefix="stim2resp_lim"))
    return columns


def add_qc_flags(metrics, min_usable_pct=None, min_snr=None):
    """Flag recordings that fail quality thresholds; never drop rows.

    Adds boolean ``qc_usable_pct_ok`` (``usable_pct >= min_usable_pct``)
    and/or ``qc_snr_ok`` (polarity-sum RMS SNR, else the combined-evoked
    ``rms_snr``, ``>= min_snr``) columns for each threshold given, plus
    ``qc_include`` (all checks pass) and ``qc_reason`` (the failed checks,
    ``n/a`` when none). A missing value (NaN) fails its check. The metrics
    table keeps every subject and the grand averages are unchanged, so
    downstream analyses choose whether to filter on ``qc_include``.

    Returns ``metrics`` unchanged when no threshold is given.
    """
    if min_usable_pct is None and min_snr is None:
        return metrics

    flagged = metrics.copy()
    ok_columns = []
    reasons = [[] for _ in range(len(flagged))]

    def _check(column, threshold, ok_name, label):
        values = flagged[column]
        passed = (values >= threshold).fillna(False).astype(bool)
        flagged[ok_name] = passed
        ok_columns.append(ok_name)
        for i, ok in enumerate(passed):
            if not ok:
                reasons[i].append(f"{label}<{threshold:g}")

    if min_usable_pct is not None:
        if "usable_pct" not in flagged:
            raise ValueError("--min-usable-pct needs --n-trials-presented (no usable_pct column)")
        _check("usable_pct", min_usable_pct, "qc_usable_pct_ok", "usable_pct")
    if min_snr is not None:
        snr_column = "rms_snr_polarity_sum" if "rms_snr_polarity_sum" in flagged else "rms_snr"
        _check(snr_column, min_snr, "qc_snr_ok", snr_column)

    flagged["qc_include"] = flagged[ok_columns].all(axis=1)
    flagged["qc_reason"] = ["; ".join(r) or "n/a" for r in reasons]
    return flagged


def merge_covariates(metrics, covariate_paths, return_sources=False):
    """Left-join subject-level covariate TSVs onto a metrics table.

    Each TSV needs a ``participant_id`` column (``sub-<label>``, as in a
    BIDS ``participants.tsv`` or ``phenotype/*.tsv``). Columns already
    present in ``metrics`` are not overwritten. Missing files are skipped
    with a warning.

    Returns
    -------
    pandas.DataFrame
        ``metrics`` with the covariate columns appended. With
        ``return_sources=True`` a ``(table, sources)`` tuple, where
        ``sources`` maps each appended column to the file it came from.
    """
    merged = metrics.copy()
    sources = {}
    key = "sub-" + merged["subject"].astype(str)
    for path in covariate_paths or []:
        path = Path(path)
        if not path.exists():
            print(f"Warning: covariate file not found, skipping: {path}")
            continue
        table = pd.read_csv(path, sep="\t", dtype={"participant_id": str})
        if "participant_id" not in table.columns:
            print(f"Warning: {path} has no 'participant_id' column, skipping.")
            continue
        new_columns = [c for c in table.columns if c != "participant_id" and c not in merged.columns]
        if not new_columns:
            continue
        lookup = table.drop_duplicates("participant_id").set_index("participant_id")[new_columns]
        for column in new_columns:
            merged[column] = key.map(lookup[column])
            sources[column] = path
    return (merged, sources) if return_sources else merged


def _covariate_sidecar_entry(path, column):
    """The ``column`` entry of the JSON sidecar next to a covariate TSV, or ``None``."""
    sidecar = Path(path).with_suffix(".json")
    if not sidecar.exists():
        return None
    try:
        entry = json.loads(sidecar.read_text()).get(column)
    except (OSError, ValueError):
        return None
    return entry if isinstance(entry, dict) else None


def build_metrics_dictionary(
    columns,
    *,
    response_window=(0.100, 0.200),
    harmonics=None,
    stimulus=None,
    n_trials_presented=None,
    min_usable_pct=None,
    min_snr=None,
    covariate_sources=None,
):
    """Build a BIDS-style data dictionary for the group metrics TSV.

    Parameters
    ----------
    columns : sequence of str
        Column names of the saved metrics table, in order.
    response_window, harmonics, stimulus, n_trials_presented, min_usable_pct, min_snr
        The options the metrics were computed with (as returned by
        ``_metric_options_from_args``); descriptions embed the actual
        windows and thresholds so the dictionary documents the run.
    covariate_sources : dict, optional
        ``{column: path}`` from ``merge_covariates(..., return_sources=True)``.
        A covariate's description (and levels/units) is copied from the JSON
        sidecar next to its TSV (``participants.json``, ``phenotype/*.json``)
        when present.

    Returns
    -------
    dict
        ``{column: {"Description": ..., "Units": ...}}`` with one entry per
        column, in the order of ``columns``. Unknown columns get a stub entry.
    """
    resp = f"{response_window[0] * 1e3:g}-{response_window[1] * 1e3:g} ms"
    info = {
        "subject": {"Description": "Participant label (without the 'sub-' prefix)"},
        "n_avg": {"Description": "Number of trials averaged in the combined evoked (kept after rejection)",
                  "Units": "trials"},
        "usable_pct": {
            "Description": f"100 * n_avg / {n_trials_presented} trials presented"
            if n_trials_presented else "100 * n_avg / trials presented",
            "Units": "%",
        },
        "rms_snr": {
            "Description": f"RMS of the response window ({resp}) divided by RMS of the baseline, "
                           "combined evoked response",
            "Units": "ratio",
        },
        "band_power_90_110hz": {
            "Description": f"Mean 90-110 Hz power in the response window ({resp}), combined evoked response",
            "Units": "V^2",
        },
        "response_consistency": {
            "Description": "Mean pairwise trial-to-trial correlation of the preprocessed epochs "
                           "(averaged over runs; empty when a recording has fewer than 10 epochs)",
            "Units": "r",
        },
        "rms_snr_polarity_sum": {
            "Description": f"RMS of the response window ({resp}) divided by RMS of the baseline, "
                           "sum of the two per-trial-type averages",
            "Units": "ratio",
        },
        "qc_usable_pct_ok": {
            "Description": f"usable_pct >= {min_usable_pct:g}" if min_usable_pct is not None
            else "usable_pct passes the threshold",
            "Levels": {"True": "passes", "False": "fails or missing"},
        },
        "qc_snr_ok": {
            "Description": f"SNR (rms_snr_polarity_sum when present, else rms_snr) >= {min_snr:g}"
            if min_snr is not None else "SNR passes the threshold",
            "Levels": {"True": "passes", "False": "fails or missing"},
        },
        "qc_include": {
            "Description": "All requested QC checks pass; subjects are flagged, never removed",
            "Levels": {"True": "include", "False": "flagged"},
        },
        "qc_reason": {"Description": "Failed QC checks ('n/a' when none)"},
    }
    if harmonics:
        window = f"{harmonics['tmin'] * 1e3:g}-{harmonics['tmax'] * 1e3:g} ms"
        band = f"+/-{harmonics['bin_hz'] / 2:g} Hz"
        info["f0_uv"] = {
            "Description": f"Mean FFT amplitude (2|FFT|/L) within {band} of {harmonics['f0']:g} Hz, "
                           f"{window}, sum of the two per-trial-type averages",
            "Units": "uV",
        }
        info["upper_harmonics_uv"] = {
            "Description": f"Sum over harmonics 2..{harmonics['n_harmonics']} of the mean FFT amplitude "
                           f"within {band} of k*{harmonics['f0']:g} Hz, {window}, sum of the two "
                           "per-trial-type averages",
            "Units": "uV",
        }
    if stimulus:
        stim_win = f"{stimulus['stim_window'][0] * 1e3:g}-{stimulus['stim_window'][1] * 1e3:g} ms"
        resp_win = f"{stimulus['resp_window'][0] * 1e3:g}-{stimulus['resp_window'][1] * 1e3:g} ms"
        base = (f"cross-correlation ('coeff' normalization) of the stimulus ({stim_win}) with the "
                f"polarity-summed response ({resp_win})")
        info["stim2resp_r"] = {"Description": f"Maximum {base} over all lags", "Units": "r"}
        info["stim2resp_z"] = {"Description": "Fisher z transform (arctanh) of stim2resp_r", "Units": "z"}
        info["stim2resp_lag_ms"] = {
            "Description": "Lag of the maximum stimulus-to-response correlation "
                           "(positive = response delayed)", "Units": "ms",
        }
        lag_range = stimulus.get("lag_range_ms")
        if lag_range:
            lim = f"restricted to lags of {lag_range[0]:g} to {lag_range[1]:g} ms"
            info["stim2resp_lim_r"] = {"Description": f"Maximum {base}, {lim}", "Units": "r"}
            info["stim2resp_lim_z"] = {"Description": "Fisher z transform of stim2resp_lim_r", "Units": "z"}
            info["stim2resp_lim_lag_ms"] = {
                "Description": f"Lag of the maximum lag-restricted correlation ({lim})", "Units": "ms",
            }

    dictionary = {}
    for column in columns:
        entry = info.get(column)
        if entry is None and column in (covariate_sources or {}):
            path = covariate_sources[column]
            entry = _covariate_sidecar_entry(path, column) or {
                "Description": f"Joined from {Path(path).name}",
            }
        dictionary[column] = entry or {"Description": "No description available"}
    return dictionary


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


def save_group_metrics(derivatives_root, task, session, run, metrics, dictionary=None):
    """Save a per-subject metrics table to a group-level TSV file.

    When ``dictionary`` (see :func:`build_metrics_dictionary`) is given it
    is written as ``<base>_metrics.json`` next to the TSV.
    """
    derivatives_root = Path(derivatives_root)
    group_dir = derivatives_root / "ffrprep-group"
    group_dir.mkdir(parents=True, exist_ok=True)
    _write_group_dataset_description(group_dir)

    base = _basename(task, session, run)
    metrics_path = group_dir / f"{base}_metrics.tsv"
    metrics.to_csv(metrics_path, sep="\t", index=False)
    if dictionary is not None:
        metrics_path.with_suffix(".json").write_text(json.dumps(dictionary, indent=2) + "\n")
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


def _qc_summary(metrics):
    """Report summary entry for the QC flags (empty when no threshold was set)."""
    if "qc_include" not in metrics:
        return {}
    n_flagged = int((~metrics["qc_include"]).sum())
    return {"QC flagged": f"{n_flagged} of {len(metrics)}"}


def _metric_options_from_args(args):
    """Translate group-level CLI args into ``compute_subject_metrics`` options.

    Every attribute is read with ``getattr`` so callers that build a bare
    namespace (tests, scripts) keep working; unset options leave the
    corresponding metrics off.
    """
    harmonics = None
    f0 = getattr(args, "f0", None)
    if f0:
        tmin, tmax = getattr(args, "harmonic_window", None) or (0.060, 0.180)
        harmonics = {
            "f0": float(f0),
            "n_harmonics": int(getattr(args, "n_harmonics", None) or 10),
            "bin_hz": float(getattr(args, "harmonic_bin_hz", None) or 60.0),
            "tmin": float(tmin),
            "tmax": float(tmax),
        }

    stimulus = None
    stimulus_path = getattr(args, "stimulus", None)
    if stimulus_path:
        stimulus = {
            "path": Path(stimulus_path),
            "stim_window": tuple(getattr(args, "xcorr_stim_window", None) or (0.050, 0.170)),
            "resp_window": tuple(getattr(args, "xcorr_resp_window", None) or (0.060, 0.180)),
        }
        lag_range = getattr(args, "xcorr_lag_range", None)
        if lag_range:
            stimulus["lag_range_ms"] = tuple(lag_range)
            lag_resp_window = getattr(args, "xcorr_lag_resp_window", None)
            if lag_resp_window:
                stimulus["lag_resp_window"] = tuple(lag_resp_window)

    return {
        "harmonics": harmonics,
        "stimulus": stimulus,
        "n_trials_presented": getattr(args, "n_trials_presented", None),
    }


def _covariate_paths_from_args(args):
    """``participants.tsv`` (if present next to bids_dir) plus ``--covariates``."""
    paths = []
    bids_dir = getattr(args, "bids_dir", None)
    if bids_dir:
        participants = Path(bids_dir) / "participants.tsv"
        if participants.exists():
            paths.append(participants)
    paths.extend(Path(p) for p in (getattr(args, "covariates", None) or []))
    return paths


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
    already computed; it does not run any group-level statistics. When
    ``args.f0`` / ``args.stimulus`` are set, the metrics table also gets
    harmonic-amplitude and stimulus-to-response columns computed from the
    sum of the per-trial-type averages, and subject-level covariates from
    ``<bids_dir>/participants.tsv`` and ``args.covariates`` are joined onto
    the saved TSV (never used for inference here). ``args.min_usable_pct`` /
    ``args.min_snr`` add ``qc_*`` flag columns (see :func:`add_qc_flags`);
    flagged subjects stay in the table and the grand averages.
    """
    output_dir = Path(args.output_dir)
    response_window = tuple(getattr(args, "response_window", None) or (0.100, 0.200))
    participant_label = getattr(args, "participant_label", None)
    task_filter = getattr(args, "task", None)
    metric_options = _metric_options_from_args(args)
    covariate_paths = _covariate_paths_from_args(args)
    min_usable_pct = getattr(args, "min_usable_pct", None)
    min_snr = getattr(args, "min_snr", None)
    if min_usable_pct is not None and not metric_options["n_trials_presented"]:
        raise ValueError("--min-usable-pct needs --n-trials-presented")

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
        metrics = compute_subject_metrics(
            evoked_paths, epochs_paths, response_window=response_window,
            by_type_paths=inputs["by_type"].get((task, session, run), {}),
            **metric_options,
        )
        metrics = add_qc_flags(metrics, min_usable_pct=min_usable_pct, min_snr=min_snr)
        metrics_with_covariates, covariate_sources = merge_covariates(
            metrics, covariate_paths, return_sources=True,
        )
        dictionary = build_metrics_dictionary(
            metrics_with_covariates.columns,
            response_window=response_window,
            min_usable_pct=min_usable_pct,
            min_snr=min_snr,
            covariate_sources=covariate_sources,
            **metric_options,
        )
        metrics_path = save_group_metrics(
            output_dir, task, session, run, metrics_with_covariates, dictionary=dictionary,
        )
        written.extend([metrics_path, metrics_path.with_suffix(".json")])

        sections = [
            reports.build_evoked_section(
                grand_average,
                section_id=f"{label}-grand-average",
                title="Grand-average evoked response",
                label=f"N={len(subjects)} subjects",
                response_window=response_window,
            ),
            reports.build_metrics_table_section(
                metrics.drop(columns=[c for c in metrics.columns if c.startswith("qc_")]),
                section_id=f"{label}-metrics",
                title="Per-subject metrics",
                extra_summary=_qc_summary(metrics),
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
