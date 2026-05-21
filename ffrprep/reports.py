"""
This module provides functions for creating, updating, and saving MNE reports.

For FFRPREP BIDS datasets, including support for figures, HTML blocks, and
special MNE objects.
"""

import base64
import io
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from matplotlib.gridspec import GridSpec
from scipy import stats


_TEMPLATE_DIR = Path(__file__).parent / "templates"
_jinja_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def _fig_to_data_uri(fig, dpi=150):
    """Encode a matplotlib Figure as a ``data:image/png;base64,...`` URI.

    Used to embed plots inline so the rendered report is a single
    self-contained HTML file with no sibling assets.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_raw_section(raw, *, section_id, title, label=None, events_fpath=None):
    """Build a section descriptor from a Raw object.

    Wraps :func:`raw_qa` to produce waveform + PSD figures, encodes them
    as inline data URIs, and pairs them with a small summary table
    (sampling rate, duration, channel count). The returned dict is
    consumable by :func:`build_subject_report`.
    """
    sfreq = float(raw.info["sfreq"])
    n_channels = len(raw.ch_names)
    duration_s = raw.n_times / sfreq

    summary = {}
    if label:
        summary["Stage"] = label
    summary["Sampling rate"] = f"{sfreq:g} Hz"
    summary["Duration"] = f"{duration_s:.2f} s"
    summary["Channels"] = str(n_channels)

    figures = []
    for fig, fig_title, caption in raw_qa(raw, events_fpath=events_fpath, save_dir=None):
        figures.append({
            "title": fig_title,
            "caption": caption,
            "data_uri": _fig_to_data_uri(fig),
        })
        plt.close(fig)

    return {
        "id": section_id,
        "title": title,
        "summary": summary,
        "figures": figures,
    }


def build_epoch_section(epochs, *, section_id, title, extra_summary=None):
    """Build a section descriptor from an Epochs object.

    Wraps :func:`epoch_qa` to produce overview / rejection / average /
    drift figures, encodes them as inline data URIs, and pairs them
    with a summary table (n_epochs, channels, sfreq, time window).
    For Epochs with ``>= 10`` trials the summary also surfaces the
    mean pairwise trial-to-trial Pearson correlation
    (:func:`ffrprep.analysis.response_consistency`); below 10 trials
    the metric is unstable so the row is omitted.

    `extra_summary` is appended to the summary table after the standard
    metadata. Use it to surface info that isn't on the Epochs object
    itself — e.g. pre-rejection counts read from a BIDS sidecar.
    """
    from .analysis import response_consistency

    sfreq = float(epochs.info["sfreq"])
    n_channels = len(epochs.ch_names)
    n_epochs = len(epochs)

    summary = {
        "Number of epochs": str(n_epochs),
        "Channels": str(n_channels),
        "Sampling rate": f"{sfreq:g} Hz",
        "Time window": f"{epochs.tmin * 1000:.0f} to {epochs.tmax * 1000:.0f} ms",
    }
    if n_epochs >= 10:
        mean_r, _ = response_consistency(epochs)
        summary["Mean trial-to-trial r"] = f"{mean_r:.3f}"
    if extra_summary:
        summary.update(extra_summary)

    figures = []
    for fig, fig_title, caption in epoch_qa(epochs, save_dir=None):
        figures.append({
            "title": fig_title,
            "caption": caption,
            "data_uri": _fig_to_data_uri(fig),
        })
        plt.close(fig)

    return {
        "id": section_id,
        "title": title,
        "summary": summary,
        "figures": figures,
    }


def evoked_qa(evoked, save_dir=None, prefix="ffr_evoked"):
    """Generate FFR-specific QA figures for an mne.Evoked object.

    Returns a list of (fig, title, caption) tuples covering the views
    that matter for FFR analysis: time-domain waveform, post-stimulus
    PSD, time-frequency representation across the FFR band,
    autocorrelation with confidence interval, and a pitch / confidence
    tracking pair (delegated to analysis.plot_pitch_and_conf).
    """
    from .analysis import (
        autocorrelation, compute_pitch_and_conf, plot_pitch_and_conf,
    )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    ch_names = evoked.ch_names
    pick = "Cz" if "Cz" in ch_names else ch_names[0]
    pick_idx = ch_names.index(pick)

    times_ms = evoked.times * 1000
    data = evoked.data[pick_idx] * 1e6  # µV
    sfreq = float(evoked.info["sfreq"])
    n_ave = evoked.nave

    colors = {
        "blue": "#0173B2",
        "purple": "#924E7D",
        "grey": "#737373",
        "cyan": "#029E73",
        "vermillion": "#D55E00",
    }

    figs = []

    # 1) Time-domain waveform.
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times_ms, data, color=colors["blue"], linewidth=1.5)
    ax.axvline(0, color=colors["purple"], linestyle="--", linewidth=1.5,
               alpha=0.7, label="Stimulus onset")
    ax.axhline(0, color=colors["grey"], linewidth=0.5)
    ax.set_xlabel("Time (ms)", fontsize=10)
    ax.set_ylabel("Amplitude (µV)", fontsize=10)
    ax.set_title(f"Evoked Response - {pick} (N={n_ave} epochs averaged)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(save_dir, f"{prefix}_waveform.png"),
                    dpi=150, bbox_inches="tight")
    figs.append((fig, "Evoked Waveform",
                 f"Time-domain average across {n_ave} epochs"))

    # 2) Power spectral density via Welch on the post-stimulus segment.
    post_mask = evoked.times >= 0
    post_data = data[post_mask]
    n_post = len(post_data)
    if n_post > 16:
        from scipy.signal import welch
        nperseg = min(2048, n_post)
        freqs, psd = welch(post_data, fs=sfreq, nperseg=nperseg)
        fmax = min(2000.0, sfreq / 2.0)
        mask = (freqs >= 65) & (freqs <= fmax)

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.semilogy(freqs[mask], psd[mask], color=colors["purple"], linewidth=1.5)
        ax2.set_xlim([65, fmax])
        ax2.set_xlabel("Frequency (Hz)", fontsize=10)
        ax2.set_ylabel("Power (µV²/Hz)", fontsize=10)
        ax2.set_title(f"Power Spectral Density - {pick} (post-stimulus)",
                      fontsize=11, fontweight="bold")
        ax2.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        if save_dir:
            fig2.savefig(os.path.join(save_dir, f"{prefix}_psd.png"),
                         dpi=150, bbox_inches="tight")
        figs.append((fig2, "Power Spectral Density",
                     f"Spectral content of the post-stimulus response "
                     f"(65-{fmax:.0f} Hz)"))

    # 3) Time-frequency representation across the FFR band.
    tfr_fmin, tfr_fmax = 70.0, min(300.0, sfreq / 2.0 - 1)
    if tfr_fmax > tfr_fmin + 5:
        tfr_freqs = np.arange(tfr_fmin, tfr_fmax + 1, 2.0)
        tfr = evoked.compute_tfr(
            method="multitaper", freqs=tfr_freqs,
            n_cycles=tfr_freqs / 4.0, time_bandwidth=4.0,
            verbose=False,
        )
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        tfr.copy().pick([pick]).plot(
            picks=0, axes=ax3, show=False, colorbar=True, verbose=False,
        )
        ax3.set_title(f"Time-Frequency Representation - {pick}",
                      fontsize=11, fontweight="bold")
        plt.tight_layout()
        if save_dir:
            fig3.savefig(os.path.join(save_dir, f"{prefix}_tfr.png"),
                         dpi=150, bbox_inches="tight")
        figs.append((fig3, "Time-Frequency Representation",
                     f"Multitaper TFR across the FFR band "
                     f"({tfr_fmin:.0f}-{tfr_fmax:.0f} Hz)"))

    # 4) Autocorrelation with 95% CI from the analysis helper.
    # ``autocorrelation`` operates on the first channel by convention;
    # pick that channel before calling so the ACF reflects the FFR pick.
    evoked_pick = evoked.copy().pick([pick])
    acf, ci = autocorrelation(evoked_pick)
    lag_samples = np.arange(len(acf))
    lag_ms = lag_samples / sfreq * 1000.0
    # CI from statsmodels has shape (n_lags, 2) — the lower/upper bounds
    # around each lag. Plot the band only when finite.
    fig4, ax4 = plt.subplots(figsize=(12, 4))
    show_n = min(len(acf), int(0.05 * sfreq))  # show first ~50 ms
    ax4.plot(lag_ms[:show_n], acf[:show_n], color=colors["cyan"], linewidth=1.5)
    if ci.ndim == 2 and ci.shape[1] == 2:
        ax4.fill_between(lag_ms[:show_n], ci[:show_n, 0], ci[:show_n, 1],
                         color=colors["cyan"], alpha=0.15)
    ax4.axhline(0, color=colors["grey"], linewidth=0.5)
    ax4.set_xlabel("Lag (ms)", fontsize=10)
    ax4.set_ylabel("Autocorrelation", fontsize=10)
    ax4.set_title(f"Autocorrelation Function - {pick}",
                  fontsize=11, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        fig4.savefig(os.path.join(save_dir, f"{prefix}_acf.png"),
                     dpi=150, bbox_inches="tight")
    figs.append((fig4, "Autocorrelation",
                 "ACF with 95% confidence interval (first 50 ms of lags)"))

    # 5) Pitch tracking via the analysis helper. plot_pitch_and_conf
    # creates its own figure and does not return it; capture it via
    # plt.gcf() immediately after calling.
    pitch_results = compute_pitch_and_conf(evoked_pick)
    if np.any(~np.isnan(pitch_results.get("pitch_hz_smooth", np.array([])))):
        plot_pitch_and_conf(pitch_results)
        fig5 = plt.gcf()
        fig5.set_size_inches(12, 6)
        if save_dir:
            fig5.savefig(os.path.join(save_dir, f"{prefix}_pitch.png"),
                         dpi=150, bbox_inches="tight")
        figs.append((fig5, "Pitch Track + Confidence",
                     "Sliding-window pitch estimates (autocorrelation "
                     "based) with per-frame confidence metrics"))

    return figs


def build_evoked_section(
    evoked, *, section_id, title, label=None, extra_summary=None,
):
    """Build a section descriptor from an Evoked object.

    Summary table includes the standard metadata plus two FFR-specific
    scalar metrics: RMS SNR over the 100-200 ms response window and
    average band power across 90-110 Hz over the same window (defaults
    matching :func:`ffrprep.analysis.rms_snr` and
    :func:`ffrprep.analysis.compute_power`).

    ``extra_summary`` is folded into the summary table last, so caller-
    supplied keys override any same-named defaults (e.g. for
    surfacing ``corr_stim_to_resp`` scalars computed at the CLI layer
    where the BIDS events.tsv + stimulus files are accessible).
    """
    from .analysis import compute_power, rms_snr

    sfreq = float(evoked.info["sfreq"])
    n_channels = len(evoked.ch_names)
    pick = "Cz" if "Cz" in evoked.ch_names else evoked.ch_names[0]
    evoked_pick = evoked.copy().pick([pick])

    summary = {}
    if label:
        summary["Stage"] = label
    summary["Epochs averaged"] = str(evoked.nave)
    summary["Channels"] = str(n_channels)
    summary["Sampling rate"] = f"{sfreq:g} Hz"
    summary["Time window"] = (
        f"{evoked.tmin * 1000:.0f} to {evoked.tmax * 1000:.0f} ms"
    )
    if getattr(evoked, "comment", None):
        summary["Condition"] = str(evoked.comment)

    # Scalar FFR metrics on the FFR pick. These rely on the analysis
    # helpers' default windows; they're indicative, not authoritative.
    if evoked.baseline is not None:
        snr = rms_snr(evoked_pick)
        summary["RMS SNR (100-200 ms)"] = f"{snr:.2f}"
    band_power = compute_power(evoked_pick, f_low=90, f_high=110, t_low=0.1, t_high=0.2)
    summary["Mean power 90-110 Hz, 100-200 ms"] = f"{band_power:.3e} V²"

    if extra_summary:
        summary.update(extra_summary)

    figures = []
    for fig, fig_title, caption in evoked_qa(evoked, save_dir=None):
        figures.append({
            "title": fig_title,
            "caption": caption,
            "data_uri": _fig_to_data_uri(fig),
        })
        plt.close(fig)

    return {
        "id": section_id,
        "title": title,
        "summary": summary,
        "figures": figures,
    }


def build_phase_consistency_section(
    epochs_a, epochs_b, *, section_id, title, alpha=0.01,
):
    """Build a phase-consistency section from two polarities of Epochs.

    Pairs :func:`ffrprep.analysis.compute_phase_consistency` with
    :func:`ffrprep.analysis.plot_phase_consistency_masked`; the masked
    plot is embedded as a single inline PNG data URI alongside a
    summary table recording the number of sweeps used and the
    significance threshold applied to the mask.

    Parameters
    ----------
    epochs_a, epochs_b : mne.Epochs
        Two polarities of Epochs (typically positive / negative for an
        FFR experiment), matched in sampling rate, time axis, and
        channel count.
    section_id : str
        Anchor used by the report template.
    title : str
        Heading shown in the report.
    alpha : float, default 0.01
        Significance level forwarded to
        ``plot_phase_consistency_masked``; controls the per-cell
        masking cutoff.
    """
    from .analysis import (
        compute_phase_consistency,
        plot_phase_consistency_masked,
    )

    phasecon, xaxis, yaxis, numsweeps = compute_phase_consistency(
        epochs_a, epochs_b,
    )
    fig = plot_phase_consistency_masked(
        phasecon, xaxis, yaxis, numsweeps, alpha=alpha,
    )

    summary = {
        "Number of sweeps used": str(numsweeps),
        "Significance threshold (alpha)": str(alpha),
    }
    figures = [{
        "title": "Phase Consistency (masked)",
        "caption": (
            "Phase consistency across two polarities (A, B) plus "
            "their sum (add) and difference (sub), masked at the "
            "given significance threshold."
        ),
        "data_uri": _fig_to_data_uri(fig),
    }]
    plt.close(fig)

    return {
        "id": section_id,
        "title": title,
        "summary": summary,
        "figures": figures,
    }


def make_group(*, sections, session=None, task=None, run=None,
               title=None, group_id=None):
    """Construct a group descriptor consumable by the report builders.

    A group represents one (optional session, optional task, optional
    run) combination and bundles all sections that belong to it (e.g.
    raw + epoched for the same task/run). The TOC renders groups as
    foldable containers; the main content renders one nested card per
    group with the contained sections as sub-cards.

    `title` and `group_id` default to a BIDS-style label / anchor
    derived from the session/task/run identifiers.
    """
    parts = []
    id_parts = []
    if session is not None:
        parts.append(f"ses-{session}")
        id_parts.append(f"ses-{session}")
    if task is not None:
        parts.append(f"task-{task}")
        id_parts.append(f"task-{task}")
    if run is not None:
        parts.append(f"run-{run}")
        id_parts.append(f"run-{run}")

    if title is None:
        title = " / ".join(parts) if parts else None
    if group_id is None:
        group_id = "-".join(id_parts) if id_parts else "group"

    return {
        "id": group_id,
        "title": title,
        "session": session,
        "task": task,
        "run": run,
        "sections": list(sections),
    }


def _normalize_groups(groups, sections):
    """Return a list of group dicts from either the new `groups` arg or
    the legacy `sections` arg. If both are None, returns []."""
    if groups is not None:
        return list(groups)
    if sections:
        return [{
            "id": "all",
            "title": None,
            "session": None,
            "task": None,
            "run": None,
            "sections": list(sections),
        }]
    return []


def _sortable(value):
    """Sort key that puts None last; otherwise compares the str form."""
    return (value is None, str(value) if value is not None else "")


def _build_nav_tree(flat_groups):
    """Reorganize flat groups into a session > task > run nested tree.

    Each tree node is a dict with ``id``, ``label``, ``children`` (list
    of nodes for further nesting) and ``sections`` (section dicts at
    that level). Intermediate nodes have ``children`` set; leaf nodes
    have ``sections`` set; a node may have both (e.g. task-level
    sections alongside per-run subgroups).

    Empty levels are unwrapped — a dataset with no sessions doesn't
    add a session level to the tree.
    """
    if not flat_groups:
        return []

    by_ses = {}
    for g in flat_groups:
        by_ses.setdefault(g.get("session"), []).append(g)

    root_nodes = []
    for ses in sorted(by_ses.keys(), key=_sortable):
        ses_groups = by_ses[ses]
        ses_prefix = f"ses-{ses}-" if ses is not None else ""

        by_task = {}
        for g in ses_groups:
            by_task.setdefault(g.get("task"), []).append(g)

        task_nodes = []
        for task in sorted(by_task.keys(), key=_sortable):
            task_groups = by_task[task]
            task_prefix = (f"{ses_prefix}task-{task}-"
                           if task is not None else ses_prefix)

            run_children = []
            task_level_sections = []
            for g in sorted(task_groups,
                            key=lambda x: _sortable(x.get("run"))):
                run = g.get("run")
                if run is not None:
                    run_children.append({
                        "id": g.get("id") or f"{task_prefix}run-{run}",
                        "label": f"run-{run}",
                        "children": [],
                        "sections": g.get("sections", []),
                    })
                else:
                    # No run identity — sections live at the task level.
                    task_level_sections.extend(g.get("sections", []))

            if task is not None:
                task_nodes.append({
                    "id": f"{ses_prefix}task-{task}",
                    "label": f"task-{task}",
                    "children": run_children,
                    "sections": task_level_sections,
                })
            else:
                # No task identity — promote children + sections to ses
                task_nodes.extend(run_children)
                # Pseudo-node for any session-level sections.
                if task_level_sections:
                    task_nodes.append({
                        "id": (f"ses-{ses}-sections"
                               if ses is not None else "sections"),
                        "label": None,
                        "children": [],
                        "sections": task_level_sections,
                    })

        if ses is not None:
            root_nodes.append({
                "id": f"ses-{ses}",
                "label": f"ses-{ses}",
                "children": task_nodes,
                "sections": [],
            })
        else:
            # No session identity — promote tasks to root.
            root_nodes.extend(task_nodes)

    return root_nodes


def build_analysis_report(bids_root, subject, out_dir, sections=None,
                          title=None, overview=None, groups=None):
    """Render a single-file HTML analysis report for a subject.

    Same template + layout as :func:`build_subject_report`; the only
    differences are the default page title and the output filename
    (``sub-<id>_analysis_report.html``).

    Either `groups` (preferred, supports nested session/task/run
    structure) or `sections` (flat list, ungrouped) may be passed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if title is None:
        title = f"ffrprep analysis report sub-{subject}"

    template = _jinja_env.get_template("subject_report.html.j2")
    html_text = template.render(
        title=title,
        subject=subject,
        overview=overview,
        nav_tree=_build_nav_tree(_normalize_groups(groups, sections)),
        bids_root=str(bids_root),
    )

    out_path = out_dir / f"sub-{subject}_analysis_report.html"
    out_path.write_text(html_text, encoding="utf-8")
    return str(out_path)


def build_subject_report(bids_root, subject, out_dir, sections=None,
                         title=None, overview=None, groups=None):
    """Render a single-file HTML preprocessing report for a subject.

    Parameters
    ----------
    bids_root : str | Path
        Top-level BIDS dataset directory. Recorded for provenance and
        used to resolve relative paths displayed in the report body.
    subject : str
        BIDS subject identifier without the ``sub-`` prefix
        (e.g. ``"01"``).
    out_dir : str | Path
        Directory where the rendered ``.html`` file is written. Created
        if missing.
    sections : list of dict, optional
        Flat list of section descriptors. Use this when there is no
        natural session/task/run grouping. Mutually exclusive with
        `groups`; if both are passed, `groups` wins.
    groups : list of dict, optional
        Nested structure. Each group dict carries ``id``, ``title``,
        ``session``, ``task``, ``run``, and ``sections`` (list of
        section dicts). Construct via :func:`make_group` for the BIDS
        defaults. The TOC renders one foldable container per group.
    overview : dict, optional
        Top-of-report summary card. Recognised keys: ``summary``
        (key→value table) and ``command`` (rendered as a code block).
    title : str | None
        Page title and ``<h1>`` text. Defaults to
        ``"ffrprep preprocessing report sub-{subject}"``.

    Returns
    -------
    str
        Absolute path to the written HTML file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if title is None:
        title = f"ffrprep preprocessing report sub-{subject}"

    template = _jinja_env.get_template("subject_report.html.j2")
    html_text = template.render(
        title=title,
        subject=subject,
        overview=overview,
        nav_tree=_build_nav_tree(_normalize_groups(groups, sections)),
        bids_root=str(bids_root),
    )

    out_path = out_dir / f"sub-{subject}_preprocessing_report.html"
    out_path.write_text(html_text, encoding="utf-8")
    return str(out_path)


def epoch_qa(epochs, save_dir=None, prefix="ffr_qa"):
    """
    Generate QA figures for an mne.Epochs object.

    Returns a list of tuples (fig, title, caption).
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # Load epochs if path provided. preload=True so downstream
    # operations (get_data, channel selection) work without further I/O.
    if isinstance(epochs, str):
        epochs = mne.read_epochs(epochs, preload=True, verbose=False)

    if not hasattr(epochs, "info") or not epochs.info.get("ch_names"):
        raise ValueError("Epochs object has no channels")

    ch_names = epochs.ch_names
    if "Cz" in ch_names:
        pick_idx = ch_names.index("Cz")
    else:
        pick_idx = 0

    times_ms = epochs.times * 1000
    data = epochs.get_data()[:, pick_idx, :] * 1e6  # µV

    n_epochs = len(epochs)

    # Colorblind-friendly palette
    colors_cb = {
        "blue": "#0173B2",
        "orange": "#DE8F05",
        "cyan": "#029E73",
        "vermillion": "#D55E00",
        "purple": "#924E7D",
        "grey": "#737373",
        "yellow": "#F0E442",
        "black": "#000000",
    }

    figs = []

    # QA FIG 1: Overview
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1a: All epochs with average
    ax1 = fig.add_subplot(gs[0, :])
    for epoch_data in data:
        ax1.plot(times_ms, epoch_data, color=colors_cb["grey"], alpha=0.15, linewidth=0.5)

    avg = data.mean(axis=0)
    std = data.std(axis=0)
    ax1.plot(times_ms, avg, color=colors_cb["black"], linewidth=2, label=f"Average (N={n_epochs})")
    ax1.fill_between(times_ms, avg - std, avg + std, alpha=0.3, color=colors_cb["blue"])
    ax1.axvline(0, color=colors_cb["purple"], linestyle="--", linewidth=1.5, alpha=0.8, label="Stimulus onset")
    ax1.set_xlabel("Time (ms)", fontsize=10)
    ax1.set_ylabel("Amplitude (µV)", fontsize=10)
    ax1.set_title("All Epochs with Average ± 1SD", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 1b: Amplitude distribution
    ax2 = fig.add_subplot(gs[1, 0])
    epoch_ptp = np.ptp(data, axis=1)
    ax2.hist(epoch_ptp, bins=20, color=colors_cb["blue"], edgecolor="black", alpha=0.7)
    ax2.axvline(
        epoch_ptp.mean(),
        color=colors_cb["vermillion"],
        linestyle="--",
        linewidth=2,
        label=f"Mean: {epoch_ptp.mean():.1f} µV",
    )
    ax2.axvline(75, color=colors_cb["orange"], linestyle="--", linewidth=2, label="Rejection threshold: 75 µV")
    ax2.set_xlabel("Peak-to-Peak Amplitude (µV)", fontsize=10)
    ax2.set_ylabel("Number of Epochs", fontsize=10)
    ax2.set_title("Epoch Amplitude Distribution", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 1c: RMS
    ax3 = fig.add_subplot(gs[1, 1])
    epoch_rms = np.sqrt(np.mean(data**2, axis=1))
    ax3.plot(epoch_rms, "o-", color=colors_cb["blue"], markersize=4, linewidth=1)
    ax3.axhline(
        epoch_rms.mean(),
        color=colors_cb["vermillion"],
        linestyle="--",
        linewidth=2,
        label=f"Mean RMS: {epoch_rms.mean():.2f} µV",
    )
    ax3.set_xlabel("Epoch Number", fontsize=10)
    ax3.set_ylabel("RMS Amplitude (µV)", fontsize=10)
    ax3.set_title("RMS Amplitude Across Epochs", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 1d: SNR
    ax4 = fig.add_subplot(gs[1, 2])
    signal_power = avg**2
    noise_power = std**2
    snr_time = 10 * np.log10(signal_power / (noise_power + 1e-10))
    ax4.plot(times_ms, snr_time, color=colors_cb["cyan"], linewidth=2)
    ax4.axhline(0, color=colors_cb["black"], linestyle="-", linewidth=0.5)
    ax4.axvline(0, color=colors_cb["purple"], linestyle="--", linewidth=1.5, alpha=0.8)
    ax4.set_xlabel("Time (ms)", fontsize=10)
    ax4.set_ylabel("SNR (dB)", fontsize=10)
    ax4.set_title("Signal-to-Noise Ratio Over Time", fontsize=11, fontweight="bold")
    ax4.grid(True, alpha=0.3)

    # 1e: Heatmap
    ax5 = fig.add_subplot(gs[2, :2])
    im = ax5.imshow(
        data, aspect="auto", cmap="RdBu_r", extent=[times_ms[0], times_ms[-1], len(data), 0], vmin=-50, vmax=50
    )
    ax5.axvline(0, color=colors_cb["yellow"], linestyle="--", linewidth=2, alpha=0.9)
    ax5.set_xlabel("Time (ms)", fontsize=10)
    ax5.set_ylabel("Epoch Number", fontsize=10)
    ax5.set_title(f"Epochs Heatmap ({ch_names[pick_idx]})", fontsize=11, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax5)
    cbar.set_label("Amplitude (µV)", fontsize=9)

    # 1f: PSD. Use welch with an n_fft that fits the available samples to
    # avoid edge cases where 2048 exceeds the per-epoch length.
    ax6 = fig.add_subplot(gs[2, 2])
    sfreq = epochs.info.get("sfreq", 1000)
    fmax_psd = min(2000, sfreq / 2)
    n_samples_per_epoch = epochs.get_data().shape[-1]
    n_fft_psd = min(2048, n_samples_per_epoch)
    psd = epochs.compute_psd(
        method="welch", fmin=65, fmax=fmax_psd, n_fft=n_fft_psd, verbose=False,
    )
    psds, freqs = psd.get_data(return_freqs=True)
    psd_mean = psds.mean(axis=0)[pick_idx]
    ax6.semilogy(freqs, psd_mean, color=colors_cb["purple"], linewidth=2)
    ax6.set_xlim([65, fmax_psd])

    ax6.set_xlabel("Frequency (Hz)", fontsize=10)
    ax6.set_ylabel("Power (µV²/Hz)", fontsize=10)
    ax6.set_title("Power Spectral Density", fontsize=11, fontweight="bold")
    ax6.grid(True, alpha=0.3, which="both")

    plt.suptitle(f"FFR Quality Assessment - {n_epochs} Epochs", fontsize=14, fontweight="bold", y=0.995)

    if save_dir:
        overview_path = os.path.join(save_dir, f"{prefix}_overview.png")
        fig.savefig(overview_path, dpi=150, bbox_inches="tight")

    figs.append((fig, "Epoch QA Overview", f"Quality assessment of {n_epochs} epochs"))

    # QA FIG 2: Rejection statistics — only render when drop_log
    # actually carries rejections. The saved BIDS-derivatives epochs
    # file only contains accepted epochs (rejection happened pre-save
    # and the discarded events leave no trace in this object's
    # drop_log), so a pie built from this data would just say "100%
    # accepted" — misleading. Pre-rejection counts belong in the
    # section's summary table, sourced from the BIDS sidecar.
    n_good = len(epochs)
    n_rejected = sum(1 for log in epochs.drop_log if len(log) > 0)
    if n_rejected > 0:
        n_total = n_good + n_rejected
        fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
        pie_colors = [colors_cb["blue"], colors_cb["orange"]]
        axes[0].pie(
            [n_good, n_rejected],
            labels=["Accepted", "Rejected"],
            colors=pie_colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 12, "fontweight": "bold"},
        )
        axes[0].set_title(
            f"Epoch Acceptance Rate\n({n_good}/{n_total} epochs)",
            fontsize=12, fontweight="bold",
        )

        rejection_reasons = {}
        for log in epochs.drop_log:
            if len(log) > 0:
                reason = ", ".join(log)
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

        reasons = list(rejection_reasons.keys())
        counts = list(rejection_reasons.values())
        axes[1].barh(reasons, counts, color=colors_cb["orange"], edgecolor="black")
        axes[1].set_xlabel("Number of Epochs", fontsize=11)
        axes[1].set_title("Rejection Reasons", fontsize=12, fontweight="bold")
        axes[1].grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        if save_dir:
            rej_path = os.path.join(save_dir, f"{prefix}_rejection.png")
            fig2.savefig(rej_path, dpi=150, bbox_inches="tight")

        figs.append((
            fig2, "Epoch Rejection Statistics",
            f"{n_rejected} of {n_total} epochs rejected",
        ))

    # QA FIG 3: Average and derivative
    fig3, axes = plt.subplots(2, 1, figsize=(12, 8))
    sem = (data.std(axis=0) / np.sqrt(len(data))) if len(data) > 0 else np.zeros_like(avg)

    axes[0].plot(times_ms, avg, color=colors_cb["black"], linewidth=2, label="Average")
    axes[0].fill_between(times_ms, avg - sem, avg + sem, alpha=0.4, color=colors_cb["blue"], label="±SEM")
    axes[0].axvline(0, color=colors_cb["purple"], linestyle="--", linewidth=1.5, alpha=0.8, label="Stimulus onset")
    axes[0].axhline(0, color=colors_cb["grey"], linestyle="-", linewidth=0.5)
    axes[0].set_ylabel("Amplitude (µV)", fontsize=11)
    axes[0].set_title(f"Average FFR Response (N={n_epochs})", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    derivative = np.diff(avg)
    axes[1].plot(times_ms[:-1], derivative, color=colors_cb["cyan"], linewidth=1.5)
    axes[1].axvline(0, color=colors_cb["purple"], linestyle="--", linewidth=1.5, alpha=0.8)
    axes[1].axhline(0, color=colors_cb["grey"], linestyle="-", linewidth=0.5)
    axes[1].set_xlabel("Time (ms)", fontsize=11)
    axes[1].set_ylabel("dV/dt (µV/sample)", fontsize=11)
    axes[1].set_title("First Derivative (Rate of Change)", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        avg_path = os.path.join(save_dir, f"{prefix}_average.png")
        fig3.savefig(avg_path, dpi=150, bbox_inches="tight")

    figs.append((fig3, "Average FFR Response", "Average response with derivative"))

    # QA FIG 4: Drift check
    epoch_means = data.mean(axis=1)
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(epoch_means)), epoch_means)

    fig4 = plt.figure(figsize=(10, 4))
    plt.plot(epoch_means, "o-", color=colors_cb["blue"], label="Epoch mean amplitude", markersize=5, linewidth=1)
    x = np.arange(len(epoch_means))
    plt.plot(
        x,
        slope * x + intercept,
        color=colors_cb["vermillion"],
        linestyle="--",
        linewidth=2,
        label=f"Linear fit (p={p_value:.4f})",
    )
    plt.xlabel("Epoch number", fontsize=11)
    plt.ylabel("Mean amplitude (µV)", fontsize=11)
    plt.title("Check for Systematic Drift Across Recording", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    if p_value < 0.05:
        plt.text(
            0.5,
            0.95,
            "⚠️ Significant drift detected!",
            transform=plt.gca().transAxes,
            ha="center",
            va="top",
            color=colors_cb["vermillion"],
            fontweight="bold",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor=colors_cb["yellow"], alpha=0.8),
        )
    else:
        plt.text(
            0.5,
            0.95,
            "✓ No significant drift",
            transform=plt.gca().transAxes,
            ha="center",
            va="top",
            color=colors_cb["cyan"],
            fontweight="bold",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="#E8F4F8", alpha=0.8),
        )

    plt.tight_layout()

    if save_dir:
        drift_path = os.path.join(save_dir, f"{prefix}_drift.png")
        fig4.savefig(drift_path, dpi=150, bbox_inches="tight")

    drift_status = "Significant drift detected" if p_value < 0.05 else "No significant drift"
    figs.append((fig4, "Drift Analysis", f"{drift_status} (p={p_value:.4f})"))

    return figs


def raw_qa(raw, events_fpath=None, save_dir=None, prefix="ffr_raw"):
    """
    Create QA figures for a Raw object: waveform snippet and PSD.

    Returns list of (fig, title, caption).
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # Prefer Cz channel
    ch_names = raw.ch_names
    pick = "Cz" if "Cz" in ch_names else ch_names[0]

    # Prepare a short snippet (first 10 seconds)
    sfreq = raw.info.get("sfreq", 1000.0)
    duration = min(10.0, raw.n_times / float(sfreq))
    start = 0.0
    stop = start + duration
    start_samp = int(start * sfreq)
    stop_samp = int(stop * sfreq)

    data, times_sec = raw.get_data(
        picks=[pick], start=start_samp, stop=stop_samp, return_times=True,
    )
    data = data[0] * 1e6  # µV
    times = times_sec * 1000  # ms

    figs = []

    # Waveform figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    ax.plot(times, data, color="#0173B2", linewidth=0.8)
    ax.set_xlabel("Time (ms)", fontsize=10)
    ax.set_ylabel("Amplitude (µV)", fontsize=10)
    ax.set_title(f"Raw Data Snippet ({pick}, {duration:.1f}s)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Mark events if provided. We only attempt to read the file when it
    # exists; a malformed events file is a real error and propagates.
    if events_fpath and os.path.exists(events_fpath):
        ev_df = pd.read_csv(events_fpath, sep="\t")
        if "onset" in ev_df.columns:
            onsets = (ev_df["onset"].values - start) * 1000.0
            for o in onsets:
                if 0 <= o <= duration * 1000:
                    ax.axvline(o, color="#924E7D", linestyle="--", alpha=0.7, linewidth=1)

    plt.tight_layout()

    if save_dir:
        wf_path = os.path.join(save_dir, f"{prefix}_waveform.png")
        fig.savefig(wf_path, dpi=150, bbox_inches="tight")

    figs.append((fig, f"Raw Data Snippet - {pick}", f"First {duration:.1f}s of recording"))

    # PSD figure. n_fft is clamped so it never exceeds the available
    # number of samples (avoids the only common compute_psd failure mode).
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4))
    fmax_psd = min(2000, sfreq / 2.0)
    n_fft_psd = min(2048, raw.n_times)
    psd = raw.compute_psd(picks=[pick], fmin=1, fmax=fmax_psd, n_fft=n_fft_psd, verbose=False)
    psds, freqs = psd.get_data(return_freqs=True)
    psd_mean = psds[0]
    ax2.semilogy(freqs, psd_mean, color="#924E7D", linewidth=2)
    ax2.set_xlim([1, fmax_psd])
    ax2.set_xlabel("Frequency (Hz)", fontsize=10)
    ax2.set_ylabel("Power (V²/Hz)", fontsize=10)
    ax2.set_title(f"Power Spectral Density - {pick}", fontsize=11, fontweight="bold")
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()

    if save_dir:
        psd_path = os.path.join(save_dir, f"{prefix}_psd.png")
        fig2.savefig(psd_path, dpi=150, bbox_inches="tight")

    figs.append((fig2, "Power Spectral Density", "Frequency content of the data"))

    return figs
