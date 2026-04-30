"""
This module provides functions for creating, updating, and saving MNE reports.

For FFRPREP BIDS datasets, including support for figures, HTML blocks, and
special MNE objects.
"""

import os
import time
from mne import Report, open_report
import mne
from pathlib import Path
import html as _html
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy import stats
import pandas as pd
import re
from jinja2 import Environment, FileSystemLoader, select_autoescape


_TEMPLATE_DIR = Path(__file__).parent / "templates"
_jinja_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def build_subject_report(bids_root, subject, out_dir, sections, title=None):
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
    sections : list of dict
        Per-stage section descriptors. Each dict must provide ``id``
        (anchor target, used by the table of contents) and ``title``
        (header text). Phase B extends this with summary fields and
        embedded plots; Phase A renders only headers.
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
        sections=sections,
        bids_root=str(bids_root),
    )

    out_path = out_dir / f"sub-{subject}_preprocessing_report.html"
    out_path.write_text(html_text, encoding="utf-8")
    return str(out_path)


def create_report(bids_root, out_dir=None, filename=None, title=None, overwrite=False):
    """
    Initialize an MNE report.

    Parameters
    ----------
    bids_root : string
        The top-level directory of the BIDS dataset.
    out_dir : string
        Location to save the report. If `None` (default),
        will write the report to '{bidsroot}/derivatives/'.
    filename : string
        Filename for the report. If `None` (default),
        will write the report as 'ffrprep_report_{YYYY:MM:DD:HH:MM:SS}.h5'.
    title : string
        Title for the report. If `None` (default),
        will write the title as 'FFRPREP report {YYYY:MM:DD:HH:MM:SS}'.
    overwrite : boolean
        Whether to overwrite an existing file (`True`).
        If `False` (default) and file exists, report will be saved as
        '{filename}_1.h5'.

    Returns
    -------
    report_fpath : string
        Full filepath to the created report file.
    """
    timenow = time.strftime("%Y-%m-%d_%Hh%Mm%Ss")

    # Define and create the output directory
    if out_dir is None:
        out_dir = os.path.join(bids_root, "derivatives")

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Define and create the output filename - use .h5 for incremental building
    if filename is None:
        filename = f"ffrprep_report_{timenow}.h5"
    else:
        # Ensure .h5 extension for working file
        if filename.endswith(".html"):
            filename = filename[:-5] + ".h5"
        elif not filename.endswith(".h5"):
            filename = f"{filename}.h5"

    report_fpath = os.path.join(out_dir, filename)

    if os.path.isfile(report_fpath):
        if overwrite:
            os.remove(report_fpath)  # Remove old file before creating new one
        else:
            # Find a unique filename by adding a suffix
            fname_base, fname_ext = os.path.splitext(filename)
            counter = 1
            while True:
                new_filename = f"{fname_base}_{counter}{fname_ext}"
                new_path = os.path.join(out_dir, new_filename)
                if not os.path.isfile(new_path):
                    report_fpath = new_path
                    break
                counter += 1

    # Define the report title, if None
    if title is None:
        timenow = time.strftime("%Y-%m-%d_%Hh%Mm%Ss")
        title = f"FFRPREP report {timenow}"

    # Create the report and save as HDF5
    report = Report(title=title, verbose=False)
    report.save(report_fpath, overwrite=True, open_browser=False)

    return report_fpath


def add_to_report(
    report_fpath,
    figure=None,
    figure_title=None,
    figure_caption=None,
    html_text=None,
    html_title=None,
):
    """
    Add content to an existing FFRPREP report.

    Parameters
    ----------
    report_fpath : string
        Full filepath to the existing FFRPREP report (.h5 file).
    figure : matplotlib figure object
        Figure to be added to the existing FFRPREP report.
    figure_title : string
        Title of the figure to be added.
    figure_caption : string
        Caption below the figure to be added.
    html_text : string
        HTML-formatted text string to be added.
    html_title : string
        Title of the HTML text to be added.

    Returns
    -------
    report_fpath : string
        Full filepath to the report file.
    """
    # Open existing HDF5 report
    report = open_report(report_fpath)

    if figure is not None:
        title = figure_title if figure_title is not None else "Figure"
        caption = figure_caption if figure_caption is not None else ""
        report.add_figure(fig=figure, title=title, caption=caption)

    if html_text is not None:
        title_text = html_title or "Content"
        report.add_html(html=html_text, title=title_text)

    # Save back to HDF5
    report.save(report_fpath, overwrite=True, open_browser=False)

    return report_fpath


def save_report(report_fpath, overwrite=True):
    """
    Save the existing FFRPREP report to an HTML file.

    Parameters
    ----------
    report_fpath : string
        Full filepath to the existing .h5 FFRPREP report.
    overwrite : boolean
        Whether to overwrite an existing HTML file.

    Returns
    -------
    html_fpath : string
        Full filepath to the saved HTML report file.
    """
    if report_fpath.endswith(".h5"):
        html_fpath = report_fpath[:-3] + ".html"
    else:
        html_fpath = f"{report_fpath}.html"

    if overwrite is False:
        if os.path.isfile(html_fpath):
            fname_base, fname_ext = os.path.splitext(html_fpath)
            new_filename = f"{fname_base}_1{fname_ext}"
            html_fpath = new_filename

    # If the input is already an HTML file, there's nothing to convert.
    if os.path.exists(report_fpath) and report_fpath.endswith(".html"):
        return report_fpath

    report = open_report(report_fpath)
    report.save(html_fpath, overwrite=True, open_browser=False)
    return html_fpath


def create_subject_report(bids_root, out_dir=None, filename=None, subject_id=None, command=None, overwrite=False):
    """
    Create a report file for a specific subject with a standardized title.

    Parameters
    ----------
    bids_root : str
        Top-level BIDS directory.
    out_dir, filename, overwrite : see `create_report`.
    subject_id : str
        The BIDS subject identifier (e.g. '01' or 'sub-01').
    command : str | None
        Optional command string to include in the summary.

    Returns
    -------
    report_fpath : str
        Filepath to the created report (.h5)
    """
    title = None
    if subject_id is not None:
        sid = subject_id
        if not sid.startswith("sub-"):
            sid = f"sub-{sid}"
        title = f"ffrprep report {sid}"

    report_fpath = create_report(bids_root, out_dir=out_dir, filename=filename, title=title, overwrite=overwrite)

    return report_fpath


def add_report_summary(
    report_fpath,
    command=None,
    raw_files=None,
    events_files=None,
    referenced_files=None,
    filtered_files=None,
    epoched_files=None,
):
    """
    Add a summary block to the report describing the command run and
    the top-level lists/counts of files processed.

    Parameters
    ----------
    report_fpath : str
        Path to the report file (.h5).
    command : str
        Command that was run.
    raw_files, events_files, referenced_files, filtered_files, epoched_files : dict
        Dictionaries mapping task -> run -> list of files.
    """
    parts = []
    parts.append('<div style="margin: 20px 0;">')
    parts.append("<h2>Processing Summary</h2>")

    if command:
        parts.append("<p><strong>Command:</strong></p>")
        parts.append(
            '<pre style="background-color: #f5f5f5; padding: 10px;'
            ' border-radius: 5px; overflow-x: auto;">'
            f"{_html.escape(command)}</pre>"
        )

    def _count_files(obj):
        if obj is None:
            return 0
        if isinstance(obj, dict):
            total = 0
            for task, runs in obj.items():
                if isinstance(runs, dict):
                    for run, files in runs.items():
                        total += len(files) if files else 0
                elif isinstance(runs, (list, tuple)):
                    total += len(runs)
            return total
        if hasattr(obj, "__len__"):
            return len(obj)
        return 0

    parts.append('<table style="border-collapse: collapse; width: 100%; margin-top: 15px;">')
    parts.append('<tr style="background-color: #0173B2; color: white;">')
    parts.append('<th style="padding: 10px; text-align: left;">Processing Stage</th>')
    parts.append('<th style="padding: 10px; text-align: right;">Files Processed</th>')
    parts.append("</tr>")

    stages = [
        ("Raw data loaded", raw_files),
        ("Events files", events_files),
        ("Referenced data", referenced_files),
        ("Filtered data", filtered_files),
        ("Epoched data", epoched_files),
    ]

    for i, (name, obj) in enumerate(stages):
        bg_color = "#f9f9f9" if i % 2 == 0 else "white"
        count = _count_files(obj)
        parts.append(f'<tr style="background-color: {bg_color};">')
        parts.append(f'<td style="padding: 10px;">{name}</td>')
        parts.append(f'<td style="padding: 10px; text-align: right;"><strong>{count}</strong></td>')
        parts.append("</tr>")

    parts.append("</table>")
    parts.append("</div>")

    html_text = "\n".join(parts)
    return add_to_report(report_fpath, html_text=html_text, html_title="Summary")


def add_processing_stages(
    report_fpath, raw_files=None, events_files=None, referenced_files=None, filtered_files=None, epoched_files=None
):
    """
    Add all processing stages organized by task and run.

    Structure: Task -> Run -> [Raw, Referenced, Filtered, Epoched stages]

    Parameters
    ----------
    report_fpath : str
        Path to report .h5 file.
    raw_files, events_files, referenced_files, filtered_files, epoched_files : dict
        Dictionaries with structure: {task: {run: [filepaths]}}
    """
    report = open_report(report_fpath)

    # Collect all unique task/run combinations
    task_run_combos = set()
    for file_dict in [raw_files, events_files, referenced_files, filtered_files, epoched_files]:
        if file_dict:
            for task, runs in file_dict.items():
                if isinstance(runs, dict):
                    for run in runs.keys():
                        task_run_combos.add((task, run))

    if not task_run_combos:
        report.add_html(html="<p>No data files found to process.</p>", title="No data")
        report.save(report_fpath, overwrite=True, open_browser=False)
        return report_fpath

    print(f"Building report for {len(task_run_combos)} task/run combinations")

    # Group by task, sorted for consistent ordering
    tasks_dict = {}
    for task, run in sorted(task_run_combos):
        tasks_dict.setdefault(task, []).append(run)

    for task in sorted(tasks_dict.keys()):
        runs_list = sorted(tasks_dict[task])

        # Add task header
        task_title = task if task else "unknown"
        print(f"\nProcessing Task: {task_title}")
        task_html = (
            '<div style="margin-top: 40px; padding: 15px; background-color: #E8F4F8;'
            ' border-left: 5px solid #0173B2;">'
            f'<h2 style="margin: 0; color: #0173B2;">Task: {_html.escape(task_title)}</h2></div>'
        )
        report.add_html(html=task_html, title=f"Task: {task_title}")

        for run in runs_list:
            run_title = run if run else "no-run"
            print(f"  Processing Run: {run_title}")

            raw_file = _get_file(raw_files, task, run)
            events_file = _get_file(events_files, task, run)
            ref_file = _get_file(referenced_files, task, run)
            filt_file = _get_file(filtered_files, task, run)
            epoch_file = _get_file(epoched_files, task, run)

            # Add run header
            run_html = (
                '<div style="margin-top: 20px; padding: 10px; background-color: #F5F5F5;'
                ' border-left: 3px solid #924E7D;">'
                f'<h3 style="margin: 0; color: #924E7D;">Run: {_html.escape(run_title)}</h3></div>'
            )
            report.add_html(html=run_html, title=f"Run: {run_title}")

            # Find events file if not provided
            if events_file is None and raw_file:
                events_file = _find_events_tsv(Path(raw_file))

            _add_events_section(report, events_file)

            _add_raw_stage(
                report, raw_file, "Raw Data",
                qa_fn=lambda r: raw_qa(r, events_fpath=events_file, save_dir=None),
                add_kwargs=dict(title_prefix="Raw"),
            )
            _add_raw_stage(
                report, ref_file, "Referenced Data",
                qa_fn=lambda r: raw_qa(r, events_fpath=events_file, save_dir=None),
                add_kwargs=dict(title_prefix="Referenced"),
            )
            _add_raw_stage(
                report, filt_file, "Filtered Data (65-2000 Hz)",
                qa_fn=lambda r: raw_qa(r, events_fpath=events_file, save_dir=None),
                add_kwargs=dict(title_prefix="Filtered"),
            )
            _add_epoch_stage(report, epoch_file)

    # Save report
    report.save(report_fpath, overwrite=True, open_browser=False)
    print("\nReport saved successfully")

    return report_fpath


def _add_events_section(report, events_file):
    """Add events.tsv preview to the report, or a 'not found' note."""
    if not (events_file and os.path.exists(events_file)):
        report.add_html(
            html='<p style="color: #D55E00;"><em>⚠️ Events file not found</em></p>',
            title="Events",
        )
        return
    ev_df_full = pd.read_csv(events_file, sep="\t")
    n_events = len(ev_df_full)
    ev_df = ev_df_full.head(5)
    events_html = (
        f'<p><strong>Events file:</strong> <code>{Path(events_file).name}</code>'
        f' ({n_events} events)</p>'
        '<details><summary>Preview (first 5 events)</summary>'
        '<div style="overflow-x: auto; margin-top: 10px;">'
        f'{ev_df.to_html(index=False, border=0)}</div></details>'
    )
    report.add_html(html=events_html, title="Events")


def _add_raw_stage(report, fpath, label, qa_fn, add_kwargs):
    """Add a raw-data stage (raw / referenced / filtered) to the report."""
    if not (fpath and os.path.exists(fpath)):
        return
    print(f"    Adding {label.lower()}: {Path(fpath).name}")
    section_html = (
        '<div style="margin: 20px 0; padding: 10px; border-left: 3px solid #029E73;">'
        f'<h4 style="color: #029E73; margin-top: 0;">{label}</h4></div>'
    )
    report.add_html(html=section_html, title=f"{label} section")

    # preload=True so .pick() can drop channels (modern MNE requires data
    # to be in memory for channel-modification operations).
    raw = mne.io.read_raw_fif(fpath, preload=True, verbose=False)
    pick = "Cz" if "Cz" in raw.ch_names else raw.ch_names[0]
    raw_picked = raw.copy().pick([pick])
    report.add_raw(
        raw=raw_picked,
        title=f"{add_kwargs['title_prefix']} - {Path(fpath).name}",
        psd=True,
    )

    for fig, fig_title, caption in qa_fn(raw):
        report.add_figure(fig=fig, title=fig_title, caption=caption)
        plt.close(fig)


def _add_epoch_stage(report, fpath):
    """Add the epoched-data stage to the report."""
    if not (fpath and os.path.exists(fpath)):
        return
    print(f"    Adding epoched data: {Path(fpath).name}")
    section_html = (
        '<div style="margin: 20px 0; padding: 10px; border-left: 3px solid #029E73;">'
        '<h4 style="color: #029E73; margin-top: 0;">Epoched Data</h4></div>'
    )
    report.add_html(html=section_html, title="Epoched data section")

    # preload=True so .pick() can drop channels (modern MNE requires data
    # to be in memory for channel-modification operations).
    epochs = mne.read_epochs(fpath, preload=True, verbose=False)
    pick = "Cz" if "Cz" in epochs.ch_names else epochs.ch_names[0]
    epochs_picked = epochs.copy().pick([pick])
    report.add_epochs(
        epochs=epochs_picked,
        title=f"Epochs - {Path(fpath).name}",
    )

    for fig, fig_title, caption in epoch_qa(epochs, save_dir=None):
        report.add_figure(fig=fig, title=fig_title, caption=caption)
        plt.close(fig)


def _get_file(file_dict, task, run):
    """Helper to extract a single file from the file dictionary."""
    if not file_dict:
        return None

    task_data = file_dict.get(task, {})
    if not isinstance(task_data, dict):
        return None

    run_files = task_data.get(run, [])
    if run_files and len(run_files) > 0:
        filepath = str(run_files[0])
        print(f"    Found file for task={task}, run={run}: {Path(filepath).name}")
        return filepath

    print(f"    No file found for task={task}, run={run}")
    return None


def build_toc_html(sections):
    """
    Build a simple HTML table-of-contents.

    Parameters
    ----------
    sections : list
        List of section names.

    Returns
    -------
    toc_html : str
        HTML string for the table of contents.
    """
    parts = ['<div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0;">']
    parts.append("<h2>Contents</h2>")
    parts.append('<ul style="list-style-type: none; padding-left: 0;">')

    for i, sec in enumerate(sections):
        safe = _html.escape(sec)
        parts.append(
            f'<li style="padding: 5px 0;"><a href="#section-{i}"'
            f' style="color: #0173B2; text-decoration: none;">→ {safe}</a></li>'
        )

    parts.append("</ul>")
    parts.append("</div>")
    return "\n".join(parts)


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

    # QA FIG 2: Rejection statistics
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    n_good = len(epochs)
    n_rejected = len([log for log in epochs.drop_log if len(log) > 0])
    n_total = n_good + n_rejected

    pie_colors = [colors_cb["blue"], colors_cb["orange"]]
    axes[0].pie(
        [n_good, n_rejected],
        labels=["Accepted", "Rejected"],
        colors=pie_colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12, "fontweight": "bold"},
    )
    axes[0].set_title(f"Epoch Acceptance Rate\n({n_good}/{n_total} epochs)", fontsize=12, fontweight="bold")

    rejection_reasons = {}
    for log in epochs.drop_log:
        if len(log) > 0:
            reason = ", ".join(log)
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    if rejection_reasons:
        reasons = list(rejection_reasons.keys())
        counts = list(rejection_reasons.values())
        axes[1].barh(reasons, counts, color=colors_cb["orange"], edgecolor="black")
        axes[1].set_xlabel("Number of Epochs", fontsize=11)
        axes[1].set_title("Rejection Reasons", fontsize=12, fontweight="bold")
        axes[1].grid(True, alpha=0.3, axis="x")
    else:
        axes[1].text(
            0.5,
            0.5,
            "No epochs rejected!\n✓ All epochs passed QA",
            ha="center",
            va="center",
            fontsize=14,
            transform=axes[1].transAxes,
            color=colors_cb["cyan"],
            fontweight="bold",
        )
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1])
        axes[1].axis("off")

    plt.tight_layout()

    if save_dir:
        rej_path = os.path.join(save_dir, f"{prefix}_rejection.png")
        fig2.savefig(rej_path, dpi=150, bbox_inches="tight")

    figs.append((fig2, "Epoch Rejection Statistics", f"{n_rejected} of {n_total} epochs rejected"))

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


def _find_events_tsv(pth):
    """Find an events.tsv file in the same directory or parent directories."""
    parent = Path(pth).parent
    if not parent.is_dir():
        print(f"    Parent directory does not exist: {parent}")
        return None
    print(f"    Looking for events file in: {parent}")

    for f in parent.iterdir():
        if f.is_file() and "events" in f.name.lower() and f.suffix == ".tsv":
            print(f"    Found events file: {f.name}")
            return str(f)

    # Try going up to find BIDS events files (eeg folder -> subject folder)
    subject_dir = parent.parent
    if not subject_dir.is_dir():
        print(f"    Subject directory does not exist: {subject_dir}")
        return None
    print(f"    Looking for events file in: {subject_dir}")

    for f in subject_dir.rglob("*events*.tsv"):
        if _files_match(pth, f):
            print(f"    Found matching events file: {f.name}")
            return str(f)

    print(f"    No events file found for {pth.name}")
    return None


def _files_match(data_file, events_file):
    """Check if data file and events file belong to same recording."""
    data_name = Path(data_file).name
    events_name = Path(events_file).name

    # Extract task and run from both
    data_task = re.search(r"task-([^_]+)", data_name)
    events_task = re.search(r"task-([^_]+)", events_name)

    data_run = re.search(r"run-([^_]+)", data_name)
    events_run = re.search(r"run-([^_]+)", events_name)

    # Must match on task
    if data_task and events_task:
        if data_task.group(1) != events_task.group(1):
            return False

    # If both have run, must match
    if data_run and events_run:
        if data_run.group(1) != events_run.group(1):
            return False

    return True


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
