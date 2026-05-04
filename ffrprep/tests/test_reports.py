"""Unit tests for the reporting functions in ffrprep.reports."""

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import mne
from ffrprep.reports import (
    build_subject_report,
    _fig_to_data_uri,
    build_raw_section,
    build_epoch_section,
)

# Use non-interactive backend for testing
matplotlib.use("Agg")


@pytest.fixture
def synthetic_raw():
    """A 4-channel, 5-second, 1 kHz Raw object suitable for QA helpers."""
    n_channels = 4
    sfreq = 1000.0
    n_times = int(sfreq * 5)
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1e-6, size=(n_channels, n_times))
    info = mne.create_info(
        ch_names=["Cz", "F3", "F4", "Pz"],
        sfreq=sfreq,
        ch_types=["eeg"] * n_channels,
    )
    return mne.io.RawArray(data, info, verbose=False)


@pytest.fixture
def synthetic_epochs():
    """A 10-epoch, 4-channel, 100-sample EpochsArray for QA helpers."""
    n_channels = 4
    n_epochs = 10
    sfreq = 1000.0
    n_times = 100
    rng = np.random.default_rng(43)
    data = rng.normal(0, 1e-6, size=(n_epochs, n_channels, n_times))
    info = mne.create_info(
        ch_names=["Cz", "F3", "F4", "Pz"],
        sfreq=sfreq,
        ch_types=["eeg"] * n_channels,
    )
    return mne.EpochsArray(data, info, tmin=-0.04, verbose=False)


def setup_function():
    """Clear any existing matplotlib figures before each test."""
    plt.close("all")


def teardown_function():
    """Clear matplotlib figures after each test."""
    plt.close("all")


def test_build_subject_report_writes_single_file_html(tmp_path):
    """Smoke test for the new build_subject_report API (Phase A skeleton).

    The new reporter renders a single-file HTML page from a Jinja2
    template. No HDF5 intermediate, no MNE.Report. The page shell must
    contain the subject identifier and a navigable structure even when
    the sections list is empty.
    """
    preproc_dir = tmp_path / "derivatives" / "ffrprep-preprocessing" / "sub-01" / "eeg"
    preproc_dir.mkdir(parents=True)

    out_path = build_subject_report(
        bids_root=str(tmp_path),
        subject="01",
        out_dir=str(preproc_dir),
        sections=[],
        title="ffrprep preprocessing report sub-01",
    )

    out = Path(out_path)
    assert out.exists()
    assert out.suffix == ".html"
    assert out.parent == preproc_dir

    html = out.read_text(encoding="utf-8")
    assert "<html" in html
    assert "</html>" in html
    assert "sub-01" in html
    # The page shell must include a TOC container even when sections is
    # empty, so the layout is stable and Phase B can append entries.
    assert 'id="toc"' in html or '<nav' in html


def test_build_subject_report_renders_section_titles(tmp_path):
    """Each entry in `sections` must produce a header in the output HTML.

    Phase A only requires structural rendering — no plots yet. Each
    section is a dict with at least a `title`; richer fields are added
    in Phase B (summary tables, plot gallery).
    """
    preproc_dir = tmp_path / "derivatives" / "ffrprep-preprocessing" / "sub-02" / "eeg"
    preproc_dir.mkdir(parents=True)

    sections = [
        {"id": "raw-active-1", "title": "Raw - task-active run-1"},
        {"id": "epoched-active-1", "title": "Epoched - task-active run-1"},
    ]

    out_path = build_subject_report(
        bids_root=str(tmp_path),
        subject="02",
        out_dir=str(preproc_dir),
        sections=sections,
        title="ffrprep preprocessing report sub-02",
    )

    html = Path(out_path).read_text(encoding="utf-8")
    for section in sections:
        assert section["title"] in html
        # Anchor target so the TOC link can scroll to the section.
        assert f'id="{section["id"]}"' in html


def test_fig_to_data_uri_returns_png_data_uri():
    """The fig→data-URI helper must produce an inline-embeddable PNG."""
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1, 2], [0, 1, 0])
    uri = _fig_to_data_uri(fig)
    plt.close(fig)
    assert uri.startswith("data:image/png;base64,")
    # A real (even tiny) PNG is at least a few hundred base64 chars.
    assert len(uri) > 300


def test_build_raw_section_returns_summary_and_figures(synthetic_raw):
    """build_raw_section wraps raw_qa and produces an embeddable section dict."""
    section = build_raw_section(
        synthetic_raw,
        section_id="raw-active-1",
        title="Raw - task-active run-1",
    )
    assert section["id"] == "raw-active-1"
    assert section["title"] == "Raw - task-active run-1"

    summary = section["summary"]
    assert isinstance(summary, dict)
    summary_blob = " ".join(str(v) for v in summary.values())
    assert "1000" in summary_blob  # sfreq
    assert "4" in summary_blob     # n_channels

    figures = section["figures"]
    assert len(figures) >= 2  # raw_qa returns waveform + PSD at minimum
    for f in figures:
        assert "title" in f
        assert "data_uri" in f
        assert f["data_uri"].startswith("data:image/png;base64,")


def test_build_epoch_section_returns_summary_and_figures(synthetic_epochs):
    """build_epoch_section wraps epoch_qa and embeds figures as data URIs."""
    section = build_epoch_section(
        synthetic_epochs,
        section_id="epoched-active-1",
        title="Epoched - task-active run-1",
    )
    assert section["id"] == "epoched-active-1"
    summary = section["summary"]
    assert isinstance(summary, dict)
    summary_blob = " ".join(str(v) for v in summary.values())
    assert "10" in summary_blob  # n_epochs
    assert "4" in summary_blob   # n_channels

    figures = section["figures"]
    assert len(figures) >= 1
    for f in figures:
        assert f["data_uri"].startswith("data:image/png;base64,")


def test_build_subject_report_renders_summary_and_figures(tmp_path):
    """Sections with summary + figures render a table and embedded plots."""
    out_dir = tmp_path / "derivatives" / "ffrprep-preprocessing" / "sub-01" / "eeg"
    out_dir.mkdir(parents=True)

    sections = [
        {
            "id": "raw-active-1",
            "title": "Raw - task-active run-1",
            "summary": {
                "Sampling rate": "1000.0 Hz",
                "Duration": "5.00 s",
                "Channels": "4",
            },
            "figures": [
                {
                    "title": "Waveform",
                    "caption": "First 5 seconds",
                    "data_uri": (
                        "data:image/png;base64,"
                        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA"
                    ),
                },
            ],
        },
    ]

    out_path = build_subject_report(
        bids_root=str(tmp_path),
        subject="01",
        out_dir=str(out_dir),
        sections=sections,
    )
    html = Path(out_path).read_text(encoding="utf-8")
    # Summary key + value should both appear in the rendered page.
    assert "Sampling rate" in html
    assert "1000.0 Hz" in html
    # Figure title + caption + the data URI itself.
    assert "Waveform" in html
    assert "First 5 seconds" in html
    assert "data:image/png;base64,iVBORw" in html
