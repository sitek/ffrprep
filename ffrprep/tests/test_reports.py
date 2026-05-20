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


def test_build_epoch_section_includes_response_consistency_row(synthetic_epochs):
    """With >= 10 epochs, summary surfaces a mean trial-to-trial r row."""
    section = build_epoch_section(
        synthetic_epochs,
        section_id="epoched-active-1",
        title="Epoched - task-active run-1",
    )
    summary = section["summary"]
    consistency_key = "Mean trial-to-trial r"
    assert consistency_key in summary, (
        f"summary must include {consistency_key!r} when there are >= 10 "
        f"epochs; got keys {list(summary.keys())}"
    )
    # Pearson r is in [-1, 1]; the formatted string should parse to a
    # float in that range (allow some tolerance for the trailing format
    # like a leading sign or scientific notation).
    raw = summary[consistency_key]
    value = float(raw)
    assert -1.0 <= value <= 1.0, (
        f"trial-to-trial r value out of range: {raw}"
    )


@pytest.fixture
def synthetic_two_polarity_epochs():
    """Two Epochs objects, one per polarity, with matched sampling.

    Uses sfreq=8000 to match the FFR sampling rate range the
    PR-35 ``compute_phase_consistency`` was designed for: its
    default ``freqcap=2000`` allocates a ``(freqcap+1, …)``
    array indexed by ``[:freqcap+1, :]`` against an
    ``np.fft.fft(..., n=int(sfreq))`` output, so sfreq must be
    >= freqcap+1 to fill that slice.
    """
    n_channels = 1
    n_epochs = 5
    sfreq = 8000.0
    n_times = 8000  # 1 second of post-onset data
    rng_a = np.random.default_rng(101)
    rng_b = np.random.default_rng(102)
    data_a = rng_a.normal(0, 1e-6, size=(n_epochs, n_channels, n_times))
    data_b = rng_b.normal(0, 1e-6, size=(n_epochs, n_channels, n_times))
    info = mne.create_info(
        ch_names=["Cz"], sfreq=sfreq, ch_types=["eeg"],
    )
    epochs_a = mne.EpochsArray(data_a, info, tmin=-0.04, verbose=False)
    epochs_b = mne.EpochsArray(data_b, info, tmin=-0.04, verbose=False)
    return epochs_a, epochs_b


def test_build_phase_consistency_section_returns_summary_and_figure(
    synthetic_two_polarity_epochs,
):
    """build_phase_consistency_section pairs the PR-35 plot with a summary."""
    from ffrprep.reports import build_phase_consistency_section

    epochs_a, epochs_b = synthetic_two_polarity_epochs
    section = build_phase_consistency_section(
        epochs_a, epochs_b,
        section_id="phase-active-1",
        title="Phase Consistency - task-active run-1",
    )
    assert section["id"] == "phase-active-1"
    assert "Phase Consistency" in section["title"]
    # Summary surfaces the sweep count + significance threshold so
    # report readers can interpret the masked plot.
    summary = section["summary"]
    assert "Number of sweeps used" in summary
    assert "Significance threshold (alpha)" in summary
    # Exactly one masked-plot figure is embedded as a data URI.
    figures = section["figures"]
    assert len(figures) == 1
    assert figures[0]["data_uri"].startswith("data:image/png;base64,")


def test_build_phase_consistency_section_honors_alpha(
    synthetic_two_polarity_epochs,
):
    """An explicit ``alpha`` overrides the default significance level."""
    from ffrprep.reports import build_phase_consistency_section

    epochs_a, epochs_b = synthetic_two_polarity_epochs
    section = build_phase_consistency_section(
        epochs_a, epochs_b,
        section_id="phase-active-1",
        title="Phase Consistency",
        alpha=0.05,
    )
    assert section["summary"]["Significance threshold (alpha)"] == "0.05"


def test_build_evoked_section_accepts_extra_summary():
    """Caller-supplied extra_summary entries are folded into the summary table.

    Mirrors the same affordance build_epoch_section has so the CLI can
    pass through computed scalars (e.g. corr_stim_to_resp's peak r and
    lag) without having to monkey-patch the return dict.
    """
    n_channels = 1
    sfreq = 1000.0
    n_times = 100
    rng = np.random.default_rng(53)
    data = rng.normal(0, 1e-6, size=(n_channels, n_times))
    info = mne.create_info(
        ch_names=["Cz"], sfreq=sfreq, ch_types=["eeg"],
    )
    evoked = mne.EvokedArray(data, info, tmin=-0.04, verbose=False)
    evoked.baseline = (-0.04, 0.0)  # so RMS SNR row is present too

    from ffrprep.reports import build_evoked_section

    section = build_evoked_section(
        evoked,
        section_id="evoked-active-1-0-0",
        title="Evoked (positive)",
        label="Evoked",
        extra_summary={
            "Stim correlation (peak r)": "0.347",
            "Stim correlation (lag, ms)": "5.0",
        },
    )
    summary = section["summary"]
    assert summary.get("Stim correlation (peak r)") == "0.347"
    assert summary.get("Stim correlation (lag, ms)") == "5.0"


def test_build_evoked_section_extra_summary_overrides_defaults():
    """extra_summary keys can override values populated by the builder.

    Useful for cases where the caller has more accurate metadata than
    what's discoverable from the Evoked object alone (e.g. a corrected
    average count read from a sidecar).
    """
    n_channels = 1
    sfreq = 1000.0
    n_times = 100
    rng = np.random.default_rng(57)
    data = rng.normal(0, 1e-6, size=(n_channels, n_times))
    info = mne.create_info(
        ch_names=["Cz"], sfreq=sfreq, ch_types=["eeg"],
    )
    evoked = mne.EvokedArray(data, info, tmin=-0.04, verbose=False)

    from ffrprep.reports import build_evoked_section

    section = build_evoked_section(
        evoked,
        section_id="evoked-active-1-0-0",
        title="Evoked",
        extra_summary={"Channels": "OVERRIDDEN"},
    )
    assert section["summary"]["Channels"] == "OVERRIDDEN"


def test_build_epoch_section_skips_response_consistency_for_few_epochs():
    """With < 10 epochs, response_consistency is omitted to avoid noise.

    Pairwise Pearson r is unstable at low N; gating on epoch count
    keeps the report row honest.
    """
    n_channels = 4
    n_epochs = 5
    sfreq = 1000.0
    n_times = 100
    rng = np.random.default_rng(91)
    data = rng.normal(0, 1e-6, size=(n_epochs, n_channels, n_times))
    info = mne.create_info(
        ch_names=["Cz", "F3", "F4", "Pz"],
        sfreq=sfreq,
        ch_types=["eeg"] * n_channels,
    )
    epochs = mne.EpochsArray(data, info, tmin=-0.04, verbose=False)

    section = build_epoch_section(
        epochs,
        section_id="epoched-active-1",
        title="Epoched - task-active run-1",
    )
    assert "Mean trial-to-trial r" not in section["summary"], (
        "response_consistency must be omitted when len(epochs) < 10"
    )


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
