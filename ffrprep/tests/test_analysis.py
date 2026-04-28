"""Analysis-module tests using the shared ``bids_dataset`` fixture.

All tests run sub-21 / task-passive / run-1 through the load → epoch →
evoke chain, then exercise one analysis function each. Reusing the
session-scoped fixture avoids per-test re-downloads.
"""
import pytest

from ffrprep.preproc import epoch_data, load_data, make_evoked
from ffrprep.analysis import (
    autocorrelation,
    compute_pitch_and_conf,
    compute_power,
    plot_pitch_and_conf,
    rms_snr,
)


pytestmark = pytest.mark.integration


@pytest.fixture
def evoked_sub21_passive(bids_dataset):
    """Evoked response for sub-21 / task-passive / run-1.

    Built once per test (function-scoped) so each analysis function gets
    a fresh object — the underlying raw data is cached by ``bids_dataset``.

    Note: ``load_data`` returns ``(raw, bids_path, original_filename,
    events_file)`` and ``epoch_data`` returns ``(epochs, time_window)``,
    so both tuples need unpacking.
    """
    raw, _bids_path, _orig_name, _events_file = load_data(
        bids_root=bids_dataset,
        sub_label="21",
        session_label=None,
        task_label="passive",
        run_label=1,
    )
    epochs, _time_window = epoch_data(raw, baseline=[-0.05, 0.2])
    # by_event_type=False returns a single mne.Evoked; True returns a dict
    # keyed by event name, which the analysis functions can't consume.
    return make_evoked(epochs=epochs, by_event_type=False)


def test_compute_power(evoked_sub21_passive):
    compute_power(evoked_sub21_passive)


def test_rms_snr(evoked_sub21_passive):
    rms_snr(evoked_sub21_passive)


def test_autocorrelation(evoked_sub21_passive):
    autocorrelation(evoked_sub21_passive)


def test_compute_pitch_and_conf(evoked_sub21_passive):
    compute_pitch_and_conf(evoked_sub21_passive)


def test_plot_pitch_and_conf(evoked_sub21_passive):
    results = compute_pitch_and_conf(evoked_sub21_passive)
    plot_pitch_and_conf(results)
