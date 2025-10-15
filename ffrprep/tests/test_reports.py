"""Unit tests for the reporting functions in ffrprep.reports."""

import os.path as op
import matplotlib.pyplot as plt
from ffrprep.reports import create_report, add_to_report


def test_create_report(tmp_path):
    """Test the create_report function with various parameters and options."""
    # Create a report with defaults
    create_report(tmp_path)
    create_report(tmp_path, title='Test report')
    create_report(tmp_path, out_dir=op.join(tmp_path, 'test'))

    # User-defined output filename
    create_report(tmp_path, filename='test.hdf5')

    # Existing output filename, but don't overwrite
    create_report(tmp_path, filename='test.hdf5', overwrite=False)

    # Existing output filename, but do overwrite
    create_report(tmp_path, filename='test.hdf5', overwrite=True)

    # Replace .html extension with .hdf5
    report_fpath = create_report(tmp_path, filename='test.html')
    assert report_fpath.endswith('.hdf5')


def test_add_to_report(tmp_path):
    """Test the add_to_report function with various parameters and options."""
    # First, create a report
    report_fpath = create_report(tmp_path)

    # Open and close the existing report
    add_to_report(report_fpath)

    # Add figure to the existing report
    fig = plt.figure()
    add_to_report(figure=fig)
    add_to_report(figure=fig,
                  figure_title='test title')
    add_to_report(figure=fig,
                  figure_title='test title',
                  figure_caption='test caption')

    # Add HTML text to the existing report
    html_text = """
    <p>Ticking away the moments that make up a dull day</p>
    <ol>
    <li>You fritter and waste the hours in an offhand way</li>
    <li>Kicking around on a piece of ground in your hometown</li>
    </ol>
    <p>Waiting for someone or something to show you the way.</p>
    """
    add_to_report(html_text=html_text)
    add_to_report(html_text=html_text,
                  html_title='test title')

    # Add both a figure and HTML text
    add_to_report(figure=fig,
                  figure_title='test title',
                  html_text=html_text,
                  html_title='test title')

    # Test special cases with MNE objects
    from mne import create_info, EpochsArray, pick_types
    import numpy as np
    sfreq = 1000
    ch_names = ['Cz', 'Fz', 'Pz', 'Oz', 'EOG']
    ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eog']
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    data = np.random.randn(10, 5, 1000) * 1e-6
    events = np.array([[i, 0, 1] for i in range(10)])
    epochs = EpochsArray(data, info, events)
    epochs.pick(pick_types(epochs.info, eeg=True, eog=False, exclude=[]))

    add_to_report(report_fpath,
                  special_case='raw',
                  special_data=epochs.average().to_raw())

    add_to_report(report_fpath,
                  special_case='events',
                  special_data=events,
                  kwargs={'sfreq': sfreq,
                          'first_samp': 0,
                          'event_id': {'1': 1}})

    add_to_report(report_fpath,
                  special_case='epochs',
                  special_data=epochs)

    add_to_report(report_fpath,
                  special_case='evoked',
                  special_data=epochs.average())


def test_save_report(tmp_path):
    """Test the save_report function with various parameters and options."""
    from ffrprep.reports import save_report

    # First, create a report
    report_fpath = create_report(tmp_path)

    # Save the report to HTML, overwriting existing file
    html_fpath = save_report(report_fpath, overwrite=True)
    assert html_fpath.endswith('.html')

    # Save the report to HTML, not overwriting existing file
    html_fpath = save_report(report_fpath, overwrite=False)
    assert html_fpath.endswith('_1.html')
