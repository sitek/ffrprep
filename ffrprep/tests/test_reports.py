import shutil
import pytest
import mne
import os.path as op
import matplotlib.pyplot as plt
from ffrprep.reports import create_report, add_to_report


def test_create_report(tmp_path):
    # Create a report with defaults
    create_report(tmp_path)
    create_report(tmp_path, title='Test report')
    create_report(tmp_path, out_dir=op.join(tmp_path, 'test'))
    
    # User-defined output filename
    create_report(tmp_path, filename='test.html')
    
    # Existing output filename, but don't overwrite
    create_report(tmp_path, filename='test.html', overwrite=False)

    # Existing output filename, but do overwrite
    create_report(tmp_path, filename='test.html', overwrite=True)
    
def test_add_to_report(tmp_path):
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
    html_text= """
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
