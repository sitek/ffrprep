"""
This module provides functions for creating, updating, and saving MNE reports.

For FFRPREP BIDS datasets, including support for figures, HTML blocks, and
special MNE objects.
"""

import os
import time
from mne import Report, open_report


def create_report(bids_root, out_dir=None, filename=None, title=None,
                  overwrite=False):
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
        Note: Uses .h5 extension for MNE compatibility.
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

    Examples
    --------
    Create an FFRPREP report in the default location
    with the default filename.

    >>> report_fpath = create)report(bids_root)
    """
    timenow = time.strftime("%Y-%m-%d_%Hh%Mm%Ss")

    # Define and create the output directory
    if out_dir is None:
        out_dir = os.path.join(bids_root, "derivatives")

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Define and create the output filename
    if filename is None:
        filename = f'ffrprep_report_{timenow}.hdf5'
    elif filename.endswith('.hdf5') is False:
        filename = f'{filename}.hdf5'

    report_fpath = os.path.join(out_dir, filename)
    if os.path.isfile(report_fpath):
        if overwrite:
            # Will overwrite the existing file
            pass
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

    # Create the report
    report = Report(title=title)
    # Save in HDF5 format for MNE compatibility
    if report_fpath.endswith(".html"):
        # If user specified .html, save as both .h5 and .html
        h5_path = report_fpath.replace(".html", ".h5")
        report.save(h5_path, overwrite=overwrite)
        # Also save HTML version for viewing
        report.save(report_fpath, overwrite=overwrite, open_browser=False)
        return h5_path  # Return .h5 path for add_to_report compatibility
    else:
        # Default: save as .h5 for MNE compatibility
        report.save(report_fpath, overwrite=overwrite)
        return report_fpath


def add_to_report(
    report_fpath,
    figure=None,
    figure_title=None,
    figure_caption=None,
    html_text=None,
    html_title=None,
    export_html=True,
):
    """
    Add a figure and/or html text to an existing FFRPREP report.

    Parameters
    ----------
    report_fpath : string
        Full filepath to the existing FFRPREP report.
    figure : matplotlib figure object
        Figure to be added to the existing FFRPREP report.
    figure_title : string
        Title of the figure to be added to the existing FFRPREP report.
    figure_caption : string
        Caption below the figure to be added to the existing FFRPREP report.
    html_text : string
        HTML-formatted text string to be added to the existing FFRPREP report.
    html_title : String
        Title of the HTML text to be added to the existing FFRPREP report.
    export_html : bool
        Whether to also export an HTML version of the report. Default True.

    Returns
    -------
    report_fpath : string
        Full filepath to the created report file.

    Examples
    --------
    Add an html text block to an existing FFRPREP report.

    >>> report_fpath = add_to_report(report_fpath, html_text=html_text,
                                     html_title='New section')
    """
    # Try to open existing report, fallback to creating new one if needed
    try:
        with open_report(report_fpath) as report:
            if figure:
                # Use correct signature for add_figure
                title = figure_title if figure_title is not None else "Figure"
                if figure_caption is not None:
                    report.add_figure(figure, title=title,
                                      caption=figure_caption)
                else:
                    report.add_figure(figure, title=title)
            if html_text:
                title_text = html_title or "Content"
                report.add_html(title=title_text, html=html_text)
            report.save(report_fpath, overwrite=True)

            # Also export HTML version if requested
            if export_html and report_fpath.endswith(".h5"):
                html_path = report_fpath.replace(".h5", ".html")
                report.save(html_path, overwrite=True, open_browser=False)

    except (OSError, ValueError, IOError) as e:
        # If opening fails, create a new report and add content
        print(f"Warning: Could not open existing report ({e}). "
              "Creating new report.")
        report = Report(title="Updated Report")
        if figure:
            # Use correct signature for add_figure
            title = figure_title if figure_title is not None else "Figure"
            if figure_caption is not None:
                report.add_figure(figure, title=title,
                                  caption=figure_caption)
            else:
                report.add_figure(figure, title=title)
        if html_text:
            title_text = html_title or "Content"
            report.add_html(title=title_text, html=html_text)
        report.save(report_fpath, overwrite=True)

        # Also export HTML version if requested
        if export_html and report_fpath.endswith(".h5"):
            html_path = report_fpath.replace(".h5", ".html")
            report.save(html_path, overwrite=True, open_browser=False)

    return report_fpath


def save_report(report_fpath, overwrite=True):
    """
    Save the existing FFRPREP report to an HTML file.

    Parameters
    ----------
    report_fpath : string
        Full filepath to the existing .hdf5 FFRPREP report.
    overwrite : boolean
        Whether to overwrite an existing HTML file (`True`).
        If `False` (default) and file exists, report will be saved as
        '{filename}_1.html'.

    Returns
    -------
    html_fpath : string
        Full filepath to the saved HTML report file.

    Examples
    --------
    Save an existing FFRPREP report to an HTML file.
    >>> html_fpath = save_report(report_fpath)
    """
    html_fpath = report_fpath.replace('.hdf5', '.html')
    if overwrite is False:
        if os.path.isfile(html_fpath):
            fname_base, fname_ext = os.path.splitext(html_fpath)
            new_filename = f'{fname_base}_1{fname_ext}'
            html_fpath = new_filename

    with open_report(report_fpath) as report:
        report.save(html_fpath, overwrite=True)
    return html_fpath
