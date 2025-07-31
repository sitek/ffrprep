import os
import time
from mne import Report, open_report


def create_report(bids_root,
                  out_dir=None,
                  filename=None,
                  title=None,
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
        will write the report as 'ffrprep_report_{YYYY:MM:DD:HH:MM:SS}.html'.
    title : string
        Title for the report. If `None` (default),
        will write the title as 'FFRPREP report {YYYY:MM:DD:HH:MM:SS}'.
    overwrite : boolean
        Whether to overwrite an existing file (`True`).
        If `False` (default) and file exists, report will be saved as
        '{filename}_1.html'.

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
        out_dir = os.path.join(bids_root, 'derivatives')

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Define and create the output filename
    if filename is None:
        filename = f'ffrprep_report_{timenow}.html'

    if os.path.isfile(os.path.join(out_dir, filename)):
        if overwrite:
            report_fpath = os.path.join(out_dir, filename)
        else:
            fname_base, fname_ext = os.path.splitext(filename)
            new_filename = f'{fname_base}_1{fname_ext}'
            report_fpath = os.path.join(out_dir, new_filename)
    else:
        report_fpath = os.path.join(out_dir, filename)

    # Define the report title, if None
    if title is None:
        timenow = time.strftime("%Y-%m-%d_%Hh%Mm%Ss")
        filename = f'FFRPREP report {timenow}'

    # Create the report
    report = Report(title=title)
    report.save(report_fpath)

    return report_fpath


def add_to_report(report_fpath,
                  figure=None,
                  figure_title=None,
                  figure_caption=None,
                  html_text=None,
                  html_title=None):
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
    with open_report(report_fpath) as report:
        if figure:
            report.add_figure(figure=figure,
                              title=figure_title,
                              caption=figure_caption)
        if html_text:
            report.add_html(title=html_title, html=html_text)
        report.save(report_fpath, overwrite=True)

    return report_fpath
