"""Unit tests for the reporting functions in ffrprep.reports."""

import os.path as op
import matplotlib
import matplotlib.pyplot as plt
from ffrprep.reports import create_report, add_to_report

# Use non-interactive backend for testing
matplotlib.use("Agg")


def setup_function():
    """Clear any existing matplotlib figures before each test."""
    plt.close("all")


def teardown_function():
    """Clear matplotlib figures after each test."""
    plt.close("all")


def test_create_report(tmp_path):
    # Create a report with defaults (unique filename)
    report1_path = create_report(tmp_path, filename="report1.h5")
    assert op.exists(report1_path)

    # Create another report with different title and filename
    report2_path = create_report(tmp_path, filename="report2.h5",
                                 title="Test report")
    assert op.exists(report2_path)

    # Create report in subdirectory
    subdir_path = create_report(tmp_path, out_dir=op.join(tmp_path, "test"))
    assert op.exists(subdir_path)

    # User-defined output filename (new file)
    test_path = create_report(tmp_path, filename="test.h5")
    assert op.exists(test_path)

    # Existing filename, don't overwrite (should create test.h5_1.hdf5)
    test_path_no_overwrite = create_report(tmp_path, filename="test.h5",
                                           overwrite=False)
    assert op.exists(test_path_no_overwrite)
    assert "test.h5_1.hdf5" in test_path_no_overwrite

    # Existing filename, do overwrite (should overwrite test.h5)
    test_path_overwrite = create_report(tmp_path, filename="test.h5",
                                        overwrite=True)
    assert op.exists(test_path_overwrite)
    assert test_path_overwrite == test_path


def test_add_to_report(tmp_path):
    """Test the add_to_report function with various parameters and options."""
    # First, create a report
    report_fpath = create_report(tmp_path, filename="add_test.h5")
    assert op.exists(report_fpath)

    # Open and close the existing report (should not fail)
    result_path = add_to_report(report_fpath)
    assert result_path == report_fpath

    # Add figure to the existing report
    fig = plt.figure(figsize=(5, 4))
    plt.plot([1, 2, 3], [1, 4, 2])

    add_to_report(report_fpath, figure=fig)
    add_to_report(report_fpath, figure=fig, figure_title="test title")
    add_to_report(report_fpath, figure=fig, figure_title="test title",
                  figure_caption="test caption")

    # Add HTML text to the existing report
    html_text = """
    <p>Testing if text is added to an existing report.</p>
    <ol>
    <li>Hopefully this works.</li>
    <li>Automated reports are pretty cool.</li>
    </ol>
    <p>This is the final line of the test.</p>
    """
    add_to_report(report_fpath, html_text=html_text)
    add_to_report(report_fpath, html_text=html_text, html_title="test title")

    # Add both a figure and HTML text
    add_to_report(
        report_fpath, figure=fig, figure_title="test title",
        html_text=html_text, html_title="test title"
    )

    # Close the figure to clean up
    plt.close(fig)
