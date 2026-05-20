"""Command-line interface for downloading ffrprep datasets."""
import argparse
from pathlib import Path

from importlib.metadata import version as _pkg_version
from ffrprep.datasets import (
    download_epoch_data,
    download_example_data,
    download_raw_data,
)


def _parse_subjects(values):
    """Coerce --subjects values to int (count) or list (specific IDs).

    A bare integer like '3' is treated as a count of subjects to fetch.
    A zero-padded value like '03' (or any non-numeric token) is treated
    as a BIDS subject identifier and kept inside a list, so it is not
    silently misrouted as "the first 3 subjects".
    """
    if values is None:
        return 1
    if len(values) == 1:
        single = values[0]
        if single.isdigit() and not single.startswith("0"):
            return int(single)
        return [single]
    return list(values)


def _add_out_argument(parser):
    """Attach the shared ``--out`` destination-directory option to `parser`."""
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination directory (defaults to current working dir).",
    )


def _add_subjects_argument(parser):
    """Attach the shared ``--subjects`` selector option to `parser`."""
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="A count (e.g. 3) or specific subject IDs (e.g. 03 21).",
    )


def _add_with_stimuli_argument(parser):
    """Attach the shared ``--with-stimuli`` opt-in flag to `parser`.

    BooleanOptionalAction so users get both ``--with-stimuli`` and the
    explicit ``--no-with-stimuli`` inverse; default is False so a
    plain ``ffrprep-download example`` keeps today's behavior.
    """
    parser.add_argument(
        "--with-stimuli",
        dest="with_stimuli",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Additionally download the BIDS /stimuli/ directory needed "
            "by stimulus-aware analyses (e.g. corr_stim_to_resp)."
        ),
    )


def get_parser():
    """Create the argument parser for ffrprep-download."""
    __version__ = _pkg_version("ffrprep")

    parser = argparse.ArgumentParser(
        prog="ffrprep-download",
        description="Download FFR datasets distributed via OSF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="ffrprep-download version {}".format(__version__),
    )

    subparsers = parser.add_subparsers(dest="dataset", required=True)

    example_parser = subparsers.add_parser(
        "example",
        help="Download the single-subject example dataset.",
    )
    _add_out_argument(example_parser)
    _add_with_stimuli_argument(example_parser)

    raw_parser = subparsers.add_parser(
        "raw",
        help="Download raw EEG data for one or more subjects.",
    )
    _add_subjects_argument(raw_parser)
    _add_out_argument(raw_parser)

    epoch_parser = subparsers.add_parser(
        "epoch",
        help="Download epoched EEG data for one or more subjects.",
    )
    _add_subjects_argument(epoch_parser)
    _add_out_argument(epoch_parser)

    return parser


def _dispatch_example(args):
    """Run the ``ffrprep-download example`` subcommand."""
    return download_example_data(
        dataset_path=args.out,
        with_stimuli=args.with_stimuli,
    )


def _dispatch_raw(args):
    """Run the ``ffrprep-download raw`` subcommand."""
    return download_raw_data(
        subjects=_parse_subjects(args.subjects),
        dataset_path=args.out,
    )


def _dispatch_epoch(args):
    """Run the ``ffrprep-download epoch`` subcommand."""
    return download_epoch_data(
        subjects=_parse_subjects(args.subjects),
        dataset_path=args.out,
    )


_DISPATCH = {
    "example": _dispatch_example,
    "raw": _dispatch_raw,
    "epoch": _dispatch_epoch,
}


def run_download(argv=None):
    """Entry point for the ffrprep-download console script."""
    args = get_parser().parse_args(argv)
    return _DISPATCH[args.dataset](args)
