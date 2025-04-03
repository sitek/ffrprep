import argparse
import os
from pathlib import Path

from ffrprep.utils import validate_input_dir
from ._version import get_versions


# Define parser to collect required inputs
def get_parser():
    """Create and return an argument parser."""

    # get version
    __version__ = get_versions()['version']

    # define parser description
    parser = argparse.ArgumentParser(
        description='A standardized and robust processing pipeline '
                    'for FFR data.')
    # add parser argument for version
    parser.add_argument(
        '-v', '--version',
        action='version',
        version='ffrprep version {}'.format(__version__)
    )
    # add parser argument for BIDS directory
    parser.add_argument('bids_dir', action='store', type=Path,
                        help='The directory with the input dataset '
                        'formatted according to the BIDS standard.')
    # add parser argument for participant label
    parser.add_argument('--participant_label',
                        help=('The label(s) of the participant(s) that should '
                              'be preprocessed. The label corresponds to '
                              'sub-<participant_label> from the BIDS spec '
                              '(so it does not include "sub-"). '
                              'Multiple participants can be specified with '
                              'a space-separated list.'),
                        nargs="+")
    # add parser argument for BIDS validation
    parser.add_argument('--skip_bids_validation', default=False,
                        help='Assume the input dataset is BIDS compliant \
                          and skip the validation (default: False).',
                        action="store_true")
    return parser


# Define the CLI
def run_ffrprep():
    """Run the FFR processing pipeline."""
    # Get arguments from parser
    args = get_parser().parse_args()

    # set computational environment
    if os.getenv('IS_DOCKER'):
        exec_env = 'singularity'
        cgroup = Path('/proc/1/cgroup')
        if cgroup.exists() and 'docker' in cgroup.read_text():
            exec_env = 'docker'
    else:
        exec_env = 'local'

    # Validate input data
    if args.skip_bids_validation:
        print("Input data will not be checked for BIDS compliance.")
    else:
        print("Making sure the input data is BIDS compliant "
              "(warnings can be ignored in most cases).")
        validate_input_dir(exec_env, args.bids_dir, args.participant_label)


# Run the CLI
if __name__ == "__main__":
    run_ffrprep()
