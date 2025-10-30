import argparse
import os
from pathlib import Path

from ffrprep.utils import validate_input_dir
from ffrprep.preproc import (
    create_preprocessing_workflow,
    create_analysis_workflow,
    get_participants,
    setup_derivatives_directories,
    check_preprocessing_exists,
)
from ._version import get_versions


# Define parser to collect required inputs
def get_parser():
    """Create and return an argument parser for BIDS-App."""

    # get version
    __version__ = get_versions()["version"]

    # define parser description
    parser = argparse.ArgumentParser(
        description=("ffrprep: A BIDS-App for standardized FFR preprocessing "
                     "and analysis"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # add parser argument for version
    parser.add_argument("-v", "--version", action="version",
                        version="ffrprep version {}".format(__version__))

    # BIDS-App standard arguments
    parser.add_argument(
        "bids_dir",
        action="store",
        type=Path,
        help=("The directory with the input dataset "
              "formatted according to the BIDS standard."),
    )
    parser.add_argument(
        "output_dir",
        action="store",
        type=Path,
        help="The directory where the output files "
        "should be stored. If you are running group level analysis "
        "this folder should be prepopulated with the results of the "
        "participant level analysis.",
    )
    parser.add_argument(
        "analysis_level",
        choices=["participant", "group"],
        help="Level of the analysis that will be performed. "
        "Multiple participant level analyses can be run independently "
        "(in parallel) using the same output_dir.",
    )

    # Participant selection
    parser.add_argument(
        "--participant_label",
        help=(
            "The label(s) of the participant(s) that should "
            "be processed. The label corresponds to "
            "sub-<participant_label> from the BIDS spec "
            '(so it does not include "sub-"). '
            "Multiple participants can be specified with "
            "a space-separated list."
        ),
        nargs="+",
    )

    # Processing stage control
    parser.add_argument(
        "--stage",
        choices=["preprocessing", "analysis", "both"],
        default="both",
        help=("Processing stage to run: preprocessing only, "
              "analysis only, or both stages sequentially."),
    )

    # Preprocessing parameters
    preproc_group = parser.add_argument_group("Preprocessing options")
    preproc_group.add_argument(
        "--ref_channels",
        help="Reference channel(s) for re-referencing. "
        'Can be "average" for average reference, a single '
        "channel name, or comma-separated list of channels.",
        default="average",
    )
    preproc_group.add_argument(
        "--high_pass", type=float,
        help="High-pass filter cutoff frequency in Hz.", default=1.0
    )
    preproc_group.add_argument(
        "--low_pass", type=float,
        help="Low-pass filter cutoff frequency in Hz.", default=40.0
    )
    preproc_group.add_argument(
        "--baseline",
        help=("Baseline correction period. Format: start,end "
              '(e.g., "-0.2,0" for -200ms to 0ms).'),
        default="-0.2,0",
    )
    preproc_group.add_argument(
        "--tmin", type=float,
        help="Start time of epochs relative to event onset (s).", default=-0.2
    )
    preproc_group.add_argument(
        "--tmax", type=float,
        help="End time of epochs relative to event onset (s).", default=0.6
    )

    # Analysis parameters
    analysis_group = parser.add_argument_group("Analysis options")
    analysis_group.add_argument(
        "--by_event_type", action="store_true", help=(
            "Create separate evoked responses for each event type.")
    )

    # General options
    parser.add_argument(
        "--skip_bids_validation",
        action="store_true",
        help=("Assume the input dataset is BIDS compliant "
              "and skip the validation."),
    )
    parser.add_argument(
        "--n_procs", type=int, default=1, help=(
            "Number of processors to use for parallel execution.")
    )
    parser.add_argument("--work_dir",
                        type=Path,
                        help="Path where intermediate results should be "
                             "stored.")

    return parser


def parse_baseline(baseline_str):
    """Parse baseline string to tuple of floats."""
    if "," in baseline_str:
        start, end = baseline_str.split(",")
        return float(start), float(end)
    else:
        return float(baseline_str), 0.0


def parse_ref_channels(ref_str):
    """Parse reference channels string."""
    if ref_str.lower() == "average":
        return None  # Average reference
    elif "," in ref_str:
        return ref_str.split(",")  # List of channels
    else:
        return ref_str  # Single channel


def run_ffrprep():
    """Run the FFR processing pipeline as a BIDS-App."""
    # Get arguments from parser
    args = get_parser().parse_args()

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set computational environment
    if os.getenv("IS_DOCKER"):
        exec_env = "singularity"
        cgroup = Path("/proc/1/cgroup")
        if cgroup.exists() and "docker" in cgroup.read_text():
            exec_env = "docker"
    else:
        exec_env = "local"

    # Validate input data
    if args.skip_bids_validation:
        print("Input data will not be checked for BIDS compliance.")
    else:
        print("Making sure the input data is BIDS compliant "
              "(warnings can be ignored in most cases).")
        validate_input_dir(exec_env, args.bids_dir, args.participant_label)

    # Only run participant-level analysis for now
    if args.analysis_level != "participant":
        print("Currently only participant-level analysis is supported.")
        return

    # Parse processing parameters
    baseline = parse_baseline(args.baseline)
    ref_channels = parse_ref_channels(args.ref_channels)

    # Get participant labels using pybids for robust querying
    print("Discovering participants using pybids...")
    subjects = get_participants(str(args.bids_dir), args.participant_label)

    if not subjects:
        if args.participant_label:
            print(f"No participants found matching {args.participant_label}")
        else:
            print("No participants with EEG data found in the dataset")
        return

    print(f"Processing subjects: {subjects}")

    # Process each subject
    for subject in subjects:
        print(f"Processing subject: sub-{subject}")

        # Set up derivatives directories within the BIDS dataset
        derivatives_info = setup_derivatives_directories(
            args.bids_dir,
            subject,
            create_preprocessing=args.stage in ["preprocessing", "both"],
            create_analysis=args.stage in ["analysis", "both"],
        )

        print(
            f"Derivatives will be stored in: "
            f"{derivatives_info['derivatives_root']}"
        )

        # Create workflow based on stage
        if args.stage in ["preprocessing", "both"]:
            print("Running preprocessing workflow...")

            # Create preprocessing workflow
            preproc_wf = create_preprocessing_workflow()

            # Set working directory for nipype
            if args.work_dir:
                work_dir = args.work_dir / f"sub-{subject}" / "preprocessing"
            else:
                work_dir = (derivatives_info["preprocessing_dir"] / "work" /
                            f"sub-{subject}")

            preproc_wf.base_dir = str(work_dir)

            # Set inputs
            preproc_wf.inputs.inputnode.bids_root = str(args.bids_dir)
            preproc_wf.inputs.inputnode.sub_label = subject
            preproc_wf.inputs.inputnode.ref_channels = ref_channels
            preproc_wf.inputs.inputnode.high_pass = args.high_pass
            preproc_wf.inputs.inputnode.low_pass = args.low_pass
            preproc_wf.inputs.inputnode.baseline = baseline
            preproc_wf.inputs.inputnode.tmin = args.tmin
            preproc_wf.inputs.inputnode.tmax = args.tmax

            # Set output directory for results
            preproc_wf.inputs.inputnode.output_dir = str(
                    derivatives_info["preprocessing_subject_dir"])

            # Run preprocessing workflow
            preproc_wf.run()
            print(
                f"Preprocessing completed. Outputs saved to: "
                f"{derivatives_info['preprocessing_subject_dir']}"
            )

        if args.stage in ["analysis", "both"]:
            print("Running analysis workflow...")

            # Check if preprocessing outputs exist (required for analysis)
            if args.stage == "analysis":  # Analysis-only mode
                preproc_exists, preproc_files = check_preprocessing_exists(
                        args.bids_dir, subject)

                if not preproc_exists:
                    print(
                        f"ERROR: No preprocessing outputs found for "
                        f"subject {subject}."
                    )
                    print(
                        f"Expected location: {args.bids_dir}/derivatives/"
                        f"ffrprep-preprocessing/sub-{subject}/"
                    )
                    print("Please run preprocessing stage first or use "
                          "'both' stage.")
                    continue
                else:
                    print(
                        f"Found preprocessing outputs: "
                        f"{[f.name for f in preproc_files]}"
                    )

            # Create analysis workflow
            analysis_wf = create_analysis_workflow()

            # Set working directory for nipype
            if args.work_dir:
                work_dir = args.work_dir / f"sub-{subject}" / "analysis"
            else:
                work_dir = (derivatives_info["analysis_dir"] / "work" /
                            f"sub-{subject}")

            analysis_wf.base_dir = str(work_dir)

            # Set inputs
            analysis_wf.inputs.inputnode.by_event_type = args.by_event_type
            analysis_wf.inputs.inputnode.bids_root = str(args.bids_dir)
            analysis_wf.inputs.inputnode.subject = subject

            # Set output directory for analysis results
            analysis_wf.inputs.inputnode.output_dir = str(
                derivatives_info["analysis_subject_dir"])

            # Run analysis workflow
            analysis_wf.run()
            print(
                f"Analysis completed. Outputs saved to: "
                f"{derivatives_info['analysis_subject_dir']}"
            )

    print("ffrprep processing completed successfully!")


if __name__ == "__main__":
    run_ffrprep()
