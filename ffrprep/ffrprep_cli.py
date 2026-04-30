import argparse
import logging
import os
import sys
import time
from pathlib import Path

from ffrprep.utils import validate_input_dir
from ffrprep.preproc import (
    create_preprocessing_workflow,
    create_analysis_workflow,
    get_participants,
    get_sessions_tasks_runs,
    setup_derivatives_directories,
)
import ffrprep.reports as reports
from importlib.metadata import version as _pkg_version
import re


def _propagate_run_provenance(preproc_file, analysis_dir):
    """Copy ConcatenatedRuns / Run from a preproc sidecar to its analysis siblings.

    The analysis stage only sees the preprocessing .fif filename; for
    concatenated-run inputs the filename has no ``_run-N`` token so the
    downstream save can't know it represents multiple runs. This helper
    reads the preproc sidecar and writes the same provenance fields into
    every analysis sidecar that came from this preproc file.
    """
    import json

    preproc_sidecar = preproc_file.with_suffix(".json")
    if not preproc_sidecar.exists():
        return
    with open(preproc_sidecar) as f:
        preproc_meta = json.load(f)
    fields = {}
    if "ConcatenatedRuns" in preproc_meta:
        fields["ConcatenatedRuns"] = preproc_meta["ConcatenatedRuns"]
    if "Run" in preproc_meta and "Run" not in fields:
        fields["Run"] = preproc_meta["Run"]
    if not fields:
        return

    # The analysis sidecar(s) for this preproc file share its base stem
    # minus the desc-preproc_epo suffix.
    base_stem = preproc_file.stem.replace("_desc-preproc_epo", "")
    for analysis_sidecar in analysis_dir.glob(f"{base_stem}_desc-*.json"):
        with open(analysis_sidecar) as f:
            data = json.load(f)
        data.update(fields)
        with open(analysis_sidecar, "w") as f:
            json.dump(data, f, indent=2)


def _setup_subject_log(derivatives_info, subject, stage):
    """Tee stdout-style run information into a per-subject log file.

    Creates ``sub-<id>_<descriptor>.log`` next to the subject's outputs
    for the requested stage. The file captures the CLI invocation,
    ffrprep version, timestamp, and any subsequent log messages emitted
    via the root or nipype loggers. A handler is attached per call so
    each subject ends up with its own log file.

    For "preprocessing" or "both" stages, the log lives in the
    preprocessing subject dir; for "analysis", in the analysis subject
    dir. The descriptor in the filename matches.
    """
    if stage in ("preprocessing", "both"):
        log_dir = derivatives_info.get("preprocessing_subject_dir")
        descriptor = "preprocessing"
    else:
        log_dir = derivatives_info.get("analysis_subject_dir")
        descriptor = "analysis"
    if log_dir is None:
        return
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"sub-{subject}_{descriptor}.log"

    # Clean up the legacy log filename (pre-descriptor naming) if present.
    legacy_log_path = log_dir / f"sub-{subject}_ffrprep.log"
    if legacy_log_path.exists() and legacy_log_path != log_path:
        legacy_log_path.unlink()

    # Detach any per-subject handler left over from a previous subject in
    # the same process (otherwise its log file keeps capturing this
    # subject's messages too — log bleed between subjects).
    root_logger = logging.getLogger()
    for existing in list(root_logger.handlers):
        if getattr(existing, "_ffrprep_subject_handler", False):
            root_logger.removeHandler(existing)
            existing.close()

    handler = logging.FileHandler(log_path, mode="w")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    )
    # Tag so the next call can find and remove it.
    handler._ffrprep_subject_handler = True

    # Attach the handler only to the root logger. nipype's logger
    # propagates to root by default, so the messages get captured exactly
    # once. (Adding the handler to both root and nipype causes duplicate
    # log lines.)
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)

    root_logger.info("ffrprep version: %s", _pkg_version("ffrprep"))
    root_logger.info("invocation: %s", " ".join(sys.argv))
    root_logger.info("started at: %s", time.strftime("%Y-%m-%d %H:%M:%S"))
    root_logger.info("derivatives root: %s", derivatives_info.get("derivatives_root"))
    print(f"Run log: {log_path}")


# Define parser to collect required inputs
def get_parser():
    """Create and return an argument parser for BIDS-App."""

    # get version
    __version__ = _pkg_version("ffrprep")

    # define parser description
    parser = argparse.ArgumentParser(
        description=("ffrprep: A BIDS-App for standardized FFR preprocessing " "and analysis"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # add parser argument for version
    parser.add_argument("-v", "--version", action="version", version="ffrprep version {}".format(__version__))

    # BIDS-App standard arguments
    parser.add_argument(
        "bids_dir",
        action="store",
        type=Path,
        help=("The directory with the input dataset " "formatted according to the BIDS standard."),
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
        help=("Processing stage to run: preprocessing only, " "analysis only, or both stages sequentially."),
    )

    # Preprocessing parameters
    preproc_group = parser.add_argument_group("Preprocessing options")
    preproc_group.add_argument(
        "--ref_channels",
        help=(
            "Reference channel(s) for re-referencing. Provide space-separated "
            "channel names (e.g. --ref_channels M1 M2). Use 'average' for "
            "average reference. Comma-separated single-argument styles are "
            "also accepted for backward compatibility (e.g. 'M1,M2')."
        ),
        nargs="+",
        type=str,
        default=None,
    )
    preproc_group.add_argument(
        "--high_pass", type=float, help="High-pass filter cutoff frequency in Hz.", default=1.0
    )
    preproc_group.add_argument(
        "--low_pass", type=float, help="Low-pass filter cutoff frequency in Hz.", default=40.0
    )
    # New MNE-style arguments (l_freq/h_freq) to make mapping explicit.
    # These override the legacy --high_pass/--low_pass when provided.
    preproc_group.add_argument(
        "--l_freq",
        type=float,
        help=(
            "Lower-pass edge in Hz (MNE name: l_freq). "
            "If provided, overrides --high_pass. Use 'None' via --no-filter to disable."
        ),
        default=None,
    )
    preproc_group.add_argument(
        "--h_freq",
        type=float,
        help=(
            "Upper-pass edge in Hz (MNE name: h_freq). "
            "If provided, overrides --low_pass. Use 'None' via --no-filter to disable."
        ),
        default=None,
    )
    preproc_group.add_argument(
        "--no-filter",
        action="store_true",
        help=(
            "Disable filtering entirely (equivalent to l_freq=None and h_freq=None). "
            "When set, any provided l_freq/h_freq/high_pass/low_pass are ignored."
        ),
        default=False,
    )
    preproc_group.add_argument(
        "--baseline",
        help=(
            "Baseline correction period. Provide two numbers: START END "
            "in seconds (e.g. --baseline -0.2 0 for -200ms to 0ms)."
        ),
        nargs=2,
        type=float,
        metavar=("START", "END"),
        default=[-0.2, 0.0],
    )
    preproc_group.add_argument(
        "--picks",
        help=(
            "Channels to include in epochs. Comma-separated list or single "
            "channel name (e.g. 'Cz' or 'Cz,Fz'). If not provided, all "
            "channels are considered."
        ),
        default=None,
    )
    preproc_group.add_argument(
        "--on-missing",
        choices=["warn", "raise", "ignore"],
        default="warn",
        help=(
            "Behavior when events referenced by event_id are missing: "
            "warn, raise, or ignore. Matches MNE's on_missing option."
        ),
    )
    preproc_group.add_argument(
        "--event-id",
        help=(
            'Event id mapping. Provide a JSON string like \'{"A":1,"B":2}\' '
            "or comma-separated pairs like 'A:1,B:2'. If not provided, "
            "event ids will be inferred from events file or annotations."
        ),
        default=None,
    )
    preproc_group.add_argument(
        "--events-file",
        help=(
            "Optional path to an events.tsv file to use for epoching. "
            "If not provided, events will be inferred from annotations."
        ),
        default=None,
    )
    preproc_group.add_argument(
        "--tmin", type=float, help="Start time of epochs relative to event onset (s).", default=-0.2
    )
    preproc_group.add_argument(
        "--tmax", type=float, help="End time of epochs relative to event onset (s).", default=0.6
    )
    preproc_group.add_argument(
        "--reject-eeg",
        type=float,
        help=(
            "Peak-to-peak rejection threshold for EEG channels in Volts. "
            "Set to 0 to disable automatic rejection."
        ),
        default=75e-6,
    )
    preproc_group.add_argument(
        "--no-auto-reject",
        action="store_true",
        help=("Disable automatic amplitude-based epoch rejection."),
        default=False,
    )
    preproc_group.add_argument(
        "--concat-runs",
        action="store_true",
        help=(
            "Concatenate multiple runs for a subject and process them as a "
            "single recording. By default runs are processed separately."
        ),
        default=False,
    )
    preproc_group.add_argument(
        "--save-each-node",
        action="store_true",
        help=(
            "Write intermediate outputs to disk after each preprocessing "
            "step (disk-backed mode). This reduces peak memory at the "
            "expense of increased I/O and runtime."
        ),
        default=False,
    )
    preproc_group.add_argument(
        "--run",
        dest="run",
        nargs="+",
        help=(
            "Run label(s) to process for the participant (without the 'run-' "
            "prefix). If not provided, all runs found for the subject will "
            "be processed. Multiple runs can be provided as space-separated "
            "values, or as a single comma-separated string (e.g. '1 2' or "
            "'1,2')."
        ),
    )
    preproc_group.add_argument(
        "--task",
        dest="task",
        nargs="+",
        help=(
            "Task label(s) to process for the participant (without the 'task-' "
            "prefix). If not provided, all tasks found for the subject will "
            "be processed. Multiple tasks can be provided as space-separated "
            "values, or as a single comma-separated string (e.g. 'active passive')."
        ),
    )

    # Analysis parameters
    analysis_group = parser.add_argument_group("Analysis options")
    analysis_group.add_argument(
        "--by_event_type", action="store_true", help=("Create separate evoked responses for each event type.")
    )

    # General options
    parser.add_argument(
        "--skip_bids_validation",
        action="store_true",
        help=("Assume the input dataset is BIDS compliant " "and skip the validation."),
    )
    parser.add_argument(
        "--n_procs", type=int, default=1, help=("Number of processors to use for parallel execution.")
    )
    parser.add_argument("--work_dir", type=Path, help="Path where intermediate results should be " "stored.")

    return parser


def parse_baseline(baseline_str):
    """Parse baseline input to a (start, end) tuple of floats.

    Accepts:
    - a list/tuple of two floats (from argparse with nargs=2)
    - a comma-separated string like '-0.2,0'
    - a single numeric string which is interpreted as (value, 0.0)
    """
    # If argparse provided two floats (nargs=2), just return them
    if isinstance(baseline_str, (list, tuple)) and len(baseline_str) == 2:
        return float(baseline_str[0]), float(baseline_str[1])

    # If a comma-separated string was provided
    if isinstance(baseline_str, str) and "," in baseline_str:
        start, end = baseline_str.split(",")
        return float(start), float(end)

    # Otherwise treat as single numeric value (start) with 0 as end
    return float(baseline_str), 0.0


def parse_ref_channels(ref_str):
    """Parse reference channels string."""
    if not ref_str:
        return None

    if isinstance(ref_str, (list, tuple)):
        # Already a sequence; ensure items are stripped and handle
        # comma-separated tokens inside any element for backward
        # compatibility (e.g. ['M1,M2'] -> ['M1','M2']).
        out = []
        for item in ref_str:
            if isinstance(item, str) and "," in item:
                out.extend([c.strip() for c in item.split(",") if c.strip()])
            else:
                out.append(str(item).strip())
        return out

    s = str(ref_str)
    if s.lower() == "average":
        return None  # Average reference
    if "," in s:
        # Split and strip whitespace around channel names
        return [c.strip() for c in s.split(",") if c.strip()]
    # Single channel name; strip whitespace
    return s.strip()


def parse_picks(picks_str):
    """Parse picks string into list or return None."""
    if not picks_str:
        return None
    if isinstance(picks_str, (list, tuple)):
        return picks_str
    # comma-separated
    return [p.strip() for p in str(picks_str).split(",") if p.strip()]


def parse_event_id(event_id_str):
    """Parse event-id mapping. Accept JSON string or key:val pairs.

    Examples:
      '{"A": 1, "B": 2}'  OR  'A:1,B:2'
    """
    if not event_id_str:
        return None
    import json

    # JSON form first — detect by leading '{' (after stripping whitespace).
    s = str(event_id_str).strip()
    if s.startswith("{"):
        return json.loads(s)

    # Otherwise: key:val comma-separated format.
    mapping = {}
    for part in s.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        v = v.strip()
        # Coerce to int when the token looks like an integer literal.
        if v.lstrip("-").isdigit():
            v = int(v)
        mapping[k] = v
    return mapping if mapping else None


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
        print("Making sure the input data is BIDS compliant " "(warnings can be ignored in most cases).")
        validate_input_dir(exec_env, args.bids_dir, args.participant_label)

    # Only run participant-level analysis for now
    if args.analysis_level != "participant":
        print("Currently only participant-level analysis is supported.")
        return

    # Parse processing parameters
    baseline = parse_baseline(args.baseline)
    ref_channels = parse_ref_channels(args.ref_channels)
    # Determine rejection criteria: allow disabling automatic rejection via
    # --no-auto-reject, otherwise use the provided --reject-eeg threshold.
    if args.no_auto_reject:
        reject_value = None
    else:
        reject_value = {"eeg": float(args.reject_eeg)}

    # Determine filtering parameters. Support new MNE-style l_freq/h_freq
    # CLI flags while preserving backward compatibility with
    # --high_pass/--low_pass. --no-filter forces both to None.
    if args.no_filter:
        effective_l = None
        effective_h = None
        print("Filtering disabled by --no-filter; l_freq/h_freq set to None.")
    else:
        # Prefer explicit MNE-style args when provided
        effective_l = args.l_freq if getattr(args, "l_freq", None) is not None else args.high_pass
        effective_h = args.h_freq if getattr(args, "h_freq", None) is not None else args.low_pass
        print(f"Using filter settings: l_freq={effective_l}, h_freq={effective_h}")

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
        print(f"\n{'='*60}")
        print(f"Processing subject: sub-{subject}")
        print(f"{'='*60}")

        # Set up derivatives directories within the BIDS dataset
        derivatives_info = setup_derivatives_directories(
            args.bids_dir,
            subject,
            create_preprocessing=args.stage in ["preprocessing", "both"],
            create_analysis=args.stage in ["analysis", "both"],
        )

        print(f"Derivatives will be stored in: " f"{derivatives_info['derivatives_root']}")

        # Persist a structured run log next to the subject's derivatives so
        # there is a record of inputs/outputs/computing logs after the run.
        # The handler is added per-subject so each subject gets its own log
        # file in its own derivatives directory.
        _setup_subject_log(derivatives_info, subject, args.stage)

        # Create workflow based on stage
        if args.stage in ["preprocessing", "both"]:
            print("\n" + "=" * 60)
            print("Running preprocessing workflow...")
            print("=" * 60)

            # Determine runs for this subject. Default behavior: process
            # runs separately (one workflow run per run label). If the user
            # asked to concatenate runs via `--concat-runs`, run once
            # with run_label=None and let `load_data()` perform
            # concatenation.
            meta = get_sessions_tasks_runs(str(args.bids_dir), subject)
            # Allow overriding runs via CLI --run. Accept space-separated
            # values or a single comma-separated string per argument.
            if args.run:
                # Flatten any comma-separated entries
                provided = []
                for entry in args.run:
                    if isinstance(entry, str) and "," in entry:
                        provided.extend([r.strip() for r in entry.split(",") if r.strip()])
                    else:
                        provided.append(entry)
                runs = provided
            else:
                runs = meta.get("runs", [None])

            # Allow overriding tasks via CLI --task (same parsing rules)
            if args.task:
                provided_tasks = []
                for entry in args.task:
                    if isinstance(entry, str) and "," in entry:
                        provided_tasks.extend([t.strip() for t in entry.split(",") if t.strip()])
                    else:
                        provided_tasks.append(entry)
                tasks = provided_tasks
            else:
                tasks = meta.get("tasks", [None])

            # Validate requested tasks and runs against dataset metadata
            valid_tasks = [t for t in meta.get("tasks", []) if t is not None]
            valid_runs = [r for r in meta.get("runs", []) if r is not None]

            if args.task:
                # Identify missing/invalid task entries
                missing_tasks = [t for t in tasks if t not in valid_tasks]
                # If the user explicitly requested tasks but none are valid,
                # this is a user error: fail loudly so the caller can fix the CLI
                # invocation. If some are valid, warn and proceed with the
                # subset that exists.
                if missing_tasks and len(missing_tasks) == len(tasks):
                    print(f"ERROR: None of the requested task(s) {tasks} were found for subject sub-{subject}.")
                    if valid_tasks:
                        print(f"Available tasks for this subject: {valid_tasks}")
                    else:
                        print("No tasks available for this subject in the dataset.")
                    sys.exit(2)
                if missing_tasks:
                    print(
                        f"Warning: Requested task(s) {missing_tasks} not found for "
                        f"subject sub-{subject}; they will be ignored."
                    )
                    tasks = [t for t in tasks if t in valid_tasks]
                if not tasks:
                    # If tasks ended up empty after filtering, skip this subject
                    print(f"No valid tasks to process for subject sub-{subject}; skipping.")
                    continue

            if args.run:
                # If the dataset doesn't list any runs but the user explicitly
                # requested runs, treat as an error (likely CLI/dataset mismatch).
                if not valid_runs:
                    print(
                        f"ERROR: No runs available for subject sub-{subject} in "
                        f"dataset but --run was provided ({runs})."
                    )
                    sys.exit(2)

                missing_runs = [r for r in runs if r not in valid_runs]
                if missing_runs and len(missing_runs) == len(runs):
                    # User requested runs but none exist
                    print(f"ERROR: None of the requested run(s) {runs} were found for subject sub-{subject}.")
                    print(f"Available runs for this subject: {valid_runs}")
                    sys.exit(2)
                if missing_runs:
                    print(
                        f"Warning: Requested run(s) {missing_runs} not found for "
                        f"subject sub-{subject}; they will be ignored."
                    )
                    runs = [r for r in runs if r in valid_runs]
                if not runs:
                    print(f"No valid runs to process for subject sub-{subject}; skipping.")
                    continue

            # If concatenation requested and specific runs provided, we'll
            # pass the selected run list into the loader so only those runs
            # are concatenated. If no runs provided, all runs for the task
            # will be concatenated.

            if args.concat_runs:
                # Run one workflow per task; load_data will concatenate when
                # multiple files exist and run_label is None.
                for task_label in tasks:
                    print("\nConcatenating runs and processing as a single recording" f" for task {task_label}")
                    preproc_wf_run = create_preprocessing_workflow(disk_backed=bool(args.save_each_node))

                    if args.work_dir:
                        work_dir = args.work_dir / f"sub-{subject}" / "preprocessing"
                    else:
                        work_dir = derivatives_info["preprocessing_dir"] / "work" / f"sub-{subject}"

                    preproc_wf_run.base_dir = str(work_dir)

                    # Set inputs directly on the inputnode Node to avoid
                    # trait-notifier propagation issues when connections
                    # already exist on the workflow. nipype's get_node
                    # returns None when the node is missing.
                    in_node = preproc_wf_run.get_node("inputnode")

                    if in_node is not None:
                        in_node.inputs.bids_root = str(args.bids_dir)
                        in_node.inputs.sub_label = subject
                        # If user supplied --run, pass that list so the loader
                        # concatenates only the selected runs; otherwise leave
                        # run_label as None to concatenate all runs.
                        in_node.inputs.run_label = runs if args.run else None
                        in_node.inputs.task_label = task_label
                        in_node.inputs.ref_channels = ref_channels
                        in_node.inputs.high_pass = effective_l
                        in_node.inputs.low_pass = effective_h
                        in_node.inputs.baseline = baseline
                        in_node.inputs.tmin = args.tmin
                        in_node.inputs.tmax = args.tmax
                        # Pass reject criteria into the per-run workflow
                        in_node.inputs.reject = reject_value
                        # Epoching-specific inputs
                        in_node.inputs.picks = parse_picks(getattr(args, "picks", None))
                        in_node.inputs.on_missing = getattr(args, "on_missing", "warn")
                        in_node.inputs.event_id = parse_event_id(getattr(args, "event_id", None))
                        if getattr(args, "events_file", None) is not None:
                            in_node.inputs.events_file = args.events_file
                        in_node.inputs.derivatives_root = str(derivatives_info.get("derivatives_root", ""))
                        in_node.inputs.output_dir = str(derivatives_info["preprocessing_subject_dir"])
                    else:
                        # Fallback: best-effort assign to workflow.inputs
                        preproc_wf_run.inputs.inputnode.bids_root = str(args.bids_dir)
                        preproc_wf_run.inputs.inputnode.sub_label = subject
                        preproc_wf_run.inputs.inputnode.run_label = runs if args.run else None
                        preproc_wf_run.inputs.inputnode.task_label = task_label
                        preproc_wf_run.inputs.inputnode.ref_channels = ref_channels
                        preproc_wf_run.inputs.inputnode.high_pass = effective_l
                        preproc_wf_run.inputs.inputnode.low_pass = effective_h
                        preproc_wf_run.inputs.inputnode.baseline = baseline
                        preproc_wf_run.inputs.inputnode.tmin = args.tmin
                        preproc_wf_run.inputs.inputnode.tmax = args.tmax
                        preproc_wf_run.inputs.inputnode.reject = reject_value
                        preproc_wf_run.inputs.inputnode.picks = parse_picks(getattr(args, "picks", None))
                        preproc_wf_run.inputs.inputnode.on_missing = getattr(args, "on_missing", "warn")
                        preproc_wf_run.inputs.inputnode.event_id = parse_event_id(getattr(args, "event_id", None))
                        if getattr(args, "events_file", None) is not None:
                            preproc_wf_run.inputs.inputnode.events_file = args.events_file
                        preproc_wf_run.inputs.inputnode.derivatives_root = str(
                            derivatives_info.get("derivatives_root", "")
                        )
                        preproc_wf_run.inputs.inputnode.output_dir = str(
                            derivatives_info["preprocessing_subject_dir"]
                        )

                    # Run with requested parallelism: use MultiProc when
                    # multiple processors requested, otherwise Linear.
                    if args.n_procs and int(args.n_procs) > 1:
                        preproc_wf_run.run(
                            plugin="MultiProc",
                            plugin_args={"n_procs": int(args.n_procs)},
                        )
                    else:
                        preproc_wf_run.run(plugin="Linear")

                    # Same nipype-cache safety check as the per-run branch:
                    # confirm the concat output landed on disk. Concatenated
                    # outputs have no run token in the filename.
                    expected_dir = Path(derivatives_info["preprocessing_subject_dir"])
                    candidates = sorted(expected_dir.glob(
                        f"sub-{subject}_*task-{task_label}_desc-preproc_epo.fif"
                    ))
                    candidates = [c for c in candidates if "_run-" not in c.name]
                    if not candidates:
                        raise FileNotFoundError(
                            f"Concatenated preprocessing reported success but "
                            f"no _desc-preproc_epo.fif appeared in {expected_dir} "
                            f"for sub-{subject}, task-{task_label}. Most likely "
                            f"cause: stale nipype cache pointing at a "
                            f"previously-deleted output. Wipe the work directory "
                            f"({preproc_wf_run.base_dir}) and re-run."
                        )

                    print("\nPreprocessing (concatenated) completed.")
                    print(f"Outputs saved to: {derivatives_info['preprocessing_subject_dir']}")

                # After all tasks complete, generate report
                print("\n" + "=" * 60)
                print("Generating preprocessing report...")
                print("=" * 60)

                preproc_dir = Path(derivatives_info["preprocessing_subject_dir"])

                # Collect files grouped by task/run for each stage
                def _collect_grouped(base_dir, patterns):
                    grouped = {}
                    for pat in patterns:
                        for p in base_dir.glob(pat):
                            name = p.name
                            # Try to extract task-<task> and run-<run>
                            m_task = re.search(r"task-([^_]+)", name)
                            m_run = re.search(r"run-([^_]+)", name)
                            task = m_task.group(1) if m_task else ""
                            run = m_run.group(1) if m_run else ""
                            grouped.setdefault(task, {}).setdefault(run, []).append(str(p))
                    return grouped

                raw_patterns = ["*_desc-loaded_raw.fif"]
                events_patterns = ["*_events.tsv"]
                referenced_patterns = ["*_desc-referenced_raw.fif"]
                filtered_patterns = ["*_desc-filtered_raw.fif"]
                epoched_patterns = ["*_desc-preproc_epo.fif"]

                raw_files = _collect_grouped(preproc_dir, raw_patterns)
                events_files = _collect_grouped(preproc_dir, events_patterns)
                referenced_files = _collect_grouped(preproc_dir, referenced_patterns)
                filtered_files = _collect_grouped(preproc_dir, filtered_patterns)
                epoched_files = _collect_grouped(preproc_dir, epoched_patterns)

                def _total(grouped):
                    if not grouped:
                        return 0
                    return sum(len(r) for t in grouped.values() for r in t.values())

                print("\nCollected files:")
                print(f"  Raw files: {_total(raw_files)}")
                print(f"  Events files: {_total(events_files)}")
                print(f"  Referenced files: {_total(referenced_files)}")
                print(f"  Filtered files: {_total(filtered_files)}")
                print(f"  Epoched files: {_total(epoched_files)}")

                report_name = f"sub-{subject}_preprocessing_report.h5"
                report_path = reports.create_subject_report(
                    str(args.bids_dir),
                    out_dir=str(preproc_dir),
                    filename=report_name,
                    subject_id=subject,
                    command=" ".join(sys.argv),
                    overwrite=True,
                )

                # Add summary
                reports.add_report_summary(
                    report_path,
                    command=" ".join(sys.argv),
                    raw_files=raw_files,
                    events_files=events_files,
                    referenced_files=referenced_files,
                    filtered_files=filtered_files,
                    epoched_files=epoched_files,
                )

                # Add all processing stages organized by task/run
                reports.add_processing_stages(
                    report_path,
                    raw_files=raw_files,
                    events_files=events_files,
                    referenced_files=referenced_files,
                    filtered_files=filtered_files,
                    epoched_files=epoched_files,
                )

                # Export to HTML
                html_path = reports.save_report(report_path, overwrite=True)
                print(f"\n{'=' * 60}")
                print(f"Preprocessing report written to: {html_path}")
                print(f"{'=' * 60}")

            else:
                # Process each task and run separately
                for task_label in tasks:
                    for run in runs:
                        if run is None:
                            print(f"\nProcessing subject sub-{subject}, task {task_label} (no run label)")
                            run_label = None
                        else:
                            print(f"\nProcessing subject sub-{subject}, task {task_label}, run {run}")
                            run_label = run

                        # Create a fresh workflow instance per run to avoid state
                        # contamination between runs
                        preproc_wf_run = create_preprocessing_workflow(disk_backed=bool(args.save_each_node))

                        # Set working directory for nipype per run
                        if args.work_dir:
                            work_dir = (
                                args.work_dir
                                / f"sub-{subject}"
                                / "preprocessing"
                                / (f"task-{task_label}" if task_label is not None else "task-None")
                                / (f"run-{run}" if run is not None else "single")
                            )
                        else:
                            work_dir = (
                                derivatives_info["preprocessing_dir"]
                                / "work"
                                / f"sub-{subject}"
                                / (f"task-{task_label}" if task_label is not None else "task-None")
                                / (f"run-{run}" if run is not None else "single")
                            )

                        preproc_wf_run.base_dir = str(work_dir)

                        # Set inputs for this run on the inputnode Node.
                        # nipype's get_node returns None when missing.
                        in_node = preproc_wf_run.get_node("inputnode")

                        if in_node is not None:
                            in_node.inputs.bids_root = str(args.bids_dir)
                            in_node.inputs.sub_label = subject
                            in_node.inputs.run_label = run_label
                            in_node.inputs.task_label = task_label
                            in_node.inputs.ref_channels = ref_channels
                            in_node.inputs.high_pass = effective_l
                            in_node.inputs.low_pass = effective_h
                            in_node.inputs.baseline = baseline
                            in_node.inputs.tmin = args.tmin
                            in_node.inputs.tmax = args.tmax
                            in_node.inputs.reject = reject_value
                            # Epoching-specific inputs
                            in_node.inputs.picks = parse_picks(getattr(args, "picks", None))
                            in_node.inputs.on_missing = getattr(args, "on_missing", "warn")
                            in_node.inputs.event_id = parse_event_id(getattr(args, "event_id", None))
                            if getattr(args, "events_file", None) is not None:
                                in_node.inputs.events_file = args.events_file
                            in_node.inputs.derivatives_root = str(derivatives_info.get("derivatives_root", ""))
                            in_node.inputs.output_dir = str(derivatives_info["preprocessing_subject_dir"])
                        else:
                            preproc_wf_run.inputs.inputnode.bids_root = str(args.bids_dir)
                            preproc_wf_run.inputs.inputnode.sub_label = subject
                            preproc_wf_run.inputs.inputnode.run_label = run_label
                            preproc_wf_run.inputs.inputnode.task_label = task_label
                            preproc_wf_run.inputs.inputnode.ref_channels = ref_channels
                            preproc_wf_run.inputs.inputnode.high_pass = effective_l
                            preproc_wf_run.inputs.inputnode.low_pass = effective_h
                            preproc_wf_run.inputs.inputnode.baseline = baseline
                            preproc_wf_run.inputs.inputnode.tmin = args.tmin
                            preproc_wf_run.inputs.inputnode.tmax = args.tmax
                            preproc_wf_run.inputs.inputnode.reject = reject_value
                            preproc_wf_run.inputs.inputnode.picks = parse_picks(getattr(args, "picks", None))
                            preproc_wf_run.inputs.inputnode.on_missing = getattr(args, "on_missing", "warn")
                            preproc_wf_run.inputs.inputnode.event_id = parse_event_id(
                                getattr(args, "event_id", None)
                            )
                            if getattr(args, "events_file", None) is not None:
                                preproc_wf_run.inputs.inputnode.events_file = args.events_file
                            preproc_wf_run.inputs.inputnode.derivatives_root = str(
                                derivatives_info.get("derivatives_root", "")
                            )
                            preproc_wf_run.inputs.inputnode.output_dir = str(
                                derivatives_info["preprocessing_subject_dir"]
                            )

                        # Run the preprocessing workflow for this run
                        # Use requested parallelism for per-run workflow as well
                        if args.n_procs and int(args.n_procs) > 1:
                            preproc_wf_run.run(
                                plugin="MultiProc",
                                plugin_args={"n_procs": int(args.n_procs)},
                            )
                        else:
                            preproc_wf_run.run(plugin="Linear")

                        # Verify the expected output file actually exists. nipype
                        # caches by input hash and reports "Cached, collecting
                        # precomputed outputs" without checking that the recorded
                        # output file still exists on disk. If a previous run was
                        # cleaned up but the work cache wasn't, the workflow
                        # silently "succeeds" without writing anything. Catch
                        # that here with a clear actionable error.
                        expected_dir = Path(derivatives_info["preprocessing_subject_dir"])
                        run_token = f"run-{run_label}" if run_label is not None else None
                        candidates = sorted(expected_dir.glob(
                            f"sub-{subject}_*task-{task_label}_*desc-preproc_epo.fif"
                        ))
                        if run_token is not None:
                            candidates = [c for c in candidates if run_token in c.name]
                        if not candidates:
                            raise FileNotFoundError(
                                f"Preprocessing reported success but no "
                                f"_desc-preproc_epo.fif appeared in {expected_dir} "
                                f"for sub-{subject}, task-{task_label}, "
                                f"run-{run_label}. Most likely cause: stale "
                                f"nipype cache pointing at a previously-deleted "
                                f"output. Wipe the work directory "
                                f"({preproc_wf_run.base_dir}) and re-run."
                            )

                        print(f"\nPreprocessing for run {run_label} completed.")
                        print(f"Outputs saved to: {derivatives_info['preprocessing_subject_dir']}")

                # After all runs complete, generate report
                print("\n" + "=" * 60)
                print("Generating preprocessing report...")
                print("=" * 60)

                preproc_dir = Path(derivatives_info["preprocessing_subject_dir"])

                # Collect files grouped by task/run for each stage
                def _collect_grouped(base_dir, patterns):
                    grouped = {}
                    for pat in patterns:
                        for p in base_dir.glob(pat):
                            name = p.name
                            # Try to extract task-<task> and run-<run>
                            m_task = re.search(r"task-([^_]+)", name)
                            m_run = re.search(r"run-([^_]+)", name)
                            task = m_task.group(1) if m_task else ""
                            run = m_run.group(1) if m_run else ""
                            grouped.setdefault(task, {}).setdefault(run, []).append(str(p))
                    return grouped

                raw_patterns = ["*_desc-loaded_raw.fif"]
                events_patterns = ["*_events.tsv"]
                referenced_patterns = ["*_desc-referenced_raw.fif"]
                filtered_patterns = ["*_desc-filtered_raw.fif"]
                epoched_patterns = ["*_desc-preproc_epo.fif"]

                raw_files = _collect_grouped(preproc_dir, raw_patterns)
                events_files = _collect_grouped(preproc_dir, events_patterns)
                referenced_files = _collect_grouped(preproc_dir, referenced_patterns)
                filtered_files = _collect_grouped(preproc_dir, filtered_patterns)
                epoched_files = _collect_grouped(preproc_dir, epoched_patterns)

                def _total(grouped):
                    if not grouped:
                        return 0
                    return sum(len(r) for t in grouped.values() for r in t.values())

                print("\nCollected files:")
                print(f"  Raw files: {_total(raw_files)}")
                print(f"  Events files: {_total(events_files)}")
                print(f"  Referenced files: {_total(referenced_files)}")
                print(f"  Filtered files: {_total(filtered_files)}")
                print(f"  Epoched files: {_total(epoched_files)}")

                report_name = f"sub-{subject}_preprocessing_report.h5"
                report_path = reports.create_subject_report(
                    str(args.bids_dir),
                    out_dir=str(preproc_dir),
                    filename=report_name,
                    subject_id=subject,
                    command=" ".join(sys.argv),
                    overwrite=True,
                )

                # Add summary
                reports.add_report_summary(
                    report_path,
                    command=" ".join(sys.argv),
                    raw_files=raw_files,
                    events_files=events_files,
                    referenced_files=referenced_files,
                    filtered_files=filtered_files,
                    epoched_files=epoched_files,
                )

                # Add all processing stages organized by task/run
                reports.add_processing_stages(
                    report_path,
                    raw_files=raw_files,
                    events_files=events_files,
                    referenced_files=referenced_files,
                    filtered_files=filtered_files,
                    epoched_files=epoched_files,
                )

                # Export to HTML
                html_path = reports.save_report(report_path, overwrite=True)
                print(f"\n{'=' * 60}")
                print(f"Preprocessing report written to: {html_path}")
                print(f"{'=' * 60}")

        if args.stage in ["analysis", "both"]:
            print("\n" + "=" * 60)
            print("Running analysis workflow...")
            print("=" * 60)

            # Locate the preprocessing outputs (epoched .fif files) we need
            # to feed to the analysis workflow.
            preproc_subject_dir = Path(derivatives_info["preprocessing_subject_dir"])
            preproc_files = sorted(preproc_subject_dir.glob("*_desc-preproc_epo.fif"))

            if not preproc_files:
                print(f"ERROR: No preprocessing outputs found for subject {subject}.")
                print(f"Expected location: {preproc_subject_dir}")
                print("Please run preprocessing stage first or use 'both' stage.")
                continue
            print(f"Found {len(preproc_files)} preprocessing output(s) to analyze.")

            # Run the analysis workflow once per preprocessed file so each
            # (task, run) combination produces its own evoked output.
            import mne

            for preproc_file in preproc_files:
                print(f"\nAnalyzing: {preproc_file.name}")
                epochs = mne.read_epochs(str(preproc_file), preload=True, verbose=False)

                analysis_wf = create_analysis_workflow()

                if args.work_dir:
                    work_dir = (
                        args.work_dir / f"sub-{subject}" / "analysis" / preproc_file.stem
                    )
                else:
                    work_dir = (
                        derivatives_info["analysis_dir"] / "work"
                        / f"sub-{subject}" / preproc_file.stem
                    )
                analysis_wf.base_dir = str(work_dir)

                analysis_wf.inputs.inputnode.epochs = epochs
                analysis_wf.inputs.inputnode.by_event_type = args.by_event_type
                analysis_wf.inputs.inputnode.bids_root = str(args.bids_dir)
                analysis_wf.inputs.inputnode.subject = subject
                analysis_wf.inputs.inputnode.original_filename = preproc_file.stem
                analysis_wf.inputs.inputnode.output_dir = str(
                    derivatives_info["analysis_subject_dir"]
                )

                if args.n_procs and int(args.n_procs) > 1:
                    analysis_wf.run(
                        plugin="MultiProc",
                        plugin_args={"n_procs": int(args.n_procs)},
                    )
                else:
                    analysis_wf.run(plugin="Linear")

                # Propagate ConcatenatedRuns / Run from the preproc sidecar
                # to each analysis sidecar produced from this preproc file.
                # The analysis stage doesn't know about concat semantics on
                # its own (it only sees the preproc filename), so carry
                # forward the field by reading the source sidecar.
                _propagate_run_provenance(
                    preproc_file=preproc_file,
                    analysis_dir=Path(derivatives_info["analysis_subject_dir"]),
                )

            print(f"Analysis completed. Outputs saved to: {derivatives_info['analysis_subject_dir']}")
            # Create analysis report using reports submodule
            analysis_dir = Path(derivatives_info["analysis_subject_dir"])
            # Look for analysis outputs
            analysis_patterns = ["*_desc-evoked.fif", "*_desc-evoked*.fif"]
            found_outputs = []
            for pat in analysis_patterns:
                found_outputs.extend(list(analysis_dir.glob(pat)))

            report_name = f"sub-{subject}_analysis_report.h5"
            report_path = reports.create_report(
                str(args.bids_dir), out_dir=str(analysis_dir),
                filename=report_name, overwrite=True,
            )
            html_text = f"<h2>Analysis summary</h2><p>Subject: sub-{subject}</p>"
            if found_outputs:
                html_text += "<p>Saved analysis outputs:</p><ul>"
                for p in found_outputs:
                    html_text += f"<li>{str(p)}</li>"
                html_text += "</ul>"
            reports.add_to_report(report_path, html_text=html_text, html_title="Analysis summary")
            reports.save_report(report_path, overwrite=True)
            print(f"Analysis report written to: {report_path}")

    print("\n" + "=" * 60)
    print("ffrprep processing completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_ffrprep()
