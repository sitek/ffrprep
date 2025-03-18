import sys
import tempfile
import subprocess
import json


def validate_input_dir(exec_env, bids_dir, participant_label):
    """
    Validate BIDS directory and structure via the BIDS-validator.
    Functionality copied from fmriprep.

    Parameters
    ----------
    exec_env : str
        Environment BIDSonym is run in.
    bids_dir : str
        Path to BIDS root directory.
    participant_label: str
        Label(s) of subject to be checked (without 'sub-').
    """

    validator_config_dict = {
        "ignore": [
            "TSV_EQUAL_ROWS",
            "TSV_EMPTY_CELL",
            "TSV_IMPROPER_NA",
            "VOLUME_COUNT_MISMATCH",
            "BVAL_MULTIPLE_ROWS",
            "BVEC_NUMBER_ROWS",
            "DWI_MISSING_BVAL",
            "INCONSISTENT_SUBJECTS",
            "INCONSISTENT_PARAMETERS",
            "BVEC_ROW_LENGTH",
            "B_FILE",
            "PARTICIPANT_ID_COLUMN",
            "PARTICIPANT_ID_MISMATCH",
            "TASK_NAME_MUST_DEFINE",
            "PHENOTYPE_SUBJECTS_MISSING",
            "STIMULUS_FILE_MISSING",
            "DWI_MISSING_BVEC",
            "TSV_IMPROPER_NA",
            "ACQTIME_FMT",
            "Participants age 89 or higher",
            "DATASET_DESCRIPTION_JSON_MISSING",
            "FILENAME_COLUMN",
            "WRONG_NEW_LINE",
            "MISSING_TSV_COLUMN_IEEG_CHANNELS",
            "MISSING_TSV_COLUMN_IEEG_ELECTRODES",
            "UNUSED_STIMULUS",
            "CHANNELS_COLUMN_SFREQ",
            "CHANNELS_COLUMN_LOWCUT",
            "CHANNELS_COLUMN_HIGHCUT",
            "CHANNELS_COLUMN_NOTCH",
            "CUSTOM_COLUMN_WITHOUT_DESCRIPTION",
            "ACQTIME_FMT",
            "SUSPICIOUSLY_LONG_EVENT_DESIGN",
            "SUSPICIOUSLY_SHORT_EVENT_DESIGN",
            "MALFORMED_BVEC",
            "MALFORMED_BVAL",
            "MISSING_TSV_COLUMN_EEG_ELECTRODES",
            "MISSING_SESSION"
        ],
        "error": ["NO_EGG", "NO_EVENTS_TSV"],
        "ignoredFiles": ['/dataset_description.json', '/participants.tsv']
    }

    # Limit validation only to data from requested participants
    if participant_label:
        all_subs = set([s.name[4:] for s in bids_dir.glob('sub-*')])
        selected_subs = set([s[4:] if s.startswith('sub-') else s
                             for s in participant_label])
        bad_labels = selected_subs.difference(all_subs)
        if bad_labels:
            error_msg = (
                'Data for requested participant(s) label(s) not found. '
                'Could not find data for participant(s): %s. '
            ) + 'Please verify the requested ' \
                'participant labels.'
            if exec_env == 'docker':
                error_msg += (
                    ' This error can be caused by the input data not being '
                    'accessible inside the docker container. Please make sure '
                    'all volumes are mounted properly (see '
                    'https://docs.docker.com/engine/reference/commandline/run/'
                    '#mount-volume--v---read-only)'
                )
            if exec_env == 'singularity':
                error_msg += (
                    'accessible inside the singularity container. Please make '
                    'sure all paths are mapped properly (see '
                    'https://www.sylabs.io/guides/3.0/user-guide/'
                    'bind_paths_and_mounts.html)'
                    'guides/3.0/user-guide/bind_paths_and_mounts.html)'
                )
            raise RuntimeError(error_msg % ','.join(bad_labels))

        ignored_subs = all_subs.difference(selected_subs)
        if ignored_subs:
            for sub in ignored_subs:
                validator_config_dict["ignoredFiles"].append(
                    "/sub-%s/**" % sub)
    with tempfile.NamedTemporaryFile('w+') as temp:
        temp.write(json.dumps(validator_config_dict))
        temp.flush()
        try:
            subprocess.check_call([
                'bids-validator', bids_dir, '-c', temp.name
            ])
        except FileNotFoundError:
            print("bids-validator does not appear to be installed",
                  file=sys.stderr)
