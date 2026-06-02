#!/usr/bin/env bash
set -e

# Ensure USER is set so libraries that call whoami don't fail under arbitrary UIDs
export USER=${USER:-container}

# Matplotlib needs a writable config dir
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplconfig}
mkdir -p "$MPLCONFIGDIR"
chmod 1777 "$MPLCONFIGDIR" || true

# Activate the uv-managed project virtualenv
if [ -f "/home/ffrprep/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source /home/ffrprep/.venv/bin/activate
fi

# Dispatch:
#   download ...  → ffrprep-download    (toolbox CLI)
#   test     ...  → pytest               (test runner; needs --with-tests image)
#   anything else → ffrprep              (BIDS-App contract preserved)
case "${1:-}" in
    download)
        shift
        exec ffrprep-download "$@"
        ;;
    test)
        shift
        # cd into the project so pytest picks up testpaths from pyproject.toml
        cd /home/ffrprep
        exec pytest "$@"
        ;;
    *)
        exec ffrprep "$@"
        ;;
esac
