#!/usr/bin/env bash
set -e

# Ensure USER is set to avoid whoami lookups failing for unknown UIDs
export USER=${USER:-container}

# Ensure MPLCONFIGDIR is writable to avoid matplotlib trying to write to '/.config'
# Default to /tmp/mplconfig if not provided by the caller.
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplconfig}
mkdir -p "$MPLCONFIGDIR"
# Make it writable by everyone but keep the sticky bit (like /tmp)
chmod 1777 "$MPLCONFIGDIR" || true

# If conda exists, initialize shell hook and activate the ffrprep environment
if [ -x "/opt/miniconda-latest/bin/conda" ]; then
    # Initialize conda for bash
    eval "$(/opt/miniconda-latest/bin/conda 'shell.bash' 'hook' 2>/dev/null)" || true
    # Activate the ffrprep environment if present
    if conda info --envs | grep -q "ffrprep"; then
        conda activate ffrprep || true
    fi
fi

# Dispatch: route `download ...` to the ffrprep-download CLI; everything
# else continues to hit `ffrprep` so the BIDS-App contract is preserved.
if [ "${1:-}" = "download" ]; then
    shift
    exec /neurodocker/startup.sh ffrprep-download "$@"
else
    exec /neurodocker/startup.sh ffrprep "$@"
fi
