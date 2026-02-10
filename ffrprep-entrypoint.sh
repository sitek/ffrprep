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

# Exec the original neurodocker startup script (preserves previous behavior)
exec /neurodocker/startup.sh "$@"
