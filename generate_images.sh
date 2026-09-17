#!/bin/bash

# Script: generate_images.sh
# Purpose: Generate Docker and/or Singularity container definitions for ffrprep
# Description: Uses neurodocker for the OS layer; uv manages Python and project
#              dependencies inside the container (no conda). Reproducibility comes
#              from the committed uv.lock — re-running uv lock locally before a
#              rebuild keeps the image deterministic.

set -e

# OS packages required at runtime (curl/unzip pull installers, git lets
# setuptools_scm read the version, ca-certificates is needed for HTTPS).
OS_PACKAGES="ca-certificates curl unzip git"

# uv installer — drops the binary at /usr/local/bin/uv
UV_INSTALL="curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh"

# deno installer — bids-validator-deno shells out to this binary
DENO_INSTALL="curl -fsSL https://deno.land/install.sh | env DENO_INSTALL=/usr/local sh -s -- --no-modify-path"

# Default uv sync flag set: omit the dev dependency group. --with-tests flips
# this so pytest/flake8/etc. get baked into the image.
INSTALL_GROUP_FLAG="--no-dev"

generate_docker() {
  docker run --rm repronim/neurodocker:2.1.1 generate docker \
    --base-image debian:bookworm-slim \
    --pkg-manager apt \
    --arg DEBIAN_FRONTEND=noninteractive \
    --install $OS_PACKAGES \
    --run "$DENO_INSTALL" \
    --run "$UV_INSTALL" \
    --env UV_LINK_MODE=copy \
    --env UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    --env UV_PYTHON=3.11 \
    --env UV_PROJECT_ENVIRONMENT=/home/ffrprep/.venv \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --copy . /home/ffrprep \
    --workdir /home/ffrprep \
    --run "uv sync --frozen $INSTALL_GROUP_FLAG" \
    --run "chmod +x /home/ffrprep/ffrprep-entrypoint.sh" \
    --run "chmod -R a+rX /opt/uv-python /home/ffrprep/.venv" \
    --run "chmod -R a+rwX /home/ffrprep" \
    --env IS_DOCKER=1 \
    --workdir /tmp \
    --entrypoint "/home/ffrprep/ffrprep-entrypoint.sh"
}

generate_singularity() {
  docker run --rm repronim/neurodocker:2.1.1 generate singularity \
    --base-image debian:bookworm-slim \
    --pkg-manager apt \
    --arg DEBIAN_FRONTEND=noninteractive \
    --install $OS_PACKAGES \
    --run "$DENO_INSTALL" \
    --run "$UV_INSTALL" \
    --env UV_LINK_MODE=copy \
    --env UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    --env UV_PYTHON=3.11 \
    --env UV_PROJECT_ENVIRONMENT=/home/ffrprep/.venv \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --copy . /home/ffrprep \
    --workdir /home/ffrprep \
    --run "uv sync --frozen $INSTALL_GROUP_FLAG" \
    --run "chmod +x /home/ffrprep/ffrprep-entrypoint.sh" \
    --run "chmod -R a+rX /opt/uv-python /home/ffrprep/.venv" \
    --run "chmod -R a+rwX /home/ffrprep" \
    --env IS_DOCKER=1 \
    --workdir /tmp \
    --entrypoint "/home/ffrprep/ffrprep-entrypoint.sh"
}

build_docker() {
    echo "  → Building ffrprep:local Docker image..."
    docker build -t ffrprep:local .
    echo "  → Build complete. Run with: docker run --rm ffrprep:local --help"
}

build_singularity() {
    echo "  → Building ffrprep.sif (may require sudo or fakeroot)..."
    singularity build ffrprep.sif Singularity.def
    echo "  → Build complete. Run with: singularity run ffrprep.sif --help"
}

show_usage() {
    echo "Usage: $0 [docker|singularity|both] [local] [--with-tests]"
    echo ""
    echo "DESCRIPTION:"
    echo "  Generate container definition files (and optionally build images) for"
    echo "  the ffrprep project. Uses neurodocker for the OS layer and uv to"
    echo "  install Python deps from the committed uv.lock."
    echo ""
    echo "ARGUMENTS:"
    echo "  docker        Generate Dockerfile only"
    echo "  singularity   Generate Singularity.def only"
    echo "  both          Generate both (default)"
    echo "  local         Build the image(s) locally after generating definitions"
    echo "  --with-tests  Include the [dependency-groups] dev group (pytest,"
    echo "                flake8, etc.) so the image can run the test suite."
    echo ""
    echo "EXAMPLES:"
    echo "  $0                            # Generate both definition files"
    echo "  $0 docker local               # Generate Dockerfile and build runtime image"
    echo "  $0 docker local --with-tests  # Build image with test dependencies"
    echo ""
    echo "REQUIREMENTS:"
    echo "  - Docker installed and running (used for both generation and Docker builds)"
    echo "  - Singularity installed (only needed when building .sif locally)"
    echo "  - uv.lock checked in (run 'uv lock' locally if it's missing or stale)"
    echo "  - repronim/neurodocker:2.1.1 will be pulled automatically"
}

echo "=== ffrprep Container Generation Script ==="

GENERATE_DOCKER=false
GENERATE_SINGULARITY=false
BUILD_LOCAL=false

if [ $# -eq 0 ]; then
    GENERATE_DOCKER=true
    GENERATE_SINGULARITY=true
fi

for arg in "$@"; do
    case $arg in
        docker)
            GENERATE_DOCKER=true
            ;;
        singularity)
            GENERATE_SINGULARITY=true
            ;;
        both)
            GENERATE_DOCKER=true
            GENERATE_SINGULARITY=true
            ;;
        local)
            BUILD_LOCAL=true
            ;;
        --with-tests)
            INSTALL_GROUP_FLAG=""
            ;;
        help|--help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument '$arg'"
            show_usage
            exit 1
            ;;
    esac
done

if [ "$GENERATE_DOCKER" = false ] && [ "$GENERATE_SINGULARITY" = false ]; then
    echo "Error: nothing to generate. Pass docker, singularity, or both."
    show_usage
    exit 1
fi

echo "Configuration:"
echo "  Generate Dockerfile:    $GENERATE_DOCKER"
echo "  Generate Singularity:   $GENERATE_SINGULARITY"
echo "  Build images locally:   $BUILD_LOCAL"
if [ -z "$INSTALL_GROUP_FLAG" ]; then
    echo "  uv sync flags:          (default groups, includes dev)"
else
    echo "  uv sync flags:          $INSTALL_GROUP_FLAG"
fi

if [ "$GENERATE_DOCKER" = true ]; then
    echo "→ Generating Dockerfile..."
    generate_docker > Dockerfile
    echo "  Wrote ./Dockerfile"
fi

if [ "$GENERATE_SINGULARITY" = true ]; then
    echo "→ Generating Singularity.def..."
    generate_singularity > Singularity.def
    echo "  Wrote ./Singularity.def"
fi

if [ "$BUILD_LOCAL" = true ]; then
    if [ "$GENERATE_DOCKER" = true ]; then
        build_docker
    fi
    if [ "$GENERATE_SINGULARITY" = true ]; then
        build_singularity
    fi
fi

echo "Done at $(date)"
