#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
export PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH:-$HOME}/open-course-rag-benchmark}"
export VENV_DIR="${VENV_DIR:-$REPO_DIR/venv}"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$PROJECT_ROOT/cache}"

module purge || true
module load cray-python/3.11.7 2>/dev/null || true
module load cudatoolkit/24.11_12.6 2>/dev/null || true

source "$VENV_DIR/bin/activate"
