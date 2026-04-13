#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH:-$HOME}/open-course-rag-benchmark}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/open-course-rag-benchmark}"

module purge || true
module load cray-python/3.11.7 2>/dev/null || true
module load cudatoolkit/24.11_12.6 2>/dev/null || true

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$REPO_DIR/requirements.txt"

mkdir -p \
  "$PROJECT_ROOT/raw" \
  "$PROJECT_ROOT/processed" \
  "$PROJECT_ROOT/indexes" \
  "$PROJECT_ROOT/results" \
  "$PROJECT_ROOT/cache"

echo "Environment ready at $VENV_DIR"
