#!/usr/bin/env bash
#SBATCH --job-name=ocrb-setup
#SBATCH --account=brics.u6ef
#SBATCH --partition=workq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/u6ef/rajantripathi.u6ef/open-course-rag-benchmark}"
REPO_DIR="${REPO_DIR:-$SCRATCH_ROOT/repo}"
cd "$REPO_DIR"
bash scripts/isambard/setup_env.sh

