#!/usr/bin/env bash
#SBATCH --job-name=ocrb-plots
#SBATCH --account=brics.u6ef
#SBATCH --partition=workq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out

set -euo pipefail
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/u6ef/rajantripathi.u6ef/open-course-rag-benchmark}"
cd "$SCRATCH_ROOT/repo"
source scripts/isambard/slurm_env.sh
ocrb plots --results-dir "$SCRATCH_ROOT/results" --output-dir "$SCRATCH_ROOT/results/figures"

