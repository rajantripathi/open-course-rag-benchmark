#!/usr/bin/env bash
#SBATCH --job-name=ocrb-genq
#SBATCH --account=brics.u6ef
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out

set -euo pipefail
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/u6ef/rajantripathi.u6ef/open-course-rag-benchmark}"
cd "$SCRATCH_ROOT/repo"
source scripts/isambard/slurm_env.sh
ocrb generate-questions --chunks "$SCRATCH_ROOT/processed/chunks.jsonl" --output "$SCRATCH_ROOT/benchmark_candidates/candidates.jsonl"

