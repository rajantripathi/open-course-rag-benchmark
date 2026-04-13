#!/usr/bin/env bash
#SBATCH --job-name=ocrb-hybrid
#SBATCH --account=brics.u6ef
#SBATCH --partition=workq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%j.out

set -euo pipefail
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/u6ef/rajantripathi.u6ef/open-course-rag-benchmark}"
cd "$SCRATCH_ROOT/repo"
source scripts/isambard/slurm_env.sh
ocrb hybrid --bm25-results "$SCRATCH_ROOT/results/retrieval/bm25_results.jsonl" --dense-results "$SCRATCH_ROOT/results/retrieval/dense_results.jsonl" --config configs/hybrid.yaml --output "$SCRATCH_ROOT/results/retrieval/hybrid_results.jsonl"

