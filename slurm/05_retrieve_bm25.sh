#!/usr/bin/env bash
#SBATCH --job-name=ocrb-bm25
#SBATCH --account=brics.u6ef
#SBATCH --partition=workq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out

set -euo pipefail
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/u6ef/rajantripathi.u6ef/open-course-rag-benchmark}"
cd "$SCRATCH_ROOT/repo"
source scripts/isambard/slurm_env.sh
ocrb bm25 --chunks "$SCRATCH_ROOT/processed/chunks.jsonl" --questions data/benchmark/questions.jsonl --config configs/bm25.yaml --output "$SCRATCH_ROOT/results/retrieval/bm25_results.jsonl"

