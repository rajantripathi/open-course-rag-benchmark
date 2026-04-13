#!/usr/bin/env bash
#SBATCH --job-name=ocrb-dense
#SBATCH --account=brics.u6ef
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out

set -euo pipefail
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/u6ef/rajantripathi.u6ef/open-course-rag-benchmark}"
cd "$SCRATCH_ROOT/repo"
source scripts/isambard/slurm_env.sh
ocrb dense --chunks "$SCRATCH_ROOT/processed/chunks.jsonl" --questions data/benchmark/questions.jsonl --config configs/dense.yaml --index-output "$SCRATCH_ROOT/indexes/e5_large.faiss" --output "$SCRATCH_ROOT/results/retrieval/dense_results.jsonl"

