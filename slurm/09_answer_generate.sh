#!/usr/bin/env bash
#SBATCH --job-name=ocrb-answer
#SBATCH --account=brics.u6ef
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%j.out

set -euo pipefail
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/u6ef/rajantripathi.u6ef/open-course-rag-benchmark}"
cd "$SCRATCH_ROOT/repo"
source scripts/isambard/slurm_env.sh
mkdir -p "$SCRATCH_ROOT/results/answers"
ocrb answer \
  --chunks "$SCRATCH_ROOT/processed/chunks.jsonl" \
  --retrieval "$SCRATCH_ROOT/results/retrieval/hybrid_results.jsonl" \
  --questions data/benchmark/questions.jsonl \
  --output "$SCRATCH_ROOT/results/answers/answers.jsonl" \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --max-new-tokens 96 \
  --append
