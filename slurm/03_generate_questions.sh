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
export HF_HOME="$SCRATCH_ROOT/cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"
mkdir -p "$SCRATCH_ROOT/benchmark_candidates"

python -u -m open_course_rag_benchmark.generate_questions \
  --chunks "$SCRATCH_ROOT/processed/chunks.jsonl" \
  --model-name "Qwen/Qwen2.5-3B-Instruct" \
  --output "$SCRATCH_ROOT/benchmark_candidates/candidates.jsonl"
