#!/usr/bin/env bash
#SBATCH --job-name=ocrb-translate
#SBATCH --account=brics.u6ef
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/u6ef/rajantripathi.u6ef/open-course-rag-benchmark}"
START_INDEX="${START_INDEX:-0}"
MAX_RECORDS="${MAX_RECORDS:-25}"
cd "$SCRATCH_ROOT/repo"
source scripts/isambard/slurm_env.sh
export HF_HOME="$SCRATCH_ROOT/cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"
mkdir -p "$SCRATCH_ROOT/benchmark_candidates"

python -u -m open_course_rag_benchmark.translate_questions \
  --input "$SCRATCH_ROOT/benchmark_candidates/candidates.jsonl" \
  --model-name "Qwen/Qwen2.5-1.5B-Instruct" \
  --output "$SCRATCH_ROOT/benchmark_candidates/translations.jsonl" \
  --start-index "$START_INDEX" \
  --max-records "$MAX_RECORDS" \
  --append

echo ""
echo "=== Job complete ==="
echo "start_index=$START_INDEX max_records=$MAX_RECORDS"
wc -l "$SCRATCH_ROOT/benchmark_candidates/translations.jsonl" 2>/dev/null || echo "0"
