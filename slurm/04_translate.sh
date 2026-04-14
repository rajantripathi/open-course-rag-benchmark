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
cd "$SCRATCH_ROOT/repo"
source scripts/isambard/slurm_env.sh
export HF_HOME="$SCRATCH_ROOT/cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"
mkdir -p "$SCRATCH_ROOT/benchmark_candidates"

python -u -m open_course_rag_benchmark.translate_questions \
  --input "$SCRATCH_ROOT/benchmark_candidates/candidates.jsonl" \
  --model-name "Qwen/Qwen2.5-3B-Instruct" \
  --output "$SCRATCH_ROOT/benchmark_candidates/translations.jsonl"

echo ""
echo "=== Job complete ==="
wc -l "$SCRATCH_ROOT/benchmark_candidates/translations.jsonl" 2>/dev/null || echo "0"
