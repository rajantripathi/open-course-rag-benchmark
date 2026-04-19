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

if [[ -z "${START_INDEX:-}" ]]; then
  BATCH_SIZE="${BATCH_SIZE:-20}"
  TOTAL_RECORDS="${TOTAL_RECORDS:-240}"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
  first_jid=""
  last_jid=""
  for start in $(seq 0 "$BATCH_SIZE" $((TOTAL_RECORDS - 1))); do
    count="$BATCH_SIZE"
    if (( start + count > TOTAL_RECORDS )); then
      count=$((TOTAL_RECORDS - start))
    fi
    jid=$(sbatch --parsable --export=ALL,START_INDEX="$start",MAX_RECORDS="$count",MAX_NEW_TOKENS="$MAX_NEW_TOKENS" slurm/09_answer_generate.sh)
    [[ -z "$first_jid" ]] && first_jid="$jid"
    last_jid="$jid"
  done
  echo "Submitted answer-generation batches: $first_jid -> $last_jid"
  exit 0
fi

MAX_RECORDS="${MAX_RECORDS:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
source scripts/isambard/slurm_env.sh
mkdir -p "$SCRATCH_ROOT/results/answers"
ocrb answer \
  --chunks "$SCRATCH_ROOT/processed/chunks.jsonl" \
  --retrieval "$SCRATCH_ROOT/results/retrieval/hybrid_results.jsonl" \
  --questions data/benchmark/questions.jsonl \
  --output "$SCRATCH_ROOT/results/answers/answers.jsonl" \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --start-index "$START_INDEX" \
  --max-records "$MAX_RECORDS" \
  --append
