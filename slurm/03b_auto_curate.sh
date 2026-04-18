#!/usr/bin/env bash
#SBATCH --job-name=ocrb-auto-curate
#SBATCH --account=brics.u6ef
#SBATCH --partition=workq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/u6ef/rajantripathi.u6ef/open-course-rag-benchmark}"
cd "$SCRATCH_ROOT/repo"
source scripts/isambard/slurm_env.sh

AUTO_SHEET="$SCRATCH_ROOT/benchmark_candidates/auto_curated_sheet.csv"
REVIEW_FLAGS="$SCRATCH_ROOT/benchmark_candidates/review_flags.csv"
QUESTIONS_OUT="$SCRATCH_ROOT/repo/data/benchmark/questions.jsonl"
LABELS_OUT="$SCRATCH_ROOT/repo/data/benchmark/gold_labels.jsonl"

python scripts/auto_curate.py \
  --input "$SCRATCH_ROOT/benchmark_candidates/curation_sheet.csv" \
  --output "$AUTO_SHEET" \
  --review-flags "$REVIEW_FLAGS"

python scripts/import_curated_sheet.py \
  --sheet "$AUTO_SHEET" \
  --translations "$SCRATCH_ROOT/benchmark_candidates/translations.jsonl" \
  --questions-output "$QUESTIONS_OUT" \
  --labels-output "$LABELS_OUT"

python -m open_course_rag_benchmark.build_benchmark \
  --questions "$QUESTIONS_OUT" \
  --gold "$LABELS_OUT" \
  --chunks "$SCRATCH_ROOT/processed/chunks.jsonl"

echo ---
wc -l "$REVIEW_FLAGS"
