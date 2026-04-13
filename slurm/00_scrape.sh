#!/usr/bin/env bash
#SBATCH --job-name=scrape_openstax
#SBATCH --account=brics.u6ef
#SBATCH --partition=workq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/u6ef/rajantripathi.u6ef/open-course-rag-benchmark}"
cd "$SCRATCH_ROOT/repo"
source venv/bin/activate

python -m open_course_rag_benchmark.scrape_openstax \
  --book principles-data-science \
  --output "$SCRATCH_ROOT/raw/openstax_data_science"

python -m open_course_rag_benchmark.scrape_openstax \
  --book introduction-philosophy \
  --output "$SCRATCH_ROOT/raw/openstax_philosophy"

find "$SCRATCH_ROOT/raw" -type f | head -50 || true
