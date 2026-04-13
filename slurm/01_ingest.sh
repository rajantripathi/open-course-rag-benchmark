#!/usr/bin/env bash
#SBATCH --job-name=ocrb-ingest
#SBATCH --account=brics.u6ef
#SBATCH --partition=workq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"
source scripts/isambard/slurm_env.sh
ocrb ingest \
  --manifest data/raw/source_manifest_template.csv \
  --base-dir "$PROJECT_ROOT" \
  --output data/processed/documents.jsonl

