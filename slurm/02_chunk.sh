#!/usr/bin/env bash
#SBATCH --job-name=ocrb-chunk
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
ocrb chunk \
  --documents data/processed/documents.jsonl \
  --output data/processed/chunks.jsonl

