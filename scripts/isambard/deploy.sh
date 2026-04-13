#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-u6ef.aip2.isambard}"
REMOTE_ROOT="${REMOTE_ROOT:-\$SCRATCH/open-course-rag-benchmark/repo}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

rsync -avz \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='data/raw/downloads' \
  --exclude='data/processed/*.jsonl' \
  --exclude='results/**/*.json' \
  --exclude='results/**/*.csv' \
  "$LOCAL_ROOT/" "$REMOTE_HOST:$REMOTE_ROOT/"

