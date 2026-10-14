#!/bin/bash
# Copy every cluster run's small result files to the laptop, plus the job
# logs. Weights stay on the cluster (copy one with rsync when it's needed,
# e.g. for the store photos in M1 D3).
#
#   scripts/pull_cluster_results.sh
#
# Needs the `bbhpc` SSH alias (docs/m1-date-detector.md, A1).
set -euo pipefail
cd "$(dirname "$0")/.."

rsync -a --prune-empty-dirs \
    --include '*/' --include 'run.json' --include 'val.json' --include 'data.yaml' \
    --include '*.json' --exclude '*' \
    bbhpc:bestbefore/results/runs/ results/runs/
rsync -a bbhpc:bestbefore/slurm/logs/ slurm/logs/

echo "runs now on the laptop:"
ls -1 results/runs | grep -- '-m1-' || echo "  (none yet)"
