#!/bin/bash
# Copy every cluster run's small result files to the laptop, plus the job
# logs. Weights and photos stay on the cluster (copy one with rsync when
# it's needed, e.g. for the store photos in M1 D3 and M2 S8).
#
#   scripts/pull_cluster_results.sh
#
# Brings every JSON file of results/runs/: run.json, val.json (M1),
# suite.json (M2), suite-m2b.json (M2b), the scores on a test set, the
# chains' decisions and the generation check records. Each M2 and M2b run's
# per-epoch log, Ultralytics' results.csv, sits beside its weights on the
# cluster; it lands in the run's own folder here, where the loader rule and
# the timings read it (an M2b two-stage run's first stage in <id>-stage1/).
# Never an image: training mosaics of store photos stay on the cluster, and
# no run keeps a box of a store photo there.
#
# Needs the `bbhpc` SSH alias (docs/m1-date-detector.md, A1).
set -euo pipefail
cd "$(dirname "$0")/.."

rsync -a --prune-empty-dirs \
    --include '*/' --include 'run.json' --include 'val.json' --include 'data.yaml' \
    --include '*.json' --exclude '*' \
    bbhpc:bestbefore/results/runs/ results/runs/
for milestone in m2 m2b; do
    rsync -a --prune-empty-dirs \
        --include '/*/' --include '/*/results.csv' --exclude '*' \
        "bbhpc:bestbefore-data/checkpoints/$milestone/" results/runs/
done
rsync -a bbhpc:bestbefore/slurm/logs/ slurm/logs/

echo "runs now on the laptop:"
ls -1 results/runs | grep -E -- '-m(1|2|2b)-' || echo "  (none yet)"
