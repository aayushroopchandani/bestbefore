#!/bin/bash
# M2b S14: ExpDate test, once, for the M2b detector's three seeds (n1280-ft
# s0-s2, each at its threshold_store, with the counts at M2's rule's
# threshold_m2 beside them from the same pass), on a CPU node
# (docs/m2b-store-and-size.md, S14: the heat rule keeps only store set v2's
# pass on the laptop). ExpDate's photos only: no store photo is read here.
#
#   cd ~/bestbefore && sbatch slurm/m2b_test_cpu.sbatch
#
# The dry run first: every run found, its weights matching m2b.LAST_PT_SHA256
# and none scored on real-test yet, or the job stops there. eval_detector.py
# refuses a run already scored, and the job isn't requeued if its node
# fails, so nothing is scored twice. Then, on the laptop, copy each run's
# real-test.json back and compare it with the cluster's.
#SBATCH --job-name=m2b-test-cpu
#SBATCH --partition=short
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
cd ~/bestbefore
module load Python/3.11.11      # the venv needs the module's libpython
source .venv/bin/activate
export YOLO_OFFLINE=1           # compute nodes have no internet: never try
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"   # PyTorch's threads: the job's CPUs, not the node's
echo "started $(date '+%F %T') on $(hostname)"
python scripts/eval_detector.py --set real-test --dry-run --runs m2b-n1280-ft-s0 m2b-n1280-ft-s1 m2b-n1280-ft-s2
python scripts/eval_detector.py --set real-test --device cpu --runs m2b-n1280-ft-s0 m2b-n1280-ft-s1 m2b-n1280-ft-s2
echo "ended $(date '+%F %T')"
