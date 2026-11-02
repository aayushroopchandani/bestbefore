#!/bin/bash
# M2b's smoke run on a CPU node, before GPU job A is released
# (docs/m2b-store-and-size.md, S2): yolo11s at 1,280 px, 1 epoch, 1% of
# gen-rot-1280's own lists, through train_detector.py's m2b path. The
# laptop can't hold it: a batch of 16 photos at 1,280 px keeps about 26 GiB
# for the backward pass.
#
#   cd ~/bestbefore && sbatch slurm/m2b_smoke_cpu.sbatch
#SBATCH --job-name=m2b-smoke-cpu
#SBATCH --partition=short
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
cd ~/bestbefore

module load Python/3.11.11      # the venv needs the module's libpython
source .venv/bin/activate
export YOLO_OFFLINE=1           # compute nodes have no internet: never try
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"   # PyTorch's threads: the job's CPUs, not the node's

echo "started $(date '+%F %T') on $(hostname)"
python scripts/train_detector.py --milestone m2b --arm s1280 --epochs 1 --fraction 0.01 \
    --device cpu --name smoke-cpu --workers 8
echo "ended $(date '+%F %T')"
