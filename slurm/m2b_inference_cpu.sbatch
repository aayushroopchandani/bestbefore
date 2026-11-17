#!/bin/bash
# M2b S13: R3's arm's prediction passes on real-val, on a CPU node
# (docs/m2b-store-and-size.md, S13): 3 seeds x 4 turns x 1,280 and 1,536
# px on ExpDate's 165 validation photos, for R4's real-val check and the AP
# check against Ultralytics. No store photo is read here; v1-val's passes
# are made on the laptop, and the add-ons are fused and scored there.
#
#   cd ~/bestbefore && sbatch slurm/m2b_inference_cpu.sbatch
#
# Then, on the laptop, copy the run's folder back (its run.json and
# passes.json.gz) and run scripts/m2b_inference.py score.
#SBATCH --job-name=m2b-inference-cpu
#SBATCH --partition=short
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
cd ~/bestbefore
module load Python/3.11.11      # the venv needs the module's libpython
source .venv/bin/activate
export YOLO_OFFLINE=1           # compute nodes have no internet: never try
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"   # PyTorch's threads: the job's CPUs, not the node's
echo "started $(date '+%F %T') on $(hostname)"
python scripts/m2b_inference.py predict --set real-val --device cpu
echo "ended $(date '+%F %T')"
