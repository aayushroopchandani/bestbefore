#!/bin/bash
# M2b's smoke runs on a CPU node (docs/m2b-store-and-size.md): the laptop
# can't hold them, as a batch of 16 photos at 1,280 px keeps about 26 GiB
# for s's backward pass (S2).
#
#   cd ~/bestbefore && sbatch slurm/m2b_smoke_cpu.sbatch          # step 1 (S2)
#   cd ~/bestbefore && sbatch slurm/m2b_smoke_cpu.sbatch step2    # step 2 (S10)
#
# step1: yolo11s at 1,280 px, 1 epoch, 1% of gen-rot-1280's own lists,
#        through train_detector.py's m2b path, before GPU job A's release.
# step2: n1280-ft and n1280-ft2 from the control's seed 0, one epoch a stage
#        on an epoch of 400 photos drawn by m2b.mix (store lines and every
#        replay pool), before GPU job B's release; ft2's run records how its
#        frozen stage moved the frozen layers.
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

task="${1:-step1}"
echo "started $(date '+%F %T') on $(hostname): $task"
case "$task" in
    step1)
        python scripts/train_detector.py --milestone m2b --arm s1280 --epochs 1 --fraction 0.01 \
            --device cpu --name smoke-cpu --workers 8
        ;;
    step2)
        for arm in n1280-ft n1280-ft2; do
            python scripts/train_detector.py --milestone m2b --arm "$arm" --seed 0 --epochs 1 \
                --epoch-photos 400 --device cpu --name "smoke-$arm-cpu" --workers 8
            echo "$arm done $(date '+%F %T')"
        done
        ;;
    *)
        echo "unknown task $task: step1 or step2"
        exit 1
        ;;
esac
echo "ended $(date '+%F %T')"
