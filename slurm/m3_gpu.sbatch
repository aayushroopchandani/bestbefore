#!/bin/bash
# Holds a place in the GPU queue for M3, submitted before its code exists
# (docs/m3-date-reader.md, S1).
#
#   cd ~/bestbefore && sbatch --hold slurm/m3_gpu.sbatch   # once or twice: S1's queue rule
#   scontrol release <job id>                              # S9, once M3's chain is ready
#
# The queue is first come, first served by submit time, and a release keeps
# the submit time, so a held job keeps its place without starting early.
# Slurm copies this script at submission but reads slurm/m3_chain.sh when
# the job starts, so the chain can be written after submitting. If it isn't
# there yet, the job exits at once, before it touches the GPU.
#SBATCH --job-name=m3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm/logs/%x-%j.out

echo "started $(date '+%F %T') on $(hostname)"
if [ ! -f slurm/m3_chain.sh ]; then
    echo "slurm/m3_chain.sh is not there yet: exiting without using the GPU"
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
exec bash slurm/m3_chain.sh
