#!/bin/bash
# Holds two places in the GPU queue for M2b, submitted before its code exists.
#
#   cd ~/bestbefore && sbatch --hold slurm/m2b_gpu.sbatch   # twice
#   scontrol release <job id>                               # once M2b's code and data are ready
#
# The queue is first come, first served by submit time, and a release keeps
# the submit time, so a held job keeps its place without starting early.
# Slurm copies this script at submission but reads slurm/m2b_chain.sh when
# the job starts, so the chain can be written after submitting. If it isn't
# there yet, the job exits at once rather than keep a GPU idle.
#SBATCH --job-name=m2b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm/logs/%x-%j.out

echo "started $(date '+%F %T') on $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
if [ ! -f slurm/m2b_chain.sh ]; then
    echo "slurm/m2b_chain.sh is not there yet: exiting without using the GPU"
    exit 1
fi
exec bash slurm/m2b_chain.sh
