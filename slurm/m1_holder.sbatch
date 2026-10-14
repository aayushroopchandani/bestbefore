#!/bin/bash
# Holds our place in the GPU queue for all of M1's training.
#
#   cd ~/bestbefore && sbatch slurm/m1_holder.sbatch
#
# The queue on this cluster is first come, first served, so the job is
# submitted early. Slurm reads the files a job runs when the job STARTS,
# so slurm/m1_chain.sh (smoke test, then every training, back to back on
# this one GPU) can be written and pulled in after submitting. If it isn't
# there yet when the job starts, the job exits at once rather than keep a
# GPU idle.
#SBATCH --job-name=m1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x-%j.out

echo "started $(date '+%F %T') on $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

if [ ! -f slurm/m1_chain.sh ]; then
    echo "slurm/m1_chain.sh is not there yet: exiting without using the GPU"
    exit 1
fi
exec bash slurm/m1_chain.sh
