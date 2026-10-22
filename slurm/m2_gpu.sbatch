#!/bin/bash
# Holds a place in the GPU queue for M2's training (docs/m2-synthetic-generator.md, S1).
#
#   cd ~/bestbefore && sbatch --hold slurm/m2_gpu.sbatch   # twice: jobs A and B
#   scontrol release <job id>                              # once its work is ready
#
# The queue is first come, first served by submit time, and a release
# keeps the submit time, so a held job keeps its place without starting
# early. Slurm copies this script at submission but reads
# slurm/m2_chain.sh when the job starts, so the chain can be written
# after submitting. If it isn't there yet, the job exits at once rather
# than keep a GPU idle.
#SBATCH --job-name=m2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x-%j.out

echo "started $(date '+%F %T') on $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
df -h /tmp | sed 's/^/local disk: /'
if [ ! -f slurm/m2_chain.sh ]; then
    echo "slurm/m2_chain.sh is not there yet: exiting without using the GPU"
    exit 1
fi
exec bash slurm/m2_chain.sh
