#!/bin/bash
# M2's derived sets on the cluster (docs/m2-synthetic-generator.md, S2):
# the turned copies of the training photos, the two hard validation sets,
# and the synthetic list without the 14 copies.
#
#   cd ~/bestbefore && sbatch slurm/m2_derive.sbatch
#
# CPU only. Needs GENERATED_DIR in .env (~/bestbefore-data/generated).
#SBATCH --job-name=m2-derive
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
cd ~/bestbefore

module load Python/3.11.11      # the venv needs the module's libpython
source .venv/bin/activate

echo "started $(date '+%F %T') on $(hostname), $SLURM_CPUS_PER_TASK CPUs"
python scripts/derive_sets.py --workers "$SLURM_CPUS_PER_TASK"
echo "finished $(date '+%F %T')"
