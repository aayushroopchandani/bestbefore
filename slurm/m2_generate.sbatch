#!/bin/bash
# M2's generated data, checked, then GPU job B released
# (docs/m2-synthetic-generator.md, S6). One job, not an array: every job of
# ours counts toward the account's caps of 5 submitted and 2 running.
#
#   cd ~/bestbefore && python scripts/fetch_fonts.py   # login node first: compute nodes have no internet
#   sbatch slurm/m2_generate.sbatch
#
# In order: the render sheet (seed 0), to compare with the laptop's byte for
# byte; tasks 0 to 50 (gen photos 0 to 49,999, then synth-val's 2,100) on
# one worker per CPU; the check, which writes the lists and the manifests
# the GPU chain waits for; and, only if the check passes, the release of
# every held m2 job. A failed check releases nothing and says why below.
# Nothing is downloaded: the check compares the fonts with their pinned
# SHA-256s instead.
#
# Sized from the laptop (S6): a worker peaks at about 1.1 GB (its font cache
# and the largest dot-matrix blocks), so 24 x 1.5 GB plus the check's
# sidecars; 52,100 photos at 55 to 120 ms each take minutes on 24 CPUs, and
# writing and reading back about 10 GB on the shared filesystem perhaps 20.
#SBATCH --job-name=m2-generate
#SBATCH --partition=short
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
cd ~/bestbefore

module load Python/3.11.11      # the venv needs the module's libpython
source .venv/bin/activate

echo "started $(date '+%F %T') on $(hostname), $SLURM_CPUS_PER_TASK CPUs, commit $(git rev-parse --short HEAD)"
python scripts/render_sheet.py --seed 0
python scripts/generate.py --tasks 0-50 --workers "$SLURM_CPUS_PER_TASK"
python scripts/generate.py --check --workers "$SLURM_CPUS_PER_TASK" --release m2
squeue --me --name=m2 --format="%.10i %.9T %.20r"
echo "finished $(date '+%F %T')"
