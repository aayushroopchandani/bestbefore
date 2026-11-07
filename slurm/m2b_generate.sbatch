#!/bin/bash
# M2b's generated data and its 180° sets, checked (docs/m2b-store-and-size.md,
# S9). One job, as M2's (slurm/m2_generate.sbatch): every job of ours counts
# toward the account's caps of 5 submitted and 2 running. It releases
# nothing: job B also waits for S10's code.
#
#   cd ~/bestbefore && python scripts/fetch_fonts.py   # login node first: compute nodes have no internet
#   sbatch slurm/m2b_generate.sbatch
#
# In order: gen-v2's render sheet (seed 0) and gen-v2-sample (500 photos,
# seed 6), both to compare with the laptop's; M2b's tasks 0 to 200
# (gen-v2's 200,000 photos, then synth-val-v2's 2,400) on one worker per
# CPU; their check, which writes the lists and the manifests that step 2
# waits for, and stops the job, saying why, if anything is wrong; then the
# three 180° sets. Nothing is downloaded (the check compares the fonts with
# their pinned SHA-256s), and nothing of M2's is written: M2's manifests and
# lists are hashed at the start and at the end, to show it.
#
# Sized from the laptop's trial (S9): a worker peaked at 885 MB (M2's at 773
# on the laptop, 900 on the cluster), so 24 near 25 GB if all peak at once,
# and the check's main process about 3.3 GB for 200,000 sidecar lines; 48 GB
# (the owner's call) also leaves room for the page cache of 42 GB written.
# Photos at about 160 to 225 ms each per worker on 24 CPUs take 22 to 31
# minutes; with the check, the 180-degree sets and slow file opens, under an
# hour of the 4 asked (short allows 6).
#SBATCH --job-name=m2b-generate
#SBATCH --partition=short
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
cd ~/bestbefore

module load Python/3.11.11      # the venv needs the module's libpython
source .venv/bin/activate

m2_files() { sha256sum ~/bestbefore-data/generated/m2/manifests/*.json ~/bestbefore-data/generated/m2/lists/*.txt; }

echo "started $(date '+%F %T') on $(hostname), $SLURM_CPUS_PER_TASK CPUs, commit $(git rev-parse --short HEAD)"
m2_files > "slurm/logs/m2b-generate-$SLURM_JOB_ID-m2-before.sha256"
python scripts/render_sheet.py --v2
python scripts/generate.py --set gen-v2-sample --count 500 --workers "$SLURM_CPUS_PER_TASK"
python scripts/generate.py --milestone m2b --tasks 0-200 --workers "$SLURM_CPUS_PER_TASK"
python scripts/generate.py --milestone m2b --check --workers "$SLURM_CPUS_PER_TASK"
python scripts/derive_sets.py --only rot180 --workers "$SLURM_CPUS_PER_TASK"
m2_files | diff "slurm/logs/m2b-generate-$SLURM_JOB_ID-m2-before.sha256" - && echo "M2's manifests and lists: unchanged"
echo "finished $(date '+%F %T')"
