#!/bin/bash
# M2c's CPU job, one script, a task per argument (docs/m2c-live-boxes.md).
#
# Tasks:
#   parity  S1: the ONNX files made on the laptop, copied into
#           $CHECKPOINT_DIR/m2c/, checked against their manifest, and M2b's
#           last.pt against m2b.py (scripts/export_live.py check); int8
#           calibrated twice, MinMax and Percentile 99.999, on 200 training
#           photos (export_live.py int8); then 7 variants over the five
#           real sets, 570 photos, one a call (scripts/live_parity.py run),
#           and R0, R1 and R2's accuracy half read (live_parity.py rules).
#           Everything made is the owner's only (umask 077): the int8 files
#           were calibrated on store photos, and v1-val's records hold boxes
#           on them.
#
# Submitting, from campus (the cluster can't be reached from home):
#   1. On the laptop, after the commits are pushed, copy the float files and
#      their manifest (scp doesn't expand braces in a remote path; local
#      paths are the laptop's shell's, so list them):
#        ssh bbhpc 'mkdir -p -m 700 ~/bestbefore-data/checkpoints/m2c'
#        scp data/checkpoints/m2c/det-fp32.onnx data/checkpoints/m2c/det-fp16.onnx \
#            data/checkpoints/m2c/det-fp16-head32.onnx data/checkpoints/m2c/manifest.json \
#            bbhpc:bestbefore-data/checkpoints/m2c/
#        ssh bbhpc 'chmod 600 ~/bestbefore-data/checkpoints/m2c/*'
#   2. On the cluster: cd ~/bestbefore && git pull. It adds M2c's files and
#      detector.predict's rect (default unchanged); M3's held jobs will run
#      that detector.py when released, unchanged in what they call.
#   3. squeue --me: 5 jobs submitted per user, every project's counted, M3's
#      two held GPU jobs included. Submit only if 4 or fewer are listed.
#   4. cd ~/bestbefore && sbatch slurm/m2c_cpu.sbatch parity
#
# Fetching the results back to the laptop (scp doesn't expand braces; tar does):
#   ssh bbhpc 'cd ~/bestbefore && tar cf - results/runs/m2c/RUN' | tar xf -
#   ssh bbhpc 'cd ~/bestbefore-data/checkpoints/m2c && tar cf - det-int8.onnx det-int8-pct.onnx manifest.json' \
#       | (cd data/checkpoints/m2c && tar xf -)
#   python scripts/export_live.py check --variant fp32 --variant fp16 --variant fp16-head32 \
#       --variant int8 --variant int8-pct
#SBATCH --job-name=m2c-cpu
#SBATCH --partition=short
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
cd ~/bestbefore

module load Python/3.11.11      # the venv needs the module's libpython
source .venv/bin/activate
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"   # PyTorch's threads: the job's CPUs, not the node's
# Torch's OpenMP threads sleep rather than spin between steps. Ultralytics'
# ONNX backend pre- and post-processes with torch, and spinning OpenMP
# threads beside ONNX Runtime's own made a 52 ms inference a 328 to 491 ms
# call on 24 CPUs of cn1; passive, 72 ms (S1, jobs 74149 to 74152).
export OMP_WAIT_POLICY=PASSIVE
export YOLO_OFFLINE=1                           # compute nodes have no internet: never try
export YOLO_AUTOINSTALL=false                   # and never pip-install: a missing package fails loudly

echo "started $(date '+%F %T') on $(hostname), task ${1:-none}, commit $(git rev-parse --short HEAD)"
case "${1:-}" in
parity)
    umask 077
    MODELS=$(python -c 'from bestbefore import live; print(live.models_dir())')
    python scripts/export_live.py check
    python scripts/export_live.py int8 --threads "$SLURM_CPUS_PER_TASK"
    # The run's output into the log and a file, whose last line is the run's
    # folder. Not `tee /dev/stderr`: in a job stderr is this log, and tee
    # reopens it truncated (job 74149 lost its first lines).
    OUT=$(mktemp)
    python scripts/live_parity.py run --threads "$SLURM_CPUS_PER_TASK" | tee "$OUT"
    RUN=$(tail -n 1 "$OUT")
    rm -f "$OUT"
    python scripts/live_parity.py rules "$RUN"
    open=$(find "$RUN" "$MODELS" -perm /077 | wc -l)
    echo "entries open to others in the run and the model folder: $open"
    ;;
*)
    echo "unknown task '${1:-}': one of parity"
    exit 2
    ;;
esac
echo "ended $(date '+%F %T')"
