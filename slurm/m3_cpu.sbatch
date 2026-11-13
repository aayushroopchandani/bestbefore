#!/bin/bash
# M3's CPU jobs, one script, a task per argument (docs/m3-date-reader.md).
# The header fits S2's crop experiment; smaller tasks ask for less on the
# command line.
#
#   cd ~/bestbefore && sbatch --cpus-per-task=4 --mem=16G --time=00:30:00 slurm/m3_cpu.sbatch env
#
# Tasks:
#   env   S1: both venvs on a CPU node with no internet: imports, the local
#         models, one real-val crop read by PaddleOCR (recogniser alone and
#         full pipeline) and RapidOCR (scripts/check_m3_env.py); llama.cpp
#         generating 8 tokens from the Q8_0 GGUF, naming its CPU variant
#   crop-experiment
#         S2: untrained readers on hand-drawn date boxes (scripts/crop_experiment.py):
#         cut (main venv), read (ocr-cpu), score (main venv). Store crops and
#         per-crop reads stay in an owner-only folder under $GENERATED_DIR/m3/;
#         results/runs/ gets counts only, and R1 in m3-decisions.json.
#   crops-smoke [N]
#         S3's sizing: the first N photos (default 2000) of every pile, cut,
#         timed and checked in $GENERATED_DIR/m3/crops-smoke/ (made again each run)
#   crops S3: every crop pile into its LMDB shard in $GENERATED_DIR/m3/crops/
#         (scripts/make_crops.py), every record read back and the counts
#         checked against the sources (--check), a sheet of 64 crops a pile,
#         and the size on disk. ExpDate and generated crops only: the store
#         pile waits for S4's text. Sized from crops-smoke.
#
#   cd ~/bestbefore && sbatch slurm/m3_cpu.sbatch crop-experiment
#SBATCH --job-name=m3-cpu
#SBATCH --partition=short
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
cd ~/bestbefore

module load Python/3.11.11      # both venvs need the module's libpython
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"   # the job's CPUs, not the node's
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1  # compute nodes have no internet: never try
export YOLO_OFFLINE=1
MAIN=.venv/bin/python
OCR=~/venvs/ocr-cpu/bin/python
LLAMA=~/opt/llama.cpp/build/bin

echo "started $(date '+%F %T') on $(hostname), task ${1:-none}, commit $(git rev-parse --short HEAD)"
case "${1:-}" in
env)
    lscpu | grep "Model name"
    for flag in avx2 avx512f avx512_bf16 amx_tile; do   # which llama.cpp CPU variant fits
        grep -qw "$flag" /proc/cpuinfo && echo "cpu flag $flag: yes" || echo "cpu flag $flag: no"
    done
    DIR=$($MAIN -c 'from bestbefore.config import generated_dir; print(generated_dir() / "m3" / "env-check")')
    $MAIN scripts/check_m3_env.py main "$DIR"
    $OCR -I scripts/check_m3_env.py ocr "$DIR"
    "$LLAMA/llama-server" --version
    # 8 tokens of text from the official Q8_0 GGUF; the verbose log names the
    # CPU variant llama.cpp loaded on this node (the photo path is S7's).
    # Its AMX variant is kept out of build/bin: it dies with SIGILL (S1).
    PRE=$($MAIN -c 'from bestbefore.config import checkpoint_dir; print(checkpoint_dir() / "pretrained")')
    "$LLAMA/llama-completion" -m "$PRE/Qwen3-VL-2B-Instruct-GGUF/Qwen3VL-2B-Instruct-Q8_0.gguf" \
        -p "The expiry date 2027-03-01 is" -n 8 -t "$SLURM_CPUS_PER_TASK" --temp 0 -no-cnv -v \
        < /dev/null > "$DIR/llama.log" 2>&1
    grep -E "loaded CPU backend|AMX|eval time" "$DIR/llama.log" | grep -v "amx repack" || true
    echo "llama.cpp: ok"
    ;;
crop-experiment)
    umask 077                   # store crops: the owner's only (CLAUDE.md)
    RUN=$($MAIN scripts/crop_experiment.py cut | tail -n 1)
    $OCR -I scripts/crop_experiment.py read "$RUN" --threads "$SLURM_CPUS_PER_TASK"
    $MAIN scripts/crop_experiment.py score "$RUN"
    open=$(find "$RUN" -perm /077 | wc -l)
    echo "entries open to others in the run folder: $open"
    ;;
crops-smoke)
    ROOT="$($MAIN -c 'from bestbefore.config import generated_dir; print(generated_dir())')/m3/crops-smoke"
    rm -rf "$ROOT"
    $MAIN scripts/make_crops.py --pile all --limit "${2:-2000}" --root "$ROOT" --workers "$SLURM_CPUS_PER_TASK"
    $MAIN scripts/make_crops.py --check --pile all --root "$ROOT" --workers "$SLURM_CPUS_PER_TASK"
    du -sh "$ROOT"/*
    ;;
crops)
    ROOT="$($MAIN -c 'from bestbefore.config import generated_dir; print(generated_dir())')/m3/crops"
    $MAIN scripts/make_crops.py --pile all --workers "$SLURM_CPUS_PER_TASK"
    $MAIN scripts/make_crops.py --check --pile all --workers "$SLURM_CPUS_PER_TASK"
    for pile in expdate-real expdate-synth gen real-val real-val-rot real-val-rot180 synth-val synth-val-v2; do
        $MAIN scripts/crop_sheet.py --pile "$pile"
    done
    du -sh "$ROOT"/*
    ;;
*)
    echo "unknown task '${1:-}': one of env, crop-experiment, crops-smoke, crops"
    exit 2
    ;;
esac
echo "ended $(date '+%F %T')"
