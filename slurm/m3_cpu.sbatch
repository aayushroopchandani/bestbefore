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
*)
    echo "unknown task '${1:-}': one of env"
    exit 2
    ;;
esac
echo "ended $(date '+%F %T')"
