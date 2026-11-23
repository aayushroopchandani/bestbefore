#!/bin/bash
# M3 S7: an HF Qwen3-VL folder as GGUF for llama.cpp (docs/m3-date-reader.md,
# "Qwen3-VL-2B", decision 16): the language model at Q8_0 (and at each other
# type asked for: bf16 is the check's diagnosis), the projector at F16, with
# convert_hf_to_gguf.py from llama.cpp's own tree (~/opt/llama.cpp, b12017)
# run in the main venv, which holds what it needs (torch, transformers,
# numpy; sentencepiece since S7: on a Qwen model it imports it before falling
# back to the BPE vocabulary). Its gguf package comes from the tree's gguf-py.
# Each file's log and SHA-256 go beside it.
#
#   scripts/vlm_gguf.sh HF_DIR OUT_DIR [TYPE ...]      TYPE: q8_0 (the default), bf16, f16, f32
set -euo pipefail
if [ $# -lt 2 ]; then
    echo "usage: $0 HF_DIR OUT_DIR [TYPE ...]"
    exit 2
fi
HF=$1
OUT=$2
shift 2
TYPES=("$@")
[ ${#TYPES[@]} -gt 0 ] || TYPES=(q8_0)
SRC=${LLAMA_SRC:-$HOME/opt/llama.cpp}
PY=${PYTHON:-.venv/bin/python}
if [ -e "$OUT" ]; then
    echo "$OUT exists"
    exit 1
fi
mkdir -p "$OUT"
echo "from $HF with llama.cpp $(git -C "$SRC" describe --tags) ($(git -C "$SRC" rev-parse --short HEAD))," \
     "$(date '+%F %T')" > "$OUT/made.txt"
for t in "${TYPES[@]}"; do
    start=$(date +%s)
    "$PY" "$SRC/convert_hf_to_gguf.py" "$HF" --outtype "$t" --outfile "$OUT/model-$t.gguf" \
        > "$OUT/convert-$t.log" 2>&1
    echo "model-$t.gguf in $(( $(date +%s) - start )) s" | tee -a "$OUT/made.txt"
done
start=$(date +%s)
"$PY" "$SRC/convert_hf_to_gguf.py" "$HF" --mmproj --outtype f16 --outfile "$OUT/mmproj-f16.gguf" \
    > "$OUT/convert-mmproj.log" 2>&1
echo "mmproj-f16.gguf in $(( $(date +%s) - start )) s" | tee -a "$OUT/made.txt"
(cd "$OUT" && sha256sum ./*.gguf 2>/dev/null || shasum -a 256 ./*.gguf) > "$OUT/SHA256SUMS"
ls -l "$OUT"/*.gguf
