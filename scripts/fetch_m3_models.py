"""Fetch M3's pretrained models at pinned revisions, and check them.

    python scripts/fetch_m3_models.py           # fetch what's missing, then check every file
    python scripts/fetch_m3_models.py --check   # check only

Compute nodes have no internet, so this runs on the cluster's login node
(docs/m3-date-reader.md, S1), in the main venv (huggingface_hub comes with
the vlm extra). Into $CHECKPOINT_DIR/pretrained/, beside the YOLO weights:

  paddlex/official_models/PP-OCRv6_small_rec, PP-OCRv6_small_det,
      PP-LCNet_x1_0_textline_ori      PaddleOCR's own models, found there when
                                      PADDLE_PDX_CACHE_HOME=<that>/paddlex
  PP-OCRv6_small_rec_onnx, PP-OCRv6_small_rec_safetensors
  PP-OCRv6_small_rec_pretrained.pdparams   training weights (S6's Paddle route)
  Qwen3-VL-2B-Instruct                the HF snapshot
  Qwen3-VL-2B-Instruct-GGUF           Q8_0 and the F16 projector only

Every Hugging Face repo is pinned to a commit, and every weights file to its
SHA-256 (docs/datasets.md lists all files). Prints each file's SHA-256 in
full; exits 1 on any mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from bestbefore.config import checkpoint_dir

# repo, revision, folder under pretrained/, files to fetch (None: all)
REPOS = (
    ("PaddlePaddle/PP-OCRv6_small_rec", "bd619643acac4b9650c040234da8d944476ee3f1",
     "paddlex/official_models/PP-OCRv6_small_rec", None),
    ("PaddlePaddle/PP-OCRv6_small_det", "106c97591b235f607453300d9fc8c1cad1b25488",
     "paddlex/official_models/PP-OCRv6_small_det", None),
    ("PaddlePaddle/PP-LCNet_x1_0_textline_ori", "cd237a44b0e359d4fe38310a416203cf7403faa5",
     "paddlex/official_models/PP-LCNet_x1_0_textline_ori", None),
    ("PaddlePaddle/PP-OCRv6_small_rec_onnx", "b8f84f0b80c529de40b4fbb3544b84fa7233a513",
     "PP-OCRv6_small_rec_onnx", None),
    ("PaddlePaddle/PP-OCRv6_small_rec_safetensors", "fe049fb103f57443fe8840c54ed06b702f3c1de5",
     "PP-OCRv6_small_rec_safetensors", None),
    ("Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203",
     "Qwen3-VL-2B-Instruct", None),
    ("Qwen/Qwen3-VL-2B-Instruct-GGUF", "52d6c8ffea26cc873ac5ad116f8631268d7eb503",
     "Qwen3-VL-2B-Instruct-GGUF",
     ["Qwen3VL-2B-Instruct-Q8_0.gguf", "mmproj-Qwen3VL-2B-Instruct-F16.gguf", "README.md"]),
)
PDPARAMS_URL = ("https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/"
                "PP-OCRv6_small_rec_pretrained.pdparams")

# Every weights file, as fetched on 2026-11-12 (LFS hashes, checked against Hugging Face's).
SHA256 = {
    "paddlex/official_models/PP-OCRv6_small_rec/inference.pdiparams":
        "406e1e689c9a7fbb04178007a3fe10cf852afe7bf8bb3bc6dbb9d532b13bd907",
    "paddlex/official_models/PP-OCRv6_small_rec/inference.yml":
        "ab078671bb49f06228eadccd34f1bb501e157f7a047095ffb943ba81512c77d1",
    "paddlex/official_models/PP-OCRv6_small_det/inference.pdiparams":
        "5043d4ccc8d63402ccea8feefcee4db57077431a873e78d2191836a178a492da",
    "paddlex/official_models/PP-LCNet_x1_0_textline_ori/inference.pdiparams":
        "0de2bcf996cf553e2b848dd7b1769dafffc6917b1ccdf55c1d8efe7909fbf743",
    "PP-OCRv6_small_rec_onnx/inference.onnx":
        "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634",
    "PP-OCRv6_small_rec_onnx/inference.yml":
        "ab078671bb49f06228eadccd34f1bb501e157f7a047095ffb943ba81512c77d1",
    "PP-OCRv6_small_rec_safetensors/model.safetensors":
        "f65a332afe5aa663f0b9d5706f4ae8457b5b4058a842d5c1eb22df505c27d642",
    "Qwen3-VL-2B-Instruct/model.safetensors":
        "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0",
    "Qwen3-VL-2B-Instruct/tokenizer.json":
        "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
    "Qwen3-VL-2B-Instruct-GGUF/Qwen3VL-2B-Instruct-Q8_0.gguf":
        "1e8db19207c8ce0733ddd78c2eff8a9e22c27c82f4443df94c25792ed8fe04f2",
    "Qwen3-VL-2B-Instruct-GGUF/mmproj-Qwen3VL-2B-Instruct-F16.gguf":
        "c3d5afbef5287953acd57b4043d2269456e5761a4eaccb3b71b062996970aea5",
    "PP-OCRv6_small_rec_pretrained.pdparams":
        "25c9bd54b0e5900916e8bb6ada938abeffb1eac1baedac0ca54a45b1c9310825",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def fetch(target: Path) -> None:
    from huggingface_hub import snapshot_download

    for repo, revision, folder, files in REPOS:
        snapshot_download(repo_id=repo, revision=revision, local_dir=target / folder,
                          allow_patterns=files)
    path = target / PDPARAMS_URL.rsplit("/", 1)[1]
    if not path.exists():
        with urllib.request.urlopen(PDPARAMS_URL, timeout=60) as r:
            path.write_bytes(r.read())


def check(target: Path) -> bool:
    ok = True
    for name, want in SHA256.items():
        path = target / name
        got = sha256(path) if path.exists() else "missing"
        ok &= got == want
        print(f"{'ok ' if got == want else 'BAD'} {got}  {name}")
    for _, _, folder, _ in REPOS:      # every other file: printed, pinned by its repo's revision
        for path in sorted(p for p in (target / folder).rglob("*")
                           if p.is_file() and ".cache" not in p.parts):
            name = path.relative_to(target).as_posix()
            if name not in SHA256:
                print(f"    {sha256(path)}  {name}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="hash only, fetch nothing")
    args = parser.parse_args()
    target = checkpoint_dir() / "pretrained"
    target.mkdir(parents=True, exist_ok=True)
    if not args.check:
        fetch(target)
    if not check(target):
        sys.exit("a file is missing or differs from its pinned SHA-256")
    print(f"all {len(SHA256)} weights files match")


if __name__ == "__main__":
    main()
