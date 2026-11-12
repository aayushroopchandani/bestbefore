"""M3 S1's check of the cluster's two environments, run on a CPU node (no
internet), by `slurm/m3_cpu.sbatch env` (docs/m3-date-reader.md, S1).

    python scripts/check_m3_env.py main DIR                  # the main venv
    ~/venvs/ocr-cpu/bin/python -I scripts/check_m3_env.py ocr DIR

`main`: the read and vlm extras import beside torch and Ultralytics;
Qwen3-VL-2B's config and processor load from the local snapshot; PP-OCRv6
small's recogniser opens in ONNX Runtime and its safetensors config loads;
the GGUF files are there. Then one real-val date crop is cut: the first
real-val photo by name with a `date` box that has text, the box with 0.15
of its short side as margin (a check, not S2's padding), saved as PNG in
DIR with check.json (its text and the model folders).

`ocr`: PaddleOCR reads that crop with the recogniser alone (PP-OCRv6 small)
and with its full pipeline (PP-OCRv6 small detector, the textline angle
classifier, the recogniser; the page orientation and unwarping models off),
and RapidOCR 1.2.3 with its own pipeline. Every model comes from local
files. This part imports nothing of bestbefore: the ocr-cpu venv doesn't
hold it.

Only ExpDate's real-val photos: never a store photo. A wrong read is
printed, not failed: the check is that every model loads and reads.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

MARGIN = 0.15
QWEN = "Qwen3-VL-2B-Instruct"
GGUF = ("Qwen3-VL-2B-Instruct-GGUF/Qwen3VL-2B-Instruct-Q8_0.gguf",
        "Qwen3-VL-2B-Instruct-GGUF/mmproj-Qwen3VL-2B-Instruct-F16.gguf")
PP_REC, PP_DET, PP_ORI = "PP-OCRv6_small_rec", "PP-OCRv6_small_det", "PP-LCNet_x1_0_textline_ori"


def versions(*modules) -> None:
    for m in modules:
        print(f"  {m.__name__} {getattr(m, '__version__', '?')}")


def folded(text: str) -> str:
    return "".join(text.split()).upper()


def main_part(out: Path) -> None:
    import accelerate
    import cv2
    import lmdb
    import numpy
    import onnx
    import onnxruntime
    import peft
    import PIL
    import torch
    import transformers
    import ultralytics
    from transformers import AutoConfig, AutoProcessor

    from bestbefore import images, sets
    from bestbefore.config import checkpoint_dir

    print("main venv:")
    versions(torch, ultralytics, numpy, PIL, cv2, transformers, peft, accelerate, lmdb, onnx,
             onnxruntime)
    pre = checkpoint_dir() / "pretrained"

    t0 = time.time()
    config = AutoConfig.from_pretrained(pre / QWEN, local_files_only=True)
    processor = AutoProcessor.from_pretrained(pre / QWEN, local_files_only=True)
    ip = processor.image_processor
    print(f"{QWEN}: {config.model_type}, {type(processor).__name__} / {type(ip).__name__}, "
          f"patch {getattr(ip, 'patch_size', '?')}, merge {getattr(ip, 'merge_size', '?')}, "
          f"size {getattr(ip, 'size', '?')} ({time.time() - t0:.1f} s, offline)")

    rec = AutoConfig.from_pretrained(pre / f"{PP_REC}_safetensors", local_files_only=True)
    print(f"{PP_REC}_safetensors: model_type {rec.model_type}, architectures {rec.architectures}")
    session = onnxruntime.InferenceSession(str(pre / f"{PP_REC}_onnx" / "inference.onnx"),
                                           providers=["CPUExecutionProvider"])
    for kind, nodes in (("in", session.get_inputs()), ("out", session.get_outputs())):
        for n in nodes:
            print(f"{PP_REC}_onnx {kind}: {n.name} {n.shape} {n.type}")
    for name in (*GGUF, f"{PP_REC}_pretrained.pdparams"):
        print(f"{name}: {(pre / name).stat().st_size:,} bytes")

    sample, box = next((s, b) for s in sorted(sets.load_set("real-val"), key=lambda s: s.id)
                       for b in s.boxes if b.label == "date" and (b.text or "").strip())
    image = images.load_upright(sample.image)
    w, h = image.size
    x1, y1, x2, y2 = box.xyxy[0] * w, box.xyxy[1] * h, box.xyxy[2] * w, box.xyxy[3] * h
    m = MARGIN * min(x2 - x1, y2 - y1)
    xyxy = [max(0, round(x1 - m)), max(0, round(y1 - m)), min(w, round(x2 + m)), min(h, round(y2 + m))]
    out.mkdir(parents=True, exist_ok=True)
    image.crop(xyxy).save(out / "crop.png")
    info = {"photo": sample.id, "set": "real-val", "label": box.label, "text": box.text,
            "crop_xyxy": xyxy, "crop": "crop.png", "pretrained": str(pre),
            "paddlex_home": str(pre / "paddlex")}
    (out / "check.json").write_text(json.dumps(info, indent=1))
    print(f"crop: real-val {sample.id}, {xyxy[2] - xyxy[0]} x {xyxy[3] - xyxy[1]} px, "
          f"text {box.text!r} -> {out / 'crop.png'}")


def ocr_part(out: Path) -> None:
    info = json.loads((out / "check.json").read_text())
    os.environ["PADDLE_PDX_CACHE_HOME"] = info["paddlex_home"]     # read when paddlex is imported
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"   # no internet: never probe
    import cv2
    import numpy
    import onnxruntime
    import paddle
    import paddleocr
    import paddlex
    import PIL
    from paddleocr import PaddleOCR, TextRecognition
    from rapidocr_onnxruntime import RapidOCR

    print("ocr-cpu venv:")
    versions(paddle, paddleocr, paddlex, onnxruntime, numpy, PIL, cv2)
    crop, truth = str(out / info["crop"]), info["text"]
    print(f"crop: real-val {info['photo']}, text {truth!r}")

    def report(what: str, texts: list[str], scores: list[float], seconds: float) -> None:
        read = " | ".join(texts)
        exact = folded(read) == folded(truth)
        print(f"{what}: {read!r} scores {[round(float(s), 3) for s in scores]} "
              f"exact {exact} ({seconds:.2f} s)")

    rec = TextRecognition(model_name=PP_REC, device="cpu")
    t0 = time.time()
    (r,) = rec.predict(crop)
    report(f"paddleocr {PP_REC} alone", [r["rec_text"]], [r["rec_score"]], time.time() - t0)

    full = PaddleOCR(text_detection_model_name=PP_DET, text_recognition_model_name=PP_REC,
                     textline_orientation_model_name=PP_ORI, use_doc_orientation_classify=False,
                     use_doc_unwarping=False, use_textline_orientation=True, device="cpu")
    t0 = time.time()
    (r,) = full.predict(crop)
    report(f"paddleocr full ({PP_DET}, {PP_ORI}, {PP_REC}), {len(r['rec_polys'])} lines",
           list(r["rec_texts"]), list(r["rec_scores"]), time.time() - t0)

    engine = RapidOCR()
    t0 = time.time()
    result, _ = engine(crop)
    result = result or []
    report(f"rapidocr full, {len(result)} lines", [t for _, t, _ in result],
           [s for _, _, s in result], time.time() - t0)


if __name__ == "__main__":
    part, folder = sys.argv[1], Path(sys.argv[2])
    {"main": main_part, "ocr": ocr_part}[part](folder)
    print(f"{part}: ok")
