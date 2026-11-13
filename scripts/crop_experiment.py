"""M3 S2's crop experiment: untrained readers on hand-drawn date boxes
(docs/m3-date-reader.md, S2; experiments/m3-crop-reading/README.md).

    python scripts/crop_experiment.py cut [--limit N]        # prints the run folder
    ~/venvs/ocr-cpu/bin/python -I scripts/crop_experiment.py read RUN [--threads N]
    python scripts/crop_experiment.py score RUN

`cut` (main venv): real-val's date boxes (ExpDate) and every expiry and
mfg box with a value of store sets v1 and v3, training and validation, from
the cluster's export ($STORE_TRAIN_DIR) and its text.json. Never store set
v2, never ExpDate test. Each photo is read as M3 reads it, upright, its long
side at most 1,600 px; each box is cut at every padding in m3.PADS and
saved as PNG in RUN, an owner-only folder under $GENERATED_DIR/m3/
crop-experiment/ (store crops never go anywhere else). The run's record,
aggregates only, goes to results/runs/. --limit takes the first N crops of
each set: the laptop's smoke.

`read` (~/venvs/ocr-cpu): every crop at every padding through RapidOCR 1.2.3
and PaddleOCR 3.7's PP-OCRv6 small, each as its full pipeline and as its
recogniser alone (as cut, and at the crop's two quarter turns), one crop
at a time, every model from local files. Imports nothing of bestbefore:
that venv doesn't hold it.

`score` (main venv): every read through the frozen parser (m3.PARSE_SHA256),
right / wrong / none against each crop's truth, R1 (m3.crop_recipe) into
results/runs/m3-decisions.json. Per-crop rows stay in RUN; the summary in
results/runs/ holds counts only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PADDLE_MODELS = {"rec": "PP-OCRv6_small_rec", "det": "PP-OCRv6_small_det", "ori": "PP-LCNet_x1_0_textline_ori"}
CROPS, READS, SCORED = "crops.json", "reads.jsonl", "scored.jsonl"


# --- cut (main venv) --------------------------------------------------------------

def cut(limit: int | None) -> None:
    os.umask(0o077)                 # every file and folder made here is the owner's only
    import hashlib
    from PIL import Image
    from bestbefore import derived, m3, runs, sets, store_text, store_train
    from bestbefore.config import checkpoint_dir, generated_dir, store_train_dir
    from bestbefore.images import load_upright
    from bestbefore.reader import crops
    from bestbefore.store_train import YOLO_LABEL

    folder = store_train_dir()
    manifest_bytes = (folder / store_train.MANIFEST).read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    text_bytes = (folder / store_text.NAME).read_bytes()
    text = json.loads(text_bytes)
    if problems := store_text.check(text, manifest, manifest_sha256):
        sys.exit("text.json doesn't fit the export:\n  " + "\n  ".join(problems))

    config = {"pads": m3.PADS, "max_side": store_train.MAX_SIDE, "limit": limit,
              "export_manifest_sha256": manifest_sha256,
              "text_sha256": hashlib.sha256(text_bytes).hexdigest(),
              "code_sha256": runs.code_sha256(crops, store_text, sys.modules[__name__])}
    name = "m3-crop-experiment" + ("-smoke" if limit else "")
    record = runs.start_run(name, config)
    run = generated_dir() / "m3" / "crop-experiment" / record.name
    run.mkdir(parents=True)
    os.chmod(run, 0o700)
    (run / "crops").mkdir()

    records: list[dict] = []

    def add(image: Image.Image, box_px, key: str, **fields) -> None:
        files = {}
        for pad in m3.PADS:
            rect = crops.padded(box_px, pad, image.size)
            path = f"crops/{key}-p{pad:g}.png"
            image.crop(rect).save(run / path)
            files[f"{pad:g}"] = path
        records.append({"key": key, "shape": "tall" if crops.is_tall(box_px) else "wide",
                        "box_px": [round(v, 2) for v in box_px], "image_size": list(image.size),
                        "files": files, **fields})

    # real-val: ExpDate's validation photos (public); every date box has text.
    n = 0
    for s in sorted(sets.load_set("real-val"), key=lambda s: s.id):
        image = store_train.shrink(load_upright(s.image))
        for k, b in enumerate(s.boxes):
            if b.label != "date" or not (b.text or "").strip() or (limit and n >= limit):
                continue
            n += 1
            add(image, crops.to_pixels(b.xyxy, image.size), f"r{n:04d}", set="real-val", part="val",
                photo=s.id, box=k, label="date", tag=None, truth_text=b.text, truth=None)

    # store: v1 and v3 as exported, training (upright copies) and validation.
    n = 0
    files = manifest["files"]
    for photo_name in sorted(text["photos"]):
        entry, meta = text["photos"][photo_name], manifest["photos"][photo_name]
        wanted = [b for b in entry["boxes"] if b["label"] in store_text.VALUE_LABELS and b["value"]]
        if not wanted or (limit and n >= limit):
            continue
        rel = f"images/{next(iter(store_train.FOLDERS[meta['part']]))}/{photo_name}.jpg"
        data = (folder / rel).read_bytes()
        if hashlib.sha256(data).hexdigest() != files[rel]:
            sys.exit(f"{rel} differs from the export's manifest")
        image_path = folder / rel
        boxes = derived.read_boxes(image_path)
        if [b.label for b in boxes] != [YOLO_LABEL[label] for label in meta["labels"]]:
            sys.exit(f"{rel}: its label file doesn't follow the manifest")
        image = store_train.shrink(load_upright(image_path))
        for b in wanted:
            if limit and n >= limit:
                break
            n += 1
            add(image, crops.to_pixels(boxes[b["index"]].xyxy, image.size), f"s{n:04d}", set="store",
                part=meta["part"], photo=photo_name, box=b["index"], label=b["label"], tag=meta["tag"],
                truth_text=None, truth=b["value"])

    pre = checkpoint_dir() / "pretrained"
    out = {"run": str(record), "made": time.strftime("%Y-%m-%d %H:%M:%S"), **config,
           "paddlex_home": str(pre / "paddlex"), "crops": records}
    (run / CROPS).write_text(json.dumps(out, indent=1))
    counts = {s: sum(r["set"] == s for r in records) for s in ("real-val", "store")}
    runs.update_run(record, crop_folder=str(run), crops=counts)
    print(f"cut {counts} crops at pads {m3.PADS} into {run}", file=sys.stderr)
    print(run)


# --- read (~/venvs/ocr-cpu, nothing of bestbefore) ---------------------------------------

def read(run: Path, threads: int) -> None:
    os.umask(0o077)
    info = json.loads((run / CROPS).read_text())
    os.environ["PADDLE_PDX_CACHE_HOME"] = info["paddlex_home"]      # read when paddlex is imported
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"    # no internet: never probe
    import numpy as np
    import onnxruntime
    import rapidocr_onnxruntime.utils as rapid_utils
    from PIL import Image

    # RapidOCR 1.2.3 makes its ONNX Runtime sessions with default options,
    # which start a thread per core of the whole node and pin each (failing
    # inside a job's cgroup). Its sessions take their options from this name.
    def options() -> onnxruntime.SessionOptions:
        o = onnxruntime.SessionOptions()
        o.intra_op_num_threads = threads
        o.inter_op_num_threads = 1
        return o
    rapid_utils.SessionOptions = options
    # Every ONNX Runtime session any library makes here is recorded with its
    # threads, so one left to the node's count can't pass unseen.
    made: list[int | None] = []
    ort_init = onnxruntime.InferenceSession.__init__

    def counted(self, path_or_bytes, sess_options=None, *args, **kwargs):
        made.append(sess_options.intra_op_num_threads if sess_options else None)
        ort_init(self, path_or_bytes, sess_options, *args, **kwargs)
    onnxruntime.InferenceSession.__init__ = counted
    from rapidocr_onnxruntime import RapidOCR
    from paddleocr import PaddleOCR, TextRecognition

    t0 = time.time()
    rapid = RapidOCR()
    sessions = [rapid.text_detector.infer.session, rapid.text_cls.infer.session,
                rapid.text_recognizer.session.session]
    rapid_threads = [s.get_session_options().intra_op_num_threads for s in sessions]
    if rapid_threads != [threads] * 3:
        sys.exit(f"RapidOCR's sessions run {rapid_threads} threads, not {threads}")
    paddle_rec = TextRecognition(model_name=PADDLE_MODELS["rec"], device="cpu", cpu_threads=threads)
    paddle_full = PaddleOCR(text_detection_model_name=PADDLE_MODELS["det"],
                            text_recognition_model_name=PADDLE_MODELS["rec"],
                            textline_orientation_model_name=PADDLE_MODELS["ori"],
                            use_doc_orientation_classify=False, use_doc_unwarping=False,
                            use_textline_orientation=True, device="cpu", cpu_threads=threads)
    load_s = time.time() - t0
    if any(n != threads for n in made):
        sys.exit(f"ONNX Runtime sessions made with {made} threads, not {threads} each")

    def rapid_full(rgb):
        result, _ = rapid(rgb)
        lines = result or []
        return [t for _, t, _ in lines], [float(s) for _, _, s in lines]

    def rapid_alone(rgb):
        (text, score), = rapid.text_recognizer([rgb])[0]
        return [text], [float(score)]

    def paddle_full_read(rgb):
        (r,) = paddle_full.predict(np.ascontiguousarray(rgb[:, :, ::-1]))     # BGR, as from a file
        return list(r["rec_texts"]), [float(s) for s in r["rec_scores"]]

    def paddle_alone(rgb):
        (r,) = paddle_rec.predict(np.ascontiguousarray(rgb[:, :, ::-1]))
        return [r["rec_text"]], [float(r["rec_score"])]

    readers = {("rapidocr", "full"): rapid_full, ("rapidocr", "alone"): rapid_alone,
               ("ppocrv6", "full"): paddle_full_read, ("ppocrv6", "alone"): paddle_alone}
    t0, n = time.time(), 0
    with open(run / READS, "w") as out:
        for i, c in enumerate(info["crops"], 1):
            ks = (0, 1, 3) if c["shape"] == "tall" else (0, 2)
            for pad, path in c["files"].items():
                with Image.open(run / path) as im:
                    rgb = np.asarray(im.convert("RGB"))
                for (reader, mode), fn in readers.items():
                    for k in (ks if mode == "alone" else (0,)):
                        img = np.ascontiguousarray(np.rot90(rgb, k))
                        t1 = time.perf_counter()
                        texts, scores = fn(img)
                        ms = (time.perf_counter() - t1) * 1000
                        out.write(json.dumps({
                            "key": c["key"], "reader": reader, "mode": mode, "pad": float(pad), "k": k,
                            "text": " ".join(texts), "lines": len(texts), "scores": scores,
                            "score": sum(scores) / len(scores) if scores else None,
                            "ms": round(ms, 2)}, ensure_ascii=False) + "\n")
                        n += 1
            if i % 25 == 0 or i == len(info["crops"]):
                print(f"{i}/{len(info['crops'])} crops, {n} reads, {time.time() - t0:.0f} s", file=sys.stderr)

    import cv2
    import paddle
    import paddleocr
    import paddlex
    import rapidocr_onnxruntime
    versions = {"onnxruntime": onnxruntime.__version__, "paddle": paddle.__version__,
                "paddleocr": paddleocr.__version__, "paddlex": paddlex.__version__, "opencv": cv2.__version__,
                "numpy": np.__version__, "rapidocr_onnxruntime": "1.2.3", "python": sys.version.split()[0]}
    rapid_dir = Path(rapidocr_onnxruntime.__file__).parent
    (run / "read.json").write_text(json.dumps({
        "threads": threads, "rapidocr_session_threads": rapid_threads, "ort_sessions_made": made,
        "load_seconds": round(load_s, 1),
        "read_seconds": round(time.time() - t0, 1), "reads": n, "versions": versions,
        "rapidocr_models": sorted(p.name for p in (rapid_dir / "models").iterdir()),
        "paddle_models": PADDLE_MODELS, "host": os.uname().nodename}, indent=1))
    print(f"read {n} times in {time.time() - t0:.0f} s (models loaded in {load_s:.0f} s)", file=sys.stderr)


# --- score (main venv) ----------------------------------------------------------------

def table(summary: dict, groups: list[str]) -> str:
    from bestbefore.m3 import CROP_MODES, CROP_READERS, PADS
    head = "| reader | runs | pad | " + " | ".join(groups) + " | ms |"
    lines = [head, "|" + "---|" * 3 + "---:|" * (len(groups) + 1)]
    for reader in CROP_READERS:
        for mode in CROP_MODES:
            for pad in PADS:
                cfg = f"{reader}/{mode}/{pad:g}"
                cells = []
                for g in groups:
                    t = summary["counts"][cfg].get(g)
                    cells.append(f"{t['right']} / {t['wrong']} / {t['none']}" if t else "–")
                lines.append(f"| {reader} | {mode} | {pad:g} | " + " | ".join(cells) + f" | {summary['ms'][cfg]} |")
    return "\n".join(lines)


def score_run(run: Path) -> None:
    os.umask(0o077)
    from bestbefore import m3, runs
    from bestbefore.reader import crop_scores

    digest = m3.check_parser()
    info = json.loads((run / CROPS).read_text())
    with open(run / READS) as f:
        reads = [json.loads(line) for line in f]
    rows = crop_scores.score_rows(info["crops"], reads)
    with open(run / SCORED, "w") as out:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = crop_scores.tally(rows)
    groups = crop_scores.group_order(next(iter(counts.values())))
    recipe = m3.crop_recipe(rows)
    record = Path(info["run"])
    read_info = json.loads((run / "read.json").read_text())
    real = [c for c in info["crops"] if c["set"] == "real-val"]
    truths = [crop_scores.truth_of(c) for c in real]
    summary = {"parse_sha256": digest, "counts": counts, "ms": crop_scores.mean_ms(rows),
               "real_val_truths": {"crops": len(real), "no_date": sum(t is None for t, _ in truths),
                                   "whole_read_rule": sum(w for _, w in truths)},
               "crop_recipe": recipe, "read": read_info}
    runs.write_json(record / "summary.json", summary)
    runs.update_run(record, parse_sha256=digest, crop_recipe=recipe)
    if not info["limit"]:
        decisions = runs.read_json(m3.DECISIONS_PATH) or {}
        decisions["crop_recipe"] = {**recipe, "run": record.name, "parse_sha256": digest}
        runs.write_json(m3.DECISIONS_PATH, decisions)
    print(f"{record}\n\nright / wrong / none, by configuration and group:\n")
    print(table(summary, groups))
    exact = {cfg: c["real-val"]["exact"] for cfg, c in counts.items() if "real-val" in c}
    print(f"\nexact text on real-val: {json.dumps(exact)}")
    print(f"real-val truths: {summary['real_val_truths']}")
    print(f"\nR1: {json.dumps(recipe)}" + ("" if info["limit"] else f"\n-> {m3.DECISIONS_PATH}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    steps = parser.add_subparsers(dest="step", required=True)
    p = steps.add_parser("cut")
    p.add_argument("--limit", type=int, help="the first N crops of each set (a smoke)")
    p = steps.add_parser("read")
    p.add_argument("run", type=Path)
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    p = steps.add_parser("score")
    p.add_argument("run", type=Path)
    args = parser.parse_args()
    if args.step == "cut":
        cut(args.limit)
    elif args.step == "read":
        read(args.run, args.threads)
    else:
        score_run(args.run)


if __name__ == "__main__":
    main()
