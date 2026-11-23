"""M3 S6: PP-OCRv6 small's ONNX path, and the check that decides its
fine-tuning route (docs/m3-date-reader.md, "PP-OCRv6 small", S6).

    python scripts/ppocr_route.py check --threads 24     # in m3_cpu.sbatch ppocr-route
    python scripts/train_reader.py --reader ppocrv6 --smoke --device cpu --threads 22 --no-amp
    python scripts/ppocr_route.py decide                 # the route, into results/runs/m3-decisions.json

`check` reads real-val's date crops exactly as S2 cut them (its run's PNG
files, every padding and turn S2 read with PaddleOCR's recogniser alone),
through M3's one pre- and post-processing with PP-OCRv6's spec:

  - the official ONNX file against S2's PaddleOCR reads: one crop a call
    (as S2 read them), in batches of one padded width (as scoring reads)
    and in batches of mixed widths (as the CRNN's);
    the same at three simpler layouts, to show what each part of the spec
    does (M3's own: RGB, padded to the crop, squeezed past 480; then BGR;
    then padded to at least 320);
  - transformers' module (reader/ppocr.py) against the ONNX file, in
    batches: the second route condition;
  - its CTC loss on 64 of those crops (upright, unpadded, their ExpDate
    text) and every parameter's gradient: the first.

Only real-val, which is ExpDate's (public): the run's store crops are
never opened. The record (results/runs/<time>-m3-ppocr-route/) holds
counts, times and up to 40 differing reads with their real-val keys.

`decide` takes the newest check and the newest PP-OCRv6 CPU smoke, both of
the same commit, whose probe loss before and after its 200 steps is the
third condition, and applies m3.ppocr_route.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from bestbefore import m3, runs
from bestbefore.config import generated_dir
from bestbefore.reader import data, onnx_run, ppocr

GRADIENT_CROPS = 64
SHOWN = 40              # differing reads kept in the record


def s2_folder(override: Path | None) -> Path:
    if override:
        return override
    decisions = runs.read_json(m3.DECISIONS_PATH) or {}
    if "crop_recipe" not in decisions:
        sys.exit(f"no crop recipe in {m3.DECISIONS_PATH}: S2's run is needed")
    return generated_dir() / "m3" / "crop-experiment" / decisions["crop_recipe"]["run"]


def s2_reads(folder: Path, limit: int | None) -> list[dict]:
    """S2's PP-OCRv6 reads of real-val crops (recogniser alone), each with
    the crop as read: the PNG turned by its k."""
    info = json.loads((folder / "crops.json").read_text())
    real = [c for c in info["crops"] if c["set"] == "real-val"][:limit]
    wanted = {c["key"]: c for c in real}
    out = []
    with open(folder / "reads.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["reader"] != "ppocrv6" or r["mode"] != "alone" or r["key"] not in wanted:
                continue
            c = wanted[r["key"]]
            with Image.open(folder / c["files"][f"{r['pad']:g}"]) as im:
                rgb = np.asarray(im.convert("RGB"))
            out.append({"key": r["key"], "pad": r["pad"], "k": r["k"], "text": r["text"], "score": r["score"],
                        "truth": c["truth_text"], "image": np.ascontiguousarray(np.rot90(rgb, r["k"]))})
    return out


def agreement(reads: list[dict], ours: list, theirs: list[str] | None = None) -> dict:
    """How many of `ours` give the same text as S2's (or as `theirs`):
    over all reads, upright ones (k 0) and turned ones; and the differing
    reads, with both texts and confidences."""
    theirs = theirs if theirs is not None else [r["text"] for r in reads]
    same = [o.text == t for o, t in zip(ours, theirs)]
    out = {}
    for part, keep in (("all", lambda r: True), ("upright", lambda r: r["k"] == 0), ("turned", lambda r: r["k"] != 0)):
        idx = [i for i, r in enumerate(reads) if keep(r)]
        if idx:
            n = sum(same[i] for i in idx)
            out[part] = {"same": n, "reads": len(idx), "share": n / len(idx)}
    out["differing"] = [{"key": reads[i]["key"], "pad": reads[i]["pad"], "k": reads[i]["k"], "theirs": theirs[i],
                         "s2_score": reads[i]["score"], "ours": ours[i].text, "ours_mean": round(ours[i].mean, 4),
                         "ours_least": round(ours[i].confidence, 4)}
                        for i in range(len(reads)) if not same[i]][:SHOWN]
    return out


def chosen_agreement(reads: list[dict], ours: list) -> dict:
    """Decision 14 on both sides: per crop and padding, the read of its two
    turns that holds a date (then the more confident), S2's by its score,
    ours by its least sure character; how often both choose the same text."""
    from bestbefore.reader import parse
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(reads):
        groups.setdefault((r["key"], r["pad"]), []).append(i)
    same = 0
    for idx in groups.values():
        s2 = max(idx, key=lambda i: (bool(parse.read_dates(reads[i]["text"])), reads[i]["score"] or 0.0))
        mine = idx[onnx_run.choose([ours[i] for i in idx])]
        same += reads[s2]["text"] == ours[mine].text
    return {"same": same, "crops": len(groups), "share": same / len(groups)}


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def check(threads: int, folder: Path, limit: int | None) -> Path:
    torch.set_num_threads(threads)
    spec = ppocr.spec()
    reads = s2_reads(folder, limit)
    images = [r["image"] for r in reads]
    pre = ppocr.pretrained()
    config = {"threads": threads, "s2_run": folder.name, "limit": limit, "reads": len(reads),
              "spec": {"classes": len(spec.classes), "max_width": spec.max_width, "min_batch": spec.min_batch,
                       "bgr": spec.bgr, "same_width": spec.same_width},
              "files": {f: ppocr.sha256(pre / f) for f in (ppocr.ONNX_FILE, ppocr.DICTIONARY_FILE,
                                                         f"{ppocr.SAFETENSORS}/model.safetensors")},
              "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY"), "host": os.uname().nodename}
    record = runs.start_run("m3-ppocr-route", config)
    print(f"run {record}: {len(reads)} of S2's real-val reads, {threads} threads", file=sys.stderr, flush=True)

    official = ppocr.onnx_reader(threads)
    layouts = {"m3": onnx_run.Spec(classes=spec.classes),
               "bgr": onnx_run.Spec(classes=spec.classes, bgr=True),
               "bgr-min320": onnx_run.Spec(classes=spec.classes, bgr=True, min_batch=ppocr.MIN_BATCH),
               "ppocrv6": spec}
    results: dict = {"onnx_vs_s2": {}, "ms_per_read": {}}
    for name, layout in layouts.items():
        ours, s = timed(lambda: onnx_run.read(official, images, batch=1, spec=layout))
        results["onnx_vs_s2"][f"{name}-one-a-call"] = agreement(reads, ours)
        results["ms_per_read"][f"onnx-{name}-one-a-call"] = round(1000 * s / len(reads), 2)
        print(f"onnx, {name}, one a call: {results['onnx_vs_s2'][f'{name}-one-a-call']['all']}", file=sys.stderr,
              flush=True)
    mixed = dataclasses.replace(spec, same_width=False)
    mixed_reads, s = timed(lambda: onnx_run.read(official, images, batch=64, spec=mixed))
    results["ms_per_read"]["onnx-ppocrv6-mixed-batches"] = round(1000 * s / len(reads), 2)
    results["onnx_vs_s2"]["ppocrv6-mixed-batches"] = agreement(reads, mixed_reads)
    onnx_reads, s = timed(lambda: onnx_run.read(official, images, batch=64, spec=spec))
    results["ms_per_read"]["onnx-ppocrv6-batches"] = round(1000 * s / len(reads), 2)
    results["onnx_vs_s2"]["ppocrv6-batches"] = agreement(reads, onnx_reads)
    results["onnx_vs_s2"]["ppocrv6-batches-chosen"] = chosen_agreement(reads, onnx_reads)

    model = ppocr.load_torch()
    torch_reads, s = timed(lambda: onnx_run.read(onnx_run.TorchReader(model), images, batch=64, spec=spec))
    results["ms_per_read"]["torch-ppocrv6-batches"] = round(1000 * s / len(reads), 2)
    results["torch_vs_onnx"] = agreement(reads, torch_reads, [r.text for r in onnx_reads])
    results["torch_vs_s2"] = agreement(reads, torch_reads)
    sample = [r for r in reads if r["pad"] == 0 and r["k"] == 0 and r["truth"]][:GRADIENT_CROPS]
    batch = data.collate([r["image"] for r in sample], [r["truth"] for r in sample], [0] * len(sample), spec=spec)
    results["softmax_gap"] = ppocr.softmax_gap(model, batch["x"].numpy())
    results["gradients"] = ppocr.gradient_check(model, batch)
    results["versions"] = {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
                           "onnxruntime": __import__("onnxruntime").__version__, "torch_threads": torch.get_num_threads()}
    runs.update_run(record, results=results)
    for k, v in results["onnx_vs_s2"].items():
        print(f"onnx vs S2, {k}: " + json.dumps({p: v[p] for p in ("all", "upright", "turned", "same", "share") if p in v}))
    print(f"torch vs onnx: {json.dumps({p: results['torch_vs_onnx'][p] for p in ('all', 'upright', 'turned')})}")
    print(f"gradients: passed {results['gradients']['passed']}, loss {results['gradients']['loss']:.3f}, "
          f"{len(results['gradients']['not_finite'])} not finite, {len(results['gradients']['no_gradient'])} none; "
          f"softmax gap {results['softmax_gap']:.2e}; ms a read {results['ms_per_read']}")
    print(record)
    return record


def decide() -> None:
    check_dir, smoke_dir = runs.latest_run("m3-ppocr-route"), runs.latest_run("m3-ppocrv6-smoke-cpu")
    if check_dir is None:
        sys.exit("no m3-ppocr-route run: run `check` first")
    check_rec = runs.read_json(check_dir / "run.json")
    smoke_rec = runs.read_json(smoke_dir / "run.json") if smoke_dir else None
    if smoke_rec and smoke_rec["git_commit"] != check_rec["git_commit"]:
        sys.exit(f"the check ({check_dir.name}) and the smoke ({smoke_dir.name}) ran at different commits")
    results = check_rec["results"]
    probe = ((smoke_rec or {}).get("result") or {}).get("probe_loss") or {}
    route = m3.ppocr_route(loads=results["gradients"]["passed"], parity=results["torch_vs_onnx"]["all"]["share"],
                           probe_before=probe.get("before"), probe_after=probe.get("after"))
    route |= {"check": check_dir.name, "smoke": smoke_dir.name if smoke_rec else None,
              "commit": check_rec["git_commit"],
              "onnx_vs_s2": {k: v.get("all", v) for k, v in results["onnx_vs_s2"].items()},
              "smoke_passed": (smoke_rec or {}).get("smoke", {}).get("passed")}
    decisions = runs.read_json(m3.DECISIONS_PATH) or {}
    decisions["ppocr_route"] = route
    runs.write_json(m3.DECISIONS_PATH, decisions)
    runs.update_run(check_dir, route=route["route"])
    print(json.dumps(route, indent=1))
    print(f"-> {m3.DECISIONS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    steps = parser.add_subparsers(dest="step", required=True)
    p = steps.add_parser("check")
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    p.add_argument("--run", type=Path, help="S2's run folder (default: the crop recipe's, from m3-decisions.json)")
    p.add_argument("--limit", type=int, help="the first N real-val crops (a smoke)")
    steps.add_parser("decide")
    args = parser.parse_args()
    if args.step == "check":
        check(args.threads, s2_folder(args.run), args.limit)
    else:
        decide()


if __name__ == "__main__":
    main()
