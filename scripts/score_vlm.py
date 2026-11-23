"""M3 S7: Qwen3-VL-2B read through llama.cpp as GGUF, checked against the
HF model (docs/m3-date-reader.md, "Qwen3-VL-2B", decision 16).

    python scripts/score_vlm.py tensors --pair model=OURS.gguf,OFFICIAL.gguf     # two files, tensor by tensor
    python scripts/score_vlm.py compare --photos 50 --hf DIR --gguf ours=MODEL,MMPROJ --gguf official=MODEL,MMPROJ
    python scripts/score_vlm.py val --gguf untrained=MODEL,MMPROJ         # every real-val and store-val photo
    python scripts/score_vlm.py one --gguf ours=MODEL,MMPROJ --gpu-layers 99 --bin DIR   # the laptop (Metal)

Every reader sees the same pixels: a photo is opened upright, shrunk to
1,600 px and resized to its model size (vlm.load_photo); the HF model gets
that image, llama-server the same image as PNG (lossless: a JPEG would be
decoded by two different decoders). Every answer is greedy, with each
token's probability from a plain softmax of its logits, and parsed by
vlm.parse_answer. The HF model runs in float32 on a CPU (its bf16 weights
exactly, no bf16 arithmetic: the reference).

compare: on N real-val photos (a sample fixed by SEED), each reader's
answers; then for each pair of readers, how many photos get the same text,
the same answer and the same expiry, and how far the expiry's confidence
moves where the text agrees; and each reader's seconds a photo.
val: every real-val and store-val photo, scored field by field against
its answer (vlm.score_answer). Store photos' answers stay in an owner-only
folder under $CHECKPOINT_DIR/m3/vlm/; results/runs/ gets counts only.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from bestbefore import runs
from bestbefore.config import checkpoint_dir
from bestbefore.reader import parse, vlm

SEED = 0
LLAMA_SRC = Path.home() / "opt" / "llama.cpp"


def png(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, "PNG", compress_level=1)
    return buf.getvalue()


def readers(specs: list[str]) -> dict[str, tuple[Path, Path]]:
    """--gguf NAME=MODEL,MMPROJ, as {name: (model, mmproj)}."""
    out = {}
    for spec in specs:
        name, files = spec.split("=", 1)
        model, mmproj = (Path(f) for f in files.split(","))
        for f in (model, mmproj):
            if not f.exists():
                sys.exit(f"{f} isn't there")
        out[name] = (model, mmproj)
    return out


# --- Our files against the official ones ---------------------------------------------------------------

def compare_files(ours: Path, official: Path) -> dict:
    """Two GGUF files key by key and tensor by tensor: names, types, shapes,
    bytes; where bytes differ, the values dequantized and their largest
    difference."""
    sys.path.insert(0, str(LLAMA_SRC / "gguf-py"))
    import gguf
    a, b = gguf.GGUFReader(ours), gguf.GGUFReader(official)
    fa = {k: f.contents() for k, f in a.fields.items() if not k.startswith("GGUF.")}
    fb = {k: f.contents() for k, f in b.fields.items() if not k.startswith("GGUF.")}
    keys = {"same": sorted(k for k in fa if k in fb and fa[k] == fb[k]),
            "differ": {k: [str(fa[k])[:80], str(fb[k])[:80]] for k in sorted(fa) if k in fb and fa[k] != fb[k]},
            "ours_only": {k: str(fa[k])[:80] for k in sorted(set(fa) - set(fb))},
            "official_only": {k: str(fb[k])[:80] for k in sorted(set(fb) - set(fa))}}
    ta, tb = {t.name: t for t in a.tensors}, {t.name: t for t in b.tensors}
    tensors = {"count": [len(ta), len(tb)], "ours_only": sorted(set(ta) - set(tb)),
               "official_only": sorted(set(tb) - set(ta)), "identical": 0, "differ": {}}
    for name in sorted(set(ta) & set(tb)):
        x, y = ta[name], tb[name]
        if x.tensor_type != y.tensor_type or list(x.shape) != list(y.shape):
            tensors["differ"][name] = {"types": [x.tensor_type.name, y.tensor_type.name],
                                       "shapes": [list(map(int, x.shape)), list(map(int, y.shape))]}
            continue
        if np.array_equal(x.data, y.data):
            tensors["identical"] += 1
            continue
        dx = gguf.quants.dequantize(x.data, x.tensor_type).astype(np.float32)
        dy = gguf.quants.dequantize(y.data, y.tensor_type).astype(np.float32)
        tensors["differ"][name] = {"type": x.tensor_type.name, "max_abs": float(np.abs(dx - dy).max()),
                                   "share_of_values": float(np.mean(dx != dy))}
    return {"files": [str(ours), str(official)], "sizes": [ours.stat().st_size, official.stat().st_size],
            "keys": keys, "tensors": tensors}


def tensors_cmd(args) -> None:
    pairs = {}
    for spec in args.pair:
        name, files = spec.split("=", 1)
        pairs[name] = tuple(Path(f) for f in files.split(","))
    record = runs.start_run(args.name, {"pairs": {k: [str(f) for f in v] for k, v in pairs.items()}})
    out = {}
    for name, (ours, official) in pairs.items():
        t0 = time.time()
        out[name] = compare_files(ours, official)
        t = out[name]["tensors"]
        print(f"{name}: {t['identical']} of {t['count']} tensors byte-identical, {len(t['differ'])} differ, "
              f"first only {len(t['ours_only'])}, second only {len(t['official_only'])}; keys differing "
              f"{sorted(out[name]['keys']['differ'])}, first only {sorted(out[name]['keys']['ours_only'])}, "
              f"second only {sorted(out[name]['keys']['official_only'])} ({time.time() - t0:.0f} s)", flush=True)
        runs.update_run(record, compared=out)
    print(f"run record {record}")


# --- Reading photos ---------------------------------------------------------------------------------------

def read_hf(model_dir: Path, photos: list[vlm.ValPhoto], threads: int) -> tuple[list[dict], dict]:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    torch.set_num_threads(threads)
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32).eval()
    processor = AutoProcessor.from_pretrained(model_dir)
    info = {"load_seconds": round(time.time() - t0, 1), "dtype": "float32", "threads": torch.get_num_threads()}
    rows = []
    for k, photo in enumerate(photos):
        rows.append(vlm.answer_with(model, processor, vlm.load_photo(photo.image)))
        print(f"hf {k + 1}/{len(photos)} {rows[-1]['seconds']:.1f} s {rows[-1]['text']!r}", flush=True)
    del model
    return rows, info


def read_gguf(name: str, files: tuple[Path, Path], photos: list[vlm.ValPhoto], threads: int, logs: Path,
              binary: Path | None = None, gpu_layers: int = 0) -> tuple[list[dict], dict]:
    rows = []
    with vlm.LlamaServer(*files, threads=threads, log=logs / f"llama-{name}.log", binary=binary,
                         gpu_layers=gpu_layers) as server:
        peak = RssWatch(server.proc.pid)
        for k, photo in enumerate(photos):
            rows.append(server.ask(png(vlm.load_photo(photo.image))))
            print(f"{name} {k + 1}/{len(photos)} {rows[-1]['seconds']:.1f} s {rows[-1]['text']!r}", flush=True)
        info = {"load_seconds": server.load_seconds, "threads": threads, "gpu_layers": gpu_layers,
                "files": [str(f) for f in files], "peak_rss_gib": peak.stop()}
    return rows, info


class RssWatch:
    """The largest resident memory a process reaches, sampled every 0.2 s
    with ps (macOS and Linux alike)."""

    def __init__(self, pid: int):
        self.pid, self.peak, self._stop = pid, 0, threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            out = subprocess.run(["ps", "-o", "rss=", "-p", str(self.pid)], capture_output=True, text=True).stdout
            if out.strip():
                self.peak = max(self.peak, int(out.split()[0]))
            self._stop.wait(0.2)

    def stop(self) -> float:
        self._stop.set()
        self._thread.join()
        return round(self.peak / 2**20, 3)          # ps reports KiB


def sample(photos: list[vlm.ValPhoto], n: int | None) -> list[vlm.ValPhoto]:
    if not n or n >= len(photos):
        return photos
    pick = sorted(np.random.default_rng(SEED).choice(len(photos), size=n, replace=False))
    return [photos[i] for i in pick]


def agreement(a: list[dict], b: list[dict]) -> dict:
    same_text = [x["text"] == y["text"] for x, y in zip(a, b)]
    conf = [abs(x["confidence"] - y["confidence"]) for x, y, s in zip(a, b, same_text)
            if s and x["confidence"] is not None and y["confidence"] is not None]
    probs = [abs(p - q) for x, y, s in zip(a, b, same_text) if s
             for (_, p), (_, q) in zip(x["tokens"], y["tokens"])]
    return {"photos": len(a), "same_text": sum(same_text), "same_answer": sum(x["answer"] == y["answer"] for x, y in zip(a, b)),
            "same_expiry": sum(x["answer"]["expiry"] == y["answer"]["expiry"] for x, y in zip(a, b)),
            "expiry_confidence_max_diff": round(max(conf), 5) if conf else None,
            "token_prob_mean_diff": round(float(np.mean(probs)), 6) if probs else None,
            "token_prob_max_diff": round(float(np.max(probs)), 5) if probs else None,
            "differ": [i for i, s in enumerate(same_text) if not s]}


def summary(rows: list[dict], photos: list[vlm.ValPhoto]) -> dict:
    tally: dict[str, Counter] = {}
    for row, photo in zip(rows, photos):
        for f, s in vlm.score_answer(photo.answer, row["answer"]).items():
            tally.setdefault(f, Counter())[s] += 1
    seconds = [r["seconds"] for r in rows]
    return {"photos": len(rows), "scored": sum(p.answer is not None for p in photos),
            **{f: dict(sorted(c.items())) for f, c in tally.items()},
            "problems": dict(Counter(p for r in rows for p in r["problems"])),
            "seconds_a_photo": {"mean": round(float(np.mean(seconds)), 2), "median": round(float(np.median(seconds)), 2),
                                "max": round(max(seconds), 2)} if seconds else None,
            "prompt_tokens": {"min": min(r["prompt_tokens"] or 0 for r in rows),
                              "max": max(r["prompt_tokens"] or 0 for r in rows)} if rows else None}


def compare_cmd(args) -> None:
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    photos = sample(vlm.val_photos("real-val"), args.photos)
    arms = readers(args.gguf)
    record = runs.start_run(args.name, {"photos": [p.key for p in photos], "hf": str(args.hf) if args.hf else None,
                                        "gguf": {k: [str(f) for f in v] for k, v in arms.items()},
                                        "threads": args.threads, "seed": SEED, "prompt": vlm.LLAMA_PROMPT,
                                        "code_sha256": runs.code_sha256(vlm, parse, sys.modules[__name__])})
    answers, info = {}, {}
    if args.hf:
        answers["hf"], info["hf"] = read_hf(args.hf, photos, args.threads)
        runs.update_run(record, info=info)
    for name, files in arms.items():
        answers[name], info[name] = read_gguf(name, files, photos, args.threads, record)
        runs.update_run(record, info=info)
    result = {"info": info, "by_reader": {k: summary(v, photos) for k, v in answers.items()},
              "agreement": {f"{x} vs {y}": agreement(answers[x], answers[y]) for x, y in combinations(answers, 2)}}
    runs.write_json(record / "answers.json", {"photos": [p.key for p in photos], **answers})
    runs.update_run(record, **result)
    for pair, a in result["agreement"].items():
        print(f"{pair}: same text {a['same_text']}/{a['photos']}, same answer {a['same_answer']}, "
              f"same expiry {a['same_expiry']}, expiry confidence moves at most {a['expiry_confidence_max_diff']}")
    for name, s in result["by_reader"].items():
        print(f"{name}: expiry {s.get('expiry')}, mfg {s.get('mfg')}, {s['seconds_a_photo']} s a photo")
    print(f"run record {record}")


def val_cmd(args) -> None:
    os.umask(0o077)                 # store-val's answers: owner-only
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    (name, files), = readers(args.gguf).items()
    record = runs.start_run(f"m3-vlm-val-{name}", {"gguf": [str(f) for f in files], "threads": args.threads,
                                                   "prompt": vlm.LLAMA_PROMPT,
                                                   "code_sha256": runs.code_sha256(vlm, parse, sys.modules[__name__])})
    private = checkpoint_dir() / "m3" / "vlm" / record.name
    private.mkdir(parents=True)
    out = {}
    for set_name in vlm.VAL_SETS:
        photos = vlm.val_photos(set_name)
        rows, info = read_gguf(name, files, photos, args.threads, private)
        out[set_name] = {**summary(rows, photos), "server": info,
                         "by_tag": by_tag(rows, photos) if set_name == "store-val" else None}
        runs.write_json(private / f"{set_name}.json", [{"key": p.key, "truth": None if p.answer is None else p.answer.as_dict(),
                                                        "tags": p.tags, **r} for p, r in zip(photos, rows)])
        runs.update_run(record, results=out, answers=str(private))
        print(f"{set_name}: {out[set_name]}", flush=True)
    print(f"run record {record}")


def by_tag(rows: list[dict], photos: list[vlm.ValPhoto]) -> dict:
    out: dict[str, Counter] = {}
    for row, photo in zip(rows, photos):
        for f, s in vlm.score_answer(photo.answer, row["answer"]).items():
            out.setdefault(f"{f}: {photo.tags.get('tag')}", Counter())[s] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


def one_cmd(args) -> None:
    (name, files), = readers(args.gguf).items()
    photos = [p for p in vlm.val_photos("real-val") if p.key == args.photo] if args.photo else \
        sample(vlm.val_photos("real-val"), 1)
    record = runs.start_run(f"m3-vlm-one-{name}", {"gguf": [str(f) for f in files], "photo": photos[0].key,
                                                   "threads": args.threads, "gpu_layers": args.gpu_layers,
                                                   "bin": str(args.bin)})
    rows, info = read_gguf(name, files, photos * args.repeat, args.threads, record, args.bin, args.gpu_layers)
    runs.write_json(record / "answers.json", rows)
    runs.update_run(record, info=info, answers=[{k: r[k] for k in ("text", "answer", "seconds", "timings")}
                                                for r in rows])
    print(json.dumps({"info": info, "answers": [{k: r[k] for k in ("text", "seconds", "timings")} for r in rows]},
                     indent=1))
    print(f"run record {record}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("tensors")
    t.add_argument("--pair", action="append", required=True, help="NAME=OURS,OTHER: two GGUF files")
    t.add_argument("--name", default="m3-vlm-gguf-files")
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    for name in ("compare", "val", "one"):
        p = sub.add_parser(name)
        p.add_argument("--gguf", action="append", default=[], help="NAME=MODEL,MMPROJ")
        p.add_argument("--threads", type=int, default=threads)
    c = sub.choices["compare"]
    c.add_argument("--photos", type=int, default=50)
    c.add_argument("--hf", type=Path, help="an HF model folder, read in float32 on the CPU")
    c.add_argument("--name", default="m3-vlm-gguf-check")
    o = sub.choices["one"]
    o.add_argument("--photo", help="a real-val key, e.g. real-val/img_00003.jpg (default: the sample's first)")
    o.add_argument("--gpu-layers", type=int, default=99)
    o.add_argument("--bin", type=Path, help="llama.cpp's bin folder")
    o.add_argument("--repeat", type=int, default=2, help="read it this many times (the first warms up)")
    args = parser.parse_args()
    {"tensors": tensors_cmd, "compare": compare_cmd, "val": val_cmd, "one": one_cmd}[args.command](args)


if __name__ == "__main__":
    main()
