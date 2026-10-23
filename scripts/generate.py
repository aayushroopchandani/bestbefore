"""Generate photos: dates drawn onto ExpDate backgrounds (docs/m2-synthetic-generator.md,
S5; the scale-out is S6).

    python scripts/generate.py --set gen-sample --count 500 --workers 4

Writes under $GENERATED_DIR/m2/: gen/images/<list>/NNNNNN.jpg, its labels in
gen/labels/<list>/, gen/sidecars/<list>.jsonl (one line per photo, in photo
order), lists/<list>.txt and manifests/<set>.json; and a run record under
results/runs/ with the shares the photos show against the spec, and where
the time went. Photo i comes from default_rng([set seed, i]) alone, so any
number of workers writes the same bytes. Each worker uses one CPU (OpenCV's
own threads off), as S6's will.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import cv2

from bestbefore import derived, m2
from bestbefore.runs import code_sha256, start_run
from bestbefore.synth import camera, compose, fonts, make, render, text, warp
from bestbefore.synth.spec import GEN, SET_SEEDS, spec_hash

_worker: dict = {}


def _start(pool: compose.Backgrounds, seed: int, images: Path) -> None:
    cv2.setNumThreads(1)
    _worker.update(pool=pool, seed=seed, images=images)


def _one(i: int) -> tuple[dict, dict]:
    """Make and write photo i: its sidecar line and the seconds per stage."""
    clock = make.Clock()
    made = make.make_one(i, GEN, _worker["pool"], _worker["seed"], clock)
    with clock("write"):
        make.write_one(made, _worker["images"])
    return made.sidecar, dict(clock.seconds)


def timing(per_photo: list[dict], wall: float, workers: int) -> dict:
    """Mean milliseconds per photo, by stage and in all, on one CPU each."""
    stages = Counter()
    for seconds in per_photo:
        stages.update(seconds)
    totals = sorted(sum(s.values()) for s in per_photo)
    n = len(per_photo)
    return {"photos": n, "workers": workers, "wall_seconds": round(wall, 1),
            "mean_ms": round(1000 * sum(totals) / n, 1),
            "median_ms": round(1000 * totals[n // 2], 1),
            "p95_ms": round(1000 * totals[min(n - 1, round(0.95 * n))], 1),
            "max_ms": round(1000 * totals[-1], 1),
            "by_stage_mean_ms": {k: round(1000 * v / n, 1) for k, v in stages.most_common()}}


def counts(sidecars: list[dict]) -> dict:
    labels = Counter(b.label for s in sidecars for b in make.label_boxes(s))
    return {"photos": len(sidecars),
            "blocks": sum(len(s["blocks"]) for s in sidecars),
            "skipped": dict(Counter(k["why"] for s in sidecars for k in s["skipped"])),
            "redrawn": sum(b["redraws"] > 0 for s in sidecars for b in s["blocks"]),
            "shrunk": sum(b["shrunk"] for s in sidecars for b in s["blocks"]),
            "boxes": dict(sorted(labels.items())),
            "generated_dates": sum(t["cls"] == "date" for s in sidecars for b in s["blocks"]
                                   for t in b["tokens"])}


def shares_table(rows: list[dict]) -> str:
    lines = ["| what | spec | measured | n |", "|---|---:|---:|---:|"]
    for r in rows:
        measured = "" if r["measured"] is None else f"{r['measured']:.3f}"
        lines.append(f"| {r['what']} | {r['spec']:.3f} | {measured} | {r['n']} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", required=True, choices=m2.GENERATED_SETS)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    name, seed = m2.GENERATED_SETS[args.set], SET_SEEDS[args.set]
    if not fonts.fetched():
        raise SystemExit("fonts not fetched: python scripts/fetch_fonts.py")

    start = time.time()
    pool = compose.load_backgrounds(GEN.backgrounds, GEN.compose.min_side)
    loaded = time.time() - start
    images, labels = m2.gen_dir() / "images" / name, m2.gen_dir() / "labels" / name
    for folder in (images, labels):             # no photo from an earlier run stays
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
    sidecar = m2.sidecar_path(name)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    run_dir = start_run(f"m2-generate-{args.set}", {**vars(args), "seed": seed, "root": m2.root()}, seed=seed)
    record = json.loads((run_dir / "run.json").read_text())

    start = time.time()
    sidecars, per_photo = [], []
    with Pool(args.workers, initializer=_start, initargs=(pool, seed, images)) as workers, \
            sidecar.open("w") as out:
        for line, seconds in workers.imap(_one, range(args.count), chunksize=4):
            out.write(json.dumps(line, separators=(",", ":")) + "\n")
            sidecars.append(line)
            per_photo.append(seconds)
    wall = time.time() - start

    summary = {
        "set": args.set, "list": name, "seed": seed, **counts(sidecars),
        "listed": derived.write_list(m2.list_path(name), [images / make.image_name(i) for i in range(args.count)]),
        "spec_sha256": spec_hash(GEN), "spec": asdict(GEN),
        "backgrounds": {n: len(pool.eligible(n, GEN.compose.min_side)) for n, _ in GEN.backgrounds},
        "backgrounds_small_print": {n: len(pool.eligible(n, GEN.compose.min_side_small)) for n, _ in GEN.backgrounds},
        "fonts": {face.file: fonts.SHA256[face.file] for f in fonts.families(GEN.fonts) for face in fonts.faces(f)},
        "git_commit": record["git_commit"], "git_dirty": record["git_dirty"], "run": run_dir.name,
        "code_sha256": code_sha256(compose, camera, make, render, text, warp),
        "timing": {**timing(per_photo, wall, args.workers), "backgrounds_loaded_seconds": round(loaded, 1)},
        "shares": make.shares(sidecars, GEN),
    }
    manifest = json.dumps(summary, indent=2) + "\n"
    m2.manifest_path(args.set).parent.mkdir(parents=True, exist_ok=True)
    m2.manifest_path(args.set).write_text(manifest)
    (run_dir / "summary.json").write_text(manifest)

    table = shares_table(summary["shares"])
    (run_dir / "shares.md").write_text(table + "\n")
    print(table)
    print(json.dumps({k: summary[k] for k in ("photos", "blocks", "skipped", "redrawn", "shrunk", "boxes",
                                              "generated_dates", "timing")}, indent=2))
    print(run_dir)


if __name__ == "__main__":
    main()
