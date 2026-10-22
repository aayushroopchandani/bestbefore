"""Write M2's derived sets under $GENERATED_DIR/m2/ (docs/m2-synthetic-generator.md, S2).

    python scripts/derive_sets.py --only val      # the two validation sets (laptop)
    python scripts/derive_sets.py --only lists    # train_synth_clean alone (laptop: no new photos)
    python scripts/derive_sets.py --workers 16    # all five (slurm/m2_derive.sbatch)

The sets, each listed in lists/<set>.txt (absolute paths):
  real_val_rot        the 165 validation photos, each turned 90 or 270 degrees
  real_val_small      the validation photos in 2 x 2 grids, four of similar size
  train_synth_clean   ExpDate's synthetic training list without the 14 copies
                      of a validation photo (a list only: no new photos)
  real_rot            the 937 real training photos, turned
  synth_rot           the 11,844 clean synthetic photos, turned

The sources are ExpDate's YOLO lists and label files, exactly what M1
trained and validated on. A photo turns 90 or 270 degrees by a coin seeded
with its set and file name (real and synthetic file names repeat). New
photos are JPEGs (quality 95) in derived/images/<set>/ with labels in
derived/labels/<set>/, and every one is read back and checked against the
boxes it was given. manifests/derived.json keeps, per set: counts, seed,
git commit, code hashes and, for the grids, which photos went where.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter
from contextlib import nullcontext
from multiprocessing import Pool
from pathlib import Path

from bestbefore import derived, m2
from bestbefore.config import expdate_dir
from bestbefore.images import load_upright
from bestbefore.runs import code_sha256, start_run
from bestbefore.samples import DATE_LABELS, Sample


# --- One photo each, run in the worker processes ------------------------------

def turn_one(src: Path, root: Path, set_name: str, quarter: int) -> tuple[Sample, float]:
    """Turn a photo, write it, read it back: the sample and its worst box error (px)."""
    image, boxes = derived.turn(load_upright(src), derived.read_boxes(src), quarter)
    sample = derived.write_yolo(root, set_name, src.name, image, boxes)
    return sample, derived.check_written(sample)


def grid_one(srcs: list[Path], root: Path, set_name: str, name: str) -> tuple[Sample, float]:
    """Put four photos in a grid, write it, read it back."""
    image, boxes = derived.grid([(load_upright(p), derived.read_boxes(p)) for p in srcs])
    sample = derived.write_yolo(root, set_name, name, image, boxes)
    return sample, derived.check_written(sample)


# --- Whole sets ---------------------------------------------------------------

def write_set(pool, root: Path, set_name: str, job, tasks: list[tuple]) -> dict:
    """Run `job` over `tasks` into an emptied set folder, list the photos
    in task order, and count what was written."""
    start = time.time()
    for kind in ("images", "labels"):           # no photo from an earlier run stays
        shutil.rmtree(root / kind / set_name, ignore_errors=True)
    results = pool.starmap(job, tasks)
    labels = Counter(b.label for s, _ in results for b in s.boxes)
    return {
        "photos": derived.write_list(m2.list_path(set_name), [s.image for s, _ in results]),
        "dates": sum(n for label, n in labels.items() if label in DATE_LABELS),
        "boxes": dict(sorted(labels.items())),
        "worst_box_error_px": round(max(err for _, err in results), 4),
        "seconds": round(time.time() - start, 1),
    }


def turned(pool, root: Path, set_name: str, sources: list[Path], seed: int) -> dict:
    quarters = [derived.quarter_for(f"{set_name}/{p.name}", seed) for p in sources]
    entry = write_set(pool, root, set_name, turn_one,
                      [(p, root, set_name, q) for p, q in zip(sources, quarters)])
    return {**entry, "turned": {"90": quarters.count(1), "270": quarters.count(3)}}


def gridded(pool, root: Path, set_name: str, sources: list[Path]) -> dict:
    groups, left = derived.grid_groups(derived.load_sample(p) for p in sources)
    names = [f"grid_{i:03d}.jpg" for i in range(len(groups))]
    entry = write_set(pool, root, set_name, grid_one,
                      [([s.image for s in g], root, set_name, n) for g, n in zip(groups, names)])
    return {**entry, "grids": {n: [s.id for s in g] for n, g in zip(names, groups)},
            "left_over": [s.id for s in left]}


def without_copies(synth: list[Path]) -> list[Path]:
    """The synthetic list without the 14 copies. Stops if any of them isn't
    in it exactly once: then this isn't the list M1 trained on."""
    seen = Counter(p.name for p in synth)
    wrong = sorted(name for name in m2.REUSED_VAL_COPIES if seen[name] != 1)
    if wrong:
        raise SystemExit(f"not exactly once in train_synth.txt: {', '.join(wrong)}")
    return [p for p in synth if p.name not in m2.REUSED_VAL_COPIES]


def save_manifest(made: dict[str, dict], common: dict, run_dir: Path) -> dict:
    """Merge into manifests/derived.json, so `--only val` and `--only lists`
    keep the other sets' entries; a copy goes into the run folder."""
    path = m2.manifest_path("derived")
    manifest = json.loads(path.read_text()) if path.exists() else {"sets": {}}
    manifest["sets"].update({name: {**entry, **common} for name, entry in made.items()})
    text = json.dumps(manifest, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    (run_dir / "manifest.json").write_text(text)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", choices=["val", "lists"],
                        help="val: only the two validation sets; lists: only "
                             "train_synth_clean, which writes no photos")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0, help="for the turns")
    args = parser.parse_args()
    make_val = args.only in (None, "val")          # real_val_rot, real_val_small
    make_list = args.only in (None, "lists")       # train_synth_clean
    make_turned = args.only is None                # real_rot, synth_rot

    # Read and check every source before writing anything.
    yolo, root = expdate_dir() / "yolo", m2.derived_dir()
    if make_val:
        val = derived.read_list(yolo / "val.txt")
    if make_list or make_turned:
        synth = without_copies(derived.read_list(yolo / "train_synth.txt"))
    if make_turned:
        real = derived.read_list(yolo / "train_real.txt")

    run_dir = start_run("m2-derive", {"only": args.only, "workers": args.workers,
                                      "root": m2.root()}, seed=args.seed)
    record = json.loads((run_dir / "run.json").read_text())
    common = {"seed": args.seed, "git_commit": record["git_commit"],
              "git_dirty": record["git_dirty"], "run": run_dir.name,
              "code_sha256": code_sha256(derived, m2)}

    made: dict[str, dict] = {}
    with Pool(args.workers) if make_val or make_turned else nullcontext() as pool:
        if make_val:
            made["real_val_rot"] = turned(pool, root, "real_val_rot", val, args.seed)
            made["real_val_small"] = gridded(pool, root, "real_val_small", val)
        if make_list:
            made["train_synth_clean"] = {
                "photos": derived.write_list(m2.list_path("train_synth_clean"), synth),
                "left_out": sorted(m2.REUSED_VAL_COPIES)}
        if make_turned:
            made["real_rot"] = turned(pool, root, "real_rot", real, args.seed)
            made["synth_rot"] = turned(pool, root, "synth_rot", synth, args.seed)
    save_manifest(made, common, run_dir)

    for name, e in made.items():
        extra = (f", turned 90/270: {e['turned']['90']}/{e['turned']['270']}" if "turned" in e
                 else f", left over: {', '.join(e['left_over'])}" if "left_over" in e else "")
        written = (f", {e['dates']} dates, worst box error {e['worst_box_error_px']} px,"
                   f" {e['seconds']} s" if "dates" in e else "")
        print(f"{name}: {e['photos']} photos{written}{extra}")
    print(m2.manifest_path("derived"))


if __name__ == "__main__":
    main()
