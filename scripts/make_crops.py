"""M3 S3: cut every crop pile into an LMDB shard (docs/m3-date-reader.md,
decision 4, "The data", S3), and check it.

    python scripts/make_crops.py --pile real-val --pile gen-sample --workers 4   # the laptop's trial
    python scripts/make_crops.py --pile all --workers 24                         # the cluster's job
    python scripts/make_crops.py --check --pile all --workers 24
    python scripts/make_crops.py --pile all --limit 2000 --root DIR              # a smoke, elsewhere

Each pile (reader/piles.py) goes into <root>/<pile>/ (default root
$GENERATED_DIR/m3/crops): LMDB's data.mdb and manifest.json. Every photo is
opened upright and shrunk to at most 1,600 px on its long side as M3 reads
photos (decision 3, store_train.shrink); every crop is cut with half its
box's short side of margin, shrunk so the box's short side is at most
64 px, turned as its pile says, and saved as JPEG 95 (reader/crops.py,
reader/shards.py). One worker pool cuts a pile's photos; one process
writes its LMDB, in the photos' order, so a pile is the same however
many workers cut it. A pile is written into a hidden folder beside its
place and moved there once whole; an existing pile is replaced only with
--replace.

The padding a reader sees is cut from the margin when a crop is loaded,
so nothing here depends on S2's choice of padding, but the margin must
hold it: the crop recipe in use (m3.crop_recipe_used, R1 with the owner's
departures) plus the loader's edge jitter must fit in the margin, or
nothing is cut. The recipe goes into every manifest.

--check reads every record back (it decodes, its box lies inside it with
a short side of at most 64 px, its text isn't empty, its class, role, turns
and parts are sound), counts the records by class, role and tags against
the manifest, recounts the crops the sources call for and compares (records
+ skipped = planned), and compares the planned dates with what the
sources' own manifests say they hold. The result goes to check.json beside
the manifest.

A run record (counts, timing, sizes: no images) goes to results/runs/.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import socket
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from bestbefore import m3, runs, store_train
from bestbefore.config import generated_dir
from bestbefore.images import load_upright
from bestbefore.reader import crops, parse, piles, shards

BATCH = 2000                    # photos handed to the pool at a time
COUNTED_TAGS = ("split", "style", "kind", "orientation", "surface", "band", "stratum", "upright",
                "photo_turn", "parts_from")


def default_root() -> Path:
    return generated_dir() / "m3" / "crops"


# --- Cutting, in the worker processes ---------------------------------------------------

def cut_photo(photo: piles.Photo) -> tuple[list[tuple], float, float]:
    """A photo's crops: per item (jpeg, box in the crop, crop size) or
    (None, why); and the seconds spent opening and cutting."""
    t0 = time.perf_counter()
    image = store_train.shrink(load_upright(photo.image))
    t1 = time.perf_counter()
    out = []
    for item in photo.items:
        if not item.text:
            out.append((None, "no text"))
            continue
        try:
            crop, box = crops.cut(image, crops.to_pixels(item.xyxy, image.size), item.turn)
        except ValueError:
            out.append((None, "no area"))
            continue
        out.append((shards.encode_jpeg(crop), box, crop.size))
    return out, t1 - t0, time.perf_counter() - t1


# --- Making a pile --------------------------------------------------------------------

def recipe_in_use() -> dict:
    """The crop recipe M3 uses, checked to fit in the margin."""
    recipe = m3.crop_recipe_used(runs.read_json(m3.DECISIONS_PATH) or {})
    if recipe["pad"] + m3.EDGE_JITTER > crops.MARGIN:
        sys.exit(f"the recipe's padding {recipe['pad']} plus the edge jitter {m3.EDGE_JITTER} "
                 f"doesn't fit in the margin {crops.MARGIN}: cut with a wider margin first")
    return recipe


def tally(counts: dict[str, Counter], crop: shards.Crop) -> None:
    counts["cls"][str(crop.cls)] += 1
    if crop.cls == "date":
        counts["role"][str(crop.role)] += 1
        counts["parts"]["yes" if crop.parts else "none"] += 1
    for name in COUNTED_TAGS:
        if name in crop.tags:
            counts[name][str(crop.tags[name])] += 1
    counts["turn"][str(crop.turn)] += 1
    w, h = crop.box[2] - crop.box[0], crop.box[3] - crop.box[1]
    counts["shape"]["tall" if h > w else "wide"] += 1
    counts["characters"].update(crop.text)


def plain(counts: dict[str, Counter]) -> dict:
    return {k: dict(sorted(v.items(), key=lambda kv: (-kv[1], kv[0]))) for k, v in counts.items()}


def make(pile: str, root: Path, workers: int, limit: int | None, replace: bool, recipe: dict,
         record: Path) -> dict:
    if why := piles.missing(pile):
        sys.exit(f"{pile} can't be cut: {why}")
    final, part = root / pile, root / f".{pile}.part"
    if final.exists() and not replace:
        sys.exit(f"{final} exists: --replace to make it again")
    shutil.rmtree(part, ignore_errors=True)
    t0 = time.time()
    source_hashes = {name: piles.sha256_file(path) for name, path in piles.sources(pile).items()}
    t_hash = time.time() - t0

    no_box = [0]
    todo = piles.photos(pile, no_box)
    if limit:
        todo = itertools.islice(todo, limit)
    counts: dict[str, Counter] = {k: Counter() for k in ("cls", "role", "parts", "turn", "shape", "characters",
                                                          *COUNTED_TAGS)}
    skipped, planned = Counter(), Counter()
    n_photos, jpeg_bytes, t_open, t_cut = 0, 0, 0.0, 0.0
    t0 = time.time()
    with shards.Writer(part, pile) as writer, Pool(workers) as pool:
        while batch := list(itertools.islice(todo, BATCH)):
            for photo, (results, t1, t2) in zip(batch, pool.imap(cut_photo, batch, chunksize=8)):
                n_photos += 1
                t_open += t1
                t_cut += t2
                for item, result in zip(photo.items, results):
                    planned[str(item.cls)] += 1
                    if result[0] is None:
                        skipped[f"{item.cls}: {result[1]}"] += 1
                        continue
                    jpeg, box, _ = result
                    jpeg_bytes += len(jpeg)
                    crop = writer.add(jpeg=jpeg, box=tuple(round(v, 2) for v in box), text=item.text,
                                      cls=item.cls, source=item.source, role=item.role, parts=item.parts,
                                      turn=item.turn, tags=item.tags)
                    tally(counts, crop)
            print(f"{pile}: {n_photos} photos, {writer.count} crops, {time.time() - t0:.0f} s",
                  file=sys.stderr, flush=True)
        n = writer.count
    seconds = time.time() - t0
    manifest = {
        "pile": pile, "kind": "training" if pile in piles.TRAINING else "suite" if pile in piles.SUITES else "trial",
        "count": n, "photos": n_photos, "limit": limit, "planned": dict(planned), "skipped": dict(skipped),
        "wanted_without_box": no_box[0], "counts": plain(counts),
        "settings": {"margin": crops.MARGIN, "max_short": crops.MAX_SHORT, "max_side": store_train.MAX_SIDE,
                     "photo_resample": "PIL BOX (store_train.shrink)", "crop_resample": "PIL BOX",
                     "jpeg_quality": shards.JPEG_QUALITY, "subsampling": "4:4:4",
                     "min_classless": piles.MIN_CLASSLESS, "edge_jitter": m3.EDGE_JITTER},
        "crop_recipe": recipe, "parse_sha256": m3.check_parser(), "sources": source_hashes,
        "upstream": piles.upstream(pile),
        "code_sha256": runs.code_sha256(crops, shards, piles, parse, sys.modules[__name__]),
        "run": record.name, **runs.git_state(), "host": socket.gethostname(), "workers": workers,
        "made": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": {"hash_sources": round(t_hash, 1), "cut_and_write": round(seconds, 1),
                    "open_in_workers": round(t_open, 1), "cut_in_workers": round(t_cut, 1),
                    "per_photo_ms": round(1000 * seconds / max(n_photos, 1), 2)},
        "bytes": {"jpeg": jpeg_bytes, "data.mdb": (part / "data.mdb").stat().st_size},
    }
    runs.write_json(part / "manifest.json", manifest)
    if final.exists():
        old = root / f".{pile}.old"
        shutil.rmtree(old, ignore_errors=True)
        final.rename(old)
        part.rename(final)
        shutil.rmtree(old)
    else:
        part.rename(final)
    print(f"{pile}: {n} crops from {n_photos} photos in {seconds:.0f} s, "
          f"{manifest['bytes']['data.mdb'] / 2**20:.1f} MiB, skipped {dict(skipped)} -> {final}", file=sys.stderr)
    return {k: manifest[k] for k in ("count", "photos", "skipped", "wanted_without_box", "seconds", "bytes")}


# --- Checking a pile ------------------------------------------------------------------

def check_range(args: tuple[Path, str, int, int]) -> tuple[dict, list[str]]:
    folder, pile, start, stop = args
    env = shards.open_pile(folder)
    counts: dict[str, Counter] = {k: Counter() for k in ("cls", "role", "parts", "turn", "shape", "characters",
                                                          *COUNTED_TAGS)}
    found = []
    seen = 0
    try:
        for crop in shards.records(env, pile, start, stop):
            seen += 1
            found += shards.problems(crop, crops.MAX_SHORT)
            tally(counts, crop)
    except KeyError as e:
        found.append(str(e))
    finally:
        env.close()
    if seen != stop - start:
        found.append(f"records {start}..{stop - 1}: {seen} read")
    return counts, found[:50]


def check(pile: str, root: Path, workers: int) -> dict:
    folder = root / pile
    manifest = json.loads((folder / "manifest.json").read_text())
    t0 = time.time()
    env = shards.open_pile(folder)
    n = shards.count(env)
    env.close()
    step = max(1, -(-n // (workers * 4)))
    ranges = [(folder, pile, s, min(n, s + step)) for s in range(0, n, step)]
    counts: dict[str, Counter] = {}
    found: list[str] = []
    with Pool(workers) as pool:
        for c, f in pool.imap(check_range, ranges):
            for k, v in c.items():
                counts.setdefault(k, Counter()).update(v)
            found += f
    t_read = time.time() - t0
    issues = list(found[:50])
    if n != manifest["count"]:
        issues.append(f"{n} records, the manifest says {manifest['count']}")
    if plain(counts) != manifest["counts"]:
        issues.append("the records' counts differ from the manifest's")

    # The crops the sources call for, recounted, against the records.
    t0 = time.time()
    planned, no_box = Counter(), [0]
    todo = piles.photos(pile, no_box)
    photos = 0
    for photo in (itertools.islice(todo, manifest["limit"]) if manifest["limit"] else todo):
        photos += 1
        planned.update(str(item.cls) for item in photo.items)
    skipped = Counter()
    for reason, k in manifest["skipped"].items():
        skipped[reason.split(":")[0]] += k
    by_cls = {cls: counts["cls"].get(cls, 0) + skipped.get(cls, 0) for cls in planned}
    if by_cls != dict(planned) or set(counts["cls"]) - set(planned):
        issues.append(f"records + skipped {by_cls}, the sources call for {dict(planned)}")
    if photos != manifest["photos"] or no_box[0] != manifest["wanted_without_box"]:
        issues.append(f"{photos} photos and {no_box[0]} tokens without a box in the sources, "
                      f"the manifest says {manifest['photos']} and {manifest['wanted_without_box']}")
    up = piles.upstream(pile)
    if not manifest["limit"] and up.get("date") != planned.get("date"):
        issues.append(f"{planned.get('date')} dates planned, {up.get('from')} says {up.get('date')}")
    for name, path in piles.sources(pile).items():
        if piles.sha256_file(path) != manifest["sources"][name]:
            issues.append(f"{name} has changed since the pile was cut")
    result = {"pile": pile, "ok": not issues, "issues": issues, "records": n, "planned": dict(planned),
              "skipped": manifest["skipped"], "upstream": up, "data_mdb_bytes": (folder / "data.mdb").stat().st_size,
              "seconds": {"read_back": round(t_read, 1), "recount_sources": round(time.time() - t0, 1)},
              "checked": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname(), **runs.git_state()}
    runs.write_json(folder / "check.json", result)
    print(f"{pile}: {'ok' if result['ok'] else 'FAILED'}, {n} records read back in {t_read:.0f} s; "
          f"planned {dict(planned)}, skipped {manifest['skipped']}, upstream {up.get('date')} dates"
          + "".join(f"\n  {i}" for i in issues), file=sys.stderr)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pile", action="append", required=True,
                        help=f"one of {', '.join(piles.PILES)}, or all (the job's: {', '.join(piles.JOB_PILES)})")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    parser.add_argument("--root", type=Path, help="where piles go (default $GENERATED_DIR/m3/crops)")
    parser.add_argument("--limit", type=int, help="the first N photos of each pile (a smoke)")
    parser.add_argument("--replace", action="store_true", help="make a pile again if it exists")
    parser.add_argument("--check", action="store_true", help="check piles already made")
    args = parser.parse_args()
    chosen = [p for name in args.pile for p in (piles.JOB_PILES if name == "all" else (name,))]
    if unknown := [p for p in chosen if p not in piles.PILES]:
        sys.exit(f"unknown pile {', '.join(unknown)}; one of {', '.join(piles.PILES)}")
    root = args.root or default_root()
    if args.limit and args.root is None and not args.check:
        sys.exit("a smoke (--limit) goes elsewhere: give --root")
    if args.check:
        results = [check(p, root, args.workers) for p in chosen]
        bad = [r["pile"] for r in results if not r["ok"]]
        print(f"checked {len(results)} piles: " + (f"FAILED {', '.join(bad)}" if bad else "all ok"))
        sys.exit(1 if bad else 0)

    m3.check_parser()
    recipe = recipe_in_use()
    root.mkdir(parents=True, exist_ok=True)
    record = runs.start_run("m3-crops" + ("-smoke" if args.limit else ""),
                            {"piles": chosen, "root": str(root), "workers": args.workers, "limit": args.limit,
                             "crop_recipe": recipe, "margin": crops.MARGIN, "max_short": crops.MAX_SHORT})
    made = {}
    for pile in chosen:
        made[pile] = make(pile, root, args.workers, args.limit, args.replace, recipe, record)
        runs.update_run(record, piles=made)
    total = sum(m["count"] for m in made.values())
    print(f"made {len(made)} piles, {total} crops -> {root}\nrun record {record}")


if __name__ == "__main__":
    main()
