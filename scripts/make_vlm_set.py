"""M3 S7: the VLM's training set (docs/m3-date-reader.md, "Qwen3-VL-2B",
decision 18), and its check.

    python scripts/make_vlm_set.py --trial --root data/store/vlm-trial --workers 4   # the laptop's trial
    python scripts/make_vlm_set.py --workers 24                                     # the cluster's job
    python scripts/make_vlm_set.py --check --workers 24

What goes in, each answer from reader/vlm.py:

    store           v1-train and v3-train from the store export, its 4 turned
                    copies (train_0 to train_270); listed 3 times in the epoch
                    (vlm.REPEATS), stored once
    expdate-real    ExpDate real-train without real-val, upright and M2's
                    turned copy (real_rot, 90 or 270 degrees), by the keyword rule
    expdate-synth   SIZES of Products-Synth without the copies of a validation
                    photo (m2.REUSED_VAL_COPIES), sampled, by the keyword rule
    gen             SIZES of gen-v2's photos, sampled, each as a panel round one
                    of its blocks (chosen at random), cut back from the
                    background's dates and the other blocks; the block's answer
    negative-empty  regions of ExpDate photos holding no date: null, null
    negative-mfg    regions holding the mfg date only: null and the mfg date

Every image is opened upright, shrunk to 1,600 px on its long side
(decision 3), cut, resized to the model's size (vlm.model_size: sides
multiples of 32, at most 1,920,000 pixels) and stored as JPEG 95 in one
LMDB, <root>/data.mdb, with manifest.json. Everything is drawn from one
seed (SEED), so the set is the same however many workers cut it.

The set holds store photos: it is made and checked under umask 077 (the
folder 700, files 600), and a root inside this checkout must lie under
data/store/. A store example's source is its numbered photo, never a
name. A run record (counts only) goes to results/runs/.

--trial (the laptop): TRIAL examples of each kind, gen-v2-sample for
gen-v2 and no turned ExpDate copies (M2's real_rot is the cluster's).
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import sys
import time
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from bestbefore import derived, expdate, m2, m2b, runs, store_train
from bestbefore.config import REPO_ROOT, expdate_dir, generated_dir, store_train_dir
from bestbefore.reader import parse, piles, shards, vlm

SIZES = {"expdate-synth": 5_000, "gen": 15_000, "negative-empty": 1_000, "negative-mfg": 1_000}
SEED = 0
TRIAL = 12                  # examples of each kind in a trial
GEN_SPARE = 1.2             # gen-v2 photos drawn for each panel wanted (some give none)


def default_root() -> Path:
    return generated_dir() / "m3" / "vlm-set"


@dataclass(frozen=True)
class Job:
    """One example to cut: a photo, the region of it (0..1, None for the
    whole photo), and what it is trained to answer."""
    path: Path
    rect: vlm.XYXY | None
    answer: vlm.Answer
    kind: str
    source: str
    tags: dict


def fractions(rect: vlm.XYXY, size: tuple[int, int]) -> vlm.XYXY:
    w, h = size
    return (rect[0] / w, rect[1] / h, rect[2] / w, rect[3] / h)


# --- Planning, in one process -----------------------------------------------------------

def plan_store(limit: int | None, counts: Counter) -> list[Job]:
    if why := piles.store_missing():
        sys.exit(f"the store's examples can't be made: {why}")
    import json
    root = store_train_dir()
    text = json.loads(piles.store_text_path().read_text())
    manifest = json.loads((root / store_train.MANIFEST).read_text())
    jobs = []
    for number, entry in sorted(text["photos"].items()):
        if entry["part"] != "train":
            continue
        answer, how = vlm.store_answer(entry)
        counts[f"store expiry {how}"] += 1
        meta = manifest["photos"][number]
        for folder, quarter in store_train.FOLDERS["train"].items():
            jobs.append(Job(root / "images" / folder / f"{number}.jpg", None, answer, "store",
                            f"store/{number}@{90 * quarter}",
                            {"tag": meta["tag"], "design": meta["design"], "expiry_from": how, "turn": quarter}))
    return jobs[:limit] if limit else jobs


def plan_expdate_real(trial: bool, limit: int | None, counts: Counter) -> list[Job]:
    val = piles.val_ids()
    ann = expdate.read_annotations("real-train")
    folder = expdate.split_dir("real-train") / "images"
    turned = {} if trial else {p.name: p for p in derived.read_list(m2.list_path("real_rot"))}
    jobs = []
    for name in sorted(ann):
        if name in val:
            continue
        roles = vlm.expdate_roles(ann[name])
        counts[f"expdate-real {roles.why}"] += 1
        if roles.answer is None:
            continue
        tags = {"why": roles.why, "copy": "upright"}
        jobs.append(Job(folder / name, None, roles.answer, "expdate-real", f"real-train/{name}", tags))
        if not trial:
            if name not in turned:
                sys.exit(f"real-train's {name} has no turned copy in M2's real_rot")
            jobs.append(Job(turned[name], None, roles.answer, "expdate-real", f"real_rot/{name}",
                            {**tags, "copy": "real_rot"}))
    return jobs[:limit] if limit else jobs


def plan_expdate_synth(n: int, rng, counts: Counter) -> list[Job]:
    ann = expdate.read_annotations("synth")
    folder = expdate.split_dir("synth") / "images"
    kept = []
    for name in sorted(ann):
        if name in m2.REUSED_VAL_COPIES:
            continue
        roles = vlm.expdate_roles(ann[name])
        counts[f"expdate-synth {roles.why}"] += 1
        if roles.answer is not None:
            kept.append((name, roles))
    pick = sorted(rng.choice(len(kept), size=min(n, len(kept)), replace=False))
    return [Job(folder / kept[i][0], None, kept[i][1].answer, "expdate-synth", f"synth/{kept[i][0]}",
                {"why": kept[i][1].why}) for i in pick]


def plan_gen(n: int, list_name: str, rng, counts: Counter) -> list[Job]:
    import json
    images = derived.read_list(m2b.list_path(list_name))
    order = rng.permutation(len(images))[:min(len(images), int(n * GEN_SPARE) + 50)]
    wanted = set(int(i) for i in order)
    lines = {}
    with open(m2b.sidecar_path(list_name)) as f:
        for i, raw in enumerate(f):
            if i in wanted:
                lines[i] = json.loads(raw)
    if len(lines) != len(wanted):
        sys.exit(f"{list_name}: the sidecar holds {len(lines)} of the {len(wanted)} photos drawn")
    val = piles.val_ids()
    jobs = []
    for i in order:
        if len(jobs) >= n:
            break
        line, path = lines[int(i)], images[int(i)]
        if line["image"] != path.name:
            sys.exit(f"{list_name}: line {i} is {line['image']}, the list has {path.name}")
        background = line.get("background") or {}
        # ExpDate's real and synth photos share names (img_00382.jpg): only a
        # train_real background can be a real-val photo (M2's guard forbids it)
        if background.get("list") == "train_real" and background.get("image") in val:
            counts["gen background is a real-val photo"] += 1
        w, h = line["size"]
        places = [(b["place"][0], b["place"][1], b["place"][0] + b["place"][2], b["place"][1] + b["place"][3])
                  for b in line["blocks"]]
        dates = [(x1 * w, y1 * h, x2 * w, y2 * h) for cls, x1, y1, x2, y2 in background.get("boxes", [])
                 if cls == "date"]
        if not places:
            counts["gen photo without a block"] += 1
            continue
        for b in rng.permutation(len(places)):
            block = line["blocks"][int(b)]
            try:
                answer = vlm.gen_answer(block)
            except ValueError:
                counts["gen block without an answer"] += 1
                continue
            rect = vlm.panel(places[b], (w, h), dates + [p for k, p in enumerate(places) if k != b], rng)
            if rect is None:
                counts["gen block that can't be had alone"] += 1
                continue
            g = block["geometry"]
            jobs.append(Job(path, fractions(rect, (w, h)), answer, "gen", f"{list_name}/{line['image']}#b{int(b)}",
                            {"block": block["kind"], "style": block["style"], "orientation": g["orientation"],
                             "quarter": g["quarter"], "surface": g["surface"], "band": block["band"]}))
            break
        else:
            counts["gen photo without a panel"] += 1
    if len(jobs) < n:
        counts["gen panels short of the size"] += n - len(jobs)
    return jobs


def plan_negatives(n_empty: int, n_mfg: int, rng, counts: Counter) -> list[Job]:
    val = piles.val_ids()
    pool = []
    for split, skip in (("real-train", val), ("synth", m2.REUSED_VAL_COPIES)):
        ann = expdate.read_annotations(split)
        folder = expdate.split_dir(split) / "images"
        pool += [(split, folder / name, ann[name]) for name in sorted(ann) if name not in skip]
    jobs = []
    order = rng.permutation(len(pool))
    for i in order:
        if sum(j.kind == "negative-empty" for j in jobs) >= n_empty:
            break
        split, path, entry = pool[int(i)]
        size = (entry["width"], entry["height"])
        dates = [tuple(a["bbox"]) for a in entry["ann"] if a["cls"] == "date"]
        if (rect := vlm.empty_region(size, dates, rng)) is not None:
            jobs.append(Job(path, fractions(rect, size), vlm.Answer(None, None), "negative-empty",
                            f"{split}/{path.name}", {"split": split}))
        else:
            counts["negative-empty photo without a free region"] += 1
    made = 0
    for i in rng.permutation(len(pool)):
        if made >= n_mfg:
            break
        split, path, entry = pool[int(i)]
        roles = vlm.expdate_roles(entry)
        if roles.answer is None or roles.answer.mfg is None:
            continue
        size = (entry["width"], entry["height"])
        mfg_box = next(box for box, value, role in roles.dates if role == "prod")
        marks = [tuple(a["bbox"]) for a in entry["ann"] if a["cls"] == "prod"]
        near = [(g, m) for m in marks if (g := vlm.mark_gap(mfg_box, m)) is not None]
        mark = min(near)[1] if near else None
        others = [box for box, _, _ in roles.dates if box != mfg_box]
        if (rect := vlm.mfg_region(size, mfg_box, mark, others, rng)) is None:
            counts["negative-mfg photo whose mfg date can't be had alone"] += 1
            continue
        jobs.append(Job(path, fractions(rect, size), vlm.Answer(None, roles.answer.mfg), "negative-mfg",
                        f"{split}/{path.name}", {"split": split}))
        made += 1
    for kind, n in (("negative-empty", n_empty), ("negative-mfg", n_mfg)):
        if (short := n - sum(j.kind == kind for j in jobs)) > 0:
            counts[f"{kind} short of the size"] += short
    return jobs


def plan(trial: bool, counts: Counter) -> list[Job]:
    """Every example, kind by kind, from one seed. A trial takes TRIAL of
    each kind."""
    rng = np.random.default_rng(SEED)
    sizes = {k: TRIAL for k in SIZES} if trial else SIZES
    limit = TRIAL if trial else None
    return (plan_store(limit, counts) + plan_expdate_real(trial, limit, counts)
            + plan_expdate_synth(sizes["expdate-synth"], rng, counts)
            + plan_gen(sizes["gen"], "gen_v2_sample" if trial else "gen_v2", rng, counts)
            + plan_negatives(sizes["negative-empty"], sizes["negative-mfg"], rng, counts))


# --- Cutting, in the worker processes ------------------------------------------------------

def cut(job: Job) -> tuple[bytes | None, str]:
    try:
        image = vlm.load_photo(job.path, job.rect)
    except (OSError, ValueError) as e:
        return None, f"{type(e).__name__}: {e}"
    return vlm.encode(image), f"{image.width}x{image.height}"


# --- The set's tallies ------------------------------------------------------------------------

def tally(counts: dict[str, Counter], ex: vlm.Example, size: tuple[int, int]) -> None:
    counts["kind"][ex.kind] += 1
    a = ex.answer
    counts["expiry"][f"{ex.kind}: {'null' if a.expiry is None else 'day' if a.expiry.day else 'month'}"] += 1
    counts["mfg"][f"{ex.kind}: {'null' if a.mfg is None else 'date'}"] += 1
    for name in ("why", "expiry_from", "block", "orientation", "style", "tag", "copy"):
        if name in ex.tags:
            counts[name][f"{ex.kind}: {ex.tags[name]}"] += 1
    counts["tokens"][ex.kind] += vlm.visual_tokens(*size)


def plain(counts: dict[str, Counter]) -> dict:
    return {k: dict(sorted(v.items())) for k, v in counts.items()}


# --- Making the set -----------------------------------------------------------------------------

def sources(trial: bool) -> dict[str, Path]:
    out = {"real-train/annotations.json": expdate.split_dir("real-train") / "annotations.json",
           "synth/annotations.json": expdate.split_dir("synth") / "annotations.json",
           "yolo/val.txt": expdate_dir() / "yolo" / "val.txt",
           "store/manifest.json": store_train_dir() / store_train.MANIFEST, "store/text.json": piles.store_text_path()}
    gen = "gen_v2_sample" if trial else "gen_v2"
    out[f"m2b/lists/{gen}.txt"] = m2b.list_path(gen)
    out[f"m2b/sidecars/{gen}.jsonl"] = m2b.sidecar_path(gen)
    if not trial:
        out["m2/lists/real_rot.txt"] = m2.list_path("real_rot")
    return out


def make(root: Path, workers: int, trial: bool, replace: bool) -> dict:
    final, part = root, root.with_name(f".{root.name}.part")
    if final.exists() and not replace:
        sys.exit(f"{final} exists: --replace to make it again")
    shutil.rmtree(part, ignore_errors=True)
    t0 = time.time()
    source_hashes = {name: piles.sha256_file(path) for name, path in sources(trial).items()}
    plan_counts: Counter = Counter()
    jobs = plan(trial, plan_counts)
    t_plan = time.time() - t0
    print(f"planned {len(jobs)} examples in {t_plan:.0f} s: {dict(Counter(j.kind for j in jobs))}",
          file=sys.stderr, flush=True)
    counts: dict[str, Counter] = {k: Counter() for k in ("kind", "expiry", "mfg", "why", "expiry_from", "block",
                                                          "orientation", "style", "tag", "copy", "tokens")}
    failed, sizes = Counter(), {k: [] for k in vlm.KINDS}
    jpeg_bytes = 0
    t0 = time.time()
    with vlm.Writer(part) as writer, Pool(workers) as pool:
        for k, (job, (jpeg, what)) in enumerate(zip(jobs, pool.imap(cut, jobs, chunksize=4))):
            if jpeg is None:
                failed[f"{job.kind}: {what}"] += 1
                continue
            w, h = (int(v) for v in what.split("x"))
            ex = writer.add(jpeg, job.answer, job.kind, job.source, job.tags)
            tally(counts, ex, (w, h))
            sizes[job.kind].append(vlm.visual_tokens(w, h))
            jpeg_bytes += len(jpeg)
            if (k + 1) % 2000 == 0:
                print(f"{k + 1} of {len(jobs)} in {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
        n = writer.count
    seconds = time.time() - t0
    listed = sum(vlm.REPEATS.get(k, 1) * c for k, c in counts["kind"].items())
    manifest = {
        "kind": "trial" if trial else "set", "count": n, "listed_in_an_epoch": listed,
        "planned": dict(Counter(j.kind for j in jobs)), "failed": dict(failed), "plan": dict(sorted(plan_counts.items())),
        "counts": plain(counts),
        "visual_tokens": {k: {"min": min(v), "median": int(np.median(v)), "max": max(v)} for k, v in sizes.items() if v},
        "settings": {"seed": SEED, "sizes": SIZES, "trial": TRIAL if trial else None, "repeats": vlm.REPEATS,
                     "max_pixels": vlm.MAX_PIXELS, "min_pixels": vlm.MIN_PIXELS, "factor": vlm.FACTOR,
                     "max_side": store_train.MAX_SIDE, "resample": "PIL BOX to 1,600 px, then bicubic to the model size",
                     "jpeg_quality": shards.JPEG_QUALITY, "subsampling": "4:4:4", "panel_margin": vlm.PANEL_MARGIN,
                     "min_panel": vlm.MIN_PANEL, "region_side": vlm.REGION_SIDE, "mfg_margin": vlm.MFG_MARGIN,
                     "line_gap": vlm.LINE_GAP, "above_gap": vlm.ABOVE_GAP, "glue": vlm.GLUE, "prompt": vlm.PROMPT},
        "sources": source_hashes,
        "code_sha256": runs.code_sha256(vlm, parse, sys.modules[__name__]),
        **runs.git_state(), "host": socket.gethostname(), "workers": workers,
        "made": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": {"plan": round(t_plan, 1), "cut_and_write": round(seconds, 1)},
        "bytes": {"jpeg": jpeg_bytes, "data.mdb": (part / "data.mdb").stat().st_size},
    }
    runs.write_json(part / "manifest.json", manifest)
    if final.exists():
        old = root.with_name(f".{root.name}.old")
        shutil.rmtree(old, ignore_errors=True)
        final.rename(old)
        part.rename(final)
        shutil.rmtree(old)
    else:
        part.rename(final)
    owner_only(final)
    print(f"{n} examples ({listed} listed in an epoch) in {seconds:.0f} s, "
          f"{manifest['bytes']['data.mdb'] / 2**30:.2f} GiB, failed {dict(failed)} -> {final}", file=sys.stderr)
    return {k: manifest[k] for k in ("count", "listed_in_an_epoch", "planned", "failed", "plan", "visual_tokens",
                                     "seconds", "bytes")}


def private_root_problem(root: Path) -> str | None:
    """Why the set can't go under `root`, or None: inside this checkout
    only under data/store/, which git ignores (it holds store photos)."""
    root, repo, data = root.resolve(), REPO_ROOT.resolve(), (REPO_ROOT / "data" / "store").resolve()
    if root.is_relative_to(repo) and not root.is_relative_to(data):
        return f"{root} is inside the checkout but not under {data}: the set holds store photos"
    return None


def owner_only(folder: Path) -> None:
    os.chmod(folder, 0o700)
    for path in folder.rglob("*"):
        os.chmod(path, 0o700 if path.is_dir() else 0o600)


# --- Checking the set ---------------------------------------------------------------------------

def check_range(args: tuple[Path, int, int]) -> tuple[dict, list[str], int]:
    folder, start, stop = args
    env = shards.open_pile(folder)
    counts: dict[str, Counter] = {k: Counter() for k in ("kind", "expiry", "mfg", "why", "expiry_from", "block",
                                                          "orientation", "style", "tag", "copy", "tokens")}
    found, seen = [], 0
    try:
        for ex in vlm.examples(env, start, stop):
            seen += 1
            found += vlm.example_problems(ex)
            if ex.kind == "store" and not ex.source.split("/", 1)[1].split("@")[0].startswith(("v1-", "v3-")):
                found.append(f"{ex.key}: a store source that isn't a numbered photo")
            with ex.image() as im:
                tally(counts, ex, im.size)
    except KeyError as e:
        found.append(str(e))
    finally:
        env.close()
    if seen != stop - start:
        found.append(f"records {start}..{stop - 1}: {seen} read")
    return counts, found[:50], seen


def check(root: Path, workers: int) -> dict:
    manifest = runs.read_json(root / "manifest.json")
    if manifest is None:
        sys.exit(f"no set in {root}")
    env = shards.open_pile(root)
    n = shards.count(env)
    env.close()
    t0 = time.time()
    step = max(1, -(-n // (workers * 4)))
    ranges = [(root, s, min(s + step, n)) for s in range(0, n, step)]
    counts: dict[str, Counter] = {}
    issues, seen = [], 0
    with Pool(workers) as pool:
        for c, f, s in pool.imap(check_range, ranges):
            for k, v in c.items():
                counts.setdefault(k, Counter()).update(v)
            issues += f
            seen += s
    if n != manifest["count"]:
        issues.append(f"{n} records, the manifest says {manifest['count']}")
    if plain(counts) != manifest["counts"]:
        issues.append("the counts read back differ from the manifest's")
    changed = [name for name, path in sources(manifest["kind"] == "trial").items()
               if path.exists() and piles.sha256_file(path) != manifest["sources"].get(name)]
    issues += [f"source {name} changed since the set was made" for name in changed]
    result = {"ok": not issues, "records": n, "read": seen, "issues": issues[:100],
              "seconds": round(time.time() - t0, 1), "checked": time.strftime("%Y-%m-%d %H:%M:%S"),
              **runs.git_state()}
    runs.write_json(root / "check.json", result)
    owner_only(root)
    print(f"{'ok' if result['ok'] else 'FAILED'}: {seen} of {n} records read back in {result['seconds']} s"
          + "".join(f"\n  {i}" for i in issues[:20]), file=sys.stderr)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    parser.add_argument("--root", type=Path, help="where the set goes (default $GENERATED_DIR/m3/vlm-set)")
    parser.add_argument("--trial", action="store_true", help=f"{TRIAL} examples of each kind, the laptop's sources")
    parser.add_argument("--replace", action="store_true", help="make the set again if it exists")
    parser.add_argument("--check", action="store_true", help="check a set already made")
    args = parser.parse_args()
    root = args.root or default_root()
    if args.trial and args.root is None:
        sys.exit("a trial goes elsewhere: give --root (under data/store/ on the laptop)")
    if why := private_root_problem(root):
        sys.exit(why)
    os.umask(0o077)
    if args.check:
        sys.exit(0 if check(root, args.workers)["ok"] else 1)
    root.parent.mkdir(parents=True, exist_ok=True)
    record = runs.start_run("m3-vlm-set" + ("-trial" if args.trial else ""),
                            {"root": str(root), "workers": args.workers, "trial": args.trial, "seed": SEED,
                             "sizes": SIZES}, seed=SEED)
    made = make(root, args.workers, args.trial, args.replace)
    runs.update_run(record, made=made)
    print(f"run record {record}")


if __name__ == "__main__":
    main()
