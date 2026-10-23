"""Generate photos: dates drawn onto ExpDate backgrounds (docs/m2-synthetic-generator.md,
S5 and S6).

    python scripts/generate.py --set gen-sample --count 500 --workers 4   # S5: a sample to look at
    python scripts/generate.py --tasks 0-50 --workers 24                  # S6: gen (tasks 0-49), synth-val (50)
    python scripts/generate.py --check --workers 24 --release m2          # S6: check, vouch, release

Everything goes under $GENERATED_DIR/m2/: gen/images/<list>/NNNNNN.jpg,
their labels in gen/labels/<list>/, the sidecars in gen/sidecars/, lists/
and manifests/ (synth/batch.py has the layout). Photo i comes from
default_rng([set seed, i]) alone, so any split into tasks, runs or worker
processes writes the same bytes. Each worker uses one CPU (OpenCV's own
threads off).

--tasks makes S6's tasks on one pool of --workers processes; each task
writes its own sidecar file and then its record. --check reads every set
back and writes the lists and manifests (complete: true only if nothing
is wrong), plus a run record under results/runs/ with the shares and every
file's SHA-256. It exits with an error, saying why, if anything is missing
or wrong. --release NAME then releases every held job of that name (the
GPU chain's); a check that fails releases nothing.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict

from bestbefore import derived, m2
from bestbefore.runs import code_sha256, start_run, write_json
from bestbefore.synth import batch, fonts, make
from bestbefore.synth.spec import GEN, SET_SEEDS, spec_hash


def sample(name: str, count: int, workers: int) -> None:
    """S5's sample: photos 0 to count - 1 of a set drawn with GEN, one sidecar."""
    start = time.time()
    pools = batch.load_pools([name])
    loaded = time.time() - start
    images = batch.images_dir(name)
    for folder in (images, derived.label_path(images / "x.jpg").parent):     # no photo from an earlier run stays
        shutil.rmtree(folder, ignore_errors=True)
    run_dir = start_run(f"m2-generate-{name}", {"set": name, "count": count, "workers": workers,
                                                "seed": SET_SEEDS[name], "root": m2.root()}, seed=SET_SEEDS[name])
    record = json.loads((run_dir / "run.json").read_text())

    start = time.time()
    sidecars, per_photo = [], []
    sidecar = m2.sidecar_path(m2.GENERATED_SETS[name])
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("w") as out:
        for line, seconds, _ in batch.produce([(name, i) for i in range(count)], workers, pools):
            out.write(line + "\n")
            sidecars.append(json.loads(line))
            per_photo.append(seconds)
    wall = time.time() - start

    pool = pools[name]
    summary = {
        "set": name, "list": m2.GENERATED_SETS[name], "seed": SET_SEEDS[name], **batch.counts(sidecars),
        "listed": derived.write_list(m2.list_path(m2.GENERATED_SETS[name]),
                                     [images / make.image_name(i) for i in range(count)]),
        "spec_sha256": spec_hash(GEN), "spec": asdict(GEN),
        "backgrounds": {n: len(pool.eligible(n, GEN.compose.min_side)) for n, _ in GEN.backgrounds},
        "backgrounds_small_print": {n: len(pool.eligible(n, GEN.compose.min_side_small)) for n, _ in GEN.backgrounds},
        "fonts": {face.file: fonts.SHA256[face.file] for f in fonts.families(GEN.fonts) for face in fonts.faces(f)},
        "git_commit": record["git_commit"], "git_dirty": record["git_dirty"], "run": run_dir.name,
        "code_sha256": code_sha256(*batch.CODE),
        "timing": {**batch.timing(per_photo), "workers": workers, "wall_seconds": round(wall, 1),
                   "backgrounds_loaded_seconds": round(loaded, 1)},
        "shares": batch.checked_shares(sidecars, GEN),
    }
    m2.manifest_path(name).parent.mkdir(parents=True, exist_ok=True)
    write_json(m2.manifest_path(name), summary)
    write_json(run_dir / "summary.json", summary)
    table = batch.shares_table(summary["shares"])
    (run_dir / "shares.md").write_text(table + "\n")
    print(table)
    print(json.dumps({k: summary[k] for k in ("photos", "blocks", "skipped", "redrawn", "fresh_ink", "shrunk",
                                              "boxes", "generated_dates", "timing")}, indent=2))
    print(run_dir)


def tasks(spec: str, workers: int) -> None:
    """Make S6's tasks, timed, with each worker's peak memory."""
    chosen = batch.parse_tasks(spec)
    run_dir = start_run("m2-generate", {"tasks": [t.number for t in chosen], "workers": workers,
                                        "layout": asdict(batch.LAYOUT), "root": m2.root()})
    start = time.time()
    pools = batch.load_pools(t.set for t in chosen)
    loaded = time.time() - start
    start = time.time()
    result = batch.make_tasks(chosen, workers, pools, run=run_dir.name)
    result |= {"tasks": [t.number for t in chosen], "wall_seconds": round(time.time() - start, 1),
               "backgrounds_loaded_seconds": round(loaded, 1)}
    write_json(run_dir / "summary.json", result)
    print(json.dumps(result, indent=2))
    print(run_dir)


def check(workers: int, release: str | None) -> None:
    """Check both sets, write what the check vouches for, and release the
    held jobs named `release` only if every set is complete."""
    run_dir = start_run("m2-generate-check", {"workers": workers, "layout": asdict(batch.LAYOUT),
                                              "release": release, "root": m2.root()})
    start = time.time()
    results = batch.check(batch.LAYOUT, workers)
    seconds = round(time.time() - start, 1)
    for name, r in results.items():
        manifest = r["manifest"]
        write_json(run_dir / f"{name}.json", manifest)
        tables = (manifest["shares"].items() if name == "synth-val" else [("", manifest["shares"])])
        (run_dir / f"{name}-shares.md").write_text(
            "\n\n".join(f"{f'### {k}' if k else ''}\n\n{batch.shares_table(rows)}".strip() for k, rows in tables)
            + "\n")
        print(f"{name}: {manifest['photos']} of {manifest['expected']} photos, "
              f"{'complete' if manifest['complete'] else 'NOT complete'}, {manifest['problem_count']} problems")
        for problem in manifest["problems"]:
            print(f"  {problem}")
    write_json(run_dir / "hashes.json", {name: r["hashes"] for name, r in results.items()})
    outcome = {"passed": batch.passed(results), "check_seconds": seconds, "released": None}
    if not outcome["passed"]:
        write_json(run_dir / "outcome.json", outcome)
        sys.exit(f"check failed: {'releasing nothing; ' if release else ''}see the problems above ({run_dir})")
    if release:
        try:
            outcome["released"] = batch.release_held(release)
        except RuntimeError as e:
            outcome["release_error"] = str(e)
            write_json(run_dir / "outcome.json", outcome)
            sys.exit(f"check passed, but the release failed: {e}")
        print(f"released held {release} jobs: {', '.join(outcome['released']) or 'none were held'}")
    write_json(run_dir / "outcome.json", outcome)
    print(f"check passed in {seconds} s: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--set", choices=[s for s in m2.GENERATED_SETS if s not in batch.SETS],
                      help="a sample of --count photos (S5)")
    mode.add_argument("--tasks", help='S6\'s tasks, e.g. "0-50" or "0-4,50"')
    mode.add_argument("--check", action="store_true", help="check every set and vouch for it (S6)")
    parser.add_argument("--count", type=int, help="with --set")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--release", metavar="NAME", help="with --check: release our held jobs of this name")
    args = parser.parse_args()
    if args.set and not args.count:
        parser.error("--set needs --count")
    if args.release and not args.check:
        parser.error("--release goes with --check")
    if not fonts.fetched():
        raise SystemExit("fonts not fetched: python scripts/fetch_fonts.py (on the cluster, on the login node)")
    if args.set:
        sample(args.set, args.count, args.workers)
    elif args.tasks:
        tasks(args.tasks, args.workers)
    else:
        check(args.workers, args.release)


if __name__ == "__main__":
    main()
