"""M2b's inference add-ons, measured (docs/m2b-store-and-size.md, S13): R3's
arm's prediction passes saved, every configuration of the add-ons fused and
scored, R4 and S13's "what if" applied, and n timed against m.

    python scripts/m2b_inference.py predict --set real-val --device cpu   # the cluster: slurm/m2b_inference_cpu.sbatch
    python scripts/m2b_inference.py predict --set v1-val --device cpu     # the laptop
    python scripts/m2b_inference.py score                                 # the laptop, from both
    python scripts/m2b_inference.py time --device cpu                     # the laptop

predict: every pass of ensemble.every_pass() (R3's arm's 3 seeds x 4 turns
x 1,280 and 1,536 px) on every photo of the set, one photo at a time, each
pass timed with its turning; each model runs once at each size on the
first photo before anything is timed. Its record, run.json in
results/runs/<id>-m2b-inference-passes-<set>/, holds the config, the code's
commit and each pass's seconds a photo, in photo order, never a photo's
name. The boxes go to passes.json.gz beside it, or, for v1-val's store
photos, into data/store/runs/<same id>/, which never goes into results/.
The weights are checked against m2b.LAST_PT_SHA256 first.

score: the newest finished pass record of each set; each of the 8
configurations (none to all three add-ons) fused (ensemble.fuse) and
scored on both sets with our own AP (detection_eval.average_precision);
each pass at 1,280 px checked against Ultralytics' mAP in its run's GPU
suite-m2b.json (within 0.01 on real-val, or the script says so and exits
with 1); then R4 with the noise band, and the "what if" with
m2b.WHAT_IF_BAND, both m2b.decide_addons. Counts at each configuration's
own F2 threshold on v1-val; CPU time a photo from v1-val's passes (the
laptop) and the fusion timed here. Writes inference.json in
results/runs/<id>-m2b-inference/: counts only.

time: R3's run at 1,280 and 1,536 px and m1280-ft-s0 at 1,280 px on
v1-val's photos, loaded upright into memory first; one photo a call,
rect=False, 2 warm-up calls each left out, then 3 repeats, the three taken
in turn within each repeat so the laptop's heat falls on all alike. Writes
timing.json in results/runs/<id>-m2b-inference-timing/: times only.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path

from bestbefore import detector, ensemble, m2, m2b, runs, sets, store
from bestbefore.detection_eval import Prediction, average_precision, best_threshold, evaluate
from bestbefore.images import load_upright
from bestbefore.yolo import CLASSES

SETS = (m2b.STORE_SET, "real-val")
AP_CHECK = 0.01           # ours against Ultralytics' on real-val, single models (the plan)
TIMED_OTHER = "m1280-ft-s0"   # timed beside R3's run, the owner's call (S13); no add-on, no switch
WARMUP, REPEATS = 2, 3


def chosen_arm() -> str:
    """R3's arm, from the chain's record, which must be m2b.decisions() worked out here."""
    chain, ours = m2.read_json(m2b.decisions_path()), m2b.decisions()
    if chain is None or "R3" not in ours:
        raise SystemExit("R3 isn't decided here: pull the runs (scripts/pull_cluster_results.sh)")
    if chain != ours:
        raise SystemExit("the chain's record differs from m2b.decisions() here: pull again and check")
    return ours["R3"]["arm"]


def processor() -> str:
    """The CPU's name: the laptop's from sysctl, a node's from /proc/cpuinfo."""
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        return out.stdout.strip()
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def machine(device: str | None) -> dict:
    import torch
    return {"device": device, "torch_threads": torch.get_num_threads(), "processor": processor(),
            "platform": platform.platform()}


def seconds(values: list[float]) -> dict:
    """A pass's or a model's seconds a photo: how many, mean, median, total."""
    return {"photos": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
            "total": sum(values), "per_photo": values}


# --- predict ------------------------------------------------------------------------------

def predict(name: str, device: str | None, limit: int | None = None) -> Path:
    from ultralytics import YOLO

    arm = chosen_arm()
    weights = {seed: m2b.last_pt(f"{arm}-s{seed}") for seed in m2b.SEEDS}
    samples = sets.load_set(name)[:limit]
    every = ensemble.every_pass()
    models = {seed: YOLO(str(path)) for seed, (path, _) in weights.items()}

    def predictor(seed: int, imgsz: int):
        return lambda image: detector.predict(models[seed], [image], imgsz, device)[0]

    started = time.time()
    first = load_upright(samples[0].image)
    for seed in models:                                     # warm-up, untimed, not kept
        for imgsz in (ensemble.IMGSZ, ensemble.IMGSZ_UP):
            predictor(seed, imgsz)(first)
    # Read after the warm-up: Ultralytics sets torch's threads when it first
    # predicts on a CPU (min(8, CPUs - 1)), whatever OMP_NUM_THREADS said.
    config = {"set": name, "arm": arm, "photos": len(samples), "limit": limit,
              "weights": {f"s{seed}": {"run": run, "sha256": m2b.LAST_PT_SHA256[f"{arm}-s{seed}"]}
                          for seed, (_, run) in weights.items()},
              "passes": [p.name for p in every], "predict_conf": detector.PREDICT_CONF, "rect": False,
              **machine(device)}
    # A smoke (--limit) is named apart, so `score` never takes it for the set's passes.
    record = runs.start_run(f"m2b-inference-passes-{name}" + ("-smoke" if limit else ""), config)
    private = name == m2b.STORE_SET
    folder = store.RUNS_DIR / record.name if private else record
    folder.mkdir(parents=True, exist_ok=True)
    boxes = {p.name: {} for p in every}
    taken = {p.name: [] for p in every}
    for k, sample in enumerate(samples):
        image = load_upright(sample.image)
        for p in every:
            t0 = time.perf_counter()
            found = ensemble.predict_turned(predictor(p.seed, p.imgsz), image, p.quarter)
            taken[p.name].append(time.perf_counter() - t0)
            boxes[p.name][sample.id] = [[CLASSES.index(b.label), *b.xyxy, b.conf] for b in found]
        if (k + 1) % 10 == 0 or k + 1 == len(samples):
            print(f"{k + 1}/{len(samples)} photos, {(time.time() - started) / 60:.1f} min", flush=True)
    data = gzip.compress(json.dumps({"set": name, "passes": boxes}).encode())
    (folder / "passes.json.gz").write_bytes(data)
    import torch
    runs.update_run(record, status="finished", torch_threads_used=torch.get_num_threads(),
                    boxes=str(folder / "passes.json.gz"),
                    boxes_sha256=hashlib.sha256(data).hexdigest(),
                    seconds={name: seconds(values) for name, values in taken.items()},
                    minutes=round((time.time() - started) / 60, 2))
    print(f"-> {record}" + (f"\n   boxes (store photos, private): {folder}" if private else ""))
    return record


# --- score --------------------------------------------------------------------------------

def pass_record(name: str, root: Path | None = None) -> tuple[Path, dict, dict[str, dict[str, list[Prediction]]]]:
    """The newest finished pass record of a set, and its boxes, by pass and
    photo, checked against the record's SHA-256. Its boxes file lies where
    the record says (v1-val's under data/store/runs/), else beside it (a
    record made on the cluster and copied here)."""
    base = root or runs.RUNS_DIR
    found = [d for d in sorted(base.glob(f"*-m2b-inference-passes-{name}"))
             if (m2.read_json(d / "run.json") or {}).get("status") == "finished"]
    if not found:
        raise SystemExit(f"no finished pass record for {name}: run `predict --set {name}` first")
    run_dir = found[-1]
    record = m2.read_json(run_dir / "run.json")
    path = Path(record["boxes"])
    if not path.exists():
        path = (store.RUNS_DIR if name == m2b.STORE_SET else run_dir.parent) / run_dir.name / "passes.json.gz"
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != record["boxes_sha256"]:
        raise SystemExit(f"{path} isn't the file {run_dir.name} recorded")
    raw = json.loads(gzip.decompress(data))["passes"]
    boxes = {p: {photo: [Prediction(CLASSES[int(b[0])], tuple(b[1:5]), b[5]) for b in found_]
                 for photo, found_ in photos.items()}
             for p, photos in raw.items()}
    return run_dir, record, boxes


def all_configurations() -> list[tuple[str, ...]]:
    """None, each add-on, each pair, and all three, in ADDONS' order."""
    return [c for k in range(len(m2b.ADDONS) + 1) for c in combinations(m2b.ADDONS, k)]


def fused(boxes: dict, addons: tuple[str, ...], photos: list[str]) -> tuple[dict[str, list[Prediction]], list[float]]:
    """Each photo's boxes for a configuration, and the seconds each fusion took."""
    names = [p.name for p in ensemble.passes(addons)]
    out, taken = {}, []
    for photo in photos:
        t0 = time.perf_counter()
        out[photo] = ensemble.fuse([boxes[name][photo] for name in names])
        taken.append(time.perf_counter() - t0)
    return out, taken


def scored(samples, preds, conf: float | None, with_size: bool) -> dict:
    """A set's entry, shaped as the M2b suite's so m2b_results' tables read
    it: found at IoU 0.5 and 0.3 and boxed at any confidence at `conf` (its
    own best F2 when None), by tag, v1-val's dates by size, and our AP as
    map50's date."""
    at_05, at_03 = evaluate(samples, preds, 0.5), evaluate(samples, preds, 0.3)
    if conf is None:
        conf, f2 = best_threshold(at_05, beta=m2.THRESHOLD_BETA)
    else:
        f2 = None
    entry = {"threshold": conf, "f2_at_threshold": f2, **detector._summaries(at_05, at_03, conf),
             "map50": {"date": average_precision(at_05)}}
    tags = [t for t in dict.fromkeys(s.tag for s in samples) if t]
    if tags:
        entry["by_tag"] = {t: detector._summaries(at_05, at_03, conf, {t}) for t in tags}
    if with_size:
        entry["by_size"] = {band: {k: v for k, v in counts.items() if k != "found_at_threshold_m2"}
                            for band, counts in m2b.found_by_size(samples, at_05, conf, conf).items()}
    return entry


def score() -> Path:
    arm = chosen_arm()
    decided = m2b.decisions()
    samples = {name: sets.load_set(name) for name in SETS}
    loaded = {name: pass_record(name) for name in SETS}
    for name, (run_dir, record, boxes) in loaded.items():
        if record["config"]["arm"] != arm:
            raise SystemExit(f"{run_dir.name} predicted with {record['config']['arm']}, not R3's {arm}")
        ids = [s.id for s in samples[name]]
        if any(set(photos) != set(ids) for photos in boxes.values()) or len(boxes) != len(ensemble.every_pass()):
            raise SystemExit(f"{run_dir.name} doesn't hold every pass on every photo of {name}")
    v1_seconds = loaded[m2b.STORE_SET][1]["seconds"]
    ensemble.fuse([[], []])        # loads ensemble-boxes (and numba) before any fusion is timed

    configurations = {}
    for addons in all_configurations():
        name = m2b.configuration(addons)
        boxes, fusion = fused(loaded[m2b.STORE_SET][2], addons, [s.id for s in samples[m2b.STORE_SET]])
        v1 = scored(samples[m2b.STORE_SET], boxes, None, with_size=True)
        real_boxes, _ = fused(loaded["real-val"][2], addons, [s.id for s in samples["real-val"]])
        real = scored(samples["real-val"], real_boxes, v1["threshold"], with_size=False)
        used = ensemble.passes(addons)
        predict_ms = sum(v1_seconds[p.name]["mean"] for p in used) * 1000
        fusion_ms = statistics.fmean(fusion) * 1000
        configurations[name] = {"addons": list(addons), "passes": [p.name for p in used],
                                "threshold": v1["threshold"], m2b.STORE_SET: v1, "real-val": real,
                                "predict_ms_per_photo": predict_ms, "fusion_ms_per_photo": fusion_ms,
                                "cpu_ms_per_photo": predict_ms + fusion_ms}
        print(f"{name:32s} v1-val AP {v1['map50']['date']:.4f}  real-val AP {real['map50']['date']:.4f}  "
              f"{predict_ms + fusion_ms:7.0f} ms a photo", flush=True)

    check = ap_check(arm, loaded, samples)
    aps = {name: {s: c[s]["map50"]["date"] for s in SETS} for name, c in configurations.items()}
    record_threshold = decided["R3"]["threshold"]
    walks = {}
    for key, band in (("R4", decided["noise_store"]["band"]), ("what_if", m2b.WHAT_IF_BAND)):
        walk = m2b.decide_addons(aps, band)
        walk["threshold"] = (configurations[walk["configuration"]]["threshold"] if walk["adopted"]
                             else record_threshold)
        walks[key] = walk
    out = {"arm": arm, "run": decided["R3"]["run"], "threshold_record": record_threshold,
           "threshold_m2_record": m2b.suite_of(decided["R3"]["run"])["threshold_m2"],
           "sources": {name: {"run": run_dir.name, "boxes_sha256": record["boxes_sha256"],
                              "git_commit": record["git_commit"], "git_dirty": record["git_dirty"],
                              **{k: record["config"][k] for k in ("device", "torch_threads", "processor",
                                                                   "platform", "photos")},
                              # what it predicted with: None in a record made before 4f1e220 that
                              # no one noted (its config's count was read too early)
                              "torch_threads_used": record.get("torch_threads_used")}
                       for name, (run_dir, record, _) in loaded.items()},
           "wbf": {"iou_thr": ensemble.WBF_IOU, "conf_type": ensemble.WBF_CONF,
                   "skip_box_thr": detector.PREDICT_CONF},
           "configurations": configurations, "ap_check": check, **walks}
    run_dir = runs.start_run("m2b-inference", {"arm": arm, "sources": {k: v["run"] for k, v in out["sources"].items()}})
    runs.write_json(run_dir / "inference.json", out)
    runs.update_run(run_dir, status="finished")
    for key in ("R4", "what_if"):
        w = walks[key]
        print(f"{key} (band {w['band']:.4f}): adopts {', '.join(w['adopted']) or 'nothing'}; "
              f"threshold {w['threshold']:.4f}")
    print(f"-> {run_dir / 'inference.json'}")
    if not check["agrees"]:
        print(f"**AP check fails**: ours and Ultralytics' differ by more than {AP_CHECK} on real-val; stop.")
        sys.exit(1)
    return run_dir


def ap_check(arm: str, loaded: dict, samples: dict) -> dict:
    """Each seed's pass at 1,280 px, upright and unfused, scored with our AP
    on both sets, against Ultralytics' mAP@0.5 for dates in its run's own
    GPU suite-m2b.json. Only real-val's must agree within AP_CHECK."""
    seeds = {}
    for seed in m2b.SEEDS:
        suite = m2b.suite_of(f"{arm}-s{seed}")
        name = ensemble.Pass(seed, 0, ensemble.IMGSZ).name
        row = {}
        for set_name in SETS:
            ours = average_precision(evaluate(samples[set_name], loaded[set_name][2][name], 0.5))
            theirs = suite[set_name]["map50"]["date"]
            row[set_name] = {"ours": ours, "ultralytics": theirs, "difference": ours - theirs}
        seeds[f"s{seed}"] = row
    worst = max(abs(r["real-val"]["difference"]) for r in seeds.values())
    return {"within": AP_CHECK, "largest_real_val_difference": worst, "agrees": worst <= AP_CHECK,
            "seeds": seeds}


# --- time ---------------------------------------------------------------------------------

def time_models(device: str | None) -> Path:
    from ultralytics import YOLO

    arm = chosen_arm()
    run = f"{arm}-s0"
    timed = ((run, ensemble.IMGSZ), (run, ensemble.IMGSZ_UP), (TIMED_OTHER, ensemble.IMGSZ))
    weights = {name: m2b.last_pt(name) for name in dict.fromkeys(n for n, _ in timed)}
    samples = m2b.load_v1_val()
    images = [load_upright(s.image) for s in samples]
    models = {name: YOLO(str(path)) for name, (path, _) in weights.items()}
    for name, size in timed:
        for _ in range(WARMUP):
            detector.predict(models[name], [images[0]], size, device)
    config = {"timed": [f"{n}@{size}" for n, size in timed], "photos": len(images),
              "photo_long_side_median": statistics.median(max(im.size) for im in images),
              "warmup": WARMUP, "repeats": REPEATS, "batch": 1, "rect": False,
              "weights": {n: {"run": r, "sha256": m2b.LAST_PT_SHA256[n]} for n, (_, r) in weights.items()},
              **machine(device)}           # after the warm-up, as predict's
    record = runs.start_run("m2b-inference-timing", config)
    taken = {f"{n}@{size}": [[] for _ in range(REPEATS)] for n, size in timed}
    started = time.time()
    for r in range(REPEATS):
        for name, size in timed:
            for image in images:
                t0 = time.perf_counter()
                detector.predict(models[name], [image], size, device)
                taken[f"{name}@{size}"][r].append(time.perf_counter() - t0)
            print(f"repeat {r + 1}: {name}@{size} {statistics.fmean(taken[f'{name}@{size}'][r]) * 1000:.0f} ms "
                  f"a photo", flush=True)
    out = {key: {"median_ms": statistics.median(sum(reps, [])) * 1000,
                 "mean_ms": statistics.fmean(sum(reps, [])) * 1000,
                 "repeat_mean_ms": [statistics.fmean(r) * 1000 for r in reps],
                 "per_photo_s": reps}
           for key, reps in taken.items()}
    runs.write_json(record / "timing.json", {"config": config, "timed": out})
    runs.update_run(record, status="finished", minutes=round((time.time() - started) / 60, 2))
    print(f"-> {record / 'timing.json'}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict", help="save every pass on a set")
    p.add_argument("--set", required=True, choices=SETS)
    p.add_argument("--device", default=None, help="e.g. cpu; default: auto")
    p.add_argument("--limit", type=int, default=None, help="the first N photos only (a smoke)")
    sub.add_parser("score", help="fuse, score, apply R4 and the what-if")
    t = sub.add_parser("time", help="n against m on the CPU")
    t.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "predict":
        predict(args.set, args.device, args.limit)
    elif args.command == "score":
        score()
    else:
        time_models(args.device)


if __name__ == "__main__":
    main()
