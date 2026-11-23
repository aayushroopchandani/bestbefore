"""M2c S1: does the exported detector see what PyTorch sees? (docs/m2c-live-boxes.md, R0 to R2)

    python scripts/live_parity.py run [--set NAME ...] [--variant NAME ...] [--limit N]
                                      [--threads N] [--int8-dir DIR] [--name NAME]
    python scripts/live_parity.py rules RUN

`run`: each variant (live.VARIANTS) loaded once through Ultralytics, M2b's
last.pt through PyTorch and each .onnx file through Ultralytics' own ONNX
backend, so the letterbox and the NMS are Ultralytics' for both; then every
photo of each set (default: the five real sets, 570 photos), one photo a
call at conf 0.001, square (rect=False) or tight (rect=True). Every ONNX
Runtime session runs --threads threads (Ultralytics sets none: CLAUDE.md,
M3 S1), and PyTorch too once Ultralytics has set its own at the first
prediction. The input shape each photo reached the model with is recorded
and checked: 1,280 square, or tight (each side a multiple of 32, the long
one 1,280). --limit N takes each set's first N photos (a dry run).

Each set and variant gets parity-<set>-<variant>.json in the run's folder
under results/runs/m2c/: every box at conf >= 0.001 per photo, photos named
by set and index only; AP@0.5 for dates (detection_eval), found at IoU 0.5
at M2b's threshold_store (live.threshold, never chosen again), expiries
found on v1-val, false boxes a photo; each labelled date's matched
confidence; ms a call (the photo already decoded). The last line printed is
the run's folder.

`rules`: R0 (PyTorch square against ONNX float32 square), R1 (float32 tight
against square) and R2's accuracy half (int8 and int8-pct against float32,
at each padding) on every set the run holds, into RUN/rules.json. A rule
whose five sets aren't all there is read per set only.
"""
from __future__ import annotations

import os

os.environ["YOLO_AUTOINSTALL"] = "false"     # before anything imports Ultralytics
os.environ.setdefault("YOLO_OFFLINE", "1")

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path


def write_compact(path: Path, data) -> None:
    """JSON without indentation (the boxes make it large), atomically."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")))
    os.replace(tmp, path)


def input_shapes(backend) -> list[list[int]]:
    """The shape of each input the model is given from now on: ONNX
    Runtime's session.run wrapped, or a forward pre-hook on PyTorch's module."""
    shapes: list[list[int]] = []
    if hasattr(backend, "session"):
        run = backend.session.run

        def capture(output_names, feed, *args, **kwargs):
            shapes.append(list(next(iter(feed.values())).shape))
            return run(output_names, feed, *args, **kwargs)

        backend.session.run = capture
    else:
        def hook(module, args, kwargs):
            x = args[0] if args else kwargs["x"]
            shapes.append(list(x.shape))

        backend.model.register_forward_pre_hook(hook, with_kwargs=True)
    return shapes


def padding_problems(shapes: list[list[int]], tight: bool, imgsz: int, stride: int) -> list[str]:
    """Inputs that aren't the padding asked for."""
    if not tight:
        return [f"{s} isn't the {imgsz} square" for s in shapes if s[2:] != [imgsz, imgsz]]
    return [f"{s} isn't tight" for s in shapes if s[2] % stride or s[3] % stride or max(s[2:]) != imgsz]


def run(set_names: list[str], variants: list[str], limit: int | None, threads: int,
        int8_dir: Path, name: str) -> Path:
    import torch
    from ultralytics import YOLO

    from bestbefore import detector, live, m2b, runs, sets
    from bestbefore.detection_eval import average_precision, evaluate, summary
    from bestbefore.images import load_upright
    from bestbefore.samples import DATE_LABELS
    from bestbefore.yolo import CLASSES

    live.check_no_autoinstall()
    folder = live.models_dir()
    files = {live.VARIANTS[v][0] for v in variants} - {"pt"}
    problems = (live.check_files(folder, sorted(files & set(live.FLOAT_FILES)))
                + live.check_files(int8_dir, sorted(files & set(live.INT8_FILES))))
    if problems:
        sys.exit("\n".join(problems))
    weights, _ = m2b.last_pt(live.SOURCE)       # PyTorch's: M2b's own last.pt, its hash checked
    paths = {"pt": weights, **{f: (int8_dir if f in live.INT8_FILES else folder) / live.FILES[f] for f in files}}
    manifests = {"models": live.read_manifest(folder), "int8": live.read_manifest(int8_dir)}
    threshold = live.threshold()
    samples = {s: sets.load_set(s)[:limit] for s in set_names}
    config = {"sets": set_names, "variants": variants, "limit": limit, "threads": threads,
              "imgsz": live.IMGSZ, "conf": live.PREDICT_CONF, "threshold": threshold,
              "models": {f: {"path": str(p), "sha256": m2b.LAST_PT_SHA256[live.SOURCE] if f == "pt"
                             else (manifests["int8"] if f in live.INT8_FILES else manifests["models"])["files"][f]["sha256"]}
                         for f, p in paths.items()},
              "int8_calibration": manifests["int8"].get("int8", {}).get("calibration"),
              "machine": f"{platform.system()} {platform.machine()} {platform.processor()}",
              "environment": live.environment()}
    run_dir = runs.start_run(name, config, root=runs.RUNS_DIR / "m2c")
    print(f"{run_dir}: threshold {threshold:.4f}, {threads} threads, "
          f"{sum(len(v) for v in samples.values())} photos x {len(variants)} variants", flush=True)
    first = load_upright(samples[set_names[0]][0].image)
    threads_used = {}
    for variant in variants:
        key, tight = live.VARIANTS[variant]
        with live.ort_threads(threads) as made:
            model = YOLO(str(paths[key]), task="detect")
            detector.predict(model, [first], live.IMGSZ, device="cpu", rect=tight)     # warm-up: builds the session
        backend = model.predictor.model.backend
        if key == "pt":
            torch.set_num_threads(threads)      # Ultralytics set its own at the warm-up
            threads_used[variant] = {"torch": torch.get_num_threads()}
        else:
            session = backend.session.get_session_options().intra_op_num_threads
            if made != [threads] or session != threads:
                sys.exit(f"{variant}: ONNX Runtime sessions made with {made} threads ({session}), not {threads}")
            threads_used[variant] = {"onnxruntime": session}
        shapes = input_shapes(backend)
        for set_name in set_names:
            shapes.clear()
            preds, photos = {}, []
            for i, s in enumerate(samples[set_name]):
                image = load_upright(s.image)
                t0 = time.perf_counter()
                found = detector.predict(model, [image], live.IMGSZ, device="cpu", rect=tight)[0]
                ms = (time.perf_counter() - t0) * 1000
                preds[s.id] = found
                photos.append({"index": i, "size": [s.width, s.height], "input": shapes[-1], "ms": round(ms, 1),
                               "boxes": [[CLASSES.index(b.label), *(round(v, 5) for v in b.xyxy), round(b.conf, 5)]
                                         for b in found]})
            if len(shapes) != len(photos):
                sys.exit(f"{variant} on {set_name}: {len(shapes)} model inputs for {len(photos)} photos")
            if bad := padding_problems(shapes, tight, live.IMGSZ, live.STRIDE):
                sys.exit(f"{variant} on {set_name}: {bad[:3]}")
            at_05 = evaluate(samples[set_name], preds, 0.5)
            ms = [p["ms"] for p in photos]
            record = {"set": set_name, "variant": variant, "file": key, "tight": tight,
                      "sha256": config["models"][key]["sha256"], "threshold": threshold,
                      "imgsz": live.IMGSZ, "conf": live.PREDICT_CONF, "photos": len(photos),
                      "ap50_date": average_precision(at_05), "at_threshold": summary(at_05, threshold),
                      "any_conf": summary(at_05, 0.0),
                      "dates": {"conf": at_05.truth_conf, "expiry": at_05.truth_expiry,
                                "photo": [i for i, s in enumerate(samples[set_name])
                                          for b in s.boxes if b.label in DATE_LABELS]},
                      "ms": {"median": round(statistics.median(ms), 1), "mean": round(statistics.fmean(ms), 1),
                             "max": max(ms)},
                      "threads": threads_used[variant], "per_photo": photos}
            write_compact(run_dir / f"parity-{set_name}-{variant}.json", record)
            t = record["at_threshold"]
            expiry = f", expiries {t['expiry_found']}/{t['expiry_dates']}" if "expiry_dates" in t else ""
            print(f"{variant:16} {set_name:16} AP {record['ap50_date']:.4f}, dates {t['dates_found']}/{t['dates']}"
                  f"{expiry}, false {t['false_boxes_per_photo']}/photo, {record['ms']['median']} ms (median)", flush=True)
    runs.update_run(run_dir, status="finished", threads_used=threads_used,
                    ended=time.strftime("%Y-%m-%dT%H:%M:%S"))
    print(run_dir)
    return run_dir


def rules(run_dir: Path) -> dict:
    from bestbefore import live, runs

    records = {}
    for path in sorted(run_dir.glob("parity-*.json")):
        r = json.loads(path.read_text())
        records[(r["set"], r["variant"])] = r
    per_set = {}
    for name in live.SETS:
        get = {v: records.get((name, v)) for v in live.VARIANTS}
        if not any(get.values()):
            continue
        entry = {}
        if get["pt-square"] and get["fp32-square"]:
            entry["R0"] = live.r0(get["pt-square"], get["fp32-square"])
        if get["fp32-square"] and get["fp32-tight"]:
            entry["R1"] = live.r1(get["fp32-square"], get["fp32-tight"])
        for low in live.INT8_FILES:
            for pad in ("square", "tight"):
                if get[f"fp32-{pad}"] and get[f"{low}-{pad}"]:
                    entry[f"R2 {low} {pad}"] = live.r2_accuracy(get[f"fp32-{pad}"], get[f"{low}-{pad}"])
        per_set[name] = entry

    def verdict(rule: str) -> bool | str:
        missing = [s for s in live.SETS if rule not in per_set.get(s, {})]
        if missing:
            return f"incomplete: no {', '.join(missing)}"
        return all(per_set[s][rule]["passed"] for s in live.SETS)

    out = {"run": run_dir.name, "sets": per_set, "R0": verdict("R0"), "R1": verdict("R1")}
    pad = "tight" if out["R1"] is True else "square" if out["R1"] is False else None
    if pad:
        for low in live.INT8_FILES:
            out[f"R2 {low} {pad}"] = verdict(f"R2 {low} {pad}")
        # R2's accuracy half: int8 if it passes, else its fallback, else float32
        out["R2 webassembly precision (accuracy half)"] = next(
            (low for low in live.INT8_FILES if out[f"R2 {low} {pad}"] is True),
            "fp32" if all(isinstance(out[f"R2 {low} {pad}"], bool) for low in live.INT8_FILES) else "incomplete")
    runs.write_json(run_dir / "rules.json", out)
    for name, entry in per_set.items():
        for rule, reading in entry.items():
            print(f"{name:16} {rule:18} {json.dumps(reading)}")
    for key in [k for k in out if k.startswith("R")]:
        print(f"{key}: {out[key]}")
    return out


def main() -> None:
    from bestbefore import live

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", action="append", choices=live.SETS)
    r.add_argument("--variant", action="append", choices=list(live.VARIANTS))
    r.add_argument("--limit", type=int)
    r.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    r.add_argument("--int8-dir", type=Path, default=None, help="default: $CHECKPOINT_DIR/m2c")
    r.add_argument("--name", default="m2c-parity")
    q = sub.add_parser("rules")
    q.add_argument("run", type=Path)
    args = p.parse_args()
    if args.cmd == "run":
        run(args.set or list(live.SETS), args.variant or list(live.VARIANTS), args.limit, args.threads,
            args.int8_dir or live.models_dir(), args.name)
    else:
        rules(args.run)


if __name__ == "__main__":
    main()
