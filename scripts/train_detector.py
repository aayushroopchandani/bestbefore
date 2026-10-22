"""Train one date detector: an M1 run, or one of M2's arms.

M1 (the default, unchanged):

    python scripts/train_detector.py --weights yolo11s --imgsz 1280 \\
        --train train_synth,train_real --seed 0 --name s1280

Writes results/runs/<id>/:
  run.json   flags, git commit, seed, versions, device, checkpoint path
  data.yaml  the dataset file this run trained on
  val.json   validation scores: mAP@0.5 per class, the F1-best threshold
             for dates, recall and precision at it (IoU 0.5 and 0.3), and
             Ultralytics' own date recall at that threshold as a cross-check
Ultralytics' output (weights, curves) goes to $CHECKPOINT_DIR/m1/<id>/.
A run whose val.json exists is complete; slurm/m1_chain.sh skips it.

M2 (docs/m2-synthetic-generator.md, S3):

    python scripts/train_detector.py --milestone m2 --arm gen --seed 0 --name gen-s0

The arm gives the lists, the real photos' repeats, the input size and the
schedule; the recipe is m2.RECIPE. --budget-share trains on part of the
budget (the pilot: 0.1); --epochs and --fraction override it for smoke and
speed runs; --workers and --cache ram set the loader (the RAM cache only
if m2.ram_check finds it fits in the job's memory). Writes
results/runs/<id>/run.json and data.yaml. run.json holds the arm's lists
and counts, the schedule, the loader, the manifests' hashes, what
Ultralytics applied (checked against the recipe when training starts: the
run stops if anything differs) and a status, running then finished. The
model is the final weights, $CHECKPOINT_DIR/m2/<id>/weights/last.pt, which
scripts/score_suite.py scores.
"""
from __future__ import annotations

import argparse
import json
import time

from bestbefore import detector, m2
from bestbefore.config import checkpoint_dir, expdate_dir
from bestbefore.runs import start_run, update_run
from bestbefore.sets import load_set

# What an M1 run records as its config, in the order it always has.
M1_FLAGS = ("weights", "imgsz", "train", "seed", "name", "epochs", "patience",
            "batch", "fraction", "workers", "device")
M1_ONLY = ("weights", "imgsz", "train", "patience", "batch")


def device_name(device) -> str:
    import torch
    if torch.cuda.is_available() and device not in ("cpu", "mps"):
        return torch.cuda.get_device_name(0)
    return str(device or "cpu")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--milestone", choices=("m1", "m2"), default="m1")
    parser.add_argument("--name", required=True, help="short run name, e.g. s1280 or base-s0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int,
                        help=f"m1: default {detector.SETTINGS['epochs']}; m2: replaces the arm's")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="share of the training photos to use (smoke tests only)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None, help="e.g. 0, cpu or mps; default: auto")
    only_m1 = parser.add_argument_group("m1 only")
    only_m1.add_argument("--weights", help="yolo11n or yolo11s")
    only_m1.add_argument("--imgsz", type=int)
    only_m1.add_argument("--train", help="comma-separated image lists, e.g. train_synth,train_real")
    only_m1.add_argument("--patience", type=int)
    only_m1.add_argument("--batch", type=int)
    only_m2 = parser.add_argument_group("m2 only")
    only_m2.add_argument("--arm", choices=list(m2.ARMS))
    only_m2.add_argument("--budget-share", type=float, default=1.0,
                         help="train on this share of the budget (the pilot: 0.1)")
    only_m2.add_argument("--cache", choices=["ram"], help="RAM cache, if it fits the job's memory")
    args = parser.parse_args(argv)

    if args.milestone == "m1":
        if None in (args.weights, args.imgsz, args.train):
            parser.error("an m1 run needs --weights, --imgsz and --train")
        if args.arm or args.cache or args.budget_share != 1.0:
            parser.error("--arm, --budget-share and --cache are for m2 runs")
        for flag in ("epochs", "patience", "batch"):
            if getattr(args, flag) is None:
                setattr(args, flag, detector.SETTINGS[flag])
    else:
        if args.arm is None:
            parser.error("an m2 run needs --arm")
        given = [f"--{flag}" for flag in M1_ONLY if getattr(args, flag) is not None]
        if given:
            parser.error(f"{', '.join(given)}: an m2 run takes these from its arm and m2.RECIPE")
    return args


def train_m1(args: argparse.Namespace) -> None:
    yolo_dir = expdate_dir() / "yolo"
    run_dir = start_run(f"m1-{args.name}", {k: getattr(args, k) for k in M1_FLAGS},
                        seed=args.seed)
    data = detector.dataset_yaml(yolo_dir, args.train.split(","), run_dir / "data.yaml")
    project = checkpoint_dir() / "m1"
    update_run(run_dir, device_name=device_name(args.device),
               checkpoint_dir=str(project / run_dir.name))
    print(f"run {run_dir.name} on {device_name(args.device)}", flush=True)

    started = time.time()
    best = detector.train(
        detector.pretrained_path(args.weights), data, args.imgsz, args.seed,
        project, run_dir.name, epochs=args.epochs, patience=args.patience,
        batch=args.batch, fraction=args.fraction, workers=args.workers,
        device=args.device)
    update_run(run_dir, checkpoint=str(best), epochs_run=detector.epochs_run(best),
               train_minutes=round((time.time() - started) / 60, 1))

    scores = detector.score(best, load_set("real-val"), args.imgsz, data=data,
                            split="val", project=project / run_dir.name,
                            device=args.device)
    (run_dir / "val.json").write_text(json.dumps(scores, indent=2))
    s = scores["iou0.5"]
    print(f"validation: mAP@0.5 date {scores['map50'].get('date')}, "
          f"threshold {scores['threshold']}, date recall {s['date_recall']}, "
          f"precision {s['precision']}  ->  {run_dir}")


def train_m2(args: argparse.Namespace) -> None:
    arm = m2.ARMS[args.arm]
    missing = [reason for reason in map(m2.list_missing, arm.names()) if reason]
    if missing:
        raise SystemExit(f"can't train {args.arm} yet: {'; '.join(missing)}")
    counts = {name: m2.count_listed(name) for name in arm.names()}
    mix = m2.mix(arm, counts)
    schedule = m2.schedule(mix.n_listed, args.budget_share, args.epochs)
    images = m2.expected_images(mix.n_listed, args.fraction)
    loader, ram = {"workers": args.workers, "cache": False}, None
    if args.cache == "ram":
        ram = m2.ram_check(m2.listed_images(mix.lists), arm.imgsz)
        loader["cache"] = "ram" if ram["fits"] else False

    config = {"milestone": "m2", "arm": args.arm, "weights": m2.WEIGHTS, "imgsz": arm.imgsz,
              "seed": args.seed, "name": args.name, "lists": list(mix.lists), "counts": counts,
              "real_repeats": mix.real_repeats, "n_listed": mix.n_listed,
              "budget_share": args.budget_share, "fraction": args.fraction,
              "train_images": images, "schedule": schedule, "recipe": m2.RECIPE,
              "loader": loader, "ram_check": ram, "device": args.device,
              "manifests": m2.manifest_hashes(arm.names())}
    run_dir = start_run(f"m2-{args.name}", config, seed=args.seed)
    data = detector.dataset_yaml(expdate_dir() / "yolo", [m2.list_spec(s) for s in mix.lists],
                                 run_dir / "data.yaml")
    project = checkpoint_dir() / "m2"
    update_run(run_dir, status="running", device_name=device_name(args.device),
               checkpoint_dir=str(project / run_dir.name))
    print(f"run {run_dir.name} on {device_name(args.device)}: {args.arm}, "
          f"{mix.n_listed} photos listed ({', '.join(mix.lists)}), "
          f"{schedule['epochs']} epochs at {arm.imgsz} px", flush=True)

    wanted = {**m2.RECIPE, **schedule, **loader, "imgsz": arm.imgsz, "seed": args.seed,
              "fraction": args.fraction}

    def check_start(trainer) -> None:
        got = detector.applied(trainer)
        update_run(run_dir, applied=got)        # training has started: the watchdog's progress
        problems = m2.recipe_problems(got, wanted, images)
        if problems:
            raise RuntimeError("Ultralytics isn't training M2's recipe: " + "; ".join(problems))
        print(f"applied as asked: {got['optimizer']}, {got['train_images']} training photos "
              f"({got['distinct_images']} distinct), RAM cache {got['ram_cache']}", flush=True)

    def record_end(trainer) -> None:
        update_run(run_dir, momentum_after_warmup=detector.momenta(trainer))

    started = time.time()
    last = detector.train(
        detector.pretrained_path(m2.WEIGHTS), data, arm.imgsz, args.seed, project,
        run_dir.name, keep="last",
        callbacks={"on_train_start": check_start, "on_train_end": record_end},
        fraction=args.fraction, device=args.device, **loader, **m2.RECIPE, **schedule)
    update_run(run_dir, status="finished", checkpoint=str(last),
               epochs_run=detector.epochs_run(last),
               train_minutes=round((time.time() - started) / 60, 1))
    print(f"finished: {last}  ->  {run_dir}", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    (train_m1 if args.milestone == "m1" else train_m2)(args)


if __name__ == "__main__":
    main()
