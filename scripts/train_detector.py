"""Train one date detector: an M1 run, one of M2's arms, or an M2b step-1 arm.

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

M2b (docs/m2b-store-and-size.md, S2):

    python scripts/train_detector.py --milestone m2b --arm s1280 --seed 0 --name s1280-s0

A step-1 arm (m2b.STEP1: s1280, m1280) is M2's gen-rot-1280 trained from
yolo11s's or yolo11m's COCO weights: M2's path, with the same lists,
schedule, recipe and checks. Only the weights change, and the record adds
the size, the M2 arm whose lists it trains on, and the starting weights'
SHA-256. Ultralytics' output goes to $CHECKPOINT_DIR/m2b/<id>/. n1280 is
never trained here: it is M2's own runs, recorded by m2b.write_control.
--smoke-lists base trains on base's lists instead, keeping the input size,
for the laptop's smoke run (the laptop has base's lists, not
gen-rot-1280's); it needs --fraction under 1.

M2b step 2 (docs/m2b-store-and-size.md, S10):

    python scripts/train_detector.py --milestone m2b --arm n1280-ft --seed 0 --name n1280-ft-s0

A step-2 arm (m2b.STEP2) starts from its step-1 run's weights, seed for
seed, as that run's record names them (n1280: the control's, M2's last.pt,
hash checked). The store export is checked against its manifest first.
m2b.mix draws the run's lists with its seed: the store lines and a replay
sample, written into results/runs/<id>/lists/. Each stage of the recipe is
one Ultralytics training (ft one, ft2 two: the backbone frozen, then all
at half the rate), each checked against its recipe when it starts and
recorded under "stages"; ft2's first stage goes to $CHECKPOINT_DIR/m2b/
<id>-stage1/, the last to <id>/. After a frozen stage the frozen layers
are compared with the weights it started from: a convolution or
batch-norm weight that moved stops the run; batch-norm statistics are
measured and recorded. --epoch-photos lists fewer photos per epoch, for
smoke runs only (with --epochs).
"""
from __future__ import annotations

import argparse
import json
import time

from bestbefore import detector, m2, m2b, store_train
from bestbefore.config import checkpoint_dir, expdate_dir, store_train_dir
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
    parser.add_argument("--milestone", choices=("m1", "m2", "m2b"), default="m1")
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
    only_m2 = parser.add_argument_group("m2 and m2b only")
    only_m2.add_argument("--arm", choices=[*m2.ARMS, *m2b.STEP1, *m2b.STEP2],
                         help="m2: one of M2's arms; m2b: a step-1 arm (s1280 or m1280) "
                              "or a step-2 arm (n1280-ft, ...)")
    only_m2.add_argument("--budget-share", type=float, default=1.0,
                         help="train on this share of the budget (the pilot: 0.1)")
    only_m2.add_argument("--cache", choices=["ram"], help="RAM cache, if it fits the job's memory")
    only_m2b = parser.add_argument_group("m2b only")
    only_m2b.add_argument("--smoke-lists", choices=list(m2.ARMS),
                          help="step 1: train on this M2 arm's lists instead, keeping the input "
                               "size (the laptop's smoke run: base); needs --fraction under 1")
    only_m2b.add_argument("--epoch-photos", type=int,
                          help="step 2: photos listed per epoch instead of "
                               f"{m2b.EPOCH_PHOTOS:,} (smoke runs; needs --epochs)")
    args = parser.parse_args(argv)

    if args.milestone == "m1":
        if None in (args.weights, args.imgsz, args.train):
            parser.error("an m1 run needs --weights, --imgsz and --train")
        if args.arm or args.cache or args.budget_share != 1.0:
            parser.error("--arm, --budget-share and --cache are for m2 runs")
        if args.smoke_lists or args.epoch_photos is not None:
            parser.error("--smoke-lists and --epoch-photos are for m2b smoke runs")
        for flag in ("epochs", "patience", "batch"):
            if getattr(args, flag) is None:
                setattr(args, flag, detector.SETTINGS[flag])
    else:
        if args.arm is None:
            parser.error(f"an {args.milestone} run needs --arm")
        arms = m2.ARMS if args.milestone == "m2" else {**m2b.STEP1, **m2b.STEP2}
        if args.arm not in arms:
            parser.error(f"--arm {args.arm}: an {args.milestone} arm is one of {', '.join(arms)}")
        given = [f"--{flag}" for flag in M1_ONLY if getattr(args, flag) is not None]
        if given:
            parser.error(f"{', '.join(given)}: an {args.milestone} run takes these from its arm "
                         "and its recipe")
        step2 = args.milestone == "m2b" and args.arm in m2b.STEP2
        if args.smoke_lists and (args.milestone != "m2b" or args.fraction >= 1 or step2):
            parser.error("--smoke-lists is for m2b step-1 smoke runs, with --fraction under 1")
        if args.epoch_photos is not None and (not step2 or args.epochs is None):
            parser.error("--epoch-photos is for m2b step-2 smoke runs, with --epochs")
        if step2 and (args.cache or args.budget_share != 1.0):
            parser.error("--cache and --budget-share aren't for step 2: no RAM cache, "
                         "and its budget is m2b.BUDGET_FT")
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
    train_arm(args, m2.ARMS[args.arm], m2.WEIGHTS)


def train_m2b(args: argparse.Namespace) -> None:
    """A step-1 arm: its M2 arm with the size's COCO weights, or, for the
    laptop's smoke run, another M2 arm's lists at the same input size. A
    step-2 arm: train_step2."""
    if args.arm in m2b.STEP2:
        train_step2(args)
        return
    step = m2b.STEP1[args.arm]
    arm = step.arm
    if args.smoke_lists:
        lists = m2.ARMS[args.smoke_lists]
        arm = m2.Arm(lists.real, lists.other, imgsz=arm.imgsz)
    weights_sha256 = m2b.sha256(detector.pretrained_path(step.weights))
    train_arm(args, arm, step.weights, size=step.size, lists_from=args.smoke_lists or step.m2_arm,
              weights_sha256=weights_sha256)


def train_arm(args: argparse.Namespace, arm: m2.Arm, weights: str, **more) -> None:
    """M2's training path, for an M2 arm (args.milestone m2) or an M2b
    step-1 arm (m2b): `more` joins the record's config after M2's fields."""
    if why := m2.missing({name: m2.list_missing(name) for name in arm.names()}):
        raise SystemExit(f"can't train {args.arm} yet: {why}")
    counts = {name: m2.count_listed(name) for name in arm.names()}
    mix = m2.mix(arm, counts)
    schedule = m2.schedule(mix.n_listed, args.budget_share, args.epochs)
    images = m2.expected_images(mix.n_listed, args.fraction)
    loader, ram = {"workers": args.workers, "cache": False}, None
    if args.cache == "ram":
        ram = m2.ram_check(m2.listed_images(mix.lists), arm.imgsz)
        loader["cache"] = "ram" if ram["fits"] else False

    config = {"milestone": args.milestone, "arm": args.arm, "weights": weights, "imgsz": arm.imgsz,
              "seed": args.seed, "name": args.name, "lists": list(mix.lists), "counts": counts,
              "real_repeats": mix.real_repeats, "n_listed": mix.n_listed,
              "budget_share": args.budget_share, "fraction": args.fraction,
              "train_images": images, "schedule": schedule, "recipe": m2.RECIPE,
              "loader": loader, "ram_check": ram, "device": args.device,
              "manifests": m2.manifest_hashes(arm.names()), **more}
    run_dir = start_run(f"{args.milestone}-{args.name}", config, seed=args.seed)
    data = detector.dataset_yaml(expdate_dir() / "yolo", [m2.list_spec(s) for s in mix.lists],
                                 run_dir / "data.yaml")
    project = checkpoint_dir() / args.milestone
    update_run(run_dir, status="running", device_name=device_name(args.device),
               checkpoint_dir=str(project / run_dir.name))
    print(f"run {run_dir.name} on {device_name(args.device)}: {args.arm}, "
          f"{mix.n_listed} photos listed ({', '.join(mix.lists)}), "
          f"{schedule['epochs']} epochs at {arm.imgsz} px", flush=True)

    # The loader isn't checked, only recorded: Ultralytics uses no workers
    # on a CPU, on purpose.
    wanted = {**m2.RECIPE, **schedule, "imgsz": arm.imgsz, "seed": args.seed,
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
        detector.pretrained_path(weights), data, arm.imgsz, args.seed, project,
        run_dir.name, keep="last",
        callbacks={"on_train_start": check_start, "on_train_end": record_end},
        fraction=args.fraction, device=args.device, **loader, **m2.RECIPE, **schedule)
    update_run(run_dir, status="finished", checkpoint=str(last),
               epochs_run=detector.epochs_run(last),
               train_minutes=round((time.time() - started) / 60, 1))
    print(f"finished: {last}  ->  {run_dir}", flush=True)


def train_step2(args: argparse.Namespace) -> None:
    """A step-2 arm (module docstring): its run's lists drawn and written,
    then one Ultralytics training per stage, each from the last one's
    weights, the first from the step-1 run's."""
    arm = m2b.STEP2[args.arm]
    weights, weights_sha256, start_name = m2b.start_weights(arm, args.seed)
    names = [name for lists in m2b.replay_lists(arm).values() for name in lists]
    if why := m2.missing({name: m2b.any_list_missing(name) for name in names}):
        raise SystemExit(f"can't train {args.arm} yet: {why}")
    folder = store_train_dir()
    if problems := store_train.check_export(folder):
        raise SystemExit(f"the store export in {folder} doesn't match its manifest "
                         f"({len(problems)} problems): " + "; ".join(problems[:5]))
    store_images, designs = m2b.store_pool(folder)
    pools = {"store": store_images, **m2b.replay_pools(arm)}
    epoch_photos = args.epoch_photos or m2b.EPOCH_PHOTOS
    lists = m2b.mix(pools, designs, args.seed, epoch_photos)
    n_listed = sum(map(len, lists.values()))
    schedules = [m2b.schedule(n_listed, stage, args.epochs) for stage in arm.stages]
    images = m2.expected_images(n_listed, args.fraction)
    loader = {"workers": args.workers, "cache": False}

    config = {"milestone": "m2b", "arm": args.arm, "size": arm.size, "recipe": arm.recipe,
              "weights": m2b.SIZES[arm.size], "imgsz": arm.imgsz, "seed": args.seed,
              "name": args.name,
              "start": {"run": start_name, "checkpoint": str(weights), "sha256": weights_sha256},
              "generated": arm.generated, "pools": {pool: len(p) for pool, p in pools.items()},
              "lists": {pool: m2b.list_record(paths) for pool, paths in lists.items()},
              "epoch_photos": epoch_photos, "n_listed": n_listed, "fraction": args.fraction,
              "train_images": images,
              "stages": [{"share": stage.share, "recipe": stage.recipe(), "schedule": schedule}
                         for stage, schedule in zip(arm.stages, schedules)],
              "loader": loader, "device": args.device, "manifests": m2b.manifest_hashes(names),
              "store_export": m2b.sha256(folder / "manifest.json")}
    run_dir = start_run(f"m2b-{args.name}", config, seed=args.seed)
    (run_dir / "lists").mkdir()
    for pool, paths in lists.items():
        (run_dir / "lists" / f"{pool}.txt").write_text(m2b.list_text(paths))
    data = detector.dataset_yaml(expdate_dir() / "yolo",
                                 [str(run_dir / "lists" / f"{pool}.txt") for pool in lists],
                                 run_dir / "data.yaml")
    project = checkpoint_dir() / "m2b"
    update_run(run_dir, status="running", device_name=device_name(args.device),
               checkpoint_dir=str(project / run_dir.name))
    print(f"run {run_dir.name} on {device_name(args.device)}: {args.arm} from {start_name}, "
          f"{n_listed} photos listed an epoch ({len(lists['store'])} store lines), "
          f"{len(arm.stages)} stage(s) of {', '.join(str(s['epochs']) for s in schedules)} epochs "
          f"at {arm.imgsz} px", flush=True)

    start, stages = weights, []
    for k, (stage, schedule) in enumerate(zip(arm.stages, schedules), start=1):
        name = run_dir.name if k == len(arm.stages) else f"{run_dir.name}-stage{k}"
        wanted = {**stage.recipe(), **schedule, "imgsz": arm.imgsz, "seed": args.seed,
                  "fraction": args.fraction}
        record = {"stage": k, "start": str(start)}

        def check_start(trainer, record=record, wanted=wanted, k=k) -> None:
            got = detector.applied(trainer)
            record["applied"] = got
            update_run(run_dir, applied=got, stage=k)     # training has started: the watchdog's progress
            problems = m2.recipe_problems(got, wanted, images)
            if problems:
                raise RuntimeError(f"Ultralytics isn't training stage {k} as asked: " + "; ".join(problems))
            print(f"stage {k} applied as asked: {got['optimizer']}, lr0 {got['args']['lr0']}, "
                  f"freeze {got['args']['freeze']}, {got['train_images']} training photos "
                  f"({got['distinct_images']} distinct), RAM cache {got['ram_cache']}", flush=True)

        def record_end(trainer, record=record) -> None:
            record["momentum_after_warmup"] = detector.momenta(trainer)

        started = time.time()
        last = detector.train(
            start, data, arm.imgsz, args.seed, project, name, keep="last",
            callbacks={"on_train_start": check_start, "on_train_end": record_end},
            fraction=args.fraction, device=args.device, **loader, **stage.recipe(), **schedule)
        record.update(checkpoint=str(last), epochs_run=detector.epochs_run(last),
                      train_minutes=round((time.time() - started) / 60, 1))
        problems = []
        if stage.freeze:
            record["frozen"] = m2b.frozen_drift(detector.weights_state(start),
                                                detector.weights_state(last), stage.freeze)
            problems = m2b.frozen_problems(record["frozen"])
        stages.append(record)
        update_run(run_dir, stages=stages)
        if problems:
            raise RuntimeError(f"stage {k} moved its frozen layers: " + "; ".join(problems))
        start = last

    update_run(run_dir, status="finished", checkpoint=str(last),
               epochs_run=sum(s["epochs_run"] or 0 for s in stages),
               train_minutes=round(sum(s["train_minutes"] for s in stages), 1))
    print(f"finished: {last}  ->  {run_dir}", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    {"m1": train_m1, "m2": train_m2, "m2b": train_m2b}[args.milestone](args)


if __name__ == "__main__":
    main()
