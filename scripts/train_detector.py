"""Train one date detector and score it on the validation photos.

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
"""
from __future__ import annotations

import argparse
import json
import time

from bestbefore import detector
from bestbefore.config import checkpoint_dir, expdate_dir
from bestbefore.runs import start_run, update_run
from bestbefore.sets import load_set


def device_name(device) -> str:
    import torch
    if torch.cuda.is_available() and device not in ("cpu", "mps"):
        return torch.cuda.get_device_name(0)
    return str(device or "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--weights", required=True, help="yolo11n or yolo11s")
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--train", required=True,
                        help="comma-separated image lists, e.g. train_synth,train_real")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", required=True, help="short run name, e.g. s1280")
    parser.add_argument("--epochs", type=int, default=detector.SETTINGS["epochs"])
    parser.add_argument("--patience", type=int, default=detector.SETTINGS["patience"])
    parser.add_argument("--batch", type=int, default=detector.SETTINGS["batch"])
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="share of the training photos to use (smoke tests only)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None, help="e.g. 0, cpu or mps; default: auto")
    args = parser.parse_args()

    yolo_dir = expdate_dir() / "yolo"
    run_dir = start_run(f"m1-{args.name}", vars(args), seed=args.seed)
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
    update_run(run_dir, checkpoint=str(best),
               train_minutes=round((time.time() - started) / 60, 1))

    scores = detector.score(best, load_set("real-val"), args.imgsz, data=data,
                            split="val", project=project / run_dir.name,
                            device=args.device)
    (run_dir / "val.json").write_text(json.dumps(scores, indent=2))
    s = scores["iou0.5"]
    print(f"validation: mAP@0.5 date {scores['map50'].get('date')}, "
          f"threshold {scores['threshold']}, date recall {s['date_recall']}, "
          f"precision {s['precision']}  ->  {run_dir}")


if __name__ == "__main__":
    main()
