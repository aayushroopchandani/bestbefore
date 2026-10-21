"""Score trained M1 detectors on a labelled set, at their validation threshold.

    python scripts/eval_detector.py --set real-test --runs n640 n640-seed1 n640-seed2
    python scripts/eval_detector.py --set store --runs n640 n640-seed1 n640-seed2
    python scripts/eval_detector.py --set real-val --runs n640     # device check

Each run is scored with the confidence threshold it chose on the
validation photos (val.json), unchanged: nothing is tuned here. Writes
results/runs/<run>/<set>.json (counts only) and prints one table, with
mean and range over the runs; on the store photos, a second table by
failure tag. The store photos' per-photo predictions, which name photos,
go to data/store/runs/ instead, for scripts/detector_misses.py.

The ExpDate test split and the store photos are each meant to be scored
once (docs/m1-date-detector.md, decision 6). So a run already scored on
either is refused unless --again is given on purpose.

Weights are read from $CHECKPOINT_DIR/m1/<run>/weights/best.pt. Runs
trained on the cluster need them copied first, e.g.
    rsync -a bbhpc:bestbefore-data/checkpoints/m1/<run>/weights/best.pt \\
        data/checkpoints/m1/<run>/weights/
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime

from bestbefore import detector, m1
from bestbefore.config import REPO_ROOT, checkpoint_dir, expdate_dir
from bestbefore.runs import start_run
from bestbefore.sets import NAMES, load_set, runs_root

SPLITS = {"real-test": "test", "real-val": "val"}   # sets with YOLO labels
SCORED_ONCE = ("real-test", "store")


def score_run(name: str, set_name: str, samples, device: str, again: bool) -> dict:
    run_dir = m1.finished_run(name)
    if run_dir is None:
        raise SystemExit(f"no finished run called {name!r} in results/runs")
    out = run_dir / f"{set_name}.json"
    if set_name in SCORED_ONCE and out.exists() and not again:
        raise SystemExit(f"{name} was already scored on {set_name} ({out}). "
                         "Test sets are scored once; pass --again only on purpose.")
    best = checkpoint_dir() / "m1" / run_dir.name / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"{best} is missing: copy it from the cluster (see --help)")

    record = json.loads((run_dir / "run.json").read_text())
    val = json.loads((run_dir / "val.json").read_text())
    cfg = record["config"]

    data = None
    if set_name in SPLITS:
        data = detector.dataset_yaml(expdate_dir() / "yolo", cfg["train"].split(","),
                                     run_dir / "data-local.yaml")
    preds = detector.predict_checkpoint(best, samples, cfg["imgsz"], device)
    if set_name == "store":
        keep_private(name, run_dir, val["threshold"], preds)
    scores = detector.score(best, samples, cfg["imgsz"], conf=val["threshold"],
                            data=data, split=SPLITS.get(set_name),
                            project=checkpoint_dir() / "m1" / run_dir.name / "eval",
                            device=device, predictions=preds)
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT,
                              capture_output=True, text=True).stdout.strip()

    scores["scored"] = {"set": set_name, "device": device, "git_commit": git("rev-parse", "HEAD"),
                        "git_dirty": bool(git("status", "--porcelain")),
                        "date": datetime.now().isoformat(timespec="seconds")}
    out.write_text(json.dumps(scores, indent=2))
    return scores


def keep_private(name: str, run_dir, threshold: float, preds: dict) -> None:
    """Per-photo date boxes on the store photos, in data/store/runs/ only."""
    private = start_run(f"detector-store-{name}", {"run": run_dir.name, "threshold": threshold},
                        root=runs_root("store"))
    boxes = {pid: [[*(round(v, 5) for v in p.xyxy), round(p.conf, 4)]
                   for p in ps if p.label == "date"] for pid, ps in preds.items()}
    (private / "predictions.json").write_text(json.dumps(boxes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", required=True, choices=NAMES)
    parser.add_argument("--runs", nargs="+", required=True, help="run names, e.g. n640")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--again", action="store_true",
                        help="re-score a run already scored on the test split")
    args = parser.parse_args()

    samples = load_set(args.set)
    results = {name: score_run(name, args.set, samples, args.device, args.again)
               for name in args.runs}
    print(m1.scores_table(results))
    if args.set == "store":
        print("\nBy failure tag (expiry found at IoU 0.5; means over the runs):\n")
        print(m1.tag_table(results))


if __name__ == "__main__":
    main()
