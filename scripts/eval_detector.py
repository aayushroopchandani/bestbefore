"""Score trained M1 and M2 detectors on a labelled set, at their validation threshold.

    python scripts/eval_detector.py --set real-test --runs n640 n640-seed1 n640-seed2
    python scripts/eval_detector.py --set store --runs n640 n640-seed1 n640-seed2
    python scripts/eval_detector.py --set real-val --runs n640     # device check
    python scripts/eval_detector.py --set store --runs m2-base-s0 m2-rot-s0

A name starting with `m2-` is an M2 run (results/runs/<id>-m2-<run>/); any
other is M1's (`m1-n640` and `n640` are the same run). Each run is scored
with the confidence threshold it chose on the validation photos,
unchanged: nothing is tuned here. M1's is in its val.json; M2's is the
pooled F2 threshold in its suite.json, used exactly as stored, never
rounded (docs/m2-synthetic-generator.md, S7). Writes
results/runs/<run>/<set>.json (counts only) and prints one table, with
mean and range over the runs; on the store photos, a second table by
failure tag. The store photos' per-photo predictions, which name photos,
go to data/store/runs/ instead, for scripts/detector_misses.py and M2's
store-by-cause tables.

The ExpDate test split and the store photos are each meant to be scored
once (docs/m1-date-detector.md, decision 6; M2 decision 17). So a run
already scored on either is refused unless --again is given on purpose.

Weights are read from $CHECKPOINT_DIR/m1/<run>/weights/best.pt (M1) or
$CHECKPOINT_DIR/m2/<run>/weights/last.pt (M2, decision 10). Runs trained
on the cluster need them copied first, e.g.
    rsync -a bbhpc:bestbefore-data/checkpoints/m1/<run>/weights/best.pt \\
        data/checkpoints/m1/<run>/weights/
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from bestbefore import detector, m1, m2
from bestbefore.config import REPO_ROOT, checkpoint_dir, expdate_dir
from bestbefore.runs import start_run
from bestbefore.sets import NAMES, load_set, runs_root

SPLITS = {"real-test": "test", "real-val": "val"}   # sets with YOLO labels
SCORED_ONCE = ("real-test", "store")


@dataclass(frozen=True)
class Detector:
    """A trained run, as this script scores it."""
    name: str               # n640 (M1) or m2-base-s0 (M2)
    milestone: str          # m1 or m2
    run_dir: Path           # its record, results/runs/<id>-m1-n640/
    threshold: float        # chosen on validation, used unchanged
    imgsz: int
    train: list[str]        # the dataset file's train lists (needed for mAP, never read)

    @property
    def checkpoints(self) -> Path:
        return checkpoint_dir() / self.milestone / self.run_dir.name

    @property
    def weights(self) -> Path:
        """M1 kept the best epoch on validation; M2 keeps the final weights."""
        return self.checkpoints / "weights" / ("best.pt" if self.milestone == "m1" else "last.pt")


def find(name: str) -> Detector:
    """The newest finished run with this name, and how to score it."""
    if name.startswith("m2-"):
        return _m2_run(name)
    name = name.removeprefix("m1-")
    run_dir = m1.finished_run(name)
    if run_dir is None:
        raise SystemExit(f"no finished run called {name!r} in results/runs")
    cfg = json.loads((run_dir / "run.json").read_text())["config"]
    val = json.loads((run_dir / "val.json").read_text())
    return Detector(name, "m1", run_dir, val["threshold"], cfg["imgsz"], cfg["train"].split(","))


def _m2_run(name: str) -> Detector:
    """An M2 run: its final weights and the threshold in its suite.json."""
    run = name.removeprefix("m2-")
    run_dir = m2.finished_run(run)
    if run_dir is None:
        raise SystemExit(f"no finished run called {name!r} in results/runs")
    suite = m2.read_json(run_dir / "suite.json")
    if suite is None:
        raise SystemExit(f"{name} has no suite.json yet, so no threshold: "
                         f"python scripts/score_suite.py --run {run}")
    cfg = m2.read_json(run_dir / "run.json")["config"]
    # Validating never reads the train entry, and an arm's own lists
    # (turned copies, generated photos) aren't on the laptop.
    return Detector(name, "m2", run_dir, suite["threshold"], cfg["imgsz"], ["train_real"])


def output(run: Detector, set_name: str, again: bool) -> Path:
    """Where the run's scores on the set go; a test set already scored is
    refused unless `again`."""
    out = run.run_dir / f"{set_name}.json"
    if set_name in SCORED_ONCE and out.exists() and not again:
        raise SystemExit(f"{run.name} was already scored on {set_name} ({out}). "
                         "Test sets are scored once; pass --again only on purpose.")
    if not run.weights.exists():
        raise SystemExit(f"{run.weights} is missing: copy it from the cluster (see --help)")
    return out


def score_run(name: str, set_name: str, samples, device: str, again: bool) -> dict:
    run = find(name)
    out = output(run, set_name, again)

    data = None
    if set_name in SPLITS:
        data = detector.dataset_yaml(expdate_dir() / "yolo", run.train, run.run_dir / "data-local.yaml")
    preds = detector.predict_checkpoint(run.weights, samples, run.imgsz, device)
    if set_name == "store":
        keep_private(run, preds)
    scores = detector.score(run.weights, samples, run.imgsz, conf=run.threshold,
                            data=data, split=SPLITS.get(set_name),
                            project=run.checkpoints / "eval",
                            device=device, predictions=preds)
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT,
                              capture_output=True, text=True).stdout.strip()

    scores["scored"] = {"set": set_name, "device": device, "git_commit": git("rev-parse", "HEAD"),
                        "git_dirty": bool(git("status", "--porcelain")),
                        "date": datetime.now().isoformat(timespec="seconds")}
    out.write_text(json.dumps(scores, indent=2))
    return scores


def keep_private(run: Detector, preds: dict) -> None:
    """Per-photo date boxes on the store photos, in data/store/runs/ only.
    M1 rounded them; M2 keeps them exact, as its threshold is exact: a
    confidence rounded to 4 decimals could cross it."""
    private = start_run(f"detector-store-{run.name}",
                        {"run": run.run_dir.name, "threshold": run.threshold},
                        root=runs_root("store"))

    def row(p) -> list[float]:
        if run.milestone == "m2":
            return [*p.xyxy, p.conf]
        return [*(round(v, 5) for v in p.xyxy), round(p.conf, 4)]

    boxes = {pid: [row(p) for p in ps if p.label == "date"] for pid, ps in preds.items()}
    (private / "predictions.json").write_text(json.dumps(boxes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", required=True, choices=NAMES)
    parser.add_argument("--runs", nargs="+", required=True, help="run names, e.g. n640 or m2-base-s0")
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
