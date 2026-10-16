"""Score trained M1 detectors on a labelled set, at their validation threshold.

    python scripts/eval_detector.py --set real-test --runs n640 n640-seed1 n640-seed2
    python scripts/eval_detector.py --set real-val --runs n640     # device check

Each run is scored with the confidence threshold it chose on the
validation photos (val.json), unchanged: nothing is tuned here. Writes
results/runs/<run>/<set>.json and prints one table, with mean and range
over the runs.

The ExpDate test split is meant to be scored once (docs/m1-date-detector.md,
decision 6). So a run already scored on it is refused unless --again is
given on purpose.

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

import numpy as np

from bestbefore import detector, m1
from bestbefore.config import REPO_ROOT, checkpoint_dir, expdate_dir
from bestbefore.sets import NAMES, load_set

SPLITS = {"real-test": "test", "real-val": "val"}   # sets with YOLO labels


def score_run(name: str, set_name: str, samples, device: str, again: bool) -> dict:
    run_dir = m1.finished_run(name)
    if run_dir is None:
        raise SystemExit(f"no finished run called {name!r} in results/runs")
    out = run_dir / f"{set_name}.json"
    if set_name == "real-test" and out.exists() and not again:
        raise SystemExit(f"{name} was already scored on the test split ({out}). "
                         "The test split is scored once; pass --again only on purpose.")
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
    scores = detector.score(best, samples, cfg["imgsz"], conf=val["threshold"],
                            data=data, split=SPLITS.get(set_name),
                            project=checkpoint_dir() / "m1" / run_dir.name / "eval",
                            device=device)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                            capture_output=True, text=True).stdout.strip()
    scores["scored"] = {"set": set_name, "device": device, "git_commit": commit,
                        "date": datetime.now().isoformat(timespec="seconds")}
    out.write_text(json.dumps(scores, indent=2))
    return scores


def table(results: dict[str, dict]) -> str:
    rows = []
    for name, s in results.items():
        a5, a3 = s["iou0.5"], s["iou0.3"]
        row = {"run": name, "threshold": f"{s['threshold']:.3f}",
               "mAP@0.5 date": f"{s['map50']['date']:.3f}" if "map50" in s else "-",
               "dates found": f"{a5['dates_found']}/{a5['dates']}",
               "precision": f"{a5['precision']:.3f}",
               "false boxes / photo": f"{a5['false_boxes_per_photo']:.2f}",
               "dates found, IoU 0.3": f"{a3['dates_found']}/{a3['dates']}"}
        if "expiry_dates" in a5:
            row["expiry found"] = f"{a5['expiry_found']}/{a5['expiry_dates']}"
            row["expiry found, IoU 0.3"] = f"{a3['expiry_found']}/{a3['expiry_dates']}"
        rows.append(row)
    text = m1.markdown_table(rows)
    if len(results) > 1:
        def spread(values):
            v = np.array(values, float)
            return f"{v.mean():.3f} (range {v.min():.3f} to {v.max():.3f})"
        lines = [f"- date recall: {spread([s['iou0.5']['date_recall'] for s in results.values()])}"]
        if all("expiry_recall" in s["iou0.5"] for s in results.values()):
            lines.append(f"- expiry recall: {spread([s['iou0.5']['expiry_recall'] for s in results.values()])}")
        if all("map50" in s for s in results.values()):
            lines.insert(0, f"- mAP@0.5 date: {spread([s['map50']['date'] for s in results.values()])}")
        text += f"\n\nMean over {len(results)} runs:\n" + "\n".join(lines)
    return text


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
    print(table(results))


if __name__ == "__main__":
    main()
