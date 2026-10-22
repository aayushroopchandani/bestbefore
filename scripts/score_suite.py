"""Score an M2 run on the validation suite (docs/m2-synthetic-generator.md, S3).

    python scripts/score_suite.py --run base-s0
    python scripts/score_suite.py --run smoke-laptop --device cpu

Scores the newest finished attempt at the run, results/runs/<id>-m2-<run>/,
with its final weights (last.pt), on the four suite sets. A set that isn't
made yet (synth-val, until S6) is marked missing, never skipped. Writes
suite.json there: the threshold that maximises F2 for dates on the three
real sets pooled (decision 11), and per set the dates found at IoU 0.5 and
0.3 at that threshold, boxed at any confidence, precision, false boxes per
photo and Ultralytics' mAP per class. Each set's dataset file is kept in
suite/<set>.yaml. scripts/m2_chain.py counts a run as scored once its
suite.json exists, so it is written whole or not at all.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from bestbefore import detector, m2, tables
from bestbefore.runs import git_state, write_json


def table(suite: dict) -> str:
    """One row per suite set, at the run's threshold (IoU 0.5)."""
    rows = []
    for name in m2.SUITE:
        s = suite[name]
        if "missing" in s:
            rows.append({"set": name, "photos": "-", "dates found": "missing",
                         "boxed at any confidence": "-", "precision": "-",
                         "false boxes / photo": "-", "mAP@0.5 date": "-"})
            continue
        at5, anyc = s["iou0.5"], s["any_conf"]
        rows.append({"set": name, "photos": str(at5["photos"]),
                     "dates found": f"{at5['dates_found']}/{at5['dates']}",
                     "boxed at any confidence": f"{anyc['dates_found']}/{anyc['dates']}",
                     "precision": f"{at5['precision']:.3f}" if at5["precision"] is not None else "-",
                     "false boxes / photo": f"{at5['false_boxes_per_photo']:.2f}",
                     "mAP@0.5 date": f"{s['map50'].get('date', 0):.3f}"})
    return tables.markdown_table(rows, left=("set",))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True, help="run name, e.g. base-s0 or smoke-laptop")
    parser.add_argument("--device", default=None, help="e.g. 0 or cpu; default: auto")
    args = parser.parse_args()

    run_dir = m2.finished_run(args.run)
    if run_dir is None:
        raise SystemExit(f"no finished run m2-{args.run} in results/runs")
    record = json.loads((run_dir / "run.json").read_text())
    started = time.time()
    suite = detector.score_suite(Path(record["checkpoint"]), record["config"]["imgsz"], run_dir,
                                 args.device)
    suite["scored"] = {"device": args.device, **git_state(),
                       "date": datetime.now().isoformat(timespec="seconds"),
                       "minutes": round((time.time() - started) / 60, 1)}
    write_json(run_dir / "suite.json", suite)

    print(f"{run_dir.name}: threshold {suite['threshold']:.4f} ({suite['threshold_rule']}, "
          f"F2 {suite['f2_at_threshold']})\n")
    print(table(suite))
    for name, why in suite["missing"].items():
        print(f"\nmissing: {name}: {why}")
    print(f"\n-> {run_dir / 'suite.json'}")


if __name__ == "__main__":
    main()
