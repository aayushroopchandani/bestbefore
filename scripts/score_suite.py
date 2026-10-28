"""Score an M2 run on the validation suite (docs/m2-synthetic-generator.md, S3).

    python scripts/score_suite.py --run base-s0
    python scripts/score_suite.py --run smoke-laptop --device cpu
    python scripts/score_suite.py --run gen-rot-1280-s0 --device cpu \\
        --out results/runs/<id>-m2-gen-rot-1280-s0/device-check-cpu     # S8's device check

Scores the newest finished attempt at the run, results/runs/<id>-m2-<run>/,
with its final weights (last.pt), on the four suite sets. A set that isn't
made yet (synth-val, until S6) is marked missing, never skipped. Writes
suite.json there: the threshold that maximises F2 for dates on the three
real sets pooled (decision 11), and per set the dates found at IoU 0.5 and
0.3 at that threshold, boxed at any confidence, precision, false boxes per
photo and Ultralytics' mAP per class. Each set's dataset file is kept in
suite/<set>.yaml. scripts/m2_chain.py counts a run as scored once its
suite.json exists, so it is written whole or not at all.

--out writes suite.json and suite/ into another folder and leaves the
run's own suite.json, its record, as it is: that one's threshold is what
the store photos and ExpDate test are cut at (S8). It then prints the two
side by side. The weights are the ones run.json names where that path
exists (the cluster); otherwise the copy scripts/eval_detector.py reads,
$CHECKPOINT_DIR/m2/<run>/weights/last.pt (the laptop).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from bestbefore import detector, m2, tables
from bestbefore.config import checkpoint_dir
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


def weights(run_dir: Path, record: dict) -> Path:
    """The run's last.pt: where run.json says, if it is there (the cluster),
    else the copy eval_detector.py reads (the laptop)."""
    recorded = Path(record["checkpoint"])
    return recorded if recorded.exists() else checkpoint_dir() / "m2" / run_dir.name / "weights" / "last.pt"


# What the comparison shows per set: (label, level, count), at each suite's own threshold.
COMPARED = (("dates found", "iou0.5", "dates"), ("found at IoU 0.3", "iou0.3", "dates"),
            ("boxed at any confidence", "any_conf", "dates"),
            ("targets found", "iou0.5", "expiry"), ("targets boxed at any confidence", "any_conf", "expiry"))


def compare(record: dict, here: dict) -> str:
    """The run's own suite.json beside one scored again (the device
    check): the threshold, then per set and per synth-val stratum the
    counts at each one's own threshold and mAP@0.5 for dates."""
    rows = [{"set": "all", "measure": "threshold", "record": f"{record['threshold']:.6f}",
             "here": f"{here['threshold']:.6f}",
             "difference": f"{here['threshold'] - record['threshold']:+.6f}"}]

    def add(name: str, a: dict, b: dict, with_map: bool) -> None:
        for label, level, what in COMPARED:
            key, of = ("expiry_found", "expiry_dates") if what == "expiry" else ("dates_found", "dates")
            if of not in a[level] or not a[level][of]:
                continue
            ka, kb = a[level][key], b[level][key]
            rows.append({"set": name, "measure": label, "record": f"{ka}/{a[level][of]}",
                         "here": f"{kb}/{b[level][of]}", "difference": f"{kb - ka:+d}"})
        if with_map:
            ma, mb = a["map50"].get("date", 0.0), b["map50"].get("date", 0.0)
            rows.append({"set": name, "measure": "mAP@0.5 date", "record": f"{ma:.4f}",
                         "here": f"{mb:.4f}", "difference": f"{mb - ma:+.4f}"})

    for name in m2.SUITE:
        a, b = record.get(name, {}), here.get(name, {})
        if "missing" in a or "missing" in b:
            rows.append({"set": name, "measure": "missing", "record": a.get("missing", ""),
                         "here": b.get("missing", ""), "difference": ""})
            continue
        add(name, a, b, with_map=True)
        for tag in a.get("by_tag", {}):
            add(f"{name}: {tag}", a["by_tag"][tag], b["by_tag"][tag], with_map=False)
    return tables.markdown_table(rows, left=("set", "measure"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True, help="run name, e.g. base-s0 or smoke-laptop")
    parser.add_argument("--device", default=None, help="e.g. 0 or cpu; default: auto")
    parser.add_argument("--out", type=Path, default=None,
                        help="write suite.json here, not in the run's folder (S8's device check)")
    args = parser.parse_args()

    run_dir = m2.finished_run(args.run)
    if run_dir is None:
        raise SystemExit(f"no finished run m2-{args.run} in results/runs")
    out_dir = args.out or run_dir
    if args.out is not None and out_dir.resolve() == run_dir.resolve():
        raise SystemExit(f"--out {out_dir} is the run's own folder, whose suite.json is its record")
    out_dir.mkdir(parents=True, exist_ok=True)
    record = json.loads((run_dir / "run.json").read_text())
    checkpoint = weights(run_dir, record)
    started = time.time()
    suite = detector.score_suite(checkpoint, record["config"]["imgsz"], out_dir, args.device)
    suite["scored"] = {"device": args.device, **git_state(),
                       "date": datetime.now().isoformat(timespec="seconds"),
                       "minutes": round((time.time() - started) / 60, 1)}
    write_json(out_dir / "suite.json", suite)

    print(f"{run_dir.name}: threshold {suite['threshold']:.4f} ({suite['threshold_rule']}, "
          f"F2 {suite['f2_at_threshold']})\n")
    print(table(suite))
    for name, why in suite["missing"].items():
        print(f"\nmissing: {name}: {why}")
    print(f"\n-> {out_dir / 'suite.json'}")
    own = m2.read_json(run_dir / "suite.json")
    if args.out is not None and own is not None:
        print(f"\nThe run's own suite.json ({own['scored'].get('device') or 'auto'} device, "
              f"{own['scored']['date']}) beside this one, each at its own threshold:\n")
        print(compare(own, suite))


if __name__ == "__main__":
    main()
