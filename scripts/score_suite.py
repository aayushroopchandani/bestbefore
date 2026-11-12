"""Score an M2 run on the validation suite (docs/m2-synthetic-generator.md, S3).

    python scripts/score_suite.py --run base-s0
    python scripts/score_suite.py --run smoke-laptop --device cpu
    python scripts/score_suite.py --milestone m2b --run s1280-s0     # M2b's step 1, on M2's suite
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

--milestone m2b scores an M2b run, results/runs/<id>-m2b-<run>/, on the
same suite (docs/m2b-store-and-size.md, S2): its step-1 runs are scored on
M2's suite as soon as they finish, by scripts/m2b_chain.py.

--suite m2b scores an M2b run on the M2b suite instead (S10): M2's four
sets, real-val-rot180, synth-val-v2 and v1-val, with two thresholds, the
best F2 on v1-val (threshold_store, the run's threshold from M2b on) and
M2's, both recorded (detector.score_suite_m2b). It writes suite-m2b.json,
and leaves suite.json, M2's suite, as it is. Counts only: no box of a
store photo is kept. The control's weights are M2's last.pt, found on
either machine.

    python scripts/score_suite.py --milestone m2b --suite m2b --run n1280-ft-s0

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

from bestbefore import detector, m2, m2b, tables
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


def table_m2b(suite: dict) -> str:
    """One row per M2b suite set: found at its threshold (threshold_store)
    and at M2's (IoU 0.5), and the rest at its threshold."""
    rows = []
    for name in m2b.SUITE:
        s = suite[name]
        at5, at_m2, anyc = s["iou0.5"], s["at_threshold_m2"]["iou0.5"], s["any_conf"]
        rows.append({"set": name, "photos": str(at5["photos"]),
                     "dates found": f"{at5['dates_found']}/{at5['dates']}",
                     "at M2's threshold": f"{at_m2['dates_found']}/{at_m2['dates']}",
                     "boxed at any confidence": f"{anyc['dates_found']}/{anyc['dates']}",
                     "precision": f"{at5['precision']:.3f}" if at5["precision"] is not None else "-",
                     "false boxes / photo": f"{at5['false_boxes_per_photo']:.2f}",
                     "mAP@0.5 date": f"{s['map50'].get('date', 0):.3f}"})
    return tables.markdown_table(rows, left=("set",))


def weights(run_dir: Path, record: dict, milestone: str = "m2") -> Path:
    """The run's last.pt: where run.json says, if it is there (the cluster),
    else the copy eval_detector.py reads (the laptop), in the milestone's
    folder of $CHECKPOINT_DIR. An M2b control's are M2's run's, which its
    record names (weights_from)."""
    recorded = Path(record["checkpoint"])
    if recorded.exists():
        return recorded
    if "weights_from" in record:
        return checkpoint_dir() / "m2" / record["weights_from"] / "weights" / "last.pt"
    return checkpoint_dir() / milestone / run_dir.name / "weights" / "last.pt"


# What the comparison shows per set: (label, level, count), at each suite's own threshold.
COMPARED = (("dates found", "iou0.5", "dates"), ("found at IoU 0.3", "iou0.3", "dates"),
            ("boxed at any confidence", "any_conf", "dates"),
            ("targets found", "iou0.5", "expiry"), ("targets boxed at any confidence", "any_conf", "expiry"))


def compare(record: dict, here: dict, names=tuple(m2.SUITE)) -> str:
    """The run's own suite.json (or suite-m2b.json) beside one scored again
    (the device check): the threshold, then per set and per stratum or tag
    the counts at each one's own threshold and mAP@0.5 for dates."""
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

    for name in names:
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
    parser.add_argument("--milestone", choices=("m2", "m2b"), default="m2",
                        help="whose run: M2's (the default) or M2b's")
    parser.add_argument("--suite", choices=("m2", "m2b"), default="m2",
                        help="M2's suite (the default, suite.json) or the M2b suite (suite-m2b.json)")
    parser.add_argument("--device", default=None, help="e.g. 0 or cpu; default: auto")
    parser.add_argument("--out", type=Path, default=None,
                        help="write suite.json here, not in the run's folder (S8's device check)")
    args = parser.parse_args()
    if args.suite == "m2b" and args.milestone != "m2b":
        parser.error("--suite m2b scores M2b runs: add --milestone m2b")

    file = m2b.SUITE_FILE if args.suite == "m2b" else "suite.json"
    run_dir = (m2b if args.milestone == "m2b" else m2).finished_run(args.run)
    if run_dir is None:
        raise SystemExit(f"no finished run {args.milestone}-{args.run} in results/runs")
    out_dir = args.out or run_dir
    if args.out is not None and out_dir.resolve() == run_dir.resolve():
        raise SystemExit(f"--out {out_dir} is the run's own folder, whose {file} is its record")
    out_dir.mkdir(parents=True, exist_ok=True)
    record = json.loads((run_dir / "run.json").read_text())
    checkpoint = weights(run_dir, record, args.milestone)
    started = time.time()
    score = detector.score_suite_m2b if args.suite == "m2b" else detector.score_suite
    suite = score(checkpoint, record["config"]["imgsz"], out_dir, args.device)
    suite["scored"] = {"device": args.device, **git_state(),
                       "date": datetime.now().isoformat(timespec="seconds"),
                       "minutes": round((time.time() - started) / 60, 1)}
    write_json(out_dir / file, suite)

    print(f"{run_dir.name}: threshold {suite['threshold']:.4f} ({suite['threshold_rule']}, "
          f"F2 {suite['f2_at_threshold']})")
    if args.suite == "m2b":
        print(f"M2's threshold {suite['threshold_m2']:.4f} ({suite['threshold_m2_rule']}, "
              f"F2 {suite['f2_at_threshold_m2']})")
    print()
    print(table_m2b(suite) if args.suite == "m2b" else table(suite))
    for name, why in suite["missing"].items():
        print(f"\nmissing: {name}: {why}")
    print(f"\n-> {out_dir / file}")
    own = m2.read_json(run_dir / file)
    if args.out is not None and own is not None:
        print(f"\nThe run's own {file} ({own['scored'].get('device') or 'auto'} device, "
              f"{own['scored']['date']}) beside this one, each at its own threshold:\n")
        print(compare(own, suite, m2b.SUITE if args.suite == "m2b" else tuple(m2.SUITE)))


if __name__ == "__main__":
    main()
