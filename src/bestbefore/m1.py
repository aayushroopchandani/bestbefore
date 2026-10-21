"""M1's experiment grid and decision rules, fixed before any run
(docs/m1-date-detector.md, C1 and C2). scripts/m1_chain.py applies them,
and experiments/m1-*/summarise.py reads the results back."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import runs, store, tables

# C1: run name -> (weights, input size). Listed cheapest first, which is
# the order the tie-break uses: smaller model first, then smaller input.
C1 = {
    "n640": ("yolo11n", 640),
    "n1280": ("yolo11n", 1280),
    "s640": ("yolo11s", 640),
    "s1280": ("yolo11s", 1280),
}
BOTH = "train_synth,train_real"
REAL = "train_real"
NOISE = 0.01   # 1 point of mAP@0.5: smaller differences are treated as noise


def map50_date(scores: dict) -> float:
    return scores["map50"]["date"]


def decide_c1(scores: dict[str, dict]) -> str:
    """Highest validation mAP@0.5 for dates; within NOISE of the best, the
    cheapest run (the first in C1's order)."""
    best = max(map50_date(s) for s in scores.values())
    return next(n for n in C1 if n in scores and map50_date(scores[n]) >= best - NOISE)


def keep_synthetic(with_synth: dict, real_only: dict) -> bool:
    """Keep synthetic data unless real-only is better by more than NOISE."""
    return not map50_date(real_only) > map50_date(with_synth) + NOISE


# --- Reading results back (experiments/m1-*/summarise.py) -----------------

START, END = "<!-- results:start -->", "<!-- results:end -->"


def finished_run(name: str, runs_dir=None) -> Path | None:
    """The newest run directory with this name that has a val.json."""
    base = Path(runs_dir) if runs_dir else runs.RUNS_DIR
    for run in sorted(base.glob(f"*-m1-{name}"), reverse=True):
        if (run / "val.json").exists():
            return run
    return None


def load_run(name: str, runs_dir=None) -> tuple[dict, dict] | None:
    """(run.json, val.json) of the newest finished run with this name."""
    run = finished_run(name, runs_dir)
    if run is None:
        return None
    return (json.loads((run / "run.json").read_text()),
            json.loads((run / "val.json").read_text()))


def row(name: str, record: dict, scores: dict) -> dict[str, str]:
    """One results-table row for a finished run."""
    cfg, at5, at3 = record["config"], scores["iou0.5"], scores["iou0.3"]
    ours, theirs = at5["date_recall"], scores.get("ultralytics_date_recall_at_threshold")
    check = "ok" if ours is not None and theirs is not None and abs(ours - theirs) <= 0.02 else "CHECK"
    return {
        "run": name,
        "model": cfg["weights"],
        "input": str(cfg["imgsz"]),
        "training photos": "synthetic + real" if "train_synth" in cfg["train"] else "real only",
        "mAP@0.5 date": f"{map50_date(scores):.3f}",
        "mAP@0.5:0.95 date": f"{scores['map50_95']['date']:.3f}",
        "threshold": f"{scores['threshold']:.3f}",
        "date recall": f"{at5['dates_found']}/{at5['dates']}",
        "precision": f"{at5['precision']:.3f}",
        "false boxes / photo": f"{at5['false_boxes_per_photo']:.2f}",
        "recall at IoU 0.3": f"{at3['dates_found']}/{at3['dates']}",
        "epochs": str(record.get("epochs_run", "?")),
        "minutes": str(record.get("train_minutes", "?")),
        "recall cross-check": check,
    }


def markdown_table(rows: list[dict[str, str]]) -> str:
    return tables.markdown_table(rows, left=("run", "model", "training photos"))


def replace_results(readme: str, block: str) -> str:
    """Swap the text between the results markers of a README."""
    head, rest = readme.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n{block.strip()}\n{END}{tail}"


# --- Scores on a test set (scripts/eval_detector.py, scripts/m1_report.py) ---

def spread(values: list[float]) -> str:
    """'0.267 (range 0.222 to 0.309)': the mean and range over runs."""
    v = np.array(values, float)
    return f"{v.mean():.3f} (range {v.min():.3f} to {v.max():.3f})"


def tag_table(results: dict[str, dict]) -> str:
    """Store photos by failure tag: expiry found at IoU 0.5 per run, then
    means over the runs. The last row is every photo, other-face included."""
    first = next(iter(results.values()))
    groups = [(t, [r["by_tag"].get(t) for r in results.values()]) for t in store.TAGS]
    groups.append((f"**all {first['iou0.5']['photos']} photos**", list(results.values())))
    rows = []
    for name, per_run in groups:
        if per_run[0] is None:          # no photo has this tag (curved)
            continue
        at5 = [r["iou0.5"] for r in per_run]
        has_expiry = "expiry_dates" in at5[0]

        def mean(key: str, field: str = "expiry_recall") -> str:
            return spread([r[key][field] for r in per_run])

        row = {"tag": name, "photos": str(at5[0]["photos"]),
               "expiry dates": str(at5[0].get("expiry_dates", 0))}
        for run, s in zip(results, at5):
            row[run] = f"{s['expiry_found']}/{s['expiry_dates']}" if has_expiry else "-"
        row["expiry recall, mean"] = mean("iou0.5") if has_expiry else "-"
        row["at IoU 0.3"] = mean("iou0.3") if has_expiry else "-"
        row["boxed at any confidence"] = mean("any_conf") if has_expiry else "-"
        row["all dates found"] = mean("iou0.5", "date_recall") if at5[0]["dates"] else "-"
        row["false boxes / photo"] = spread([s["false_boxes_per_photo"] for s in at5])
        rows.append(row)
    return tables.markdown_table(rows, left=("tag",))


def scores_table(results: dict[str, dict]) -> str:
    """One row per run scored on a set, then means over the runs."""
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
    text = markdown_table(rows)
    if len(results) > 1:
        lines = [f"- date recall: {spread([s['iou0.5']['date_recall'] for s in results.values()])}"]
        if all("expiry_recall" in s["iou0.5"] for s in results.values()):
            lines.append(f"- expiry recall: {spread([s['iou0.5']['expiry_recall'] for s in results.values()])}")
        if all("map50" in s for s in results.values()):
            lines.insert(0, f"- mAP@0.5 date: {spread([s['map50']['date'] for s in results.values()])}")
        text += f"\n\nMean over {len(results)} runs:\n" + "\n".join(lines)
    return text
