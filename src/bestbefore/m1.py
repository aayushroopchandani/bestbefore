"""M1's experiment grid and decision rules, fixed before any run
(docs/m1-date-detector.md, C1 and C2). scripts/m1_chain.py applies them,
and experiments/m1-*/summarise.py reads the results back."""
from __future__ import annotations

import json
from pathlib import Path

from . import runs

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


def load_run(name: str, runs_dir=None) -> tuple[dict, dict] | None:
    """(run.json, val.json) of the newest finished run with this name."""
    base = Path(runs_dir) if runs_dir else runs.RUNS_DIR
    for run in sorted(base.glob(f"*-m1-{name}"), reverse=True):
        if (run / "val.json").exists():
            return (json.loads((run / "run.json").read_text()),
                    json.loads((run / "val.json").read_text()))
    return None


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
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" if c in ("run", "model", "training photos") else "---:"
                            for c in cols) + "|"]
    lines += ["| " + " | ".join(r[c] for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def replace_results(readme: str, block: str) -> str:
    """Swap the text between the results markers of a README."""
    head, rest = readme.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n{block.strip()}\n{END}{tail}"
