"""Write results/m1_detector.md: M1's results, rebuilt from the run records
(docs/m1-date-detector.md, D4).

    python scripts/m1_report.py

Reads only what the runs saved: each run's run.json and val.json
(C1, C2, the threshold) and the real-test.json and store.json that
scripts/eval_detector.py wrote (D2, D3). Nothing is re-scored, so the
report can't drift from the runs. Counts only: no store photo is named.
"""
from __future__ import annotations

import json

from bestbefore import m1, store
from bestbefore.config import REPO_ROOT
from bestbefore.tables import markdown_table

OUT = REPO_ROOT / "results" / "m1_detector.md"


def scores(name: str, set_name: str) -> dict:
    run = m1.finished_run(name)
    path = run / f"{set_name}.json" if run else None
    if path is None or not path.exists():
        raise SystemExit(f"{name} isn't scored on {set_name}: "
                         f"python scripts/eval_detector.py --set {set_name} --runs {name}")
    return json.loads(path.read_text())


def validation_tables() -> tuple[str, str, str]:
    """C1 and C2 tables, and the name of the chosen run."""
    c1 = {n: m1.load_run(n) for n in m1.C1}
    winner = m1.decide_c1({n: val for n, (_, val) in c1.items()})
    real_only = m1.load_run(f"{winner}-real")
    c1_table = markdown_table([m1.row(n, *c1[n]) for n in m1.C1],
                              left=("run", "model", "training photos"))
    c2_table = markdown_table([m1.row(winner, *c1[winner]), m1.row(f"{winner}-real", *real_only)],
                              left=("run", "model", "training photos"))
    keep = m1.keep_synthetic(c1[winner][1], real_only[1])
    c2_table += f"\n\nDecision: {'keep' if keep else 'drop'} the synthetic photos."
    return c1_table, c2_table, winner


def headline(test: dict, store_scores: dict, chosen: str) -> str:
    def cell(results: dict, found: str, total: str, recall: str) -> tuple[str, str]:
        s = results[chosen]["iou0.5"]
        mean = m1.spread([r["iou0.5"][recall] for r in results.values()])
        return f"{s[found]}/{s[total]}", mean

    rows = []
    for label, keys in (("expiry dates found", ("expiry_found", "expiry_dates", "expiry_recall")),
                        ("all dates found", ("dates_found", "dates", "date_recall"))):
        (t, tm), (s, sm) = cell(test, *keys), cell(store_scores, *keys)
        rows.append({"": f"{label}, `{chosen}`", "ExpDate test": t, "store photos": s})
        rows.append({"": f"{label}, mean of {len(test)} seeds", "ExpDate test": tm, "store photos": sm})
    rows.append({"": "precision, `" + chosen + "`",
                 "ExpDate test": f"{test[chosen]['iou0.5']['precision']:.3f}",
                 "store photos": f"{store_scores[chosen]['iou0.5']['precision']:.3f}"})
    rows.append({"": "mAP@0.5 for dates, mean of 3 seeds",
                 "ExpDate test": m1.spread([r["map50"]["date"] for r in test.values()]),
                 "store photos": "not computed (no YOLO labels for the store set)"})
    return markdown_table(rows, left=("",))


def cross_check(test: dict) -> str:
    gaps = [abs(r["iou0.5"]["date_recall"] - r["ultralytics_date_recall_at_threshold"])
            for r in test.values()]
    return (f"Our date recall and Ultralytics' own, at the same threshold, differ by at "
            f"most {max(gaps):.3f} over the {len(test)} runs.")


def runs_list(results: dict[str, dict[str, dict]]) -> str:
    lines = []
    for set_name, per_run in results.items():
        for name, s in per_run.items():
            info = s["scored"]
            dirty = " + uncommitted changes" if info.get("git_dirty") else ""
            lines.append(f"- `{name}` on {set_name}: run `{m1.finished_run(name).name}`, "
                         f"scored {info['date'][:10]} on the {info['device']} at "
                         f"`{info['git_commit'][:7]}`{dirty}, threshold {s['threshold']:.4f}.")
    return "\n".join(lines)


def main() -> None:
    c1_table, c2_table, chosen = validation_tables()
    seeds = (chosen, f"{chosen}-seed1", f"{chosen}-seed2")
    test = {n: scores(n, "real-test") for n in seeds}
    store_scores = {n: scores(n, "store") for n in seeds}
    thresholds = ", ".join(f"`{n}` {test[n]['threshold']:.3f}" for n in seeds)

    md = f"""# M1: the date detector

A small YOLO (`yolo11n`, 640 px input) fine-tuned to find every printed
date on a pack, trained on ExpDate's Products-Synth plus Products-Real
train. Everything was chosen on 165 validation photos; the ExpDate test
split and the store photos were each scored once, afterwards. M1 is judged
only on **finding** dates; reading them is M3's job. Plan and reasoning:
[`docs/m1-date-detector.md`](../docs/m1-date-detector.md).

## Headline

A date counts as found when a box at or above the run's threshold overlaps
it with IoU 0.5 or more. `{chosen}` is the M1 detector (seed 0); the
means are over three training seeds, with their range.

{headline(test, store_scores, chosen)}

**The gap is the result.** On ExpDate, the kind of photo it trained on, the
detector finds almost every expiry date. On the retailer's store photos it
finds about a quarter. Sideways and small print are 0 of 15 each, and dates
with the month written as a word are almost never found
([`experiments/m1-store-misses`](../experiments/m1-store-misses/README.md)).
Closing that gap is M2's job.

## How the detector was chosen (validation, 187 dates)

**C1, model size and input size.** Rule fixed before the runs: the
highest mAP@0.5 for dates; within 1 point of the best, the cheapest run.
All four found 185 to 187 of 187 dates, so the rule took `{chosen}`
([`experiments/m1-size-and-input`](../experiments/m1-size-and-input/README.md)).

{c1_table}

**C2, synthetic data.** Rule: keep the synthetic photos unless real only
is better by more than 1 point
([`experiments/m1-synthetic-data`](../experiments/m1-synthetic-data/README.md)).
The two runs also got different optimisers from Ultralytics'
`optimizer=auto`, so part of the gap may be the optimiser.

{c2_table}

**The threshold.** Each run keeps the confidence that gave the best F1 for
dates on validation: {thresholds}. F1 weighs a missed date and a false box
equally; for this project a miss costs more, so the next detector should
pick its threshold for recall, on validation (plan, Later).

## ExpDate Products-Real test (D2)

665 photos, 733 dates, 665 of them the expiry.

{m1.scores_table(test)}

{cross_check(test)} In D2, `{chosen}` put a box on every test expiry date
at some confidence: the ones it missed were under its threshold (plan, D2).

## Store photos, by failure tag (D3)

The frozen store test set v1 (`{store.TEST_SET_SHA256[:12]}…`): 99 photos,
199 dates, 81 of them a printed expiry. Six more expiries are worked out
from a mfg date and shelf life, so they have no box to find.

{m1.scores_table(store_scores)}

{m1.tag_table(store_scores)}

"Boxed at any confidence" counts expiry dates that got any box at all,
however unsure: it separates "never seen" from "seen, but under the
threshold". It changes no headline number. Every photo is padded to the
same 640 px square (`rect=False`), as on validation, so the thresholds mean
the same thing here.

## Compared with the ExpDate paper

No comparable number. Seker & Ahn (2022) report about 97.74% accuracy for
recognising expiry dates: that is reading them end to end, not finding
them, so it belongs beside M3's reader, not this detector. No public
source gives their detection recall or mAP on Products-Real test.

## Runs

Made by `python scripts/m1_report.py` from:

{runs_list({"ExpDate test": test, "the store photos": store_scores})}
"""
    OUT.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
