"""Score a baseline reader on a labelled set: right, wrong or none per photo.

    python scripts/run_baseline.py --reader a --set real-test
    python scripts/run_baseline.py --reader a --set store

Per-photo results (the OCR text, the dates found, the one read as the
expiry, the score) go to <run>/results.jsonl: under results/runs/ for
ExpDate, under data/store/runs/ for the store photos. The summary is
printed and saved in the run's run.json, along with a hash of the reader's
code, so a number can't drift from the code that made it.

Photos with nothing to score against are counted, not scored: ExpDate
photos with no marked expiry, and store photos whose expiry isn't in the
photo (other-face) or can't be read.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from bestbefore import dates, store
from bestbefore.baselines import rapidocr_regex
from bestbefore.m1 import markdown_table
from bestbefore.runs import start_run, update_run
from bestbefore.scoring import OUTCOMES, score, summarise
from bestbefore.sets import load_set, runs_root

READERS = {"a": rapidocr_regex}
SETS = ("real-test", "store")


def code_hashes(reader) -> dict[str, str]:
    """SHA-256 of the files that decide what a reader returns."""
    return {f.__name__: store.sha256(f.__file__) for f in (reader, dates)}


def table(summary: dict) -> str:
    def row(name: str, t: dict) -> dict[str, str]:
        return {"photos": name, "n": f"{t['n']}",
                **{o: f"{t[o]}/{t['n']}" for o in OUTCOMES}}

    by_tag = summary["by_tag"]
    rows = [row(t, by_tag[t]) for t in store.TAGS if t in by_tag]
    rows.append(row("**all scored**", summary["all"]))
    text = markdown_table(rows)
    for group, c in summary["not_scored"].items():
        text += f"\n\nNot scored, {group}: {c['n']} photos; a date was read from {c['date_read']}."
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reader", required=True, choices=READERS)
    parser.add_argument("--set", required=True, choices=SETS)
    args = parser.parse_args()

    reader = READERS[args.reader]
    samples = load_set(args.set)
    run_dir = start_run(f"baseline-{args.reader}-{args.set}",
                        {**vars(args), "code_sha256": code_hashes(reader)},
                        root=runs_root(args.set))
    started, rows = time.time(), []
    with open(run_dir / "results.jsonl", "w") as out:
        results = reader.read(s.image for s in samples)
        for i, (s, r) in enumerate(zip(samples, results), 1):
            read = r.pop("expiry")
            row = {"id": s.id, "tag": s.tag,
                   "truth": str(s.expiry) if s.expiry else None,
                   "read": str(read) if read else None,
                   "score": score(s.expiry, read) if s.expiry else None, **r}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            if i % 50 == 0 or i == len(samples):
                print(f"{i}/{len(samples)} photos", file=sys.stderr)

    summary = summarise(rows)
    update_run(run_dir, summary=summary, seconds=round(time.time() - started))
    print(f"{run_dir}\n\n{table(summary)}")


if __name__ == "__main__":
    main()
