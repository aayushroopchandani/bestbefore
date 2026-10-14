"""Rebuild the C1 results (model size and input size) from the runs.

    scripts/pull_cluster_results.sh                              # runs -> laptop
    python experiments/m1-size-and-input/summarise.py            # print
    python experiments/m1-size-and-input/summarise.py --write    # and update README.md

Applies the same decision rule as the chain (bestbefore.m1.decide_c1) and
checks that it agrees with the chain's own record, m1-decisions.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bestbefore import m1
from bestbefore.runs import RUNS_DIR

README = Path(__file__).with_name("README.md")


def block(runs_dir: Path) -> str:
    done = {n: r for n in m1.C1 if (r := m1.load_run(n, runs_dir))}
    if not done:
        return "_Not run yet: the runs are queued in GPU job 72789._"
    table = m1.markdown_table([m1.row(n, *done[n]) for n in m1.C1 if n in done])
    if len(done) < len(m1.C1):
        return f"_{len(done)} of {len(m1.C1)} runs finished so far._\n\n{table}"

    scores = {n: val for n, (_, val) in done.items()}
    winner = m1.decide_c1(scores)
    top = max(scores, key=lambda n: m1.map50_date(scores[n]))
    if winner == top:
        why = "the highest validation mAP@0.5 for dates"
    else:
        gap = m1.map50_date(scores[top]) - m1.map50_date(scores[winner])
        why = (f"within {gap:.3f} of the best ({top}, "
               f"{m1.map50_date(scores[top]):.3f}), which is under 1 point, and cheaper")
    text = f"{table}\n\n**Decision, by the rule above: `{winner}`**, {why}."

    chain = runs_dir / "m1-decisions.json"
    if chain.exists():
        chose = json.loads(chain.read_text())["c1_winner"]
        if chose != winner:
            text += f"\n\n**Warning:** the chain chose `{chose}`. Check the runs."
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write", action="store_true", help="update README.md")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args()

    text = block(args.runs_dir)
    print(text)
    if args.write:
        README.write_text(m1.replace_results(README.read_text(), text))
        print(f"\nupdated {README}")


if __name__ == "__main__":
    main()
