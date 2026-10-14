"""Rebuild the C2 results (does synthetic data help?) from the runs.

    scripts/pull_cluster_results.sh                              # runs -> laptop
    python experiments/m1-synthetic-data/summarise.py            # print
    python experiments/m1-synthetic-data/summarise.py --write    # and update README.md

The C1 winner is worked out again from the C1 runs with the same rule as
the chain, then compared with its real-only twin. The decision uses the
same rule as the chain (bestbefore.m1.keep_synthetic) and is checked
against the chain's own record, m1-decisions.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bestbefore import m1
from bestbefore.runs import RUNS_DIR

README = Path(__file__).with_name("README.md")


def block(runs_dir: Path) -> str:
    c1 = {n: r for n in m1.C1 if (r := m1.load_run(n, runs_dir))}
    if len(c1) < len(m1.C1):
        return "_Not run yet: C2 starts once all four C1 runs have finished (GPU job 72789)._"
    winner = m1.decide_c1({n: val for n, (_, val) in c1.items()})
    real_only = m1.load_run(f"{winner}-real", runs_dir)
    if real_only is None:
        return f"_C1 chose `{winner}`; its real-only run hasn't finished yet._"

    with_synth = c1[winner]
    table = m1.markdown_table([m1.row(winner, *with_synth),
                               m1.row(f"{winner}-real", *real_only)])
    a, b = m1.map50_date(with_synth[1]), m1.map50_date(real_only[1])
    keep = m1.keep_synthetic(with_synth[1], real_only[1])
    diff = f"synthetic + real {a:.3f} vs real only {b:.3f} ({a - b:+.3f})"
    verdict = ("keep the synthetic photos" if keep
               else "drop the synthetic photos (real only is better by more than 1 point)")
    text = f"{table}\n\n**Decision, by the rule above: {verdict}.** mAP@0.5 for dates: {diff}."

    chain = runs_dir / "m1-decisions.json"
    if chain.exists() and json.loads(chain.read_text())["c2_keep_synthetic"] != keep:
        text += "\n\n**Warning:** the chain decided the other way. Check the runs."
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
