"""Rebuild the input-size results (M2: gen-rot at 640 against 1,280 px) from the runs.

    scripts/pull_cluster_results.sh                           # runs -> laptop
    python experiments/m2-input-size/summarise.py             # print
    python experiments/m2-input-size/summarise.py --write     # and update README.md

The plan fixes no rule here beyond reading real-val-small first; whether
1,280 px is used is the M2 detector rule's call (experiments/m2-ablation).
Runs not in yet show as "not yet".
"""
from pathlib import Path

from bestbefore import m2_results

if __name__ == "__main__":
    m2_results.summarise(m2_results.input_size_block, Path(__file__).with_name("README.md"),
                         __doc__.split("\n")[0])
