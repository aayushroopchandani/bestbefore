"""Rebuild the loader speed test's results (M2, decision 15) from the runs.

    scripts/pull_cluster_results.sh                            # runs -> laptop
    python experiments/m2-loader-speed/summarise.py            # print
    python experiments/m2-loader-speed/summarise.py --write    # and update README.md

Applies the chain's own rule (bestbefore.m2.loader_choice, through
m2.decisions()) to the three speed runs' results.csv, and checks that it
agrees with the chain's record, results/runs/m2-decisions.json: if not, it
says so and exits with 1.
"""
from pathlib import Path

from bestbefore import m2_results

if __name__ == "__main__":
    m2_results.summarise(m2_results.loader_block, Path(__file__).with_name("README.md"),
                         __doc__.split("\n")[0])
