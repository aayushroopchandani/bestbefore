"""Rebuild the model-size results (M2b: n, s and m at step 1 and step 2, and R3) from the runs.

    scripts/pull_cluster_results.sh                            # runs -> laptop
    python experiments/m2b-model-size/summarise.py             # print
    python experiments/m2b-model-size/summarise.py --write     # and update README.md

Reads every run's suite-m2b.json and run.json, step 1's suite.json (M2's
suite, the early read), M2's records for n's step 1, and each run's pulled
results.csv (compute). Works the noise band, R1 and R3 out again with the
chain's own function, bestbefore.m2b.decisions(), and checks each, and the
chain's whole record, against results/runs/m2b-decisions.json: if any
differs, it says so and exits with 1. Runs not in yet show as "not yet".
Counts only: no store photo is named.
"""
from pathlib import Path

from bestbefore import m2_results, m2b_results

if __name__ == "__main__":
    m2_results.summarise(m2b_results.model_size_block, Path(__file__).with_name("README.md"),
                         __doc__.split("\n")[0])
