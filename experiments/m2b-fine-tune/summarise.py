"""Rebuild the fine-tuning results (M2b step 2 on n: ft, ft2 and the control) from the runs.

    scripts/pull_cluster_results.sh                           # runs -> laptop
    python experiments/m2b-fine-tune/summarise.py             # print
    python experiments/m2b-fine-tune/summarise.py --write     # and update README.md

Reads every run's suite-m2b.json and run.json, M2's suite.json for its
thresholds, and each run's pulled results.csv (compute). Works the noise
band, R1 and R2 out again with the chain's own function,
bestbefore.m2b.decisions(), and checks each, and the chain's whole record,
against results/runs/m2b-decisions.json: if any differs, it says so and
exits with 1. Runs not in yet show as "not yet". Counts only: no store
photo is named.
"""
from pathlib import Path

from bestbefore import m2_results, m2b_results

if __name__ == "__main__":
    m2_results.summarise(m2b_results.fine_tune_block, Path(__file__).with_name("README.md"),
                         __doc__.split("\n")[0])
