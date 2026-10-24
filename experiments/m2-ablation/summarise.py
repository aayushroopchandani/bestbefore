"""Rebuild the ablation's results (M2: base, rot, gen, gen-rot) from the runs.

    scripts/pull_cluster_results.sh                         # runs -> laptop
    python experiments/m2-ablation/summarise.py             # print
    python experiments/m2-ablation/summarise.py --write     # and update README.md

Reads every run's suite.json and, once S8 has scored them, store.json and
the per-photo store predictions in data/store/runs/ (for the causes; counts
only are printed). Applies the chain's own rules (bestbefore.m2.pilot_passes
and decide_detector, through m2.decisions()) and checks that they agree
with the chain's record, results/runs/m2-decisions.json: if not, it says so
and exits with 1. Runs not in yet show as "not yet".
"""
from pathlib import Path

from bestbefore import m2_results

if __name__ == "__main__":
    m2_results.summarise(m2_results.ablation_block, Path(__file__).with_name("README.md"),
                         __doc__.split("\n")[0])
