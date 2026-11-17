"""Rebuild the gen-v2 side arm's results (M2b: n1280-ft against n1280-ft-gen50k) from the runs.

    scripts/pull_cluster_results.sh                         # runs -> laptop
    python experiments/m2b-gen-v2/summarise.py              # print
    python experiments/m2b-gen-v2/summarise.py --write      # and update README.md

Reads every run's suite-m2b.json, in the order the plan fixed for this
side question: synth-val-v2 per stratum, then real-val-small and v1-val's
dates by size, then real-val-rot180. No rule decides it; the chain's whole
record, results/runs/m2b-decisions.json, is still worked out again with
bestbefore.m2b.decisions() and checked: if it differs, it says so and
exits with 1. Runs not in yet show as "not yet". Counts only.
"""
from pathlib import Path

from bestbefore import m2_results, m2b_results

if __name__ == "__main__":
    m2_results.summarise(m2b_results.gen_v2_block, Path(__file__).with_name("README.md"),
                         __doc__.split("\n")[0])
