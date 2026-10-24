"""Rebuild the data-size results (M2: gen against gen10k) from the runs.

    scripts/pull_cluster_results.sh                          # runs -> laptop
    python experiments/m2-data-size/summarise.py             # print
    python experiments/m2-data-size/summarise.py --write     # and update README.md

Applies the rule fixed in the plan (bestbefore.m2.more_photos_help) once
all six runs are scored on the validation suite. Runs not in yet show as
"not yet".
"""
from pathlib import Path

from bestbefore import m2_results

if __name__ == "__main__":
    m2_results.summarise(m2_results.data_size_block, Path(__file__).with_name("README.md"),
                         __doc__.split("\n")[0])
