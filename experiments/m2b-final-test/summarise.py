"""Rebuild the final test's results (M2b S14: store set v2 once, by tag, cause and design; ExpDate test for the M2b detector) from the records.

    python scripts/eval_detector.py --set store-v2 --dry-run --runs ...   # the pre-flight, final_test.RUNS's 30
    python scripts/eval_detector.py --set store-v2 --runs ...             # the laptop, in S14's 12 parts
    sbatch slurm/m2b_test_cpu.sbatch                  # the cluster: ExpDate test; copy each real-test.json back
    python experiments/m2b-final-test/summarise.py                  # print
    python experiments/m2b-final-test/summarise.py --write          # and update README.md

Reads each final-test run's store-v2.json and real-test.json
(scripts/eval_detector.py), the device checks (score_suite.py --sets v1-val
--out <run>/device-check-cpu, and -batch1 for one photo a call) and the M2b
suites. Checks that every record was cut at its own GPU record's threshold,
on a CPU, by the weights hashed for it, v2's one photo a call; that v2's
counts by cause and by design add up to each record's own; that the
per-photo boxes kept in data/store/v2/runs/ give each record's counts
again (the laptop only); that every device check passes; and the chain's
whole record against results/runs/m2b-decisions.json. If any check fails,
it says so and exits with 1. Counts only: no store photo is named.
"""
from pathlib import Path

from bestbefore import m2_results, m2b_results

if __name__ == "__main__":
    m2_results.summarise(m2b_results.final_test_block, Path(__file__).with_name("README.md"),
                         __doc__.split("\n")[0])
