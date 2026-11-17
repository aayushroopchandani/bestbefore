"""Rebuild the inference add-on results (M2b S13: R4, the what-if, every configuration, n timed against m) from the runs.

    sbatch slurm/m2b_inference_cpu.sbatch                      # the cluster: real-val's passes; copy its run back
    python scripts/m2b_inference.py predict --set v1-val --device cpu      # the laptop
    python scripts/m2b_inference.py score
    python scripts/m2b_inference.py time --device cpu
    python experiments/m2b-inference/summarise.py              # print
    python experiments/m2b-inference/summarise.py --write      # and update README.md

Reads the newest inference.json and timing.json that scripts/m2b_inference.py
wrote, the device check (score_suite.py --sets v1-val --out
<run>/device-check-cpu) and each run's suite-m2b.json. Works R4 (the noise
band) and the what-if (band 0.03) out again with the rules' own function,
bestbefore.m2b.decide_addons(), from the record's APs, checks them and the
detector's arm, run and threshold against R3 in m2b.decisions(), and the
chain's whole record against results/runs/m2b-decisions.json: if any
differs, or the AP or device check fails, it says so and exits with 1.
Counts only: no store photo is named.
"""
from pathlib import Path

from bestbefore import m2_results, m2b_results

if __name__ == "__main__":
    m2_results.summarise(m2b_results.inference_block, Path(__file__).with_name("README.md"),
                         __doc__.split("\n")[0])
