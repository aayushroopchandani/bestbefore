"""M2b's GPU work queue, run by every M2b GPU job (slurm/m2b_chain.sh).

    python scripts/m2b_chain.py            # what a GPU job runs
    python scripts/m2b_chain.py --dry-run  # every task's state; runs, claims and writes nothing

docs/m2b-store-and-size.md, "The chain". It runs on M2's machinery
(bestbefore.chain): claims in results/runs/m2b-claims/, the watchdog, time
left, failures in results/runs/m2b-failures/, replacements and the dry run
work as in M2. Jobs A and B walk one task list:

  stage 1 (public data)  m2b-smoke: 1 epoch, s at 1,280 px, 2% of gen-rot-1280's mix;
                         m2b-n1280-s0..s2: the control's records (M2's weights, no training);
                         m2b-s1280-s0..s2, then m2b-m1280-s0 (the longest last)
  stage 2 (S10)          step 2 on n, then on s and m with R2's recipe: waits for v2
                         frozen, v1 on the cluster, gen-v2, the 180° sets and S10's code;
                         m2b-m1280-s1, -s2 wait for rule R1
  scoring                suite.json, M2's suite, for every trained step-1 run, at once

- Done: the run's run.json says finished (training, a control's record),
  or its suite.json exists (scoring).
- Training runs use M2's loader decision: 8 workers, no RAM cache.
- Nothing ready: the job ends; with tasks left it submits a held
  replacement, as M2's did. Job B stays held until S10.
- Decisions go to results/runs/m2b-decisions.json: none before S10.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from bestbefore import chain, m2, m2b, slurm, store
from bestbefore.config import checkpoint_dir

GPU_SCRIPT = "slurm/m2b_gpu.sbatch"
JOB_NAME = "m2b"                 # its #SBATCH --job-name
SCORE_TIMEOUT = 3600             # a suite takes minutes
LOADER_FLAGS = ("--workers", str(m2b.LOADER["workers"]))    # no RAM cache: M2's decision


# --- The tasks ----------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    name: str                        # m2b-s1280-s0, or score-m2b-s1280-s0
    run: str                         # the run it trains, records or scores: s1280-s0
    arm: str                         # a step-1 arm, n1280 (the control) or a step-2 arm
    stage: int
    hours: float                     # the plan's estimate
    seed: int = 0
    flags: tuple[str, ...] = ()      # more train_detector.py flags (the smoke)
    control: bool = False            # the control's record: M2's weights, no training
    score: bool = False              # a scoring task, on M2's suite


def training(run: str, arm: str, stage: int, hours: float, **kw) -> Task:
    return Task(f"m2b-{run}", run, arm, stage, hours, **kw)


def scoring(task: Task) -> Task:
    return replace(task, name=f"score-{task.name}", hours=0.25, score=True)


# Step 2's arms on n (S10). s and m take R2's recipe, named once R2 chooses.
STEP2_N = ("n1280-ft", "n1280-ft2", "n1280-ft-gen50k")
RECIPE = "<recipe>"


def task_list() -> list[Task]:
    """Every task, in the order the chain takes them."""
    stage1 = [training("smoke", "s1280", 1, 0.1, flags=("--epochs", "1", "--fraction", "0.02"))]
    stage1 += [training(f"{m2b.CONTROL}-s{s}", m2b.CONTROL, 1, 0.01, seed=s, control=True)
               for s in m2b.SEEDS]
    stage1 += [training(f"s1280-s{s}", "s1280", 1, 4.0, seed=s) for s in m2b.SEEDS]
    stage1 += [training("m1280-s0", "m1280", 1, 7.0)]
    stage2 = [training(f"{arm}-s{s}", arm, 2, 0.4, seed=s) for s in m2b.SEEDS for arm in STEP2_N]
    stage2 += [training(f"s1280-{RECIPE}-s{s}", f"s1280-{RECIPE}", 2, 0.5, seed=s)
               for s in m2b.SEEDS]
    stage2 += [training(f"m1280-s{s}", "m1280", 2, 7.0, seed=s) for s in m2b.SEEDS[1:]]
    stage2 += [training(f"m1280-{RECIPE}-s{s}", f"m1280-{RECIPE}", 2, 0.8, seed=s)
               for s in m2b.SEEDS]
    trained = stage1 + stage2
    return trained + [scoring(t) for t in trained if t.arm in m2b.STEP1]


TASKS = task_list()


# --- Where each task stands ----------------------------------------------------

def done(task: Task) -> Path | None:
    """The finished attempt at the task's run, if the task is done."""
    run_dir = m2b.finished_run(task.run)
    if run_dir is None or (task.score and not (run_dir / "suite.json").exists()):
        return None
    return run_dir


def step2_missing(task: Task) -> str:
    """What a step-2 run waits for: the data S4 to S9 make, and S10's code,
    without which no step-2 run can start, whatever data exists."""
    why = []
    if store.TEST_SET_V2_SHA256 is None:
        why.append("store set v2's labels frozen (M0 D6)")
    if reason := m2b.store_export_missing():
        why.append(reason)
    lists = [n for n in m2b.STEP2_LISTS if not (task.arm.endswith("-gen50k") and n == "gen_v2")]
    if reason := m2.missing({n: m2b.list_missing(n) for n in lists}):
        why.append(reason)
    return "; ".join([*why, "step 2's code and rules (S10)"])


def waiting_for(task: Task, decisions: dict) -> str | None:
    """What a task still needs, or None if it can start."""
    if task.score:
        if m2b.finished_run(task.run) is None:
            return f"waiting for m2b-{task.run} to finish"
        why = m2.missing({name: m2.suite_missing(name) for name in m2.SUITE})
        return f"waiting for {why}" if why else None
    if task.control:
        why = m2b.control_missing(task.seed)
        return f"waiting for {why}" if why else None
    if task.stage == 2:
        if task.arm in m2b.STEP1:                    # m1280's seeds 1 and 2
            return "waiting for R1: m1280-s0 against every s1280 seed on v1-val (S10)"
        return f"waiting for {step2_missing(task)}"
    step = m2b.STEP1[task.arm]
    if why := m2.missing({name: m2.list_missing(name) for name in step.arm.names()}):
        return f"waiting for {why}"
    if not (checkpoint_dir() / "pretrained" / f"{step.weights}.pt").exists():
        return f"waiting for the pretrained weights, {step.weights}.pt (scripts/fetch_pretrained.py)"
    return None


# --- Running a task --------------------------------------------------------------

def run_task(task: Task, decisions: dict) -> str | None:
    """Run one task; None if it succeeded, else why it failed."""
    if task.score:
        return chain.score([sys.executable, "scripts/score_suite.py", "--milestone", "m2b",
                            "--run", task.run], SCORE_TIMEOUT)
    if task.control:
        try:
            m2b.write_control(task.seed)
        except Exception as e:          # the task's failure, recorded as a run's would be
            return f"the control's record failed: {e!r}"
        return None
    if task.stage == 2:
        return "stage 2 isn't written yet (S10)"
    cmd = [sys.executable, "scripts/train_detector.py", "--milestone", "m2b", "--arm", task.arm,
           "--seed", str(task.seed), "--name", task.run, *task.flags, *LOADER_FLAGS]
    return CHAIN.train(cmd, task.run)


# --- Main ----------------------------------------------------------------------------

GROUPS = (("stage 1: public data", lambda t: t.stage == 1 and not t.score),
          ("stage 2: needs v2 frozen, v1 on the cluster, gen-v2, the 180° sets and S10's code",
           lambda t: t.stage == 2 and not t.score),
          ("scoring: M2's suite, every trained step-1 run", lambda t: t.score))


def summary(decisions: dict) -> str:
    """The dry run's line of decisions."""
    return ", ".join(decisions) or "none yet"


CHAIN = chain.Chain(milestone="m2b", title="M2b", gpu_script=GPU_SCRIPT, job_name=JOB_NAME,
                    tasks=TASKS, done=done, waiting_for=waiting_for, run_task=run_task,
                    decide=m2b.decisions, run_dirs=m2b.run_dirs, groups=GROUPS, summary=summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="print every task's state; run, claim and write nothing")
    args = parser.parse_args()
    job = os.environ.get("SLURM_JOB_ID")
    claims = slurm.Claims(CHAIN.claims_dir(), job)
    if args.dry_run:
        CHAIN.dry_run(claims)
        return
    if job is None:
        raise SystemExit("m2b_chain.py runs inside a Slurm job; elsewhere, use --dry-run")
    CHAIN.run_job(job, claims)


if __name__ == "__main__":
    main()
