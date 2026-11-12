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
  stage 2 (S10)          step 2 on n: n1280-ft, -ft2 and -ft-gen50k, the s0s first; step 2
                         on s with R2's recipe; m1280-s1, -s2 and step 2 on m (R2's recipe)
                         only if R1 passes. Each waits for v2 frozen, the store export, its
                         replay lists, its step-1 run and its rules
  scoring                suite.json, M2's suite: every trained step-1 run, at once;
                         suite-m2b.json, the M2b suite: every run, the control's three seeds
                         first, every other once the noise band is fixed

- Done: the run's run.json says finished (training, a control's record),
  or its suite.json or suite-m2b.json exists (scoring).
- Training runs use M2's loader decision: 8 workers, no RAM cache.
- Nothing ready: the job ends; with tasks left it submits a held
  replacement, as M2's did.
- Decisions go to results/runs/m2b-decisions.json as their inputs are
  complete (m2b.decisions): the noise band, R1, R2 and R3. Until R2 names a
  recipe, step 2 on s and m is listed as <recipe>; a rule that says no
  leaves its tasks blocked, for a person.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from bestbefore import chain, m2, m2b, slurm, store
from bestbefore.config import checkpoint_dir

GPU_SCRIPT = "slurm/m2b_gpu.sbatch"
JOB_NAME = "m2b"                 # its #SBATCH --job-name
SCORE_TIMEOUT = 3600             # a suite takes minutes
LOADER_FLAGS = ("--workers", str(m2b.LOADER["workers"]))    # no RAM cache: M2's decision


# --- The tasks ----------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    name: str                        # m2b-s1280-s0, score-m2b-s1280-s0 or suite-m2b-s1280-s0
    run: str                         # the run it trains, records or scores: s1280-s0
    arm: str                         # a step-1 arm, n1280 (the control) or a step-2 arm
    stage: int
    hours: float                     # the plan's estimate
    seed: int = 0
    flags: tuple[str, ...] = ()      # more train_detector.py flags (the smoke)
    control: bool = False            # the control's record: M2's weights, no training
    score: bool = False              # a scoring task
    suite: str = "m2"                # a scoring task's suite: M2's (suite.json) or M2b's (suite-m2b.json)


def training(run: str, arm: str, stage: int, hours: float, **kw) -> Task:
    return Task(f"m2b-{run}", run, arm, stage, hours, **kw)


def scoring(task: Task, suite: str) -> Task:
    """The task that scores a training task's run on M2's suite (score-...)
    or on the M2b suite (suite-...)."""
    prefix = "score" if suite == "m2" else "suite"
    return replace(task, name=f"{prefix}-{task.name}", hours=0.25, score=True, suite=suite)


# Step 2's arms on n. s and m take R2's recipe, <recipe> until R2 names one.
STEP2_N = ("n1280-ft", "n1280-ft2", "n1280-ft-gen50k")
RECIPE = "<recipe>"


def recipe(decisions: Mapping) -> str:
    """R2's recipe, once R2 names one; <recipe> until then."""
    return (decisions.get("R2") or {}).get("recipe") or RECIPE


def task_list(decisions: Mapping = MappingProxyType({})) -> list[Task]:
    """Every task, in the order the chain takes them, named by the decisions so far."""
    stage1 = [training("smoke", "s1280", 1, 0.1, flags=("--epochs", "1", "--fraction", "0.02"))]
    stage1 += [training(f"{m2b.CONTROL}-s{s}", m2b.CONTROL, 1, 0.01, seed=s, control=True)
               for s in m2b.SEEDS]
    stage1 += [training(f"s1280-s{s}", "s1280", 1, 4.0, seed=s) for s in m2b.SEEDS]
    stage1 += [training("m1280-s0", "m1280", 1, 7.0)]
    r = recipe(decisions)
    stage2 = [training(f"{arm}-s{s}", arm, 2, 0.4, seed=s) for s in m2b.SEEDS for arm in STEP2_N]
    stage2 += [training(f"s1280-{r}-s{s}", f"s1280-{r}", 2, 0.5, seed=s) for s in m2b.SEEDS]
    stage2 += [training(f"m1280-s{s}", "m1280", 2, 7.0, seed=s) for s in m2b.SEEDS[1:]]
    stage2 += [training(f"m1280-{r}-s{s}", f"m1280-{r}", 2, 0.8, seed=s) for s in m2b.SEEDS]
    trained = stage1 + stage2
    return (trained + [scoring(t, "m2") for t in trained if t.arm in m2b.STEP1]
            + [scoring(t, "m2b") for t in trained if t.run != "smoke"])


TASKS = task_list()


# --- Where each task stands ----------------------------------------------------

def done(task: Task) -> Path | None:
    """The finished attempt at the task's run, if the task is done."""
    run_dir = m2b.finished_run(task.run)
    if run_dir is None:
        return None
    if task.score and not (run_dir / (m2b.SUITE_FILE if task.suite == "m2b" else "suite.json")).exists():
        return None
    return run_dir


def step2_missing(task: Task) -> str | None:
    """The data a step-2 run still waits for, or None: v2 frozen (M0 D6),
    the store export (S6) and its replay lists (M2's, ExpDate's and S9's)."""
    why = []
    if store.TEST_SET_V2_SHA256 is None:
        why.append("store set v2's labels frozen (M0 D6)")
    if reason := m2b.store_export_missing():
        why.append(reason)
    names = [name for lists in m2b.replay_lists(m2b.STEP2[task.arm]).values() for name in lists]
    if reason := m2.missing({name: m2b.any_list_missing(name) for name in names}):
        why.append(reason)
    return "; ".join(why) or None


def rules_missing(task: Task, decisions: Mapping) -> str | None:
    """The rules a stage-2 task still waits for, or None: R1 for m's runs,
    R2 for step 2 on s and m; "blocked: ..." where a rule said no."""
    size = task.arm[0]
    if size == "m":
        if "R1" not in decisions:
            return "waiting for R1: m1280-s0 against every s1280 seed on v1-val (the M2b suite)"
        if not decisions["R1"]["passes"]:
            return "blocked: R1: m1280-s0 didn't earn m more seeds"
    if size in ("s", "m") and task.arm not in m2b.STEP1:
        if "R2" not in decisions:
            return "waiting for R2: n1280-ft against n1280-ft2 on the M2b suite"
        if decisions["R2"]["recipe"] is None:
            return "blocked: R2: no recipe passed the guard, for a person to look at"
    return None


def waiting_for(task: Task, decisions: Mapping) -> str | None:
    """What a task still needs, or None if it can start."""
    if task.score:
        if m2b.finished_run(task.run) is None:
            return f"waiting for m2b-{task.run} to finish"
        if task.suite == "m2":
            why = m2.missing({name: m2.suite_missing(name) for name in m2.SUITE})
            return f"waiting for {why}" if why else None
        if why := m2.missing({name: m2b.suite_missing(name) for name in m2b.SUITE}):
            return f"waiting for {why}"
        if not task.control and "noise_store" not in decisions:
            return "waiting for the noise band: the control's three seeds on the M2b suite"
        return None
    if task.control:
        why = m2b.control_missing(task.seed)
        return f"waiting for {why}" if why else None
    if task.stage == 2 and (why := rules_missing(task, decisions)):
        return why
    if task.arm in m2b.STEP1:
        step = m2b.STEP1[task.arm]
        if why := m2.missing({name: m2.list_missing(name) for name in step.arm.names()}):
            return f"waiting for {why}"
        if not (checkpoint_dir() / "pretrained" / f"{step.weights}.pt").exists():
            return f"waiting for the pretrained weights, {step.weights}.pt (scripts/fetch_pretrained.py)"
        return None
    if why := step2_missing(task):
        return f"waiting for {why}"
    if why := m2b.start_missing(m2b.STEP2[task.arm], task.seed):
        return f"waiting for {why}"
    return None


# --- Running a task --------------------------------------------------------------

def run_task(task: Task, decisions: Mapping) -> str | None:
    """Run one task; None if it succeeded, else why it failed."""
    if task.score:
        suite = ("--suite", "m2b") if task.suite == "m2b" else ()
        return chain.score([sys.executable, "scripts/score_suite.py", "--milestone", "m2b",
                            "--run", task.run, *suite], SCORE_TIMEOUT)
    if task.control:
        try:
            m2b.write_control(task.seed)
        except Exception as e:          # the task's failure, recorded as a run's would be
            return f"the control's record failed: {e!r}"
        return None
    cmd = [sys.executable, "scripts/train_detector.py", "--milestone", "m2b", "--arm", task.arm,
           "--seed", str(task.seed), "--name", task.run, *task.flags, *LOADER_FLAGS]
    return CHAIN.train(cmd, task.run)


# --- Main ----------------------------------------------------------------------------

GROUPS = (("stage 1: public data", lambda t: t.stage == 1 and not t.score),
          ("stage 2: store fine-tuning, and m's seeds by R1", lambda t: t.stage == 2 and not t.score),
          ("scoring: M2's suite (suite.json), every trained step-1 run",
           lambda t: t.score and t.suite == "m2"),
          ("scoring: the M2b suite (suite-m2b.json), every run, the control first",
           lambda t: t.score and t.suite == "m2b"))


def summary(decisions: Mapping) -> str:
    """The dry run's line of decisions: each one's outcome."""
    parts = []
    if "noise_store" in decisions:
        parts.append(f"noise band {decisions['noise_store']['band']:.4f}")
    if "R1" in decisions:
        parts.append(f"R1 {'passes' if decisions['R1']['passes'] else 'fails'}")
    if "R2" in decisions:
        parts.append(f"R2 {decisions['R2']['recipe'] or 'no recipe passed the guard'}")
    if "R3" in decisions:
        parts.append(f"R3 {decisions['R3']['arm']}")
    return ", ".join(parts) or "none yet"


CHAIN = chain.Chain(milestone="m2b", title="M2b", gpu_script=GPU_SCRIPT, job_name=JOB_NAME,
                    tasks=task_list, done=done, waiting_for=waiting_for, run_task=run_task,
                    decide=m2b.decisions, run_dirs=m2b.run_dirs, groups=GROUPS, summary=summary,
                    width=30)


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
