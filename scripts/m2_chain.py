"""M2's GPU work queue, run by every M2 GPU job (slurm/m2_chain.sh).

    python scripts/m2_chain.py            # what a GPU job runs
    python scripts/m2_chain.py --dry-run  # every task's state; runs, claims and writes nothing

docs/m2-synthetic-generator.md, "The chain". Jobs A and B walk one fixed
task list, and each runs the first task that is ready (its data and
inputs exist), not done and not claimed, scoring tasks first:

  stage 1 (S2's derived sets)    m2-smoke, scored at once (synth-val missing);
                                 m2-speed-a, -b, -c -> the loader decision;
                                 m2-base-s0, m2-rot-s0, then the s1s, then the s2s
  stage 2 (S6's generated sets)  m2-pilot, then scored -> the pilot gate;
                                 m2-gen-rot-1280-s0..s2 (longest first);
                                 m2-gen-s0, m2-gen-rot-s0, m2-gen10k-s0, the s1s, the s2s
  scoring                        suite.json for every run but the speed runs,
                                 once all four suite sets exist

- Claims are folders in results/runs/m2-claims/ (bestbefore.slurm.Claims);
  a claim whose job Slurm no longer lists is taken over.
- Done: the run's run.json says finished (training), or its suite.json
  exists (scoring). A run cut off midway restarts from scratch, in a new
  folder.
- Watchdog: a training run that makes no progress for 2 hours (training
  starting, after the label scan and RAM cache, or a new results.csv row)
  is stopped and marked failed. A task that failed twice is left for a
  person (results/runs/m2-failures/).
- A task starts only if twice its estimate fits in the job's time left.
- Nothing to do: the job ends, except while the other job runs the pilot
  (it waits up to 30 minutes, checking every 5). With tasks left, it
  submits its replacement: held unless some task's data is ready, at most
  two M2 GPU jobs, and nothing if sbatch refuses (the 5-job cap).
- Decisions go to results/runs/m2-decisions.json as soon as their inputs
  are complete.

The machinery (claims, the watchdog, time left, failures, replacements,
the dry run) is bestbefore.chain's, shared with M2b's chain; this script
holds M2's tasks, their readiness, how each runs and the decisions.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from bestbefore import chain, m2, slurm
from bestbefore.chain import State
from bestbefore.config import checkpoint_dir

GPU_SCRIPT = "slurm/m2_gpu.sbatch"
JOB_NAME = "m2"                  # its #SBATCH --job-name
STALL = chain.STALL              # the watchdog: seconds without progress
POLL = chain.POLL                # how often a training run is checked
PILOT_WAIT, PILOT_POLL = 30 * 60, 5 * 60
SCORE_TIMEOUT = 3600             # a suite takes minutes
MAX_FAILURES = chain.MAX_FAILURES
MARGIN = chain.MARGIN            # start a task only if MARGIN x its estimate fits in the time left


def claims_dir() -> Path:
    return _chain().claims_dir()


# --- The tasks ----------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    name: str                        # m2-base-s0, or score-m2-base-s0
    run: str                         # the run it trains or scores: base-s0
    arm: str
    stage: int
    hours: float                     # the plan's estimate, at M1's speed
    seed: int = 0
    flags: tuple[str, ...] = ()      # more train_detector.py flags (smoke, speed, pilot)
    loader: bool = True              # trains with the loader decision's settings
    gated: bool = False              # waits for the pilot gate
    score: bool = False              # a scoring task
    synth_val: bool = True           # scoring waits for synth-val too (all but the smoke)


def loader_flags(loader: dict) -> tuple[str, ...]:
    return ("--workers", str(loader["workers"]), *(("--cache", "ram") if loader["cache"] else ()))


def training(run: str, arm: str, stage: int, hours: float, **kw) -> Task:
    return Task(f"m2-{run}", run, arm, stage, hours, **kw)


def scoring(task: Task) -> Task:
    return replace(task, name=f"score-{task.name}", hours=0.25, score=True,
                   synth_val=task.run != "smoke")


def task_list() -> list[Task]:
    """Every task, in the order the chain takes them."""
    stage1 = [training("smoke", "base", 1, 0.1, loader=False,
                       flags=("--epochs", "1", "--fraction", "0.02")),
              *(training(f"speed-{k}", "base", 1, 0.1, loader=False,
                         flags=("--epochs", "3", *loader_flags(m2.LOADERS[k])))
                for k in m2.LOADERS)]
    stage1 += [training(f"{arm}-s{s}", arm, 1, 2.0, seed=s)
               for s in m2.SEEDS for arm in ("base", "rot")]
    stage2 = [training("pilot", "gen", 2, 0.25, flags=("--budget-share", str(m2.PILOT_SHARE)))]
    stage2 += [training(f"gen-rot-1280-s{s}", "gen-rot-1280", 2, 3.3, seed=s, gated=True)
               for s in m2.SEEDS]
    stage2 += [training(f"{arm}-s{s}", arm, 2, 2.0, seed=s, gated=True)
               for s in m2.SEEDS for arm in ("gen", "gen-rot", "gen10k")]
    trained = stage1 + stage2
    return trained + [scoring(t) for t in trained if not t.run.startswith("speed-")]


TASKS = task_list()


# --- Where each task stands ----------------------------------------------------

def done(task: Task) -> Path | None:
    """The finished attempt at the task's run, if the task is done."""
    run_dir = m2.finished_run(task.run)
    if run_dir is None or (task.score and not (run_dir / "suite.json").exists()):
        return None
    return run_dir


def failures(task: Task) -> int:
    return _chain().failures(task)


def waiting_for(task: Task, decisions: dict) -> str | None:
    """What a task still needs, or None if it can start."""
    if task.score:
        if m2.finished_run(task.run) is None:
            return f"waiting for m2-{task.run} to finish"
        needed = m2.SUITE if task.synth_val else m2.THRESHOLD_SETS
        why = m2.missing({name: m2.suite_missing(name) for name in needed})
        return f"waiting for {why}" if why else None
    if why := m2.missing({name: m2.list_missing(name) for name in m2.ARMS[task.arm].names()}):
        return f"waiting for {why}"
    if not (checkpoint_dir() / "pretrained" / f"{m2.WEIGHTS}.pt").exists():
        return "waiting for the pretrained weights (scripts/fetch_pretrained.py)"
    if task.loader and "loader" not in decisions:
        return "waiting for the loader decision (m2-speed-a, -b, -c)"
    if task.gated:
        if "pilot" not in decisions:
            return "waiting for the pilot gate (m2-pilot, scored)"
        if not decisions["pilot"]["passes"]:
            return "blocked: the pilot gate failed, so the generated labels need a look"
    return None


def state(task: Task, decisions: dict, claims: slurm.Claims, live: set[str] | None) -> State:
    return _chain().state(task, decisions, claims, live)


def states(decisions: dict, claims: slurm.Claims, live: set[str] | None) -> dict[Task, State]:
    return _chain().states(decisions, claims, live)


def pick(current: dict[Task, State], time_left: float, skip: set[Task] = frozenset()) -> Task | None:
    """The next task to run: scoring first, then the list's order; only a
    task whose estimate fits MARGIN times in the time left, and none this
    job just failed to claim."""
    return _chain().pick(current, time_left, skip)


# --- Decisions (m2.py's rules, applied as soon as their inputs are complete) ---

def decide() -> dict:
    """m2.decisions(), the same function experiments/m2-*/summarise.py re-check with."""
    return m2.decisions()


def save(decisions: dict) -> None:
    _chain().save(decisions)


# --- Running a task --------------------------------------------------------------

def attempt_dir(run: str, launched: float) -> Path | None:
    """This attempt's run folder: the newest one started since `launched`."""
    return _chain().attempt_dir(run, launched)


def progress(run: str, launched: float) -> tuple[bool, int]:
    """How far this attempt has got: whether training has started (after
    the label scan and RAM cache), and how many epochs results.csv holds."""
    return _chain().progress(run, launched)


def watch(proc, run: str, launched: float) -> str | None:
    """Wait for a training run; None if it finished, else why it failed.
    STALL seconds without progress stop it."""
    return _chain().watch(proc, run, launched)


def run_task(task: Task, decisions: dict) -> str | None:
    """Run one task; None if it succeeded, else why it failed."""
    if task.score:
        return chain.score([sys.executable, "scripts/score_suite.py", "--run", task.run],
                           SCORE_TIMEOUT)
    cmd = [sys.executable, "scripts/train_detector.py", "--milestone", "m2", "--arm", task.arm,
           "--seed", str(task.seed), "--name", task.run, *task.flags]
    if task.loader:
        cmd += loader_flags(decisions["loader"])
    return _chain().train(cmd, task.run)


def record_failure(task: Task, why: str, job: str) -> None:
    _chain().record_failure(task, why, job)


# --- The job's end ----------------------------------------------------------------

def replace_job(current: dict[Task, State], job: str) -> None:
    """A job that ends with tasks left submits its replacement: held unless
    some task's data is ready, and never more than two M2 GPU jobs."""
    _chain().replace_job(current, job)


# --- Main ----------------------------------------------------------------------------

GROUPS = (("stage 1: needs S2's derived sets", lambda t: t.stage == 1 and not t.score),
          ("stage 2: needs S6's generated sets", lambda t: t.stage == 2 and not t.score),
          ("scoring", lambda t: t.score))


def summary(decisions: dict) -> str:
    """The dry run's line of decisions: each one's choice."""
    return ", ".join(f"{k}: {v.get('choice', v.get('arm', v.get('passes')))}"
                     for k, v in decisions.items()) or "none yet"


def pilot_elsewhere(current: dict[Task, State]) -> str | None:
    """With nothing ready, wait while the other job runs the pilot: it
    unlocks most of stage 2 within minutes."""
    if any(t.run == "pilot" and s.status == "claimed" for t, s in current.items()):
        return "the other job is running the pilot"
    return None


def _chain() -> chain.Chain:
    """M2's chain, from this script's names as they are when it is called
    (the tests swap TASKS, run_task, STALL and POLL)."""
    return chain.Chain(
        milestone="m2", title="M2", gpu_script=GPU_SCRIPT, job_name=JOB_NAME, tasks=TASKS,
        done=done, waiting_for=waiting_for, run_task=run_task, decide=decide,
        run_dirs=m2.run_dirs, groups=GROUPS, summary=summary, wait=pilot_elsewhere,
        wait_limit=PILOT_WAIT, wait_poll=PILOT_POLL, stall=STALL, poll=POLL,
        max_failures=MAX_FAILURES, margin=MARGIN)


def dry_run(claims: slurm.Claims) -> None:
    _chain().dry_run(claims)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="print every task's state; run, claim and write nothing")
    args = parser.parse_args()
    job = os.environ.get("SLURM_JOB_ID")
    claims = slurm.Claims(claims_dir(), job)
    if args.dry_run:
        dry_run(claims)
        return
    if job is None:
        raise SystemExit("m2_chain.py runs inside a Slurm job; elsewhere, use --dry-run")
    _chain().run_job(job, claims)


if __name__ == "__main__":
    main()
