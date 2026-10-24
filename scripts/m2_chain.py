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
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from bestbefore import m2, runs, slurm
from bestbefore.config import REPO_ROOT, checkpoint_dir
from bestbefore.runs import update_run, write_json

GPU_SCRIPT = "slurm/m2_gpu.sbatch"
JOB_NAME = "m2"                  # its #SBATCH --job-name
STALL = 2 * 3600                 # the watchdog: seconds without progress
POLL = 60                        # how often a training run is checked
PILOT_WAIT, PILOT_POLL = 30 * 60, 5 * 60
SCORE_TIMEOUT = 3600             # a suite takes minutes
MAX_FAILURES = 2
MARGIN = 2.0                     # start a task only if MARGIN x its estimate fits in the time left


def claims_dir() -> Path:
    return runs.RUNS_DIR / "m2-claims"


def failures_dir() -> Path:
    return runs.RUNS_DIR / "m2-failures"


def decisions_path() -> Path:
    return m2.decisions_path()


def log(message: str) -> None:
    print(f"{datetime.now():%F %T}  {message}", flush=True)


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

@dataclass(frozen=True)
class State:
    status: str                      # done, ready, claimed, waiting, blocked or failed
    note: str = ""


def done(task: Task) -> Path | None:
    """The finished attempt at the task's run, if the task is done."""
    run_dir = m2.finished_run(task.run)
    if run_dir is None or (task.score and not (run_dir / "suite.json").exists()):
        return None
    return run_dir


def failures(task: Task) -> int:
    path = failures_dir() / f"{task.name}.txt"
    return len(path.read_text().splitlines()) if path.exists() else 0


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
    if (run_dir := done(task)) is not None:
        return State("done", run_dir.name)
    if (n := failures(task)) >= MAX_FAILURES:
        return State("failed", f"failed {n} times: results/runs/m2-failures/{task.name}.txt")
    if (why := waiting_for(task, decisions)) is not None:
        return State("blocked" if why.startswith("blocked") else "waiting", why)
    held = claims.holder(task.name)
    if held is not None and not claims.is_stale(task.name, held, live):
        return State("claimed", f"claimed by job {held[0]}")
    return State("ready", f"stale claim of job {held[0]}: will be taken over" if held else "")


def states(decisions: dict, claims: slurm.Claims, live: set[str] | None) -> dict[Task, State]:
    return {t: state(t, decisions, claims, live) for t in TASKS}


def pick(current: dict[Task, State], time_left: float, skip: set[Task] = frozenset()) -> Task | None:
    """The next task to run: scoring first, then the list's order; only a
    task whose estimate fits MARGIN times in the time left, and none this
    job just failed to claim."""
    fits = [t for t, s in current.items() if s.status == "ready" and t not in skip
            and MARGIN * t.hours * 3600 < time_left]
    return next((t for t in fits if t.score), None) or next(iter(fits), None)


# --- Decisions (m2.py's rules, applied as soon as their inputs are complete) ---

def decide() -> dict:
    """m2.decisions(), the same function experiments/m2-*/summarise.py re-check with."""
    return m2.decisions()


def save(decisions: dict) -> None:
    if decisions != m2.read_json(decisions_path()):
        decisions_path().parent.mkdir(parents=True, exist_ok=True)
        write_json(decisions_path(), decisions)
        log(f"[decisions] {', '.join(decisions) or 'none yet'}")


# --- Running a task --------------------------------------------------------------

def attempt_dir(run: str, launched: float) -> Path | None:
    """This attempt's run folder: the newest one started since `launched`."""
    for d in reversed(m2.run_dirs(run)):
        if datetime.strptime(d.name[:15], "%Y%m%d-%H%M%S").timestamp() >= launched - 1:
            return d
    return None


def progress(run: str, launched: float) -> tuple[bool, int]:
    """How far this attempt has got: whether training has started (after
    the label scan and RAM cache), and how many epochs results.csv holds."""
    run_dir = attempt_dir(run, launched)
    applied = (m2.read_json(run_dir / "run.json") or {}).get("applied") if run_dir else None
    if not applied:
        return False, 0
    csv = Path(applied["save_dir"]) / "results.csv"
    return True, max(0, len(csv.read_text().splitlines()) - 1) if csv.exists() else 0


def stop(proc: subprocess.Popen) -> None:
    """Stop a training run and its loader workers: its whole process group."""
    for sig, grace in ((signal.SIGTERM, 60), (signal.SIGKILL, 30)):
        try:
            os.killpg(proc.pid, sig)
            proc.wait(timeout=grace)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            continue


def watch(proc: subprocess.Popen, run: str, launched: float) -> str | None:
    """Wait for a training run; None if it finished, else why it failed.
    STALL seconds without progress stop it."""
    seen, since = (False, 0), launched
    while True:
        try:
            code = proc.wait(timeout=POLL)
            return None if code == 0 else f"train_detector.py exited with {code}"
        except subprocess.TimeoutExpired:
            pass
        now = progress(run, launched)
        if now != seen:
            seen, since = now, time.time()
        elif time.time() - since > STALL:
            stop(proc)
            started, epochs = seen
            where = f"after epoch {epochs}" if epochs else ("in epoch 1" if started else "in setup")
            return f"watchdog: no progress for {STALL / 3600:g} hours, {where}"


def run_task(task: Task, decisions: dict) -> str | None:
    """Run one task; None if it succeeded, else why it failed."""
    if task.score:
        cmd = [sys.executable, "scripts/score_suite.py", "--run", task.run]
        try:
            code = subprocess.run(cmd, cwd=REPO_ROOT, timeout=SCORE_TIMEOUT).returncode
        except subprocess.TimeoutExpired:
            return f"scoring took over {SCORE_TIMEOUT // 60} minutes"
        return None if code == 0 else f"score_suite.py exited with {code}"
    cmd = [sys.executable, "scripts/train_detector.py", "--milestone", "m2", "--arm", task.arm,
           "--seed", str(task.seed), "--name", task.run, *task.flags]
    if task.loader:
        cmd += loader_flags(decisions["loader"])
    launched = time.time()
    why = watch(subprocess.Popen(cmd, cwd=REPO_ROOT, start_new_session=True), task.run, launched)
    if why and (run_dir := attempt_dir(task.run, launched)) is not None:
        update_run(run_dir, status="failed", failure=why)
    return why


def record_failure(task: Task, why: str, job: str) -> None:
    path = failures_dir() / f"{task.name}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(f"{datetime.now():%F %T}  job {job}: {why}\n")


# --- The job's end ----------------------------------------------------------------

def replace_job(current: dict[Task, State], job: str) -> None:
    """A job that ends with tasks left submits its replacement: held unless
    some task's data is ready, and never more than two M2 GPU jobs."""
    left = [s for s in current.values() if s.status in ("ready", "claimed", "waiting")]
    if not left:
        log("[end] nothing left that a GPU job could do")
        return
    jobs = slurm.jobs_named(JOB_NAME)
    if jobs is None:
        log(f"[end] {len(left)} tasks left, but squeue didn't answer: no replacement")
        return
    others = sorted(set(jobs) - {job})
    if len(others) >= 2:
        log(f"[end] {len(left)} tasks left; M2 jobs {', '.join(others)} remain: no replacement")
        return
    hold = not any(s.status in ("ready", "claimed") for s in left)
    ok, answer = slurm.submit(GPU_SCRIPT, hold=hold, cwd=REPO_ROOT)
    if ok:
        log(f"[end] {len(left)} tasks left: replacement job {answer}" + (" (held)" if hold else ""))
    else:
        log(f"[end] {len(left)} tasks left; sbatch refused a replacement: {answer}")


# --- Main ----------------------------------------------------------------------------

def dry_run(claims: slurm.Claims) -> None:
    decisions = decide()
    current = states(decisions, claims, slurm.live_jobs())
    groups = (("stage 1: needs S2's derived sets", lambda t: t.stage == 1 and not t.score),
              ("stage 2: needs S6's generated sets", lambda t: t.stage == 2 and not t.score),
              ("scoring", lambda t: t.score))
    for title, member in groups:
        print(title)
        for task, s in current.items():
            if member(task):
                print(f"  {task.name:26s} {s.status:8s} {s.note}")
    print("decisions:", ", ".join(f"{k}: {v.get('choice', v.get('arm', v.get('passes')))}"
                                  for k, v in decisions.items()) or "none yet")


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

    log(f"job {job} on {socket.gethostname()}, {slurm.time_left() / 3600:.1f} hours left")
    waited, unclaimed = 0, set()
    while True:
        decisions = decide()
        save(decisions)
        live = slurm.live_jobs()
        current = states(decisions, claims, live)
        task = pick(current, slurm.time_left(), unclaimed)
        if task is not None:
            if not claims.claim(task.name, live):
                unclaimed.add(task)     # another job got there first
                continue
            log(f"[run] {task.name}")
            started = time.time()
            try:
                why = run_task(task, decisions)
                if why:
                    record_failure(task, why, job)
            finally:
                claims.release(task.name)
            log(f"[{'failed' if why else 'done'}] {task.name} after "
                f"{(time.time() - started) / 60:.0f} min" + (f": {why}" if why else ""))
            waited, unclaimed = 0, set()
            continue
        pilot_elsewhere = any(t.run == "pilot" and s.status == "claimed" for t, s in current.items())
        if pilot_elsewhere and waited < PILOT_WAIT:
            log("[wait] the other job is running the pilot")
            time.sleep(PILOT_POLL)
            waited, unclaimed = waited + PILOT_POLL, set()
            continue
        break
    decisions = decide()
    save(decisions)
    replace_job(states(decisions, claims, slurm.live_jobs()), job)


if __name__ == "__main__":
    main()
