"""A milestone's GPU work queue: the machinery M2's chain built
(docs/m2-synthetic-generator.md, "The chain"), moved out of
scripts/m2_chain.py so that M2b's chain runs on the same code
(docs/m2b-store-and-size.md, S2).

Every GPU job of a milestone walks one fixed task list and runs the first
task that is ready (its data and inputs exist), not done and not claimed:
scoring tasks first, then the list's order. The milestone's script passes
in its tasks, when one is done, what it still waits for, how to run it and
its decisions (a Chain); the rest is here:

- Claims are folders in results/runs/<milestone>-claims/ (slurm.Claims);
  a claim whose job Slurm no longer lists is taken over.
- Watchdog: a training run that makes no progress for STALL seconds
  (training starting, after the label scan and RAM cache, or a new
  results.csv row) is stopped and marked failed in its run.json. A task
  that failed MAX_FAILURES times is left for a person
  (results/runs/<milestone>-failures/).
- A task starts only if MARGIN times its estimate fits in the job's time left.
- Nothing to do: the job ends, unless the milestone gives a reason to wait
  (M2: the other job running the pilot). With tasks left, it submits its
  replacement: held unless some task's data is ready, at most MAX_JOBS GPU
  jobs of the milestone, and nothing if sbatch refuses (the 5-job cap).
- Decisions go to results/runs/<milestone>-decisions.json as soon as their
  inputs are complete.

A task is any object with the fields read here: `name` (unique, e.g.
m2-base-s0), `run` (the run it trains or scores, base-s0), `hours` (its
estimate) and `score` (whether it is a scoring task).
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from . import runs, slurm
from .config import REPO_ROOT
from .runs import read_json, update_run, write_json

STALL = 2 * 3600                 # the watchdog: seconds without progress
POLL = 60                        # how often a training run is checked
MAX_FAILURES = 2
MARGIN = 2.0                     # start a task only if MARGIN x its estimate fits in the time left
MAX_JOBS = 2                     # GPU jobs of one milestone kept in the queue


def log(message: str) -> None:
    print(f"{datetime.now():%F %T}  {message}", flush=True)


class Task(Protocol):
    name: str
    run: str
    hours: float
    score: bool


@dataclass(frozen=True)
class State:
    status: str                      # done, ready, claimed, waiting, blocked or failed
    note: str = ""


def score(cmd: list[str], timeout: float) -> str | None:
    """Run a scoring command; None if it succeeded, else why it failed."""
    try:
        code = subprocess.run(cmd, cwd=REPO_ROOT, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        return f"scoring took over {timeout // 60:g} minutes"
    return None if code == 0 else f"{Path(cmd[1]).name} exited with {code}"


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


def never(current: dict) -> None:
    """No reason to wait: with nothing ready, the job ends."""
    return None


@dataclass(frozen=True)
class Chain:
    """One milestone's chain: its name, its GPU job, and the parts its
    script passes in."""
    milestone: str                                  # m2: runs <stamp>-m2-<run>, results/runs/m2-claims/
    title: str                                      # M2, as the log names its jobs
    gpu_script: str                                 # what a replacement job runs
    job_name: str                                   # that script's #SBATCH --job-name
    tasks: Sequence[Task]                           # in the order the chain takes them
    done: Callable[[Task], Path | None]             # the run folder that makes a task done, or None
    waiting_for: Callable[[Task, dict], str | None]  # what it still needs ("blocked: ..." if it never will)
    run_task: Callable[[Task, dict], str | None]    # None if the task succeeded, else why it failed
    decide: Callable[[], dict]                      # every decision whose inputs are complete
    run_dirs: Callable[[str], list[Path]]           # every attempt at a run, oldest first
    groups: Sequence[tuple[str, Callable[[Task], bool]]]   # the dry run's sections
    summary: Callable[[dict], str]                  # the dry run's line of decisions
    wait: Callable[[dict], str | None] = never      # why to wait rather than end, if anything
    wait_limit: float = 0                           # for at most this many seconds,
    wait_poll: float = 0                            # checking this often
    stall: float = STALL
    poll: float = POLL
    max_failures: int = MAX_FAILURES
    margin: float = MARGIN
    width: int = 26                                 # the dry run's column of task names

    def claims_dir(self) -> Path:
        return runs.RUNS_DIR / f"{self.milestone}-claims"

    def failures_dir(self) -> Path:
        return runs.RUNS_DIR / f"{self.milestone}-failures"

    def decisions_path(self) -> Path:
        return runs.RUNS_DIR / f"{self.milestone}-decisions.json"

    # --- Where each task stands -------------------------------------------------

    def failures(self, task: Task) -> int:
        path = self.failures_dir() / f"{task.name}.txt"
        return len(path.read_text().splitlines()) if path.exists() else 0

    def state(self, task: Task, decisions: dict, claims: slurm.Claims,
              live: set[str] | None) -> State:
        if (run_dir := self.done(task)) is not None:
            return State("done", run_dir.name)
        if (n := self.failures(task)) >= self.max_failures:
            return State("failed", f"failed {n} times: "
                                   f"results/runs/{self.milestone}-failures/{task.name}.txt")
        if (why := self.waiting_for(task, decisions)) is not None:
            return State("blocked" if why.startswith("blocked") else "waiting", why)
        held = claims.holder(task.name)
        if held is not None and not claims.is_stale(task.name, held, live):
            return State("claimed", f"claimed by job {held[0]}")
        return State("ready", f"stale claim of job {held[0]}: will be taken over" if held else "")

    def states(self, decisions: dict, claims: slurm.Claims,
               live: set[str] | None) -> dict[Task, State]:
        return {t: self.state(t, decisions, claims, live) for t in self.tasks}

    def pick(self, current: dict[Task, State], time_left: float,
             skip: set[Task] = frozenset()) -> Task | None:
        """The next task to run: scoring first, then the list's order; only a
        task whose estimate fits `margin` times in the time left, and none
        this job just failed to claim."""
        fits = [t for t, s in current.items() if s.status == "ready" and t not in skip
                and self.margin * t.hours * 3600 < time_left]
        return next((t for t in fits if t.score), None) or next(iter(fits), None)

    def save(self, decisions: dict) -> None:
        if decisions != read_json(self.decisions_path()):
            self.decisions_path().parent.mkdir(parents=True, exist_ok=True)
            write_json(self.decisions_path(), decisions)
            log(f"[decisions] {', '.join(decisions) or 'none yet'}")

    # --- Running a training task ---------------------------------------------------

    def attempt_dir(self, run: str, launched: float) -> Path | None:
        """This attempt's run folder: the newest one started since `launched`."""
        for d in reversed(self.run_dirs(run)):
            if datetime.strptime(d.name[:15], "%Y%m%d-%H%M%S").timestamp() >= launched - 1:
                return d
        return None

    def progress(self, run: str, launched: float) -> tuple[bool, int]:
        """How far this attempt has got: whether training has started (after
        the label scan and RAM cache), and how many epochs results.csv holds."""
        run_dir = self.attempt_dir(run, launched)
        applied = (read_json(run_dir / "run.json") or {}).get("applied") if run_dir else None
        if not applied:
            return False, 0
        csv = Path(applied["save_dir"]) / "results.csv"
        return True, max(0, len(csv.read_text().splitlines()) - 1) if csv.exists() else 0

    def watch(self, proc: subprocess.Popen, run: str, launched: float) -> str | None:
        """Wait for a training run; None if it finished, else why it failed.
        `stall` seconds without progress stop it."""
        seen, since = (False, 0), launched
        while True:
            try:
                code = proc.wait(timeout=self.poll)
                return None if code == 0 else f"train_detector.py exited with {code}"
            except subprocess.TimeoutExpired:
                pass
            now = self.progress(run, launched)
            if now != seen:
                seen, since = now, time.time()
            elif time.time() - since > self.stall:
                stop(proc)
                started, epochs = seen
                where = f"after epoch {epochs}" if epochs else ("in epoch 1" if started else "in setup")
                return f"watchdog: no progress for {self.stall / 3600:g} hours, {where}"

    def train(self, cmd: list[str], run: str) -> str | None:
        """Run a training command in its own process group, under the
        watchdog; None if it finished, else why it failed, which the
        attempt's run.json then records."""
        launched = time.time()
        why = self.watch(subprocess.Popen(cmd, cwd=REPO_ROOT, start_new_session=True), run, launched)
        if why and (run_dir := self.attempt_dir(run, launched)) is not None:
            update_run(run_dir, status="failed", failure=why)
        return why

    def record_failure(self, task: Task, why: str, job: str) -> None:
        path = self.failures_dir() / f"{task.name}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(f"{datetime.now():%F %T}  job {job}: {why}\n")

    # --- The job's end ----------------------------------------------------------------

    def replace_job(self, current: dict[Task, State], job: str) -> None:
        """A job that ends with tasks left submits its replacement: held unless
        some task's data is ready, and never more than MAX_JOBS GPU jobs of
        the milestone."""
        left = [s for s in current.values() if s.status in ("ready", "claimed", "waiting")]
        if not left:
            log("[end] nothing left that a GPU job could do")
            return
        jobs = slurm.jobs_named(self.job_name)
        if jobs is None:
            log(f"[end] {len(left)} tasks left, but squeue didn't answer: no replacement")
            return
        others = sorted(set(jobs) - {job})
        if len(others) >= MAX_JOBS:
            log(f"[end] {len(left)} tasks left; {self.title} jobs {', '.join(others)} remain: "
                "no replacement")
            return
        hold = not any(s.status in ("ready", "claimed") for s in left)
        ok, answer = slurm.submit(self.gpu_script, hold=hold, cwd=REPO_ROOT)
        if ok:
            log(f"[end] {len(left)} tasks left: replacement job {answer}" + (" (held)" if hold else ""))
        else:
            log(f"[end] {len(left)} tasks left; sbatch refused a replacement: {answer}")

    # --- The dry run and the job's loop ---------------------------------------------------

    def dry_run(self, claims: slurm.Claims) -> None:
        """Every task's state, by section, then the decisions so far; runs,
        claims and writes nothing."""
        decisions = self.decide()
        current = self.states(decisions, claims, slurm.live_jobs())
        for title, member in self.groups:
            print(title)
            for task, s in current.items():
                if member(task):
                    print(f"  {task.name:{self.width}s} {s.status:8s} {s.note}")
        print("decisions:", self.summary(decisions))

    def run_job(self, job: str, claims: slurm.Claims) -> None:
        """What a GPU job does: run ready tasks until none is left that fits,
        then submit a replacement if tasks remain."""
        log(f"job {job} on {socket.gethostname()}, {slurm.time_left() / 3600:.1f} hours left")
        waited, unclaimed = 0, set()
        while True:
            decisions = self.decide()
            self.save(decisions)
            live = slurm.live_jobs()
            current = self.states(decisions, claims, live)
            task = self.pick(current, slurm.time_left(), unclaimed)
            if task is not None:
                if not claims.claim(task.name, live):
                    unclaimed.add(task)     # another job got there first
                    continue
                log(f"[run] {task.name}")
                started = time.time()
                try:
                    why = self.run_task(task, decisions)
                    if why:
                        self.record_failure(task, why, job)
                finally:
                    claims.release(task.name)
                log(f"[{'failed' if why else 'done'}] {task.name} after "
                    f"{(time.time() - started) / 60:.0f} min" + (f": {why}" if why else ""))
                waited, unclaimed = 0, set()
                continue
            reason = self.wait(current)
            if reason and waited < self.wait_limit:
                log(f"[wait] {reason}")
                time.sleep(self.wait_poll)
                waited, unclaimed = waited + self.wait_poll, set()
                continue
            break
        decisions = self.decide()
        self.save(decisions)
        self.replace_job(self.states(decisions, claims, slurm.live_jobs()), job)
