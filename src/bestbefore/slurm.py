"""Slurm, for M2's GPU work queue (scripts/m2_chain.py): which of our jobs
Slurm still lists, how long this job has left, submitting a replacement,
and claiming tasks so that two jobs never run the same one. Also releasing
held jobs, once the generated data they wait for is checked (S6)."""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

UNFINISHED_CLAIM_AGE = 600     # seconds: a claim or takeover left half-made is abandoned


def squeue(*args: str, tries: int = 3, pause: float = 20) -> list[str] | None:
    """squeue's lines for our own jobs, or None where there is no squeue
    (the laptop) or it keeps failing (Slurm's controller busy)."""
    for attempt in range(tries):
        try:
            out = subprocess.run(["squeue", "--me", "--noheader", *args],
                                 capture_output=True, text=True, timeout=60)
        except FileNotFoundError:
            return None
        except subprocess.TimeoutExpired:
            out = None
        if out is not None and out.returncode == 0:
            return [line.strip() for line in out.stdout.splitlines() if line.strip()]
        if attempt + 1 < tries:
            time.sleep(pause)
    return None


def live_jobs() -> set[str] | None:
    """Ids of our jobs Slurm still lists, waiting or running."""
    lines = squeue("--format=%i")
    return None if lines is None else set(lines)


def jobs_named(name: str) -> dict[str, str] | None:
    """Our jobs with this name: id -> state (PENDING, RUNNING, ...)."""
    lines = squeue(f"--name={name}", "--format=%i %T")
    return None if lines is None else dict(line.split(maxsplit=1) for line in lines)


def held(name: str) -> list[str] | None:
    """Ids of our jobs with this name that wait held by us (reason
    JobHeldUser), oldest first; None where squeue can't be asked."""
    lines = squeue(f"--name={name}", "--states=PENDING", "--format=%i %r")
    if lines is None:
        return None
    jobs = [line.split(maxsplit=1) for line in lines]
    return sorted((job for job, reason in jobs if reason == "JobHeldUser"), key=int)


def release(job: str) -> tuple[bool, str]:
    """scontrol release one job: whether Slurm took it, and what it said."""
    try:
        out = subprocess.run(["scontrol", "release", job], capture_output=True, text=True, timeout=120)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    return out.returncode == 0, (out.stdout + out.stderr).strip()


def time_left() -> float:
    """Seconds until this job's time limit (SLURM_JOB_END_TIME); infinite
    outside a Slurm job."""
    end = os.environ.get("SLURM_JOB_END_TIME")
    return float(end) - time.time() if end else math.inf


def submit(script: str, hold: bool, cwd: Path | None = None) -> tuple[bool, str]:
    """sbatch a job script, held if asked (it can't start until released).
    Returns whether Slurm took it, and the job id or Slurm's refusal."""
    cmd = ["sbatch", "--parsable", *(["--hold"] if hold else []), script]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=cwd)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    return out.returncode == 0, (out.stdout if out.returncode == 0 else out.stderr).strip()


class Claims:
    """Claims on tasks: a folder `<root>/<task>/` holding the claiming job's
    id and a token. os.mkdir is atomic, so of two jobs claiming the same
    task at once exactly one succeeds.

    A claim is stale when Slurm no longer lists its job, or when it holds
    this job's own id: then it dates from before Slurm requeued this job
    (after a node failure). To take a stale claim over, a job first makes
    `<task>.takeover-<token>`, again with os.mkdir, so of two jobs taking
    over the same claim only one gets through. A marker older than
    UNFINISHED_CLAIM_AGE whose claim is still there was left by a job that
    died halfway through the takeover, and no longer blocks it.
    """

    def __init__(self, root: Path, job: str | None):
        self.root, self.job = root, job
        self.tokens: dict[str, str] = {}

    def holder(self, task: str) -> tuple[str, str] | None:
        """(job id, token) of the task's claim; None if it isn't claimed.
        A claim whose job file isn't written yet holds job "?"."""
        folder = self.root / task
        try:
            job, token = (folder / "job").read_text().split()
            return job, token
        except (FileNotFoundError, ValueError):     # no job file yet, or half written
            try:
                return "?", f"unfinished-{folder.stat().st_ino}"
            except FileNotFoundError:
                return None

    def is_stale(self, task: str, holder: tuple[str, str], live: set[str] | None) -> bool:
        """Whether a claim can be taken over. `live`: the job ids squeue
        lists, or None if unknown (then only this job's own claims are)."""
        job = holder[0]
        if job == "?":                  # its maker died between mkdir and writing the job file
            try:
                return time.time() - (self.root / task).stat().st_mtime > UNFINISHED_CLAIM_AGE
            except FileNotFoundError:
                return False
        return job == self.job or (live is not None and job not in live)

    def claim(self, task: str, live: set[str] | None) -> bool:
        """Claim a task for this job. False if a live job holds it, or another
        job got there first."""
        self.root.mkdir(parents=True, exist_ok=True)
        folder = self.root / task
        held = self.holder(task)
        if held is not None:
            if not self.is_stale(task, held, live):
                return False
            marker = self.root / f"{task}.takeover-{held[1]}"
            try:                        # only one job may remove this stale claim
                marker.mkdir()
            except FileExistsError:
                if time.time() - marker.stat().st_mtime < UNFINISHED_CLAIM_AGE:
                    return False
            shutil.rmtree(folder, ignore_errors=True)
        try:
            folder.mkdir()
        except FileExistsError:
            return False
        token = uuid.uuid4().hex
        (folder / "job").write_text(f"{self.job} {token}\n")
        self.tokens[task] = token
        return True

    def release(self, task: str) -> None:
        """Remove this job's claim on a task; never another job's."""
        held = self.holder(task)
        if held is not None and held[1] == self.tokens.pop(task, None):
            shutil.rmtree(self.root / task, ignore_errors=True)
