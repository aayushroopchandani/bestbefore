"""The GPU chain's machinery (bestbefore.chain), on a made-up milestone `mx`:
states, the order of tasks, claims, the watchdog, failures, replacements,
the dry run and a job's loop. M2's own chain is tested through
scripts/m2_chain.py in test_m2_rules.py, unchanged since the machinery
moved (docs/m2b-store-and-size.md, S2)."""
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from bestbefore import chain, m2, runs, slurm


@dataclass(frozen=True)
class Job:
    """A made-up task, with the fields the machinery reads (and a stage)."""
    name: str
    run: str
    hours: float = 1.0
    score: bool = False
    stage: int = 1


TASKS = (Job("mx-a-s0", "a-s0"), Job("mx-b-s0", "b-s0", hours=3.0),
         Job("score-mx-a-s0", "a-s0", hours=0.25, score=True))


def run_dirs(name: str) -> list[Path]:
    return sorted(d for d in runs.RUNS_DIR.glob(f"*-mx-{name}") if d.name[16:] == f"mx-{name}")


@pytest.fixture
def runs_dir(monkeypatch, tmp_path) -> Path:
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    return tmp_path / "runs"


def made_up(finished=(), scored=(), waiting=None, run_task=None, **kw) -> chain.Chain:
    """mx's chain: training tasks are done once their run is `finished`,
    scoring tasks once it is `scored`; `waiting` maps a task to what it
    waits for."""
    waiting = waiting or {}

    def done(task):
        if task.run in (scored if task.score else finished):
            return Path(f"/runs/20261031-120000-mx-{task.run}")
        return None

    def waiting_for(task, decisions):
        if task.score and task.run not in finished:
            return f"waiting for mx-{task.run} to finish"
        return waiting.get(task.name)

    parts = dict(milestone="mx", title="MX", gpu_script="slurm/mx_gpu.sbatch", job_name="mx",
                 tasks=TASKS, done=done, waiting_for=waiting_for,
                 run_task=run_task or (lambda task, decisions: None), decide=lambda: {},
                 run_dirs=run_dirs,
                 groups=(("training", lambda t: not t.score), ("scoring", lambda t: t.score)),
                 summary=lambda decisions: ", ".join(decisions) or "none yet")
    return chain.Chain(**{**parts, **kw})


def by_name(current: dict) -> dict:
    return {t.name: s for t, s in current.items()}


def test_records_live_under_the_milestones_own_names(runs_dir):
    mx = made_up()
    assert (mx.claims_dir(), mx.failures_dir(), mx.decisions_path()) == (
        runs_dir / "mx-claims", runs_dir / "mx-failures", runs_dir / "mx-decisions.json")
    # M2's chain, built on this, keeps M2's own record of its decisions.
    assert made_up(milestone="m2").decisions_path() == m2.decisions_path()


def test_a_tasks_state_done_then_failed_then_waiting_then_claimed(runs_dir):
    claims = slurm.Claims(runs_dir / "mx-claims", "72914")
    blocked = {"mx-b-s0": "blocked: a rule said no"}
    current = by_name(made_up(finished={"a-s0"}, waiting=blocked).states({}, claims, set()))
    assert current["mx-a-s0"] == chain.State("done", "20261031-120000-mx-a-s0")
    assert current["mx-b-s0"] == chain.State("blocked", "blocked: a rule said no")
    assert current["score-mx-a-s0"].status == "ready"           # its run finished
    # A claim held by a live job; one held by a job Slurm no longer lists.
    assert slurm.Claims(runs_dir / "mx-claims", "72915").claim("mx-b-s0", {"72915"})
    mx = made_up()
    assert mx.state(TASKS[1], {}, claims, {"72914", "72915"}) == chain.State(
        "claimed", "claimed by job 72915")
    assert mx.state(TASKS[1], {}, claims, {"72914"}) == chain.State(
        "ready", "stale claim of job 72915: will be taken over")
    # Two failures leave a task for a person, even with its data ready.
    for why in ("exited with 1", "watchdog"):
        mx.record_failure(TASKS[0], why, "72915")
    assert mx.failures(TASKS[0]) == 2
    assert mx.state(TASKS[0], {}, claims, set()) == chain.State(
        "failed", "failed 2 times: results/runs/mx-failures/mx-a-s0.txt")
    assert not (runs_dir / "m2-failures").exists()             # never another milestone's


def test_pick_takes_scoring_first_then_the_lists_order_within_the_time_left(runs_dir):
    mx = made_up()
    ready = {t: chain.State("ready") for t in TASKS}
    assert mx.pick(ready, math.inf).name == "score-mx-a-s0"
    training = {t: s for t, s in ready.items() if not t.score}
    assert mx.pick(training, math.inf).name == "mx-a-s0"
    assert mx.pick(training, math.inf, skip={TASKS[0]}).name == "mx-b-s0"
    # Twice the estimate must fit: 1 h needs 2 h left, 3 h needs 6 h.
    assert mx.pick({TASKS[1]: chain.State("ready")}, 5.9 * 3600) is None
    assert mx.pick(training, 2.1 * 3600).name == "mx-a-s0"
    assert made_up(margin=1.0).pick({TASKS[1]: chain.State("ready")}, 3.1 * 3600).name == "mx-b-s0"
    assert mx.pick({t: chain.State("waiting") for t in TASKS}, math.inf) is None


# A stand-in for train_detector.py: its run.json, then training starts (the
# record gains "applied"), then an epoch to results.csv every 0.4 s, four
# times; or, hung, nothing after its run.json.
TRAINING = """
import json, sys, time
from datetime import datetime
from pathlib import Path
runs_dir, save_dir, hung = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == "hung"
run_dir = runs_dir / f"{datetime.now():%Y%m%d-%H%M%S}-mx-a-s0"
run_dir.mkdir(parents=True)
(run_dir / "run.json").write_text(json.dumps({"status": "running"}))
if hung:
    time.sleep(60)
time.sleep(0.4)
(run_dir / "run.json").write_text(json.dumps({"status": "running", "applied": {"save_dir": str(save_dir)}}))
save_dir.mkdir(parents=True)
for epoch in range(1, 5):
    time.sleep(0.4)
    with (save_dir / "results.csv").open("a") as f:
        f.write(("epoch,time\\n" if epoch == 1 else "") + f"{epoch},{epoch}\\n")
"""


def test_the_watchdog_stops_a_run_without_progress_and_marks_it_failed(runs_dir, tmp_path):
    mx = made_up(stall=1.0, poll=0.2)
    cmd = [sys.executable, "-c", TRAINING, str(runs_dir), str(tmp_path / "ckpt"), "hung"]
    why = mx.train(cmd, "a-s0")
    assert why.startswith("watchdog: no progress") and why.endswith("in setup")
    [attempt] = run_dirs("a-s0")
    record = json.loads((attempt / "run.json").read_text())
    assert (record["status"], record["failure"]) == ("failed", why)


def test_the_watchdog_lets_a_run_that_writes_epochs_finish(runs_dir, tmp_path):
    mx = made_up(stall=1.0, poll=0.2)                 # shorter than the run, longer than an epoch
    started = time.time()
    proc = subprocess.Popen([sys.executable, "-c", TRAINING, str(runs_dir), str(tmp_path / "ckpt"),
                             "writes"], start_new_session=True)
    assert mx.watch(proc, "a-s0", started) is None
    assert time.time() - started > mx.stall
    assert mx.progress("a-s0", started) == (True, 4)


def test_score_says_why_a_scoring_command_failed(tmp_path):
    script = tmp_path / "score_it.py"
    script.write_text("import sys, time\nif sys.argv[1] == 'slow':\n    time.sleep(30)\nsys.exit(int(sys.argv[1]))\n")
    assert chain.score([sys.executable, str(script), "0"], 60) is None
    assert chain.score([sys.executable, str(script), "3"], 60) == "score_it.py exited with 3"
    assert chain.score([sys.executable, str(script), "slow"], 120 / 60 / 60).startswith(
        "scoring took over")


def test_a_job_ending_with_tasks_left_submits_its_replacement(runs_dir, fake_slurm, monkeypatch,
                                                             capsys):
    mx = made_up()
    waiting = {t: chain.State("waiting", "data") for t in TASKS}
    mx.replace_job(waiting, "72914")                # 72915 waits too: room for one more
    assert (fake_slurm / "sbatch.args").read_text().split() == [
        "--parsable", "--hold", "slurm/mx_gpu.sbatch"]               # held: nothing ready
    mx.replace_job({**waiting, TASKS[0]: chain.State("ready")}, "72914")
    assert "--hold" not in (fake_slurm / "sbatch.args").read_text()
    monkeypatch.setenv("SBATCH_REFUSE", "1")
    mx.replace_job(waiting, "72914")
    assert "sbatch refused a replacement" in capsys.readouterr().out
    (fake_slurm / "sbatch.args").unlink()
    mx.replace_job(waiting, "70000")                # two other jobs of the milestone: none
    assert "MX jobs 72914, 72915 remain: no replacement" in capsys.readouterr().out
    mx.replace_job({t: chain.State("failed") for t in TASKS}, "72914")
    assert "nothing left that a GPU job could do" in capsys.readouterr().out
    assert not (fake_slurm / "sbatch.args").exists()


def test_the_dry_run_prints_every_section_and_the_decisions(runs_dir, capsys):
    mx = made_up(finished={"a-s0"}, width=14)
    mx.dry_run(slurm.Claims(mx.claims_dir(), None))
    assert capsys.readouterr().out == (
        "training\n"
        "  mx-a-s0        done     20261031-120000-mx-a-s0\n"
        "  mx-b-s0        ready    \n"
        "scoring\n"
        "  score-mx-a-s0  ready    \n"
        "decisions: none yet\n")
    assert not runs_dir.exists()                    # it wrote nothing


def test_a_jobs_loop_runs_ready_tasks_waits_while_told_then_ends(runs_dir, fake_slurm, monkeypatch,
                                                                capsys):
    finished, scored, ran, asked = set(), set(), [], []

    def run_task(task, decisions):
        assert (runs_dir / "mx-claims" / task.name / "job").read_text().startswith("72914 ")
        ran.append(task.name)
        if task.name == "mx-b-s0":
            return "train_detector.py exited with 1"
        (scored if task.score else finished).add(task.run)
        return None

    def wait(current):
        asked.append(1)
        return "the other job is busy"

    mx = made_up(finished=finished, scored=scored, run_task=run_task, wait=wait, wait_limit=0.025,
                 wait_poll=0.01)
    mx.run_job("72914", slurm.Claims(mx.claims_dir(), "72914"))
    # The training run first, its scoring as soon as it is ready, then the
    # failing run twice; then 3 waits of 0.01 s fit in 0.025 s.
    assert ran == ["mx-a-s0", "score-mx-a-s0", "mx-b-s0", "mx-b-s0"]
    out = capsys.readouterr().out
    assert out.count("[wait] the other job is busy") == 3 and len(asked) == 4
    assert "[failed] mx-b-s0 after 0 min: train_detector.py exited with 1" in out
    assert "nothing left that a GPU job could do" in out           # b failed twice: a person's
    assert list((runs_dir / "mx-claims").iterdir()) == []          # every claim released
    assert len((runs_dir / "mx-failures" / "mx-b-s0.txt").read_text().splitlines()) == 2
    assert json.loads((runs_dir / "mx-decisions.json").read_text()) == {}
    assert not (fake_slurm / "sbatch.args").exists()
