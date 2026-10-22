"""The work queue's Slurm side: claims taken atomically, stale claims found
with a fake squeue, replacements submitted with a fake sbatch."""
import math
import os
import time

from bestbefore import slurm

LIVE = {"72914", "72915"}


def test_squeue_lists_our_jobs(fake_slurm):
    assert slurm.live_jobs() == LIVE
    assert slurm.jobs_named("m2") == {"72914": "RUNNING", "72915": "PENDING"}


def test_without_squeue_the_jobs_are_unknown(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert slurm.live_jobs() is None and slurm.jobs_named("m2") is None


def test_of_two_jobs_one_claims_a_task(tmp_path):
    a, b = slurm.Claims(tmp_path, "72914"), slurm.Claims(tmp_path, "72915")
    assert a.claim("m2-smoke", LIVE)
    assert not b.claim("m2-smoke", LIVE)
    b.release("m2-smoke")                        # not b's to release
    assert a.holder("m2-smoke")[0] == "72914"
    a.release("m2-smoke")
    assert a.holder("m2-smoke") is None
    assert b.claim("m2-smoke", LIVE)


def test_a_stale_claim_found_with_a_fake_squeue_is_taken_over(fake_slurm, tmp_path):
    assert slurm.Claims(tmp_path, "70000").claim("m2-base-s0", None)    # a job that has ended
    live = slurm.live_jobs()
    a, b = slurm.Claims(tmp_path, "72914"), slurm.Claims(tmp_path, "72915")
    assert a.is_stale("m2-base-s0", a.holder("m2-base-s0"), live)
    assert a.claim("m2-base-s0", live)
    assert a.holder("m2-base-s0")[0] == "72914"
    # Now a live job holds it: the other job can't take it.
    assert not b.is_stale("m2-base-s0", b.holder("m2-base-s0"), live)
    assert not b.claim("m2-base-s0", live)
    # Without squeue nothing is known to have ended, so nothing is stale.
    assert not b.is_stale("m2-base-s0", b.holder("m2-base-s0"), None)


def test_of_two_jobs_taking_over_one_stale_claim_only_one_does(tmp_path):
    slurm.Claims(tmp_path, "70000").claim("m2-rot-s0", None)
    a, b = slurm.Claims(tmp_path, "72914"), slurm.Claims(tmp_path, "72915")
    stale = b.holder("m2-rot-s0")                 # b reads the stale claim ...
    assert a.claim("m2-rot-s0", LIVE)              # ... a takes it over first ...
    b.holder = lambda task: stale                  # ... while b still acts on what it read
    assert not b.claim("m2-rot-s0", LIVE)          # a's takeover marker stops it
    assert a.holder("m2-rot-s0")[0] == "72914"     # and a's claim is untouched


def test_after_a_requeue_a_job_takes_back_its_own_claim(tmp_path):
    assert slurm.Claims(tmp_path, "72914").claim("m2-gen-s0", LIVE)
    again = slurm.Claims(tmp_path, "72914")        # the same job, restarted by Slurm
    assert again.claim("m2-gen-s0", LIVE)


def test_a_half_made_claim_is_abandoned_after_ten_minutes(tmp_path):
    (tmp_path / "m2-speed-a").mkdir()              # its maker died before writing the job file
    claims = slurm.Claims(tmp_path, "72914")
    assert claims.holder("m2-speed-a")[0] == "?"
    assert not claims.claim("m2-speed-a", LIVE)
    old = time.time() - slurm.UNFINISHED_CLAIM_AGE - 1
    os.utime(tmp_path / "m2-speed-a", (old, old))
    assert claims.claim("m2-speed-a", LIVE)


def test_time_left(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    assert slurm.time_left() == math.inf
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(int(time.time()) + 3600))
    assert 3590 < slurm.time_left() <= 3600


def test_submit_held_and_refused(fake_slurm, monkeypatch):
    assert slurm.submit("slurm/m2_gpu.sbatch", hold=True) == (True, "72999")
    assert (fake_slurm / "sbatch.args").read_text().split() == [
        "--parsable", "--hold", "slurm/m2_gpu.sbatch"]
    monkeypatch.setenv("SBATCH_REFUSE", "1")
    ok, answer = slurm.submit("slurm/m2_gpu.sbatch", hold=False)
    assert not ok and "QOSMaxSubmitJobPerUserLimit" in answer
