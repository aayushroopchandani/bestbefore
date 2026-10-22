"""M2's arms, budget, recipe and decision rules (docs/m2-synthetic-generator.md),
against the plan's own numbers and on made-up results."""
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import pytest
from PIL import Image

from bestbefore import detector, m2, runs, slurm
from bestbefore.derived import write_list
from bestbefore.yolo import split_repeats

# The list sizes the plan's arms table was worked out from.
COUNTS = {"train_real": 937, "real_rot": 937, "train_synth_clean": 11_844,
          "synth_rot": 11_844, "gen50k": 50_000, "gen10k": 10_000}

# "The arms": real repeats, photos listed, epochs, input size.
ARMS_TABLE = {
    "base": (1, 12_781, 200, 640),
    "rot": (1, 25_562, 100, 640),
    "gen": (5, 66_529, 38, 640),
    "gen-rot": (3, 79_310, 32, 640),
    "gen10k": (5, 66_529, 38, 640),
    "gen-rot-1280": (3, 79_310, 32, 1280),
}


@pytest.mark.parametrize("name", ARMS_TABLE)
def test_every_arm_matches_the_plans_table(name):
    arm = m2.ARMS[name]
    mix = m2.mix(arm, COUNTS)
    epochs = m2.schedule(mix.n_listed)["epochs"]
    assert (mix.real_repeats, mix.n_listed, epochs, arm.imgsz) == ARMS_TABLE[name]
    assert 2_525_000 <= epochs * mix.n_listed <= 2_560_000       # "2.53 to 2.56 million"
    n_real = mix.real_repeats * sum(COUNTS[n] for n in arm.real)
    assert 0.065 <= n_real / mix.n_listed <= 0.075                # about 7% real (decision 8)


def test_the_lists_spell_out_the_repeats():
    assert m2.mix(m2.ARMS["gen-rot"], COUNTS).lists == (
        "train_real*3", "real_rot*3", "train_synth_clean", "synth_rot", "gen50k")
    assert m2.mix(m2.ARMS["gen10k"], COUNTS).lists == (
        "train_real*5", "train_synth_clean", "gen10k*5")
    assert m2.mix(m2.ARMS["base"], COUNTS).lists == ("train_real", "train_synth_clean")
    assert m2.ARMS["gen10k"].names() == ("train_real", "train_synth_clean", "gen10k")


def test_split_repeats():
    assert split_repeats("train_real*5") == ("train_real", 5)
    assert split_repeats("/abs/lists/gen50k.txt") == ("/abs/lists/gen50k.txt", 1)
    for bad in ("train_real*", "*3", "x*0", "x*two"):
        with pytest.raises(ValueError):
            split_repeats(bad)


def test_schedule_keeps_m1s_shares_of_warmup_and_mosaic_off():
    assert m2.schedule(12_781) == dict(epochs=200, warmup_epochs=6.0, close_mosaic=20)
    # The pilot: gen at 10% of the budget.
    assert m2.schedule(66_529, m2.PILOT_SHARE) == dict(epochs=4, warmup_epochs=0.12,
                                                       close_mosaic=1)
    # A 3-epoch speed run. Ultralytics turns mosaic off from epoch index
    # epochs - close_mosaic, so epoch 2 (index 1) must come before it.
    speed = m2.schedule(12_781, epochs=3)
    assert speed == dict(epochs=3, warmup_epochs=0.09, close_mosaic=1)
    assert speed["epochs"] - speed["close_mosaic"] > 1


def test_expected_images_follows_ultralytics_fraction():
    assert m2.expected_images(12_781) == 12_781
    assert m2.expected_images(12_781, 0.02) == 256
    assert m2.expected_images(10, 0.01) == 1


def test_the_recipe_is_the_plans_and_replaces_every_m1_setting():
    assert m2.RECIPE == dict(batch=16, fliplr=0.0, deterministic=True, optimizer="MuSGD",
                             lr0=0.01, momentum=0.937, warmup_bias_lr=0.0, patience=0,
                             val=False)
    # detector.train starts from M1's SETTINGS; the recipe and the schedule
    # set every one of them again, so none of M1's can leak into an M2 run.
    assert set(detector.SETTINGS) <= set(m2.RECIPE) | set(m2.schedule(1000))


def ultralytics_has(train_images=66_529, **changes):
    """What detector.applied would read off a trainer."""
    args = {**m2.RECIPE, "epochs": 38, **changes}
    return {"args": args, "optimizer": args["optimizer"], "train_images": train_images}


def test_recipe_problems_names_whatever_ultralytics_changed():
    wanted = {**m2.RECIPE, "epochs": 38}
    assert m2.recipe_problems(ultralytics_has(), wanted, 66_529) == []
    # optimizer=auto would swap the optimiser and lr0 for a short run.
    auto = ultralytics_has(optimizer="AdamW", lr0=0.00125)
    assert len(m2.recipe_problems(auto, wanted, 66_529)) == 3
    assert m2.recipe_problems(ultralytics_has(momentum=0.9), wanted, 66_529) == [
        "momentum: asked 0.937, Ultralytics has 0.9"]
    # Repeats lost: the real photos would be listed once instead of five times.
    lost = 66_529 - 4 * 937
    assert m2.recipe_problems(ultralytics_has(train_images=lost), wanted, 66_529) == [
        f"training photos: expected 66529, Ultralytics has {lost}"]


def test_loader_choice_takes_the_simplest_within_10_percent():
    assert m2.loader_choice({"a": 300, "b": 400, "c": 500}) == "c"
    assert m2.loader_choice({"a": 300, "b": 460, "c": 500}) == "b"     # 460 >= 0.9 x 500
    assert m2.loader_choice({"a": 451, "b": 460, "c": 500}) == "a"
    assert m2.loader_choice({"a": 449, "b": 440, "c": 500}) == "c"


def test_epoch_seconds_reads_the_cumulative_time_column(tmp_path):
    csv = tmp_path / "results.csv"
    csv.write_text("epoch,time,train/box_loss\n1,95.5,1.2\n2,131,1.1\n")
    assert m2.epoch_seconds(csv, 2) == pytest.approx(35.5)
    assert m2.epoch_seconds(csv, 1) == pytest.approx(95.5)
    assert m2.epoch_seconds(csv, 3) is None


def test_cached_bytes_resizes_the_long_side_up_or_down():
    assert m2.cached_bytes(1280, 960, 640) == 640 * 480 * 3
    assert m2.cached_bytes(320, 200, 640) == 640 * 400 * 3


def test_ram_check_scales_the_sample_to_the_whole_list(tmp_path):
    photo = tmp_path / "a.jpg"
    Image.new("RGB", (1280, 960)).save(photo)
    listed = [photo] * 1000          # a repeated photo is cached once per listing
    # 640 x 480 x 3 bytes x 1000 = 0.86 GB, against 60% of the job's memory.
    roomy = m2.ram_check(listed, 640, mem_mb="2048")
    assert roomy["fits"] and roomy["need_gb"] == pytest.approx(0.9)
    assert not m2.ram_check(listed, 640, mem_mb="1024")["fits"]
    assert not m2.ram_check(listed, 640, mem_mb="")["fits"]      # not a Slurm job


def test_pilot_gate():
    def suite(recall):
        return {"synth-val": {"any_conf": {"expiry_recall": recall}}}

    assert m2.pilot_passes(suite(0.5)) and not m2.pilot_passes(suite(0.49))
    with pytest.raises(ValueError, match="synth-val"):
        m2.pilot_passes({"synth-val": {"missing": "synth_val not made yet"}})


def suites(real_val, hard):
    """Three seeds' suites: one mAP@0.5 for dates on real-val, one on the hard sets."""
    one = {"real-val": {"map50": {"date": real_val}},
           **{name: {"map50": {"date": hard}} for name in m2.HARD}}
    return [one] * 3


def test_decide_detector_a_tie_goes_to_the_cheapest():
    scores = {"base": suites(0.990, 0.50), "rot": suites(0.990, 0.80),
              "gen": suites(0.985, 0.85), "gen-rot": suites(0.990, 0.855),
              "gen-rot-1280": suites(0.990, 0.86)}
    # gen-rot-1280 is best on the hard sets, but gen is exactly 1 point
    # behind (a tie) and cheaper.
    assert m2.decide_detector(scores) == "gen"


def test_decide_detector_passes_over_an_ineligible_winner():
    scores = {"base": suites(0.990, 0.50), "rot": suites(0.990, 0.70),
              "gen": suites(0.975, 0.95),          # best on the hard sets, 1.5 points under base
              "gen-rot": suites(0.985, 0.80), "gen-rot-1280": suites(0.982, 0.79)}
    assert m2.decide_detector(scores) == "gen-rot"


def test_a_list_is_ready_once_its_manifest_vouches_for_it(monkeypatch, tmp_path):
    monkeypatch.setenv("GENERATED_DIR", str(tmp_path))
    monkeypatch.setenv("EXPDATE_DIR", str(tmp_path / "expdate"))
    assert "S6" in m2.list_missing("gen50k")
    m2.list_path("gen50k").parent.mkdir(parents=True)
    m2.list_path("gen50k").write_text("/g/1.jpg\n")
    assert "manifests/gen.json" in m2.list_missing("gen50k")
    m2.manifest_path("gen").parent.mkdir(parents=True)
    m2.manifest_path("gen").write_text(json.dumps({"complete": False}))
    assert m2.list_missing("gen50k") is not None
    m2.manifest_path("gen").write_text(json.dumps({"complete": True}))
    assert m2.list_missing("gen50k") is None
    # S2's manifest names its sets instead.
    m2.list_path("real_rot").write_text("/d/1.jpg\n")
    m2.manifest_path("derived").write_text(json.dumps({"sets": {"real_val_rot": {}}}))
    assert m2.list_missing("real_rot") is not None
    m2.manifest_path("derived").write_text(json.dumps({"sets": {"real_rot": {}}}))
    assert m2.list_missing("real_rot") is None
    # ExpDate's own lists only need to exist.
    assert m2.list_missing("train_real") is not None
    (tmp_path / "expdate" / "yolo").mkdir(parents=True)
    (tmp_path / "expdate" / "yolo" / "train_real.txt").write_text("./images/real-train/a.jpg\n")
    assert m2.list_missing("train_real") is None
    assert m2.suite_missing("synth-val") is not None
    assert m2.missing({"real_rot": None, "synth_rot": "not made yet (S2)",
                       "gen50k": "not made yet (S6)", "gen10k": "not made yet (S6)"}) == (
        "synth_rot not made yet (S2); gen50k, gen10k not made yet (S6)")
    assert m2.missing({"train_real": None}) is None


def test_listed_images_keep_repeats(monkeypatch, tmp_path):
    monkeypatch.setenv("GENERATED_DIR", str(tmp_path))
    monkeypatch.setenv("EXPDATE_DIR", str(tmp_path / "expdate"))
    yolo = tmp_path / "expdate" / "yolo"
    yolo.mkdir(parents=True)
    (yolo / "train_real.txt").write_text("./images/real-train/a.jpg\n./images/real-train/b.jpg\n")
    m2.list_path("gen10k").parent.mkdir(parents=True)
    m2.list_path("gen10k").write_text("/g/1.jpg\n\n")
    assert (m2.count_listed("train_real"), m2.count_listed("gen10k")) == (2, 1)
    images = m2.listed_images(["train_real*2", "gen10k*3"])
    assert len(images) == 7 and images.count(Path("/g/1.jpg")) == 3
    assert m2.list_spec("train_real*2") == "train_real*2"       # in the dataset file's folder
    assert m2.list_spec("gen10k*5") == f"{m2.list_path('gen10k')}*5"


# --- The chain: task readiness (scripts/m2_chain.py) ---------------------------

DERIVED_LISTS = ("train_synth_clean", "real_rot", "synth_rot", "real_val_rot", "real_val_small")


@pytest.fixture
def stage1(monkeypatch, tmp_path):
    """The cluster as S2 leaves it: stage 1's lists and the pretrained
    weights, nothing of S6's, no runs yet."""
    monkeypatch.setenv("EXPDATE_DIR", str(tmp_path / "expdate"))
    monkeypatch.setenv("GENERATED_DIR", str(tmp_path / "generated"))
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    for name in ("train_real", "val"):
        write_list(tmp_path / "expdate" / "yolo" / f"{name}.txt", ["./images/real-train/a.jpg"])
    for name in DERIVED_LISTS:
        write_list(m2.list_path(name), [Path("/d/a.jpg")])
    write_json_file(m2.manifest_path("derived"), {"sets": {name: {} for name in DERIVED_LISTS}})
    (tmp_path / "ckpt" / "pretrained").mkdir(parents=True)
    (tmp_path / "ckpt" / "pretrained" / "yolo11n.pt").write_bytes(b"weights")
    return tmp_path


def write_json_file(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def finish(run: str, epoch_times=(40.0,), images=12_781, suite=None) -> Path:
    """A finished run's record, its results.csv and maybe its suite.json,
    as train_detector.py and score_suite.py leave them."""
    save_dir = runs.RUNS_DIR.parent / "ckpt" / "m2" / run
    save_dir.mkdir(parents=True)
    (save_dir / "results.csv").write_text(
        "epoch,time\n" + "".join(f"{i + 1},{t}\n" for i, t in enumerate(epoch_times)))
    run_dir = runs.start_run(f"m2-{run}", {})
    runs.update_run(run_dir, status="finished",
                    applied={"save_dir": str(save_dir), "train_images": images})
    if suite is not None:
        (run_dir / "suite.json").write_text(json.dumps(suite))
    return run_dir


def by_name(chain, decisions, claims, live=frozenset()) -> dict:
    return {t.name: s for t, s in chain.states(decisions, claims, set(live)).items()}


def test_the_chain_starts_with_stage_1_and_waits_for_the_generated_data(load_script, stage1):
    chain = load_script("m2_chain")
    current = by_name(chain, chain.decide(), slurm.Claims(chain.claims_dir(), "72914"))
    assert [n for n, s in current.items() if s.status == "ready"] == [
        "m2-smoke", "m2-speed-a", "m2-speed-b", "m2-speed-c"]
    assert all(s.status == "waiting" for n, s in current.items() if n not in (
        "m2-smoke", "m2-speed-a", "m2-speed-b", "m2-speed-c"))
    assert "loader decision" in current["m2-base-s0"].note
    assert current["m2-pilot"].note == "waiting for gen50k not made yet (S6's generation)"
    assert current["score-m2-smoke"].note == "waiting for m2-smoke to finish"
    # The speed runs aren't scored; every other run is.
    assert "score-m2-speed-a" not in current and "score-m2-gen10k-s2" in current


def test_the_speed_runs_decide_the_loader_and_scoring_goes_first(load_script, stage1):
    chain = load_script("m2_chain")
    finish("smoke")
    # Epoch 2 takes 30, 25 and 24 s: c is the fastest, b within 10% of it.
    for k, times in (("a", (40, 70, 100)), ("b", (40, 65, 90)), ("c", (90, 114, 138))):
        finish(f"speed-{k}", times)
    decisions = chain.decide()
    assert decisions["loader"]["choice"] == "b"
    assert (decisions["loader"]["workers"], decisions["loader"]["cache"]) == (24, False)
    claims = slurm.Claims(chain.claims_dir(), "72914")
    current = chain.states(decisions, claims, set())
    named = {t.name: s for t, s in current.items()}
    assert named["m2-speed-a"].status == "done" and named["m2-base-s0"].status == "ready"
    assert named["score-m2-smoke"].status == "ready"        # the smoke needs only the real sets
    assert named["score-m2-base-s0"].note == "waiting for m2-base-s0 to finish"
    assert chain.pick(current, math.inf).name == "score-m2-smoke"
    training = {t: s for t, s in current.items() if not t.score}
    assert chain.pick(training, math.inf).name == "m2-base-s0"
    assert [t.name for t, s in training.items() if s.status == "ready"][:3] == [
        "m2-base-s0", "m2-rot-s0", "m2-base-s1"]                  # seed 0 of every arm first
    # A 2-hour run starts only with 4 hours left.
    assert chain.pick(training, 3.9 * 3600) is None
    assert chain.pick(current, 3.9 * 3600).name == "score-m2-smoke"
    # And the run it trains takes the loader decision's settings.
    base = next(t for t in chain.TASKS if t.name == "m2-base-s0")
    assert chain.loader_flags(decisions["loader"]) == ("--workers", "24")
    assert base.loader and not base.flags
    # Once trained, an arm's run is scored only with all four suite sets.
    finish("base-s0")
    assert by_name(chain, decisions, claims)["score-m2-base-s0"].note == (
        "waiting for synth-val not made yet (S6's generation)")


def test_claims_failures_and_stale_claims(load_script, stage1):
    chain = load_script("m2_chain")
    a = slurm.Claims(chain.claims_dir(), "72914")
    assert slurm.Claims(chain.claims_dir(), "72915").claim("m2-smoke", {"72915"})
    speed_a = next(t for t in chain.TASKS if t.name == "m2-speed-a")
    chain.record_failure(speed_a, "train_detector.py exited with 1", "72915")
    assert by_name(chain, {}, a, {"72914", "72915"})["m2-speed-a"].status == "ready"
    chain.record_failure(speed_a, "watchdog: no progress for 2 hours, in setup", "72915")
    current = by_name(chain, {}, a, {"72914", "72915"})
    assert current["m2-smoke"] == chain.State("claimed", "claimed by job 72915")
    assert current["m2-speed-a"].status == "failed"            # twice: left for a person
    # Once job 72915 has ended, its claim is stale and the smoke ready again.
    assert by_name(chain, {}, a, {"72914"})["m2-smoke"].status == "ready"


def test_the_pilot_gate_holds_back_stage_2(load_script, stage1):
    chain = load_script("m2_chain")
    for name in ("gen50k", "gen10k", "synth_val"):
        write_list(m2.list_path(name), [Path("/g/a.jpg")])
    write_json_file(m2.manifest_path("gen"), {"complete": True})
    write_json_file(m2.manifest_path("synth-val"), {"complete": True})
    claims = slurm.Claims(chain.claims_dir(), "72914")
    loader = {"loader": {"workers": 8, "cache": False}}
    assert by_name(chain, loader, claims)["m2-pilot"].status == "ready"
    assert "pilot gate" in by_name(chain, loader, claims)["m2-gen-s0"].note
    failed = {**loader, "pilot": {"passes": False}}
    assert by_name(chain, failed, claims)["m2-gen-rot-1280-s0"].status == "blocked"
    passed = {**loader, "pilot": {"passes": True}}
    assert by_name(chain, passed, claims)["m2-gen-rot-1280-s0"].status == "ready"
    # The pilot's suite makes the gate's decision.
    finish("pilot", suite={"synth-val": {"any_conf": {"expiry_recall": 0.62}}})
    assert chain.decide()["pilot"] == {"synth_val_any_conf": 0.62, "passes": True}


def test_a_job_ending_with_tasks_left_submits_its_replacement(load_script, stage1, fake_slurm,
                                                             monkeypatch, capsys):
    chain = load_script("m2_chain")
    claims = slurm.Claims(chain.claims_dir(), "72914")
    waiting = {t: chain.State("waiting", "gen50k") for t in chain.TASKS[:2]}
    chain.replace_job(waiting, "72914")       # 72915 waits too: room for one more
    assert (fake_slurm / "sbatch.args").read_text().split() == [
        "--parsable", "--hold", chain.GPU_SCRIPT]                 # held: no data ready
    chain.replace_job(chain.states({}, claims, set()), "72914")
    assert "--hold" not in (fake_slurm / "sbatch.args").read_text()   # the smoke is ready
    monkeypatch.setenv("SBATCH_REFUSE", "1")
    chain.replace_job(waiting, "72914")       # the 5-job cap: logged, no error
    assert "sbatch refused a replacement" in capsys.readouterr().out
    (fake_slurm / "sbatch.args").unlink()
    chain.replace_job(waiting, "70000")       # two other M2 jobs already: none
    chain.replace_job({t: chain.State("done") for t in chain.TASKS}, "72914")
    assert not (fake_slurm / "sbatch.args").exists()


# A stand-in for train_detector.py: starts training after a moment (its
# run.json gains "applied"), then writes an epoch to results.csv every
# 0.4 s, four times.
TRAINING = """
import json, sys, time
from datetime import datetime
from pathlib import Path
runs_dir, save_dir = Path(sys.argv[1]), Path(sys.argv[2])
run_dir = runs_dir / f"{datetime.now():%Y%m%d-%H%M%S}-m2-base-s0"
run_dir.mkdir(parents=True)
time.sleep(0.4)
(run_dir / "run.json").write_text(json.dumps({"applied": {"save_dir": str(save_dir)}}))
save_dir.mkdir(parents=True)
for epoch in range(1, 5):
    time.sleep(0.4)
    with (save_dir / "results.csv").open("a") as f:
        f.write(("epoch,time\\n" if epoch == 1 else "") + f"{epoch},{epoch}\\n")
"""


def test_the_watchdog_stops_a_run_without_progress(load_script, monkeypatch, tmp_path):
    chain = load_script("m2_chain")
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(chain, "STALL", 1.0)
    monkeypatch.setattr(chain, "POLL", 0.2)
    hung = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                            start_new_session=True)
    why = chain.watch(hung, "base-s0", time.time())
    assert why.startswith("watchdog: no progress") and why.endswith("in setup")
    assert hung.poll() is not None                   # stopped, with its process group


def test_the_watchdog_lets_a_run_that_writes_epochs_finish(load_script, monkeypatch, tmp_path):
    chain = load_script("m2_chain")
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(chain, "STALL", 1.0)          # shorter than the run, longer than an epoch
    monkeypatch.setattr(chain, "POLL", 0.2)
    started = time.time()
    proc = subprocess.Popen([sys.executable, "-c", TRAINING, str(tmp_path / "runs"),
                             str(tmp_path / "ckpt")], start_new_session=True)
    assert chain.watch(proc, "base-s0", started) is None
    assert time.time() - started > chain.STALL
    assert chain.progress("base-s0", started) == (True, 4)


def test_the_chain_loop_runs_ready_tasks_then_ends(load_script, stage1, fake_slurm, monkeypatch,
                                                  capsys):
    chain = load_script("m2_chain")
    smoke = [t for t in chain.TASKS if t.run == "smoke"]           # m2-smoke and its scoring
    monkeypatch.setattr(chain, "TASKS", smoke)
    ran = []

    def fake_run_task(task, decisions):
        assert chain.claims_dir().joinpath(task.name, "job").read_text().startswith("72914 ")
        ran.append(task.name)
        if task.score:
            (m2.finished_run("smoke") / "suite.json").write_text("{}")
        else:
            finish("smoke")

    monkeypatch.setattr(chain, "run_task", fake_run_task)
    monkeypatch.setenv("SLURM_JOB_ID", "72914")
    monkeypatch.setattr(sys, "argv", ["m2_chain.py"])
    chain.main()
    assert ran == ["m2-smoke", "score-m2-smoke"]
    assert list(chain.claims_dir().iterdir()) == []                 # every claim released
    assert "nothing left that a GPU job could do" in capsys.readouterr().out
    assert not (fake_slurm / "sbatch.args").exists()               # so no replacement
