"""M2's arms, budget, recipe and decision rules (docs/m2-synthetic-generator.md),
against the plan's own numbers and on made-up results."""
import json
from pathlib import Path

import pytest
from PIL import Image

from bestbefore import detector, m2
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
