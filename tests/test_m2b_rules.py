"""M2b's step 1 and its control (docs/m2b-store-and-size.md, S2): step 1 is
M2's gen-rot-1280 with other weights, and the control is M2's runs as they
are. Against M2's own code and records, on made-up lists and runs."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from bestbefore import m2, m2b, runs, sets, slurm, store
from bestbefore.derived import write_list

# The list sizes M2's arms table was worked out from (test_m2_rules.py).
COUNTS = {"train_real": 937, "real_rot": 937, "train_synth_clean": 11_844,
          "synth_rot": 11_844, "gen50k": 50_000}


def test_step_1_is_gen_rot_1280_with_another_sizes_weights():
    assert m2b.SIZES == {"n": "yolo11n", "s": "yolo11s", "m": "yolo11m"}
    assert {name: (s.size, s.weights) for name, s in m2b.STEP1.items()} == {
        "s1280": ("s", "yolo11s"), "m1280": ("m", "yolo11m")}
    gen_rot_1280 = m2.ARMS["gen-rot-1280"]
    for step in m2b.STEP1.values():
        assert step.arm is gen_rot_1280                          # the same lists and input size
        mix = m2.mix(step.arm, COUNTS)
        assert mix == m2.mix(gen_rot_1280, COUNTS)
        # "The arms": 3 real repeats, 79,310 photos listed, 32 epochs, 1,280 px.
        assert (mix.real_repeats, mix.n_listed, m2.schedule(mix.n_listed)["epochs"],
                step.arm.imgsz) == (3, 79_310, 32, 1280)
    assert m2b.SEEDS == m2.SEEDS == (0, 1, 2)
    assert m2b.LOADER == {"workers": 8, "cache": False}          # M2's loader decision, a
    assert m2b.CONTROL not in m2b.STEP1                          # n1280 is never trained again


# --- train_detector.py --milestone m2b ------------------------------------------------

@pytest.fixture
def lists(monkeypatch, tmp_path):
    """gen-rot-1280's five lists as the cluster holds them, vouched for by
    their manifests, and the three COCO models, under tmp_path."""
    monkeypatch.setenv("EXPDATE_DIR", str(tmp_path / "expdate"))
    monkeypatch.setenv("GENERATED_DIR", str(tmp_path / "generated"))
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    yolo = tmp_path / "expdate" / "yolo"
    yolo.mkdir(parents=True)
    for name in ("val", "test"):
        (yolo / f"{name}.txt").write_text("./images/real-train/v.jpg\n")
    (yolo / "train_real.txt").write_text("./images/real-train/a.jpg\n" * 2)
    sizes = {"real_rot": 2, "train_synth_clean": 20, "synth_rot": 20, "gen50k": 30}
    for name, n in sizes.items():
        write_list(m2.list_path(name), [Path(f"/{name}/{i}.jpg") for i in range(n)])
    m2.manifest_path("derived").parent.mkdir(parents=True)
    m2.manifest_path("derived").write_text(json.dumps(
        {"sets": {n: {} for n in ("real_rot", "train_synth_clean", "synth_rot")}}))
    m2.manifest_path("gen").write_text(json.dumps({"complete": True}))
    (tmp_path / "ckpt" / "pretrained").mkdir(parents=True)
    for size, weights in m2b.SIZES.items():
        (tmp_path / "ckpt" / "pretrained" / f"{weights}.pt").write_bytes(f"COCO {size}".encode())
    return tmp_path


class FakeDataset(list):
    cache = None

    @property
    def im_files(self):
        return list(self)


def train(script, monkeypatch, argv: list[str]) -> tuple[dict, dict, str]:
    """Run train_detector.py with a stand-in for Ultralytics; what it was
    asked to train, its run record and its dataset file."""
    asked = {}

    def fake_train(weights, data, imgsz, seed, project, name, keep, callbacks, **settings):
        asked.update(weights=weights, imgsz=imgsz, seed=seed, project=project, keep=keep,
                     settings=settings)
        n = m2.expected_images(json.loads((data.parent / "run.json").read_text())["config"]["n_listed"],
                               settings["fraction"])

        class MuSGD:
            param_groups = [{"momentum": settings["momentum"]}]
        trainer = SimpleNamespace(
            args=SimpleNamespace(**{**settings, "imgsz": imgsz, "seed": seed, "warmup_momentum": 0.8}),
            optimizer=MuSGD(), save_dir=str(project / name),
            train_loader=SimpleNamespace(dataset=FakeDataset(["x.jpg"] * n), num_workers=8))
        callbacks["on_train_start"](trainer)
        callbacks["on_train_end"](trainer)
        return project / name / "weights" / f"{keep}.pt"

    monkeypatch.setattr(script.detector, "train", fake_train)
    monkeypatch.setattr(script, "device_name", lambda device: "cpu")
    before = set(runs.RUNS_DIR.iterdir()) if runs.RUNS_DIR.exists() else set()
    script.main(argv)
    [run_dir] = set(runs.RUNS_DIR.iterdir()) - before
    return asked, json.loads((run_dir / "run.json").read_text()), (run_dir / "data.yaml").read_text()


def test_s1280_and_m1280_train_what_gen_rot_1280_trained_but_the_weights(load_script, monkeypatch,
                                                                         lists):
    script = load_script("train_detector")
    common = ["--seed", "1", "--device", "cpu", "--workers", "8"]
    m2_asked, m2_record, m2_data = train(script, monkeypatch, [
        "--milestone", "m2", "--arm", "gen-rot-1280", "--name", "gen-rot-1280-s1", *common])
    for arm, size in (("s1280", "s"), ("m1280", "m")):
        asked, record, data = train(script, monkeypatch, [
            "--milestone", "m2b", "--arm", arm, "--name", f"{arm}-s1", *common])
        weights = f"yolo11{size}"
        # Ultralytics is asked for the same training, from other weights, into M2b's folder.
        assert asked["weights"] == lists / "ckpt" / "pretrained" / f"{weights}.pt"
        assert m2_asked["weights"] == lists / "ckpt" / "pretrained" / "yolo11n.pt"
        assert asked["settings"] == m2_asked["settings"]        # recipe, schedule, fraction, loader
        assert (asked["imgsz"], asked["seed"], asked["keep"]) == (1280, 1, "last")
        assert (asked["imgsz"], asked["seed"], asked["keep"]) == (
            m2_asked["imgsz"], m2_asked["seed"], m2_asked["keep"])
        assert asked["project"] == lists / "ckpt" / "m2b"
        assert data == m2_data                                   # the same lists, repeats and val
        # The record: M2's, but the milestone, arm and weights, plus three fields.
        config, m2_config = record["config"], m2_record["config"]
        added = {"size": size, "lists_from": "gen-rot-1280",
                 "weights_sha256": m2b.sha256(lists / "ckpt" / "pretrained" / f"{weights}.pt")}
        assert config == {**m2_config, "milestone": "m2b", "arm": arm, "weights": weights,
                          "name": f"{arm}-s1", **added}
        assert config["recipe"] == m2.RECIPE
        assert record["name"] == f"m2b-{arm}-s1" and record["status"] == "finished"
        assert record["applied"]["args"]["epochs"] == m2_record["applied"]["args"]["epochs"]
    assert m2_record["name"] == "m2-gen-rot-1280-s1" and "size" not in m2_record["config"]


def test_the_laptop_smoke_trains_on_bases_lists_at_1280(load_script, monkeypatch, lists):
    script = load_script("train_detector")
    parse = script.parse_args
    with pytest.raises(SystemExit):                              # only a smoke run
        parse(["--milestone", "m2b", "--arm", "s1280", "--name", "x", "--smoke-lists", "base"])
    with pytest.raises(SystemExit):                              # only m2b
        parse(["--milestone", "m2", "--arm", "base", "--name", "x", "--smoke-lists", "base",
               "--fraction", "0.01"])
    asked, record, data = train(script, monkeypatch, [
        "--milestone", "m2b", "--arm", "s1280", "--smoke-lists", "base", "--epochs", "1",
        "--fraction", "0.01", "--device", "cpu", "--name", "smoke-laptop"])
    config = record["config"]
    assert config["lists"] == ["train_real", "train_synth_clean"] and config["imgsz"] == 1280
    assert (config["lists_from"], config["weights"], config["train_images"]) == ("base", "yolo11s", 1)
    assert set(config["manifests"]) == {"derived"}
    assert asked["imgsz"] == 1280 and asked["settings"]["epochs"] == 1
    assert "train: [train_real.txt" in data and "real_rot" not in data


def test_m2b_trains_only_its_own_arms(load_script):
    parse = load_script("train_detector").parse_args
    for argv in (["--milestone", "m2b", "--arm", "n1280"],        # M2's runs: the control
                 ["--milestone", "m2b", "--arm", "gen-rot-1280"],  # an M2 arm
                 ["--milestone", "m2", "--arm", "s1280"],          # a step-1 arm, as M2
                 ["--milestone", "m2", "--arm", "n1280-ft"],       # a step-2 arm, as M2
                 ["--milestone", "m2b"],
                 ["--milestone", "m2b", "--arm", "s1280", "--weights", "yolo11m"]):
        with pytest.raises(SystemExit):
            parse([*argv, "--name", "x"])
    args = parse(["--milestone", "m2b", "--arm", "m1280", "--name", "m1280-s0"])
    assert (args.arm, args.smoke_lists, args.batch) == ("m1280", None, None)


def test_step_2s_own_flags(load_script):
    parse = load_script("train_detector").parse_args
    step2 = ["--milestone", "m2b", "--arm", "n1280-ft2", "--name", "x"]
    for more in (["--epoch-photos", "400"],                                  # only with --epochs
                 ["--smoke-lists", "base", "--fraction", "0.01"],            # step 1's smoke only
                 ["--cache", "ram"],                                         # no RAM cache
                 ["--budget-share", "0.1"]):                                 # the budget is BUDGET_FT
        with pytest.raises(SystemExit):
            parse([*step2, *more])
    for argv in (["--milestone", "m2b", "--arm", "s1280", "--name", "x", "--epochs", "1",
                  "--epoch-photos", "400"],                                  # a step-1 arm
                 ["--milestone", "m2", "--arm", "base", "--name", "x", "--epochs", "1",
                  "--epoch-photos", "400"]):
        with pytest.raises(SystemExit):
            parse(argv)
    args = parse([*step2, "--epochs", "1", "--epoch-photos", "400"])
    assert (args.arm, args.epochs, args.epoch_photos) == ("n1280-ft2", 1, 400)


# --- The control (decision 12) -----------------------------------------------------------

@pytest.fixture
def m2_runs(monkeypatch, tmp_path):
    """M2's gen-rot-1280-s0 as the laptop holds it: its record from the
    cluster, naming the cluster's last.pt, and the laptop's copy."""
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    run_dir = tmp_path / "runs" / "20261025-003853-m2-gen-rot-1280-s0"
    run_dir.mkdir(parents=True)
    cluster = f"/home/someone/bestbefore-data/checkpoints/m2/{run_dir.name}/weights/last.pt"
    (run_dir / "run.json").write_text(json.dumps(
        {"status": "finished", "checkpoint": cluster,
         "config": {"arm": "gen-rot-1280", "weights": "yolo11n", "imgsz": 1280}}))
    (run_dir / "suite.json").write_text(json.dumps({"threshold": 0.342413991689682}))
    copy = tmp_path / "ckpt" / "m2" / run_dir.name / "weights" / "last.pt"
    copy.parent.mkdir(parents=True)
    copy.write_bytes(b"M2's trained weights")
    return run_dir, copy


def snapshot(*folders: Path) -> dict:
    return {p: (p.read_bytes(), p.stat().st_mtime_ns) for f in folders for p in f.rglob("*")
            if p.is_file()}


def test_the_control_is_m2s_last_pt_as_it_is(m2_runs, tmp_path):
    run_dir, copy = m2_runs
    before = snapshot(run_dir, tmp_path / "ckpt")
    control = m2b.write_control(0)
    record = json.loads((control / "run.json").read_text())
    assert control.name.endswith("-m2b-n1280-s0") and m2b.finished_run("n1280-s0") == control
    assert record["status"] == "finished" and record["weights_from"] == run_dir.name
    assert record["checkpoint"] == str(copy)                    # the cluster's path isn't here
    assert record["checkpoint_sha256"] == m2b.sha256(copy)
    assert record["config"] == {"milestone": "m2b", "arm": "n1280", "size": "n",
                                "weights": "yolo11n", "imgsz": 1280, "seed": 0,
                                "name": "n1280-s0", "trained": False,
                                "weights_from": run_dir.name}
    # Nothing of M2's changed, and no weights were copied or written anywhere.
    assert snapshot(run_dir, tmp_path / "ckpt") == before
    assert [p.name for p in control.iterdir()] == ["run.json"]
    # M2's runs stay M2's: its own lookups don't see the control.
    assert m2.run_dirs("n1280-s0") == [] and m2.finished_run("gen-rot-1280-s0") == run_dir
    # A seed whose M2 run isn't there has no control.
    assert m2b.control_missing(1) == "M2's gen-rot-1280-s1 to finish"
    with pytest.raises(FileNotFoundError, match="gen-rot-1280-s1"):
        m2b.write_control(1)
    copy.unlink()
    assert m2b.control_missing(0) == "M2's gen-rot-1280-s0 last.pt on this machine"


def test_m2b_runs_are_found_by_their_whole_name(monkeypatch, tmp_path):
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path)
    for name in ("20261101-010101-m2b-s1280-s0", "20261101-020202-m2b-m1280-s0",
                 "20261101-030303-m2-s1280-s0", "20261101-040404-m2b-xs1280-s0"):
        (tmp_path / name).mkdir()
    (tmp_path / "20261101-010101-m2b-s1280-s0" / "run.json").write_text('{"status": "finished"}')
    assert [d.name for d in m2b.run_dirs("s1280-s0")] == ["20261101-010101-m2b-s1280-s0"]
    assert m2b.finished_run("s1280-s0").name == "20261101-010101-m2b-s1280-s0"
    assert m2b.finished_run("m1280-s0") is None                 # no record says it finished
    assert m2b.decisions_path() == tmp_path / "m2b-decisions.json"


# --- The chain (scripts/m2b_chain.py) ---------------------------------------------------

M2_SUITE_LISTS = ("real_val_rot", "real_val_small", "synth_val")


@pytest.fixture
def cluster(lists, monkeypatch):
    """The cluster as S2 leaves it: gen-rot-1280's lists, M2's suite sets,
    the COCO models, M2's three gen-rot-1280 runs with their last.pt; no
    M2b run, nothing of stage 2's."""
    root = lists
    for name in M2_SUITE_LISTS:
        write_list(m2.list_path(name), [Path(f"/{name}/0.jpg")])
    derived = json.loads(m2.manifest_path("derived").read_text())
    derived["sets"].update({n: {} for n in ("real_val_rot", "real_val_small")})
    m2.manifest_path("derived").write_text(json.dumps(derived))
    m2.manifest_path("synth-val").write_text(json.dumps({"complete": True}))
    for s in m2b.SEEDS:
        run_dir = root / "runs" / f"2026102{5 + s}-003853-m2-gen-rot-1280-s{s}"
        last = root / "ckpt" / "m2" / run_dir.name / "weights" / "last.pt"
        last.parent.mkdir(parents=True)
        last.write_bytes(f"M2 seed {s}".encode())
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(json.dumps(
            {"status": "finished", "checkpoint": str(last),
             "config": {"arm": "gen-rot-1280", "weights": "yolo11n", "imgsz": 1280}}))
    monkeypatch.delenv("STORE_TRAIN_DIR", raising=False)
    return root


def states(script, live=frozenset()) -> dict:
    claims = slurm.Claims(script.CHAIN.claims_dir(), "73831")
    return {t.name: s for t, s in script.CHAIN.states({}, claims, set(live)).items()}


STAGE_1 = ["m2b-smoke", "m2b-n1280-s0", "m2b-n1280-s1", "m2b-n1280-s2",
           "m2b-s1280-s0", "m2b-s1280-s1", "m2b-s1280-s2", "m2b-m1280-s0"]


def test_the_chain_starts_with_stage_1_and_stage_2_waits_for_its_data(load_script, cluster, monkeypatch):
    script = load_script("m2b_chain")
    monkeypatch.setattr(script.store, "TEST_SET_V2_SHA256", None)    # as before M0 D6
    current = states(script)
    assert [n for n, s in current.items() if s.status == "ready"] == STAGE_1
    stage2 = {t.name: current[t.name] for t in script.TASKS if t.stage == 2 and not t.score}
    assert len(stage2) == 17 and all(s.status == "waiting" for s in stage2.values())
    for name, s in stage2.items():
        if name.startswith("m2b-m1280"):                        # m's seeds and step 2: R1 first
            assert s.note.startswith("waiting for R1"), name
        elif name.startswith("m2b-s1280"):                       # step 2 on s: R2 first
            assert s.note.startswith("waiting for R2") and "<recipe>" in name, name
        else:
            for needed in ("store set v2's labels frozen (M0 D6)", "store set v1's export",
                           "real_rot180, synth_rot180 not made yet"):
                assert needed in s.note, (name, needed)
            assert ("gen_v2" in s.note) == ("gen50k" not in name)      # the side arm keeps M2's gen50k
    assert current["score-m2b-smoke"].note == "waiting for m2b-smoke to finish"
    assert sum(t.score and t.suite == "m2" for t in script.TASKS) == 7   # smoke, s x 3, m x 3
    assert not any(t.score and t.suite == "m2" and t.arm == m2b.CONTROL for t in script.TASKS)
    # Every run but the smoke is scored on the M2b suite, the control's seeds first.
    m2b_suite = [t for t in script.TASKS if t.score and t.suite == "m2b"]
    assert len(m2b_suite) == 24 and [t.name for t in m2b_suite[:3]] == [
        "suite-m2b-n1280-s0", "suite-m2b-n1280-s1", "suite-m2b-n1280-s2"]


def test_step_2_on_n_waits_for_each_of_its_data(load_script, step2, monkeypatch):
    script = load_script("m2b_chain")
    root, _ = step2
    n_tasks = [t for t in script.TASKS if t.stage == 2 and t.arm in STEP2_N and not t.score]
    assert len(n_tasks) == 9
    assert all(states(script)[t.name].status == "ready" for t in n_tasks)       # S10's dry run
    # Each piece of data, missing alone: v2 frozen, the store export on the cluster,
    # gen-v2 (not the side arm's), the 180° sets.
    frozen = script.store.TEST_SET_V2_SHA256
    monkeypatch.setattr(script.store, "TEST_SET_V2_SHA256", None)
    assert all("store set v2's labels frozen (M0 D6)" in states(script)[t.name].note for t in n_tasks)
    monkeypatch.setattr(script.store, "TEST_SET_V2_SHA256", frozen)
    monkeypatch.setenv("STORE_TRAIN_DIR", str(root / "nowhere"))
    assert all("store set v1's export ($STORE_TRAIN_DIR/manifest.json; S6)" in states(script)[t.name].note
               for t in n_tasks)
    monkeypatch.setenv("STORE_TRAIN_DIR", str(root / "store"))
    m2b.manifest_path("gen-v2").write_text(json.dumps({"complete": False}))
    current = states(script)
    for t in n_tasks:
        assert (current[t.name].status == "ready") == t.arm.endswith("-gen50k"), t.name
        if not t.arm.endswith("-gen50k"):
            assert current[t.name].note == ("waiting for gen_v2 not yet vouched for by "
                                            "m2b/manifests/gen-v2.json (S9's generation)")
    m2b.manifest_path("gen-v2").write_text(json.dumps({"complete": True}))
    m2b.list_path("real_rot180").unlink()
    assert all(states(script)[t.name].note == "waiting for real_rot180 not made yet "
               "(S9's derive_sets.py --only rot180)" for t in n_tasks)


STEP2_N = ("n1280-ft", "n1280-ft2", "n1280-ft-gen50k")


def test_what_stage_1_waits_for(load_script, cluster):
    script = load_script("m2b_chain")
    (cluster / "ckpt" / "pretrained" / "yolo11m.pt").unlink()
    m2b.m2_weights(*(lambda d: (d, json.loads((d / "run.json").read_text())))(
        m2.finished_run("gen-rot-1280-s2"))).unlink()
    current = states(script)
    assert current["m2b-m1280-s0"].note == (
        "waiting for the pretrained weights, yolo11m.pt (scripts/fetch_pretrained.py)")
    assert current["m2b-n1280-s2"].note == "waiting for M2's gen-rot-1280-s2 last.pt on this machine"
    assert current["m2b-s1280-s0"].status == current["m2b-n1280-s0"].status == "ready"
    m2.list_path("gen50k").unlink()
    assert "gen50k not made yet" in states(script)["m2b-s1280-s0"].note


def test_the_smoke_first_then_the_controls_s_and_m_and_scoring_as_soon_as_it_can(
        load_script, cluster):
    script = load_script("m2b_chain")
    ready = {t: s for t, s in script.CHAIN.states({}, slurm.Claims(script.CHAIN.claims_dir(), "73831"),
                                                   set()).items() if s.status == "ready"}
    order = []
    while (task := script.CHAIN.pick(ready, 72 * 3600)) is not None:
        order.append(task.name)
        del ready[task]
    assert order == STAGE_1                                      # the longest, m, last
    # Twice its estimate must fit: with 9 hours left an s run starts, m doesn't.
    left = {t: s for t, s in script.CHAIN.states({}, slurm.Claims(script.CHAIN.claims_dir(), "73831"),
                                                  set()).items() if t.name in ("m2b-m1280-s0",)}
    assert script.CHAIN.pick(left, 9 * 3600) is None and script.CHAIN.pick(left, 14.1 * 3600)
    # Once the smoke has finished, its scoring goes first.
    smoke = runs.start_run("m2b-smoke", {"imgsz": 1280})
    runs.update_run(smoke, status="finished", checkpoint="/ckpt/m2b/x/weights/last.pt")
    current = script.CHAIN.states({}, slurm.Claims(script.CHAIN.claims_dir(), "73831"), set())
    assert script.CHAIN.pick(current, 72 * 3600).name == "score-m2b-smoke"


def test_each_kind_of_task_runs_as_planned(load_script, cluster, monkeypatch):
    script = load_script("m2b_chain")
    asked = []
    monkeypatch.setattr(script.chain.Chain, "train", lambda self, cmd, run: asked.append(cmd[1:]))
    monkeypatch.setattr(script.chain, "score", lambda cmd, timeout: asked.append(cmd[1:]))
    named = {t.name: t for t in script.TASKS}
    for name in ("m2b-smoke", "m2b-s1280-s1", "m2b-m1280-s0", "score-m2b-s1280-s1", "m2b-n1280-ft2-s1",
                 "suite-m2b-n1280-s0", "m2b-m1280-s2"):
        script.run_task(named[name], {})
    assert asked[4:] == [
        ["scripts/train_detector.py", "--milestone", "m2b", "--arm", "n1280-ft2", "--seed", "1",
         "--name", "n1280-ft2-s1", "--workers", "8"],
        ["scripts/score_suite.py", "--milestone", "m2b", "--run", "n1280-s0", "--suite", "m2b"],
        ["scripts/train_detector.py", "--milestone", "m2b", "--arm", "m1280", "--seed", "2",
         "--name", "m1280-s2", "--workers", "8"]]
    del asked[4:]
    assert asked == [
        ["scripts/train_detector.py", "--milestone", "m2b", "--arm", "s1280", "--seed", "0",
         "--name", "smoke", "--epochs", "1", "--fraction", "0.02", "--workers", "8"],
        ["scripts/train_detector.py", "--milestone", "m2b", "--arm", "s1280", "--seed", "1",
         "--name", "s1280-s1", "--workers", "8"],
        ["scripts/train_detector.py", "--milestone", "m2b", "--arm", "m1280", "--seed", "0",
         "--name", "m1280-s0", "--workers", "8"],
        ["scripts/score_suite.py", "--milestone", "m2b", "--run", "s1280-s1"]]
    # A control writes its record and is done; M2's run is only read.
    m2_run = m2.finished_run("gen-rot-1280-s1")
    before = snapshot(m2_run, cluster / "ckpt")
    assert script.run_task(named["m2b-n1280-s1"], {}) is None
    assert states(script)["m2b-n1280-s1"].status == "done"
    assert snapshot(m2_run, cluster / "ckpt") == before
    # One that can't be written is the task's failure, not the job's end.
    seed_7 = script.Task("m2b-n1280-s7", "n1280-s7", "n1280", 1, 0.01, seed=7, control=True)
    why = script.run_task(seed_7, {})
    assert why.startswith("the control's record failed") and "gen-rot-1280-s7 to finish" in why


def test_m2bs_chain_keeps_its_own_records_and_writes_nothing_in_a_dry_run(load_script, cluster,
                                                                         capsys):
    script = load_script("m2b_chain")
    chain_ = script.CHAIN
    assert (chain_.claims_dir(), chain_.failures_dir(), chain_.decisions_path()) == (
        runs.RUNS_DIR / "m2b-claims", runs.RUNS_DIR / "m2b-failures", m2b.decisions_path())
    before = sorted(p for p in cluster.rglob("*"))
    chain_.dry_run(slurm.Claims(chain_.claims_dir(), None))
    assert sorted(p for p in cluster.rglob("*")) == before
    out = capsys.readouterr().out
    assert out.startswith(f"stage 1: public data\n  {'m2b-smoke':30s} ready")
    assert out.endswith("decisions: none yet\n")


def test_the_job_script_runs_this_chain(load_script):
    repo = Path(__file__).resolve().parents[1]
    sbatch = (repo / "slurm" / "m2b_gpu.sbatch").read_text()
    chain_sh = (repo / "slurm" / "m2b_chain.sh").read_text()
    script = load_script("m2b_chain")
    assert script.GPU_SCRIPT == "slurm/m2b_gpu.sbatch"
    assert f"#SBATCH --job-name={script.JOB_NAME}\n" in sbatch
    for line in ("#SBATCH --partition=gpu\n", "#SBATCH --gres=gpu:1\n", "#SBATCH --mem=96G\n",
                 "#SBATCH --time=3-00:00:00\n", "exec bash slurm/m2b_chain.sh\n"):
        assert line in sbatch
    assert chain_sh.rstrip().endswith("python scripts/m2b_chain.py")
    assert "export YOLO_OFFLINE=1" in chain_sh and "module load Python/3.11.11" in chain_sh


# --- The store lines (decision 7, S6) ------------------------------------------------

def made_up_pool(photos_per_design: dict[str, int]) -> tuple[list[Path], dict[str, str]]:
    """A training list as the export writes it: each photo in its 4 turns."""
    names = {f"v1-{k:04d}": d for k, d in enumerate(
        (d for d, n in photos_per_design.items() for _ in range(n)), start=1)}
    folders = ("train_0", "train_90", "train_180", "train_270")
    return [Path(f"/store/images/{f}/{n}.jpg") for f in folders for n in names], names


def test_store_lines_give_each_design_the_same_count():
    images, designs = made_up_pool({"d01": 1, "d02": 10, "d03": 3})
    lines = m2b.store_lines(images, designs, 5_000, seed=0)
    per_design = Counter(designs[p.stem] for p in lines)
    assert per_design == {"d01": 1667, "d02": 1667, "d03": 1667}     # 5,000 / 3, rounded half up
    for d, n in {"d01": 1, "d02": 10, "d03": 3}.items():
        copies = Counter(p for p in lines if designs[p.stem] == d)
        assert len(copies) == 4 * n                                    # every photo, in all 4 turns
        assert max(copies.values()) - min(copies.values()) <= 1        # as even as can be
    assert m2b.store_lines(images, designs, 5_000, seed=0) == lines    # the same seed, the same lines
    assert m2b.store_lines(images, designs, 5_000, seed=1) != lines    # another seed, other extras
    assert len(m2b.store_lines(*made_up_pool({"a": 2, "b": 5}), 5, seed=0)) == 6    # 2.5 rounds up to 3


def test_the_store_pool_is_the_exports_training_list_and_designs(tmp_path):
    images, designs = made_up_pool({"d01": 1, "d02": 2})
    (tmp_path / "train.txt").write_text("".join(f"./images/{p.parent.name}/{p.name}\n" for p in images))
    manifest = {"settings": {"lists": {"train": "train.txt", "val": "val.txt"}},
                "photos": {**{n: {"part": "train", "design": d} for n, d in designs.items()},
                           "v1-0009": {"part": "val", "design": "d07"}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    pool, found = m2b.store_pool(tmp_path)
    assert pool == [tmp_path / "images" / p.parent.name / p.name for p in images]
    assert found == designs                                        # validation photos are no part of it


# --- Step 2 (decisions 7 to 11, S10) ----------------------------------------------------

def test_step_2s_arms_recipes_and_schedules():
    assert set(m2b.STEP2) == {"n1280-ft", "n1280-ft2", "s1280-ft", "s1280-ft2", "m1280-ft",
                              "m1280-ft2", "n1280-ft-gen50k"}
    assert {a.start for a in m2b.STEP2.values()} == {"n1280", "s1280", "m1280"}
    assert m2b.STEP2["n1280-ft-gen50k"] == m2b.StepTwo("n", "ft", "gen50k")   # only the generated pool differs
    assert all(a.imgsz == 1280 for a in m2b.STEP2.values())
    assert m2b.FT_RECIPE == {**m2.RECIPE, "lr0": 0.001}                       # a tenth of M2's rate
    assert m2b.BUDGET_FT * 10 == m2.BUDGET and m2b.RECIPES == ("ft", "ft2")
    [ft] = m2b.STAGES["ft"]
    assert ft.recipe() == {**m2b.FT_RECIPE, "freeze": None}
    first, second = m2b.STAGES["ft2"]
    assert first.share == second.share == 0.5
    assert first.recipe() == {**m2b.FT_RECIPE, "freeze": 11}                  # the backbone, layers 0 to 10
    assert second.recipe() == {**m2b.FT_RECIPE, "freeze": None, "lr0": 0.0005}
    # An epoch of 24,992 listed photos (the plan's mix): ft 10 epochs, ft2 5 + 5.
    assert m2b.schedule(24_992, ft) == {"epochs": 10, "warmup_epochs": 0.3, "close_mosaic": 1}
    assert m2b.schedule(24_992, first) == m2b.schedule(24_992, second) == {
        "epochs": 5, "warmup_epochs": 0.15, "close_mosaic": 1}
    assert m2b.schedule(400, first, epochs=1) == {"epochs": 1, "warmup_epochs": 0.03, "close_mosaic": 1}


def test_the_replay_keeps_gen_rot_1280s_shares():
    shares = m2b.REPLAY_SHARES
    assert sum(shares.values()) == pytest.approx(1.0)
    # gen-rot-1280 lists 79,310 photos: real ones 3 times, upright and turned alike.
    assert shares["real"] == shares["real_turned"] == pytest.approx(2_811 / 79_310)
    assert shares["synthetic"] == shares["synthetic_turned"] == pytest.approx(11_844 / 79_310)
    assert shares["generated"] == pytest.approx(50_000 / 79_310)
    # The plan's shares of an epoch, the store's 20% aside: real 5.7%, synthetic 23.9%, generated 50.4%.
    rest = 1 - m2b.STORE_SHARE
    assert round(100 * rest * (shares["real"] + shares["real_turned"]), 1) == 5.7
    assert round(100 * rest * (shares["synthetic"] + shares["synthetic_turned"]), 1) == 23.9
    assert round(100 * rest * shares["generated"], 1) == 50.4
    # The 180° copies join the turned pools; the side arm swaps only its generated photos.
    assert m2b.replay_lists(m2b.STEP2["n1280-ft"]) == {
        "real": ("train_real",), "real_turned": ("real_rot", "real_rot180"),
        "synthetic": ("train_synth_clean",), "synthetic_turned": ("synth_rot", "synth_rot180"),
        "generated": ("gen_v2",)}
    assert m2b.replay_lists(m2b.STEP2["n1280-ft-gen50k"])["generated"] == ("gen50k",)


def made_up_replay(sizes: dict[str, int]) -> dict[str, list[Path]]:
    return {pool: [Path(f"/{pool}/{i:06d}.jpg") for i in range(n)] for pool, n in sizes.items()}


def test_the_mix_draws_its_shares_with_its_seed():
    sizes = {"real": 937, "real_turned": 2 * 937, "synthetic": 11_844, "synthetic_turned": 2 * 11_844,
             "generated": 20_000}
    images, designs = made_up_pool({f"d{k:02d}": 1 + k % 5 for k in range(31)})   # 31 designs, as v1-train
    pools = {"store": images, **made_up_replay(sizes)}
    lists = m2b.mix(pools, designs, seed=0)
    counts = {pool: len(paths) for pool, paths in lists.items()}
    assert counts == {"store": 4_991, "real": 709, "real_turned": 709, "synthetic": 2_987,
                      "synthetic_turned": 2_987, "generated": 12_609}
    assert sum(counts.values()) == 24_992                       # ft: 10 epochs, 249,920 images
    for pool in sizes:                                          # each replay photo at most once
        assert len(set(lists[pool])) == counts[pool] and set(lists[pool]) <= set(pools[pool])
    assert Counter(designs[p.stem] for p in lists["store"]) == {d: 161 for d in set(designs.values())}
    # The same seed, the same lists, in whatever order the pools come; another seed, another sample.
    assert m2b.mix(dict(reversed(list(pools.items()))), designs, seed=0) == lists
    other = m2b.mix(pools, designs, seed=1)
    assert all(other[pool] != lists[pool] for pool in lists)
    # The turned pools hold their quarter turns and their half turns about equally.
    halves = made_up_replay({**sizes, "real_turned": 0})
    halves["real_turned"] = ([Path(f"/real_rot/{i}.jpg") for i in range(937)]
                             + [Path(f"/real_rot180/{i}.jpg") for i in range(937)])
    turned = m2b.mix({"store": images, **halves}, designs, seed=0)["real_turned"]
    assert 0.45 < sum(p.parent.name == "real_rot180" for p in turned) / len(turned) < 0.55
    with pytest.raises(ValueError, match="the generated pool holds 100 photos"):
        m2b.mix({**pools, "generated": pools["generated"][:100]}, designs, seed=0)


def test_frozen_drift_counts_what_a_frozen_stage_moved():
    import numpy as np
    before = {"model.0.conv.weight": np.ones((2, 3)), "model.0.bn.weight": np.ones(2),
              "model.0.bn.bias": np.zeros(2), "model.0.bn.running_mean": np.zeros(2),
              "model.0.bn.running_var": np.ones(2), "model.0.bn.num_batches_tracked": np.array(7),
              "model.1.conv.weight": np.ones(4), "model.10.m.0.attn.qkv.bn.running_var": np.ones(3),
              "model.11.cv1.conv.weight": np.ones(5), "model.23.dfl.conv.weight": np.arange(4.0)}
    after = {**before, "model.0.bn.running_mean": np.array([0.0, 0.25]),
             "model.0.bn.num_batches_tracked": np.array(9), "model.11.cv1.conv.weight": np.full(5, 0.5)}
    drift = m2b.frozen_drift(before, after, layers=11)
    assert drift["layers"] == 11
    assert drift["conv"] == {"tensors": 2, "values": 10, "changed": 0, "max_abs": 0.0, "mean_abs": 0.0}
    assert (drift["bn_affine"]["values"], drift["bn_affine"]["changed"]) == (4, 0)
    assert drift["bn_running_mean"] == {"tensors": 1, "values": 2, "changed": 1, "max_abs": 0.25,
                                        "mean_abs": 0.125}
    assert drift["bn_running_var"]["tensors"] == 2                    # model.10's too: frozen, not model.1's
    assert (drift["trained"]["values"], drift["trained"]["changed"]) == (9, 5)   # model.11 on, dfl included
    assert m2b.frozen_problems(drift) == []                         # statistics are only measured
    moved = m2b.frozen_drift(before, {**after, "model.1.conv.weight": np.full(4, 1.5)}, layers=11)
    assert m2b.frozen_problems(moved) == ["conv: 4 of 10 values changed, by up to 0.5"]


BOXES = ("0.5 0.5 0.2 0.1", "0.2 0.5 0.1 0.1")     # each made-up photo's two boxes, in YOLO's form


def made_up_export(folder: Path, photos_per_design: dict[str, int]) -> None:
    """A store export as export_store_yolo.py writes it, small: each training
    photo in its 4 turns, one validation photo, every file in the manifest
    with its SHA-256, so store_train.check_export passes on it."""
    _, designs = made_up_pool(photos_per_design)
    photos = {n: {"part": "train", "design": d, "tag": "clean", "labels": ["expiry", "due"]}
              for n, d in designs.items()}
    photos["v1-0900"] = {"part": "val", "design": "d99", "tag": "sideways", "labels": ["mfg", "expiry"]}
    folders = {"train": {"train_0": 0, "train_90": 1, "train_180": 2, "train_270": 3}, "val": {"val": 0}}
    lines = {"train": [], "val": []}
    for k, (name, p) in enumerate(photos.items()):
        for f in folders[p["part"]]:
            image, labels = folder / "images" / f / f"{name}.jpg", folder / "labels" / f / f"{name}.txt"
            for path in (image, labels):
                path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 48), (40 * k % 256, 90, 160)).save(image, quality=95)
            classes = [{"expiry": 0, "mfg": 0, "due": 1}[label] for label in p["labels"]]
            labels.write_text("".join(f"{c} {xywh}\n" for c, xywh in zip(classes, BOXES)))
            lines[p["part"]].append(f"./images/{f}/{name}.jpg\n")
    for part, name in (("train", "train.txt"), ("val", "val.txt")):
        (folder / name).write_text("".join(lines[part]))
    files = {f.relative_to(folder).as_posix(): m2b.sha256(f) for f in sorted(folder.rglob("*")) if f.is_file()}
    images = {f: sum(p["part"] == part for p in photos.values()) for part, fs in folders.items() for f in fs}
    (folder / "manifest.json").write_text(json.dumps(
        {"settings": {"folders": folders, "lists": {"train": "train.txt", "val": "val.txt"}},
         "counts": {"images": images}, "photos": photos, "files": files}))


@pytest.fixture
def step2(cluster, monkeypatch):
    """The cluster as S10 finds it: stage 1's lists, M2b's 180° sets and
    gen-v2, the store export, and the control's records. Lists large enough
    for an epoch of 400 photos (the smoke's)."""
    root = cluster
    sizes = {"real_rot": 40, "train_synth_clean": 100, "synth_rot": 100, "gen50k": 300}
    for name, n in sizes.items():
        write_list(m2.list_path(name), [Path(f"/m2/{name}/{i}.jpg") for i in range(n)])
    (root / "expdate" / "yolo" / "train_real.txt").write_text(
        "".join(f"./images/real-train/r{i}.jpg\n" for i in range(40)))
    for name, n in {"real_rot180": 40, "synth_rot180": 100, "gen_v2": 300}.items():
        write_list(m2b.list_path(name), [Path(f"/m2b/{name}/{i}.jpg") for i in range(n)])
    m2b.manifest_path("derived").parent.mkdir(parents=True)
    m2b.manifest_path("derived").write_text(json.dumps({"sets": {"real_rot180": {}, "synth_rot180": {}}}))
    m2b.manifest_path("gen-v2").write_text(json.dumps({"complete": True}))
    made_up_export(root / "store", {"d01": 1, "d02": 2, "d03": 1, "d04": 3})
    monkeypatch.setenv("STORE_TRAIN_DIR", str(root / "store"))
    controls = {s: m2b.write_control(s) for s in m2b.SEEDS}
    return root, controls


def fake_ultralytics(script, monkeypatch) -> list[dict]:
    """A stand-in for Ultralytics' training, as train() above, that keeps
    every call."""
    calls = []

    def fake_train(weights, data, imgsz, seed, project, name, keep, callbacks, **settings):
        calls.append(dict(weights=weights, imgsz=imgsz, seed=seed, project=project, name=name, keep=keep,
                          settings=settings))
        n = json.loads((data.parent / "run.json").read_text())["config"]["n_listed"]

        class MuSGD:
            param_groups = [{"momentum": settings["momentum"]}]
        trainer = SimpleNamespace(
            args=SimpleNamespace(**{**settings, "imgsz": imgsz, "seed": seed, "warmup_momentum": 0.8}),
            optimizer=MuSGD(), save_dir=str(project / name),
            train_loader=SimpleNamespace(dataset=FakeDataset(["x.jpg"] * n), num_workers=8))
        callbacks["on_train_start"](trainer)
        callbacks["on_train_end"](trainer)
        return project / name / "weights" / f"{keep}.pt"

    monkeypatch.setattr(script.detector, "train", fake_train)
    monkeypatch.setattr(script, "device_name", lambda device: "cpu")
    return calls


def new_run(script, argv: list[str]) -> Path:
    before = set(runs.RUNS_DIR.iterdir())
    try:
        script.main(argv)
    finally:
        [run_dir] = set(runs.RUNS_DIR.iterdir()) - before
    return run_dir


SMOKE = ["--seed", "0", "--epochs", "1", "--epoch-photos", "400", "--device", "cpu", "--workers", "8"]


def test_ft_trains_its_mix_from_the_controls_weights(load_script, monkeypatch, step2):
    root, controls = step2
    script = load_script("train_detector")
    calls = fake_ultralytics(script, monkeypatch)
    m2_run = m2.finished_run("gen-rot-1280-s0")
    m2_last = m2b.m2_weights(m2_run, json.loads((m2_run / "run.json").read_text()))
    before = snapshot(m2_run, root / "ckpt" / "m2")
    run_dir = new_run(script, ["--milestone", "m2b", "--arm", "n1280-ft", "--name", "smoke-ft", *SMOKE])
    record = json.loads((run_dir / "run.json").read_text())
    config = record["config"]
    # One training, from the control's weights: M2's last.pt, as its record names and hashes it.
    [call] = calls
    assert call["weights"] == m2_last and call["name"] == run_dir.name and call["keep"] == "last"
    assert call["project"] == root / "ckpt" / "m2b" and (call["imgsz"], call["seed"]) == (1280, 0)
    assert config["start"] == {"run": controls[0].name, "checkpoint": str(m2_last),
                               "sha256": json.loads((controls[0] / "run.json").read_text())["checkpoint_sha256"]}
    settings = call["settings"]
    assert {k: settings[k] for k in m2b.FT_RECIPE} == m2b.FT_RECIPE and settings["freeze"] is None
    assert (settings["epochs"], settings["warmup_epochs"], settings["close_mosaic"]) == (1, 0.03, 1)
    assert (settings["workers"], settings["cache"], settings["fraction"]) == (8, False, 1.0)
    # Its lists: 400 photos, the store's 20% design-balanced, each pool's share; written and listed.
    assert {pool: v["photos"] for pool, v in config["lists"].items()} == {
        "store": 80, "real": 11, "real_turned": 11, "synthetic": 48, "synthetic_turned": 48, "generated": 202}
    assert config["n_listed"] == config["train_images"] == 400 and config["epoch_photos"] == 400
    store = (run_dir / "lists" / "store.txt").read_text().splitlines()
    assert len(store) == 80 and all(line.startswith(str(root / "store" / "images" / "train_")) for line in store)
    assert all(line.startswith("/m2b/gen_v2/") for line in (run_dir / "lists" / "generated.txt").read_text().splitlines())
    data = (run_dir / "data.yaml").read_text()
    assert all(json.dumps(str(run_dir / "lists" / f"{pool}.txt")) in data for pool in config["lists"])
    assert "\nval: val.txt\n" in data                               # validated on real-val, as M2
    for pool, v in config["lists"].items():
        assert v["sha256"] == hashlib.sha256((run_dir / "lists" / f"{pool}.txt").read_bytes()).hexdigest()
    assert set(config["manifests"]) == {"derived", "m2b/derived", "m2b/gen-v2"}     # no gen50k: gen-v2
    assert config["store_export"] == m2b.sha256(root / "store" / "manifest.json")
    # The record: finished, one stage as asked, the final weights in the run's own folder.
    assert record["status"] == "finished" and record["stage"] == 1
    assert record["checkpoint"] == str(root / "ckpt" / "m2b" / run_dir.name / "weights" / "last.pt")
    [stage] = record["stages"]
    assert stage["applied"]["args"]["freeze"] is None and "frozen" not in stage
    assert stage["momentum_after_warmup"] == [0.937]
    # M2's run and weights were only read; the same seed draws the same lists again.
    assert snapshot(m2_run, root / "ckpt" / "m2") == before
    again = new_run(script, ["--milestone", "m2b", "--arm", "n1280-ft", "--name", "smoke-ft-again", *SMOKE])
    assert json.loads((again / "run.json").read_text())["config"]["lists"] == config["lists"]


def test_ft2_trains_two_stages_and_checks_its_frozen_layers(load_script, monkeypatch, step2):
    import numpy as np
    root, controls = step2
    script = load_script("train_detector")
    m2_run = m2.finished_run("gen-rot-1280-s0")
    start = {"model.0.conv.weight": np.ones(3), "model.0.bn.running_var": np.ones(2),
             "model.11.cv1.conv.weight": np.ones(4)}
    trained = {**start, "model.11.cv1.conv.weight": np.full(4, 0.9)}
    calls = fake_ultralytics(script, monkeypatch)
    monkeypatch.setattr(script.detector, "weights_state",
                        lambda path: start if Path(path).parent.parent.name == m2_run.name else trained)
    run_dir = new_run(script, ["--milestone", "m2b", "--arm", "n1280-ft2", "--name", "smoke-ft2", *SMOKE])
    record = json.loads((run_dir / "run.json").read_text())
    first, second = calls
    stage1_last = root / "ckpt" / "m2b" / f"{run_dir.name}-stage1" / "weights" / "last.pt"
    assert (first["name"], first["settings"]["freeze"], first["settings"]["lr0"]) == (
        f"{run_dir.name}-stage1", 11, 0.001)
    assert first["weights"] == Path(record["config"]["start"]["checkpoint"])
    assert (second["weights"], second["name"], second["settings"]["freeze"], second["settings"]["lr0"]) == (
        stage1_last, run_dir.name, None, 0.0005)
    assert [s["schedule"]["epochs"] for s in record["config"]["stages"]] == [1, 1]
    s1, s2 = record["stages"]
    assert (s1["stage"], s1["checkpoint"], s2["start"]) == (1, str(stage1_last), str(stage1_last))
    assert s1["frozen"]["conv"]["changed"] == 0 and s1["frozen"]["trained"]["changed"] == 4
    assert "frozen" not in s2 and record["status"] == "finished" and record["epochs_run"] is not None
    assert record["applied"]["args"]["freeze"] is None and record["stage"] == 2   # the watchdog sees stage 2
    # A frozen convolution that moved stops the run after stage 1, recorded.
    moved = {**trained, "model.0.conv.weight": np.full(3, 1.01)}
    monkeypatch.setattr(script.detector, "weights_state",
                        lambda path: start if Path(path).parent.parent.name == m2_run.name else moved)
    calls.clear()
    with pytest.raises(RuntimeError, match="stage 1 moved its frozen layers: conv: 3 of 3"):
        new_run(script, ["--milestone", "m2b", "--arm", "n1280-ft2", "--name", "smoke-ft2-moved", *SMOKE])
    assert len(calls) == 1
    failed = json.loads((runs.latest_run("m2b-smoke-ft2-moved") / "run.json").read_text())
    assert failed["status"] == "running" and failed["stages"][0]["frozen"]["conv"]["changed"] == 3


def test_step_2_starts_only_from_its_step_1_runs_record(load_script, monkeypatch, step2):
    root, controls = step2
    weights, digest, name = m2b.start_weights(m2b.STEP2["n1280-ft2"], 2)
    assert (name, digest) == (controls[2].name, m2b.sha256(weights))
    assert m2b.start_missing(m2b.STEP2["s1280-ft"], 1) == "m2b-s1280-s1 to finish"
    with pytest.raises(FileNotFoundError, match="m2b-s1280-s1 hasn't finished"):
        m2b.start_weights(m2b.STEP2["s1280-ft"], 1)
    s1280 = runs.start_run("m2b-s1280-s1", {"arm": "s1280"})
    last = root / "ckpt" / "m2b" / s1280.name / "weights" / "last.pt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"s1280 seed 1")
    runs.update_run(s1280, status="finished", checkpoint=str(last))
    assert m2b.start_weights(m2b.STEP2["s1280-ft2"], 1) == (last, m2b.sha256(last), s1280.name)
    assert m2b.start_missing(m2b.STEP2["s1280-ft"], 1) is None
    # M2's weights changed since the control recorded them: refused.
    m2b.m2_weights(*(lambda d: (d, json.loads((d / "run.json").read_text())))(
        m2.finished_run("gen-rot-1280-s0"))).write_bytes(b"something else")
    with pytest.raises(RuntimeError, match="has changed since m2b-n1280-s0 recorded it"):
        m2b.start_weights(m2b.STEP2["n1280-ft"], 0)
    # A store export that no longer matches its manifest: nothing trains.
    script = load_script("train_detector")
    calls = fake_ultralytics(script, monkeypatch)
    (root / "store" / "images" / "train_0" / "v1-0001.jpg").write_text("changed")
    with pytest.raises(SystemExit, match="doesn't match its manifest"):
        script.main(["--milestone", "m2b", "--arm", "n1280-ft", "--name", "x", *SMOKE[2:], "--seed", "1"])
    assert calls == []


# --- The M2b suite (S10) ---------------------------------------------------------------

def test_the_m2b_suite_is_m2s_and_three_more_sets(monkeypatch, step2):
    root, _ = step2
    assert m2b.SUITE == ("real-val", "real-val-rot", "real-val-small", "synth-val", "real-val-rot180",
                         "synth-val-v2", "v1-val")
    assert m2b.suite_list("v1-val") == root / "store" / "val.txt"
    assert m2b.suite_list("real-val") == root / "expdate" / "yolo" / "val.txt"
    assert m2b.suite_list("real-val-rot180") == m2b.list_path("real_val_rot180")
    assert m2b.suite_list("synth-val-v2") == m2b.list_path("synth_val_v2")
    assert m2b.suite_missing("v1-val") is None and m2b.suite_missing("synth-val") is None
    assert m2b.suite_missing("real-val-rot180").startswith("not made yet")
    assert m2b.suite_missing("synth-val-v2").startswith("not made yet")
    monkeypatch.delenv("STORE_TRAIN_DIR")
    assert "$STORE_TRAIN_DIR unset" in m2b.suite_missing("v1-val")


def test_v1_val_is_the_exports_validation_part_with_its_roles_and_tags(step2):
    root, _ = step2
    [sample] = sets.load_set("v1-val")
    assert sample.id == "v1-0900.jpg" and sample.tag == "sideways" and (sample.width, sample.height) == (64, 48)
    assert [b.label for b in sample.boxes] == ["mfg", "expiry"]          # the manifest's, in label-file order
    assert sample.boxes[0].xyxy == pytest.approx((0.4, 0.45, 0.6, 0.55))
    assert sets.runs_root("v1-val") == sets.runs_root("store") == store.RUNS_DIR
    # A label file whose classes don't follow the manifest: refused.
    (root / "store" / "labels" / "val" / "v1-0900.txt").write_text("1 0.5 0.5 0.2 0.1\n0 0.2 0.5 0.1 0.1\n")
    with pytest.raises(ValueError, match="v1-0900.jpg: its label file doesn't follow"):
        m2b.load_v1_val()


def test_dates_by_size_band_as_the_store_measures_them():
    from bestbefore.detection_eval import Prediction, evaluate
    from bestbefore.samples import Box, Sample
    boxes = {"small": (0.0, 0.0, 0.1, 0.01),          # 100 x 8 px on 1,000 x 800: 0.8%
             "middle": (0.2, 0.2, 0.4, 0.22),         # 200 x 16 px: 1.6%
             "sideways": (0.5, 0.1, 0.53, 0.4)}       # 30 x 240 px: its short side, 3%
    photo = Sample("a.jpg", Path("/a.jpg"), 1000, 800,
                   (Box("expiry", boxes["small"]), Box("due", (0.6, 0.6, 0.7, 0.7)), Box("mfg", boxes["middle"])))
    other = Sample("b.jpg", Path("/b.jpg"), 1000, 800, (Box("other_date", boxes["sideways"]),))
    preds = {"a.jpg": [Prediction("date", boxes["small"], 0.9), Prediction("date", boxes["middle"], 0.2)],
             "b.jpg": []}
    at_05 = evaluate([photo, other], preds, 0.5)
    assert m2b.found_by_size([photo, other], at_05, threshold=0.5, threshold_m2=0.1) == {
        "under 1.5%": {"dates": 1, "found": 1, "found_at_threshold_m2": 1, "boxed_at_any_confidence": 1},
        "1.5-2.5%": {"dates": 1, "found": 0, "found_at_threshold_m2": 1, "boxed_at_any_confidence": 1},
        "2.5% and up": {"dates": 1, "found": 0, "found_at_threshold_m2": 0, "boxed_at_any_confidence": 0}}
    with pytest.raises(ValueError):
        m2b.found_by_size([photo], at_05, 0.5, 0.1)


# --- The decision rules (written before any M2b run, "Decision rules") ------------------

def m2b_suite(v1: float, real: float = 0.99, synth: float = 0.99, threshold: float = 0.25) -> dict:
    """An M2b suite with the mAP@0.5 for dates the rules read."""
    return {"threshold": threshold, "real-val": {"map50": {"date": real}},
            "synth-val": {"map50": {"date": synth}}, "v1-val": {"map50": {"date": v1}}}


def three(v1: float, **kw) -> list[dict]:
    return [m2b_suite(v1, **kw)] * 3


CONTROL = [m2b_suite(0.80), m2b_suite(0.85), m2b_suite(0.82)]


def test_the_noise_band_and_the_guard():
    assert m2b.noise_store(CONTROL) == pytest.approx(0.05)                      # the control's spread
    assert m2b.noise_store(three(0.80)) == m2b.NOISE_STORE_MIN == 0.03          # at least 0.03
    assert m2b.NOISE == m2.NOISE == 0.01
    assert m2b.guard(three(0.5, real=0.98, synth=0.98), CONTROL)                # 0.01 under: still in
    assert not m2b.guard(three(0.9, real=0.979), CONTROL)                       # forgot real-val
    assert not m2b.guard(three(0.9, synth=0.979), CONTROL)                      # forgot synth-val
    assert m2b.guard(CONTROL, CONTROL)


def test_r1_gives_m_its_seeds_only_if_it_beats_every_s_seed():
    s = [m2b_suite(0.86), m2b_suite(0.90), m2b_suite(0.88)]
    assert m2b.probe_m(m2b_suite(0.91), s, CONTROL)
    assert not m2b.probe_m(m2b_suite(0.90), s, CONTROL)            # a tie with the best s seed doesn't beat it
    assert not m2b.probe_m(m2b_suite(0.95, real=0.97), s, CONTROL)  # beats them all, but forgets real-val


def test_r2_takes_the_simplest_recipe_within_the_band():
    def recipes(ft, ft2, **kw):
        return {"ft": three(ft), "ft2": three(ft2, **kw)}
    assert m2b.decide_recipe(recipes(0.85, 0.87), 0.03, CONTROL) == "ft"     # within the band: the simplest
    assert m2b.decide_recipe(recipes(0.85, 0.88), 0.03, CONTROL) == "ft"     # the band's edge: still in it
    assert m2b.decide_recipe(recipes(0.85, 0.89), 0.03, CONTROL) == "ft2"    # beyond it
    assert m2b.decide_recipe(recipes(0.85, 0.95, synth=0.95), 0.03, CONTROL) == "ft"   # ft2 forgets: out
    assert m2b.decide_recipe({"ft": three(0.9, real=0.9), "ft2": three(0.9, synth=0.9)}, 0.03, CONTROL) is None


def test_r3_takes_the_cheapest_candidate_within_the_band():
    suites = {"n1280": CONTROL, "n1280-ft": three(0.86), "s1280": three(0.84), "s1280-ft": three(0.88),
              "m1280": [m2b_suite(0.99), None, None], "m1280-ft": [None] * 3}
    assert m2b.decide_detector(suites, 0.03, CONTROL) == "n1280-ft"            # within 0.03 of s1280-ft
    assert m2b.decide_detector({**suites, "s1280-ft": three(0.90)}, 0.03, CONTROL) == "s1280-ft"
    assert m2b.decide_detector({**suites, "s1280-ft": three(0.95, real=0.95)}, 0.03, CONTROL) == "n1280-ft"
    # The control's mean is 0.823: s1280 must beat it by more than the band to be chosen over it.
    only_s = {**suites, "n1280-ft": three(0.70), "s1280": three(0.86), "s1280-ft": three(0.70)}
    assert m2b.decide_detector(only_s, 0.03, CONTROL) == "s1280"
    assert m2b.decide_detector({**only_s, "s1280": three(0.84)}, 0.03, CONTROL) == "n1280"   # M2's weights
    assert m2b.decide_detector({**suites, "m1280": three(0.99)}, 0.03, CONTROL) == "m1280"   # all 3 seeds: in
    assert m2b.candidate_runs("ft2", with_m=False) == {"n1280": "n1280", "n1280-ft": "n1280-ft2",
                                                       "s1280": "s1280", "s1280-ft": "s1280-ft2"}


def scored(name: str, v1: float, **kw) -> Path:
    """A finished M2b run with its M2b suite."""
    run_dir = runs.start_run(f"m2b-{name}", {})
    runs.update_run(run_dir, status="finished")
    (run_dir / "suite-m2b.json").write_text(json.dumps(m2b_suite(v1, **kw)))
    return run_dir


def test_decisions_come_as_their_inputs_complete(monkeypatch, tmp_path):
    for root, m in ((tmp_path / "a", 0.85), (tmp_path / "b", 0.90)):     # R1 fails, then passes
        monkeypatch.setattr(runs, "RUNS_DIR", root)
        root.mkdir()
        assert m2b.decisions() == {}
        for s, v in enumerate((0.80, 0.82, 0.81)):
            scored(f"n1280-s{s}", v)
        assert m2b.decisions() == {"noise_store": {"band": 0.03, "spread": 0.02,
                                                   "control_v1_val_map50": [0.80, 0.82, 0.81]}}
        for s, v in enumerate((0.84, 0.85, 0.86)):
            scored(f"s1280-s{s}", v)
        scored("m1280-s0", m)
        r1 = m2b.decisions()["R1"]
        assert r1 == {"passes": m > 0.86, "guard": True, "m1280_s0_v1_val_map50": m,
                      "s1280_v1_val_map50": [0.84, 0.85, 0.86]}
        for s in m2b.SEEDS:
            scored(f"n1280-ft-s{s}", 0.88)
            scored(f"n1280-ft2-s{s}", 0.90)
        r2 = m2b.decisions()["R2"]
        assert (r2["recipe"], r2["v1_val_map50"], r2["guard"]) == (
            "ft", {"ft": 0.88, "ft2": 0.9}, {"ft": True, "ft2": True})     # ft2 within the band of 0.03
        for s in m2b.SEEDS:
            scored(f"s1280-ft-s{s}", 0.95)
        d = m2b.decisions()
        if not r1["passes"]:          # without m, R3 has its four candidates
            assert (d["R3"]["arm"], d["R3"]["run"], d["R3"]["threshold"]) == ("s1280-ft", "s1280-ft-s0", 0.25)
            assert list(d["R3"]["v1_val_map50"]) == ["n1280", "n1280-ft", "s1280", "s1280-ft"]
            continue
        assert "R3" not in d          # R1 passed: R3 waits for m's seeds and its step 2
        for s in (1, 2):
            scored(f"m1280-s{s}", 0.90)
        for s in m2b.SEEDS:
            scored(f"m1280-ft-s{s}", 0.97)
        d = m2b.decisions()
        assert d["R3"]["arm"] == "s1280-ft"                        # 0.95 within 0.03 of m1280-ft's 0.97
        assert list(d["R3"]["v1_val_map50"]) == ["n1280", "n1280-ft", "s1280", "s1280-ft", "m1280", "m1280-ft"]


def test_step_2_on_s_and_m_waits_for_its_rules_and_takes_r2s_recipe(load_script, step2):
    script = load_script("m2b_chain")
    claims = slurm.Claims(script.CHAIN.claims_dir(), "73832")

    def at(decisions: dict) -> dict:
        return {t.name: s for t, s in script.CHAIN.states(decisions, claims, set()).items()}

    State = script.chain.State
    current = at({"R1": {"passes": False}, "R2": {"recipe": None}})
    assert current["m2b-m1280-s1"] == State("blocked", "blocked: R1: m1280-s0 didn't earn m more seeds")
    assert current["m2b-m1280-<recipe>-s0"].status == "blocked"
    assert current["m2b-s1280-<recipe>-s2"] == State(
        "blocked", "blocked: R2: no recipe passed the guard, for a person to look at")
    current = at({"R1": {"passes": True}, "R2": {"recipe": "ft2"}})
    assert not any("<recipe>" in name for name in current)          # named by R2's recipe
    assert current["m2b-s1280-ft2-s0"] == State("waiting", "waiting for m2b-s1280-s0 to finish")
    assert current["m2b-m1280-s1"].status == "ready"                # R1 passed: m's seed 1, a step-1 run
    assert current["m2b-m1280-ft2-s1"] == State("waiting", "waiting for m2b-m1280-s1 to finish")
    scored("s1280-s0", 0.9)
    assert at({"R1": {"passes": True}, "R2": {"recipe": "ft2"}})["m2b-s1280-ft2-s0"].status == "ready"
    assert script.summary({"noise_store": {"band": 0.03}, "R1": {"passes": False}, "R2": {"recipe": "ft2"},
                           "R3": {"arm": "s1280-ft2"}}) == "noise band 0.0300, R1 fails, R2 ft2, R3 s1280-ft2"


def test_the_m2b_suite_scores_the_control_first(load_script, step2):
    script = load_script("m2b_chain")
    current = states(script)
    assert current["suite-m2b-n1280-s0"].note == (
        "waiting for real-val-rot180 not made yet (S9's derive_sets.py --only rot180); "
        "synth-val-v2 not made yet (S9's generation)")
    write_list(m2b.list_path("real_val_rot180"), [Path("/m2b/rot180/0.jpg")])
    write_list(m2b.list_path("synth_val_v2"), [Path("/m2b/synth_val_v2/0.jpg")])
    derived = json.loads(m2b.manifest_path("derived").read_text())
    derived["sets"]["real_val_rot180"] = {}
    m2b.manifest_path("derived").write_text(json.dumps(derived))
    m2b.manifest_path("synth-val-v2").write_text(json.dumps({"complete": True}))
    s1280 = scored("s1280-s0", 0.9)                 # trained and scored on M2's suite, as on the cluster
    s1280.joinpath("suite-m2b.json").rename(s1280 / "suite.json")
    current = states(script)
    assert [current[f"suite-m2b-n1280-s{s}"].status for s in m2b.SEEDS] == ["ready"] * 3
    assert current["suite-m2b-s1280-s0"].note == (
        "waiting for the noise band: the control's three seeds on the M2b suite")
    claims = slurm.Claims(script.CHAIN.claims_dir(), "73832")
    ready = script.CHAIN.states({}, claims, set())
    assert script.CHAIN.pick(ready, 72 * 3600).name == "suite-m2b-n1280-s0"   # scoring first, the control first
    banded = script.CHAIN.states({"noise_store": {"band": 0.03}}, claims, set())
    assert {t.name: s.status for t, s in banded.items()}["suite-m2b-s1280-s0"] == "ready"
