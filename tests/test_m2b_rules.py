"""M2b's step 1 and its control (docs/m2b-store-and-size.md, S2): step 1 is
M2's gen-rot-1280 with other weights, and the control is M2's runs as they
are. Against M2's own code and records, on made-up lists and runs."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bestbefore import m2, m2b, runs, slurm
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


def test_m2b_trains_only_its_step_1_arms(load_script):
    parse = load_script("train_detector").parse_args
    for argv in (["--milestone", "m2b", "--arm", "n1280"],        # M2's runs: the control
                 ["--milestone", "m2b", "--arm", "gen-rot-1280"],  # an M2 arm
                 ["--milestone", "m2", "--arm", "s1280"],          # a step-1 arm, as M2
                 ["--milestone", "m2b"],
                 ["--milestone", "m2b", "--arm", "s1280", "--weights", "yolo11m"]):
        with pytest.raises(SystemExit):
            parse([*argv, "--name", "x"])
    args = parse(["--milestone", "m2b", "--arm", "m1280", "--name", "m1280-s0"])
    assert (args.arm, args.smoke_lists, args.batch) == ("m1280", None, None)


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


def test_the_chain_starts_with_stage_1_and_stage_2_waits_for_its_data(load_script, cluster):
    script = load_script("m2b_chain")
    current = states(script)
    assert [n for n, s in current.items() if s.status == "ready"] == STAGE_1
    stage2 = {t.name: current[t.name] for t in script.TASKS if t.stage == 2 and not t.score}
    assert len(stage2) == 17 and all(s.status == "waiting" for s in stage2.values())
    for name, s in stage2.items():
        if name.startswith(("m2b-m1280-s1", "m2b-m1280-s2")):
            assert "R1" in s.note
            continue
        for needed in ("store set v2's labels frozen (M0 D6)", "store set v1's export",
                       "real_rot180, synth_rot180 not made yet", "step 2's code and rules (S10)"):
            assert needed in s.note, (name, needed)
        assert ("gen_v2" in s.note) == ("gen50k" not in name)      # the side arm keeps M2's gen50k
    assert current["score-m2b-smoke"].note == "waiting for m2b-smoke to finish"
    assert sum(t.score for t in script.TASKS) == 7               # smoke, s x 3, m x 3
    assert not any(t.score and t.arm == m2b.CONTROL for t in script.TASKS)   # M2 scored them


def test_stage_2_waits_for_s10_even_with_all_its_data(load_script, cluster, monkeypatch):
    script = load_script("m2b_chain")
    monkeypatch.setattr(script.store, "TEST_SET_V2_SHA256", "0" * 64)
    (cluster / "store").mkdir()
    (cluster / "store" / "manifest.json").write_text("{}")
    monkeypatch.setenv("STORE_TRAIN_DIR", str(cluster / "store"))
    for name in m2b.STEP2_LISTS:
        write_list(m2b.list_path(name), [Path(f"/{name}/0.jpg")])
    m2b.manifest_path("derived").parent.mkdir(parents=True)
    m2b.manifest_path("derived").write_text(json.dumps({"sets": {"real_rot180": {}, "synth_rot180": {}}}))
    m2b.manifest_path("gen-v2").write_text(json.dumps({"complete": True}))
    current = states(script)
    for task in script.TASKS:
        if task.stage == 2 and task.arm not in m2b.STEP1:
            assert current[task.name] == script.chain.State(
                "waiting", "waiting for step 2's code and rules (S10)")
    assert all(current[t.name].status == "waiting" for t in script.TASKS if t.stage == 2)
    assert script.run_task(next(t for t in script.TASKS if t.stage == 2), {}) == (
        "stage 2 isn't written yet (S10)")


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
    for name in ("m2b-smoke", "m2b-s1280-s1", "m2b-m1280-s0", "score-m2b-s1280-s1"):
        script.run_task(named[name], {})
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
    assert out.startswith("stage 1: public data\n  m2b-smoke                  ready")
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
