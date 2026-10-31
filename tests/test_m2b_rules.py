"""M2b's step 1 and its control (docs/m2b-store-and-size.md, S2): step 1 is
M2's gen-rot-1280 with other weights, and the control is M2's runs as they
are. Against M2's own code and records, on made-up lists and runs."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bestbefore import m2, m2b, runs
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
