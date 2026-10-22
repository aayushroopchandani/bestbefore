import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from bestbefore import detector, m2, runs, sets
from bestbefore.derived import write_list, write_yolo
from bestbefore.detection_eval import Prediction
from bestbefore.samples import Box, Sample
from bestbefore.yolo import CLASSES

# What dataset_yaml wrote for M1's runs before M2 changed it (commit 7b3fc0b).
M1_DATA_YAML = """# Made by bestbefore.detector.dataset_yaml
path: {path}
train: [train_synth.txt, train_real.txt]
val: val.txt
test: test.txt
names:
  0: date
  1: due
  2: prod
  3: code
"""


def _yolo_dir(root: Path) -> Path:
    """An ExpDate-like yolo/ folder holding M1's four lists."""
    root.mkdir(parents=True, exist_ok=True)
    for name in ("train_real", "train_synth", "val", "test"):
        (root / f"{name}.txt").write_text("./images/x/a.jpg\n")
    return root


# --- M1 stays as it was --------------------------------------------------------

def test_m1_settings_are_unchanged():
    assert detector.SETTINGS == dict(epochs=100, patience=30, batch=16, fliplr=0.0,
                                     deterministic=True)


def test_m1_dataset_file_is_unchanged(tmp_path):
    yolo = _yolo_dir(tmp_path)
    out = detector.dataset_yaml(yolo, ["train_synth", "train_real"], tmp_path / "data.yaml")
    assert out.read_text() == M1_DATA_YAML.format(path=yolo.resolve())


def test_train_detector_m1_records_trains_and_scores_as_before(load_script, monkeypatch,
                                                               tmp_path):
    script = load_script("train_detector")
    yolo = _yolo_dir(tmp_path / "expdate" / "yolo")
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("EXPDATE_DIR", str(tmp_path / "expdate"))
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    best = tmp_path / "ckpt" / "m1" / "x" / "weights" / "best.pt"
    calls = {}

    def fake_train(*args, **kwargs):
        calls["train"] = args, kwargs
        return best

    def fake_score(*args, **kwargs):
        calls["score"] = args, kwargs
        return {"map50": {"date": 0.99}, "threshold": 0.73,
                "iou0.5": {"date_recall": 1.0, "precision": 0.95}}

    monkeypatch.setattr(script.detector, "pretrained_path", lambda w: Path(f"/pre/{w}.pt"))
    monkeypatch.setattr(script.detector, "train", fake_train)
    monkeypatch.setattr(script.detector, "score", fake_score)
    monkeypatch.setattr(script, "load_set", lambda name: [name])
    monkeypatch.setattr(script, "device_name", lambda device: "cpu")
    script.main(["--weights", "yolo11n", "--imgsz", "640", "--train", "train_synth,train_real",
                 "--name", "n640"])

    [run_dir] = (tmp_path / "runs").iterdir()
    assert run_dir.name.endswith("-m1-n640")
    assert json.loads((run_dir / "run.json").read_text())["config"] == {
        "weights": "yolo11n", "imgsz": 640, "train": "train_synth,train_real", "seed": 0,
        "name": "n640", "epochs": 100, "patience": 30, "batch": 16, "fraction": 1.0,
        "workers": 8, "device": None}
    assert (run_dir / "data.yaml").read_text() == M1_DATA_YAML.format(path=yolo.resolve())
    project = tmp_path / "ckpt" / "m1"
    assert calls["train"] == ((Path("/pre/yolo11n.pt"), run_dir / "data.yaml", 640, 0, project,
                               run_dir.name),
                              dict(epochs=100, patience=30, batch=16, fraction=1.0, workers=8,
                                   device=None))
    assert calls["score"] == ((best, ["real-val"], 640),
                              dict(data=run_dir / "data.yaml", split="val",
                                   project=project / run_dir.name, device=None))
    assert json.loads((run_dir / "val.json").read_text())["threshold"] == 0.73


def test_train_detector_keeps_m1_and_m2_flags_apart(load_script):
    parse = load_script("train_detector").parse_args
    with pytest.raises(SystemExit):
        parse(["--name", "x", "--weights", "yolo11n", "--imgsz", "640"])          # no --train
    with pytest.raises(SystemExit):
        parse(["--name", "x", "--weights", "yolo11n", "--imgsz", "640", "--train", "t",
               "--arm", "base"])
    with pytest.raises(SystemExit):
        parse(["--milestone", "m2", "--name", "x"])                               # no --arm
    with pytest.raises(SystemExit):
        parse(["--milestone", "m2", "--name", "x", "--arm", "base", "--batch", "32"])
    args = parse(["--milestone", "m2", "--name", "x", "--arm", "base"])
    assert (args.epochs, args.batch, args.budget_share) == (None, None, 1.0)


# --- M2's additions ------------------------------------------------------------

def test_dataset_yaml(tmp_path):
    for name in ("train_real", "train_synth", "val", "test"):
        (tmp_path / f"{name}.txt").write_text("./images/x/a.jpg\n")
    out = detector.dataset_yaml(tmp_path, ["train_real"], tmp_path / "data.yaml")
    text = out.read_text()
    assert "train: [train_real.txt]\n" in text
    assert "val: val.txt\n" in text and "  0: date\n" in text
    both = detector.dataset_yaml(tmp_path, ["train_synth", "train_real"], tmp_path / "b.yaml")
    assert "train: [train_synth.txt, train_real.txt]\n" in both.read_text()


def test_dataset_yaml_rejects_a_missing_list(tmp_path):
    with pytest.raises(FileNotFoundError):
        detector.dataset_yaml(tmp_path, ["train_m2"], tmp_path / "data.yaml")


def test_pretrained_path_never_downloads(monkeypatch, tmp_path):
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="fetch_pretrained"):
        detector.pretrained_path("yolo11s")
    (tmp_path / "pretrained").mkdir()
    (tmp_path / "pretrained" / "yolo11s.pt").write_bytes(b"x")
    assert detector.pretrained_path("yolo11s.pt").name == "yolo11s.pt"


def test_epochs_run(tmp_path):
    best = tmp_path / "weights" / "best.pt"
    assert detector.epochs_run(best) is None
    (tmp_path / "results.csv").write_text("epoch,time\n1,4.7\n2,4.6\n3,4.6\n")
    assert detector.epochs_run(best) == 3


class FakeModel:
    """Stands in for an Ultralytics model: one result per image."""
    def __init__(self):
        self.seen, self.rect = [], None

    def predict(self, source, imgsz, conf, device, rect, verbose):
        self.seen, self.rect = source, rect
        boxes = SimpleNamespace(xyxyn=np.array([[0.1, 0.4, 0.3, 0.45]]),
                                cls=np.array([0.0]), conf=np.array([0.87]))
        return [SimpleNamespace(boxes=boxes, names={0: "date", 1: "due"})
                for _ in source]


def test_predict_returns_named_normalised_boxes():
    model = FakeModel()
    images = [Image.new("RGB", (40, 20)), Image.new("RGB", (40, 20))]
    out = detector.predict(model, images, imgsz=640)
    assert len(out) == 2
    assert out[0] == [Prediction("date", (0.1, 0.4, 0.3, 0.45), 0.87)]
    assert model.seen is images        # PIL images go in, never file paths
    assert model.rect is False          # padding never depends on the batch


def test_dataset_yaml_repeats_lists_takes_absolute_paths_and_a_chosen_val(tmp_path):
    yaml = pytest.importorskip("yaml")
    yolo = _yolo_dir(tmp_path / "yolo")
    other = tmp_path / "Local Projects" / "lists" / "gen50k.txt"      # a space, as on the laptop
    other.parent.mkdir(parents=True)
    other.write_text("/g/1.jpg\n")
    out = detector.dataset_yaml(yolo, ["train_real*3", str(other)], tmp_path / "d.yaml",
                                val=str(other))
    data = yaml.safe_load(out.read_text())
    assert data["train"] == ["train_real.txt"] * 3 + [str(other)]
    assert data["val"] == str(other) and data["path"] == str(yolo.resolve())
    with pytest.raises(FileNotFoundError):
        detector.dataset_yaml(yolo, ["train_real"], tmp_path / "e.yaml", val="real_val_rot")


def test_train_returns_the_weights_asked_for(monkeypatch, tmp_path):
    seen = {}

    class FakeYOLO:
        def __init__(self, weights):
            self.callbacks = {}

        def add_callback(self, event, fn):
            self.callbacks[event] = fn

        def train(self, **kwargs):
            seen.update(kwargs, callbacks=self.callbacks)
            self.trainer = SimpleNamespace(save_dir=str(tmp_path / "run"))

    fake = types.ModuleType("ultralytics")
    fake.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    last = detector.train(Path("w.pt"), Path("d.yaml"), 640, 0, tmp_path, "run", keep="last",
                          callbacks={"on_train_start": print}, epochs=3, patience=0)
    assert last == tmp_path / "run" / "weights" / "last.pt"
    assert (seen["epochs"], seen["patience"], seen["batch"]) == (3, 0, 16)
    assert seen["callbacks"] == {"on_train_start": print}
    assert detector.train(Path("w.pt"), Path("d.yaml"), 640, 0, tmp_path, "run").name == "best.pt"
    with pytest.raises(ValueError, match="keep"):
        detector.train(Path("w.pt"), Path("d.yaml"), 640, 0, tmp_path, "run", keep="final")


class FakeDataset(list):
    """Ultralytics' training set: its photos, repeats included."""
    cache = None

    @property
    def im_files(self):
        return list(self)


def fake_trainer(args: dict, images: list[str], momentum: float = 0.937):
    class MuSGD:                       # applied() reports the optimiser's class name
        param_groups = [{"momentum": momentum}, {"momentum": momentum}]
    return SimpleNamespace(args=SimpleNamespace(**args), optimizer=MuSGD(),
                           train_loader=SimpleNamespace(dataset=FakeDataset(images), num_workers=8),
                           save_dir="/ckpt/m2/run")


def test_applied_reads_what_ultralytics_trains_with():
    args = {k: f"value of {k}" for k in detector.APPLIED_ARGS}
    trainer = fake_trainer(args, ["a.jpg", "a.jpg", "b.jpg"], momentum=0.93694)
    got = detector.applied(trainer)
    assert got["args"] == args and got["optimizer"] == "MuSGD"
    assert (got["train_images"], got["distinct_images"], got["ram_cache"]) == (3, 2, False)
    assert got["save_dir"] == "/ckpt/m2/run"
    assert detector.momenta(trainer) == [0.9369]


def test_ultralytics_keeps_a_lists_repeats(tmp_path):
    yaml = pytest.importorskip("yaml")
    pytest.importorskip("ultralytics")
    from ultralytics.data.dataset import YOLODataset

    photos = [write_yolo(tmp_path, "s", f"{i}.jpg", Image.new("RGB", (64, 48), "white"),
                         [Box("date", (0.1, 0.1, 0.5, 0.3))]).image for i in range(2)]
    listed = tmp_path / "lists" / "two.txt"
    write_list(listed, photos)
    data = yaml.safe_load(detector.dataset_yaml(tmp_path, [f"{listed}*3"], tmp_path / "d.yaml",
                                                val=str(listed)).read_text())
    dataset = YOLODataset(img_path=data["train"], imgsz=64, augment=False,
                          data={"names": dict(enumerate(CLASSES)), "channels": 3})
    assert len(dataset) == 6 and len(set(dataset.im_files)) == 2


def _m2_base_lists(tmp_path: Path, monkeypatch) -> None:
    """base's two lists, as S2 leaves them, under tmp_path."""
    monkeypatch.setenv("EXPDATE_DIR", str(tmp_path / "expdate"))
    monkeypatch.setenv("GENERATED_DIR", str(tmp_path / "generated"))
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    yolo = _yolo_dir(tmp_path / "expdate" / "yolo")
    (yolo / "train_real.txt").write_text("./images/real-train/a.jpg\n" * 2)
    write_list(m2.list_path("train_synth_clean"), [Path(f"/s/{i}.jpg") for i in range(26)])
    m2.manifest_path("derived").parent.mkdir(parents=True)
    m2.manifest_path("derived").write_text(json.dumps({"sets": {"train_synth_clean": {}}}))


@pytest.mark.parametrize("ultralytics_momentum", [0.937, 0.9])
def test_train_detector_m2_checks_the_recipe_when_training_starts(
        load_script, monkeypatch, tmp_path, ultralytics_momentum):
    script = load_script("train_detector")
    _m2_base_lists(tmp_path, monkeypatch)

    def fake_train(weights, data, imgsz, seed, project, name, keep, callbacks, **settings):
        args = {**settings, "imgsz": imgsz, "seed": seed, "warmup_momentum": 0.8,
                "momentum": ultralytics_momentum}
        trainer = fake_trainer(args, ["x.jpg"] * 28)
        callbacks["on_train_start"](trainer)
        callbacks["on_train_end"](trainer)
        return project / name / "weights" / f"{keep}.pt"

    monkeypatch.setattr(script.detector, "pretrained_path", lambda w: Path(f"/pre/{w}.pt"))
    monkeypatch.setattr(script.detector, "train", fake_train)
    monkeypatch.setattr(script, "device_name", lambda device: "cpu")
    argv = ["--milestone", "m2", "--arm", "base", "--name", "base-s0", "--device", "cpu"]
    if ultralytics_momentum != m2.RECIPE["momentum"]:
        with pytest.raises(RuntimeError, match="momentum: asked 0.937, Ultralytics has 0.9"):
            script.main(argv)
    else:
        script.main(argv)

    [run_dir] = (tmp_path / "runs").iterdir()
    record = json.loads((run_dir / "run.json").read_text())
    assert record["config"]["lists"] == ["train_real", "train_synth_clean"]
    assert record["config"]["n_listed"] == 28 and record["config"]["schedule"]["epochs"] > 0
    assert set(record["config"]["manifests"]) == {"derived"}
    assert record["applied"]["args"]["momentum"] == ultralytics_momentum
    data = (run_dir / "data.yaml").read_text()
    assert f'train: [train_real.txt, "{m2.list_path("train_synth_clean")}"]' in data
    if ultralytics_momentum == m2.RECIPE["momentum"]:
        assert record["status"] == "finished" and record["checkpoint"].endswith("last.pt")
        assert record["momentum_after_warmup"] == [0.937]
    else:
        assert record["status"] == "running"        # the chain marks it failed


def _fake_ultralytics(monkeypatch) -> None:
    fake = types.ModuleType("ultralytics")
    fake.YOLO = lambda weights: SimpleNamespace(weights=weights)
    monkeypatch.setitem(sys.modules, "ultralytics", fake)


def test_score_suite_pools_the_real_sets_and_marks_missing_ones(monkeypatch, tmp_path):
    _m2_base_lists(tmp_path, monkeypatch)
    for name in ("real_val_rot", "real_val_small"):
        write_list(m2.list_path(name), [Path("/d/1.jpg")])
    _fake_ultralytics(monkeypatch)
    date, wrong = (0.1, 0.4, 0.3, 0.45), (0.6, 0.6, 0.7, 0.65)
    # One date per set, found at a different confidence in each, plus a false box.
    found_at = {"real-val": 0.9, "real-val-rot": 0.4, "real-val-small": 0.2}
    samples = {name: [Sample(id=name, image=Path(f"/{name}.jpg"), width=100, height=100,
                             boxes=(Box("date", date),))] for name in found_at}
    validated = {}

    def fake_val_metrics(model, data, split, imgsz, device, project, name, conf):
        validated[name] = data.read_text()
        return {"map50": {"date": 0.9}}

    monkeypatch.setattr(m2, "suite_missing",
                        lambda name: "synth_val not made yet (S6)" if name == "synth-val" else None)
    monkeypatch.setattr(sets, "load_set", lambda name: samples[name])
    monkeypatch.setattr(detector, "predict_samples", lambda model, ss, imgsz, device: {
        s.id: [Prediction("date", date, found_at[s.id]), Prediction("date", wrong, 0.5)]
        for s in ss})
    monkeypatch.setattr(detector, "_val_metrics", fake_val_metrics)

    suite = detector.score_suite(tmp_path / "ckpt" / "run" / "weights" / "last.pt", 640, tmp_path)
    # Alone, real-val would keep only boxes at 0.9 (F2 = 1). Pooled, every
    # date counts: keeping down to 0.2 gives P = 1/2, R = 1, F2 = 0.833.
    assert suite["threshold"] == pytest.approx(0.2)
    assert suite["f2_at_threshold"] == pytest.approx(0.8333, abs=1e-4)
    assert suite["synth-val"] == {"missing": "synth_val not made yet (S6)"}
    assert suite["missing"] == {"synth-val": "synth_val not made yet (S6)"}
    for name in found_at:
        at5 = suite[name]["iou0.5"]
        assert (at5["dates_found"], at5["false_boxes_per_photo"]) == (1, 1.0)
        assert suite[name]["map50"] == {"date": 0.9}
    assert "val: val.txt" in validated["real-val"]
    assert f'val: "{m2.list_path("real_val_rot")}"' in validated["real-val-rot"]
    assert (tmp_path / "suite" / "real-val-small.yaml").exists()

    # Without all three real sets there is no threshold.
    monkeypatch.setattr(m2, "suite_missing",
                        lambda name: None if name == "real-val" else f"{name} not made yet")
    with pytest.raises(FileNotFoundError, match="real-val-rot"):
        detector.score_suite(tmp_path / "last.pt", 640, tmp_path)
