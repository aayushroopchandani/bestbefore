from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from bestbefore import detector
from bestbefore.detection_eval import Prediction


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


class FakeModel:
    """Stands in for an Ultralytics model: one result per image."""
    def __init__(self):
        self.seen = []

    def predict(self, source, imgsz, conf, device, verbose):
        self.seen = source
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
