"""The inference add-ons (docs/m2b-store-and-size.md, S13): boxes turned and
turned back, weighted boxes fusion, test-time rotation with a fake model,
the passes each configuration fuses, the weights' hashes, and
scripts/m2b_inference.py end to end on made-up photos and records."""
import gzip
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from bestbefore import derived, detector, ensemble, m2b, runs, sets, store
from bestbefore.detection_eval import Prediction, evaluate, iou
from bestbefore.samples import Box, Sample

# Fusing needs the `ensemble` extra (the laptop's venv); the rest doesn't.
needs_wbf = pytest.mark.skipif(importlib.util.find_spec("ensemble_boxes") is None,
                               reason="ensemble-boxes isn't installed (the ensemble extra)")

WIDE = (0.20, 0.40, 0.60, 0.50)     # an upright date: wider than tall
TALL = (0.40, 0.20, 0.50, 0.80)     # a sideways one


def test_a_box_turned_and_turned_back_is_itself():
    for quarter in ensemble.TURNS:
        turned = Prediction("date", derived.turn_box(WIDE, quarter), 0.7)
        back = ensemble.turn_back([turned], quarter)[0]
        assert back.xyxy == pytest.approx(WIDE) and (back.label, back.conf) == ("date", 0.7)
    assert ensemble.turn_back([Prediction("date", WIDE, 0.7)], 1)[0].xyxy == pytest.approx(
        derived.turn_box(WIDE, 3))                                    # back is the rest of a whole turn


@needs_wbf
def test_wbf_of_identical_boxes_is_that_box():
    same = [[Prediction("date", WIDE, 0.8), Prediction("due", TALL, 0.5)] for _ in range(3)]
    fused = ensemble.fuse(same)
    assert [p.label for p in fused] == ["date", "due"]
    assert fused[0].xyxy == pytest.approx(WIDE, abs=1e-6) and fused[0].conf == pytest.approx(0.8)
    assert fused[1].xyxy == pytest.approx(TALL, abs=1e-6) and fused[1].conf == pytest.approx(0.5)


@needs_wbf
def test_wbf_keeps_a_box_some_passes_missed_at_their_share_and_one_pass_as_it_is():
    fused = ensemble.fuse([[Prediction("date", WIDE, 0.9)], [], []])
    assert len(fused) == 1 and fused[0].conf == pytest.approx(0.3)   # the mean, times 1 of 3
    near = (0.21, 0.40, 0.61, 0.50)
    fused = ensemble.fuse([[Prediction("date", WIDE, 0.9)], [Prediction("date", near, 0.6)]])
    assert len(fused) == 1 and fused[0].conf == pytest.approx(0.75)
    assert fused[0].xyxy[0] == pytest.approx((0.20 * 0.9 + 0.21 * 0.6) / 1.5, abs=1e-6)   # weighted by confidence
    one = [Prediction("date", TALL, 0.2), Prediction("date", WIDE, 0.9)]
    assert ensemble.fuse([one]) == [one[1], one[0]]                    # one pass: unfused, surest first


def photo_with(box, size=(400, 240)) -> Image.Image:
    """A white photo with one black block where `box` says (0..1)."""
    w, h = size
    array = np.full((h, w, 3), 255, np.uint8)
    x1, y1, x2, y2 = box
    array[round(y1 * h):round(y2 * h), round(x1 * w):round(x2 * w)] = 0
    return Image.fromarray(array)


def sees_upright(image: Image.Image, conf: float = 0.9) -> list[Prediction]:
    """A fake detector that finds a dark block only when it's wider than tall."""
    dark = np.asarray(image.convert("L")) < 128
    ys, xs = np.nonzero(dark)
    if not len(xs) or xs.max() - xs.min() <= ys.max() - ys.min():
        return []
    w, h = image.size
    return [Prediction("date", (xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h), conf)]


@needs_wbf
def test_rotation_finds_a_sideways_date_a_model_of_upright_dates_misses():
    image = photo_with(TALL)
    assert sees_upright(image) == []
    turned = [ensemble.predict_turned(sees_upright, image, q) for q in ensemble.TURNS]
    assert [len(t) for t in turned] == [0, 1, 0, 1]                  # seen when turned a quarter either way
    for found in (turned[1][0], turned[3][0]):
        assert iou(found.xyxy, TALL) == pytest.approx(1.0)           # turned back onto the photo
    fused = ensemble.fuse(turned)
    sample = Sample("a.png", Path("a.png"), 400, 240, (Box("date", TALL),))
    assert evaluate([sample], {"a.png": fused}).truth_conf == [pytest.approx(0.45)]   # 2 of 4 turns


def test_each_configuration_fuses_its_passes():
    assert [p.name for p in ensemble.passes([])] == ["s0-q0-1280"]
    assert [p.name for p in ensemble.passes(["ensemble"])] == ["s0-q0-1280", "s1-q0-1280", "s2-q0-1280"]
    assert [p.name for p in ensemble.passes(["rotation", "imgsz-1536"])] == [
        "s0-q0-1536", "s0-q1-1536", "s0-q2-1536", "s0-q3-1536"]
    assert len(ensemble.passes(m2b.ADDONS)) == 12
    every = ensemble.every_pass()
    assert len(every) == len(set(every)) == 24
    assert {p for c in ((), ("ensemble", "rotation"), ("ensemble", "rotation", "imgsz-1536"))
            for p in ensemble.passes(c)} <= set(every)
    assert ensemble.Pass.parse("s2-q3-1536") == ensemble.Pass(2, 3, 1536)
    with pytest.raises(ValueError):
        ensemble.passes(["tta"])


def test_weights_are_used_only_if_they_are_the_ones_hashed_on_the_cluster(monkeypatch, tmp_path):
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    run_dir = runs.start_run("m2b-n1280-ft-s0", {})
    runs.update_run(run_dir, status="finished", checkpoint="/cluster/only/last.pt")
    with pytest.raises(FileNotFoundError, match="copy it"):
        m2b.last_pt("n1280-ft-s0")
    copy = tmp_path / "ckpt" / "m2b" / run_dir.name / "weights" / "last.pt"
    copy.parent.mkdir(parents=True)
    copy.write_bytes(b"not the weights")
    with pytest.raises(RuntimeError, match="isn't the last.pt hashed"):
        m2b.last_pt("n1280-ft-s0")
    monkeypatch.setitem(m2b.LAST_PT_SHA256, "n1280-ft-s0", store.sha256(copy))
    assert m2b.last_pt("n1280-ft-s0") == (copy, run_dir.name)
    # Every trained run is hashed (S12, S13, then the rest in S14); the control's weights are M2's.
    trained = (*m2b.STEP1, *(f"{size}1280-ft" for size in m2b.SIZES), "n1280-ft2", "n1280-ft-gen50k")
    assert set(m2b.LAST_PT_SHA256) == {f"{arm}-s{s}" for arm in trained for s in m2b.SEEDS}
    assert all(len(h) == 64 and int(h, 16) >= 0 for h in m2b.LAST_PT_SHA256.values())


# --- scripts/m2b_inference.py on made-up photos and records ---------------------------------

@pytest.fixture
def inference(load_script, monkeypatch, tmp_path):
    """Two photos (an upright date, a sideways one) as v1-val and as
    real-val; R3's arm's three seeds as fake models that see only upright
    dates (seed 0 surest); the chain's record and each seed's GPU suite."""
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(store, "RUNS_DIR", tmp_path / "store-runs")
    (tmp_path / "runs").mkdir()
    images = {"p1.png": photo_with(WIDE), "p2.png": photo_with(TALL)}
    for name, image in images.items():
        image.save(tmp_path / name)
    v1 = [Sample("p1.png", tmp_path / "p1.png", 400, 240, (Box("expiry", WIDE),), tag="clean"),
          Sample("p2.png", tmp_path / "p2.png", 400, 240, (Box("expiry", TALL),), tag="sideways")]
    real = [Sample(s.id, s.image, 400, 240, (Box("date", s.boxes[0].xyxy),)) for s in v1]
    monkeypatch.setattr(sets, "load_set", lambda name: {"v1-val": v1, "real-val": real}[name])
    monkeypatch.setattr(m2b, "load_v1_val", lambda: v1)
    decided = {"noise_store": {"band": 0.6}, "R3": {"arm": "n1280-ft", "run": "n1280-ft-s0", "threshold": 0.1641}}
    monkeypatch.setattr(m2b, "decisions", lambda root=None: decided)
    m2b.decisions_path().write_text(json.dumps(decided))
    for seed in m2b.SEEDS:
        run_dir = runs.start_run(f"m2b-n1280-ft-s{seed}", {})
        runs.update_run(run_dir, status="finished")
        (run_dir / "suite-m2b.json").write_text(json.dumps(
            {"threshold": 0.1641, "threshold_m2": 0.3, "v1-val": {"map50": {"date": 0.6}},
             "real-val": {"map50": {"date": 0.5}}}))
    monkeypatch.setattr(m2b, "last_pt", lambda run: (Path(f"/w/{run}.pt"), f"x-{run}"))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda path: path))
    confs = {"/w/n1280-ft-s0.pt": 0.9, "/w/n1280-ft-s1.pt": 0.8, "/w/n1280-ft-s2.pt": 0.7,
             "/w/m1280-ft-s0.pt": 0.95}
    monkeypatch.setattr(detector, "predict",
                        lambda model, images, imgsz, device: [sees_upright(im, confs[model]) for im in images])
    return load_script("m2b_inference"), tmp_path


@needs_wbf
def test_the_passes_fusion_r4_and_the_what_if_end_to_end(inference, capsys):
    script, tmp_path = inference
    v1_record = script.predict("v1-val", "cpu")
    real_record = script.predict("real-val", "cpu")
    # Store photos' boxes go to data/store/runs/, never beside the record in results/.
    assert not (v1_record / "passes.json.gz").exists()
    assert (tmp_path / "store-runs" / v1_record.name / "passes.json.gz").exists()
    assert (real_record / "passes.json.gz").exists()
    for record in (v1_record, real_record):
        text = (record / "run.json").read_text()
        assert "p1.png" not in text and "p2.png" not in text
        assert json.loads(text)["config"]["passes"] == [p.name for p in ensemble.every_pass()]
    boxes = json.loads(gzip.decompress((real_record / "passes.json.gz").read_bytes()))["passes"]
    assert boxes["s0-q0-1280"]["p2.png"] == [] and len(boxes["s0-q1-1280"]["p2.png"]) == 1

    out = script.score()
    text = (out / "inference.json").read_text()
    assert "p1.png" not in text and "p2.png" not in text                # counts only
    result = json.loads(text)
    configs = result["configurations"]
    assert list(configs) == ["none", "ensemble", "rotation", "imgsz-1536", "ensemble+rotation",
                             "ensemble+imgsz-1536", "rotation+imgsz-1536", "ensemble+rotation+imgsz-1536"]
    none, rotation = configs["none"], configs["rotation"]
    assert none["v1-val"]["iou0.5"]["dates_found"] == 1 and rotation["v1-val"]["iou0.5"]["dates_found"] == 2
    assert none["v1-val"]["by_tag"]["sideways"]["any_conf"]["expiry_found"] == 0
    assert rotation["v1-val"]["by_tag"]["sideways"]["iou0.5"]["expiry_found"] == 1
    assert sum(b["dates"] for b in rotation["v1-val"]["by_size"].values()) == 2
    assert rotation["v1-val"]["map50"]["date"] > none["v1-val"]["map50"]["date"] + 0.4
    assert rotation["cpu_ms_per_photo"] > none["cpu_ms_per_photo"] > 0
    assert len(configs["ensemble+rotation+imgsz-1536"]["passes"]) == 12
    # R4 with the noise band (0.6 here) keeps nothing, so the record's threshold stays;
    # the what-if (0.03) keeps rotation alone, at its own F2 threshold on v1-val.
    assert result["R4"]["adopted"] == [] and result["R4"]["threshold"] == 0.1641
    assert result["what_if"]["adopted"] == ["rotation"] and result["what_if"]["band"] == 0.03
    assert result["what_if"]["threshold"] == rotation["threshold"]
    assert [s["against"] for s in result["what_if"]["steps"]] == ["none", "none", "rotation"]
    # The AP check: each seed's upright pass against its GPU suite's Ultralytics mAP.
    check = result["ap_check"]
    assert check["agrees"] and set(check["seeds"]) == {"s0", "s1", "s2"}
    assert check["seeds"]["s0"]["real-val"]["ultralytics"] == 0.5
    assert "R4 (band 0.6000): adopts nothing" in capsys.readouterr().out


def test_score_refuses_passes_that_miss_a_photo_and_a_smoke_is_never_scored(inference):
    script, tmp_path = inference
    smoke = script.predict("v1-val", "cpu", limit=1)
    assert smoke.name.endswith("-m2b-inference-passes-v1-val-smoke")
    script.predict("real-val", "cpu")
    with pytest.raises(SystemExit, match="no finished pass record for v1-val"):
        script.score()


def test_n_is_timed_beside_m_without_keeping_a_box(inference):
    script, tmp_path = inference
    record = script.time_models("cpu")
    timing = json.loads((record / "timing.json").read_text())
    assert list(timing["timed"]) == ["n1280-ft-s0@1280", "n1280-ft-s0@1536", "m1280-ft-s0@1280"]
    one = timing["timed"]["m1280-ft-s0@1280"]
    assert len(one["per_photo_s"]) == script.REPEATS and all(len(r) == 2 for r in one["per_photo_s"])
    assert timing["config"]["warmup"] == 2 and timing["config"]["batch"] == 1
    assert "p1.png" not in json.dumps(timing)


def test_torchs_threads_are_read_after_ultralytics_sets_them(inference, monkeypatch):
    """Ultralytics sets torch's threads when it first predicts on a CPU
    (min(8, CPUs - 1)), so a record reads them after the warm-up."""
    import torch

    script, _ = inference
    before, fake = torch.get_num_threads(), detector.predict

    def predict_and_set(model, images, imgsz, device):
        torch.set_num_threads(3)
        return fake(model, images, imgsz, device)

    monkeypatch.setattr(detector, "predict", predict_and_set)
    try:
        torch.set_num_threads(1)
        record = script.predict("v1-val", "cpu", limit=1)
        timing = script.time_models("cpu")
    finally:
        torch.set_num_threads(before)
    assert json.loads((record / "run.json").read_text())["config"]["torch_threads"] == 3
    assert json.loads((record / "run.json").read_text())["torch_threads_used"] == 3
    assert json.loads((timing / "timing.json").read_text())["config"]["torch_threads"] == 3
