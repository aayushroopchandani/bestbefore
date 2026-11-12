"""scripts/score_suite.py: --out leaves the run's own suite.json, its record,
untouched, and the weights are found on either machine
(docs/m2-synthetic-generator.md, S8)."""
import json
import sys
from pathlib import Path

import pytest

from bestbefore import detector, m2, m2b, runs


def _summary(found: int, of: int, targets: tuple[int, int] | None = None) -> dict:
    s = {"photos": 10, "dates_found": found, "dates": of, "precision": 0.9,
         "false_boxes_per_photo": 0.1}
    if targets:
        s |= {"expiry_found": targets[0], "expiry_dates": targets[1]}
    return s


def _suite(threshold: float, found: int, synth_target: int) -> dict:
    """A suite.json as score_suite writes it, with the counts given."""
    def entry(found, targets=None):
        return {"iou0.5": _summary(found, 187, targets), "iou0.3": _summary(found, 187, targets),
                "any_conf": _summary(187, 187, targets), "map50": {"date": found / 187}}
    suite = {"threshold": threshold, "threshold_rule": "best F2", "f2_at_threshold": 0.9,
             "missing": {}, **{name: entry(found) for name in m2.THRESHOLD_SETS}}
    suite["synth-val"] = entry(found, (synth_target, 300))
    suite["synth-val"]["by_tag"] = {"clean": {level: _summary(found, 187, (synth_target, 300))
                                              for level in ("iou0.5", "iou0.3", "any_conf")}}
    return suite


@pytest.fixture
def run(monkeypatch, tmp_path):
    """A finished cluster run whose suite.json the GPU wrote, and the laptop's
    copy of its weights."""
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    run_dir = tmp_path / "runs" / "20261025-003853-m2-gen-rot-1280-s0"
    run_dir.mkdir(parents=True)
    cluster = f"/home/someone/bestbefore-data/checkpoints/m2/{run_dir.name}/weights/last.pt"
    (run_dir / "run.json").write_text(json.dumps(
        {"status": "finished", "checkpoint": cluster, "config": {"imgsz": 1280}}))
    gpu = _suite(0.342413991689682, 180, 280) | {"scored": {"device": None, "date": "2026-10-25"}}
    (run_dir / "suite.json").write_text(json.dumps(gpu))
    laptop = tmp_path / "ckpt" / "m2" / run_dir.name / "weights" / "last.pt"
    laptop.parent.mkdir(parents=True)
    laptop.write_bytes(b"weights")
    return run_dir, laptop


def test_weights_are_the_recorded_path_where_it_exists_else_the_laptop_copy(load_script, run, tmp_path):
    script = load_script("score_suite")
    run_dir, laptop = run
    record = json.loads((run_dir / "run.json").read_text())
    assert script.weights(run_dir, record) == laptop                 # the cluster's path isn't here
    here = tmp_path / "elsewhere" / "last.pt"
    here.parent.mkdir()
    here.write_bytes(b"weights")
    assert script.weights(run_dir, {**record, "checkpoint": str(here)}) == here   # as on the cluster


def test_out_leaves_the_runs_own_suite_and_compares(load_script, run, monkeypatch, tmp_path, capsys):
    script = load_script("score_suite")
    run_dir, laptop = run
    record = (run_dir / "suite.json").read_bytes()
    scored = {}

    def fake_score_suite(checkpoint, imgsz, out_dir, device):
        scored.update(checkpoint=checkpoint, imgsz=imgsz, out_dir=out_dir, device=device)
        return _suite(0.3424, 179, 281)

    monkeypatch.setattr(detector, "score_suite", fake_score_suite)
    out = run_dir / "device-check-cpu"
    monkeypatch.setattr(sys, "argv", ["score_suite.py", "--run", "gen-rot-1280-s0", "--device", "cpu",
                                      "--out", str(out)])
    script.main()

    assert (run_dir / "suite.json").read_bytes() == record           # the GPU's record, untouched
    assert scored == {"checkpoint": laptop, "imgsz": 1280, "out_dir": out, "device": "cpu"}
    written = json.loads((out / "suite.json").read_text())
    assert written["threshold"] == 0.3424 and written["scored"]["device"] == "cpu"
    text = capsys.readouterr().out
    assert "| all | threshold | 0.342414 | 0.342400 | -0.000014 |" in text
    assert "| real-val | dates found | 180/187 | 179/187 | -1 |" in text
    assert "| synth-val: clean | targets found | 280/300 | 281/300 | +1 |" in text

    # --out naming the run's own folder would replace its record: refused.
    monkeypatch.setattr(sys, "argv", ["score_suite.py", "--run", "gen-rot-1280-s0", "--out", str(run_dir)])
    with pytest.raises(SystemExit, match="its record"):
        script.main()
    assert (run_dir / "suite.json").read_bytes() == record


def test_without_out_the_suite_goes_into_the_run_as_before(load_script, run, monkeypatch, capsys):
    script = load_script("score_suite")
    run_dir, _ = run
    monkeypatch.setattr(detector, "score_suite", lambda checkpoint, imgsz, out_dir, device: _suite(0.2, 187, 300))
    monkeypatch.setattr(sys, "argv", ["score_suite.py", "--run", "gen-rot-1280-s0"])
    script.main()
    assert json.loads((run_dir / "suite.json").read_text())["threshold"] == 0.2
    assert "beside this one" not in capsys.readouterr().out          # nothing to compare with


def test_m2b_runs_are_scored_from_their_own_records_and_folder(load_script, monkeypatch, tmp_path):
    script = load_script("score_suite")
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    run_dir = tmp_path / "runs" / "20261102-010101-m2b-s1280-s0"
    run_dir.mkdir(parents=True)
    record = {"status": "finished", "checkpoint": "/cluster/ckpt/m2b/last.pt", "config": {"imgsz": 1280}}
    (run_dir / "run.json").write_text(json.dumps(record))
    laptop = tmp_path / "ckpt" / "m2b" / run_dir.name / "weights" / "last.pt"
    assert script.weights(run_dir, record, "m2b") == laptop       # M2b's own folder
    scored = {}

    def score_suite(checkpoint, imgsz, out_dir, device):
        scored.update(checkpoint=checkpoint, imgsz=imgsz, out_dir=out_dir)
        return _suite(0.2, 187, 300)

    monkeypatch.setattr(detector, "score_suite", score_suite)
    monkeypatch.setattr(sys, "argv", ["score_suite.py", "--milestone", "m2b", "--run", "s1280-s0"])
    script.main()
    assert scored == {"checkpoint": laptop, "imgsz": 1280, "out_dir": run_dir}
    assert json.loads((run_dir / "suite.json").read_text())["threshold"] == 0.2
    # M2's lookup, the default, doesn't take an M2b run.
    monkeypatch.setattr(sys, "argv", ["score_suite.py", "--run", "s1280-s0"])
    with pytest.raises(SystemExit, match="no finished run m2-s1280-s0"):
        script.main()


# --- The M2b suite (docs/m2b-store-and-size.md, S10) ----------------------------------

def _suite_m2b(threshold: float, found: int) -> dict:
    """A suite-m2b.json as score_suite writes it, with the counts given."""
    def entry(found, targets=None):
        return {"iou0.5": _summary(found, 187, targets), "iou0.3": _summary(found, 187, targets),
                "any_conf": _summary(187, 187, targets), "map50": {"date": found / 187},
                "at_threshold_m2": {"iou0.5": _summary(found - 1, 187, targets),
                                    "iou0.3": _summary(found - 1, 187, targets)}}
    return {"suite": "m2b", "threshold": threshold, "threshold_rule": "best F2 for dates on v1-val",
            "f2_at_threshold": 0.9, "threshold_m2": 0.34, "threshold_m2_rule": "best F2 ..., pooled",
            "f2_at_threshold_m2": 0.95, "missing": {}, **{name: entry(found) for name in m2b.SUITE}}


@pytest.fixture
def control(monkeypatch, tmp_path):
    """The control's record as the cluster wrote it (m2b.write_control): M2's
    last.pt, by the cluster's path; and the laptop's copy of M2's weights."""
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    m2_run = "20261025-003853-m2-gen-rot-1280-s0"
    run_dir = tmp_path / "runs" / "20261108-170142-m2b-n1280-s0"
    run_dir.mkdir(parents=True)
    cluster = f"/home/someone/bestbefore-data/checkpoints/m2/{m2_run}/weights/last.pt"
    (run_dir / "run.json").write_text(json.dumps(
        {"status": "finished", "checkpoint": cluster, "weights_from": m2_run,
         "config": {"arm": "n1280", "imgsz": 1280, "weights_from": m2_run}}))
    (run_dir / "suite.json").write_text(json.dumps({"threshold": 0.3424}))     # M2's suite, as written before
    laptop = tmp_path / "ckpt" / "m2" / m2_run / "weights" / "last.pt"
    laptop.parent.mkdir(parents=True)
    laptop.write_bytes(b"M2's weights")
    return run_dir, laptop


def test_the_m2b_suite_goes_to_its_own_file_from_the_controls_weights(load_script, control, monkeypatch,
                                                                       capsys):
    script = load_script("score_suite")
    run_dir, laptop = control
    m2_suite = (run_dir / "suite.json").read_bytes()
    assert script.weights(run_dir, json.loads((run_dir / "run.json").read_text()), "m2b") == laptop
    scored = {}

    def score_suite_m2b(checkpoint, imgsz, out_dir, device):
        scored.update(checkpoint=checkpoint, imgsz=imgsz, out_dir=out_dir)
        return _suite_m2b(0.21, 180)

    monkeypatch.setattr(detector, "score_suite_m2b", score_suite_m2b)
    monkeypatch.setattr(detector, "score_suite", lambda *a: pytest.fail("M2's suite isn't asked for"))
    monkeypatch.setattr(sys, "argv", ["score_suite.py", "--milestone", "m2b", "--suite", "m2b",
                                      "--run", "n1280-s0"])
    script.main()
    assert scored == {"checkpoint": laptop, "imgsz": 1280, "out_dir": run_dir}
    written = json.loads((run_dir / "suite-m2b.json").read_text())
    assert (written["threshold"], written["threshold_m2"]) == (0.21, 0.34)
    assert (run_dir / "suite.json").read_bytes() == m2_suite                  # M2's suite stays
    out = capsys.readouterr().out
    assert "M2's threshold 0.3400" in out and "| v1-val | 10 | 180/187 | 179/187 | 187/187 |" in out
    # The M2b suite is for M2b's runs.
    monkeypatch.setattr(sys, "argv", ["score_suite.py", "--suite", "m2b", "--run", "gen-rot-1280-s0"])
    with pytest.raises(SystemExit):
        script.main()
    # --out compares with the run's own suite-m2b.json, set by set.
    monkeypatch.setattr(detector, "score_suite_m2b", lambda checkpoint, imgsz, out_dir, device: _suite_m2b(0.2, 179))
    monkeypatch.setattr(sys, "argv", ["score_suite.py", "--milestone", "m2b", "--suite", "m2b",
                                      "--run", "n1280-s0", "--out", str(run_dir / "device-check-cpu")])
    script.main()
    out = capsys.readouterr().out
    assert "The run's own suite-m2b.json" in out and "| v1-val | dates found | 180/187 | 179/187 | -1 |" in out
    assert json.loads((run_dir / "suite-m2b.json").read_text()) == written


def test_score_suite_m2b_scores_every_set_at_two_thresholds(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from bestbefore import sets
    from bestbefore.detection_eval import Prediction
    from bestbefore.samples import Box, Sample

    monkeypatch.setenv("EXPDATE_DIR", str(tmp_path / "expdate"))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda path: "a model"))
    date = (0.1, 0.1, 0.3, 0.14)
    lists = {}
    for name in m2b.SUITE:
        lists[name] = tmp_path / "lists" / f"{name}.txt"
        lists[name].parent.mkdir(exist_ok=True)
        lists[name].write_text("./x.jpg\n")
    samples = {name: [Sample(f"{name}.jpg", Path("/x.jpg"), 1000, 800, (Box("date", date),))]
               for name in m2b.SUITE}
    samples["synth-val-v2"] = [Sample("t.jpg", Path("/t.jpg"), 1000, 800, (Box("exp", date),), tag="glued")]
    samples["v1-val"] = [Sample("v1-0004.jpg", Path("/v.jpg"), 1000, 800,
                                (Box("expiry", date), Box("mfg", (0.5, 0.5, 0.52, 0.6))), tag="sideways")]
    confidences = {"v1-0004.jpg": (0.6, 0.3)}          # every other photo's date: 0.8

    def predict(model, given, imgsz, device):
        return {s.id: [Prediction("date", b.xyxy, c) for b, c in zip(s.boxes, confidences.get(s.id, (0.8,)))]
                for s in given}

    validated = []

    def val_metrics(model, data, split, imgsz, device, project, name, conf):
        validated.append((name, data, project, conf))
        return {"map50": {"date": 0.9}, "map50_95": {"date": 0.7}, "ultralytics_date_recall_at_threshold": 1.0}

    monkeypatch.setattr(m2b, "suite_missing", lambda name: None)
    monkeypatch.setattr(m2b, "suite_list", lambda name: lists[name])
    monkeypatch.setattr(sets, "load_set", lambda name: samples[name])
    monkeypatch.setattr(detector, "predict_samples", predict)
    monkeypatch.setattr(detector, "_val_metrics", val_metrics)
    out_dir = tmp_path / "runs" / "20261108-170142-m2b-n1280-s0"
    suite = detector.score_suite_m2b(Path("/m2/last.pt"), 1280, out_dir)
    # Two thresholds: v1-val's own (keeping both its dates), and M2's on the real sets pooled.
    assert (suite["threshold"], suite["threshold_m2"]) == (0.3, 0.8)
    v1 = suite["v1-val"]
    assert (v1["iou0.5"]["dates_found"], v1["iou0.5"]["expiry_found"]) == (2, 1)
    assert v1["at_threshold_m2"]["iou0.5"]["dates_found"] == 0
    assert set(v1["by_tag"]) == {"sideways"} and v1["by_tag"]["sideways"]["at_threshold_m2"]
    assert v1["by_size"] == {      # the expiry's box is 3.2% of the long side, the mfg's 2.0%
        "under 1.5%": {"dates": 0, "found": 0, "found_at_threshold_m2": 0, "boxed_at_any_confidence": 0},
        "1.5-2.5%": {"dates": 1, "found": 1, "found_at_threshold_m2": 0, "boxed_at_any_confidence": 1},
        "2.5% and up": {"dates": 1, "found": 1, "found_at_threshold_m2": 0, "boxed_at_any_confidence": 1}}
    assert suite["synth-val-v2"]["by_tag"]["glued"]["iou0.5"]["expiry_found"] == 1
    assert "by_size" not in suite["real-val"] and v1["map50"]["date"] == 0.9
    # Ultralytics validates each set from the run's own folder, at the M2b suite's threshold.
    assert [n for n, *_ in validated] == list(m2b.SUITE)
    assert {(project, conf) for _, _, project, conf in validated} == {(out_dir / "suite-m2b" / "val", 0.3)}
    assert all(data == out_dir / "suite-m2b" / f"{n}.yaml" for n, data, *_ in validated)
    assert json.dumps(str(lists["v1-val"])) in (out_dir / "suite-m2b" / "v1-val.yaml").read_text()
    # A set not made: nothing is scored.
    monkeypatch.setattr(m2b, "suite_missing", lambda name: "not made yet" if name == "synth-val-v2" else None)
    with pytest.raises(FileNotFoundError, match="synth-val-v2 not made yet"):
        detector.score_suite_m2b(Path("/m2/last.pt"), 1280, out_dir)
