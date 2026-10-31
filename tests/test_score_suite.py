"""scripts/score_suite.py: --out leaves the run's own suite.json, its record,
untouched, and the weights are found on either machine
(docs/m2-synthetic-generator.md, S8)."""
import json
import sys

import pytest

from bestbefore import detector, m2, runs


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
