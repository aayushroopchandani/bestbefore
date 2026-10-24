"""scripts/eval_detector.py picks each milestone's weights and threshold,
and scores a test set only once (docs/m2-synthetic-generator.md, S7)."""
import json

import pytest

from bestbefore import runs, store
from bestbefore.detection_eval import Prediction


@pytest.fixture
def trained(monkeypatch, tmp_path):
    """One finished M1 run and one finished, scored M2 run, as the laptop
    has them once their weights are copied."""
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(store, "RUNS_DIR", tmp_path / "store-runs")
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    m1_run = tmp_path / "runs" / "20261015-211752-m1-n640"
    m2_run = tmp_path / "runs" / "20261024-121631-m2-base-s0"
    files = {
        m1_run / "run.json": {"config": {"imgsz": 640, "train": "train_synth,train_real"}},
        m1_run / "val.json": {"threshold": 0.7279},
        m2_run / "run.json": {"status": "finished", "config": {"imgsz": 640}},
        m2_run / "suite.json": {"threshold": 0.18234567891234},
        m2_run / "val.json": {"threshold": 0.9},            # never M2's threshold
    }
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    for weights in (tmp_path / "ckpt" / "m1" / m1_run.name / "weights" / "best.pt",
                    tmp_path / "ckpt" / "m2" / m2_run.name / "weights" / "last.pt"):
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"weights")
    return tmp_path


def test_an_m1_run_uses_best_pt_and_its_validation_threshold(load_script, trained):
    ev = load_script("eval_detector")
    run = ev.find("n640")
    assert run.milestone == "m1" and run.threshold == 0.7279
    assert run.weights == trained / "ckpt" / "m1" / "20261015-211752-m1-n640" / "weights" / "best.pt"
    assert run.train == ["train_synth", "train_real"]
    assert ev.find("m1-n640") == run


def test_an_m2_run_uses_last_pt_and_its_suite_threshold_unrounded(load_script, trained):
    ev = load_script("eval_detector")
    run = ev.find("m2-base-s0")
    assert run.milestone == "m2" and run.threshold == 0.18234567891234
    assert run.weights == trained / "ckpt" / "m2" / "20261024-121631-m2-base-s0" / "weights" / "last.pt"
    assert run.checkpoints.parent.name == "m2"
    with pytest.raises(SystemExit, match="no finished run called 'm2-rot-s0'"):
        ev.find("m2-rot-s0")
    (run.run_dir / "suite.json").unlink()
    with pytest.raises(SystemExit, match="no suite.json yet.*score_suite.py --run base-s0"):
        ev.find("m2-base-s0")


@pytest.mark.parametrize("name", ["n640", "m2-base-s0"])
def test_a_test_set_is_scored_once_for_either_milestone(load_script, trained, name):
    ev = load_script("eval_detector")
    run = ev.find(name)
    for set_name in ("store", "real-test"):
        out = ev.output(run, set_name, again=False)
        out.write_text("{}")
        with pytest.raises(SystemExit, match="already scored"):
            ev.output(run, set_name, again=False)
        assert ev.output(run, set_name, again=True) == out
    (run.run_dir / "real-val.json").write_text("{}")
    assert ev.output(run, "real-val", again=False).name == "real-val.json"   # validation: any time
    run.weights.unlink()
    with pytest.raises(SystemExit, match="copy it from the cluster"):
        ev.output(run, "real-val", again=False)


def test_store_predictions_stay_private_rounded_for_m1_and_exact_for_m2(load_script, trained):
    ev = load_script("eval_detector")
    preds = {"samples1/a.jpg": [Prediction("date", (0.123456789, 0.2, 0.3, 0.4), 0.72794321),
                                Prediction("due", (0.1, 0.1, 0.2, 0.2), 0.9)]}
    for name, box in (("n640", [0.12346, 0.2, 0.3, 0.4, 0.7279]),
                      ("m2-base-s0", [0.123456789, 0.2, 0.3, 0.4, 0.72794321])):
        run = ev.find(name)
        ev.keep_private(run, preds)
        kept = runs.latest_run(f"detector-store-{name}", store.RUNS_DIR)
        assert json.loads((kept / "predictions.json").read_text()) == {"samples1/a.jpg": [box]}
        assert json.loads((kept / "run.json").read_text())["config"] == {
            "run": run.run_dir.name, "threshold": run.threshold}
    assert not list((trained / "runs").glob("*detector-store*"))    # nothing private in results/
