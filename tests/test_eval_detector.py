"""scripts/eval_detector.py picks each milestone's weights and threshold,
checks the weights against their hash, and scores a test set only once
(docs/m2-synthetic-generator.md, S7); store set v2 once, for the final
test's records only (docs/m2b-store-and-size.md, S14)."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

from bestbefore import detector, final_test, m1, m2, m2b, runs, store
from bestbefore.detection_eval import Prediction
from bestbefore.samples import Box, Sample

M2_RUN = "20261025-003853-m2-gen-rot-1280-s0"
CONTROL = "20261112-000000-m2b-n1280-s0"
FT = "20261113-092835-m2b-n1280-ft-s0"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def trained(monkeypatch, tmp_path):
    """One finished M1 run and two finished, scored M2 runs, and M2b's
    control (M2's gen-rot-1280-s0, as it is) and one fine-tuned run, as
    the laptop has them once their weights are copied; each weights file
    with its hash recorded, but M2's base-s0, which has none."""
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(store, "RUNS_DIR", tmp_path / "store-runs")
    monkeypatch.setattr(final_test, "RUNS_DIR", tmp_path / "store-v2-runs")
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    m1_run = tmp_path / "runs" / "20261015-211752-m1-n640"
    m2_run = tmp_path / "runs" / "20261024-121631-m2-base-s0"
    files = {
        m1_run / "run.json": {"config": {"imgsz": 640, "train": "train_synth,train_real"}},
        m1_run / "val.json": {"threshold": 0.7279},
        m2_run / "run.json": {"status": "finished", "config": {"imgsz": 640}},
        m2_run / "suite.json": {"threshold": 0.18234567891234},
        m2_run / "val.json": {"threshold": 0.9},            # never M2's threshold
        tmp_path / "runs" / M2_RUN / "run.json": {"status": "finished", "config": {"imgsz": 1280}},
        tmp_path / "runs" / M2_RUN / "suite.json": {"threshold": 0.342413991689682},
        tmp_path / "runs" / CONTROL / "run.json": {
            "status": "finished", "config": {"imgsz": 1280, "seed": 0, "trained": False},
            "weights_from": M2_RUN, "checkpoint": "/cluster/only/m2/last.pt",
            "checkpoint_sha256": sha(b"m2 weights")},
        tmp_path / "runs" / CONTROL / "suite-m2b.json": {"threshold": 0.0054761, "threshold_m2": 0.342413991689682},
        tmp_path / "runs" / FT / "run.json": {"status": "finished", "config": {"imgsz": 1280},
                                              "checkpoint": "/cluster/only/m2b/last.pt"},
        tmp_path / "runs" / FT / "suite-m2b.json": {"threshold": 0.1640664041042328, "threshold_m2": 0.25},
    }
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    for weights, data in ((tmp_path / "ckpt" / "m1" / m1_run.name / "weights" / "best.pt", b"weights"),
                          (tmp_path / "ckpt" / "m2" / m2_run.name / "weights" / "last.pt", b"weights"),
                          (tmp_path / "ckpt" / "m2" / M2_RUN / "weights" / "last.pt", b"m2 weights"),
                          (tmp_path / "ckpt" / "m2b" / FT / "weights" / "last.pt", b"ft weights")):
        weights.parent.mkdir(parents=True)
        weights.write_bytes(data)
    monkeypatch.setitem(m1.BEST_PT_SHA256, "n640", sha(b"weights"))
    monkeypatch.setitem(m2.LAST_PT_SHA256, "gen-rot-1280-s0", sha(b"m2 weights"))
    monkeypatch.setitem(m2b.LAST_PT_SHA256, "n1280-ft-s0", sha(b"ft weights"))
    return tmp_path


def test_an_m1_run_uses_best_pt_and_its_validation_threshold(load_script, trained):
    ev = load_script("eval_detector")
    run = ev.find("n640")
    assert run.milestone == "m1" and run.threshold == 0.7279
    assert run.weights == trained / "ckpt" / "m1" / "20261015-211752-m1-n640" / "weights" / "best.pt"
    assert run.train == ["train_synth", "train_real"]
    assert run.sha256 == sha(b"weights") and run.threshold_from.startswith("val.json")
    assert ev.find("m1-n640") == run


def test_an_m2_run_uses_last_pt_and_its_suite_threshold_unrounded(load_script, trained):
    ev = load_script("eval_detector")
    run = ev.find("m2-base-s0")
    assert run.milestone == "m2" and run.threshold == 0.18234567891234
    assert run.weights == trained / "ckpt" / "m2" / "20261024-121631-m2-base-s0" / "weights" / "last.pt"
    assert run.checkpoints.parent.name == "m2"
    assert run.sha256 is None                       # no hash recorded: base-s0 isn't the final test's
    with pytest.raises(SystemExit, match="no finished run called 'm2-rot-s0'"):
        ev.find("m2-rot-s0")
    (run.run_dir / "suite.json").unlink()
    with pytest.raises(SystemExit, match="no suite.json yet.*score_suite.py --run base-s0"):
        ev.find("m2-base-s0")


def test_an_m2b_run_uses_its_threshold_store_and_its_copied_last_pt(load_script, trained):
    ev = load_script("eval_detector")
    run = ev.find("m2b-n1280-ft-s0")
    assert run.milestone == "m2b" and run.imgsz == 1280
    assert run.threshold == 0.1640664041042328 and run.threshold_m2 == 0.25      # unrounded, both
    assert run.threshold_from.startswith("suite-m2b.json: threshold_store")
    assert run.weights == trained / "ckpt" / "m2b" / FT / "weights" / "last.pt"   # the cluster's path isn't here
    assert run.sha256 == sha(b"ft weights")
    (trained / "runs" / FT / "suite-m2b.json").unlink()
    with pytest.raises(SystemExit, match="no suite-m2b.json yet.*--suite m2b --run n1280-ft-s0"):
        ev.find("m2b-n1280-ft-s0")


def test_the_control_reads_m2s_weights_and_hash_at_its_own_threshold(load_script, trained):
    ev = load_script("eval_detector")
    control, m2_run = ev.find("m2b-n1280-s0"), ev.find("m2-gen-rot-1280-s0")
    assert control.weights == m2_run.weights == trained / "ckpt" / "m2" / M2_RUN / "weights" / "last.pt"
    assert control.sha256 == m2_run.sha256 == sha(b"m2 weights")
    assert (control.threshold, m2_run.threshold) == (0.0054761, 0.342413991689682)
    record = trained / "runs" / CONTROL / "run.json"
    record.write_text(json.dumps({**json.loads(record.read_text()), "checkpoint_sha256": sha(b"other")}))
    with pytest.raises(SystemExit, match="record holds"):
        ev.find("m2b-n1280-s0")


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


def test_weights_that_arent_the_ones_hashed_are_refused(load_script, trained):
    ev = load_script("eval_detector")
    run = ev.find("m2b-n1280-ft-s0")
    assert ev.output(run, "real-test", again=False).name == "real-test.json"
    run.weights.write_bytes(b"retrained")
    with pytest.raises(SystemExit, match="isn't the file hashed for it"):
        ev.output(run, "real-test", again=False)


def test_store_v2_is_scored_once_and_only_for_the_final_tests_records(load_script, trained, monkeypatch):
    ev = load_script("eval_detector")
    assert "store-v2" not in ev.NAMES and "store-v2" in ev.SETS     # no other script loads v2 by name
    ready = ev.find("m2b-n1280-ft-s0")
    out = ev.output(ready, "store-v2", again=False)
    assert out == ready.run_dir / "store-v2.json"
    assert any("--again is refused" in p for p in ev.problems(ready, "store-v2", again=True))
    assert any("final test's records" in p for p in ev.problems(ev.find("m2-base-s0"), "store-v2", False))
    out.write_text("{}")
    for again in (False, True):
        with pytest.raises(SystemExit, match="already scored"):
            ev.output(ready, "store-v2", again=again)
    out.unlink()
    runs.start_run("detector-store-v2-m2b-n1280-ft-s0", {}, root=final_test.RUNS_DIR)
    with pytest.raises(SystemExit, match="per-photo folder"):
        ev.output(ready, "store-v2", again=False)
    monkeypatch.delitem(m2b.LAST_PT_SHA256, "n1280-ft-s0")
    with pytest.raises(SystemExit, match="no SHA-256"):
        ev.output(ev.find("m2b-n1280-ft-s0"), "store-v2", again=False)


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


def test_store_v2_boxes_stay_in_its_own_folder_exact_with_the_keyword_boxes(load_script, trained):
    ev = load_script("eval_detector")
    preds = {"testset2/a.jpeg": [Prediction("date", (0.123456789, 0.2, 0.3, 0.4), 0.72794321),
                                 Prediction("prod", (0.1, 0.1, 0.2, 0.2), 0.6)]}
    run = ev.find("n640")                              # exact even for M1 on v2
    kept = ev.keep_private(run, preds, "store-v2")
    assert kept.parent == final_test.RUNS_DIR and kept.name.endswith("-detector-store-v2-n640")
    assert json.loads((kept / "predictions.json").read_text()) == {
        "testset2/a.jpeg": [[0.123456789, 0.2, 0.3, 0.4, 0.72794321]]}
    assert json.loads((kept / "keyword_boxes.json").read_text()) == {
        "testset2/a.jpeg": {"due": [], "prod": [[0.1, 0.1, 0.2, 0.2, 0.6]], "code": []}}
    assert json.loads((kept / "run.json").read_text())["config"]["weights_sha256"] == sha(b"weights")
    for root in (trained / "runs", store.RUNS_DIR):
        assert not list(root.glob("*detector-store*"))


@pytest.fixture
def one_photo_v2():
    """Store set v2 as one made-up photo with an expiry, in an unseen design."""
    sample = Sample(id="testset2/a.jpeg", image=Path("a.jpeg"), width=1200, height=1600,
                    boxes=(Box("expiry", (0.1, 0.5, 0.4, 0.52)),), tag="clean")
    return final_test.V2([sample], {sample.id: final_test.UNSEEN},
                         {sample.id: ["numbers: print 1.5 to 2.5%"]})


def test_runs_that_share_weights_share_one_pass_and_say_so(load_script, trained, monkeypatch, one_photo_v2):
    ev = load_script("eval_detector")
    calls = []

    def predict(weights, samples, imgsz, device):
        calls.append(weights)
        return {"testset2/a.jpeg": [Prediction("date", (0.1, 0.5, 0.4, 0.52), 0.2)]}

    def score(weights, samples, imgsz, conf, predictions, **kwargs):
        from bestbefore.detection_eval import evaluate, summary
        at_05 = evaluate(samples, predictions, 0.5)
        return {"threshold": round(conf, 4), "iou0.5": summary(at_05, conf),
                "by_tag": {"clean": {"iou0.5": summary(at_05, conf, {"clean"})}}}

    monkeypatch.setattr(detector, "predict_checkpoint", predict)
    monkeypatch.setattr(detector, "score", score)
    cache = {}
    first = ev.score_run(ev.find("m2-gen-rot-1280-s0"), "store-v2", one_photo_v2.samples, "cpu", False,
                         cache, one_photo_v2)
    second = ev.score_run(ev.find("m2b-n1280-s0"), "store-v2", one_photo_v2.samples, "cpu", False,
                          cache, one_photo_v2)
    assert len(calls) == 1                                   # one pass, two thresholds
    assert first["scored"]["predicted_by"] == second["scored"]["predicted_by"] == "m2-gen-rot-1280-s0"
    assert second["scored"]["predict_seconds"] == 0.0
    assert (first["iou0.5"]["expiry_found"], second["iou0.5"]["expiry_found"]) == (0, 1)   # 0.2: under M2's, over the control's
    assert second["by_design"][final_test.UNSEEN]["iou0.5"]["expiry_found"] == 1
    assert second["by_cause"]["numbers: print 1.5 to 2.5%"]["iou0.5"]["expiry_found"] == 1
    assert second["ap50"]["date"] is not None and second["scored"]["weights_sha256"] == sha(b"m2 weights")
    for name, run_dir in (("m2-gen-rot-1280-s0", M2_RUN), ("m2b-n1280-s0", CONTROL)):
        assert json.loads((trained / "runs" / run_dir / "store-v2.json").read_text())["scored"]["set"] == "store-v2"
        assert runs.latest_run(f"detector-store-v2-{name}", final_test.RUNS_DIR) is not None
    with pytest.raises(SystemExit, match="already scored"):    # and never again
        ev.score_run(ev.find("m2b-n1280-s0"), "store-v2", one_photo_v2.samples, "cpu", False, {}, one_photo_v2)


def test_expdate_test_gives_an_m2b_run_threshold_m2s_counts_beside_its_own(load_script, trained, monkeypatch):
    ev = load_script("eval_detector")
    sample = Sample(id="test_1.jpg", image=Path("test_1.jpg"), width=1000, height=1000,
                    boxes=(Box("exp", (0.1, 0.5, 0.4, 0.55)),))
    monkeypatch.setattr(detector, "predict_checkpoint", lambda *a, **k: {
        "test_1.jpg": [Prediction("date", (0.1, 0.5, 0.4, 0.55), 0.2)]})
    monkeypatch.setattr(detector, "score", lambda *a, conf, **k: {"threshold": round(conf, 4)})
    monkeypatch.setattr(detector, "dataset_yaml", lambda *a, **k: Path("data.yaml"))
    got = ev.score_run(ev.find("m2b-n1280-ft-s0"), "real-test", [sample], "cpu", False)
    assert got["at_threshold_m2"]["threshold"] == 0.25
    assert got["at_threshold_m2"]["iou0.5"]["expiry_found"] == 0          # 0.2 is under M2's 0.25
    assert got["ap50_ours"]["date"] is not None
    for root in (final_test.RUNS_DIR, store.RUNS_DIR):        # ExpDate photos aren't private
        assert not root.exists() or not any(root.iterdir())


def run_main(ev, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["eval_detector.py", *argv])
    with pytest.raises(SystemExit) as stop:
        ev.main()
    return stop.value.code


def written(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def test_a_dry_run_checks_every_run_and_writes_nothing(load_script, trained, monkeypatch, one_photo_v2, capsys):
    ev = load_script("eval_detector")
    monkeypatch.setattr(final_test, "load", lambda: one_photo_v2)
    monkeypatch.setattr(detector, "predict_checkpoint", lambda *a, **k: pytest.fail("a dry run predicted"))
    before = written(trained)
    ready = ("m2-gen-rot-1280-s0", "m2b-n1280-s0", "m2b-n1280-ft-s0")
    assert run_main(ev, monkeypatch, "--set", "store-v2", "--dry-run", "--runs", *ready) == 0
    shown = capsys.readouterr().out
    assert shown.count("| ready |") == 3 and "m2-gen-rot-1280-s0's" in shown      # the control shares M2's pass
    assert "| design | unseen | 1 | 1 | 1 |" in shown
    assert run_main(ev, monkeypatch, "--set", "store-v2", "--dry-run", "--runs", *ready, "m2-base-s0") == 1
    assert "final test's records" in capsys.readouterr().out
    assert run_main(ev, monkeypatch, "--set", "store-v2", "--dry-run", "--runs", *ready, "m2b-s1280-s0") == 1
    assert "| m2b-s1280-s0 | – |" in capsys.readouterr().out          # no such run here: listed, not a crash
    assert run_main(ev, monkeypatch, "--set", "store-v2", "--dry-run", "--again", "--runs", *ready) == 1
    assert written(trained) == before                          # nothing written, nothing private


def test_a_problem_with_any_run_stops_them_all_before_any_is_scored(load_script, trained, monkeypatch, one_photo_v2):
    ev = load_script("eval_detector")
    monkeypatch.setattr(final_test, "load", lambda: one_photo_v2)
    monkeypatch.setattr(detector, "predict_checkpoint", lambda *a, **k: pytest.fail("scored before checking"))
    before = written(trained)
    code = run_main(ev, monkeypatch, "--set", "store-v2", "--runs", "m2b-n1280-ft-s0", "m2b-n1280-ft-s9")
    assert "nothing scored" in code and "m2b-n1280-ft-s9" in code
    assert run_main(ev, monkeypatch, "--set", "store-v2", "--runs", "m2b-n1280-ft-s0", "m2b-n1280-ft-s0") \
        == "a run is named twice"
    assert written(trained) == before
