"""PP-OCRv6 small: its spec, its ONNX path, its PyTorch module and the
fine-tuning route (reader/ppocr.py, reader/onnx_run.py's Spec,
docs/m3-date-reader.md "PP-OCRv6 small", S6).

The tests marked `pretrained` need the official files fetched in S1
($CHECKPOINT_DIR/pretrained/PP-OCRv6_small_rec_onnx, _safetensors) and
transformers (the vlm extra); they skip where either is missing."""
import dataclasses
import importlib.util
import json

import numpy as np
import pytest
import torch

from bestbefore import m3
from bestbefore.reader import charset, ctc, onnx_run, ppocr, train


def _have_pretrained() -> bool:
    try:
        root = ppocr.pretrained()
    except Exception:
        return False
    return (importlib.util.find_spec("transformers") is not None and (root / ppocr.ONNX_FILE).exists()
            and (root / ppocr.SAFETENSORS / "model.safetensors").exists())


pretrained = pytest.mark.skipif(not _have_pretrained(), reason="PP-OCRv6's files or transformers not here")

TOY = onnx_run.Spec(classes=("", "a", "B", "1", " "), max_width=3200, min_batch=320, bgr=True, same_width=True)


def _image(w: int, h: int, rgb=(255, 0, 0)) -> np.ndarray:
    im = np.zeros((h, w, 3), np.uint8)
    im[:] = rgb
    return im


# --- the spec and the layout (no model needed) ------------------------------------------

def test_the_decoder_reads_any_class_table():
    probs = np.zeros((5, 5), np.float32)
    for t, k in enumerate([1, 1, 0, 1, 2]):
        probs[t, k] = 1.0
    assert ctc.greedy_decode(probs, classes=TOY.classes).text == "aaB"     # a blank keeps the repeat


def test_m3s_spec_folds_and_another_keeps_case_and_drops_what_it_lacks():
    assert onnx_run.M3_SPEC.encode("exp 08/27") == charset.encode("EXP 08/27")
    assert TOY.encode("aB 1") == [1, 2, 4, 3]
    assert TOY.encode("a  \tB") == [1, 4, 2]                  # runs of white space: one space
    assert TOY.encode("aXb1") == [1, 3]                       # 'X' and 'b' aren't its classes: dropped


def test_ppocr_layout_pads_to_320_swaps_channels_and_squeezes_only_past_3200():
    x, widths = onnx_run.preprocess([_image(100, 48)], TOY)
    assert widths == [100] and x.shape == (1, 3, 48, 320)              # padded to its training width
    assert np.allclose(x[0, 2, :, :100], 1.0) and np.allclose(x[0, 0, :, :100], -1.0)   # red is channel 2: BGR
    assert np.allclose(x[0, :, :, 100:], 0.0)
    _, widths = onnx_run.preprocess([_image(2000, 48)], TOY)
    assert widths == [2000]                                            # M3's CRNN would squeeze it to 480
    _, widths = onnx_run.preprocess([_image(4000, 48)], TOY)
    assert widths == [3200]
    x, _ = onnx_run.preprocess([_image(100, 48)])                     # M3's own layout is unchanged
    assert x.shape == (1, 3, 48, 104) and np.allclose(x[0, 0, :, :100], 1.0)


def test_same_width_batches_hold_one_padded_width():
    sizes = [(100, 48), (300, 48), (330, 48), (331, 48), (600, 48), (20, 48), (336, 48)]
    batches = onnx_run.width_batches(sizes, 2, TOY)
    assert sorted(i for b in batches for i in b) == list(range(len(sizes)))
    for b in batches:
        assert len({onnx_run.padded_width([onnx_run.resized_width(*sizes[i], TOY.max_width)], TOY) for i in b}) == 1
    assert [sorted(b) for b in batches] == [[0, 5], [1], [2, 3], [6], [4]]     # 320 (3 crops), 336 (3), 600
    mixed = onnx_run.width_batches(sizes, 2)                           # M3's: neighbours by width only
    assert [len(b) for b in mixed] == [2, 2, 2, 1]


def test_reads_in_one_width_batches_do_not_depend_on_their_neighbours():
    def probs_of(x):          # 'a' in each frame that has any ink: a read that sees the padding
        frames = x.shape[3] // 8
        out = np.zeros((x.shape[0], frames, 5), np.float32)
        out[:, :, 0] = 1
        for i in range(x.shape[0]):
            out[i, :, 0] = 0
            out[i, :, 1 + (frames > 40)] = 1     # 'B' if its batch is wider than 320
        return out
    narrow, wide = _image(100, 48), _image(900, 48)
    alone = onnx_run.read(probs_of, [narrow], spec=TOY)
    together = onnx_run.read(probs_of, [narrow, wide], spec=TOY)
    assert together[0].text == alone[0].text == "a"
    mixed = onnx_run.read(probs_of, [narrow, wide], spec=dataclasses.replace(TOY, same_width=False))
    assert mixed[0].text == "B"


def test_the_dictionary_is_blank_then_its_characters_then_the_space(tmp_path):
    path = tmp_path / "inference.yml"
    path.write_text("Global:\n  model_name: x\nPostProcess:\n  name: CTCLabelDecode\n  character_dict:\n"
                    "  - '0'\n  - a\n  - A\n")
    assert ppocr.dictionary(path) == ("", "0", "a", "A", " ")


def test_the_route_is_pytorch_only_when_all_three_hold():
    assert m3.ppocr_route(True, 0.996, 1.2, 1.1)["route"] == "pytorch"
    assert m3.ppocr_route(True, 0.995, 1.2, 1.1)["route"] == "pytorch"
    for args in [(False, 1.0, 1.2, 1.1), (True, 0.994, 1.2, 1.1), (True, 1.0, 1.2, 1.2), (True, 1.0, None, None)]:
        route = m3.ppocr_route(*args)
        assert route["route"] == "paddle" and not all(route["conditions"].values())


def test_ppocr_takes_its_recipe():
    cfg = train.Config.for_reader("ppocrv6", seed=1, lr=1e-4)
    assert (cfg.budget, cfg.batch, cfg.warmup, cfg.lr, cfg.seed) == (2_000_000, 128, 0.03, 1e-4, 1)
    assert train.Config.for_reader("crnn") == train.Config()
    with pytest.raises(ValueError):
        train.Config.for_reader("trocr")


def test_the_route_scripts_agreement_counts_by_turn(load_script):
    script = load_script("ppocr_route")
    reads = [{"key": "r1", "pad": 0.0, "k": 0, "text": "21.05.26", "score": 0.99},
             {"key": "r1", "pad": 0.0, "k": 2, "text": "92'90'12", "score": 0.6},
             {"key": "r2", "pad": 0.0, "k": 0, "text": "MAR 2027", "score": 0.98},
             {"key": "r2", "pad": 0.0, "k": 2, "text": "LZOZ", "score": 0.5}]
    ours = [ctc.Read(t, (0.9,) * len(t)) for t in ("21.05.26", "92'90'1", "MAR 2027", "LZOZ")]
    a = script.agreement(reads, ours)
    assert a["all"]["same"] == 3 and a["upright"]["share"] == 1.0 and a["turned"]["same"] == 1
    assert [d["key"] for d in a["differing"]] == ["r1"]
    assert script.chosen_agreement(reads, ours) == {"same": 2, "crops": 2, "share": 1.0}


# --- with the pretrained files -----------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2)
    return ppocr.load_torch()


@pretrained
def test_its_spec_has_18710_classes():
    spec = ppocr.spec()
    assert len(spec.classes) == 18_710 and spec.classes[0] == "" and spec.classes[-1] == " "
    assert (spec.min_batch, spec.max_width, spec.bgr, spec.same_width) == (320, 3200, True, True)
    assert spec.encode("Exp 12/2027") == [spec.classes.index(c) for c in "Exp 12/2027"]


@pretrained
@pytest.mark.parametrize("width, frames", [(320, 40), (100, 12), (8, 1), (800, 100)])
def test_a_frame_is_8_px_and_our_logits_are_its_softmax(model, width, frames):
    x = torch.randn(2, 3, 48, width)
    with torch.no_grad():
        logits = model.eval()(x)
        assert logits.shape == (2, frames, 18_710) and logits.dtype == torch.float32
        assert (logits.softmax(-1) - model.library_probs(x)).abs().max() < 1e-6


@pretrained
def test_the_module_reads_as_the_official_onnx_file(model):
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for w, h in [(200, 40), (90, 30), (900, 20)]]
    spec = ppocr.spec()
    x, _ = onnx_run.preprocess(images[:1], spec)
    official, mine = ppocr.onnx_reader(threads=2), onnx_run.TorchReader(model)
    assert np.abs(official(x) - mine(x)).max() < 1e-3
    assert onnx_run.parity(onnx_run.read(official, images, spec=spec), onnx_run.read(mine, images, spec=spec)) == 1.0


@pretrained
def test_its_ctc_logits_take_gradients_and_the_check_leaves_it_as_it_was(model):
    from bestbefore.reader import data
    spec = ppocr.spec()
    rng = np.random.default_rng(1)
    views = [rng.integers(0, 256, (24, w, 3), dtype=np.uint8) for w in (60, 150, 400)]
    batch = data.collate(views, ["12/2027", "EXP 01.02.2026", "Best Before 12 months"], [0, 0, 0], spec=spec)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    check = ppocr.gradient_check(model, batch)
    assert check["passed"] and not check["not_finite"] and not check["no_gradient"]
    assert check["parameters"] == 5_275_238 and check["loss"] > 0
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())    # BatchNorm's statistics too


@pretrained
def test_its_export_reads_as_pytorch(model, tmp_path):
    spec = ppocr.spec()
    path = onnx_run.export(model, tmp_path / "ppocr.onnx")
    rng = np.random.default_rng(2)
    images = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for w, h in [(200, 40), (90, 30), (700, 20)]]
    x, _ = onnx_run.preprocess(images[2:], spec)
    mine, exported = onnx_run.TorchReader(model), onnx_run.OnnxReader(path, threads=2)
    assert np.abs(mine(x) - exported(x)).max() < 1e-4
    assert onnx_run.parity(onnx_run.read(mine, images, spec=spec), onnx_run.read(exported, images, spec=spec)) == 1.0


@pretrained
def test_a_ppocr_run_end_to_end(fake_piles, tmp_path):
    """The fine-tune in M3's loop on made-up piles: it trains from the
    pretrained weights, measures its probe before and after, keeps a
    checkpoint, and exports it."""
    cfg = train.Config.for_reader("ppocrv6", arm="all", budget=8, batch=4, evals=1, workers=0, device="cpu",
                                  amp=False, visible_min=0.0, log_every=1, eval_batch=16, probe=8, threads=2)
    summary = train.run(cfg, fake_piles, tmp_path / "ckpt", say=lambda s: None)
    assert summary["steps"] == 2 and summary["crops"] == 8 and summary["widest_batch"] >= 320
    probe = summary["probe_loss"]
    assert probe["crops"] == 100 and probe["before"] > 0 and probe["after"] > 0    # one batch of 100
    assert summary["spec"] == {"classes": 18_710, "max_width": 3200, "min_batch": 320, "bgr": True}
    model, saved = train.load(tmp_path / "ckpt" / "best.pt")
    assert isinstance(model, ppocr.PPOCRv6) and saved["config"]["reader"] == "ppocrv6"
    exported = train.export_checked(tmp_path / "ckpt" / "best.pt", fake_piles, threads=2)
    assert exported["passed"] and exported["reads"] == 16


def test_the_route_script_decides_from_its_records(load_script, monkeypatch, tmp_path):
    from bestbefore import runs
    script = load_script("ppocr_route")
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(m3, "DECISIONS_PATH", tmp_path / "m3-decisions.json")
    check = runs.start_run("m3-ppocr-route", {}, root=tmp_path)
    runs.update_run(check, results={"gradients": {"passed": True}, "torch_vs_onnx": {"all": {"share": 1.0}},
                                    "onnx_vs_s2": {"ppocrv6-batches": {"all": {"share": 0.99}}}})
    smoke = runs.start_run("m3-ppocrv6-smoke-cpu", {}, root=tmp_path)
    runs.update_run(smoke, result={"probe_loss": {"before": 2.0, "after": 1.5}}, smoke={"passed": True})
    script.decide()
    route = json.loads((tmp_path / "m3-decisions.json").read_text())["ppocr_route"]
    assert route["route"] == "pytorch" and route["check"] == check.name and route["smoke"] == smoke.name


@pretrained
def test_the_ppocr_smoke_script_runs_end_to_end(load_script, monkeypatch, tmp_path, fake_piles):
    from bestbefore import runs
    script = load_script("train_reader")
    decisions = tmp_path / "m3-decisions.json"
    decisions.write_text(json.dumps({"crop_recipe": {"run": "a-run", "pad": 0.0, "turn": True}}))
    monkeypatch.setattr(m3, "DECISIONS_PATH", decisions)
    monkeypatch.setattr(m3, "VISIBLE_MIN", 0.0)
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    real = ppocr.pretrained()
    monkeypatch.setattr(ppocr, "pretrained", lambda: real)          # the run's checkpoints go to tmp, not these
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    monkeypatch.setitem(script.SMOKE, "ppocrv6", {**script.SMOKE["ppocrv6"], "evals": 1, "probe": 8})
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr("sys.argv", ["train_reader.py", "--reader", "ppocrv6", "--smoke", "--device", "cpu",
                                     "--root", str(fake_piles), "--budget", "8", "--batch", "4", "--workers", "0",
                                     "--threads", "2", "--no-amp"])
    with pytest.raises(SystemExit):
        script.main()
    (run,) = (tmp_path / "runs").iterdir()
    assert run.name.endswith("m3-ppocrv6-smoke-cpu")
    record = json.loads((run / "run.json").read_text())
    assert record["config"]["lr"] == 5e-5 and record["config"]["warmup"] == 0.03 and record["config"]["steps"] == 2
    probe = record["result"]["probe_loss"]
    assert record["smoke"]["loss_fell"] == (probe["after"] < probe["before"]) and record["export"]["passed"]
