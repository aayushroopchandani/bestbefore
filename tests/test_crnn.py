"""The CRNN, its training loop and its export (reader/crnn.py, reader/train.py,
docs/m3-date-reader.md "CRNN + CTC", S5)."""
import numpy as np
import pytest
import torch

from bestbefore.reader import charset, crnn, data, metrics, onnx_run, train
from conftest import render_crop


def test_the_crnn_is_the_plans_size():
    assert crnn.parameters(crnn.CRNN()) == 4_552_048       # "about 4.5M", near PP-OCRv6 small's 5.2M


@pytest.mark.parametrize("width, frames", [(320, 80), (100, 25), (8, 2), (480, 120)])
def test_a_frame_is_4_px_of_the_crop(width, frames):
    model = crnn.CRNN().eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 48, width))
    assert out.shape == (2, frames, len(charset.CLASSES)) and out.dtype == torch.float32


def test_bf16_convolutions_only_while_training():
    model = crnn.CRNN(channels=(8, 16, 24, 32), hidden=16)
    model.amp = True
    seen = {}
    model.cnn.register_forward_hook(lambda m, i, o: seen.__setitem__("cnn", o.dtype))
    model.rnn.register_forward_hook(lambda m, i, o: seen.__setitem__("rnn", i[0].dtype))
    x = torch.randn(2, 3, 48, 64)
    model.train()
    out = model(x)
    assert seen == {"cnn": torch.bfloat16, "rnn": torch.float32} and out.dtype == torch.float32
    model.eval()
    with torch.no_grad():
        model(x)
    assert seen["cnn"] == torch.float32                    # evaluation and export: float32 throughout


def test_lr_warms_up_then_falls_on_a_cosine():
    f = [train.lr_factor(s, 1000, 0.02) for s in range(1000)]
    assert f[0] == pytest.approx(1 / 20) and f[19] == pytest.approx(1.0)
    assert all(a <= b for a, b in zip(f[:20], f[1:20]))
    assert all(a >= b for a, b in zip(f[20:], f[21:]))
    assert f[-1] < 1e-4


@pytest.mark.parametrize("text, frames", [("2027", 4), ("2007", 5), ("00", 3), ("000", 5), ("08/2027", 7)])
def test_the_frames_ctc_needs(text, frames):
    assert train.needed_frames(charset.encode(text)) == frames     # a blank between equal neighbours


def test_a_text_too_long_for_its_frames_is_counted_and_dropped():
    views = [np.zeros((48, 16, 3), np.uint8), np.zeros((48, 160, 3), np.uint8)]  # 4 and 40 frames
    batch = data.collate(views, ["08/2027", "08/2027"], [0, 0])
    model = crnn.CRNN(channels=(8, 16, 24, 32), hidden=16)
    loss, infeasible = train.ctc_loss(model, batch, "cpu", torch.nn.CTCLoss(zero_infinity=True))
    assert infeasible == 1 and torch.isfinite(loss)


def test_a_tiny_crnn_overfits_50_crops():
    """The loop's pieces learn: 50 rendered crops, read back exactly once the
    loss is near zero (about 40 s on the laptop: a narrow CRNN, its plateau
    of mostly blanks breaking after some 150 steps)."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    views, texts = [], []
    for _ in range(50):
        text = "".join(rng.choice(list("0123456789/"), int(rng.integers(3, 6))))
        crop, box = render_crop(text, rng, size=16)
        views.append(data.plain_view(np.asarray(crop), box, 0.0))
        texts.append(text)
    batch = data.collate(views, texts, [0] * 50)
    model = crnn.CRNN(channels=(8, 16, 24, 32), hidden=32)
    optimiser = torch.optim.AdamW(model.parameters(), lr=5e-3)
    loss_fn = torch.nn.CTCLoss(zero_infinity=True)
    model.train()
    for _ in range(600):
        loss, infeasible = train.ctc_loss(model, batch, "cpu", loss_fn)
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimiser.step()
        if loss.item() < 0.05:
            break
    assert loss.item() < 0.1 and infeasible == 0
    reads = onnx_run.read(onnx_run.TorchReader(model), views)
    assert sum(metrics.exact(r.text, t) for r, t in zip(reads, texts)) >= 48


def test_onnx_reads_the_crnn_as_pytorch_does(tmp_path):
    torch.manual_seed(1)
    model = crnn.CRNN()
    for m in model.modules():                   # running statistics unlike any batch's
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-1, 1)
            m.running_var.uniform_(0.5, 2)
    model.eval()
    path = onnx_run.export(model, tmp_path / "crnn.onnx")
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for w, h in [(200, 40), (90, 30), (600, 20), (30, 60)]]
    torch_reader, onnx_reader = onnx_run.TorchReader(model), onnx_run.OnnxReader(path, threads=2)
    x, _ = onnx_run.preprocess(images)
    assert np.abs(torch_reader(x) - onnx_reader(x)).max() < 1e-4
    assert onnx_run.parity(onnx_run.read(torch_reader, images), onnx_run.read(onnx_reader, images)) == 1.0


def test_a_run_end_to_end(fake_piles, tmp_path):
    """train.run on made-up piles, a narrow CRNN, no loader workers: it
    trains, scores its checkpoints, keeps the best, and exports it."""
    cfg = train.Config(arm="all", budget=48, batch=8, evals=2, workers=0, device="cpu", amp=True,
                       visible_min=0.0, channels=(8, 16, 24, 32), hidden=16, log_every=2, eval_batch=16)
    said = []
    summary = train.run(cfg, fake_piles, tmp_path / "ckpt", say=said.append)
    assert summary["steps"] == 6 and summary["crops"] == 48
    assert sum(summary["drawn"].values()) == 48 and set(summary["drawn"]) == {"expdate-real", "expdate-synth",
                                                                               "gen", "store"}
    assert [e["step"] for e in summary["evals"]] == [3, 6]
    assert {"real-val", "synth-val-v2", "pooled"} <= set(summary["evals"][0])
    assert len(summary["history"]) == 3 and summary["crops_per_second"] > 0
    assert (tmp_path / "ckpt" / "best.pt").exists() and (tmp_path / "ckpt" / "last.pt").exists()
    model, saved = train.load(tmp_path / "ckpt" / "best.pt")
    assert saved["step"] == summary["best"]["step"] and not model.training
    exported = train.export_checked(tmp_path / "ckpt" / "best.pt", fake_piles, threads=2)
    assert exported["passed"] and exported["parity"] == 1.0 and exported["reads"] == 16
    assert (tmp_path / "ckpt" / "reader.onnx").exists()


def test_the_smokes_loss_test(load_script):
    script = load_script("train_reader")
    falling = [{"loss": v} for v in np.linspace(20, 2, 40)]
    flat = [{"loss": v} for v in np.linspace(3, 2, 40)]
    assert script.loss_fell(falling) and not script.loss_fell(flat)


def _script_world(load_script, monkeypatch, tmp_path, fake_piles):
    """train_reader.py pointed at made-up piles, a made-up crop recipe and
    temporary run and checkpoint folders, with a cut-off set."""
    import json
    from bestbefore import m3, runs
    script = load_script("train_reader")
    decisions = tmp_path / "m3-decisions.json"
    decisions.write_text(json.dumps({"crop_recipe": {"run": "a-run", "pad": 0.0, "turn": True}}))
    monkeypatch.setattr(m3, "DECISIONS_PATH", decisions)
    monkeypatch.setattr(m3, "VISIBLE_MIN", 0.0)
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    (tmp_path / "runs").mkdir()
    return script


def test_the_smoke_script_runs_end_to_end(load_script, monkeypatch, tmp_path, fake_piles):
    import json
    script = _script_world(load_script, monkeypatch, tmp_path, fake_piles)
    monkeypatch.setattr("sys.argv", ["train_reader.py", "--smoke", "--device", "cpu", "--root", str(fake_piles),
                                     "--budget", "16", "--batch", "8", "--workers", "0", "--threads", "2"])
    with pytest.raises(SystemExit):          # 2 steps can't halve the loss: the smoke says FAILED, exit 1
        script.main()
    (run,) = (tmp_path / "runs").iterdir()
    assert run.name.endswith("m3-crnn-smoke-cpu")
    record = json.loads((run / "run.json").read_text())
    assert record["config"]["steps"] == 2 and record["config"]["visible_min"] == 0.0
    assert record["config"]["parse_sha256"] and set(record["config"]["piles"]["gen"]) == {"manifest", "visibility"}
    assert record["result"]["crops"] == 16 and record["export"]["passed"]
    assert record["smoke"] == {"loss_fell": False, "parity_passed": True, "passed": False}
    assert (tmp_path / "checkpoints" / "m3" / run.name / "reader.onnx").exists()


def test_the_loader_alone_is_timed(load_script, monkeypatch, tmp_path, fake_piles):
    import json
    script = _script_world(load_script, monkeypatch, tmp_path, fake_piles)
    monkeypatch.setattr("sys.argv", ["train_reader.py", "--loader-only", "--root", str(fake_piles),
                                     "--batch", "8", "--workers", "0", "--batches", "3"])
    monkeypatch.setattr(script, "WARMUP_BATCHES", 1)
    script.main()
    (run,) = (tmp_path / "runs").iterdir()
    loader = json.loads((run / "run.json").read_text())["loader"]
    assert loader["crops_per_second"] > 0 and loader["lmdb"]["pile"] == "gen" and loader["lmdb"]["reads"] == 5000
