"""The one pre- and post-processing of CTC readers, ONNX export and parity
(reader/onnx_run.py, docs/m3-date-reader.md "Shared", decisions 14 and 16)."""
import numpy as np
import pytest
import torch
from PIL import Image

from bestbefore.reader import charset, crop_scores, onnx_run
from bestbefore.reader.ctc import Read


def _image(w: int, h: int, value: int = 255) -> np.ndarray:
    return np.full((h, w, 3), value, np.uint8)


def test_preprocess_height_aspect_padding_and_normalisation():
    x, widths = onnx_run.preprocess([_image(100, 20), _image(30, 30, 0)])
    assert x.dtype == np.float32 and x.shape == (2, 3, 48, 240)     # 100/20 x 48 = 240
    assert widths == [240, 48]
    assert np.allclose(x[0], 1.0)                                    # white -> 1
    assert np.allclose(x[1, :, :, :48], -1.0)                        # black -> -1
    assert np.allclose(x[1, :, :, 48:], 0.0)                         # padding: mid-grey


def test_preprocess_widths_round_up_to_a_step_squeeze_past_the_max_and_keep_a_minimum():
    x, widths = onnx_run.preprocess([_image(101, 48)])
    assert widths == [101] and x.shape[3] == 104                     # padded to a multiple of 8
    x, widths = onnx_run.preprocess([_image(2000, 48), _image(10, 48)])
    assert widths == [480, 10] and x.shape[3] == 480                 # wider than 480 is squeezed
    _, widths = onnx_run.preprocess([_image(5, 200)])
    assert widths == [8]                                             # never under 8 px


def test_own_frames_drop_the_padding():
    assert onnx_run.own_frames(40, [160, 100, 8], 160) == [40, 25, 2]


def test_postprocess_reads_each_crop_from_its_own_frames():
    C = len(charset.CLASSES)
    probs = np.zeros((2, 4, C), np.float32)
    for t, k in enumerate([charset.index("1"), charset.index("2"), charset.BLANK, charset.index("3")]):
        probs[:, t, k] = 1.0
    reads = onnx_run.postprocess(probs, [8, 4], 8)      # the second crop owns only frames 0-1
    assert [r.text for r in reads] == ["123", "12"]


def test_width_batches_sort_by_width_and_keep_every_crop_once():
    sizes = [(300, 30), (40, 40), (200, 40), (50, 50), (600, 30)]
    batches = onnx_run.width_batches(sizes, 2)
    assert sorted(i for b in batches for i in b) == list(range(5))
    widths = [onnx_run.resized_width(*sizes[i]) for b in batches for i in b]
    assert widths == sorted(widths)


def test_read_keeps_the_order_given():
    def probs_of(x):          # reads each crop's width class: wide crops '1', narrow ones '2'
        out = np.zeros((x.shape[0], x.shape[3] // 4, len(charset.CLASSES)), np.float32)
        for i in range(x.shape[0]):
            own = (x[i, 0, 0] != 0).sum() // 4
            out[i, :, charset.BLANK] = 1
            out[i, 0, :] = 0
            out[i, 0, charset.index("1" if own > 20 else "2")] = 1
        return out
    images = [_image(200, 20), _image(20, 20), _image(300, 20), _image(10, 20)]
    assert [r.text for r in onnx_run.read(probs_of, images, batch=2)] == ["1", "2", "1", "2"]


@pytest.mark.parametrize("texts, confs, best", [
    (["03/2027", "LZ0Z"], [0.5, 0.9], 0),    # the read holding a date wins, even less confident
    (["2027", "LZ0Z"], [0.5, 0.9], 1),       # a year alone is no date to the parser
    (["LZ0Z", "08/2027"], [0.9, 0.5], 1),
    (["08/2027", "08/2021"], [0.6, 0.9], 1),  # both hold one: the more confident
    (["", "abc"], [0.0, 0.3], 1),             # neither: the more confident
    (["08/2027", "08/2027"], [0.7, 0.7], 0),  # a tie: the first
])
def test_choose_is_decision_14_as_s2_scored_it(texts, confs, best):
    reads = [Read(t, (c,) if t else ()) for t, c in zip(texts, confs)]
    assert onnx_run.choose(reads) == best
    s2 = [{"text": t, "score": c} for t, c in zip(texts, confs)]
    assert s2.index(crop_scores.choose_turn(s2)) == best


def _spelling(text: str, p: float, frames: int) -> np.ndarray:
    out = np.zeros((frames, len(charset.CLASSES)), np.float32)
    out[:, charset.BLANK] = 1
    for t, c in enumerate(text):
        out[2 * t] = 0
        out[2 * t, charset.index(c)] = p
    return out


def test_read_oriented_tries_two_turns_and_keeps_the_read_with_a_date():
    # a wide crop, marked top-left: read upright it gives a date (unsure), any other way a
    # confident non-date, so only decision 14's rule can pick the upright read
    im = np.zeros((20, 60, 3), np.uint8)
    im[:4, :4] = 255
    tall = np.ascontiguousarray(np.rot90(im, 3))         # the same crop, lying on its side

    def probs_of(x):
        out = np.zeros((x.shape[0], 16, len(charset.CLASSES)), np.float32)
        for i in range(x.shape[0]):
            upright = x[i, 0, 0, 0] > 0.5 and x[i, 0, -1, -1] < 0
            out[i] = _spelling("08/2027", 0.6, 16) if upright else _spelling("LZ0Z", 0.99, 16)
        return out
    (r_wide, q_wide), (r_tall, q_tall) = onnx_run.read_oriented(probs_of, [im, tall])
    assert (r_wide.text, q_wide) == ("08/2027", 0)
    assert (r_tall.text, q_tall) == ("08/2027", 1)       # a tall crop is read at 1 and 3 only
    pil = np.asarray(Image.fromarray(im).transpose(Image.Transpose.ROTATE_90))
    assert np.array_equal(np.rot90(im, 1), pil)          # np.rot90 turns as PIL's ROTATE_90 (and crops.cut)


class TinyCtc(torch.nn.Module):
    """A small CTC reader with the CRNN's frame arithmetic (W/4 frames)."""

    def __init__(self):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.BatchNorm2d(8), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(8, 16, 3, padding=1), torch.nn.BatchNorm2d(16), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(16, 16, (12, 1)))
        self.rnn = torch.nn.LSTM(16, 8, num_layers=2, bidirectional=True, batch_first=True)
        self.head = torch.nn.Linear(16, len(charset.CLASSES))

    def forward(self, x):
        f = self.cnn(x).squeeze(2).permute(0, 2, 1)
        return self.head(self.rnn(f)[0])


@pytest.fixture
def tiny():
    torch.manual_seed(0)
    model = TinyCtc()
    for m in model.modules():          # running statistics far from batch ones, so a BatchNorm left
        if isinstance(m, torch.nn.BatchNorm2d):     # in training mode would show
            m.running_mean.uniform_(-2, 2)
            m.running_var.uniform_(0.2, 3)
    return model.eval()


def test_export_reads_as_pytorch_does(tmp_path, tiny):
    path = onnx_run.export(tiny, tmp_path / "tiny.onnx")
    assert not tiny.training and all(not m.training for m in tiny.modules())    # left as it came
    onnx_reader, torch_reader = onnx_run.OnnxReader(path, threads=2), onnx_run.TorchReader(tiny)
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for w, h in
              [(120, 30), (300, 40), (40, 40), (900, 30), (64, 64), (16, 48)]]
    for idx in onnx_run.width_batches([(im.shape[1], im.shape[0]) for im in images], 4):
        x, _ = onnx_run.preprocess([images[i] for i in idx])
        a, b = torch_reader(x), onnx_reader(x)
        assert a.shape == b.shape == (len(idx), x.shape[3] // 4, len(charset.CLASSES))
        assert np.abs(a - b).max() < 1e-5
    reads_torch = onnx_run.read(torch_reader, images, batch=4)
    reads_onnx = onnx_run.read(onnx_reader, images, batch=4)
    assert onnx_run.parity(reads_torch, reads_onnx) == 1.0


def test_export_puts_a_training_model_back_in_training(tmp_path, tiny):
    tiny.train()
    onnx_run.export(tiny, tmp_path / "tiny.onnx")
    assert tiny.training and all(m.training for m in tiny.modules())


def test_the_onnx_session_uses_the_threads_it_is_given(tmp_path, tiny):
    reader = onnx_run.OnnxReader(onnx_run.export(tiny, tmp_path / "tiny.onnx"), threads=3)
    assert reader.session.get_session_options().intra_op_num_threads == 3


def test_parity_needs_matching_reads():
    a = [Read("1", (0.9,)), Read("2", (0.9,))]
    assert onnx_run.parity(a, [Read("1", (0.5,)), Read("3", (0.9,))]) == 0.5
    with pytest.raises(ValueError):
        onnx_run.parity(a, a[:1])
