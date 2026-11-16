"""Any CTC reader, through one pre- and post-processing
(docs/m3-date-reader.md, "Shared", decisions 14 and 16).

Every CTC reader in M3 is read the same way, by training's evaluation
(PyTorch), by scoring (ONNX Runtime, on both machines) and later by the
phone:

  preprocess   RGB crops -> one batch: each resized to HEIGHT px high
               keeping its aspect (at least MIN_WIDTH px wide; wider than
               MAX_WIDTH is squeezed), normalised to (x/255 - 0.5)/0.5 as
               PP-OCR does, padded on the right with 0 (mid-grey) to the
               batch's widest, a multiple of WIDTH_STEP
  reader       the batch -> probabilities [N, frames, classes]
  postprocess  each crop's own frames (those over its padding dropped),
               greedy CTC decoding (reader/ctc.py)

Crops are read in batches of similar width (width_batches), so a batch
pads little. Decision 14's orientation step reads a crop at two quarter
turns and keeps the read that holds a date (read_oriented).

The export writes any CTC module (images [N, 3, 48, W] -> logits [N, T, C])
as an ONNX file that gives probabilities, its batch and width dynamic.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np

from . import crops, parse
from .ctc import Read, greedy_decode

HEIGHT = 48
MAX_WIDTH = 480
MIN_WIDTH = 8
WIDTH_STEP = 8
OPSET = 17
PARITY_MIN = 0.995      # the share of reads an exported file must give as its framework did (decision 16)


def resized_width(width: int, height: int) -> int:
    """A crop's width once HEIGHT px high, its aspect kept, within
    MIN_WIDTH..MAX_WIDTH (rounded up, as PP-OCR does)."""
    return min(MAX_WIDTH, max(MIN_WIDTH, math.ceil(HEIGHT * width / height)))


def preprocess(images: Sequence[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    """RGB uint8 crops [H, W, 3] -> a batch [N, 3, HEIGHT, W] in float32,
    and each crop's own width in it (before the padding)."""
    widths = [resized_width(im.shape[1], im.shape[0]) for im in images]
    padded = min(MAX_WIDTH, math.ceil(max(widths) / WIDTH_STEP) * WIDTH_STEP)
    batch = np.zeros((len(images), 3, HEIGHT, padded), np.float32)
    for i, (im, w) in enumerate(zip(images, widths)):
        resized = cv2.resize(im, (w, HEIGHT), interpolation=cv2.INTER_LINEAR)
        batch[i, :, :, :w] = (resized.transpose(2, 0, 1).astype(np.float32) / 255.0 - 0.5) / 0.5
    return batch, widths


def own_frames(frames: int, widths: Sequence[int], padded: int) -> list[int]:
    """How many of a batch's `frames` belong to each crop: the share of the
    padded width its own width takes, rounded up."""
    return [min(frames, math.ceil(frames * w / padded)) for w in widths]


def postprocess(probs: np.ndarray, widths: Sequence[int], padded: int) -> list[Read]:
    """[N, frames, classes] probabilities -> one Read a crop, from its own frames."""
    return [greedy_decode(p, n) for p, n in zip(probs, own_frames(probs.shape[1], widths, padded))]


def width_batches(sizes: Sequence[tuple[int, int]], batch: int) -> list[list[int]]:
    """Indices of crops of `sizes` (width, height), sorted by their resized
    width and cut into batches: a batch pads little."""
    order = sorted(range(len(sizes)), key=lambda i: resized_width(*sizes[i]))
    return [order[i:i + batch] for i in range(0, len(order), batch)]


Probs = Callable[[np.ndarray], np.ndarray]     # a batch -> probabilities [N, frames, classes]


def read(probs_of: Probs, images: Sequence[np.ndarray], batch: int = 64) -> list[Read]:
    """Every crop read, in batches of similar width, in the order given."""
    out: list[Read | None] = [None] * len(images)
    for idx in width_batches([(im.shape[1], im.shape[0]) for im in images], batch):
        x, widths = preprocess([images[i] for i in idx])
        for i, r in zip(idx, postprocess(probs_of(x), widths, x.shape[3])):
            out[i] = r
    return out


def choose(reads: Sequence[Read]) -> int:
    """Decision 14: of a crop's reads at its two turns, the one whose text
    holds a date (the frozen parser judges); if both or neither do, the
    more confident; a tie, the first. S2's crop_scores.choose_turn is the
    same rule on its records."""
    return max(range(len(reads)), key=lambda i: (bool(parse.read_dates(reads[i].text)), reads[i].confidence))


def read_oriented(probs_of: Probs, images: Sequence[np.ndarray], batch: int = 64) -> list[tuple[Read, int]]:
    """Each crop read at its two quarter turns (crops.turns: a wide crop at
    0 and 2, a tall one at 1 and 3, anticlockwise), all in one pass; per
    crop the chosen read and the turn it was read at."""
    turned, owner = [], []
    for k, im in enumerate(images):
        for q in crops.turns(im.shape[0] > im.shape[1]):
            turned.append(np.ascontiguousarray(np.rot90(im, q)))
            owner.append(q)
    reads = read(probs_of, turned, batch)
    out = []
    for k in range(len(images)):
        pair = reads[2 * k:2 * k + 2]
        best = choose(pair)
        out.append((pair[best], owner[2 * k + best]))
    return out


# --- ONNX Runtime -----------------------------------------------------------------

class OnnxReader:
    """An exported CTC reader in ONNX Runtime, on the CPU, at a fixed number
    of threads: unless told, ONNX Runtime starts a thread per core of the
    whole node and pins each, which fails inside a job (CLAUDE.md)."""

    def __init__(self, path: Path, threads: int):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.threads = threads
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input: x})[0]


# --- PyTorch: training's evaluation and the export ----------------------------------

class TorchReader:
    """A PyTorch CTC module (images -> logits [N, T, C]) read as ONNX reads
    it: in eval mode, fp32, softmax over the classes."""

    def __init__(self, model, device: str = "cpu"):
        self.model, self.device = model, device

    def __call__(self, x: np.ndarray) -> np.ndarray:
        import torch
        was = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                logits = self.model(torch.from_numpy(x).to(self.device))
            return logits.float().softmax(-1).cpu().numpy()
        finally:
            self.model.train(was)


def _probs_module(model):
    import torch

    class Probs(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x).float().softmax(-1)

    return Probs(model)


def export(model, path: Path) -> Path:
    """Write a CTC module as ONNX: input `x` [N, 3, HEIGHT, W], output
    `probs` [N, frames, classes], N and W dynamic; checked by onnx.

    Exported in eval mode (BatchNorm's running statistics), and the module
    is left in the mode it came in. torch.onnx.export puts the module it is
    given back in that module's own mode afterwards, and a wrapper made
    fresh starts in training mode, which would leave the model's BatchNorm
    on batch statistics (found in S5): so the wrapper is put in eval mode
    first. The legacy TorchScript exporter (dynamo=False) handles the LSTM
    with a dynamic width; the default one needs onnxscript, which neither
    machine has."""
    import onnx
    import torch
    was = model.training
    wrapper = _probs_module(model).eval()
    example = torch.zeros(2, 3, HEIGHT, 160, device=next(model.parameters()).device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # the legacy exporter's deprecation and LSTM notes
        torch.onnx.export(wrapper, (example,), str(path), dynamo=False, input_names=["x"],
                          output_names=["probs"], opset_version=OPSET,
                          dynamic_axes={"x": {0: "n", 3: "w"}, "probs": {0: "n", 1: "frames"}})
    model.train(was)
    onnx.checker.check_model(str(path))
    return path


def parity(a: Sequence[Read], b: Sequence[Read]) -> float:
    """The share of crops whose two reads give the same text."""
    if len(a) != len(b) or not a:
        raise ValueError("parity needs two non-empty lists of reads of the same crops")
    return sum(x.text == y.text for x, y in zip(a, b)) / len(a)
