"""PP-OCRv6 small, M3's second reader (docs/m3-date-reader.md, "PP-OCRv6
small", S6): its dictionary, its batch layout, the official ONNX file read
through M3's one pre- and post-processing, and its recogniser in PyTorch
(transformers) for the fine-tuning route.

The network: a PP-LCNetV4 backbone turns a crop 48 px high into one row
of features, a frame every 8 px of width; a light SVTR neck (two
transformer blocks, self-attention over every frame of the row); a linear
head to 18,710 classes a frame: the blank, a dictionary of 18,708
characters (Chinese, Latin, digits, symbols, emoji), and the space.

Its layout (SPEC), PaddleX 3.7's own, which M3 S6 measured matters:

  BGR        its config decodes images as BGR (`DecodeImage img_mode: BGR`)
             and S2 handed PaddleOCR BGR crops; preprocess swaps the
             channels of M3's RGB batch
  min 320    its training input is 3 x 48 x 320, and PaddleX pads every
             narrower crop to 320. The neck's attention looks at every
             frame, padding included, so a crop padded less reads
             differently
  max 3200   PaddleX squeezes only past 3,200 px, M3's CRNN past 480
  one width  read in batches of one padded width only: with no attention
             mask, a crop padded to a wider neighbour's width reads
             otherwise (on 4 real-val crops, 24 reads, one batch: 9 matched
             PaddleOCR's, one a call 22). Training's batches still pad to
             their widest; the width buckets keep that small

On the laptop's first look (40 real-val crops, the official ONNX file
against S2's PaddleOCR reads, upright): BGR padded to the crop's own width
and squeezed past 480 matched 113 of 120, padded to at least 320 117, and
squeezed only past 3,200 too 120. The full measure is S6's job's.

Case is kept (decision 10): its dictionary has both cases. A training
text's characters outside the dictionary are dropped, as PaddleOCR's own
label encoder does; every score folds case alike (charset.fold).

The PyTorch module (`PPOCRv6`) is transformers' PPOCRV6SmallRecForTextRecognition
(5.22.2) from the official safetensors, giving the head's logits: the
library's own forward ends in a softmax, which a CTC loss can't take
(log of probabilities that underflow to 0). The same weights as the ONNX
file: on random input their probabilities differ by at most 6e-5.
"""
from __future__ import annotations

import hashlib
import math
from functools import cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..config import checkpoint_dir
from . import onnx_run

NAME = "PP-OCRv6_small_rec"
ONNX_FILE = "PP-OCRv6_small_rec_onnx/inference.onnx"
SAFETENSORS = "PP-OCRv6_small_rec_safetensors"
DICTIONARY_FILE = "PP-OCRv6_small_rec_onnx/inference.yml"
# scripts/fetch_m3_models.py's hashes (docs/datasets.md): another file is refused
ONNX_SHA256 = "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634"
DICTIONARY_SHA256 = "ab078671bb49f06228eadccd34f1bb501e157f7a047095ffb943ba81512c77d1"
SAFETENSORS_SHA256 = "f65a332afe5aa663f0b9d5706f4ae8457b5b4058a842d5c1eb22df505c27d642"

CLASSES = 18_710
MIN_BATCH = 320         # RecResizeImg's image_shape [3, 48, 320]: PaddleX pads every crop to at least this
MAX_WIDTH = 3200        # PaddleX's OCRReisizeNormImg.max_imgW: wider is squeezed
BGR = True              # DecodeImage img_mode: BGR
FRAME_PX = 8            # crop pixels per frame


def pretrained() -> Path:
    return checkpoint_dir() / "pretrained"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(path: Path, digest: str) -> Path:
    if sha256(path) != digest:
        raise RuntimeError(f"{path} isn't the file fetched in S1 (its SHA-256 differs): scripts/fetch_m3_models.py --check")
    return path


def dictionary(path: Path) -> tuple[str, ...]:
    """Its classes from its inference.yml, as PaddleX's CTCLabelDecode
    builds them: the blank (here ""), the dictionary, then the space."""
    import yaml
    characters = yaml.safe_load(path.read_text())["PostProcess"]["character_dict"]
    return ("", *characters, " ")


@cache
def spec(root: Path | None = None) -> onnx_run.Spec:
    """PP-OCRv6's onnx_run.Spec, its dictionary read from the pretrained
    files (checked by hash)."""
    path = checked((root or pretrained()) / DICTIONARY_FILE, DICTIONARY_SHA256)
    classes = dictionary(path)
    if len(classes) != CLASSES:
        raise RuntimeError(f"{path}: {len(classes)} classes, not {CLASSES}")
    return onnx_run.Spec(classes=classes, max_width=MAX_WIDTH, min_batch=MIN_BATCH, bgr=BGR, same_width=True)


def onnx_reader(threads: int, root: Path | None = None) -> onnx_run.OnnxReader:
    """The official ONNX file (off the shelf, decision 19's baseline C), at
    `threads` threads."""
    return onnx_run.OnnxReader(checked((root or pretrained()) / ONNX_FILE, ONNX_SHA256), threads)


class PPOCRv6(nn.Module):
    """Images [N, 3, 48, W] (BGR, normalised) -> logits [N, W/8, 18,710].

    transformers' module, called part by part so its head's linear gives
    logits: backbone (PP-LCNetV4) and its pooling, the SVTR neck, the
    linear. bf16 (`amp`, while training only) covers the backbone and
    the neck; the head and the loss stay in float32, as in the CRNN."""

    def __init__(self, hf: nn.Module):
        super().__init__()
        self.hf = hf
        self.amp = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=self.amp and self.training):
            features = self.hf.model(pixel_values=x).last_hidden_state          # [N, 384, 1, W/8]
            frames = self.hf.head.encoder(features).last_hidden_state           # [N, W/8, 120]
        return self.hf.head.head(frames.float())                                 # [N, W/8, classes]

    def library_probs(self, x: torch.Tensor) -> torch.Tensor:
        """transformers' own forward: its softmax, for checking ours."""
        return self.hf(pixel_values=x).last_hidden_state


def load_torch(root: Path | None = None, attention: str = "eager") -> PPOCRv6:
    """The pretrained recogniser from the official safetensors, in eval
    mode. Eager attention: the same arithmetic on every device, and the
    legacy ONNX exporter traces it plainly."""
    from transformers import PPOCRV6SmallRecForTextRecognition
    folder = (root or pretrained()) / SAFETENSORS
    checked(folder / "model.safetensors", SAFETENSORS_SHA256)
    hf = PPOCRV6SmallRecForTextRecognition.from_pretrained(folder, attn_implementation=attention)
    if hf.config.head_out_channels != CLASSES:
        raise RuntimeError(f"{folder}: {hf.config.head_out_channels} classes, not {CLASSES}")
    return PPOCRv6(hf).eval()


# --- S6's route check: what the PyTorch route needs ------------------------------------

def gradient_check(model: nn.Module, batch: dict) -> dict:
    """The CTC loss of one training batch through `model`'s logits, and its
    gradients: the first route condition holds when the loss is finite and
    every parameter gets a finite gradient, not all zero.

    On a copy: a forward pass in training mode moves BatchNorm's running
    statistics, which would leave `model` reading otherwise (found in S6:
    afterwards it no longer read as the ONNX file did)."""
    import copy
    from .train import ctc_loss
    model = copy.deepcopy(model).train()
    loss, infeasible = ctc_loss(model, batch, "cpu", nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True))
    loss.backward()
    named = list(model.named_parameters())
    missing = [n for n, p in named if p.grad is None]
    not_finite = [n for n, p in named if p.grad is not None and not torch.isfinite(p.grad).all()]
    zero = [n for n, p in named if p.grad is not None and not p.grad.any()]
    norm = math.sqrt(sum(float(p.grad.double().pow(2).sum()) for _, p in named if p.grad is not None))
    loss = loss.item()
    return {"loss": loss, "infeasible": infeasible, "crops": int(len(batch["lengths"])),
            "tensors": len(named), "parameters": sum(p.numel() for _, p in named),
            "no_gradient": missing, "not_finite": not_finite, "all_zero": zero, "grad_norm": norm,
            "passed": math.isfinite(loss) and not missing and not not_finite and len(zero) < len(named)}


def softmax_gap(model: PPOCRv6, x: np.ndarray) -> float:
    """The largest difference between our logits' softmax and transformers'
    own probabilities on a batch (eval mode)."""
    with torch.no_grad():
        t = torch.from_numpy(x)
        return float((model.eval()(t).softmax(-1) - model.library_probs(t)).abs().max())
