"""The CRNN (docs/m3-date-reader.md, "CRNN + CTC"): a convolutional network
that turns a crop 48 px high into a row of feature columns, a bidirectional
LSTM that reads that row left to right and right to left, and a linear head
giving, for every column (a frame), a score for each of the 48 classes.

| Block    | Layers                                          | Output (H x W)  |
|----------|-------------------------------------------------|-----------------|
| 1        | conv 3x3, 64, BN, ReLU; max-pool 2x2            | 24 x W/2        |
| 2        | conv 3x3, 128, BN, ReLU; max-pool 2x2           | 12 x W/4        |
| 3        | conv 3x3, 256 x2, BN, ReLU; max-pool (2,1)      | 6 x W/4         |
| 4        | conv 3x3, 384 x2, BN, ReLU; max-pool (2,1)      | 3 x W/4         |
| 5        | conv (3,1), 384, no padding, BN, ReLU           | 1 x W/4         |
| sequence | 2-layer bidirectional LSTM, 128 a direction     | W/4 frames x 256 |
| head     | linear to 48 classes                            | W/4 x 48        |

So a frame is 4 px of the crop's width: a 320 px crop gives 80 frames. The
pools after block 2 halve the height only, keeping the frames narrow
enough to place 25 characters with CTC's blanks between repeats.

The LSTM is what lets one frame's guess use its neighbours: a column that
shows the left half of an 8 looks like a 3 alone, and like an 8 once the
next columns are seen. Bidirectional: one LSTM reads the frames left to
right, another right to left, and each frame gets both.

bf16 (`amp`): while training, the convolutions run in bfloat16 (autocast);
the LSTM, the head and the loss stay in float32. Evaluation and the ONNX
export are float32 throughout.
"""
from __future__ import annotations

import torch
from torch import nn

from .charset import CLASSES

CHANNELS = (64, 128, 256, 384)
HIDDEN = 128
FRAME_PX = 4            # crop pixels per frame


def _block(cin: int, cout: int, kernel=3, padding=1) -> list[nn.Module]:
    return [nn.Conv2d(cin, cout, kernel, padding=padding, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class CRNN(nn.Module):
    """Images [N, 3, 48, W] -> logits [N, W/4, classes]."""

    def __init__(self, classes: int = len(CLASSES), channels: tuple[int, int, int, int] = CHANNELS,
                 hidden: int = HIDDEN):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.cnn = nn.Sequential(
            *_block(3, c1), nn.MaxPool2d(2, 2),
            *_block(c1, c2), nn.MaxPool2d(2, 2),
            *_block(c2, c3), *_block(c3, c3), nn.MaxPool2d((2, 1), (2, 1)),
            *_block(c3, c4), *_block(c4, c4), nn.MaxPool2d((2, 1), (2, 1)),
            *_block(c4, c4, (3, 1), 0),
        )
        self.rnn = nn.LSTM(c4, hidden, num_layers=2, bidirectional=True, batch_first=True)
        self.head = nn.Linear(2 * hidden, classes)
        self.amp = False            # bf16 convolutions, while training only

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=self.amp and self.training):
            features = self.cnn(x)                               # [N, C, 1, W/4]
        frames = features.float().squeeze(2).permute(0, 2, 1)    # [N, W/4, C]: one feature vector a frame
        frames, _ = self.rnn(frames)                             # [N, W/4, 2 x hidden], float32
        return self.head(frames)                                 # [N, W/4, classes]


def parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
