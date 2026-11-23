"""Greedy CTC decoding (docs/m3-date-reader.md, "Shared", decision 15),
written against tests/test_ctc.py.

A CTC reader doesn't say where each character is. It gives, for every frame
(a thin vertical slice of the crop), a probability for each class: the blank
or a character. A character usually spans several frames, so its class
repeats; frames between characters say blank. Decoding greedily takes each
frame's best class, merges a run of one class into one character, and drops
the blanks. A blank between two equal characters keeps both: "0 - 0" is "00",
while "0 0" is "0".

One confidence a read (decision 15): its least sure character, since one
wrong digit is the danger; the mean beside it.

The classes are M3's own (charset.CLASSES, the CRNN's) unless a reader
brings its own table, blank first: PP-OCRv6's 18,710 (reader/ppocr.py, S6).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .charset import BLANK, CLASSES


@dataclass(frozen=True)
class Read:
    """One crop's read: its text, in its reader's classes (M3's are upper
    case; PP-OCRv6 keeps its own case), and for each character the highest
    probability over the frames it spans."""
    text: str
    probs: tuple[float, ...]

    @property
    def confidence(self) -> float:
        """The least sure character's probability; 0 for an empty read."""
        return min(self.probs, default=0.0)

    @property
    def mean(self) -> float:
        return float(np.mean(self.probs)) if self.probs else 0.0


def greedy_decode(probs: np.ndarray, length: int | None = None,
                  classes: tuple[str, ...] = CLASSES) -> Read:
    """`probs`: [frames, classes], each frame's probabilities (a softmax).
    Only the first `length` frames are read (a crop's own frames, when a
    batch padded it on the right). `classes[k]` is class k's character,
    classes[BLANK] the blank."""
    probs = probs[:length]
    best = probs.argmax(axis=1)                       # each frame's best class
    p = probs[np.arange(len(best)), best]             # and its probability
    chars: list[str] = []
    char_probs: list[float] = []
    previous = BLANK
    for k, pk in zip(best.tolist(), p.tolist()):
        if k != BLANK:
            if k != previous:                         # a new character starts here
                chars.append(classes[k])
                char_probs.append(pk)
            else:                                     # the same character, one more frame
                char_probs[-1] = max(char_probs[-1], pk)
        previous = k                                  # a blank resets: the next k starts anew
    return Read("".join(chars), tuple(char_probs))
