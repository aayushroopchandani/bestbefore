"""The greedy CTC decoder's spec (reader/ctc.py, docs/m3-date-reader.md S5),
written before the decoder.

A CTC reader gives, for each frame (a 4 px wide slice of the crop for the
CRNN), a probability for every class: the blank (class 0, "no character
here") and each character. Greedy decoding takes the best class of each
frame, merges a run of the same class into one character, and drops the
blanks. A blank between two equal characters keeps both: that is how CTC
writes "00".

Frames are written here as a string, one token a frame, "-" the blank.
"""
import numpy as np
import pytest

from bestbefore.reader import charset, ctc


def frames(spec: str, p: float = 0.9, ps: list[float] | None = None) -> np.ndarray:
    """Per-frame probabilities [T, C] whose best class spells `spec`: the
    named class gets p (or ps[t]), the rest share what is left."""
    tokens = spec.split()
    ps = ps or [p] * len(tokens)
    out = np.empty((len(tokens), len(charset.CLASSES)), np.float32)
    for t, (tok, pt) in enumerate(zip(tokens, ps)):
        k = charset.BLANK if tok == "-" else charset.index(tok)
        out[t] = (1 - pt) / (len(charset.CLASSES) - 1)
        out[t, k] = pt
    return out


@pytest.mark.parametrize("spec, text", [
    ("2 0 0", "20"),                        # a repeat merges
    ("2 0 - 0", "200"),                     # a blank between keeps both
    ("- - 2 2 - - 0 7 - -", "207"),         # blanks at the ends and inside dropped
    ("0 0 - 8 / / 2 - 2 0 2 7 7", "08/22027"),   # the owner's question (they guessed 08/2027)
    ("- - - -", ""),                        # all blanks: an empty read
    ("A A A", "A"),
    ("J A N - 2 0 2 7", "JAN2027"),
    ("J A N", "JAN"),
])
def test_greedy_decode_spells(spec, text):
    assert ctc.greedy_decode(frames(spec)).text == text


def test_space_is_a_character_not_a_blank():
    assert ctc.greedy_decode(frames("J A N - 2 0")).text == "JAN20"
    probs = frames("J A N - 2 0")
    probs[3] = (1 - 0.9) / (len(charset.CLASSES) - 1)
    probs[3, charset.index(" ")] = 0.9     # frame 3 now says space, not blank
    assert ctc.greedy_decode(probs).text == "JAN 20"


def test_empty_frames():
    read = ctc.greedy_decode(np.zeros((0, len(charset.CLASSES)), np.float32))
    assert read.text == "" and read.probs == ()


def test_each_kept_character_gets_the_highest_probability_of_its_frames():
    # 2 spans frames 0-1 (0.6, 0.9); 0 spans frame 3 (0.7): blanks carry none
    read = ctc.greedy_decode(frames("2 2 - 0", ps=[0.6, 0.9, 0.99, 0.7]))
    assert read.text == "20"
    assert read.probs == pytest.approx((0.9, 0.7))


def test_confidence_is_the_least_sure_character_and_the_mean_beside_it():
    read = ctc.greedy_decode(frames("0 8 / 2 0 2 7", ps=[0.99, 0.98, 0.97, 0.99, 0.95, 0.99, 0.4]))
    assert read.confidence == pytest.approx(0.4)         # decision 15: one unsure digit is the danger
    assert read.mean == pytest.approx(np.mean([0.99, 0.98, 0.97, 0.99, 0.95, 0.99, 0.4]))


def test_an_empty_read_has_confidence_zero():
    read = ctc.greedy_decode(frames("- - -"))
    assert read.confidence == 0.0 and read.mean == 0.0


def test_the_best_class_wins_even_below_one_half():
    probs = np.full((1, len(charset.CLASSES)), 0.5 / (len(charset.CLASSES) - 2), np.float32)
    probs[0, charset.BLANK] = 0.2
    probs[0, charset.index("7")] = 0.3          # 0.3 is under one half, but the most of any class
    assert ctc.greedy_decode(probs).text == "7"


def test_frames_beyond_a_length_are_not_read():
    probs = frames("1 2 - 3 4")
    assert ctc.greedy_decode(probs, length=3).text == "12"
