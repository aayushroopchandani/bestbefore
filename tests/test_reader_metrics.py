"""Text metrics' spec (reader/metrics.py, docs/m3-date-reader.md "Shared"),
written before the code: exact text and the character error rate, both
after folding case and runs of spaces (decision 10)."""
import pytest

from bestbefore.reader import metrics


@pytest.mark.parametrize("a, b, d", [
    ("", "", 0),
    ("08/2027", "08/2027", 0),
    ("08/2021", "08/2027", 1),      # one substitution
    ("08/22027", "08/2027", 1),     # one insertion
    ("8/2027", "08/2027", 1),       # one deletion
    ("kitten", "sitting", 3),       # the textbook case: k->s, e->i, +g
    ("", "08/2027", 7),
    ("2027", "7202", 2),
])
def test_edit_distance(a, b, d):
    assert metrics.edit_distance(a, b) == d
    assert metrics.edit_distance(b, a) == d     # the same both ways


def test_cer_is_edits_over_the_truths_length():
    assert metrics.cer("08/2021", "08/2027") == pytest.approx(1 / 7)    # the slice's case
    assert metrics.cer("08/2027", "08/2027") == 0


def test_cer_of_an_empty_read_is_one_and_a_long_read_can_pass_one():
    assert metrics.cer("", "08/2027") == 1                           # the owner guessed right
    assert metrics.cer("08/2027 08/2027", "08/2027") == pytest.approx(8 / 7)


def test_cer_folds_case_and_spaces():
    assert metrics.cer("jan  2027", "JAN 2027") == 0
    assert metrics.cer(" Jan 2027 ", "JAN 2027") == 0
    assert metrics.cer("JAN2027", "JAN 2027") == pytest.approx(1 / 8)  # a missing space is still an edit


def test_cer_needs_a_truth():
    with pytest.raises(ValueError):
        metrics.cer("08/2027", "  ")


def test_exact_text():
    assert metrics.exact("Exp: Jan 2027", "EXP: JAN  2027")
    assert not metrics.exact("08/2021", "08/2027")
    assert metrics.exact("08–2027", "08—2027")      # both dashes fall outside the set: '?'
