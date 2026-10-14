from bestbefore.m1 import decide_c1, keep_synthetic


def scores(**map50):
    return {name: {"map50": {"date": v}} for name, v in map50.items()}


def test_clear_winner():
    assert decide_c1(scores(n640=0.80, n1280=0.84, s640=0.82, s1280=0.90)) == "s1280"


def test_within_one_point_the_cheapest_wins():
    # s1280 is best, but n1280 is within 0.01 and comes first in the order.
    assert decide_c1(scores(n640=0.80, n1280=0.895, s640=0.89, s1280=0.90)) == "n1280"
    # Exactly 1 point behind still counts as a tie.
    assert decide_c1(scores(n640=0.89, n1280=0.85, s640=0.85, s1280=0.90)) == "n640"


def test_keep_synthetic_unless_real_only_clearly_wins():
    assert keep_synthetic({"map50": {"date": 0.90}}, {"map50": {"date": 0.905}})
    assert keep_synthetic({"map50": {"date": 0.90}}, {"map50": {"date": 0.85}})
    assert not keep_synthetic({"map50": {"date": 0.90}}, {"map50": {"date": 0.92}})
