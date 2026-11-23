"""M2c's decision rules as S1 reads them (docs/m2c-live-boxes.md, "Decision
rules"): R0, R1 and R2's accuracy half, on made-up parity records."""
from bestbefore import live

T = 0.1641


def record(confs, ap, expiry=None, set_name="v1-val", variant="fp32-square"):
    """A parity record holding only what the rules read."""
    return {"set": set_name, "variant": variant, "threshold": T, "ap50_date": ap,
            "dates": {"conf": confs, "expiry": expiry or [False] * len(confs)}}


def test_r0_needs_the_same_dates_and_ap_within_0002():
    pt = record([0.9, 0.5, 0.1], 0.9596)
    assert live.r0(pt, record([0.9, 0.49, 0.12], 0.9586))["passed"]              # 0.001 apart, same dates
    assert live.r0(pt, record([0.9, 0.5, 0.1], 0.9616))["passed"]                # exactly 0.002: within
    assert not live.r0(pt, record([0.9, 0.5, 0.1], 0.9617))["passed"]
    swapped = live.r0(pt, record([0.9, 0.1, 0.5], 0.9596))                       # as many found, not the same
    assert not swapped["passed"] and not swapped["same_dates_found"]
    assert live.r0(pt, record([0.9, 0.5, -1.0], 0.9596))["passed"]     # missed by both (-1: never boxed): same


def test_r1_allows_one_date_fewer_and_001_of_ap_but_no_lost_expiry():
    square = record([0.9, 0.5, 0.3, 0.2], 0.95, expiry=[True, False, False, True])
    assert live.r1(square, record([0.9, 0.5, 0.3, 0.2], 0.94, expiry=[True, False, False, True]))["passed"]
    one_fewer = live.r1(square, record([0.9, 0.1, 0.3, 0.2], 0.945, expiry=[True, False, False, True]))
    assert one_fewer["passed"] and one_fewer["dates_found"] == [4, 3] and one_fewer["expiries_lost"] == 0
    assert not live.r1(square, record([0.9, 0.1, 0.1, 0.2], 0.95, expiry=[True, False, False, True]))["passed"]
    assert not live.r1(square, record([0.9, 0.5, 0.3, 0.2], 0.9399, expiry=[True, False, False, True]))["passed"]
    lost = live.r1(square, record([0.9, 0.5, 0.3, 0.1], 0.95, expiry=[True, False, False, True]))
    assert not lost["passed"] and lost["expiries_lost"] == 1                     # one date fewer, but an expiry
    gained = live.r1(record([0.1, 0.5], 0.9), record([0.5, 0.1], 0.9))          # no expiries: as many found
    assert gained["passed"] and "expiries_lost" not in gained


def test_r2s_accuracy_half_reads_like_r1_without_expiries():
    fp32 = record([0.9, 0.5, 0.3], 0.95, expiry=[True, True, True])
    assert live.r2_accuracy(fp32, record([0.9, 0.5, 0.1], 0.94))["passed"]
    assert not live.r2_accuracy(fp32, record([0.9, 0.1, 0.1], 0.95))["passed"]
    assert not live.r2_accuracy(fp32, record([0.9, 0.5, 0.3], 0.9399))["passed"]
