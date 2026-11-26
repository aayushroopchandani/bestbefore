"""M2c's decision rules as S1 reads them (docs/m2c-live-boxes.md, "Decision
rules"): R0, R1 and R2's accuracy half, on made-up parity records; G, on
made-up Motion records."""
import json

import pytest

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


# --- Rule G: the stillness gate's threshold T, from a Motion record (S3) ---------------

def motion_record(still, moving, still2=None, moving2=None):
    """A Motion record as the app posts it: per frame its phase and its motion
    values against the frame before (value1) and the one before that (value2)."""
    still2 = still if still2 is None else still2
    moving2 = moving if moving2 is None else moving2
    return {"frames": {"phase": ["still"] * len(still) + ["moving"] * len(moving),
                       "value1": [*still, *moving], "value2": [*still2, *moving2]}}


def test_rule_g_is_the_geometric_mean_of_still_p95_and_moving_p5():
    still = [0.1 * i for i in range(1, 21)]             # 0.1 .. 2.0: p95 1.905
    moving = [5.0 + i for i in range(20)]              # 5 .. 24: p5 5.95
    g = live.rule_g(still, moving)
    assert g["on"] and g["still_p95"] == pytest.approx(1.905) and g["moving_p5"] == pytest.approx(5.95)
    assert g["t"] == pytest.approx((1.905 * 5.95) ** 0.5)
    assert (g["still"], g["moving"]) == (20, 20)


def test_rule_g_turns_the_gate_off_when_still_and_moving_overlap():
    g = live.rule_g([1.0] * 10 + [6.0] * 10, [6.0] * 20)  # p95 of still 6.0 = p5 of moving: not below
    assert not g["on"] and g["t"] is None
    assert not live.rule_g([5.0] * 20, [1.0] * 20)["on"]


def test_gate_reads_a_motion_record_at_the_gates_cadence(tmp_path):
    path = tmp_path / "motion-001.json"
    path.write_text(json.dumps(motion_record([1.0, 1.0, None], [8.0, 9.0], still2=[2.0, 2.0, None], moving2=[16.0, 18.0])))
    every_frame = live.gate(path, every=1)
    assert every_frame["t"] == pytest.approx((1.0 * 8.05) ** 0.5) and every_frame["every"] == 1
    assert every_frame["still"] == 2                    # a frame with no value (the first) is left out
    every_other = live.gate(path, every=2)
    assert every_other["t"] == pytest.approx((2.0 * 16.1) ** 0.5) and every_other["every"] == 2
    assert every_other["record"] == str(path)


def test_gate_is_none_until_a_motion_record_is_pinned(monkeypatch):
    monkeypatch.setattr(live, "GATE_MOTION", None)
    assert live.gate() is None


def test_a_pinned_record_is_named_under_the_runs_folder_not_by_its_full_path(monkeypatch, tmp_path):
    # The config goes to the phone: the laptop's own folders stay out of it.
    from bestbefore import runs
    (tmp_path / "m2c" / "run").mkdir(parents=True)
    (tmp_path / "m2c" / "run" / "motion-001.json").write_text(json.dumps(motion_record([1.0, 1.0], [8.0, 9.0])))
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(live, "GATE_MOTION", "run/motion-001.json")
    monkeypatch.setattr(live, "GATE_EVERY", 1)
    g = live.gate()
    assert g["on"] and g["record"] == "run/motion-001.json"
