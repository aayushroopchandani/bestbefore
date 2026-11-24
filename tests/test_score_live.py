"""M2c: the app's parity benches scored against Ultralytics (scripts/score_live.py;
docs/m2c-live-boxes.md, "Records and scores"), on made-up photos and records."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import score_live  # noqa: E402

from bestbefore.samples import Box, Sample  # noqa: E402

T = 0.1641
DATE_A, DATE_B = (0.1, 0.1, 0.3, 0.2), (0.5, 0.5, 0.7, 0.6)


def samples():
    """Two photos: one date each, the second an expiry."""
    return [Sample("p0.jpg", Path("p0.jpg"), 100, 100, (Box("date", DATE_A),)),
            Sample("p1.jpg", Path("p1.jpg"), 100, 100, (Box("exp", DATE_B),))]


def photos(conf_a, conf_b, extra=()):
    """Per-photo boxes as the app and the job write them: [class, x1, y1, x2, y2, conf]."""
    return [{"index": 0, "boxes": [[0, *DATE_A, conf_a], *extra]}, {"index": 1, "boxes": [[0, *DATE_B, conf_b]]}]


def app_record(conf_a, conf_b, variant="fp32", extra=()):
    return {"kind": "bench-parity", "set": "s", "backend": "webgpu", "variant": variant, "threshold": T,
            "device": {"userAgent": "a browser"}, "model": {"sha256": "x"}, "photos": photos(conf_a, conf_b, extra)}


def test_scored_gives_detection_evals_ap_and_dates_found():
    s = score_live.scored(app_record(0.9, 0.1), samples())
    assert s["dates"]["conf"] == [0.9, 0.1] and s["dates"]["expiry"] == [False, True]
    assert s["at_threshold"]["dates_found"] == 1 and s["at_threshold"]["expiry_found"] == 0
    # Both found and nothing false gives 0.995, not 1: the AP integrates its 101
    # recall points by trapezoids down to a last point of precision 0 (Ultralytics' way).
    assert s["ap50_date"] == pytest.approx(0.995)
    with pytest.raises(ValueError, match="in order"):
        score_live.scored({**app_record(0.9, 0.9), "photos": photos(0.9, 0.9)[::-1]}, samples())


def test_the_check_allows_one_lost_date_and_001_of_ap():
    ref = {"threshold": T, "ap50_date": 0.95, "dates": {"conf": [0.9, 0.5, 0.3]}}
    app = lambda confs, ap: {"threshold": T, "ap50_date": ap, "dates": {"conf": confs}}  # noqa: E731
    assert score_live.s2_check(app([0.9, 0.5, 0.3], 0.95), ref)["passed"]
    one = score_live.s2_check(app([0.9, 0.1, 0.3], 0.945), ref)
    assert one["passed"] and one["lost"] == 1 and one["dates_found"] == [3, 2]
    assert not score_live.s2_check(app([0.9, 0.1, 0.1], 0.95), ref)["passed"]
    assert score_live.s2_check(app([0.9, 0.5, 0.3], 0.96), ref)["passed"]        # 0.01 over: within
    assert not score_live.s2_check(app([0.9, 0.5, 0.3], 0.9601), ref)["passed"]  # within either way, not only under
    assert not score_live.s2_check(app([0.9, 0.5, 0.3], 0.9399), ref)["passed"]


def test_box_by_box_pairs_boxes_of_one_class_at_iou_09():
    ref = photos(0.9, 0.5)
    same = score_live.box_by_box(photos(0.9, 0.5), ref, T)
    assert same["pairs"] == 2 and same["reference_missing_in_app"] == same["app_missing_in_reference"] == 0
    assert same["conf_move"]["max_abs"] == 0
    # A box that fell under the threshold still pairs, and its move is measured.
    fell = score_live.box_by_box(photos(0.9, 0.1), ref, T)
    assert fell["pairs"] == 2 and fell["conf_move"]["max_abs"] == pytest.approx(0.4)
    # Shifted to IoU under 0.9, or another class: unpaired on both sides.
    shifted = [{"index": 0, "boxes": [[0, 0.1, 0.12, 0.3, 0.22, 0.9]]}, ref[1]]
    moved = score_live.box_by_box(shifted, ref, T)
    assert (moved["pairs"], moved["reference_missing_in_app"], moved["app_missing_in_reference"]) == (1, 1, 1)
    other = [{"index": 0, "boxes": [[3, *DATE_A, 0.9]]}, ref[1]]
    assert score_live.box_by_box(other, ref, T)["reference_missing_in_app"] == 1
    # A box under the threshold on both sides isn't counted.
    low = score_live.box_by_box(photos(0.9, 0.1), photos(0.9, 0.1), T)
    assert low["kept"] == {"reference": 1, "app": 1} and low["pairs"] == 1


def test_records_are_scored_against_the_reference_with_r2_for_float16(tmp_path, monkeypatch):
    from bestbefore import sets

    monkeypatch.setattr(sets, "load_set", lambda name: samples())
    job = {"photos": 2, "threshold": T, "ap50_date": 0.995, "per_photo": photos(0.9, 0.5)}
    (tmp_path / "parity-s-fp32-square.json").write_text(json.dumps(job))
    fp32, fp16 = tmp_path / "bench-parity-s-fp32.json", tmp_path / "bench-parity-s-fp16.json"
    fp32.write_text(json.dumps(app_record(0.9, 0.5)))
    fp16.write_text(json.dumps(app_record(0.9, 0.1, variant="fp16")))
    out = score_live.score_records([fp32, fp16], tmp_path)
    assert out[fp32.name]["check"]["passed"] and out[fp32.name]["whole_set"]
    assert out[fp32.name]["reference"]["ap50_date"] == pytest.approx(out[fp32.name]["reference"]["job_ap50_date"])
    assert "R2_accuracy" not in out[fp32.name]
    r2 = out[fp16.name]["R2_accuracy"]
    assert r2["against"] == fp32.name and r2["dates_found"] == [2, 1] and r2["passed"]   # one date fewer: allowed
    assert out[fp16.name]["box_by_box"]["conf_move"]["max_abs"] == pytest.approx(0.4)
    bad = tmp_path / "bench-speed-s-fp32.json"
    bad.write_text(json.dumps({**app_record(0.9, 0.5), "kind": "bench-speed"}))
    with pytest.raises(SystemExit, match="isn't a parity bench"):
        score_live.score_records([bad], tmp_path)
