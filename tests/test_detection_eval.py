from pathlib import Path

import pytest

from bestbefore.detection_eval import (Prediction, best_threshold, evaluate,
                                       iou, match_photo, pooled, summary)
from bestbefore.samples import Box, Sample

DATE = (0.10, 0.40, 0.30, 0.45)        # a labelled date
EXPIRY = (0.10, 0.50, 0.30, 0.55)      # a labelled expiry date


def photo(id, *boxes, tag=None):
    return Sample(id=id, image=Path(id), width=1000, height=1000,
                  boxes=tuple(Box(label, xyxy) for label, xyxy in boxes), tag=tag)


def pred(xyxy, conf, label="date"):
    return Prediction(label, xyxy, conf)


def test_iou():
    assert iou(DATE, DATE) == pytest.approx(1.0)
    assert iou(DATE, (0.5, 0.5, 0.6, 0.6)) == 0.0
    # Same height, half the width, inside: overlap / union = 0.5
    assert iou(DATE, (0.10, 0.40, 0.20, 0.45)) == pytest.approx(0.5)


def test_perfect_match():
    s = photo("a", ("date", DATE), ("exp", EXPIRY))
    m = evaluate([s], {"a": [pred(DATE, 0.9), pred(EXPIRY, 0.8)]})
    r = summary(m, conf=0.5)
    assert (r["dates"], r["dates_found"], r["precision"]) == (2, 2, 1.0)
    assert (r["expiry_dates"], r["expiry_found"]) == (1, 1)
    assert r["false_boxes_per_photo"] == 0.0


def test_near_miss_counts_only_at_the_lower_iou():
    # Box 0.45 of the way into the date's area: IoU = 0.45.
    near = (0.10, 0.40, 0.19, 0.45)
    s = photo("a", ("date", DATE))
    assert iou(DATE, near) == pytest.approx(0.45)
    at_05 = summary(evaluate([s], {"a": [pred(near, 0.9)]}, iou_min=0.5), 0.5)
    at_03 = summary(evaluate([s], {"a": [pred(near, 0.9)]}, iou_min=0.3), 0.5)
    assert at_05["dates_found"] == 0 and at_05["precision"] == 0.0
    assert at_03["dates_found"] == 1


def test_one_box_cannot_count_for_two_dates():
    # Two dates on adjacent lines and one tall box covering both: IoU 0.5
    # with each, so at the "roughly found" level (0.3) it could match either.
    top, bottom = (0.1, 0.40, 0.3, 0.45), (0.1, 0.45, 0.3, 0.50)
    both = (0.1, 0.40, 0.3, 0.50)
    pred_tp, taken = match_photo([pred(both, 0.9)], [top, bottom], iou_min=0.3)
    assert [tp for _, tp in pred_tp] == [True]
    assert sum(c is not None for c in taken) == 1


def test_most_confident_prediction_matches_first():
    s = photo("a", ("date", DATE))
    m = evaluate([s], {"a": [pred(DATE, 0.3), pred(DATE, 0.9)]})
    assert m.truth_conf == [0.9]               # matched by the 0.9 box
    assert sorted(m.pred_tp) == [False, True]  # the 0.3 duplicate is a false box


def test_empty_photo_with_a_false_box():
    s = photo("a")
    r = summary(evaluate([s], {"a": [pred(DATE, 0.9)]}), conf=0.5)
    assert r["dates"] == 0 and r["date_recall"] is None
    assert r["false_boxes_per_photo"] == 1.0


def test_only_date_predictions_count():
    s = photo("a", ("date", DATE))
    r = summary(evaluate([s], {"a": [pred(DATE, 0.9, label="due")]}), 0.5)
    assert r["dates_found"] == 0 and r["boxes_kept"] == 0


def test_threshold_and_tags():
    a = photo("a", ("expiry", EXPIRY), tag="dot-matrix")
    b = photo("b", ("expiry", EXPIRY), tag="clean")
    m = evaluate([a, b], {"a": [pred(EXPIRY, 0.4)], "b": [pred(EXPIRY, 0.8)]})
    assert summary(m, 0.5)["expiry_found"] == 1      # only b's box survives
    assert summary(m, 0.3)["expiry_found"] == 2
    assert summary(m, 0.5, tags={"dot-matrix"})["expiry_found"] == 0
    assert summary(m, 0.5, tags={"clean"})["photos"] == 1


def test_best_threshold():
    # Three dates; boxes at 0.9 (right), 0.8 (wrong), 0.7 (right), 0.2 (right).
    # Keeping down to 0.7: P = 2/3, R = 2/3, F1 = 0.667.
    # Keeping down to 0.2: P = 3/4, R = 3/3, F1 = 0.857  <- best
    d1, d2, d3 = (0.1, 0.1, 0.2, 0.15), (0.1, 0.3, 0.2, 0.35), (0.1, 0.5, 0.2, 0.55)
    s = photo("a", ("date", d1), ("date", d2), ("date", d3))
    preds = [pred(d1, 0.9), pred((0.6, 0.6, 0.7, 0.65), 0.8),
             pred(d2, 0.7), pred(d3, 0.2)]
    conf, f1 = best_threshold(evaluate([s], {"a": preds}))
    assert conf == pytest.approx(0.2)
    assert f1 == pytest.approx(2 * 0.75 * 1.0 / 1.75)


def test_f2_picks_a_lower_threshold_than_f1():
    # Four dates; boxes at 0.9, 0.8, 0.7 (right), 0.4, 0.3 (wrong), 0.2 (right).
    # Keeping down to 0.7: P = 1,   R = 3/4  ->  F1 0.857 (best), F2 0.789
    # Keeping down to 0.2: P = 4/6, R = 1    ->  F1 0.800,        F2 0.909 (best)
    dates = [(0.1, y, 0.2, y + 0.05) for y in (0.1, 0.3, 0.5, 0.7)]
    wrong = [(0.6, y, 0.7, y + 0.05) for y in (0.1, 0.3)]
    s = photo("a", *(("date", d) for d in dates))
    preds = [pred(dates[0], 0.9), pred(dates[1], 0.8), pred(dates[2], 0.7),
             pred(wrong[0], 0.4), pred(wrong[1], 0.3), pred(dates[3], 0.2)]
    m = evaluate([s], {"a": preds})
    assert best_threshold(m) == pytest.approx((0.7, 6 / 7))
    assert best_threshold(m, beta=1) == best_threshold(m)          # M1's calls unchanged
    assert best_threshold(m, beta=2) == pytest.approx((0.2, 5 * (2 / 3) / (4 * (2 / 3) + 1)))


def test_pooled_matches_are_the_sets_scored_together():
    a = photo("a", ("date", DATE), tag="x")
    b = photo("b", ("exp", EXPIRY), ("date", DATE))
    preds = {"a": [pred(DATE, 0.9)], "b": [pred(EXPIRY, 0.6), pred((0.5, 0.5, 0.6, 0.6), 0.7)]}
    together = evaluate([a, b], preds)
    pool = pooled([evaluate([a], preds), evaluate([b], preds)])
    assert pool == together
    assert summary(pool, 0.5) == summary(together, 0.5)
    with pytest.raises(ValueError, match="IoU"):
        pooled([evaluate([a], preds, 0.5), evaluate([b], preds, 0.3)])
