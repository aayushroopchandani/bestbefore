"""bestbefore.final_test: the final test's records, store set v2's causes
and design groups, and one run's breakdown (docs/m2b-store-and-size.md,
S14). Made-up photos and labels only."""
from pathlib import Path

import pytest

from bestbefore import final_test as ft
from bestbefore import m1, m2, m2b
from bestbefore.detection_eval import Prediction, average_precision, evaluate
from bestbefore.samples import Box, Sample

SIZE = (1200, 1600)                     # upright width, height: the long side is 1,600
WIDE = (0.10, 0.50, 0.40, 0.52)         # a line across the photo, 2% of the long side high
TALL = (0.50, 0.10, 0.52, 0.40)         # a line up the photo
SMALL = (0.10, 0.70, 0.30, 0.71)        # 1% high
BIG = (0.10, 0.20, 0.50, 0.25)          # 5% high


def label(tag, fmt, *boxes, derived=False):
    return {"tag": tag, "format": fmt, "derived": derived,
            "boxes": [{"label": kind, "xyxy": list(xyxy)} for kind, xyxy in boxes]}


def test_the_records_are_the_plans_thirty_in_its_order():
    assert len(ft.RUNS) == len(set(ft.RUNS)) == 30
    assert ft.RUNS[:3] == ("n640", "n640-seed1", "n640-seed2")
    assert ft.RUNS[3:6] == tuple(f"m2-gen-rot-1280-s{s}" for s in range(3))
    assert ft.RUNS[6:9] == tuple(f"m2b-n1280-s{s}" for s in range(3))     # the control: M2's weights again
    assert ft.DETECTOR == "m2b-n1280-ft-s0" and ft.DETECTOR in ft.RUNS
    assert ft.EXPDATE_TEST == tuple(f"m2b-n1280-ft-s{s}" for s in range(3))
    # Every weights file the final test reads has its SHA-256 recorded, and nothing else does.
    assert {r.removeprefix("m2b-") for r in ft.RUNS[9:]} == set(m2b.LAST_PT_SHA256)
    assert set(m1.BEST_PT_SHA256) == set(ft.M1_RUNS)
    assert set(m2.LAST_PT_SHA256) == {r.removeprefix("m2-") for r in ft.M2_RUNS}


def test_causes_are_m2s_with_month_words_as_one_and_upside_down_apart():
    assert ft.tall(TALL, SIZE) and not ft.tall(WIDE, SIZE)
    assert not ft.tall((0.0, 0.0, 0.4, 0.3), SIZE)       # 480 px each way: not taller, so wide
    cases = [
        (label("clean", "MON-YYYY", ("expiry", WIDE)), [ft.MONTH_WORD]),
        (label("clean", "Month YYYY", ("expiry", WIDE)), [ft.MONTH_WORD]),      # v1's full names too
        (label("dot-matrix", "DDMONYYYY", ("expiry", SMALL)), [ft.MONTH_WORD]),
        (label("clean", "DD/MM/YYYY", ("expiry", WIDE), ("mfg", SMALL)), ["numbers: print 1.5 to 2.5%"]),
        (label("small-print", "MM/YYYY", ("expiry", SMALL)), ["numbers: print under 1.5%"]),
        (label("curved", "YYYYMMDD", ("expiry", BIG)), ["numbers: print 2.5% and up"]),
        (label("sideways", "MON-YYYY", ("expiry", TALL)), [ft.SIDEWAYS]),        # sideways comes first
        (label("sideways", "DD/MM/YYYY", ("expiry", WIDE)), [ft.UPSIDE_DOWN]),
        (label("sideways", "DD/MM/YYYY", ("expiry", TALL), ("expiry", WIDE)), [ft.SIDEWAYS, ft.UPSIDE_DOWN]),
        (label("other-face", "", ("mfg", WIDE)), []),
        (label("clean", "DD/MM/YYYY", ("mfg", WIDE), derived=True), []),        # worked out: no box
    ]
    for lab, causes in cases:
        assert ft.expiry_causes(lab, SIZE) == causes, lab
    assert set(ft.CAUSES) == {ft.MONTH_WORD, *ft.NUMBERS, ft.SIDEWAYS, ft.UPSIDE_DOWN}
    with pytest.raises(ValueError, match="other-face"):
        ft.expiry_causes(label("other-face", "", ("expiry", WIDE)), SIZE)


def test_design_groups_count_v3_train_as_training():
    designs = {"v1/a": "A", "v1/b": "B", "v1/c": "C"}
    split = {"v1/a": "train", "v1/b": "val", "v1/c": "val"}
    designs_v3 = {"v3/p": "A", "v3/q": "D", "v3/r": "E", "v3/s": "C"}
    split_v3 = {"v3/p": "train", "v3/q": "train", "v3/r": "val", "v3/s": "train"}
    train, validation = ft.design_sides(designs, split, designs_v3, split_v3)
    assert train == {"A", "C", "D"} and validation == {"B", "E"}      # trained on wins over validated on
    groups = {d: ft.design_group(d, train, validation) for d in "ABCDEZ"}
    assert groups == {"A": ft.SEEN, "B": ft.VALIDATION, "C": ft.SEEN, "D": ft.SEEN,
                      "E": ft.VALIDATION, "Z": ft.UNSEEN}


def photo(id, tag, *boxes):
    return Sample(id=id, image=Path(id), width=SIZE[0], height=SIZE[1],
                  boxes=tuple(Box(kind, xyxy) for kind, xyxy in boxes), tag=tag)


@pytest.fixture
def v2():
    """Four photos: seen, validation-only and unseen designs; a found
    expiry with a false box beside it, a sideways one boxed below the
    threshold, an other-face photo, and a dot-matrix month word missed."""
    samples = [photo("a", "clean", ("expiry", WIDE), ("mfg", SMALL)),
               photo("b", "sideways", ("expiry", TALL)),
               photo("c", "other-face", ("mfg", SMALL)),
               photo("d", "dot-matrix", ("expiry", WIDE))]
    groups = {"a": ft.SEEN, "b": ft.UNSEEN, "c": ft.UNSEEN, "d": ft.VALIDATION}
    causes = {"a": ["numbers: print 1.5 to 2.5%"], "b": [ft.SIDEWAYS], "c": [], "d": [ft.MONTH_WORD]}
    return ft.V2(samples, groups, causes)


PREDS = {"a": [Prediction("date", WIDE, 0.9), Prediction("date", (0.7, 0.7, 0.8, 0.8), 0.5)],
         "b": [Prediction("date", TALL, 0.2)],
         "c": [Prediction("date", SMALL, 0.6), Prediction("due", (0.0, 0.0, 0.05, 0.05), 0.9)],
         "d": []}


def test_a_runs_breakdown_by_cause_adds_up_to_its_expiries(v2):
    got = ft.breakdown(v2, PREDS, 0.3)["by_cause"]
    found = {c: (got[c]["iou0.5"]["expiry_found"], got[c]["iou0.5"]["expiry_dates"]) for c in ft.CAUSES}
    assert found == {ft.MONTH_WORD: (0, 1), "numbers: print under 1.5%": (0, 0),
                     "numbers: print 1.5 to 2.5%": (1, 1), "numbers: print 2.5% and up": (0, 0),
                     ft.SIDEWAYS: (0, 1), ft.UPSIDE_DOWN: (0, 0)}
    assert got[ft.SIDEWAYS]["any_conf"]["expiry_found"] == 1       # boxed, below the threshold
    total = evaluate(v2.samples, PREDS)
    assert sum(got[c]["iou0.5"]["expiry_dates"] for c in ft.CAUSES) == sum(total.truth_expiry) == 3
    with pytest.raises(ValueError, match="more causes"):
        ft.cause_counts(v2.samples, total, 0.3, {**v2.causes, "c": [ft.MONTH_WORD]})


def test_a_runs_breakdown_by_design_has_precision_and_our_ap_per_group(v2):
    out = ft.breakdown(v2, PREDS, 0.3)
    seen, unseen, val = (out["by_design"][g]["iou0.5"] for g in (ft.SEEN, ft.UNSEEN, ft.VALIDATION))
    assert (seen["photos"], seen["dates"], seen["dates_found"], seen["expiry_found"]) == (1, 2, 1, 1)
    assert seen["precision"] == 0.5 and seen["false_boxes_per_photo"] == 1.0
    assert (unseen["photos"], unseen["dates"], unseen["dates_found"], unseen["expiry_found"]) == (2, 2, 1, 0)
    assert (val["photos"], val["expiry_dates"], val["expiry_found"], val["boxes_kept"]) == (1, 1, 0, 0)
    assert out["by_design"][ft.UNSEEN]["any_conf"]["dates_found"] == 2
    assert sum(out["by_design"][g]["iou0.5"]["dates"] for g in ft.DESIGN_GROUPS) == 5
    # Our AP per group is the AP of that group's photos scored alone.
    for group in ft.DESIGN_GROUPS:
        alone = [s for s in v2.samples if v2.groups[s.id] == group]
        assert out["ap50"]["by_design"][group] == average_precision(evaluate(alone, PREDS))
    assert out["ap50"]["by_tag"]["sideways"] == average_precision(evaluate(v2.samples[1:2], PREDS))
    assert out["ap50"]["date"] == average_precision(evaluate(v2.samples, PREDS))
    assert "curved" not in out["ap50"]["by_tag"]              # no photo has that tag here


def test_v2s_counts_name_no_photo(v2):
    text = ft.counts(v2)
    assert "| design | seen in training | 1 | 1 | 2 |" in text
    assert "| cause | sideways | – | 1 | – |" in text
    assert "| all | every photo | 4 | 3 | 5 |" in text
    assert not any(f"| {name} |" in text for name in "abcd")
