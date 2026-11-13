"""S2's crop experiment scored (reader/crop_scores.py), on made-up reads."""
from bestbefore import m3
from bestbefore.reader import crop_scores


def read(key, reader, mode, pad, k, text, score=0.9, ms=10.0):
    return {"key": key, "reader": reader, "mode": mode, "pad": pad, "k": k, "text": text,
            "score": score, "ms": ms}


def reads_for(crop, text_of):
    """Every read the read step makes of `crop`; text_of(reader, mode, pad, k) -> (text, score)."""
    ks = (0, 1, 3) if crop["shape"] == "tall" else (0, 2)
    out = []
    for reader in m3.CROP_READERS:
        for pad in m3.PADS:
            for mode, turns in (("full", (0,)), ("alone", ks)):
                for k in turns:
                    text, score = text_of(reader, mode, pad, k)
                    out.append(read(crop["key"], reader, mode, pad, k, text, score))
    return out


STORE = {"key": "s0001", "set": "store", "part": "val", "tag": "sideways", "shape": "tall",
         "label": "expiry", "photo": "v1-0001", "truth": "2027-03", "truth_text": None}
REAL = {"key": "r0001", "set": "real-val", "part": "val", "tag": None, "shape": "wide",
        "label": "date", "photo": "img_1.jpg", "truth": None, "truth_text": "2021.08.15"}


def test_fold():
    assert crop_scores.fold("  exp  03/27 ") == "EXP 03/27"


def test_truths():
    assert str(crop_scores.truth_of(STORE)[0]) == "2027-03"
    assert str(crop_scores.truth_of(REAL)[0]) == "2021-08-15"
    assert crop_scores.truth_of({**REAL, "truth_text": "02.18"})[1] is True
    assert crop_scores.truth_of({**REAL, "truth_text": "LOT 5"})[0] is None


def test_choose_turn():
    a, b = {"text": "LZOZ EVW", "score": 0.95}, {"text": "MAR 2027", "score": 0.6}
    assert crop_scores.choose_turn([a, b]) is b                       # the read with a date wins
    c = {"text": "MAR 2021", "score": 0.7}
    assert crop_scores.choose_turn([b, c]) is c                       # both: the more confident
    d = {"text": "####", "score": 0.7}
    assert crop_scores.choose_turn([d, a]) is a                       # neither: the more confident
    assert crop_scores.choose_turn([d, dict(d)]) is d                 # a tie: the first
    assert crop_scores.choose_turn([{"text": "", "score": None}, d]) is d


def test_score_rows_and_tally():
    def store_text(reader, mode, pad, k):
        if mode == "full":
            return ("", None) if pad == 0.0 else ("MAR 2021", 0.8)
        return {0: ("|||", 0.3), 1: ("MAR 2027", 0.9), 3: ("LZOZ", 0.5)}[k]

    def real_text(reader, mode, pad, k):
        return ("2021.08.15", 0.99) if k == 0 else ("SI'80'IZOZ", 0.4)

    reads = reads_for(STORE, store_text) + reads_for(REAL, real_text)
    rows = crop_scores.score_rows([STORE, REAL], reads)
    assert len(rows) == 2 * len(m3.CROP_READERS) * len(m3.CROP_MODES) * len(m3.PADS)
    by = {(r["key"], r["config"]): r for r in rows}
    assert by[("s0001", "rapidocr/full/0")]["outcome"] == "none"
    assert by[("s0001", "rapidocr/full/0.4")]["outcome"] == "wrong"
    assert by[("s0001", "rapidocr/alone/0.4")]["outcome"] == "none"
    turned = by[("s0001", "ppocrv6/turned/0.15")]
    assert (turned["outcome"], turned["k"], turned["ms"]) == ("right", 1, 20.0)
    real = by[("r0001", "ppocrv6/turned/0")]
    assert (real["outcome"], real["exact"], real["k"]) == ("right", True, 0)

    counts = crop_scores.tally(rows)
    assert counts["ppocrv6/turned/0"]["tag:sideways"] == {
        "n": 1, "right": 1, "wrong": 0, "none": 0, "no_truth": 0, "exact": 0, "multi": 0}
    assert counts["rapidocr/alone/0"]["real-val"]["exact"] == 1
    assert set(counts["rapidocr/alone/0"]) == {"real-val", "store", "store-val", "tag:sideways",
                                               "shape:tall", "label:expiry"}
    assert crop_scores.group_order(counts["rapidocr/alone/0"])[:3] == ["real-val", "store", "store-val"]
    ms = crop_scores.mean_ms(rows)
    assert (ms["rapidocr/alone/0"], ms["rapidocr/turned/0"]) == (10.0, 20.0)   # turned: both reads
    assert m3.crop_recipe(rows)["turn"] is True
