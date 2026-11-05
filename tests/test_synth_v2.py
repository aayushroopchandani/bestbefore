"""gen-v2's text and line spacing (docs/m2b-store-and-size.md, "The generator,
v2", S7).

M2's generator must not move: GEN and synth-val's strata hash as before;
M2's text draws exactly what it drew (a digest of 10,000 blocks and of the
random generator's state after them, taken from M2's code before S7);
gen-v2's print changes nothing for M2's kinds; and S5's 500-photo sample is
remade byte for byte (it needs ExpDate, the fonts and the sample on disk,
and is skipped without them). Then each new form, its text and its boxes;
gen-v2's shares within 3 points over 10,000 blocks; and the rupee sign only
where every font has it.

The text tests need nothing; those that draw need the fonts
(python scripts/fetch_fonts.py).
"""
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, replace
from multiprocessing import get_context

import numpy as np
import pytest

from bestbefore.synth import fonts, render as R, text as T
from bestbefore.synth.spec import (GEN, SET_SEEDS, STYLE, SYNTH_VAL, TEXT, TEXT_V2, V2, spec_dict, spec_hash,
                                   strata_hash)

needs_fonts = pytest.mark.skipif(not fonts.fetched(), reason="fonts not fetched: python scripts/fetch_fonts.py")

GEN_SHA256 = "3fc141781fcfe5bfe2827c08159bf0ce79b713d53a6f1634200a3fca8dfd0103"       # M2's manifests
STRATA_SHA256 = "ba77f1bd918d9c445c603645881c14b85a5bfd5583b048b6503fc690d88058ba"
M2_TEXT_DIGEST = "c74bec80b7986cfa11c9a00f7e1c8199eea915454918981e53840014963d48ca"   # M2's code, before S7
RUPEE = "₹"
WORDS = ("due", "prod")
BARCODE = r"\d{13}|\d \d{6} \d{6}"         # run together, or spaced 1-6-6


def _blocks(n: int, seed: int, spec=TEXT_V2, v2=V2.text) -> list[T.Block]:
    rng = np.random.default_rng(seed)
    return [T.random_block(rng, spec, v2) for _ in range(n)]


def _only(kind: str):
    return replace(TEXT_V2, kinds=((kind, 1.0),))


def _word_dates(block: T.Block) -> list[tuple[T.Token, T.Token]]:
    """Every word printed before a date on its line, with that date."""
    return [(line[i - 1], t) for line in block.lines for i, t in enumerate(line)
            if t.cls == "date" and i and line[i - 1].cls in WORDS]


def _months(a: T.Token, b: T.Token) -> int:
    return (b.parts[0] - a.parts[0]) * 12 + b.parts[1] - a.parts[1]


def _mfg_first(block: T.Block) -> bool:
    """Whether the mfg line comes above the expiry's (a relative block's:
    above its shelf life's)."""
    mfg = [i for i, line in enumerate(block.lines) if any(t.role == "mfg" for t in line)]
    other = [i for i, line in enumerate(block.lines)
             if any(t.role == "expiry" for t in line) or line[0].text == T.RELATIVE_DUE]
    return mfg[0] < other[0]


# --- M2's generator, unmoved ---------------------------------------------------------------------

def test_m2s_spec_and_strata_hash_as_before():
    assert spec_hash(GEN) == GEN_SHA256 and strata_hash() == STRATA_SHA256
    assert GEN.v2 is None and all(spec.v2 is None for _, spec in SYNTH_VAL)
    assert spec_dict(GEN) == {k: v for k, v in asdict(GEN).items() if k != "v2"}
    with_v2 = replace(GEN, v2=V2)
    assert spec_dict(with_v2)["v2"] == asdict(V2) and spec_hash(with_v2) != GEN_SHA256
    other = replace(GEN, v2=replace(V2, text=replace(V2.text, glued=0.31)))
    assert spec_hash(other) != spec_hash(with_v2)                # every v2 number is in a v2 spec's hash


def test_m2s_text_draws_exactly_what_it_drew():
    rng = np.random.default_rng(12345)
    digest = hashlib.sha256()
    for _ in range(10_000):
        b = T.random_block(rng)
        digest.update(repr((b.kind, tuple((t.text, t.cls, t.role, t.parts, t.format, t.space_before)
                                          for t in b.tokens), tuple(len(line) for line in b.lines))).encode())
    digest.update(repr(rng.bit_generator.state).encode())       # not one draw more or fewer
    assert digest.hexdigest() == M2_TEXT_DIGEST


@needs_fonts
@pytest.mark.parametrize("style", R.STYLES)
def test_gen_v2s_print_changes_nothing_for_m2s_kinds(style):
    rng = np.random.default_rng(R.STYLES.index(style))
    for _ in range(15):
        block, family = T.random_block(rng), fonts.random_family(rng, "train")
        a, b = np.random.default_rng(1), np.random.default_rng(1)
        m2_layer = R.render(block, style, family, 12, a)
        v2_layer = R.render(block, style, family, 12, b, v2=V2.style)
        assert m2_layer.params == v2_layer.params and np.array_equal(R.ink(m2_layer), R.ink(v2_layer))
        assert a.bit_generator.state == b.bit_generator.state


def _sample_files():
    """S5's sample on disk (images, sidecar lines) and its background pool,
    or None where there is none: no ExpDate, fonts or sample."""
    if not fonts.fetched():
        return None
    try:
        from bestbefore import m2
        from bestbefore.synth import batch
        images = batch.images_dir("gen-sample")
        lines = m2.sidecar_path(m2.GENERATED_SETS["gen-sample"]).read_text().splitlines()
        manifest = json.loads(m2.manifest_path("gen-sample").read_text())
    except (RuntimeError, OSError):
        return None
    if len(lines) != 500 or manifest["spec_sha256"] != GEN_SHA256:
        return None
    return images, lines


_remade: dict = {}


def _remake_start(pool) -> None:
    import cv2
    cv2.setNumThreads(1)
    _remade["pool"] = pool


def _remake(i: int) -> tuple[int, bytes, str, str]:
    from bestbefore.synth import make
    made = make.make_one(i, GEN, _remade["pool"], SET_SEEDS["gen-sample"])
    return i, made.jpeg, made.labels, json.dumps(made.sidecar, separators=(",", ":"))


@pytest.mark.skipif(_sample_files() is None, reason="needs ExpDate, the fonts and S5's gen-sample on disk")
def test_s5s_500_photo_sample_is_remade_byte_for_byte():
    """Every JPEG, label file and sidecar line of S5's sample (made with
    M2's code), made again with GEN, in memory: nothing is written."""
    from bestbefore import derived
    from bestbefore.synth import batch, make
    images, lines = _sample_files()
    pool = batch.load_pools(["gen-sample"])["gen-sample"]
    differ = []
    with get_context("spawn").Pool(4, initializer=_remake_start, initargs=(pool,)) as workers:
        for i, jpeg, labels, line in workers.imap(_remake, range(500), chunksize=8):
            path = images / make.image_name(i)
            for what, ok in (("jpeg", jpeg == path.read_bytes()),
                             ("labels", labels == derived.label_path(path).read_text()),
                             ("sidecar", line == lines[i])):
                if not ok:
                    differ.append((i, what))
    assert not differ, differ[:10]


# --- Glued words ---------------------------------------------------------------------------------

def test_a_glued_date_follows_a_word_that_ends_in_a_colon_or_a_full_stop():
    blocks = _blocks(3000, seed=1)
    pairs = [p for b in blocks for p in _word_dates(b)]
    glued = [(w, d) for w, d in pairs if not d.space_before]
    assert glued and len(glued) < len(pairs)
    assert all(w.text.endswith((":", ".")) for w, _ in glued)
    assert {w.text for w, _ in glued} == set(T._GLUABLE[T.DUE] + T._GLUABLE[T.PROD] + T.IMPORT)
    for b in blocks:                                             # printed against each other, as written
        for line in b.lines:
            printed = T.line_text(line)
            for i, t in enumerate(line):
                if t.cls == "date" and i and line[i - 1].cls in WORDS:
                    assert (line[i - 1].text + ("" if not t.space_before else " ") + t.text) in printed
    m2_pairs = [p for b in _blocks(3000, seed=1, spec=TEXT, v2=None) for p in _word_dates(b)]
    assert m2_pairs and all(d.space_before for _, d in m2_pairs)          # M2's text never glues


# --- Line-pairs, dense panels and import dates ----------------------------------------------------

def test_a_line_pair_is_two_words_and_two_dates_on_one_line():
    for b in _blocks(2000, seed=2, spec=_only("line-pair")):
        (line,) = b.lines
        w1, d1, mark, w2, d2 = line
        assert (w1.cls in WORDS, d1.cls, mark.cls, w2.cls in WORDS, d2.cls) == (True, "date", None, True, "date")
        assert {d1.role, d2.role} in ({"mfg", "expiry"}, {"mfg", "import"}) and d1.format == d2.format
        for w, d in ((w1, d1), (w2, d2)):
            assert w.text in {"mfg": T.PROD, "expiry": T.DUE, "import": T.IMPORT}[d.role]
            assert w.cls == ("due" if d.role == "expiry" else "prod")
        between = T.line_text(line[:4])[len(T.line_text(line[:2])):-len(w2.text)]
        assert between in V2.text.separators, between
        assert mark.text == between.strip()


def test_a_dense_panel_is_a_pair_and_3_to_5_more_lines():
    for b in _blocks(2000, seed=3, spec=_only("dense-panel")):
        assert 3 <= len(b.lines) - 2 <= 5
        roles = [t.role for t in b.tokens if t.cls == "date"]
        assert sorted(roles) in (["expiry", "mfg"], ["expiry", "import", "mfg"])
        for line in b.lines:
            if any(t.cls == "date" for t in line):
                word, d = line
                assert word.text in {"mfg": T.PROD, "expiry": T.DUE, "import": T.IMPORT}[d.role]
            else:
                (t,) = line
                assert t.cls in ("code", None)
        pair = [t.role for t in b.tokens if t.role in ("mfg", "expiry")]
        assert len(pair) == 2                                    # one mfg, one expiry: the pair


def test_an_import_date_falls_1_to_6_months_after_mfg_and_before_the_expiry():
    rng = np.random.default_rng(4)
    gaps = Counter()
    for _ in range(20_000):
        mfg, expiry = T.date_pair(rng)
        imported = T.import_date(rng, mfg, expiry, V2.text)
        months = (imported.year - mfg.year) * 12 + imported.month - mfg.month
        shelf = (expiry.year - mfg.year) * 12 + expiry.month - mfg.month
        assert 1 <= months <= 6 and months < shelf and imported.day <= mfg.day
        gaps[months] += 1
    assert set(gaps) == set(range(1, 7))


def test_import_words_are_prod_and_come_only_in_gen_v2s_new_kinds():
    seen = Counter()
    for b in _blocks(5000, seed=5):
        dates = {t.role: t for t in b.tokens if t.cls == "date"}
        for w, d in _word_dates(b):
            assert (w.text in T.IMPORT) == (d.role == "import")
            if d.role == "import":
                assert w.cls == "prod" and d.format == dates["mfg"].format
                assert 1 <= _months(dates["mfg"], d) <= 6
                seen[b.kind] += 1
    assert set(seen) == {"line-pair", "dense-panel"}
    m2_tokens = [t for b in _blocks(5000, seed=5, spec=TEXT, v2=None) for t in b.tokens]
    assert not any(t.role == "import" or t.text in T.IMPORT for t in m2_tokens)


# --- The new distractors --------------------------------------------------------------------------

def test_barcode_numbers_and_unit_prices_join_the_distractors():
    rng = np.random.default_rng(6)
    drawn = [T.distractor(rng, V2.text) for _ in range(6000)]
    barcodes = [d for d in drawn if re.fullmatch(BARCODE, d)]
    prices = [d for d in drawn if d.startswith(("USP", "Unit Sale Price"))]
    for b in barcodes:                                           # EAN-13: the weighted sum, check digit too
        assert sum(int(c) * (3 if i % 2 else 1) for i, c in enumerate(b.replace(" ", ""))) % 10 == 0, b
    for p in prices:
        assert re.fullmatch(r"USP Rs\. \d\.\d\d/(g|ml)|Unit Sale Price: Rs\. \d\.\d per (g|ml)", p), p
    n = len(drawn)                                               # every template equally likely: 2 of 10 each
    assert abs(len(barcodes) / n - 0.2) < 0.03 and abs(len(prices) / n - 0.2) < 0.03
    assert abs(sum(" " in b for b in barcodes) / len(barcodes) - 0.5) < 0.05
    m2_drawn = [T.distractor(rng) for _ in range(3000)]
    assert not any(re.fullmatch(BARCODE, d) or d.startswith(("USP", "Unit")) for d in m2_drawn)


def test_the_rupee_sign_only_if_every_font_has_it():
    """The plan prints it only if every font, training and held-out, has
    the glyph; S7 found 4 families without it, so money prints Rs."""
    printed = "".join(b.text for b in _blocks(5000, seed=7))
    assert RUPEE not in printed and "Rs." in printed
    assert (RUPEE in V2.text.currency) == (RUPEE in T.CHARSET)  # so the font check would test it


@needs_fonts
def test_four_families_lack_the_rupee_sign():
    from fontTools.ttLib import TTFont
    lacking = [f.name for f in fonts.FAMILIES
               if any(ord(RUPEE) not in TTFont(fonts.path(face), lazy=True)["cmap"].getBestCmap()
                      for face in fonts.faces(f))]
    assert lacking == ["Open Sans", "Barlow Condensed", "Roboto Mono", "Share Tech Mono"]
    assert V2.text.currency == (RUPEE if not lacking else "Rs.")


def test_every_character_gen_v2_prints_is_in_the_charset():
    printed = set("".join(b.text for b in _blocks(10_000, seed=8))) - {"\n"}
    assert printed <= T.CHARSET and {"|", "&"} <= printed


def test_gen_v2s_kinds_need_its_text():
    with pytest.raises(ValueError, match="need its text"):
        T.random_block(np.random.default_rng(0), TEXT_V2)


# --- Shares -------------------------------------------------------------------------------------

def _case(token: T.Token) -> str:
    letters = "".join(c for c in token.text if c.isalpha())
    return "upper" if letters.isupper() else "lower" if letters.islower() else "title"


class _Shares:
    """Measured shares against the spec's; off() lists those more than 3 points away."""

    def __init__(self):
        self.got, self.want = {}, {}

    def __call__(self, name: str, expected: float, hits: list[bool]) -> None:
        assert hits, name
        self.got[name], self.want[name] = sum(hits) / len(hits), expected

    def off(self) -> dict:
        return {k: (round(self.got[k], 3), round(w, 3)) for k, w in self.want.items() if abs(self.got[k] - w) > 0.03}


def test_gen_v2s_shares_are_within_3_points_over_10000_blocks():
    blocks = _blocks(10_000, seed=9)
    share = _Shares()
    kinds = Counter(b.kind for b in blocks)
    for kind, p in TEXT_V2.kinds:
        share(f"kind {kind}", p, [b.kind == kind for b in blocks])
    share("glued", V2.text.glued, [not d.space_before for b in blocks for _, d in _word_dates(b)])
    panels = [b for b in blocks if b.kind in ("panel", "dense-panel")]
    extra = [line[0] for b in panels for line in b.lines if not any(t.cls == "date" for t in line)]
    share("panel codes", TEXT.panel_codes, [t.cls == "code" for t in extra])
    new = [t.text for t in extra if t.cls is None]
    share("distractors: barcode", 0.2, [bool(re.fullmatch(BARCODE, t)) for t in new])
    share("distractors: unit price", 0.2, [t.startswith(("USP", "Unit Sale")) for t in new])
    # M2's shares hold in gen-v2's text: the month-word and the other format shares unchanged.
    firsts = [next(t for t in b.tokens if t.cls == "date") for b in blocks if b.kind != "range"]
    word = [T.FAMILY_OF[t.format].startswith("word") for t in firsts]
    share("month word", TEXT.month_word, word)
    words = [t for t, w in zip(firsts, word) if w]
    numbers = [t for t, w in zip(firsts, word) if not w]
    share("full name", TEXT.full_name, [T.FAMILY_OF[t.format] == "word-full" for t in words])
    share("day, numbers", TEXT.day_numbers, [t.parts[2] is not None for t in numbers])
    share("day, words", TEXT.day_words, [t.parts[2] is not None for t in words])
    share("ordinal", TEXT.ordinal, ["DDth" in t.format for t in words if t.parts[2] is not None])
    for case, p in TEXT.case:
        share(f"case {case}", p, [_case(t) == case for t in words])
    share("single word", TEXT.single_word, [b.tokens[0].cls == "due" for b in blocks if b.kind == "single"])
    share("mfg first", TEXT.mfg_first, [_mfg_first(b) for b in blocks
                                         if b.kind in ("pair", "panel", "relative", "dense-panel")])
    assert not share.off(), share.off()
    assert sum(kinds.values()) == len(blocks) and set(kinds) == {k for k, _ in TEXT_V2.kinds}


def test_line_pair_shares_over_10000_line_pairs():
    """A new kind's own shares over 10,000 of its blocks (as M2's over
    10,000 ranges): at 10% of 10,000 blocks, 3 points is under 2 standard
    errors."""
    pairs = _blocks(10_000, seed=13, spec=_only("line-pair"))
    share = _Shares()
    share("import", V2.text.pair_import, [any(t.role == "import" for t in b.tokens) for b in pairs])
    share("mfg first", TEXT.mfg_first, [b.lines[0][1].role == "mfg" for b in pairs])
    share("glued", V2.text.glued, [not d.space_before for b in pairs for _, d in _word_dates(b)])
    marks = [b.lines[0][2].text for b in pairs]
    for sep in V2.text.separators:
        share(f"separator {sep!r}", 1 / len(V2.text.separators), [m == sep.strip() for m in marks])
    assert not share.off(), share.off()


def test_dense_panel_shares_over_10000_dense_panels():
    dense = _blocks(10_000, seed=14, spec=_only("dense-panel"))
    share = _Shares()
    share("import", V2.text.dense_import, [any(t.role == "import" for t in b.tokens) for b in dense])
    lo, hi = V2.text.dense_lines
    for n in range(lo, hi + 1):
        share(f"+{n} lines", 1 / (hi - lo + 1), [len(b.lines) - 2 == n for b in dense])
    extra = [line[0] for b in dense for line in b.lines if not any(t.cls == "date" for t in line)]
    share("codes", TEXT.panel_codes, [t.cls == "code" for t in extra])
    share("mfg first", TEXT.mfg_first, [_mfg_first(b) for b in dense])
    assert not share.off(), share.off()


def test_the_same_seed_gives_the_same_gen_v2_blocks():
    assert _blocks(300, seed=10) == _blocks(300, seed=10) != _blocks(300, seed=11)


# --- Line spacing per kind ----------------------------------------------------------------------

def test_a_dense_panel_is_spaced_closer_and_every_other_kind_as_m2():
    assert R.line_spacing("dense-panel", STYLE, V2.style) == (1.0, 1.2)
    for kind, _ in TEXT_V2.kinds:
        if kind != "dense-panel":
            assert R.line_spacing(kind, STYLE, V2.style) == STYLE.line_spacing == (1.2, 1.6), kind
    assert R.line_spacing("dense-panel", STYLE) == STYLE.line_spacing             # M2's print: as before


@needs_fonts
def test_a_dense_panels_lines_are_drawn_at_its_spacing():
    rng = np.random.default_rng(12)
    for block in _blocks(60, seed=12, spec=replace(TEXT_V2, kinds=(("dense-panel", 0.5), ("panel", 0.5)))):
        layer = R.render(block, R.STYLES[int(rng.integers(4))], fonts.random_family(rng, "train"), 10, rng,
                         v2=V2.style)
        lo, hi = R.line_spacing(block.kind, STYLE, V2.style)
        assert lo - 1e-4 <= layer.params["line_spacing"] <= hi + 1e-4, block.kind      # rounded to 4 places


# --- Boxes from the ink ---------------------------------------------------------------------------

def _own_ink(layer4: R.Layer, i: int) -> np.ndarray:
    """Token i's ink found another way: its 4x sprite alone on a canvas the
    layer's size, shrunk by a plain mean over 4 x 4, thresholded at INK."""
    k, s = layer4.scale, layer4.tokens[i]
    canvas = np.zeros((layer4.height, layer4.width), np.float32)
    if s is not None:
        h, w = s.alpha.shape
        canvas[s.y:s.y + h, s.x:s.x + w] = s.alpha
    h, w = canvas.shape
    return canvas.reshape(h // k, k, w // k, k).mean(axis=(1, 3)) >= R.INK


def _boxes_hug_their_own_ink(block: T.Block, layer4: R.Layer) -> list:
    """Every boxed token's box holds all of its own ink and touches it on
    all four sides (within 1 px); returns the boxes."""
    boxes = R.boxes(R.shrink(layer4))
    for i, (token, box) in enumerate(zip(block.tokens, boxes)):
        if token.cls is None:
            continue
        assert box is not None, f"{token.text!r} ({token.cls}) has no ink to box"
        ys, xs = np.nonzero(_own_ink(layer4, i))
        x1, y1, x2, y2 = box
        assert x1 <= xs.min() <= x1 + 1 and y1 <= ys.min() <= y1 + 1, token.text
        assert x2 - 2 <= xs.max() < x2 and y2 - 2 <= ys.max() < y2, token.text
    return boxes


@needs_fonts
@pytest.mark.parametrize("style", R.STYLES)
def test_a_glued_dates_box_holds_the_date_only(style):
    rng = np.random.default_rng(20 + R.STYLES.index(style))
    always = replace(V2.text, glued=1.0)
    checked = 0
    for block in _blocks(40, seed=20 + R.STYLES.index(style), spec=replace(TEXT, kinds=(("pair", 0.5), ("single", 0.5))),
                         v2=always):
        layer = R.render(block, style, fonts.random_family(rng, "train"), float(rng.choice([6, 11, 24])), rng,
                         v2=V2.style)
        boxes = _boxes_hug_their_own_ink(block, layer)
        index = {id(t): i for i, t in enumerate(block.tokens)}
        for word, date in _word_dates(block):
            w, d = index[id(word)], index[id(date)]
            assert not date.space_before
            ys, xs = np.nonzero(_own_ink(layer, w))              # the word's ink stays out of the date's box,
            x1, y1, x2, y2 = boxes[d]                            # but for a column they may share
            inside = (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)
            assert not inside.any() or xs[inside].max() <= x1, (word.text, date.text)
            assert boxes[w][2] <= x1 + 1 and boxes[w][0] < x1     # the word's box ends where the date's starts
            checked += 1
    assert checked >= 30                                         # about 54 expected


@needs_fonts
@pytest.mark.parametrize("style", R.STYLES)
def test_a_line_pair_gives_two_date_boxes_and_two_keyword_boxes(style):
    rng = np.random.default_rng(30 + R.STYLES.index(style))
    for block in _blocks(25, seed=30 + R.STYLES.index(style), spec=_only("line-pair")):
        layer = R.render(block, style, fonts.random_family(rng, "train"), float(rng.choice([6, 11, 24])), rng,
                         v2=V2.style)
        boxes = _boxes_hug_their_own_ink(block, layer)
        classes = Counter(t.cls for t, b in zip(block.tokens, boxes) if t.cls and b)
        assert classes["date"] == 2 and classes["due"] + classes["prod"] == 2
        w1, d1, mark, w2, d2 = boxes
        assert block.tokens[2].cls is None                      # the separator: no box
        assert w1[2] <= d1[0] + 1 and d1[2] <= w2[0] and w2[2] <= d2[0] + 1   # left to right, apart
        if mark is not None:                                    # its mark between the two pairs
            assert d1[2] <= mark[0] + 1 and mark[2] <= w2[0] + 1


@needs_fonts
@pytest.mark.parametrize("style", R.STYLES)
def test_every_gen_v2_box_holds_all_its_ink_and_touches_it(style):
    seed = 40 + R.STYLES.index(style)
    rng = np.random.default_rng(seed)
    blocks = (_blocks(15, seed, _only("dense-panel")) + _blocks(15, seed, _only("line-pair"))
              + _blocks(30, seed))                               # the new kinds for certain, then the mix
    for block in blocks:
        layer = R.render(block, style, fonts.random_family(rng, "train"), float(rng.choice([6, 9, 14, 24])), rng,
                         v2=V2.style)
        _boxes_hug_their_own_ink(block, layer)
