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

from bestbefore.synth import compose, fonts, make, render as R, text as T, warp as W
from bestbefore.synth.spec import (COMPOSE, GEN, GEOMETRY, GEOMETRY_V2, SET_SEEDS, STYLE, STYLE_V2, SYNTH_VAL, TEXT,
                                   TEXT_V2, V2, spec_dict, spec_hash, strata_hash)
from bestbefore.synth.warp import Geometry

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
    """M2's kinds in M2's styles draw exactly as M2's under gen-v2's print,
    but for dot-matrix: since S8 a gen-v2 dot-matrix block also draws its
    nozzle rows, so only the draws before its first dot are M2's."""
    rng = np.random.default_rng(R.STYLES.index(style))
    for _ in range(15):
        block, family = T.random_block(rng), fonts.random_family(rng, "train")
        a, b = np.random.default_rng(1), np.random.default_rng(1)
        m2_layer = R.render(block, style, family, 12, a)
        v2_layer = R.render(block, style, family, 12, b, v2=V2.style)
        if style == "dot-matrix":
            v2_dots = {k: v for k, v in v2_layer.params["dots"].items() if k != "nozzle"}
            assert m2_layer.params == v2_layer.params | {"dots": v2_dots}
            continue
        assert m2_layer.params == v2_layer.params and np.array_equal(R.ink(m2_layer), R.ink(v2_layer))
        assert a.bit_generator.state == b.bit_generator.state


@needs_fonts
@pytest.mark.parametrize("style", R.STYLES_V2)
def test_gen_v2s_line_spacing_changes_nothing_for_m2s_kinds(style):
    """S7's line spacing per kind, alone: M2's kinds draw the same with it
    as without it, in every style, dot-matrix and the window included."""
    rng = np.random.default_rng(10 + R.STYLES_V2.index(style))
    for _ in range(15):
        block, family = T.random_block(rng), fonts.random_family(rng, "train")
        a, b = np.random.default_rng(1), np.random.default_rng(1)
        plain = R.render(block, style, family, 12, a, v2=replace(V2.style, line_spacing=()))
        spaced = R.render(block, style, family, 12, b, v2=V2.style)
        assert plain.params == spaced.params and np.array_equal(R.ink(plain), R.ink(spaced))
        assert a.bit_generator.state == b.bit_generator.state


def _sample_files(name: str = "gen-sample", spec_sha256: str = GEN_SHA256):
    """A 500-photo sample on disk (images, sidecar lines): S5's (M2's
    gen-sample) or S8's (gen-v2-sample), or None where there is none: no
    ExpDate, fonts or sample, or one made from another spec."""
    if not fonts.fetched():
        return None
    try:
        from bestbefore.synth import batch
        home = batch.home(name)
        images = batch.images_dir(name)
        lines = home.sidecar_path(home.GENERATED_SETS[name]).read_text().splitlines()
        manifest = json.loads(home.manifest_path(name).read_text())
    except (RuntimeError, OSError):
        return None
    if len(lines) != 500 or manifest["spec_sha256"] != spec_sha256:
        return None
    return images, lines


_remade: dict = {}


def _remake_start(pool, spec=GEN, seed=SET_SEEDS["gen-sample"]) -> None:
    import cv2
    cv2.setNumThreads(1)
    _remade.update(pool=pool, spec=spec, seed=seed)


def _remake(i: int) -> tuple[int, bytes, str, str]:
    from bestbefore.synth import make
    made = make.make_one(i, _remade["spec"], _remade["pool"], _remade["seed"])
    return i, made.jpeg, made.labels, json.dumps(made.sidecar, separators=(",", ":"))


def _remade_differ(name: str, spec, images, lines) -> list:
    """Each photo of a sample on disk made again in memory (nothing is
    written), on 4 processes: the (photo, file) pairs that differ."""
    from bestbefore import derived
    from bestbefore.synth import batch, make
    pool = batch.load_pools([name])[name]
    differ = []
    with get_context("spawn").Pool(4, initializer=_remake_start, initargs=(pool, spec, SET_SEEDS[name])) as workers:
        for i, jpeg, labels, line in workers.imap(_remake, range(len(lines)), chunksize=8):
            path = images / make.image_name(i)
            for what, ok in (("jpeg", jpeg == path.read_bytes()),
                             ("labels", labels == derived.label_path(path).read_text()),
                             ("sidecar", line == lines[i])):
                if not ok:
                    differ.append((i, what))
    return differ


@pytest.mark.skipif(_sample_files() is None, reason="needs ExpDate, the fonts and S5's gen-sample on disk")
def test_s5s_500_photo_sample_is_remade_byte_for_byte():
    """Every JPEG, label file and sidecar line of S5's sample (made with
    M2's code), made again with GEN, in memory: nothing is written."""
    differ = _remade_differ("gen-sample", GEN, *_sample_files())
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


# === S8: gen-v2's print, geometry and ink ======================================================
#
# Nozzle rows, the laser window, upside-down blocks and worn patches; M2's
# geometry, ink, print and defects draw exactly what they drew (digests
# taken from M2's code before S8).

M2_GEOMETRY_DIGEST = "a4aad09d28dc0c6fcae82b59f34df84577ca4b208c119c6c5571e84dc04a795e"
M2_INK_DIGEST = "e9c01efae4c6844c44cbd1349d103234ca486cf0b5b06beee8a22dfba927cec9"
M2_PRINT_DIGEST = "6622a64db7cb1dea6dea550087f6158c9b0a9a84bee5be185ab55ab92cb4e79e"
# The pixels behind M2_PRINT_DIGEST: the laptop's builds (S8), as M2 S6 found
# that another machine's draw a few pixels differently.
PRINT_DIGEST_BUILDS = {"pillow": "12.3.0", "freetype": "2.14.3", "opencv": "5.0.0", "numpy": "2.4.6",
                       "platform": "darwin-arm64"}
V2_PRINT = replace(GEN, text=TEXT_V2, style=STYLE_V2, geometry=GEOMETRY_V2, v2=V2)   # gen-v2, M2's sizes
FLAT = Geometry("upright", 0, 0.0)


def _builds() -> dict:
    import platform

    import cv2
    import PIL
    from PIL import features
    return {"pillow": PIL.__version__, "freetype": features.version("freetype2"), "opencv": cv2.__version__,
            "numpy": np.__version__, "platform": f"{platform.system().lower()}-{platform.machine()}"}


def test_m2s_geometry_and_ink_draw_exactly_what_they_drew():
    rng = np.random.default_rng(23456)
    digest = hashlib.sha256()
    for _ in range(10_000):
        digest.update(repr(W.random_geometry(rng).record()).encode())
    digest.update(repr(rng.bit_generator.state).encode())
    assert digest.hexdigest() == M2_GEOMETRY_DIGEST
    rng = np.random.default_rng(34567)
    digest = hashlib.sha256()
    for _ in range(10_000):
        style = R.STYLES[int(rng.integers(4))]
        digest.update(repr(compose.choose_ink(rng, style, float(rng.random())).record()).encode())
    digest.update(repr(rng.bit_generator.state).encode())
    assert digest.hexdigest() == M2_INK_DIGEST


@needs_fonts
@pytest.mark.skipif(_builds() != PRINT_DIGEST_BUILDS, reason="the digest's pixels are the laptop's builds'")
def test_m2s_print_and_ink_defects_draw_exactly_what_they_drew():
    """120 of M2's blocks in M2's styles, warped, with their ink defects:
    every layer's record, boxes and pixels, and the generator's state."""
    rng = np.random.default_rng(45678)
    digest = hashlib.sha256()
    for _ in range(120):
        block = T.random_block(rng)
        style = R.random_style(rng)
        family = fonts.random_family(rng, "train")
        geometry = W.random_geometry(rng)
        layer = W.apply(R.render(block, style, family, float(rng.uniform(5, 30)), rng), geometry)
        layer, record = compose.defects(layer, style, [t.cls is not None for t in block.tokens], rng)
        small = R.shrink(layer)
        digest.update(repr((sorted(layer.params.items(), key=str), record, R.boxes(small))).encode())
        digest.update(np.ascontiguousarray(R.ink(small)).tobytes())
    digest.update(repr(rng.bit_generator.state).encode())
    assert digest.hexdigest() == M2_PRINT_DIGEST


def test_gen_v2s_print_and_geometry_need_v2():
    family = fonts.FAMILIES[0]
    block = T.Block(((T.Token("EXP", "due"), T.Token("08/2027", "date")),), "single")
    with pytest.raises(ValueError, match="needs gen-v2"):
        R.render(block, "window", family, 10, np.random.default_rng(0))
    with pytest.raises(ValueError, match="unknown style"):
        R.render(block, "stencil", family, 10, np.random.default_rng(0), v2=V2.style)
    with pytest.raises(ValueError, match="needs gen-v2"):
        compose.choose_ink(np.random.default_rng(0), "window", 0.5)
    with pytest.raises(ValueError, match="upside-down"):
        W.random_geometry(np.random.default_rng(0), GEOMETRY_V2)
    assert {name for name, _ in STYLE_V2.styles} == set(R.STYLES_V2)
    assert dict(STYLE_V2.styles)["dot-matrix"] == dict(STYLE.styles)["dot-matrix"]     # the plan keeps it


# --- Nozzle rows ---------------------------------------------------------------------------------

def test_nozzle_rows_drop_1_or_2_rows_inside_a_digit_with_their_share():
    rng = np.random.default_rng(50)
    grids = [int(r) for r in rng.choice(STYLE.dot_rows, 10_000)]
    drawn = [(rows, R.nozzle_rows(rng, V2.style, rows)) for rows in grids]
    dropped = [(rows, d) for rows, d in drawn if d]
    assert abs(len(dropped) / len(drawn) - V2.style.nozzle) < 0.03
    assert abs(np.mean([len(d) == 1 for _, d in dropped]) - 0.5) < 0.03
    for rows, d in dropped:
        assert list(d) == sorted(set(d)) and all(1 <= r <= rows - 2 for r in d)    # never a top or bottom row
    for rows in STYLE.dot_rows:                                     # every inside row comes up
        assert {r for n, d in dropped if n == rows for r in d} == set(range(1, rows - 1))


@pytest.mark.parametrize("radius, blur", [(0.40, 0.20), (0.48, 0.35)])      # crisp, and the spec's fattest
def test_dotify_puts_no_dot_on_a_skipped_row(radius, blur):
    """A covered block 7 pitches tall, dots on every cell: with rows 2 and 4
    skipped, every other dot is as before, and the skipped rows' centre
    lines hold no ink, even beside the spec's fattest, blurriest dots."""
    pitch = 8.0
    p = {"pitch_x": pitch, "pitch_y": pitch, "cover": 0.4, "drop": 0.0, "jitter": 0.0,
         "kernel": R._dot_kernel(radius * pitch, blur * pitch)}
    cov = R.Sprite(0, 0, np.ones((56, 80), np.float32))
    full = R.dotify(cov, 0.0, p, np.random.default_rng(0))
    gaps = R.dotify(cov, 0.0, p | {"skip": (2, 4)}, np.random.default_rng(0))
    assert (full.x, full.y, full.alpha.shape) == (gaps.x, gaps.y, gaps.alpha.shape)
    assert (gaps.alpha <= full.alpha + 1e-6).all()
    for r in range(7):
        line = gaps.alpha[round((r + 0.5) * pitch) - gaps.y]
        assert (line.max() < R.INK) if r in (2, 4) else (line.max() > 0.5), r


def _line_tokens(block: T.Block) -> list[list[int]]:
    """Each line's token indices, in Block.tokens order."""
    out, k = [], 0
    for line in block.lines:
        out.append(list(range(k, k + len(line))))
        k += len(line)
    return out


@needs_fonts
@pytest.mark.parametrize("rows", STYLE.dot_rows)
def test_a_dropped_dot_row_leaves_every_line_and_its_tokens_box_holds_the_rest(rows):
    """The same dot-matrix block with nozzle rows and without (one seed): the
    same dots but the dropped rows', no ink where those were (crisp dots,
    still, so neighbours don't blur into the gap), a loss in every line,
    and every box still on its token's remaining ink."""
    crisp = replace(STYLE, dot_rows=(rows,), dot_radius=(0.32, 0.32), dot_blur=(0.15, 0.15), dot_jitter=0.0)
    dropping, keeping = replace(V2.style, nozzle=1.0), replace(V2.style, nozzle=0.0)
    rng = np.random.default_rng(60 + rows)
    blocks = _blocks(30, seed=60 + rows, spec=replace(TEXT_V2, kinds=(("panel", 0.5), ("dense-panel", 0.5))))
    for block in blocks:
        family, seed = fonts.random_family(rng, "train"), int(rng.integers(1 << 30))
        size = float(rng.choice([10, 16, 24]))
        whole = R.render(block, "dot-matrix", family, size, np.random.default_rng(seed), crisp, keeping)
        gappy = R.render(block, "dot-matrix", family, size, np.random.default_rng(seed), crisp, dropping)
        assert gappy.params["dots"]["nozzle"] and not whole.params["dots"]["nozzle"]
        assert whole.params | {"dots": None} == gappy.params | {"dots": None}
        assert (whole.width, whole.height) == (gappy.width, gappy.height)     # dropping rows moves nothing
        a, b = R.ink(whole), R.ink(gappy)
        assert (b <= a + 1e-5).all()
        removed = a - b
        assert (b[removed >= 0.5] < R.INK).all()
        pitch = whole.params["line_spacing"] * whole.params["px"]   # rounded in the record: lines to 1 px
        starts, shapes = [], set()
        for n, line in enumerate(_line_tokens(block)):
            sprites = [whole.tokens[i] for i in line if whole.tokens[i] is not None]
            x0, y0 = min(s.x for s in sprites), min(s.y for s in sprites)
            x1 = max(s.x + s.alpha.shape[1] for s in sprites)
            y1 = max(s.y + s.alpha.shape[0] for s in sprites)
            assert removed[y0:y1, x0:x1].max() >= 0.5                # every line lost its rows' dots
            heights = set()                                          # the same rows in every token and line:
            for i in line:                                           # the heights of the dots it lost
                s, t = whole.tokens[i], gappy.tokens[i]
                if s is not None and t is not None:
                    ys = np.flatnonzero(((s.alpha - t.alpha) >= 0.5).any(axis=1))
                    if len(ys):
                        heights.add(tuple(ys + s.y))
            assert len(heights) == 1, heights                        # within a line, exactly
            (h,) = heights
            starts.append(h[0] - round(n * pitch))
            shapes.add(tuple(np.array(h) - h[0]))
        assert len(shapes) == 1 and max(starts) - min(starts) <= 1, (shapes, starts)
        _boxes_hug_their_own_ink(block, gappy)


@needs_fonts
def test_with_the_specs_own_dots_nozzle_rows_only_take_ink_away():
    rng = np.random.default_rng(65)
    for block in _blocks(40, seed=65):
        family, seed, size = fonts.random_family(rng, "train"), int(rng.integers(1 << 30)), float(rng.uniform(6, 24))
        whole = R.render(block, "dot-matrix", family, size, np.random.default_rng(seed), STYLE,
                         replace(V2.style, nozzle=0.0))
        gappy = R.render(block, "dot-matrix", family, size, np.random.default_rng(seed), STYLE,
                         replace(V2.style, nozzle=1.0))
        assert (whole.width, whole.height) == (gappy.width, gappy.height)
        assert (R.ink(gappy) <= R.ink(whole) + 1e-5).all()
        assert R.ink(gappy).sum() < R.ink(whole).sum()
        _boxes_hug_their_own_ink(block, gappy)


# --- The laser window ---------------------------------------------------------------------------

def test_window_looks_follow_their_shares_and_ranges():
    rng = np.random.default_rng(70)
    drawn = [R.window_params(rng, V2.style) for _ in range(10_000)]
    assert abs(np.mean([w["dots"] for w in drawn]) - V2.style.window_dots) < 0.03
    colours = Counter(tuple(w["colour"]) for w in drawn)
    assert set(colours) == set(V2.style.window_colours)
    assert all(abs(n / len(drawn) - 0.25) < 0.03 for n in colours.values())
    for name in ("pad", "corner", "soft", "glow", "glow_spread"):
        lo, hi = getattr(V2.style, f"window_{name}")
        assert all(lo <= w[name] <= hi for w in drawn), name
    assert V2.style.window_glow[1] < R.INK
    assert all(compose.luminance(c) < 0.25 for c in V2.style.window_colours)       # a dark printed rectangle


def test_a_glow_alone_is_never_ink():
    dot = R.Sprite(40, 40, np.pad(np.ones((6, 6), np.float32), 2))
    lit = R.laser(dot, soft=0.0, glow=V2.style.window_glow[1], spread=12.0)
    outside = np.ones_like(lit.alpha, bool)
    outside[dot.y - lit.y:dot.y - lit.y + 10, dot.x - lit.x:dot.x - lit.x + 10] = False
    assert 0 < lit.alpha[outside].max() <= V2.style.window_glow[1] < R.INK
    assert lit.alpha[dot.y - lit.y + 5, dot.x - lit.x + 5] == pytest.approx(1.0)


@needs_fonts
@pytest.mark.parametrize("dots", [False, True])
def test_a_windows_text_is_lighter_than_its_rectangle(dots):
    style = replace(V2.style, window_dots=float(dots))
    rng = np.random.default_rng(71 + dots)
    for block in _blocks(20, seed=71 + dots):
        layer = R.shrink(R.render(block, "window", fonts.random_family(rng, "train"), float(rng.choice([12, 16, 24])),
                                  rng, v2=style))
        assert layer.params["window"]["dots"] is dots and ("dots" in layer.params) is dots
        ink = R.ink(layer)
        rectangle = compose._canvas(layer.backing, layer.height, layer.width)
        strokes, bare = ink >= 0.8 * ink.max(), (rectangle >= 0.99) & (ink < 0.01)
        assert ink.max() > 0.25 and bare.any()                       # M2's faintest dots peak near 0.3
        for grey in (40.0, 220.0):                                   # on a dark pack and on a light one
            photo = np.full((layer.height, layer.width, 3), grey, np.float32)
            compose.blend(photo, layer, 0, 0, compose.choose_ink(rng, "window", grey / 255, v2=V2.compose),
                          compose.Surface(photo.copy()))
            lum = photo @ compose.LUMA / 255
            # light strokes on a dark window: at least 0.8 x 0.75 x (0.75 - 0.25) of their alpha lighter
            assert lum[strokes].min() > lum[bare].max() + 0.25 * ink.max()


@needs_fonts
@pytest.mark.parametrize("dots", [False, True])
def test_every_window_box_holds_all_its_ink_and_touches_it(dots):
    style = replace(V2.style, window_dots=float(dots))
    rng = np.random.default_rng(73 + dots)
    blocks = _blocks(15, 73, _only("dense-panel")) + _blocks(15, 74, _only("line-pair")) + _blocks(20, 75)
    for block in blocks:
        layer = R.render(block, "window", fonts.random_family(rng, "train"), float(rng.choice([6, 9, 14, 24])), rng,
                         v2=style)
        _boxes_hug_their_own_ink(block, layer)


@needs_fonts
def test_white_strokes_on_a_white_pack_are_measured_on_their_window():
    """The floor on faint ink measures a window's strokes against its dark
    rectangle, what they are seen on: white strokes on a white pack change
    its pixels less than they stand out from their window."""
    white = replace(V2.compose, window_ink=(1.0, 1.0))
    solid = replace(COMPOSE, opacity=(1.0, 1.0))
    block = T.Block(((T.Token("EXP", "due"), T.Token("08/2027", "date")),), "single")
    layer = R.shrink(R.render(block, "window", fonts.families("train")[0], 16, np.random.default_rng(0),
                              v2=replace(V2.style, window_dots=0.0)))
    photo = np.full((layer.height + 20, layer.width + 20, 3), 252.0, np.float32)
    before = photo.copy()
    inks, shown = make.ink_block(photo, layer, block, "window", 10, 10, compose.Surface(before),
                                 np.random.default_rng(1), solid, white)
    assert len(inks) == 1 and shown >= COMPOSE.visible
    region = (slice(10, 10 + layer.height), slice(10, 10 + layer.width))
    against = compose.backdrop(before, layer, 10, 10, compose.Surface(before))
    assert shown == pytest.approx(min(compose.ink_change(against, photo[region], s) for s in layer.tokens))
    assert shown > min(compose.ink_change(before[region], photo[region], s) for s in layer.tokens) + 0.1


# --- Upside-down blocks --------------------------------------------------------------------------

def test_upside_down_blocks_take_their_share_turned_a_half_turn():
    rng = np.random.default_rng(80)
    drawn = [W.random_geometry(rng, GEOMETRY_V2, V2.geometry) for _ in range(10_000)]
    for name, share in GEOMETRY_V2.orientation:
        assert abs(np.mean([g.orientation == name for g in drawn]) - share) < 0.03, name
    upside = [g for g in drawn if g.orientation == "upside-down"]
    assert {g.quarter for g in upside} == {2} and all(abs(g.angle) <= V2.geometry.upside_down_angle for g in upside)
    assert max(abs(g.angle) for g in upside) > 0.9 * V2.geometry.upside_down_angle
    assert {g.quarter for g in drawn if g.orientation == "sideways"} == {1, 3}
    assert GEOMETRY_V2.surface == GEOMETRY.surface and GEOMETRY_V2.perspective == GEOMETRY.perspective


def _half_turned(box, size):
    (x1, y1, x2, y2), (w, h) = box, size
    return (w - x2, h - y2, w - x1, h - y1)


@needs_fonts
@pytest.mark.parametrize("style", R.STYLES_V2)
def test_a_half_turn_maps_every_box_exactly(style):
    rng = np.random.default_rng(81 + R.STYLES_V2.index(style))
    for block in _blocks(8, seed=81):
        layer = R.render(block, style, fonts.random_family(rng, "train"), float(rng.choice([7, 16])), rng,
                         v2=V2.style)
        small, turned = R.shrink(layer), R.shrink(W.turn(layer, 2))
        assert R.boxes(turned) == tuple(None if b is None else _half_turned(b, (small.width, small.height))
                                        for b in R.boxes(small))
        assert np.array_equal(R.ink(W.turn(layer, 2)), np.rot90(R.ink(layer), 2))
        # Through the whole warp, upside down at 0 degrees is upright at 0 degrees turned a half turn.
        up = R.shrink(W.apply(layer, FLAT))
        down = R.shrink(W.apply(layer, Geometry("upside-down", 2, 0.0)))
        assert R.boxes(down) == tuple(None if b is None else _half_turned(b, (up.width, up.height))
                                      for b in R.boxes(up))


# --- Worn patches --------------------------------------------------------------------------------

def _dated_layer() -> tuple[T.Block, R.Layer]:
    """A made-up block drawn at 4x, no font needed: a word and a date on
    each of two lines, each token a filled bar."""
    block = T.Block(((T.Token("EXP", "due"), T.Token("08/2027", "date")),
                     (T.Token("MFG", "prod"), T.Token("08/2025", "date"))), "pair")
    rects = [(8, 8, 60, 48), (80, 8, 260, 48), (8, 72, 60, 112), (80, 72, 260, 112)]
    sprites = tuple(R.aligned(R.Sprite(x1, y1, np.ones((y2 - y1, x2 - x1), np.float32)), 4) for x1, y1, x2, y2 in rects)
    return block, R.Layer(272, 120, 4, sprites, None, {"size": 10.0, "px": 56, "line_spacing": 1.4})


def test_worn_patches_take_their_share_on_solid_dot_matrix_and_window_print():
    block, layer = _dated_layer()
    rng = np.random.default_rng(90)
    for style in ("label", "embossed"):
        assert compose.random_wear(rng, layer, block, style, V2.compose) is None
    for style in V2.compose.worn_styles:
        drawn = [compose.random_wear(rng, layer, block, style, V2.compose) for _ in range(10_000)]
        assert abs(np.mean([w.on for w in drawn]) - V2.compose.worn) < 0.03, style
        assert {w.token for w in drawn} == {1, 3}                         # on a date, either one
        char, line = 10.0 * 4, 1.4 * 56
        for w in drawn:
            x1, _, x2, _ = R.ink_box(layer.tokens[w.token])
            assert x1 <= w.cx <= x2 and w.cx == pytest.approx(x1 + w.along * (x2 - x1))
            assert w.width == pytest.approx(w.chars * char) and w.height == pytest.approx(w.lines * line)
            assert 1.0 <= w.chars <= 3.0 and 1.0 <= w.lines <= 2.0 and 0.6 <= w.fade <= 0.95


def test_a_patch_fades_by_its_fade_over_its_core_and_nothing_beyond_its_edge():
    block, layer = _dated_layer()
    patch = compose.Wear(True, 1, 0.5, 2.0, 1.2, 0.8, 0.25, cx=170.0, cy=28.0, width=80.0, height=100.0)
    worn = compose.wear(layer, patch)
    for s, t in zip(layer.tokens, worn.tokens):
        assert (t.x, t.y, t.alpha.shape) == (s.x, s.y, s.alpha.shape)
    date = worn.tokens[1]
    ys, xs = np.mgrid[:date.alpha.shape[0], :date.alpha.shape[1]] + 0.5
    ys, xs = ys + date.y, xs + date.x
    core = (np.abs(xs - 170) <= 40 * 0.75) & (np.abs(ys - 28) <= 50 * 0.75)
    beyond = (np.abs(xs - 170) >= 40) | (np.abs(ys - 28) >= 50)
    ink = layer.tokens[1].alpha > 0
    assert np.allclose(date.alpha[core & ink], 0.2)
    assert np.array_equal(date.alpha[beyond], layer.tokens[1].alpha[beyond])
    assert np.array_equal(worn.tokens[0].alpha, layer.tokens[0].alpha)      # the word is clear of it


def _worn(share: float, **more):
    return replace(V2_PRINT, v2=replace(V2, compose=replace(V2.compose, worn=share, **more)))


@needs_fonts
@pytest.mark.parametrize("style", V2.compose.worn_styles)
def test_a_worn_blocks_boxes_are_its_unworn_twins_and_its_patch_holds_60_to_95_percent_less_ink(style):
    rng = np.random.default_rng(91 + V2.compose.worn_styles.index(style))
    for i, block in enumerate(_blocks(24, seed=91)):
        family, seed = fonts.random_family(rng, "train"), int(rng.integers(1 << 60))
        size = float(rng.choice([12, 18, 26]))
        geometry = FLAT if i % 2 == 0 else W.random_geometry(rng, GEOMETRY_V2, V2.geometry)
        clean = make.draw_v2(block, style, family, size, geometry, seed, _worn(0.0))
        worn = make.draw_v2(block, style, family, size, geometry, seed, _worn(1.0))
        assert worn.boxes == clean.boxes                                   # the ink before the wear
        assert clean.defects["worn"] is None and worn.defects["worn"] is not None
        assert {k: v for k, v in clean.defects.items() if k != "worn"} == \
            {k: v for k, v in worn.defects.items() if k != "worn"}
        a, b = R.ink(clean.layer), R.ink(worn.layer)
        assert a.shape == b.shape and (b <= a + 1e-5).all() and b.sum() < a.sum()
        if style == "window":                                              # its dark rectangle isn't ink
            assert np.array_equal(clean.layer.backing.alpha, worn.layer.backing.alpha)
        if geometry is FLAT:                                               # its core, as drawn
            inked = a > 0.05
            least = float((b[inked] / a[inked]).min())
            fade = worn.defects["worn"]["fade"]
            assert least == pytest.approx(1 - fade, abs=0.01) and 0.05 - 0.01 <= least <= 0.40 + 0.01


@needs_fonts
def test_a_date_worn_to_nothing_gets_its_block_drawn_again():
    """A patch over the whole block, fading it by 99%: every draw wears a
    boxed token to nothing, so the block is drawn again and again, and
    then skipped rather than boxed round nothing."""
    gone = _worn(1.0, worn_width=(80.0, 80.0), worn_height=(12.0, 12.0), worn_fade=(0.99, 0.99))
    block = T.Block(((T.Token("EXP", "due"), T.Token("08/2027", "date")),), "single")
    family = fonts.families("train")[0]
    d = make.draw_v2(block, "solid", family, 14, FLAT, 5, gone)
    assert d.worn_out and all(b is not None for b in d.boxes)            # its boxes from before the wear
    assert make._draw(block, "solid", family, 14, 1.0, FLAT, np.random.default_rng(6), gone) is None
    kept = make._draw(block, "solid", family, 14, 1.0, FLAT, np.random.default_rng(6), _worn(1.0))
    assert kept is not None and kept.redraws == 0 and kept.defects["worn"] is not None


# --- Whole gen-v2 photos (M2's sizes until S8's second part) -----------------------------------------

@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    """Three made-up backgrounds, as test_synth_compose.py's: 1,200 and 1,800
    px (both fit small print; the second is shrunk to 1,600) and 700 px."""
    from PIL import Image

    from bestbefore.samples import Box
    from bestbefore.synth.compose import Background, Backgrounds
    root = tmp_path_factory.mktemp("backgrounds")
    made = []
    for name, (w, h), boxes, seed in (
            ("train_real", (1200, 900), (Box("date", (0.1, 0.1, 0.3, 0.14)),), 1),
            ("train_synth_clean", (1300, 1800), (Box("code", (0.4, 0.4, 0.6, 0.43)),), 2),
            ("train_synth_clean", (700, 500), (), 3)):
        rng = np.random.default_rng(seed)
        coarse = rng.uniform(40, 230, (6, 8, 3)).astype(np.uint8)
        im = np.asarray(Image.fromarray(coarse).resize((w, h), Image.Resampling.BICUBIC), np.float32)
        path = root / f"{seed}.jpg"
        Image.fromarray(np.clip(im + rng.normal(0, 6, im.shape), 0, 255).astype(np.uint8)).save(path, quality=92)
        made.append(Background(name, path, w, h, boxes))
    return Backgrounds({"train_real": (made[0],), "train_synth_clean": tuple(made[1:])})


@needs_fonts
def test_gen_v2_photos_box_every_date_inside_and_come_out_the_same(pool):
    seen = Counter()
    for i in range(40):
        made = make.make_one(i, V2_PRINT, pool, 7)
        assert make.make_one(i, V2_PRINT, pool, 7).jpeg == made.jpeg
        s = made.sidecar
        w, h = s["size"]
        for b in s["blocks"]:
            x, y, bw, bh = b["place"]
            assert b["lines"] >= 1
            for t in b["tokens"]:
                assert t["cls"] is None or t["box"] is not None, t
                if t["box"]:
                    x1, y1, x2, y2 = t["box"]
                    assert x <= x1 < x2 <= x + bw and y <= y1 < y2 <= y + bh
                if t["cls"] == "date":
                    assert isinstance(t["space_before"], bool)
            seen[b["style"]] += 1
            seen[b["geometry"]["orientation"]] += 1
            seen["worn"] += b["defects"].get("worn") is not None
        assert made.labels == make.label_text(s)
    assert {"window", "upside-down"} <= set(seen) and seen["worn"]
    m2 = make.make_one(3, GEN, pool, 7).sidecar                     # M2's sidecar keeps its shape
    assert all("lines" not in b and all("space_before" not in t for t in b["tokens"]) for b in m2["blocks"])


def _placed(kind: str, style: str, tokens: list[tuple], lines: int = 1, **record) -> dict:
    """A made-up placed block as the sidecar keeps it: tokens (text, cls)
    or (text, "date", role, space_before)."""
    out = {"kind": kind, "style": style, "lines": lines, "render": {}, "defects": {"worn": record.get("worn")},
           "tokens": [{"text": t[0], "cls": t[1]} | ({"role": t[2], "space_before": t[3]} if t[1] == "date" else {})
                      for t in tokens]}
    if style == "dot-matrix":
        out["render"]["dots"] = {"nozzle": record.get("nozzle", [])}
    if style == "window":
        out["render"]["window"] = {"dots": record.get("dots", False)}
    return out


def test_gen_v2s_share_rows_count_what_they_should():
    placed = [
        _placed("line-pair", "solid", [("MFG:", "prod"), ("02/2025", "date", "mfg", False), ("|", None),
                                       ("EXP", "due"), ("02/2027", "date", "expiry", True)], worn={"fade": 0.8}),
        _placed("dense-panel", "dot-matrix", [("MFG", "prod"), ("01/25", "date", "mfg", True), ("L 12345", "code"),
                                              ("8901234567890", None), ("IMPORT:", "prod"),
                                              ("03/25", "date", "import", True), ("EXP:", "due"),
                                              ("01/27", "date", "expiry", False)], lines=5, nozzle=[3]),
        _placed("panel", "window", [("EXP", "due"), ("05/2026", "date", "expiry", True),
                                    ("USP Rs. 1.50/g", None)], lines=2, dots=True),
        _placed("single", "dot-matrix", [("05/2026", "date", "expiry", True)]),
        _placed("pair", "label", [("MFG", "prod"), ("05/2024", "date", "mfg", True), ("EXP", "due"),
                                  ("05/2026", "date", "expiry", True)], lines=2, worn={"fade": 0.7}),
        _placed("range", "solid", [("BB", "due"), ("05/2024", "date", "mfg", True), ("-", None),
                                   ("05/2026", "date", "expiry", False)]),
    ]
    rows = {}
    make._v2_shares(lambda what, spec, hits: rows.__setitem__(what, (spec, list(hits))), placed, V2_PRINT)
    # glued: every date after a word, its space; a range's second date follows its dash, so isn't counted
    assert rows["glued: of words printed before a date"] == (0.3, [True, False, False, False, True, False,
                                                                   False, False, False])
    assert rows["line-pairs: an import date"][1] == [False]
    assert [rows[f"line-pairs: separator {sep!r}"][1] for sep in V2.text.separators] == [[True], [False], [False],
                                                                                         [False]]
    assert rows["dense panels: an import line"][1] == [True]
    assert [rows[f"dense panels: {n} more lines"][1] for n in (3, 4, 5)] == [[True], [False], [False]]
    assert rows["panels' other lines: codes"][1] == [True, False, False]
    assert rows["distractors: barcode numbers"] == (0.2, [True, False])
    assert rows["distractors: unit prices"] == (0.2, [False, True])
    assert rows["dot-matrix: nozzle rows"] == (0.15, [True, False])
    assert rows["windows: dotified"] == (0.3, [True])
    assert rows["worn patch (solid, dot-matrix, window)"] == (0.1, [True, False, False, False, False])  # no label


@needs_fonts
def test_gen_v2s_shares_are_read_from_its_sidecars(pool):
    rows = {r["what"]: r for r in make.shares([make.make_one(i, V2_PRINT, pool, 8).sidecar for i in range(30)],
                                              V2_PRINT)}
    for what in ("style: window", "orientation: upside-down", "glued: of words printed before a date",
                 "line-pairs: an import date", "dense panels: an import line", "distractors: barcode numbers",
                 "dot-matrix: nozzle rows", "windows: dotified", "worn patch (solid, dot-matrix, window)"):
        assert what in rows and rows[what]["n"] > 0, what
    assert rows["kind: line-pair"]["spec"] == 0.10 and rows["style: window"]["spec"] == 0.10
    m2_rows = make.shares([make.make_one(i, GEN, pool, 8).sidecar for i in range(5)], GEN)
    assert not any(r["what"].startswith(("glued", "dot-matrix: nozzle", "worn")) for r in m2_rows)


# === S8, part 2: sizes like the store's =======================================================
#
# Each block's first date's box follows v1-train's date sizes: a target
# drawn from their deciles, hit by drawing the block, measuring the box and
# drawing it again from the same draws (corrected once more while off, then
# from fresh draws: the owner's call, 2026-11-05). A tilted block is sized
# by its digits instead: v1-train has no tilted date (2026-11-06).

DECILES = (0.00659, 0.01118, 0.01296, 0.01451, 0.01573, 0.01736, 0.01974, 0.02181, 0.02559, 0.03247, 0.045)


def _within(box: float, target: float) -> bool:
    """The plan's tolerance, written out here rather than taken from make.fits."""
    return abs(box - target) <= max(1.0, 0.05 * target)


def test_gen_v2_is_m2s_spec_with_every_gap_filled_and_the_stores_sizes():
    from bestbefore.synth.spec import GEN_V2, V2Size
    assert GEN_V2 == replace(GEN, text=TEXT_V2, style=STYLE_V2, geometry=GEOMETRY_V2, v2=V2)
    assert (GEN_V2.size, GEN_V2.compose, GEN_V2.camera, GEN_V2.fonts, GEN_V2.backgrounds) == \
        (GEN.size, GEN.compose, GEN.camera, GEN.fonts, GEN.backgrounds)          # the rest is M2's
    assert V2Size().deciles == DECILES and len(DECILES) == 11 and list(DECILES) == sorted(DECILES)
    assert DECILES[0] >= 0.006 and DECILES[-1] == 0.045                        # M2's floor and ceiling
    assert spec_dict(GEN_V2)["v2"]["size"]["deciles"] == DECILES
    assert spec_hash(GEN_V2) != GEN_SHA256 and spec_hash(GEN) == GEN_SHA256
    assert SET_SEEDS["gen-v2-sample"] == 6 and len(set(SET_SEEDS.values())) == len(SET_SEEDS)


def test_targets_follow_v1_trains_deciles():
    from bestbefore.synth.spec import V2Size
    rng = np.random.default_rng(100)
    drawn = [compose.store_size(rng, V2Size()) for _ in range(10_000)]
    for k in range(10):
        inside = [t for d, t in drawn if d == k]
        assert abs(len(inside) / len(drawn) - 0.1) < 0.03, k
        lo, hi = DECILES[k], DECILES[k + 1]
        assert all(lo <= t <= hi for t in inside)
        assert abs(np.median(np.log(inside)) - (np.log(lo) + np.log(hi)) / 2) < 0.1 * (np.log(hi) - np.log(lo))
    assert abs(np.median([t for _, t in drawn]) - DECILES[5]) < 0.0005         # the store's median, 1.736%


def _one_way(orientation: str):
    return replace(GEOMETRY_V2, orientation=((orientation, 1.0),))


def _drawn_to_size(rng, style: str, geometry_spec, spec) -> tuple[str, float]:
    """A random gen-v2 block drawn to a target as gen-v2 draws one, on a
    photo of 1,000 to 1,600 px: "fits" or "off", or for a tilted block
    "digits", which are its target over 1.3 and no drawing more; "lost" if
    every try lost its ink, when make_one skips it. And its drawings."""
    block = T.random_block(rng, TEXT_V2, V2.text)
    geometry = W.random_geometry(rng, geometry_spec, V2.geometry)
    target = compose.store_size(rng, V2.size)[1] * float(rng.choice([1000, 1280, 1600]))
    drawn = make._sized(block, style, fonts.random_family(rng, "train"), target, 1.0, geometry, rng, spec)
    if drawn is None:
        return "lost", 0
    first = next(i for i, t in enumerate(block.tokens) if t.cls == "date")
    x1, y1, x2, y2 = drawn.boxes[first]
    assert drawn.sizing["reached"] == min(x2 - x1, y2 - y1)                   # its first date's box, as drawn
    if geometry.orientation == "tilted":
        assert drawn.sizing["fits"] is None and drawn.sizing["digits"] == pytest.approx(target / 1.3)
        assert drawn.layer.params["size"] == pytest.approx(target / 1.3, abs=0.001)
        return "digits", drawn.sizing["drawings"]
    assert drawn.sizing["fits"] == _within(drawn.sizing["reached"], target)
    return ("fits" if drawn.sizing["fits"] else "off"), drawn.sizing["drawings"]


@needs_fonts
def test_a_block_drawn_to_a_target_lands_within_1px_or_5_percent():
    """At least 99% of blocks drawn as gen-v2 draws them land within 1 px or
    5% of their target (the owner's call, 2026-11-05: a box of dots or relief
    grows in steps, so a few fit no size and keep their closest drawing),
    tilted blocks aside, which aren't drawn to a box. Every style and
    orientation, 20 blocks each, at least 90%: a guard, as 20 blocks can't
    tell 95% from 90%; S8's record measures each on 100 or more."""
    from bestbefore.synth.spec import GEN_V2
    rng = np.random.default_rng(101)
    drawn = [_drawn_to_size(rng, R.random_style(rng, GEN_V2.style), GEOMETRY_V2, GEN_V2) for _ in range(500)]
    outcome = Counter(f for f, _ in drawn)
    fits = [f == "fits" for f, _ in drawn if f in ("fits", "off")]
    assert outcome["lost"] <= 0.02 * len(drawn) and np.mean(fits) >= 0.99, outcome
    assert abs(outcome["digits"] / len(drawn) - 0.15) < 0.05, outcome          # tilted blocks: 15% of them
    assert np.mean([n for f, n in drawn if f != "lost"]) < 2.3
    assert {n for f, n in drawn if f == "digits"} <= {1, 2, 3}                 # one drawing a try
    cells = {}
    for style in R.STYLES_V2:
        for orientation, _ in GEOMETRY_V2.orientation:
            found = [_drawn_to_size(rng, style, _one_way(orientation), GEN_V2)[0] for _ in range(20)]
            if orientation == "tilted":
                assert set(found) <= {"digits", "lost"} and found.count("digits") >= 15, (style, found)
            else:
                cells[style, orientation] = np.mean([f == "fits" for f in found if f != "lost"])
    assert min(cells.values()) >= 0.90, {k: v for k, v in cells.items() if v < 1}


def _boxes_of(shorts: list[int], calls: list):
    """A stand-in for make.draw_v2: each call's first date box has the next
    short side of `shorts` (no ink once they run out); calls are logged as
    (digits, seed)."""
    def draw(block, style, family, size, geometry, seed, spec):
        calls.append((size, seed))
        short = shorts[len(calls) - 1] if len(calls) <= len(shorts) else None
        return make.Drawing(None, (None if short is None else (0, 0, 3 * short, short), None), {})
    return draw


@pytest.mark.parametrize("shorts, kept, drawings, fits", [
    ([30, 10], 10, 2, True),                    # the first correction fits: done
    ([30, 15, 10], 10, 3, True),                # off, so corrected once more, from the same draws
    ([30, 12, 14], 12, 3, False),               # still off: the closer drawing is kept
])
def test_a_try_corrects_once_more_while_off_and_keeps_its_closest_drawing(monkeypatch, shorts, kept, drawings, fits):
    from bestbefore.synth.spec import GEN_V2
    calls = []
    monkeypatch.setattr(make, "draw_v2", _boxes_of(shorts, calls))
    block = T.Block(((T.Token("08/2027", "date"), T.Token("EXP", None)),), "single")
    d, sizing = make.draw_to_size(block, "solid", None, 10.0, FLAT, 77, GEN_V2)
    assert (sizing["reached"], sizing["drawings"], sizing["fits"]) == (kept, drawings, fits)
    assert d.boxes[0][3] == kept and {seed for _, seed in calls} == {77}
    digits = [size for size, _ in calls]                                  # each at the size its box asks for
    assert digits[0] == pytest.approx(10 / V2.size.provisional)
    for k in range(1, len(calls)):
        assert digits[k] == pytest.approx(digits[k - 1] * 10 / shorts[k - 1])


def test_a_block_still_off_is_drawn_from_fresh_draws_and_keeps_its_closest(monkeypatch):
    from bestbefore.synth.spec import GEN_V2
    calls = []
    monkeypatch.setattr(make, "draw_v2", _boxes_of([30, 14, 13, 30, 12, 13, 30, 16, 15], calls))
    block = T.Block(((T.Token("08/2027", "date"),),), "single")
    drawn = make._sized(block, "solid", None, 10.0, 1.0, FLAT, np.random.default_rng(0), GEN_V2)
    seeds = [seed for _, seed in calls]
    assert len(calls) == 9 and len(set(seeds)) == 3 and seeds[0] == seeds[2] != seeds[3]   # 3 tries, fresh each
    assert (drawn.sizing["reached"], drawn.sizing["fits"], drawn.redraws) == (12, False, 1)   # the closest: try 2
    assert (drawn.sizing["drawings"], drawn.sizing["tries"]) == (9, 3)


TILTED = Geometry("tilted", 0, 30.0)


def test_a_tilted_block_keeps_its_first_drawing_whatever_its_box(monkeypatch):
    """Its digits are its target over 1.3, as an upright block's; a box
    six times its target is its tilt, and draws no correction."""
    from bestbefore.synth.spec import GEN_V2
    calls = []
    monkeypatch.setattr(make, "draw_v2", _boxes_of([60], calls))
    block = T.Block(((T.Token("08/2027", "date"),),), "single")
    drawn = make._sized(block, "solid", None, 10.0, 1.0, TILTED, np.random.default_rng(0), GEN_V2)
    assert calls == [(pytest.approx(10 / 1.3), calls[0][1])]                  # one drawing, one try
    assert drawn.sizing == {"digits": pytest.approx(10 / 1.3), "reached": 60, "fits": None, "drawings": 1, "tries": 1}
    assert drawn.redraws == 0
    for orientation in ("upright", "sideways", "upside-down"):                # the others are corrected, as before
        calls.clear()
        monkeypatch.setattr(make, "draw_v2", _boxes_of([60, 10], calls))
        kept, sizing = make.draw_to_size(block, "solid", None, 10.0, Geometry(orientation, 0, 0.0), 1, GEN_V2)
        assert len(calls) == 2 and (sizing["reached"], sizing["fits"]) == (10, True)


def test_a_tilted_block_that_lost_its_ink_is_drawn_from_fresh_draws(monkeypatch):
    from bestbefore.synth.spec import GEN_V2
    block = T.Block(((T.Token("08/2027", "date"),),), "single")
    calls = []

    def lost_then_kept(blk, style, family, size, geometry, seed, spec):
        calls.append(seed)
        return make.Drawing(None, (None if len(calls) == 1 else (0, 0, 90, 30),), {})
    monkeypatch.setattr(make, "draw_v2", lost_then_kept)
    drawn = make._sized(block, "solid", None, 10.0, 1.0, TILTED, np.random.default_rng(0), GEN_V2)
    assert len(calls) == 2 and calls[0] != calls[1] and drawn.redraws == 1 and drawn.sizing["drawings"] == 2
    calls.clear()
    monkeypatch.setattr(make, "draw_v2", lambda *a: calls.append(a) or make.Drawing(None, (None,), {}))
    assert make._sized(block, "solid", None, 10.0, 1.0, TILTED, np.random.default_rng(0), GEN_V2) is None
    assert len(calls) == GEN_V2.compose.redraws                                # 3 tries, then skipped, as M2
    worn = make.Drawing(None, ((0, 0, 90, 30),), {}, worn_out=True)            # worn to nothing: no better
    monkeypatch.setattr(make, "draw_v2", lambda *a: worn)
    assert make.draw_to_size(block, "solid", None, 10.0, TILTED, 1, GEN_V2)[0] is None


@needs_fonts
@pytest.mark.parametrize("style", R.STYLES_V2)
def test_the_corrections_make_the_first_drawings_choices(style):
    """Drawn again at another size from the same seed, a block keeps its
    font, weight, spacing, dots, window, relief, label and ink defects: only
    its size changes. The ink is chosen once, after the last drawing."""
    from bestbefore.synth.spec import GEN_V2
    rng = np.random.default_rng(102 + R.STYLES_V2.index(style))
    for block in _blocks(12, seed=102):
        family, seed = fonts.random_family(rng, "train"), int(rng.integers(1 << 62))
        geometry = W.random_geometry(rng, GEOMETRY_V2, V2.geometry)
        a = make.draw_v2(block, style, family, float(rng.uniform(5, 12)), geometry, seed, _worn(0.5))
        b = make.draw_v2(block, style, family, float(rng.uniform(14, 30)), geometry, seed, _worn(0.5))
        drop = ("size", "px")
        assert {k: v for k, v in a.layer.params.items() if k not in drop} == \
            {k: v for k, v in b.layer.params.items() if k not in drop}
        assert a.defects == b.defects
        target = compose.store_size(rng, V2.size)[1] * 1280
        kept, sizing = make.draw_to_size(block, style, family, target, geometry, seed, GEN_V2)
        if kept is not None:                    # the drawing kept is the seed's own, at the digits it records
            again = make.draw_v2(block, style, family, sizing["digits"], geometry, seed, GEN_V2)
            assert again.boxes == kept.boxes and np.array_equal(R.ink(again.layer), R.ink(kept.layer))


@needs_fonts
def test_gen_v2_records_each_blocks_target_and_the_box_it_reached(pool):
    from bestbefore.synth.spec import GEN_V2
    rows, kept, skipped = [], 0, 0
    for i in range(40):
        s = make.make_one(i, GEN_V2, pool, 9).sidecar
        long_side = max(s["size"])
        if any(b["target"] < V2.size.small for b in s["blocks"] + s["skipped"]):
            assert max(s["background"]["size"]) >= GEN.compose.min_side_small     # small print, M2's rule
        for b in s["blocks"]:
            lo, hi = DECILES[b["decile"]], DECILES[b["decile"] + 1]
            assert lo <= b["target"] <= hi and b["band"] == ("small" if b["target"] < 0.015 else "normal")
            first = next(t for t in b["tokens"] if t["cls"] == "date")
            x1, y1, x2, y2 = first["box"]
            assert b["reached"] == round(min(x2 - x1, y2 - y1) / long_side, 5)    # the box, as written
            want = b["target"] * long_side * (GEN.compose.shrink if b["shrunk"] else 1.0)
            assert abs(b["print_height"] - b["size"] / long_side) < 1e-5
            kept += 1
            if b["geometry"]["orientation"] == "tilted":                    # by its digits, not its box
                assert b["fits"] is None and b["size"] == pytest.approx(want / 1.3, abs=0.015)   # both rounded
                continue
            assert b["fits"] == _within(min(x2 - x1, y2 - y1), want) and b["drawings"] >= 2
            rows.append(b["fits"])
        for k in s["skipped"]:
            assert {"decile", "target"} <= set(k) and "size" not in k
            skipped += 1
    assert kept > 40 and np.mean(rows) >= 0.95


@needs_fonts
def test_gen_v2s_size_shares_are_its_deciles_and_its_bands_are_the_stores(pool):
    from bestbefore.synth.spec import GEN_V2
    sidecars = [make.make_one(i, GEN_V2, pool, 10).sidecar for i in range(20)]
    rows = {r["what"]: r for r in make.shares(sidecars, GEN_V2)}
    assert [rows[f"size: decile {k} of 10"]["spec"] for k in range(1, 11)] == [0.1] * 10
    assert not any(what.startswith("band:") for what in rows)                  # M2's bands are not gen-v2's
    sizes = make.date_sizes(sidecars)
    assert len(sizes) == sum(t["cls"] == "date" and t["box"] is not None for s in sidecars for b in s["blocks"]
                             for t in b["tokens"])
    tilted = [{"size": [1000, 800], "blocks": [
        {"geometry": {"orientation": "tilted"}, "tokens": [{"cls": "date", "box": [0, 0, 100, 40]}]},
        {"geometry": {"orientation": "sideways"}, "tokens": [{"cls": "date", "box": [0, 0, 20, 90]}]}]}]
    assert make.date_sizes(tilted) == [0.04, 0.02] and make.date_sizes(tilted, tilted=False) == [0.02]
    bands = make.size_bands(sizes)
    assert bands["dates"] == len(sizes) and sum(bands[k] for k in ("under 1.5%", "1.5-2.5%", "2.5% and up")) == \
        pytest.approx(1.0, abs=1e-3)
    m2_rows = {r["what"] for r in make.shares([make.make_one(i, GEN, pool, 10).sidecar for i in range(5)], GEN)}
    assert {"band: normal", "band: small"} <= m2_rows and not any(w.startswith("size: decile") for w in m2_rows)


# --- scripts/store_sizes.py: v1-train's dates only -----------------------------------------------------

def _fake_export(root, monkeypatch):
    """A made-up store export (made-up numbers for names): two training
    photos, one validation photo, each with dates and a keyword box, its
    manifest naming every file with its SHA-256, as S6's export has it."""
    from PIL import Image
    folders = {"train": {"train_0": 0, "train_90": 1, "train_180": 2, "train_270": 3}, "val": {"val": 0}}
    photos = {"v1-0001": ("train", (1600, 1200), [("0", 0.40, 0.50, 0.30, 0.020), ("1", 0.2, 0.5, 0.1, 0.02),
                                                    ("0", 0.40, 0.60, 0.010, 0.25)]),
              "v1-0002": ("train", (1000, 800), [("0", 0.5, 0.5, 0.2, 0.05)]),
              "v1-0003": ("val", (1200, 900), [("0", 0.5, 0.5, 0.3, 0.30)])}
    files, counts = {}, {name: 0 for part in folders.values() for name in part}
    manifest = {"photos": {}, "settings": {"folders": folders, "lists": {"train": "train.txt", "val": "val.txt"}}}
    lines = {"train": [], "val": []}
    for name, (part, size, boxes) in photos.items():
        manifest["photos"][name] = {"part": part, "labels": ["expiry" if c == "0" else "due" for c, *_ in boxes]}
        for folder in folders[part]:
            image, labels = root / "images" / folder / f"{name}.jpg", root / "labels" / folder / f"{name}.txt"
            for f in (image, labels):
                f.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", size if folders[part][folder] % 2 == 0 else size[::-1], "white").save(image)
            labels.write_text("".join(f"{c} {x} {y} {w} {h}\n" for c, x, y, w, h in boxes))
            for f in (image, labels):
                files[f.relative_to(root).as_posix()] = hashlib.sha256(f.read_bytes()).hexdigest()
            counts[folder] += 1
            lines[part].append(f"./images/{folder}/{name}.jpg")
    for part, name in manifest["settings"]["lists"].items():
        (root / name).write_text("".join(f"{line}\n" for line in lines[part]))
    manifest |= {"files": files, "counts": {"images": counts}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setenv("STORE_TRAIN_DIR", str(root))
    return root


def test_store_sizes_measures_v1_trains_dates_as_the_store_does(load_script, tmp_path, monkeypatch, capsys):
    store_sizes = load_script("store_sizes")
    folder = _fake_export(tmp_path / "export", monkeypatch)
    sizes, roles = store_sizes.training_sizes(folder)
    # the short side over the long side; the keyword box and the validation photo's date left out
    assert sizes == pytest.approx([0.020 * 1200 / 1600, 0.010, 0.05 * 800 / 1000]) and roles == {"expiry": 3}
    edges = store_sizes.deciles([0.001, 0.02, 0.03, 0.09])
    assert edges[0] == 0.006 and edges[-1] == 0.045 and len(edges) == 11            # ends clipped to M2's
    store_sizes.main()
    printed = capsys.readouterr().out
    assert "3 (expiry 3)" in printed and "V2Size holds other deciles" in printed
    (folder / "labels" / "train_0" / "v1-0002.txt").write_text("0 0.5 0.5 0.9 0.9\n")      # a changed label file
    with pytest.raises(SystemExit, match="doesn't match its manifest"):
        store_sizes.main()


# --- generate.py --set gen-v2-sample ------------------------------------------------------------------

@needs_fonts
def test_the_gen_v2_sample_is_made_with_gen_v2_apart_from_m2s(load_script, fake_expdate, tmp_path, monkeypatch):
    from bestbefore import m2, m2b, runs, sets
    from bestbefore.synth.spec import GEN_V2
    generate = load_script("generate")
    monkeypatch.setattr(generate, "start_run", lambda name, config, seed=None: runs.start_run(
        name, config, seed=seed, root=tmp_path / "runs"))
    generate.sample("gen-v2-sample", 12, 1)
    manifest = json.loads(m2b.manifest_path("gen-v2-sample").read_text())
    assert manifest["spec_sha256"] == spec_hash(GEN_V2) and manifest["photos"] == 12
    assert set(manifest["date_sizes"]) == {"dates", "under 1.5%", "1.5-2.5%", "2.5% and up", "median"}
    assert set(manifest["date_sizes_all"]) == set(manifest["date_sizes"])
    images = m2b.gen_dir() / "images" / "gen_v2_sample"
    assert sorted(p.name for p in images.iterdir()) == [make.image_name(i) for i in range(12)]
    assert not m2.gen_dir().exists() and not m2.manifest_path("gen-v2-sample").exists()     # M2's untouched
    samples = sets.load_set("gen-v2-sample")
    assert [s.id for s in samples] == [make.image_name(i) for i in range(12)]
    lines = [json.loads(line) for line in m2b.sidecar_path("gen_v2_sample").read_text().splitlines()]
    assert all("target" in b for s in lines for b in s["blocks"])
    dates = [(b["geometry"]["orientation"], t) for s in lines for b in s["blocks"] for t in b["tokens"]
             if t["cls"] == "date" and t["box"]]
    assert manifest["date_sizes_all"]["dates"] == len(dates)                  # the check's: its untilted dates
    assert manifest["date_sizes"]["dates"] == sum(o != "tilted" for o, _ in dates) < len(dates)
    deciles = [r["what"] for r in manifest["shares"] if r["what"].startswith("size: decile")]
    assert deciles[0] == "size: decile 1 of 10" and len(deciles) == 10
    assert (tmp_path / "runs").is_dir() and any(d.name.endswith("m2b-generate-gen-v2-sample")
                                                for d in (tmp_path / "runs").iterdir())


# === S9: synth-val-v2's strata, at M2's sizes ==================================================
#
# Each stratum is M2's clean (nozzle-rows: M2's dot-matrix stratum) drawn by
# gen-v2's code with every new thing off and M2's sizes, then one thing on.
# Its sizes must be M2's clean's, the normal band on the digits, or it
# wouldn't compare with M2's strata: a v2 spec draws the store's sizes
# unless its v2.size is None.

def _leaves(spec, prefix="") -> dict:
    from dataclasses import fields, is_dataclass
    out = {}
    for f in fields(spec):
        value = getattr(spec, f.name)
        out |= _leaves(value, f"{prefix}{f.name}.") if is_dataclass(value) else {f"{prefix}{f.name}": value}
    return out


def _changed(spec, base) -> list[str]:
    a, b = _leaves(spec), _leaves(base)
    return sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))


def test_gen_v2s_specs_say_whose_sizes_they_draw():
    from bestbefore.synth.spec import CLEAN_V2, GEN_V2, SYNTH_VAL_V2, V2Size
    assert make.store_sized(GEN_V2) and not make.store_sized(GEN) and not make.store_sized(CLEAN_V2)
    assert not any(make.store_sized(spec) for _, spec in SYNTH_VAL_V2)
    assert spec_hash(GEN_V2) == "2c8b24d5ec86c565a66f47ceaa0f61cfc91785a1bc578817ea0b71a2578c9a7b"   # S8's sample's
    assert spec_dict(replace(GEN_V2, v2=replace(V2, size=None)))["v2"]["size"] is None
    assert V2.size == V2Size()


def test_synth_val_v2s_strata_each_change_one_thing_from_m2s_clean():
    from bestbefore.synth.spec import CLEAN, CLEAN_V2, SYNTH_VAL_V2
    assert [name for name, _ in SYNTH_VAL_V2] == ["upside-down", "glued", "line-pair", "dense-panel", "window",
                                                  "embossed", "worn", "nozzle-rows"]
    # the base: M2's clean, every new thing off and M2's sizes
    assert replace(CLEAN_V2, v2=None) == CLEAN
    assert sorted({k.split(".")[0] if k.startswith("size") else k for k in _changed(CLEAN_V2.v2, V2)}) == \
        ["compose.worn", "size", "style.nozzle", "text.glued"]
    assert (CLEAN_V2.v2.text.glued, CLEAN_V2.v2.compose.worn, CLEAN_V2.v2.style.nozzle, CLEAN_V2.v2.size) == \
        (0.0, 0.0, 0.0, None)
    dot_matrix = replace(dict(SYNTH_VAL)["dot-matrix"], v2=CLEAN_V2.v2)
    changed = {name: _changed(spec, dot_matrix if name == "nozzle-rows" else CLEAN_V2) for name, spec in SYNTH_VAL_V2}
    assert changed == {"upside-down": ["geometry.orientation"],
                       "glued": ["text.single_word", "v2.text.glued"],        # a glued date needs its word
                       "line-pair": ["text.kinds"], "dense-panel": ["text.kinds"],
                       "window": ["style.styles"], "embossed": ["style.styles"],
                       "worn": ["style.styles", "v2.compose.worn"],          # solid only: labels aren't worn
                       "nozzle-rows": ["v2.style.nozzle"]}
    spec = dict(SYNTH_VAL_V2)
    assert spec["glued"].text.single_word == 1.0 and spec["glued"].v2.text.glued == 1.0
    assert spec["worn"].style.styles == (("solid", 1.0),) and spec["worn"].v2.compose.worn == 1.0
    assert spec["upside-down"].geometry.orientation == (("upside-down", 1.0),)
    assert spec["nozzle-rows"].style.styles == (("dot-matrix", 1.0),) and spec["nozzle-rows"].v2.style.nozzle == 1.0
    for name, s in SYNTH_VAL_V2:        # every one a validation set's: held-out fonts, val photos, one target
        assert s.fonts == "held-out" and s.backgrounds == (("val", 1.0),), name
        assert s.text.single_role == "target" and s.compose.two_blocks == 0 and s.compose.block_tries == 10, name
        assert s.size.bands == (("normal", 1.0),) and s.v2.size is None, name       # clean's sizes
        assert (s.camera, s.compose, s.text.month_word) == (CLEAN.camera, CLEAN.compose, 0.0), name
    assert strata_hash(SYNTH_VAL_V2) != strata_hash() == STRATA_SHA256
    assert SET_SEEDS["gen-v2"] == 4 and SET_SEEDS["synth-val-v2"] == 5 and len(set(SET_SEEDS.values())) == len(SET_SEEDS)


def test_a_line_pairs_or_dense_panels_target_is_one_of_its_dates():
    from bestbefore.synth.spec import SYNTH_VAL_V2
    strata = dict(SYNTH_VAL_V2)
    for name in ("line-pair", "dense-panel"):
        rng, place = np.random.default_rng(5), Counter()
        for _ in range(3000):
            block = T.random_block(rng, strata[name].text, strata[name].v2.text)
            dates = [t for t in block.tokens if t.cls == "date"]
            assert [t.role for t in dates].count("target") == 1 and len(dates) >= 2, block
            others = {t.role for t in dates if t.role != "target"}
            assert others <= {"mfg", "expiry", "import"}
            place[len(dates), [t.role for t in dates].index("target")] += 1
        for (n, k), count in place.items():              # each of its dates equally likely
            assert abs(count / sum(c for (m, _), c in place.items() if m == n) - 1 / n) < 0.05, (name, n, k)
    for name in ("upside-down", "glued", "window", "worn"):         # a single line's date is the target already
        block = T.random_block(np.random.default_rng(1), strata[name].text, strata[name].v2.text)
        assert [t.role for t in block.tokens if t.cls == "date"] == ["target"]
    rng = np.random.default_rng(2)                       # gen-v2's own blocks have no target
    assert not any(t.role == "target" for _ in range(2000) for t in T.random_block(rng, TEXT_V2, V2.text).tokens)


def _strata_photos(pool, per: int = 6, spec_of=lambda s: s):
    """`per` photos of each synth-val-v2 stratum, made in memory as task 200
    makes them (set seed 5, photo i of the stratum's block of 300)."""
    from bestbefore.synth.spec import SYNTH_VAL_V2
    return {name: [make.make_one(300 * k + i, spec_of(spec), pool, SET_SEEDS["synth-val-v2"], stratum=name)
                   for i in range(per)] for k, (name, spec) in enumerate(SYNTH_VAL_V2)}


@pytest.fixture(scope="module")
def val_pool(tmp_path_factory):
    """synth-val-v2's backgrounds: conftest's made-up validation photos,
    loaded through the real guards."""
    from conftest import build_fake_expdate

    from bestbefore.synth.spec import CLEAN_V2
    with pytest.MonkeyPatch.context() as mp:
        build_fake_expdate(tmp_path_factory.mktemp("expdate"), mp)
        return compose.load_backgrounds(CLEAN_V2.backgrounds, CLEAN_V2.compose.min_side)


@needs_fonts
def test_a_stratums_digits_fall_in_cleans_band(val_pool):
    """Every block of every stratum is drawn at M2's normal band, its digits
    1.5 to 4.5% of the photo's long side: the height make_one drew for it
    with M2's print_height, from the photo's own generator (or, after 10
    blocks found no place, a fresh one in the band), and a shrunk block 0.8
    of it, as M2's. Nothing of the store's sizes is drawn."""
    from bestbefore.synth.spec import SYNTH_VAL_V2
    lo, hi = GEN.size.normal
    strata = dict(SYNTH_VAL_V2)
    for name, made in _strata_photos(val_pool).items():
        for m in made:
            s = m.sidecar
            rng = np.random.default_rng(s["seed"])
            rng.random()                                       # one block or two (synth-val: always one)
            band, height = compose.print_height(rng, strata[name].size)
            assert band == "normal" and lo <= height <= hi
            long_side = max(s["size"])
            first = len(s["skipped"]) < strata[name].compose.block_tries      # placed at the first size
            for b in s["blocks"]:
                scale = GEN.compose.shrink if b["shrunk"] else 1.0
                assert b["band"] == "normal" and lo * scale - 1e-5 <= b["print_height"] <= hi * scale + 1e-5
                if first:
                    assert b["print_height"] == round(height * scale, 5), (name, b)
                    assert b["size"] == pytest.approx(height * scale * long_side, abs=0.006)
                assert not {"decile", "target", "reached", "fits", "drawings"} & set(b) and b["lines"] >= 1
            for k in s["skipped"]:
                assert k["band"] == "normal" and "target" not in k and k["size"] > 0
            assert m.labels == make.label_text(s)
    store = replace(strata["window"], v2=replace(strata["window"].v2, size=V2.size))    # without None: the store's
    assert all({"decile", "target"} <= set(b) for b in make.make_one(1200, store, val_pool, 5).sidecar["blocks"])


@needs_fonts
def test_a_block_at_m2s_sizes_is_drawn_once_at_its_digits(monkeypatch):
    """At M2's sizes a gen-v2 block is drawn at the digits asked, once, and
    keeps that drawing whatever its box (no correction to a box), as a
    tilted block is; a drawing that lost its ink is drawn from fresh
    draws."""
    from bestbefore.synth.spec import CLEAN_V2
    rng = np.random.default_rng(103)
    for style in ("solid", "label"):
        block = T.random_block(rng, CLEAN_V2.text, CLEAN_V2.v2.text)
        drawn = make._sized(block, style, fonts.random_family(rng, "held-out"), 21.0, 1.0, FLAT, rng, CLEAN_V2)
        assert drawn.layer.params["size"] == pytest.approx(21.0)
        assert drawn.sizing["digits"] == 21.0 and drawn.sizing["fits"] is None and drawn.sizing["drawings"] == 1
    calls = []
    monkeypatch.setattr(make, "draw_v2", _boxes_of([60], calls))
    block = T.Block(((T.Token("08/2027", "date", "target"),),), "single")
    drawn = make._sized(block, "solid", None, 10.0, 1.0, FLAT, np.random.default_rng(0), CLEAN_V2)
    assert calls == [(10.0, calls[0][1])] and drawn.sizing == {"digits": 10.0, "reached": 60, "fits": None,
                                                                "drawings": 1, "tries": 1}


@needs_fonts
def test_every_stratum_photo_has_one_target_and_its_shares_are_exact(val_pool):
    """One target per photo (a line-pair's or dense panel's among its other
    dates, which stay dates), loading as exp; and every share a stratum
    fixes at 0 or 1 is exactly that in what was drawn."""
    from bestbefore import derived
    from bestbefore.samples import Sample
    from bestbefore.synth import batch
    from bestbefore.synth.spec import SYNTH_VAL_V2
    strata = dict(SYNTH_VAL_V2)
    for name, made in _strata_photos(val_pool).items():
        sidecars = [m.sidecar for m in made]
        for m in made:
            dates = [t for b in m.sidecar["blocks"] for t in b["tokens"] if t["cls"] == "date"]
            assert [t["role"] for t in dates].count("target") == 1 and m.sidecar["stratum"] == name
            assert len(dates) == 1 if name not in ("line-pair", "dense-panel") else len(dates) >= 2
            w, h = m.sidecar["size"]
            sample = Sample(m.name, m.name, w, h, tuple(make.label_boxes(m.sidecar)))
            assert [b.label for b in derived.with_targets(sample, m.sidecar).boxes].count("exp") == 1
        rows = make.shares(sidecars, strata[name])
        fixed = [r for r in rows if r["spec"] in (0, 1) and r["measured"] is not None]
        assert fixed and all(batch.share_ok(r) for r in fixed), (name, [r for r in fixed if not batch.share_ok(r)])
        assert not any(r["what"].startswith("size: decile") for r in rows)
        hits = {r["what"]: r["measured"] for r in rows}
        assert hits["band: normal"] == 1.0
    one = {name: {r["what"]: r["measured"] for r in make.shares([m.sidecar for m in made], strata[name])}
           for name, made in _strata_photos(val_pool, per=3).items()}
    assert one["upside-down"]["orientation: upside-down"] == 1.0 and one["glued"]["orientation: upright"] == 1.0
    assert one["glued"]["glued: of words printed before a date"] == 1.0
    assert one["line-pair"]["glued: of words printed before a date"] == 0.0         # no glue but the glued stratum's
    assert one["window"]["style: window"] == 1.0 and one["embossed"]["style: embossed"] == 1.0
    assert one["worn"]["worn patch (solid, dot-matrix, window)"] == 1.0
    assert one["upside-down"]["worn patch (solid, dot-matrix, window)"] in (0.0, None)
    assert one["nozzle-rows"]["dot-matrix: nozzle rows"] == 1.0 and one["nozzle-rows"]["style: dot-matrix"] == 1.0


@pytest.mark.skipif(_sample_files("gen-v2-sample", "2c8b24d5ec86c565a66f47ceaa0f61cfc91785a1bc578817ea0b71a2578c9a7b")
                    is None, reason="needs ExpDate, the fonts and S8's gen-v2-sample on disk")
def test_s8s_gen_v2_sample_is_remade_byte_for_byte():
    """gen-v2 draws as S8 drew it: every JPEG, label file and sidecar line of
    S8's 500-photo sample, made again with GEN_V2, in memory."""
    from bestbefore.synth.spec import GEN_V2
    differ = _remade_differ("gen-v2-sample", GEN_V2, *_sample_files("gen-v2-sample", spec_hash(GEN_V2)))
    assert not differ, differ[:10]


def test_an_import_date_counts_by_its_word_even_as_the_target():
    """In synth-val-v2 a line-pair's or dense panel's import date may be the
    target, its role then "target": the share rows find it by its word
    (S9's trial measured a quarter, half the spec, by role)."""
    pair = _placed("line-pair", "solid", [("MFG:", "prod"), ("02/2025", "date", "mfg", True), ("|", None),
                                          ("IMPORT:", "prod"), ("03/2025", "date", "target", True)])
    dense = _placed("dense-panel", "solid", [("MFG", "prod"), ("01/25", "date", "mfg", True), ("L 12345", "code"),
                                             ("Imported on:", "prod"), ("03/25", "date", "target", True),
                                             ("EXP:", "due"), ("01/27", "date", "expiry", True)], lines=4)
    rows = {}
    make._v2_shares(lambda what, spec, hits: rows.__setitem__(what, list(hits)), [pair, dense], V2_PRINT)
    assert rows["line-pairs: an import date"] == [True] and rows["dense panels: an import line"] == [True]
    assert set(T.IMPORT).isdisjoint(T.PROD + T.DUE)                  # no other word is an import word


@needs_fonts
def test_a_strata_block_without_a_place_is_drawn_at_a_fresh_size_after_10_tries(val_pool, monkeypatch):
    """The owner's call (S9): synth-val-v2's block that finds no place in
    its 10 fresh draws is drawn at a fresh size in its band, 10 more times,
    and so on, so every photo keeps its target. M2's synth-val keeps one
    size through its 10 draws, and gen-v2 draws one block, as before."""
    from bestbefore.synth.spec import CLEAN, GEN_V2, SYNTH_VAL_V2
    real, sizes = make._place, []

    def fails(n):
        def place(block, style, family, size, *args, **kwargs):
            sizes.append(size)
            drawn, spot = real(block, style, family, size, *args, **kwargs)
            return (drawn, None) if len(sizes) <= n else (drawn, spot)
        return place

    line_pair = dict(SYNTH_VAL_V2)["line-pair"]
    assert make.fresh_sizes(line_pair) and not make.fresh_sizes(CLEAN) and not make.fresh_sizes(GEN_V2)
    monkeypatch.setattr(make, "_place", fails(25))
    s = make.make_one(600, line_pair, val_pool, 5, stratum="line-pair").sidecar
    assert len(sizes) == 26 and len(s["skipped"]) == 25 and len(s["blocks"]) == 1
    assert len(set(sizes[:10])) == 1 and len(set(sizes[10:20])) == 1 and len(set(sizes[20:])) == 1
    assert len({sizes[0], sizes[10], sizes[20]}) == 3                       # a fresh size every 10 draws
    long_side = max(s["size"])
    assert all(0.015 <= x / long_side <= 0.045 for x in sizes)              # each in clean's band
    assert [k["size"] for k in s["skipped"]] == [round(x, 2) for x in sizes[:25]]
    sizes.clear()
    monkeypatch.setattr(make, "_place", fails(10 * make.FRESH_SIZES))       # never a place: no target, as M2's
    s = make.make_one(600, line_pair, val_pool, 5, stratum="line-pair").sidecar
    assert len(sizes) == 100 and not s["blocks"] and len(s["skipped"]) == 100
    sizes.clear()
    monkeypatch.setattr(make, "_place", fails(1000))
    s = make.make_one(0, CLEAN, val_pool, 2, stratum="clean").sidecar        # M2's synth-val: one size, 10 draws
    assert len(sizes) == 10 and len(set(sizes)) == 1 and not s["blocks"]


@needs_fonts
def test_a_worn_patch_skips_a_date_the_ink_defects_emptied():
    """Met at scale (S9's job crashed at gen-v2's photo 17,868 on the
    cluster): small dot-matrix print can lose every dot under the ink line
    to the density field, before the wear is chosen, and the patch then
    looked for that date's ink. It now picks among dates with ink; a drawing
    that lost a date's ink is drawn again anyway, as before."""
    from bestbefore.synth.spec import GEN_V2
    rng = np.random.default_rng(0)
    for _ in range(249):                     # the scratch search's first crash, block 248 (dot-matrix, 3.1 px)
        block = T.random_block(rng, GEN_V2.text, V2.text)
        style = ("dot-matrix", "solid", "window")[int(rng.integers(3))]
        family = fonts.random_family(rng, "train")
        geometry = W.random_geometry(rng, GEN_V2.geometry, V2.geometry)
        size, seed = float(rng.uniform(3.0, 9.0)), int(rng.integers(2**62))
    assert (style, round(size, 3)) == ("dot-matrix", 3.119)
    d = make.draw_v2(block, style, family, size, geometry, seed, GEN_V2)          # raised TypeError before
    boxed = [t.cls is not None for t in block.tokens]
    assert any(box is None for box, b in zip(d.boxes, boxed) if b)                # it lost a date: drawn again
    # the patch, given a layer whose one date has no ink, picks none rather than that one
    layer = R.render(block, "dot-matrix", family, 12, np.random.default_rng(1), GEN_V2.style, V2.style)
    dates = [i for i, t in enumerate(block.tokens) if t.cls == "date"]
    faint = replace(layer, tokens=tuple(replace(s, alpha=s.alpha * 0.1) if i == dates[0] and s is not None else s
                                        for i, s in enumerate(layer.tokens)))
    for k in range(20):
        wear = compose.random_wear(np.random.default_rng(k), faint, block, "dot-matrix",
                                   replace(V2.compose, worn=1.0))
        assert wear is None or wear.token in dates[1:]
