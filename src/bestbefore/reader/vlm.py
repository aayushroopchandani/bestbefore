"""Qwen3-VL-2B, M3's third reader (docs/m3-date-reader.md, "Qwen3-VL-2B",
decision 18): its prompt, the answers it is trained to give, the images it
sees, and the examples of its training set.

The VLM takes a whole photo, or for a generated photo a panel round one
generated block, and answers `{"expiry": ..., "mfg": ...}` in JSON, each a
date as YYYY-MM-DD, YYYY-MM when no day is printed, or null. No detector,
no parser, no rules: that is what it is there to test.

Where the answers come from (`*_answer`):
  store    the frozen labels: an expiry box's value, else on a photo that
           prints a shelf life, its mfg date plus that shelf life (the 9
           derived photos, S4); an mfg box's value. Nothing in frame, null.
  ExpDate  the keyword rule (`expdate_answer`): each date takes the role of
           its nearest due or prod mark; the expiry is the date beside a
           due mark, else the only date, else the latest, never one beside
           a prod mark; the mfg the date beside a prod mark. ExpDate's
           train and synth splits mark no expiry, so a rule decides, and
           ExpDate test shows once (S13) how far it agrees with ExpDate's
           own marks.
  gen      the block's own sidecar: its expiry and mfg tokens' parts; a
           relative block's expiry worked out from its mfg date and its
           shelf-life phrase.

Every image is stored at the size the model reads it (`model_size`: sides
multiples of 32, at most MAX_PIXELS), so the HF processor (told not to
resize) and llama.cpp's own resize both leave it as it is and read the same
pixels.
"""
from __future__ import annotations

import io
import json
import math
import re
import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import lmdb
from PIL import Image

from ..samples import Expiry
from ..store import parse_date
from . import parse, shards

# --- The prompt and the answers ---------------------------------------------------------

PROMPT = ("Read the dates printed on this product. Answer with JSON only: "
          '{"expiry": date or null, "mfg": date or null}. Write a date as YYYY-MM-DD, '
          "or YYYY-MM when no day is printed. If the expiry is printed as a shelf life from "
          "the manufacture date, work it out. If no expiry is visible, write null.")
FIELDS = ("expiry", "mfg")


@dataclass(frozen=True)
class Answer:
    """What the VLM says, or is trained to say, about one image."""
    expiry: Expiry | None
    mfg: Expiry | None

    def json(self) -> str:
        """The answer as the model writes it: '{"expiry": "2027-03", "mfg": null}'."""
        return json.dumps({f: None if getattr(self, f) is None else str(getattr(self, f)) for f in FIELDS})

    def as_dict(self) -> dict:
        return json.loads(self.json())


@dataclass(frozen=True)
class Parsed:
    """An output read back: its answer, and what was wrong with it, if
    anything ("no json", "bad json", "bad expiry", "bad mfg"). A field
    that isn't a real date is None in the answer (a none, never a guess)."""
    answer: Answer
    problems: tuple[str, ...] = ()


_DATE = re.compile(r"^\s*(\d{4})\s*[-/.]\s*(\d{1,2})(?:\s*[-/.]\s*(\d{1,2}))?\s*$")
_NULLS = (None, "", "null", "none", "n/a", "na")


def to_expiry(value) -> Expiry | None:
    """A date as the model wrote it ('2027-03-01', '2027-03'; '/' and '.'
    allowed) as an Expiry; None for null and for anything that isn't a
    real date ('2027-13', '2027-02-30', '03/2027')."""
    if value is None or not isinstance(value, str):
        return None
    m = _DATE.match(value)
    if not m:
        return None
    year, month, day = int(m.group(1)), int(m.group(2)), m.group(3)
    try:
        return Expiry(year, month, None if day is None else int(day))
    except ValueError:
        return None


def _object(text: str) -> str | None:
    """The first {...} in a text (an answer never nests)."""
    start = text.find("{")
    if start < 0:
        return None
    end = text.find("}", start)
    return text[start:end + 1] if end >= 0 else None


def _loads(raw: str) -> dict | None:
    """JSON, leniently: as written, then with single quotes, Python's None
    and trailing commas mended."""
    for attempt in (raw, re.sub(r",\s*}", "}", raw.replace("'", '"').replace("None", "null"))):
        try:
            value = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_answer(text: str) -> Parsed:
    """The first {...} in a model's output, parsed as JSON (leniently: code
    fences, single quotes, trailing commas); each field a real date or
    None. Keys are matched without case."""
    raw = _object(text or "")
    if raw is None:
        return Parsed(Answer(None, None), ("no json",))
    value = _loads(raw)
    if value is None:
        return Parsed(Answer(None, None), ("bad json",))
    value = {str(k).strip().lower(): v for k, v in value.items()}
    out, problems = {}, []
    for f in FIELDS:
        given = value.get(f)
        out[f] = to_expiry(given)
        if out[f] is None and not (given is None or (isinstance(given, str) and given.strip().lower() in _NULLS)):
            problems.append(f"bad {f}")
    return Parsed(Answer(**out), tuple(problems))


def value_confidence(tokens: Sequence[tuple[str, float]], field_name: str = "expiry") -> float | None:
    """Decision 15 for the VLM: the lowest probability among the tokens
    that write the field's value (its date's characters), from the output's
    tokens and each one's probability. None when the field holds no date."""
    text = "".join(t for t, _ in tokens)
    m = re.search(r'"%s"\s*:\s*"([^"]*)"' % field_name, text, re.I)
    if m is None or to_expiry(m.group(1)) is None:
        return None
    start, end = m.span(1)
    probs, at = [], 0
    for t, p in tokens:
        if at < end and at + len(t) > start:
            probs.append(p)
        at += len(t)
    return min(probs) if probs else None


# --- Image size ---------------------------------------------------------------------------

FACTOR = 32                 # Qwen3-VL: 16 px patches merged 2 x 2, one visual token per 32 x 32 px
MAX_PIXELS = 1_920_000      # about 1,875 visual tokens (1,600 x 1,200)
MIN_PIXELS = 65_536         # the processor's own floor (256 x 256)


def model_size(width: int, height: int) -> tuple[int, int]:
    """The (width, height) the model reads an image at: Qwen's smart_resize
    (each side rounded to a multiple of FACTOR, half to even as Python
    rounds; then scaled down under MAX_PIXELS or up over MIN_PIXELS, the
    aspect kept), as transformers and llama.cpp both compute it."""
    h = max(FACTOR, round(height / FACTOR) * FACTOR)
    w = max(FACTOR, round(width / FACTOR) * FACTOR)
    if h * w > MAX_PIXELS:
        beta = math.sqrt(height * width / MAX_PIXELS)
        h = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR)
        w = max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif h * w < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h = math.ceil(height * beta / FACTOR) * FACTOR
        w = math.ceil(width * beta / FACTOR) * FACTOR
    return w, h


def to_model_size(image: Image.Image) -> Image.Image:
    """The image as the model reads it: RGB, resized (bicubic, as Qwen's
    processor and llama.cpp resize) to model_size; unchanged if already so."""
    image = image.convert("RGB")
    size = model_size(*image.size)
    return image if image.size == size else image.resize(size, Image.Resampling.BICUBIC)


def visual_tokens(width: int, height: int) -> int:
    """How many visual tokens an image of this (model) size becomes."""
    return (width // FACTOR) * (height // FACTOR)


# --- Answers from labels, rules and sidecars ------------------------------------------------

def store_answer(entry: dict) -> tuple[Answer, str]:
    """A store photo's answer from its entry in the export's text.json (S4):
    the expiry box's frozen value; on a photo with no expiry box, its mfg
    date plus a printed shelf life (a derived expiry); the mfg box's value.
    Returns the answer and how its expiry was found: "printed", "derived"
    or "none"."""
    values = {f: {b["value"] for b in entry["boxes"] if b["label"] == f and b["value"]} for f in FIELDS}
    for f in FIELDS:
        if len(values[f]) > 1:
            raise ValueError(f"the photo's {f} boxes hold {sorted(values[f])}: one value expected")
    mfg = parse_date(next(iter(values["mfg"]))) if values["mfg"] else None
    if values["expiry"]:
        return Answer(parse_date(next(iter(values["expiry"]))), mfg), "printed"
    for life in entry.get("life", []):
        months = parse.shelf_life(life["text"])
        if mfg is not None and months is not None:
            return Answer(parse.add_months(mfg, months), mfg), "derived"
    return Answer(None, mfg), "none"


def gen_answer(block: dict) -> Answer:
    """A generated block's answer from its sidecar: its expiry token's
    parts and its mfg token's (an import date is neither); a relative
    block's expiry is its mfg date plus its shelf-life phrase."""
    found = {}
    for t in block["tokens"]:
        if t["cls"] == "date" and t.get("role") in FIELDS:
            if t["role"] in found:
                raise ValueError(f"a block with two {t['role']} dates")
            found[t["role"]] = Expiry(*t["parts"])
    if block["kind"] == "relative" and "expiry" not in found:
        months = [m for t in block["tokens"] if t["cls"] is None and (m := parse.shelf_life(t["text"]))]
        if len(months) != 1 or "mfg" not in found:
            raise ValueError("a relative block needs one mfg date and one shelf-life phrase")
        found["expiry"] = parse.add_months(found["mfg"], months[0])
    return Answer(found.get("expiry"), found.get("mfg"))


XYXY = tuple[float, float, float, float]
# The keyword rule's geometry, set on ExpDate's training photos (S7: its
# marks sit left, right, above and below their dates; synth's prod marks up
# to 10 date-heights to the left at the 90th percentile, label columns).
LINE_GAP = 12.0     # a mark on a date's line, left or right: at most this many date-heights away
ABOVE_GAP = 2.0     # a mark above or below a date: at most this many date-heights away
GLUE = 0.3          # a mark may overlap its date by this much of the date's height (a glued word)
TWO_NUMBERS = re.compile(r"^\s*\d{1,2}\s*[./-]\s*\d{1,2}\s*$")    # '02.18': no year, or which is the year?


def mark_gap(date: XYXY, mark: XYXY) -> float | None:
    """How far a mark sits from a date, in date-heights, if it is beside
    it: on its line (vertical overlap at least half the smaller height),
    left or right, within LINE_GAP; or above or below it (some horizontal
    overlap), within ABOVE_GAP. 0 for a mark overlapping the date; None
    when it isn't beside it."""
    dh = date[3] - date[1]
    if dh <= 0:
        return None
    small = min(dh, mark[3] - mark[1])
    v_overlap = min(date[3], mark[3]) - max(date[1], mark[1])
    if v_overlap >= 0.5 * small:
        if mark[2] <= date[0] + GLUE * dh:
            gap = (date[0] - mark[2]) / dh
        elif mark[0] >= date[2] - GLUE * dh:
            gap = (mark[0] - date[2]) / dh
        else:
            gap = 0.0
        return max(gap, 0.0) if gap <= LINE_GAP else None
    if min(date[2], mark[2]) - max(date[0], mark[0]) > 0:
        if mark[3] <= date[1] + GLUE * dh:
            gap = (date[1] - mark[3]) / dh
        elif mark[1] >= date[3] - GLUE * dh:
            gap = (mark[1] - date[3]) / dh
        else:
            return None
        return max(gap, 0.0) if gap <= ABOVE_GAP else None
    return None


def _later(a: Expiry, b: Expiry) -> bool:
    return (a.year, a.month, a.day or 0) > (b.year, b.month, b.day or 0)


@dataclass(frozen=True)
class Roles:
    """The keyword rule on one ExpDate photo: its answer (None when the
    photo is dropped), why, and each date's box, value and role (the
    nearest mark's class, due or prod, or None)."""
    answer: Answer | None
    why: str
    dates: tuple[tuple[XYXY, Expiry | None, str | None], ...] = ()


def expdate_roles(entry: dict) -> Roles:
    """The keyword rule on an ExpDate annotation entry (its `ann` boxes in
    pixels). Each date takes the class of its nearest beside mark; the
    expiry is the date beside a due mark, else the only date beside no
    mark, else the latest of those; never a date beside a prod mark. The
    mfg is the date beside a prod mark. The photo is dropped when a date's
    transcription gives the frozen parser no date; when it is two numbers
    alone ('01.31', '02.18': month and day, or month and two-digit year,
    can't be told, and the parser's month-and-year reading would teach
    2031-01 for 31 January); when two dates take one role (which is which
    can't be told); or when the mfg comes after the expiry. `why`: "due",
    "only", "latest" or "no expiry" (an mfg date only, or no date), or the
    reason it was dropped."""
    marks = [(a["cls"], tuple(a["bbox"])) for a in entry["ann"] if a["cls"] in ("due", "prod")]
    dates, bare = [], False
    for a in entry["ann"]:
        if a["cls"] != "date":
            continue
        box, text = tuple(a["bbox"]), (a.get("transcription") or "").strip()
        bare = bare or bool(TWO_NUMBERS.match(text))
        value = parse.crop_date(parse.read_dates(text))
        near = [(g, cls) for cls, m in marks if (g := mark_gap(box, m)) is not None]
        dates.append((box, value, min(near)[1] if near else None))
    dates = tuple(dates)
    if bare:
        return Roles(None, "dropped: a date of two numbers alone", dates)
    if any(v is None for _, v, _ in dates):
        return Roles(None, "dropped: a date doesn't parse", dates)
    due = [v for _, v, r in dates if r == "due"]
    prod = [v for _, v, r in dates if r == "prod"]
    free = [v for _, v, r in dates if r is None]
    if len(due) > 1 or len(prod) > 1:
        return Roles(None, "dropped: two dates take one role", dates)
    if due:
        expiry, why = due[0], "due"
    elif free:
        expiry = free[0]
        for v in free[1:]:
            expiry = v if _later(v, expiry) else expiry
        why = "only" if len(free) == 1 else "latest"
    else:
        expiry, why = None, "no expiry"
    mfg = prod[0] if prod else None
    if expiry is not None and mfg is not None and _later(mfg, expiry):
        return Roles(None, "dropped: the mfg comes after the expiry", dates)
    return Roles(Answer(expiry, mfg), why, dates)


# --- Panels and regions --------------------------------------------------------------------

def overlaps(a: XYXY, b: XYXY) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def inside(inner: XYXY, outer: XYXY) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def cut_back(rect: XYXY, keep: Sequence[XYXY], foreign: Sequence[XYXY]) -> XYXY | None:
    """`rect` with its edges moved in until it overlaps no foreign box,
    still holding every box in `keep`: for each foreign box it overlaps,
    the one edge whose move loses the least area. None when that can't be
    done (a foreign box overlaps a kept one, or sits between them)."""
    x1, y1, x2, y2 = rect
    for _ in range(4 * len(foreign) + 1):
        hit = next((f for f in foreign if overlaps((x1, y1, x2, y2), f)), None)
        if hit is None:
            return (x1, y1, x2, y2)
        options = [(hit[2], y1, x2, y2), (x1, y1, hit[0], y2), (x1, hit[3], x2, y2), (x1, y1, x2, hit[1])]
        options = [o for o in options if o[2] > o[0] and o[3] > o[1] and all(inside(k, o) for k in keep)]
        if not options:
            return None
        x1, y1, x2, y2 = max(options, key=lambda o: (o[2] - o[0]) * (o[3] - o[1]))
    return None


PANEL_MARGIN = (0.25, 1.0)      # a panel's margin on each side, as a share of its block's longer side
MIN_PANEL = 256                 # a panel's shorter side, if the photo allows: about MIN_PIXELS' side


def panel(place: XYXY, size: tuple[int, int], foreign: Sequence[XYXY], rng) -> XYXY | None:
    """A panel round one generated block (`place`, its pixel box): each
    side's margin drawn from PANEL_MARGIN times the block's longer side,
    widened to MIN_PANEL where the photo allows, clipped to the photo,
    then cut back from every foreign box (the background's dates, other
    blocks). None when the block can't be had alone."""
    w, h = size
    long = max(place[2] - place[0], place[3] - place[1])
    m = rng.uniform(*PANEL_MARGIN, size=4) * long
    x1, y1, x2, y2 = place[0] - m[0], place[1] - m[1], place[2] + m[2], place[3] + m[3]
    if (grow := (MIN_PANEL - (x2 - x1)) / 2) > 0:
        x1, x2 = x1 - grow, x2 + grow
    if (grow := (MIN_PANEL - (y2 - y1)) / 2) > 0:
        y1, y2 = y1 - grow, y2 + grow
    if any(overlaps(place, f) for f in foreign):
        return None
    return cut_back((max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)), [place], foreign)


REGION_SIDE = (0.3, 0.7)        # a no-date region's sides, as shares of the photo's


def empty_region(size: tuple[int, int], dates: Sequence[XYXY], rng, tries: int = 50) -> XYXY | None:
    """A region of a photo holding no date (each date box kept a quarter of
    its height clear): sides drawn from REGION_SIDE of the photo's, placed
    at random; None if `tries` draws all hit a date."""
    w, h = size
    clear = [_grown(d, 0.25 * (d[3] - d[1])) for d in dates]
    for _ in range(tries):
        rw, rh = rng.uniform(*REGION_SIDE) * w, rng.uniform(*REGION_SIDE) * h
        x, y = rng.uniform(0, w - rw), rng.uniform(0, h - rh)
        rect = (x, y, x + rw, y + rh)
        if not any(overlaps(rect, c) for c in clear):
            return rect
    return None


MFG_MARGIN = (1.0, 4.0)         # an mfg-only region's margin, in date-heights


def mfg_region(size: tuple[int, int], mfg: XYXY, mark: XYXY | None, others: Sequence[XYXY], rng) -> XYXY | None:
    """A region holding the mfg date (and its mark) but no other date: their
    box grown by a margin drawn from MFG_MARGIN date-heights a side, clipped,
    then cut back from every other date (each kept a quarter of its height
    clear). None when the mfg date can't be had alone."""
    w, h = size
    keep = [mfg] + ([mark] if mark else [])
    box = (min(k[0] for k in keep), min(k[1] for k in keep), max(k[2] for k in keep), max(k[3] for k in keep))
    m = rng.uniform(*MFG_MARGIN, size=4) * (mfg[3] - mfg[1])
    rect = (max(0.0, box[0] - m[0]), max(0.0, box[1] - m[1]), min(float(w), box[2] + m[2]), min(float(h), box[3] + m[3]))
    clear = [_grown(o, 0.25 * (o[3] - o[1])) for o in others]
    if any(overlaps(k, c) for k in keep for c in clear):
        return None
    return cut_back(rect, keep, clear)


def _grown(b: XYXY, by: float) -> XYXY:
    return (b[0] - by, b[1] - by, b[2] + by, b[3] + by)


# --- Examples: the VLM set's records ----------------------------------------------------------

KINDS = ("store", "expdate-real", "expdate-synth", "gen", "negative-empty", "negative-mfg")
REPEATS = {"store": 3}      # the plan's table: the store's 4 turns listed 3 times in the epoch


@dataclass(frozen=True)
class Example:
    """One training example: an image at the model's size and the answer it
    is trained to give. `source`: where it was cut from (a store photo by
    its number only, never its name)."""
    key: str                    # "vlm/000001234"
    jpeg: bytes
    answer: Answer
    kind: str
    source: str
    tags: dict = field(default_factory=dict)

    def image(self) -> Image.Image:
        with Image.open(io.BytesIO(self.jpeg)) as im:
            return im.convert("RGB")


def encode(image: Image.Image) -> bytes:
    """An image at the model's size as JPEG 95, 4:4:4 (shards' settings)."""
    if image.size != model_size(*image.size):
        raise ValueError(f"{image.size} isn't a model size")
    return shards.encode_jpeg(image)


def to_bytes(ex: Example) -> bytes:
    head = json.dumps({"answer": ex.answer.as_dict(), "kind": ex.kind, "source": ex.source, "tags": ex.tags},
                      ensure_ascii=False, separators=(",", ":")).encode()
    return struct.pack(">I", len(head)) + head + ex.jpeg


def from_bytes(key: str, value: bytes) -> Example:
    (n,) = struct.unpack(">I", value[:4])
    head = json.loads(value[4:4 + n])
    answer = Answer(*(to_expiry(head["answer"][f]) for f in FIELDS))
    return Example(key, bytes(value[4 + n:]), answer, head["kind"], head["source"], head["tags"])


def key_of(i: int) -> str:
    return f"vlm/{i:09d}"


class Writer:
    """Appends examples to a new LMDB folder (shards.Writer's way: batches,
    a map that doubles when full, the count written on leaving)."""

    def __init__(self, folder: Path, batch: int = 200):
        folder.mkdir(parents=True, exist_ok=False)
        self.env = lmdb.open(str(folder), map_size=shards.MAP_START, subdir=True, lock=False,
                             sync=False, metasync=False, meminit=False, readahead=False)
        self.count, self.batch, self._pending = 0, batch, []

    def add(self, jpeg: bytes, answer: Answer, kind: str, source: str, tags: dict | None = None) -> Example:
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}")
        ex = Example(key_of(self.count), jpeg, answer, kind, source, tags or {})
        self._pending.append((ex.key.encode(), to_bytes(ex)))
        self.count += 1
        if len(self._pending) >= self.batch:
            self._flush()
        return ex

    def _flush(self, extra=()) -> None:
        items = self._pending + list(extra)
        while True:
            try:
                with self.env.begin(write=True) as txn:
                    for k, v in items:
                        txn.put(k, v)
                break
            except lmdb.MapFullError:
                self.env.set_mapsize(self.env.info()["map_size"] * 2)
        self._pending = []

    def close(self) -> int:
        self._flush(((shards.COUNT_KEY, str(self.count).encode()),))
        self.env.sync(True)
        self.env.close()
        return self.count

    def __enter__(self) -> Writer:
        return self

    def __exit__(self, kind, value, tb) -> None:
        if kind is None:
            self.close()
        else:
            self.env.close()


def examples(env: lmdb.Environment, start: int = 0, stop: int | None = None) -> Iterator[Example]:
    """Records start..stop-1 in key order."""
    stop = shards.count(env) if stop is None else stop
    with env.begin() as txn:
        for i in range(start, stop):
            value = txn.get(key_of(i).encode())
            if value is None:
                raise KeyError(key_of(i))
            yield from_bytes(key_of(i), value)


def get(env: lmdb.Environment, i: int) -> Example:
    with env.begin(buffers=True) as txn:
        value = txn.get(key_of(i).encode())
        if value is None:
            raise KeyError(key_of(i))
        return from_bytes(key_of(i), bytes(value))


def example_problems(ex: Example) -> list[str]:
    """What is wrong with one record: it must decode at a model size, its
    kind be known, and its answer round-trip through its own JSON."""
    out = []
    try:
        im = ex.image()
    except Exception as e:          # noqa: BLE001 (any decoding failure is the problem reported)
        return [f"{ex.key}: doesn't decode ({e})"]
    if im.size != model_size(*im.size):
        out.append(f"{ex.key}: {im.size} isn't a model size")
    if ex.kind not in KINDS:
        out.append(f"{ex.key}: kind {ex.kind!r}")
    if parse_answer(ex.answer.json()).answer != ex.answer:
        out.append(f"{ex.key}: its answer doesn't round-trip")
    return out


def epoch(kinds: Sequence[str], rng) -> list[int]:
    """One epoch's order: every record once, a store record REPEATS times,
    shuffled."""
    order = [i for i, k in enumerate(kinds) for _ in range(REPEATS.get(k, 1))]
    rng.shuffle(order)
    return order


# --- Photos to read: the source images, and the validation photos -------------------------

def load_photo(path: Path, rect: XYXY | None = None) -> Image.Image:
    """A photo as the VLM reads it: upright, its long side at most 1,600 px
    (decision 3, as store_train.shrink makes the store export), `rect` (0..1
    of the photo) cut from it, at the model's size."""
    from ..images import load_upright
    from ..store_train import shrink
    image = shrink(load_upright(path))
    if rect is not None:
        w, h = image.size
        image = image.crop((round(rect[0] * w), round(rect[1] * h), round(rect[2] * w), round(rect[3] * h)))
    return to_model_size(image)


VAL_SETS = ("real-val", "store-val")


@dataclass(frozen=True)
class ValPhoto:
    """A validation photo: whole, upright, and the answer it should get.
    `answer` None: nothing to score it against (a real-val photo the
    keyword rule drops). A store photo is named by its number only."""
    key: str
    image: Path
    answer: Answer | None
    tags: dict = field(default_factory=dict)


def val_photos(name: str) -> list[ValPhoto]:
    """real-val: ExpDate's 165 validation photos, each answer by the keyword
    rule, as the training set's ExpDate answers (ExpDate's train split
    marks no expiry). store-val: v1-val and v3-val's 34 photos from the
    store export, each answer from the frozen labels (store_answer)."""
    from .. import expdate
    from ..config import store_train_dir
    from . import piles
    if name == "real-val":
        ann = expdate.read_annotations("real-train")
        folder = expdate.split_dir("real-train") / "images"
        out = []
        for photo in sorted(piles.val_ids()):
            roles = expdate_roles(ann[photo])
            out.append(ValPhoto(f"real-val/{photo}", folder / photo, roles.answer, {"why": roles.why}))
        return out
    if name == "store-val":
        if why := piles.store_missing():
            raise RuntimeError(f"store-val can't be read: {why}")
        root = store_train_dir()
        text = json.loads(piles.store_text_path().read_text())
        manifest = json.loads((root / "manifest.json").read_text())
        out = []
        for number, entry in sorted(text["photos"].items()):
            if entry["part"] != "val":
                continue
            answer, how = store_answer(entry)
            meta = manifest["photos"][number]
            out.append(ValPhoto(f"store-val/{number}", root / "images" / "val" / f"{number}.jpg", answer,
                                {"tag": meta["tag"], "design": meta["design"], "expiry_from": how,
                                 "month": entry.get("month")}))
        return out
    raise ValueError(f"unknown validation set {name!r}; one of {', '.join(VAL_SETS)}")


# --- The model's inputs: training and answering (S7's loop, scripts/train_vlm.py) -----------

# LoRA on the language model's attention and MLP projections only; the
# vision tower and its merger stay frozen (decision 18).
LORA_TARGETS = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05}
IGNORE = -100           # a label the loss skips


def prompt_text(processor) -> str:
    """The chat as Qwen's template writes it, up to the assistant's turn:
    '<|im_start|>user\\n<|vision_start|><|image_pad|><|vision_end|>PROMPT<|im_end|>\\n<|im_start|>assistant\\n'."""
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def answer_ids(processor, answer: Answer) -> list[int]:
    """The answer's tokens as the model should write them: its JSON, then
    <|im_end|>, which ends its turn."""
    return processor.tokenizer(answer.json() + "<|im_end|>", add_special_tokens=False)["input_ids"]


def model_inputs(processor, image: Image.Image, answer: Answer | None = None) -> dict:
    """One image's inputs: the prompt's tokens with the image's visual
    tokens (the processor told not to resize: the image is at its model
    size), and, given an answer, its tokens after them with `labels` that
    are IGNORE everywhere else: the loss falls on the answer only."""
    import torch
    if image.size != model_size(*image.size):
        raise ValueError(f"{image.size} isn't a model size")
    out = dict(processor(text=[prompt_text(processor)], images=[image], return_tensors="pt", do_resize=False))
    if answer is None:
        return out
    ids = torch.tensor([answer_ids(processor, answer)], dtype=out["input_ids"].dtype)
    n = out["input_ids"].shape[1]
    out["input_ids"] = torch.cat([out["input_ids"], ids], 1)
    out["attention_mask"] = torch.cat([out["attention_mask"], torch.ones_like(ids)], 1)
    if "mm_token_type_ids" in out:                  # 1 on visual tokens, 0 on text
        out["mm_token_type_ids"] = torch.cat([out["mm_token_type_ids"], torch.zeros_like(ids)], 1)
    labels = torch.full_like(out["input_ids"], IGNORE)
    labels[:, n:] = ids
    out["labels"] = labels
    return out


def collate(items: Sequence[dict], pad_id: int) -> dict:
    """Examples into one batch: token rows padded on the right (pad tokens,
    attention 0, labels IGNORE), every image's patches stacked in order."""
    import torch
    width = max(it["input_ids"].shape[1] for it in items)
    fill = {"input_ids": pad_id, "attention_mask": 0, "labels": IGNORE, "mm_token_type_ids": 0}
    out = {}
    for name, value in fill.items():
        if name in items[0]:
            out[name] = torch.cat([torch.nn.functional.pad(it[name], (0, width - it[name].shape[1]), value=value)
                                   for it in items])
    out["pixel_values"] = torch.cat([it["pixel_values"] for it in items])
    out["image_grid_thw"] = torch.cat([it["image_grid_thw"] for it in items])
    return out


def answer_with(model, processor, image: Image.Image, max_new_tokens: int = 64) -> dict:
    """The model's greedy answer to one image: its text, each token with the
    probability the raw logits gave it (a plain softmax, as llama.cpp gives
    with a negative temperature), the parsed answer and the seconds taken."""
    import time
    import torch
    t0 = time.perf_counter()
    inputs = {k: v.to(model.device) for k, v in model_inputs(processor, image).items()}
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                             top_p=None, top_k=None, output_logits=True, return_dict_in_generate=True)
    new = out.sequences[0, inputs["input_ids"].shape[1]:].tolist()
    tokens = []
    for tok, logits in zip(new, out.logits):
        prob = torch.softmax(logits[0].float(), -1)[tok].item()
        if tok in (processor.tokenizer.eos_token_id, processor.tokenizer.pad_token_id):
            break
        tokens.append((processor.tokenizer.decode([tok]), prob))
    text = "".join(t for t, _ in tokens)
    parsed = parse_answer(text)
    return {"text": text, "tokens": tokens, "answer": parsed.answer.as_dict(), "problems": list(parsed.problems),
            "confidence": value_confidence(tokens), "prompt_tokens": inputs["input_ids"].shape[1],
            "seconds": round(time.perf_counter() - t0, 3)}


def kinds(env: lmdb.Environment) -> list[str]:
    """Every record's kind, in key order, reading only each record's head
    (LMDB maps the values: the JPEGs behind them are never touched)."""
    out = []
    with env.begin(buffers=True) as txn:
        for i in range(shards.count(env)):
            value = txn.get(key_of(i).encode())
            (n,) = struct.unpack(">I", bytes(value[:4]))
            out.append(json.loads(bytes(value[4:4 + n]))["kind"])
    return out


def score_answer(truth: Answer | None, said: dict) -> dict:
    """An answer against its truth, field by field: right, wrong or none
    (scoring.score: the day only where the truth prints one); a field whose
    truth is null scores "claimed" when the model gave a date (a date where
    none is in view is a wrong read) and "null" when it gave none. No truth:
    nothing scored."""
    from .. import scoring
    if truth is None:
        return {}
    out = {}
    for f in FIELDS:
        read, want = to_expiry(said.get(f)), getattr(truth, f)
        out[f] = scoring.score(want, read) if want is not None else ("claimed" if read is not None else "null")
    return out
