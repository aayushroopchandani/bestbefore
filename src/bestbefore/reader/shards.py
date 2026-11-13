"""M3's crop shards (docs/m3-date-reader.md, decision 4 and "A crop
record"): every crop of a pile in one LMDB file, each with its text, its
box inside the crop and the turn it was given.

Why LMDB: the cluster opens a file in about 25 ms, so 700,000 crop files
would cost hours a pass; one LMDB file costs nothing to open.

A pile is a folder, `<root>/<pile>/`, holding LMDB's data.mdb (and
manifest.json, written by scripts/make_crops.py). Record i sits under the
key "<pile>/<i:09d>", i counting from 0 in the order the crops were cut;
COUNT_KEY holds how many there are. A record's value is a 4-byte big-endian
length, that many bytes of JSON (every field but the key and the JPEG), then
the JPEG itself: readable from any language, no pickle.

No lock file is used (lock=False): a pile has one writer, which finishes
before anything reads it, and Lustre may refuse the locks LMDB takes.
"""
from __future__ import annotations

import io
import json
import struct
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import lmdb
from PIL import Image

JPEG_QUALITY = 95       # decision 4
SUBSAMPLING = 0         # 4:4:4: no colour halved round small print (PIL's 0)
COUNT_KEY = b"__count__"
MAP_START = 1 << 30     # LMDB's map, 1 GiB at first, doubled whenever it fills
CLASSES = ("date", "code", None)    # decision 9: dates, codes and class-less tokens
ROLES = ("expiry", "mfg", "import", "other", "target", None)


@dataclass(frozen=True)
class Crop:
    """One stored crop. `box` is the token's box inside the crop, in its
    pixels (floats, rounded to 0.01). `turn`: the quarter turns
    (anticlockwise) it was given when cut, to stand upright in a training
    pile, 0 in a suite cut as it lies; tags["upright"]: the quarter turns
    still needed to stand it upright (0 for every training crop; a suite's
    truth for the orientation step, decision 14)."""
    key: str                                    # "gen/000123456"
    jpeg: bytes                                 # upright or as cut, with its margin
    box: tuple[float, float, float, float]
    text: str                                   # as printed, case kept; never ""
    cls: str | None                             # date, code, None (class-less)
    source: str                                 # the photo and the token in it
    role: str | None = None                     # dates: expiry, mfg, import, other, target
    parts: tuple[int, int, int | None] | None = None    # dates: (year, month, day or None)
    turn: int = 0
    tags: dict = field(default_factory=dict)

    def image(self) -> Image.Image:
        """The crop as RGB."""
        with Image.open(io.BytesIO(self.jpeg)) as im:
            return im.convert("RGB")


def encode_jpeg(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=JPEG_QUALITY, subsampling=SUBSAMPLING)
    return buf.getvalue()


def key_of(pile: str, i: int) -> str:
    return f"{pile}/{i:09d}"


def to_bytes(crop: Crop) -> bytes:
    head = json.dumps({"box": [round(v, 2) for v in crop.box], "text": crop.text, "cls": crop.cls,
                       "source": crop.source, "role": crop.role,
                       "parts": None if crop.parts is None else list(crop.parts),
                       "turn": crop.turn, "tags": crop.tags},
                      ensure_ascii=False, separators=(",", ":")).encode()
    return struct.pack(">I", len(head)) + head + crop.jpeg


def from_bytes(key: str, value: bytes) -> Crop:
    (n,) = struct.unpack(">I", value[:4])
    head = json.loads(value[4:4 + n])
    parts = head["parts"]
    return Crop(key=key, jpeg=bytes(value[4 + n:]), box=tuple(head["box"]), text=head["text"],
                cls=head["cls"], source=head["source"], role=head["role"],
                parts=None if parts is None else tuple(parts), turn=head["turn"], tags=head["tags"])


# --- Writing ----------------------------------------------------------------------

class Writer:
    """Appends crops to a new pile folder, in batches of `batch` records a
    transaction; LMDB's map doubles whenever it fills. Use as a context
    manager: leaving it writes COUNT_KEY and syncs to disk."""

    def __init__(self, folder: Path, pile: str, batch: int = 2000):
        folder.mkdir(parents=True, exist_ok=False)
        self.folder, self.pile, self.batch = folder, pile, batch
        self.env = lmdb.open(str(folder), map_size=MAP_START, subdir=True, lock=False,
                             sync=False, metasync=False, meminit=False, readahead=False)
        self.count = 0
        self._pending: list[tuple[bytes, bytes]] = []

    def add(self, **fields) -> Crop:
        """A record of the next key from Crop's fields (key excepted)."""
        crop = Crop(key=key_of(self.pile, self.count), **fields)
        self._pending.append((crop.key.encode(), to_bytes(crop)))
        self.count += 1
        if len(self._pending) >= self.batch:
            self.flush()
        return crop

    def flush(self, extra: tuple[tuple[bytes, bytes], ...] = ()) -> None:
        items = self._pending + list(extra)
        while True:
            try:
                with self.env.begin(write=True) as txn:
                    for k, v in items:
                        txn.put(k, v)
                break
            except lmdb.MapFullError:   # the transaction was aborted: grow the map, write it again
                self.env.set_mapsize(self.env.info()["map_size"] * 2)
        self._pending = []

    def close(self) -> int:
        self.flush(((COUNT_KEY, str(self.count).encode()),))
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


# --- Reading ----------------------------------------------------------------------

def open_pile(folder: Path) -> lmdb.Environment:
    if not (folder / "data.mdb").exists():
        raise FileNotFoundError(f"no pile in {folder}")
    return lmdb.open(str(folder), readonly=True, lock=False, subdir=True, readahead=False)


def count(env: lmdb.Environment) -> int:
    with env.begin() as txn:
        value = txn.get(COUNT_KEY)
    if value is None:
        raise ValueError("the pile has no count: its writer didn't finish")
    return int(value)


def get(env: lmdb.Environment, pile: str, i: int) -> Crop:
    key = key_of(pile, i)
    with env.begin(buffers=True) as txn:
        value = txn.get(key.encode())
        if value is None:
            raise KeyError(key)
        return from_bytes(key, bytes(value))


def records(env: lmdb.Environment, pile: str, start: int = 0, stop: int | None = None) -> Iterator[Crop]:
    """Records start..stop-1 in key order, by one cursor."""
    stop = count(env) if stop is None else stop
    with env.begin() as txn:
        cursor = txn.cursor()
        if start >= stop or not cursor.set_key(key_of(pile, start).encode()):
            return
        for i in range(start, stop):
            key, value = cursor.item()
            if key.decode() != key_of(pile, i):
                raise KeyError(f"record {key_of(pile, i)} missing: found {key.decode()}")
            yield from_bytes(key.decode(), value)
            cursor.next()


def problems(crop: Crop, max_short: float) -> list[str]:
    """What is wrong with one record, if anything: it must decode, its box
    lie inside it with an area, its box's short side be at most
    `max_short` (rounding allowed), its text be non-empty, its class,
    role and turns be ones a pile holds, and a date carry real parts."""
    out = []
    try:
        w, h = crop.image().size
    except Exception as e:         # noqa: BLE001 - any failure to decode is the finding
        return [f"{crop.key}: doesn't decode ({e})"]
    x1, y1, x2, y2 = crop.box
    if not (0 <= x1 < x2 <= w + 0.01 and 0 <= y1 < y2 <= h + 0.01):
        out.append(f"{crop.key}: box {crop.box} not inside its {w} x {h} crop")
    elif min(x2 - x1, y2 - y1) > max_short + 0.51:
        out.append(f"{crop.key}: box short side {min(x2 - x1, y2 - y1):.2f} over {max_short}")
    if not crop.text.strip():
        out.append(f"{crop.key}: empty text")
    if crop.cls not in CLASSES:
        out.append(f"{crop.key}: class {crop.cls!r}")
    if crop.role not in ROLES or (crop.role is not None and crop.cls != "date"):
        out.append(f"{crop.key}: role {crop.role!r} on class {crop.cls!r}")
    if crop.turn not in (0, 1, 2, 3) or crop.tags.get("upright") not in (0, 1, 2, 3):
        out.append(f"{crop.key}: turn {crop.turn}, upright {crop.tags.get('upright')!r}")
    if crop.parts is not None:
        if crop.cls != "date":
            out.append(f"{crop.key}: parts on class {crop.cls!r}")
        else:
            from ..samples import Expiry
            try:
                Expiry(*crop.parts)
            except (TypeError, ValueError) as e:
                out.append(f"{crop.key}: parts {crop.parts} ({e})")
    return out
