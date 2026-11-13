"""Crop shards: the LMDB record, its writer and reader, its checks (reader/shards.py)."""
import io

import pytest
from PIL import Image

from bestbefore.reader import shards


def _jpeg(size=(40, 20), colour=(30, 30, 30)) -> bytes:
    return shards.encode_jpeg(Image.new("RGB", size, colour))


def _fields(i: int = 0, **over) -> dict:
    fields = {"jpeg": _jpeg(), "box": (10.0, 5.0, 30.0, 15.0), "text": f"0{i % 9 + 1}/2027", "cls": "date",
              "source": f"gen_v2/{i:06d}.jpg#b0t1", "role": "expiry", "parts": (2027, i % 9 + 1, None),
              "turn": 3, "tags": {"style": "dot-matrix", "upright": 0}}
    return fields | over


def test_a_record_round_trips():
    crop = shards.Crop(key="gen/000000007", **_fields(7, text="Best Before 1 Year from Mfg. Dt. – é"))
    back = shards.from_bytes(crop.key, shards.to_bytes(crop))
    assert back == crop
    assert back.image().size == (40, 20)


def test_a_record_is_json_then_the_jpeg():
    value = shards.to_bytes(shards.Crop(key="k", **_fields()))
    n = int.from_bytes(value[:4], "big")
    assert value[4:5] == b"{" and value[4 + n:4 + n + 2] == b"\xff\xd8"     # a JPEG starts FF D8


def test_jpegs_are_quality_95_without_halved_colour():
    with Image.open(io.BytesIO(_jpeg((64, 64)))) as im:
        assert im.layer[0][1:3] == im.layer[1][1:3] == (1, 1)            # 4:4:4: one sample a pixel


def test_writer_and_reader(tmp_path):
    folder = tmp_path / "gen"
    with shards.Writer(folder, "gen", batch=3) as w:
        written = [w.add(**_fields(i)) for i in range(10)]
    env = shards.open_pile(folder)
    assert shards.count(env) == 10
    assert [c.key for c in written] == [f"gen/{i:09d}" for i in range(10)]
    assert list(shards.records(env, "gen")) == written
    assert list(shards.records(env, "gen", 4, 7)) == written[4:7]
    assert shards.get(env, "gen", 9) == written[9]
    with pytest.raises(KeyError):
        shards.get(env, "gen", 10)
    env.close()


def test_the_map_grows_when_full(tmp_path, monkeypatch):
    monkeypatch.setattr(shards, "MAP_START", 64 * 1024)
    big = shards.encode_jpeg(Image.effect_noise((200, 200), 80).convert("RGB"))
    with shards.Writer(tmp_path / "p", "p", batch=5) as w:
        for i in range(40):
            w.add(**_fields(i, jpeg=big))
    env = shards.open_pile(tmp_path / "p")
    assert shards.count(env) == 40 and env.info()["map_size"] > 64 * 1024
    assert shards.get(env, "p", 39).jpeg == big
    env.close()


def test_a_writer_that_fails_leaves_no_count(tmp_path):
    with pytest.raises(RuntimeError):
        with shards.Writer(tmp_path / "p", "p", batch=2) as w:
            for i in range(5):
                w.add(**_fields(i))
            raise RuntimeError("a worker died")
    env = shards.open_pile(tmp_path / "p")
    with pytest.raises(ValueError):
        shards.count(env)
    env.close()


def test_a_writer_never_writes_into_an_existing_folder(tmp_path):
    (tmp_path / "p").mkdir()
    with pytest.raises(FileExistsError):
        shards.Writer(tmp_path / "p", "p")


@pytest.mark.parametrize("over, finding", [
    ({}, None),
    ({"box": (10.0, 5.0, 45.0, 15.0)}, "not inside"),
    ({"box": (10.0, 5.0, 10.0, 15.0)}, "not inside"),
    ({"jpeg": _jpeg((200, 100)), "box": (0.0, 0.0, 200.0, 100.0)}, "short side"),
    ({"text": "  "}, "empty text"),
    ({"cls": "due", "role": None, "parts": None}, "class"),
    ({"cls": "code", "role": "expiry", "parts": None}, "role"),
    ({"cls": "code", "role": None, "parts": (2027, 1, None)}, "parts on class"),
    ({"parts": (2027, 2, 30)}, "parts"),
    ({"turn": 4}, "turn"),
    ({"tags": {}}, "upright"),
    ({"jpeg": b"not a jpeg"}, "doesn't decode"),
])
def test_problems(over, finding):
    found = shards.problems(shards.Crop(key="k", **_fields(**over)), max_short=64)
    assert (found == []) if finding is None else (len(found) == 1 and finding in found[0])
