"""The turned and gridded boxes must sit exactly on the pixels they label.

Each test draws coloured rectangles off-centre, so a box turned the wrong
way, mirrored, or put in the wrong cell lands on the wrong pixels.
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bestbefore.derived import (
    check_written, grid, grid_groups, label_path, load_yolo_set, quarter_for,
    read_list, turn, turn_box, write_list, write_yolo)
from bestbefore.samples import Box, Sample

RED, GREEN, BLUE, PINK = (255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 0, 255)


def _photo(size, rect, colour):
    """A white photo with one filled rectangle, `rect` in pixels (ends exclusive)."""
    im = Image.new("RGB", size, "white")
    im.paste(colour, rect)
    return im


def _box(rect, size, label="date"):
    w, h = size
    return Box(label, (rect[0] / w, rect[1] / h, rect[2] / w, rect[3] / h))


def _ink(im, colour):
    """The tight pixel box (ends exclusive) around every pixel of `colour`."""
    ys, xs = np.nonzero(np.all(np.asarray(im) == colour, axis=-1))
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)


def _px(box, size):
    w, h = size
    x1, y1, x2, y2 = box.xyxy
    return (x1 * w, y1 * h, x2 * w, y2 * h)


# --- Turning ----------------------------------------------------------------

SIZE, RECT = (200, 100), (20, 10, 60, 30)     # a wide photo, the red patch top-left


@pytest.mark.parametrize("quarter", [0, 1, 2, 3])
def test_turned_box_covers_exactly_the_red_pixels(quarter):
    im, (box,) = turn(_photo(SIZE, RECT, RED), [_box(RECT, SIZE)], quarter)
    assert _px(box, im.size) == pytest.approx(_ink(im, RED))


def test_quarter_turns_are_anticlockwise():
    # Anticlockwise by 90: the top-right corner ends up top-left.
    im = _photo(SIZE, (190, 0, 200, 10), RED)
    turned, _ = turn(im, [], 1)
    assert turned.size == (100, 200)
    assert _ink(turned, RED) == (0, 0, 10, 10)


def test_formulas_match_the_plan():
    x1, y1, x2, y2 = 0.1, 0.2, 0.4, 0.3
    assert turn_box((x1, y1, x2, y2), 1) == pytest.approx((y1, 1 - x2, y2, 1 - x1))
    assert turn_box((x1, y1, x2, y2), 3) == pytest.approx((1 - y2, x1, 1 - y1, x2))


def test_four_turns_give_back_the_original():
    im, boxes = _photo(SIZE, RECT, RED), (_box(RECT, SIZE), _box((0, 0, 200, 100), SIZE, "code"))
    out, out_boxes = im, boxes
    for _ in range(4):
        out, out_boxes = turn(out, out_boxes, 1)
    assert np.array_equal(np.asarray(out), np.asarray(im))
    for a, b in zip(out_boxes, boxes):
        assert a.label == b.label and a.xyxy == pytest.approx(b.xyxy)
    # A turn one way and one back is nothing too.
    back, (box,) = turn(*turn(im, boxes[:1], 1), 3)
    assert np.array_equal(np.asarray(back), np.asarray(im))
    assert box.xyxy == pytest.approx(boxes[0].xyxy)


def test_quarter_for_is_a_seeded_fair_coin():
    keys = [f"real_rot/img_{i:05d}.jpg" for i in range(2000)]
    quarters = [quarter_for(k, seed=0) for k in keys]
    assert set(quarters) == {1, 3}
    assert 0.45 < quarters.count(1) / len(keys) < 0.55
    assert quarters == [quarter_for(k, seed=0) for k in keys]       # same every time
    assert quarters != [quarter_for(k, seed=1) for k in keys]       # the seed matters


# --- Grids ------------------------------------------------------------------

# Four photos of different shapes, each with its own colour off-centre.
TILES = [((200, 100), (20, 10, 60, 30), RED),
         ((120, 160), (5, 100, 40, 150), GREEN),
         ((90, 90), (60, 5, 85, 20), BLUE),
         ((160, 40), (130, 2, 158, 38), PINK)]


def test_grid_boxes_land_on_their_pixels_in_their_cells():
    canvas, boxes = grid([(_photo(s, r, c), [_box(r, s)]) for s, r, c in TILES])
    cell = 200                                   # the longest side of the four
    assert canvas.size == (2 * cell, 2 * cell)
    assert canvas.getpixel((0, 0)) == (114, 114, 114)   # padding above photo 0
    for i, ((_, _, colour), box) in enumerate(zip(TILES, boxes)):
        x1, y1, x2, y2 = _px(box, canvas.size)
        assert (x1, y1, x2, y2) == pytest.approx(_ink(canvas, colour))
        col, row = i % 2, i // 2
        assert col * cell <= x1 and x2 <= (col + 1) * cell
        assert row * cell <= y1 and y2 <= (row + 1) * cell


def test_grid_needs_four_photos():
    with pytest.raises(ValueError, match="4 photos"):
        grid([(_photo((10, 10), (0, 0, 1, 1), RED), [])] * 3)


def test_grid_groups_are_by_size_and_the_largest_is_left_over():
    samples = [Sample(id=f"{i}.jpg", image=Path(f"{i}.jpg"), width=w, height=100, boxes=())
               for i, w in enumerate([900, 100, 500, 300, 800, 200, 700, 400, 600])]
    groups, left = grid_groups(samples)
    assert [[s.width for s in g] for g in groups] == [[100, 200, 300, 400], [500, 600, 700, 800]]
    assert [s.width for s in left] == [900]


# --- Writing and reading YOLO sets ------------------------------------------

def test_label_path_swaps_the_last_images_folder():
    assert (label_path(Path("/d/images/m2/images/real_rot/img_1.jpg"))
            == Path("/d/images/m2/labels/real_rot/img_1.txt"))


def test_read_list_resolves_dot_slash_and_keeps_repeats(tmp_path):
    (tmp_path / "a.txt").write_text("./images/x/1.jpg\n/abs/2.jpg\n\n./images/x/1.jpg\n")
    assert read_list(tmp_path / "a.txt") == [
        tmp_path / "images/x/1.jpg", Path("/abs/2.jpg"), tmp_path / "images/x/1.jpg"]


def test_write_then_load_round_trip(tmp_path):
    size = (200, 100)
    boxes = (_box(RECT, size), _box((70, 10, 90, 30), size, "exp"),   # exp is a date to YOLO
             _box((0, 50, 40, 60), size, "due"), _box((50, 50, 80, 60), size, "prod"),
             _box((100, 0, 199, 99), size, "code"),
             Box("code", (0.5, 0.5, 0.5, 0.7)))                       # no area: dropped
    written = write_yolo(tmp_path / "derived", "real_val_rot", "img_00001.jpg",
                         _photo(size, RECT, RED), boxes)
    assert label_path(written.image).exists()
    assert check_written(written) < 0.01                              # pixels

    write_list(tmp_path / "lists" / "real_val_rot.txt", [written.image])
    [back] = load_yolo_set(tmp_path / "lists" / "real_val_rot.txt")
    assert (back.id, back.width, back.height) == ("img_00001.jpg", 200, 100)
    assert [b.label for b in back.boxes] == ["date", "date", "due", "prod", "code"]
    for a, b in zip(back.boxes, boxes):
        assert a.xyxy == pytest.approx(b.xyxy, abs=1e-6)


def test_check_written_catches_a_lost_box(tmp_path):
    size = (200, 100)
    written = write_yolo(tmp_path, "s", "a.jpg", _photo(size, RECT, RED),
                         [_box(RECT, size), _box((0, 50, 40, 60), size, "due")])
    labels = label_path(written.image)
    labels.write_text(labels.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="1 boxes on disk, meant 2"):
        check_written(written)
