from PIL import Image

from bestbefore.images import ORIENTATION, load_upright, orientation, upright_size


def _save(path, size, flag=None):
    im = Image.new("RGB", size, "white")
    im.paste((255, 0, 0), (0, 0, 10, 10))   # a red corner to follow
    exif = Image.Exif()
    if flag is not None:
        exif[ORIENTATION] = flag
    im.save(path, exif=exif)


def test_no_flag_loads_as_stored(tmp_path):
    path = tmp_path / "plain.jpg"
    _save(path, (40, 20))
    assert orientation(path) == 1
    assert load_upright(path).size == (40, 20)
    assert upright_size(path) == (40, 20)


def test_rotate_flag_turns_the_image(tmp_path):
    # Flag 6: stored sideways, shown turned 90 degrees clockwise.
    path = tmp_path / "sideways.jpg"
    _save(path, (40, 20), flag=6)
    im = load_upright(path)
    assert im.size == (20, 40)
    assert upright_size(path) == (20, 40)
    # The stored top-left corner ends up top-right.
    r, g, b = im.getpixel((15, 4))
    assert r > 200 and g < 80 and b < 80
