"""Contact sheets: many image tiles packed into one picture, to check
labels and predictions by eye."""
from __future__ import annotations

from PIL import Image, ImageFont

FONT = ImageFont.load_default(size=13)


def grid(tiles: list[Image.Image], cols: int) -> Image.Image:
    """Tiles of one size in a grid, `cols` across."""
    w, h = tiles[0].size
    rows = -(-len(tiles) // cols)
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * w, (i // cols) * h))
    return sheet


def rows_of(tiles: list[Image.Image], width: int = 1800) -> Image.Image:
    """Tiles of one height and different widths, packed left to right into rows."""
    rows, row, x = [], [], 0
    for t in tiles:
        if x + t.width > width and row:
            rows.append(row)
            row, x = [], 0
        row.append(t)
        x += t.width + 4
    rows.append(row)
    h = tiles[0].height + 4
    sheet = Image.new("RGB", (width, len(rows) * h), "white")
    for r, row in enumerate(rows):
        x = 0
        for t in row:
            sheet.paste(t, (x, r * h))
            x += t.width + 4
    return sheet
