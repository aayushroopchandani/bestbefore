"""Make the README's figures from ExpDate, into docs/assets/.

    python scripts/readme_figures.py

  expdate_dates.jpg         12 labelled expiry dates from the ExpDate test
                            split, each showing a different difficulty
  expdate_small_dates.jpg   3 whole test photos with the expiry boxed, and
                            the same date zoomed in
  date_box_sizes_{light,dark}.png
                            how big date boxes are at YOLO's 640 px input,
                            real vs synthetic (one image per README theme)

Only ExpDate photos are used (CC BY 4.0, credited in the README). Store
photos never go into a figure.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont, ImageOps  # noqa: E402

from bestbefore.config import REPO_ROOT  # noqa: E402
from bestbefore.expdate import load_split  # noqa: E402
from bestbefore.images import load_upright  # noqa: E402
from bestbefore.samples import DATE_LABELS, Sample  # noqa: E402

OUT = REPO_ROOT / "docs" / "assets"

SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
HIGHLIGHT = "#eb6834"

# Picked by eye from the test split: one difficulty each.
DATES = [
    ("test_00415.jpg", "Dot-matrix print"),
    ("test_00324.jpg", "Curved cap"),
    ("test_00562.jpg", "Faint ink"),
    ("test_00484.jpg", "Out of focus"),
    ("test_00573.jpg", "Expiry above a production date"),
    ("test_00064.jpg", "Month as a word"),
    ("test_00445.jpg", "Curved jar"),
    ("test_00104.jpg", "Under a lot code"),
    ("test_00627.jpg", "Low contrast"),
    ("test_00498.jpg", "Dot-matrix, month first"),
    ("test_00143.jpg", "Clean"),
    ("test_00598.jpg", "Clean, bold"),
]
SMALL = ["test_00545.jpg", "test_00210.jpg", "test_00634.jpg"]


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def expiry_box(s: Sample) -> tuple[float, float, float, float]:
    return next(b.xyxy for b in s.boxes if b.label == "exp")


def crop_around(im: Image.Image, xyxy, size: tuple[int, int]) -> Image.Image:
    """A crop centred on the box, with margin, at the tile's aspect ratio."""
    W, H = im.size
    x1, y1, x2, y2 = xyxy[0] * W, xyxy[1] * H, xyxy[2] * W, xyxy[3] * H
    pad = 0.7 * (y2 - y1)
    w, h = (x2 - x1) + 2 * pad, (y2 - y1) + 2 * pad
    aspect = size[0] / size[1]
    w, h = (h * aspect, h) if w / h < aspect else (w, w / aspect)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = min(max(cx - w / 2, 0), max(W - w, 0))
    top = min(max(cy - h / 2, 0), max(H - h, 0))
    crop = im.crop((int(left), int(top), int(left + w), int(top + h)))
    return ImageOps.pad(crop, size, color=SURFACE)


def dates_mosaic(test: dict[str, Sample]) -> Image.Image:
    tile, caption, gap, cols = (440, 147), 40, 12, 4
    rows = -(-len(DATES) // cols)
    sheet = Image.new("RGB", (cols * tile[0] + (cols + 1) * gap,
                              rows * (tile[1] + caption) + (rows + 1) * gap), SURFACE)
    draw = ImageDraw.Draw(sheet)
    for i, (name, label) in enumerate(DATES):
        s = test[name]
        x = gap + (i % cols) * (tile[0] + gap)
        y = gap + (i // cols) * (tile[1] + caption + gap)
        sheet.paste(crop_around(load_upright(s.image), expiry_box(s), tile), (x, y))
        draw.text((x + 2, y + tile[1] + 9), label, fill=INK_2, font=font(22))
    return sheet


def small_dates(test: dict[str, Sample]) -> Image.Image:
    col_w, photo_h, zoom_h, caption, gap = 420, 560, 140, 44, 16
    sheet = Image.new("RGB", (len(SMALL) * col_w + (len(SMALL) + 1) * gap,
                              photo_h + zoom_h + caption + 3 * gap), SURFACE)
    draw = ImageDraw.Draw(sheet)
    for i, name in enumerate(SMALL):
        s = test[name]
        im = load_upright(s.image)
        xyxy = expiry_box(s)
        x = gap + i * (col_w + gap)

        photo = ImageOps.contain(im, (col_w, photo_h))
        pw, ph = photo.size
        d = ImageDraw.Draw(photo)
        bx = (xyxy[0] * pw - 6, xyxy[1] * ph - 6, xyxy[2] * pw + 6, xyxy[3] * ph + 6)
        d.rectangle(bx, outline=INK, width=6)
        d.rectangle(bx, outline=HIGHLIGHT, width=3)
        sheet.paste(photo, (x + (col_w - pw) // 2, gap + (photo_h - ph) // 2))

        sheet.paste(crop_around(im, xyxy, (col_w, zoom_h)), (x, 2 * gap + photo_h))
        area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]) * 100
        draw.text((x + 2, 2 * gap + photo_h + zoom_h + 10),
                  f"The date is {area:.2f}% of the photo", fill=INK_2, font=font(22))
    return sheet


def short_sides_at(samples: list[Sample], size: int = 640) -> np.ndarray:
    out = []
    for s in samples:
        scale = size / max(s.width, s.height)
        for b in s.boxes:
            if b.label in DATE_LABELS:
                w = (b.xyxy[2] - b.xyxy[0]) * s.width
                h = (b.xyxy[3] - b.xyxy[1]) * s.height
                out.append(min(w, h) * scale)
    return np.array(out)


THEMES = {
    "light": dict(surface=SURFACE, ink=INK, ink_2=INK_2, muted=MUTED,
                  grid="#e1e0d9", axis="#c3c2b7", series=("#2a78d6", "#eb6834")),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink_2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", series=("#3987e5", "#d95926")),
}


def size_chart(real: np.ndarray, synth: np.ndarray, theme: str) -> plt.Figure:
    t = THEMES[theme]
    bins = np.arange(0, 82, 2)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    medians = []
    for values, colour, label in ((real, t["series"][0], "Real photos"),
                                  (synth, t["series"][1], "Synthetic photos")):
        share = np.histogram(values, bins=bins)[0] / len(values) * 100
        ax.stairs(share, bins, color=colour, linewidth=2,
                  label=f"{label} (median {np.median(values):.0f} px)")
        medians.append(np.median(values))

    # Direct labels, placed in the clear space beside each curve.
    ax.text(19, 21, f"Real photos\nmedian {medians[0]:.0f} px",
            color=t["ink_2"], fontsize=9, va="top")
    ax.text(36, 5.6, f"Synthetic photos\nmedian {medians[1]:.0f} px",
            color=t["ink_2"], fontsize=9, va="bottom")

    ax.axvline(8, color=t["muted"], linewidth=1, linestyle=(0, (4, 3)))
    ax.text(7.3, 24, "hard to\ndetect", color=t["muted"], fontsize=8,
            ha="right", va="top")

    ax.set_xlim(0, 80)
    ax.set_ylim(0, 25)
    ax.set_xlabel("Date box height after resizing the photo to 640 px (pixels)",
                  color=t["ink_2"], fontsize=9)
    ax.set_ylabel("Share of date boxes (%)", color=t["ink_2"], fontsize=9)
    ax.tick_params(colors=t["muted"], labelsize=8.5, length=0)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    legend = ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    for text in legend.get_texts():
        text.set_color(t["ink_2"])

    fig.text(0.07, 0.95, "Dates are small in real photos", color=t["ink"],
             fontsize=12, fontweight="bold", va="top")
    fig.text(0.07, 0.89, f"ExpDate date boxes: {len(real):,} real, "
             f"{len(synth):,} synthetic", color=t["ink_2"], fontsize=9, va="top")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.14)
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    test = {s.id: s for s in load_split("real-test")}

    dates_mosaic(test).save(OUT / "expdate_dates.jpg", quality=86, optimize=True)
    small_dates(test).save(OUT / "expdate_small_dates.jpg", quality=86, optimize=True)

    real = short_sides_at(load_split("real-train") + list(test.values()))
    synth = short_sides_at(load_split("synth"))
    for theme in THEMES:
        fig = size_chart(real, synth, theme)
        fig.savefig(OUT / f"date_box_sizes_{theme}.png", facecolor=fig.get_facecolor())
        plt.close(fig)

    for p in sorted(OUT.iterdir()):
        print(f"{p.relative_to(REPO_ROOT)}  {p.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
