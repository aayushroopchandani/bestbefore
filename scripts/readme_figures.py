"""Make the README's figures into docs/assets/: ExpDate's, and M2b's charts.

    python scripts/readme_figures.py                 # every figure
    python scripts/readme_figures.py --only m2b      # M2b's two charts alone (S15)

  expdate_dates.jpg         12 labelled expiry dates from the ExpDate test
                            split, each showing a different difficulty
  expdate_small_dates.jpg   3 whole test photos with the expiry boxed, and
                            the same date zoomed in
  date_box_sizes_{light,dark}.png
                            how big date boxes are at YOLO's 640 px input,
                            real vs synthetic (one image per README theme)
  m2b_v2_by_tag_{light,dark}.png
                            store set v2's expiry dates found by failure tag,
                            M1, M2 and the M2b detector's arm (3 seeds each)
  m2b_v2_found_precision_{light,dark}.png
                            store set v2's expiry dates found against
                            precision, one point per arm

ExpDate's figures use only ExpDate photos (CC BY 4.0, credited in the
README). M2b's charts are drawn from the run records alone
(bestbefore.m2b_results.chart_data), counts only; each prints its alt
text, which the README carries. Store photos never go into a figure.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from PIL import Image, ImageDraw, ImageFont, ImageOps  # noqa: E402

from bestbefore import m2b_results  # noqa: E402
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


# --- M2b's charts: store set v2, from the run records (docs/m2b-store-and-size.md, S15) -----

# Each milestone's colour: one hue ordered by milestone, the later the further from the
# surface (an ordinal ramp, checked with the dataviz skill's validate_palette.js --ordinal
# in both modes: monotone lightness, steps at least 0.06 apart, the near end at 2.06:1 on
# the light surface and 2.15:1 on the dark one, so every bar is direct-labelled too).
ORDINAL = {"light": ("#86b6ef", "#3987e5", "#184f95"), "dark": ("#184f95", "#3987e5", "#86b6ef")}
MILESTONES = tuple(label for label, _ in m2b_results.CHART_MODELS)       # M1, M2, M2b
TAG_NAMES = {"every expiry": "Every expiry", "sideways": "Sideways", "dot-matrix": "Dot-matrix",
             "small-print": "Small print", "clean": "Clean", "curved": "Curved"}
# Chart (b)'s points: each arm's milestone colour, and where its label sits (points from the
# marker, the label's alignment).
POINT_STYLE = {"M1 (n640)": (0, (0, -20), "center"),
               "M2 at its threshold": (1, (-28, 0), "right"),
               "M2's weights at the store threshold": (1, (-12, -12), "right"),
               "n1280-ft": (2, (9, -20), "left"),
               "s1280-ft": (2, (12, 0), "left"),
               "m1280-ft": (2, (-28, 0), "right")}
POINT_NAMES = {"n1280-ft": "n1280-ft,\nthe M2b detector's arm"}


def count(entry: dict) -> str:
    """'64/68': a mean count over the seeds, as the report's cells give it."""
    return f"{round(entry['mean'], 1):g}/{entry['n']}"


def header(fig: plt.Figure, t: dict, title: str, subtitle: str) -> None:
    fig.text(0.04, 0.965, title, color=t["ink"], fontsize=12, fontweight="bold", va="top")
    fig.text(0.04, 0.915, subtitle, color=t["ink_2"], fontsize=8.5, va="top")


def milestone_legend(fig: plt.Figure, t: dict, colours: tuple[str, ...], marker: str | None = None) -> None:
    """The three milestones' colours in one row under the subtitle."""
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=7, markerfacecolor=c,
                      markeredgecolor=t["surface"]) if marker else
               Patch(facecolor=c, edgecolor="none") for c in colours]
    legend = fig.legend(handles, MILESTONES, loc="upper left", bbox_to_anchor=(0.035, 0.89), ncol=3,
                        frameon=False, fontsize=8.5, handlelength=1.2, columnspacing=1.6)
    for text in legend.get_texts():
        text.set_color(t["ink_2"])


def rounded_hbar(ax: plt.Axes, y: float, length: float, thickness: float, colour: str,
                 radius_px: float = 4.0) -> None:
    """A horizontal bar from 0 to `length`, square at the baseline, its data
    end rounded by radius_px pixels (drawn once the axes' limits are set)."""
    if length <= 0:
        return
    (x0, y0), (x1, y1) = ax.transData.transform([(0, 0), (1, 1)])
    rx = min(radius_px / abs(x1 - x0), length)
    ry = min(radius_px / abs(y1 - y0), thickness / 2)
    lo, hi, k = y - thickness / 2, y + thickness / 2, 0.5523         # k: a cubic Bezier's quarter circle
    verts = [(0, lo), (length - rx, lo), (length - rx + k * rx, lo), (length, lo + ry - k * ry), (length, lo + ry),
             (length, hi - ry), (length, hi - ry + k * ry), (length - rx + k * rx, hi), (length - rx, hi),
             (0, hi), (0, lo)]
    codes = [MPath.MOVETO, MPath.LINETO, *[MPath.CURVE4] * 3, MPath.LINETO, *[MPath.CURVE4] * 3, MPath.LINETO,
             MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=colour, edgecolor="none", zorder=2))


def by_tag_chart(data: dict, theme: str) -> plt.Figure:
    """(a) Store set v2's expiry dates found by failure tag: for each tag of
    three or more expiries, M1's, M2's and the M2b detector's arm, the share
    found (the bar, the mean of 3 seeds), the seed range (the whisker) and
    the count at the tip."""
    t, colours = THEMES[theme], ORDINAL[theme]
    tags = list(data["tags"])
    pitch, thickness, group = 1.0, 0.86, 3.9          # a bar every 1.0, groups 3.9 apart
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    fig.subplots_adjust(left=0.2, right=0.97, top=0.79, bottom=0.12)
    ax.set_xlim(0, 121)
    ax.set_ylim(group * (len(tags) - 1) + 2 * pitch + 0.75, -0.75)
    centres = []
    for g, tag in enumerate(tags):
        for i, name in enumerate(MILESTONES):
            y = g * group + i * pitch
            entry = data["tags"][tag][name]
            share = {k: 100 * entry[k] / entry["n"] for k in ("mean", "low", "high")}
            rounded_hbar(ax, y, share["mean"], thickness, colours[i])
            if share["low"] != share["high"]:
                ax.plot([share["low"], share["high"]], [y, y], color=t["ink"], linewidth=1, zorder=3,
                        solid_capstyle="butt")
                for x in (share["low"], share["high"]):
                    ax.plot([x, x], [y - 0.2, y + 0.2], color=t["ink"], linewidth=1, zorder=3)
            ax.annotate(count(entry), (max(share["mean"], share["high"]), y), xytext=(5, 0),
                        textcoords="offset points", va="center", ha="left", fontsize=8, color=t["ink_2"])
        centres.append(g * group + pitch)
    ax.set_yticks(centres, [f"{TAG_NAMES[tag]}\n{data['tags'][tag][MILESTONES[0]]['n']} expiries" for tag in tags])
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(colors=t["muted"], labelsize=8.5, length=0)
    for label in ax.get_yticklabels():
        label.set_color(t["ink_2"])
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(t["axis"])
    ax.set_xlabel("Share of the tag's expiry dates found, each run at its own threshold", color=t["ink_2"],
                  fontsize=8.5)
    header(fig, t, "Store set v2: expiry dates found, by failure tag",
           "Each milestone's detector arm: the bar is the mean of 3 seeds, the whisker the lowest to the highest "
           "seed")
    milestone_legend(fig, t, colours)
    listed = "; ".join(f"{TAG_NAMES.get(tag, tag)}: one or two expiries, not charted ("
                       + ", ".join(f"{name.split(' ')[0]} {cell}" for name, cell in cells.items()) + ")"
                       for tag, cells in data["listed"].items())
    if listed:
        fig.text(0.04, 0.025, listed + ".", color=t["muted"], fontsize=8, va="bottom")
    return fig


def found_precision_chart(data: dict, theme: str) -> plt.Figure:
    """(b) Store set v2's expiry dates found against precision: one point per
    arm (the mean of 3 seeds) with its seed range on both axes, coloured by
    milestone and named beside it; arrows from M2's threshold to the store
    threshold (the same weights) and on to fine-tuning."""
    t, colours = THEMES[theme], ORDINAL[theme]
    points = data["points"]
    n = next(iter(points.values()))["found"]["n"]
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    fig.subplots_adjust(left=0.1, right=0.97, top=0.79, bottom=0.12)
    ax.set_xlim(14, n + 11)                           # room right of the last tick for a label
    ax.set_ylim(0.38, 1.02)
    halo = [pe.withStroke(linewidth=3.2, foreground=t["surface"])]
    centre = {}
    for name, p in points.items():
        k, offset, align = POINT_STYLE[name]
        x, y = p["found"]["mean"], p["precision"]["mean"]
        centre[name] = (x, y)
        for xs, ys in (((p["found"]["low"], p["found"]["high"]), (y, y)),
                       ((x, x), (p["precision"]["low"], p["precision"]["high"]))):
            ax.plot(xs, ys, color=colours[k], linewidth=1.6, zorder=3, solid_capstyle="round", path_effects=halo)
        ax.plot([x], [y], marker="o", markersize=8, markerfacecolor=colours[k], markeredgecolor=t["surface"],
                markeredgewidth=1.6, zorder=4)
        label = f"{POINT_NAMES.get(name, name)}\n{count(p['found'])}, precision {p['precision']['mean']:.3f}"
        ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", ha=align, va="center", fontsize=8,
                    color=t["ink_2"], zorder=5, path_effects=halo, linespacing=1.25, multialignment=align)
    for start, end, words, side in (("M2 at its threshold", "M2's weights at the store threshold", "threshold only",
                                     (-12, 0)),
                                    ("M2's weights at the store threshold", "n1280-ft", "fine-tuning", (8, -6))):
        ax.annotate("", centre[end], xytext=centre[start], zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=t["muted"], linewidth=1, shrinkA=8, shrinkB=8,
                                    mutation_scale=9))
        mid = tuple((a + b) / 2 for a, b in zip(centre[start], centre[end]))
        ax.annotate(words, mid, xytext=side, textcoords="offset points", ha="right" if side[0] < 0 else "left",
                    va="center", fontsize=8,
                    color=t["muted"], fontstyle="italic", path_effects=halo)
    ax.set_xticks([20, 30, 40, 50, 60, n], ["20", "30", "40", "50", "60", f"{n} (all)"])
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ["0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
    ax.tick_params(colors=t["muted"], labelsize=8.5, length=0)
    ax.grid(color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
    ax.set_xlabel(f"Expiry dates found, of {n}", color=t["ink_2"], fontsize=8.5)
    ax.set_ylabel("Precision: boxes kept that are dates", color=t["ink_2"], fontsize=8.5)
    header(fig, t, "Store set v2: expiry dates found against precision",
           "Each arm at its own threshold: the dot is the mean of 3 seeds, the whiskers the lowest to the highest "
           "seed")
    milestone_legend(fig, t, colours, marker="o")
    return fig


def alt_texts(data: dict) -> dict[str, str]:
    """Each chart's alt text, holding its key numbers, from the same data."""
    names = {MILESTONES[0]: "M1", MILESTONES[1]: "M2", MILESTONES[2]: "M2b"}
    tags = "; ".join(f"{TAG_NAMES[tag].lower()} " + ", ".join(
        f"{names[m]} {count(entry)}" for m, entry in data["tags"][tag].items()) for tag in data["tags"])
    listed = "; ".join(f"the {TAG_NAMES.get(tag, tag).lower()} group has one or two expiries and isn't charted ("
                       + ", ".join(f"{names[m]} {cell}" for m, cell in cells.items()) + ")"
                       for tag, cells in data["listed"].items())
    points = "; ".join(f"{POINT_NAMES.get(name, name).replace(chr(10), ' ')} {count(p['found'])} at "
                       f"{p['precision']['mean']:.3f}"
                       for name, p in data["points"].items())
    return {
        "m2b_v2_by_tag": ("Horizontal bars of store set v2's expiry dates found by failure tag, mean of 3 seeds for "
                          "M1 (n640), M2 (gen-rot-1280) and M2b (n1280-ft), each at its own threshold: "
                          f"{tags}." + (f" {listed[0].upper()}{listed[1:]}." if listed else "")),
        "m2b_v2_found_precision": ("Scatter of store set v2's expiry dates found against precision, one point per "
                                   f"arm, mean of 3 seeds: {points}. Lowering M2's threshold alone moves right and "
                                   "down; fine-tuning moves right and back up."),
    }


def m2b_charts() -> None:
    data = m2b_results.chart_data()
    if data is None:
        print("M2b's charts: not drawn, store set v2's records aren't all here")
        return
    for theme in THEMES:
        for stem, chart in (("m2b_v2_by_tag", by_tag_chart), ("m2b_v2_found_precision", found_precision_chart)):
            fig = chart(data, theme)
            fig.savefig(OUT / f"{stem}_{theme}.png", facecolor=fig.get_facecolor())
            plt.close(fig)
    for stem, text in alt_texts(data).items():
        print(f"{stem} alt text: {text}")


def expdate_figures() -> None:
    test = {s.id: s for s in load_split("real-test")}

    dates_mosaic(test).save(OUT / "expdate_dates.jpg", quality=86, optimize=True)
    small_dates(test).save(OUT / "expdate_small_dates.jpg", quality=86, optimize=True)

    real = short_sides_at(load_split("real-train") + list(test.values()))
    synth = short_sides_at(load_split("synth"))
    for theme in THEMES:
        fig = size_chart(real, synth, theme)
        fig.savefig(OUT / f"date_box_sizes_{theme}.png", facecolor=fig.get_facecolor())
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", choices=("expdate", "m2b"), help="make only these figures")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "expdate"):
        expdate_figures()
    if args.only in (None, "m2b"):
        m2b_charts()
    for p in sorted(OUT.iterdir()):
        print(f"{p.relative_to(REPO_ROOT)}  {p.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
