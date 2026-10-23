"""The generator's shares and ranges, fixed by rule before any photo is made
(docs/m2-synthetic-generator.md, "The spec"). Never tuned on the store
photos, which nothing here was designed from.

S4 wrote the text and the print styles; S5 adds geometry, size, composing
and camera effects, and GenSpec, which holds them all. GenSpec's hash goes
into every manifest and training record, so each model names exactly the
data it saw. Numbers the plan left open are marked "fixed in S5".
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

Shares = tuple[tuple[str, float], ...]     # (name, share) pairs that sum to 1
Range = tuple[float, float]                # drawn uniformly
RGB = tuple[int, int, int]


@dataclass(frozen=True)
class TextSpec:
    """What is printed ("Text")."""
    years: tuple[int, int] = (2018, 2035)       # every date falls in these years
    shelf_months: tuple[int, int] = (6, 36)     # two dates: expiry = mfg + this many months
    kinds: Shares = (("single", 0.35), ("pair", 0.35), ("panel", 0.20),
                     ("range", 0.05), ("relative", 0.05))
    month_word: float = 0.45     # of dates outside ranges; the rest are numbers
    full_name: float = 0.50      # of month words; the rest are abbreviations
    day_numbers: float = 0.70    # numeric dates that print a day
    day_words: float = 0.40      # month-word dates that print a day
    ordinal: float = 0.50        # of month-word dates with a day: 19th
    case: Shares = (("upper", 0.50), ("title", 0.45), ("lower", 0.05))   # month words
    single_word: float = 0.70    # a single date with its expiry word
    mfg_first: float = 0.80      # an mfg line printed above the expiry's
    panel_lines: tuple[int, int] = (1, 3)    # a panel's extra lines: codes and distractors
    # Left open by the plan, fixed in S4:
    panel_codes: float = 0.50    # of a panel's extra lines, codes; the rest distractors
    range_word: float = 0.70     # a range with an expiry word, as a single date
    relative_months: tuple[int, ...] = (6, 9, 12, 18, 24, 36)   # Best Before N months from Mfg.


@dataclass(frozen=True)
class StyleSpec:
    """How it is printed ("Print styles")."""
    styles: Shares = (("solid", 0.40), ("dot-matrix", 0.35), ("label", 0.15),
                      ("embossed", 0.10))
    supersample: int = 4                  # drawn at 4x, then shrunk with an area filter
    line_spacing: Range = (1.2, 1.6)      # baseline to baseline, in em
    # dotify (dot-matrix and inkjet)
    dot_rows: tuple[int, ...] = (7, 9)       # dots across a digit's height (at 5 they garble: S4)
    dot_aspect: Range = (0.8, 1.2)        # horizontal pitch / vertical pitch
    dot_cover: float = 0.40               # a dot where the text covers this much of its cell
    dot_radius: Range = (0.30, 0.48)      # x vertical pitch
    dot_jitter: float = 0.06              # sigma of a dot's position, x vertical pitch
    dot_drop: Range = (0.02, 0.08)        # share of dots left out
    dot_blur: Range = (0.15, 0.35)        # sigma, x vertical pitch: the inkjet look
    # Left open by the plan, fixed in S4:
    bold: float = 0.50                    # solid and label text in bold (dotify and emboss: always)
    letter_spacing: Range = (0.0, 0.12)   # extra space after every character, em
    label_pad: Range = (0.3, 0.6)         # the sticker's margin around the text, em
    label_corner: Range = (0.0, 0.3)      # the sticker's corner radius, em
    label_colours: tuple[tuple[int, int, int], ...] = ((250, 250, 246), (240, 232, 208))  # white, cream
    emboss_soft: Range = (0.02, 0.05)     # sigma of the relief's slopes, em
    emboss_strength: Range = (0.5, 0.9)   # alpha where the relief is steepest


@dataclass(frozen=True)
class GeometrySpec:
    """How a block is turned and bent ("Geometry"). Angles in degrees."""
    orientation: Shares = (("upright", 0.55), ("tilted", 0.15), ("sideways", 0.30))
    upright_angle: float = 5.0                  # upright: within this either way
    tilt_angle: Range = (5.0, 35.0)             # tilted: this far, either way
    sideways_angle: float = 8.0                 # sideways: 90 or 270, then within this either way
    surface: Shares = (("flat", 0.75), ("cylinder", 0.15), ("bend", 0.10))
    perspective: float = 0.40                   # blocks seen at an angle
    # Fixed in S5:
    cylinder_arc: Range = (40.0, 100.0)         # how much of a can's round the block's width covers
    cylinder_edge: float = 70.0                 # neither end turns further than this from the camera
    bend_arc: Range = (20.0, 70.0)              # how much of a circle the bent lines follow
    perspective_tilt: Range = (10.0, 30.0)      # the block's plane turned from facing the camera
    perspective_distance: Range = (2.0, 5.0)    # the camera's distance, in block diagonals


@dataclass(frozen=True)
class SizeSpec:
    """How tall a block's digits print, over the photo's long side ("Size"),
    drawn log-uniformly within a band."""
    bands: Shares = (("normal", 0.70), ("small", 0.30))
    normal: Range = (0.015, 0.045)
    small: Range = (0.006, 0.015)


@dataclass(frozen=True)
class ComposeSpec:
    """Backgrounds, a free place, the ink and how it is laid on
    ("Placement and blending")."""
    two_blocks: float = 0.40          # photos with 2 blocks; the rest have 1
    min_side: int = 640               # a background's long side, at least
    min_side_small: int = 1000        # ... when a block is small print
    max_side: int = 1600              # backgrounds are shrunk to at most this
    candidates: int = 20              # free places to choose from
    tries: int = 30                   # places tried before the block shrinks
    plainest: float = 0.70            # take the plainest free place; else a random one
    shrink: float = 0.80              # a block with no free place shrinks once, then is skipped
    margin: float = 0.5               # clear of other boxes and blocks by this many lines
    dark_inks: tuple[RGB, ...] = ((24, 24, 27), (30, 45, 110), (90, 55, 30), (120, 25, 32))
    #                              black,        dark blue,     brown,        dark red   (RGB fixed in S5)
    light_inks: tuple[RGB, ...] = ((246, 246, 242), (238, 228, 196), (196, 199, 205), (242, 208, 64))
    #                               white,           cream,           silver,          yellow
    colourful: float = 0.15           # a saturated ink instead
    low_contrast: float = 0.20        # ink luminance only `low_contrast_gap` from the surface's
    low_contrast_gap: Range = (0.08, 0.15)    # in luminance 0..1, as blended
    opacity: Range = (0.75, 1.0)
    # Fixed in S5:
    light_surface: float = 0.5        # a surface at least this light (mean luminance 0..1) takes dark ink
    colourful_saturation: Range = (0.6, 1.0)
    colourful_dark: Range = (0.35, 0.70)      # HSV value of a colourful ink on a light surface
    colourful_light: Range = (0.75, 1.0)      # ... on a dark one
    shading_window: float = 0.05      # the light on the surface: its brightest within this share of the long side,
    shading_sigma: float = 0.02       #   then blurred over this share (dark print narrower than the window is print)
    shading_floor: float = 0.6        # light ink and stickers darken with the light down to this share
    density: Range = (0.0, 0.4)       # uneven density: up to this much lighter in places, about a character apart
    spread: Range = (0.0, 0.4)        # sigma in final pixels; solid and dot-matrix (a label stays crisp)
    scratch: float = 0.10             # inked blocks with a scratch through one boxed token
    scratch_width: Range = (0.5, 1.0)     # final pixels
    redraws: int = 3                  # draws of a block before it is skipped for a boxed token with no ink


@dataclass(frozen=True)
class GenSpec:
    """Everything a generated set is drawn with. synth-val's strata (S6)
    are overrides of it."""
    text: TextSpec = TextSpec()
    style: StyleSpec = StyleSpec()
    geometry: GeometrySpec = GeometrySpec()
    size: SizeSpec = SizeSpec()
    compose: ComposeSpec = ComposeSpec()
    fonts: str = "train"            # fonts.random_family's split; synth-val's: "held-out"
    backgrounds: Shares = (("train_real", 0.5), ("train_synth_clean", 0.5))   # m2 lists


TEXT = TextSpec()
STYLE = StyleSpec()
GEOMETRY = GeometrySpec()
SIZE = SizeSpec()
COMPOSE = ComposeSpec()
GEN = GenSpec()

# Photo i of a set is drawn from numpy.random.default_rng([seed, i]).
SET_SEEDS = {"gen": 1, "synth-val": 2, "gen-sample": 3}


def spec_hash(spec: GenSpec) -> str:
    """SHA-256 of the spec as JSON: what a manifest and a training record
    name, so each model says exactly what it saw."""
    text = json.dumps(asdict(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()
