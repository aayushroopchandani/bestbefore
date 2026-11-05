"""The generator's shares and ranges, fixed by rule before any photo is made
(docs/m2-synthetic-generator.md, "The spec"). Never tuned on the store
photos, which nothing here was designed from.

S4 wrote the text and the print styles; S5 adds geometry, size, composing
and camera effects, and GenSpec, which holds them all. GenSpec's hash goes
into every manifest and training record, so each model names exactly the
data it saw. Numbers the plan left open are marked "fixed in S5". S6 adds
synth-val's strata, each an override of the training spec.

M2b adds gen-v2's parts (docs/m2b-store-and-size.md, "The generator, v2"):
V2Spec, in GenSpec.v2. It is None in M2's specs and left out of their hash
(spec_dict), so GEN still hashes 3fc14178... and synth-val's strata
ba77f1bd..., and every v2 feature draws its random numbers only when a
spec's v2 is set. S7 wrote its text and line spacing; S8 adds its print,
geometry and sizes.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace

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
    # Fixed in S6:
    single_role: str = "expiry"  # a single line's date; synth-val's is its "target"


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
    # Decided 2026-10-23 (owner), after S5's look check: no box round ink that barely shows.
    visible: float = 0.04             # a boxed token's ink must change the photo this much (RGB distance, as
    inks: int = 3                     #   luminance 0..1 for grey), or the block gets a fresh ink; skipped after 3
    # Fixed in S6:
    block_tries: int = 1              # blocks drawn for each one wanted, a fresh one each time one is skipped
    #                                   (synth-val: 10, so every photo has its target)


@dataclass(frozen=True)
class CameraSpec:
    """Effects on the whole finished photo, mild on purpose ("Camera
    effects"). Each is applied to its share of photos."""
    defocus: float = 0.40
    defocus_sigma: Range = (0.3, 1.2)           # Gaussian, pixels
    motion: float = 0.10
    motion_length: tuple[int, int] = (3, 7)     # a line kernel, pixels
    noise: float = 0.50
    noise_sigma: Range = (2.0, 6.0)             # Gaussian, of 255
    glare: float = 0.15
    glare_count: tuple[int, int] = (1, 2)
    glare_size: Range = (0.05, 0.25)            # the ellipse's long radius, of the photo's short side
    glare_near_block: float = 0.5               # centred near a block, else anywhere
    uneven: float = 0.30
    uneven_ramp: Range = (0.10, 0.25)           # brightness up and down by this much across the photo
    dim: float = 0.10
    dim_level: Range = (0.5, 0.8)               # brightness x this, plus noise
    cast: float = 0.30
    cast_shift: float = 0.08                    # white balance off by up to this per channel
    jpeg_quality: tuple[int, int] = (70, 95)    # every photo, as it is saved
    # Fixed in S5:
    glare_strength: Range = (0.4, 0.9)          # how far towards white at a glare's centre
    glare_aspect: Range = (0.4, 1.0)            # an ellipse's short radius over its long one
    uneven_radial: float = 0.5                  # of uneven light, a radial ramp; the rest linear


# --- gen-v2 (M2b) -----------------------------------------------------------------------------
# What gen-v2 adds to M2's generator, drawn only when a spec's v2 is set.
# The shares of its new block kinds sit in TextSpec.kinds (TEXT_V2), beside
# M2's own; these say how the new kinds and forms are drawn.

@dataclass(frozen=True)
class V2Text:
    """gen-v2's text (S7): a date glued to its word, two dates on one line,
    dense panels, import dates, barcode numbers and unit prices."""
    glued: float = 0.30                     # of words before a date: the date printed against it, no space
    pair_import: float = 0.50               # line-pairs whose second date is an import date, not the expiry
    separators: tuple[str, ...] = (" | ", "  ", " / ", ", ")    # between a line-pair's two pairs, as printed
    dense_lines: tuple[int, int] = (3, 5)   # a dense panel's lines beside its pair
    dense_import: float = 0.50              # dense panels with an import line among them
    # Left open by the plan, fixed in S7:
    import_months: tuple[int, int] = (1, 6)    # an import date: this many months after mfg, before the expiry
    currency: str = "Rs."                   # money in the new distractors: the rupee sign only if every
    #                                         font has it, and 4 of the 20 families don't (S7)


@dataclass(frozen=True)
class V2Style:
    """gen-v2's print: line spacing per kind (S7); nozzle rows and the laser
    window (S8). The window's share sits in StyleSpec.styles (STYLE_V2)."""
    line_spacing: tuple[tuple[str, Range], ...] = (("dense-panel", (1.0, 1.2)),)   # em; other kinds StyleSpec's
    # Nozzle rows (S8): a clogged print head leaves whole rows of dots out.
    nozzle: float = 0.15                    # dot-matrix blocks with rows left out
    nozzle_rows: tuple[int, ...] = (1, 2)   # how many, each equally likely: the same rows in every line
    # The laser window (S8): a dark printed rectangle with light strokes, slightly soft, a faint glow.
    window_colours: tuple[RGB, ...] = ((24, 24, 27), (30, 45, 110), (25, 70, 45), (90, 55, 30))
    #                                   near-black,   dark blue,     dark green,   dark brown
    window_dots: float = 0.30               # windows whose text is dotified
    # Left open by the plan, fixed in S8 (lengths in em, as the label's):
    window_pad: Range = (0.3, 0.8)          # the rectangle's margin around the text
    window_corner: Range = (0.0, 0.1)       # its corner radius
    window_soft: Range = (0.01, 0.03)       # sigma of the strokes' softness (M2's ink spread comes on top)
    window_glow: Range = (0.06, 0.15)       # the glow's strength, under render.INK: a glow alone is never ink
    window_glow_spread: Range = (0.10, 0.25)    # the glow's sigma


@dataclass(frozen=True)
class V2Geometry:
    """gen-v2's geometry (S8): blocks turned upside down. Their share sits in
    GeometrySpec.orientation (GEOMETRY_V2), beside M2's own."""
    upside_down_angle: float = 8.0          # 180 degrees, then within this either way, as sideways


@dataclass(frozen=True)
class V2Compose:
    """gen-v2's ink (S8): worn patches, rubbed or smeared over part of a
    date, and a laser window's light strokes."""
    worn: float = 0.10                      # of solid, dot-matrix and window blocks: a worn patch
    worn_styles: tuple[str, ...] = ("solid", "dot-matrix", "window")
    worn_width: Range = (1.0, 3.0)          # characters across (a character: the digits' height, as M2's density)
    worn_height: Range = (1.0, 2.0)         # lines tall ("up to two"; a line: the block's line pitch)
    worn_fade: Range = (0.60, 0.95)         # the ink fades by this much at its core
    # Left open by the plan, fixed in S8:
    worn_edge: Range = (0.15, 0.35)         # its soft edge, a share of its half width and half height
    window_ink: Range = (0.75, 1.0)         # a window's strokes: a grey of this luminance (light grey to white)


@dataclass(frozen=True)
class V2Size:
    """gen-v2's sizes (S8), like the store's (Cause 4): each block is drawn
    so its first date's box, its short side over the photo's long side as
    the store measures its dates, follows v1-train's dates: one of the ten
    bands between their deciles, uniformly, then log-uniform inside it.
    Replaces M2's two bands (SizeSpec), which set the digits' height."""
    # v1-train's 169 dates, v3's training part included (scripts/store_sizes.py, 2026-11-05): aggregates
    # only, the two ends clipped to M2's floor and ceiling, 0.6% (8 px at 1,280 px input) and 4.5%.
    deciles: tuple[float, ...] = (0.00659, 0.01118, 0.01296, 0.01451, 0.01573, 0.01736, 0.01974, 0.02181,
                                  0.02559, 0.03247, 0.045)
    provisional: float = 1.3        # a first drawing's digits: the target over this (M2 S5's upright ratio)
    # Changed 2026-11-05 (S8, owner's call): a box off by more than this after its first correction is
    # corrected once more from the same draws, then drawn afresh, ComposeSpec.redraws tries in all.
    corrections: int = 2            # drawings from the same draws that correct its size, the second only while off
    tolerance_px: float = 1.0       # a first date's box within this many pixels of its target,
    tolerance: float = 0.05         #   or within this share of it, fits
    small: float = 0.015            # a target under this is small print: a background of at least min_side_small


@dataclass(frozen=True)
class V2Spec:
    """Everything gen-v2 adds to M2's spec, by the part of GenSpec it
    extends: its text (S7), print, geometry, ink and sizes (S8)."""
    text: V2Text = V2Text()
    style: V2Style = V2Style()
    geometry: V2Geometry = V2Geometry()
    compose: V2Compose = V2Compose()
    size: V2Size = V2Size()


@dataclass(frozen=True)
class GenSpec:
    """Everything a generated set is drawn with. synth-val's strata (S6)
    are overrides of it."""
    text: TextSpec = TextSpec()
    style: StyleSpec = StyleSpec()
    geometry: GeometrySpec = GeometrySpec()
    size: SizeSpec = SizeSpec()
    compose: ComposeSpec = ComposeSpec()
    camera: CameraSpec = CameraSpec()
    fonts: str = "train"            # fonts.random_family's split; synth-val's: "held-out"
    backgrounds: Shares = (("train_real", 0.5), ("train_synth_clean", 0.5))   # m2 lists
    v2: V2Spec | None = None        # gen-v2's additions (M2b); None in M2's specs, and then left out of the hash


TEXT = TextSpec()
STYLE = StyleSpec()
GEOMETRY = GeometrySpec()
SIZE = SizeSpec()
COMPOSE = ComposeSpec()
CAMERA = CameraSpec()
GEN = GenSpec()

# gen-v2's block kinds (M2b): each new kind about 10%, taken from single,
# pair and panel blocks; range and relative as M2's.
TEXT_V2 = replace(TEXT, kinds=(("single", 0.25), ("pair", 0.30), ("panel", 0.15), ("dense-panel", 0.10),
                               ("line-pair", 0.10), ("range", 0.05), ("relative", 0.05)))
# gen-v2's print and geometry (S8): the window takes its 10% from solid
# print, upside-down blocks theirs from upright ones; dot-matrix keeps 35%.
STYLE_V2 = replace(STYLE, styles=(("solid", 0.30), ("dot-matrix", 0.35), ("label", 0.15), ("embossed", 0.10),
                                  ("window", 0.10)))
GEOMETRY_V2 = replace(GEOMETRY, orientation=(("upright", 0.45), ("tilted", 0.15), ("sideways", 0.30),
                                             ("upside-down", 0.10)))
V2 = V2Spec()
# gen-v2 (M2b S8): M2's generator with every gap filled, its sizes like the
# store's. Everything it doesn't name is M2's spec.
GEN_V2 = replace(GEN, text=TEXT_V2, style=STYLE_V2, geometry=GEOMETRY_V2, v2=V2)

# Photo i of a set is drawn from numpy.random.default_rng([seed, i]).
# gen-v2-sample is M2b S8's 500 photos to look at (gen-v2 and synth-val-v2,
# 4 and 5, are S9's).
SET_SEEDS = {"gen": 1, "synth-val": 2, "gen-sample": 3, "gen-v2-sample": 6}


# --- synth-val's strata (S6) ------------------------------------------------------------------

def only(shares: Shares, *names: str) -> Shares:
    """The named entries of a share table, in their training proportions."""
    kept = [(name, share) for name, share in shares if name in names]
    total = sum(share for _, share in kept)
    return tuple((name, share / total) for name, share in kept)


# Every synth-val photo: one block, a single line on a validation photo in a
# held-out font, its one date the target. A skipped block is replaced by a
# fresh one, so no photo goes without its target.
_TARGET = replace(GEN, text=replace(TEXT, kinds=(("single", 1.0),), single_role="target"),
                  compose=replace(COMPOSE, two_blocks=0.0, block_tries=10),
                  fonts="held-out", backgrounds=(("val", 1.0),))
# clean: upright, numbers, normal size, solid or label style, flat. What the
# plan doesn't name follows the training spec, as its camera effects do.
CLEAN = replace(_TARGET, text=replace(_TARGET.text, month_word=0.0),
                style=replace(STYLE, styles=only(STYLE.styles, "solid", "label")),
                geometry=replace(GEOMETRY, orientation=(("upright", 1.0),), surface=(("flat", 1.0),)),
                size=replace(SIZE, bands=(("normal", 1.0),)))
# Each stratum changes one thing from clean; mixed is the training spec.
SYNTH_VAL = (
    ("clean", CLEAN),
    ("sideways", replace(CLEAN, geometry=replace(CLEAN.geometry, orientation=(("sideways", 1.0),)))),
    ("month-word", replace(CLEAN, text=replace(CLEAN.text, month_word=1.0))),
    ("small", replace(CLEAN, size=replace(CLEAN.size, bands=(("small", 1.0),)))),
    ("dot-matrix", replace(CLEAN, style=replace(CLEAN.style, styles=(("dot-matrix", 1.0),)))),
    ("curved", replace(CLEAN, geometry=replace(CLEAN.geometry, surface=only(GEOMETRY.surface, "cylinder", "bend")))),
    ("mixed", _TARGET),
)


def _sha256(data) -> str:
    text = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def spec_dict(spec: GenSpec) -> dict:
    """The spec as data, for its hash and the manifests. A spec without v2
    (M2's) leaves the field out, so it hashes and reads as it did before
    the field existed."""
    data = asdict(spec)
    if data["v2"] is None:
        del data["v2"]
    return data


def spec_hash(spec: GenSpec) -> str:
    """SHA-256 of the spec as JSON: what a manifest and a training record
    name, so each model says exactly what it saw."""
    return _sha256(spec_dict(spec))


def strata_hash(strata: tuple[tuple[str, GenSpec], ...] = SYNTH_VAL) -> str:
    """SHA-256 of synth-val's strata, in order, as JSON."""
    return _sha256([[name, spec_dict(spec)] for name, spec in strata])
