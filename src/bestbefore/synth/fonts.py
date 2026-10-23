"""The generator's fonts: 20 families from the Google Fonts repository,
pinned to one commit and checked by SHA-256 (docs/m2-synthetic-generator.md,
"Fonts"). scripts/fetch_fonts.py downloads them into $GENERATED_DIR/fonts/;
they are never committed.

16 families draw training text. The 4 held-out families draw only
synth-val's (S6), so synth-val can tell learning dates from learning fonts.

Every family is drawn in two instances, chosen on purpose: regular (weight
400) and bold (700). A variable font names every axis it has, because its
default instance is not always regular: Montserrat's is Thin (100), Source
Sans 3's and Source Code Pro's ExtraLight (200), Merriweather's Light (300).
"""
from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import ImageFont

from ..config import generated_dir

# google/fonts main on 2026-10-23. Every family below is in its ofl/ folder
# (SIL Open Font License 1.1).
COMMIT = "e7db6ca603fd4862010383ff17ffd6cc4ffac1c0"
RAW_URL = "https://raw.githubusercontent.com/google/fonts"
LICENCES = {"ofl": "Open Font License", "apache": "Apache License"}   # folder: name-table text
SPLITS = ("train", "held-out")
DIGITS = "0123456789"


@dataclass(frozen=True)
class Face:
    """One instance to draw with: a file and, for a variable font, the
    value of each of its axes."""
    file: str                                   # path in the Google Fonts repository
    axes: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class Family:
    name: str
    look: str               # sans, serif, mono, condensed or pixel (the plan's "style")
    split: str              # "train" or "held-out"
    regular: Face
    bold: Face | None       # None: the family has no bold

    @property
    def licence(self) -> str:
        """The repository's licence folder: ofl or apache."""
        return self.regular.file.split("/")[0]

    def face(self, bold: bool) -> Face:
        """The bold instance if asked and there is one, else the regular."""
        return self.bold if bold and self.bold else self.regular


def _variable(file: str, **other_axes: float) -> tuple[Face, Face]:
    """Regular and bold of a variable font: weight 400 and 700, every other
    axis at the value given (width 100 is the normal width)."""
    return tuple(Face(file, tuple(sorted({"wght": w, **other_axes}.items()))) for w in (400, 700))


def _static(regular: str, bold: str | None = None) -> tuple[Face, Face | None]:
    return Face(regular), Face(bold) if bold else None


FAMILIES = (
    Family("Roboto", "sans", "train", *_variable("ofl/roboto/Roboto[wdth,wght].ttf", wdth=100)),
    Family("Open Sans", "sans", "train", *_variable("ofl/opensans/OpenSans[wdth,wght].ttf", wdth=100)),
    Family("Lato", "sans", "train", *_static("ofl/lato/Lato-Regular.ttf", "ofl/lato/Lato-Bold.ttf")),
    Family("Montserrat", "sans", "train", *_variable("ofl/montserrat/Montserrat[wght].ttf")),
    Family("Source Sans 3", "sans", "train", *_variable("ofl/sourcesans3/SourceSans3[wght].ttf")),
    Family("Noto Sans", "sans", "train", *_variable("ofl/notosans/NotoSans[wdth,wght].ttf", wdth=100)),
    Family("Work Sans", "sans", "train", *_variable("ofl/worksans/WorkSans[wght].ttf")),
    Family("Oswald", "condensed", "train", *_variable("ofl/oswald/Oswald[wght].ttf")),
    Family("Roboto Condensed", "condensed", "train",
           *_variable("ofl/robotocondensed/RobotoCondensed[wght].ttf")),
    Family("Barlow Condensed", "condensed", "train",
           *_static("ofl/barlowcondensed/BarlowCondensed-Regular.ttf",
                    "ofl/barlowcondensed/BarlowCondensed-Bold.ttf")),
    Family("Noto Serif", "serif", "train", *_variable("ofl/notoserif/NotoSerif[wdth,wght].ttf", wdth=100)),
    # opsz 18: the design for text, the size dates are printed at.
    Family("Merriweather", "serif", "train",
           *_variable("ofl/merriweather/Merriweather[opsz,wdth,wght].ttf", opsz=18, wdth=100)),
    Family("Roboto Mono", "mono", "train", *_variable("ofl/robotomono/RobotoMono[wght].ttf")),
    Family("Source Code Pro", "mono", "train", *_variable("ofl/sourcecodepro/SourceCodePro[wght].ttf")),
    Family("Share Tech Mono", "mono", "train", *_static("ofl/sharetechmono/ShareTechMono-Regular.ttf")),
    Family("VT323", "pixel", "train", *_static("ofl/vt323/VT323-Regular.ttf")),
    # Held out: synth-val's text only.
    Family("IBM Plex Sans", "sans", "held-out",
           *_variable("ofl/ibmplexsans/IBMPlexSans[wdth,wght].ttf", wdth=100)),
    Family("Archivo Narrow", "condensed", "held-out",
           *_variable("ofl/archivonarrow/ArchivoNarrow[wght].ttf")),
    Family("PT Serif", "serif", "held-out",
           *_static("ofl/ptserif/PT_Serif-Web-Regular.ttf", "ofl/ptserif/PT_Serif-Web-Bold.ttf")),
    Family("IBM Plex Mono", "mono", "held-out",
           *_static("ofl/ibmplexmono/IBMPlexMono-Regular.ttf", "ofl/ibmplexmono/IBMPlexMono-Bold.ttf")),
)

# SHA-256 of every file at COMMIT, recorded in S4 after downloading.
SHA256 = {
    "ofl/roboto/Roboto[wdth,wght].ttf": "d7598e12c5dbef095ff8272cfc55da0250bd07fbdecbac8a530b9b277872a134",
    "ofl/opensans/OpenSans[wdth,wght].ttf": "36643644f318a812aab2d2ed3bb98f8cf0872527f835fe9398d95fe6b9adb878",
    "ofl/lato/Lato-Regular.ttf": "d636e4683231f931eda222d588e944d082bfd3bdba02f928bee461c0f185b251",
    "ofl/lato/Lato-Bold.ttf": "8a0aace75d33794eece4b28187bfc1df0bbd2888b5d8a56e01788c8d65d16be1",
    "ofl/montserrat/Montserrat[wght].ttf": "0f7b311b2f3279e4eef9b2f968bcdbab6e28f4daeb1f049f4f278a902bcd82f7",
    "ofl/sourcesans3/SourceSans3[wght].ttf": "042fe2cc0b933e328410d7acbd0aa6a1873dca5aef81875f4bc214b08825c7b9",
    "ofl/notosans/NotoSans[wdth,wght].ttf": "bfb7bb691513f12e734dc346c03a03f784912432d7e3fa8e56efcf906fe86b3d",
    "ofl/worksans/WorkSans[wght].ttf": "f50f61f2ba738e239442d40bf1069adb195c224b6a5a73a581fc2f3ed62a9f63",
    "ofl/oswald/Oswald[wght].ttf": "5b38c246e255a12f5712d640d56bcced0472466fc68983d2d0410ec0457c2817",
    "ofl/robotocondensed/RobotoCondensed[wght].ttf": "dace262afcee68a5276f200d8026c57221735c0118ab5fda8c2c0d3dc409a8d0",
    "ofl/barlowcondensed/BarlowCondensed-Regular.ttf": "583cec5da3b84bc4dc7c9c72e2a565c94d34e431518b19d7e250b7830ad5f996",
    "ofl/barlowcondensed/BarlowCondensed-Bold.ttf": "e476562ec9c1e16cf16475895b511f08c804f438cc9a9f80a44ea50a0eeb5b65",
    "ofl/notoserif/NotoSerif[wdth,wght].ttf": "4d8e6761424656867019081a1a01336f3cb086982682698714054fc33f782713",
    "ofl/merriweather/Merriweather[opsz,wdth,wght].ttf": "d0ed0e359e396af7ad05e73dffd11a3a4c326ea0d0283c56bd9361cb2cc86a96",
    "ofl/robotomono/RobotoMono[wght].ttf": "66a80e79d17e4c7cabd162e2916578a4cc08fd19eef6e2a643305eae9c567b2b",
    "ofl/sourcecodepro/SourceCodePro[wght].ttf": "b400fc584e10aff25d0e775ce181b4fc1c5ea1b5dc37b81aeb2084375b945790",
    "ofl/sharetechmono/ShareTechMono-Regular.ttf": "9ceab1f87414829af259c0f537573ae03ef7dd3147c0b27a36a1a0beb6732677",
    "ofl/vt323/VT323-Regular.ttf": "cf4de751ada78ceac033dbe16a687742939995b77bc2a052ae17a4957958594d",
    "ofl/ibmplexsans/IBMPlexSans[wdth,wght].ttf": "3b031aa4216174205bd8471f88a49b91f093169e9e87bd5262242bc5967fe2e3",
    "ofl/archivonarrow/ArchivoNarrow[wght].ttf": "adbe027f625c8393ae0f6e174e32e233dda485bc3eda5153ce428275394ef97f",
    "ofl/ptserif/PT_Serif-Web-Regular.ttf": "a4951fade06ff8f09b7673aa81ffb65a8cd409e24d3289a6dc670bc4dda2557a",
    "ofl/ptserif/PT_Serif-Web-Bold.ttf": "038ba7336bd7ea14f12ad155bed51a4345cac5153275d521dec3ba04021c526e",
    "ofl/ibmplexmono/IBMPlexMono-Regular.ttf": "6a3412f058c7d8dfd9170c41e85ade48e5156ecb89356110ca57a0a27734af46",
    "ofl/ibmplexmono/IBMPlexMono-Bold.ttf": "ac27abd6450a64dd94467580a02fe6235156d5b92f2926ebbc8e7489df64e0be",
}


def families(split: str) -> tuple[Family, ...]:
    """The families of one split: "train" for training text, "held-out" for synth-val's."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; one of {', '.join(SPLITS)}")
    return tuple(f for f in FAMILIES if f.split == split)


def random_family(rng: np.random.Generator, split: str) -> Family:
    """A family of the split, each equally likely: training text draws from
    "train" only, so the held-out families stay unseen."""
    found = families(split)
    return found[int(rng.integers(len(found)))]


def faces(family: Family) -> tuple[Face, ...]:
    return (family.regular,) + ((family.bold,) if family.bold else ())


def font_dir() -> Path:
    return generated_dir() / "fonts"


def path(face: Face) -> Path:
    return font_dir() / face.file


def fetched() -> bool:
    """Whether every font file is on disk; False on a machine with no
    $GENERATED_DIR, too."""
    try:
        return all(path(face).exists() for f in FAMILIES for face in faces(f))
    except RuntimeError:
        return False


def sha256(file: Path) -> str:
    with open(file, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


# --- Loading (each font once per size) ---------------------------------------

@functools.cache
def _axis_order(file: str) -> tuple[str, ...]:
    """A variable font's axis tags, in the order FreeType takes their values."""
    from fontTools.ttLib import TTFont
    return tuple(a.axisTag for a in TTFont(font_dir() / file, lazy=True)["fvar"].axes)


@functools.lru_cache(maxsize=512)
def load(face: Face, px: int) -> ImageFont.FreeTypeFont:
    """The face at `px` pixels per em. Basic layout (no shaping): the same
    on every machine, whether or not Pillow has libraqm."""
    font = ImageFont.truetype(str(path(face)), px, layout_engine=ImageFont.Layout.BASIC)
    if face.axes:
        values = dict(face.axes)
        font.set_variation_by_axes([values[tag] for tag in _axis_order(face.file)])
    return font


@functools.cache
def digit_band(face: Face) -> tuple[float, float]:
    """Where the face's digits reach, in em from the baseline (negative is
    up): the median top and bottom over the ten digits, so one taller or
    descending digit (Merriweather's 6 and 9) doesn't stretch it."""
    em = 1000
    font = load(face, em)
    tops, bottoms = zip(*(font.getbbox(d, anchor="ls")[1::2] for d in DIGITS))
    return float(np.median(tops)) / em, float(np.median(bottoms)) / em


def digit_height(face: Face) -> float:
    """The height of the face's digits, in em: what makes a date's size the
    same in every font."""
    top, bottom = digit_band(face)
    return bottom - top


# --- Checks (scripts/fetch_fonts.py) ----------------------------------------

def problems(face: Face, chars: frozenset[str]) -> list[str]:
    """What is wrong with a downloaded face, if anything: its hash, a
    character the text prints that it lacks, axes other than its own, or a
    licence other than its folder's."""
    from fontTools.ttLib import TTFont

    file = path(face)
    if not file.exists():
        return [f"{face.file}: not downloaded (python scripts/fetch_fonts.py)"]
    found = []
    if (actual := sha256(file)) != SHA256[face.file]:
        found.append(f"{face.file}: SHA-256 {actual} is not the recorded {SHA256[face.file]}")
    font = TTFont(file, lazy=True)
    cmap = font["cmap"].getBestCmap()
    if missing := "".join(sorted(c for c in chars if ord(c) not in cmap)):
        found.append(f"{face.file}: no glyph for {missing!r}")
    axes = {a.axisTag: (a.minValue, a.maxValue) for a in font["fvar"].axes} if "fvar" in font else {}
    if set(axes) != {tag for tag, _ in face.axes}:
        found.append(f"{face.file}: axes {sorted(axes)}, the instance names {[t for t, _ in face.axes]}")
    found += [f"{face.file}: {tag} {v} outside {axes[tag]}"
              for tag, v in face.axes if tag in axes and not axes[tag][0] <= v <= axes[tag][1]]
    # The name table's own words. Roboto Condensed's still say Apache 2.0,
    # though the repository now files it under OFL: either is allowed.
    licence = font["name"].getDebugName(13) or ""
    if not any(name in licence for name in LICENCES.values()):
        found.append(f"{face.file}: licence {licence[:60]!r} is neither OFL nor Apache 2.0")
    return found
