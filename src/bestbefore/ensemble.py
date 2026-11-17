"""The inference add-ons R4 weighs (docs/m2b-store-and-size.md, "Inference
add-ons", S13): several prediction passes over one photo, from the chosen
arm's other seeds (the ensemble), from the photo turned a quarter, a half
or three quarters (test-time rotation), or at 1,536 px instead of 1,280,
fused into one set of boxes by weighted boxes fusion.

Every pass predicts one photo at a time with rect=False, as every M2b
score does (detector.predict). A pass on a turned photo gets its boxes
turned back onto the upright photo, so every pass's boxes share one frame
and can be fused. A configuration of one pass is that pass's boxes as they
are: fusion only ever merges two passes or more.

The settings were fixed in S13's plan note before any add-on was measured:
ensemble-boxes' weighted_boxes_fusion with its defaults (IoU 0.55, the
average confidence, every pass weighing the same), boxes under
PREDICT_CONF skipped, one fusion over all of a configuration's passes,
class by class. With "avg", a box only some passes found keeps its mean
confidence times the share of passes that found it.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from PIL import Image

from . import derived, m2b
from .detection_eval import Prediction
from .detector import PREDICT_CONF
from .yolo import CLASSES

TURNS = (0, 1, 2, 3)          # quarter turns anticlockwise, as derived.turn
IMGSZ, IMGSZ_UP = 1280, 1536  # the arm's input size, and the add-on's
WBF_IOU = 0.55                # weighted_boxes_fusion's defaults
WBF_CONF = "avg"


@dataclass(frozen=True, order=True)
class Pass:
    """One prediction pass: a seed's weights on the photo turned `quarter`
    quarter turns anticlockwise, at `imgsz`."""
    seed: int
    quarter: int
    imgsz: int

    @property
    def name(self) -> str:
        return f"s{self.seed}-q{self.quarter}-{self.imgsz}"

    @classmethod
    def parse(cls, name: str) -> Pass:
        seed, quarter, imgsz = name.split("-")
        return cls(int(seed[1:]), int(quarter[1:]), int(imgsz))


def passes(addons: Sequence[str], seeds: Sequence[int] = m2b.SEEDS) -> tuple[Pass, ...]:
    """The passes a configuration fuses: seed 0 alone (R3's run), or every
    seed with the ensemble; upright alone, or all four turns with
    rotation; 1,280 px, or 1,536 with that add-on."""
    m2b.configuration(addons)                      # refuses an unknown add-on
    chosen = seeds if "ensemble" in addons else seeds[:1]
    quarters = TURNS if "rotation" in addons else (0,)
    imgsz = IMGSZ_UP if "imgsz-1536" in addons else IMGSZ
    return tuple(Pass(s, q, imgsz) for s in chosen for q in quarters)


def every_pass(seeds: Sequence[int] = m2b.SEEDS) -> tuple[Pass, ...]:
    """Every pass any configuration uses: 3 seeds x 4 turns x 2 sizes."""
    return tuple(Pass(s, q, size) for s in seeds for q in TURNS for size in (IMGSZ, IMGSZ_UP))


def turn_back(predictions: Sequence[Prediction], quarter: int) -> list[Prediction]:
    """Boxes predicted on a photo turned `quarter` quarter turns
    anticlockwise, as boxes on the photo itself: turned on by the rest of
    a whole turn, the inverse of derived.turn_box(·, quarter)."""
    back = (4 - quarter) % 4
    return [Prediction(p.label, derived.turn_box(p.xyxy, back), p.conf) for p in predictions]


def predict_turned(predict: Callable[[Image.Image], list[Prediction]], image: Image.Image,
                   quarter: int) -> list[Prediction]:
    """`predict` (an upright photo to its boxes, 0..1 of that photo) on the
    photo turned `quarter` quarter turns anticlockwise, its boxes turned
    back onto the photo. Turning by quarters moves pixels without
    resampling them (derived.turn)."""
    turned, _ = derived.turn(image, (), quarter)
    return turn_back(predict(turned), quarter)


def fuse(lists: Sequence[Sequence[Prediction]]) -> list[Prediction]:
    """One photo's boxes from several passes, merged class by class by
    weighted boxes fusion (ensemble-boxes, pinned in the `ensemble` extra),
    surest first. One pass is returned as it is."""
    if len(lists) == 1:
        return sorted(lists[0], key=lambda p: -p.conf)
    from ensemble_boxes import weighted_boxes_fusion

    boxes = [[list(p.xyxy) for p in ps] for ps in lists]
    scores = [[p.conf for p in ps] for ps in lists]
    labels = [[CLASSES.index(p.label) for p in ps] for ps in lists]
    fused, conf, cls = weighted_boxes_fusion(boxes, scores, labels, iou_thr=WBF_IOU,
                                             skip_box_thr=PREDICT_CONF, conf_type=WBF_CONF)
    return [Prediction(CLASSES[int(c)], tuple(float(v) for v in box), float(p))
            for box, p, c in zip(fused, conf, cls)]
