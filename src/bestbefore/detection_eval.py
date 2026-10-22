"""Did the detector find the dates? Scores predictions against `Sample`s.

The same code scores ExpDate validation and test and the store photos, so
"found" means the same thing everywhere: a predicted `date` box overlaps a
labelled date with IoU at or above a threshold (0.5 headline, 0.3 "roughly
found").

Matching is greedy and one-to-one, in order of confidence: each
prediction takes the unmatched labelled date it overlaps most, so one box
can never count for two dates. Because a prediction's match depends only
on the predictions more confident than it, matching once at the lowest
confidence gives the right answer for every higher threshold too. That
makes scoring at any threshold, and finding the best one, a single pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .samples import DATE_LABELS, Sample

EXPIRY_LABELS = {"exp", "expiry"}   # ExpDate test / store photos


@dataclass(frozen=True)
class Prediction:
    label: str                                  # date, due, prod or code
    xyxy: tuple[float, float, float, float]     # 0..1 of the upright image
    conf: float


def iou(a, b) -> float:
    """Intersection over union of two xyxy boxes (0..1 coords are fine:
    scaling both axes scales every area alike, so the ratio is unchanged)."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    overlap = w * h
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
             - overlap)
    return overlap / union


def match_photo(preds: list[Prediction], truths: list, iou_min: float):
    """Greedy matching for one photo.

    Returns (pred_tp, truth_conf): for each prediction (by descending
    confidence) whether it matched a labelled date, and for each labelled
    date the confidence of the prediction that matched it, or None.
    """
    order = sorted(preds, key=lambda p: -p.conf)
    taken: list[float | None] = [None] * len(truths)
    pred_tp = []
    for p in order:
        best, best_iou = None, iou_min
        for j, t in enumerate(truths):
            if taken[j] is None:
                v = iou(p.xyxy, t)
                if v >= best_iou:
                    best, best_iou = j, v
        if best is not None:
            taken[best] = p.conf
        pred_tp.append((p.conf, best is not None))
    return pred_tp, taken


@dataclass
class Matches:
    """Every prediction and every labelled date of a set, matched once."""
    iou_min: float
    n_photos: dict[str | None, int] = field(default_factory=dict)  # by tag
    pred_conf: list[float] = field(default_factory=list)
    pred_tp: list[bool] = field(default_factory=list)
    pred_tag: list[str | None] = field(default_factory=list)
    truth_conf: list[float] = field(default_factory=list)     # -1: not matched
    truth_expiry: list[bool] = field(default_factory=list)
    truth_tag: list[str | None] = field(default_factory=list)


def evaluate(samples: list[Sample], predictions: dict[str, list[Prediction]],
             iou_min: float = 0.5) -> Matches:
    """Match every photo's `date` predictions against its labelled dates."""
    m = Matches(iou_min)
    for s in samples:
        m.n_photos[s.tag] = m.n_photos.get(s.tag, 0) + 1
        truths = [b for b in s.boxes if b.label in DATE_LABELS]
        preds = [p for p in predictions.get(s.id, []) if p.label == "date"]
        pred_tp, taken = match_photo(preds, [b.xyxy for b in truths], iou_min)
        for conf, tp in pred_tp:
            m.pred_conf.append(conf)
            m.pred_tp.append(tp)
            m.pred_tag.append(s.tag)
        for b, conf in zip(truths, taken):
            m.truth_conf.append(-1.0 if conf is None else conf)
            m.truth_expiry.append(b.label in EXPIRY_LABELS)
            m.truth_tag.append(s.tag)
    return m


def summary(m: Matches, conf: float, tags: set | None = None) -> dict:
    """Recall, precision and false positives at one confidence threshold,
    over every photo, or only the photos whose failure tag is in `tags`."""
    def keep(t):
        return tags is None or t in tags

    pc = np.array([c for c, t in zip(m.pred_conf, m.pred_tag) if keep(t)])
    tp = np.array([x for x, t in zip(m.pred_tp, m.pred_tag) if keep(t)], bool)
    tc = np.array([c for c, t in zip(m.truth_conf, m.truth_tag) if keep(t)])
    te = np.array([e for e, t in zip(m.truth_expiry, m.truth_tag) if keep(t)], bool)
    photos = sum(n for t, n in m.n_photos.items() if keep(t))

    kept = pc >= conf
    found = tc >= conf
    n_kept, n_tp = int(kept.sum()), int((kept & tp).sum())
    out = {
        "iou": m.iou_min, "conf": round(conf, 4), "photos": photos,
        "dates": int(len(tc)), "dates_found": int(found.sum()),
        "date_recall": _ratio(found.sum(), len(tc)),
        "boxes_kept": n_kept, "precision": _ratio(n_tp, n_kept),
        "false_boxes_per_photo": _ratio(n_kept - n_tp, photos),
    }
    if te.any():
        out["expiry_dates"] = int(te.sum())
        out["expiry_found"] = int((found & te).sum())
        out["expiry_recall"] = _ratio((found & te).sum(), te.sum())
    return out


def pooled(matches: list[Matches]) -> Matches:
    """Several sets' matches as one, so a threshold can be chosen over all
    of them together (M2 pools its three real validation sets)."""
    if len({m.iou_min for m in matches}) != 1:
        raise ValueError("pooled matches need the same IoU level")
    out = Matches(matches[0].iou_min)
    for m in matches:
        for tag, n in m.n_photos.items():
            out.n_photos[tag] = out.n_photos.get(tag, 0) + n
        out.pred_conf += m.pred_conf
        out.pred_tp += m.pred_tp
        out.pred_tag += m.pred_tag
        out.truth_conf += m.truth_conf
        out.truth_expiry += m.truth_expiry
        out.truth_tag += m.truth_tag
    return out


def best_threshold(m: Matches, beta: float = 1.0) -> tuple[float, float]:
    """The confidence that maximises F-beta for dates, and that F-beta.

    beta 1 is F1 (M1). beta 2 is F2 (M2, decision 11): a missed date
    weighs twice as much as a false box, so the threshold sits lower.
    Candidates are the predictions' own confidences: between two of them,
    nothing changes.
    """
    if not m.pred_conf or not m.truth_conf:
        return 0.0, 0.0
    order = np.argsort(-np.array(m.pred_conf), kind="stable")
    conf = np.array(m.pred_conf)[order]
    tp = np.cumsum(np.array(m.pred_tp, bool)[order])
    kept = np.arange(1, len(conf) + 1)
    recall = tp / len(m.truth_conf)
    precision = tp / kept
    b2 = beta * beta
    f = (1 + b2) * precision * recall / np.maximum(b2 * precision + recall, 1e-12)
    # Among equal confidences only the last position counts (all kept at once).
    last = np.r_[conf[1:] != conf[:-1], True]
    i = int(np.argmax(np.where(last, f, -1)))
    return float(conf[i]), float(f[i])


def _ratio(a, b) -> float | None:
    return round(float(a) / float(b), 4) if b else None
