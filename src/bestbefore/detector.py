"""Training and running the date detector (Ultralytics YOLO11).

Every run uses the same fixed settings except the ones under test (model
size, input size, training data, seed): see docs/m1-date-detector.md,
decision 4. Pretrained weights are never downloaded here: compute nodes
have no internet, so scripts/fetch_pretrained.py puts them in
$CHECKPOINT_DIR/pretrained/ beforehand, on the login node.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import checkpoint_dir, expdate_dir
from .detection_eval import Matches, Prediction, best_threshold, evaluate, pooled, summary
from .images import load_upright
from .samples import Sample
from .yolo import CLASSES, split_repeats

# Fixed for every M1 run (decision 4). Horizontal flips are off: a flipped
# date is mirrored text, which no pack prints. Rotation stays at YOLO's
# default of 0, so M2 can measure what sideways synthetic dates add.
SETTINGS = dict(
    epochs=100,
    patience=30,
    batch=16,
    fliplr=0.0,
    deterministic=True,
)
PREDICT_CONF = 0.001   # keep nearly every box; thresholds are applied later

# The arguments `applied` reads back off Ultralytics' trainer.
APPLIED_ARGS = ("optimizer", "lr0", "momentum", "warmup_momentum", "warmup_bias_lr",
                "warmup_epochs", "epochs", "close_mosaic", "patience", "val", "batch",
                "fliplr", "deterministic", "imgsz", "seed", "fraction", "workers", "cache")


def pretrained_path(name: str) -> Path:
    """$CHECKPOINT_DIR/pretrained/<name>.pt, which must already exist."""
    path = checkpoint_dir() / "pretrained" / f"{Path(name).stem}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run scripts/fetch_pretrained.py first "
            "(on the cluster: on the login node, which has internet).")
    return path


def dataset_yaml(yolo_dir: Path, train_lists: list[str], out: Path,
                 val: str = "val") -> Path:
    """A YOLO dataset file training on the chosen image lists.

    This is how "synthetic + real" vs "real only" is switched without
    touching the converter's expdate.yaml. A list is a name in `yolo_dir`
    (`train_real` is train_real.txt) or the absolute path of a list file
    (M2's live elsewhere). `name*5` lists it five times: Ultralytics keeps
    repeats, so its photos are seen five times as often. `val` is chosen
    the same way (M2 scores each validation set through its own file).
    """
    train = []
    for spec in train_lists:
        name, k = split_repeats(spec)
        train += [_list_entry(yolo_dir, name)] * k
    out.write_text(
        "# Made by bestbefore.detector.dataset_yaml\n"
        f"path: {yolo_dir.resolve()}\n"
        f"train: [{', '.join(train)}]\n"
        f"val: {_list_entry(yolo_dir, val)}\n"
        "test: test.txt\n"
        "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    return out


def _list_entry(yolo_dir: Path, name: str) -> str:
    """A list as the dataset file names it: a list in `yolo_dir` by its
    file name, as M1 always wrote it; any other by its absolute path,
    quoted, since a path may hold spaces."""
    if Path(name).is_absolute():
        path, entry = Path(name), json.dumps(name)
    else:
        path, entry = yolo_dir / f"{name}.txt", f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    return entry


def train(weights: Path, data: Path, imgsz: int, seed: int, project: Path,
          name: str, keep: str = "best", callbacks: dict | None = None,
          **overrides) -> Path:
    """Fine-tune `weights` on `data`; returns the path of best.pt, or with
    keep="last" of last.pt, the final weights (M2, decision 10).

    `callbacks` maps Ultralytics events (e.g. "on_train_start") to
    functions of the trainer.
    """
    if keep not in ("best", "last"):
        raise ValueError(f"keep is 'best' or 'last', not {keep!r}")
    from ultralytics import YOLO

    model = YOLO(str(weights))
    for event, fn in (callbacks or {}).items():
        model.add_callback(event, fn)
    settings = {**SETTINGS, **overrides}
    model.train(data=str(data), imgsz=imgsz, seed=seed, project=str(project),
                name=name, exist_ok=False, **settings)
    return Path(model.trainer.save_dir) / "weights" / f"{keep}.pt"


def applied(trainer) -> dict:
    """What Ultralytics trains with, read off its trainer when training
    starts (an "on_train_start" callback), after optimizer="auto" would
    have been resolved: its arguments, the optimiser it built, how many
    photos the training set holds (repeats included), and whether they
    are cached in RAM (Ultralytics' own memory check can refuse)."""
    dataset = trainer.train_loader.dataset
    return {"args": {k: getattr(trainer.args, k) for k in APPLIED_ARGS},
            "optimizer": type(trainer.optimizer).__name__,
            "train_images": len(dataset),
            "distinct_images": len(set(dataset.im_files)),
            "ram_cache": dataset.cache == "ram",
            "loader_workers": trainer.train_loader.num_workers,
            "save_dir": str(trainer.save_dir)}


def momenta(trainer) -> list[float]:
    """The momentum of every parameter group. After warmup it stays where
    warmup left it, close to the `momentum` argument (decision 9)."""
    return sorted({round(g["momentum"], 4) for g in trainer.optimizer.param_groups})


def epochs_run(best: Path) -> int | None:
    """How many epochs a run trained for (fewer than asked = early stopping),
    from the results.csv Ultralytics writes one row per epoch to."""
    csv = best.parent.parent / "results.csv"
    if not csv.exists():
        return None
    return max(0, len(csv.read_text().strip().splitlines()) - 1)


def predict(model, images: list, imgsz: int, device=None) -> list[list[Prediction]]:
    """Boxes for a batch of upright PIL images, in 0..1 coordinates.

    Always given images from load_upright, never file paths: a phone photo
    with an EXIF rotation flag must be seen the same way as its labels.

    `rect=False` pads every photo to the same imgsz square. Ultralytics'
    default pads a batch of same-shaped photos less, which changes the
    confidences (by up to 0.15 on one validation date), so a threshold
    chosen on mixed-shape validation batches would mean something else on
    the store photos, which share a few shapes (docs/learnings/m1.md, §3.4).
    """
    results = model.predict(source=images, imgsz=imgsz, conf=PREDICT_CONF,
                            device=device, rect=False, verbose=False)
    out = []
    for r in results:
        b = r.boxes
        xyxyn, cls, conf = _numpy(b.xyxyn), _numpy(b.cls), _numpy(b.conf)
        out.append([Prediction(r.names[int(c)], tuple(float(v) for v in box), float(p))
                    for box, c, p in zip(xyxyn, cls, conf)])
    return out


def predict_samples(model, samples: list[Sample], imgsz: int, device=None,
                    batch: int = 16) -> dict[str, list[Prediction]]:
    """Predictions for every sample, keyed by sample id."""
    out = {}
    for i in range(0, len(samples), batch):
        chunk = samples[i:i + batch]
        preds = predict(model, [load_upright(s.image) for s in chunk], imgsz, device)
        out.update({s.id: p for s, p in zip(chunk, preds)})
    return out


def predict_checkpoint(checkpoint: Path, samples: list[Sample], imgsz: int,
                       device=None) -> dict[str, list[Prediction]]:
    """predict_samples for a checkpoint on disk."""
    from ultralytics import YOLO
    return predict_samples(YOLO(str(checkpoint)), samples, imgsz, device)


def score(checkpoint: Path, samples: list[Sample], imgsz: int, conf: float | None = None,
          data: Path | None = None, split: str | None = None,
          project: Path | None = None, device=None,
          predictions: dict[str, list[Prediction]] | None = None) -> dict:
    """Score a trained detector on labelled photos.

    With `conf` None (validation), the threshold is the one that maximises
    F1 for dates; otherwise (test, store photos) the given one is used
    unchanged. With a YOLO dataset file and split, Ultralytics' mAP is
    added, plus its own date recall at the same threshold as a cross-check.
    Photos with failure tags (the store set) are also scored per tag.

    `predictions`, when given, are this checkpoint's for these samples
    (predict_samples), for a caller that keeps them: they aren't made twice.
    """
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    preds = predictions if predictions is not None else predict_samples(model, samples, imgsz, device)
    at_05, at_03 = evaluate(samples, preds, 0.5), evaluate(samples, preds, 0.3)
    if conf is None:
        conf, f1 = best_threshold(at_05)
    else:
        f1 = None
    out = {"checkpoint": str(checkpoint), "imgsz": imgsz,
           "threshold": round(conf, 4), "f1_at_threshold": f1,
           **_summaries(at_05, at_03, conf)}
    tags = [t for t in dict.fromkeys(s.tag for s in samples) if t]
    if tags:
        out["by_tag"] = {t: _summaries(at_05, at_03, conf, {t}) for t in tags}
    if data is not None:
        out.update(_val_metrics(model, data, split, imgsz, device, project, f"val-{split}", conf))
    return out


def score_suite(checkpoint: Path, imgsz: int, out_dir: Path, device=None) -> dict:
    """Score an M2 run on its validation suite (docs/m2-synthetic-generator.md,
    "The validation suite").

    Every set is predicted with rect=False, as M1 D3. The threshold is the
    one that maximises F2 for dates over the three real sets pooled
    (decision 11), kept unrounded. Per set: found at IoU 0.5 and 0.3 at
    that threshold, boxed at any confidence, precision, false boxes per
    photo, and Ultralytics' mAP per class through the set's own dataset
    file (kept in out_dir/suite/); per stratum too where photos carry tags
    (synth-val, from S6). A set that isn't made yet is marked missing,
    never skipped silently; without all three real sets there is no
    threshold, so that raises.
    """
    from ultralytics import YOLO

    from . import m2, sets

    missing = {name: why for name in m2.SUITE if (why := m2.suite_missing(name))}
    if lacking := m2.missing({n: missing.get(n) for n in m2.THRESHOLD_SETS}):
        raise FileNotFoundError(f"the threshold needs every real set: {lacking}")
    model = YOLO(str(checkpoint))
    matched = {}
    for name in m2.SUITE:
        if name not in missing:
            samples = sets.load_set(name)
            preds = predict_samples(model, samples, imgsz, device)
            matched[name] = samples, evaluate(samples, preds, 0.5), evaluate(samples, preds, 0.3)
    conf, f_beta = best_threshold(pooled([matched[n][1] for n in m2.THRESHOLD_SETS]),
                                  beta=m2.THRESHOLD_BETA)
    out = {"checkpoint": str(checkpoint), "imgsz": imgsz, "threshold": conf,
           "threshold_rule": f"best F{m2.THRESHOLD_BETA:g} for dates on "
                             f"{' + '.join(m2.THRESHOLD_SETS)}, pooled",
           "f2_at_threshold": round(f_beta, 4), "missing": missing}
    (out_dir / "suite").mkdir(parents=True, exist_ok=True)
    for name in m2.SUITE:
        if name in missing:
            out[name] = {"missing": missing[name]}
            continue
        samples, at_05, at_03 = matched[name]
        out[name] = _summaries(at_05, at_03, conf)
        tags = [t for t in dict.fromkeys(s.tag for s in samples) if t]
        if tags:
            out[name]["by_tag"] = {t: _summaries(at_05, at_03, conf, {t}) for t in tags}
        listed = m2.list_spec(m2.SUITE[name])
        # Ultralytics wants a train entry too; validating never reads it.
        data = dataset_yaml(expdate_dir() / "yolo", [listed], out_dir / "suite" / f"{name}.yaml",
                            val=listed)
        out[name].update(_val_metrics(model, data, "val", imgsz, device,
                                      checkpoint.parent.parent / "suite", name, conf))
    return out


def _val_metrics(model, data: Path, split: str, imgsz: int, device, project: Path | None,
                 name: str, conf: float) -> dict:
    """Ultralytics' mAP per class on one split of a dataset file, and its own
    date recall at `conf`, to cross-check ours."""
    metrics = model.val(data=str(data), split=split, imgsz=imgsz, batch=16,
                        device=device, plots=False, verbose=False,
                        project=str(project) if project else None, name=name)
    box = metrics.box
    classes = np.asarray(box.ap_class_index).tolist()
    return {"map50": {CLASSES[c]: round(float(v), 4) for c, v in zip(classes, box.ap50)},
            "map50_95": {CLASSES[c]: round(float(v), 4) for c, v in zip(classes, box.ap)},
            "ultralytics_date_recall_at_threshold": ultralytics_recall_at(metrics, conf)}


def _summaries(at_05: Matches, at_03: Matches, conf: float, tags: set | None = None) -> dict:
    """Found at IoU 0.5 and 0.3 at the threshold, and at IoU 0.5 at any
    confidence: the last separates "never boxed" from "boxed, not sure
    enough" (added after D2; it changes no headline number)."""
    return {"iou0.5": summary(at_05, conf, tags), "iou0.3": summary(at_03, conf, tags),
            "any_conf": summary(at_05, 0.0, tags)}


def ultralytics_recall_at(metrics, conf: float, class_id: int = 0) -> float | None:
    """Ultralytics' own recall for one class at a confidence, read off the
    recall curve it computed during validation (IoU 0.5). Used to
    cross-check our evaluator."""
    box = metrics.box
    rows = list(np.asarray(box.ap_class_index).tolist())
    if class_id not in rows:
        return None
    return float(np.interp(conf, box.px, box.r_curve[rows.index(class_id)]))


def _numpy(x) -> np.ndarray:
    return np.asarray(x.cpu() if hasattr(x, "cpu") else x)
