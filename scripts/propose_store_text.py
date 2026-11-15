"""M3 S4: proposals for the store text, as a Label Studio project
(docs/m3-date-reader.md, "Store text (S4)", S4; store_text_review.py).

    python scripts/propose_store_text.py cut                                       # prints RUN
    ~/venvs/ocr-cpu/bin/python -I scripts/propose_store_text.py read RUN --threads 4
    python scripts/propose_store_text.py tasks RUN

S2's crop experiment read the 198 expiry and mfg boxes with a value; the
other dates (the frozen labels' other_date boxes and the keyword review's
added dates) have no read yet. `cut` (main venv) cuts each of those from
the export the way S2 cut its boxes ($STORE_TRAIN_DIR: the upright train_0
copy, or val; shrunk to 1,600 px; the recipe's padding, m3.crop_recipe_used)
into an owner-only run folder under data/store/runs/. `read`
(~/venvs/ocr-cpu, which doesn't hold bestbefore) reads each with PP-OCRv6
small's recogniser alone at its two quarter turns, as S2's `turned` mode
did; only that, as nothing else is proposed (seconds on the laptop).
`tasks` (main venv) takes S2's PP-OCRv6 alone-and-turned read at the
recipe's padding for each box it read, the fresh reads for the rest (the
same rule picks the turn, crop_scores.choose_turn), and writes into
data/store/text/ (owner-only):

    proposals.json  each box's proposed text and turn, by numbered photo and line
    tasks.json      one task per exported photo, the proposals as a prediction
    project.json    the project's title, interface and instructions (with a
                    picture of the four ways print can read: a CSS background,
                    as Label Studio 1.23 strips an <img> whose source is data:)

Only store sets v1 and v3, as exported; never v2. Prints counts only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

S2_RUN = "20261113-185541-m3-crop-experiment"   # S2's reads, copied to data/store/runs/
CROPS, READS = "crops.json", "reads.jsonl"
PADDLE_REC = "PP-OCRv6_small_rec"
TITLE = "Store text (M3 S4)"
# The turn choices as the interface lists them (store_text.TURNS maps them).
CHOICE_ORDER = ("upright", "reads bottom to top", "reads top to bottom", "upside down")

INTERFACE = """<View>
  <Header value="$note" size="5"/>
  <Image name="img" value="$image" zoom="true" zoomControl="true"/>
  <Header value="Date boxes: leave their place and label as they are" size="6"/>
  <RectangleLabels name="date" toName="img">
    <Label value="expiry" background="#d62728"/>
    <Label value="mfg" background="#1f77b4"/>
    <Label value="other_date" background="#7f7f7f"/>
  </RectangleLabels>
  <Header value="Shelf-life phrase (e.g. 24 months from Mfg.): pick life, then draw a box round the phrase" size="6"/>
  <RectangleLabels name="life" toName="img">
    <Label value="life" background="#ff7f0e"/>
  </RectangleLabels>
  <View visibleWhen="region-selected">
    <Header value="The selected box's text, exactly as printed" size="6"/>
    <TextArea name="text" toName="img" perRegion="true" editable="true" maxSubmissions="1" required="true" rows="1" placeholder="as printed, e.g. EXP 04/12/2026"/>
    <Header value="Which way does its print read? If you tilt your head LEFT to read it: bottom to top. RIGHT: top to bottom." size="6"/>
    <Choices name="turn" toName="img" perRegion="true" required="true" choice="single-radio">
      <Choice value="upright" hint="left to right, like this line"/>
      <Choice value="reads bottom to top" hint="tilt your head LEFT to read it"/>
      <Choice value="reads top to bottom" hint="tilt your head RIGHT to read it"/>
      <Choice value="upside down" hint="you'd turn the photo over to read it"/>
    </Choices>
  </View>
</View>
"""

INSTRUCTIONS = """<h3>Store text: what each date box says, and which way it reads</h3>
<p>Every date box already has OCR's guess at its text and its direction. On every photo:</p>
<ol>
<li><b>Click each date box</b> (red expiry, blue mfg, grey other date). Below the photo:
 <ul>
 <li><b>Text</b>: fix it to exactly what is printed inside the box: every character, its case
 (<code>Feb</code> or <code>FEB</code>), its punctuation, one space where the print has a gap.
 If the box also holds a word (<code>EXP.</code>, <code>MFG:</code>), type it too. Two lines in one box:
 the top line first, one space between. A character you can't read even zoomed in: type <code>#</code> for it.</li>
 <li><b>Which way it reads</b>: pick one of the four, by the picture below.</li>
 </ul></li>
<li><b>Shelf-life phrase</b> (<code>24 months from Mfg.</code>, <code>Best before 1 year from manufacture</code>):
 pick <b>life</b>, draw one box round the phrase, then type its text and pick its direction as for a date.
 The line above the photo says when the photo's expiry is worked out from one: that photo must have a life box.
 Box it on other photos too when one is printed.</li>
<li>Never move, resize, relabel or delete a date box, and draw no new date box.</li>
<li><b>Submit</b> (after changing a photo already submitted, <b>Update</b>). A photo with nothing to box is
 submitted as it is.</li>
</ol>
<p><b>Which way it reads.</b> Look at the print as the photo shows it:</p>
<style>.bestbefore-turns {{width: 100%; max-width: 1160px; aspect-ratio: 1160 / 330;
background: url(data:image/png;base64,{picture}) no-repeat left top / contain;}}</style>
<div class="bestbefore-turns" title="the four ways print can read"></div>
<p>Tilt your head to the <b>left</b> to read it: <b>reads bottom to top</b>. To the <b>right</b>:
<b>reads top to bottom</b>. Turned right over: <b>upside down</b>.</p>
"""


# --- cut (main venv) --------------------------------------------------------------------

def export_text() -> tuple[Path, dict, str, dict]:
    """The export's folder, manifest, its hash and its text.json, checked to fit."""
    import hashlib
    from bestbefore import store_text, store_train
    from bestbefore.config import store_train_dir
    folder = store_train_dir()
    manifest_bytes = (folder / store_train.MANIFEST).read_bytes()
    manifest, digest = json.loads(manifest_bytes), hashlib.sha256(manifest_bytes).hexdigest()
    text = json.loads((folder / store_text.NAME).read_bytes())
    if problems := store_text.check(text, manifest, digest):
        sys.exit("text.json doesn't fit the export:\n  " + "\n  ".join(problems))
    return folder, manifest, digest, text


def s2_crops(manifest_sha256: str) -> dict[tuple[str, int], str]:
    """S2's store crops: (numbered photo, line) -> its key, once S2's run is
    shown to be cut from this export."""
    from bestbefore import store
    info = json.loads((store.RUNS_DIR / S2_RUN / CROPS).read_text())
    if info["export_manifest_sha256"] != manifest_sha256:
        sys.exit(f"S2's run {S2_RUN} was cut from another export")
    return {(c["photo"], c["box"]): c["key"] for c in info["crops"] if c["set"] == "store"}


def cut() -> None:
    os.umask(0o077)                 # every file and folder made here is the owner's only
    from bestbefore import derived, m3, runs, store, store_train
    from bestbefore.config import checkpoint_dir
    from bestbefore.images import load_upright
    from bestbefore.reader import crops

    folder, manifest, digest, text = export_text()
    read = s2_crops(digest)
    pad = m3.crop_recipe_used(runs.read_json(m3.DECISIONS_PATH) or {})["pad"]
    run = runs.start_run("m3-store-text-reads", {"pad": pad, "export_manifest_sha256": digest,
                                                  "s2_run": S2_RUN}, root=store.RUNS_DIR)
    os.chmod(run, 0o700)
    (run / "crops").mkdir()
    records = []
    for name in sorted(text["photos"]):
        entry, meta = text["photos"][name], manifest["photos"][name]
        todo = [b for b in entry["boxes"] if (name, b["index"]) not in read]
        if not todo:
            continue
        path = folder / "images" / next(iter(store_train.FOLDERS[meta["part"]])) / f"{name}.jpg"
        boxes = derived.read_boxes(path)
        if [b.label for b in boxes] != [store_train.YOLO_LABEL[label] for label in meta["labels"]]:
            sys.exit(f"{path.name}: its label file doesn't follow the manifest")
        image = store_train.shrink(load_upright(path))
        for b in todo:
            box = crops.to_pixels(boxes[b["index"]].xyxy, image.size)
            key = f"t{len(records) + 1:04d}"
            image.crop(crops.padded(box, pad, image.size)).save(run / "crops" / f"{key}.png")
            records.append({"key": key, "photo": name, "box": b["index"], "label": b["label"],
                            "part": meta["part"], "shape": "tall" if crops.is_tall(box) else "wide",
                            "file": f"crops/{key}.png"})
    (run / CROPS).write_text(json.dumps({"pad": pad, "export_manifest_sha256": digest,
                                         "paddlex_home": str(checkpoint_dir() / "pretrained" / "paddlex"),
                                         "crops": records}, indent=1))
    print(f"cut {len(records)} boxes S2 didn't read into {run}", file=sys.stderr)
    print(run)


# --- read (~/venvs/ocr-cpu, nothing of bestbefore) ----------------------------------------

def read(run: Path, threads: int) -> None:
    os.umask(0o077)
    info = json.loads((run / CROPS).read_text())
    os.environ["PADDLE_PDX_CACHE_HOME"] = info["paddlex_home"]
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    import numpy as np
    from PIL import Image
    from paddleocr import TextRecognition

    t0 = time.time()
    rec = TextRecognition(model_name=PADDLE_REC, device="cpu", cpu_threads=threads)
    load_s, t0, n = time.time() - t0, time.time(), 0
    with open(run / READS, "w") as out:
        for c in info["crops"]:
            with Image.open(run / c["file"]) as im:
                rgb = np.asarray(im.convert("RGB"))
            for k in ((1, 3) if c["shape"] == "tall" else (0, 2)):      # crops.turns, decision 14
                (r,) = rec.predict(np.ascontiguousarray(np.rot90(rgb, k)[:, :, ::-1]))   # BGR, as S2
                out.write(json.dumps({"key": c["key"], "k": k, "text": r["rec_text"],
                                      "score": float(r["rec_score"])}, ensure_ascii=False) + "\n")
                n += 1
    import paddle
    import paddleocr
    (run / "read.json").write_text(json.dumps({
        "threads": threads, "model": PADDLE_REC, "load_seconds": round(load_s, 1),
        "read_seconds": round(time.time() - t0, 1), "reads": n, "host": os.uname().nodename,
        "versions": {"paddle": paddle.__version__, "paddleocr": paddleocr.__version__,
                     "numpy": np.__version__}}, indent=1))
    print(f"read {len(info['crops'])} crops, {n} reads in {time.time() - t0:.1f} s "
          f"(model loaded in {load_s:.1f} s)", file=sys.stderr)


# --- tasks (main venv) --------------------------------------------------------------------

def picture() -> str:
    """The four ways print can read, as a PNG in base64: an example line in
    each, with the choice's name under it. Each is drawn by turning the
    upright line against store_text.TURNS, so turning it back by the
    choice's quarter turns stands it up."""
    import base64
    import io
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from bestbefore import store_text

    font, small = ImageFont.load_default(size=30), ImageFont.load_default(size=17)
    line = Image.new("RGB", (250, 44), "white")
    ImageDraw.Draw(line).text((8, 4), "EXP 08/2027", fill="black", font=font)
    upright = np.asarray(line)
    cell, out = 290, Image.new("RGB", (4 * 290, 330), "white")
    draw = ImageDraw.Draw(out)
    hints = {"upright": "as you read this", "reads bottom to top": "tilt your head LEFT",
             "reads top to bottom": "tilt your head RIGHT", "upside down": "turn it right over"}
    for i, choice in enumerate(CHOICE_ORDER):
        shown = Image.fromarray(np.ascontiguousarray(np.rot90(upright, -store_text.TURNS[choice])))
        x = i * cell + (cell - shown.width) // 2
        out.paste(shown, (x, 10 + (260 - shown.height) // 2))
        draw.rectangle((x - 1, 9 + (260 - shown.height) // 2, x + shown.width, 10 + (260 - shown.height) // 2 + shown.height),
                       outline=(214, 39, 40), width=2)
        draw.text((i * cell + 12, 278), choice, fill="black", font=small)
        draw.text((i * cell + 12, 302), hints[choice], fill=(90, 90, 90), font=small)
    buf = io.BytesIO()
    out.save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def proposals(run: Path, manifest_sha256: str, pad: float) -> dict[str, dict[int, dict]]:
    """Each date box's proposed text and turn: S2's PP-OCRv6 recogniser
    alone, read both ways, at the recipe's padding; the fresh reads for the
    boxes S2 didn't read, by the same rule."""
    from bestbefore import store
    from bestbefore.reader import crop_scores
    keys = {v: k for k, v in s2_crops(manifest_sha256).items()}
    out: dict[str, dict[int, dict]] = {}
    config = crop_scores.config_name("ppocrv6", "turned", pad)
    with open(store.RUNS_DIR / S2_RUN / "scored.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["config"] == config and r["key"] in keys:
                photo, index = keys[r["key"]]
                out.setdefault(photo, {})[index] = {"text": r["text"], "turn": r["k"], "score": r["score"], "from": "S2"}
    info = json.loads((run / CROPS).read_text())
    if info["export_manifest_sha256"] != manifest_sha256 or info["pad"] != pad:
        sys.exit(f"{run} was cut from another export, or at another padding")
    reads: dict[str, list[dict]] = {}
    with open(run / READS) as f:
        for line in f:
            r = json.loads(line)
            reads.setdefault(r["key"], []).append(r)
    for c in info["crops"]:
        if len(reads.get(c["key"], [])) != 2:
            sys.exit(f"{c['key']}: {len(reads.get(c['key'], []))} reads, not 2")
        r = crop_scores.choose_turn(reads[c["key"]])
        out.setdefault(c["photo"], {})[c["box"]] = {"text": r["text"], "turn": r["k"], "score": r["score"], "from": "S4"}
    return out


def write_private(path: Path, data) -> str:
    from bestbefore import store_text
    return store_text.write_private(path, data)


def tasks(run: Path) -> None:
    os.umask(0o077)
    from collections import Counter
    from datetime import date
    from bestbefore import m3, runs, store, store_text, store_text_review as review, store_train
    from bestbefore.config import store_photos_dir
    from bestbefore.images import upright_size
    from bestbefore.runs import code_sha256

    folder, manifest, digest, text = export_text()
    pad = m3.crop_recipe_used(runs.read_json(m3.DECISIONS_PATH) or {})["pad"]
    proposed = proposals(run, digest, pad)
    sets, designs, split, sources = store_train.export_inputs()
    labels = {p: lab for given in sets.values() for p, lab in given["labels"].items()}
    root = store_photos_dir()
    out_tasks, counts, by_name = [], Counter(), {}
    for p in store_train.export_plan(sets, designs, split):
        first = next(iter(store_train.FOLDERS[p.part]))
        p = store_train.without_slips(p, upright_size(folder / "images" / first / f"{p.name}.jpg"))
        values = store_text.box_values(p, labels[p.photo])
        if [(b["index"], b["label"]) for b in values] != [(b["index"], b["label"]) for b in text["photos"][p.name]["boxes"]]:
            sys.exit(f"{p.name}: its date boxes aren't text.json's")
        dated = [b for b in p.boxes if b.label in store.BOX_LABELS]
        boxes = [{"index": v["index"], "label": v["label"], "xyxy": list(b.xyxy)} for b, v in zip(dated, values)]
        mine = proposed.get(p.name, {})
        if missing := [b["index"] for b in boxes if b["index"] not in mine]:
            sys.exit(f"{p.name}: no proposal for date lines {missing}")
        out_tasks.append(review.task(p.photo, upright_size(root / p.photo), boxes, mine, labels[p.photo]["derived"]))
        by_name[p.name] = mine
        counts["photos"] += 1
        counts["derived photos"] += labels[p.photo]["derived"]
        counts["photos without a date box"] += not boxes
        for b in boxes:
            counts[f"{b['label']} boxes"] += 1
            counts[f"proposals from {mine[b['index']]['from']}"] += 1
            counts[f"proposed {review.CHOICE[mine[b['index']]['turn']]}"] += 1
            counts["proposals holding no text"] += not mine[b["index"]]["text"].strip()

    out = store_text.REVIEW_DIR
    out.mkdir(mode=0o700, exist_ok=True)
    os.chmod(out, 0o700)
    record = {"made": date.today().isoformat(), "model_version": review.MODEL_VERSION, "pad": pad,
              "export_manifest_sha256": digest, "s2_run": S2_RUN, "fresh_run": run.name,
              "code_sha256": code_sha256(review, store_text, sys.modules[__name__])}
    write_private(out / "proposals.json", {**record, "photos": {n: {str(i): v for i, v in sorted(b.items())}
                                                                for n, b in sorted(by_name.items())}})
    write_private(out / "tasks.json", out_tasks)
    project = {"title": TITLE, "label_config": INTERFACE, "expert_instruction": INSTRUCTIONS.format(picture=picture()),
               "show_instruction": True, "show_skip_button": False, "show_collab_predictions": True,
               "model_version": review.MODEL_VERSION, "maximum_annotations": 1}
    write_private(out / "project.json", project)
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")
    print(f"wrote {out}/: proposals.json, tasks.json ({len(out_tasks)} tasks), project.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    steps = parser.add_subparsers(dest="step", required=True)
    steps.add_parser("cut")
    p = steps.add_parser("read")
    p.add_argument("run", type=Path)
    p.add_argument("--threads", type=int, default=4)
    p = steps.add_parser("tasks")
    p.add_argument("run", type=Path)
    args = parser.parse_args()
    if args.step == "cut":
        cut()
    elif args.step == "read":
        read(args.run, args.threads)
    else:
        tasks(args.run)


if __name__ == "__main__":
    main()
