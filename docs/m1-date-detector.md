# M1: Date detector

Planned 2026-10-14. Status: not started.
The ideas behind this plan, explained simply: [`learnings-m1.md`](learnings-m1.md).

**Goal.** Fine-tune a small YOLO model that finds every printed date in a
photo, trained on Products-Synth plus the Products-Real train split, on
the university cluster.

**Done when.** `results/m1_detector.md` reports:

- **ExpDate Products-Real test:** mAP@0.5 and recall for dates, and how
  many of the 665 expiry dates were found (IoU of at least 0.5).
- **Store photos:** how many expiry dates were found, by failure tag.

M1 is judged **only on finding dates**, not on reading them. From
Scanify, RapidOCR reads crops *worse* than whole photos (25 vs 31 of 44),
so a detector feeding crops to off-the-shelf OCR won't beat baseline B.
Reading crops well is M3's job.

---

## What we start from

**From M0, Part A.** `$EXPDATE_DIR/yolo/` holds the YOLO dataset: image
lists `train_real.txt` (937), `train_synth.txt` (11,858), `val.txt` (165
real) and `test.txt` (665 real), and `expdate.yaml` with 4 classes:
`0 date`, `1 due`, `2 prod`, `3 code`. What we learned about the data
([`datasets.md`](datasets.md)):

| Finding | What it means for M1 |
|---|---|
| Real date boxes are ~16 px tall at 640 px input, ~31 px at 1280 | compare 640 and 1280 px input |
| Synthetic dates are bigger (median 27 px at 640) | measure whether synthetic data helps instead of assuming it |
| Almost no sideways dates in ExpDate | don't expect M1 to find sideways dates; that is M2's job |
| Synthetic PROD/EXP pairs are random | the detector finds dates; telling expiry from mfg is M3's job |

**From the cluster check** (`CLAUDE.md`, HPC section):

| Fact | What it means for M1 |
|---|---|
| 2 GPU nodes, 2× NVIDIA H100 each | one H100 per run is plenty |
| QOS: 2 jobs at a time per user | run experiments 2 at a time |
| Compute nodes have **no internet** | download data, packages and weights on the login node first |
| No scratch; 950 GB home quota | data and checkpoints in `~/bestbefore-data/` |
| `Python/3.11.11` module | same Python as the laptop |

**Not yet done:** the store test set (M0 Parts B and C). Only the last
evaluation slice (D3) needs it; everything else can go ahead now.

---

## Decisions

Each comes with its reason. **(new)** marks a choice not already written
down in `CLAUDE.md` or the M0 plan.

1. **Ultralytics YOLO11, pinned to one version.** It's the best-documented
   small detector, it fine-tunes with a few lines of code, and it reads
   the YOLO format M0 produced. The exact version is pinned in
   `pyproject.toml` (the `train` extra) so the laptop and the cluster run
   the same code. A newer family (e.g. YOLO26) is a Later comparison, not
   the starting point.
2. **(new) Train all 4 classes, judge on `date`.** The `due` ("EXP") and
   `prod` ("MFG") marks cost nothing extra to learn, and M3 can use them:
   a date next to an "EXP" mark is probably the expiry. Every headline
   number is for the `date` class.
3. **(new) Two questions decide the final setup, both answered on the
   validation split only:**
   - model size, `yolo11n` (2.6 M parameters) or `yolo11s` (9.4 M), and
     input size, 640 or 1280 px: a 2 × 2 grid;
   - whether Products-Synth helps: synthetic + real vs real only.
4. **(new) Every run uses the same fixed settings** except the one under
   test: 100 epochs with early stopping (patience 30), batch 16, seed 0,
   YOLO's default augmentation **with horizontal flipping turned off**.
   A flipped date is mirrored text, which never appears on a pack.
   Rotation augmentation stays off too, so that M2 can measure cleanly
   what sideways synthetic data adds.
5. **(new) The confidence threshold is chosen on validation:** the one
   that maximises F1 for `date`. It is then fixed and used unchanged on
   the test split and the store photos.
6. **(new) The test split is used once.** After both questions are
   answered and the threshold is fixed, the final setup is trained with 3
   seeds (0, 1, 2) and all three are scored on test once. Results are
   reported as mean and range. Seed 0's model is declared the M1
   detector **in advance**, so no model is picked by its test score.
7. **(new) The store photos never go to the cluster.** `CLAUDE.md` says
   only aggregate numbers leave the laptop, and the cluster is another
   machine. The trained weights (a few MB) come to the laptop instead, and
   the store evaluation runs there.
8. **(new) Store-photo statistics don't choose settings.** It would be
   tempting to pick the input size from the store photos' box sizes, but
   that is tuning on the test set. Input size is chosen on ExpDate
   validation only, and the store result is reported as it comes out.
9. **(new) Our own evaluator for "was the date found".** Ultralytics
   computes mAP@0.5. But "how many expiry dates were found" needs to know
   which box is the expiry, which the YOLO labels don't keep. So
   `bestbefore.detection_eval` scores predictions against `Sample`s with
   IoU, the same way on ExpDate and the store photos. As a cross-check,
   its `date` recall must agree with Ultralytics' recall at the same
   threshold.
10. **(new) Predictions always take the image from `load_upright`**, never
    a file path. Phone photos carry an EXIF rotation flag, and this keeps
    the model and the labels in the same frame, as in M0.
11. **(new) The cluster only pulls code; the laptop does all the
    committing.** The cluster checkout is a `git clone` (so every run
    records its exact commit), results come back with `rsync`, and the
    summary tables are written and committed on the laptop.

---

## Slices

### Part A: Cluster setup (one-time)

#### A1. Get the code onto the cluster

**Goal:** the repo checked out on the cluster with a working Python env,
and an easy login from the laptop.

1. **SSH key and alias (laptop).** Make a key with `ssh-keygen -t
   ed25519`, copy it over with `ssh-copy-id`, and add an alias to
   `~/.ssh/config`:

   ```
   Host bbhpc
       HostName <cluster-address>
       User <username>
   ```

   From then on, `ssh bbhpc` and `rsync … bbhpc:…` work without typing
   the password. Login details live only in `~/.ssh/config`, never in the
   repo.
2. **GitHub.** Create the GitHub repo and push `main` (the very first
   question in this project). On the cluster, `git clone` it into
   `~/bestbefore`. If the repo is private, add a read-only deploy key.
3. **Python env (cluster).**

   ```
   module load Python/3.11.11
   cd ~/bestbefore
   python3 -m venv .venv          # the module has python3, not python
   .venv/bin/pip install -e ".[dev,download,train]"
   mkdir -p slurm/logs
   ```

   Run on the login node, which has internet. `slurm/logs/` must exist
   before any job is submitted: Slurm doesn't create it, and a job whose
   log folder is missing fails without leaving a log.
4. **Code changes in this slice:**
   - `pyproject.toml`: a `train` extra with `ultralytics==<pinned>`.
   - `config.py` and `.env.example`: a new `CHECKPOINT_DIR` (where
     Ultralytics writes weights and plots). On the cluster it's
     `~/bestbefore-data/checkpoints`; on the laptop, `data/checkpoints`.
   - `.gitignore`: `slurm/logs/` is already ignored, and `*.pt` too.
5. **Ultralytics settings (cluster):** `yolo settings sync=False`, which
   turns off anonymous usage reporting. It would fail anyway without
   internet.

**Check:** `python -c "import torch, ultralytics; print(torch.__version__,
ultralytics.__version__)"` runs on the login node.
**Commit:** `feat: train extra (ultralytics) and CHECKPOINT_DIR`

#### A2. Get the data onto the cluster

**Goal:** the same ExpDate files and the same YOLO dataset as on the
laptop.

1. On the cluster, `.env` sets `EXPDATE_DIR=~/bestbefore-data/expdate` and
   `CHECKPOINT_DIR=~/bestbefore-data/checkpoints`. The store-photo and
   Scanify variables stay unset, because they don't exist there.
2. On the **login node**:
   `python scripts/download_expdate.py products-real products-synth`. The
   script checks both SHA-256 hashes, so the files are proven identical to
   the laptop's.
3. `python scripts/check_expdate.py` and `python
   scripts/expdate_to_yolo.py`. Both take seconds, so the login node is
   fine.

**Check:** the counts match the laptop exactly (1,102 / 665 / 11,858
photos; 937 / 165 / 11,858 / 665 in the lists). The validation split
uses seed 0, so it's the same 165 photos.
**Commit:** none (no code changes).

#### A3. Smoke tests: catch problems before the real runs

**Goal:** find crashes and hidden downloads in minutes, not after a long
wait in the queue.

1. **On the login node, pre-fetch everything a job would download:** the
   pretrained weights `yolo11n.pt` and `yolo11s.pt` (into
   `~/bestbefore-data/weights/`) and the font Ultralytics uses for its
   plots. Compute nodes have no internet, so any download inside a job
   fails.
2. **Laptop smoke test:** train `yolo11n` for 1 epoch on 2% of the data
   (`fraction=0.02`) on CPU. This catches bugs in our own code without
   using the cluster at all.
3. **Cluster smoke test:** `slurm/smoke_test.sbatch` asks for 1 GPU for
   20 minutes and runs the same tiny training. It prints the GPU name,
   PyTorch's CUDA version and the time per epoch. It must finish with no
   download attempts in the log.
4. Write the measured time per epoch into this plan's GPU budget table
   (below), replacing the estimates.

**Commit:** `feat: smoke-test job for the cluster`

### Part B: Training code

#### B1. The training script

**Goal:** one command trains one detector, records everything, and scores
it on validation.

- `src/bestbefore/detector.py`:
  - `dataset_yaml(train_lists, out_path)` writes a YOLO dataset file for
    a chosen mix of training lists. This is how "synthetic + real" vs
    "real only" is switched, without touching `expdate.yaml`.
  - `train(...)` wraps `YOLO(weights).train(...)` with the fixed settings
    from decision 4.
  - `predict(model, image)` takes a PIL image from `load_upright` and
    returns boxes as normalised `xyxy` with class and confidence.
- `src/bestbefore/detection_eval.py`, needed here to score validation
  and reused unchanged in D2 and D3:
  - `match(predictions, truth, iou=0.5)`: greedy one-to-one matching,
    highest confidence first, so one prediction can't count for two
    labels.
  - `score_set(...)` returns recall for all dates, **recall for the
    expiry date** (when the set marks one), precision for dates and false
    positives per photo, at IoU 0.5 (headline) and 0.3 (secondary:
    "roughly found", enough for a reader that crops with a margin).
  - `best_threshold(...)`: the confidence that maximises F1 for `date`.
- `scripts/train_detector.py`, for example:

  ```
  python scripts/train_detector.py --weights yolo11s.pt --imgsz 1280 \
      --train train_synth,train_real --seed 0 --name s1280
  ```

  It starts a run record (`results/runs/<id>/run.json`: flags, git
  commit, seed, versions, GPU name). Ultralytics' own output (weights,
  curves, `args.yaml`) goes to `$CHECKPOINT_DIR/m1/<id>/`, and its path is
  written into `run.json`. After training, it scores `best.pt` on the
  validation split and writes `results/runs/<id>/val.json`: mAP@0.5 per
  class, precision and recall for `date` at the F1-best threshold, and
  that threshold.
- `tests/test_detector.py`: `dataset_yaml` writes the right lists, and
  `predict` returns boxes in the upright frame (a tiny fake model output,
  so no GPU is needed).
- `tests/test_detection_eval.py`: hand-made boxes for a perfect match, a
  near miss at IoU 0.45, one prediction covering two labels, an empty
  photo, the 0.3 threshold, and the F1-best threshold on a small example.

**Commits:** `feat: detection evaluator (matching, recall, F1-best threshold)`,
then `feat: detector training script with run records and validation scoring`

#### B2. The job script

**Goal:** submit any training with one `sbatch` line.

`slurm/train_detector.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=m1-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%j.out

module load Python/3.11.11
source ~/bestbefore/.venv/bin/activate
cd ~/bestbefore
python scripts/train_detector.py "$@"
```

Experiments with several runs use a **job array** with `%2` (at most 2
running at once, the QOS limit). Each array task reads its own flags
from a small table in the experiment folder.

**Check:** a 1-epoch run submitted with `sbatch` finishes, and its
`run.json` and `val.json` are where they should be.
**Commit:** `feat: Slurm job script for detector training`

### Part C: Experiments (validation split only)

Each question gets a folder under `experiments/` with a README (question,
method, result, decision), the array table, and a `summarise.py` that
rebuilds the README's table from the runs' `val.json` files.

#### C1. Model size and input size

**Question:** which of `yolo11n` / `yolo11s` and 640 / 1280 px finds
dates best on validation?

| Run | Weights | Input | Training data |
|---|---|---|---|
| n640 | `yolo11n.pt` | 640 | synthetic + real |
| n1280 | `yolo11n.pt` | 1280 | synthetic + real |
| s640 | `yolo11s.pt` | 640 | synthetic + real |
| s1280 | `yolo11s.pt` | 1280 | synthetic + real |

**Decision rule, written before the runs:** pick the highest validation
mAP@0.5 for `date`. With only 165 validation photos, differences under 1
point are noise, so within 1 point prefer the cheaper run (smaller model
first, then smaller input).
**Folder:** `experiments/m1-size-and-input/`.

#### C2. Does synthetic data help?

**Question:** with the C1 winner's settings, does adding the 11,858
synthetic photos beat training on the 937 real ones alone?

| Run | Training data |
|---|---|
| C1 winner | synthetic + real (already trained in C1) |
| real-only | real only |

**Decision rule:** keep synthetic data unless real-only is better by more
than 1 point of validation mAP@0.5 for `date`. Either answer is a result
worth writing up: M0 showed the synthetic dates are bigger and easier, so
"synthetic data doesn't help" would be believable.
**Folder:** `experiments/m1-synthetic-data/`.

### Part D: The final detector and its evaluation

#### D1. Train the final setup with 3 seeds

The setup chosen in C1 and C2, trained with seeds 0, 1 and 2 (an array of
3, two at a time). Report validation mean and range. Seed 0's model is the
M1 detector, as declared in decision 6. Its validation F1-best threshold
becomes **the** M1 threshold.

#### D2. Score on the ExpDate test split (once)

- `scripts/eval_detector.py --run <id> --set real-test` scores with
  `detection_eval` (from B1) at the M1 threshold, and writes
  `results/runs/<id>/real-test.json`. mAP@0.5 on test comes from
  Ultralytics (`model.val(split="test")`).
- **Cross-check:** our `date` recall must match Ultralytics' at the same
  threshold, to within rounding.
All 3 seeds are scored; nothing is retrained or re-chosen afterwards.

**Commit:** `feat: M1 scored on ExpDate test`

#### D3. Score on the store photos (laptop)

**Needs:** the frozen store test set from M0 B3.

1. Copy the weights: `rsync bbhpc:~/bestbefore-data/checkpoints/m1/<id>/weights/best.pt data/checkpoints/m1/<id>/`.
2. `python scripts/eval_detector.py --run <id> --set store` on the
   laptop, using `load_test_set()`, which checks the frozen hash first.
   Input size and threshold are the ones fixed on validation.
3. Results by tag: **expiry found** (`k/n`, IoU 0.5 and 0.3), all dates
   found, and false positives per photo. `other-face` photos have no
   expiry in frame, so they're reported only as "false dates found".
4. A gallery of misses, for error analysis, stays in `data/store/`: it
   shows store photos, so it's never committed.

**Commit:** `feat: M1 scored on the store photos`

#### D4. Report

- `results/m1_detector.md` (committed): the C1 and C2 tables, the
  threshold, ExpDate test results (mean and range over 3 seeds), and
  store results by tag (`k/n`). If the ExpDate paper reports a comparable
  detection number, it goes alongside, with a note on any difference in
  how it was measured.
- `README.md`: the M1 row in the roadmap, and the headline numbers.
- `docs/learnings-m1.md`: the results section filled in.

**Commit:** `docs: M1 results`

---

## Order

```
A1 ─ A2 ─ A3 ─ B1 ─ B2 ─ C1 ─ C2 ─ D1 ─ D2 ─┬─ D4
                                            │
M0 B3 (store test set frozen) ──────── D3 ──┘
```

A3's laptop smoke test can run as soon as B1's code exists. Labelling
the store photos (M0 B1) can happen at any time before D3.

## GPU budget

Rough estimates, to be replaced by A3's measured time per epoch.

| Runs | Count | Estimate per run (H100) |
|---|---:|---|
| C1 at 640 px | 2 | under an hour |
| C1 at 1280 px | 2 | a few hours (4× the pixels) |
| C2 real-only | 1 | short (937 photos) |
| D1 extra seeds | 2 | as the C1 winner |
| **Total** | **7** | about a day of GPU time, two runs at a time |

Checkpoints are about 5 to 20 MB each, so storage is no concern.

## Files at the end of M1

```
docs/              m1-date-detector.md, learnings-m1.md
src/bestbefore/    detector.py, detection_eval.py
scripts/           train_detector.py, eval_detector.py
slurm/             smoke_test.sbatch, train_detector.sbatch
experiments/       m1-size-and-input/, m1-synthetic-data/
tests/             test_detector.py, test_detection_eval.py
results/           m1_detector.md        (committed; runs/ is not)
```

## Risks

- **Store dates may be much smaller than ExpDate's.** Store photos are
  large phone photos, and a date at 640 or 1280 px may be only a few
  pixels tall. If the store result is poor for `small-print`, that is a
  finding, and tiled inference is the fix to try in a later milestone. It
  is not something to tune on the store photos now.
- **The validation split is small** (165 photos). That's why there are
  1-point decision rules, written before the runs, and 3 seeds.
- **Hidden downloads in jobs.** A3 exists to find them.
- **The GPUs are shared.** Jobs may wait in the queue; two at a time is
  the limit anyway.
- **Ultralytics is AGPL-3.0.** That's fine for this open repo. If a
  detector ever ships inside Scanify (a closed app), it would need
  Ultralytics' enterprise licence or a detector under a permissive
  licence. Decide before deployment; it doesn't block M1.

## Later

- **Tiled inference** for very small dates: cut a large photo into
  overlapping tiles, detect in each, merge the boxes.
- **90° rotation augmentation** as a cheap rival to M2's sideways
  synthetic dates: does simple augmentation get most of the gain?
- **Compare a newer family** (YOLO26) or a detector under a permissive
  licence (RT-DETR).
- **A detection baseline with no training:** RapidOCR's own text boxes,
  scored for whether they cover the expiry box.
- **Retrain on train + validation** once all settings are fixed, for a
  little more data.
- **Export the detector to ONNX** for use inside Scanify.
