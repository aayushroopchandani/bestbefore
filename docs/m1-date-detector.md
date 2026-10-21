# M1: Date detector

Planned 2026-10-14. Status: **A, B, C, D1 and D2 done** (2026-10-16). The
M1 detector is **`n640`** (`yolo11n`, 640 px, synthetic + real, seed 0),
threshold 0.728. On the ExpDate test split it finds **637 of 665 expiry
dates** (mAP@0.5 0.993). Next: D3, once the store photos are labelled.
The ideas behind this plan, explained simply: [`learnings/m1.md`](learnings/m1.md).

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
   .venv/bin/pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
       --index-url https://download.pytorch.org/whl/cu128
   .venv/bin/pip install -e ".[dev,download,train]"
   mkdir -p slurm/logs
   ```

   **PyTorch must be the CUDA 12.8 build.** The GPU driver is version 570,
   which supports CUDA up to 12.8. Plain `pip install` picks the newest
   PyTorch, built for CUDA 13, which needs driver 580 or newer and would
   not see the GPUs. torch 2.11.0 is the newest CUDA 12.8 build, so it's
   pinned in `pyproject.toml` for the laptop too.

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

1. **On the login node, pre-fetch everything a job would download:**
   `scripts/fetch_pretrained.py` puts `yolo11n.pt`, `yolo11s.pt` and
   `yolo26n.pt` (only used by Ultralytics' mixed-precision check) into
   `$CHECKPOINT_DIR/pretrained/`, fetches the plot font (`Arial.ttf`,
   loaded with every dataset), and points Ultralytics' `weights_dir`
   setting at that folder. Compute nodes have no internet, so any download
   inside a job fails; the chain also sets `YOLO_OFFLINE=1` so Ultralytics
   doesn't even try.
2. **Laptop smoke test:** train `yolo11n` for 1 epoch on 2% of the data
   (`fraction=0.02`) on CPU. This catches bugs in our own code without
   using the cluster at all. **Passed 2026-10-14:** 19 training photos,
   320 px, 26 seconds end to end; `run.json`, `data.yaml`, `val.json` and
   `best.pt` all written.
3. **Cluster smoke test:** the first step of `scripts/m1_chain.py`
   (inside GPU job 72789): 1 epoch on 2% of the photos at 640 px. The
   chain stops if it fails. Before it, `slurm/m1_chain.sh` checks that
   PyTorch sees the GPU.
4. Write the measured time per epoch into this plan's GPU budget table
   (below), replacing the estimates.

**Commit:** `feat: smoke-test job for the cluster`

### Part B: Training code

#### B1. The training script

**Goal:** one command trains one detector, records everything, and scores
it on validation.

- `src/bestbefore/sets.py`: a `real-val` set, the 165 validation photos
  read from the converter's `val.txt` (the one record of which they are).
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

**Queue strategy (decided 2026-10-14).** This cluster's queue is first
come, first served: every job has priority 1 and all priority weights are
zero (`sprio`). So instead of one job per run, which would mean queueing
again for each, M1 holds **one GPU for one long job** and runs everything
back to back:

- `slurm/m1_holder.sbatch` (job 72789, submitted 2026-10-14, 1 GPU for up
  to 2 days) holds our place in the queue. When it starts, it runs
  `slurm/m1_chain.sh`, which Slurm reads only at that moment, so it can be
  written after submitting. If the chain isn't there yet, the job exits
  at once rather than keep a GPU idle.
- `slurm/m1_chain.sh` runs, in order: the GPU smoke test (A3), the 4 C1
  runs, the C1 decision, the C2 run, the C2 decision, and the 2 extra D1
  seeds. The decisions apply the rules written in C1 and C2 **before** any
  run, so no one has to choose by hand in the middle of the night.
- The chain is **resumable**: a run whose `val.json` exists is skipped, so
  if the 2-day limit cuts it short, a resubmit carries on where it
  stopped.
- `slurm/train_detector.sbatch` stays, for one-off runs later.

**Deadline:** the chain and the B1 code must be pushed and pulled on the
cluster before job 72789 starts. Slurm's estimate is 2026-10-15 about
21:15, but a running job finishing early could bring it forward.

**Check:** a 1-epoch run submitted with `sbatch` finishes, and its
`run.json` and `val.json` are where they should be.
**Commit:** `feat: Slurm job script for detector training`

### Part C: Experiments (validation split only)

Each question gets a folder under `experiments/` with a README (question,
why, method, decision rule, result, what it means) and a `summarise.py`
that rebuilds the README's results table from the runs. It applies the
same decision rule as the chain (`bestbefore.m1`) and warns if the two
disagree. To read results on the laptop:

```bash
scripts/pull_cluster_results.sh          # small result files only, no weights
python experiments/m1-size-and-input/summarise.py --write
python experiments/m1-synthetic-data/summarise.py --write
```

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

The setup chosen in C1 and C2, trained with seeds 0, 1 and 2. Report
validation mean and range. Seed 0's model is the M1 detector, as declared
in decision 6. Its validation F1-best threshold becomes **the** M1
threshold.

**Done 2026-10-16.** C1 chose `n640` (all four within the 1-point noise
band) and C2 kept the synthetic photos
([`experiments/m1-size-and-input`](../experiments/m1-size-and-input/README.md),
[`experiments/m1-synthetic-data`](../experiments/m1-synthetic-data/README.md)).
On the 165 validation photos (187 dates):

| Seed | mAP@0.5 date | Dates found | Precision | Threshold |
|---:|---:|---:|---:|---:|
| 0 (the M1 detector) | 0.991 | 185 | 0.974 | 0.728 |
| 1 | 0.991 | 183 | 0.989 | 0.667 |
| 2 | 0.994 | 184 | 0.979 | 0.664 |

Mean mAP@0.5 0.992, range 0.003: the result doesn't depend on the seed.
Our evaluator's date recall agreed with Ultralytics' own, within 0.02, on
every run.

#### D2. Score on the ExpDate test split (once)

- `scripts/eval_detector.py --set real-test --runs n640 n640-seed1
  n640-seed2` scores with `detection_eval` (from B1), each run at its own
  validation threshold, and writes `results/runs/<run>/real-test.json`.
  mAP@0.5 on test comes from Ultralytics (`model.val(split="test")`). It
  refuses to re-score a run on the test split unless `--again` is given.
- **Cross-check:** our `date` recall must match Ultralytics' at the same
  threshold, to within rounding.
All 3 seeds are scored; nothing is retrained or re-chosen afterwards.

**Done 2026-10-16, on the laptop's CPU** (the store photos must be scored
there too, so both use the same device).

- **Device check first.** Validation re-scored on the laptop matched the
  cluster's GPU numbers (mAP@0.5 0.9909 against 0.9908), except one date
  out of 187: its confidence is 0.7226 on the CPU, 0.005 under the
  threshold, and the GPU's half-precision arithmetic put it just over.
- **Results** on the 665 test photos (733 dates, 665 of them expiry dates):

  | Run | Threshold | mAP@0.5 date | Dates found | Expiry found | Precision | False boxes / photo |
  |---|---:|---:|---:|---:|---:|---:|
  | **`n640`, seed 0 (the M1 detector)** | 0.728 | 0.993 | 697/733 | **637/665** | 0.987 | 0.01 |
  | `n640`, seed 1 | 0.667 | 0.993 | 713/733 | 651/665 | 0.983 | 0.02 |
  | `n640`, seed 2 | 0.664 | 0.992 | 712/733 | 650/665 | 0.974 | 0.03 |

  Mean over the seeds: mAP@0.5 0.993 (0.992 to 0.993), date recall 0.965
  (0.951 to 0.973), expiry recall 0.971 (0.958 to 0.979). At IoU 0.3 only
  1 or 2 more dates count as found, so the boxes are in the right place.
- **Cross-check passed:** our date recall and Ultralytics' agree within
  0.003 on all three runs.
- **Finding: the threshold, not the model, limits recall.** The seeds
  have the same mAP, but seed 0 chose a stricter threshold on validation
  (0.728 against about 0.665) and finds 14 fewer expiry dates. Seed 0
  puts a box on **all 665** test expiry dates at some confidence; the 28
  it misses are found with confidence under 0.728 (21 of them between 0.5
  and 0.728). F1 weighs a missed date and a false box equally, but for
  this project a missed date costs more. **The threshold stays 0.728 for
  M1**, because changing it now would mean choosing it from the test
  photos. The lesson goes to Later: pick the threshold for recall, on
  validation, before the next test.

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
   *(Added after D2, as a diagnostic only:)* how many expiry dates get a
   box at **any** confidence, which separates "can't see it" from "saw it
   but wasn't sure". It doesn't change the threshold or the headline
   numbers.
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
- `docs/learnings/m1.md`: the results section filled in.

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

Measured in GPU job 72789 (one H100 NVL, 2026-10-15 21:13 to 2026-10-16
04:55). The estimates made before the runs were about twice as long.

| Runs | Count | Estimated per run | Measured per run |
|---|---:|---|---|
| GPU smoke test | 1 | minutes | 2.6 min |
| C1 at 640 px (`n640`, `s640`) | 2 | under an hour | 66 and 62 min |
| C1 at 1280 px (`n1280`, `s1280`) | 2 | a few hours | 97 and 102 min |
| C2 real-only | 1 | short | 6 min |
| D1 extra seeds (`n640`) | 2 | as the C1 winner | 61 and 62 min |
| **Total** | **8** | about a day | **7 h 42 min**, in one job |

`s640` took no longer than `n640`, so loading the photos, not the GPU,
set the pace.

Checkpoints are about 5 to 20 MB each, so storage is no concern.

## Files at the end of M1

```
docs/              m1-date-detector.md, learnings/m1.md
src/bestbefore/    detector.py, detection_eval.py, m1.py (grid and decision rules)
scripts/           fetch_pretrained.py, train_detector.py, m1_chain.py,
                   pull_cluster_results.sh, eval_detector.py
slurm/             m1_holder.sbatch, m1_chain.sh, train_detector.sbatch
experiments/       m1-size-and-input/, m1-synthetic-data/
tests/             test_detector.py, test_detection_eval.py, test_m1_rules.py
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

- **Leave the 14 synthetic copies of validation photo `img_00277.jpg`
  out of training** the next time the detector is trained (M2), so
  validation is fully clean. Found in
  [`experiments/expdate-near-duplicates`](../experiments/expdate-near-duplicates/README.md);
  it changes none of M1's results.
- **Pick the threshold for recall, not F1** (found in D2). For example:
  the highest threshold that keeps validation recall at 99% or more.
  Decide it on validation, before the next test, never from test or
  store results.
- **Set `optimizer` and `lr0` explicitly** in `detector.SETTINGS`.
  `optimizer=auto` picks by the number of training steps, so C2's two
  runs got different optimisers (MuSGD for synthetic + real, AdamW for
  real only), which makes C2 a two-variable comparison. Found
  2026-10-20; the C2 decision stands
  ([`experiments/m1-synthetic-data`](../experiments/m1-synthetic-data/README.md)).
- **Train `n640` for longer.** It ran all 100 epochs without early
  stopping, so it may still have been improving.
- **Faster data loading** (caching the photos in memory, more loader
  workers): **planned for M2's setup** (decided 2026-10-21): test
  `cache="ram"` (not `"disk"`, whose `.npy` files would sit on the same
  network filesystem) and more `workers` on a 2-epoch `n640` run, then
  use the faster setting in every M2 arm.
  `s640` trained no faster than `n640`, so the GPU was waiting
  on the photos.
- **Tiled inference** for very small dates: cut a large photo into
  overlapping tiles, detect in each, merge the boxes.
- **90° rotation augmentation** as a cheap rival to M2's sideways
  synthetic dates: does simple augmentation get most of the gain?
  **Moved to M2** (decided 2026-10-21): an arm of M2's ablation (none /
  90° rotation / synthetic / both), scored on the store photos by tag.
  Exactly 90°, not YOLO's small-angle `degrees`, which loosens the
  upright boxes around long, thin dates.
- **Compare a newer family** (YOLO26) or a detector under a permissive
  licence (RT-DETR).
- **A detection baseline with no training:** RapidOCR's own text boxes,
  scored for whether they cover the expiry box.
- **Retrain on train + validation** once all settings are fixed, for a
  little more data.
- **Export the detector to ONNX** for use inside Scanify.
