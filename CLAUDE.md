# BestBefore

Trained models that read the expiry date printed on product packaging, and
tell from a camera photo whether that date will be readable at all. A
student portfolio project: trained on public data on a university HPC
cluster, tested on real store photos.

---

## Why this exists

It grows out of **Scanify**, an app the owner built for a pet-supplies
retailer. The retailer's inventory system has no expiry dates, so staff
photograph each pack and the app reads two things from the photo: the
barcode (`zxing-cpp`) and the printed expiry date (`RapidOCR`, then a
hand-written rule-based date parser).

What Scanify measured on its store photos:

- **32 of 44** photos with a known expiry are read right, 12 missed, 0 read
  wrong.
- The parser extracts every date the OCR returns. The limit is **reading**
  the print, not parsing the text.
- The misses: dot-matrix print on curved caps and jar bottoms, sideways
  stamped dates, small print, and some large, clear "Best Before" lines the
  OCR skipped entirely.
- A blur score (variance of the Laplacian) did **not** predict failures.
  Photos that were read and photos that were missed scored the same. The
  score mostly measured how much fine print was in the frame, and the one
  photo known to be out of focus was read.

This project replaces hand-written rules with trained models, starting
where the rules ran out.

## Goal

Two models, each beating a measured baseline on real photos held out from
training:

1. **The date reader.** Find every date on a pack, tell the expiry date from
   the manufacture date, and read it as day / month / year.
2. **The readability checker.** Given a photo, predict whether the date
   reader will read it right. This is the useful version of a blur score:
   it learns what actually makes reading fail, not just focus. It must be
   small enough to run in a phone browser on the live camera preview, so
   staff are warned before they take the photo.

Success is a results table **broken down by failure type** (dot-matrix,
sideways, curved, small print, clean), not one headline number.

---

## Data

### Public: for training

**ExpDate** (Seker & Ahn, 2022). Licence CC BY 4.0: cite the paper and
credit KIST (Korea Institute of Science and Technology). Official page,
with a Google Drive download:
https://felizang.github.io/expdate/index_expdate.html. Annotations are JSON.

| Subset | Size | Used for | Milestone |
|---|---|---|---|
| Products-Real | 1,767 real photos (1,102 train / 665 test) | Detecting dates. Classes `date`, `due mark`, `production mark`, `code mark` | M1 |
| Products-Synth | ~12,000 synthetic | Detection, training only | M1 |
| Date-Synth / Date-Real | 128,000 synthetic + a real test set | Finding day / month / year inside a date crop | M3 |
| Components-Synth / -Real | 450,000 synthetic + a real test set | Reading the characters | M3 |

Citation: Seker, A. C. & Ahn, S. C. (2022). *A generalized framework for
recognition of expiration dates on product packages using fully
convolutional networks.* Expert Systems with Applications, 203, 117310.

### Public: for testing the checker only (M4)

- **SmartDoc-QA**: 4,260 smartphone photos of paper documents with
  defocus, motion blur and bad lighting, each labelled with its OCR
  accuracy. Its authors say it is not for training OCR, so use it only to
  test the checker.
- **SPAQ**: 11,125 photos from 66 phones with human sharpness ratings.
  Tests whether the checker responds to focus at all.

### Private: store photos

About 99 photos (**store set v1**) taken in the retailer's stores. **They
belong to the retailer: never commit them, never publish them** (no
GitHub, no README figures, no public dataset). Anything committed holds
aggregate numbers only, never a photo or a photo name.

**Permission from the retailer's owner, 2026-10-28:** the store photos may
be used to **train** models and may be **shown to AI tools** (such as
Claude) for analysis; the products on them are public. Publishing is not
part of this permission. Until then they were test-only.

- **Store set v1 becomes training data** from the first milestone that
  trains on it. From then on it can't judge a model.
- **A new set of store photos is the test set:** labelled and frozen
  (hashed, like v1) **before** any training on store photos. Never train
  on it, never tune on it. This is **store set v2**: 99 photos in
  `testset2/`, photo list frozen 2026-10-28, labels still to do
  (`docs/m0-data-and-test-set.md`, Part D). The Scanify repo doesn't
  ignore `testset2/`; only this laptop's `.git/info/exclude` there does.
- Results on v1 from M0 to M2 were measured before any store training and
  stay as recorded.

- 44 have a known expiry (Scanify `backend/tests/ground_truth.py`). The
  rest (`samples1/`, `samples3/`) still need expiry labels. That is M0.
- They need **tight** boxes drawn around each date for M1. Scanify's
  `date_panel_boxes.py` boxes are deliberately loose (the whole
  batch/MFG/expiry block), so they can't score a detector.
- Each photo gets one failure-type tag: `dot-matrix`, `sideways`, `curved`,
  `small-print`, `other-face`, `clean`.

Paths come from environment variables in `.env` (gitignored):

| Variable | Points to |
|---|---|
| `EXPDATE_DIR` | the unpacked ExpDate download |
| `STORE_PHOTOS_DIR` | the folder holding `samples1/`, `samples2/`, `samples3/` (store set v1) and `testset2/` (v2) |
| `SCANIFY_DIR` | the Scanify repo, for baseline B |

### Date formats on the store packs

Indian pet-product packs print the expiry in at least 9 formats:
`04/12/2026`, `April 2031`, `10/2027`, `20260518`, `2028.JAN.21`,
`MM/YYYY-MM/YYYY` ranges and more. Some print only a manufacture date plus
a shelf life ("Best Before 1 Year from Mfg. Dt."), so the expiry has to be
worked out. ExpDate may not cover every one of these formats; the M2
generator must.

---

## Baselines

- **A: RapidOCR off the shelf, plus a simple date regex.** Public and
  reproducible. Run it on ExpDate.
- **B: Scanify's reader** (`app.scanner.scan` in `$SCANIFY_DIR`), on ExpDate
  and on the store photos. Call it by path. Never copy Scanify code into
  this repo.

## Milestones

Plan each milestone as a sliced doc in `docs/` before writing its code.

| # | Milestone | Done when |
|---|---|---|
| **M0** | **Data and test set.** Download Products-Real and Products-Synth; draw the boxes on ~50 photos as a contact sheet; collect stats (box size relative to the photo, dates per photo, class counts); convert the JSON to YOLO format. Label the remaining store photos, draw tight date boxes, tag failure types, then freeze the test set. Run baselines A and B. | Baseline table on ExpDate test and on the store photos, by tag |
| **M1** | **Date detector.** Fine-tune a small YOLO on Products-Synth plus the Products-Real train split. | Recall and mAP@0.5 on Products-Real test; expiry-box recall on store photos |
| **M2** | **Synthetic generator.** Dot-matrix and inkjet fonts, the store date formats, wrapped onto cylinders, rotated sideways, with glare and small print. Retrain M1 with and without it. | An ablation showing what synthetic data adds on the store photos |
| **M2b** | **Store photos and a bigger detector.** Fine-tune on store set v1, split by pack design, with the public data replayed; gen-v2 draws the layouts and date sizes the generator missed; `s` and an `m` probe against `n`; the threshold chosen on store photos (`docs/m2b-store-and-size.md`). | Model size, the fine-tuning recipe and gen-v2 against the no-training control (M2's detector at a store-chosen threshold); the M2b detector scored once on store set v2 by tag, cause and seen or unseen pack design, next to M1 and M2 |
| **M3** | **Date reader.** A recogniser trained on crops (Date-Synth, Components-Synth, M2 data), or a small vision-language model fine-tuned to return `{expiry, mfg, format}` as JSON. | End-to-end date accuracy vs baselines A and B, by tag |
| **M4** | **Readability checker.** Make millions of damaged copies (defocus, motion blur, glare, low light, JPEG, rotation), label each by whether the M3 reader still reads it, and train a small classifier on that label. | AUROC on held-out copies; correlation with OCR accuracy on SmartDoc-QA and with sharpness on SPAQ; results on store photos |
| **M5** | **On the phone.** Export the checker to ONNX, quantise it, run it in a phone browser. | Size, latency per frame, accuracy kept after quantising |

## Metrics

- **Detection:** recall and mAP@0.5. On store photos, did it find the
  expiry date (IoU ≥ 0.5)?
- **Reading:** exact date match on day, month and year; on month and year
  only when the pack prints no day. Count **wrong reads separately from
  no reads.** A wrong date is worse than none: reading `08/2027` as
  `08/2021` would mark a whole batch as expired.
- **Checker:** AUROC for "will be read right". Also report the agreement
  with OCR accuracy on SmartDoc-QA and with sharpness ratings on SPAQ.
- Always report **per failure tag**. Never train on, or tune
  hyperparameters against, any test split: the frozen store test set
  (v1 until store training starts, the new set after) or ExpDate test.

## Lessons already measured (don't re-learn them)

- **Crops hurt off-the-shelf OCR.** RapidOCR read fewer dates from perfect
  hand-drawn crops (25/44) than from the whole photo (31/44). So detector
  plus RapidOCR on crops will **not** beat baseline B. The gain needs a
  recogniser trained on crops (M3). Judge M1 only on finding dates.
- **Heavy preprocessing hurts OCR.** Adaptive thresholding made it worse;
  mild sharpening helped slightly. Models trained on natural photos want
  natural photos.
- **Resolution:** for RapidOCR, a longest side of 1000 px beat 1600, 2200
  and the original size.
- **A blur score over the whole frame measures detail, not focus.** If blur
  is scored again, score only the region around the text.
- About a third of the store packs have the date and the barcode on
  different faces. That is a capture problem, not a reading problem; tag
  it `other-face` and leave it out of reading accuracy.

---

## HPC

Checked on 2026-10-14. Login details stay in `~/.ssh/config`, never in
this repo.

- **Cluster and scheduler:** the university cluster (OpenHPC on Red Hat),
  **Slurm**.
- **GPUs:** NVIDIA **H100 NVL** (95,830 MiB each, `nvidia-smi` in M1's
  and M2's jobs), 2 per node, on 2 GPU nodes (`cn5`, `cn6`: 48 CPUs and
  about 500 GB RAM each, shared with other users). Partitions with GPUs:
  `gpu` (3-day limit), `medium` (2 days), `long` (7 days). CPU-only:
  `short` (6 h; `cn1` to `cn4`, 48 CPUs and 510 GB each, checked
  2026-11-02), `defaultq` (default, 3 days), `smp` (7 days). One H100 per
  job is plenty for every model here: at batch 16 and 1,280 px,
  Ultralytics' `GPU_mem` peaked at 10.1 GB for yolo11n (M2) and 21.1 GB
  for yolo11s (M1); yolo11m is estimated at about 44 GB (M2b S2).
- **Job limits** (checked 2026-10-22): QOS `limit_gpu_qos` is the
  account's only QOS, so it covers **every** job on every partition,
  CPU-only ones and other projects' too: **2 running** and **5 submitted**
  (waiting + running) per user. Each task of a job array counts as one
  job.
- **Queue:** first come, first served by submit time (every job has
  priority 1, every partition the same tier). A job submitted with
  `--hold` keeps its submit time when released, and a waiting job's CPUs,
  memory and time limit can be changed with `scontrol update`. Other
  users are capped at 2 running jobs too, so `squeue --start` can be far
  too early. It gives the time the first GPU frees up (on 2026-10-22 the
  same time to five jobs), and leaves out older jobs held back only by
  their owners' caps. To estimate a start, count the older GPU jobs
  ahead and the running ones' end times. A waiting job's reason
  `ReqNodeNotAvail, UnavailableNodes:cn[5-6]` or `Nodes required for job
  are DOWN, DRAINED or reserved for jobs in higher priority partitions`
  only means every GPU is busy: there was no reservation (`sinfo -T`) and
  every partition is tier 1 (checked 2026-11-02). A job just submitted
  can show it for a few seconds before it starts.
- **Inside a job** (Slurm 23.11.4, checked 2026-10-22):
  `SLURM_JOB_END_TIME` holds the job's end (Unix time), so a job can tell
  how long it has left. At the time limit or a `scancel`, every process
  gets SIGTERM, then SIGKILL 30 s later (`KillWait`). Processes are
  tracked by cgroup (`proctrack/cgroup`), so even one in its own process
  group ends with its job. A job whose node fails is requeued under the
  same id and runs its script again from the top (`JobRequeue=1`).
  `squeue --me` lists only our own jobs. While 2 of our jobs run, a waiting
  one shows reason `QOSMaxJobsPerUserLimit` (seen 2026-10-23).
  `sacct`'s `MaxRSS` is sampled, so it can be far under the true peak
  (4.7 GB reported for 24 workers that each peaked near 0.9 GB, M2 S6).
- **Storage:** home is on a shared 114 TB parallel filesystem
  (`/fs1/home`). **Quota: 950 GB** (hard limit 1 TB), no limit on the
  number of files; about 65 GB was already used on 2026-10-14. **There is
  no scratch filesystem** and `$SCRATCH` is unset; no purge policy was
  found. Millions of full-size damaged photos (M4) would not fit: make
  them on the fly, or store small crops (a million 30 KB crops is 30 GB).
  **Opening files is slow** (measured 2026-10-23, M2 S6): one process
  opening 16,700 small files took 7 minutes on the login node, about
  25 ms each; on 32 threads it took 8 s. Writing a 185 kB JPEG from a job
  took 15 to 40 ms on average, with stalls of up to 17 s. A GPU node's
  file cache makes a second read fast: Ultralytics read `base`'s training
  photos at 7.6 MB/s cold on `cn5`, and at about 2.8 GB/s in the next run
  minutes later (2026-10-24, M2 S7).
- **Internet:** the login node has it; **compute nodes don't** (`curl`
  exits with code 7, "couldn't connect"). Download data, pip packages and
  model weights on the login node before submitting a job.
- **Copying store photos there** (M2b S6, 2026-11-04): the laptop's `rsync`
  is macOS's openrsync (protocol 29; the cluster has rsync 3.1.3). It
  refuses `--chmod=D700,F600`, accepts `--chmod=u=rwX,go=` but applies
  nothing, and `-a` copies the laptop's permissions (755, 644), even onto a
  folder made 700. So `chmod` on the cluster after every sync (folders 700,
  files 600) and check with `find ... -perm /077`. The home folder is 700;
  `~/bestbefore-data` is 775 and the umask 0002. The store export lives in
  `~/bestbefore-data/store/`. A `du` over all of `~/bestbefore-data` (M2's
  generated photos) ran over 5 minutes on the login node without finishing.
- **CPUs:** compute nodes are Intel Xeon Gold 6442Y (x86-64); the laptop
  is arm64. With the same library versions, floating-point results still
  differ in their last bits, so generated photos match the laptop's
  byte for byte only mostly (85% in M2 S6; the rest within a few grey
  levels). A busy 24-CPU job ran the generator about 1.6 times slower per
  photo than the laptop's cores. The laptop (8 GB of RAM) can't train at
  1,280 px: a batch of 16 keeps 13 to 55 GiB for the backward pass
  (yolo11n to yolo11m), and even a 1-photo run froze it in Ultralytics'
  validation at the end. A 1-epoch yolo11s smoke at 1,280 px on 793
  photos took 11 minutes on 24 CPUs of `short` (30.5 GB by `sacct`'s
  sample), so training smoke runs go there (`slurm/m2b_smoke_cpu.sbatch`,
  M2b S2).
- **Python:** modules `Python/3.11.11` (same minor version as the laptop),
  `cuda/12.6.3`, `anaconda3/2024.10`. The system `python3` is 3.9: don't
  use it. The module provides `python3`, not `python`, and the venv only
  runs after `module load Python/3.11.11` (it needs the module's
  `libpython3.11.so`).
- **GPU driver:** 560.35.05 on both GPU nodes (`nvidia-smi` in M1's job on
  `cn6`, 2026-10-15, and M2's on `cn5`, 2026-10-24; the 570.86 first
  written here was seen on neither). Install PyTorch from the `cu128`
  index (`torch==2.11.0+cu128`), which trains on this driver in both
  jobs; the default PyPI build needs CUDA 13 and won't see the GPUs.

Conventions:

- Job scripts live in `slurm/`.
- Data and checkpoints go outside the git checkout, in
  `~/bestbefore-data/`. (The original rule said "on scratch, never in
  home", but this cluster has no scratch.)
- Synthetic generation and OCR labelling run as **job arrays**: every image
  is independent. With the 5-job cap, an array has a few tasks, each using
  many CPUs.
- Every run records its config, git commit and random seed.

## Repo layout

```
bestbefore/
├── CLAUDE.md
├── README.md            the public face: problem, results, how to reproduce
├── .env.example         EXPDATE_DIR, STORE_PHOTOS_DIR, SCANIFY_DIR
├── docs/                milestone plans (sliced), dataset notes
├── src/bestbefore/      datasets, format conversion, synthesis, models, eval
├── scripts/             command-line entry points
├── slurm/               job scripts
├── experiments/         one folder per question: script + README
├── data/                gitignored
└── results/             gitignored, except the summary tables
```

---

## Working with the owner

- The owner is a student building this for their portfolio. Explain why a
  choice is made in a sentence or two, and give a recommendation rather
  than a survey of options.
- **Plans:** simple, optimised and clean, split into slices, as markdown in
  `docs/`. Read the existing docs first. Call out any departure from a
  decision already written down. Put deferred ideas in a "Later" section;
  don't drop them.
- **Experiments:** one folder per question under `experiments/`, holding
  the script and a README with the question, method, result and decision.
  The README's numbers must be reproducible from its script.
- **Report honestly.** A model that doesn't beat the baseline is still a
  result; write it down.
- **Commits:** lowercase conventional prefixes (`feat:`, `fix:`, `test:`,
  `docs:`, `refactor:`), one logical change each, plan doc first, and every
  commit should still run on its own. **No `Co-Authored-By` trailer.** When
  the owner asks for commit text, give each commit as `git add <files>` and
  `git commit -m "..."` together in one code block.
- **Never commit:** store photos, ExpDate files, model weights, `.env`.
