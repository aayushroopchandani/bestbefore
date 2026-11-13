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

- **Crops hurt RapidOCR's full pipeline, not its recogniser.** On tight
  hand-drawn boxes of v1's 81 expiries, RapidOCR's full pipeline read 34
  right / 2 wrong (A on the whole photo 49/7, B 57/2), but its recogniser
  alone, sideways crops turned, read 70/0, and PP-OCRv6 small's 78/0 (M3
  S2). A detector feeding a recogniser alone can beat B, if it finds the
  dates. Judge M1 only on finding dates.
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
- **A box round a tilted date measures the tilt, not the print.** Store
  dates are barely tilted (none of v1-train's 169 past about 8°), so their
  box size is their print height. Size generated tilted text by its digits,
  and compare sizes with the store over untilted dates (M2b S8).

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
  Ultralytics' `GPU_mem` peaked at 10.1 GB for yolo11n (M2), 21.1 GB for
  yolo11s validating every epoch (M1) and 16.3 GB without, and 30.8 GB for
  yolo11m (M2b step 1; S2 had estimated 44). At 1,280 px one H100 trains
  208 images a second with yolo11n, 181 with yolo11s and 107 with yolo11m
  (M2, M2b step 1).
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
  can show it for a few seconds before it starts. The waiting job first in
  line shows `Resources`, one behind it `Priority` (seen 2026-11-12).
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
  (4.7 GB reported for 24 workers that each peaked near 0.9 GB, M2 S6),
  and it can overcount too: 27.5 GB for a job limited to 16 GB that wasn't
  killed (M3 S1, job 73950). **ONNX Runtime** starts a thread per core of
  the whole node (48) and pins each unless told how many; inside a job's
  cgroup every pin fails (`pthread_setaffinity_np failed`). Set
  `intra_op_num_threads` to the job's CPUs in every session (M3 S1).
  RapidOCR 1.2.3 has no option for it: replace
  `rapidocr_onnxruntime.utils.SessionOptions` before building it
  (`scripts/crop_experiment.py`). PaddleOCR 3.7 runs on Paddle, not ONNX
  Runtime, and takes `cpu_threads` (M3 S2).
- **OCR on CPU nodes is slower than the laptop** (M3 S2, job 73956, `cn1`,
  8 CPUs): a crop took 27 to 40 ms through RapidOCR's recogniser, 87 to
  102 ms through PP-OCRv6 small's and 1.2 to 1.5 s through PaddleOCR's
  full pipeline, 3 to 20 times the laptop's smoke (Apple M2, 4 threads);
  7,176 reads took 37 minutes where the smoke promised about 6. TotalCPU
  was 4 h 58 min in 38:55, the threads kept busy; Paddle prints
  `ReduceMeanCheckIfOneDNNSupport` and an `OMP_NUM_THREADS` warning. Why
  wasn't looked into (oneDNN on crops of every size, likely). Size such
  jobs from a cluster smoke. Loading the three PaddleX models took 56 s
  cold. The reads themselves matched the laptop's: 156 of 156 texts
  identical, scores within 0.0002. A `short` job of 8 CPUs and 16 GB
  started at once with one GPU job of ours running.
- **Storage:** home is on a shared 114 TB parallel filesystem
  (`/fs1/home`). **Quota: 950 GB** (hard limit 1 TB), no limit on the
  number of files; about 65 GB was already used on 2026-10-14. It is
  Lustre: `quota` prints nothing, and `lfs quota -h -u $USER /home` shows
  it (182 GB used on 2026-11-12 after M2b S9, 191 GB after M3 S1's
  models, venv and llama.cpp build). **There is
  no scratch filesystem** and `$SCRATCH` is unset; no purge policy was
  found. Millions of full-size damaged photos (M4) would not fit: make
  them on the fly, or store small crops (a million 30 KB crops is 30 GB).
  **Opening files is slow** (measured 2026-10-23, M2 S6): one process
  opening 16,700 small files took 7 minutes on the login node, about
  25 ms each; on 32 threads it took 8 s. Writing a 185 kB JPEG from a job
  took 15 to 40 ms on average, with stalls of up to 17 s. A GPU node's
  file cache makes a second read fast: Ultralytics read `base`'s training
  photos at 7.6 MB/s cold on `cn5`, and at about 2.8 GB/s in the next run
  minutes later (2026-10-24, M2 S7). A CPU node's cache helps the same way:
  a second job on `cn1` loaded the generator's backgrounds in 2.8 s,
  against 24.8 s cold half an hour before (2026-11-12, M2b S9).
- **Internet:** the login node has it; **compute nodes don't** (`curl`
  exits with code 7, "couldn't connect"). Download data, pip packages and
  model weights on the login node before submitting a job. From there,
  Hugging Face gave about 100 MB/s (Qwen3-VL-2B's 4.27 GB in 40 s) and
  Baidu's BOS about 6 MB/s (M3 S1, 2026-11-12).
- **Pretrained models** live in `$CHECKPOINT_DIR/pretrained/`: YOLO's
  (`scripts/fetch_pretrained.py`), PP-OCRv6's and Qwen3-VL's
  (`scripts/fetch_m3_models.py`, pinned revisions and SHA-256s; `--check`
  re-hashes them). PaddleOCR finds its models without internet when
  `PADDLE_PDX_CACHE_HOME=$CHECKPOINT_DIR/pretrained/paddlex` and
  `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`; otherwise it fetches them
  from Hugging Face on first use. `slurm/m3_cpu.sbatch` also sets
  `HF_HUB_OFFLINE=1` for transformers.
- **Copying store photos there** (M2b S6, 2026-11-04): the laptop's `rsync`
  is macOS's openrsync (protocol 29; the cluster has rsync 3.1.3). It
  refuses `--chmod=D700,F600`, accepts `--chmod=u=rwX,go=` but applies
  nothing, and `-a` copies the laptop's permissions (755, 644), even onto a
  folder made 700. So make the laptop's copy owner-only first (folders
  700, files 600): `-a` then carries that across, and nothing is open even
  mid-sync (S6's second sync: 0 entries open after it). Still `chmod` on
  the cluster after every sync and check with `find ... -perm /077`. A
  re-export gives every file a new time, so rsync lists all 710 files but
  sends only what changed (1 MB of 166 MB in 11 s). The home folder is 700;
  `~/bestbefore-data` is 775 and the umask 0002. The store export lives in
  `~/bestbefore-data/store/`, `STORE_TRAIN_DIR` in the cluster's `.env`
  since 2026-11-12 (M2b S10). Beside it, `text.json` (M3 S2): each
  numbered photo's date values, written on the laptop with mode 600 and
  copied the same way. Store crops cut on the cluster go into owner-only
  run folders under `~/bestbefore-data/generated/m3/` (the job sets
  `umask 077`). Ultralytics writes v1-val's label cache
  there, `labels/val.cache`, with mode 664: `chmod 600` it after its first
  scoring (it is rewritten only if v1-val's photos change). A `du` over all of `~/bestbefore-data`
  (M2's generated photos) ran over 5 minutes on the login node without
  finishing.
- **Fetching files from it:** `scp` doesn't expand braces in a remote
  path (it speaks SFTP); fetch several at once with
  `ssh bbhpc 'cd DIR && tar cf - FILES' | tar xf -` (M2b S9).
- **CPUs:** compute nodes are Intel Xeon Gold 6442Y (x86-64); the laptop
  is arm64. The login node `mn1` is the same model (2 × 24 cores, 96
  threads, AVX-512 BF16 and AMX; checked 2026-11-12). With the same
  library versions, floating-point results still differ in their last
  bits, so generated photos match the laptop's
  byte for byte only mostly (85% in M2 S6; the rest within a few grey
  levels). gen-v2 (M2b S9) matched for about 90% of photos, and 11 of
  7,400 sidecar lines differed (a block placed elsewhere, a box 1 px off,
  the JPEG quality drawn after the noise), never a block's size. A photo
  that drifts draws its later blocks its own way, so a crash can be met on
  the cluster only: S9's first job crashed on a photo the laptop draws
  fine. A gen-v2 worker peaked at 1.8 GB there, twice the laptop's
  measure, so its 24 workers asked 48 GB. A busy 24-CPU job ran the generator about 1.6 times slower per
  photo than the laptop's cores. The laptop (8 GB of RAM) can't train at
  1,280 px: a batch of 16 keeps 13 to 55 GiB for the backward pass
  (yolo11n to yolo11m), and even a 1-photo run froze it in Ultralytics'
  validation at the end. A 1-epoch yolo11s smoke at 1,280 px on 793
  photos took 11 minutes on 24 CPUs of `short` (30.5 GB by `sacct`'s
  sample), so training smoke runs go there (`slurm/m2b_smoke_cpu.sbatch`,
  M2b S2). Its `step2` task (M2b S10, job 73951) trained yolo11n's ft and
  two-stage smokes on 413 photos in 3 and 4 minutes and scored the M2b
  suite's 5,070 photos in 12 (about 40 ms an image): 19 minutes in all,
  17.1 GB by `sacct`'s sample.
- **Python:** modules `Python/3.11.11` (same minor version as the laptop),
  `cuda/12.6.3`, `anaconda3/2024.10`. The system `python3` is 3.9: don't
  use it. The module provides `python3`, not `python`, and the venv only
  runs after `module load Python/3.11.11` (it needs the module's
  `libpython3.11.so`). Two environments (M3 S1): the main venv
  `~/bestbefore/.venv` (the train, synth, read and vlm extras), and
  `~/venvs/ocr-cpu` for CPU jobs (Paddle 3.2.0 CPU, PaddleOCR 3.7.0,
  RapidOCR 1.2.3), which holds **one OpenCV**: PaddleX pins
  `opencv-contrib-python==4.10.0.84` and RapidOCR asks for `opencv-python`,
  two packages writing one `cv2/`, so RapidOCR goes in with `--no-deps`.
  `~/venvs` holds other projects' venvs too. A fresh venv from the module
  starts with pip 24.0. **To add packages without moving any:** `pip
  freeze --exclude-editable > constraints.txt`, then `pip install -c
  constraints.txt ...` (`--dry-run --report r.json` first shows what would
  come); M3 S1 added 28 packages that way and changed none. The whole test
  suite (737 tests) takes 3 to 5 minutes on the login node, niced.
- **Building code** on the login node: it is shared, so a few threads
  (`-j 4`), niced. The default compiler is OpenHPC's gcc 12.3.0 (module
  `gnu12`, loaded), with the system's binutils 2.30 (RHEL 8.8), which
  **can't assemble AMX instructions**; `source
  /opt/rh/gcc-toolset-12/enable` gives gcc 12.2.1 with binutils 2.38, which
  can. cmake: the system's 3.20.2, modules `cmake/3.24.2` and `3.31.4`;
  also `gnu14/14.2.0`, `llvm/latest`, `intel/2025.0.4`. llama.cpp `b12017`
  is built in `~/opt/llama.cpp` (CPU only, every CPU variant, picked at
  run time; 269 s on 4 threads); its configure downloads a web UI from a
  moving tag unless `LLAMA_USE_PREBUILT_UI=OFF` (M3 S1). **Its AMX variant
  (`sapphirerapids`) dies with `Illegal instruction`** at the first
  computation on `cn1`, though AMX itself runs there (a C test, job
  73949), so it is kept in `build/bin/disabled/` and ggml picks `zen4`
  (AVX-512 VNNI, BF16): Qwen3-VL-2B Q8_0 at 23 tokens a second on 4 CPUs.
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
